#!/usr/bin/env python3
"""PHASE D: frozen inferred relation-conditioned topology refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from run_phase_c_oracle_refinement_v1 import (
    alignment,
    as_bool,
    candidate_cost_lookup,
    dominant_state_lookup,
    edge_key,
    mapped_edge_set,
    prf,
    replace_edge,
    truth_edge_set,
)


GAMMA = 0.35
MAX_BOUNDARY_SWAPS_PER_EVENT = 1
MIN_DOWNSTREAM_SUPPORT_FRACTION = 0.20
EPS = 1e-12


def parse_tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {token for token in str(value).split(";") if token and token != "nan"}


def parse_genes(value: object) -> list[str]:
    genes: list[str] = []
    for token in str(value).split(";"):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", maxsplit=1)
            genes.extend(f"G{index}" for index in range(int(left[1:]), int(right[1:]) + 1))
        else:
            genes.append(token)
    return genes


def topology_localization_support(
    run_dir: Path,
    relation,
    mapping: pd.DataFrame,
    assignments: pd.DataFrame,
    catalog_path: Path,
) -> dict[str, float]:
    """Relocalize the structural entry without re-testing relation significance."""
    adata = ad.read_h5ad(run_dir / "event" / "synthetic.h5ad")
    matrix = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    matrix = np.log1p(np.maximum(matrix.astype(float), 0.0))
    gene_index = {str(gene): index for index, gene in enumerate(adata.var_names)}
    catalog = pd.read_csv(catalog_path)
    primary = catalog[catalog.program_role.eq("primary_relation")].iloc[0]
    signature = catalog[catalog.program_role.eq("type_signature")].iloc[0]
    source_genes = parse_genes(primary.source_genes)
    receiver_genes = parse_genes(primary.receiver_genes)
    if str(relation.event_type) == "collision_convergence":
        receiver_genes += parse_genes(signature.receiver_genes)
    source_columns = [gene_index[gene] for gene in source_genes if gene in gene_index]
    receiver_columns = [gene_index[gene] for gene in receiver_genes if gene in gene_index]
    if not source_columns or not receiver_columns:
        return {}
    source_score = matrix[:, source_columns].mean(axis=1)
    receiver_score = matrix[:, receiver_columns].mean(axis=1)
    cell_to_mc = mapping.set_index("cell_name").metacell_id.astype(str)
    metacells = cell_to_mc.reindex(adata.obs_names.astype(str)).to_numpy()
    primary_assignment = assignments.sort_values(
        ["membership", "root_id"], ascending=[False, True], kind="mergesort"
    ).drop_duplicates("unit_id")
    mc_to_root = primary_assignment.set_index("unit_id").root_id.astype(str).to_dict()
    roots = np.asarray([mc_to_root.get(str(unit), "none") for unit in metacells])
    source_positions = np.flatnonzero(roots == str(relation.source_root))
    receiver_positions = np.flatnonzero(roots == str(relation.receiver_root))
    if len(source_positions) < 10 or len(receiver_positions) < 10:
        return {}
    source_cut = float(np.quantile(source_score[source_positions], 0.60))
    selected_source = source_positions[source_score[source_positions] >= source_cut]
    coords = adata.obs[["spatial_x", "spatial_y"]].to_numpy(float)
    distance = np.sqrt(
        ((coords[receiver_positions, None, :] - coords[selected_source][None, :, :]) ** 2).sum(axis=2)
    )
    nearest = distance.min(axis=1)
    radius = max(float(np.quantile(nearest, 0.35)), EPS)
    exposure = np.exp(-np.square(nearest / radius))
    exposure_rank = pd.Series(exposure).rank(method="average", pct=True).to_numpy(float)
    local_response = receiver_score[receiver_positions]
    response_rank = pd.Series(local_response).rank(method="average", pct=True).to_numpy(float)
    if str(relation.event_type) == "inhibition":
        response_rank = 1.0 - response_rank
    support_cell = exposure_rank * response_rank
    active = (exposure_rank >= 0.50) & (response_rank >= 0.80)
    frame = pd.DataFrame({
        "metacell_id": metacells[receiver_positions].astype(str),
        "support": support_cell,
        "active": active.astype(float),
    })
    grouped = frame.groupby("metacell_id").agg(
        mean_support=("support", "mean"), active_fraction=("active", "mean")
    )
    grouped["topology_support"] = grouped.mean_support * (
        0.25 + 0.75 * grouped.active_fraction
    )
    positive = grouped[grouped.active_fraction.gt(0)].topology_support
    if positive.empty or float(positive.max()) <= 0:
        return {}
    return (positive / float(positive.max())).astype(float).to_dict()


def relation_support(
    relation, mapping: pd.DataFrame, node_names: set[str]
) -> dict[str, float]:
    cells = parse_tokens(relation.localized_support_cell_ids)
    units = parse_tokens(relation.localized_support_unit_ids) & node_names
    local = mapping[mapping.cell_name.astype(str).isin(cells)].copy()
    local = local[local.metacell_id.astype(str).isin(node_names)]
    total = mapping.groupby(mapping.metacell_id.astype(str)).size().astype(float)
    hit = local.groupby(local.metacell_id.astype(str)).size().astype(float)
    fraction = (hit / total.reindex(hit.index)).fillna(0.0).to_dict()
    for unit in units:
        fraction.setdefault(unit, 1.0 / max(float(total.get(unit, 1.0)), 1.0))
    maximum = max(fraction.values(), default=0.0)
    return {
        str(unit): float(value / maximum)
        for unit, value in fraction.items() if maximum > 0 and value > 0
    }


def refine_relation(
    t0: pd.DataFrame,
    nodes: pd.DataFrame,
    candidate: pd.DataFrame,
    mapping: pd.DataFrame,
    relation,
    topology_support_by_name: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    root_id = str(relation.receiver_root)
    part = t0[t0.root_id.astype(str).eq(root_id)].copy()
    node_part = nodes[nodes.root_id.astype(str).eq(root_id)].copy()
    tau = dict(zip(node_part.node_id.astype(int), node_part.pseudotime.astype(float)))
    names = dict(zip(node_part.node_id.astype(int), node_part.unit_id.astype(str)))
    ids = {name: node for node, name in names.items()}
    branches = dict(zip(node_part.node_id.astype(int), node_part.branch_id.astype(str)))
    root_rows = node_part[node_part.is_root.map(as_bool)]
    if len(root_rows) != 1:
        raise ValueError(f"Expected one anchor in inferred receiver {root_id}")
    anchor = int(root_rows.iloc[0].node_id)
    graph = nx.Graph()
    graph.add_edges_from(part[["source", "target"]].astype(int).itertuples(index=False, name=None))
    if not nx.is_tree(graph) or anchor not in graph:
        raise ValueError(f"Frozen/current tree invalid for {root_id}")
    paths = nx.single_source_shortest_path(graph, anchor)
    parent = {node: path[-2] for node, path in paths.items() if len(path) > 1}
    support_by_name = relation_support(relation, mapping, set(names.values()))
    evidence_support_units = set(support_by_name)
    for name, value in topology_support_by_name.items():
        if name in names.values():
            support_by_name[name] = max(support_by_name.get(name, 0.0), float(value))
    start = float(relation.receiver_window_start)
    end = float(relation.receiver_window_end)
    support_nodes = {
        ids[name]: value for name, value in support_by_name.items()
        if name in ids and ids[name] != anchor
    }
    if not support_nodes:
        return t0.copy(), pd.DataFrame(), {
            "accepted": False,
            "reason": "no_localized_receiver_nodes",
            "n_selected_swaps": 0,
            "upstream_violation": False,
        }
    relative_tau = pd.Series({node: tau[node] for node in support_nodes})
    onset = (
        1.0 - relative_tau.rank(method="average", pct=True)
        + 1.0 / len(relative_tau)
    ).clip(0.0, 1.0).to_dict()
    support = {
        node: float(value * (0.25 + 0.75 * onset[node]))
        for node, value in support_nodes.items()
    }
    maximum = max(support.values())
    support = {node: value / maximum for node, value in support.items()}
    costs = candidate_cost_lookup(candidate)
    receiver_nodes = set(names)
    candidate_pairs = {
        edge for edge in costs if edge[0] in receiver_nodes and edge[1] in receiver_nodes
    }
    trials: list[dict[str, object]] = []
    for child in sorted(support):
        if child not in parent:
            continue
        old_parent = parent[child]
        old_key = edge_key(old_parent, child)
        old_cost = costs.get(old_key)
        if old_cost is None:
            old_cost = float(part[
                ((part.source.astype(int) == old_parent) & (part.target.astype(int) == child))
                | ((part.source.astype(int) == child) & (part.target.astype(int) == old_parent))
            ].edge_cost.iloc[0])
        without = graph.copy()
        without.remove_edge(old_parent, child)
        subtree = nx.node_connected_component(without, child)
        downstream_support_mass = float(sum(
            support.get(node, 0.0) for node in subtree
        ))
        total_support_mass = float(sum(support.values()))
        downstream_support_fraction = (
            downstream_support_mass / max(total_support_mass, EPS)
        )
        downstream_high_support_count = int(len(set(subtree) & set(support)))
        downstream_node_fraction = float(len(subtree) / max(len(receiver_nodes), 1))
        downstream_support_enrichment = float(
            downstream_support_fraction / max(downstream_node_fraction, EPS)
        )
        for new_parent in sorted(receiver_nodes):
            key = edge_key(new_parent, child)
            if key not in candidate_pairs or new_parent in subtree or new_parent == old_parent:
                continue
            if tau.get(new_parent, 1.0) >= tau.get(child, 0.0):
                continue
            parent_affected = float(
                branches.get(new_parent) == str(relation.affected_branch)
            )
            score, upstream_shift, branch_escape = alignment(
                str(relation.event_type), support[child], tau[old_parent], tau[new_parent],
                parent_affected,
                branches.get(child) != branches.get(new_parent),
            )
            if score <= 0:
                continue
            new_cost = costs[key]
            modified_new = new_cost * (1.0 - GAMMA * score)
            modified_old = old_cost * (1.0 + GAMMA * support[child])
            trials.append({
                "source_root": str(relation.source_root),
                "receiver_root": root_id,
                "event_type": str(relation.event_type),
                "child": child,
                "child_name": names[child],
                "old_parent": old_parent,
                "old_parent_name": names[old_parent],
                "new_parent": new_parent,
                "new_parent_name": names[new_parent],
                "child_tau": tau[child],
                "old_parent_tau": tau[old_parent],
                "new_parent_tau": tau[new_parent],
                "event_support": support[child],
                "downstream_support_mass": downstream_support_mass,
                "downstream_support_fraction": downstream_support_fraction,
                "downstream_support_node_count": downstream_high_support_count,
                "downstream_node_fraction": downstream_node_fraction,
                "downstream_support_enrichment": downstream_support_enrichment,
                "event_onset_localization": onset[child],
                "alignment": score,
                "upstream_shift": upstream_shift,
                "branch_escape": branch_escape,
                "base_old_cost": old_cost,
                "base_new_cost": new_cost,
                "modified_old_cost": modified_old,
                "modified_new_cost": modified_new,
                "cost_advantage": modified_old - modified_new,
                "candidate_legal": True,
                "truth_used_for_selection": False,
            })
    trial_frame = pd.DataFrame(trials)
    if trial_frame.empty:
        return t0.copy(), trial_frame, {
            "accepted": False,
            "reason": "no_candidate_legal_alternative",
            "n_selected_swaps": 0,
            "upstream_violation": False,
        }
    support_gate = float(np.median(list(support.values())))
    high_support_nodes = {
        node for node, value in support.items() if value >= support_gate
    }
    boundary_children = {
        node for node in high_support_nodes
        if node in parent and parent[node] not in high_support_nodes
    }
    if not boundary_children and high_support_nodes:
        boundary_children = {
            min(high_support_nodes, key=lambda node: (tau[node], node))
        }
    accepted = trial_frame[
        trial_frame.cost_advantage.gt(0)
        & trial_frame.event_support.ge(support_gate)
        & trial_frame.child.astype(int).isin(boundary_children)
        & trial_frame.downstream_support_fraction.ge(
            MIN_DOWNSTREAM_SUPPORT_FRACTION
        )
    ].copy()
    if accepted.empty:
        return t0.copy(), trial_frame, {
            "accepted": False,
            "reason": "significant_but_no_structurally_supported_cost_advantage",
            "n_selected_swaps": 0,
            "upstream_violation": False,
        }
    ordered = accepted.sort_values(
        [
            "downstream_support_fraction", "alignment", "cost_advantage",
            "base_new_cost", "child", "new_parent",
        ],
        ascending=[False, False, False, True, True, True], kind="mergesort",
    )
    refined = t0.copy()
    current = graph.copy()
    selected_rows: list[tuple[int, pd.Series]] = []
    changed_children: set[int] = set()
    for index, row in ordered.iterrows():
        child, old_parent, new_parent = int(row.child), int(row.old_parent), int(row.new_parent)
        if child in changed_children or not current.has_edge(old_parent, child):
            continue
        without = current.copy()
        without.remove_edge(old_parent, child)
        if new_parent in nx.node_connected_component(without, child):
            continue
        proposed = without.copy()
        proposed.add_edge(new_parent, child)
        if not nx.is_tree(proposed):
            continue
        refined = replace_edge(
            refined, root_id, old_parent, child, new_parent,
            names, float(row.base_new_cost),
        )
        current = proposed
        changed_children.add(child)
        selected_rows.append((index, row))
        if len(selected_rows) >= MAX_BOUNDARY_SWAPS_PER_EVENT:
            break
    if not selected_rows:
        return t0.copy(), trial_frame, {
            "accepted": False,
            "reason": "significant_but_sequential_legality_abstention",
            "n_selected_swaps": 0,
            "upstream_violation": False,
        }
    trial_frame["selected"] = False
    trial_frame.loc[[index for index, _ in selected_rows], "selected"] = True
    selected = [row for _, row in selected_rows]
    return refined, trial_frame, {
        "accepted": True,
        "reason": "inferred_event_boundary_positive_cost_advantage",
        "n_selected_swaps": len(selected),
        "support_gate": support_gate,
        "n_boundary_children": len(boundary_children),
        "n_significance_localized_units": len(evidence_support_units),
        "n_topology_localized_units": len(topology_support_by_name),
        "stale_tau_window_crossed": bool(any(
            float(row.child_tau) < start - 0.05 for row in selected
        )),
        "upstream_violation": bool(any(
            int(row.child) not in boundary_children for row in selected
        )),
        "selected_child": ";".join(str(row.child_name) for row in selected),
        "selected_old_parent": ";".join(str(row.old_parent_name) for row in selected),
        "selected_new_parent": ";".join(str(row.new_parent_name) for row in selected),
    }


def prune_unanchored_components(
    edges: pd.DataFrame, nodes: pd.DataFrame, protected_names: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop only root-unreachable components that carry no inferred event boundary."""
    kept_frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for root_id, node_part in nodes.groupby("root_id", sort=True):
        part = edges[edges.root_id.astype(str).eq(str(root_id))].copy()
        graph = nx.Graph()
        graph.add_nodes_from(node_part.node_id.astype(int))
        graph.add_edges_from(part[["source", "target"]].astype(int).itertuples(index=False, name=None))
        root_rows = node_part[node_part.is_root.map(as_bool)]
        if len(root_rows) != 1:
            raise ValueError(f"Expected one pruning anchor for {root_id}")
        anchor = int(root_rows.iloc[0].node_id)
        names = dict(zip(node_part.node_id.astype(int), node_part.unit_id.astype(str)))
        anchor_component = set(nx.node_connected_component(graph, anchor))
        detached = set(graph.nodes) - anchor_component
        protected_detached = {
            node for node in detached if names.get(node) in protected_names
        }
        if protected_detached:
            raise ValueError(
                f"Event boundary detached from root in {root_id}: {sorted(protected_detached)}"
            )
        keep = part[
            part.source.astype(int).isin(anchor_component)
            & part.target.astype(int).isin(anchor_component)
        ].copy()
        kept_frames.append(keep)
        audits.append({
            "root_id": str(root_id),
            "n_nodes_before": int(len(node_part)),
            "n_nodes_retained": int(len(anchor_component)),
            "n_unanchored_nodes_pruned": int(len(detached)),
            "n_edges_before": int(len(part)),
            "n_edges_retained": int(len(keep)),
            "event_boundary_pruned": False,
            "pruning_rule": "remove_unanchored_non_event_components_only",
        })
    return pd.concat(kept_frames, ignore_index=True), pd.DataFrame(audits)


def recompute_pseudotime(
    edges: pd.DataFrame, nodes: pd.DataFrame, drop_unreachable: bool
) -> pd.DataFrame:
    updated: list[pd.DataFrame] = []
    for root_id, node_part in nodes.groupby("root_id", sort=True):
        part = edges[edges.root_id.astype(str).eq(str(root_id))]
        graph = nx.Graph()
        graph.add_nodes_from(node_part.node_id.astype(int))
        for row in part.itertuples(index=False):
            graph.add_edge(
                int(row.source), int(row.target), weight=max(float(row.edge_cost), EPS)
            )
        root_rows = node_part[node_part.is_root.map(as_bool)]
        if len(root_rows) != 1:
            raise ValueError(f"Expected one pseudotime anchor for {root_id}")
        anchor = int(root_rows.iloc[0].node_id)
        reachable = set(nx.node_connected_component(graph, anchor))
        distances = nx.single_source_dijkstra_path_length(graph, anchor, weight="weight")
        scale = max(distances.values(), default=1.0)
        local = node_part.copy()
        local["root_reachable"] = local.node_id.astype(int).isin(reachable)
        local["pseudotime"] = [
            float(distances[int(node)] / max(scale, EPS))
            if int(node) in distances else np.nan
            for node in local.node_id
        ]
        if drop_unreachable:
            local = local[local.root_reachable].copy()
        updated.append(local)
    return pd.concat(updated, ignore_index=True)


def topology_metrics(
    before: pd.DataFrame,
    after: pd.DataFrame,
    state_lookup: dict[tuple[str, str], str],
    truth_t0: set[tuple[str, str, str]],
    truth_t1: set[tuple[str, str, str]],
) -> dict[str, float]:
    mapped_before = mapped_edge_set(before, state_lookup)
    mapped_after = mapped_edge_set(after, state_lookup)
    true_removed, true_added = truth_t0 - truth_t1, truth_t1 - truth_t0
    pred_removed, pred_added = mapped_before - mapped_after, mapped_after - mapped_before
    true_rewire = {("removed", edge) for edge in true_removed} | {
        ("added", edge) for edge in true_added
    }
    pred_rewire = {("removed", edge) for edge in pred_removed} | {
        ("added", edge) for edge in pred_added
    }
    precision, recall, rewire_f1 = prf(true_rewire, pred_rewire)
    before_distance = len(mapped_before ^ truth_t1)
    after_distance = len(mapped_after ^ truth_t1)
    return {
        "topology_distance_before": before_distance,
        "topology_distance_after": after_distance,
        "inferred_topology_gain": before_distance - after_distance,
        "rewiring_precision": precision,
        "rewiring_recall": recall,
        "rewiring_F1": rewire_f1,
        "n_predicted_removed_edges": len(pred_removed),
        "n_predicted_added_edges": len(pred_added),
    }


def legality(edges: pd.DataFrame) -> tuple[bool, bool, bool]:
    legal = bool(edges.candidate_legal.map(as_bool).all())
    connected = True
    acyclic = True
    for _, part in edges.groupby("root_id"):
        graph = nx.Graph(part[["source", "target"]].astype(int).itertuples(index=False, name=None))
        connected &= nx.is_connected(graph)
        acyclic &= nx.is_tree(graph)
    return legal, connected, acyclic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase-a-dir", type=Path, required=True)
    parser.add_argument("--relation-dir", type=Path, required=True)
    parser.add_argument("--program-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    t0 = pd.read_csv(args.phase_a_dir / "T0_FINE_TRAJECTORY_EDGES.csv")
    nodes = pd.read_csv(args.phase_a_dir / "T0_STATE_NODE_ASSIGNMENTS.csv")
    candidate = pd.read_csv(args.phase_a_dir / "C1_MULTIRESOLUTION_CANDIDATE_EDGES.csv")
    mapping = pd.read_csv(args.run_dir / "formal_metacell" / "cell_to_metacell_mapping.csv")
    evidence = pd.read_csv(args.relation_dir / "PHASE_B_CMI_E_EVIDENCE.csv")
    calls = evidence[evidence.is_called.map(as_bool)].sort_values(
        ["call_strength", "source_root", "receiver_root"],
        ascending=[False, True, True], kind="mergesort",
    )
    current = t0.copy()
    audits: list[dict[str, object]] = []
    trials: list[pd.DataFrame] = []
    for relation in calls.itertuples(index=False):
        topology_support = topology_localization_support(
            args.run_dir, relation, mapping, nodes, args.program_catalog
        )
        refined, trial, audit = refine_relation(
            current, nodes, candidate, mapping, relation, topology_support
        )
        audit.update({
            "source_root": str(relation.source_root),
            "receiver_root": str(relation.receiver_root),
            "event_type": str(relation.event_type),
            "primary_channel_q": float(relation.primary_channel_q),
            "call_strength": float(relation.call_strength),
        })
        audits.append(audit)
        if not trial.empty:
            trials.append(trial)
        current = refined
    protected_names = set()
    for audit in audits:
        if audit.get("accepted"):
            protected_names.update(parse_tokens(audit.get("selected_child", "")))
            protected_names.update(parse_tokens(audit.get("selected_new_parent", "")))
    final, pruning_audit = prune_unanchored_components(
        current, nodes, protected_names
    )
    tau1_nodes = recompute_pseudotime(current, nodes, drop_unreachable=False)
    final_nodes = recompute_pseudotime(final, nodes, drop_unreachable=True)
    state = pd.read_csv(args.run_dir / "formal_metacell" / "METACELL_STATE_COMPOSITION.csv")
    state_lookup = dominant_state_lookup(state, "T1")
    truth_t0 = truth_edge_set(pd.read_csv(args.run_dir / "event" / "truth" / "TRUE_T0_EDGES.csv"))
    truth_t1 = truth_edge_set(pd.read_csv(args.run_dir / "event" / "truth" / "TRUE_T1_EDGES.csv"))
    metrics = topology_metrics(t0, current, state_lookup, truth_t0, truth_t1)
    final_metrics = topology_metrics(t0, final, state_lookup, truth_t0, truth_t1)
    legal, connected, acyclic = legality(current)
    final_legal, final_connected, final_acyclic = legality(final)
    before_pairs = {
        (str(row.root_id), edge_key(row.source, row.target))
        for row in t0.itertuples(index=False)
    }
    full_pairs = {
        (str(row.root_id), edge_key(row.source, row.target))
        for row in current.itertuples(index=False)
    }
    final_pairs = {
        (str(row.root_id), edge_key(row.source, row.target))
        for row in final.itertuples(index=False)
    }
    event_added = full_pairs - before_pairs
    retained_event_added = event_added & final_pairs
    causal_truth = pd.read_csv(args.run_dir / "event" / "truth" / "TRUE_EVENTS.csv")
    causal = bool(causal_truth.causal.map(as_bool).any())
    result = {
        "dataset": args.dataset,
        "n_relation_calls": int(len(calls)),
        "n_accepted_relation_events": int(sum(bool(row.get("accepted")) for row in audits)),
        "significant_but_no_rewire": int(sum(not bool(row.get("accepted")) for row in audits)),
        "causal_scenario": causal,
        **metrics,
        "candidate_legal": legal,
        "connected": connected,
        "acyclic": acyclic,
        "upstream_violation": bool(any(row.get("upstream_violation", False) for row in audits)),
        "tfinal_topology_distance": final_metrics["topology_distance_after"],
        "tfinal_topology_gain": final_metrics["inferred_topology_gain"],
        "tfinal_candidate_legal": final_legal,
        "tfinal_connected": final_connected,
        "tfinal_acyclic": final_acyclic,
        "event_added_edges": len(event_added),
        "event_added_edges_retained_after_pruning": len(retained_event_added),
        "event_boundary_retained": len(event_added) == len(retained_event_added),
        "n_unanchored_nodes_pruned": int(pruning_audit.n_unanchored_nodes_pruned.sum()),
        "truth_used_for_selection": False,
        "truth_used_for_scoring_only": True,
    }
    t0.to_csv(args.output / "M4_T0_RELATION_INFERENCE_NO_TOPOLOGY_CHANGE.csv", index=False)
    current.to_csv(args.output / "M5_T1_FULL_INFERRED_RELATION_REFINED.csv", index=False)
    tau1_nodes.to_csv(args.output / "M5_TAU1_NODE_ASSIGNMENTS.csv", index=False)
    final.to_csv(args.output / "M6_TFINAL_PRUNED.csv", index=False)
    final_nodes.to_csv(args.output / "M6_TFINAL_NODE_ASSIGNMENTS.csv", index=False)
    pruning_audit.to_csv(args.output / "PHASE_D_PRUNING_RETENTION.csv", index=False)
    pd.DataFrame(audits).to_csv(args.output / "PHASE_D_EVENT_REFINEMENT_AUDIT.csv", index=False)
    if trials:
        pd.concat(trials, ignore_index=True).to_csv(
            args.output / "PHASE_D_CANDIDATE_SWAP_TRIALS.csv", index=False
        )
    pd.DataFrame([result]).to_csv(args.output / "PHASE_D_INFERRED_REFINEMENT_METRICS.csv", index=False)
    status = {
        "phase": "D",
        "operator_version": "INFERRED_REFINEMENT_V5_SUPPORTED_PROGRAM_BOUNDARY",
        "pruning_version": "ROOT_CONNECTED_EVENT_PROTECTED_V1",
        "relation_detector": "MULTICHANNEL_RELATION_V3_8",
        "gamma": GAMMA,
        "max_boundary_swaps_per_event": MAX_BOUNDARY_SWAPS_PER_EVENT,
        "minimum_downstream_support_fraction": MIN_DOWNSTREAM_SUPPORT_FRACTION,
        "frozen_phase_a_modified": False,
        "relation_significance_recomputed_after_T1": False,
        "truth_used_for_selection": False,
        "tau1_defined_only_from_t1": True,
        "complete": True,
    }
    (args.output / "RUN_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
