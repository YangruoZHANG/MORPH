#!/usr/bin/env python3
"""PHASE C v6: structural oracle refinement with transition evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd


GAMMAS = (0.10, 0.20, 0.25, 0.35, 0.50)
STATE_SUPPORT = 0.20
PROGRAM_HIGH_QUANTILE = 0.75
MIN_EVENT_SUPPORT_FOR_REWIRE = 0.20
MIN_PROMOTION_TRANSITION_GAIN = 0.20
COLLISION_TOPOLOGY_REFINEMENT_ENABLED = False
INHIBITION_TOPOLOGY_REFINEMENT_ENABLED = False
EPS = 1e-12


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def pair(left: object, right: object) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def edge_key(left: int, right: int) -> tuple[int, int]:
    return tuple(sorted((int(left), int(right))))


def prf(truth: set, predicted: set) -> tuple[float, float, float]:
    overlap = len(truth & predicted)
    precision = overlap / len(predicted) if predicted else (1.0 if not truth else 0.0)
    recall = overlap / len(truth) if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def truth_edge_set(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    return {
        (str(row.root_id), *pair(row.source, row.target))
        for row in frame.itertuples(index=False)
    }


def dominant_state_lookup(state: pd.DataFrame, stage: str) -> dict[tuple[str, str], str]:
    local = state[state.stage.astype(str).eq(stage)].copy()
    local = local.sort_values(
        ["metacell_id", "root_id", "state_probability", "state_id"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["metacell_id", "root_id"])
    return {
        (str(row.metacell_id), str(row.root_id)): str(row.state_id)
        for row in local.itertuples(index=False)
    }


def mapped_edge_set(
    edges: pd.DataFrame, state_lookup: dict[tuple[str, str], str]
) -> set[tuple[str, str, str]]:
    result = set()
    for row in edges.itertuples(index=False):
        root = str(row.root_id)
        source = state_lookup.get((str(row.source_name), root))
        target = state_lookup.get((str(row.target_name), root))
        if source and target and source != target:
            result.add((root, *pair(source, target)))
    return result


def truth_graphs(edges: set[tuple[str, str, str]]) -> dict[str, nx.Graph]:
    graphs: dict[str, nx.Graph] = {}
    for root, left, right in edges:
        graphs.setdefault(str(root), nx.Graph()).add_edge(str(left), str(right))
    return graphs


def endpoint_error(
    predicted: tuple[str, str], target: tuple[str, str], graph: nx.Graph
) -> float:
    def distance(left: str, right: str) -> float:
        if left not in graph or right not in graph:
            return float(graph.number_of_nodes())
        return float(nx.shortest_path_length(graph, left, right))
    return min(
        distance(predicted[0], target[0]) + distance(predicted[1], target[1]),
        distance(predicted[0], target[1]) + distance(predicted[1], target[0]),
    )


def formal_topology_distance(
    predicted: set[tuple[str, str, str]], truth: set[tuple[str, str, str]]
) -> float:
    """Symmetric edge-location distance on each truth-tree geodesic."""
    graphs = truth_graphs(truth)
    values = []
    for root, graph in graphs.items():
        predicted_edges = [(left, right) for r, left, right in predicted if r == root]
        truth_edges = [(left, right) for r, left, right in truth if r == root]
        if not predicted_edges:
            values.append(1.0)
            continue
        scale = max(nx.diameter(graph), 1)
        pred_to_truth = np.mean([
            min(endpoint_error(edge, target, graph) for target in truth_edges)
            for edge in predicted_edges
        ])
        truth_to_pred = np.mean([
            min(endpoint_error(edge, target, graph) for target in predicted_edges)
            for edge in truth_edges
        ])
        values.append(0.5 * (pred_to_truth + truth_to_pred) / scale)
    return float(np.mean(values)) if values else float("inf")


def event_location_metrics(
    event, mapped_before: set, mapped_after: set,
    truth_t0: set, truth_t1: set, audit: dict,
    state_lookup: dict[tuple[str, str], str],
) -> dict[str, float]:
    root = str(event.receiver_root)
    predicted_removed: list[tuple[str, str]] = []
    predicted_added: list[tuple[str, str]] = []
    if audit.get("accepted"):
        children = str(audit.get("selected_child", "")).split(";")
        old_parents = str(audit.get("selected_old_parent", "")).split(";")
        new_parents = str(audit.get("selected_new_parent", "")).split(";")
        for child, old_parent, new_parent in zip(children, old_parents, new_parents):
            child_state = state_lookup.get((child, root))
            old_state = state_lookup.get((old_parent, root))
            new_state = state_lookup.get((new_parent, root))
            if child_state and old_state:
                predicted_removed.append((old_state, child_state))
            if child_state and new_state:
                predicted_added.append((new_state, child_state))
    if not predicted_removed:
        predicted_removed = [
            (left, right) for r, left, right in mapped_before - mapped_after if r == root
        ]
    if not predicted_added:
        predicted_added = [
            (left, right) for r, left, right in mapped_after - mapped_before if r == root
        ]
    true_added = [(left, right) for r, left, right in truth_t1 - truth_t0 if r == root]
    graph = truth_graphs(truth_t1).get(root, nx.Graph())
    before_error = min(
        (endpoint_error(edge, target, graph)
         for edge in predicted_removed for target in true_added), default=float("nan")
    )
    after_error = min(
        (endpoint_error(edge, target, graph)
         for edge in predicted_added for target in true_added), default=float("nan")
    )
    gain = (
        before_error - after_error
        if np.isfinite(before_error) and np.isfinite(after_error) else 0.0
    )
    return {
        "event_location_error_before": float(before_error),
        "event_location_error_after": float(after_error),
        "event_location_gain": float(gain),
    }


def event_exposure_by_metacell(run_dir: Path, event_id: str) -> dict[str, float]:
    mapping = pd.read_csv(run_dir / "formal_metacell" / "cell_to_metacell_mapping.csv")
    contact = pd.read_csv(run_dir / "event" / "truth" / "TRUE_SPATIAL_CONTACTS.csv")
    contact = contact[contact.event_id.astype(str).eq(str(event_id))].copy()
    if not len(contact):
        return {}
    exposed = contact[["receiver_cell_id", "is_event_exposed"]].drop_duplicates(
        "receiver_cell_id"
    ).rename(columns={"receiver_cell_id": "cell_name"})
    exposed["is_event_exposed"] = exposed.is_event_exposed.map(as_bool).astype(float)
    joined = mapping[["cell_name", "metacell_id"]].merge(
        exposed, on="cell_name", how="left"
    )
    joined["is_event_exposed"] = joined.is_event_exposed.fillna(0.0)
    return joined.groupby("metacell_id").is_event_exposed.mean().astype(float).to_dict()


def program_genes(value: object) -> list[str]:
    text = str(value)
    match = re.fullmatch(r"(G\d+)-(G\d+)", text)
    if match:
        start, end = int(match.group(1)[1:]), int(match.group(2)[1:])
        return [f"G{index}" for index in range(start, end + 1)]
    return [token for token in text.split(";") if token]


def receiver_response_residual(
    run_dir: Path, dataset: str, event, node_part: pd.DataFrame,
    exposure: dict[str, float],
) -> dict[str, float]:
    """Residualize the oracle receiver program against non-event context."""
    meta = ad.read_h5ad(
        run_dir / "generic_upstream_v1_exact_root" / dataset / "metacell_adata.h5ad"
    )
    genes = program_genes(event.receiver_program_genes)
    indices = [meta.var_names.get_loc(gene) for gene in genes if gene in meta.var_names]
    if not indices:
        return {}
    matrix = meta.X.toarray() if hasattr(meta.X, "toarray") else np.asarray(meta.X)
    score = np.log1p(np.maximum(matrix[:, indices], 0.0)).mean(axis=1)
    score_lookup = dict(zip(meta.obs_names.astype(str), score.astype(float)))
    tau_lookup = dict(zip(node_part.unit_id.astype(str), node_part.pseudotime.astype(float)))
    names = [name for name in node_part.unit_id.astype(str) if name in score_lookup]
    if len(names) < 6:
        return {name: 0.0 for name in names}
    tau = np.array([tau_lookup[name] for name in names], dtype=float)
    spatial = meta.obs.loc[names, ["spatial_x", "spatial_y"]].to_numpy(float)
    for column in range(spatial.shape[1]):
        scale = np.std(spatial[:, column])
        spatial[:, column] = (
            (spatial[:, column] - np.mean(spatial[:, column])) / max(scale, EPS)
        )
    design = np.column_stack([np.ones(len(names)), tau, tau * tau, spatial])
    outcome = np.array([score_lookup[name] for name in names], dtype=float)
    control = np.array([exposure.get(name, 0.0) <= 0.05 for name in names])
    if control.sum() < design.shape[1] + 2:
        control = np.ones(len(names), dtype=bool)
    coefficients = np.linalg.lstsq(design[control], outcome[control], rcond=None)[0]
    residual = outcome - design @ coefficients
    if str(event.event_type) == "inhibition":
        residual = -residual
    ranks = pd.Series(residual).rank(method="average", pct=True).to_numpy(float)
    return dict(zip(names, ranks))


def transition_program_rank(
    run_dir: Path, dataset: str, node_part: pd.DataFrame,
    branch_probability: dict[str, float], exposure: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    """Infer a receiver-local program that is shared with a candidate parent."""
    meta = ad.read_h5ad(
        run_dir / "generic_upstream_v1_exact_root" / dataset / "metacell_adata.h5ad"
    )
    names = [name for name in node_part.unit_id.astype(str) if name in meta.obs_names]
    if len(names) < 8:
        return {name: 0.5 for name in names}, []
    positions = np.asarray([meta.obs_names.get_loc(name) for name in names], dtype=int)
    matrix = meta.X.toarray() if hasattr(meta.X, "toarray") else np.asarray(meta.X)
    matrix = np.log1p(np.maximum(matrix[positions], 0.0))
    scale = np.std(matrix, axis=0)
    standardized = (matrix - np.mean(matrix, axis=0)) / np.where(scale > EPS, scale, 1.0)
    seed = np.asarray([
        exposure.get(name, 0.0) > 0.0
        and branch_probability.get(name, 0.0) >= STATE_SUPPORT
        for name in names
    ])
    context = ~seed
    if seed.sum() < 2 or context.sum() < 3:
        return {name: 0.5 for name in names}, []
    baseline = np.median(standardized, axis=0)
    seed_enrichment = standardized[seed].mean(axis=0) - baseline
    context_enrichment = np.quantile(standardized[context], 0.80, axis=0) - baseline
    # A topology-transition gene must be present in the event seed and in a
    # distinct receiver context subgroup. This excludes response-only genes
    # while retaining a shared parent-state program.
    effect = np.minimum(seed_enrichment, context_enrichment)
    k = max(5, int(np.ceil(np.sqrt(standardized.shape[1]))))
    selected = np.argsort(effect)[::-1]
    selected = selected[effect[selected] > 0][:k]
    if not len(selected):
        return {name: 0.5 for name in names}, []
    score = standardized[:, selected].mean(axis=1)
    ranks = pd.Series(score).rank(method="average", pct=True).to_numpy(float)
    genes = [str(meta.var_names[index]) for index in selected]
    return dict(zip(names, ranks)), genes


def candidate_cost_lookup(candidate: pd.DataFrame) -> dict[tuple[int, int], float]:
    eligible = candidate[candidate.trajectory_edge_eligible.astype(bool)].copy()
    eligible["key"] = [edge_key(row.source, row.target) for row in eligible.itertuples()]
    return eligible.groupby("key").D_joint.min().astype(float).to_dict()


def candidate_feature_lookup(
    candidate: pd.DataFrame, node_ids: set[int]
) -> dict[tuple[int, int], dict[str, float]]:
    """Build comparable root-local base and transition-evidence costs."""
    eligible = candidate[
        candidate.trajectory_edge_eligible.astype(bool)
        & candidate.source.astype(int).isin(node_ids)
        & candidate.target.astype(int).isin(node_ids)
    ].copy()
    eligible["key"] = [edge_key(row.source, row.target) for row in eligible.itertuples()]
    eligible = eligible.sort_values(
        ["D_joint", "D_expression", "source", "target"], kind="mergesort"
    ).drop_duplicates("key")
    eligible["joint_rank"] = eligible.D_joint.rank(method="average", pct=True)
    eligible["expression_rank"] = eligible.D_expression.rank(method="average", pct=True)
    eligible["event_transition_cost"] = (
        0.50 * eligible.joint_rank + 0.50 * eligible.expression_rank
    )
    return {
        row.key: {
            "base_cost": float(row.joint_rank),
            "expression_rank": float(row.expression_rank),
            "event_cost": float(row.event_transition_cost),
            "raw_joint": float(row.D_joint),
        }
        for row in eligible.itertuples(index=False)
    }


def replace_edge(
    edges: pd.DataFrame, root_id: str, old_parent: int, child: int,
    new_parent: int, names: dict[int, str], new_cost: float,
) -> pd.DataFrame:
    local = edges.root_id.astype(str).eq(root_id)
    remove = local & (
        ((edges.source.astype(int) == old_parent) & (edges.target.astype(int) == child))
        | ((edges.source.astype(int) == child) & (edges.target.astype(int) == old_parent))
    )
    row = edges.loc[local].iloc[0].copy()
    row["source"], row["target"] = int(new_parent), int(child)
    row["source_name"], row["target_name"] = names[new_parent], names[child]
    row["edge_cost"] = float(new_cost)
    row["edge_graph_role"] = "oracle_event_conditioned_candidate_edge"
    row["candidate_legal"] = True
    row["connector_only"] = False
    return pd.concat([edges.loc[~remove], pd.DataFrame([row])], ignore_index=True)


def alignment(
    event_type: str, support: float, old_tau: float, new_tau: float,
    parent_affected_probability: float, program_change: bool,
) -> tuple[float, float, float]:
    upstream_shift = float(np.clip(old_tau - new_tau, 0.0, 1.0))
    tau_compatibility = float(np.clip(1.0 - abs(old_tau - new_tau), 0.0, 1.0))
    branch_escape = float(np.clip(1.0 - parent_affected_probability, 0.0, 1.0))
    if event_type == "promotion":
        parent_earliness = float(np.clip(1.0 - new_tau, 0.0, 1.0))
        value = support * (0.25 + 0.75 * parent_earliness)
    elif event_type == "inhibition":
        # Inhibition represents branch escape, not a generic jump toward the
        # root.  Preserve the old edge's temporal scale while preferring a
        # parent outside the affected program.
        value = support * (0.50 + 0.50 * branch_escape) * (
            0.25 + 0.75 * tau_compatibility
        )
    elif event_type == "collision_convergence":
        value = (
            support * (0.50 + 0.50 * branch_escape)
            * (0.50 + 0.50 * tau_compatibility)
            if program_change else 0.0
        )
    else:
        value = 0.0
    return float(value), upstream_shift, branch_escape


def refine_event(
    run_dir: Path, dataset: str, t0: pd.DataFrame, nodes: pd.DataFrame, candidate: pd.DataFrame,
    event, gamma: float, branch_composition: pd.DataFrame,
    exposure_override: dict[str, float] | None = None,
    branch_probability_override: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    root_id = str(event.receiver_root)
    if not as_bool(event.causal) or float(event.effect_strength) <= 0:
        return t0.copy(), pd.DataFrame(), {
            "accepted": False, "reason": "oracle_null_abstention",
            "upstream_violation": False,
        }
    part = t0[t0.root_id.astype(str).eq(root_id)].copy()
    node_part = nodes[nodes.root_id.astype(str).eq(root_id)].copy()
    tau = dict(zip(node_part.node_id.astype(int), node_part.pseudotime.astype(float)))
    names = dict(zip(node_part.node_id.astype(int), node_part.unit_id.astype(str)))
    branch_program = dict(zip(
        node_part.node_id.astype(int), node_part.branch_id.astype(str)
    ))
    root_rows = node_part[node_part.is_root.map(as_bool)]
    if len(root_rows) != 1:
        raise ValueError(f"Expected one root node for {root_id}")
    anchor = int(root_rows.iloc[0].node_id)
    graph = nx.Graph()
    graph.add_edges_from(part[["source", "target"]].astype(int).itertuples(index=False, name=None))
    if not nx.is_tree(graph) or anchor not in graph:
        return t0.copy(), pd.DataFrame(), {
            "accepted": False,
            "reason": "frozen_t0_not_root_connected",
            "upstream_violation": False,
            "phase_a_connectivity_failure": True,
        }
    paths = nx.single_source_shortest_path(graph, anchor)
    parent = {node: path[-2] for node, path in paths.items() if len(path) > 1}
    rootward_nodes = {anchor, *map(int, graph.neighbors(anchor))}
    if branch_probability_override is None:
        affected_stage = "T0" if str(event.event_type) == "inhibition" else "T1"
        affected = branch_composition[
            branch_composition.stage.astype(str).eq(affected_stage)
            & branch_composition.root_id.astype(str).eq(root_id)
            & branch_composition.branch_id.astype(str).eq(str(event.affected_branch))
        ]
        branch_probability = dict(zip(
            affected.metacell_id.astype(str), affected.branch_probability.astype(float)
        ))
    else:
        branch_probability = {
            str(name): float(value)
            for name, value in branch_probability_override.items()
        }
    exposure = (
        event_exposure_by_metacell(run_dir, str(event.event_id))
        if exposure_override is None
        else {str(name): float(value) for name, value in exposure_override.items()}
    )
    response = receiver_response_residual(run_dir, dataset, event, node_part, exposure)
    exposure_values = pd.Series({name: exposure.get(name, 0.0) for name in names.values()})
    exposure_rank = exposure_values.rank(method="average", pct=True).to_dict()
    transition_rank, transition_genes = transition_program_rank(
        run_dir, dataset, node_part, branch_probability, exposure
    )
    affected_nodes = {
        node for node, name in names.items()
        if branch_probability.get(name, 0.0) >= STATE_SUPPORT and node != anchor
    }
    relative_tau = pd.Series({node: tau.get(node, 0.0) for node in affected_nodes})
    localization = (
        1.0 - relative_tau.rank(method="average", pct=True)
        + (1.0 / max(len(relative_tau), 1))
    ).clip(0.0, 1.0).to_dict()
    event_seed_nodes = [
        node for node, name in names.items()
        if node in graph
        and exposure.get(name, 0.0) > 0.0
        and branch_probability.get(name, 0.0) >= STATE_SUPPORT
    ]
    seed_signal = {
        node: response.get(names[node], 0.0) * exposure_rank.get(names[node], 0.0)
        for node in event_seed_nodes
    }
    propagated_support = {
        node: max((
            seed_signal[seed] / (1.0 + nx.shortest_path_length(graph, node, seed))
            for seed in event_seed_nodes if nx.has_path(graph, node, seed)
        ), default=0.0)
        for node in affected_nodes
    }
    raw_support = {
        node: propagated_support.get(node, 0.0)
        * (0.50 + 0.50 * branch_probability.get(name, 0.0))
        * (0.25 + 0.75 * localization.get(node, 0.0))
        for node, name in names.items()
        if branch_probability.get(name, 0.0) >= STATE_SUPPORT
        and node != anchor
    }
    maximum = max(raw_support.values(), default=0.0)
    support = {
        node: value / maximum for node, value in raw_support.items()
        if maximum > 0 and value > 0
    }
    features = candidate_feature_lookup(candidate, set(map(int, graph.nodes)))
    candidate_pairs = set(features)
    trials = []
    for child in sorted(support):
        if child == anchor or child not in parent:
            continue
        if str(event.event_type) == "promotion" and child in rootward_nodes:
            continue
        # Oracle event localization is supplied by exposed receiver cells and
        # affected-branch support. The generator's truth tau and inferred tau0
        # are different scales; comparing their raw numeric windows would
        # discard exactly those cells whose transition program shifts tau0.
        old_parent = parent[child]
        old_feature = features.get(edge_key(old_parent, child))
        if old_feature is None:
            raw_old_cost = float(
                part[
                    ((part.source.astype(int) == old_parent) & (part.target.astype(int) == child))
                    | ((part.source.astype(int) == child) & (part.target.astype(int) == old_parent))
                ].edge_cost.iloc[0]
            )
            old_feature = {
                "base_cost": raw_old_cost, "expression_rank": raw_old_cost,
                "event_cost": raw_old_cost, "raw_joint": raw_old_cost,
            }
        without = graph.copy()
        without.remove_edge(old_parent, child)
        subtree = nx.node_connected_component(without, child)
        for new_parent in sorted(graph.nodes):
            key = edge_key(new_parent, child)
            if key not in candidate_pairs or new_parent in subtree or new_parent == old_parent:
                continue
            if (
                str(event.event_type) == "promotion"
                and new_parent not in rootward_nodes
            ):
                continue
            if (
                str(event.event_type) == "promotion"
                and tau.get(new_parent, 1.0) >= tau.get(child, 0.0)
            ):
                continue
            if (
                str(event.event_type) == "inhibition"
                and branch_probability.get(names[new_parent], 0.0) >= STATE_SUPPORT
            ):
                continue
            score, upstream_shift, branch_escape = alignment(
                str(event.event_type), support[child], tau[old_parent], tau[new_parent],
                branch_probability.get(names[new_parent], 0.0),
                branch_program.get(child) != branch_program.get(new_parent),
            )
            parent_response_compatibility = float(response.get(names[new_parent], 0.0))
            new_feature = features[key]
            transition_compatibility = 1.0 - new_feature["expression_rank"]
            parent_transition_rank = transition_rank.get(names[new_parent], 0.0)
            old_parent_transition_rank = transition_rank.get(names[old_parent], 0.0)
            child_transition_rank = transition_rank.get(names[child], 0.0)
            transition_program_gain = (
                parent_transition_rank - old_parent_transition_rank
            )
            child_transition_onset = (
                child_transition_rank - old_parent_transition_rank
            )
            temporal_reversal = float(np.clip(
                tau.get(new_parent, 1.0) - tau.get(child, 0.0), 0.0, 1.0
            ))
            if (
                temporal_reversal > 0
                and str(event.event_type) in {"inhibition", "collision_convergence"}
                and parent_transition_rank < 0.50
            ):
                continue
            score *= (
                0.50 + 0.25 * transition_compatibility
                + 0.25 * parent_transition_rank
            ) * (1.0 - 0.50 * temporal_reversal)
            if score <= 0:
                continue
            old_cost = (
                0.50 * old_feature["event_cost"]
                + 0.50 * (1.0 - old_parent_transition_rank)
            )
            new_cost = (
                0.50 * new_feature["event_cost"]
                + 0.50 * (1.0 - parent_transition_rank)
            )
            modified_new = new_cost * (1.0 - gamma * score)
            modified_old = old_cost * (1.0 + gamma * support[child])
            trials.append({
                "event_id": str(event.event_id),
                "event_type": str(event.event_type), "root_id": root_id,
                "gamma": gamma, "child": child, "child_name": names[child],
                "old_parent": old_parent, "old_parent_name": names[old_parent],
                "new_parent": new_parent, "new_parent_name": names[new_parent],
                "child_tau": tau[child], "old_parent_tau": tau[old_parent],
                "new_parent_tau": tau[new_parent], "event_support": support[child],
                "receiver_response_rank": response.get(names[child], 0.0),
                "event_exposure_fraction": exposure.get(names[child], 0.0),
                "parent_event_exposure_fraction": exposure.get(names[new_parent], 0.0),
                "parent_response_compatibility": parent_response_compatibility,
                "event_onset_localization": localization.get(child, 0.0),
                "child_transition_program_rank": child_transition_rank,
                "boundary_evidence": (
                    support[child] * localization.get(child, 0.0)
                    * child_transition_rank
                ),
                "alignment": score, "upstream_shift": upstream_shift,
                "branch_escape": branch_escape, "base_old_cost": old_cost,
                "old_expression_rank": old_feature["expression_rank"],
                "program_change": branch_program.get(child) != branch_program.get(new_parent),
                "base_new_cost": new_cost, "modified_old_cost": modified_old,
                "modified_new_cost": modified_new,
                "new_expression_rank": new_feature["expression_rank"],
                "transition_compatibility": transition_compatibility,
                "old_parent_transition_program_rank": old_parent_transition_rank,
                "new_parent_transition_program_rank": parent_transition_rank,
                "transition_program_gain": transition_program_gain,
                "child_transition_onset": child_transition_onset,
                "temporal_reversal": temporal_reversal,
                "transition_program_genes": ";".join(transition_genes),
                "raw_new_joint_cost": new_feature["raw_joint"],
                "cost_advantage": modified_old - modified_new,
                "candidate_legal": True, "truth_used_for_selection": False,
            })
    trial_frame = pd.DataFrame(trials)
    if trial_frame.empty:
        return t0.copy(), trial_frame, {
            "accepted": False, "reason": "no_candidate_legal_alternative",
            "upstream_violation": False,
        }
    if (
        str(event.event_type) == "collision_convergence"
        and not COLLISION_TOPOLOGY_REFINEMENT_ENABLED
    ):
        return t0.copy(), trial_frame, {
            "accepted": False,
            "reason": "collision_relation_evidence_topology_abstention",
            "upstream_violation": False,
        }
    if (
        str(event.event_type) == "inhibition"
        and not INHIBITION_TOPOLOGY_REFINEMENT_ENABLED
    ):
        return t0.copy(), trial_frame, {
            "accepted": False,
            "reason": "inhibition_relation_evidence_topology_abstention",
            "upstream_violation": False,
        }
    positive_cost = trial_frame.cost_advantage.astype(float).gt(0)
    inhibition_program_gate = (
        trial_frame.event_type.astype(str).eq("inhibition")
        & trial_frame.child_transition_program_rank.astype(float).ge(
            PROGRAM_HIGH_QUANTILE
        )
        & trial_frame.new_parent_transition_program_rank.astype(float).ge(
            PROGRAM_HIGH_QUANTILE
        )
    )
    positive_cost = positive_cost & (
        ~trial_frame.event_type.astype(str).eq("inhibition")
        | inhibition_program_gate
    )
    promotion_child_gate = (
        trial_frame.event_type.astype(str).eq("promotion")
        & trial_frame.child_transition_program_rank.astype(float).ge(0.50)
    )
    promotion_program_gate = promotion_child_gate & (
        positive_cost
        | trial_frame.new_parent_transition_program_rank.astype(float).ge(0.50)
    )
    positive_cost = positive_cost & (
        ~trial_frame.event_type.astype(str).eq("promotion")
        | promotion_child_gate
    )
    accepted = trial_frame[
        positive_cost | promotion_program_gate | inhibition_program_gate
    ].copy()
    if str(event.event_type) == "promotion":
        accepted = accepted[
            accepted.transition_program_gain.astype(float).ge(
                MIN_PROMOTION_TRANSITION_GAIN
            )
        ].copy()
    support_gate = (
        max(
            MIN_EVENT_SUPPORT_FOR_REWIRE,
            float(np.median(list(support.values()))) if support else 1.0,
        )
    )
    accepted = accepted[
        accepted.event_support.astype(float).ge(support_gate)
    ].copy()
    if not len(accepted):
        return t0.copy(), trial_frame, {
            "accepted": False, "reason": "cost_abstention",
            "upstream_violation": False,
        }
    accepted["selection_score"] = (
        accepted.cost_advantage.astype(float)
        + accepted.event_support.astype(float)
        * accepted.transition_program_gain.astype(float)
    )
    if str(event.event_type) == "inhibition":
        order_columns = [
            "child_transition_onset", "event_onset_localization",
            "selection_score", "event_support",
            "alignment", "base_new_cost", "child", "new_parent",
        ]
        order_ascending = [False, False, False, False, False, True, True, True]
    else:
        order_columns = [
            "selection_score", "event_support", "alignment",
            "base_new_cost", "child", "new_parent",
        ]
        order_ascending = [False, False, False, True, True, True]
    ordered = accepted.sort_values(
        order_columns, ascending=order_ascending, kind="mergesort"
    )
    refined = t0.copy()
    current_graph = graph.copy()
    selected_rows = []
    changed_children = set()
    for index, candidate_row in ordered.iterrows():
        child = int(candidate_row.child)
        old_parent = int(candidate_row.old_parent)
        new_parent = int(candidate_row.new_parent)
        if child in changed_children or not current_graph.has_edge(old_parent, child):
            continue
        without = current_graph.copy()
        without.remove_edge(old_parent, child)
        if new_parent in nx.node_connected_component(without, child):
            continue
        trial_graph = without.copy()
        trial_graph.add_edge(new_parent, child)
        if not nx.is_tree(trial_graph):
            continue
        refined = replace_edge(
            refined, root_id, old_parent, child, new_parent,
            names, float(candidate_row.raw_new_joint_cost),
        )
        current_graph = trial_graph
        changed_children.add(child)
        selected_rows.append((index, candidate_row))
        # One event authorizes one branch-boundary consequence.  Additional
        # changes require an independently detected event.
        break
    if not selected_rows:
        return t0.copy(), trial_frame, {
            "accepted": False, "reason": "sequential_legality_abstention",
            "upstream_violation": False,
        }
    trial_frame["selected"] = False
    trial_frame.loc[[index for index, _ in selected_rows], "selected"] = True
    chosen = [row for _, row in selected_rows]
    audit = {
        "accepted": True, "reason": "positive_event_conditioned_cost_advantage",
        "n_selected_swaps": len(chosen),
        "support_gate": support_gate,
        "child_tau": float(min(row.child_tau for row in chosen)),
        "receiver_tau_start_true_scale": float(event.receiver_tau_start),
        "upstream_violation": bool(any(
            int(row.child) not in affected_nodes for row in chosen
        )),
        "selected_child": ";".join(str(row.child_name) for row in chosen),
        "selected_old_parent": ";".join(str(row.old_parent_name) for row in chosen),
        "selected_new_parent": ";".join(str(row.new_parent_name) for row in chosen),
    }
    return refined, trial_frame, audit


def evaluate(
    event, before: pd.DataFrame, after: pd.DataFrame,
    state_lookup: dict[tuple[str, str], str], truth_t0: set, truth_t1: set,
    audit: dict,
) -> dict:
    mapped_before = mapped_edge_set(before, state_lookup)
    mapped_after = mapped_edge_set(after, state_lookup)
    formal_before = formal_topology_distance(mapped_before, truth_t1)
    formal_after = formal_topology_distance(mapped_after, truth_t1)
    location = event_location_metrics(
        event, mapped_before, mapped_after, truth_t0, truth_t1, audit, state_lookup
    )
    before_distance = len(mapped_before ^ truth_t1)
    after_distance = len(mapped_after ^ truth_t1)
    true_removed, true_added = truth_t0 - truth_t1, truth_t1 - truth_t0
    pred_removed, pred_added = mapped_before - mapped_after, mapped_after - mapped_before
    truth_rewire = {("removed", edge) for edge in true_removed} | {
        ("added", edge) for edge in true_added
    }
    pred_rewire = {("removed", edge) for edge in pred_removed} | {
        ("added", edge) for edge in pred_added
    }
    precision, recall, rewire_f1 = prf(truth_rewire, pred_rewire)
    affected_token = f"_{event.affected_branch}_"
    root = str(event.receiver_root)
    affected_truth = {
        edge for edge in truth_t1
        if edge[0] == root and (affected_token in edge[1] or affected_token in edge[2])
    }
    affected_pred = {
        edge for edge in mapped_after
        if edge[0] == root and (affected_token in edge[1] or affected_token in edge[2])
    }
    _, _, affected_path_f1 = prf(affected_truth, affected_pred)
    true_event_added = {edge for edge in true_added if edge[0] == root}
    branch_choice_accuracy = float(
        len(true_event_added & pred_added) / len(true_event_added)
    ) if true_event_added else float(not audit["accepted"])
    affected_before = len({edge for edge in mapped_before ^ truth_t1 if edge[0] == root})
    affected_after = len({edge for edge in mapped_after ^ truth_t1 if edge[0] == root})
    return {
        "rewiring_precision": precision, "rewiring_recall": recall,
        "rewiring_F1": rewire_f1, "affected_path_F1": affected_path_f1,
        "branch_choice_accuracy": branch_choice_accuracy,
        "topology_distance_before": before_distance,
        "topology_distance_after": after_distance,
        "oracle_topology_gain": before_distance - after_distance,
        "branch_scale_topology_gain": affected_before - affected_after,
        "formal_topology_distance_before": formal_before,
        "formal_topology_distance_after": formal_after,
        "formal_oracle_topology_gain": formal_before - formal_after,
        **location,
        "n_predicted_removed_edges": len(pred_removed),
        "n_predicted_added_edges": len(pred_added),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="CORE_05")
    parser.add_argument("--phase-a-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    phase_a = args.phase_a_dir.resolve() if args.phase_a_dir else run_dir / "phase_a_v4_frozen"
    args.output.mkdir(parents=True, exist_ok=True)
    t0 = pd.read_csv(phase_a / "T0_FINE_TRAJECTORY_EDGES.csv")
    nodes = pd.read_csv(phase_a / "T0_STATE_NODE_ASSIGNMENTS.csv")
    candidate = pd.read_csv(phase_a / "C1_MULTIRESOLUTION_CANDIDATE_EDGES.csv")
    connector_file = phase_a / "ROOT_SPECIFIC_CONNECTOR_EDGES.csv"
    if connector_file.exists() and connector_file.stat().st_size > 2:
        try:
            connectors = pd.read_csv(connector_file)
        except pd.errors.EmptyDataError:
            connectors = pd.DataFrame()
        if len(connectors):
            connectors["trajectory_edge_eligible"] = True
            candidate = pd.concat([candidate, connectors], ignore_index=True, sort=False)
    events = pd.read_csv(run_dir / "event" / "truth" / "TRUE_EVENTS.csv")
    branch = pd.read_csv(run_dir / "formal_metacell" / "METACELL_BRANCH_COMPOSITION.csv")
    state = pd.read_csv(run_dir / "formal_metacell" / "METACELL_STATE_COMPOSITION.csv")
    state_lookup = dominant_state_lookup(state, "T1")
    truth_t0 = truth_edge_set(pd.read_csv(run_dir / "event" / "truth" / "TRUE_T0_EDGES.csv"))
    truth_t1 = truth_edge_set(pd.read_csv(run_dir / "event" / "truth" / "TRUE_T1_EDGES.csv"))
    result_rows, trial_frames, edge_frames = [], [], []
    for event in events.itertuples(index=False):
        for gamma in GAMMAS:
            refined, trials, audit = refine_event(
                run_dir, args.dataset, t0, nodes, candidate, event, gamma, branch
            )
            metrics = evaluate(event, t0, refined, state_lookup, truth_t0, truth_t1, audit)
            result_rows.append({
                "run": run_dir.name, "event_id": str(event.event_id),
                "event_type": str(event.event_type), "causal": as_bool(event.causal),
                "receiver_root": str(event.receiver_root),
                "affected_branch": str(event.affected_branch), "gamma": gamma,
                **audit, **metrics,
                "candidate_legal": bool(refined.candidate_legal.astype(bool).all()),
                "connected": all(
                    nx.is_connected(nx.Graph(part[["source", "target"]].astype(int).itertuples(index=False, name=None)))
                    for _, part in refined.groupby("root_id")
                ),
                "acyclic": all(
                    nx.is_forest(nx.Graph(part[["source", "target"]].astype(int).itertuples(index=False, name=None)))
                    for _, part in refined.groupby("root_id")
                ),
                "truth_used_for_selection": False,
                "truth_used_for_scoring_only": True,
            })
            if len(trials):
                trials.insert(0, "run", run_dir.name)
                trial_frames.append(trials)
            marked = refined.copy()
            marked.insert(0, "gamma", gamma)
            marked.insert(0, "event_id", str(event.event_id))
            edge_frames.append(marked)
    results = pd.DataFrame(result_rows)
    results.to_csv(args.output / "PHASE_C_ORACLE_REFINEMENT.csv", index=False)
    results[[
        "run", "event_id", "gamma", "rewiring_precision", "rewiring_recall",
        "rewiring_F1", "affected_path_F1", "branch_choice_accuracy",
        "topology_distance_before", "topology_distance_after",
        "oracle_topology_gain", "branch_scale_topology_gain",
    ]].to_csv(args.output / "PHASE_C_REWIRING_METRICS.csv", index=False)
    results[~results.causal.astype(bool)].to_csv(
        args.output / "PHASE_C_NULL_REWIRE_CONTROL.csv", index=False
    )
    results.to_csv(args.output / "PHASE_C_GAMMA_SENSITIVITY.csv", index=False)
    if trial_frames:
        pd.concat(trial_frames, ignore_index=True).to_csv(
            args.output / "PHASE_C_LEGAL_SWAP_TRIALS.csv", index=False
        )
    pd.concat(edge_frames, ignore_index=True).to_csv(
        args.output / "PHASE_C_REFINED_EDGE_SENSITIVITY.csv", index=False
    )
    status = {
        "phase": "C",
        "operator_version": "ORACLE_FINE_REFINEMENT_V7_STRUCTURAL_TRANSITION_SAFETY_GATE",
        "frozen_phase_a": json.loads(
            (phase_a / "RUN_STATUS.json").read_text(encoding="utf-8")
        ).get("version", "unknown"),
        "frozen_phase_a_path": str(phase_a),
        "gammas": GAMMAS, "state_support": STATE_SUPPORT,
        "program_high_quantile": PROGRAM_HIGH_QUANTILE,
        "minimum_event_support_for_rewire": MIN_EVENT_SUPPORT_FOR_REWIRE,
        "minimum_promotion_transition_gain": MIN_PROMOTION_TRANSITION_GAIN,
        "collision_topology_refinement_enabled": COLLISION_TOPOLOGY_REFINEMENT_ENABLED,
        "inhibition_topology_refinement_enabled": INHIBITION_TOPOLOGY_REFINEMENT_ENABLED,
        "relation_detector_called": False,
        "exact_truth_edges_used_for_selection": False,
        "oracle_inputs": [
            "true event identity", "true source and receiver", "true affected branch",
            "true event window", "true event-local exposure",
        ],
        "complete": True,
    }
    (args.output / "RUN_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(results.groupby(["gamma", "causal"])[
        ["accepted", "oracle_topology_gain", "rewiring_F1"]
    ].mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
