#!/usr/bin/env python3
"""PHASE A v1: candidate/path audits and four-way T0 oracle decomposition.

Synthetic truth is loaded only by audit functions. Oracle true membership is
an explicitly labelled evaluation input and is never used by the deployable
case D constructor.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score


HERE = Path(__file__).resolve().parent
T0_ACTIVE_ROLES = {
    "trajectory_local_reciprocal",
    "trajectory_mesoscale_reciprocal",
    "trajectory_crossscale_mst_bridge",
}
EPS = 1e-12


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V1 = load_module("construct_branch_backbone.py", "phase_a_c1")
C1V2 = load_module("run_branch_backbone_experiment.py", "phase_a_c1")
C3V2 = load_module("construct_branch_state_backbone.py", "phase_a_c3")


def pair(a: object, b: object) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def mode(values: list[str]) -> str | None:
    if not values:
        return None
    counts = Counter(map(str, values))
    return min(counts, key=lambda value: (-counts[value], value))


def safe_mean(values) -> float:
    finite = [float(value) for value in values if pd.notna(value) and np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else math.nan


def prf(true_set: set, pred_set: set) -> tuple[float, float, float]:
    overlap = len(true_set & pred_set)
    precision = overlap / len(pred_set) if pred_set else (1.0 if not true_set else 0.0)
    recall = overlap / len(true_set) if true_set else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


@dataclass
class TruthContext:
    roots: list[str]
    anchors: dict[str, str]
    state: dict[tuple[str, str], str]
    branch: dict[tuple[str, str], str]
    root_probability: dict[tuple[str, str], float]
    pseudotime: dict[tuple[str, str], float]
    graphs: dict[str, nx.DiGraph]


def load_truth_context(run_dir: Path) -> TruthContext:
    truth_dir = run_dir / "event" / "truth"
    meta_dir = run_dir / "formal_metacell"
    fixed = pd.read_csv(run_dir / "supplied_root_inputs" / "FIXED_ROOT_INPUT.csv")
    roots = fixed["truth_root_id"].astype(str).tolist()
    anchors = dict(zip(fixed["truth_root_id"].astype(str), fixed["root_anchor_id"].astype(str)))

    state_frame = pd.read_csv(meta_dir / "METACELL_STATE_COMPOSITION.csv")
    state_frame = state_frame[state_frame["stage"].astype(str).eq("T0")]
    dominant = (
        state_frame.sort_values(["metacell_id", "root_id", "state_probability", "state_id"],
                                ascending=[True, True, False, True])
        .drop_duplicates(["metacell_id", "root_id"])
    )
    state = {
        (str(row.metacell_id), str(row.root_id)): str(row.state_id)
        for row in dominant.itertuples(index=False)
        if float(row.state_probability) > 0
    }

    root_frame = pd.read_csv(meta_dir / "METACELL_ROOT_COMPOSITION.csv")
    root_probability = {
        (str(row.metacell_id), str(row.root_id)): float(row.root_probability)
        for row in root_frame.itertuples(index=False)
    }

    mapping = pd.read_csv(meta_dir / "cell_to_metacell_mapping.csv")[["cell_name", "metacell_id"]]
    pt = pd.read_csv(truth_dir / "TRUE_PSEUDOTIME.csv").rename(columns={"cell_id": "cell_name"})
    pt = mapping.merge(pt, on="cell_name", how="inner")
    medians = pt.groupby(["metacell_id", "root_id"], as_index=False)["pseudotime"].median()
    pseudotime = {
        (str(row.metacell_id), str(row.root_id)): float(row.pseudotime)
        for row in medians.itertuples(index=False)
    }

    branch_cells = pd.read_csv(truth_dir / "TRUE_CELL_BRANCH.csv")
    branch_cells = branch_cells[branch_cells["stage"].astype(str).eq("T0")].rename(columns={"cell_id": "cell_name"})
    branch_cells = mapping.merge(branch_cells[["cell_name", "root_id", "branch_id"]], on="cell_name", how="inner")
    branch = {}
    for key, group in branch_cells.groupby(["metacell_id", "root_id"]):
        branch[(str(key[0]), str(key[1]))] = str(mode(group["branch_id"].astype(str).tolist()))

    truth_edges = pd.read_csv(truth_dir / "TRUE_T0_EDGES.csv")
    graphs = {}
    for root in roots:
        graph = nx.DiGraph()
        local = truth_edges[truth_edges["root_id"].astype(str).eq(root)]
        graph.add_edges_from((str(row.source), str(row.target)) for row in local.itertuples(index=False))
        graphs[root] = graph
    return TruthContext(roots, anchors, state, branch, root_probability, pseudotime, graphs)


def inferred_membership(method_dir: Path, n_nodes: int):
    membership, roots, active = V1.load_membership(method_dir, n_nodes)
    supports = {
        str(root["root_id"]): V1.root_support(active, int(root["root_pos"]), int(root["root_index"]))
        for root in roots
    }
    return membership, roots, supports


def true_membership(meta: ad.AnnData, roots: list[dict], truth: TruthContext):
    membership = np.zeros((meta.n_obs, len(roots)), dtype=float)
    supports = {}
    name_to_index = {str(name): i for i, name in enumerate(meta.obs_names)}
    for root in roots:
        root_id = str(root["root_id"])
        k = int(root["root_pos"])
        for name, index in name_to_index.items():
            membership[index, k] = truth.root_probability.get((name, root_id), 0.0)
        support = set(np.flatnonzero(membership[:, k] > 0).astype(int))
        support.add(int(root["root_index"]))
        supports[root_id] = support
    row_sum = membership.sum(axis=1, keepdims=True)
    membership = np.divide(membership, row_sum, out=np.zeros_like(membership), where=row_sum > 0)
    return membership, supports


def current_candidate(method_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(method_dir / "trajectory_candidate_edges.csv")
    frame = frame[frame["trajectory_edge_eligible"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    frame["D_joint"] = frame["d_global"].astype(float)
    frame["joint_distance_ecdf"] = V1.empirical_rank(frame["D_joint"].to_numpy(float))
    frame["edge_graph_role"] = "trajectory_current_single_scale"
    frame["trajectory_edge_eligible"] = True
    return frame


def new_candidates(meta: ad.AnnData, geometry: dict[str, np.ndarray]):
    full, audit = C1V2.candidate(V1, meta, geometry)
    full["T0_role_active"] = full["edge_graph_role"].astype(str).isin(T0_ACTIVE_ROLES)
    full["T1_reserve_eligible"] = full["trajectory_edge_eligible"].astype(bool) & ~full["T0_role_active"]
    active = full[full["T0_role_active"]].copy()
    active["joint_distance_ecdf"] = V1.empirical_rank(active["D_joint"].to_numpy(float))
    return full, active, audit


def topology_branch_labels(forest: nx.Graph, root: int) -> dict[int, str]:
    labels = {}
    for cid, nodes_set in enumerate(sorted(nx.connected_components(forest), key=lambda nodes: (root not in nodes, min(nodes)))):
        nodes = set(map(int, nodes_set))
        if root not in nodes:
            labels.update({node: f"component_{cid}" for node in nodes})
            continue
        local = forest.subgraph(nodes)
        for node in sorted(nodes):
            if node == root:
                labels[node] = "trunk"
                continue
            path = nx.shortest_path(local, root, node)
            split_index = next((i for i, value in enumerate(path[:-1]) if local.degree(value) >= 3), None)
            if split_index is None:
                labels[node] = "trunk"
            else:
                labels[node] = f"branch_from_{path[split_index + 1]}"
    return labels


def state_backbone(
    root_id: str,
    fine: nx.Graph,
    branches: dict[int, str],
    tau: dict[int, float],
    names: list[str],
) -> pd.DataFrame:
    best = {}
    for i, j, data in fine.edges(data=True):
        source_state, target_state = str(branches[int(i)]), str(branches[int(j)])
        if source_state == target_state:
            continue
        if (float(tau.get(int(i), 0.0)), int(i)) > (float(tau.get(int(j), 0.0)), int(j)):
            i, j = j, i
            source_state, target_state = target_state, source_state
        key = (source_state, target_state)
        trial = (float(data.get("weight", 0.0)), int(i), int(j), str(data.get("edge_graph_role", "")))
        if key not in best or trial[:3] < best[key][:3]:
            best[key] = trial
    rows = []
    for (source_state, target_state), (cost, i, j, role) in sorted(best.items()):
        rows.append({
            "root_id": root_id,
            "source_state": source_state,
            "target_state": target_state,
            "support_source": names[i],
            "support_target": names[j],
            "edge_cost": cost,
            "edge_graph_role": role,
        })
    return pd.DataFrame(rows, columns=[
        "root_id", "source_state", "target_state", "support_source", "support_target",
        "edge_cost", "edge_graph_role",
    ])


def construct_t0(
    meta: ad.AnnData,
    geometry: dict[str, np.ndarray],
    candidate: pd.DataFrame,
    membership: np.ndarray,
    roots: list[dict],
    supports: dict[str, set[int]],
    branch_scale: bool,
) -> dict[str, pd.DataFrame]:
    edge_rows, node_rows, backbone_rows = [], [], []
    names = [str(name) for name in meta.obs_names]
    legal = {V1.edge_key(int(row.source), int(row.target)) for row in candidate.itertuples(index=False)}
    for root in roots:
        root_id = str(root["root_id"])
        k, root_index = int(root["root_pos"]), int(root["root_index"])
        graph = V1.graph_for_support(candidate, supports[root_id], membership[:, k])
        preliminary = V1.component_forest(graph)
        tau_pre, _ = V1.rooted_tau(preliminary, root_index, membership[:, k])
        if branch_scale:
            branches, _ = C3V2.raw_expression_diagonal_gmm(geometry["expression"], preliminary, tau_pre)
            fine = V1.quotient_branch_forest(graph, branches)
        else:
            fine = preliminary
            branches = topology_branch_labels(fine, root_index)
        tau, component = V1.rooted_tau(fine, root_index, membership[:, k])
        for i, j, data in fine.edges(data=True):
            edge_rows.append({
                "root_id": root_id,
                "source": int(i),
                "target": int(j),
                "source_name": names[int(i)],
                "target_name": names[int(j)],
                "edge_cost": float(data.get("weight", 0.0)),
                "edge_graph_role": str(data.get("edge_graph_role", "")),
                "candidate_legal": V1.edge_key(int(i), int(j)) in legal,
            })
        for node in sorted(supports[root_id]):
            node_rows.append({
                "root_id": root_id,
                "node_id": int(node),
                "unit_id": names[int(node)],
                "membership": float(membership[int(node), k]),
                "pseudotime": float(tau.get(int(node), 0.0)),
                "branch_id": f"{root_id}_{branches[int(node)]}",
                "component_id": int(component.get(int(node), -1)),
                "is_root": int(node) == root_index,
            })
        backbone_rows.append(state_backbone(root_id, fine, branches, tau, names))
    edges = pd.DataFrame(edge_rows)
    nodes = pd.DataFrame(node_rows)
    backbone = pd.concat(backbone_rows, ignore_index=True) if backbone_rows else pd.DataFrame()
    return {"edges": edges, "nodes": nodes, "backbone": backbone}


def orient_by_truth(graph: nx.Graph, root_id: str, truth: TruthContext) -> nx.DiGraph:
    directed = nx.DiGraph()
    directed.add_nodes_from(graph.nodes())
    for a, b in graph.edges():
        ta = truth.pseudotime.get((str(a), root_id), math.nan)
        tb = truth.pseudotime.get((str(b), root_id), math.nan)
        if not np.isfinite(ta) or not np.isfinite(tb):
            continue
        if tb >= ta:
            directed.add_edge(str(a), str(b))
        if ta >= tb:
            directed.add_edge(str(b), str(a))
    return directed


def transition_path_coverage(graph: nx.Graph, root_id: str, truth: TruthContext):
    state = {name: value for (name, root), value in truth.state.items() if root == root_id}
    directed = orient_by_truth(graph, root_id, truth)
    truth_graph = truth.graphs[root_id]
    covered = {}
    for source, target in truth_graph.edges():
        source_nodes = {node for node, value in state.items() if value == source and node in directed}
        target_nodes = {node for node, value in state.items() if value == target and node in directed}
        allowed = source_nodes | target_nodes
        local = directed.subgraph(allowed)
        covered[(source, target)] = bool(source_nodes and target_nodes and any(
            nx.has_path(local, a, b) for a in source_nodes for b in target_nodes
        ))
    terminals = [node for node in truth_graph if truth_graph.out_degree(node) == 0]
    roots = [node for node in truth_graph if truth_graph.in_degree(node) == 0]
    path_scores = []
    for source in roots:
        for terminal in terminals:
            if not nx.has_path(truth_graph, source, terminal):
                continue
            path = nx.shortest_path(truth_graph, source, terminal)
            transitions = list(zip(path[:-1], path[1:]))
            path_scores.append(safe_mean([float(covered.get(edge, False)) for edge in transitions]))
    return safe_mean(covered.values()), safe_mean(path_scores), covered


def candidate_audit(
    run_label: str,
    candidate_name: str,
    candidate: pd.DataFrame,
    truth: TruthContext,
) -> pd.DataFrame:
    eligible = candidate[candidate["trajectory_edge_eligible"].astype(bool)]
    graph = nx.Graph()
    graph.add_edges_from((str(row.source_name), str(row.target_name)) for row in eligible.itertuples(index=False))
    rows = []
    total_pairs = len({str(value) for key in truth.state for value in [key[0]]})
    for root_id in truth.roots:
        truth_graph = truth.graphs[root_id]
        root_nodes = {name for (name, root), _ in truth.state.items() if root == root_id}
        anchor = truth.anchors[root_id]
        local = graph.subgraph(root_nodes | {anchor}).copy()
        reachable = nx.node_connected_component(local, anchor) if anchor in local else set()
        terminals = {node for node in truth_graph if truth_graph.out_degree(node) == 0}
        branchpoints = {node for node in truth_graph if truth_graph.out_degree(node) > 1}
        terminal_nodes = {name for (name, root), state in truth.state.items() if root == root_id and state in terminals}
        branchpoint_nodes = {name for (name, root), state in truth.state.items() if root == root_id and state in branchpoints}
        direct_pairs = {
            pair(truth.state[(str(a), root_id)], truth.state[(str(b), root_id)])
            for a, b in local.edges()
            if (str(a), root_id) in truth.state and (str(b), root_id) in truth.state
            and truth.state[(str(a), root_id)] != truth.state[(str(b), root_id)]
        }
        true_pairs = {pair(a, b) for a, b in truth_graph.edges()}
        edge_recall = len(direct_pairs & true_pairs) / max(len(true_pairs), 1)
        transition_coverage, path_coverage, _ = transition_path_coverage(local, root_id, truth)
        projected = []
        for a, b in local.edges():
            sa, sb = truth.state.get((str(a), root_id)), truth.state.get((str(b), root_id))
            if sa and sb and sa != sb:
                projected.append((sa, sb))
        truth_undirected = truth_graph.to_undirected()
        shortcuts = [
            nx.shortest_path_length(truth_undirected, a, b) > 1
            for a, b in projected if a in truth_undirected and b in truth_undirected and nx.has_path(truth_undirected, a, b)
        ]
        cross_branch = [
            not nx.has_path(truth_graph, a, b) and not nx.has_path(truth_graph, b, a)
            for a, b in projected if a in truth_graph and b in truth_graph
        ]
        rows.append({
            "run": run_label,
            "candidate": candidate_name,
            "root_id": root_id,
            "candidate_true_edge_recall": edge_recall,
            "candidate_transition_path_coverage": transition_coverage,
            "candidate_true_path_coverage": path_coverage,
            "candidate_root_reachable_fraction": len(reachable & root_nodes) / max(len(root_nodes), 1),
            "candidate_terminal_reachable_fraction": len(reachable & terminal_nodes) / max(len(terminal_nodes), 1),
            "candidate_branchpoint_reachable_fraction": len(reachable & branchpoint_nodes) / max(len(branchpoint_nodes), 1),
            "candidate_shortcut_rate": safe_mean(shortcuts),
            "candidate_cross_branch_false_connection_rate": safe_mean(cross_branch),
            "n_candidate_nodes": local.number_of_nodes(),
            "n_candidate_edges": local.number_of_edges(),
            "n_projectable_cross_state_edges": len(projected),
            "n_unsupported_possible_pairs": max(total_pairs * (total_pairs - 1) // 2 - graph.number_of_edges(), 0),
            "truth_use": "audit_only",
        })
    numeric = [column for column in rows[0] if column not in {"run", "candidate", "root_id", "truth_use"}]
    aggregate = {"run": run_label, "candidate": candidate_name, "root_id": "ALL", "truth_use": "audit_only"}
    aggregate.update({column: safe_mean([row[column] for row in rows]) for column in numeric})
    return pd.DataFrame(rows + [aggregate])


def mapped_backbone_graph(root_id: str, result: dict[str, pd.DataFrame], truth: TruthContext):
    nodes = result["nodes"]
    nodes = nodes[nodes["root_id"].astype(str).eq(root_id)]
    predicted_to_true = {}
    for branch_id, group in nodes.groupby("branch_id"):
        values = [truth.state.get((str(unit), root_id)) for unit in group["unit_id"]]
        predicted_to_true[str(branch_id)] = mode([value for value in values if value])
    graph = nx.DiGraph()
    backbone = result["backbone"]
    backbone = backbone[backbone["root_id"].astype(str).eq(root_id)] if len(backbone) else backbone
    prefix = f"{root_id}_"
    for row in backbone.itertuples(index=False):
        source = predicted_to_true.get(prefix + str(row.source_state))
        target = predicted_to_true.get(prefix + str(row.target_state))
        if source and target and source != target:
            graph.add_edge(source, target)
    root_rows = nodes[nodes["is_root"].astype(bool)]
    predicted_root_branch = str(root_rows.iloc[0]["branch_id"]) if len(root_rows) else None
    root_state = predicted_to_true.get(predicted_root_branch)
    return graph, predicted_to_true, root_state


def shared_prefix_score(true_graph: nx.DiGraph, pred_graph: nx.DiGraph, true_root: str, pred_root: str | None):
    terminals = [node for node in true_graph if true_graph.out_degree(node) == 0]
    scores = []
    for i, left in enumerate(terminals):
        for right in terminals[i + 1:]:
            if not (nx.has_path(true_graph, true_root, left) and nx.has_path(true_graph, true_root, right)):
                continue
            left_true = nx.shortest_path(true_graph, true_root, left)
            right_true = nx.shortest_path(true_graph, true_root, right)
            true_prefix = 0
            for a, b in zip(left_true, right_true):
                if a != b:
                    break
                true_prefix += 1
            true_fraction = max(true_prefix - 1, 0) / max(min(len(left_true), len(right_true)) - 1, 1)
            if pred_root is None or not (pred_root in pred_graph and left in pred_graph and right in pred_graph):
                scores.append(0.0)
                continue
            if not (nx.has_path(pred_graph, pred_root, left) and nx.has_path(pred_graph, pred_root, right)):
                scores.append(0.0)
                continue
            left_pred = nx.shortest_path(pred_graph, pred_root, left)
            right_pred = nx.shortest_path(pred_graph, pred_root, right)
            pred_prefix = 0
            for a, b in zip(left_pred, right_pred):
                if a != b:
                    break
                pred_prefix += 1
            pred_fraction = max(pred_prefix - 1, 0) / max(min(len(left_pred), len(right_pred)) - 1, 1)
            scores.append(max(0.0, 1.0 - abs(true_fraction - pred_fraction)))
    return safe_mean(scores)


def t0_metrics(run_label: str, case: str, result: dict[str, pd.DataFrame], truth: TruthContext) -> pd.DataFrame:
    rows = []
    for root_id in truth.roots:
        edges = result["edges"][result["edges"]["root_id"].astype(str).eq(root_id)]
        nodes = result["nodes"][result["nodes"]["root_id"].astype(str).eq(root_id)]
        fine = nx.Graph()
        fine.add_nodes_from(nodes["unit_id"].astype(str))
        fine.add_edges_from((str(row.source_name), str(row.target_name)) for row in edges.itertuples(index=False))
        truth_graph = truth.graphs[root_id]
        true_pairs = {pair(a, b) for a, b in truth_graph.edges()}
        pred_pairs = set()
        projected = []
        for a, b in fine.edges():
            sa, sb = truth.state.get((a, root_id)), truth.state.get((b, root_id))
            if sa and sb and sa != sb:
                pred_pairs.add(pair(sa, sb))
                projected.append((sa, sb))
        edge_p, edge_r, edge_f1 = prf(true_pairs, pred_pairs)
        _, path_coverage, _ = transition_path_coverage(fine, root_id, truth)
        truth_undirected = truth_graph.to_undirected()
        shortcut = safe_mean([
            nx.shortest_path_length(truth_undirected, a, b) > 1
            for a, b in projected if a in truth_undirected and b in truth_undirected
            and nx.has_path(truth_undirected, a, b)
        ])

        branch_pairs = [
            (truth.branch.get((str(row.unit_id), root_id)), str(row.branch_id))
            for row in nodes.itertuples(index=False)
            if truth.branch.get((str(row.unit_id), root_id)) is not None
        ]
        branch_ari = adjusted_rand_score(
            [value[0] for value in branch_pairs], [value[1] for value in branch_pairs]
        ) if len(branch_pairs) >= 2 else math.nan

        mapped, _, mapped_root = mapped_backbone_graph(root_id, result, truth)
        true_branchpoints = {node for node in truth_graph if truth_graph.out_degree(node) > 1}
        true_terminals = {node for node in truth_graph if truth_graph.out_degree(node) == 0}
        pred_branchpoints = {node for node in mapped if mapped.out_degree(node) > 1}
        pred_terminals = {node for node in mapped if mapped.out_degree(node) == 0 and node != mapped_root}
        bp_p, bp_r, bp_f1 = prf(true_branchpoints, pred_branchpoints)
        term_p, term_r, term_f1 = prf(true_terminals, pred_terminals)
        ancestor = safe_mean([
            float(source in truth_graph and target in truth_graph and nx.has_path(truth_graph, source, target))
            for source, target in mapped.edges()
        ])
        true_roots = [node for node in truth_graph if truth_graph.in_degree(node) == 0]
        true_root = true_roots[0] if true_roots else None
        terminal_errors = []
        if true_root is not None:
            for terminal in true_terminals:
                true_distance = nx.shortest_path_length(truth_graph, true_root, terminal)
                if mapped_root is None or mapped_root not in mapped or terminal not in mapped or not nx.has_path(mapped, mapped_root, terminal):
                    terminal_errors.append(1.0)
                else:
                    pred_distance = nx.shortest_path_length(mapped, mapped_root, terminal)
                    terminal_errors.append(min(abs(pred_distance - true_distance) / max(true_distance, 1), 1.0))
        root_terminal_similarity = 1.0 - safe_mean(terminal_errors) if terminal_errors else math.nan
        shared_prefix = shared_prefix_score(truth_graph, mapped, true_root, mapped_root) if true_root else math.nan
        key_nodes = true_branchpoints | true_terminals
        order_trials = []
        for source in key_nodes:
            for target in key_nodes:
                if source != target and nx.has_path(truth_graph, source, target):
                    order_trials.append(float(source in mapped and target in mapped and nx.has_path(mapped, source, target)))
        branch_order = safe_mean(order_trials)

        pt_true, pt_pred, root_distance = [], [], []
        anchor = truth.anchors[root_id]
        reachable = nx.node_connected_component(fine, anchor) if anchor in fine else set()
        lengths = nx.single_source_shortest_path_length(fine.subgraph(reachable), anchor) if anchor in fine else {}
        pred_lookup = dict(zip(nodes["unit_id"].astype(str), nodes["pseudotime"].astype(float)))
        for name in pred_lookup:
            true_value = truth.pseudotime.get((name, root_id))
            if true_value is not None and np.isfinite(true_value):
                pt_true.append(float(true_value))
                pt_pred.append(float(pred_lookup[name]))
                if name in lengths:
                    root_distance.append((float(true_value), float(lengths[name])))
        pt_sr = float(spearmanr(pt_true, pt_pred).statistic) if len(pt_true) >= 3 and len(set(pt_true)) > 1 and len(set(pt_pred)) > 1 else math.nan
        distance_sr = float(spearmanr([x[0] for x in root_distance], [x[1] for x in root_distance]).statistic) if len(root_distance) >= 3 else math.nan
        rows.append({
            "run": run_label,
            "case": case,
            "root_id": root_id,
            "fine_edge_precision": edge_p,
            "fine_edge_recall": edge_r,
            "fine_edge_F1": edge_f1,
            "fine_true_path_coverage": path_coverage,
            "fine_shortcut_rate": shortcut,
            "branch_ARI_rootwise": branch_ari,
            "branchpoint_precision": bp_p,
            "branchpoint_recall": bp_r,
            "branchpoint_F1": bp_f1,
            "terminal_precision": term_p,
            "terminal_recall": term_r,
            "terminal_F1": term_f1,
            "ancestor_descendant_consistency": ancestor,
            "root_terminal_path_similarity": root_terminal_similarity,
            "shared_prefix_recovery": shared_prefix,
            "branching_order_accuracy": branch_order,
            "pseudotime_spearman": pt_sr,
            "root_distance_consistency": distance_sr,
            "n_fine_edges": fine.number_of_edges(),
            "n_state_backbone_edges": int(len(result["backbone"][result["backbone"]["root_id"].astype(str).eq(root_id)])) if len(result["backbone"]) else 0,
            "n_components": nx.number_connected_components(fine) if len(fine) else 0,
            "is_forest": nx.is_forest(fine),
            "truth_use": "audit_only",
        })
    numeric = [column for column in rows[0] if column not in {"run", "case", "root_id", "truth_use", "is_forest"}]
    aggregate = {"run": run_label, "case": case, "root_id": "ALL", "truth_use": "audit_only", "is_forest": all(row["is_forest"] for row in rows)}
    aggregate.update({column: safe_mean([row[column] for row in rows]) for column in numeric})
    return pd.DataFrame(rows + [aggregate])


def event_feasibility(
    run_label: str,
    run_dir: Path,
    candidate: pd.DataFrame,
    result: dict[str, pd.DataFrame],
    truth: TruthContext,
) -> pd.DataFrame:
    truth_dir = run_dir / "event" / "truth"
    events = pd.read_csv(truth_dir / "TRUE_EVENTS.csv")
    added = pd.read_csv(truth_dir / "TRUE_ADDED_EDGES.csv")
    removed = pd.read_csv(truth_dir / "TRUE_REMOVED_EDGES.csv")
    true_t1 = pd.read_csv(truth_dir / "TRUE_T1_EDGES.csv")
    eligible = candidate[candidate["trajectory_edge_eligible"].astype(bool)]
    rows = []
    for event in events.itertuples(index=False):
        event_id, root_id = str(event.event_id), str(event.receiver_root)
        add_rows = added[added["event_id"].astype(str).eq(event_id)]
        remove_rows = removed[removed["event_id"].astype(str).eq(event_id)]
        add_pairs = {pair(row.source, row.target) for row in add_rows.itertuples(index=False)}
        remove_pairs = {pair(row.source, row.target) for row in remove_rows.itertuples(index=False)}
        candidate_edges = []
        for row in eligible.itertuples(index=False):
            source_state = truth.state.get((str(row.source_name), root_id))
            target_state = truth.state.get((str(row.target_name), root_id))
            if source_state and target_state and pair(source_state, target_state) in add_pairs:
                candidate_edges.append((str(row.source_name), str(row.target_name)))
        local_edges = result["edges"][result["edges"]["root_id"].astype(str).eq(root_id)]
        removable_edges = []
        for row in local_edges.itertuples(index=False):
            source_state = truth.state.get((str(row.source_name), root_id))
            target_state = truth.state.get((str(row.target_name), root_id))
            if source_state and target_state and pair(source_state, target_state) in remove_pairs:
                removable_edges.append((str(row.source_name), str(row.target_name)))
        graph = nx.Graph()
        graph.add_edges_from((str(row.source_name), str(row.target_name)) for row in local_edges.itertuples(index=False))
        legal_swaps = 0
        for remove_edge in removable_edges:
            for add_edge in candidate_edges:
                trial = graph.copy()
                if trial.has_edge(*remove_edge):
                    trial.remove_edge(*remove_edge)
                trial.add_edge(*add_edge)
                legal_swaps += int(nx.is_forest(trial))
        represented = set()
        for row in eligible.itertuples(index=False):
            source_state = truth.state.get((str(row.source_name), root_id))
            target_state = truth.state.get((str(row.target_name), root_id))
            if source_state and target_state and source_state != target_state:
                represented.add(pair(source_state, target_state))
        local_true_t1 = true_t1[true_t1["root_id"].astype(str).eq(root_id)]
        required_t1 = {pair(row.source, row.target) for row in local_true_t1.itertuples(index=False)}
        rows.append({
            "run": run_label,
            "event_id": event_id,
            "event_type": str(event.event_type),
            "receiver_root": root_id,
            "topology_changed": bool(event.topology_changed),
            "n_truth_added_state_edges": len(add_pairs),
            "n_candidate_added_edge_realizations": len(candidate_edges),
            "n_T0_removable_edge_realizations": len(removable_edges),
            "n_strict_legal_downstream_swaps": legal_swaps,
            "strict_legal_alternative_exists": legal_swaps > 0,
            "truth_T1_state_edge_coverage": len(required_t1 & represented) / max(len(required_t1), 1),
            "all_truth_T1_state_edges_representable": required_t1 <= represented,
            "truth_use": "oracle_audit_only_no_refinement_performed",
        })
    return pd.DataFrame(rows)


def write_case(output: Path, name: str, result: dict[str, pd.DataFrame]) -> None:
    target = output / "oracle_cases" / name
    target.mkdir(parents=True, exist_ok=True)
    result["backbone"].to_csv(target / "T0_STATE_BACKBONE_EDGES.csv", index=False)
    result["edges"].to_csv(target / "T0_FINE_TRAJECTORY_EDGES.csv", index=False)
    result["nodes"].to_csv(target / "T0_NODE_ASSIGNMENTS.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="CORE_05")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir, output = args.run_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    method_dir = run_dir / "supplied_root_inputs" / args.dataset
    meta = ad.read_h5ad(method_dir / "metacell_adata.h5ad")
    geometry = V1.prepare_geometry(meta)
    inferred_a, roots, inferred_supports = inferred_membership(method_dir, meta.n_obs)
    truth = load_truth_context(run_dir)
    true_a, true_supports = true_membership(meta, roots, truth)
    current = current_candidate(method_dir)
    new_full, new_active, candidate_build_audit = new_candidates(meta, geometry)

    cases = {
        "A_true_membership_current_T0": construct_t0(meta, geometry, current, true_a, roots, true_supports, False),
        "B_inferred_membership_current_T0": construct_t0(meta, geometry, current, inferred_a, roots, inferred_supports, False),
        "C_true_membership_new_T0": construct_t0(meta, geometry, new_active, true_a, roots, true_supports, True),
        "D_inferred_membership_new_T0": construct_t0(meta, geometry, new_active, inferred_a, roots, inferred_supports, True),
    }
    for name, result in cases.items():
        write_case(output, name, result)
    selected = cases["D_inferred_membership_new_T0"]
    selected["backbone"].to_csv(output / "T0_STATE_BACKBONE_EDGES.csv", index=False)
    selected["edges"].to_csv(output / "T0_FINE_TRAJECTORY_EDGES.csv", index=False)

    candidate_metrics = pd.concat([
        candidate_audit(args.run_label, "current_single_scale", current, truth),
        candidate_audit(args.run_label, "new_multiresolution_full", new_full, truth),
    ], ignore_index=True)
    candidate_metrics.to_csv(output / "PHASE_A_CANDIDATE_AUDIT.csv", index=False)
    t0_frames = [t0_metrics(args.run_label, name, result, truth) for name, result in cases.items()]
    t0_frame = pd.concat(t0_frames, ignore_index=True)
    t0_frame.to_csv(output / "PHASE_A_T0_METRICS.csv", index=False)
    oracle = t0_frame[t0_frame["root_id"].astype(str).eq("ALL")].copy()
    oracle.to_csv(output / "PHASE_A_ORACLE_DECOMPOSITION.csv", index=False)
    feasibility = event_feasibility(args.run_label, run_dir, new_full, selected, truth)
    feasibility.to_csv(output / "PHASE_A_EVENT_FEASIBILITY.csv", index=False)
    feasibility.to_csv(output / "ORACLE_EVENT_FEASIBILITY_ON_T0.csv", index=False)
    pd.DataFrame([candidate_build_audit]).to_csv(output / "CANDIDATE_BUILD_AUDIT.csv", index=False)

    status = {
        "phase": "A",
        "version": "PHASE_A",
        "run": args.run_label,
        "dataset": args.dataset,
        "relation_detector_called": False,
        "refinement_operator_called": False,
        "production_case": "D_inferred_membership_new_T0",
        "oracle_truth_membership_cases": ["A_true_membership_current_T0", "C_true_membership_new_T0"],
        "all_fine_edges_candidate_legal": bool(selected["edges"]["candidate_legal"].all()),
        "all_cases_are_forests": bool(all(frame["is_forest"].all() for frame in t0_frames)),
        "truth_used_for_construction_D": False,
        "complete": True,
    }
    (output / "RUN_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
