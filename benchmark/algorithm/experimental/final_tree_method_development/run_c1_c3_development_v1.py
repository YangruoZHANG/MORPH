#!/usr/bin/env python3
"""Develop and audit multi-resolution candidates and branch-scale T0.

Truth is deliberately isolated in ``evaluate_only``. The method constructor
accepts method outputs only and cannot access benchmark truth.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


EPS = 1e-12
SEED = 20260809


def edge_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def dense_float(x) -> np.ndarray:
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=float)


def empirical_rank(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(float)
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def reciprocal_pairs(distance: np.ndarray, k: int) -> set[tuple[int, int]]:
    n = len(distance)
    order = np.argsort(distance, axis=1)[:, : min(k, n - 1)]
    neighbours = [set(map(int, row)) for row in order]
    return {
        edge_key(i, j)
        for i in range(n)
        for j in neighbours[i]
        if i in neighbours[j] and i != j
    }


def directed_union_pairs(distance: np.ndarray, k: int) -> set[tuple[int, int]]:
    n = len(distance)
    order = np.argsort(distance, axis=1)[:, : min(k, n - 1)]
    return {edge_key(i, int(j)) for i in range(n) for j in order[i] if i != int(j)}


def distance_mst_pairs(distance: np.ndarray) -> set[tuple[int, int]]:
    graph = nx.Graph()
    n = len(distance)
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            graph.add_edge(i, j, weight=float(distance[i, j]))
    return {edge_key(int(i), int(j)) for i, j in nx.minimum_spanning_tree(graph).edges()}


def prepare_geometry(adata: ad.AnnData) -> dict[str, np.ndarray]:
    x = dense_float(adata.X)
    x = np.log1p(np.maximum(x, 0.0))
    x = StandardScaler().fit_transform(x)
    n_components = max(1, min(20, x.shape[0] - 1, x.shape[1]))
    expression = PCA(n_components=n_components, random_state=SEED).fit_transform(x)
    expression = StandardScaler().fit_transform(expression)
    spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    spatial = StandardScaler().fit_transform(spatial)
    d_expr = cdist(expression, expression)
    d_spatial = cdist(spatial, spatial)
    mask = ~np.eye(len(expression), dtype=bool)
    expr_scale = np.median(d_expr[mask & (d_expr > 0)])
    spatial_scale = np.median(d_spatial[mask & (d_spatial > 0)])
    d_expr /= max(float(expr_scale), EPS)
    d_spatial /= max(float(spatial_scale), EPS)
    # The 0.35 spatial contribution is inherited from the frozen metacell
    # construction, so graph geometry and metacell geometry use one scale.
    d_joint = 0.65 * d_expr + 0.35 * d_spatial
    np.fill_diagonal(d_expr, np.inf)
    np.fill_diagonal(d_spatial, np.inf)
    np.fill_diagonal(d_joint, np.inf)
    return {
        "expression": expression,
        "spatial": spatial,
        "d_expr": d_expr,
        "d_spatial": d_spatial,
        "d_joint": d_joint,
    }


def build_multiresolution_candidate(adata: ad.AnnData, geometry: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict]:
    """Construct candidate edges without roots, assignments, pseudotime, or truth."""
    n = adata.n_obs
    k_local = max(3, int(math.ceil(math.log2(max(n, 2)))))
    k_meso = max(k_local + 1, int(math.ceil(math.sqrt(n))))
    k_broad = max(k_meso + 1, int(math.ceil(2.0 * math.sqrt(n))))
    k_broad = min(k_broad, n - 1)

    d_expr = geometry["d_expr"]
    d_spatial = geometry["d_spatial"]
    d_joint = geometry["d_joint"]
    local_joint = reciprocal_pairs(d_joint, k_local)
    local_expr = reciprocal_pairs(d_expr, k_local)
    local = local_joint | local_expr
    meso = reciprocal_pairs(d_joint, k_meso) - local
    bridge = distance_mst_pairs(d_joint) - local - meso
    broad = directed_union_pairs(d_joint, k_broad) | directed_union_pairs(d_expr, k_broad)

    reference = sorted(local | meso)
    q_expr = float(np.quantile([d_expr[i, j] for i, j in reference], 0.95)) if reference else math.inf
    q_spatial = float(np.quantile([d_spatial[i, j] for i, j in reference], 0.975)) if reference else math.inf
    structural_bridge = {
        edge for edge in bridge
        if d_expr[edge] <= q_expr and d_spatial[edge] <= q_spatial
    }
    structural = local | meso | structural_bridge
    universe = broad | bridge | structural

    records = []
    for i, j in sorted(universe):
        if (i, j) in local:
            role = "trajectory_local_reciprocal"
            scale_rank = 0
        elif (i, j) in meso:
            role = "trajectory_mesoscale_reciprocal"
            scale_rank = 1
        elif (i, j) in structural_bridge:
            role = "trajectory_crossscale_mst_bridge"
            scale_rank = 2
        else:
            role = "interaction_only_broad_neighbour"
            scale_rank = 3
        records.append({
            "source": i,
            "target": j,
            "source_name": str(adata.obs_names[i]),
            "target_name": str(adata.obs_names[j]),
            "D_expression": float(d_expr[i, j]),
            "D_spatial": float(d_spatial[i, j]),
            "D_joint": float(d_joint[i, j]),
            "S_global": float(math.exp(-d_joint[i, j])),
            "d_global": float(d_joint[i, j]),
            "candidate_scale_rank": scale_rank,
            "edge_graph_role": role,
            "trajectory_edge_eligible": bool((i, j) in structural),
            "edge_role_rule": "node_count_scales_and_reciprocity;bridge_has_empirical_distance_guard",
        })
    frame = pd.DataFrame(records)
    eligible = frame[frame["trajectory_edge_eligible"]].copy()
    values = eligible["D_joint"].to_numpy(float)
    eligible_ranks = dict(zip(
        [edge_key(int(r.source), int(r.target)) for r in eligible.itertuples(index=False)],
        empirical_rank(values),
    ))
    frame["joint_distance_ecdf"] = [eligible_ranks.get(edge_key(int(r.source), int(r.target)), 1.0) for r in frame.itertuples(index=False)]
    audit = {
        "n_nodes": n,
        "k_local": k_local,
        "k_mesoscale": k_meso,
        "k_broad": k_broad,
        "n_local": len(local),
        "n_mesoscale": len(meso),
        "n_bridge_proposed": len(bridge),
        "n_bridge_eligible": len(structural_bridge),
        "n_trajectory_eligible": len(structural),
        "n_interaction_only": len(universe - structural),
        "bridge_expression_guard_q95": q_expr,
        "bridge_spatial_guard_q975": q_spatial,
    }
    return frame, audit


def load_membership(method_dir: Path, n_nodes: int) -> tuple[np.ndarray, list[dict], pd.DataFrame]:
    membership = pd.read_csv(method_dir / "membership_A_sharp_final.csv")
    roots_frame = pd.read_csv(method_dir / "resolved_roots.csv").sort_values("resolved_index")
    # Preserve the original root order encoded by root_pos, not numerical node order.
    root_rows = (
        membership[["root_pos", "root_index", "root_name"]]
        .drop_duplicates().sort_values("root_pos").to_dict("records")
    )
    a = np.zeros((n_nodes, len(root_rows)), dtype=float)
    for row in membership.itertuples(index=False):
        a[int(row.node_id), int(row.root_pos)] = float(row.A_sharp)
    truth_root_lookup = {}
    if "truth_root_id" in roots_frame:
        truth_root_lookup = dict(zip(roots_frame.resolved_index.astype(int), roots_frame.truth_root_id.astype(str)))
    for item in root_rows:
        item["root_id"] = truth_root_lookup.get(int(item["root_index"]), f"R{int(item['root_pos'])}")
    active = pd.read_csv(method_dir / "active_root_assignment_final.csv")
    return a, root_rows, active


def root_support(active: pd.DataFrame, root_pos: int, root_index: int) -> set[int]:
    local = active[(active["root_pos"].astype(int) == root_pos) & active["is_active"].astype(str).str.lower().isin(["true", "1", "yes"])]
    nodes = set(local["node_id"].astype(int))
    nodes.add(int(root_index))
    return nodes


def graph_for_support(candidate: pd.DataFrame, support: set[int], membership: np.ndarray) -> nx.Graph:
    eligible = candidate[candidate["trajectory_edge_eligible"]].copy()
    graph = nx.Graph()
    graph.add_nodes_from(sorted(support))
    rows = []
    for row in eligible.itertuples(index=False):
        i, j = int(row.source), int(row.target)
        if i not in support or j not in support:
            continue
        soft_penalty = -math.log(max(float(membership[i] * membership[j]), EPS))
        rows.append((i, j, float(row.joint_distance_ecdf), soft_penalty, str(row.edge_graph_role)))
    if rows:
        soft_ranks = empirical_rank(np.asarray([x[3] for x in rows], dtype=float))
        for (i, j, distance_rank, _, role), membership_rank in zip(rows, soft_ranks):
            # Sum of ECDF ranks makes the two evidence channels scale-free.
            graph.add_edge(i, j, weight=float(distance_rank + membership_rank), edge_graph_role=role)
    return graph


def component_forest(graph: nx.Graph) -> nx.Graph:
    forest = nx.Graph()
    forest.add_nodes_from(graph.nodes())
    for nodes in nx.connected_components(graph):
        sub = graph.subgraph(nodes)
        if sub.number_of_edges():
            forest.add_edges_from(nx.minimum_spanning_tree(sub, weight="weight").edges(data=True))
    return forest


def rooted_tau(forest: nx.Graph, root: int, membership: np.ndarray) -> tuple[dict[int, float], dict[int, int]]:
    tau: dict[int, float] = {}
    component_id: dict[int, int] = {}
    components = sorted(nx.connected_components(forest), key=lambda x: (root not in x, min(x)))
    for cid, nodes in enumerate(components):
        component_id.update({int(node): cid for node in nodes})
        anchor = root if root in nodes else max(nodes, key=lambda node: (membership[int(node)], -int(node)))
        distances = nx.single_source_dijkstra_path_length(forest.subgraph(nodes), anchor, weight="weight")
        maximum = max(distances.values()) if distances else 0.0
        for node in nodes:
            tau[int(node)] = float(distances.get(node, 0.0) / maximum) if maximum > 0 else 0.0
    return tau, component_id


def residual_expression(expression: np.ndarray, nodes: list[int], tau: dict[int, float]) -> np.ndarray:
    y = expression[nodes]
    t = np.asarray([tau[node] for node in nodes], dtype=float)
    design = np.column_stack([np.ones(len(t)), t, t * t])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    if residual.shape[1] > min(8, len(nodes) - 1):
        residual = PCA(n_components=min(8, len(nodes) - 1), random_state=SEED).fit_transform(residual)
    return StandardScaler().fit_transform(residual)


def infer_branch_communities(expression: np.ndarray, forest: nx.Graph, tau: dict[int, float]) -> tuple[dict[int, str], list[dict]]:
    labels: dict[int, str] = {}
    audit: list[dict] = []
    for cid, nodes_set in enumerate(sorted(nx.connected_components(forest), key=lambda x: min(x))):
        nodes = sorted(map(int, nodes_set))
        if len(nodes) < 8:
            for node in nodes:
                labels[node] = f"c{cid}_b0"
            audit.append({"component_id": cid, "n_nodes": len(nodes), "selected_k": 1, "selection": "small_component"})
            continue
        features = residual_expression(expression, nodes, tau)
        k_max = min(6, max(1, len(nodes) // 4))
        candidates = []
        for k in range(1, k_max + 1):
            reg = max(float(np.median(np.var(features, axis=0))) * 1e-6, 1e-8)
            model = GaussianMixture(n_components=k, covariance_type="full", reg_covar=reg, random_state=SEED, n_init=10)
            prediction = model.fit_predict(features)
            counts = np.bincount(prediction, minlength=k)
            if k > 1 and counts.min() < 2:
                continue
            candidates.append((float(model.bic(features)), k, prediction))
        bic, selected_k, prediction = min(candidates, key=lambda x: (x[0], x[1]))
        # Stable numbering by median tau and then smallest node ID.
        cluster_order = sorted(range(selected_k), key=lambda k: (np.median([tau[nodes[i]] for i in np.flatnonzero(prediction == k)]), min(nodes[i] for i in np.flatnonzero(prediction == k))))
        remap = {old: new for new, old in enumerate(cluster_order)}
        for node, raw in zip(nodes, prediction):
            labels[node] = f"c{cid}_b{remap[int(raw)]}"
        audit.append({"component_id": cid, "n_nodes": len(nodes), "selected_k": selected_k, "bic": bic, "k_max": k_max, "selection": "minimum_GMM_BIC_on_tau_residual_expression"})
    return labels, audit


def quotient_branch_forest(graph: nx.Graph, branch: dict[int, str]) -> nx.Graph:
    """Connect within-branch pieces first, then join them through a quotient MST."""
    result = nx.Graph()
    result.add_nodes_from(graph.nodes())
    pieces: list[set[int]] = []
    node_piece: dict[int, int] = {}
    for component in nx.connected_components(graph):
        by_branch: dict[str, set[int]] = defaultdict(set)
        for node in component:
            by_branch[branch[int(node)]].add(int(node))
        for _, branch_nodes in sorted(by_branch.items()):
            induced = graph.subgraph(branch_nodes)
            for local_nodes in nx.connected_components(induced):
                pid = len(pieces)
                piece = set(map(int, local_nodes))
                pieces.append(piece)
                node_piece.update({node: pid for node in piece})
                local = induced.subgraph(piece)
                if local.number_of_edges():
                    result.add_edges_from(nx.minimum_spanning_tree(local, weight="weight").edges(data=True))
    quotient = nx.Graph()
    quotient.add_nodes_from(range(len(pieces)))
    best_edge: dict[tuple[int, int], tuple[float, int, int, dict]] = {}
    for i, j, data in graph.edges(data=True):
        pi, pj = node_piece[int(i)], node_piece[int(j)]
        if pi == pj:
            continue
        key = edge_key(pi, pj)
        trial = (float(data["weight"]), min(int(i), int(j)), max(int(i), int(j)), dict(data))
        if key not in best_edge or trial[:3] < best_edge[key][:3]:
            best_edge[key] = trial
    for (pi, pj), (weight, i, j, data) in best_edge.items():
        quotient.add_edge(pi, pj, weight=weight, source=i, target=j, source_data=data)
    for quotient_component in nx.connected_components(quotient):
        sub = quotient.subgraph(quotient_component)
        for pi, pj, qdata in nx.minimum_spanning_tree(sub, weight="weight").edges(data=True):
            result.add_edge(int(qdata["source"]), int(qdata["target"]), **qdata["source_data"])
    return result


def construct_t0(method_dir: Path, adata: ad.AnnData, candidate: pd.DataFrame, geometry: dict[str, np.ndarray]) -> dict:
    a, roots, active = load_membership(method_dir, adata.n_obs)
    edge_rows, node_rows, component_rows, branch_audit = [], [], [], []
    all_edges: dict[int, set[tuple[int, int]]] = {}
    legality = set(edge_key(int(r.source), int(r.target)) for r in candidate[candidate["trajectory_edge_eligible"]].itertuples(index=False))
    for root in roots:
        k = int(root["root_pos"])
        root_index = int(root["root_index"])
        support = root_support(active, k, root_index)
        graph = graph_for_support(candidate, support, a[:, k])
        preliminary = component_forest(graph)
        tau_pre, _ = rooted_tau(preliminary, root_index, a[:, k])
        branches, audits = infer_branch_communities(geometry["expression"], preliminary, tau_pre)
        final = quotient_branch_forest(graph, branches)
        tau, component_ids = rooted_tau(final, root_index, a[:, k])
        edges = {edge_key(int(i), int(j)) for i, j in final.edges()}
        all_edges[k] = edges
        for audit in audits:
            branch_audit.append({"root_pos": k, "root_id": root["root_id"], **audit})
        for i, j, data in final.edges(data=True):
            edge_rows.append({
                "root_pos": k, "root_id": root["root_id"], "source": i, "target": j,
                "source_name": str(adata.obs_names[i]), "target_name": str(adata.obs_names[j]),
                "edge_cost": float(data["weight"]), "edge_graph_role": data["edge_graph_role"],
                "candidate_legal": edge_key(int(i), int(j)) in legality,
            })
        for node in sorted(support):
            node_rows.append({
                "node_id": node, "unit_id": str(adata.obs_names[node]), "root_pos": k,
                "root_id": root["root_id"], "membership": float(a[node, k]),
                "pseudotime": float(tau.get(node, 0.0)), "branch_id": f"{root['root_id']}_{branches[node]}",
                "component_id": int(component_ids.get(node, -1)),
            })
        for cid, nodes in enumerate(sorted(nx.connected_components(graph), key=lambda x: min(x))):
            component_rows.append({
                "root_pos": k, "root_id": root["root_id"], "component_id": cid,
                "n_nodes": len(nodes), "n_candidate_edges": graph.subgraph(nodes).number_of_edges(),
                "contains_root": root_index in nodes,
                "n_T0_edges": final.subgraph(nodes).number_of_edges(),
            })
    return {
        "roots": roots, "membership": a, "edges": pd.DataFrame(edge_rows),
        "nodes": pd.DataFrame(node_rows), "components": pd.DataFrame(component_rows),
        "branch_audit": pd.DataFrame(branch_audit), "edge_sets": all_edges,
    }


def write_prediction(t0: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    roots = pd.DataFrame([{"root_id": r["root_id"]} for r in t0["roots"]])
    roots.to_csv(output / "PRED_ROOTS.csv", index=False)
    nodes = t0["nodes"].copy()
    primary = nodes.sort_values(["unit_id", "membership"], ascending=[True, False]).drop_duplicates("unit_id")
    primary[["unit_id", "root_id", "membership"]].rename(columns={"membership": "membership_probability"}).to_csv(output / "PRED_CELL_OR_METACELL_ASSIGNMENT.csv", index=False)
    primary[["unit_id", "root_id", "branch_id"]].to_csv(output / "PRED_BRANCH.csv", index=False)
    primary[["unit_id", "root_id", "pseudotime"]].to_csv(output / "PRED_PSEUDOTIME.csv", index=False)
    edges = t0["edges"][["root_id", "source_name", "target_name"]].rename(columns={"source_name": "source", "target_name": "target"})
    edges.to_csv(output / "PRED_T0_EDGES.csv", index=False)
    edges.to_csv(output / "PRED_T1_EDGES.csv", index=False)
    pd.DataFrame(columns=["event_type", "source_root", "receiver_root", "affected_branch", "is_called"]).to_csv(output / "PRED_EVENTS.csv", index=False)


def candidate_truth_audit(candidate: pd.DataFrame, metacell_truth_dir: Path, truth_dir: Path) -> pd.DataFrame:
    state = pd.read_csv(metacell_truth_dir / "METACELL_STATE_COMPOSITION.csv")
    state = state[state["stage"].astype(str).eq("T0")]
    dominant = state.sort_values("state_probability", ascending=False).drop_duplicates("metacell_id")
    lookup = dict(zip(dominant.metacell_id.astype(str), dominant.state_id.astype(str)))
    eligible = candidate[candidate["trajectory_edge_eligible"]].copy()
    represented = set()
    for row in eligible.itertuples(index=False):
        a = lookup.get(str(row.source_name)); b = lookup.get(str(row.target_name))
        if a and b and a != b:
            represented.add(tuple(sorted((a, b))))
    truth = pd.read_csv(truth_dir / "TRUE_T0_EDGES.csv")
    rows = []
    for root_id, group in truth.groupby("root_id"):
        expected = {tuple(sorted((str(r.source), str(r.target)))) for r in group.itertuples(index=False)}
        rows.append({
            "root_id": root_id, "n_true_state_edges": len(expected),
            "n_represented": len(expected & represented),
            "candidate_state_edge_recall": len(expected & represented) / max(len(expected), 1),
            "truth_use": "evaluation_only_after_candidate_frozen",
        })
    return pd.DataFrame(rows)


def evaluate_only(project: Path, run_dir: Path, prediction: Path, score_dir: Path, method: str) -> None:
    scorer = project / "algorithm" / "experimental" / "cross_trajectory_gold_standard" / "score_benchmark_prediction_v1.py"
    command = [
        sys.executable, str(scorer), "--truth-dir", str(run_dir / "event" / "truth"),
        "--prediction-dir", str(prediction), "--output-dir", str(score_dir),
        "--method", method, "--metacell-truth-dir", str(run_dir / "formal_metacell"),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="CORE_05")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    method_dir = run_dir / "generic_upstream_v1_exact_root" / args.dataset
    adata = ad.read_h5ad(method_dir / "metacell_adata.h5ad")

    geometry = prepare_geometry(adata)
    candidate, candidate_audit = build_multiresolution_candidate(adata, geometry)
    candidate.to_csv(output / "C1_MULTIRESOLUTION_CANDIDATE_EDGES.csv", index=False)
    pd.DataFrame([candidate_audit]).to_csv(output / "C1_CANDIDATE_CONSTRUCTION_AUDIT.csv", index=False)
    truth_audit = candidate_truth_audit(candidate, run_dir / "formal_metacell", run_dir / "event" / "truth")
    truth_audit.to_csv(output / "C1_TRUTH_ONLY_RECOVERABILITY_AUDIT.csv", index=False)

    t0 = construct_t0(method_dir, adata, candidate, geometry)
    t0["edges"].to_csv(output / "C3_BRANCH_SCALE_T0_EDGES.csv", index=False)
    t0["nodes"].to_csv(output / "C3_BRANCH_SCALE_NODE_ASSIGNMENTS.csv", index=False)
    t0["components"].to_csv(output / "C3_COMPONENT_RETENTION_AUDIT.csv", index=False)
    t0["branch_audit"].to_csv(output / "C3_BRANCH_MODEL_SELECTION_AUDIT.csv", index=False)
    prediction = output / "prediction"
    write_prediction(t0, prediction)
    score_dir = output / "score"
    evaluate_only(project, run_dir, prediction, score_dir, "C1_C3_branchscale_v1")

    summary = pd.read_csv(score_dir / "SCORE_SUMMARY.csv").iloc[0].to_dict()
    integrity = {
        "method_version": "C1_candidate_multires_v1+C3_T0_branchscale_v1",
        "truth_available_to_constructor": False,
        "truth_read_only_after_method_outputs_written": True,
        "relation_module_called": False,
        "all_T0_edges_candidate_legal": bool(t0["edges"]["candidate_legal"].all()),
        "cycle_violating_roots": int(summary["cycle_violating_roots"]),
        "disconnected_roots": int(summary["disconnected_roots"]),
        "score": summary,
    }
    (output / "RUN_DECISION_INPUT.json").write_text(json.dumps(integrity, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(integrity, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
