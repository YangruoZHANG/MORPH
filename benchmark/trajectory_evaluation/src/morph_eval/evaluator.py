"""Evaluation pipeline over the frozen simulation and native predictions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .metrics import (
    branch_f1,
    cordist_score,
    fate_lineage_sets,
    graph_lineage_sets,
    maximal_segments,
    quotient_to_truth,
    safe_spearman,
    set_mapping_f1,
    simplify_graph,
    topology_one_minus_him,
    truth_lineage_sets,
    truth_segment_labels,
    weighted_rank_correlation,
)

METHODS = [
    "NF_v70",
    "stLearn",
    "Palantir",
    "scFates",
    "Slingshot",
    "Monocle3",
    "PAGA_DPT",
    "CellRank_DPT",
    "TSCAN",
]
DISPLAY = {
    "NF_v70": "MORPH",
    "stLearn": "stL",
    "Palantir": "Pal",
    "scFates": "scF",
    "Slingshot": "Sli",
    "Monocle3": "Mon",
    "PAGA_DPT": "PAG",
    "CellRank_DPT": "CeR",
    "TSCAN": "TSC",
}
METRICS = ["coverage", "pt", "cordist", "branch_f1", "topo_1_him", "feat_imp"]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_graph(path: Path) -> nx.Graph:
    graph = nx.Graph()
    if not path.exists():
        return graph
    table = pd.read_csv(path, dtype={"source": str, "target": str})
    if not {"source", "target"}.issubset(table.columns):
        return graph
    for row in table.itertuples(index=False):
        length = getattr(row, "length", 1.0)
        length = float(length) if pd.notna(length) and float(length) > 0 else 1.0
        if str(row.source) != str(row.target):
            graph.add_edge(str(row.source), str(row.target), length=length)
    return graph


def truth_milestone_targets(truth: pd.DataFrame, graph: nx.Graph) -> np.ndarray:
    nodes = sorted(graph.nodes, key=str)
    distance = dict(nx.all_pairs_dijkstra_path_length(graph, weight="length"))
    edge_length = {(str(u), str(v)): float(d.get("length", 1.0)) for u, v, d in graph.edges(data=True)}
    edge_length.update({(v, u): x for (u, v), x in list(edge_length.items())})
    target = np.empty((len(truth), len(nodes)), float)
    for i, (u0, v0, f0) in enumerate(zip(truth["from"], truth["to"], truth["fraction"])):
        u, v, f = str(u0), str(v0), float(f0)
        length = edge_length[(u, v)]
        for j, milestone in enumerate(nodes):
            target[i, j] = min(f * length + distance[u][milestone], (1 - f) * length + distance[v][milestone])
    return target


def prediction_targets(cells: pd.DataFrame, graph: nx.Graph, pseudotime: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    nodes = cells["node"].to_numpy(object) if "node" in cells else np.full(len(cells), None, object)
    node_ok = np.asarray([pd.notna(x) and str(x) in graph for x in nodes])
    valid_pt = np.isfinite(pseudotime)
    if graph.number_of_edges() and node_ok[valid_pt].mean() >= 0.5:
        anchors = [str(v) for v, d in graph.degree if d != 2]
        if not anchors:
            anchors = sorted((str(v) for v in graph.nodes))[:2]
        dist = dict(nx.all_pairs_dijkstra_path_length(graph, weight="length"))
        target = np.asarray([[dist.get(str(node), {}).get(anchor, np.nan) for anchor in anchors] for node in nodes])
        valid = valid_pt & np.isfinite(target).all(axis=1)
        return target, valid, "native_graph_milestone_distances"
    target = pseudotime[:, None]
    return target, valid_pt, "pseudotime_fallback"


def rf_importance(expression: np.ndarray, target: np.ndarray, valid: np.ndarray, seed: int) -> np.ndarray:
    valid = np.asarray(valid, bool) & np.isfinite(target).all(axis=1) & np.isfinite(expression).all(axis=1)
    if valid.sum() < 20:
        return np.full(expression.shape[1], np.nan)
    ids = np.flatnonzero(valid)
    if len(ids) > 250:
        ids = np.random.default_rng(seed).choice(ids, 250, replace=False)
    y = target[ids]
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    model = RandomForestRegressor(
        n_estimators=48,
        max_features=max(1, int(round(expression.shape[1] * 0.01))),
        min_samples_leaf=5,
        bootstrap=True,
        random_state=seed,
        n_jobs=2,
    )
    model.fit(expression[ids], y)
    return model.feature_importances_


def prediction_branches(cells: pd.DataFrame, graph: nx.Graph, root_cell: str) -> tuple[np.ndarray, str]:
    node = cells["node"].to_numpy(object) if "node" in cells else np.full(len(cells), None, object)
    root_node = None
    if root_cell in cells.index and "node" in cells and pd.notna(cells.loc[root_cell, "node"]):
        root_node = str(cells.loc[root_cell, "node"])
    segments = maximal_segments(graph, root_node)
    if segments:
        labels = np.asarray([segments.get(str(x), None) if pd.notna(x) else None for x in node], object)
        if pd.notna(labels).sum() >= 5:
            return labels, "maximal_nonbranching_graph_segments"
    if "native_branch" in cells:
        raw = cells["native_branch"].astype(object)
        missing = raw.isna() | raw.astype(str).str.lower().isin(["", "nan", "none", "na"])
        informative = raw.loc[~missing].astype(str)
        if len(informative) >= 5:
            # Fate-only methods leave the common progenitor unlabeled. Treating it as
            # an explicit trunk is an output interpretation, not truth imputation.
            labels = raw.astype(object).to_numpy()
            labels[missing.to_numpy()] = "__trunk__"
            return labels, "native_branch_plus_explicit_unassigned_trunk"
    return np.full(len(cells), None, object), "unavailable"


def derived_fate_star(cells: pd.DataFrame, pseudotime: np.ndarray) -> tuple[nx.Graph, np.ndarray]:
    """Build a declared topology surrogate from fate/macrostate labels.

    Palantir and CellRank return probabilistic terminal-state information rather than
    a unique tree. For visualization/completeness only, represent that native output
    as one common early trunk connected to the observed terminal labels. No truth
    information is used. This graph is used only by Topo 1-HIM.
    """
    graph = nx.Graph()
    assigned = np.full(len(cells), "__trunk__", dtype=object)
    if "native_branch" not in cells:
        return graph, assigned
    raw = cells["native_branch"].astype(object)
    missing = raw.isna() | raw.astype(str).str.lower().isin(["", "nan", "none", "na"])
    labels = sorted(raw.loc[~missing].astype(str).unique())
    if not labels:
        return graph, assigned
    valid_pt = np.isfinite(pseudotime)
    early = np.zeros(len(cells), bool)
    if valid_pt.any():
        early = valid_pt & (pseudotime <= np.nanquantile(pseudotime[valid_pt], 0.20))
    assigned[missing.to_numpy() | early] = "__trunk__"
    for label in labels:
        node = f"fate::{label}"
        graph.add_edge("__trunk__", node, length=1.0)
        mask = (~missing).to_numpy() & (~early) & raw.astype(str).eq(label).to_numpy()
        assigned[mask] = node
    return graph, assigned


def evaluate_partition(source: Path, row, method: str, cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    case_dir = source / "data" / row.case
    truth_all = pd.read_csv(case_dir / "truth.csv", index_col=0, dtype={"origin": str})
    truth = truth_all.loc[truth_all["origin"].eq(str(row.root_index))].copy()
    truth_edges = pd.read_csv(case_dir / "truth_edges.csv", dtype={"source": str, "target": str})
    prefix = f"O{row.root_index}_"
    truth_edges = truth_edges.loc[truth_edges["source"].str.startswith(prefix)]
    truth_graph = nx.Graph()
    for edge in truth_edges.itertuples(index=False):
        truth_graph.add_edge(str(edge.source), str(edge.target), length=float(edge.length))

    pred_dir = source / "results" / "predictions" / row.part / method
    status_path = source / "results" / "status" / f"{row.part}__{method}.json"
    status = json.loads(status_path.read_text(encoding="utf-8-sig")) if status_path.exists() else {"status": "NOT_RUN"}
    result = {
        "case": row.case,
        "condition": row.condition,
        "replicate": int(row.replicate),
        "root_index": int(row.root_index),
        "method_id": method,
        "method": DISPLAY[method],
        "status": status.get("status", "NOT_RUN"),
    }
    if result["status"] != "OK":
        return result

    cells = pd.read_csv(pred_dir / "cells.csv", dtype={"cell_id": str, "node": str, "native_branch": str}).set_index("cell_id")
    cells = cells.reindex(truth.index)
    pseudotime = pd.to_numeric(cells["pseudotime"], errors="coerce").to_numpy(float)
    valid_pt = np.isfinite(pseudotime)
    pred_graph = read_graph(pred_dir / "edges.csv")
    nodes = cells["node"].to_numpy(object) if "node" in cells else np.full(len(cells), None, object)
    seed = 20261123 + int(row.simulation_seed) % 100000 + int(row.root_index) * 101 + METHODS.index(method) * 1009

    result["coverage"] = float(valid_pt.mean())
    result["pt"] = safe_spearman(pseudotime[valid_pt], truth["time"].to_numpy()[valid_pt])
    result["cordist"], result["cordist_pair_coverage"], result["cordist_mode"] = cordist_score(
        truth, truth_graph, pseudotime, pred_graph, nodes, seed
    )

    root_state = str(truth.loc[row.root, "from"])
    truth_branch = truth_segment_labels(truth, truth_graph, root_state)
    pred_branch, result["branch_mode"] = prediction_branches(cells, pred_graph, str(row.root))
    # Keep the strict edge-segment mapping as an audit metric. The headline Branch
    # F1 uses root-to-leaf lineage sets, allowing the shared trunk to belong to all
    # downstream fates (the natural semantics of multi-origin trajectory forests).
    (
        result["segment_f1_audit"],
        result["segment_recovery_audit"],
        result["segment_relevance_audit"],
        result["segment_coverage_audit"],
    ) = branch_f1(truth_branch, pred_branch)
    root_node = None
    if str(row.root) in cells.index and "node" in cells and pd.notna(cells.loc[str(row.root), "node"]):
        root_node = str(cells.loc[str(row.root), "node"])
    ref_lineages = truth_lineage_sets(truth, truth_graph, root_state)
    pred_lineages = graph_lineage_sets(pred_graph, root_node, nodes)
    if pred_lineages:
        result["branch_mode"] = "overlapping_root_to_leaf_native_graph_lineages"
    elif "native_branch" in cells:
        pred_lineages = fate_lineage_sets(cells["native_branch"].to_numpy(object))
        result["branch_mode"] = "overlapping_fate_lineages_plus_shared_trunk" if pred_lineages else "unavailable"
    (
        result["branch_f1"],
        result["branch_recovery"],
        result["branch_relevance"],
        result["branch_coverage"],
    ) = set_mapping_f1(ref_lineages, pred_lineages, len(truth))

    topology_graph = pred_graph
    topology_nodes = nodes
    result["topology_mode"] = "native_graph"
    if not topology_graph.number_of_edges():
        topology_graph, topology_nodes = derived_fate_star(cells, pseudotime)
        result["topology_mode"] = "derived_fate_star" if topology_graph.number_of_edges() else "unavailable"
    quotient = quotient_to_truth(truth, topology_graph, topology_nodes) if topology_graph.number_of_edges() else nx.Graph()
    result["topo_1_him"] = topology_one_minus_him(truth_graph, quotient)
    result["topology_available"] = bool(np.isfinite(result["topo_1_him"]))

    expression = pd.read_csv(case_dir / f"R{row.root_index}" / "expr_log.csv", index_col=0).reindex(truth.index)
    x = expression.to_numpy(float)
    ref_cache = cache / f"{row.part}__truth_feature_importance.npy"
    if ref_cache.exists():
        ref_importance = np.load(ref_cache)
    else:
        truth_target = truth_milestone_targets(truth, truth_graph)
        ref_importance = rf_importance(x, truth_target, np.ones(len(truth), bool), seed=seed - METHODS.index(method) * 1009)
        np.save(ref_cache, ref_importance)
    pred_target, pred_valid, result["feature_target_mode"] = prediction_targets(cells, pred_graph, pseudotime)
    pred_cache = cache / f"{row.part}__{method}__feature_importance.npy"
    if pred_cache.exists():
        pred_importance = np.load(pred_cache)
    else:
        pred_importance = rf_importance(x, pred_target, pred_valid, seed=seed)
        np.save(pred_cache, pred_importance)
    result["feat_imp"] = weighted_rank_correlation(ref_importance, pred_importance)
    return result


def bootstrap_summary(by_dataset: pd.DataFrame, seed: int = 20261123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (method, method_id), group in by_dataset.groupby(["method", "method_id"], sort=False):
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(float)
            if len(values):
                boot = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
                mean, lo, hi = float(values.mean()), *np.quantile(boot, [0.025, 0.975])
            else:
                mean = lo = hi = np.nan
            rows.append({
                "method": method,
                "method_id": method_id,
                "metric": metric,
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_valid": len(values),
                "n_planned": len(group),
            })
    return pd.DataFrame(rows)


def write_result_tables(by_root: pd.DataFrame, result_dir: Path) -> None:
    """Write all-condition audit tables and positive/null headline summaries."""
    metadata = ["case", "condition", "replicate", "method", "method_id"]
    diagnostics = ["cordist_pair_coverage", "branch_coverage", "topology_available"]
    agg_cols = [x for x in METRICS + diagnostics if x in by_root]
    by_dataset = by_root.groupby(metadata, sort=False)[agg_cols].mean().reset_index()
    by_dataset.to_csv(result_dir / "by_dataset.csv", index=False, encoding="utf-8-sig")

    # Preserve an all-condition summary for audit. The headline accuracy panel excludes
    # spatial_null because it is a rejection diagnostic, not a trajectory-recovery task.
    summary_all = bootstrap_summary(by_dataset)
    summary_all.to_csv(result_dir / "summary_all_conditions_long.csv", index=False, encoding="utf-8-sig")
    positive = by_dataset.loc[by_dataset["condition"].ne("spatial_null")].copy()
    summary = bootstrap_summary(positive)
    summary.to_csv(result_dir / "summary_long.csv", index=False, encoding="utf-8-sig")

    order = [DISPLAY[m] for m in METHODS]
    table = summary.pivot(index="method", columns="metric", values="mean").reindex(order)
    table = table.rename(columns={
        "coverage": "Coverage",
        "pt": "PT",
        "cordist": "Cordist",
        "branch_f1": "Branch F1",
        "topo_1_him": "Topo 1-HIM",
        "feat_imp": "Feat-imp",
    })[["Coverage", "PT", "Cordist", "Branch F1", "Topo 1-HIM", "Feat-imp"]]
    table.to_csv(result_dir / "manuscript_table.csv", encoding="utf-8-sig", float_format="%.4f")

    null = by_dataset.loc[by_dataset["condition"].eq("spatial_null")]
    negative = null.groupby(["method", "method_id"], sort=False)["coverage"].agg(["mean", "std", "count"]).reset_index()
    negative = negative.rename(columns={"mean": "output_coverage", "std": "coverage_sd", "count": "n_datasets"})
    negative["rejection_rate"] = 1.0 - negative["output_coverage"]
    negative = negative.set_index("method").reindex(order).reset_index()
    negative.to_csv(result_dir / "spatial_null_negative_control.csv", index=False, encoding="utf-8-sig", float_format="%.6f")


def run(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    cache = output / "results" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(source / "data" / "partitions.csv")
    rows = []
    for part_i, row in enumerate(registry.itertuples(index=False), start=1):
        for method in METHODS:
            scored = evaluate_partition(source, row, method, cache)
            rows.append(scored)
            print(f"[{part_i:02d}/{len(registry):02d}] {row.part:28s} {DISPLAY[method]:5s} {scored['status']}", flush=True)
    by_root = pd.DataFrame(rows)
    result_dir = output / "results"
    result_dir.mkdir(exist_ok=True)
    by_root.to_csv(result_dir / "by_root.csv", index=False, encoding="utf-8-sig")

    write_result_tables(by_root, result_dir)

    source_files = [source / "code" / "prepare.py", source / "data" / "partitions.csv"]
    provenance = {
        "scorer_version": "1.0.0",
        "frozen_source": str(source),
        "simulation_reused_without_modification": True,
        "simulation_source_sha256": {str(p.relative_to(source)): file_sha256(p) for p in source_files},
        "methods": METHODS,
        "display_names": DISPLAY,
        "n_root_partitions": int(len(registry)),
        "n_datasets": int(registry["case"].nunique()),
    }
    (result_dir / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return provenance

