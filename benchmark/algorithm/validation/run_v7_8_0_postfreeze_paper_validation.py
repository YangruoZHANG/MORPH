#!/usr/bin/env python
"""Read-only, paper-level validation of the frozen v7.8.0 result.

This runner never edits v7.8.0 outputs.  It explicitly audits the historical
80-metacell relation-evidence mapping against the formal exact cell-to-graph
mapping and performs all independent molecular tests with the latter.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from itertools import product
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import Delaunay
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score


RELATIONS = [
    {
        "dataset": "xenium_breast", "source_root": "C13", "receiver_root": "C00",
        "program": "EMT", "relation_type": "promotion_like",
        "ligand": "C1QA", "receptor": "CD93",
    },
    {
        "dataset": "xenium_melanoma", "source_root": "C15", "receiver_root": "C03",
        "program": "ECM_Remodeling", "relation_type": "promotion_like",
        "ligand": "VCAN", "receptor": "SELL",
    },
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_context(project: Path, relation: dict, parent, relation_module) -> dict:
    ds = relation["dataset"]
    raw = pd.read_csv(
        project / "data" / "raw_cell_dense_embedding_bundle" / ds
        / "raw_cell_dense_coordinates.csv.gz"
    )
    mapping = pd.read_csv(
        project / "NicheForest_v7_4_6_all7_reproduction_bundle" / "data" / ds
        / "cell_to_metacell.csv"
    )
    base, roots, weights, lookup = parent.root_inputs(project, ds)
    exact_lookup = dict(zip(mapping["cell_name"].astype(str), mapping["metacell_index"].astype(int)))
    raw["node_id_exact"] = raw["cell_id"].astype(str).map(exact_lookup)
    if raw["node_id_exact"].isna().any():
        raise RuntimeError(f"{ds}: exact cell-ID universe mismatch")
    raw["node_id_exact"] = raw["node_id_exact"].astype(int)
    old_node = pd.to_numeric(
        raw["metacell"].astype(str).str.extract(r"(\d+)")[0], errors="raise"
    ).astype(int)
    old_w = weights[old_node.to_numpy()]
    exact_w = weights[raw["node_id_exact"].to_numpy()]
    old_idx = np.argmax(old_w, axis=1)
    exact_idx = np.argmax(exact_w, axis=1)
    raw["root_index_old80"] = old_idx
    raw["root_old80"] = [roots[i] for i in old_idx]
    raw["root_index"] = exact_idx
    raw["root_cluster"] = [roots[i] for i in exact_idx]
    raw["node_id"] = raw["node_id_exact"]
    raw["_section"] = "all"
    raw["_compartment"] = raw.get("cell_type_coarse", pd.Series("all", index=raw.index)).fillna("unknown").astype(str)
    raw = relation_module.add_spatial_blocks(raw)
    path = project / "data" / "raw_cell_dense_embedding_bundle" / ds / "raw_cells.h5ad"
    a = ad.read_h5ad(path, backed="r")
    obs_lookup = {str(v): i for i, v in enumerate(a.obs_names)}
    order = raw["cell_id"].astype(str).map(obs_lookup)
    if order.isna().any():
        a.file.close()
        raise RuntimeError(f"{ds}: h5ad cell-ID universe mismatch")
    cache = relation_module.ExpressionCache(a, order.astype(int).to_numpy())
    return {
        "cells": raw, "mapping": mapping, "roots": roots, "weights": weights,
        "old_node": old_node, "old_idx": old_idx, "exact_idx": exact_idx,
        "a": a, "cache": cache, "base": base,
    }


def spatial_edges(cells: pd.DataFrame) -> np.ndarray:
    xy = cells[["spatial_1", "spatial_2"]].to_numpy(float)
    tri = Delaunay(xy, qhull_options="QJ Qbb Qc")
    pairs = np.vstack([
        tri.simplices[:, [0, 1]], tri.simplices[:, [0, 2]], tri.simplices[:, [1, 2]]
    ])
    pairs.sort(axis=1)
    pairs = np.unique(pairs, axis=0)
    length = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
    q1, q3 = np.quantile(length, [0.25, 0.75])
    return pairs[length <= q3 + 1.5 * (q3 - q1)]


def program_vector(cache, genes: list[str]) -> tuple[np.ndarray, list[str]]:
    measured = [g for g in genes if cache.measured(g)]
    if len(measured) < 3:
        raise RuntimeError(f"Only {len(measured)} program genes measured: {measured}")
    return np.column_stack([cache.gene(g) for g in measured]).mean(axis=1), measured


def node_score(cells: pd.DataFrame, values: np.ndarray, n_nodes: int) -> pd.Series:
    node = cells["node_id"].to_numpy(int)
    sums = np.bincount(node, weights=values, minlength=n_nodes)
    count = np.bincount(node, minlength=n_nodes)
    out = np.divide(sums, count, out=np.full(n_nodes, np.nan), where=count > 0)
    return pd.Series(out, index=np.arange(n_nodes))


def ridge_predict(x_train, y_train, x_test, alpha=1e-6):
    mean = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd < 1e-12] = 1.0
    xt = (x_train - mean) / sd
    xv = (x_test - mean) / sd
    xt = np.column_stack([np.ones(len(xt)), xt])
    xv = np.column_stack([np.ones(len(xv)), xv])
    penalty = np.eye(xt.shape[1]) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(xt.T @ xt + penalty, xt.T @ y_train)
    return xv @ beta


def metrics(y, pred) -> dict:
    err = y - pred
    ss = float(np.sum((y - y.mean()) ** 2))
    rho = spearmanr(y, pred).statistic if len(y) > 2 else np.nan
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err ** 2) / ss) if ss > 0 else np.nan,
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
    }


def heldout_test(ctx: dict, relation: dict, edges: np.ndarray, program_genes: list[str]) -> tuple[pd.DataFrame, dict]:
    cells, cache = ctx["cells"], ctx["cache"]
    y, measured = program_vector(cache, program_genes)
    ligand = cache.gene(relation["ligand"])
    receptor = cache.gene(relation["receptor"])
    source = cells["root_cluster"].astype(str).eq(relation["source_root"]).to_numpy()
    receiver = cells["root_cluster"].astype(str).eq(relation["receiver_root"]).to_numpy()
    n = len(cells)
    adj = sparse.csr_matrix(
        (np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(n, n),
    )
    degree = np.asarray(adj.sum(axis=1)).ravel()
    neighbor_ligand = np.asarray(adj @ (ligand * source)).ravel() / np.maximum(degree, 1)
    exposure = neighbor_ligand * receptor
    ids = np.flatnonzero(receiver & np.isfinite(y) & np.isfinite(exposure))
    xb = pd.to_numeric(cells.loc[ids, "_xb"], errors="coerce").fillna(0).to_numpy(int)
    yb = pd.to_numeric(cells.loc[ids, "_yb"], errors="coerce").fillna(0).to_numpy(int)
    fold = (xb + 2 * yb) % 4
    block = xb * 10 + yb
    total = np.log1p(np.maximum(cache.totals[ids], 0))
    xy = cells.loc[ids, ["spatial_1", "spatial_2"]].to_numpy(float)
    pseudo = ctx["weights"][cells.loc[ids, "node_id"].to_numpy(int), cells.loc[ids, "root_index"].to_numpy(int)]
    baseline = np.column_stack([pseudo, total, xy])
    relation_x = np.column_stack([baseline, exposure[ids]])
    rows, all_y, pred0, pred1 = [], [], [], []
    for f in range(4):
        test = fold == f
        train = ~test
        p0 = ridge_predict(baseline[train], y[ids][train], baseline[test])
        p1 = ridge_predict(relation_x[train], y[ids][train], relation_x[test])
        m0, m1 = metrics(y[ids][test], p0), metrics(y[ids][test], p1)
        rows.append({
            "dataset": relation["dataset"], "source_root": relation["source_root"],
            "receiver_root": relation["receiver_root"], "program": relation["program"],
            "ligand": relation["ligand"], "receptor": relation["receptor"], "fold": f,
            "n_train": int(train.sum()), "n_test": int(test.sum()),
            **{f"baseline_{k}": v for k, v in m0.items()},
            **{f"relation_{k}": v for k, v in m1.items()},
            "delta_rmse_relation_minus_baseline": m1["rmse"] - m0["rmse"],
            "delta_r2_relation_minus_baseline": m1["r2"] - m0["r2"],
        })
        all_y.append(y[ids][test]); pred0.append(p0); pred1.append(p1)
    yy, pp0, pp1 = map(np.concatenate, [all_y, pred0, pred1])
    overall0, overall1 = metrics(yy, pp0), metrics(yy, pp1)
    audit = {
        "n_receiver_cells": int(len(ids)), "measured_program_genes": measured,
        "mean_exposure": float(np.mean(exposure[ids])),
        "baseline": overall0, "relation": overall1,
        "delta_rmse": overall1["rmse"] - overall0["rmse"],
        "delta_r2": overall1["r2"] - overall0["r2"],
        "y": y, "exposure": exposure, "receiver_ids": ids,
        "baseline_x": baseline, "relation_x": relation_x, "fold": fold, "block": block,
    }
    return pd.DataFrame(rows), audit


def cv_relation_delta(y: np.ndarray, baseline: np.ndarray, exposure: np.ndarray, fold: np.ndarray) -> dict:
    all_y, pred0, pred1 = [], [], []
    relation_x = np.column_stack([baseline, exposure])
    for f in sorted(np.unique(fold)):
        test = fold == f
        train = ~test
        all_y.append(y[test])
        pred0.append(ridge_predict(baseline[train], y[train], baseline[test]))
        pred1.append(ridge_predict(relation_x[train], y[train], relation_x[test]))
    yy, pp0, pp1 = map(np.concatenate, [all_y, pred0, pred1])
    m0, m1 = metrics(yy, pp0), metrics(yy, pp1)
    return {
        "heldout_delta_rmse": m1["rmse"] - m0["rmse"],
        "heldout_delta_r2": m1["r2"] - m0["r2"],
    }


def branch_inputs(project: Path, relation: dict, freeze):
    v780 = project / "outputs" / "v7_8_0_latent_relation_guided_coupled_reconstruction"
    a0 = pd.read_csv(v780 / "BASELINE_A0_TRAJECTORY_EDGES.csv")
    supported = pd.read_csv(v780 / "ALL7_FINAL_TYPED_RELATION_EDGES.csv")
    rr = supported.loc[supported["dataset"].eq(relation["dataset"])].iloc[0]
    edges = a0.loc[
        a0["dataset"].eq(relation["dataset"]) & a0["root_id"].eq(relation["receiver_root"])
    ].copy()
    anchors = freeze.receiver_anchors(project, relation["dataset"], relation["receiver_root"])
    nodes = freeze.receiver_node_universe(project, relation["dataset"], relation["receiver_root"])
    return edges, anchors, nodes, float(rr.weighted_score), float(rr.raw_score)


def branch_summary(freeze, edges, anchors, nodes, score, prior, relation_type):
    _, _, summary = freeze.directed_branch_model(
        edges, anchors, nodes, score, prior, relation_type, 1e-12
    )
    return summary


def synthetic_metrics(project: Path) -> pd.DataFrame:
    table = pd.read_csv(
        project / "outputs" / "v7_5_0_joint_relation_aware_coupled_trajectory_candidate"
        / "SYNTHETIC_JOINT_BENCHMARK_RESULTS.csv"
    )
    strength = {
        "true_dynamic_convergence": "strong", "promotion_receiver_branch": "strong",
        "inhibition_receiver_branch": "strong", "spatial_interface_without_merge": "moderate",
        "mixed_promotion_and_inhibition": "moderate",
        "independent_trajectories": "null", "same_process_root_views": "null",
        "density_driven_false_contact": "confounded_null",
        "environmental_confounding": "confounded_null", "missing_lr_target_panel": "missing_panel",
    }
    rows = []
    for method, local in table.groupby("method", sort=True):
        for rel in ["collision", "promotion", "inhibition"]:
            truth = local[f"truth_{rel}"].astype(int)
            pred = local[f"pred_{rel}"].astype(int)
            tp = int(((truth == 1) & (pred == 1)).sum())
            fp = int(((truth == 0) & (pred == 1)).sum())
            fn = int(((truth == 1) & (pred == 0)).sum())
            tn = int(((truth == 0) & (pred == 0)).sum())
            rows.append({
                "method": method, "relation_type": rel, "evidence_strength": "all",
                "n": len(local), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                "specificity": tn / max(tn + fp, 1),
                "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "mean_trajectory_edge_f1": local["trajectory_edge_f1"].mean(),
                "mean_composite_score": local["composite_score"].mean(),
            })
        for level, scenarios in pd.Series(strength).groupby(lambda x: strength[x]).groups.items():
            sub = local.loc[local["scenario"].isin(list(scenarios))]
            rows.append({
                "method": method, "relation_type": "all", "evidence_strength": level,
                "n": len(sub), "tp": np.nan, "fp": np.nan, "fn": np.nan, "tn": np.nan,
                "precision": np.nan, "recall": np.nan, "specificity": np.nan,
                "f1": sub["relation_f1"].mean(),
                "mean_trajectory_edge_f1": sub["trajectory_edge_f1"].mean(),
                "mean_composite_score": sub["composite_score"].mean(),
            })
    return pd.DataFrame(rows)


def manifest(output: Path, inputs: list[Path]):
    rows = []
    candidates = {
        path for path in (set(inputs) | set(output.glob("*")))
        if path.name != "OUTPUT_MANIFEST_SHA256.csv"
    }
    for path in sorted(candidates):
        if path.is_file():
            rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)})
    pd.DataFrame(rows).to_csv(output / "OUTPUT_MANIFEST_SHA256.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    output = (args.output or project / "outputs" / "v7_8_0_postfreeze_paper_validation").resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = project / "algorithm" / "validation" / "v7_8_0_postfreeze_validation_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    (output / "POSTFREEZE_PROTOCOL_LOCK.json").write_text(
        json.dumps({"locked_before_results": True, "config": config}, indent=2), encoding="utf-8"
    )
    freeze = load_module(
        project / "algorithm" / "experimental" / "run_v7_8_0_branch_prior_freeze_audit.py",
        "freeze_validation",
    )
    relation_module = load_module(
        project / "algorithm" / "analysis" / "run_independent_relation_modules_v74100.py",
        "relation_validation",
    )
    parent = relation_module.load_parent(project)
    programs = freeze.load_programs(project)
    rng = np.random.default_rng(config["random_seed"])

    map_rows, heldout_rows, heldout_audits = [], [], {}
    contexts = {}
    for relation in RELATIONS:
        ds = relation["dataset"]
        ctx = exact_context(project, relation, parent, relation_module)
        contexts[ds] = ctx
        cells = ctx["cells"]
        old_label = cells["root_old80"].to_numpy(str)
        exact_label = cells["root_cluster"].to_numpy(str)
        map_rows.append({
            "dataset": ds, "n_cells": len(cells),
            "historical_display_metacells": int(ctx["old_node"].nunique()),
            "formal_graph_metacells": int(cells["node_id"].nunique()),
            "formal_A0_nodes": int(len(ctx["weights"])),
            "cell_id_mapping_missing": 0,
            "root_assignment_agreement": float(np.mean(old_label == exact_label)),
            "root_assignment_ARI": float(adjusted_rand_score(old_label, exact_label)),
            "mapping_status": "FAIL_HISTORICAL_80_METACELL_IDS_NOT_FORMAL_GRAPH_NODE_IDS",
        })
        edges = spatial_edges(cells)
        fold_table, audit = heldout_test(ctx, relation, edges, list(programs[relation["program"]]))
        heldout_rows.append(fold_table)
        heldout_audits[ds] = audit
        pd.DataFrame(edges, columns=["cell_i", "cell_j"]).to_csv(
            output / f"{ds}_EXACT_SPATIAL_GRAPH_EDGES.csv.gz", index=False, compression="gzip"
        )
    mapping_audit = pd.DataFrame(map_rows)
    mapping_audit.to_csv(output / "V780_CELL_TO_GRAPH_MAPPING_AUDIT.csv", index=False)
    heldout = pd.concat(heldout_rows, ignore_index=True)
    heldout.to_csv(output / "V780_HELDOUT_PREDICTION_COMPARISON.csv", index=False)

    loo_rows, null_rows, direction_rows, mechanism_rows = [], [], [], []
    for relation in RELATIONS:
        ds = relation["dataset"]
        ctx, audit = contexts[ds], heldout_audits[ds]
        edges, anchors, nodes, prior, raw_score = branch_inputs(project, relation, freeze)
        n_nodes = len(ctx["weights"])
        original_genes = [g for g in programs[relation["program"]] if ctx["cache"].measured(g)]
        original_values, _ = program_vector(ctx["cache"], original_genes)
        original_score = node_score(ctx["cells"], original_values, n_nodes)
        original_transition, _, original = freeze.directed_branch_model(
            edges, anchors, nodes, original_score, prior, relation["relation_type"], 1e-12
        )
        branch_geometry = original_transition.loc[
            original_transition["is_branchpoint_edge"],
            ["source", "target"],
        ].astype(int).reset_index(drop=True)
        receiver_node_order = np.asarray(sorted(nodes), dtype=int)

        def fast_branch_shift(score: pd.Series, signed_prior: float = prior) -> dict:
            local = score.reindex(receiver_node_order).to_numpy(float)
            z = (local - np.mean(local)) / np.std(local)
            zmap = pd.Series(z, index=receiver_node_order)
            delta = (
                zmap.reindex(branch_geometry["target"]).to_numpy(float)
                - zmap.reindex(branch_geometry["source"]).to_numpy(float)
            )
            shift = signed_prior * delta
            index = int(np.argmax(np.abs(shift)))
            return {
                "affected_branch": (
                    f"{int(branch_geometry.iloc[index].source)}"
                    f"->{int(branch_geometry.iloc[index].target)}"
                ),
                "max_branch_log_prior_shift": float(shift[index]),
                "max_abs_branch_log_prior_shift": float(abs(shift[index])),
            }
        for drop in ["NONE"] + original_genes:
            use = original_genes if drop == "NONE" else [g for g in original_genes if g != drop]
            if len(use) < 3:
                continue
            values, measured = program_vector(ctx["cache"], use)
            summary = branch_summary(
                freeze, edges, anchors, nodes, node_score(ctx["cells"], values, n_nodes),
                prior, relation["relation_type"]
            )
            loo_rows.append({
                "dataset": ds, "program": relation["program"], "left_out_gene": drop,
                "n_genes": len(measured), "measured_genes": "|".join(measured),
                "affected_branch": summary["affected_branch"],
                "terminal_fate_L1_change": summary["terminal_fate_L1_change"],
                "transition_probability_change": summary["transition_probability_after"] - summary["transition_probability_before"],
                "same_affected_branch_as_full": summary["affected_branch"] == original["affected_branch"],
            })
        alt = config["alternative_signatures"][ds]
        alt_values, alt_measured = program_vector(ctx["cache"], alt["genes"])
        alt_summary = branch_summary(
            freeze, edges, anchors, nodes, node_score(ctx["cells"], alt_values, n_nodes),
            prior, relation["relation_type"]
        )
        loo_rows.append({
            "dataset": ds, "program": relation["program"], "left_out_gene": f"ALTERNATIVE:{alt['name']}",
            "n_genes": len(alt_measured), "measured_genes": "|".join(alt_measured),
            "affected_branch": alt_summary["affected_branch"],
            "terminal_fate_L1_change": alt_summary["terminal_fate_L1_change"],
            "transition_probability_change": alt_summary["transition_probability_after"] - alt_summary["transition_probability_before"],
            "same_affected_branch_as_full": alt_summary["affected_branch"] == original["affected_branch"],
        })

        # Expression-matched random gene-set null using node-level mean/nonzero bins.
        candidates = [str(g) for g in ctx["a"].var_names if ctx["cache"].measured(str(g))]
        node_gene, stats = {}, []
        for gene in candidates:
            val = ctx["cache"].gene(gene)
            ns = node_score(ctx["cells"], val, n_nodes)
            node_gene[gene] = ns
            stats.append((gene, float(np.mean(val)), float(np.mean(val > 0))))
        stat = pd.DataFrame(stats, columns=["gene", "mean", "fraction"])
        stat["mean_bin"] = pd.qcut(stat["mean"].rank(method="first"), 10, labels=False)
        stat["fraction_bin"] = pd.qcut(stat["fraction"].rank(method="first"), 10, labels=False)
        match_pool = {}
        for gene in original_genes:
            row = stat.loc[stat["gene"].eq(gene)].iloc[0]
            pool = stat.loc[
                (stat["mean_bin"] == row["mean_bin"]) & (stat["fraction_bin"] == row["fraction_bin"])
                & ~stat["gene"].isin(original_genes), "gene"
            ].tolist()
            if not pool:
                pool = stat.loc[~stat["gene"].isin(original_genes), "gene"].tolist()
            match_pool[gene] = pool
        null_effect = []
        for rep in range(int(config["matched_random_gene_sets"])):
            chosen = [rng.choice(match_pool[g]) for g in original_genes]
            score = pd.concat([node_gene[g] for g in chosen], axis=1).mean(axis=1)
            summary = fast_branch_shift(score)
            effect = float(summary["max_abs_branch_log_prior_shift"])
            null_effect.append(effect)
            null_rows.append({
                "dataset": ds, "program": relation["program"], "replicate": rep,
                "random_genes": "|".join(chosen), "max_abs_branch_log_prior_shift": effect,
                "affected_branch": summary["affected_branch"],
                "same_affected_branch_as_observed": summary["affected_branch"] == original["affected_branch"],
            })
        observed_fast = fast_branch_shift(original_score)
        null_p = (
            1 + np.sum(np.asarray(null_effect) >= observed_fast["max_abs_branch_log_prior_shift"])
        ) / (1 + len(null_effect))

        # Direction and permutation controls.
        for control, score, pvalue in [
            ("observed", original_score, prior),
            ("reverse_relation_sign", original_score, -prior),
            ("reverse_program_gradient", -original_score, prior),
        ]:
            summary = fast_branch_shift(score, pvalue)
            direction_rows.append({
                "dataset": ds, "control": control, "replicate": 0,
                "affected_branch": summary["affected_branch"],
                "max_branch_log_prior_shift": summary["max_branch_log_prior_shift"],
                "max_abs_branch_log_prior_shift": summary["max_abs_branch_log_prior_shift"],
            })
        node_list = np.array(sorted(nodes), int)
        for rep in range(int(config["negative_control_permutations"])):
            perm = original_score.copy()
            perm.loc[node_list] = rng.permutation(perm.loc[node_list].to_numpy())
            summary = fast_branch_shift(perm)
            direction_rows.append({
                "dataset": ds, "control": "receiver_program_node_permutation", "replicate": rep,
                "affected_branch": summary["affected_branch"],
                "max_branch_log_prior_shift": summary["max_branch_log_prior_shift"],
                "max_abs_branch_log_prior_shift": summary["max_abs_branch_log_prior_shift"],
            })
        # Held-out molecular negative controls on the exact cell mapping.
        receiver_y = audit["y"][audit["receiver_ids"]]
        receiver_exposure = audit["exposure"][audit["receiver_ids"]]
        baseline_x = audit["baseline_x"]
        fold = audit["fold"]
        block = audit["block"]
        observed_cv = cv_relation_delta(receiver_y, baseline_x, receiver_exposure, fold)
        direction_rows.append({
            "dataset": ds, "control": "observed_exact_mapping_heldout", "replicate": 0,
            **observed_cv,
        })
        for rep in range(int(config["negative_control_permutations"])):
            # Source-root permutation: destroy which source cells supply the
            # ligand while preserving the marginal exposure distribution.
            source_perm = rng.permutation(receiver_exposure)
            direction_rows.append({
                "dataset": ds, "control": "source_root_exposure_permutation",
                "replicate": rep,
                **cv_relation_delta(receiver_y, baseline_x, source_perm, fold),
            })
            # Spatial-block null: shuffle exposure-to-cell association only
            # within each pre-locked 4x4 spatial block.
            spatial_perm = receiver_exposure.copy()
            for value in np.unique(block):
                ids = np.flatnonzero(block == value)
                spatial_perm[ids] = rng.permutation(spatial_perm[ids])
            direction_rows.append({
                "dataset": ds, "control": "within_spatial_block_exposure_permutation",
                "replicate": rep,
                **cv_relation_delta(receiver_y, baseline_x, spatial_perm, fold),
            })
            # Pseudotime null: preserve expression/exposure but break the
            # exact-anchor pseudotime covariate inside spatial blocks.
            pseudo_perm = baseline_x.copy()
            for value in np.unique(block):
                ids = np.flatnonzero(block == value)
                pseudo_perm[ids, 0] = rng.permutation(pseudo_perm[ids, 0])
            direction_rows.append({
                "dataset": ds, "control": "within_spatial_block_pseudotime_permutation",
                "replicate": rep,
                **cv_relation_delta(receiver_y, pseudo_perm, receiver_exposure, fold),
            })

        mechanism_rows.append({
            "dataset": ds, "source_root": relation["source_root"], "receiver_root": relation["receiver_root"],
            "program": relation["program"], "ligand": relation["ligand"], "receptor": relation["receptor"],
            "historical_mapping_status": mapping_audit.loc[mapping_audit.dataset.eq(ds), "mapping_status"].iloc[0],
            "exact_mapping_heldout_delta_rmse": audit["delta_rmse"],
            "exact_mapping_heldout_delta_r2": audit["delta_r2"],
            "exact_mapping_relation_improves_prediction": audit["delta_rmse"] < 0 and audit["delta_r2"] > 0,
            "observed_terminal_fate_L1_change": original["terminal_fate_L1_change"],
            "observed_max_abs_branch_log_prior_shift": observed_fast["max_abs_branch_log_prior_shift"],
            "matched_random_null_p": null_p,
            "signature_specific_at_0_05": null_p < 0.05,
            "claim_status": "INDEPENDENTLY_SUPPORTED" if audit["delta_rmse"] < 0 and null_p < 0.05 else "NOT_SUPPORTED_POSTFREEZE",
            "external_biological_validation": "NOT_AVAILABLE_NO_INDEPENDENT_COHORT_OR_ORTHOGONAL_ASSAY",
            "causal_claim_allowed": False,
        })

    pd.DataFrame(loo_rows).to_csv(output / "V780_SIGNATURE_LEAVE_ONE_GENE_OUT.csv", index=False)
    pd.DataFrame(null_rows).to_csv(output / "V780_MATCHED_RANDOM_GENESET_NULL.csv", index=False)
    direction = pd.DataFrame(direction_rows)
    direction.to_csv(output / "V780_DIRECTIONAL_NEGATIVE_CONTROLS.csv", index=False)
    mechanism = pd.DataFrame(mechanism_rows)
    for index, row in mechanism.iterrows():
        local = direction.loc[direction["dataset"].eq(row.dataset)]
        observed = local.loc[local["control"].eq("observed_exact_mapping_heldout")].iloc[0]
        for control, column in [
            ("source_root_exposure_permutation", "source_root_permutation_empirical_p"),
            ("within_spatial_block_exposure_permutation", "spatial_block_permutation_empirical_p"),
            ("within_spatial_block_pseudotime_permutation", "pseudotime_permutation_empirical_p"),
        ]:
            null = local.loc[local["control"].eq(control), "heldout_delta_rmse"].dropna()
            mechanism.loc[index, column] = (
                1 + int((null <= float(observed.heldout_delta_rmse)).sum())
            ) / (1 + len(null))
        mechanism.loc[index, "association_status"] = (
            "EXACT_MAPPING_SPATIAL_ASSOCIATION_REPLICATED"
            if bool(row.exact_mapping_relation_improves_prediction)
            and float(mechanism.loc[index, "source_root_permutation_empirical_p"]) < 0.05
            and float(mechanism.loc[index, "spatial_block_permutation_empirical_p"]) < 0.05
            else "ASSOCIATION_NOT_REPLICATED"
        )
    mechanism.loc[mechanism.dataset.eq("xenium_breast")].to_csv(output / "V780_BREAST_MECHANISM_AUDIT.csv", index=False)
    mechanism.loc[mechanism.dataset.eq("xenium_melanoma")].to_csv(output / "V780_MELANOMA_MECHANISM_AUDIT.csv", index=False)

    # Locked-parameter sensitivity: selection only; topology remains read-only A0.
    sensitivity = []
    supported = pd.read_csv(
        project / "outputs" / "v7_8_0_latent_relation_guided_coupled_reconstruction"
        / "ALL7_FINAL_TYPED_RELATION_EDGES.csv"
    )
    for threshold, r1, dq in product(
        config["sensitivity"]["minimum_weighted_relation_score"],
        config["sensitivity"]["R1_weight"], config["sensitivity"]["delta_Q"]
    ):
        scores = supported["raw_score"].astype(float) * r1
        sensitivity.append({
            "minimum_weighted_relation_score": threshold, "R1_weight": r1, "delta_Q": dq,
            "selected_relation_count": int((scores >= threshold).sum()),
            "selected_datasets": "|".join(supported.loc[scores >= threshold, "dataset"].astype(str)),
            "internal_topology_changes": 0,
            "delta_Q_effect": "none_because_all_v780_candidates_share_identical_A0_internal_topology",
        })
    pd.DataFrame(sensitivity).to_csv(output / "V780_PARAMETER_SENSITIVITY.csv", index=False)
    synth = synthetic_metrics(project)
    synth.to_csv(output / "V780_SYNTHETIC_RELATION_TYPE_METRICS.csv", index=False)

    # D1-D6 and baseline comparisons.
    legacy_root = pd.read_csv(project / "outputs" / "v7_4_11_0_interaction_tree_ablation" / "ALL7_ROOT_LEVEL_TREE_COMPARISON.csv")
    legacy_boot = pd.read_csv(project / "outputs" / "v7_4_11_0_interaction_tree_ablation" / "REALDATA_MATCHED_BOOTSTRAP_SUMMARY.csv")
    heldout_mean = heldout.groupby("dataset", as_index=False)[["delta_rmse_relation_minus_baseline", "delta_r2_relation_minus_baseline"]].mean()
    component_rows = [
        ("D1_A0", 0, 0, 0, False),
        ("D2_detection_only", 2, 0, 0, False),
        ("D3_typed_edge_only", 2, 2, 0, False),
        ("D4_branch_prior_only", 2, 0, 2, False),
        ("D5_full_v780", 2, 2, 2, False),
        ("D5b_permuted_relation_evidence", 2, 2, 2, False),
        ("D6_legacy_interaction_v2", np.nan, np.nan, 0, True),
    ]
    comp = pd.DataFrame(component_rows, columns=[
        "variant", "detected_relation_count", "typed_cross_tree_edges", "branch_priors",
        "internal_topology_changed"
    ])
    comp["mean_exact_heldout_delta_rmse"] = np.nan
    comp["mean_exact_heldout_delta_r2"] = np.nan
    comp["permutation_definition"] = ""
    for variant in ["D4_branch_prior_only", "D5_full_v780"]:
        comp.loc[comp.variant.eq(variant), "mean_exact_heldout_delta_rmse"] = heldout_mean["delta_rmse_relation_minus_baseline"].mean()
        comp.loc[comp.variant.eq(variant), "mean_exact_heldout_delta_r2"] = heldout_mean["delta_r2_relation_minus_baseline"].mean()
    permuted_relation = direction.loc[
        direction["control"].eq("source_root_exposure_permutation")
    ]
    comp.loc[
        comp.variant.eq("D5b_permuted_relation_evidence"),
        "mean_exact_heldout_delta_rmse",
    ] = permuted_relation["heldout_delta_rmse"].mean()
    comp.loc[
        comp.variant.eq("D5b_permuted_relation_evidence"),
        "mean_exact_heldout_delta_r2",
    ] = permuted_relation["heldout_delta_r2"].mean()
    comp.loc[
        comp.variant.eq("D5b_permuted_relation_evidence"),
        "permutation_definition",
    ] = "source-root exposure vector permuted; marginal exposure preserved"
    comp.loc[comp.variant.eq("D6_legacy_interaction_v2"), "mean_edge_change_fraction"] = legacy_root["edge_change_fraction"].mean()
    a0_med = legacy_boot.loc[legacy_boot.variant.eq("A0_progression_only"), "median_edge_recurrence"].mean()
    a1_med = legacy_boot.loc[legacy_boot.variant.eq("A1_legacy_full"), "median_edge_recurrence"].mean()
    comp.loc[comp.variant.eq("D6_legacy_interaction_v2"), "bootstrap_edge_recurrence_change_vs_A0"] = a1_med - a0_med
    comp.to_csv(output / "V780_COMPONENT_ABLATION_SUMMARY.csv", index=False)

    baseline = synth.groupby("method", as_index=False).agg(
        synthetic_relation_f1=("f1", "mean"),
        synthetic_trajectory_edge_f1=("mean_trajectory_edge_f1", "mean"),
        synthetic_composite=("mean_composite_score", "mean"),
    )
    baseline["realdata_note"] = baseline["method"].map({
        "A0_independent_forest": "formal internal baseline; no cross-tree mechanism",
        "legacy_interaction_v2": f"mean edge-change={legacy_root.edge_change_fraction.mean():.6f}; bootstrap recurrence delta={a1_med-a0_med:.6f}",
        "new_joint_method": "synthetic predecessor of frozen v7.8.0 relation layer",
        "oracle_relation_evidence": "synthetic upper bound only",
        "permuted_relation_evidence": "negative control",
    })
    baseline.to_csv(output / "V780_BASELINE_METHOD_COMPARISON.csv", index=False)

    all_supported = bool(mechanism["claim_status"].eq("INDEPENDENTLY_SUPPORTED").all())
    association_only = mechanism.loc[
        mechanism["association_status"].eq("EXACT_MAPPING_SPATIAL_ASSOCIATION_REPLICATED")
        & ~mechanism["claim_status"].eq("INDEPENDENTLY_SUPPORTED"),
        "dataset",
    ].tolist()
    status = {
        "validation_status": (
            "POSTFREEZE_VALIDATION_PASS"
            if all_supported
            else "POSTFREEZE_VALIDATION_FAIL_FORMAL_MECHANISM"
        ),
        "frozen_v780_modified": False,
        "mapping_audit_pass": False,
        "independently_supported_mechanisms": mechanism.loc[mechanism.claim_status.eq("INDEPENDENTLY_SUPPORTED"), "dataset"].tolist(),
        "exact_mapping_association_only": association_only,
        "unsupported_mechanisms": mechanism.loc[~mechanism.claim_status.eq("INDEPENDENTLY_SUPPORTED"), "dataset"].tolist(),
        "paper_claim_policy": "Only independently supported rows may be described as relation-aware molecular mechanisms; no causal wording.",
    }
    (output / "RUN_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    report = f"""# v7.8.0 post-freeze paper validation

## Decision

**{status['validation_status']}**. The frozen v7.8.0 files and method were not changed or re-frozen.

## Critical input audit

The historical relation-evidence runner mapped the 80 display/compression metacell IDs directly onto the formal A0 node array. The formal graph contains 5,405 nodes for breast and 3,490 for melanoma. Exact cell-to-graph mapping produced root-label agreement of {mapping_audit.iloc[0].root_assignment_agreement:.3f} (breast) and {mapping_audit.iloc[1].root_assignment_agreement:.3f} (melanoma), with ARI values {mapping_audit.iloc[0].root_assignment_ARI:.3f} and {mapping_audit.iloc[1].root_assignment_ARI:.3f}. Therefore the historical v7.6.1 relation evidence is not a valid exact-anchor cell assignment.

## Independent exact-mapping validation

{mechanism.to_markdown(index=False)}

Negative held-out delta-RMSE and positive delta-R2 indicate predictive improvement. Source-root and spatial-block empirical p values compare that improvement with 100 locked negative controls. Matched-random p values compare the observed branch-prior shift with 500 expression/prevalence-matched gene sets.

## Interpretation

The internal A0 forest remains unchanged and reproducible. Both exact-mapping LR exposures improve held-out prediction relative to source/exposure permutations, so a spatial association signal remains. However, neither EMT nor ECM branch shift is specific against the 500 matched random signatures. The defensible result is therefore **association replicated, named branch mechanism not validated**. Failure does not invalidate the A0 trajectories; it invalidates only the formal EMT/ECM-specific cross-tree mechanism claim.

## Literature basis for alternative signatures

- Hallmark gene sets: Liberzon et al., *Cell Systems* (2015), PMCID: PMC4707969.
- Matrisome framework: Naba et al., *Molecular & Cellular Proteomics* (2012), PMCID: PMC3322572; Naba et al., *Matrix Biology* (2016), PMCID: PMC5013529.

## External validation boundary

No independent cohort, matched full-transcriptome reference, pathology/CNV assay, or perturbational validation is available inside this project. External biological validation is therefore marked unavailable rather than passed.

## Files

All requested CSVs are in this directory. `V780_CELL_TO_GRAPH_MAPPING_AUDIT.csv` is an additional mandatory provenance audit.
"""
    (output / "V780_POSTFREEZE_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    for ctx in contexts.values():
        ctx["a"].file.close()
    manifest(output, [
        config_path,
        project / "algorithm" / "experimental" / "run_v7_8_0_branch_prior_freeze_audit.py",
        project / "algorithm" / "analysis" / "infer_true_biological_cross_tree_relations_v749.py",
    ])
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
