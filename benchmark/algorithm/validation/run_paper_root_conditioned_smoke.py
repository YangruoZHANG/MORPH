#!/usr/bin/env python3
"""Audit root-conditioned public-method adapters on one multi-root dataset.

This is a P2.5 interface smoke test, not a performance benchmark. Each method is
run independently for every supplied root. Coverage is reported separately from
quality computed only on covered cells; uncovered cells are never imputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from run_paper_baseline_adapter_smoke import prepare_scanpy, roots_from_truth


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path(
    r"D:\桌面\FYP\BASE_CASCAT\algorithms\unified_eval_kit"
    r"\datasets_multiroot\sim_K2_sep.h5ad"
)


@dataclass
class RootPrediction:
    method: str
    root_index: int
    root_cell: str
    cell_ids: np.ndarray
    pseudotime: np.ndarray
    branch: np.ndarray | None
    n_edges: int
    method_api: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timecourse(adata: ad.AnnData) -> np.ndarray:
    value = adata.uns.get("timecourse")
    if value is None:
        raise ValueError("Dataset has no truth timecourse")
    result = np.asarray(value, dtype=float).reshape(-1)
    if len(result) != adata.n_obs:
        raise ValueError("Truth timecourse length mismatch")
    return result


def _root_component(adata: ad.AnnData, root_cell: str, seed: int) -> ad.AnnData:
    prepared = prepare_scanpy(adata, seed)
    graph = nx.from_scipy_sparse_array(prepared.obsp["connectivities"])
    root_position = int(prepared.obs_names.get_loc(root_cell))
    positions = sorted(nx.node_connected_component(graph, root_position))
    return prepared[positions].copy()


def _paga_dpt(adata: ad.AnnData, root_index: int, root_cell: str) -> RootPrediction:
    result = _root_component(adata, root_cell, 2101)
    sc.tl.paga(result, groups="smoke_cluster")
    result.uns["iroot"] = int(result.obs_names.get_loc(root_cell))
    sc.tl.diffmap(result)
    sc.tl.dpt(result)
    tree = result.uns["paga"]["connectivities_tree"]
    return RootPrediction(
        "PAGA+DPT",
        root_index,
        root_cell,
        result.obs_names.astype(str).to_numpy(),
        result.obs["dpt_pseudotime"].to_numpy(float),
        result.obs["smoke_cluster"].astype(str).to_numpy(),
        int(tree.nnz),
        "scanpy.tl.paga + scanpy.tl.dpt",
    )


def _cellrank(adata: ad.AnnData, root_index: int, root_cell: str) -> RootPrediction:
    from cellrank.estimators import GPCCA
    from cellrank.kernels import PseudotimeKernel

    result = _root_component(adata, root_cell, 2102)
    result.uns["iroot"] = int(result.obs_names.get_loc(root_cell))
    sc.tl.diffmap(result)
    sc.tl.dpt(result)
    kernel = PseudotimeKernel(result, time_key="dpt_pseudotime")
    kernel.compute_transition_matrix(show_progress_bar=False, n_jobs=1)
    requested_states = min(6, max(2, int(result.obs["smoke_cluster"].nunique())))
    branch = None
    for n_states in range(requested_states, 1, -1):
        estimator = GPCCA(kernel)
        try:
            estimator.compute_schur(n_components=max(n_states + 1, 3))
            estimator.compute_macrostates(n_states=n_states, n_cells=3)
            branch = estimator.macrostates.astype(str).to_numpy()
            break
        except ValueError:
            continue
    return RootPrediction(
        "CellRank-on-DPT",
        root_index,
        root_cell,
        result.obs_names.astype(str).to_numpy(),
        result.obs["dpt_pseudotime"].to_numpy(float),
        branch,
        int(kernel.transition_matrix.nnz),
        "CellRank PseudotimeKernel + GPCCA",
    )


def _palantir(adata: ad.AnnData, root_index: int, root_cell: str) -> RootPrediction:
    import palantir

    component = _root_component(adata, root_cell, 2103)
    result = adata[component.obs_names].copy()
    sc.pp.normalize_total(result, target_sum=1e4)
    sc.pp.log1p(result)
    sc.pp.pca(result, n_comps=min(20, result.n_vars - 1, result.n_obs - 1))
    palantir.utils.run_diffusion_maps(result, n_components=10, knn=15)
    palantir.utils.determine_multiscale_space(result)
    multiscale = pd.DataFrame(
        result.obsm["DM_EigenVectors_multiscaled"], index=result.obs_names
    )
    fit = palantir.core.run_palantir(
        multiscale,
        early_cell=root_cell,
        num_waypoints=min(250, result.n_obs),
        knn=15,
        n_jobs=1,
        seed=2103,
    )
    probabilities = fit.branch_probs
    branch = (
        probabilities.idxmax(axis=1).astype(str).to_numpy()
        if probabilities.shape[1] > 0
        else None
    )
    return RootPrediction(
        "Palantir",
        root_index,
        root_cell,
        fit.pseudotime.index.astype(str).to_numpy(),
        fit.pseudotime.to_numpy(float),
        branch,
        0,
        "palantir.core.run_palantir",
    )


def _stlearn(adata: ad.AnnData, root_index: int, root_cell: str) -> RootPrediction:
    import stlearn

    result = _root_component(adata, root_cell, 2104)
    result.obs["imagerow"] = np.asarray(result.obsm["spatial"])[:, 0]
    result.obs["imagecol"] = np.asarray(result.obsm["spatial"])[:, 1]
    root_cluster = str(result.obs.loc[root_cell, "smoke_cluster"])
    stlearn.spatial.trajectory.set_root(
        result, use_label="smoke_cluster", cluster=root_cluster
    )
    result.uns["iroot"] = int(result.obs_names.get_loc(root_cell))
    stlearn.spatial.trajectory.pseudotime(
        result,
        use_label="smoke_cluster",
        n_neighbors=15,
        eps=20,
        use_rep="X_pca",
        threshold=0.01,
        radius=50,
        use_sme=False,
        pseudotime_key="dpt_pseudotime",
        max_nodes=4,
        run_knn=False,
    )
    graph = result.uns.get("global_graph", nx.Graph())
    return RootPrediction(
        "stLearn",
        root_index,
        root_cell,
        result.obs_names.astype(str).to_numpy(),
        result.obs["dpt_pseudotime"].to_numpy(float),
        result.obs["smoke_cluster"].astype(str).to_numpy(),
        int(graph.number_of_edges()) if hasattr(graph, "number_of_edges") else 0,
        "stlearn.spatial.trajectory.pseudotime",
    )


ADAPTERS = {
    "PAGA+DPT": _paga_dpt,
    "CellRank-on-DPT": _cellrank,
    "Palantir": _palantir,
    "stLearn": _stlearn,
}


def _score_root(
    adata: ad.AnnData, prediction: RootPrediction, true_time: np.ndarray
) -> tuple[dict[str, object], pd.DataFrame]:
    all_ids = adata.obs_names.astype(str)
    all_set = set(all_ids)
    declared = prediction.cell_ids.astype(str)
    unique = len(set(declared)) == len(declared)
    lengths_match = len(declared) == len(prediction.pseudotime) and (
        prediction.branch is None or len(declared) == len(prediction.branch)
    )
    ids_valid = set(declared).issubset(all_set)
    root_present = prediction.root_cell in set(declared)
    finite = np.isfinite(prediction.pseudotime)

    frame_data: dict[str, object] = {
        "cell_id": declared,
        "pseudotime": prediction.pseudotime,
    }
    if prediction.branch is not None:
        frame_data["branch"] = prediction.branch.astype(str)
    frame = pd.DataFrame(frame_data).set_index("cell_id")
    truth = pd.DataFrame(
        {
            "origin": np.asarray(adata.obs["origin"], dtype=int),
            "truth_branch": adata.obs["branch"].astype(str).to_numpy(),
            "truth_time": true_time,
        },
        index=all_ids,
    )
    joined = frame.join(truth, how="left")
    expected = truth[truth["origin"].eq(prediction.root_index)]
    covered = joined[joined["origin"].eq(prediction.root_index) & np.isfinite(joined["pseudotime"])]
    off_target = joined[~joined["origin"].eq(prediction.root_index)]
    pt_sr = (
        float(spearmanr(covered["pseudotime"], covered["truth_time"]).statistic)
        if len(covered) >= 3
        else np.nan
    )
    branch_available = "branch" in covered and covered["branch"].notna().any()
    branch_ari = (
        float(
            adjusted_rand_score(
                covered.loc[covered["branch"].notna(), "truth_branch"],
                covered.loc[covered["branch"].notna(), "branch"],
            )
        )
        if branch_available and covered["branch"].notna().sum() >= 3
        else np.nan
    )
    exported = joined.reset_index().rename(columns={"index": "cell_id"})
    exported.insert(0, "root_cell", prediction.root_cell)
    exported.insert(0, "root_index", prediction.root_index)
    exported.insert(0, "method", prediction.method)
    row = {
        "method": prediction.method,
        "root_index": prediction.root_index,
        "root_cell": prediction.root_cell,
        "status": "PASS",
        "n_total_cells": adata.n_obs,
        "n_expected_root_cells": len(expected),
        "n_declared_cells": len(declared),
        "n_covered_root_cells": len(covered),
        "root_coverage": len(covered) / max(len(expected), 1),
        "total_declared_coverage": len(declared) / adata.n_obs,
        "off_target_fraction": len(off_target) / max(len(declared), 1),
        "conditional_pt_spearman": pt_sr,
        "conditional_branch_ari": branch_ari,
        "branch_output_available": branch_available,
        "n_edges": prediction.n_edges,
        "cell_ids_unique": unique,
        "cell_ids_valid": ids_valid,
        "lengths_match": lengths_match,
        "root_present": root_present,
        "method_api": prediction.method_api,
        "error": "",
    }
    schema_ok = unique and ids_valid and lengths_match and root_present
    if not schema_ok or row["root_coverage"] < 0.90 or len(covered) < 3:
        row["status"] = "FAIL"
    return row, exported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO
        / "outputs"
        / "NicheForest_paper_frozen_candidate"
        / "P2_5_root_conditioned_smoke",
    )
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    predictions_dir = output / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(dataset)
    roots = roots_from_truth(adata)
    true_time = _timecourse(adata)
    root_rows: list[dict[str, object]] = []
    method_predictions: dict[str, list[RootPrediction]] = {}

    for method, adapter in ADAPTERS.items():
        method_predictions[method] = []
        for root_index, root_cell in enumerate(roots):
            started = time.perf_counter()
            try:
                prediction = adapter(adata, root_index, root_cell)
                row, exported = _score_root(adata, prediction, true_time)
                method_predictions[method].append(prediction)
                exported.to_csv(
                    predictions_dir / f"{method.replace('+', '_')}_R{root_index}.csv",
                    index=False,
                )
            except Exception as exc:
                row = {
                    "method": method,
                    "root_index": root_index,
                    "root_cell": root_cell,
                    "status": "FAIL",
                    "n_total_cells": adata.n_obs,
                    "n_expected_root_cells": int(
                        np.sum(np.asarray(adata.obs["origin"], dtype=int) == root_index)
                    ),
                    "n_declared_cells": 0,
                    "n_covered_root_cells": 0,
                    "root_coverage": 0.0,
                    "total_declared_coverage": 0.0,
                    "off_target_fraction": np.nan,
                    "conditional_pt_spearman": np.nan,
                    "conditional_branch_ari": np.nan,
                    "branch_output_available": False,
                    "n_edges": 0,
                    "cell_ids_unique": False,
                    "cell_ids_valid": False,
                    "lengths_match": False,
                    "root_present": False,
                    "method_api": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                (output / f"{method.replace('+', '_')}_R{root_index}_ERROR.log").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            row["seconds"] = round(time.perf_counter() - started, 3)
            root_rows.append(row)

    pd.DataFrame(root_rows).to_csv(output / "P2_5_ROOT_RUNS.csv", index=False)
    aggregate_rows = []
    all_ids = set(adata.obs_names.astype(str))
    for method, predictions in method_predictions.items():
        sets = [set(pred.cell_ids.astype(str)) for pred in predictions]
        union = set().union(*sets) if sets else set()
        overlap = set.intersection(*sets) if len(sets) == len(roots) and sets else set()
        method_rows = [row for row in root_rows if row["method"] == method]
        aggregate_rows.append(
            {
                "method": method,
                "status": "PASS"
                if len(predictions) == len(roots)
                and all(row["status"] == "PASS" for row in method_rows)
                and len(union) / len(all_ids) >= 0.95
                else "FAIL",
                "n_roots_expected": len(roots),
                "n_roots_completed": len(predictions),
                "union_coverage": len(union) / len(all_ids),
                "root_run_overlap_fraction": len(overlap) / len(all_ids),
                "mean_root_coverage": float(
                    np.mean([float(row["root_coverage"]) for row in method_rows])
                ),
                "mean_conditional_pt_spearman": float(
                    np.nanmean([float(row["conditional_pt_spearman"]) for row in method_rows])
                ),
                "mean_conditional_branch_ari": float(
                    np.nanmean([float(row["conditional_branch_ari"]) for row in method_rows])
                ),
                "coverage_rule": "union/total; uncovered cells are not imputed",
                "quality_rule": "computed only on finite covered cells of the conditioned root",
            }
        )
    aggregate_rows.append(
        {
            "method": "Slingshot",
            "status": "NOT_RUN_ENVIRONMENT",
            "n_roots_expected": len(roots),
            "n_roots_completed": 0,
            "union_coverage": np.nan,
            "root_run_overlap_fraction": np.nan,
            "mean_root_coverage": np.nan,
            "mean_conditional_pt_spearman": np.nan,
            "mean_conditional_branch_ari": np.nan,
            "coverage_rule": "not run",
            "quality_rule": "official R runtime unavailable; no approximation permitted",
        }
    )
    pd.DataFrame(aggregate_rows).to_csv(output / "P2_5_AGGREGATE.csv", index=False)
    failures = [row for row in aggregate_rows if row["status"] == "FAIL"]
    decision = {
        "phase": "P2.5_ROOT_CONDITIONED_SMOKE",
        "dataset": str(dataset),
        "method_frozen": True,
        "formal_benchmark_started": False,
        "coverage_and_quality_separated": True,
        "uncovered_cells_imputed": False,
        "slingshot": "NOT_RUN_ENVIRONMENT",
        "decision": "PASS" if not failures else "FAIL",
        "failures": [row["method"] for row in failures],
        "gate": "all supported adapters complete every root with >=0.90 root coverage and >=0.95 aggregate union coverage",
    }
    (output / "P2_5_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    manifest = pd.DataFrame(
        [
            {"path": str(path), "sha256": _sha256(path)}
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "P2_5_SHA256_MANIFEST.csv"
        ]
    )
    manifest.to_csv(output / "P2_5_SHA256_MANIFEST.csv", index=False)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
