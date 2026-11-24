#!/usr/bin/env python3
"""Run the frozen P3 native single-start public-method benchmark.

The public methods in this lane receive exactly one start cell because their
native APIs require it. They do not receive all roots. Coverage and quality on
covered cells are reported separately, and root F1 is not scored.
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
DEFAULT_DATASETS = Path(
    r"D:\桌面\FYP\BASE_CASCAT\algorithms\unified_eval_kit\datasets_multiroot"
)


@dataclass
class Prediction:
    method: str
    cell_ids: np.ndarray
    pseudotime: np.ndarray
    branch: np.ndarray | None
    n_edges: int
    topology_supported: bool
    api: str


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _truth_time(adata: ad.AnnData) -> np.ndarray:
    return np.asarray(adata.uns["timecourse"], dtype=float).reshape(-1)


def _paga(adata: ad.AnnData, root: str) -> Prediction:
    result = prepare_scanpy(adata, 3101)
    sc.tl.paga(result, groups="smoke_cluster")
    result.uns["iroot"] = int(result.obs_names.get_loc(root))
    sc.tl.diffmap(result)
    sc.tl.dpt(result)
    tree = result.uns["paga"]["connectivities_tree"]
    return Prediction(
        "PAGA+DPT",
        result.obs_names.astype(str).to_numpy(),
        result.obs["dpt_pseudotime"].to_numpy(float),
        result.obs["smoke_cluster"].astype(str).to_numpy(),
        int(tree.nnz),
        True,
        "scanpy.tl.paga + scanpy.tl.dpt",
    )


def _cellrank(adata: ad.AnnData, root: str) -> Prediction:
    from cellrank.estimators import GPCCA
    from cellrank.kernels import PseudotimeKernel

    full = prepare_scanpy(adata, 3102)
    graph = nx.from_scipy_sparse_array(full.obsp["connectivities"])
    root_position = int(full.obs_names.get_loc(root))
    component = sorted(nx.node_connected_component(graph, root_position))
    result = full[component].copy()
    result.uns["iroot"] = int(result.obs_names.get_loc(root))
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
    return Prediction(
        "CellRank-on-DPT",
        result.obs_names.astype(str).to_numpy(),
        result.obs["dpt_pseudotime"].to_numpy(float),
        branch,
        int(kernel.transition_matrix.nnz),
        False,
        "CellRank PseudotimeKernel + GPCCA",
    )


def _palantir(adata: ad.AnnData, root: str) -> Prediction:
    import palantir

    result = adata.copy()
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
        early_cell=root,
        num_waypoints=min(250, result.n_obs),
        knn=15,
        n_jobs=1,
        seed=3103,
    )
    probabilities = fit.branch_probs
    branch = (
        probabilities.idxmax(axis=1).astype(str).to_numpy()
        if probabilities.shape[1] > 0
        else None
    )
    return Prediction(
        "Palantir",
        fit.pseudotime.index.astype(str).to_numpy(),
        fit.pseudotime.to_numpy(float),
        branch,
        0,
        False,
        "palantir.core.run_palantir",
    )


def _stlearn(adata: ad.AnnData, root: str) -> Prediction:
    import stlearn

    result = prepare_scanpy(adata, 3104)
    result.obs["imagerow"] = np.asarray(result.obsm["spatial"])[:, 0]
    result.obs["imagecol"] = np.asarray(result.obsm["spatial"])[:, 1]
    root_cluster = str(result.obs.loc[root, "smoke_cluster"])
    stlearn.spatial.trajectory.set_root(
        result, use_label="smoke_cluster", cluster=root_cluster
    )
    result.uns["iroot"] = int(result.obs_names.get_loc(root))
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
    return Prediction(
        "stLearn",
        result.obs_names.astype(str).to_numpy(),
        result.obs["dpt_pseudotime"].to_numpy(float),
        result.obs["smoke_cluster"].astype(str).to_numpy(),
        int(graph.number_of_edges()) if hasattr(graph, "number_of_edges") else 0,
        True,
        "stlearn.spatial.trajectory.pseudotime",
    )


ADAPTERS = {
    "PAGA+DPT": _paga,
    "CellRank-on-DPT": _cellrank,
    "Palantir": _palantir,
    "stLearn": _stlearn,
}


def _score(adata: ad.AnnData, prediction: Prediction) -> tuple[dict[str, object], pd.DataFrame]:
    truth = pd.DataFrame(
        {
            "truth_time": _truth_time(adata),
            "truth_branch": adata.obs["branch"].astype(str).to_numpy(),
            "origin": np.asarray(adata.obs["origin"], dtype=int),
        },
        index=adata.obs_names.astype(str),
    )
    pred_data: dict[str, object] = {"pseudotime": prediction.pseudotime}
    if prediction.branch is not None:
        pred_data["branch"] = prediction.branch.astype(str)
    pred = pd.DataFrame(pred_data, index=prediction.cell_ids.astype(str))
    joined = truth.join(pred, how="left")
    covered = joined[np.isfinite(joined["pseudotime"])]
    origin_correlations = []
    for _, group in covered.groupby("origin"):
        if len(group) >= 3:
            origin_correlations.append(
                float(spearmanr(group["pseudotime"], group["truth_time"]).statistic)
            )
    pooled = (
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
    row = {
        "method": prediction.method,
        "status": "PASS" if len(covered) >= 3 else "FAIL",
        "n_cells": adata.n_obs,
        "n_covered": len(covered),
        "coverage": len(covered) / adata.n_obs,
        "n_uncovered": adata.n_obs - len(covered),
        "pt_spearman_pooled_covered": pooled,
        "forest_pt_SR_covered": float(np.mean(origin_correlations))
        if origin_correlations
        else np.nan,
        "n_origins_with_pt_score": len(origin_correlations),
        "branch_ARI_covered": branch_ari,
        "branch_output_available": branch_available,
        "n_edges": prediction.n_edges,
        "topology_supported": prediction.topology_supported,
        "root_F1": "N/A_ROOT_SUPPLIED",
        "quality_denominator": "finite covered cells only",
        "uncovered_policy": "reported; not imputed and not scored as zero",
        "method_api": prediction.api,
        "error": "",
    }
    return row, exported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO
        / "outputs"
        / "NicheForest_paper_frozen_candidate"
        / "P3_native_single_start_benchmark",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    predictions = output / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    datasets = sorted(args.datasets.resolve().glob("*.h5ad"))
    for dataset in datasets:
        adata = ad.read_h5ad(dataset)
        supplied_start = roots_from_truth(adata)[0]
        for method, adapter in ADAPTERS.items():
            started = time.perf_counter()
            try:
                prediction = adapter(adata, supplied_start)
                row, exported = _score(adata, prediction)
                exported.insert(0, "supplied_start_cell", supplied_start)
                exported.insert(0, "dataset", dataset.stem)
                exported.insert(0, "method", method)
                exported.to_csv(
                    predictions / f"{dataset.stem}__{method.replace('+', '_')}.csv",
                    index=False,
                )
            except Exception as exc:
                row = {
                    "method": method,
                    "status": "FAIL",
                    "n_cells": adata.n_obs,
                    "n_covered": 0,
                    "coverage": 0.0,
                    "n_uncovered": adata.n_obs,
                    "pt_spearman_pooled_covered": np.nan,
                    "forest_pt_SR_covered": np.nan,
                    "n_origins_with_pt_score": 0,
                    "branch_ARI_covered": np.nan,
                    "branch_output_available": False,
                    "n_edges": 0,
                    "topology_supported": False,
                    "root_F1": "N/A_ROOT_SUPPLIED",
                    "quality_denominator": "finite covered cells only",
                    "uncovered_policy": "reported; not imputed and not scored as zero",
                    "method_api": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                (output / f"{dataset.stem}__{method.replace('+', '_')}_ERROR.log").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            row = {
                "dataset": dataset.stem,
                "n_true_roots": int(adata.uns.get("n_origins", 1)),
                "supplied_start_cell": supplied_start,
                "seconds": round(time.perf_counter() - started, 3),
                **row,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / "P3_DATASET_RESULTS.csv", index=False)

    results = pd.DataFrame(rows)
    summary = (
        results.groupby("method", as_index=False)
        .agg(
            n_datasets=("dataset", "count"),
            n_pass=("status", lambda value: int(np.sum(value == "PASS"))),
            mean_coverage=("coverage", "mean"),
            median_coverage=("coverage", "median"),
            mean_forest_pt_SR_covered=("forest_pt_SR_covered", "mean"),
            mean_branch_ARI_covered=("branch_ARI_covered", "mean"),
            branch_output_rate=("branch_output_available", "mean"),
        )
    )
    summary["comparison_role"] = "CONTEXTUAL_EXTERNAL_BASELINE"
    summary["primary_core_decision"] = "NOT_EVALUATED_IN_P3"
    summary.to_csv(output / "P3_METHOD_SUMMARY.csv", index=False)
    slingshot = pd.DataFrame(
        [
            {
                "method": "Slingshot",
                "status": "NOT_RUN_ENVIRONMENT",
                "reason": "Rscript unavailable; no Slingshot-like surrogate permitted",
            }
        ]
    )
    slingshot.to_csv(output / "P3_EXCLUSIONS.csv", index=False)
    failures = results[results["status"].eq("FAIL")]
    decision = {
        "phase": "P3_NATIVE_SINGLE_START_END_TO_END",
        "method_frozen": True,
        "root_contract": "one required start cell supplied; all roots are not supplied",
        "coverage_and_quality_separated": True,
        "root_f1_scored": False,
        "primary_comparison_evaluated": False,
        "decision": "PASS" if failures.empty else "FAIL",
        "n_dataset_method_runs": len(results),
        "n_failures": len(failures),
        "interpretation": "context only; P4 evaluates current T0 vs root-conditioned MSF",
    }
    (output / "P3_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    manifest = pd.DataFrame(
        [
            {"path": str(path), "sha256": _sha256(path)}
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "P3_SHA256_MANIFEST.csv"
        ]
    )
    manifest.to_csv(output / "P3_SHA256_MANIFEST.csv", index=False)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
