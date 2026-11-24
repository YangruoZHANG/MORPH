#!/usr/bin/env python3
"""Build formal NicheForest metacells, then project cell truth without leakage."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FORMAL_MODULE = (
    REPO
    / "algorithm"
    / "CASCAT-MR-CMI-E-Gate_v7.4.3.3_anchor_bridge"
    / "cascat_v7_3_metacell_process.py"
)


def load_formal_module():
    spec = importlib.util.spec_from_file_location("nicheforest_formal_metacell", FORMAL_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load formal metacell module: {FORMAL_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalized_entropy(probabilities: pd.Series) -> float:
    values = probabilities.to_numpy(dtype=float)
    values = values[values > 0]
    if len(values) <= 1:
        return 0.0
    return float(-np.sum(values * np.log(values)) / math.log(len(values)))


def composition_table(
    mapping: pd.DataFrame,
    truth: pd.DataFrame,
    category_cols: list[str],
    output_probability: str,
) -> pd.DataFrame:
    joined = mapping[["cell_name", "metacell_id"]].merge(
        truth, left_on="cell_name", right_on="cell_id", how="left", validate="many_to_many"
    )
    if joined[category_cols].isna().any(axis=None):
        raise ValueError(f"Missing truth while constructing {output_probability}")
    group_cols = ["metacell_id", *category_cols]
    if "membership_probability" in joined.columns:
        values = joined.groupby(group_cols, as_index=False)["membership_probability"].sum()
        denominators = mapping.groupby("metacell_id").size().rename("n_cells")
        values = values.merge(denominators, on="metacell_id", validate="many_to_one")
        values[output_probability] = values["membership_probability"] / values["n_cells"]
        return values[group_cols + [output_probability]]
    values = joined.groupby(group_cols, as_index=False).size().rename(columns={"size": "n_category"})
    denominators = mapping.groupby("metacell_id").size().rename("n_cells")
    values = values.merge(denominators, on="metacell_id", validate="many_to_one")
    values[output_probability] = values["n_category"] / values["n_cells"]
    return values[group_cols + [output_probability]]


def dominant_summary(
    composition: pd.DataFrame,
    category: str,
    probability: str,
    prefix: str,
) -> pd.DataFrame:
    ordered = composition.sort_values(
        ["metacell_id", probability, category], ascending=[True, False, True]
    )
    dominant = ordered.groupby("metacell_id", as_index=False).first()
    entropy = composition.groupby("metacell_id")[probability].apply(normalized_entropy)
    return dominant[["metacell_id", category, probability]].rename(
        columns={category: f"dominant_{prefix}", probability: f"{prefix}_purity"}
    ).merge(entropy.rename(f"{prefix}_entropy"), on="metacell_id")


def build(args: argparse.Namespace) -> None:
    input_h5ad = Path(args.input_h5ad).resolve()
    truth_dir = Path(args.truth_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    formal = load_formal_module()
    adata = ad.read_h5ad(input_h5ad)
    method_adata = ad.AnnData(
        X=adata.X.copy(),
        obs=pd.DataFrame(index=adata.obs_names.copy()),
        var=adata.var.copy(),
    )
    method_adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=float).copy()
    if "X_pca" in adata.obsm:
        method_adata.obsm["X_pca"] = np.asarray(adata.obsm["X_pca"], dtype=float).copy()
    cfg = SimpleNamespace(
        input_level="cell",
        metacell_key=None,
        annotation_key="__disabled_for_truth_safe_benchmark__",
        n_pca=args.n_pca,
        random_state=args.seed,
        metacell_target_size=args.target_size,
        metacell_min_size=args.min_size,
        metacell_max_size=args.max_size,
        metacell_birch_threshold_multiplier=args.birch_multiplier,
        metacell_spatial_weight=args.spatial_weight,
        metacell_tiny_merge_size=args.tiny_merge_size,
        tiny_merge_min_comp_similarity=args.tiny_merge_min_comp_similarity,
        tiny_merge_max_spatial_nn_factor=args.tiny_merge_max_spatial_nn_factor,
    )
    force_adaptive = pd.DataFrame({"dataset": [args.dataset], "root_id": ["__NO_TRUTH_ROOT__"]})
    result = formal.build_or_use_metacells(
        method_adata,
        force_adaptive,
        args.dataset,
        cfg,
        dataset_col="dataset",
        root_col="root_id",
    )
    formal.write_metacell_outputs(result, output_dir)
    mapping = result.cell_to_metacell.copy()

    root_truth = pd.read_csv(truth_dir / "TRUE_CELL_ROOT_MEMBERSHIP.csv")
    branch_truth = pd.read_csv(truth_dir / "TRUE_CELL_BRANCH.csv")
    pt_truth = pd.read_csv(truth_dir / "TRUE_PSEUDOTIME.csv")

    root_comp = composition_table(mapping, root_truth, ["root_id"], "root_probability")
    branch_comp = composition_table(
        mapping, branch_truth, ["stage", "root_id", "branch_id"], "branch_probability"
    )
    state_comp = composition_table(
        mapping, branch_truth, ["stage", "root_id", "state_id"], "state_probability"
    )
    root_comp.to_csv(output_dir / "METACELL_ROOT_COMPOSITION.csv", index=False)
    branch_comp.to_csv(output_dir / "METACELL_BRANCH_COMPOSITION.csv", index=False)
    state_comp.to_csv(output_dir / "METACELL_STATE_COMPOSITION.csv", index=False)

    summary = result.metacell_summary[["metacell_id", "n_cells"]].copy()
    summary = summary.merge(
        dominant_summary(root_comp, "root_id", "root_probability", "root"), on="metacell_id"
    )
    for stage in ("T0", "T1"):
        stage_comp = branch_comp[branch_comp["stage"] == stage].copy()
        dom = dominant_summary(stage_comp, "branch_id", "branch_probability", f"branch_{stage}")
        summary = summary.merge(dom, on="metacell_id", how="left")

    mapped_cells = mapping[["cell_name", "metacell_id"]].copy()
    obs_truth = adata.obs.copy()
    obs_truth["cell_name"] = adata.obs_names.astype(str)
    audit_cols = ["cell_name", "event_exposure", "confounded_exposure"]
    audit_cells = mapped_cells.merge(obs_truth[audit_cols], on="cell_name", validate="one_to_one")
    audit_summary = audit_cells.groupby("metacell_id", as_index=False).agg(
        event_exposed_fraction=("event_exposure", "mean"),
        confounded_fraction=("confounded_exposure", "mean"),
    )
    pt_cells = mapped_cells.merge(pt_truth, left_on="cell_name", right_on="cell_id", how="left")
    pt_summary = pt_cells.groupby("metacell_id", as_index=False).agg(
        median_true_pseudotime=("pseudotime", "median"),
        min_true_pseudotime=("pseudotime", "min"),
        max_true_pseudotime=("pseudotime", "max"),
    )
    summary = summary.merge(audit_summary, on="metacell_id").merge(pt_summary, on="metacell_id")
    for field in ("root", "branch_T0", "branch_T1"):
        summary[f"ambiguous_{field}"] = summary[f"{field}_purity"] < args.ambiguity_threshold
    summary["hard_truth_eligible"] = ~summary[
        ["ambiguous_root", "ambiguous_branch_T0", "ambiguous_branch_T1"]
    ].any(axis=1)
    summary.to_csv(output_dir / "METACELL_TRUTH_SUMMARY.csv", index=False)

    expected_cells = set(adata.obs_names.astype(str))
    mapped_set = set(mapping["cell_name"].astype(str))
    root_sums = root_comp.groupby("metacell_id")["root_probability"].sum()
    forbidden = ("truth", "branch", "root_id", "event_exposure", "confounded")
    leaked_columns = [c for c in result.adata.obs.columns if any(x in str(c).lower() for x in forbidden)]
    integrity = {
        "formal_module": str(FORMAL_MODULE),
        "formal_source_mode": result.source_mode,
        "n_cells": int(adata.n_obs),
        "n_metacells": int(result.adata.n_obs),
        "mapping_coverage": float(len(mapped_set & expected_cells) / max(1, len(expected_cells))),
        "mapping_is_one_to_one": bool(mapping["cell_name"].is_unique),
        "root_probability_max_abs_sum_error": float(np.max(np.abs(root_sums - 1.0))),
        "truth_columns_in_method_metacell_adata": leaked_columns,
        "method_input_obs_columns": list(map(str, method_adata.obs.columns)),
        "source_truth_obs_excluded_before_construction": len(method_adata.obs.columns) == 0,
        "truth_leakage_check_pass": not leaked_columns,
        "ambiguity_threshold": float(args.ambiguity_threshold),
        "hard_truth_eligible_fraction": float(summary["hard_truth_eligible"].mean()),
    }
    if integrity["mapping_coverage"] != 1.0 or not integrity["mapping_is_one_to_one"]:
        raise ValueError(f"Metacell mapping integrity failed: {integrity}")
    if integrity["root_probability_max_abs_sum_error"] > 1e-8:
        raise ValueError(f"Root soft truth does not sum to one: {integrity}")
    if leaked_columns:
        raise ValueError(f"Truth leaked into method metacell input: {leaked_columns}")
    (output_dir / "METACELL_TRUTH_INTEGRITY.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--truth-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="synthetic_event_gold")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-pca", type=int, default=30)
    parser.add_argument("--target-size", type=int, default=50)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--max-size", type=int, default=150)
    parser.add_argument("--birch-multiplier", type=float, default=1.5)
    parser.add_argument("--spatial-weight", type=float, default=0.35)
    parser.add_argument("--tiny-merge-size", type=int, default=5)
    parser.add_argument("--tiny-merge-min-comp-similarity", type=float, default=0.55)
    parser.add_argument("--tiny-merge-max-spatial-nn-factor", type=float, default=4.0)
    parser.add_argument("--ambiguity-threshold", type=float, default=0.70)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
