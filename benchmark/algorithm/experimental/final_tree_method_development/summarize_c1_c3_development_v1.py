#!/usr/bin/env python3
"""Summarize C1/C3 development without modifying the frozen scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


SETTINGS = ("CORE_05", "CORE_06", "CORE_19")
SEED = 7112101


def corrected_branch_metrics(run_dir: Path, prediction: Path) -> dict[str, object]:
    truth = pd.read_csv(run_dir / "formal_metacell" / "METACELL_BRANCH_COMPOSITION.csv")
    truth = truth[truth["stage"].astype(str).eq("T1")]
    truth = truth.sort_values("branch_probability", ascending=False).drop_duplicates("metacell_id")
    truth = truth[truth["branch_probability"].astype(float).ge(0.70)].rename(
        columns={"metacell_id": "unit_id", "branch_id": "true_branch"}
    )
    pred = pd.read_csv(prediction / "PRED_BRANCH.csv").rename(columns={"branch_id": "pred_branch"})
    joined = truth.merge(pred, on=["unit_id", "root_id"], how="inner")
    per_root = []
    for root_id, group in joined.groupby("root_id"):
        per_root.append((str(root_id), len(group), adjusted_rand_score(group.true_branch, group.pred_branch)))
    return {
        "root_joint_ARI": adjusted_rand_score(
            joined.root_id.astype(str) + "|" + joined.true_branch.astype(str),
            joined.root_id.astype(str) + "|" + joined.pred_branch.astype(str),
        ),
        "rootwise_macro_ARI": float(np.mean([x[2] for x in per_root])),
        "rootwise_weighted_ARI": float(np.average([x[2] for x in per_root], weights=[x[1] for x in per_root])),
        "rootwise_ARI": "|".join(f"{root}:{score:.6f}" for root, _, score in per_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = (args.output or project / "outputs" / "final_tree_method_development").resolve()
    rows, c10_rows = [], []
    for setting in SETTINGS:
        run = project / "outputs" / "cross_trajectory_gold_standard" / "protocol_runs_v1" / "core" / setting / f"replicate_01_seed_{SEED}"
        baseline = pd.read_csv(run / "STAGE_SCORE_SUMMARY.csv")
        baseline = baseline[baseline["stage"].astype(str).eq("M3")].iloc[0].to_dict()
        method_dir = output / "C1_C3_v2" / f"{setting}_seed_{SEED}"
        method = pd.read_csv(method_dir / "score" / "SCORE_SUMMARY.csv").iloc[0].to_dict()
        for label, values, prediction in (
            ("baseline_M3", baseline, run / "pred_M3"),
            ("C1_v1_C3_v2", method, method_dir / "prediction"),
        ):
            corrected = corrected_branch_metrics(run, prediction)
            rows.append({
                "setting_id": setting, "seed": SEED, "method": label,
                "forest_root_F1": values["forest_root_F1"],
                "assignment_accuracy": values["assignment_accuracy"],
                "official_forest_branch_ARI": values["forest_branch_ARI"],
                **corrected,
                "forest_pt_SR": values["forest_pt_SR"],
                "final_topology_F1": values["final_topology_F1"],
                "distance_T1_to_true_T1": values["distance_T1_to_true_T1"],
            })
            c10_rows.append({
                "setting_id": setting, "method": label,
                "official_definition": "ARI(branch_id_true, branch_id_pred) across all roots",
                "problem": "truth branch IDs B0/B1/B2 are reused across roots while predictions are root-specific",
                "official_forest_branch_ARI": values["forest_branch_ARI"],
                **corrected,
                "scorer_modified": False,
            })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "C1_C3_DEVELOPMENT_SCORE_COMPARISON.csv", index=False)
    pd.DataFrame(c10_rows).to_csv(output / "C10_BRANCH_ARI_SEMANTIC_AUDIT_ALL_SETTINGS.csv", index=False)

    mean = comparison.groupby("method", as_index=False).agg(
        official_forest_branch_ARI=("official_forest_branch_ARI", "mean"),
        rootwise_weighted_ARI=("rootwise_weighted_ARI", "mean"),
        forest_pt_SR=("forest_pt_SR", "mean"),
        final_topology_F1=("final_topology_F1", "mean"),
        topology_distance=("distance_T1_to_true_T1", "mean"),
    )
    mean.to_csv(output / "C1_C3_DEVELOPMENT_MEAN_COMPARISON.csv", index=False)
    baseline_mean = mean[mean.method.eq("baseline_M3")].iloc[0]
    method_mean = mean[mean.method.eq("C1_v1_C3_v2")].iloc[0]
    decision = {
        "C1_candidate_multires_v1": "keep_provisional",
        "C1_candidate_multires_v2": "reject_structure_regression",
        "C3_T0_branchscale_v1": "reject_branch_oversegmentation",
        "C3_T0_branchscale_v2": "keep_provisional",
        "relation_development_started": False,
        "locked_heldout_claim": False,
        "independent_seed_replication": False,
        "mean_delta_vs_M3": {
            "official_forest_branch_ARI": float(method_mean.official_forest_branch_ARI - baseline_mean.official_forest_branch_ARI),
            "rootwise_weighted_ARI": float(method_mean.rootwise_weighted_ARI - baseline_mean.rootwise_weighted_ARI),
            "forest_pt_SR": float(method_mean.forest_pt_SR - baseline_mean.forest_pt_SR),
            "final_topology_F1": float(method_mean.final_topology_F1 - baseline_mean.final_topology_F1),
            "topology_distance_reduction": float(baseline_mean.topology_distance - method_mean.topology_distance),
        },
    }
    (output / "C1_C3_DEVELOPMENT_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    report = f"""# C1/C3 development decision

## Decision

- Keep provisionally: `C1_candidate_multires_v1` and `C3_T0_branchscale_v2`.
- Reject: `C3_T0_branchscale_v1` (dominant-branch oversegmentation).
- Reject: `C1_candidate_multires_v2` (higher local coverage but worse topology).
- Do not start relation development from this script.

## Mean change versus M3

- Official global Branch ARI: {decision['mean_delta_vs_M3']['official_forest_branch_ARI']:+.4f}
- Root-wise weighted Branch ARI: {decision['mean_delta_vs_M3']['rootwise_weighted_ARI']:+.4f}
- PT Spearman: {decision['mean_delta_vs_M3']['forest_pt_SR']:+.4f}
- Final topology F1: {decision['mean_delta_vs_M3']['final_topology_F1']:+.4f}
- Topology distance reduction: {decision['mean_delta_vs_M3']['topology_distance_reduction']:+.2f}

## C10 caveat

The frozen scorer is unchanged. Its global branch ARI merges truth labels with
the same text across different roots, while predicted branch labels are
root-specific. Root-joint and root-wise ARI are therefore reported as a
semantic audit, not as a replacement score selected after seeing results.

## Remaining gate

CORE_05, CORE_06, and CORE_19 use the same seed and metacell geometry. They are
different event conditions, not independent replicates. C1/C3 are not
paper-ready until the exact frozen code is tested on predeclared independent
development seeds and finally on the locked held-out benchmark.
"""
    (output / "C1_C3_DEVELOPMENT_DECISION.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
