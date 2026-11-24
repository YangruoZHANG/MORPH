#!/usr/bin/env python3
"""Final audit for selected C1_v2 + C3_v5.1 on development seeds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUTPUT = REPO / "outputs" / "final_tree_method_development"
RUNS = (
    (1, 7112101, REPO / "outputs" / "cross_trajectory_gold_standard" / "protocol_runs_v1" / "core" / "CORE_05" / "replicate_01_seed_7112101"),
    (2, 7113110, OUTPUT / "independent_seed_protocol_runs" / "core" / "CORE_05" / "replicate_02_seed_7113110"),
    (3, 7114119, OUTPUT / "independent_seed_protocol_runs" / "core" / "CORE_05" / "replicate_03_seed_7114119"),
    (4, 7115128, OUTPUT / "independent_seed_protocol_runs" / "core" / "CORE_05" / "replicate_04_seed_7115128"),
)


def edge_digest(path: Path) -> str:
    frame = pd.read_csv(path)
    values = sorted(
        (str(row.root_id), min(int(row.source), int(row.target)), max(int(row.source), int(row.target)))
        for row in frame.itertuples(index=False)
    )
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def branch_metrics(run: Path, prediction: Path) -> dict[str, float]:
    truth = pd.read_csv(run / "formal_metacell" / "METACELL_BRANCH_COMPOSITION.csv")
    truth = truth[truth.stage.astype(str).eq("T1")].sort_values("branch_probability", ascending=False).drop_duplicates("metacell_id")
    truth = truth[truth.branch_probability.astype(float).ge(0.70)].rename(columns={"metacell_id": "unit_id", "branch_id": "truth_branch"})
    pred = pd.read_csv(prediction / "PRED_BRANCH.csv").rename(columns={"branch_id": "pred_branch"})
    joined = truth.merge(pred, on=["unit_id", "root_id"])
    scores, weights = [], []
    for _, group in joined.groupby("root_id"):
        scores.append(adjusted_rand_score(group.truth_branch, group.pred_branch))
        weights.append(len(group))
    return {
        "root_joint_ARI": adjusted_rand_score(
            joined.root_id.astype(str) + "|" + joined.truth_branch.astype(str),
            joined.root_id.astype(str) + "|" + joined.pred_branch.astype(str),
        ),
        "rootwise_macro_ARI": float(np.mean(scores)),
        "rootwise_weighted_ARI": float(np.average(scores, weights=weights)),
    }


def main() -> None:
    rows, equivalence = [], []
    for replicate, seed, run in RUNS:
        selected = OUTPUT / "C1_v2_C3_v5_1" / f"CORE_05_seed_{seed}"
        if replicate == 1:
            baseline_score = pd.read_csv(run / "STAGE_SCORE_SUMMARY.csv")
            baseline_score = baseline_score[baseline_score.stage.astype(str).eq("M3")].iloc[0]
            baseline_prediction = run / "pred_M3"
            reference = OUTPUT / "C1_C3_v2" / "CORE_05_seed_7112101"
        else:
            baseline_score = pd.read_csv(run / "score_M3" / "SCORE_SUMMARY.csv").iloc[0]
            baseline_prediction = run / "pred_M3"
            reference = run / "C1_v1_C3_v2"
        selected_score = pd.read_csv(selected / "score" / "SCORE_SUMMARY.csv").iloc[0]
        for method, score, prediction in (
            ("baseline_M3", baseline_score, baseline_prediction),
            ("selected_C1_v2_C3_v5_1", selected_score, selected / "prediction"),
        ):
            rows.append({
                "replicate": replicate, "seed": seed, "role": "diagnostic_existing" if replicate == 1 else "development_replication",
                "method": method, "forest_root_F1": score.forest_root_F1,
                "assignment_accuracy": score.assignment_accuracy,
                "official_forest_branch_ARI": score.forest_branch_ARI,
                **branch_metrics(run, prediction), "forest_pt_SR": score.forest_pt_SR,
                "final_topology_F1": score.final_topology_F1,
                "distance_T1_to_true_T1": score.distance_T1_to_true_T1,
                "cycle_violating_roots": score.cycle_violating_roots,
                "disconnected_roots": score.disconnected_roots,
            })
        selected_digest = edge_digest(selected / "C3_BRANCH_SCALE_T0_EDGES.csv")
        reference_digest = edge_digest(reference / "C3_BRANCH_SCALE_T0_EDGES.csv")
        candidate = pd.read_csv(selected / "C1_MULTIRESOLUTION_CANDIDATE_EDGES.csv")
        t0 = pd.read_csv(selected / "C3_BRANCH_SCALE_T0_EDGES.csv")
        equivalence.append({
            "replicate": replicate, "seed": seed,
            "selected_T0_sha256": selected_digest, "reference_C1v1_C3v2_T0_sha256": reference_digest,
            "T0_exactly_preserved": selected_digest == reference_digest,
            "reserve_edges_in_T0": int(t0.edge_graph_role.astype(str).eq("trajectory_local_union").sum()),
            "n_full_trajectory_eligible_candidates": int(candidate.trajectory_edge_eligible.sum()),
            "n_T0_active_candidates": int(candidate.T0_role_active.sum()),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "SELECTED_C1_C3_V5_1_SCORE_COMPARISON.csv", index=False)
    pd.DataFrame(equivalence).to_csv(OUTPUT / "SELECTED_C1_C3_V5_1_EQUIVALENCE_AUDIT.csv", index=False)

    dev = result[result.role.eq("development_replication")]
    wide = dev.pivot(index=["replicate", "seed"], columns="method")
    metrics = ["official_forest_branch_ARI", "rootwise_weighted_ARI", "forest_pt_SR", "final_topology_F1", "distance_T1_to_true_T1"]
    paired_rows = []
    for replicate, seed in wide.index:
        row = {"replicate": replicate, "seed": seed}
        for metric in metrics:
            row[f"baseline_{metric}"] = wide.loc[(replicate, seed), (metric, "baseline_M3")]
            row[f"selected_{metric}"] = wide.loc[(replicate, seed), (metric, "selected_C1_v2_C3_v5_1")]
            row[f"delta_{metric}"] = row[f"selected_{metric}"] - row[f"baseline_{metric}"]
        paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUTPUT / "SELECTED_C1_C3_V5_1_PAIRED_DEVELOPMENT_DELTAS.csv", index=False)
    summary = {
        "selected_method": "C1_candidate_multires_v2+C3_T0_highconf_v5_1",
        "development_replicates": [2, 3, 4],
        "heldout_replicates_touched": [],
        "all_T0_edges_exactly_match_frozen_highconf_reference": bool(pd.DataFrame(equivalence).T0_exactly_preserved.all()),
        "reserve_edges_entering_T0": int(pd.DataFrame(equivalence).reserve_edges_in_T0.sum()),
        "paired_mean_delta": {metric: float(paired[f"delta_{metric}"].mean()) for metric in metrics},
        "improvement_count_of_3": {
            metric: int((paired[f"delta_{metric}"] > 0).sum()) if metric != "distance_T1_to_true_T1" else int((paired[f"delta_{metric}"] < 0).sum())
            for metric in metrics
        },
        "candidate_fraction_of_complete_graph_representability_ceiling_all_4_seeds": 0.930556,
        "paper_ready": False,
        "remaining_reason": "C3 branch recovery is seed-sensitive; relation/refinement gates have not been restarted; locked heldout is untouched",
    }
    (OUTPUT / "SELECTED_C1_C3_V5_1_DECISION.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# Selected C1/C3 method

Selected: `C1_candidate_multires_v2 + C3_T0_highconf_v5_1`.

## Candidate graph

The full typed candidate graph reaches 93.1% of the state edges that are
representable at all after metacell projection across seeds 1-4. One-sided
local-union edges are retained as trajectory reserve; they do not enter T0.

## Independent development replication (replicates 2-4)

- Mean root-wise weighted Branch ARI delta: {summary['paired_mean_delta']['rootwise_weighted_ARI']:+.4f} ({summary['improvement_count_of_3']['rootwise_weighted_ARI']}/3 improved).
- Mean PT Spearman delta: {summary['paired_mean_delta']['forest_pt_SR']:+.4f} ({summary['improvement_count_of_3']['forest_pt_SR']}/3 improved).
- Mean topology F1 delta: {summary['paired_mean_delta']['final_topology_F1']:+.4f} ({summary['improvement_count_of_3']['final_topology_F1']}/3 improved).
- Mean topology distance delta: {summary['paired_mean_delta']['distance_T1_to_true_T1']:+.2f} ({summary['improvement_count_of_3']['distance_T1_to_true_T1']}/3 improved; lower is better).

## Decision

Keep C1_v2 and C3_v5.1 as the selected first-stage design. Do not claim the
complete method is paper-ready: branch recovery still regressed on one of
three independent development seeds, no relation/refinement redesign has been
started, and locked held-out replicates 21-30 remain untouched.
"""
    (OUTPUT / "SELECTED_C1_C3_V5_1_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
