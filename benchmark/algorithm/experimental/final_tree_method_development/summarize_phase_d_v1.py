#!/usr/bin/env python3
"""Aggregate frozen Phase D runs and issue a non-negotiable gate decision."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def replicate_from(path: Path) -> int:
    match = re.search(r"replicate_(\d+)", str(path))
    if not match:
        raise ValueError(f"No replicate token in {path}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-d-subdir", default="phase_d_v5_final_dev")
    parser.add_argument("--ablation-subdir", default="phase_d_stage_ablation_v1_dev")
    parser.add_argument("--development-replicates", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--confirmation-replicates", nargs="+", type=int, default=[7, 8])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metric_files = sorted(args.run_root.rglob(
        f"{args.phase_d_subdir}/PHASE_D_INFERRED_REFINEMENT_METRICS.csv"
    ))
    rows: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    pruning: list[pd.DataFrame] = []
    stage_rows: list[pd.DataFrame] = []
    for metric_file in metric_files:
        replicate = replicate_from(metric_file)
        split = (
            "DEV" if replicate in args.development_replicates
            else "CONFIRMATION" if replicate in args.confirmation_replicates
            else "EXCLUDED"
        )
        if split == "EXCLUDED":
            continue
        frame = pd.read_csv(metric_file)
        frame.insert(1, "replicate", replicate)
        frame.insert(2, "split", split)
        frame.insert(3, "run_directory", metric_file.parents[1].name)
        rows.append(frame)
        event_file = metric_file.parent / "PHASE_D_EVENT_REFINEMENT_AUDIT.csv"
        if event_file.exists() and event_file.stat().st_size:
            try:
                local = pd.read_csv(event_file)
            except EmptyDataError:
                local = pd.DataFrame()
            if len(local):
                local.insert(0, "dataset", str(frame.iloc[0].dataset))
                local.insert(1, "replicate", replicate)
                local.insert(2, "split", split)
                audits.append(local)
        prune_file = metric_file.parent / "PHASE_D_PRUNING_RETENTION.csv"
        local_prune = pd.read_csv(prune_file)
        local_prune.insert(0, "dataset", str(frame.iloc[0].dataset))
        local_prune.insert(1, "replicate", replicate)
        local_prune.insert(2, "split", split)
        pruning.append(local_prune)
        ablation_file = metric_file.parents[1] / args.ablation_subdir / "PHASE_D_STAGE_ABLATION.csv"
        local_stage = pd.read_csv(ablation_file)
        local_stage.insert(1, "replicate", replicate)
        local_stage.insert(2, "split", split)
        stage_rows.append(local_stage)
    metrics = pd.concat(rows, ignore_index=True)
    event_audit = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    pruning_audit = pd.concat(pruning, ignore_index=True)
    stages = pd.concat(stage_rows, ignore_index=True)
    metrics.to_csv(args.output / "PHASE_D_INFERRED_REFINEMENT.csv", index=False)
    event_audit.to_csv(args.output / "PHASE_D_EVENT_REFINEMENT_AUDIT.csv", index=False)
    pruning_audit.to_csv(args.output / "PHASE_D_PRUNING_RETENTION.csv", index=False)
    stages.to_csv(args.output / "PHASE_D_STAGE_ABLATION.csv", index=False)
    null_rows = metrics[~metrics.causal_scenario.map(as_bool)].copy()
    null_rows["null_no_rewire"] = null_rows.n_accepted_relation_events.astype(int).eq(0)
    null_rows["false_rewire"] = ~null_rows.null_no_rewire
    null_rows.to_csv(args.output / "PHASE_D_NULL_ABSTENTION.csv", index=False)
    summaries: list[dict[str, object]] = []
    for split, split_frame in metrics.groupby("split", sort=True):
        for causal, group in split_frame.groupby(
            split_frame.causal_scenario.map(as_bool), sort=False
        ):
            summaries.append({
                "split": split,
                "causal_scenario": bool(causal),
                "n_runs": int(len(group)),
                "n_relation_calls": int(group.n_relation_calls.astype(int).sum()),
                "n_rewired": int(group.n_accepted_relation_events.astype(int).sum()),
                "n_significant_but_no_rewire": int(group.significant_but_no_rewire.astype(int).sum()),
                "mean_t1_gain": float(group.inferred_topology_gain.astype(float).mean()),
                "median_t1_gain": float(group.inferred_topology_gain.astype(float).median()),
                "positive_t1_gain_runs": int((group.inferred_topology_gain.astype(float) > 0).sum()),
                "negative_t1_gain_runs": int((group.inferred_topology_gain.astype(float) < 0).sum()),
                "mean_tfinal_gain": float(group.tfinal_topology_gain.astype(float).mean()),
                "false_rewire_runs": int((
                    (~group.causal_scenario.map(as_bool))
                    & group.n_accepted_relation_events.astype(int).gt(0)
                ).sum()),
                "event_boundary_retention": float(
                    group.event_boundary_retained.map(as_bool).mean()
                ),
                "upstream_violation_runs": int(group.upstream_violation.map(as_bool).sum()),
                "tfinal_legality_rate": float((
                    group.tfinal_candidate_legal.map(as_bool)
                    & group.tfinal_connected.map(as_bool)
                    & group.tfinal_acyclic.map(as_bool)
                ).mean()),
            })
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / "PHASE_D_SUMMARY.csv", index=False)
    stage_summary = stages.groupby(["split", "stage"], sort=False).agg(
        n_runs=("dataset", "size"),
        root_assignment_accuracy=("root_assignment_accuracy", "mean"),
        root_assignment_brier=("root_assignment_brier", "mean"),
        fine_edge_F1=("fine_edge_F1", "mean"),
        branch_ARI=("branch_ARI_rootwise", "mean"),
        pseudotime_SR=("pseudotime_spearman", "mean"),
        topology_F1=("topology_F1", "mean"),
        topology_distance=("topology_distance_to_true_T1", "mean"),
        event_F1=("event_F1", "mean"),
        rewiring_gain=("rewiring_gain", "mean"),
        false_rewire=("false_rewire", "sum"),
    ).reset_index()
    stage_summary.to_csv(args.output / "PHASE_D_STAGE_ABLATION_SUMMARY.csv", index=False)
    development = summary[
        summary.split.eq("DEV") & summary.causal_scenario.eq(True)
    ].iloc[0]
    confirmation = summary[
        summary.split.eq("CONFIRMATION") & summary.causal_scenario.eq(True)
    ].iloc[0]
    null_summary = summary[summary.causal_scenario.eq(False)]
    expected_stages = {f"M{index}" for index in range(7)}
    complete_stages = all(
        set(stages[stages.split.eq(split)].stage.astype(str)) == expected_stages
        for split in {"DEV", "CONFIRMATION"}
    )
    checks = {
        "all_36_runs_present": len(metrics) == 36,
        "m0_m6_complete": complete_stages,
        "development_inferred_gain_positive": float(development.mean_t1_gain) > 0,
        "confirmation_inferred_gain_positive": float(confirmation.mean_t1_gain) > 0,
        "no_negative_gain_runs": int(summary.negative_t1_gain_runs.sum()) == 0,
        "null_false_rewire_zero": int(null_summary.false_rewire_runs.sum()) == 0,
        "upstream_invariant": int(summary.upstream_violation_runs.sum()) == 0,
        "tfinal_candidate_legal_connected_acyclic": bool(
            np.isclose(summary.tfinal_legality_rate.astype(float), 1.0).all()
        ),
        "event_boundaries_retained": bool(
            np.isclose(summary.event_boundary_retention.astype(float), 1.0).all()
        ),
        "confirmation_has_positive_runs": int(confirmation.positive_t1_gain_runs) >= 1,
        "significant_relation_may_abstain": int(
            metrics.significant_but_no_rewire.astype(int).sum()
        ) > 0,
    }
    decision = {
        "phase": "D",
        "method_version": "INFERRED_REFINEMENT_V5_SUPPORTED_PROGRAM_BOUNDARY",
        "pruning_version": "ROOT_CONNECTED_EVENT_PROTECTED_V1",
        "relation_detector": "MULTICHANNEL_RELATION_V3_8",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "development_mean_causal_gain": float(development.mean_t1_gain),
        "confirmation_mean_causal_gain": float(confirmation.mean_t1_gain),
        "confirmation_positive_zero_negative": [
            int(confirmation.positive_t1_gain_runs),
            int(confirmation.n_runs - confirmation.positive_t1_gain_runs - confirmation.negative_t1_gain_runs),
            int(confirmation.negative_t1_gain_runs),
        ],
        "interpretation": (
            "Conservative local correction: positive mean held-out gain with frequent abstention; "
            "core-innovation status remains conditional on locked Phase E."
        ),
        "truth_used_for_selection": False,
        "truth_used_for_scoring_only": True,
    }
    (args.output / "PHASE_D_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
