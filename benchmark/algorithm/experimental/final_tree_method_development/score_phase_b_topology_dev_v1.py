#!/usr/bin/env python3
"""Score Phase B relation calls on the topology-identifiable DEV grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize_phase_e_locked_v1 import as_bool, greedy_matches, prf, event_region_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--relation-subdir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    statuses = sorted(args.grid_root.rglob("TOPOLOGY_DEV_CASE_STATUS.json"))
    if not statuses:
        raise RuntimeError("No topology-identifiable DEV cases")

    run_rows = []
    match_rows = []
    evidence_frames = []
    total_tp = total_fp = total_fn = 0
    for status_file in statuses:
        run = status_file.parent
        status = json.loads(status_file.read_text(encoding="utf-8"))
        scenario = json.loads((run / "SCENARIO_CONFIG.json").read_text(encoding="utf-8"))
        truth = pd.read_csv(run / "event" / "truth" / "TRUE_EVENTS.csv")
        causal = truth[truth.causal.map(as_bool)].reset_index(drop=True)
        evidence = pd.read_csv(run / args.relation_subdir / "PHASE_B_CMI_E_EVIDENCE.csv")
        evidence.insert(0, "setting", str(status["setting"]))
        evidence.insert(1, "replicate", int(status["replicate"]))
        evidence.insert(2, "seed", int(status["seed"]))
        true_pairs = {
            (str(row.source_root), str(row.receiver_root))
            for row in causal.itertuples(index=False)
        }
        evidence["is_true_direction"] = [
            (str(row.source_root), str(row.receiver_root)) in true_pairs
            for row in evidence.itertuples(index=False)
        ]
        true_typed_pairs = {
            (str(row.source_root), str(row.receiver_root), str(row.event_type))
            for row in causal.itertuples(index=False)
        }
        evidence["is_true_typed_direction"] = [
            (str(row.source_root), str(row.receiver_root), str(row.event_type))
            in true_typed_pairs
            for row in evidence.itertuples(index=False)
        ]
        evidence["scenario_event_type"] = str(scenario.get("event_type", ""))
        evidence["scenario_effect_strength"] = float(scenario.get("effect_strength", 0.0))
        evidence_frames.append(evidence)

        calls = evidence[evidence.is_called.map(as_bool)].reset_index(drop=True)
        matches = greedy_matches(calls, causal)
        tp, fp, fn = len(matches), len(calls) - len(matches), len(causal) - len(matches)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        run_rows.append({
            "setting": str(status["setting"]), "replicate": int(status["replicate"]),
            "seed": int(status["seed"]), "scenario_event_type": str(scenario.get("event_type", "")),
            "effect_strength": float(scenario.get("effect_strength", 0.0)),
            "causal_scenario": bool(len(causal)), "n_true_events": len(causal),
            "n_calls": len(calls), "event_tp": tp, "event_fp": fp, "event_fn": fn,
        })
        obs = ad.read_h5ad(run / "event" / "synthetic.h5ad").obs
        for call_index, truth_index in matches:
            call, event = calls.iloc[call_index], causal.iloc[truth_index]
            region_p, region_r, region_f1, mapped_branch, branch_ok = event_region_metrics(
                call, event, obs
            )
            predicted_center = 0.5 * (
                float(call.receiver_window_start) + float(call.receiver_window_end)
            )
            truth_center = 0.5 * (
                float(event.receiver_tau_start) + float(event.receiver_tau_end)
            )
            match_rows.append({
                "setting": str(status["setting"]), "event_id": str(event.event_id),
                "truth_event_type": str(event.event_type),
                "predicted_event_type": str(call.event_type),
                "source_root_correct": float(str(call.source_root) == str(event.source_root)),
                "receiver_root_correct": float(str(call.receiver_root) == str(event.receiver_root)),
                "relation_type_correct": float(str(call.event_type) == str(event.event_type)),
                "truth_affected_branch": str(event.affected_branch),
                "predicted_affected_branch": str(call.affected_branch),
                "mapped_truth_branch": mapped_branch,
                "affected_branch_correct": branch_ok,
                "event_region_precision": region_p, "event_region_recall": region_r,
                "event_region_F1": region_f1,
                "event_location_error": abs(predicted_center - truth_center),
            })

    runs = pd.DataFrame(run_rows)
    matches = pd.DataFrame(match_rows)
    evidence_all = pd.concat(evidence_frames, ignore_index=True)
    precision, recall, event_f1 = prf(total_tp, total_fp, total_fn)
    null_runs = runs[~runs.causal_scenario]
    causal_runs = runs[runs.causal_scenario]
    summary = pd.DataFrame([{
        "n_runs": len(runs), "n_causal_runs": len(causal_runs),
        "n_null_runs": len(null_runs), "event_tp": total_tp,
        "event_fp": total_fp, "event_fn": total_fn,
        "event_precision": precision, "event_recall": recall, "event_F1": event_f1,
        "relation_type_accuracy": float(matches.relation_type_correct.mean()) if len(matches) else 0.0,
        "source_root_accuracy": float(matches.source_root_correct.mean()) if len(matches) else 0.0,
        "receiver_root_accuracy": float(matches.receiver_root_correct.mean()) if len(matches) else 0.0,
        "affected_branch_accuracy": float(matches.affected_branch_correct.mean()) if len(matches) else 0.0,
        "affected_branch_F1": float(matches.event_region_F1.mean()) if len(matches) else 0.0,
        "mean_event_location_error": float(matches.event_location_error.mean()) if len(matches) else np.nan,
        "null_false_call_rate": float(null_runs.n_calls.gt(0).mean()) if len(null_runs) else np.nan,
        "causal_detection_rate": float(causal_runs.event_tp.gt(0).mean()) if len(causal_runs) else np.nan,
    }])
    summary.to_csv(args.output / "PHASE_B_EVENT_DETECTION_METRICS.csv", index=False)
    runs.to_csv(args.output / "PHASE_B_RUN_LEVEL_METRICS.csv", index=False)
    matches.to_csv(args.output / "PHASE_B_MATCHED_EVENT_METRICS.csv", index=False)
    null_runs.to_csv(args.output / "PHASE_B_NULL_CONTROL.csv", index=False)
    evidence_all.to_csv(args.output / "PHASE_B_CMI_E_EVIDENCE.csv", index=False)
    evidence_all.to_csv(args.output / "PHASE_B_RELATION_SCORE_DISTRIBUTION.csv", index=False)

    score_rows = []
    for (event_type, effect), group in evidence_all[evidence_all.is_true_direction].groupby(
        ["scenario_event_type", "scenario_effect_strength"], dropna=False
    ):
        score_rows.append({
            "scenario_event_type": event_type, "effect_strength": effect,
            "n": len(group),
            "mean_raw_dependency": float(group.raw_dependency.mean()),
            "mean_environment_adjusted_dependency": float(
                group.environment_adjusted_dependency.mean()
            ),
            "mean_joint_statistic": float(group.joint_dependency_statistic.mean()),
            "mean_environment_drop": float(group.environment_drop.mean()),
            "mean_environment_trust": float(group.environment_trust.mean()),
            "call_rate": float(group.is_called.map(as_bool).mean()),
        })
    pd.DataFrame(score_rows).to_csv(
        args.output / "PHASE_B_EFFECT_STRENGTH_RESPONSE.csv", index=False
    )
    row = summary.iloc[0]
    passed = bool(
        row.null_false_call_rate <= 0.10
        and row.event_precision >= 0.70
        and row.event_recall >= 0.50
        and row.affected_branch_accuracy >= 0.50
    )
    decision = {
        "phase": "B", "role": "DEVELOPMENT_ONLY",
        "decision": "PASS" if passed else "FAIL",
        "criteria": {
            "null_false_call_rate_le_0.10": bool(row.null_false_call_rate <= 0.10),
            "event_precision_ge_0.70": bool(row.event_precision >= 0.70),
            "event_recall_ge_0.50": bool(row.event_recall >= 0.50),
            "affected_branch_accuracy_ge_0.50": bool(row.affected_branch_accuracy >= 0.50),
        },
        "metrics": {key: (None if pd.isna(value) else float(value)) for key, value in row.items()},
    }
    (args.output / "PHASE_B_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    failure = [
        "# Phase B failure attribution",
        "",
        f"Decision: {decision['decision']}",
        "",
        "The detector is audited on topology-identifiable development data only.",
        "Failed gates must be attributed before changing significance thresholds.",
        "",
        json.dumps(decision["criteria"], indent=2),
    ]
    (args.output / "PHASE_B_FAILURE_ATTRIBUTION.md").write_text(
        "\n".join(failure) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
