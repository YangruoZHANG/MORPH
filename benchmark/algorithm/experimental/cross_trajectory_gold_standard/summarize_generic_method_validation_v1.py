#!/usr/bin/env python3
"""Summarize frozen positive/null runs without using truth during fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--settings", nargs="+", default=["CORE_05", "CORE_06", "CORE_19"])
    parser.add_argument("--replicate", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit_rows = []
    score_rows = []
    for setting in args.settings:
        candidates = sorted((args.runs_root / "core" / setting).glob(f"replicate_{args.replicate:02d}_seed_*"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Expected one run for {setting} replicate {args.replicate}, found {len(candidates)}")
        run_dir = candidates[0]
        config = json.loads((run_dir / "SCENARIO_CONFIG.json").read_text(encoding="utf-8"))
        fixed = pd.read_csv(run_dir / "generic_upstream_v1_exact_root" / "FIXED_ROOT_INPUT.csv")
        root_to_pos = dict(zip(fixed["truth_root_id"].astype(str), range(len(fixed))))
        events = pd.read_csv(run_dir / "event" / "truth" / "TRUE_EVENTS.csv")
        event = events.iloc[0]
        diagnostics = pd.read_csv(run_dir / "generic_relation_refinement_v1" / "RELATION_DIAGNOSTICS.csv")
        target = diagnostics[
            diagnostics["source_root_pos"].astype(int).eq(root_to_pos[str(event.source_root)])
            & diagnostics["receiver_root_pos"].astype(int).eq(root_to_pos[str(event.receiver_root)])
        ]
        if len(target) != 1:
            raise ValueError(f"Target relation is not unique for {setting}")
        target = target.iloc[0]
        status = json.loads((run_dir / "generic_relation_refinement_v1" / "RUN_STATUS.json").read_text(encoding="utf-8"))
        audit_rows.append({
            "setting_id": setting,
            "event_type": config["event_type"],
            "effect_level": "null" if float(config["effect_strength"]) == 0.0 else config["effect_level"],
            "effect_strength": config["effect_strength"],
            "truth_causal": as_bool(event.causal),
            "truth_source": str(event.source_root),
            "truth_receiver": str(event.receiver_root),
            "n_exposed_receiver_cells": int(event.n_exposed_receiver_cells),
            "target_effect_size": float(target.effect_size),
            "target_permutation_p": float(target.permutation_p),
            "target_q_value": float(target.q_value),
            "target_called": as_bool(target.is_called),
            "target_decision_reason": str(target.decision_reason),
            "n_called_events": int(status["n_called_events"]),
            "n_accepted_rewires": int(status["n_accepted_rewires"]),
            "all_refined_roots_are_trees": bool(status["all_refined_roots_are_trees"]),
            "inference_unit": status.get("inference_unit", "cell"),
            "spatial_block_permutation": status.get("spatial_block_permutation", True),
        })
        stage = pd.read_csv(run_dir / "STAGE_SCORE_SUMMARY.csv")
        score_rows.append(stage)

    audit = pd.DataFrame(audit_rows)
    scores = pd.concat(score_rows, ignore_index=True)
    audit.to_csv(args.output_dir / "IDENTIFIABILITY_AUDIT.csv", index=False)
    scores.to_csv(args.output_dir / "STAGE_COMPARISON.csv", index=False)

    positives = audit[audit["truth_causal"]]
    nulls = audit[~audit["truth_causal"]]
    positive_detection_rate = float(positives["target_called"].mean()) if len(positives) else float("nan")
    null_false_call_rate = float((nulls["n_called_events"] > 0).mean()) if len(nulls) else float("nan")
    decision = {
        "protocol": "BENCHMARK_SPEC_V1",
        "evaluated_settings": args.settings,
        "evaluated_replicates_per_setting": 1,
        "positive_target_detection_rate": positive_detection_rate,
        "null_false_call_rate": null_false_call_rate,
        "null_false_rewire_rate": float((nulls["n_accepted_rewires"] > 0).mean()) if len(nulls) else float("nan"),
        "all_outputs_connected_acyclic": bool(audit["all_refined_roots_are_trees"].all()),
        "relation_aware_topology_claim_supported": False,
        "supported_interim_claim": "conservative relation-aware abstention with candidate-legal tree refinement machinery",
        "reason": (
            "The robust cell-level cross-fit gate avoided false rewiring under the tested global null, "
            "but did not recover the frozen medium or strong positive event; M4-to-M5 topology gain was zero."
        ),
        "interpretation_boundary": (
            "These three runs are an engineering/power diagnostic, not a paper-level performance estimate. "
            "The full frozen replicate grid remains required."
        ),
    }
    (args.output_dir / "VALIDATION_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    report = f"""# Generic relation-refinement validation v1

## Decision

The current implementation preserves candidate legality, connectivity, and acyclicity, and it correctly abstains in the tested global-null setting. It does **not** yet support a claim that tree-tree relations improve the final topology: neither the medium nor strong positive event was called, and M4-to-M5 topology gain was zero.

The earlier metacell-scale association statistic was rejected because the same source-receiver signal appeared in the paired null setting. The retained detector uses cell-level two-fold cross-fitting, spatial-block permutation, source competition, FDR, and constrained metacell rewiring.

## Frozen run audit

{audit.to_markdown(index=False)}

## Stage-wise truth scores

{scores[["setting_id", "stage", "forest_root_F1", "assignment_accuracy", "forest_branch_ARI", "forest_pt_SR", "event_F1", "final_topology_F1", "distance_T1_to_true_T1", "rewiring_gain_delta_d", "false_rewiring_under_global_null"]].to_markdown(index=False)}

For the global-null setting, event F1 = 1 follows the empty-truth/empty-prediction convention and must be read as a true negative, not positive-event recovery. Null safety is reported separately by false-call and false-rewire rates.

## Interpretation

- Root recovery and hard assignment are strong in these fixed-root runs.
- Branch reconstruction remains weak and is an upstream limitation.
- The null safety result is encouraging but is based on one frozen replicate.
- Positive-event power is currently inadequate, so relation-aware rewiring must remain experimental.
- The complete 900-core and 240-stress grid is required before manuscript performance claims.
"""
    (args.output_dir / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
