#!/usr/bin/env python3
"""Development-only sensitivity analysis for Phase B primary relation gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize_phase_e_locked_v1 import as_bool, greedy_matches, prf
from run_phase_b_multichannel_detector_v6 import apply_competition


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def specificity_gate(evidence: pd.DataFrame, policy: str) -> pd.Series:
    raw_margin = evidence.raw_program_specificity_margin.gt(0.01)
    adjusted_margin = evidence.program_specificity_margin.gt(0.01)
    raw_rank = evidence.raw_program_specificity_rank.ge(0.80)
    adjusted_rank = evidence.program_specificity_rank.ge(0.80)
    if policy == "strict":
        return raw_margin & adjusted_margin & raw_rank & adjusted_rank
    if policy == "rank":
        return raw_rank & adjusted_rank
    if policy == "either":
        return (raw_margin | raw_rank) & (adjusted_margin | adjusted_rank)
    if policy == "rank_two_thirds":
        return (
            evidence.raw_program_specificity_rank.ge(2.0 / 3.0)
            & evidence.program_specificity_rank.ge(2.0 / 3.0)
        )
    raise ValueError(policy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--relation-subdir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    payloads = []
    for status_file in sorted(args.grid_root.rglob("TOPOLOGY_DEV_CASE_STATUS.json")):
        run = status_file.parent
        status = json.loads(status_file.read_text(encoding="utf-8"))
        scenario = json.loads((run / "SCENARIO_CONFIG.json").read_text(encoding="utf-8"))
        truth = pd.read_csv(run / "event" / "truth" / "TRUE_EVENTS.csv")
        evidence = pd.read_csv(run / args.relation_subdir / "PHASE_B_CMI_E_EVIDENCE.csv")
        payloads.append((status, scenario, truth, evidence))

    summary_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    for q_cutoff in (0.10, 0.15, 0.20, 0.25):
        for effect_cutoff in (0.10, 0.12, 0.15):
            for specificity in ("strict", "rank", "either", "rank_two_thirds"):
                for direction_policy in ("consistent", "raw_signed"):
                  for source_policy in ("max_gap", "relative_0.60", "enrichment_0.25"):
                    variant = (
                        f"q{q_cutoff:.2f}_effect{effect_cutoff:.2f}_"
                        f"{specificity}_{direction_policy}_{source_policy}"
                    )
                    total_tp = total_fp = total_fn = 0
                    variant_runs = []
                    for status, scenario, truth, original in payloads:
                        evidence = original.copy()
                        if source_policy == "max_gap":
                            source_gate = evidence.source_hypothesis_eligible.map(as_bool)
                        elif source_policy == "relative_0.60":
                            source_gate = evidence.source_program_relative_strength.ge(0.60)
                        else:
                            source_gate = evidence.source_program_enrichment.ge(0.25)
                        eligible = (
                            evidence.status.eq("tested")
                            & source_gate
                        )
                        direction = (
                            evidence.environment_direction_consistent.map(as_bool)
                            if direction_policy == "consistent"
                            else evidence.raw_monotone_statistic.abs().ge(effect_cutoff)
                        )
                        primary = (
                            eligible
                            & evidence.primary_channel_q.le(q_cutoff)
                            & evidence.raw_monotone_statistic.abs().ge(effect_cutoff)
                            & evidence.adjusted_monotone_statistic.abs().ge(effect_cutoff)
                            & specificity_gate(evidence, specificity)
                            & direction
                            & evidence.oof_predictive_gain.ge(-0.10)
                            & evidence.exposure_variance.gt(1e-5)
                            & evidence.positivity_overlap.ge(0.50)
                        )
                        evidence["is_called"] = primary
                        evidence["call_channel"] = np.where(primary, "primary_response", "abstain")
                        evidence["event_type"] = np.where(
                            evidence.raw_monotone_statistic.lt(0), "inhibition", "promotion"
                        )
                        evidence["sensitivity_signature_q"] = 1.0
                        signature_indices = list(evidence.index[primary])
                        if signature_indices:
                            evidence.loc[
                                signature_indices, "sensitivity_signature_q"
                            ] = bh(evidence.loc[
                                signature_indices, "signature_channel_p"
                            ].to_numpy(float))
                        collision = (
                            primary
                            & evidence.sensitivity_signature_q.le(0.10)
                            & evidence.signature_joint_statistic.ge(0.50)
                            & evidence.signature_support_raw_rho.ge(0.15)
                            & evidence.signature_support_raw_specificity_rank.ge(0.80)
                            & evidence.signature_environment_direction_consistent.map(as_bool)
                            & evidence.signature_environment_trust.ge(1.0 / 3.0)
                        )
                        evidence.loc[collision, "event_type"] = "collision_convergence"
                        evidence.loc[collision, "call_channel"] = "collision_signature"
                        evidence["call_strength"] = np.where(
                            collision,
                            evidence.signature_joint_statistic.fillna(0.0),
                            evidence.environment_adjusted_dependency.fillna(0.0),
                        )
                        apply_competition(evidence, event_local=True)
                        calls = evidence[evidence.is_called].reset_index(drop=True)
                        causal = truth[truth.causal.map(as_bool)].reset_index(drop=True)
                        tp = len(greedy_matches(calls, causal))
                        fp = len(calls) - tp
                        fn = len(causal) - tp
                        total_tp += tp
                        total_fp += fp
                        total_fn += fn
                        row = {
                            "variant": variant,
                            "setting": str(status["setting"]),
                            "scenario_event_type": str(scenario["event_type"]),
                            "effect_strength": float(scenario["effect_strength"]),
                            "causal": bool(len(causal)),
                            "n_true": len(causal), "n_calls": len(calls),
                            "tp": tp, "fp": fp, "fn": fn,
                        }
                        variant_runs.append(row)
                        run_rows.append(row)
                    frame = pd.DataFrame(variant_runs)
                    precision, recall, f1 = prf(total_tp, total_fp, total_fn)
                    null = frame[~frame.causal]
                    medium_strong = frame[
                        frame.causal & frame.effect_strength.ge(0.70)
                    ]
                    summary_rows.append({
                        "variant": variant,
                        "q_cutoff": q_cutoff,
                        "effect_cutoff": effect_cutoff,
                        "specificity_policy": specificity,
                        "direction_policy": direction_policy,
                        "source_policy": source_policy,
                        "event_precision": precision,
                        "event_recall": recall,
                        "event_F1": f1,
                        "null_false_call_rate": float(null.n_calls.gt(0).mean()),
                        "medium_strong_detection_rate": float(
                            medium_strong.tp.gt(0).mean()
                        ),
                        "n_tp": total_tp, "n_fp": total_fp, "n_fn": total_fn,
                    })
    summary = pd.DataFrame(summary_rows)
    summary["passes_gate"] = (
        summary.event_precision.ge(0.70)
        & summary.event_recall.ge(0.50)
        & summary.event_F1.ge(0.58)
        & summary.null_false_call_rate.le(0.10)
    )
    summary = summary.sort_values(
        ["passes_gate", "event_F1", "event_precision", "null_false_call_rate"],
        ascending=[False, False, False, True],
    )
    summary.to_csv(args.output / "PHASE_B_PRIMARY_GATE_SENSITIVITY.csv", index=False)
    pd.DataFrame(run_rows).to_csv(
        args.output / "PHASE_B_PRIMARY_GATE_RUN_LEVEL.csv", index=False
    )
    selected = summary.iloc[0].to_dict()
    (args.output / "PHASE_B_PRIMARY_GATE_DEV_SELECTION.json").write_text(
        json.dumps({
            "role": "DEVELOPMENT_SELECTION_ONLY",
            "selected": selected,
            "requires_independent_seed_validation": True,
        }, indent=2), encoding="utf-8",
    )
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
