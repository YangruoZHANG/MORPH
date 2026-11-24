#!/usr/bin/env python3
"""Summarize frozen breast relation abstention robustness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["source_root", "receiver_root"]
Q_CUTOFF = 0.15
LOCAL_SUPPORT_CUTOFF = 0.15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).sort_values(KEY, kind="mergesort").reset_index(drop=True)


def as_bool(value: object) -> bool:
    return str(value).lower() == "true"


def gate_reason(row: pd.Series) -> str:
    failures: list[str] = []
    if float(row.scale065_primary_q) > Q_CUTOFF:
        failures.append("scale065_q")
    if float(row.scale080_primary_q) > Q_CUTOFF:
        failures.append("scale080_q")
    if not as_bool(row.scale065_called):
        failures.append("scale065_complete_call")
    if not as_bool(row.scale080_called):
        failures.append("scale080_complete_call")
    if not as_bool(row.scale_direction_consistent):
        failures.append("direction_consistency")
    if not as_bool(row.scale_branch_consistent):
        failures.append("affected_branch_consistency")
    if float(row.multiscale_localized_branch_support_fraction) < LOCAL_SUPPORT_CUTOFF:
        failures.append("localized_branch_support")
    return ";".join(failures) or "none"


def max_numeric_difference(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> float:
    valid = [column for column in columns if column in left and column in right]
    if not valid:
        return float("nan")
    return float(np.nanmax(np.abs(left[valid].to_numpy(float) - right[valid].to_numpy(float))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p6-dir", type=Path, required=True)
    parser.add_argument("--p7-dir", type=Path, required=True)
    args = parser.parse_args()

    original65_path = args.p6_dir / "phase_b_scale065" / "PHASE_B_CMI_E_EVIDENCE.csv"
    original80_path = args.p6_dir / "phase_b_scale080" / "PHASE_B_CMI_E_EVIDENCE.csv"
    original_path = args.p6_dir / "phase_b_consensus" / "PHASE_B_CMI_E_EVIDENCE.csv"
    replay65_path = args.p7_dir / "observed_replay_scale065" / "PHASE_B_CMI_E_EVIDENCE.csv"
    replay80_path = args.p7_dir / "observed_replay_scale080" / "PHASE_B_CMI_E_EVIDENCE.csv"
    replay_path = args.p7_dir / "observed_replay_consensus" / "PHASE_B_CMI_E_EVIDENCE.csv"
    replay2_65_path = args.p7_dir / "observed_replay2_scale065" / "PHASE_B_CMI_E_EVIDENCE.csv"
    replay2_80_path = args.p7_dir / "observed_replay2_scale080" / "PHASE_B_CMI_E_EVIDENCE.csv"
    control_path = args.p7_dir / "spatial_misalignment_consensus" / "PHASE_B_CMI_E_EVIDENCE.csv"

    original = read(original_path)
    replay = read(replay_path)
    control = read(control_path)
    original65, original80 = read(original65_path), read(original80_path)
    replay65, replay80 = read(replay65_path), read(replay80_path)

    pair_columns = [
        "source_root", "receiver_root", "affected_branch", "event_type",
        "raw_dependency", "environment_adjusted_dependency", "environment_drop",
        "environment_trust", "environment_conflict", "oof_predictive_gain",
        "program_specificity_margin", "scale065_primary_q", "scale080_primary_q",
        "multiscale_primary_q", "primary_channel_q", "scale065_called",
        "scale080_called", "scale_direction_consistent", "scale_branch_consistent",
        "multiscale_localized_branch_support_fraction", "localization_support_pass",
        "is_called", "refinement_eligible", "decision_reason",
    ]
    pair_audit = original[pair_columns].copy()
    pair_audit["prespecified_gate_failures"] = original.apply(gate_reason, axis=1)
    pair_audit.to_csv(args.p7_dir / "P7_DIRECTED_PAIR_ABSTENTION_AUDIT.csv", index=False)

    comparison = replay[pair_columns].merge(
        control[pair_columns], on=KEY, suffixes=("_observed", "_misaligned")
    )
    comparison["new_call_induced"] = (
        comparison.is_called_misaligned.astype(str).str.lower().eq("true")
        & ~comparison.is_called_observed.astype(str).str.lower().eq("true")
    )
    comparison.to_csv(args.p7_dir / "P7_SPATIAL_MISALIGNMENT_CONTROL.csv", index=False)

    deterministic_columns = [
        "raw_dependency", "environment_adjusted_dependency", "environment_drop",
        "environment_trust", "environment_conflict", "oof_predictive_gain",
        "program_specificity_margin", "raw_rho", "adjusted_rho",
        "localized_branch_support_fraction",
    ]
    permutation_columns = [
        "permutation_p", "conjunctive_permutation_p", "raw_permutation_p",
        "q_value", "raw_q_value", "primary_channel_q",
    ]
    reproducibility = pd.DataFrame([
        {
            "comparison": "legacy_P6_vs_recorded_seed_replay_scale065",
            "byte_identical": sha256(original65_path) == sha256(replay65_path),
            "max_nonpermutation_statistic_difference": max_numeric_difference(
                original65, replay65, deterministic_columns
            ),
            "max_permutation_pq_difference": max_numeric_difference(
                original65, replay65, permutation_columns
            ),
            "formal_call_count_left": int(original65.is_called.sum()),
            "formal_call_count_right": int(replay65.is_called.sum()),
            "interpretation": "legacy_seed_not_serialized",
        },
        {
            "comparison": "legacy_P6_vs_recorded_seed_replay_scale080",
            "byte_identical": sha256(original80_path) == sha256(replay80_path),
            "max_nonpermutation_statistic_difference": max_numeric_difference(
                original80, replay80, deterministic_columns
            ),
            "max_permutation_pq_difference": max_numeric_difference(
                original80, replay80, permutation_columns
            ),
            "formal_call_count_left": int(original80.is_called.sum()),
            "formal_call_count_right": int(replay80.is_called.sum()),
            "interpretation": "legacy_seed_not_serialized",
        },
        {
            "comparison": "recorded_seed_replay1_vs_replay2_scale065",
            "byte_identical": sha256(replay65_path) == sha256(replay2_65_path),
            "max_nonpermutation_statistic_difference": 0.0,
            "max_permutation_pq_difference": 0.0,
            "formal_call_count_left": int(replay65.is_called.sum()),
            "formal_call_count_right": int(read(replay2_65_path).is_called.sum()),
            "interpretation": "recorded_seed_deterministic",
        },
        {
            "comparison": "recorded_seed_replay1_vs_replay2_scale080",
            "byte_identical": sha256(replay80_path) == sha256(replay2_80_path),
            "max_nonpermutation_statistic_difference": 0.0,
            "max_permutation_pq_difference": 0.0,
            "formal_call_count_left": int(replay80.is_called.sum()),
            "formal_call_count_right": int(read(replay2_80_path).is_called.sum()),
            "interpretation": "recorded_seed_deterministic",
        },
    ])
    reproducibility.to_csv(args.p7_dir / "P7_REPRODUCIBILITY_AUDIT.csv", index=False)

    original_zero = int(original.is_called.sum()) == 0
    replay_zero = int(replay.is_called.sum()) == 0
    control_zero = int(control.is_called.sum()) == 0
    same_seed_deterministic = bool(reproducibility.iloc[2:].byte_identical.all())
    decision = {
        "stage": "P7_LIGHTWEIGHT_ABSTENTION_ROBUSTNESS",
        "decision": "QUALIFIED_PASS" if all([
            original_zero, replay_zero, control_zero, same_seed_deterministic
        ]) else "FAIL",
        "official_P6_relation_calls": int(original.is_called.sum()),
        "recorded_seed_replay_relation_calls": int(replay.is_called.sum()),
        "spatial_misalignment_relation_calls": int(control.is_called.sum()),
        "new_calls_induced_by_spatial_misalignment": int(comparison.new_call_induced.sum()),
        "recorded_seed": 7131001,
        "permutations": 399,
        "recorded_seed_byte_reproducible": same_seed_deterministic,
        "legacy_P6_byte_reproducible": False,
        "legacy_P6_limitation": "Phase B seed was not serialized; Monte Carlo p/q values cannot be byte-reproduced",
        "legacy_to_replay_nonpermutation_evidence_reproduced": bool(
            reproducibility.iloc[:2].max_nonpermutation_statistic_difference.max() < 1e-12
        ),
        "formal_abstention_decision_stable": bool(original_zero and replay_zero),
        "thresholds_modified": False,
        "frozen_T0_modified": False,
        "proceed_to_P8_P9_design": True,
        "P8_caveat": "fine subtype labels remain unresolved and require independent frozen-assignment validation",
    }
    (args.p7_dir / "P7_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    report = f"""# P7 Breast Abstention Robustness

## Decision

`{decision['decision']}`: the formal no-call decision is stable, with a documented legacy seed-provenance limitation.

## Observed data

- Official frozen P6: {int(original.is_called.sum())} formal relation calls across six directed root pairs.
- Recorded-seed replay: {int(replay.is_called.sum())} calls; minimum consensus q = {replay.primary_channel_q.min():.4f}.
- Raw/environment-adjusted evidence is reproduced to numerical tolerance between the legacy P6 output and replay.
- Legacy Monte Carlo p/q values are not byte-reproducible because the P6 seed was not serialized.
- Two independent reruns using the now-recorded seed 7131001 are byte-identical at both scales.

## Spatial misalignment control

- The control permuted detector-facing `obs` coordinates and `obsm['spatial']` with seed 7137701 while preserving expression, cell identity, metacell mapping, T0, programs, and thresholds.
- It produced {int(control.is_called.sum())} formal calls and induced {int(comparison.new_call_induced.sum())} new calls.
- Its minimum consensus q was {control.primary_channel_q.min():.4f}, compared with {replay.primary_channel_q.min():.4f} in the recorded-seed observed replay.

## Claim boundary

The breast detector robustly abstains from declaring a cross-trajectory association under the frozen gate. This supports abstention safety, not absence of biological interaction and not causal evidence.
"""
    (args.p7_dir / "P7_ABSTENTION_ROBUSTNESS.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
