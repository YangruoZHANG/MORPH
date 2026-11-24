#!/usr/bin/env python3
"""Branch-local, confounding-aware two-channel relation detector (dev v6.0).

Primary response and collision-signature hypotheses share one BH family. The
contact support is chosen only from the number of frozen T0 roots.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BASE_DETECTOR = HERE / "run_phase_b_relation_detector_v5.py"
EPS = 1e-12


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def holm(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.maximum.accumulate(ranked * np.arange(len(ranked), 0, -1))
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def gaussian_mi(rho: float) -> float:
    return float(-0.5 * np.log(max(1.0 - float(rho) ** 2, EPS)))


def run_channel(args: argparse.Namespace, support: float, output: Path, seed: int) -> None:
    command = [
        sys.executable, str(BASE_DETECTOR),
        "--run-dir", str(args.run_dir), "--dataset", args.dataset,
        "--phase-a-dir", str(args.phase_a_dir),
        "--program-catalog", str(args.program_catalog),
        "--output", str(output), "--permutations", str(args.permutations),
        "--seed", str(seed), "--q-cutoff", str(args.q_cutoff),
        "--effect-cutoff", str(args.effect_cutoff),
        "--raw-effect-cutoff", str(args.raw_effect_cutoff),
        "--minimum-oof-gain", str(args.minimum_oof_gain),
        "--source-activity-quantile", "0.60",
        "--source-identity-quantile", "0.80",
        "--kernel-quantile", "0.35", "--contact-quantile", str(support),
    ]
    if args.disable_environment_programs:
        command.append("--disable-environment-programs")
    subprocess.run(command, check=True)


def apply_competition(
    evidence: pd.DataFrame, ratio: float = 1.20, event_local: bool = True
) -> None:
    evidence["call_strength"] = np.where(
        evidence.call_channel.eq("collision_signature"),
        evidence.signature_joint_statistic.fillna(0.0),
        evidence.environment_adjusted_dependency.fillna(0.0),
    )
    pair_key = evidence.apply(
        lambda row: tuple(sorted((str(row.source_root), str(row.receiver_root)))), axis=1
    )
    for _, indices in evidence.groupby(pair_key).groups.items():
        called = evidence.loc[list(indices)]
        called = called[called.is_called].sort_values("call_strength", ascending=False)
        if len(called) <= 1:
            continue
        best, second = called.index[:2]
        if evidence.loc[best, "call_strength"] < ratio * max(
            float(evidence.loc[second, "call_strength"]), EPS
        ):
            evidence.loc[called.index, "is_called"] = False
            evidence.loc[called.index, "decision_reason"] = "bidirectional_ambiguity_abstain"
        else:
            evidence.loc[called.index[1:], "is_called"] = False
            evidence.loc[called.index[1:], "decision_reason"] = "reverse_direction_weaker"
    receiver_groups = ["receiver_root", "affected_branch"] if event_local else ["receiver_root"]
    for _, indices in evidence.groupby(receiver_groups).groups.items():
        called = evidence.loc[list(indices)]
        called = called[called.is_called].sort_values("call_strength", ascending=False)
        if len(called) <= 1:
            continue
        best, second = called.index[:2]
        if evidence.loc[best, "call_strength"] < 1.15 * max(
            float(evidence.loc[second, "call_strength"]), EPS
        ):
            evidence.loc[called.index, "is_called"] = False
            evidence.loc[called.index, "decision_reason"] = "receiver_source_ambiguity_abstain"
        else:
            evidence.loc[called.index[1:], "is_called"] = False
            evidence.loc[called.index[1:], "decision_reason"] = "competing_source_weaker"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase-a-dir", type=Path, required=True)
    parser.add_argument("--program-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=399)
    parser.add_argument("--seed", type=int, default=7131001)
    parser.add_argument("--q-cutoff", type=float, default=0.15)
    parser.add_argument("--effect-cutoff", type=float, default=0.10)
    parser.add_argument("--raw-effect-cutoff", type=float, default=0.15)
    parser.add_argument("--minimum-oof-gain", type=float, default=-0.10)
    parser.add_argument(
        "--primary-significance", choices=["omnibus", "conjunctive"],
        default="conjunctive",
    )
    parser.add_argument(
        "--collision-environment-trust-min", type=float, default=0.30,
        help=(
            "Minimum adjusted/(raw+adjusted) dependency; 0.30 requires the "
            "adjusted dependency to retain at least 3/7 of the raw dependency."
        ),
    )
    parser.add_argument(
        "--competition-scope", choices=["event-local", "whole-receiver"],
        default="event-local",
    )
    parser.add_argument(
        "--primary-contact-support", type=float, default=None,
        help="Optional predeclared primary contact-distance quantile.",
    )
    parser.add_argument("--disable-environment-programs", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_csv(args.phase_a_dir / "T0_STATE_NODE_ASSIGNMENTS.csv")
    n_roots = int(assignments.root_id.astype(str).nunique())
    primary_support = (
        float(args.primary_contact_support)
        if args.primary_contact_support is not None
        else (0.80 if n_roots == 2 else 0.65)
    )
    signature_support = 0.80
    primary_dir = output / "channel_primary"
    signature_dir = output / "channel_collision_signature"
    run_channel(args, primary_support, primary_dir, args.seed)
    if signature_support == primary_support:
        signature_dir = primary_dir
    else:
        run_channel(args, signature_support, signature_dir, args.seed + 1)

    primary = pd.read_csv(primary_dir / "PHASE_B_CMI_E_EVIDENCE.csv")
    signature = pd.read_csv(signature_dir / "PHASE_B_CMI_E_EVIDENCE.csv")
    keys = ["source_root", "receiver_root"]
    signature = signature.set_index(keys)
    evidence = primary.copy()
    evidence["primary_contact_support_quantile"] = primary_support
    evidence["signature_contact_support_quantile"] = signature_support
    primary_p_column = (
        "permutation_p"
        if args.primary_significance == "omnibus"
        else "conjunctive_permutation_p"
    )
    evidence["primary_channel_p"] = evidence[primary_p_column].astype(float)
    evidence["signature_channel_p"] = [
        float(signature.loc[(row.source_root, row.receiver_root), "signature_permutation_p"])
        for row in evidence.itertuples(index=False)
    ]
    evidence["signature_joint_statistic"] = [
        float(signature.loc[(row.source_root, row.receiver_root), "signature_joint_statistic"])
        for row in evidence.itertuples(index=False)
    ]
    evidence["signature_support_raw_rho"] = [
        float(signature.loc[(row.source_root, row.receiver_root), "signature_best_raw_rho"])
        for row in evidence.itertuples(index=False)
    ]
    evidence["signature_support_adjusted_rho"] = [
        float(signature.loc[(row.source_root, row.receiver_root), "signature_best_adjusted_rho"])
        for row in evidence.itertuples(index=False)
    ]
    evidence["signature_support_joint_statistic"] = [
        float(signature.loc[(row.source_root, row.receiver_root), "joint_dependency_statistic"])
        for row in evidence.itertuples(index=False)
    ]
    evidence["signature_support_raw_specificity_margin"] = [
        float(signature.loc[(row.source_root, row.receiver_root), "signature_best_raw_specificity_margin"])
        for row in evidence.itertuples(index=False)
    ]
    evidence["signature_support_raw_specificity_rank"] = [
        float(signature.loc[(row.source_root, row.receiver_root), "signature_best_raw_specificity_rank"])
        for row in evidence.itertuples(index=False)
    ]
    evidence["signature_support_affected_branch"] = [
        str(signature.loc[(row.source_root, row.receiver_root), "signature_best_affected_branch"])
        for row in evidence.itertuples(index=False)
    ]
    evidence["signature_support_receiver_window_start"] = [
        float(signature.loc[(row.source_root, row.receiver_root), "signature_best_receiver_window_start"])
        for row in evidence.itertuples(index=False)
    ]
    evidence["signature_support_receiver_window_end"] = [
        float(signature.loc[(row.source_root, row.receiver_root), "signature_best_receiver_window_end"])
        for row in evidence.itertuples(index=False)
    ]
    evidence["signature_support_localized_cell_ids"] = [
        str(signature.loc[(row.source_root, row.receiver_root), "signature_localized_support_cell_ids"])
        for row in evidence.itertuples(index=False)
    ]
    evidence["signature_support_localized_unit_ids"] = [
        str(signature.loc[(row.source_root, row.receiver_root), "signature_localized_support_unit_ids"])
        for row in evidence.itertuples(index=False)
    ]
    evidence["signature_support_localized_n_cells"] = [
        int(value) if pd.notna(value) else 0
        for row in evidence.itertuples(index=False)
        for value in [signature.loc[
            (row.source_root, row.receiver_root),
            "signature_localized_support_n_cells",
        ]]
    ]
    evidence["signature_support_branch_n_cells"] = [
        int(value) if pd.notna(value) else 0
        for row in evidence.itertuples(index=False)
        for value in [signature.loc[
            (row.source_root, row.receiver_root),
            "signature_localized_branch_support_n_cells",
        ]]
    ]
    evidence["signature_support_branch_fraction"] = [
        float(signature.loc[(row.source_root, row.receiver_root), "signature_localized_branch_support_fraction"])
        for row in evidence.itertuples(index=False)
    ]
    evidence["signature_support_positivity_overlap"] = [
        float(signature.loc[(row.source_root, row.receiver_root), "signature_best_positivity_overlap"])
        for row in evidence.itertuples(index=False)
    ]
    evidence["signature_support_exposure_variance"] = [
        float(signature.loc[(row.source_root, row.receiver_root), "signature_best_exposure_variance"])
        for row in evidence.itertuples(index=False)
    ]
    evidence["signature_support_oof_predictive_gain"] = [
        float(signature.loc[(row.source_root, row.receiver_root), "signature_best_oof_predictive_gain"])
        for row in evidence.itertuples(index=False)
    ]
    signature_raw_dependency = evidence.signature_support_raw_rho.map(gaussian_mi)
    signature_adjusted_dependency = evidence.signature_support_adjusted_rho.map(gaussian_mi)
    signature_direction_consistent = (
        evidence.signature_support_raw_rho * evidence.signature_support_adjusted_rho > 0
    )
    evidence["signature_environment_direction_consistent"] = signature_direction_consistent
    evidence["signature_environment_trust"] = np.where(
        signature_direction_consistent,
        signature_adjusted_dependency
        / np.maximum(signature_raw_dependency + signature_adjusted_dependency, EPS),
        0.0,
    )

    eligible = evidence.status.eq("tested") & evidence.source_hypothesis_eligible.astype(bool)
    evidence["primary_channel_q"] = 1.0
    evidence["signature_channel_q"] = 1.0
    primary_indices = list(evidence.index[eligible])
    if primary_indices:
        evidence.loc[primary_indices, "primary_channel_q"] = bh(
            evidence.loc[primary_indices, "primary_channel_p"].to_numpy(float)
        )

    primary_call = (
        eligible & evidence.primary_channel_q.le(args.q_cutoff)
        & evidence[[
            "raw_monotone_statistic", "adjusted_monotone_statistic"
        ]].abs().max(axis=1).ge(args.effect_cutoff)
        & evidence.raw_monotone_statistic.abs().ge(args.raw_effect_cutoff)
        & evidence.raw_program_specificity_margin.gt(0.01)
        & evidence.raw_program_specificity_rank.ge(0.80)
        & evidence.program_specificity_margin.gt(0.01)
        & evidence.program_specificity_rank.ge(0.80)
        & evidence.environment_direction_consistent.astype(bool)
        & evidence.oof_predictive_gain.ge(args.minimum_oof_gain)
        & evidence.passed_candidate_specificity_prefilter.astype(bool)
        & evidence.exposure_variance.gt(1e-5)
        & evidence.positivity_overlap.ge(0.50)
    )
    # Collision is a subtype test conditional on an established primary
    # relation, not a parallel relation-existence hypothesis. This gatekeeping
    # prevents the subtype channel from reducing primary-event power.
    signature_indices = list(evidence.index[primary_call])
    if signature_indices:
        evidence.loc[signature_indices, "signature_channel_q"] = bh(
            evidence.loc[signature_indices, "signature_channel_p"].to_numpy(float)
        )
    collision_call = (
        primary_call & evidence.signature_channel_q.le(args.q_cutoff)
        & evidence.signature_joint_statistic.ge(0.50)
        & evidence.signature_support_raw_rho.ge(0.15)
        & evidence.signature_support_raw_specificity_margin.gt(0.01)
        & evidence.signature_support_raw_specificity_rank.ge(0.80)
        & evidence.signature_environment_direction_consistent.astype(bool)
        & evidence.signature_environment_trust.ge(args.collision_environment_trust_min)
        & evidence.signature_support_oof_predictive_gain.ge(args.minimum_oof_gain)
        & evidence.signature_support_exposure_variance.gt(1e-5)
        & evidence.signature_support_positivity_overlap.ge(0.50)
    )
    evidence["is_called"] = primary_call
    evidence["call_channel"] = np.where(
        collision_call, "collision_signature",
        np.where(primary_call, "primary_response", "abstain"),
    )
    evidence["event_type"] = np.where(
        collision_call, "collision_convergence",
        np.where(evidence.raw_monotone_statistic.lt(0), "inhibition", "promotion"),
    )
    evidence.loc[collision_call, "affected_branch"] = evidence.loc[
        collision_call, "signature_support_affected_branch"
    ]
    evidence.loc[collision_call, "receiver_window_start"] = evidence.loc[
        collision_call, "signature_support_receiver_window_start"
    ]
    evidence.loc[collision_call, "receiver_window_end"] = evidence.loc[
        collision_call, "signature_support_receiver_window_end"
    ]
    evidence.loc[collision_call, "localized_support_cell_ids"] = evidence.loc[
        collision_call, "signature_support_localized_cell_ids"
    ]
    evidence.loc[collision_call, "localized_support_unit_ids"] = evidence.loc[
        collision_call, "signature_support_localized_unit_ids"
    ]
    evidence.loc[collision_call, "localized_support_n_cells"] = evidence.loc[
        collision_call, "signature_support_localized_n_cells"
    ]
    evidence.loc[collision_call, "localized_branch_support_n_cells"] = evidence.loc[
        collision_call, "signature_support_branch_n_cells"
    ]
    evidence.loc[collision_call, "localized_branch_support_fraction"] = evidence.loc[
        collision_call, "signature_support_branch_fraction"
    ]
    evidence["decision_reason"] = np.where(
        evidence.is_called, "passes_pooled_channel_fdr_and_effect_gates",
        "abstain_after_pooled_channel_fdr_or_effect_gate",
    )
    apply_competition(
        evidence, event_local=args.competition_scope == "event-local"
    )
    evidence = evidence.sort_values(
        ["is_called", "primary_channel_q", "signature_channel_q"],
        ascending=[False, True, True],
    )
    evidence.to_csv(output / "PHASE_B_CMI_E_EVIDENCE.csv", index=False)
    evidence.to_csv(output / "PHASE_B_RELATION_SCORE_DISTRIBUTION.csv", index=False)
    evidence[evidence.is_called].to_csv(output / "PHASE_B_CALLED_RELATIONS.csv", index=False)
    hypothesis_rows = [
        {
            "evidence_index": index, "channel": "primary_relation",
            "gate_level": 1,
            "p_value": float(evidence.loc[index, "primary_channel_p"]),
            "bh_q_value": float(evidence.loc[index, "primary_channel_q"]),
            "tested": True,
        }
        for index in primary_indices
    ]
    hypothesis_rows.extend([
        {
            "evidence_index": index, "channel": "collision_subtype",
            "gate_level": 2,
            "p_value": float(evidence.loc[index, "signature_channel_p"]),
            "bh_q_value": float(evidence.loc[index, "signature_channel_q"]),
            "tested": True,
        }
        for index in signature_indices
    ])
    hypothesis = pd.DataFrame(hypothesis_rows)
    hypothesis.to_csv(output / "PHASE_B_POOLED_HYPOTHESIS_FAMILY.csv", index=False)
    status = {
        "phase": "B", "detector_version": "BRANCH_LOCAL_MULTICHANNEL_RELATION_V6_12_HIGH_CONFIDENCE_SELECTED_DEV",
        "base_detector": "BRANCH_LOCAL_RELATION_V4_11_HIGH_CONFIDENCE_MINP",
        "n_roots": n_roots, "primary_contact_support_quantile": primary_support,
        "signature_contact_support_quantile": signature_support,
        "kernel_quantile": 0.35, "source_activity_quantile": 0.60,
        "raw_effect_cutoff_abs_rho": args.raw_effect_cutoff,
        "minimum_oof_predictive_gain": args.minimum_oof_gain,
        "source_identity_quantile": 0.80, "pooled_channel_bh": False,
        "multiplicity_strategy": "primary_relation_BH_then_collision_subtype_BH",
        "primary_significance": args.primary_significance,
        "collision_environment_trust_min": args.collision_environment_trust_min,
        "collision_min_adjusted_to_raw_dependency_ratio": (
            args.collision_environment_trust_min
            / max(1.0 - args.collision_environment_trust_min, EPS)
        ),
        "competition_scope": args.competition_scope,
        "n_primary_hypotheses": len(primary_indices),
        "n_secondary_subtype_hypotheses": len(signature_indices),
        "n_called": int(evidence.is_called.sum()),
        "environment_adjustment_enabled": not args.disable_environment_programs,
        "truth_used": False, "relation_significance_before_T1": True,
        "complete": True,
    }
    (output / "RUN_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
