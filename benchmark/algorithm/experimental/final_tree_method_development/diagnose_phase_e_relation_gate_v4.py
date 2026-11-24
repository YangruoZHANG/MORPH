#!/usr/bin/env python3
"""Mechanistic gate sensitivity after the failed Phase E v3 lock.

The failed locked grid is explicitly diagnostic data here. No configuration
selected by this script may be evaluated again on the same grid as locked.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_phase_e_locked_v1 import as_bool, greedy_matches, prf


EPS = 1e-12
CHANNEL_WEIGHTS = {"primary": 1.5, "signature": 0.5}


def replicate_from(run: Path) -> int:
    match = re.search(r"replicate_(\d+)", run.name)
    if not match:
        raise ValueError(run)
    return int(match.group(1))


def bh(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
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
    adjusted = np.maximum.accumulate(
        ranked * np.arange(len(ranked), 0, -1)
    )
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def tokens(value: object) -> set[str]:
    return {
        token for token in str(value).split(";")
        if token and token.lower() != "nan"
    }


def interval_iou(left: pd.Series, right: pd.Series) -> float:
    a0, a1 = float(left.receiver_window_start), float(left.receiver_window_end)
    b0, b1 = float(right.receiver_window_start), float(right.receiver_window_end)
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return overlap / max(union, EPS)


def support_jaccard(left: pd.Series, right: pd.Series) -> float:
    a, b = tokens(left.localized_support_cell_ids), tokens(right.localized_support_cell_ids)
    return len(a & b) / max(len(a | b), 1)


def compete_indices(evidence: pd.DataFrame, indices: list[int], ratio: float) -> None:
    called = evidence.loc[indices]
    called = called[called.is_called].sort_values("call_strength", ascending=False)
    if len(called) <= 1:
        return
    best, second = called.index[:2]
    if float(evidence.loc[best, "call_strength"]) < ratio * max(
        float(evidence.loc[second, "call_strength"]), EPS
    ):
        evidence.loc[called.index, "is_called"] = False
    else:
        evidence.loc[called.index[1:], "is_called"] = False


def connected_conflict_groups(evidence: pd.DataFrame, indices: list[int]) -> list[list[int]]:
    adjacency = {index: set() for index in indices}
    for position, left_index in enumerate(indices):
        left = evidence.loc[left_index]
        for right_index in indices[position + 1:]:
            right = evidence.loc[right_index]
            same_branch = str(left.affected_branch) == str(right.affected_branch)
            same_event_locus = (
                interval_iou(left, right) >= 0.50
                or support_jaccard(left, right) >= 0.50
            )
            if same_branch and same_event_locus:
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
    groups: list[list[int]] = []
    unseen = set(indices)
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        component = [seed]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node] & unseen:
                unseen.remove(neighbor)
                component.append(neighbor)
                stack.append(neighbor)
        groups.append(component)
    return groups


def apply_competition(evidence: pd.DataFrame, scope: str) -> None:
    pair_keys = evidence.apply(
        lambda row: tuple(sorted((str(row.source_root), str(row.receiver_root)))), axis=1
    )
    for indices in evidence.groupby(pair_keys).groups.values():
        compete_indices(evidence, list(indices), 1.20)
    if scope == "direction_only":
        return
    for indices in evidence.groupby("receiver_root").groups.values():
        called_indices = [index for index in indices if bool(evidence.loc[index, "is_called"])]
        groups = (
            connected_conflict_groups(evidence, called_indices)
            if scope == "event_local" else [called_indices]
        )
        for group in groups:
            compete_indices(evidence, group, 1.15)


def gate(
    source: pd.DataFrame,
    primary_p: str,
    correction: str,
    collision_trust: float,
    competition_scope: str,
    source_eligibility: str,
) -> pd.DataFrame:
    evidence = source.copy().reset_index(drop=True)
    if source_eligibility == "max_gap":
        source_gate = evidence.source_hypothesis_eligible.map(as_bool)
    elif source_eligibility == "enrichment_0.25":
        source_gate = evidence.source_program_enrichment.ge(0.25)
    else:
        threshold = float(source_eligibility.split("_")[-1])
        source_gate = evidence.source_program_relative_strength.ge(threshold)
    eligible = evidence.status.eq("tested") & source_gate
    p_primary_column = (
        "permutation_p" if primary_p == "omnibus" else "conjunctive_permutation_p"
    )
    hypothesis: list[tuple[int, str, float]] = []
    for index in evidence.index[eligible]:
        hypothesis.extend([
            (index, "primary", float(evidence.loc[index, p_primary_column])),
            (index, "signature", float(evidence.loc[index, "signature_channel_p"])),
        ])
    evidence["primary_candidate_q"] = 1.0
    evidence["signature_candidate_q"] = 1.0
    if hypothesis:
        weighted = np.asarray([
            min(p / CHANNEL_WEIGHTS[channel], 1.0)
            for _, channel, p in hypothesis
        ])
        adjusted_bh = bh(weighted)
        adjusted_holm = holm(weighted)
        adjusted = adjusted_bh if correction == "BH" else adjusted_holm
        for position, ((index, channel, _), value) in enumerate(zip(hypothesis, adjusted)):
            if correction == "MIXED" and channel == "signature":
                value = adjusted_bh[position]
            evidence.loc[index, f"{channel}_candidate_q"] = float(value)
    primary = (
        eligible & evidence.primary_candidate_q.le(0.10)
        & evidence[["raw_rho", "adjusted_rho"]].abs().min(axis=1).ge(0.12)
        & evidence.raw_rho.abs().ge(0.15)
        & evidence.raw_program_specificity_margin.gt(0.01)
        & evidence.raw_program_specificity_rank.ge(0.80)
        & evidence.program_specificity_margin.gt(0.01)
        & evidence.program_specificity_rank.ge(0.80)
        & evidence.environment_direction_consistent.map(as_bool)
        & evidence.oof_predictive_gain.ge(-0.10)
        & evidence.passed_candidate_specificity_prefilter.map(as_bool)
        & evidence.exposure_variance.gt(1e-5)
        & evidence.positivity_overlap.ge(0.50)
    )
    signature_direction = (
        evidence.signature_environment_direction_consistent.map(as_bool)
        if "signature_environment_direction_consistent" in evidence
        else evidence.environment_direction_consistent.map(as_bool)
    )
    signature_trust = (
        evidence.signature_environment_trust
        if "signature_environment_trust" in evidence else evidence.environment_trust
    )
    signature_oof = (
        evidence.signature_support_oof_predictive_gain
        if "signature_support_oof_predictive_gain" in evidence
        else evidence.oof_predictive_gain
    )
    signature_variance = (
        evidence.signature_support_exposure_variance
        if "signature_support_exposure_variance" in evidence
        else evidence.exposure_variance
    )
    signature_positivity = (
        evidence.signature_support_positivity_overlap
        if "signature_support_positivity_overlap" in evidence
        else evidence.positivity_overlap
    )
    collision = (
        eligible & evidence.signature_candidate_q.le(0.10)
        & evidence.signature_joint_statistic.ge(0.50)
        & evidence.signature_support_raw_rho.ge(0.15)
        & evidence.signature_support_raw_specificity_margin.gt(0.01)
        & evidence.signature_support_raw_specificity_rank.ge(0.80)
        & signature_direction
        & signature_trust.ge(collision_trust)
        & signature_oof.ge(-0.10)
        & signature_variance.gt(1e-5)
        & signature_positivity.ge(0.50)
    )
    evidence["is_called"] = primary | collision
    evidence["call_channel"] = np.where(
        collision, "collision_signature", np.where(primary, "primary_response", "abstain")
    )
    evidence["event_type"] = np.where(
        collision,
        "collision_convergence",
        np.where(evidence.raw_rho.lt(0), "inhibition", "promotion"),
    )
    evidence.loc[collision, "affected_branch"] = evidence.loc[
        collision, "signature_support_affected_branch"
    ]
    evidence["call_strength"] = np.where(
        collision,
        evidence.signature_joint_statistic.fillna(0.0),
        evidence.environment_adjusted_dependency.fillna(0.0),
    )
    apply_competition(evidence, competition_scope)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relation-subdir", default="phase_b_v3_8_frozen")
    parser.add_argument("--focus-competition", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    statuses = sorted(args.diagnostic_root.rglob("LOCKED_CASE_STATUS.json"))
    run_payloads = []
    for status_file in statuses:
        run = status_file.parent
        scenario = json.loads((run / "SCENARIO_CONFIG.json").read_text(encoding="utf-8"))
        truth = pd.read_csv(run / "event" / "truth" / "TRUE_EVENTS.csv")
        evidence = pd.read_csv(
            run / args.relation_subdir / "PHASE_B_CMI_E_EVIDENCE.csv"
        )
        run_payloads.append((run, scenario, truth, evidence))

    rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    primary_options = ["omnibus"] if args.focus_competition else ["omnibus", "conjunctive"]
    correction_options = ["HOLM", "MIXED"] if args.focus_competition else ["BH", "HOLM"]
    trust_options = [0.40, 0.50] if args.focus_competition else [0.0, 0.30, 0.40, 0.50]
    scope_options = (
        ["whole_receiver", "event_local", "direction_only"]
        if args.focus_competition else ["whole_receiver", "event_local"]
    )
    eligibility_options = (
        ["max_gap", "relative_0.60", "relative_0.80", "enrichment_0.25"]
        if args.focus_competition else ["max_gap"]
    )
    for primary_p in primary_options:
        for correction in correction_options:
            for collision_trust in trust_options:
                for competition_scope in scope_options:
                    for source_eligibility in eligibility_options:
                        variant = (
                            f"{primary_p}_{correction}_trust{collision_trust:.2f}_"
                            f"{competition_scope}_{source_eligibility}"
                        )
                        aggregate = {"tp": 0, "fp": 0, "fn": 0}
                        per_variant: list[dict[str, object]] = []
                        for run, scenario, truth, original in run_payloads:
                            evidence = gate(
                                original, primary_p, correction,
                                collision_trust, competition_scope, source_eligibility,
                            )
                            calls = evidence[evidence.is_called].reset_index(drop=True)
                            causal = truth[truth.causal.map(as_bool)].reset_index(drop=True)
                            tp = len(greedy_matches(calls, causal))
                            fp, fn = len(calls) - tp, len(causal) - tp
                            aggregate["tp"] += tp
                            aggregate["fp"] += fp
                            aggregate["fn"] += fn
                            per_variant.append({
                                "variant": variant,
                                "setting": str(scenario["setting_id"]),
                                "replicate": replicate_from(run),
                                "family": str(scenario["family"]),
                                "scenario_event_type": str(scenario["event_type"]),
                                "effect_level": str(scenario["effect_level"]),
                                "causal": bool(causal.shape[0]),
                                "n_true": int(len(causal)),
                                "n_calls": int(len(calls)),
                                "tp": int(tp), "fp": int(fp), "fn": int(fn),
                            })
                        frame = pd.DataFrame(per_variant)
                        p, r, f1 = prf(
                            aggregate["tp"], aggregate["fp"], aggregate["fn"]
                        )
                        null = frame[~frame.causal]
                        complex_rows = frame[frame.family.eq("complex")]
                        rows.append({
                            "variant": variant,
                            "primary_p": primary_p,
                            "multiplicity_correction": correction,
                            "collision_environment_trust_min": collision_trust,
                            "competition_scope": competition_scope,
                            "source_eligibility": source_eligibility,
                            "event_precision": p,
                            "event_recall": r,
                            "event_F1": f1,
                            "null_false_call_rate": float(null.n_calls.gt(0).mean()),
                            "complex_event_recall": float(
                                complex_rows.tp.sum() / max(complex_rows.n_true.sum(), 1)
                            ),
                            "n_tp": aggregate["tp"],
                            "n_fp": aggregate["fp"],
                            "n_fn": aggregate["fn"],
                        })
                        run_rows.extend(per_variant)
    summary = pd.DataFrame(rows)
    run_frame = pd.DataFrame(run_rows)
    summary["passes_relation_gate"] = (
        summary.event_precision.ge(0.70)
        & summary.event_recall.ge(0.50)
        & summary.event_F1.ge(0.58)
        & summary.null_false_call_rate.le(0.10)
    )
    summary = summary.sort_values(
        ["passes_relation_gate", "event_F1", "null_false_call_rate"],
        ascending=[False, False, True],
    )
    summary.to_csv(args.output / "PHASE_E_V4_GATE_SENSITIVITY.csv", index=False)
    run_frame.to_csv(args.output / "PHASE_E_V4_GATE_RUN_LEVEL.csv", index=False)
    selected = summary.iloc[0].to_dict()
    decision = {
        "role": "POST_LOCK_FAILURE_DIAGNOSTIC_ONLY",
        "same_grid_may_not_be_reused_as_locked": True,
        "selected_development_variant": selected,
        "next_validation": "untouched replicate seeds 31-40",
    }
    (args.output / "PHASE_E_V4_DEVELOPMENT_SELECTION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
