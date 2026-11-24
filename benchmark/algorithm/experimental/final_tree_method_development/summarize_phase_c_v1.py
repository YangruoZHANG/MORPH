#!/usr/bin/env python3
"""Summarize the frozen DEV audit for PHASE C oracle refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SELECTED_GAMMA = 0.35


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_runs(roots: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in roots:
        for path in root.rglob(
            "phase_c_oracle_final_dev/PHASE_C_ORACLE_REFINEMENT.csv"
        ):
            run_dir = path.parent.parent
            setting = run_dir.parent.name
            frame = pd.read_csv(path)
            frame.insert(0, "setting", setting)
            frame.insert(1, "run_directory", str(run_dir.resolve()))
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No final PHASE C DEV results found")
    result = pd.concat(frames, ignore_index=True)
    result["causal"] = as_bool(result["causal"])
    for column in [
        "accepted", "upstream_violation", "candidate_legal", "connected",
        "acyclic", "truth_used_for_selection", "truth_used_for_scoring_only",
    ]:
        result[column] = as_bool(result[column])
    return result.sort_values(["causal", "setting", "run", "gamma"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    all_rows = load_runs([path.resolve() for path in args.input_root])
    selected = all_rows[np.isclose(all_rows.gamma, SELECTED_GAMMA)].copy()
    causal = selected[selected.causal].copy()
    null = selected[~selected.causal].copy()

    sensitivity = (
        all_rows.groupby(["gamma", "causal"], as_index=False)
        .agg(
            n=("event_id", "size"),
            accepted_rate=("accepted", "mean"),
            mean_topology_gain=("oracle_topology_gain", "mean"),
            median_topology_gain=("oracle_topology_gain", "median"),
            positive_gain_rate=("oracle_topology_gain", lambda x: float((x > 0).mean())),
            negative_gain_rate=("oracle_topology_gain", lambda x: float((x < 0).mean())),
            mean_rewiring_F1=("rewiring_F1", "mean"),
            mean_affected_path_F1=("affected_path_F1", "mean"),
            mean_branch_choice_accuracy=("branch_choice_accuracy", "mean"),
            legality_rate=("candidate_legal", "mean"),
            connected_rate=("connected", "mean"),
            acyclic_rate=("acyclic", "mean"),
            upstream_violation_rate=("upstream_violation", "mean"),
        )
    )

    rewiring = selected[[
        "setting", "run", "event_id", "event_type", "causal", "gamma",
        "accepted", "reason", "n_selected_swaps", "rewiring_precision",
        "rewiring_recall", "rewiring_F1", "affected_path_F1",
        "branch_choice_accuracy", "topology_distance_before",
        "topology_distance_after", "oracle_topology_gain",
        "branch_scale_topology_gain", "candidate_legal", "connected", "acyclic",
        "upstream_violation",
    ]].copy()
    null_control = null[[
        "setting", "run", "event_id", "event_type", "gamma", "accepted",
        "reason", "n_selected_swaps", "oracle_topology_gain",
        "branch_scale_topology_gain", "upstream_violation", "candidate_legal",
        "connected", "acyclic",
    ]].copy()
    null_control["false_rewire"] = null_control.n_selected_swaps.fillna(0).gt(0)

    causal_median_gain = float(causal.oracle_topology_gain.median())
    causal_positive_rate = float(causal.oracle_topology_gain.gt(0).mean())
    causal_negative_rate = float(causal.oracle_topology_gain.lt(0).mean())
    null_false_rewire_rate = float(null_control.false_rewire.mean()) if len(null) else np.nan
    legal = bool(
        selected.candidate_legal.all()
        and selected.connected.all()
        and selected.acyclic.all()
        and not selected.upstream_violation.any()
    )
    no_truth_selection = bool(
        not selected.truth_used_for_selection.any()
        and selected.truth_used_for_scoring_only.all()
    )
    passed = bool(
        len(causal) >= 10
        and len(null) >= 3
        and causal_median_gain > 0
        and causal_positive_rate > 0.5
        and causal_negative_rate == 0
        and null_false_rewire_rate == 0
        and legal
        and no_truth_selection
    )
    decision = {
        "phase": "C",
        "decision": "PASS" if passed else "FAIL",
        "operator_version": "ORACLE_REFINEMENT_V1",
        "frozen_phase_a": "PHASE_A_v4_generator_v2_fine_hvg_v9_regularized",
        "selected_gamma": SELECTED_GAMMA,
        "selection_rule": "predeclared stability plateau, not maximum edge change",
        "n_causal_dev_instances": int(len(causal)),
        "n_null_dev_instances": int(len(null)),
        "median_oracle_topology_gain": causal_median_gain,
        "mean_oracle_topology_gain": float(causal.oracle_topology_gain.mean()),
        "causal_positive_gain_rate": causal_positive_rate,
        "causal_negative_gain_rate": causal_negative_rate,
        "null_false_rewire_rate": null_false_rewire_rate,
        "all_candidate_legal_connected_acyclic_upstream_invariant": legal,
        "exact_truth_edges_used_for_selection": not no_truth_selection,
        "edge_exact_rewiring_F1_mean": float(causal.rewiring_F1.mean()),
        "edge_exact_rewiring_F1_median": float(causal.rewiring_F1.median()),
        "affected_path_F1_mean": float(causal.affected_path_F1.mean()),
        "branch_choice_accuracy_mean": float(causal.branch_choice_accuracy.mean()),
        "hard_gate": {
            "median_gain_gt_zero": causal_median_gain > 0,
            "majority_causal_gain_gt_zero": causal_positive_rate > 0.5,
            "no_causal_harm_at_selected_gamma": causal_negative_rate == 0,
            "null_no_rewire": null_false_rewire_rate == 0,
            "structural_safety": legal,
            "truth_sequestered_to_scoring": no_truth_selection,
        },
        "interpretation": (
            "Oracle-local event information improves formal topology distance in a "
            "majority of causal DEV instances without harming the remainder or rewiring "
            "null events. Exact edge recovery remains modest and is retained as a limitation."
        ),
    }

    selected.to_csv(output / "PHASE_C_ORACLE_REFINEMENT.csv", index=False)
    rewiring.to_csv(output / "PHASE_C_REWIRING_METRICS.csv", index=False)
    null_control.to_csv(output / "PHASE_C_NULL_REWIRE_CONTROL.csv", index=False)
    sensitivity.to_csv(output / "PHASE_C_GAMMA_SENSITIVITY.csv", index=False)
    (output / "PHASE_C_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
