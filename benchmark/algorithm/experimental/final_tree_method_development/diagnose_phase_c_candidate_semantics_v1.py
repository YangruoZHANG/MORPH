#!/usr/bin/env python3
"""Audit observable features of Phase C swap candidates against held-out truth.

Truth is used only to label candidate swaps after generation.  The resulting
table is a development diagnostic and is never consumed by the operator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def pair(left: object, right: object) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def dominant_lookup(frame: pd.DataFrame, stage: str, value: str) -> dict[tuple[str, str], str]:
    local = frame[frame.stage.astype(str).eq(stage)].copy()
    probability = "state_probability" if value == "state_id" else "branch_probability"
    local = local.sort_values(
        ["metacell_id", "root_id", probability, value],
        ascending=[True, True, False, True],
    ).drop_duplicates(["metacell_id", "root_id"])
    return {
        (str(row.metacell_id), str(row.root_id)): str(getattr(row, value))
        for row in local.itertuples(index=False)
    }


def probability_lookup(frame: pd.DataFrame, stage: str) -> dict[tuple[str, str, str], float]:
    local = frame[frame.stage.astype(str).eq(stage)]
    return {
        (str(row.metacell_id), str(row.root_id), str(row.branch_id)): float(row.branch_probability)
        for row in local.itertuples(index=False)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--phase-c-subdir", default="phase_c_oracle_v2_fullgrid_dev")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[pd.DataFrame] = []
    for trial_file in args.grid_root.rglob("PHASE_C_LEGAL_SWAP_TRIALS.csv"):
        if args.phase_c_subdir not in str(trial_file):
            continue
        run = trial_file.parent.parent
        trials = pd.read_csv(trial_file)
        if trials.empty:
            continue
        events = pd.read_csv(run / "event" / "truth" / "TRUE_EVENTS.csv")
        events = events[[
            "event_id", "event_type", "receiver_root", "affected_branch",
            "decision_from", "decision_to",
        ]]
        trials = trials.merge(events, on="event_id", how="left")
        state = pd.read_csv(run / "formal_metacell" / "METACELL_STATE_COMPOSITION.csv")
        branch = pd.read_csv(run / "formal_metacell" / "METACELL_BRANCH_COMPOSITION.csv")
        state_t0 = dominant_lookup(state, "T0", "state_id")
        state_t1 = dominant_lookup(state, "T1", "state_id")
        branch_t0 = dominant_lookup(branch, "T0", "branch_id")
        branch_t1 = dominant_lookup(branch, "T1", "branch_id")
        prob_t0 = probability_lookup(branch, "T0")
        prob_t1 = probability_lookup(branch, "T1")
        added = pd.read_csv(run / "event" / "truth" / "TRUE_ADDED_EDGES.csv")
        added_lookup = {
            str(row.event_id): pair(row.source, row.target)
            for row in added.itertuples(index=False)
        }
        removed = pd.read_csv(run / "event" / "truth" / "TRUE_REMOVED_EDGES.csv")
        removed_lookup = {
            str(row.event_id): pair(row.source, row.target)
            for row in removed.itertuples(index=False)
        }
        for side in ("child", "old_parent", "new_parent"):
            names = trials[f"{side}_name"].astype(str)
            roots = trials.receiver_root.astype(str)
            trials[f"{side}_state_T0"] = [state_t0.get((n, r), "") for n, r in zip(names, roots)]
            trials[f"{side}_state_T1"] = [state_t1.get((n, r), "") for n, r in zip(names, roots)]
            trials[f"{side}_branch_T0"] = [branch_t0.get((n, r), "") for n, r in zip(names, roots)]
            trials[f"{side}_branch_T1"] = [branch_t1.get((n, r), "") for n, r in zip(names, roots)]
        trials["new_parent_decision_to_probability_T1"] = [
            prob_t1.get((str(n), str(r), str(b)), 0.0)
            for n, r, b in zip(
                trials.new_parent_name, trials.receiver_root, trials.decision_to
            )
        ]
        trials["child_affected_probability_T0"] = [
            prob_t0.get((str(n), str(r), str(b)), 0.0)
            for n, r, b in zip(
                trials.child_name, trials.receiver_root, trials.affected_branch
            )
        ]
        trials["child_affected_probability_T1"] = [
            prob_t1.get((str(n), str(r), str(b)), 0.0)
            for n, r, b in zip(
                trials.child_name, trials.receiver_root, trials.affected_branch
            )
        ]
        trials["truth_added_match_T1"] = [
            pair(a, b) == added_lookup.get(str(event_id), ("", ""))
            for a, b, event_id in zip(
                trials.new_parent_state_T1, trials.child_state_T1, trials.event_id
            )
        ]
        trials["truth_removed_match_T1"] = [
            pair(a, b) == removed_lookup.get(str(event_id), ("", ""))
            for a, b, event_id in zip(
                trials.old_parent_state_T1, trials.child_state_T1, trials.event_id
            )
        ]
        trials.insert(0, "case", str(run.relative_to(args.grid_root)))
        rows.append(trials)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    if result.empty:
        print("No candidates found")
        return
    numeric = [
        "alignment", "cost_advantage", "base_new_cost", "new_parent_tau",
        "child_affected_probability_T0", "child_affected_probability_T1",
        "new_parent_decision_to_probability_T1",
    ]
    summary = result.groupby(["event_type", "truth_added_match_T1"])[numeric].median()
    print(summary.to_string())
    print("\nCandidate counts")
    print(result.groupby(["event_type", "truth_added_match_T1"]).size().to_string())


if __name__ == "__main__":
    main()
