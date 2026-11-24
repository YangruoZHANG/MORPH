#!/usr/bin/env python3
"""Create truth-derived predictions used only to unit-test the benchmark scorer."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def write_prediction(truth_dir: Path, output_dir: Path, mode: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = pd.read_csv(truth_dir / "TRUE_ROOTS.csv")
    membership = pd.read_csv(truth_dir / "TRUE_CELL_ROOT_MEMBERSHIP.csv")
    branches = pd.read_csv(truth_dir / "TRUE_CELL_BRANCH.csv")
    pseudotime = pd.read_csv(truth_dir / "TRUE_PSEUDOTIME.csv")
    events = pd.read_csv(truth_dir / "TRUE_EVENTS.csv")
    t0 = pd.read_csv(truth_dir / "TRUE_T0_EDGES.csv")
    t1 = pd.read_csv(truth_dir / "TRUE_T1_EDGES.csv")

    roots[["root_id"]].drop_duplicates().to_csv(output_dir / "PRED_ROOTS.csv", index=False)
    membership.rename(columns={"cell_id": "unit_id"})[
        ["unit_id", "root_id", "membership_probability"]
    ].to_csv(output_dir / "PRED_CELL_OR_METACELL_ASSIGNMENT.csv", index=False)
    branches = branches[branches["stage"] == "T1"].rename(columns={"cell_id": "unit_id"})
    branches[["unit_id", "root_id", "branch_id"]].to_csv(
        output_dir / "PRED_BRANCH.csv", index=False
    )
    pseudotime.rename(columns={"cell_id": "unit_id"})[
        ["unit_id", "root_id", "pseudotime"]
    ].to_csv(output_dir / "PRED_PSEUDOTIME.csv", index=False)
    t0[["root_id", "source", "target"]].to_csv(output_dir / "PRED_T0_EDGES.csv", index=False)
    if mode == "oracle":
        pred_t1 = t1
    elif mode == "false_rewire":
        pred_t1 = t0.copy()
        remove = (
            pred_t1["root_id"].astype(str).eq("R0")
            & pred_t1["source"].astype(str).eq("R0_T1")
            & pred_t1["target"].astype(str).eq("R0_B0_S1")
        )
        if int(remove.sum()) != 1:
            raise ValueError("false_rewire sanity fixture cannot find its editable R0 edge")
        pred_t1 = pred_t1.loc[~remove].copy()
        pred_t1 = pd.concat([
            pred_t1,
            pd.DataFrame([{"root_id": "R0", "source": "R0_ROOT", "target": "R0_B0_S1", "edge_role": "false_rewire_sanity"}]),
        ], ignore_index=True)
    else:
        pred_t1 = t0
    pred_t1[["root_id", "source", "target"]].to_csv(output_dir / "PRED_T1_EDGES.csv", index=False)

    pred_events = events[
        ["event_id", "event_type", "source_root", "receiver_root", "affected_branch"]
    ].copy()
    pred_events["score"] = 1.0
    causal = events["causal"].astype(str).str.lower().isin(["true", "1", "yes"])
    pred_events["is_called"] = causal
    if mode == "false_rewire" and len(pred_events):
        pred_events.loc[pred_events.index[0], "is_called"] = True
    pred_events.to_csv(output_dir / "PRED_EVENTS.csv", index=False)

    if mode == "oracle":
        for truth_name, pred_name in (
            ("TRUE_REMOVED_EDGES.csv", "PRED_REMOVED_EDGES.csv"),
            ("TRUE_ADDED_EDGES.csv", "PRED_ADDED_EDGES.csv"),
        ):
            pd.read_csv(truth_dir / truth_name).to_csv(output_dir / pred_name, index=False)
    elif mode == "no_rewire":
        columns = ["event_id", "root_id", "source", "target"]
        pd.DataFrame(columns=columns).to_csv(output_dir / "PRED_REMOVED_EDGES.csv", index=False)
        pd.DataFrame(columns=columns).to_csv(output_dir / "PRED_ADDED_EDGES.csv", index=False)
    else:
        pd.DataFrame([{"event_id": "FALSE_REWIRE_SANITY", "root_id": "R0", "source": "R0_T1", "target": "R0_B0_S1"}]).to_csv(
            output_dir / "PRED_REMOVED_EDGES.csv", index=False
        )
        pd.DataFrame([{"event_id": "FALSE_REWIRE_SANITY", "root_id": "R0", "source": "R0_ROOT", "target": "R0_B0_S1"}]).to_csv(
            output_dir / "PRED_ADDED_EDGES.csv", index=False
        )

    pd.DataFrame(
        [{
            "cycle_violation": False,
            "connectivity_violation": False,
            "upstream_violation": False,
            "illegal_candidate_edge": False,
        }]
    ).to_csv(output_dir / "PRED_SAFETY_AUDIT.csv", index=False)
    (output_dir / "SANITY_ONLY_DO_NOT_REPORT_AS_METHOD.txt").write_text(
        f"{mode} is a truth-derived scorer sanity check, not a fitted method.\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["oracle", "no_rewire", "false_rewire"], required=True)
    args = parser.parse_args()
    write_prediction(Path(args.truth_dir), Path(args.output_dir), args.mode)


if __name__ == "__main__":
    main()
