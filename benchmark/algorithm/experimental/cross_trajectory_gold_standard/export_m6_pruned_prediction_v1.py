#!/usr/bin/env python3
"""Apply conservative event-aware terminal-leaf pruning to an M5 prediction."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import networkx as nx
import pandas as pd


FILES = [
    "PRED_ROOTS.csv", "PRED_CELL_OR_METACELL_ASSIGNMENT.csv", "PRED_BRANCH.csv",
    "PRED_PSEUDOTIME.csv", "PRED_T0_EDGES.csv", "PRED_T1_EDGES.csv", "PRED_EVENTS.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m5-prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--membership-threshold", type=float, default=0.10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copyfile(args.m5_prediction_dir / name, args.output_dir / name)

    edges = pd.read_csv(args.output_dir / "PRED_T1_EDGES.csv")
    assignment = pd.read_csv(args.output_dir / "PRED_CELL_OR_METACELL_ASSIGNMENT.csv")
    branches = pd.read_csv(args.output_dir / "PRED_BRANCH.csv")
    events = pd.read_csv(args.output_dir / "PRED_EVENTS.csv")
    called = events[events["is_called"].astype(str).str.lower().isin({"true", "1", "yes"})]
    protected_branches = set(zip(called["receiver_root"].astype(str), called["affected_branch"].astype(str)))
    branch_lookup = {
        (str(row.root_id), str(row.unit_id)): str(row.branch_id)
        for row in branches.itertuples(index=False)
    }
    membership_lookup = {
        (str(row.root_id), str(row.unit_id)): float(row.membership_probability)
        for row in assignment.itertuples(index=False)
    }
    removed = []
    keep = pd.Series(True, index=edges.index)
    for root, part in edges.groupby(edges["root_id"].astype(str)):
        graph = nx.Graph()
        graph.add_edges_from(part[["source", "target"]].astype(str).itertuples(index=False, name=None))
        changed = True
        while changed:
            changed = False
            for leaf in sorted(node for node, degree in graph.degree if degree == 1):
                branch = branch_lookup.get((root, leaf), "")
                if branch == "ROOT":
                    continue
                if (root, branch) in protected_branches:
                    continue
                if membership_lookup.get((root, leaf), 1.0) >= args.membership_threshold:
                    continue
                parent = next(iter(graph.neighbors(leaf)))
                graph.remove_edge(parent, leaf)
                removed.append({"root_id": root, "source": parent, "target": leaf, "reason": "low_membership_terminal_leaf"})
                changed = True
        local_keep = {
            tuple(sorted(edge)) for edge in graph.edges
        }
        for index, row in part.iterrows():
            keep.loc[index] = tuple(sorted((str(row.source), str(row.target)))) in local_keep
    pruned = edges.loc[keep].copy()
    pruned.to_csv(args.output_dir / "PRED_T1_EDGES.csv", index=False)
    removed_units = {(str(row["root_id"]), str(row["target"])) for row in removed}
    branch_keep = [
        (str(row.root_id), str(row.unit_id)) not in removed_units
        for row in branches.itertuples(index=False)
    ]
    branches.loc[branch_keep].to_csv(args.output_dir / "PRED_BRANCH.csv", index=False)
    pt = pd.read_csv(args.output_dir / "PRED_PSEUDOTIME.csv")
    pt_keep = [
        (str(row.root_id), str(row.unit_id)) not in removed_units
        for row in pt.itertuples(index=False)
    ]
    pt.loc[pt_keep].to_csv(args.output_dir / "PRED_PSEUDOTIME.csv", index=False)
    pd.DataFrame(removed, columns=["root_id", "source", "target", "reason"]).to_csv(args.output_dir / "PRUNING_AUDIT.csv", index=False)
    (args.output_dir / "METHOD_STAGE.txt").write_text(
        f"M6: event-protected terminal pruning at membership < {args.membership_threshold}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
