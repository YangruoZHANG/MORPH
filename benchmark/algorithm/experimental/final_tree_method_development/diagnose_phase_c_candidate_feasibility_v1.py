#!/usr/bin/env python3
"""Audit whether oracle truth transitions exist in the frozen candidate graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_phase_c_oracle_refinement_v1 import (
    as_bool, dominant_state_lookup, edge_key, truth_edge_set,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--phase-a-subdir", required=True)
    parser.add_argument("--phase-c-subdir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for status_path in sorted(args.grid_root.rglob("TOPOLOGY_DEV_CASE_STATUS.json")):
        run = status_path.parent
        phase_a = run / args.phase_a_subdir
        phase_c = run / args.phase_c_subdir
        nodes = pd.read_csv(phase_a / "T0_STATE_NODE_ASSIGNMENTS.csv")
        candidate = pd.read_csv(phase_a / "C1_MULTIRESOLUTION_CANDIDATE_EDGES.csv")
        connector = phase_a / "ROOT_SPECIFIC_CONNECTOR_EDGES.csv"
        if connector.exists() and connector.stat().st_size > 2:
            try:
                extra = pd.read_csv(connector)
            except pd.errors.EmptyDataError:
                extra = pd.DataFrame()
            if len(extra):
                extra["trajectory_edge_eligible"] = True
                candidate = pd.concat([candidate, extra], ignore_index=True, sort=False)
        candidate = candidate[candidate.trajectory_edge_eligible.map(as_bool)]
        state = pd.read_csv(run / "formal_metacell" / "METACELL_STATE_COMPOSITION.csv")
        state_lookup = dominant_state_lookup(state, "T1")
        truth_t0 = truth_edge_set(pd.read_csv(run / "event" / "truth" / "TRUE_T0_EDGES.csv"))
        truth_t1 = truth_edge_set(pd.read_csv(run / "event" / "truth" / "TRUE_T1_EDGES.csv"))
        true_added = truth_t1 - truth_t0
        result = pd.read_csv(phase_c / "PHASE_C_ORACLE_REFINEMENT.csv")
        result = result[result.gamma.astype(float).eq(0.35)]
        for event in result.itertuples(index=False):
            root = str(event.receiver_root)
            local = nodes[nodes.root_id.astype(str).eq(root)]
            node_names = dict(zip(local.node_id.astype(int), local.unit_id.astype(str)))
            node_ids = set(node_names)
            mapped_candidate = set()
            for edge in candidate.itertuples(index=False):
                left, right = int(edge.source), int(edge.target)
                if left not in node_ids or right not in node_ids:
                    continue
                left_state = state_lookup.get((node_names[left], root))
                right_state = state_lookup.get((node_names[right], root))
                if left_state and right_state and left_state != right_state:
                    state_left, state_right = sorted((left_state, right_state))
                    mapped_candidate.add((root, state_left, state_right))
            event_added = {edge for edge in true_added if edge[0] == root}
            reachable = event_added & mapped_candidate
            rows.append({
                "run": run.name,
                "setting": status_path.parents[1].name,
                "event_id": str(event.event_id),
                "event_type": str(event.event_type),
                "causal": as_bool(event.causal),
                "receiver_root": root,
                "n_true_added_edges": len(event_added),
                "n_true_added_candidate_reachable": len(reachable),
                "oracle_candidate_feasible": bool(reachable) if event_added else False,
                "accepted": as_bool(event.accepted),
                "oracle_topology_gain": float(event.oracle_topology_gain),
                "formal_oracle_topology_gain": float(event.formal_oracle_topology_gain),
                "event_location_gain": float(event.event_location_gain),
            })
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    causal = output[output.causal]
    feasible = causal[causal.oracle_candidate_feasible]
    print({
        "n_causal": len(causal),
        "n_oracle_candidate_feasible": len(feasible),
        "feasible_positive_gain": int((feasible.oracle_topology_gain > 0).sum()),
        "feasible_negative_gain": int((feasible.oracle_topology_gain < 0).sum()),
    })


if __name__ == "__main__":
    main()
