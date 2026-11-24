#!/usr/bin/env python3
"""Print endpoint-level legality for truth-defined synthetic event swaps.

This is a scorer-only diagnostic.  It never participates in construction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd


def edge_pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase-a-dir", type=Path, required=True)
    args = parser.parse_args()

    truth = args.run_dir / "event" / "truth"
    added = pd.read_csv(truth / "TRUE_ADDED_EDGES.csv")
    events = pd.read_csv(truth / "TRUE_EVENTS.csv")
    state = pd.read_csv(args.run_dir / "formal_metacell" / "METACELL_STATE_COMPOSITION.csv")
    state = state[state.stage.astype(str).eq("T1")].sort_values(
        ["metacell_id", "root_id", "state_probability", "state_id"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["metacell_id", "root_id"])
    state_lookup = {
        (str(row.metacell_id), str(row.root_id)): str(row.state_id)
        for row in state.itertuples(index=False)
    }
    candidate = pd.read_csv(args.phase_a_dir / "C1_MULTIRESOLUTION_CANDIDATE_EDGES.csv")
    candidate = candidate[candidate.trajectory_edge_eligible.astype(bool)]
    t0 = pd.read_csv(args.phase_a_dir / "T0_FINE_TRAJECTORY_EDGES.csv")
    nodes = pd.read_csv(args.phase_a_dir / "T0_STATE_NODE_ASSIGNMENTS.csv")
    rows = []
    for event in events.itertuples(index=False):
        if not bool(event.topology_changed):
            continue
        event_added = added[
            added.event_id.astype(str).eq(str(event.event_id))
        ].iloc[0]
        root = str(event.receiver_root)
        parent_state, child_state = str(event_added.source), str(event_added.target)
        local_nodes = nodes[nodes.root_id.astype(str).eq(root)]
        node_ids = set(local_nodes.node_id.astype(int))
        root_node = int(local_nodes[local_nodes.is_root.astype(bool)].iloc[0].node_id)
        local_t0 = t0[t0.root_id.astype(str).eq(root)]
        graph = nx.Graph(local_t0[["source", "target"]].astype(int).itertuples(index=False, name=None))
        tree = nx.bfs_tree(graph, root_node)
        current_parent = {int(node): int(parent) for parent, node in tree.edges()}
        local_candidate = candidate[
            candidate.source.astype(int).isin(node_ids)
            & candidate.target.astype(int).isin(node_ids)
        ]
        for edge in local_candidate.itertuples(index=False):
            left, right = int(edge.source), int(edge.target)
            left_name, right_name = str(edge.source_name), str(edge.target_name)
            left_state = state_lookup.get((left_name, root), "")
            right_state = state_lookup.get((right_name, root), "")
            if edge_pair(left_state, right_state) != edge_pair(parent_state, child_state):
                continue
            if left_state == child_state:
                child, new_parent = left, right
                child_name, new_parent_name = left_name, right_name
            else:
                child, new_parent = right, left
                child_name, new_parent_name = right_name, left_name
            old_parent = current_parent.get(child)
            old_parent_name = ""
            old_parent_state = ""
            new_parent_in_child_subtree = True
            if old_parent is not None:
                old_parent_name = str(local_nodes.set_index("node_id").loc[old_parent, "unit_id"])
                old_parent_state = state_lookup.get((old_parent_name, root), "")
                without = graph.copy()
                without.remove_edge(old_parent, child)
                new_parent_in_child_subtree = new_parent in nx.node_connected_component(without, child)
            rows.append({
                "event_id": str(event.event_id), "root_id": root,
                "candidate_child": child_name, "candidate_child_state": child_state,
                "candidate_new_parent": new_parent_name,
                "candidate_new_parent_state": parent_state,
                "current_parent": old_parent_name,
                "current_parent_state": old_parent_state,
                "child_has_current_parent": old_parent is not None,
                "new_parent_in_child_subtree": bool(new_parent_in_child_subtree),
                "strict_swap_legal": old_parent is not None and not new_parent_in_child_subtree,
                "candidate_role": str(edge.edge_graph_role),
                "candidate_joint_distance": float(edge.D_joint),
            })
    frame = pd.DataFrame(rows)
    if len(frame):
        print(frame.to_csv(index=False))
    else:
        print("no truth-added candidate realizations")


if __name__ == "__main__":
    main()
