#!/usr/bin/env python3
"""Export global-first M0 and hard root-conditioned M1 baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--fixed-root-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["M0", "M1"], required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fixed = pd.read_csv(args.fixed_root_input)
    candidates = pd.read_csv(args.upstream_dir / "trajectory_candidate_edges.csv")
    membership = pd.read_csv(args.upstream_dir / "membership_A_sharp.csv")
    node_table = membership.drop_duplicates("node_id")[["node_id", "node_name"]].copy()
    id_to_name = dict(zip(node_table["node_id"].astype(int), node_table["node_name"].astype(str)))
    name_to_id = {name: node for node, name in id_to_name.items()}
    pos_to_root = dict(enumerate(fixed["truth_root_id"].astype(str)))
    anchor_by_pos = dict(enumerate(fixed["root_anchor_id"].astype(str)))

    graph = nx.Graph()
    for row in candidates.itertuples(index=False):
        graph.add_edge(int(row.source), int(row.target), weight=float(row.d_global))
    if not nx.is_connected(graph):
        raise ValueError("Candidate graph must be connected for M0/M1 export")

    pivot = membership.pivot(index="node_id", columns="root_pos", values="A_sharp").fillna(0.0)
    primary = pivot.idxmax(axis=1).astype(int).to_dict()
    for root_pos, anchor_name in anchor_by_pos.items():
        primary[name_to_id[anchor_name]] = root_pos

    edge_records = []
    assignment = {}
    if args.stage == "M0":
        tree = nx.minimum_spanning_tree(graph, weight="weight")
        distances = {
            root_pos: nx.single_source_dijkstra_path_length(tree, name_to_id[anchor], weight="weight")
            for root_pos, anchor in anchor_by_pos.items()
        }
        for node in tree.nodes:
            assignment[node] = min(distances, key=lambda root: (distances[root].get(node, np.inf), root))
        for source, target, data in tree.edges(data=True):
            if assignment[source] == assignment[target]:
                edge_records.append((assignment[source], source, target, float(data["weight"])))
    else:
        assignment = primary
        for root_pos in sorted(pos_to_root):
            nodes = [node for node, root in assignment.items() if root == root_pos]
            local = graph.subgraph(nodes).copy()
            if len(nodes) and not nx.is_connected(local):
                raise ValueError(f"Hard root candidate subgraph {root_pos} is disconnected")
            tree = nx.minimum_spanning_tree(local, weight="weight")
            edge_records.extend((root_pos, source, target, float(data["weight"])) for source, target, data in tree.edges(data=True))

    edge_frame = pd.DataFrame(edge_records, columns=["root_pos", "source_id", "target_id", "weight"])
    edge_frame["root_id"] = edge_frame["root_pos"].map(pos_to_root)
    edge_frame["source"] = edge_frame["source_id"].map(id_to_name)
    edge_frame["target"] = edge_frame["target_id"].map(id_to_name)
    contract_edges = edge_frame[["root_id", "source", "target"]]
    contract_edges.to_csv(args.output_dir / "PRED_T0_EDGES.csv", index=False)
    contract_edges.to_csv(args.output_dir / "PRED_T1_EDGES.csv", index=False)
    pd.DataFrame({"root_id": list(pos_to_root.values())}).to_csv(args.output_dir / "PRED_ROOTS.csv", index=False)
    pd.DataFrame([
        {"unit_id": id_to_name[node], "root_id": pos_to_root[root], "membership_probability": 1.0}
        for node, root in sorted(assignment.items())
    ]).to_csv(args.output_dir / "PRED_CELL_OR_METACELL_ASSIGNMENT.csv", index=False)

    branch_rows = []
    pt_rows = []
    for root_pos, part in edge_frame.groupby("root_pos"):
        root_pos = int(root_pos)
        anchor = anchor_by_pos[root_pos]
        local = nx.Graph()
        for row in part.itertuples(index=False):
            local.add_edge(str(row.source), str(row.target), weight=float(row.weight))
        if anchor not in local:
            local.add_node(anchor)
        lengths = nx.single_source_dijkstra_path_length(local, anchor, weight="weight")
        scale = max(lengths.values()) if lengths else 1.0
        for unit in local.nodes:
            if unit == anchor:
                branch = "ROOT"
            else:
                path = nx.shortest_path(local, anchor, unit)
                branch = f"BR_{pos_to_root[root_pos]}_{path[1]}"
            branch_rows.append({"unit_id": unit, "root_id": pos_to_root[root_pos], "branch_id": branch})
            pt_rows.append({"unit_id": unit, "root_id": pos_to_root[root_pos], "pseudotime": lengths.get(unit, 0.0) / max(scale, 1e-9)})
    pd.DataFrame(branch_rows).to_csv(args.output_dir / "PRED_BRANCH.csv", index=False)
    pd.DataFrame(pt_rows).to_csv(args.output_dir / "PRED_PSEUDOTIME.csv", index=False)
    pd.DataFrame(columns=[
        "event_id", "event_type", "source_root", "receiver_root", "affected_branch", "score", "is_called",
    ]).to_csv(args.output_dir / "PRED_EVENTS.csv", index=False)
    (args.output_dir / "METHOD_STAGE.txt").write_text(
        f"{args.stage}: {'global MST with root Voronoi partition' if args.stage == 'M0' else 'hard root-conditioned candidate MST'}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
