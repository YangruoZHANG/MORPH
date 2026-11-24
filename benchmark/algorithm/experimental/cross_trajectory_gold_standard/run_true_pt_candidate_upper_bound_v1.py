#!/usr/bin/env python3
"""Truth-pseudotime upper bound for rooted candidate-tree construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd

from score_benchmark_prediction_v1 import score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    prediction = args.output_dir / "prediction"
    prediction.mkdir(parents=True, exist_ok=True)
    truth_dir = args.run_dir / "event" / "truth"
    metacell_dir = args.run_dir / "formal_metacell"
    upstream_root = args.run_dir / "generic_upstream_v1_exact_root"
    upstream = upstream_root / args.setting
    fixed = pd.read_csv(upstream_root / "FIXED_ROOT_INPUT.csv")
    candidates = pd.read_csv(upstream / "trajectory_candidate_edges.csv")
    root_comp = pd.read_csv(metacell_dir / "METACELL_ROOT_COMPOSITION.csv")
    truth_summary = pd.read_csv(metacell_dir / "METACELL_TRUTH_SUMMARY.csv")
    dominant = root_comp.sort_values(["root_probability", "root_id"], ascending=[False, True]).drop_duplicates("metacell_id")
    assignment = dict(zip(dominant["metacell_id"].astype(str), dominant["root_id"].astype(str)))
    true_pt = dict(zip(truth_summary["metacell_id"].astype(str), truth_summary["median_true_pseudotime"].astype(float)))
    anchors = dict(zip(fixed["truth_root_id"].astype(str), fixed["root_anchor_id"].astype(str)))
    for root, anchor in anchors.items():
        assignment[anchor] = root
        true_pt[anchor] = -1e-6

    edge_rows = []
    branch_rows = []
    pt_rows = []
    audit = []
    for root, anchor in anchors.items():
        nodes = {node for node, value in assignment.items() if value == root}
        directed = nx.DiGraph()
        directed.add_nodes_from(nodes)
        for edge in candidates.itertuples(index=False):
            a, b = str(edge.source_name), str(edge.target_name)
            if a not in nodes or b not in nodes:
                continue
            if true_pt[a] <= true_pt[b]:
                directed.add_edge(a, b, weight=float(edge.d_global))
            if true_pt[b] <= true_pt[a]:
                directed.add_edge(b, a, weight=float(edge.d_global))
        paths = nx.single_source_dijkstra_path(directed, anchor, weight="weight")
        reachable = set(paths)
        tree = nx.DiGraph()
        tree.add_node(anchor)
        for node, path in paths.items():
            if node == anchor:
                continue
            parent = path[-2]
            tree.add_edge(parent, node)
            edge_rows.append({"root_id": root, "source": parent, "target": node})
        for node in sorted(nodes):
            if node == anchor:
                branch = "ROOT"
            elif node not in reachable:
                branch = f"UNREACHABLE_{node}"
            else:
                path = nx.shortest_path(tree, anchor, node)
                branch = f"BR_{root}_{path[1]}"
            branch_rows.append({"unit_id": node, "root_id": root, "branch_id": branch})
            if node in reachable:
                pt_rows.append({"unit_id": node, "root_id": root, "pseudotime": max(0.0, true_pt[node])})
        audit.append({
            "root_id": root,
            "n_nodes": len(nodes),
            "n_reachable": len(reachable),
            "reachability": len(reachable) / len(nodes) if nodes else 0.0,
            "rooted_tree": bool(len(reachable) == len(nodes) and nx.is_arborescence(tree)),
            "truth_used": True,
        })

    pd.DataFrame({"root_id": list(anchors)}).to_csv(prediction / "PRED_ROOTS.csv", index=False)
    pd.DataFrame([
        {"unit_id": node, "root_id": root, "membership_probability": 1.0}
        for node, root in sorted(assignment.items())
    ]).to_csv(prediction / "PRED_CELL_OR_METACELL_ASSIGNMENT.csv", index=False)
    edges = pd.DataFrame(edge_rows, columns=["root_id", "source", "target"])
    edges.to_csv(prediction / "PRED_T0_EDGES.csv", index=False)
    edges.to_csv(prediction / "PRED_T1_EDGES.csv", index=False)
    pd.DataFrame(branch_rows).to_csv(prediction / "PRED_BRANCH.csv", index=False)
    pd.DataFrame(pt_rows).to_csv(prediction / "PRED_PSEUDOTIME.csv", index=False)
    pd.DataFrame(columns=[
        "event_id", "event_type", "source_root", "receiver_root", "affected_branch", "score", "is_called",
    ]).to_csv(prediction / "PRED_EVENTS.csv", index=False)
    pd.DataFrame(audit).to_csv(args.output_dir / "TRUE_PT_DAG_AUDIT.csv", index=False)
    score_dir = args.output_dir / "score"
    score(truth_dir, prediction, score_dir, f"true_PT_candidate_DAG_{args.setting}", metacell_dir)
    result = pd.read_csv(score_dir / "SCORE_SUMMARY.csv").iloc[0].to_dict()
    status = {
        "protocol": "ORACLE_DECOMPOSITION_SPEC_V1",
        "truth_used": True,
        "all_nodes_reachable": bool(all(row["reachability"] == 1.0 for row in audit)),
        "all_roots_are_arborescences": bool(all(row["rooted_tree"] for row in audit)),
        "forest_branch_ARI": result.get("forest_branch_ARI"),
        "forest_pt_SR": result.get("forest_pt_SR"),
        "final_topology_F1": result.get("final_topology_F1"),
        "distance_T1_to_true_T1": result.get("distance_T1_to_true_T1"),
    }
    (args.output_dir / "RUN_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
