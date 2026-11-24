#!/usr/bin/env python3
"""Oracle root-membership and candidate-representability diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from score_benchmark_prediction_v1 import score


def pair(a: object, b: object) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def build_oracle_membership_mst(run_dir: Path, setting: str, output_dir: Path) -> dict[str, object]:
    truth_dir = run_dir / "event" / "truth"
    metacell_dir = run_dir / "formal_metacell"
    upstream_root = run_dir / "generic_upstream_v1_exact_root"
    upstream = upstream_root / setting
    fixed = pd.read_csv(upstream_root / "FIXED_ROOT_INPUT.csv")
    candidates = pd.read_csv(upstream / "trajectory_candidate_edges.csv")
    root_comp = pd.read_csv(metacell_dir / "METACELL_ROOT_COMPOSITION.csv")
    dominant = root_comp.sort_values(["root_probability", "root_id"], ascending=[False, True]).drop_duplicates("metacell_id")
    assignment = dict(zip(dominant["metacell_id"].astype(str), dominant["root_id"].astype(str)))
    for row in fixed.itertuples(index=False):
        assignment[str(row.root_anchor_id)] = str(row.truth_root_id)

    graph = nx.Graph()
    for row in candidates.itertuples(index=False):
        graph.add_edge(str(row.source_name), str(row.target_name), weight=float(row.d_global))
    roots = fixed["truth_root_id"].astype(str).tolist()
    anchors = dict(zip(fixed["truth_root_id"].astype(str), fixed["root_anchor_id"].astype(str)))
    edge_rows = []
    branch_rows = []
    pt_rows = []
    connectivity_rows = []
    for root in roots:
        nodes = sorted(node for node, assigned_root in assignment.items() if assigned_root == root and node in graph)
        local = graph.subgraph(nodes).copy()
        components = list(nx.connected_components(local)) if len(local) else []
        connectivity_rows.append({
            "setting_id": setting,
            "root_id": root,
            "n_nodes": len(nodes),
            "n_candidate_components": len(components),
            "candidate_connected": len(components) == 1,
            "truth_used": True,
        })
        forest = nx.minimum_spanning_tree(local, weight="weight")
        for source, target in forest.edges:
            edge_rows.append({"root_id": root, "source": source, "target": target})
        anchor = anchors[root]
        if anchor not in forest:
            forest.add_node(anchor)
        reachable = nx.node_connected_component(forest, anchor)
        weighted_lengths = nx.single_source_dijkstra_path_length(forest, anchor, weight="weight")
        scale = max(weighted_lengths.values()) if weighted_lengths else 1.0
        for node in forest.nodes:
            if node == anchor:
                branch = "ROOT"
            elif node not in reachable:
                branch = f"UNREACHABLE_{node}"
            else:
                path = nx.shortest_path(forest, anchor, node)
                branch = f"BR_{root}_{path[1]}"
            branch_rows.append({"unit_id": node, "root_id": root, "branch_id": branch})
            if node in weighted_lengths:
                pt_rows.append({"unit_id": node, "root_id": root, "pseudotime": weighted_lengths[node] / max(scale, 1e-12)})

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"root_id": roots}).to_csv(output_dir / "PRED_ROOTS.csv", index=False)
    pd.DataFrame([
        {"unit_id": node, "root_id": root, "membership_probability": 1.0}
        for node, root in sorted(assignment.items())
    ]).to_csv(output_dir / "PRED_CELL_OR_METACELL_ASSIGNMENT.csv", index=False)
    edges = pd.DataFrame(edge_rows, columns=["root_id", "source", "target"])
    edges.to_csv(output_dir / "PRED_T0_EDGES.csv", index=False)
    edges.to_csv(output_dir / "PRED_T1_EDGES.csv", index=False)
    pd.DataFrame(branch_rows).to_csv(output_dir / "PRED_BRANCH.csv", index=False)
    pd.DataFrame(pt_rows).to_csv(output_dir / "PRED_PSEUDOTIME.csv", index=False)
    pd.DataFrame(columns=[
        "event_id", "event_type", "source_root", "receiver_root", "affected_branch", "score", "is_called",
    ]).to_csv(output_dir / "PRED_EVENTS.csv", index=False)
    pd.DataFrame(connectivity_rows).to_csv(output_dir.parent / f"{setting}_ORACLE_MEMBERSHIP_CONNECTIVITY.csv", index=False)
    return {
        "all_true_membership_candidate_subgraphs_connected": bool(all(row["candidate_connected"] for row in connectivity_rows)),
        "max_candidate_components_per_root": max((row["n_candidate_components"] for row in connectivity_rows), default=0),
    }


def state_edge_coverage(run_dir: Path, setting: str, thresholds: list[float]) -> pd.DataFrame:
    truth_dir = run_dir / "event" / "truth"
    metacell_dir = run_dir / "formal_metacell"
    upstream = run_dir / "generic_upstream_v1_exact_root" / setting
    state = pd.read_csv(metacell_dir / "METACELL_STATE_COMPOSITION.csv")
    state = state[state["stage"].astype(str).eq("T1")].copy()
    candidates = pd.read_csv(upstream / "trajectory_candidate_edges.csv")
    candidate_pairs = {pair(row.source_name, row.target_name) for row in candidates.itertuples(index=False)}
    rows = []
    for table_name in ("TRUE_T0_EDGES.csv", "TRUE_T1_EDGES.csv", "TRUE_ADDED_EDGES.csv", "TRUE_REMOVED_EDGES.csv"):
        truth = pd.read_csv(truth_dir / table_name)
        for threshold in thresholds:
            covered = 0
            complete = 0
            for edge in truth.itertuples(index=False):
                source_units = set(state[
                    state["root_id"].astype(str).eq(str(edge.root_id))
                    & state["state_id"].astype(str).eq(str(edge.source))
                    & state["state_probability"].astype(float).ge(threshold)
                ]["metacell_id"].astype(str))
                target_units = set(state[
                    state["root_id"].astype(str).eq(str(edge.root_id))
                    & state["state_id"].astype(str).eq(str(edge.target))
                    & state["state_probability"].astype(float).ge(threshold)
                ]["metacell_id"].astype(str))
                mapping_complete = bool(source_units and target_units)
                complete += int(mapping_complete)
                covered += int(mapping_complete and any(pair(a, b) in candidate_pairs for a in source_units for b in target_units if a != b))
            rows.append({
                "setting_id": setting,
                "truth_table": table_name,
                "state_probability_threshold": threshold,
                "n_truth_edges": len(truth),
                "n_edges_with_complete_mapping": complete,
                "n_edges_represented_by_candidate": covered,
                "mapping_coverage": complete / len(truth) if len(truth) else np.nan,
                "candidate_recall_among_mapped": covered / complete if complete else np.nan,
                "truth_used": True,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--settings", nargs="+", default=["CORE_19", "CORE_05", "CORE_06"])
    parser.add_argument("--replicate-dir", default="replicate_01_seed_7112101")
    parser.add_argument("--state-thresholds", nargs="+", type=float, default=[0.0, 0.2, 0.5, 0.7])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    coverage_rows = []
    for setting in args.settings:
        run_dir = args.protocol_run_root / setting / args.replicate_dir
        prediction = args.output_dir / setting / "ORACLE_MEMBERSHIP_MST_PREDICTION"
        connectivity = build_oracle_membership_mst(run_dir, setting, prediction)
        score_dir = args.output_dir / setting / "ORACLE_MEMBERSHIP_MST_SCORE"
        score(
            run_dir / "event" / "truth", prediction, score_dir,
            f"oracle_membership_candidate_MST_{setting}", run_dir / "formal_metacell",
        )
        row = pd.read_csv(score_dir / "SCORE_SUMMARY.csv").iloc[0].to_dict()
        summary_rows.append({"setting_id": setting, **connectivity, **row, "truth_used": True})
        coverage_rows.append(state_edge_coverage(run_dir, setting, args.state_thresholds))
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "ORACLE_MEMBERSHIP_MST_SUMMARY.csv", index=False)
    pd.concat(coverage_rows, ignore_index=True).to_csv(args.output_dir / "CANDIDATE_STATE_EDGE_COVERAGE.csv", index=False)
    status = {
        "protocol": "ORACLE_DECOMPOSITION_SPEC_V1",
        "truth_used": True,
        "state_probability_thresholds": args.state_thresholds,
        "completed": True,
    }
    (args.output_dir / "RUN_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
