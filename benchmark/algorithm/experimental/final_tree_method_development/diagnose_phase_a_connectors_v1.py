#!/usr/bin/env python3
"""Audit candidate-legal options for detached root-conditioned components."""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd


def edge_key(left: int, right: int) -> tuple[int, int]:
    return tuple(sorted((int(left), int(right))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase-a-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    phase_a = args.phase_a_dir
    nodes = pd.read_csv(phase_a / "T0_STATE_NODE_ASSIGNMENTS.csv")
    full = pd.read_csv(phase_a / "C1_MULTIRESOLUTION_CANDIDATE_EDGES.csv")
    existing = pd.read_csv(
        args.run_dir / "generic_upstream_v1_exact_root" / args.dataset
        / "trajectory_candidate_edges.csv"
    )
    existing = existing[
        existing.trajectory_edge_eligible.astype(str).str.lower().isin({"true", "1", "yes"})
    ].copy()
    existing["D_expression"] = existing.get("d_expression", existing.d_global)
    existing["D_spatial"] = existing.get("d_spatial", existing.d_global)
    existing["D_joint"] = existing.d_global.astype(float)
    existing["edge_graph_role"] = "trajectory_current_single_scale"
    existing["trajectory_edge_eligible"] = True
    audit = pd.read_csv(phase_a / "CANDIDATE_BUILD_AUDIT.csv").iloc[0]
    expression_guard = float(audit.bridge_expression_guard_q95)
    spatial_guard = float(audit.bridge_spatial_guard_q975)
    eligible = pd.concat([
        full[full.trajectory_edge_eligible.astype(bool)], existing
    ], ignore_index=True, sort=False)
    eligible["key"] = [edge_key(row.source, row.target) for row in eligible.itertuples()]
    eligible = eligible.sort_values(["D_joint", "source", "target"]).drop_duplicates("key")
    rows = []
    for root_id, local in nodes.groupby("root_id"):
        support = set(local.node_id.astype(int))
        anchor = int(local[local.is_root.astype(str).str.lower().isin({"true", "1", "yes"})].iloc[0].node_id)
        graph = nx.Graph()
        graph.add_nodes_from(support)
        graph.add_edges_from(
            (int(row.source), int(row.target)) for row in eligible.itertuples()
            if int(row.source) in support and int(row.target) in support
        )
        anchor_component = set(nx.node_connected_component(graph, anchor))
        membership = dict(zip(local.node_id.astype(int), local.membership.astype(float)))
        detached = [
            set(component) for component in nx.connected_components(graph)
            if anchor not in component
        ]
        for component_index, component in enumerate(sorted(detached, key=min)):
            bridges = full[
                (
                    full.source.astype(int).isin(component)
                    & full.target.astype(int).isin(anchor_component)
                ) | (
                    full.target.astype(int).isin(component)
                    & full.source.astype(int).isin(anchor_component)
                )
            ].copy()
            bridges = bridges.sort_values(
                ["D_joint", "D_expression", "D_spatial", "source", "target"],
                kind="mergesort",
            )
            for rank, bridge in enumerate(bridges.head(5).itertuples(index=False), start=1):
                rows.append({
                    "root_id": str(root_id), "detached_component": component_index,
                    "detached_nodes": ";".join(map(str, sorted(component))),
                    "n_detached_nodes": len(component), "candidate_rank": rank,
                    "source": int(bridge.source), "target": int(bridge.target),
                    "source_name": str(bridge.source_name), "target_name": str(bridge.target_name),
                    "D_expression": float(bridge.D_expression),
                    "D_spatial": float(bridge.D_spatial), "D_joint": float(bridge.D_joint),
                    "edge_graph_role": str(bridge.edge_graph_role),
                    "originally_trajectory_eligible": bool(bridge.trajectory_edge_eligible),
                    "source_membership": membership.get(int(bridge.source), 0.0),
                    "target_membership": membership.get(int(bridge.target), 0.0),
                    "minimum_endpoint_membership": min(
                        membership.get(int(bridge.source), 0.0),
                        membership.get(int(bridge.target), 0.0),
                    ),
                    "passes_expression_guard": float(bridge.D_expression) <= expression_guard,
                    "passes_spatial_guard": float(bridge.D_spatial) <= spatial_guard,
                    "passes_both_guards": (
                        float(bridge.D_expression) <= expression_guard
                        and float(bridge.D_spatial) <= spatial_guard
                    ),
                })
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False) if len(result) else "No detached components")


if __name__ == "__main__":
    main()
