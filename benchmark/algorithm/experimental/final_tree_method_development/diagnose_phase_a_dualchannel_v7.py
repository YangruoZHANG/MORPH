#!/usr/bin/env python3
"""Diagnostic comparison for a truth-blind dual-channel PHASE A T0.

Branch programs are learned once from the high-confidence multiresolution graph.
The existing k=6 local graph is admitted only to a separate topology channel.
Synthetic truth is loaded after construction and is used only by the scorer.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import anndata as ad
import networkx as nx
import pandas as pd


HERE = Path(__file__).resolve().parent


def load_version(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = load_version("run_phase_a_case_v2.py", "phase_a_v2_for_dualchannel_diagnostic")


def topology_candidate(high: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Union evidence channels while retaining the strongest provenance per pair."""
    high = high.copy()
    existing = existing.copy()
    high["channel_priority"] = 0
    existing["channel_priority"] = 1
    combined = pd.concat([high, existing], ignore_index=True, sort=False)
    combined["edge_key"] = [
        V2.P.V1.edge_key(int(row.source), int(row.target))
        for row in combined.itertuples(index=False)
    ]
    combined = combined.sort_values(
        ["channel_priority", "D_joint", "source", "target"], kind="mergesort"
    ).drop_duplicates("edge_key", keep="first")
    combined["joint_distance_ecdf"] = V2.P.V1.empirical_rank(combined.D_joint.to_numpy(float))
    return combined.drop(columns=["edge_key", "channel_priority"])


def construct_variant(
    meta,
    geometry,
    full_candidate,
    existing_candidate,
    membership,
    roots,
    supports,
    variant: str,
):
    high = full_candidate[
        full_candidate.edge_graph_role.astype(str).isin(V2.P.T0_ACTIVE_ROLES)
    ].copy()
    high["joint_distance_ecdf"] = V2.P.V1.empirical_rank(high.D_joint.to_numpy(float))
    topology = topology_candidate(high, existing_candidate)
    union = full_candidate[
        full_candidate.edge_graph_role.astype(str).eq("trajectory_local_union")
    ].copy()
    if len(union):
        union["joint_distance_ecdf"] = V2.P.V1.empirical_rank(union.D_joint.to_numpy(float))
    legal = {
        V2.P.V1.edge_key(int(row.source), int(row.target))
        for row in pd.concat([full_candidate, existing_candidate], ignore_index=True, sort=False).itertuples(index=False)
        if bool(row.trajectory_edge_eligible)
    }
    names = [str(name) for name in meta.obs_names]
    edge_rows, node_rows, backbone_frames, component_rows, model_rows = [], [], [], [], []
    connector_total = 0
    for root in roots:
        root_id = str(root["root_id"])
        root_pos, root_index = int(root["root_pos"]), int(root["root_index"])
        support = supports[root_id]

        high_graph = V2.P.V1.graph_for_support(high, support, membership[:, root_pos])
        high_preliminary = V2.P.V1.component_forest(high_graph)
        high_tau, _ = V2.P.V1.rooted_tau(high_preliminary, root_index, membership[:, root_pos])
        branches, audits = V2.P.C3V2.raw_expression_diagonal_gmm(
            geometry["expression"], high_preliminary, high_tau
        )

        topology_graph = V2.P.V1.graph_for_support(topology, support, membership[:, root_pos])
        if variant == "combined_mst":
            frozen_topology = V2.P.V1.component_forest(topology_graph)
        elif variant == "frozen_branch_quotient":
            frozen_topology = V2.P.V1.quotient_branch_forest(topology_graph, branches)
        else:
            raise ValueError(f"Unknown diagnostic variant: {variant}")

        union_graph = V2.P.V1.graph_for_support(
            union, support, membership[:, root_pos]
        ) if len(union) else nx.Graph()
        union_graph.add_nodes_from(support)
        final, connectors = V2.V4.attach_components_with_union(
            V2.P.V1, frozen_topology, union_graph
        )
        connector_keys = set(connectors)
        connector_total += len(connectors)
        for left, right in final.edges():
            final[left][right]["connector_only"] = (
                V2.P.V1.edge_key(int(left), int(right)) in connector_keys
            )

        tau, component = V2.P.V1.rooted_tau(final, root_index, membership[:, root_pos])
        states = V2.topology_state_labels(final, root_index, root_id)
        backbone_frames.append(V2.backbone_from_states(root_id, final, states, tau, names))
        for audit in audits:
            model_rows.append({"root_id": root_id, **audit})
        for left, right, data in final.edges(data=True):
            edge_rows.append({
                "root_id": root_id,
                "source": int(left),
                "target": int(right),
                "source_name": names[int(left)],
                "target_name": names[int(right)],
                "edge_cost": float(data.get("weight", 0.0)),
                "edge_graph_role": str(data.get("edge_graph_role", "")),
                "candidate_legal": V2.P.V1.edge_key(int(left), int(right)) in legal,
                "connector_only": bool(data.get("connector_only", False)),
            })
        for node in sorted(support):
            node_rows.append({
                "root_id": root_id,
                "node_id": int(node),
                "unit_id": names[int(node)],
                "membership": float(membership[int(node), root_pos]),
                "pseudotime": float(tau.get(int(node), 0.0)),
                "branch_id": f"{root_id}_{branches[int(node)]}",
                "state_id": states[int(node)],
                "component_id": int(component.get(int(node), -1)),
                "is_root": int(node) == root_index,
            })
        for cid, nodes in enumerate(sorted(nx.connected_components(final), key=min)):
            component_rows.append({
                "root_id": root_id,
                "component_id": cid,
                "n_nodes": len(nodes),
                "contains_root": root_index in nodes,
                "n_fine_edges": final.subgraph(nodes).number_of_edges(),
                "n_states": len({states[int(node)] for node in nodes}),
                "n_connectors": sum(
                    V2.P.V1.edge_key(int(left), int(right)) in connector_keys
                    for left, right in final.subgraph(nodes).edges()
                ),
            })
    return {
        "edges": pd.DataFrame(edge_rows),
        "nodes": pd.DataFrame(node_rows),
        "backbone": pd.concat(backbone_frames, ignore_index=True),
        "components": pd.DataFrame(component_rows),
        "branch_models": pd.DataFrame(model_rows),
        "connector_total": connector_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="CORE_05")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    method_dir = args.run_dir.resolve() / "generic_upstream_v1_exact_root" / args.dataset
    meta = ad.read_h5ad(method_dir / "metacell_adata.h5ad")
    geometry = V2.P.V1.prepare_geometry(meta)
    membership, roots, supports = V2.P.inferred_membership(method_dir, meta.n_obs)
    full_candidate, _, _ = V2.P.new_candidates(meta, geometry)
    existing_candidate = V2.P.current_candidate(method_dir)

    # Truth is intentionally loaded only after both production variants exist.
    variants = {
        name: construct_variant(
            meta, geometry, full_candidate, existing_candidate,
            membership, roots, supports, name,
        )
        for name in ("combined_mst", "frozen_branch_quotient")
    }
    truth = V2.Q.add_soft_state_truth(
        args.run_dir.resolve(), V2.P.load_truth_context(args.run_dir.resolve())
    )
    metric_frames = []
    for name, result in variants.items():
        metrics = V2.dual_scale_metrics(args.run_label, name, result, truth)
        metric_frames.append(metrics)
        case_dir = output / name
        case_dir.mkdir(exist_ok=True)
        result["edges"].to_csv(case_dir / "T0_FINE_TRAJECTORY_EDGES.csv", index=False)
        result["backbone"].to_csv(case_dir / "T0_STATE_BACKBONE_EDGES.csv", index=False)
        result["nodes"].to_csv(case_dir / "T0_NODE_ASSIGNMENTS.csv", index=False)
    pd.concat(metric_frames, ignore_index=True).to_csv(
        output / "DUALCHANNEL_DIAGNOSTIC_METRICS.csv", index=False
    )


if __name__ == "__main__":
    main()
