#!/usr/bin/env python3
"""C1_v2 + C3_v4: freeze high-confidence trees before quotient connectors."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import anndata as ad
import networkx as nx
import pandas as pd


HERE = Path(__file__).resolve().parent


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attach_components_with_union(v1, high_forest: nx.Graph, full_graph: nx.Graph) -> tuple[nx.Graph, list[tuple[int, int]]]:
    result = high_forest.copy()
    components = list(nx.connected_components(high_forest))
    node_component = {int(node): cid for cid, nodes in enumerate(components) for node in nodes}
    quotient = nx.Graph()
    quotient.add_nodes_from(range(len(components)))
    best = {}
    for i, j, data in full_graph.edges(data=True):
        if str(data.get("edge_graph_role")) != "trajectory_local_union":
            continue
        ci, cj = node_component[int(i)], node_component[int(j)]
        if ci == cj:
            continue
        key = v1.edge_key(ci, cj)
        trial = (float(data["weight"]), min(int(i), int(j)), max(int(i), int(j)), dict(data))
        if key not in best or trial[:3] < best[key][:3]:
            best[key] = trial
    for (ci, cj), (weight, i, j, data) in best.items():
        quotient.add_edge(ci, cj, weight=weight, source=i, target=j, source_data=data)
    used = []
    for quotient_nodes in nx.connected_components(quotient):
        sub = quotient.subgraph(quotient_nodes)
        for _, _, data in nx.minimum_spanning_tree(sub, weight="weight").edges(data=True):
            i, j = int(data["source"]), int(data["target"])
            result.add_edge(i, j, **data["source_data"])
            used.append(v1.edge_key(i, j))
    return result, used


def construct_t0_v4(v1, method_dir, meta, candidate, geometry):
    a, roots, active = v1.load_membership(method_dir, meta.n_obs)
    high_candidate = candidate[~candidate.edge_graph_role.astype(str).eq("trajectory_local_union")].copy()
    edge_rows, node_rows, component_rows, branch_audit = [], [], [], []
    legal = {v1.edge_key(int(r.source), int(r.target)) for r in candidate[candidate.trajectory_edge_eligible].itertuples(index=False)}
    connector_total = 0
    for root in roots:
        k, root_index = int(root["root_pos"]), int(root["root_index"])
        support = v1.root_support(active, k, root_index)
        high_graph = v1.graph_for_support(high_candidate, support, a[:, k])
        preliminary = v1.component_forest(high_graph)
        tau_pre, _ = v1.rooted_tau(preliminary, root_index, a[:, k])
        branches, audits = v1.infer_branch_communities(geometry["expression"], preliminary, tau_pre)
        frozen_high = v1.quotient_branch_forest(high_graph, branches)
        full_graph = v1.graph_for_support(candidate, support, a[:, k])
        final, connectors = attach_components_with_union(v1, frozen_high, full_graph)
        connector_total += len(connectors)
        tau, component_ids = v1.rooted_tau(final, root_index, a[:, k])
        for audit in audits:
            branch_audit.append({"root_pos": k, "root_id": root["root_id"], **audit})
        for i, j, data in final.edges(data=True):
            edge_rows.append({
                "root_pos": k, "root_id": root["root_id"], "source": i, "target": j,
                "source_name": str(meta.obs_names[i]), "target_name": str(meta.obs_names[j]),
                "edge_cost": float(data["weight"]), "edge_graph_role": data["edge_graph_role"],
                "candidate_legal": v1.edge_key(int(i), int(j)) in legal,
                "connector_only": v1.edge_key(int(i), int(j)) in set(connectors),
            })
        for node in sorted(support):
            node_rows.append({
                "node_id": node, "unit_id": str(meta.obs_names[node]), "root_pos": k,
                "root_id": root["root_id"], "membership": float(a[node, k]),
                "pseudotime": float(tau.get(node, 0.0)), "branch_id": f"{root['root_id']}_{branches[node]}",
                "component_id": int(component_ids.get(node, -1)),
            })
        high_count = nx.number_connected_components(high_graph)
        for cid, nodes in enumerate(sorted(nx.connected_components(final), key=lambda x: min(x))):
            component_rows.append({
                "root_pos": k, "root_id": root["root_id"], "component_id": cid,
                "n_nodes": len(nodes), "contains_root": root_index in nodes,
                "n_T0_edges": final.subgraph(nodes).number_of_edges(),
                "n_high_confidence_components_before_connector": high_count,
                "n_connector_edges_for_root": sum(1 for i, j in connectors if i in nodes and j in nodes),
            })
    return {
        "roots": roots, "membership": a, "edges": pd.DataFrame(edge_rows),
        "nodes": pd.DataFrame(node_rows), "components": pd.DataFrame(component_rows),
        "branch_audit": pd.DataFrame(branch_audit), "connector_total": connector_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=HERE.parents[2])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="CORE_05")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project, run_dir, output = args.project_root.resolve(), args.run_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    v1 = load_module(HERE / "run_c1_c3_development_v1.py", "nicheforest_v1_for_v4")
    c3v2 = load_module(HERE / "run_c1_c3_development_v2.py", "nicheforest_c3v2_for_v4")
    c1v2 = load_module(HERE / "run_c1_v2_c3_v2.py", "nicheforest_c1v2_for_v4")
    v1.infer_branch_communities = c3v2.raw_expression_diagonal_gmm
    method_dir = run_dir / "generic_upstream_v1_exact_root" / args.dataset
    meta = ad.read_h5ad(method_dir / "metacell_adata.h5ad")
    geometry = v1.prepare_geometry(meta)
    candidate, candidate_audit = c1v2.candidate_v2(v1, meta, geometry)
    candidate.to_csv(output / "C1_MULTIRESOLUTION_CANDIDATE_EDGES.csv", index=False)
    pd.DataFrame([candidate_audit]).to_csv(output / "C1_CANDIDATE_CONSTRUCTION_AUDIT.csv", index=False)
    v1.candidate_truth_audit(candidate, run_dir / "formal_metacell", run_dir / "event" / "truth").to_csv(output / "C1_TRUTH_ONLY_RECOVERABILITY_AUDIT.csv", index=False)
    t0 = construct_t0_v4(v1, method_dir, meta, candidate, geometry)
    t0["edges"].to_csv(output / "C3_BRANCH_SCALE_T0_EDGES.csv", index=False)
    t0["nodes"].to_csv(output / "C3_BRANCH_SCALE_NODE_ASSIGNMENTS.csv", index=False)
    t0["components"].to_csv(output / "C3_COMPONENT_RETENTION_AUDIT.csv", index=False)
    t0["branch_audit"].to_csv(output / "C3_BRANCH_MODEL_SELECTION_AUDIT.csv", index=False)
    t0["edges"][t0["edges"].connector_only].to_csv(output / "C3_CONNECTOR_ONLY_EDGES_USED.csv", index=False)
    prediction = output / "prediction"
    v1.write_prediction(t0, prediction)
    score_dir = output / "score"
    v1.evaluate_only(project, run_dir, prediction, score_dir, "C1_v2_C3_frozen_internal_v4")
    summary = pd.read_csv(score_dir / "SCORE_SUMMARY.csv").iloc[0].to_dict()
    integrity = {
        "method_version": "C1_candidate_multires_v2+C3_T0_frozen_internal_v4",
        "parent": "C1_candidate_multires_v2+C3_T0_role_hierarchical_v3",
        "truth_available_to_constructor": False, "relation_module_called": False,
        "connector_semantics": "post_hoc_quotient_connector_after_internal_tree_and_branch_freeze",
        "n_union_connectors_used": int(t0["connector_total"]),
        "all_T0_edges_candidate_legal": bool(t0["edges"]["candidate_legal"].all()),
        "score": summary,
    }
    (output / "RUN_DECISION_INPUT.json").write_text(json.dumps(integrity, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(integrity, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
