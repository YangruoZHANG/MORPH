#!/usr/bin/env python3
"""PHASE A v2 with dual-scale branch programs and topology states."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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


P = load_version("score_tree_geometry.py", "phase_a_for")
Q = load_version("score_tree_support.py", "phase_a_for")
R = load_version("score_tree_pseudotime.py", "phase_a_for")
V4 = load_version("run_branch_state_experiment.py", "c3_connector_for")


def topology_state_labels(forest: nx.Graph, root: int, root_id: str) -> dict[int, str]:
    labels = {}
    state_index = 0
    components = sorted(nx.connected_components(forest), key=lambda nodes: (root not in nodes, min(nodes)))
    for nodes in components:
        local = forest.subgraph(nodes).copy()
        critical = {int(node) for node in local if local.degree(node) != 2}
        if root in nodes:
            critical.add(int(root))
        for node in sorted(critical):
            labels[node] = f"{root_id}_s{state_index}"
            state_index += 1
        remainder = local.copy()
        remainder.remove_nodes_from(critical)
        for chain in sorted(nx.connected_components(remainder), key=min):
            state_id = f"{root_id}_s{state_index}"
            state_index += 1
            for node in chain:
                labels[int(node)] = state_id
    return labels


def backbone_from_states(
    root_id: str,
    forest: nx.Graph,
    states: dict[int, str],
    tau: dict[int, float],
    names: list[str],
) -> pd.DataFrame:
    best = {}
    for left, right, data in forest.edges(data=True):
        source_state, target_state = states[int(left)], states[int(right)]
        if source_state == target_state:
            continue
        if (float(tau.get(int(left), 0.0)), int(left)) > (float(tau.get(int(right), 0.0)), int(right)):
            left, right = right, left
            source_state, target_state = target_state, source_state
        key = (source_state, target_state)
        trial = (float(data.get("weight", 0.0)), int(left), int(right), dict(data))
        if key not in best or trial[:3] < best[key][:3]:
            best[key] = trial
    rows = []
    for (source_state, target_state), (cost, left, right, data) in sorted(best.items()):
        rows.append({
            "root_id": root_id,
            "source_state": source_state,
            "target_state": target_state,
            "support_source": names[left],
            "support_target": names[right],
            "edge_cost": cost,
            "edge_graph_role": str(data.get("edge_graph_role", "")),
            "connector_only": bool(data.get("connector_only", False)),
        })
    return pd.DataFrame(rows, columns=[
        "root_id", "source_state", "target_state", "support_source", "support_target",
        "edge_cost", "edge_graph_role", "connector_only",
    ])


def dual_scale_construct(
    meta: ad.AnnData,
    geometry,
    full_candidate: pd.DataFrame,
    membership,
    roots: list[dict],
    supports: dict[str, set[int]],
):
    high = full_candidate[full_candidate.edge_graph_role.astype(str).isin(P.T0_ACTIVE_ROLES)].copy()
    high["joint_distance_ecdf"] = P.V1.empirical_rank(high.D_joint.to_numpy(float))
    union = full_candidate[full_candidate.edge_graph_role.astype(str).eq("trajectory_local_union")].copy()
    if len(union):
        union["joint_distance_ecdf"] = P.V1.empirical_rank(union.D_joint.to_numpy(float))
    legal = {
        P.V1.edge_key(int(row.source), int(row.target))
        for row in full_candidate[full_candidate.trajectory_edge_eligible.astype(bool)].itertuples(index=False)
    }
    names = [str(name) for name in meta.obs_names]
    edge_rows, node_rows, backbone_frames, component_rows, model_rows = [], [], [], [], []
    connector_total = 0
    for root in roots:
        root_id = str(root["root_id"])
        root_pos, root_index = int(root["root_pos"]), int(root["root_index"])
        support = supports[root_id]
        high_graph = P.V1.graph_for_support(high, support, membership[:, root_pos])
        preliminary = P.V1.component_forest(high_graph)
        tau_pre, _ = P.V1.rooted_tau(preliminary, root_index, membership[:, root_pos])
        branches, audits = P.C3V2.raw_expression_diagonal_gmm(geometry["expression"], preliminary, tau_pre)
        frozen_high = P.V1.quotient_branch_forest(high_graph, branches)
        union_graph = P.V1.graph_for_support(union, support, membership[:, root_pos]) if len(union) else nx.Graph()
        union_graph.add_nodes_from(support)
        final, connectors = V4.attach_components_with_union(P.V1, frozen_high, union_graph)
        connector_keys = set(connectors)
        for left, right in final.edges():
            final[left][right]["connector_only"] = P.V1.edge_key(int(left), int(right)) in connector_keys
        connector_total += len(connectors)
        tau, component = P.V1.rooted_tau(final, root_index, membership[:, root_pos])
        states = topology_state_labels(final, root_index, root_id)
        backbone_frames.append(backbone_from_states(root_id, final, states, tau, names))
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
                "candidate_legal": P.V1.edge_key(int(left), int(right)) in legal,
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
                "n_connectors": sum(P.V1.edge_key(int(left), int(right)) in connector_keys for left, right in final.subgraph(nodes).edges()),
            })
    return {
        "edges": pd.DataFrame(edge_rows),
        "nodes": pd.DataFrame(node_rows),
        "backbone": pd.concat(backbone_frames, ignore_index=True),
        "components": pd.DataFrame(component_rows),
        "branch_models": pd.DataFrame(model_rows),
        "connector_total": connector_total,
    }


def topological_state_upgrade(result):
    nodes = result["nodes"].copy()
    edges = result["edges"].copy()
    states_all, backbones = {}, []
    for root_id, local_nodes in nodes.groupby("root_id"):
        local_edges = edges[edges.root_id.astype(str).eq(str(root_id))]
        forest = nx.Graph()
        forest.add_nodes_from(local_nodes.node_id.astype(int))
        for row in local_edges.itertuples(index=False):
            forest.add_edge(
                int(row.source), int(row.target), weight=float(row.edge_cost),
                edge_graph_role=str(row.edge_graph_role), connector_only=bool(getattr(row, "connector_only", False)),
            )
        root_index = int(local_nodes[local_nodes.is_root.astype(bool)].iloc[0].node_id)
        states = topology_state_labels(forest, root_index, str(root_id))
        states_all.update({(str(root_id), node): state for node, state in states.items()})
        tau = dict(zip(local_nodes.node_id.astype(int), local_nodes.pseudotime.astype(float)))
        names = {int(row.node_id): str(row.unit_id) for row in local_nodes.itertuples(index=False)}
        name_list = [names.get(index, str(index)) for index in range(max(names) + 1)]
        backbones.append(backbone_from_states(str(root_id), forest, states, tau, name_list))
    nodes["state_id"] = [states_all[(str(row.root_id), int(row.node_id))] for row in nodes.itertuples(index=False)]
    return {"edges": edges, "nodes": nodes, "backbone": pd.concat(backbones, ignore_index=True)}


def mapped_state_backbone(root_id: str, result, truth):
    nodes = result["nodes"]
    nodes = nodes[nodes.root_id.astype(str).eq(root_id)]
    mapping = {}
    for state_id, group in nodes.groupby("state_id"):
        weights = {}
        for unit in group.unit_id.astype(str):
            for truth_state in truth.state_support.get((unit, root_id), set()):
                category = R.coarse_category(truth_state)
                if category:
                    weights[category] = weights.get(category, 0.0) + Q.state_probability(
                        truth, unit, root_id, truth_state
                    )
        mapping[str(state_id)] = min(weights, key=lambda value: (-weights[value], value)) if weights else None
    graph = nx.DiGraph()
    backbone = result["backbone"]
    backbone = backbone[backbone.root_id.astype(str).eq(root_id)]
    for row in backbone.itertuples(index=False):
        source = mapping.get(str(row.source_state))
        target = mapping.get(str(row.target_state))
        if source and target and source != target:
            graph.add_edge(source, target)
    root_rows = nodes[nodes.is_root.astype(bool)]
    root_state = str(root_rows.iloc[0].state_id) if len(root_rows) else None
    return graph, mapping.get(root_state)


def dual_scale_branch_metrics(root_id: str, result, truth):
    true_graph = R.coarse_truth_graph(truth, root_id)
    pred_graph, pred_root = mapped_state_backbone(root_id, result, truth)
    true_bp = {node for node in true_graph if true_graph.out_degree(node) > 1}
    true_term = {node for node in true_graph if true_graph.out_degree(node) == 0}
    pred_bp = {node for node in pred_graph if pred_graph.out_degree(node) > 1}
    pred_term = {node for node in pred_graph if pred_graph.out_degree(node) == 0 and node != pred_root}
    bp_p, bp_r, bp_f1 = P.prf(true_bp, pred_bp)
    term_p, term_r, term_f1 = P.prf(true_term, pred_term)
    ancestor = P.safe_mean([
        float(source in true_graph and target in true_graph and nx.has_path(true_graph, source, target))
        for source, target in pred_graph.edges()
    ])
    errors = []
    for terminal in true_term:
        if pred_root is None or pred_root not in pred_graph or terminal not in pred_graph or not nx.has_path(pred_graph, pred_root, terminal):
            errors.append(1.0)
        else:
            true_distance = nx.shortest_path_length(true_graph, "TRUNK", terminal)
            pred_distance = nx.shortest_path_length(pred_graph, pred_root, terminal)
            errors.append(min(abs(pred_distance - true_distance) / max(true_distance, 1), 1.0))
    root_terminal = 1.0 - P.safe_mean(errors) if errors else math.nan
    prefix = P.shared_prefix_score(true_graph, pred_graph, "TRUNK", pred_root)
    key_nodes = true_bp | true_term
    order = P.safe_mean([
        float(source in pred_graph and target in pred_graph and nx.has_path(pred_graph, source, target))
        for source in key_nodes for target in key_nodes
        if source != target and nx.has_path(true_graph, source, target)
    ])
    return {
        "branchpoint_precision": bp_p, "branchpoint_recall": bp_r, "branchpoint_F1": bp_f1,
        "terminal_precision": term_p, "terminal_recall": term_r, "terminal_F1": term_f1,
        "ancestor_descendant_consistency": ancestor,
        "root_terminal_path_similarity": root_terminal,
        "shared_prefix_recovery": prefix,
        "branching_order_accuracy": order,
    }


def dual_scale_metrics(run_label: str, case: str, result, truth):
    frame = Q.t0_metrics(run_label, case, result, truth)
    roots = frame[~frame.root_id.astype(str).eq("ALL")].copy()
    for root_id in truth.roots:
        values = dual_scale_branch_metrics(root_id, result, truth)
        mask = roots.root_id.astype(str).eq(root_id)
        for column, value in values.items():
            roots.loc[mask, column] = value
        roots.loc[mask, "n_state_backbone_edges"] = len(
            result["backbone"][result["backbone"].root_id.astype(str).eq(root_id)]
        )
    numeric = [column for column in roots.columns if column not in {"run", "case", "root_id", "truth_use", "is_forest"}]
    aggregate = {"run": run_label, "case": case, "root_id": "ALL", "truth_use": "dual_scale_soft_state_audit_only", "is_forest": bool(roots.is_forest.all())}
    aggregate.update({column: P.safe_mean(roots[column]) for column in numeric})
    return pd.concat([roots, pd.DataFrame([aggregate])], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="CORE_05")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir, output = args.run_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    method_dir = run_dir / "supplied_root_inputs" / args.dataset
    meta = ad.read_h5ad(method_dir / "metacell_adata.h5ad")
    geometry = P.V1.prepare_geometry(meta)
    inferred_a, roots, inferred_supports = P.inferred_membership(method_dir, meta.n_obs)
    truth = Q.add_soft_state_truth(run_dir, P.load_truth_context(run_dir))
    true_a, true_supports = P.true_membership(meta, roots, truth)
    current_candidate = P.current_candidate(method_dir)
    full_candidate, _, build_audit = P.new_candidates(meta, geometry)
    current_true = topological_state_upgrade(P.construct_t0(meta, geometry, current_candidate, true_a, roots, true_supports, False))
    current_inferred = topological_state_upgrade(P.construct_t0(meta, geometry, current_candidate, inferred_a, roots, inferred_supports, False))
    new_true = dual_scale_construct(meta, geometry, full_candidate, true_a, roots, true_supports)
    new_inferred = dual_scale_construct(meta, geometry, full_candidate, inferred_a, roots, inferred_supports)
    cases = {
        "A_true_membership_current_T0": current_true,
        "B_inferred_membership_current_T0": current_inferred,
        "C_true_membership_new_T0": new_true,
        "D_inferred_membership_new_T0": new_inferred,
    }
    for name, result in cases.items():
        P.write_case(output, name, result)
    new_inferred["backbone"].to_csv(output / "T0_STATE_BACKBONE_EDGES.csv", index=False)
    new_inferred["edges"].to_csv(output / "T0_FINE_TRAJECTORY_EDGES.csv", index=False)
    new_inferred["nodes"].to_csv(output / "T0_STATE_NODE_ASSIGNMENTS.csv", index=False)
    new_inferred["components"].to_csv(output / "T0_COMPONENT_AUDIT.csv", index=False)
    new_inferred["branch_models"].to_csv(output / "T0_BRANCH_PROGRAM_MODEL_AUDIT.csv", index=False)
    candidate_frame = pd.concat([
        R.directional_candidate_audit(args.run_label, "current_single_scale", current_candidate, truth),
        R.directional_candidate_audit(args.run_label, "new_multiresolution_full", full_candidate, truth),
        R.directional_candidate_audit(args.run_label, "complete_graph_ceiling", R.complete_candidate(meta), truth),
    ], ignore_index=True)
    R.add_ceiling_columns(candidate_frame).to_csv(output / "PHASE_A_CANDIDATE_AUDIT.csv", index=False)
    t0_frame = pd.concat([
        dual_scale_metrics(args.run_label, name, result, truth) for name, result in cases.items()
    ], ignore_index=True)
    t0_frame.to_csv(output / "PHASE_A_T0_METRICS.csv", index=False)
    t0_frame[t0_frame.root_id.astype(str).eq("ALL")].to_csv(output / "PHASE_A_ORACLE_DECOMPOSITION.csv", index=False)
    feasibility = Q.event_feasibility(args.run_label, run_dir, full_candidate, new_inferred, truth)
    feasibility.to_csv(output / "PHASE_A_EVENT_FEASIBILITY.csv", index=False)
    feasibility.to_csv(output / "ORACLE_EVENT_FEASIBILITY_ON_T0.csv", index=False)
    Q.threshold_sensitivity(args.run_label, run_dir, current_candidate, full_candidate).to_csv(output / "STATE_SUPPORT_THRESHOLD_SENSITIVITY.csv", index=False)
    pd.DataFrame([build_audit]).to_csv(output / "CANDIDATE_BUILD_AUDIT.csv", index=False)
    status = {
        "phase": "A", "version": "PHASE_A_C3_dualscale", "parent": "PHASE_A",
        "construction_change": "separate expression branch programs from graph-constrained topology states",
        "run": args.run_label, "dataset": args.dataset,
        "relation_detector_called": False, "refinement_operator_called": False,
        "production_case": "D_inferred_membership_new_T0", "truth_used_for_construction_D": False,
        "n_union_connectors_used": int(new_inferred["connector_total"]),
        "all_fine_edges_candidate_legal": bool(new_inferred["edges"].candidate_legal.all()),
        "all_cases_are_forests": bool(t0_frame.is_forest.all()), "complete": True,
    }
    (output / "RUN_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
