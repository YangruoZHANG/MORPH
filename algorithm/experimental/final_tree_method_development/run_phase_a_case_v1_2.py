#!/usr/bin/env python3
"""PHASE A v1.2: ceiling-aware directional and branch-scale audits."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import re
import sys
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def load_version(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = load_version("run_phase_a_case_v1.py", "phase_a_v1_for_v12")
Q = load_version("run_phase_a_case_v1_1.py", "phase_a_v11_for_v12")


def complete_candidate(meta: ad.AnnData) -> pd.DataFrame:
    rows = []
    for source, target in itertools.combinations(range(meta.n_obs), 2):
        rows.append({
            "source": source,
            "target": target,
            "source_name": str(meta.obs_names[source]),
            "target_name": str(meta.obs_names[target]),
            "trajectory_edge_eligible": True,
            "edge_graph_role": "oracle_complete_graph",
        })
    return pd.DataFrame(rows)


def directional_candidate_audit(run_label: str, name: str, candidate: pd.DataFrame, truth) -> pd.DataFrame:
    frame = Q.candidate_audit(run_label, name, candidate, truth)
    eligible = candidate[candidate["trajectory_edge_eligible"].astype(bool)]
    global_graph = nx.Graph()
    global_graph.add_edges_from((str(row.source_name), str(row.target_name)) for row in eligible.itertuples(index=False))
    for root_id in truth.roots:
        root_nodes = {node for (node, root) in truth.state_support if root == root_id}
        anchor = truth.anchors[root_id]
        local = global_graph.subgraph(root_nodes | {anchor}).copy()
        directed = P.orient_by_truth(local, root_id, truth)
        reachable = nx.descendants(directed, anchor) | {anchor} if anchor in directed else set()
        truth_graph = truth.graphs[root_id]
        terminal_states = {str(node) for node in truth_graph if truth_graph.out_degree(node) == 0}
        branchpoint_states = {str(node) for node in truth_graph if truth_graph.out_degree(node) > 1}
        terminal_nodes = set().union(*(Q.supported_nodes(truth, root_id, state) for state in terminal_states)) if terminal_states else set()
        branchpoint_nodes = set().union(*(Q.supported_nodes(truth, root_id, state) for state in branchpoint_states)) if branchpoint_states else set()
        mask = frame["root_id"].astype(str).eq(root_id)
        frame.loc[mask, "candidate_root_reachable_fraction"] = len(reachable & root_nodes) / max(len(root_nodes), 1)
        frame.loc[mask, "candidate_terminal_reachable_fraction"] = len(reachable & terminal_nodes) / max(len(terminal_nodes), 1)
        frame.loc[mask, "candidate_branchpoint_reachable_fraction"] = len(reachable & branchpoint_nodes) / max(len(branchpoint_nodes), 1)
    metric_columns = [
        "candidate_root_reachable_fraction",
        "candidate_terminal_reachable_fraction",
        "candidate_branchpoint_reachable_fraction",
    ]
    for column in metric_columns:
        frame.loc[frame.root_id.astype(str).eq("ALL"), column] = P.safe_mean(
            frame.loc[~frame.root_id.astype(str).eq("ALL"), column]
        )
    frame["reachability_orientation"] = "truth_pseudotime_audit_only"
    return frame


def coarse_category(state: str) -> str | None:
    state = str(state)
    if state.endswith("_ROOT") or state.endswith("_T1"):
        return "TRUNK"
    match = re.search(r"_(B[^_]+)_S", state)
    return match.group(1) if match else None


def coarse_truth_graph(truth, root_id: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("TRUNK")
    categories = {
        coarse_category(state)
        for (metacell, root), states in truth.state_support.items() if root == root_id
        for state in states
    }
    for category in sorted(value for value in categories if value and value != "TRUNK"):
        graph.add_edge("TRUNK", category)
    return graph


def mapped_coarse_backbone(root_id: str, result, truth):
    nodes = result["nodes"]
    nodes = nodes[nodes.root_id.astype(str).eq(root_id)]
    mapping = {}
    for branch_id, group in nodes.groupby("branch_id"):
        weights = {}
        for unit in group.unit_id.astype(str):
            for state in truth.state_support.get((unit, root_id), set()):
                category = coarse_category(state)
                if category:
                    weights[category] = weights.get(category, 0.0) + Q.state_probability(truth, unit, root_id, state)
        mapping[str(branch_id)] = min(weights, key=lambda value: (-weights[value], value)) if weights else None
    graph = nx.DiGraph()
    backbone = result["backbone"]
    backbone = backbone[backbone.root_id.astype(str).eq(root_id)] if len(backbone) else backbone
    prefix = f"{root_id}_"
    for row in backbone.itertuples(index=False):
        source = mapping.get(prefix + str(row.source_state))
        target = mapping.get(prefix + str(row.target_state))
        if source and target and source != target:
            graph.add_edge(source, target)
    root_rows = nodes[nodes.is_root.astype(bool)]
    root_branch = str(root_rows.iloc[0].branch_id) if len(root_rows) else None
    return graph, mapping.get(root_branch)


def coarse_branch_metrics(root_id: str, result, truth) -> dict[str, float]:
    true_graph = coarse_truth_graph(truth, root_id)
    pred_graph, pred_root = mapped_coarse_backbone(root_id, result, truth)
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
        true_distance = nx.shortest_path_length(true_graph, "TRUNK", terminal)
        if pred_root is None or pred_root not in pred_graph or terminal not in pred_graph or not nx.has_path(pred_graph, pred_root, terminal):
            errors.append(1.0)
        else:
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
        "branchpoint_precision": bp_p,
        "branchpoint_recall": bp_r,
        "branchpoint_F1": bp_f1,
        "terminal_precision": term_p,
        "terminal_recall": term_r,
        "terminal_F1": term_f1,
        "ancestor_descendant_consistency": ancestor,
        "root_terminal_path_similarity": root_terminal,
        "shared_prefix_recovery": prefix,
        "branching_order_accuracy": order,
    }


def branch_scale_t0_metrics(run_label: str, case: str, result, truth) -> pd.DataFrame:
    frame = Q.t0_metrics(run_label, case, result, truth)
    root_frame = frame[~frame.root_id.astype(str).eq("ALL")].copy()
    for root_id in truth.roots:
        values = coarse_branch_metrics(root_id, result, truth)
        mask = root_frame.root_id.astype(str).eq(root_id)
        for column, value in values.items():
            root_frame.loc[mask, column] = value
    numeric = [column for column in root_frame.columns if column not in {"run", "case", "root_id", "truth_use", "is_forest", "reachability_orientation"}]
    aggregate = {"run": run_label, "case": case, "root_id": "ALL", "truth_use": "soft_state_branch_scale_audit_only", "is_forest": bool(root_frame.is_forest.all())}
    aggregate.update({column: P.safe_mean(root_frame[column]) for column in numeric})
    return pd.concat([root_frame, pd.DataFrame([aggregate])], ignore_index=True)


def add_ceiling_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    metrics = [
        "candidate_true_edge_recall", "candidate_true_path_coverage",
        "candidate_root_reachable_fraction", "candidate_terminal_reachable_fraction",
        "candidate_branchpoint_reachable_fraction",
    ]
    for (run, root_id), group in result.groupby(["run", "root_id"]):
        ceiling_rows = group[group.candidate.astype(str).eq("complete_graph_ceiling")]
        current_rows = group[group.candidate.astype(str).eq("current_single_scale")]
        if len(ceiling_rows) != 1:
            continue
        ceiling = ceiling_rows.iloc[0]
        current = current_rows.iloc[0] if len(current_rows) == 1 else None
        mask = result.run.astype(str).eq(str(run)) & result.root_id.astype(str).eq(str(root_id))
        for metric in metrics:
            denominator = float(ceiling[metric])
            result.loc[mask, f"{metric}_fraction_of_ceiling"] = np.divide(
                result.loc[mask, metric].astype(float), denominator
            ) if denominator > 0 else math.nan
            if current is not None and denominator > float(current[metric]) + 1e-12:
                result.loc[mask, f"{metric}_recoverable_gap_closed"] = (
                    result.loc[mask, metric].astype(float) - float(current[metric])
                ) / (denominator - float(current[metric]))
            else:
                result.loc[mask, f"{metric}_recoverable_gap_closed"] = math.nan
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="CORE_05")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir, output = args.run_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    method_dir = run_dir / "generic_upstream_v1_exact_root" / args.dataset
    meta = ad.read_h5ad(method_dir / "metacell_adata.h5ad")
    geometry = P.V1.prepare_geometry(meta)
    inferred_a, roots, inferred_supports = P.inferred_membership(method_dir, meta.n_obs)
    truth = Q.add_soft_state_truth(run_dir, P.load_truth_context(run_dir))
    true_a, true_supports = P.true_membership(meta, roots, truth)
    current = P.current_candidate(method_dir)
    new_full, new_active, build_audit = P.new_candidates(meta, geometry)
    complete = complete_candidate(meta)
    cases = {
        "A_true_membership_current_T0": P.construct_t0(meta, geometry, current, true_a, roots, true_supports, False),
        "B_inferred_membership_current_T0": P.construct_t0(meta, geometry, current, inferred_a, roots, inferred_supports, False),
        "C_true_membership_new_T0": P.construct_t0(meta, geometry, new_active, true_a, roots, true_supports, True),
        "D_inferred_membership_new_T0": P.construct_t0(meta, geometry, new_active, inferred_a, roots, inferred_supports, True),
    }
    for name, result in cases.items():
        P.write_case(output, name, result)
    selected = cases["D_inferred_membership_new_T0"]
    selected["backbone"].to_csv(output / "T0_STATE_BACKBONE_EDGES.csv", index=False)
    selected["edges"].to_csv(output / "T0_FINE_TRAJECTORY_EDGES.csv", index=False)
    candidate_frame = pd.concat([
        directional_candidate_audit(args.run_label, "current_single_scale", current, truth),
        directional_candidate_audit(args.run_label, "new_multiresolution_full", new_full, truth),
        directional_candidate_audit(args.run_label, "complete_graph_ceiling", complete, truth),
    ], ignore_index=True)
    candidate_frame = add_ceiling_columns(candidate_frame)
    candidate_frame.to_csv(output / "PHASE_A_CANDIDATE_AUDIT.csv", index=False)
    t0_frame = pd.concat([
        branch_scale_t0_metrics(args.run_label, name, result, truth) for name, result in cases.items()
    ], ignore_index=True)
    t0_frame.to_csv(output / "PHASE_A_T0_METRICS.csv", index=False)
    t0_frame[t0_frame.root_id.astype(str).eq("ALL")].to_csv(output / "PHASE_A_ORACLE_DECOMPOSITION.csv", index=False)
    feasibility = Q.event_feasibility(args.run_label, run_dir, new_full, selected, truth)
    feasibility.to_csv(output / "PHASE_A_EVENT_FEASIBILITY.csv", index=False)
    feasibility.to_csv(output / "ORACLE_EVENT_FEASIBILITY_ON_T0.csv", index=False)
    Q.threshold_sensitivity(args.run_label, run_dir, current, new_full).to_csv(output / "STATE_SUPPORT_THRESHOLD_SENSITIVITY.csv", index=False)
    pd.DataFrame([build_audit]).to_csv(output / "CANDIDATE_BUILD_AUDIT.csv", index=False)
    status = {
        "phase": "A", "version": "PHASE_A_v1_2", "parent": "PHASE_A_v1_1",
        "evaluation_corrections": ["monotonic_directional_reachability", "complete_graph_ceiling", "coarse_truth_branch_skeleton"],
        "construction_changed_from_v1_1": False,
        "run": args.run_label, "dataset": args.dataset,
        "relation_detector_called": False, "refinement_operator_called": False,
        "production_case": "D_inferred_membership_new_T0", "truth_used_for_construction_D": False,
        "all_fine_edges_candidate_legal": bool(selected["edges"]["candidate_legal"].all()),
        "all_cases_are_forests": bool(t0_frame.is_forest.all()), "complete": True,
    }
    (output / "RUN_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
