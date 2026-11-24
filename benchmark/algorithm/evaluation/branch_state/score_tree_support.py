#!/usr/bin/env python3
"""PHASE A v1.1: soft-state truth audit correcting dominant-state collapse."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score


HERE = Path(__file__).resolve().parent
STATE_SUPPORT_THRESHOLD = 0.20
SENSITIVITY_THRESHOLDS = [0.0, 0.2, 0.5, 0.7]


def load_parent():
    name = "phase_a_parent"
    spec = importlib.util.spec_from_file_location(name, HERE / "score_tree_geometry.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P = load_parent()


def add_soft_state_truth(run_dir: Path, truth, threshold: float = STATE_SUPPORT_THRESHOLD):
    frame = pd.read_csv(run_dir / "formal_metacell" / "METACELL_STATE_COMPOSITION.csv")
    frame = frame[frame["stage"].astype(str).eq("T0")]
    truth.state_probability = {
        (str(row.metacell_id), str(row.root_id), str(row.state_id)): float(row.state_probability)
        for row in frame.itertuples(index=False)
    }
    truth.state_support = {}
    for (metacell, root), group in frame.groupby(["metacell_id", "root_id"]):
        supported = {
            str(row.state_id) for row in group.itertuples(index=False)
            if float(row.state_probability) >= threshold
        }
        if supported:
            truth.state_support[(str(metacell), str(root))] = supported
    truth.state_support_threshold = threshold
    return truth


def state_probability(truth, metacell: str, root: str, state: str) -> float:
    return float(truth.state_probability.get((str(metacell), str(root), str(state)), 0.0))


def supported_nodes(truth, root: str, state: str) -> set[str]:
    return {
        metacell for (metacell, local_root), states in truth.state_support.items()
        if local_root == root and state in states
    }


def plausible_pairs(truth, root: str, left: str, right: str):
    left_states = truth.state_support.get((str(left), root), set())
    right_states = truth.state_support.get((str(right), root), set())
    pairs = []
    for source in left_states:
        for target in right_states:
            if source == target:
                continue
            weight = state_probability(truth, left, root, source) * state_probability(truth, right, root, target)
            pairs.append((float(weight), str(source), str(target)))
    return sorted(pairs, key=lambda value: (-value[0], value[1], value[2]))


def transition_path_coverage(graph: nx.Graph, root_id: str, truth):
    directed = P.orient_by_truth(graph, root_id, truth)
    truth_graph = truth.graphs[root_id]
    covered = {}
    for source, target in truth_graph.edges():
        source_nodes = supported_nodes(truth, root_id, str(source)) & set(directed.nodes())
        target_nodes = supported_nodes(truth, root_id, str(target)) & set(directed.nodes())
        local = directed.subgraph(source_nodes | target_nodes)
        covered[(str(source), str(target))] = bool(source_nodes and target_nodes and any(
            nx.has_path(local, left, right) for left in source_nodes for right in target_nodes
        ))
    terminals = [node for node in truth_graph if truth_graph.out_degree(node) == 0]
    roots = [node for node in truth_graph if truth_graph.in_degree(node) == 0]
    path_scores = []
    for source in roots:
        for terminal in terminals:
            if nx.has_path(truth_graph, source, terminal):
                path = nx.shortest_path(truth_graph, source, terminal)
                path_scores.append(P.safe_mean([covered.get((str(a), str(b)), False) for a, b in zip(path[:-1], path[1:])]))
    return P.safe_mean(covered.values()), P.safe_mean(path_scores), covered


def candidate_audit(run_label: str, candidate_name: str, candidate: pd.DataFrame, truth) -> pd.DataFrame:
    eligible = candidate[candidate["trajectory_edge_eligible"].astype(bool)]
    graph = nx.Graph()
    graph.add_edges_from((str(row.source_name), str(row.target_name)) for row in eligible.itertuples(index=False))
    all_nodes = {key[0] for key in truth.state_support}
    rows = []
    for root_id in truth.roots:
        truth_graph = truth.graphs[root_id]
        root_nodes = {name for (name, root) in truth.state_support if root == root_id}
        anchor = truth.anchors[root_id]
        local = graph.subgraph(root_nodes | {anchor}).copy()
        reachable = nx.node_connected_component(local, anchor) if anchor in local else set()
        terminals = {str(node) for node in truth_graph if truth_graph.out_degree(node) == 0}
        branchpoints = {str(node) for node in truth_graph if truth_graph.out_degree(node) > 1}
        terminal_nodes = set().union(*(supported_nodes(truth, root_id, state) for state in terminals)) if terminals else set()
        branchpoint_nodes = set().union(*(supported_nodes(truth, root_id, state) for state in branchpoints)) if branchpoints else set()
        represented_pairs = set()
        best_pairs = []
        for left, right in local.edges():
            options = plausible_pairs(truth, root_id, str(left), str(right))
            represented_pairs.update(P.pair(source, target) for _, source, target in options)
            if options:
                best_pairs.append((options[0][1], options[0][2]))
        true_pairs = {P.pair(source, target) for source, target in truth_graph.edges()}
        transition_coverage, path_coverage, _ = transition_path_coverage(local, root_id, truth)
        undirected = truth_graph.to_undirected()
        shortcuts = [
            nx.shortest_path_length(undirected, source, target) > 1
            for source, target in best_pairs if source in undirected and target in undirected and nx.has_path(undirected, source, target)
        ]
        cross_branch = [
            not nx.has_path(truth_graph, source, target) and not nx.has_path(truth_graph, target, source)
            for source, target in best_pairs if source in truth_graph and target in truth_graph
        ]
        rows.append({
            "run": run_label,
            "candidate": candidate_name,
            "root_id": root_id,
            "state_support_threshold": truth.state_support_threshold,
            "candidate_true_edge_recall": len(represented_pairs & true_pairs) / max(len(true_pairs), 1),
            "candidate_transition_path_coverage": transition_coverage,
            "candidate_true_path_coverage": path_coverage,
            "candidate_root_reachable_fraction": len(reachable & root_nodes) / max(len(root_nodes), 1),
            "candidate_terminal_reachable_fraction": len(reachable & terminal_nodes) / max(len(terminal_nodes), 1),
            "candidate_branchpoint_reachable_fraction": len(reachable & branchpoint_nodes) / max(len(branchpoint_nodes), 1),
            "candidate_shortcut_rate": P.safe_mean(shortcuts),
            "candidate_cross_branch_false_connection_rate": P.safe_mean(cross_branch),
            "n_candidate_nodes": local.number_of_nodes(),
            "n_candidate_edges": local.number_of_edges(),
            "n_projectable_cross_state_edges": len(best_pairs),
            "n_unsupported_possible_pairs": max(len(all_nodes) * (len(all_nodes) - 1) // 2 - graph.number_of_edges(), 0),
            "truth_use": "soft_state_audit_only",
        })
    numeric = [column for column in rows[0] if column not in {"run", "candidate", "root_id", "truth_use"}]
    aggregate = {"run": run_label, "candidate": candidate_name, "root_id": "ALL", "truth_use": "soft_state_audit_only"}
    aggregate.update({column: P.safe_mean([row[column] for row in rows]) for column in numeric})
    return pd.DataFrame(rows + [aggregate])


def mapped_backbone_graph(root_id: str, result, truth):
    nodes = result["nodes"]
    nodes = nodes[nodes["root_id"].astype(str).eq(root_id)]
    predicted_to_true = {}
    for branch_id, group in nodes.groupby("branch_id"):
        weights = {}
        for unit in group["unit_id"].astype(str):
            for state in truth.state_support.get((unit, root_id), set()):
                weights[state] = weights.get(state, 0.0) + state_probability(truth, unit, root_id, state)
        predicted_to_true[str(branch_id)] = min(weights, key=lambda state: (-weights[state], state)) if weights else None
    graph = nx.DiGraph()
    backbone = result["backbone"]
    backbone = backbone[backbone["root_id"].astype(str).eq(root_id)] if len(backbone) else backbone
    prefix = f"{root_id}_"
    for row in backbone.itertuples(index=False):
        source = predicted_to_true.get(prefix + str(row.source_state))
        target = predicted_to_true.get(prefix + str(row.target_state))
        if source and target and source != target:
            graph.add_edge(source, target)
    root_rows = nodes[nodes["is_root"].astype(bool)]
    root_branch = str(root_rows.iloc[0]["branch_id"]) if len(root_rows) else None
    return graph, predicted_to_true, predicted_to_true.get(root_branch)


def t0_metrics(run_label: str, case: str, result, truth) -> pd.DataFrame:
    rows = []
    for root_id in truth.roots:
        edges = result["edges"][result["edges"]["root_id"].astype(str).eq(root_id)]
        nodes = result["nodes"][result["nodes"]["root_id"].astype(str).eq(root_id)]
        fine = nx.Graph()
        fine.add_nodes_from(nodes["unit_id"].astype(str))
        fine.add_edges_from((str(row.source_name), str(row.target_name)) for row in edges.itertuples(index=False))
        truth_graph = truth.graphs[root_id]
        true_pairs = {P.pair(a, b) for a, b in truth_graph.edges()}
        pred_pairs, best_pairs = set(), []
        for left, right in fine.edges():
            options = plausible_pairs(truth, root_id, left, right)
            pred_pairs.update(P.pair(source, target) for _, source, target in options)
            if options:
                best_pairs.append((options[0][1], options[0][2]))
        edge_p, edge_r, edge_f1 = P.prf(true_pairs, pred_pairs)
        _, path_coverage, _ = transition_path_coverage(fine, root_id, truth)
        undirected = truth_graph.to_undirected()
        shortcut = P.safe_mean([
            nx.shortest_path_length(undirected, source, target) > 1
            for source, target in best_pairs if source in undirected and target in undirected and nx.has_path(undirected, source, target)
        ])
        branch_pairs = [
            (truth.branch.get((str(row.unit_id), root_id)), str(row.branch_id))
            for row in nodes.itertuples(index=False) if truth.branch.get((str(row.unit_id), root_id)) is not None
        ]
        branch_ari = adjusted_rand_score([x[0] for x in branch_pairs], [x[1] for x in branch_pairs]) if len(branch_pairs) >= 2 else math.nan
        mapped, _, mapped_root = mapped_backbone_graph(root_id, result, truth)
        true_bp = {node for node in truth_graph if truth_graph.out_degree(node) > 1}
        true_term = {node for node in truth_graph if truth_graph.out_degree(node) == 0}
        pred_bp = {node for node in mapped if mapped.out_degree(node) > 1}
        pred_term = {node for node in mapped if mapped.out_degree(node) == 0 and node != mapped_root}
        bp_p, bp_r, bp_f1 = P.prf(true_bp, pred_bp)
        term_p, term_r, term_f1 = P.prf(true_term, pred_term)
        ancestor = P.safe_mean([
            float(source in truth_graph and target in truth_graph and nx.has_path(truth_graph, source, target))
            for source, target in mapped.edges()
        ])
        true_roots = [node for node in truth_graph if truth_graph.in_degree(node) == 0]
        true_root = true_roots[0] if true_roots else None
        errors = []
        if true_root:
            for terminal in true_term:
                true_distance = nx.shortest_path_length(truth_graph, true_root, terminal)
                if mapped_root is None or mapped_root not in mapped or terminal not in mapped or not nx.has_path(mapped, mapped_root, terminal):
                    errors.append(1.0)
                else:
                    pred_distance = nx.shortest_path_length(mapped, mapped_root, terminal)
                    errors.append(min(abs(pred_distance - true_distance) / max(true_distance, 1), 1.0))
        root_terminal = 1.0 - P.safe_mean(errors) if errors else math.nan
        prefix = P.shared_prefix_score(truth_graph, mapped, true_root, mapped_root) if true_root else math.nan
        keys = true_bp | true_term
        order = P.safe_mean([
            float(source in mapped and target in mapped and nx.has_path(mapped, source, target))
            for source in keys for target in keys if source != target and nx.has_path(truth_graph, source, target)
        ])
        pred_pt = dict(zip(nodes["unit_id"].astype(str), nodes["pseudotime"].astype(float)))
        pt_pairs = [(truth.pseudotime.get((name, root_id)), value) for name, value in pred_pt.items()]
        pt_pairs = [(a, b) for a, b in pt_pairs if a is not None and np.isfinite(a)]
        pt_sr = float(spearmanr([x[0] for x in pt_pairs], [x[1] for x in pt_pairs]).statistic) if len(pt_pairs) >= 3 else math.nan
        anchor = truth.anchors[root_id]
        reachable = nx.node_connected_component(fine, anchor) if anchor in fine else set()
        lengths = nx.single_source_shortest_path_length(fine.subgraph(reachable), anchor) if anchor in fine else {}
        distance_pairs = [(truth.pseudotime.get((name, root_id)), distance) for name, distance in lengths.items()]
        distance_pairs = [(a, b) for a, b in distance_pairs if a is not None and np.isfinite(a)]
        distance_sr = float(spearmanr([x[0] for x in distance_pairs], [x[1] for x in distance_pairs]).statistic) if len(distance_pairs) >= 3 else math.nan
        rows.append({
            "run": run_label, "case": case, "root_id": root_id,
            "state_support_threshold": truth.state_support_threshold,
            "fine_edge_precision": edge_p, "fine_edge_recall": edge_r, "fine_edge_F1": edge_f1,
            "fine_true_path_coverage": path_coverage, "fine_shortcut_rate": shortcut,
            "branch_ARI_rootwise": branch_ari,
            "branchpoint_precision": bp_p, "branchpoint_recall": bp_r, "branchpoint_F1": bp_f1,
            "terminal_precision": term_p, "terminal_recall": term_r, "terminal_F1": term_f1,
            "ancestor_descendant_consistency": ancestor,
            "root_terminal_path_similarity": root_terminal,
            "shared_prefix_recovery": prefix, "branching_order_accuracy": order,
            "pseudotime_spearman": pt_sr, "root_distance_consistency": distance_sr,
            "n_fine_edges": fine.number_of_edges(),
            "n_state_backbone_edges": int(len(result["backbone"][result["backbone"]["root_id"].astype(str).eq(root_id)])) if len(result["backbone"]) else 0,
            "n_components": nx.number_connected_components(fine) if len(fine) else 0,
            "is_forest": nx.is_forest(fine), "truth_use": "soft_state_audit_only",
        })
    numeric = [column for column in rows[0] if column not in {"run", "case", "root_id", "truth_use", "is_forest"}]
    aggregate = {"run": run_label, "case": case, "root_id": "ALL", "truth_use": "soft_state_audit_only", "is_forest": all(row["is_forest"] for row in rows)}
    aggregate.update({column: P.safe_mean([row[column] for row in rows]) for column in numeric})
    return pd.DataFrame(rows + [aggregate])


def edge_supports_pair(row, root_id: str, target_pair: tuple[str, str], truth) -> bool:
    return any(P.pair(source, target) == target_pair for _, source, target in plausible_pairs(
        truth, root_id, str(row.source_name), str(row.target_name)
    ))


def event_feasibility(run_label: str, run_dir: Path, candidate: pd.DataFrame, result, truth) -> pd.DataFrame:
    truth_dir = run_dir / "event" / "truth"
    events = pd.read_csv(truth_dir / "TRUE_EVENTS.csv")
    added = pd.read_csv(truth_dir / "TRUE_ADDED_EDGES.csv")
    removed = pd.read_csv(truth_dir / "TRUE_REMOVED_EDGES.csv")
    true_t1 = pd.read_csv(truth_dir / "TRUE_T1_EDGES.csv")
    eligible = candidate[candidate["trajectory_edge_eligible"].astype(bool)]
    rows = []
    for event in events.itertuples(index=False):
        event_id, root_id = str(event.event_id), str(event.receiver_root)
        add_pairs = {P.pair(row.source, row.target) for row in added[added.event_id.astype(str).eq(event_id)].itertuples(index=False)}
        remove_pairs = {P.pair(row.source, row.target) for row in removed[removed.event_id.astype(str).eq(event_id)].itertuples(index=False)}
        candidate_edges = [
            (str(row.source_name), str(row.target_name)) for row in eligible.itertuples(index=False)
            if any(edge_supports_pair(row, root_id, target, truth) for target in add_pairs)
        ]
        local_edges = result["edges"][result["edges"]["root_id"].astype(str).eq(root_id)]
        removable_edges = [
            (str(row.source_name), str(row.target_name)) for row in local_edges.itertuples(index=False)
            if any(edge_supports_pair(row, root_id, target, truth) for target in remove_pairs)
        ]
        graph = nx.Graph()
        graph.add_edges_from((str(row.source_name), str(row.target_name)) for row in local_edges.itertuples(index=False))
        legal = 0
        for remove_edge in removable_edges:
            for add_edge in candidate_edges:
                trial = graph.copy()
                trial.remove_edge(*remove_edge)
                trial.add_edge(*add_edge)
                legal += int(nx.is_forest(trial))
        represented = set()
        for row in eligible.itertuples(index=False):
            represented.update(P.pair(source, target) for _, source, target in plausible_pairs(
                truth, root_id, str(row.source_name), str(row.target_name)
            ))
        required = {P.pair(row.source, row.target) for row in true_t1[true_t1.root_id.astype(str).eq(root_id)].itertuples(index=False)}
        rows.append({
            "run": run_label, "event_id": event_id, "event_type": str(event.event_type),
            "receiver_root": root_id, "topology_changed": bool(event.topology_changed),
            "state_support_threshold": truth.state_support_threshold,
            "n_truth_added_state_edges": len(add_pairs),
            "n_candidate_added_edge_realizations": len(candidate_edges),
            "n_T0_removable_edge_realizations": len(removable_edges),
            "n_strict_legal_downstream_swaps": legal,
            "strict_legal_alternative_exists": legal > 0,
            "truth_T1_state_edge_coverage": len(required & represented) / max(len(required), 1),
            "all_truth_T1_state_edges_representable": required <= represented,
            "truth_use": "soft_state_oracle_audit_only_no_refinement_performed",
        })
    return pd.DataFrame(rows)


def threshold_sensitivity(run_label: str, run_dir: Path, current, new_full) -> pd.DataFrame:
    rows = []
    for threshold in SENSITIVITY_THRESHOLDS:
        truth = add_soft_state_truth(run_dir, P.load_truth_context(run_dir), threshold)
        for name, candidate in [("current_single_scale", current), ("new_multiresolution_full", new_full)]:
            frame = candidate_audit(run_label, name, candidate, truth)
            row = frame[frame.root_id.astype(str).eq("ALL")].iloc[0].to_dict()
            rows.append(row)
    return pd.DataFrame(rows)


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
    truth = add_soft_state_truth(run_dir, P.load_truth_context(run_dir))
    true_a, true_supports = P.true_membership(meta, roots, truth)
    current = P.current_candidate(method_dir)
    new_full, new_active, build_audit = P.new_candidates(meta, geometry)
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
        candidate_audit(args.run_label, "current_single_scale", current, truth),
        candidate_audit(args.run_label, "new_multiresolution_full", new_full, truth),
    ], ignore_index=True)
    candidate_frame.to_csv(output / "PHASE_A_CANDIDATE_AUDIT.csv", index=False)
    t0_frame = pd.concat([t0_metrics(args.run_label, name, result, truth) for name, result in cases.items()], ignore_index=True)
    t0_frame.to_csv(output / "PHASE_A_T0_METRICS.csv", index=False)
    t0_frame[t0_frame.root_id.astype(str).eq("ALL")].to_csv(output / "PHASE_A_ORACLE_DECOMPOSITION.csv", index=False)
    feasibility = event_feasibility(args.run_label, run_dir, new_full, selected, truth)
    feasibility.to_csv(output / "PHASE_A_EVENT_FEASIBILITY.csv", index=False)
    feasibility.to_csv(output / "ORACLE_EVENT_FEASIBILITY_ON_T0.csv", index=False)
    threshold_sensitivity(args.run_label, run_dir, current, new_full).to_csv(output / "STATE_SUPPORT_THRESHOLD_SENSITIVITY.csv", index=False)
    pd.DataFrame([build_audit]).to_csv(output / "CANDIDATE_BUILD_AUDIT.csv", index=False)
    status = {
        "phase": "A", "version": "PHASE_A", "parent": "PHASE_A",
        "failure_corrected": "C10 dominant-state projection removed valid mixed-state support",
        "run": args.run_label, "dataset": args.dataset,
        "state_support_threshold": STATE_SUPPORT_THRESHOLD,
        "relation_detector_called": False, "refinement_operator_called": False,
        "production_case": "D_inferred_membership_new_T0",
        "truth_used_for_construction_D": False,
        "all_fine_edges_candidate_legal": bool(selected["edges"]["candidate_legal"].all()),
        "all_cases_are_forests": bool(t0_frame["is_forest"].all()), "complete": True,
    }
    (output / "RUN_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
