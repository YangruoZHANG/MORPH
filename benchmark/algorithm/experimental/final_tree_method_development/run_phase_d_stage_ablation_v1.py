#!/usr/bin/env python3
"""Truth-scored M0-M6 ablation for the frozen Phase D development method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd

import diagnose_phase_a_hvg_v8 as phase_a
from run_phase_c_oracle_refinement_v1 import (
    as_bool,
    dominant_state_lookup,
    mapped_edge_set,
    prf,
    truth_edge_set,
)


CSV_FLOAT_FORMAT = "%.15g"


def load_result(edge_path: Path, node_path: Path) -> dict[str, object]:
    result = {
        "edges": pd.read_csv(edge_path),
        "nodes": pd.read_csv(node_path),
    }
    upgraded = phase_a.V2.topological_state_upgrade(result)
    upgraded["components"] = pd.DataFrame()
    upgraded["branch_models"] = pd.DataFrame()
    return upgraded


def global_msf_result(
    meta: ad.AnnData,
    candidate: pd.DataFrame,
    roots: list[dict[str, object]],
    membership: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    names = [str(name) for name in meta.obs_names]
    graph = nx.Graph()
    graph.add_nodes_from(range(meta.n_obs))
    for row in candidate.itertuples(index=False):
        graph.add_edge(
            int(row.source), int(row.target), weight=float(row.D_joint),
            edge_graph_role="global_candidate_msf",
        )
    forest = nx.minimum_spanning_tree(graph, weight="weight", algorithm="kruskal")
    root_ids = [str(root["root_id"]) for root in roots]
    anchors = {str(root["root_id"]): int(root["root_index"]) for root in roots}
    assigned = np.full(meta.n_obs, -1, dtype=int)
    for component in nx.connected_components(forest):
        component = set(component)
        local_anchors = [root for root in root_ids if anchors[root] in component]
        if not local_anchors:
            mean_membership = membership[list(component)].mean(axis=0)
            winner = int(np.argmax(mean_membership))
            assigned[list(component)] = winner
            continue
        distances = {
            root: nx.single_source_dijkstra_path_length(
                forest.subgraph(component), anchors[root], weight="weight"
            )
            for root in local_anchors
        }
        for node in component:
            root = min(local_anchors, key=lambda value: (distances[value][node], value))
            assigned[node] = root_ids.index(root)
    for index, root in enumerate(root_ids):
        assigned[anchors[root]] = index
    edge_rows: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []
    for root_pos, root_id in enumerate(root_ids):
        root_nodes = set(np.flatnonzero(assigned == root_pos).astype(int))
        local = nx.Graph()
        local.add_nodes_from(root_nodes)
        for left, right, data in forest.edges(data=True):
            if left in root_nodes and right in root_nodes:
                local.add_edge(left, right, **data)
                edge_rows.append({
                    "root_id": root_id,
                    "source": int(left),
                    "target": int(right),
                    "source_name": names[int(left)],
                    "target_name": names[int(right)],
                    "edge_cost": float(data.get("weight", 0.0)),
                    "edge_graph_role": "global_candidate_msf",
                    "candidate_legal": True,
                    "connector_only": False,
                })
        anchor = anchors[root_id]
        for component_index, component in enumerate(nx.connected_components(local)):
            component = set(component)
            local_anchor = anchor if anchor in component else min(
                component, key=lambda node: (-membership[node, root_pos], node)
            )
            distances = nx.single_source_dijkstra_path_length(
                local.subgraph(component), local_anchor, weight="weight"
            )
            scale = max(distances.values(), default=1.0)
            paths = nx.single_source_shortest_path(local.subgraph(component), local_anchor)
            for node in sorted(component):
                path = paths[node]
                branch = "ROOT" if node == anchor else (
                    f"{root_id}_global_{path[1]}" if len(path) > 1 else f"{root_id}_detached"
                )
                node_rows.append({
                    "root_id": root_id,
                    "node_id": int(node),
                    "unit_id": names[int(node)],
                    "membership": 1.0,
                    "pseudotime": float(distances[node] / max(scale, 1e-12)),
                    "branch_id": branch,
                    "state_id": f"{root_id}_placeholder",
                    "component_id": component_index,
                    "is_root": node == anchor,
                })
    result = phase_a.V2.topological_state_upgrade({
        "edges": pd.DataFrame(edge_rows),
        "nodes": pd.DataFrame(node_rows),
    })
    result["components"] = pd.DataFrame()
    result["branch_models"] = pd.DataFrame()
    one_hot = np.zeros_like(membership)
    one_hot[np.arange(meta.n_obs), assigned] = 1.0
    return result, one_hot


def hard_root_result(
    meta: ad.AnnData,
    geometry,
    candidate: pd.DataFrame,
    roots: list[dict[str, object]],
    membership: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    hard = np.zeros_like(membership)
    hard[np.arange(meta.n_obs), np.argmax(membership, axis=1)] = 1.0
    for root in roots:
        hard[int(root["root_index"]), :] = 0.0
        hard[int(root["root_index"]), int(root["root_pos"])] = 1.0
    supports = {
        str(root["root_id"]): set(np.flatnonzero(hard[:, int(root["root_pos"])] > 0))
        for root in roots
    }
    result = phase_a.V2.P.construct_t0(
        meta, geometry, candidate, hard, roots, supports, False
    )
    upgraded = phase_a.V2.topological_state_upgrade(result)
    upgraded["components"] = result.get("components", pd.DataFrame())
    upgraded["branch_models"] = result.get("branch_models", pd.DataFrame())
    return upgraded, hard


def assignment_metrics(
    predicted: np.ndarray, truth_frame: pd.DataFrame, names: list[str], roots: list[str]
) -> dict[str, float]:
    truth = truth_frame.pivot_table(
        index="metacell_id", columns="root_id", values="root_probability", fill_value=0.0
    ).reindex(index=names, columns=roots, fill_value=0.0).to_numpy(float)
    truth /= np.maximum(truth.sum(axis=1, keepdims=True), 1e-12)
    predicted = predicted / np.maximum(predicted.sum(axis=1, keepdims=True), 1e-12)
    return {
        "root_assignment_accuracy": float(
            np.mean(np.argmax(predicted, axis=1) == np.argmax(truth, axis=1))
        ),
        "root_assignment_brier": float(np.mean(np.square(predicted - truth).sum(axis=1))),
    }


def topology_metrics(
    edges: pd.DataFrame,
    state_lookup: dict[tuple[str, str], str],
    truth_t1: set[tuple[str, str, str]],
) -> dict[str, float]:
    predicted = mapped_edge_set(edges, state_lookup)
    precision, recall, f1 = prf(truth_t1, predicted)
    return {
        "topology_precision": precision,
        "topology_recall": recall,
        "topology_F1": f1,
        "topology_distance_to_true_T1": float(len(predicted ^ truth_t1)),
    }


def event_metrics(run_dir: Path, evidence: pd.DataFrame) -> dict[str, float]:
    truth = pd.read_csv(run_dir / "event" / "truth" / "TRUE_EVENTS.csv")
    causal = truth[truth.causal.map(as_bool)]
    calls = evidence[evidence.is_called.map(as_bool)]
    if len(calls):
        correct = 0
        for call in calls.itertuples(index=False):
            correct += int(any(
                str(row.source_root) == str(call.source_root)
                and str(row.receiver_root) == str(call.receiver_root)
                for row in causal.itertuples(index=False)
            ))
        precision = correct / len(calls)
    else:
        precision = 1.0
        correct = 0
    recall = correct / len(causal) if len(causal) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_F1": float(f1),
        "n_relation_calls": int(len(calls)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase-a-dir", type=Path, required=True)
    parser.add_argument("--phase-d-dir", type=Path, required=True)
    parser.add_argument("--relation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    method_dir = args.run_dir / "generic_upstream_v1_exact_root" / args.dataset
    meta = ad.read_h5ad(method_dir / "metacell_adata.h5ad")
    geometry = phase_a.V2.P.V1.prepare_geometry(meta)
    membership, roots, _ = phase_a.V2.P.inferred_membership(method_dir, meta.n_obs)
    current_candidate = phase_a.V2.P.current_candidate(method_dir)
    truth = phase_a.V2.Q.add_soft_state_truth(
        args.run_dir, phase_a.V2.P.load_truth_context(args.run_dir)
    )
    m0, m0_membership = global_msf_result(meta, current_candidate, roots, membership)
    m1, m1_membership = hard_root_result(
        meta, geometry, current_candidate, roots, membership
    )
    m2 = load_result(
        args.phase_a_dir / "current_root_conditioned_msf" / "T0_FINE_TRAJECTORY_EDGES.csv",
        args.phase_a_dir / "current_root_conditioned_msf" / "T0_NODE_ASSIGNMENTS.csv",
    )
    m3 = load_result(
        args.phase_a_dir / "T0_FINE_TRAJECTORY_EDGES.csv",
        args.phase_a_dir / "T0_STATE_NODE_ASSIGNMENTS.csv",
    )
    m5 = load_result(
        args.phase_d_dir / "M5_T1_FULL_INFERRED_RELATION_REFINED.csv",
        args.phase_d_dir / "M5_TAU1_NODE_ASSIGNMENTS.csv",
    )
    m6 = load_result(
        args.phase_d_dir / "M6_TFINAL_PRUNED.csv",
        args.phase_d_dir / "M6_TFINAL_NODE_ASSIGNMENTS.csv",
    )
    stage_results = {
        "M0": m0, "M1": m1, "M2": m2, "M3": m3,
        "M4": m3, "M5": m5, "M6": m6,
    }
    soft_assignment = membership.copy()
    assignment_by_stage = {
        "M0": m0_membership, "M1": m1_membership,
        "M2": soft_assignment, "M3": soft_assignment,
        "M4": soft_assignment, "M5": soft_assignment, "M6": soft_assignment,
    }
    root_ids = [str(root["root_id"]) for root in roots]
    root_truth = pd.read_csv(
        args.run_dir / "formal_metacell" / "METACELL_ROOT_COMPOSITION.csv"
    )
    state = pd.read_csv(
        args.run_dir / "formal_metacell" / "METACELL_STATE_COMPOSITION.csv"
    )
    state_lookup = dominant_state_lookup(state, "T1")
    truth_t1 = truth_edge_set(
        pd.read_csv(args.run_dir / "event" / "truth" / "TRUE_T1_EDGES.csv")
    )
    evidence = pd.read_csv(args.relation_dir / "PHASE_B_CMI_E_EVIDENCE.csv")
    relation_metrics = event_metrics(args.run_dir, evidence)
    d_metrics = pd.read_csv(
        args.phase_d_dir / "PHASE_D_INFERRED_REFINEMENT_METRICS.csv"
    ).iloc[0]
    rows: list[dict[str, object]] = []
    per_root_frames: list[pd.DataFrame] = []
    for stage, result in stage_results.items():
        scored = phase_a.V2.dual_scale_metrics(
            args.dataset, stage, result, truth
        )
        scored.insert(0, "stage", stage)
        per_root_frames.append(scored)
        aggregate = scored[scored.root_id.astype(str).eq("ALL")].iloc[0].to_dict()
        aggregate.update(assignment_metrics(
            assignment_by_stage[stage], root_truth,
            [str(name) for name in meta.obs_names], root_ids,
        ))
        aggregate.update(topology_metrics(result["edges"], state_lookup, truth_t1))
        if stage in {"M4", "M5", "M6"}:
            aggregate.update(relation_metrics)
        else:
            causal = pd.read_csv(
                args.run_dir / "event" / "truth" / "TRUE_EVENTS.csv"
            ).causal.map(as_bool).any()
            aggregate.update({
                "event_precision": 1.0,
                "event_recall": 0.0 if causal else 1.0,
                "event_F1": 0.0 if causal else 1.0,
                "n_relation_calls": 0,
            })
        aggregate["rewiring_gain"] = (
            0.0 if stage in {"M0", "M1", "M2", "M3", "M4"}
            else float(d_metrics[
                "inferred_topology_gain" if stage == "M5" else "tfinal_topology_gain"
            ])
        )
        aggregate["false_rewire"] = int(
            (not bool(d_metrics.causal_scenario))
            and stage in {"M5", "M6"}
            and int(d_metrics.n_accepted_relation_events) > 0
        )
        rows.append({"dataset": args.dataset, **aggregate})
    pd.DataFrame(rows).to_csv(
        args.output / "PHASE_D_STAGE_ABLATION.csv", index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    pd.concat(per_root_frames, ignore_index=True).to_csv(
        args.output / "PHASE_D_STAGE_ABLATION_PER_ROOT.csv", index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    status = {
        "dataset": args.dataset,
        "stages": list(stage_results),
        "same_truth_scorer": True,
        "truth_used_for_construction": False,
        "truth_used_for_scoring_only": True,
        "complete": True,
    }
    (args.output / "RUN_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
