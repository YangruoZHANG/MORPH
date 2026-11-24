#!/usr/bin/env python3
"""PHASE C v4: oracle event refinement on the frozen state backbone."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load("run_phase_c_oracle_refinement_v3.py", "phase_c_v3_helpers")
GAMMAS = BASE.GAMMAS
STATE_SUPPORT = BASE.STATE_SUPPORT
EPS = 1e-12


def candidate_universe(run_dir: Path, dataset: str, phase_a: Path) -> pd.DataFrame:
    full = pd.read_csv(phase_a / "C1_MULTIRESOLUTION_CANDIDATE_EDGES.csv")
    full = full[full.trajectory_edge_eligible.astype(bool)].copy()
    full["root_scope"] = "*"
    existing = pd.read_csv(
        run_dir / "generic_upstream_v1_exact_root" / dataset / "trajectory_candidate_edges.csv"
    )
    existing = existing[
        existing.trajectory_edge_eligible.astype(str).str.lower().isin({"true", "1", "yes"})
    ].copy()
    existing["D_expression"] = 1.0 - existing.S_expr.astype(float)
    existing["D_spatial"] = 1.0 - existing.S_spatial.astype(float)
    existing["D_joint"] = existing.d_global.astype(float)
    existing["edge_graph_role"] = "trajectory_current_single_scale"
    existing["root_scope"] = "*"
    connector_file = phase_a / "ROOT_SPECIFIC_CONNECTOR_EDGES.csv"
    connectors = pd.DataFrame()
    if connector_file.exists() and connector_file.stat().st_size > 1:
        try:
            connectors = pd.read_csv(connector_file)
        except pd.errors.EmptyDataError:
            connectors = pd.DataFrame()
    if len(connectors):
        connectors["root_scope"] = connectors.root_id.astype(str)
        connectors["edge_graph_role"] = "trajectory_guarded_component_connector"
    columns = [
        "source", "target", "source_name", "target_name", "D_expression",
        "D_spatial", "D_joint",
        "edge_graph_role", "root_scope",
    ]
    combined = pd.concat([
        full[columns], existing[columns], connectors[columns] if len(connectors) else pd.DataFrame(columns=columns)
    ], ignore_index=True)
    combined["key"] = [BASE.edge_key(row.source, row.target) for row in combined.itertuples()]
    return combined.sort_values(
        ["D_joint", "source", "target"], kind="mergesort"
    ).drop_duplicates(["root_scope", "key"])


def backbone_edge_frame(backbone: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    ids = dict(zip(nodes.unit_id.astype(str), nodes.node_id.astype(int)))
    rows = []
    for row in backbone.itertuples(index=False):
        rows.append({
            "root_id": str(row.root_id), "source": ids[str(row.support_source)],
            "target": ids[str(row.support_target)],
            "source_name": str(row.support_source), "target_name": str(row.support_target),
            "edge_cost": float(row.edge_cost), "source_state": str(row.source_state),
            "target_state": str(row.target_state),
            "edge_graph_role": str(row.edge_graph_role),
            "candidate_legal": True, "connector_only": bool(row.connector_only),
        })
    return pd.DataFrame(rows)


def state_candidates(
    universe: pd.DataFrame, node_part: pd.DataFrame, root_id: str,
) -> pd.DataFrame:
    state = dict(zip(node_part.node_id.astype(int), node_part.state_id.astype(str)))
    membership = dict(zip(node_part.node_id.astype(int), node_part.membership.astype(float)))
    local = universe[
        universe.root_scope.astype(str).isin({"*", root_id})
        & universe.source.astype(int).isin(state)
        & universe.target.astype(int).isin(state)
    ].copy()
    local["source_state"] = local.source.astype(int).map(state)
    local["target_state"] = local.target.astype(int).map(state)
    local = local[local.source_state.ne(local.target_state)].copy()
    if local.empty:
        return local
    local["distance_rank"] = local.D_joint.rank(method="average", pct=True)
    local["expression_rank"] = local.D_expression.rank(method="average", pct=True)
    local["soft_penalty"] = [
        -np.log(max(membership[int(row.source)] * membership[int(row.target)], EPS))
        for row in local.itertuples()
    ]
    local["membership_rank"] = local.soft_penalty.rank(method="average", pct=True)
    local["transition_cost"] = local.distance_rank + local.membership_rank
    local["event_transition_cost"] = (
        0.50 * local.distance_rank + 0.50 * local.expression_rank
    )
    local["state_key"] = [
        tuple(sorted((str(row.source_state), str(row.target_state))))
        for row in local.itertuples()
    ]
    return local.sort_values(
        ["transition_cost", "D_joint", "source", "target"], kind="mergesort"
    ).drop_duplicates("state_key")


def inferred_state_truth_map(
    nodes: pd.DataFrame, state_composition: pd.DataFrame, stage: str = "T1"
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], float]]:
    """Map inferred states to truth states for scoring, never construction."""
    local = state_composition[state_composition.stage.astype(str).eq(stage)].copy()
    joined = nodes[["root_id", "unit_id", "state_id"]].merge(
        local[["root_id", "metacell_id", "state_id", "state_probability"]],
        left_on=["root_id", "unit_id"], right_on=["root_id", "metacell_id"],
        how="left", suffixes=("_inferred", "_truth"),
    )
    joined["state_probability"] = joined.state_probability.fillna(0.0).astype(float)
    support = joined.groupby(
        ["root_id", "state_id_inferred", "state_id_truth"], dropna=True
    ).state_probability.sum().reset_index()
    totals = support.groupby(["root_id", "state_id_inferred"]).state_probability.transform("sum")
    support["mapping_probability"] = support.state_probability / totals.clip(lower=EPS)
    selected = support.sort_values(
        ["root_id", "state_id_inferred", "mapping_probability", "state_id_truth"],
        ascending=[True, True, False, True], kind="mergesort",
    ).drop_duplicates(["root_id", "state_id_inferred"])
    mapping = {
        (str(row.root_id), str(row.state_id_inferred)): str(row.state_id_truth)
        for row in selected.itertuples(index=False)
    }
    confidence = {
        (str(row.root_id), str(row.state_id_inferred)): float(row.mapping_probability)
        for row in selected.itertuples(index=False)
    }
    return mapping, confidence


def truth_graphs(truth_edges: set[tuple[str, str, str]]) -> dict[str, nx.Graph]:
    graphs: dict[str, nx.Graph] = {}
    for root, left, right in truth_edges:
        graphs.setdefault(str(root), nx.Graph()).add_edge(str(left), str(right))
    return graphs


def scale_topology_distance(
    backbone: pd.DataFrame, nodes: pd.DataFrame,
    mapping: dict[tuple[str, str], str], truth_edges: set[tuple[str, str, str]],
) -> float:
    """Weighted state-pair path distortion on a common truth-state scale."""
    truth = truth_graphs(truth_edges)
    sizes = nodes.groupby(["root_id", "state_id"]).size().astype(float).to_dict()
    numerator = 0.0
    denominator = 0.0
    for root, local_nodes in nodes.groupby("root_id"):
        root = str(root)
        inferred = nx.Graph()
        states = sorted(local_nodes.state_id.astype(str).unique())
        inferred.add_nodes_from(states)
        local_edges = backbone[backbone.root_id.astype(str).eq(root)]
        inferred.add_edges_from(local_edges[["source_state", "target_state"]].astype(str).itertuples(
            index=False, name=None
        ))
        if not nx.is_connected(inferred) or root not in truth:
            return float("inf")
        truth_graph = truth[root]
        truth_scale = max(nx.diameter(truth_graph), 1)
        for index, left in enumerate(states):
            for right in states[index + 1:]:
                mapped_left = mapping.get((root, left))
                mapped_right = mapping.get((root, right))
                if mapped_left not in truth_graph or mapped_right not in truth_graph:
                    continue
                predicted_distance = nx.shortest_path_length(inferred, left, right)
                true_distance = nx.shortest_path_length(
                    truth_graph, mapped_left, mapped_right
                )
                weight = np.sqrt(
                    sizes.get((root, left), 1.0) * sizes.get((root, right), 1.0)
                )
                numerator += weight * abs(predicted_distance - true_distance) / truth_scale
                denominator += weight
    return float(numerator / denominator) if denominator else float("inf")


def truth_branch(state: str) -> str:
    value = str(state)
    if "_B" in value and "_S" in value:
        return value.rsplit("_S", 1)[0]
    return value


def endpoint_error(
    predicted: tuple[str, str], target: tuple[str, str], graph: nx.Graph
) -> float:
    def distance(left: str, right: str) -> float:
        if left not in graph or right not in graph:
            return float(graph.number_of_nodes())
        return float(nx.shortest_path_length(graph, left, right))
    return min(
        distance(predicted[0], target[0]) + distance(predicted[1], target[1]),
        distance(predicted[0], target[1]) + distance(predicted[1], target[0]),
    )


def scale_event_metrics(
    event, before: pd.DataFrame, after: pd.DataFrame,
    mapping: dict[tuple[str, str], str], truth_t0: set, truth_t1: set,
) -> dict[str, float]:
    root = str(event.receiver_root)
    before_pairs = {
        tuple(sorted((str(row.source_state), str(row.target_state))))
        for row in before[before.root_id.astype(str).eq(root)].itertuples(index=False)
    }
    after_pairs = {
        tuple(sorted((str(row.source_state), str(row.target_state))))
        for row in after[after.root_id.astype(str).eq(root)].itertuples(index=False)
    }
    predicted_removed = before_pairs - after_pairs
    predicted_added = after_pairs - before_pairs
    true_added = [(left, right) for r, left, right in truth_t1 - truth_t0 if r == root]
    true_removed = [(left, right) for r, left, right in truth_t0 - truth_t1 if r == root]
    graph = truth_graphs(truth_t1).get(root, nx.Graph())

    def mapped(pair_value: tuple[str, str]) -> tuple[str, str] | None:
        left = mapping.get((root, pair_value[0]))
        right = mapping.get((root, pair_value[1]))
        return (left, right) if left and right else None

    added_mapped = [value for value in map(mapped, predicted_added) if value]
    removed_mapped = [value for value in map(mapped, predicted_removed) if value]
    added_error = min(
        (endpoint_error(predicted, target, graph)
         for predicted in added_mapped for target in true_added), default=float("nan")
    )
    removed_error = min(
        (endpoint_error(predicted, target, graph)
         for predicted in removed_mapped for target in true_removed), default=float("nan")
    )
    correct_branch = 0.0
    if added_mapped and true_added:
        predicted_branches = {
            tuple(sorted(map(truth_branch, edge))) for edge in added_mapped
        }
        true_branches = {tuple(sorted(map(truth_branch, edge))) for edge in true_added}
        correct_branch = float(bool(predicted_branches & true_branches))
    return {
        "event_added_location_error_scale": float(added_error),
        "event_removed_location_error_scale": float(removed_error),
        "branch_choice_accuracy_scale": correct_branch,
    }


def refine_event(
    run_dir: Path, dataset: str, backbone: pd.DataFrame, nodes: pd.DataFrame,
    universe: pd.DataFrame, event, gamma: float, branch_composition: pd.DataFrame,
):
    root_id = str(event.receiver_root)
    if not BASE.as_bool(event.causal) or float(event.effect_strength) <= 0:
        return backbone.copy(), pd.DataFrame(), {
            "accepted": False, "reason": "oracle_null_abstention",
            "upstream_violation": False, "n_selected_swaps": 0,
        }
    local_edges = backbone[backbone.root_id.astype(str).eq(root_id)].copy()
    node_part = nodes[nodes.root_id.astype(str).eq(root_id)].copy()
    state_nodes = {
        str(state): set(group.node_id.astype(int))
        for state, group in node_part.groupby("state_id")
    }
    state_names = {
        str(state): set(group.unit_id.astype(str))
        for state, group in node_part.groupby("state_id")
    }
    state_tau = node_part.groupby("state_id").pseudotime.median().astype(float).to_dict()
    state_branch = node_part.groupby("state_id").branch_id.agg(
        lambda values: values.astype(str).value_counts().sort_index().idxmax()
    ).to_dict()
    root_rows = node_part[node_part.is_root.map(BASE.as_bool)]
    if len(root_rows) != 1:
        raise ValueError(f"Expected one root state for {root_id}")
    anchor_state = str(root_rows.iloc[0].state_id)
    graph = nx.Graph()
    graph.add_nodes_from(state_nodes)
    graph.add_edges_from(local_edges[["source_state", "target_state"]].itertuples(
        index=False, name=None
    ))
    if not nx.is_tree(graph) or anchor_state not in graph:
        return backbone.copy(), pd.DataFrame(), {
            "accepted": False, "reason": "frozen_backbone_not_rooted_tree",
            "upstream_violation": False, "phase_a_connectivity_failure": True,
            "n_selected_swaps": 0,
        }
    paths = nx.single_source_shortest_path(graph, anchor_state)
    parent = {state: path[-2] for state, path in paths.items() if len(path) > 1}
    affected_stage = "T0" if str(event.event_type) == "inhibition" else "T1"
    affected = branch_composition[
        branch_composition.stage.astype(str).eq(affected_stage)
        & branch_composition.root_id.astype(str).eq(root_id)
        & branch_composition.branch_id.astype(str).eq(str(event.affected_branch))
    ]
    branch_probability = dict(zip(
        affected.metacell_id.astype(str), affected.branch_probability.astype(float)
    ))
    exposure = BASE.event_exposure_by_metacell(run_dir, str(event.event_id))
    response = BASE.receiver_response_residual(run_dir, dataset, event, node_part, exposure)
    exposure_values = pd.Series({
        name: exposure.get(name, 0.0) for name in node_part.unit_id.astype(str)
    })
    exposure_rank = exposure_values.rank(method="average", pct=True).to_dict()
    state_support = {}
    state_affected = {}
    for state, names in state_names.items():
        affected_probability = max((branch_probability.get(name, 0.0) for name in names), default=0.0)
        state_affected[state] = affected_probability
        values = [
            response.get(name, 0.0)
            * (0.50 + 0.50 * branch_probability.get(name, 0.0))
            * (0.50 + 0.50 * exposure_rank.get(name, 0.0))
            for name in names
        ]
        state_support[state] = float(max(values, default=0.0))
    maximum = max(state_support.values(), default=0.0)
    if maximum > 0:
        state_support = {state: value / maximum for state, value in state_support.items()}
    alternatives = state_candidates(universe, node_part, root_id)
    alternative_lookup = {
        row.state_key: row for row in alternatives.itertuples(index=False)
    }
    trials = []
    for child, support in sorted(state_support.items()):
        if child == anchor_state or child not in parent or state_affected.get(child, 0.0) < STATE_SUPPORT:
            continue
        if (
            state_tau.get(child, -np.inf) < float(event.receiver_tau_start) - 0.08
            or state_tau.get(child, np.inf) > float(event.receiver_tau_end) + 0.08
        ):
            continue
        old_parent = parent[child]
        old_row = local_edges[
            local_edges.apply(
                lambda row: {str(row.source_state), str(row.target_state)} == {old_parent, child},
                axis=1,
            )
        ].iloc[0]
        old_candidate = alternative_lookup.get(tuple(sorted((old_parent, child))))
        without = graph.copy()
        without.remove_edge(old_parent, child)
        subtree = set(nx.node_connected_component(without, child))
        for new_parent in sorted(graph.nodes):
            if new_parent in subtree or new_parent == old_parent:
                continue
            if state_tau.get(new_parent, 1.0) >= state_tau.get(child, 0.0):
                continue
            key = tuple(sorted((new_parent, child)))
            candidate = alternative_lookup.get(key)
            if candidate is None:
                continue
            if (
                str(event.event_type) == "inhibition"
                and state_affected.get(new_parent, 0.0) >= STATE_SUPPORT
            ):
                continue
            score, upstream_shift, branch_escape = BASE.alignment(
                str(event.event_type), support,
                state_tau.get(old_parent, 0.0), state_tau.get(new_parent, 0.0),
                state_affected.get(new_parent, 0.0),
                state_branch.get(child) != state_branch.get(new_parent),
            )
            if score <= 0:
                continue
            # Old and new edges must be compared on the same root-local ECDF
            # scale. The frozen backbone stores its construction weight,
            # whereas alternatives use transition_cost; mixing them makes
            # gamma and abstention uninterpretable.
            old_cost = (
                float(old_candidate.transition_cost)
                if old_candidate is not None else float(old_row.edge_cost)
            )
            new_cost = float(candidate.transition_cost)
            modified_new = new_cost * (1.0 - gamma * score)
            modified_old = old_cost * (1.0 + gamma * support)
            trials.append({
                "event_id": str(event.event_id), "event_type": str(event.event_type),
                "root_id": root_id, "gamma": gamma, "child_state": child,
                "old_parent_state": old_parent, "new_parent_state": new_parent,
                "child_tau": state_tau[child], "old_parent_tau": state_tau[old_parent],
                "new_parent_tau": state_tau[new_parent], "event_support": support,
                "affected_probability": state_affected.get(child, 0.0),
                "alignment": score, "upstream_shift": upstream_shift,
                "branch_escape": branch_escape, "base_old_cost": old_cost,
                "base_new_cost": new_cost, "modified_old_cost": modified_old,
                "modified_new_cost": modified_new,
                "cost_advantage": modified_old - modified_new,
                "support_source": str(candidate.source_name),
                "support_target": str(candidate.target_name),
                "candidate_role": str(candidate.edge_graph_role),
                "candidate_legal": True, "truth_used_for_selection": False,
            })
    trial_frame = pd.DataFrame(trials)
    if trial_frame.empty:
        return backbone.copy(), trial_frame, {
            "accepted": False, "reason": "no_candidate_legal_state_alternative",
            "upstream_violation": False, "n_selected_swaps": 0,
        }
    support_gate = float(np.median(list(state_support.values())))
    accepted = trial_frame[
        trial_frame.cost_advantage.gt(0)
        & trial_frame.event_support.ge(support_gate)
    ].copy()
    if accepted.empty:
        return backbone.copy(), trial_frame, {
            "accepted": False, "reason": "state_cost_abstention",
            "upstream_violation": False, "n_selected_swaps": 0,
        }
    selected = accepted.sort_values(
        ["alignment", "event_support", "cost_advantage", "base_new_cost",
         "child_state", "new_parent_state"],
        ascending=[False, False, False, True, True, True], kind="mergesort",
    ).iloc[0]
    refined = backbone.copy()
    local_mask = refined.root_id.astype(str).eq(root_id)
    remove = local_mask & refined.apply(
        lambda row: {str(row.source_state), str(row.target_state)}
        == {str(selected.old_parent_state), str(selected.child_state)}, axis=1
    )
    row = refined.loc[local_mask].iloc[0].copy()
    row["source_state"] = str(selected.new_parent_state)
    row["target_state"] = str(selected.child_state)
    row["support_source"] = str(selected.support_source)
    row["support_target"] = str(selected.support_target)
    row["edge_cost"] = float(selected.base_new_cost)
    row["edge_graph_role"] = "oracle_event_conditioned_state_edge"
    row["connector_only"] = False
    refined = pd.concat([refined.loc[~remove], pd.DataFrame([row])], ignore_index=True)
    trial_frame["selected"] = False
    trial_frame.loc[selected.name, "selected"] = True
    return refined, trial_frame, {
        "accepted": True, "reason": "positive_state_event_cost_advantage",
        "n_selected_swaps": 1, "support_gate": support_gate,
        "child_tau": float(selected.child_tau),
        "receiver_tau_start_true_scale": float(event.receiver_tau_start),
        "upstream_violation": False,
        "selected_child": str(selected.child_state),
        "selected_old_parent": str(selected.old_parent_state),
        "selected_new_parent": str(selected.new_parent_state),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase-a-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir, phase_a = args.run_dir.resolve(), args.phase_a_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    backbone = pd.read_csv(phase_a / "T0_STATE_BACKBONE_EDGES.csv")
    nodes = pd.read_csv(phase_a / "T0_STATE_NODE_ASSIGNMENTS.csv")
    universe = candidate_universe(run_dir, args.dataset, phase_a)
    events = pd.read_csv(run_dir / "event" / "truth" / "TRUE_EVENTS.csv")
    branch = pd.read_csv(run_dir / "formal_metacell" / "METACELL_BRANCH_COMPOSITION.csv")
    state = pd.read_csv(run_dir / "formal_metacell" / "METACELL_STATE_COMPOSITION.csv")
    state_lookup = BASE.dominant_state_lookup(state, "T1")
    inferred_truth_map, inferred_truth_confidence = inferred_state_truth_map(nodes, state, "T1")
    truth_t0 = BASE.truth_edge_set(pd.read_csv(run_dir / "event" / "truth" / "TRUE_T0_EDGES.csv"))
    truth_t1 = BASE.truth_edge_set(pd.read_csv(run_dir / "event" / "truth" / "TRUE_T1_EDGES.csv"))
    result_rows, trial_frames, edge_frames = [], [], []
    before_edges = backbone_edge_frame(backbone, nodes)
    scale_before = scale_topology_distance(
        backbone, nodes, inferred_truth_map, truth_t1
    )
    for event in events.itertuples(index=False):
        for gamma in GAMMAS:
            refined, trials, audit = refine_event(
                run_dir, args.dataset, backbone, nodes, universe, event, gamma, branch
            )
            after_edges = backbone_edge_frame(refined, nodes)
            metrics = BASE.evaluate(
                event, before_edges, after_edges, state_lookup, truth_t0, truth_t1, audit
            )
            scale_after = scale_topology_distance(
                refined, nodes, inferred_truth_map, truth_t1
            )
            scale_event = scale_event_metrics(
                event, backbone, refined, inferred_truth_map, truth_t0, truth_t1
            )
            local = refined[refined.root_id.astype(str).eq(str(event.receiver_root))]
            graph = nx.Graph()
            graph.add_edges_from(local[["source_state", "target_state"]].itertuples(
                index=False, name=None
            ))
            result_rows.append({
                "run": run_dir.name, "event_id": str(event.event_id),
                "event_type": str(event.event_type), "causal": BASE.as_bool(event.causal),
                "receiver_root": str(event.receiver_root),
                "affected_branch": str(event.affected_branch), "gamma": gamma,
                **audit, **metrics, **scale_event,
                "scale_topology_distance_before": scale_before,
                "scale_topology_distance_after": scale_after,
                "scale_oracle_topology_gain": scale_before - scale_after,
                "mean_state_truth_mapping_confidence": float(np.mean(list(
                    inferred_truth_confidence.values()
                ))) if inferred_truth_confidence else np.nan,
                "candidate_legal": True,
                "connected": nx.is_connected(graph) if len(graph) else False,
                "acyclic": nx.is_forest(graph), "truth_used_for_selection": False,
                "truth_used_for_scoring_only": True,
            })
            if len(trials):
                trials.insert(0, "run", run_dir.name)
                trial_frames.append(trials)
            marked = after_edges.copy()
            marked.insert(0, "gamma", gamma)
            marked.insert(0, "event_id", str(event.event_id))
            edge_frames.append(marked)
    results = pd.DataFrame(result_rows)
    results.to_csv(args.output / "PHASE_C_ORACLE_REFINEMENT.csv", index=False)
    results.to_csv(args.output / "PHASE_C_REWIRING_METRICS.csv", index=False)
    results[~results.causal.astype(bool)].to_csv(
        args.output / "PHASE_C_NULL_REWIRE_CONTROL.csv", index=False
    )
    results.to_csv(args.output / "PHASE_C_GAMMA_SENSITIVITY.csv", index=False)
    if trial_frames:
        pd.concat(trial_frames, ignore_index=True).to_csv(
            args.output / "PHASE_C_LEGAL_SWAP_TRIALS.csv", index=False
        )
    pd.concat(edge_frames, ignore_index=True).to_csv(
        args.output / "PHASE_C_REFINED_EDGE_SENSITIVITY.csv", index=False
    )
    status = {
        "phase": "C", "operator_version": "ORACLE_BACKBONE_REFINEMENT_V4",
        "phase_a": "PHASE_A_v5_3_guarded_hierarchical_graph_partition_state",
        "gammas": GAMMAS, "relation_detector_called": False,
        "exact_truth_edges_used_for_selection": False,
        "oracle_inputs": [
            "true event identity", "true source and receiver",
            "true affected branch", "true event window", "true event-local exposure",
        ],
        "complete": True,
    }
    (args.output / "RUN_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
