#!/usr/bin/env python3
"""PHASE C: oracle-localized, candidate-legal topology refinement V2."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd


GAMMAS = (0.10, 0.20, 0.25, 0.35, 0.50)
STATE_SUPPORT = 0.20
EPS = 1e-12


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def pair(left: object, right: object) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def edge_key(left: int, right: int) -> tuple[int, int]:
    return tuple(sorted((int(left), int(right))))


def prf(truth: set, predicted: set) -> tuple[float, float, float]:
    overlap = len(truth & predicted)
    precision = overlap / len(predicted) if predicted else (1.0 if not truth else 0.0)
    recall = overlap / len(truth) if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def truth_edge_set(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    return {
        (str(row.root_id), *pair(row.source, row.target))
        for row in frame.itertuples(index=False)
    }


def dominant_state_lookup(state: pd.DataFrame, stage: str) -> dict[tuple[str, str], str]:
    local = state[state.stage.astype(str).eq(stage)].copy()
    local = local.sort_values(
        ["metacell_id", "root_id", "state_probability", "state_id"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["metacell_id", "root_id"])
    return {
        (str(row.metacell_id), str(row.root_id)): str(row.state_id)
        for row in local.itertuples(index=False)
    }


def mapped_edge_set(
    edges: pd.DataFrame, state_lookup: dict[tuple[str, str], str]
) -> set[tuple[str, str, str]]:
    result = set()
    for row in edges.itertuples(index=False):
        root = str(row.root_id)
        source = state_lookup.get((str(row.source_name), root))
        target = state_lookup.get((str(row.target_name), root))
        if source and target and source != target:
            result.add((root, *pair(source, target)))
    return result


def event_exposure_by_metacell(run_dir: Path, event_id: str) -> dict[str, float]:
    mapping = pd.read_csv(run_dir / "formal_metacell" / "cell_to_metacell_mapping.csv")
    contact = pd.read_csv(run_dir / "event" / "truth" / "TRUE_SPATIAL_CONTACTS.csv")
    contact = contact[contact.event_id.astype(str).eq(str(event_id))].copy()
    if not len(contact):
        return {}
    exposed = contact[["receiver_cell_id", "is_event_exposed"]].drop_duplicates(
        "receiver_cell_id"
    ).rename(columns={"receiver_cell_id": "cell_name"})
    exposed["is_event_exposed"] = exposed.is_event_exposed.map(as_bool).astype(float)
    joined = mapping[["cell_name", "metacell_id"]].merge(
        exposed, on="cell_name", how="left"
    )
    joined["is_event_exposed"] = joined.is_event_exposed.fillna(0.0)
    return joined.groupby("metacell_id").is_event_exposed.mean().astype(float).to_dict()


def program_genes(value: object) -> list[str]:
    text = str(value)
    match = re.fullmatch(r"(G\d+)-(G\d+)", text)
    if match:
        start, end = int(match.group(1)[1:]), int(match.group(2)[1:])
        return [f"G{index}" for index in range(start, end + 1)]
    return [token for token in text.split(";") if token]


def receiver_response_residual(
    run_dir: Path, dataset: str, event, node_part: pd.DataFrame,
    exposure: dict[str, float],
) -> dict[str, float]:
    """Residualize the oracle receiver program against non-event context."""
    meta = ad.read_h5ad(
        run_dir / "generic_upstream_v1_exact_root" / dataset / "metacell_adata.h5ad"
    )
    genes = program_genes(event.receiver_program_genes)
    indices = [meta.var_names.get_loc(gene) for gene in genes if gene in meta.var_names]
    if not indices:
        return {}
    matrix = meta.X.toarray() if hasattr(meta.X, "toarray") else np.asarray(meta.X)
    score = np.log1p(np.maximum(matrix[:, indices], 0.0)).mean(axis=1)
    score_lookup = dict(zip(meta.obs_names.astype(str), score.astype(float)))
    tau_lookup = dict(zip(node_part.unit_id.astype(str), node_part.pseudotime.astype(float)))
    names = [name for name in node_part.unit_id.astype(str) if name in score_lookup]
    if len(names) < 6:
        return {name: 0.0 for name in names}
    tau = np.array([tau_lookup[name] for name in names], dtype=float)
    spatial = meta.obs.loc[names, ["spatial_x", "spatial_y"]].to_numpy(float)
    for column in range(spatial.shape[1]):
        scale = np.std(spatial[:, column])
        spatial[:, column] = (
            (spatial[:, column] - np.mean(spatial[:, column])) / max(scale, EPS)
        )
    design = np.column_stack([np.ones(len(names)), tau, tau * tau, spatial])
    outcome = np.array([score_lookup[name] for name in names], dtype=float)
    control = np.array([exposure.get(name, 0.0) <= 0.05 for name in names])
    if control.sum() < design.shape[1] + 2:
        control = np.ones(len(names), dtype=bool)
    coefficients = np.linalg.lstsq(design[control], outcome[control], rcond=None)[0]
    residual = outcome - design @ coefficients
    if str(event.event_type) == "inhibition":
        residual = -residual
    ranks = pd.Series(residual).rank(method="average", pct=True).to_numpy(float)
    return dict(zip(names, ranks))


def candidate_cost_lookup(candidate: pd.DataFrame) -> dict[tuple[int, int], float]:
    eligible = candidate[candidate.trajectory_edge_eligible.astype(bool)].copy()
    eligible["key"] = [edge_key(row.source, row.target) for row in eligible.itertuples()]
    return eligible.groupby("key").D_joint.min().astype(float).to_dict()


def replace_edge(
    edges: pd.DataFrame, root_id: str, old_parent: int, child: int,
    new_parent: int, names: dict[int, str], new_cost: float,
) -> pd.DataFrame:
    local = edges.root_id.astype(str).eq(root_id)
    remove = local & (
        ((edges.source.astype(int) == old_parent) & (edges.target.astype(int) == child))
        | ((edges.source.astype(int) == child) & (edges.target.astype(int) == old_parent))
    )
    row = edges.loc[local].iloc[0].copy()
    row["source"], row["target"] = int(new_parent), int(child)
    row["source_name"], row["target_name"] = names[new_parent], names[child]
    row["edge_cost"] = float(new_cost)
    row["edge_graph_role"] = "oracle_event_conditioned_candidate_edge"
    row["candidate_legal"] = True
    row["connector_only"] = False
    return pd.concat([edges.loc[~remove], pd.DataFrame([row])], ignore_index=True)


def alignment(
    event_type: str, support: float, old_tau: float, new_tau: float,
    parent_affected_probability: float, program_change: bool,
) -> tuple[float, float, float]:
    upstream_shift = float(np.clip(old_tau - new_tau, 0.0, 1.0))
    tau_compatibility = float(np.clip(1.0 - abs(old_tau - new_tau), 0.0, 1.0))
    branch_escape = float(np.clip(1.0 - parent_affected_probability, 0.0, 1.0))
    if event_type == "promotion":
        parent_earliness = float(np.clip(1.0 - new_tau, 0.0, 1.0))
        value = support * (0.25 + 0.75 * parent_earliness)
    elif event_type == "inhibition":
        # Inhibition represents branch escape, not a generic jump toward the
        # root.  Preserve the old edge's temporal scale while preferring a
        # parent outside the affected program.
        value = support * (0.50 + 0.50 * branch_escape) * (
            0.25 + 0.75 * tau_compatibility
        )
    elif event_type == "collision_convergence":
        value = (
            support * (0.50 + 0.50 * branch_escape)
            * (0.50 + 0.50 * tau_compatibility)
            if program_change else 0.0
        )
    else:
        value = 0.0
    return float(value), upstream_shift, branch_escape


def refine_event(
    run_dir: Path, dataset: str, t0: pd.DataFrame, nodes: pd.DataFrame, candidate: pd.DataFrame,
    event, gamma: float, branch_composition: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    root_id = str(event.receiver_root)
    if not as_bool(event.causal) or float(event.effect_strength) <= 0:
        return t0.copy(), pd.DataFrame(), {
            "accepted": False, "reason": "oracle_null_abstention",
            "upstream_violation": False,
        }
    part = t0[t0.root_id.astype(str).eq(root_id)].copy()
    node_part = nodes[nodes.root_id.astype(str).eq(root_id)].copy()
    tau = dict(zip(node_part.node_id.astype(int), node_part.pseudotime.astype(float)))
    names = dict(zip(node_part.node_id.astype(int), node_part.unit_id.astype(str)))
    branch_program = dict(zip(
        node_part.node_id.astype(int), node_part.branch_id.astype(str)
    ))
    root_rows = node_part[node_part.is_root.map(as_bool)]
    if len(root_rows) != 1:
        raise ValueError(f"Expected one root node for {root_id}")
    anchor = int(root_rows.iloc[0].node_id)
    graph = nx.Graph()
    graph.add_edges_from(part[["source", "target"]].astype(int).itertuples(index=False, name=None))
    if not nx.is_tree(graph) or anchor not in graph:
        return t0.copy(), pd.DataFrame(), {
            "accepted": False,
            "reason": "frozen_t0_not_root_connected",
            "upstream_violation": False,
            "phase_a_connectivity_failure": True,
        }
    paths = nx.single_source_shortest_path(graph, anchor)
    parent = {node: path[-2] for node, path in paths.items() if len(path) > 1}
    affected = branch_composition[
        branch_composition.stage.astype(str).eq("T1")
        & branch_composition.root_id.astype(str).eq(root_id)
        & branch_composition.branch_id.astype(str).eq(str(event.affected_branch))
    ]
    branch_probability = dict(zip(
        affected.metacell_id.astype(str), affected.branch_probability.astype(float)
    ))
    exposure = event_exposure_by_metacell(run_dir, str(event.event_id))
    response = receiver_response_residual(run_dir, dataset, event, node_part, exposure)
    exposure_values = pd.Series({name: exposure.get(name, 0.0) for name in names.values()})
    exposure_rank = exposure_values.rank(method="average", pct=True).to_dict()
    affected_nodes = {
        node for node, name in names.items()
        if branch_probability.get(name, 0.0) >= STATE_SUPPORT and node != anchor
    }
    relative_tau = pd.Series({node: tau.get(node, 0.0) for node in affected_nodes})
    localization = (
        1.0 - relative_tau.rank(method="average", pct=True)
        + (1.0 / max(len(relative_tau), 1))
    ).clip(0.0, 1.0).to_dict()
    raw_support = {
        node: response.get(name, 0.0)
        * (0.50 + 0.50 * branch_probability.get(name, 0.0))
        * (0.50 + 0.50 * exposure_rank.get(name, 0.0))
        * (0.25 + 0.75 * localization.get(node, 0.0))
        for node, name in names.items()
        if branch_probability.get(name, 0.0) >= STATE_SUPPORT
        and node != anchor
    }
    maximum = max(raw_support.values(), default=0.0)
    support = {
        node: value / maximum for node, value in raw_support.items()
        if maximum > 0 and value > 0
    }
    costs = candidate_cost_lookup(candidate)
    candidate_pairs = set(costs)
    trials = []
    for child in sorted(support):
        if child == anchor or child not in parent:
            continue
        old_parent = parent[child]
        old_cost = costs.get(edge_key(old_parent, child))
        if old_cost is None:
            old_cost = float(
                part[
                    ((part.source.astype(int) == old_parent) & (part.target.astype(int) == child))
                    | ((part.source.astype(int) == child) & (part.target.astype(int) == old_parent))
                ].edge_cost.iloc[0]
            )
        without = graph.copy()
        without.remove_edge(old_parent, child)
        subtree = nx.node_connected_component(without, child)
        for new_parent in sorted(graph.nodes):
            key = edge_key(new_parent, child)
            if key not in candidate_pairs or new_parent in subtree or new_parent == old_parent:
                continue
            if tau.get(new_parent, 1.0) >= tau.get(child, 0.0):
                continue
            if (
                str(event.event_type) == "inhibition"
                and branch_probability.get(names[new_parent], 0.0) >= STATE_SUPPORT
            ):
                continue
            if (
                str(event.event_type) == "collision_convergence"
                and exposure.get(names[new_parent], 0.0) <= 0.0
            ):
                continue
            score, upstream_shift, branch_escape = alignment(
                str(event.event_type), support[child], tau[old_parent], tau[new_parent],
                branch_probability.get(names[new_parent], 0.0),
                branch_program.get(child) != branch_program.get(new_parent),
            )
            if score <= 0:
                continue
            new_cost = costs[key]
            modified_new = new_cost * (1.0 - gamma * score)
            modified_old = old_cost * (1.0 + gamma * support[child])
            trials.append({
                "event_id": str(event.event_id), "root_id": root_id,
                "gamma": gamma, "child": child, "child_name": names[child],
                "old_parent": old_parent, "old_parent_name": names[old_parent],
                "new_parent": new_parent, "new_parent_name": names[new_parent],
                "child_tau": tau[child], "old_parent_tau": tau[old_parent],
                "new_parent_tau": tau[new_parent], "event_support": support[child],
                "receiver_response_rank": response.get(names[child], 0.0),
                "event_exposure_fraction": exposure.get(names[child], 0.0),
                "parent_event_exposure_fraction": exposure.get(names[new_parent], 0.0),
                "event_onset_localization": localization.get(child, 0.0),
                "alignment": score, "upstream_shift": upstream_shift,
                "branch_escape": branch_escape, "base_old_cost": old_cost,
                "program_change": branch_program.get(child) != branch_program.get(new_parent),
                "base_new_cost": new_cost, "modified_old_cost": modified_old,
                "modified_new_cost": modified_new,
                "cost_advantage": modified_old - modified_new,
                "candidate_legal": True, "truth_used_for_selection": False,
            })
    trial_frame = pd.DataFrame(trials)
    if trial_frame.empty:
        return t0.copy(), trial_frame, {
            "accepted": False, "reason": "no_candidate_legal_alternative",
            "upstream_violation": False,
        }
    accepted = trial_frame[trial_frame.cost_advantage.astype(float).gt(0)].copy()
    support_gate = (
        0.0 if str(event.event_type) == "collision_convergence"
        else (float(np.median(list(support.values()))) if support else 1.0)
    )
    accepted = accepted[
        accepted.event_support.astype(float).ge(support_gate)
    ].copy()
    if not len(accepted):
        return t0.copy(), trial_frame, {
            "accepted": False, "reason": "cost_abstention",
            "upstream_violation": False,
        }
    ordered = accepted.sort_values(
        ["alignment", "cost_advantage", "base_new_cost", "child", "new_parent"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    refined = t0.copy()
    current_graph = graph.copy()
    selected_rows = []
    changed_children = set()
    for index, candidate_row in ordered.iterrows():
        child = int(candidate_row.child)
        old_parent = int(candidate_row.old_parent)
        new_parent = int(candidate_row.new_parent)
        if child in changed_children or not current_graph.has_edge(old_parent, child):
            continue
        without = current_graph.copy()
        without.remove_edge(old_parent, child)
        if new_parent in nx.node_connected_component(without, child):
            continue
        trial_graph = without.copy()
        trial_graph.add_edge(new_parent, child)
        if not nx.is_tree(trial_graph):
            continue
        refined = replace_edge(
            refined, root_id, old_parent, child, new_parent,
            names, float(candidate_row.base_new_cost),
        )
        current_graph = trial_graph
        changed_children.add(child)
        selected_rows.append((index, candidate_row))
        # One event authorizes one branch-boundary consequence.  Additional
        # changes require an independently detected event.
        break
    if not selected_rows:
        return t0.copy(), trial_frame, {
            "accepted": False, "reason": "sequential_legality_abstention",
            "upstream_violation": False,
        }
    trial_frame["selected"] = False
    trial_frame.loc[[index for index, _ in selected_rows], "selected"] = True
    chosen = [row for _, row in selected_rows]
    audit = {
        "accepted": True, "reason": "positive_event_conditioned_cost_advantage",
        "n_selected_swaps": len(chosen),
        "support_gate": support_gate,
        "child_tau": float(min(row.child_tau for row in chosen)),
        "receiver_tau_start_true_scale": float(event.receiver_tau_start),
        "upstream_violation": bool(any(
            int(row.child) not in affected_nodes for row in chosen
        )),
        "selected_child": ";".join(str(row.child_name) for row in chosen),
        "selected_old_parent": ";".join(str(row.old_parent_name) for row in chosen),
        "selected_new_parent": ";".join(str(row.new_parent_name) for row in chosen),
    }
    return refined, trial_frame, audit


def evaluate(
    event, before: pd.DataFrame, after: pd.DataFrame,
    state_lookup: dict[tuple[str, str], str], truth_t0: set, truth_t1: set,
    audit: dict,
) -> dict:
    mapped_before = mapped_edge_set(before, state_lookup)
    mapped_after = mapped_edge_set(after, state_lookup)
    before_distance = len(mapped_before ^ truth_t1)
    after_distance = len(mapped_after ^ truth_t1)
    true_removed, true_added = truth_t0 - truth_t1, truth_t1 - truth_t0
    pred_removed, pred_added = mapped_before - mapped_after, mapped_after - mapped_before
    truth_rewire = {("removed", edge) for edge in true_removed} | {
        ("added", edge) for edge in true_added
    }
    pred_rewire = {("removed", edge) for edge in pred_removed} | {
        ("added", edge) for edge in pred_added
    }
    precision, recall, rewire_f1 = prf(truth_rewire, pred_rewire)
    affected_token = f"_{event.affected_branch}_"
    root = str(event.receiver_root)
    affected_truth = {
        edge for edge in truth_t1
        if edge[0] == root and (affected_token in edge[1] or affected_token in edge[2])
    }
    affected_pred = {
        edge for edge in mapped_after
        if edge[0] == root and (affected_token in edge[1] or affected_token in edge[2])
    }
    _, _, affected_path_f1 = prf(affected_truth, affected_pred)
    true_event_added = {edge for edge in true_added if edge[0] == root}
    branch_choice_accuracy = float(
        len(true_event_added & mapped_after) / len(true_event_added)
    ) if true_event_added else float(not audit["accepted"])
    affected_before = len({edge for edge in mapped_before ^ truth_t1 if edge[0] == root})
    affected_after = len({edge for edge in mapped_after ^ truth_t1 if edge[0] == root})
    return {
        "rewiring_precision": precision, "rewiring_recall": recall,
        "rewiring_F1": rewire_f1, "affected_path_F1": affected_path_f1,
        "branch_choice_accuracy": branch_choice_accuracy,
        "topology_distance_before": before_distance,
        "topology_distance_after": after_distance,
        "oracle_topology_gain": before_distance - after_distance,
        "branch_scale_topology_gain": affected_before - affected_after,
        "n_predicted_removed_edges": len(pred_removed),
        "n_predicted_added_edges": len(pred_added),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="CORE_05")
    parser.add_argument("--phase-a-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    phase_a = args.phase_a_dir.resolve() if args.phase_a_dir else run_dir / "phase_a_v4_frozen"
    args.output.mkdir(parents=True, exist_ok=True)
    t0 = pd.read_csv(phase_a / "T0_FINE_TRAJECTORY_EDGES.csv")
    nodes = pd.read_csv(phase_a / "T0_STATE_NODE_ASSIGNMENTS.csv")
    candidate = pd.read_csv(phase_a / "C1_MULTIRESOLUTION_CANDIDATE_EDGES.csv")
    events = pd.read_csv(run_dir / "event" / "truth" / "TRUE_EVENTS.csv")
    branch = pd.read_csv(run_dir / "formal_metacell" / "METACELL_BRANCH_COMPOSITION.csv")
    state = pd.read_csv(run_dir / "formal_metacell" / "METACELL_STATE_COMPOSITION.csv")
    state_lookup = dominant_state_lookup(state, "T1")
    truth_t0 = truth_edge_set(pd.read_csv(run_dir / "event" / "truth" / "TRUE_T0_EDGES.csv"))
    truth_t1 = truth_edge_set(pd.read_csv(run_dir / "event" / "truth" / "TRUE_T1_EDGES.csv"))
    result_rows, trial_frames, edge_frames = [], [], []
    for event in events.itertuples(index=False):
        for gamma in GAMMAS:
            refined, trials, audit = refine_event(
                run_dir, args.dataset, t0, nodes, candidate, event, gamma, branch
            )
            metrics = evaluate(event, t0, refined, state_lookup, truth_t0, truth_t1, audit)
            result_rows.append({
                "run": run_dir.name, "event_id": str(event.event_id),
                "event_type": str(event.event_type), "causal": as_bool(event.causal),
                "receiver_root": str(event.receiver_root),
                "affected_branch": str(event.affected_branch), "gamma": gamma,
                **audit, **metrics,
                "candidate_legal": bool(refined.candidate_legal.astype(bool).all()),
                "connected": all(
                    nx.is_connected(nx.Graph(part[["source", "target"]].astype(int).itertuples(index=False, name=None)))
                    for _, part in refined.groupby("root_id")
                ),
                "acyclic": all(
                    nx.is_forest(nx.Graph(part[["source", "target"]].astype(int).itertuples(index=False, name=None)))
                    for _, part in refined.groupby("root_id")
                ),
                "truth_used_for_selection": False,
                "truth_used_for_scoring_only": True,
            })
            if len(trials):
                trials.insert(0, "run", run_dir.name)
                trial_frames.append(trials)
            marked = refined.copy()
            marked.insert(0, "gamma", gamma)
            marked.insert(0, "event_id", str(event.event_id))
            edge_frames.append(marked)
    results = pd.DataFrame(result_rows)
    results.to_csv(args.output / "PHASE_C_ORACLE_REFINEMENT.csv", index=False)
    results[[
        "run", "event_id", "gamma", "rewiring_precision", "rewiring_recall",
        "rewiring_F1", "affected_path_F1", "branch_choice_accuracy",
        "topology_distance_before", "topology_distance_after",
        "oracle_topology_gain", "branch_scale_topology_gain",
    ]].to_csv(args.output / "PHASE_C_REWIRING_METRICS.csv", index=False)
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
        "phase": "C", "operator_version": "ORACLE_REFINEMENT_V2_BRANCH_ESCAPE",
        "frozen_phase_a": "PHASE_A_v4_generator_v2_fine_hvg_v9_regularized",
        "gammas": GAMMAS, "state_support": STATE_SUPPORT,
        "relation_detector_called": False,
        "exact_truth_edges_used_for_selection": False,
        "oracle_inputs": [
            "true event identity", "true source and receiver", "true affected branch",
            "true event window", "true event-local exposure",
        ],
        "complete": True,
    }
    (args.output / "RUN_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(results.groupby(["gamma", "causal"])[
        ["accepted", "oracle_topology_gain", "rewiring_F1"]
    ].mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
