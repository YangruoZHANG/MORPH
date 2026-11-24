#!/usr/bin/env python3
"""Truth-fed decomposition of candidate, detector, and rewiring bottlenecks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd

from run_generic_relation_refinement_v1 import branch_assignment, export_prediction, refine_trees
from score_benchmark_prediction_v1 import score


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def undirected_pair(a: object, b: object) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def f1_sets(truth: set[str], predicted: set[str]) -> tuple[float, float, float]:
    overlap = len(truth & predicted)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def dominant_state_lookup(state_composition: pd.DataFrame, stage: str) -> dict[str, str]:
    selected = state_composition[state_composition["stage"].astype(str).eq(stage)].copy()
    # Match score_benchmark_prediction_v1 exactly, including stable input order
    # for probability ties.
    selected = selected.sort_values("state_probability", ascending=False)
    selected = selected.drop_duplicates("metacell_id")
    return dict(zip(selected["metacell_id"].astype(str), selected["state_id"].astype(str)))


def representative_metacell_by_state(state_composition: pd.DataFrame, stage: str) -> pd.DataFrame:
    selected = state_composition[state_composition["stage"].astype(str).eq(stage)].copy()
    selected = selected.sort_values(
        ["state_probability", "metacell_id"], ascending=[False, True]
    ).drop_duplicates(["root_id", "state_id"])
    return selected[["root_id", "state_id", "metacell_id", "state_probability"]]


def mapped_edge_set(edges: pd.DataFrame, state_lookup: dict[str, str], pos_to_root: dict[int, str]) -> set[tuple[str, str, str]]:
    result: set[tuple[str, str, str]] = set()
    for row in edges.itertuples(index=False):
        source = state_lookup.get(str(row.source_name))
        target = state_lookup.get(str(row.target_name))
        if source is None or target is None or source == target:
            continue
        a, b = undirected_pair(source, target)
        result.add((pos_to_root[int(row.root_pos)], a, b))
    return result


def truth_edge_set(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    return {
        (str(row.root_id), *undirected_pair(row.source, row.target))
        for row in frame.itertuples(index=False)
    }


def map_truth_edges_to_metacells(edges: pd.DataFrame, representatives: pd.DataFrame) -> pd.DataFrame:
    lookup = {
        (str(row.root_id), str(row.state_id)): (str(row.metacell_id), float(row.state_probability))
        for row in representatives.itertuples(index=False)
    }
    rows = []
    for row in edges.itertuples(index=False):
        source = lookup.get((str(row.root_id), str(row.source)))
        target = lookup.get((str(row.root_id), str(row.target)))
        rows.append({
            "root_id": str(row.root_id),
            "truth_source": str(row.source),
            "truth_target": str(row.target),
            "mapped_source": source[0] if source else "",
            "mapped_target": target[0] if target else "",
            "source_state_probability": source[1] if source else np.nan,
            "target_state_probability": target[1] if target else np.nan,
            "mapping_complete": bool(source and target and source[0] != target[0]),
        })
    return pd.DataFrame(rows)


def candidate_representability(
    truth_dir: Path,
    state_composition: pd.DataFrame,
    candidates: pd.DataFrame,
    inferred_t0: pd.DataFrame,
) -> pd.DataFrame:
    representatives = representative_metacell_by_state(state_composition, "T1")
    candidate_pairs = {undirected_pair(row.source_name, row.target_name) for row in candidates.itertuples(index=False)}
    inferred_pairs = {undirected_pair(row.source_name, row.target_name) for row in inferred_t0.itertuples(index=False)}
    rows = []
    for role, filename in (("removed", "TRUE_REMOVED_EDGES.csv"), ("added", "TRUE_ADDED_EDGES.csv")):
        truth = pd.read_csv(truth_dir / filename)
        mapped = map_truth_edges_to_metacells(truth, representatives)
        for row in mapped.itertuples(index=False):
            pair = undirected_pair(row.mapped_source, row.mapped_target)
            rows.append({
                "edge_role": role,
                **row._asdict(),
                "in_candidate_graph": bool(row.mapping_complete and pair in candidate_pairs),
                "in_inferred_T0": bool(row.mapping_complete and pair in inferred_pairs),
                "truth_used": True,
            })
    columns = [
        "edge_role", "root_id", "truth_source", "truth_target", "mapped_source",
        "mapped_target", "source_state_probability", "target_state_probability",
        "mapping_complete", "in_candidate_graph", "in_inferred_T0", "truth_used",
    ]
    return pd.DataFrame(rows, columns=columns)


def map_affected_branch(
    receiver_root: str,
    affected_branch: str,
    receiver_pos: int,
    inferred_t0: pd.DataFrame,
    branch_composition: pd.DataFrame,
    purity: float,
) -> dict[str, object]:
    part = inferred_t0[inferred_t0["root_pos"].astype(int).eq(receiver_pos)]
    anchor = str(part.iloc[0].root_id)
    predicted = branch_assignment(part, anchor)
    true_rows = branch_composition[
        branch_composition["stage"].astype(str).eq("T1")
        & branch_composition["root_id"].astype(str).eq(receiver_root)
        & branch_composition["branch_id"].astype(str).eq(affected_branch)
        & branch_composition["branch_probability"].astype(float).ge(purity)
    ]
    true_units = set(true_rows["metacell_id"].astype(str))
    records = []
    for branch in sorted(set(predicted.values()) - {"ROOT"}):
        units = {unit for unit, label in predicted.items() if label == branch}
        precision, recall, f1 = f1_sets(true_units, units)
        records.append((f1, precision, recall, branch, units))
    if not records:
        return {"mapped_branch": "none", "branch_precision": 0.0, "branch_recall": 0.0, "branch_f1": 0.0, "true_units": true_units, "predicted_units": set()}
    f1, precision, recall, branch, units = max(records, key=lambda item: (item[0], item[1], item[2], item[3]))
    return {"mapped_branch": branch, "branch_precision": precision, "branch_recall": recall, "branch_f1": f1, "true_units": true_units, "predicted_units": units}


def replace_edge_frame(frame: pd.DataFrame, receiver_pos: int, old_parent: int, child: int, new_parent: int, candidates: pd.DataFrame) -> pd.DataFrame:
    local = frame["root_pos"].astype(int).eq(receiver_pos)
    remove = local & (
        ((frame["source"].astype(int) == old_parent) & (frame["target"].astype(int) == child))
        | ((frame["source"].astype(int) == child) & (frame["target"].astype(int) == old_parent))
    )
    template = candidates[
        ((candidates["source"].astype(int) == new_parent) & (candidates["target"].astype(int) == child))
        | ((candidates["source"].astype(int) == child) & (candidates["target"].astype(int) == new_parent))
    ].iloc[0]
    new_row = frame.loc[local].iloc[0].copy()
    for column in frame.columns:
        if column in template.index:
            new_row[column] = template[column]
    new_row["source"] = new_parent
    new_row["target"] = child
    new_row["source_name"] = str(template.source_name if int(template.source) == new_parent else template.target_name)
    new_row["target_name"] = str(template.source_name if int(template.source) == child else template.target_name)
    return pd.concat([frame.loc[~remove], pd.DataFrame([new_row])], ignore_index=True)


def oracle_best_legal_swap(
    inferred_t0: pd.DataFrame,
    candidates: pd.DataFrame,
    pseudotime: pd.DataFrame,
    fixed: pd.DataFrame,
    receiver_pos: int,
    localized_units: set[str],
    truth_t1: set[tuple[str, str, str]],
    state_lookup: dict[str, str],
    pos_to_root: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = mapped_edge_set(inferred_t0, state_lookup, pos_to_root)
    baseline_distance = len(baseline ^ truth_t1)
    part = inferred_t0[inferred_t0["root_pos"].astype(int).eq(receiver_pos)]
    graph = nx.Graph()
    graph.add_edges_from(part[["source", "target"]].astype(int).itertuples(index=False, name=None))
    anchor_name = str(fixed.iloc[receiver_pos].root_anchor_id)
    name_to_id = dict(zip(part["source_name"].astype(str), part["source"].astype(int)))
    name_to_id.update(dict(zip(part["target_name"].astype(str), part["target"].astype(int))))
    anchor = name_to_id[anchor_name]
    pt = dict(zip(
        pseudotime[pseudotime["root_pos"].astype(int).eq(receiver_pos)]["node_id"].astype(int),
        pseudotime[pseudotime["root_pos"].astype(int).eq(receiver_pos)]["selected_pseudotime"].astype(float),
    ))
    trials = []
    best_frame = inferred_t0.copy()
    best_gain = 0
    for child_name in sorted(localized_units):
        child = name_to_id.get(child_name)
        if child is None or child == anchor or child not in graph:
            continue
        path = nx.shortest_path(graph, anchor, child)
        if len(path) < 2:
            continue
        old_parent = path[-2]
        without = graph.copy()
        without.remove_edge(old_parent, child)
        subtree = nx.node_connected_component(without, child)
        for edge in candidates.itertuples(index=False):
            endpoints = {int(edge.source), int(edge.target)}
            if child not in endpoints:
                continue
            new_parent = int(edge.target) if int(edge.source) == child else int(edge.source)
            if new_parent in subtree or new_parent == old_parent or new_parent not in graph:
                continue
            if pt.get(new_parent, 1.0) >= pt.get(child, 1.0):
                continue
            candidate_graph = without.copy()
            candidate_graph.add_edge(new_parent, child)
            if not nx.is_tree(candidate_graph):
                continue
            trial = replace_edge_frame(inferred_t0, receiver_pos, old_parent, child, new_parent, candidates)
            mapped = mapped_edge_set(trial, state_lookup, pos_to_root)
            distance = len(mapped ^ truth_t1)
            gain = baseline_distance - distance
            trials.append({
                "child": child_name,
                "old_parent": old_parent,
                "new_parent": new_parent,
                "baseline_topology_distance": baseline_distance,
                "trial_topology_distance": distance,
                "topology_gain": gain,
                "truth_used_for_selection": True,
            })
            if gain > best_gain:
                best_gain = gain
                best_frame = trial
    return best_frame, pd.DataFrame(trials)


def structural_oracle(truth_dir: Path) -> dict[str, object]:
    t0 = pd.read_csv(truth_dir / "TRUE_T0_EDGES.csv")
    t1 = pd.read_csv(truth_dir / "TRUE_T1_EDGES.csv")
    removed = pd.read_csv(truth_dir / "TRUE_REMOVED_EDGES.csv")
    added = pd.read_csv(truth_dir / "TRUE_ADDED_EDGES.csv")
    edited = t0.copy()
    for edge in removed.itertuples(index=False):
        mask = (
            edited["root_id"].astype(str).eq(str(edge.root_id))
            & edited.apply(lambda row: undirected_pair(row.source, row.target) == undirected_pair(edge.source, edge.target), axis=1)
        )
        edited = edited.loc[~mask].copy()
    for edge in added.itertuples(index=False):
        edited = pd.concat([edited, pd.DataFrame([{"root_id": edge.root_id, "source": edge.source, "target": edge.target, "edge_role": f"oracle:{edge.event_id}"}])], ignore_index=True)
    all_trees = True
    for _, part in edited.groupby("root_id"):
        graph = nx.Graph()
        graph.add_edges_from(part[["source", "target"]].astype(str).itertuples(index=False, name=None))
        all_trees = all_trees and nx.is_tree(graph)
    set_t0, set_t1, set_edited = truth_edge_set(t0), truth_edge_set(t1), truth_edge_set(edited)
    return {
        "structural_oracle_exact_T1": set_edited == set_t1,
        "structural_oracle_all_roots_are_trees": bool(all_trees),
        "T0_to_T1_distance": len(set_t0 ^ set_t1),
        "edited_to_T1_distance": len(set_edited ^ set_t1),
        "topology_gain": len(set_t0 ^ set_t1) - len(set_edited ^ set_t1),
        "truth_used": True,
    }


def run_setting(run_dir: Path, setting: str, output_dir: Path, purity: float, clip_fraction: float) -> dict[str, object]:
    truth_dir = run_dir / "event" / "truth"
    metacell_dir = run_dir / "formal_metacell"
    upstream_root = run_dir / "generic_upstream_v1_exact_root"
    upstream = upstream_root / setting
    relation = run_dir / "generic_relation_refinement_v1"
    fixed = pd.read_csv(upstream_root / "FIXED_ROOT_INPUT.csv")
    membership = pd.read_csv(upstream / "membership_A_sharp.csv")
    inferred_t0 = pd.read_csv(upstream / "progression_tree_edges.csv")
    candidates = pd.read_csv(upstream / "trajectory_candidate_edges.csv")
    pseudotime = pd.read_csv(upstream / "pseudotime_table.csv")
    pseudotime = pseudotime[pseudotime["tree_label"].astype(str).eq("progression")].copy()
    branch_composition = pd.read_csv(metacell_dir / "METACELL_BRANCH_COMPOSITION.csv")
    state_composition = pd.read_csv(metacell_dir / "METACELL_STATE_COMPOSITION.csv")
    summary = pd.read_csv(metacell_dir / "METACELL_TRUTH_SUMMARY.csv")
    events = pd.read_csv(truth_dir / "TRUE_EVENTS.csv")
    causal = events[events["causal"].map(bool_value)]
    representability = candidate_representability(truth_dir, state_composition, candidates, inferred_t0)
    representability.insert(0, "setting_id", setting)
    representability.to_csv(output_dir / f"{setting}_CANDIDATE_REPRESENTABILITY.csv", index=False)
    structural = structural_oracle(truth_dir) if len(causal) else {
        "structural_oracle_exact_T1": True,
        "structural_oracle_all_roots_are_trees": True,
        "T0_to_T1_distance": 0,
        "edited_to_T1_distance": 0,
        "topology_gain": 0,
        "truth_used": True,
    }
    detector = pd.read_csv(relation / "RELATION_DIAGNOSTICS.csv")
    pos_to_root = dict(enumerate(fixed["truth_root_id"].astype(str)))
    root_to_pos = {root: pos for pos, root in pos_to_root.items()}
    target_detector = []
    oracle_rows = []

    for event in events.itertuples(index=False):
        source_pos = root_to_pos[str(event.source_root)]
        receiver_pos = root_to_pos[str(event.receiver_root)]
        row = detector[
            detector["source_root_pos"].astype(int).eq(source_pos)
            & detector["receiver_root_pos"].astype(int).eq(receiver_pos)
        ].iloc[0]
        rank = int(
            detector[detector["status"].astype(str).eq("tested")]
            .sort_values("effect_size", ascending=False)
            .reset_index(drop=True)
            .index[
                detector[detector["status"].astype(str).eq("tested")]
                .sort_values("effect_size", ascending=False)
                .reset_index(drop=True)
                .apply(lambda candidate: int(candidate.source_root_pos) == source_pos and int(candidate.receiver_root_pos) == receiver_pos, axis=1)
            ][0]
        ) + 1
        target_detector.append({
            "setting_id": setting,
            "source_root": event.source_root,
            "receiver_root": event.receiver_root,
            "causal": bool_value(event.causal),
            "effect_size": float(row.effect_size),
            "permutation_p": float(row.permutation_p),
            "q_value": float(row.q_value),
            "is_called": bool_value(row.is_called),
            "rank_among_ordered_pairs": rank,
            "n_tested_pairs": int(detector["status"].astype(str).eq("tested").sum()),
            "truth_used_for_evaluation_only": True,
        })
        if not bool_value(event.causal):
            continue
        mapping = map_affected_branch(
            str(event.receiver_root), str(event.affected_branch), receiver_pos,
            inferred_t0, branch_composition, purity,
        )
        node_score = dict(zip(summary["metacell_id"].astype(str), summary["event_exposed_fraction"].astype(float)))
        called = pd.DataFrame([{
            "source_root_pos": source_pos,
            "receiver_root_pos": receiver_pos,
            "effect_size": float(event.effect_strength),
            "affected_branch": mapping["mapped_branch"],
            "event_type": str(event.event_type),
            "is_called": True,
        }])
        current_refined, changes = refine_trees(
            inferred_t0, candidates, called, {(source_pos, receiver_pos): node_score},
            fixed, pseudotime, clip_fraction,
        )
        percentile_refined, percentile_changes = refine_trees(
            inferred_t0, candidates, called, {(source_pos, receiver_pos): node_score},
            fixed, pseudotime, clip_fraction, evidence_scaling="branch_percentile",
        )
        state_lookup = dominant_state_lookup(state_composition, "T1")
        truth_t1 = truth_edge_set(pd.read_csv(truth_dir / "TRUE_T1_EDGES.csv"))
        before_distance = len(mapped_edge_set(inferred_t0, state_lookup, pos_to_root) ^ truth_t1)
        current_distance = len(mapped_edge_set(current_refined, state_lookup, pos_to_root) ^ truth_t1)
        percentile_distance = len(mapped_edge_set(percentile_refined, state_lookup, pos_to_root) ^ truth_t1)
        upper_frame, trials = oracle_best_legal_swap(
            inferred_t0, candidates, pseudotime, fixed, receiver_pos,
            mapping["predicted_units"], truth_t1, state_lookup, pos_to_root,
        )
        if len(trials):
            trials["event_evidence"] = trials["child"].astype(str).map(node_score).fillna(0.0)
            trials["event_supported"] = trials["event_evidence"].astype(float).gt(0.0)
        event_supported_trials = trials[trials.get("event_supported", pd.Series(False, index=trials.index)).astype(bool)]
        event_supported_upper_gain = (
            float(event_supported_trials["topology_gain"].max()) if len(event_supported_trials) else float("nan")
        )
        upper_distance = len(mapped_edge_set(upper_frame, state_lookup, pos_to_root) ^ truth_t1)
        trials.insert(0, "setting_id", setting)
        trials.to_csv(output_dir / f"{setting}_ORACLE_LEGAL_SWAP_TRIALS.csv", index=False)
        setting_output = output_dir / setting
        prediction_dir = setting_output / "O1_CURRENT_OPERATOR_PREDICTION"
        metacell_adata = ad.read_h5ad(upstream / "metacell_adata.h5ad")
        export_prediction(
            prediction_dir, metacell_adata, fixed, membership, inferred_t0, current_refined,
            called, candidates, pseudotime,
        )
        score(
            truth_dir, prediction_dir, setting_output / "O1_CURRENT_OPERATOR_SCORE",
            f"O1_oracle_event_current_operator_{setting}", metacell_dir,
        )
        percentile_prediction_dir = setting_output / "O1_PERCENTILE_OPERATOR_PREDICTION"
        export_prediction(
            percentile_prediction_dir, metacell_adata, fixed, membership, inferred_t0,
            percentile_refined, called, candidates, pseudotime,
        )
        percentile_score_dir = setting_output / "O1_PERCENTILE_OPERATOR_SCORE"
        score(
            truth_dir, percentile_prediction_dir, percentile_score_dir,
            f"O1_oracle_event_percentile_operator_{setting}", metacell_dir,
        )
        percentile_score = pd.read_csv(percentile_score_dir / "SCORE_SUMMARY.csv").iloc[0]
        upper_prediction_dir = setting_output / "O1_TRUTH_SELECTED_LEGAL_SWAP_PREDICTION"
        export_prediction(
            upper_prediction_dir, metacell_adata, fixed, membership, inferred_t0,
            upper_frame, called, candidates, pseudotime,
        )
        upper_score_dir = setting_output / "O1_TRUTH_SELECTED_LEGAL_SWAP_SCORE"
        score(
            truth_dir, upper_prediction_dir, upper_score_dir,
            f"O1_truth_selected_legal_swap_{setting}", metacell_dir,
        )
        upper_score = pd.read_csv(upper_score_dir / "SCORE_SUMMARY.csv").iloc[0]
        oracle_rows.append({
            "setting_id": setting,
            "event_id": event.event_id,
            "mapped_predicted_branch": mapping["mapped_branch"],
            "branch_mapping_precision": mapping["branch_precision"],
            "branch_mapping_recall": mapping["branch_recall"],
            "branch_mapping_f1": mapping["branch_f1"],
            "n_true_branch_metacells": len(mapping["true_units"]),
            "n_predicted_branch_metacells": len(mapping["predicted_units"]),
            "current_operator_accepted_rewire": bool(len(changes) and changes.get("accepted", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
            "inferred_T0_distance": before_distance,
            "current_operator_T1_distance": current_distance,
            "current_operator_topology_gain": before_distance - current_distance,
            "percentile_operator_accepted_rewire": bool(
                len(percentile_changes)
                and percentile_changes.get("accepted", pd.Series(dtype=bool)).fillna(False).astype(bool).any()
            ),
            "percentile_operator_T1_distance": percentile_distance,
            "percentile_operator_topology_gain": before_distance - percentile_distance,
            "percentile_operator_scorer_topology_gain": float(percentile_score.rewiring_gain_delta_d),
            "percentile_operator_scorer_T0_distance": float(percentile_score.distance_T0_to_true_T1),
            "percentile_operator_scorer_T1_distance": float(percentile_score.distance_T1_to_true_T1),
            "oracle_best_legal_T1_distance": upper_distance,
            "oracle_best_legal_topology_gain": before_distance - upper_distance,
            "oracle_best_legal_scorer_topology_gain": float(upper_score.rewiring_gain_delta_d),
            "oracle_best_event_supported_topology_gain": event_supported_upper_gain,
            "n_legal_local_swaps": len(trials),
            "truth_used": True,
        })

    pd.DataFrame(target_detector).to_csv(output_dir / f"{setting}_O3_DETECTOR_ONLY.csv", index=False)
    pd.DataFrame(oracle_rows).to_csv(output_dir / f"{setting}_O1_RESULTS.csv", index=False)
    return {
        "setting_id": setting,
        **{f"O2_{key}": value for key, value in structural.items() if key != "truth_used"},
        "candidate_added_edge_available": bool(
            len(representability[representability["edge_role"].eq("added")])
            and representability.loc[representability["edge_role"].eq("added"), "in_candidate_graph"].all()
        ),
        "candidate_removed_edge_present_in_T0": bool(
            len(representability[representability["edge_role"].eq("removed")])
            and representability.loc[representability["edge_role"].eq("removed"), "in_inferred_T0"].all()
        ),
        "truth_used": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--settings", nargs="+", default=["CORE_19", "CORE_05", "CORE_06"])
    parser.add_argument("--replicate-dir", default="replicate_01_seed_7112101")
    parser.add_argument("--truth-purity", type=float, default=0.70)
    parser.add_argument("--clip-fraction", type=float, default=0.35)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for setting in args.settings:
        run_dir = args.protocol_run_root / setting / args.replicate_dir
        summaries.append(run_setting(run_dir, setting, args.output_dir, args.truth_purity, args.clip_fraction))
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "ORACLE_DECOMPOSITION_SUMMARY.csv", index=False)
    status = {
        "protocol": "ORACLE_DECOMPOSITION_SPEC_V1",
        "truth_used": True,
        "settings": args.settings,
        "truth_purity": args.truth_purity,
        "clip_fraction": args.clip_fraction,
        "completed": True,
    }
    (args.output_dir / "RUN_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
