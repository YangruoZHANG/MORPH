#!/usr/bin/env python3
"""Probability-aware Phase C topology scorer for mixed metacells.

Construction and edge selection are untouched.  The scorer maps each fine
edge to a distribution over truth-state edges, then aggregates repeated
support as one minus the probability that no fine edge realizes that state
edge.  Dominant-state metrics remain in the operator output as a strict audit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def pair(left: object, right: object) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def truth_edges(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    return {
        (str(row.root_id), *pair(row.source, row.target))
        for row in frame.itertuples(index=False)
    }


def state_probabilities(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    result: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in frame.itertuples(index=False):
        probability = float(row.state_probability)
        if probability > 0:
            result[(str(row.metacell_id), str(row.root_id))][str(row.state_id)] = probability
    return result


def aggregate_edge_support(
    edges: pd.DataFrame,
    probabilities: dict[tuple[str, str], dict[str, float]],
) -> dict[tuple[str, str, str], float]:
    no_support: dict[tuple[str, str, str], float] = defaultdict(lambda: 1.0)
    for row in edges.itertuples(index=False):
        root = str(row.root_id)
        source = probabilities.get((str(row.source_name), root), {})
        target = probabilities.get((str(row.target_name), root), {})
        local: dict[tuple[str, str, str], float] = defaultdict(float)
        for source_state, source_probability in source.items():
            for target_state, target_probability in target.items():
                if source_state == target_state:
                    continue
                key = (root, *pair(source_state, target_state))
                local[key] += source_probability * target_probability
        for key, probability in local.items():
            no_support[key] *= 1.0 - min(max(float(probability), 0.0), 1.0)
    return {key: 1.0 - value for key, value in no_support.items()}


def l1_truth_distance(
    support: dict[tuple[str, str, str], float], truth: set[tuple[str, str, str]]
) -> float:
    universe = set(support) | set(truth)
    return float(sum(abs(support.get(edge, 0.0) - float(edge in truth)) for edge in universe))


def soft_prf(
    predicted: dict[tuple[str, str, str], float],
    truth: set[tuple[str, str, str]],
) -> tuple[float, float, float]:
    mass = float(sum(predicted.values()))
    true_positive = float(sum(predicted.get(edge, 0.0) for edge in truth))
    precision = true_positive / mass if mass > 0 else (1.0 if not truth else 0.0)
    recall = true_positive / len(truth) if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase-a-dir", type=Path, required=True)
    parser.add_argument("--phase-c-dir", type=Path, required=True)
    args = parser.parse_args()

    truth_dir = args.run_dir / "event" / "truth"
    events = pd.read_csv(truth_dir / "TRUE_EVENTS.csv")
    true_t0 = truth_edges(pd.read_csv(truth_dir / "TRUE_T0_EDGES.csv"))
    true_t1 = truth_edges(pd.read_csv(truth_dir / "TRUE_T1_EDGES.csv"))
    true_added, true_removed = true_t1 - true_t0, true_t0 - true_t1
    composition = pd.read_csv(
        args.run_dir / "formal_metacell" / "METACELL_STATE_COMPOSITION.csv"
    )
    probabilities = state_probabilities(
        composition[composition.stage.astype(str).eq("T1")]
    )
    before = pd.read_csv(args.phase_a_dir / "T0_FINE_TRAJECTORY_EDGES.csv")
    before_support = aggregate_edge_support(before, probabilities)
    distance_before = l1_truth_distance(before_support, true_t1)
    branch = pd.read_csv(
        args.run_dir / "formal_metacell" / "METACELL_BRANCH_COMPOSITION.csv"
    )
    branch = branch[branch.stage.astype(str).eq("T0")]
    branch_probability = {
        (str(row.metacell_id), str(row.root_id), str(row.branch_id)): float(row.branch_probability)
        for row in branch.itertuples(index=False)
    }
    operator = pd.read_csv(args.phase_c_dir / "PHASE_C_ORACLE_REFINEMENT.csv")
    operator_lookup = {
        (str(row.event_id), float(row.gamma)): row
        for row in operator.itertuples(index=False)
    }
    refined = pd.read_csv(args.phase_c_dir / "PHASE_C_REFINED_EDGE_SENSITIVITY.csv")
    rows = []
    for (event_id, gamma), after in refined.groupby(["event_id", "gamma"], sort=True):
        event = events[events.event_id.astype(str).eq(str(event_id))].iloc[0]
        root = str(event.receiver_root)
        operator_row = operator_lookup[(str(event_id), float(gamma))]
        child_name = str(getattr(operator_row, "selected_child", "")).split(";")[0]
        old_parent_name = str(getattr(operator_row, "selected_old_parent", "")).split(";")[0]
        new_parent_name = str(getattr(operator_row, "selected_new_parent", "")).split(";")[0]
        if child_name.lower() == "nan":
            child_name = ""
        if old_parent_name.lower() == "nan":
            old_parent_name = ""
        if new_parent_name.lower() == "nan":
            new_parent_name = ""
        child_affected = branch_probability.get(
            (child_name, root, str(event.affected_branch)), 0.0
        )
        old_decision_from = branch_probability.get(
            (old_parent_name, root, str(event.decision_from)), 0.0
        )
        new_decision_to = branch_probability.get(
            (new_parent_name, root, str(event.decision_to)), 0.0
        )
        after_support = aggregate_edge_support(after, probabilities)
        distance_after = l1_truth_distance(after_support, true_t1)
        predicted_added = {
            edge: max(after_support.get(edge, 0.0) - before_support.get(edge, 0.0), 0.0)
            for edge in set(after_support) | set(before_support)
        }
        predicted_removed = {
            edge: max(before_support.get(edge, 0.0) - after_support.get(edge, 0.0), 0.0)
            for edge in set(after_support) | set(before_support)
        }
        predicted_added = {edge: value for edge, value in predicted_added.items() if value > 0}
        predicted_removed = {edge: value for edge, value in predicted_removed.items() if value > 0}
        event_added = {edge for edge in true_added if edge[0] == root}
        event_removed = {edge for edge in true_removed if edge[0] == root}
        add_precision, add_recall, add_f1 = soft_prf(predicted_added, event_added)
        remove_precision, remove_recall, remove_f1 = soft_prf(predicted_removed, event_removed)
        rewire_precision = 0.5 * (add_precision + remove_precision)
        rewire_recall = 0.5 * (add_recall + remove_recall)
        rewire_f1 = (
            2 * rewire_precision * rewire_recall / (rewire_precision + rewire_recall)
            if rewire_precision + rewire_recall else 0.0
        )
        event_added_gain = float(sum(
            after_support.get(edge, 0.0) - before_support.get(edge, 0.0)
            for edge in event_added
        ))
        rows.append({
            "run": args.run_dir.name, "event_id": str(event_id),
            "event_type": str(event.event_type), "causal": bool(event.causal),
            "receiver_root": root, "gamma": float(gamma),
            "probabilistic_topology_distance_before": distance_before,
            "probabilistic_topology_distance_after": distance_after,
            "probabilistic_oracle_topology_gain": distance_before - distance_after,
            "probabilistic_rewiring_precision": rewire_precision,
            "probabilistic_rewiring_recall": rewire_recall,
            "probabilistic_rewiring_F1": rewire_f1,
            "probabilistic_added_edge_F1": add_f1,
            "probabilistic_removed_edge_F1": remove_f1,
            "true_event_added_support_gain": event_added_gain,
            "selected_child_affected_branch_probability": child_affected,
            "selected_old_parent_decision_from_probability": old_decision_from,
            "selected_new_parent_decision_to_probability": new_decision_to,
            "selected_branch_choice_joint_support": child_affected * new_decision_to,
            "truth_used_for_scoring_only": True,
        })
    pd.DataFrame(rows).to_csv(
        args.phase_c_dir / "PHASE_C_PROBABILISTIC_TOPOLOGY_METRICS.csv", index=False
    )


if __name__ == "__main__":
    main()
