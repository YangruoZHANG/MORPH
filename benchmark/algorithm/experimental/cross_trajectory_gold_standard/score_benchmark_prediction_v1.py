#!/usr/bin/env python3
"""Score a prediction directory against benchmark truth contract v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def prf(true_set: set, pred_set: set) -> tuple[float, float, float]:
    tp = len(true_set & pred_set)
    precision = safe_ratio(tp, len(pred_set))
    recall = safe_ratio(tp, len(true_set))
    if not true_set and not pred_set:
        return 1.0, 1.0, 1.0
    if np.isnan(precision) or np.isnan(recall) or precision + recall == 0:
        return (0.0 if pred_set else float("nan"), 0.0 if true_set else float("nan"), 0.0)
    return precision, recall, float(2 * precision * recall / (precision + recall))


def edge_set(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    edges: set[tuple[str, str, str]] = set()
    for row in frame.itertuples(index=False):
        a, b = sorted((str(row.source), str(row.target)))
        edges.add((str(row.root_id), a, b))
    return edges


def map_metacell_edges_to_states(
    frame: pd.DataFrame, state_composition: pd.DataFrame, stage: str
) -> pd.DataFrame:
    state = state_composition[state_composition["stage"].astype(str).eq(stage)].copy()
    state = state.sort_values("state_probability", ascending=False).drop_duplicates("metacell_id")
    lookup = dict(zip(state["metacell_id"].astype(str), state["state_id"].astype(str)))
    mapped = frame.copy()
    mapped["source"] = mapped["source"].astype(str).map(lookup)
    mapped["target"] = mapped["target"].astype(str).map(lookup)
    mapped = mapped.dropna(subset=["source", "target"])
    mapped = mapped[mapped["source"].astype(str) != mapped["target"].astype(str)]
    return mapped[["root_id", "source", "target"]].drop_duplicates()


def event_set(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    return {
        (str(r.event_type), str(r.source_root), str(r.receiver_root))
        for r in frame.itertuples(index=False)
    }


def event_key(row) -> tuple[str, str, str]:
    return str(row.event_type), str(row.source_root), str(row.receiver_root)


def graph_safety(edges: pd.DataFrame) -> dict[str, int]:
    cycles = 0
    disconnected = 0
    for _, group in edges.groupby("root_id"):
        graph = nx.Graph()
        graph.add_edges_from(group[["source", "target"]].astype(str).itertuples(index=False, name=None))
        if len(nx.cycle_basis(graph)) > 0:
            cycles += 1
        if graph.number_of_nodes() and not nx.is_connected(graph):
            disconnected += 1
    return {"cycle_violating_roots": cycles, "disconnected_roots": disconnected}


def require_files(directory: Path, names: list[str]) -> None:
    missing = [name for name in names if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {directory}: {missing}")


def score(
    truth_dir: Path,
    prediction_dir: Path,
    output_dir: Path,
    method: str,
    metacell_truth_dir: Path | None = None,
) -> None:
    truth_required = [
        "TRUE_ROOTS.csv", "TRUE_CELL_ROOT_MEMBERSHIP.csv", "TRUE_CELL_BRANCH.csv",
        "TRUE_PSEUDOTIME.csv", "TRUE_T0_EDGES.csv", "TRUE_T1_EDGES.csv", "TRUE_EVENTS.csv",
    ]
    prediction_required = [
        "PRED_ROOTS.csv", "PRED_CELL_OR_METACELL_ASSIGNMENT.csv", "PRED_BRANCH.csv",
        "PRED_PSEUDOTIME.csv", "PRED_T0_EDGES.csv", "PRED_T1_EDGES.csv", "PRED_EVENTS.csv",
    ]
    require_files(truth_dir, truth_required)
    require_files(prediction_dir, prediction_required)
    output_dir.mkdir(parents=True, exist_ok=True)

    true_roots = set(pd.read_csv(truth_dir / "TRUE_ROOTS.csv")["root_id"].astype(str))
    pred_roots = set(pd.read_csv(prediction_dir / "PRED_ROOTS.csv")["root_id"].astype(str))
    root_p, root_r, root_f1 = prf(true_roots, pred_roots)

    true_membership = pd.read_csv(truth_dir / "TRUE_CELL_ROOT_MEMBERSHIP.csv")
    pred_membership = pd.read_csv(prediction_dir / "PRED_CELL_OR_METACELL_ASSIGNMENT.csv")
    if metacell_truth_dir is not None:
        root_comp = pd.read_csv(metacell_truth_dir / "METACELL_ROOT_COMPOSITION.csv")
        true_hard = root_comp.sort_values("root_probability", ascending=False).drop_duplicates("metacell_id")
        true_hard = true_hard[true_hard["root_probability"].astype(float).ge(0.70)].copy()
        true_hard = true_hard.rename(columns={"metacell_id": "cell_id", "root_probability": "membership_probability"})
    else:
        true_hard = true_membership.sort_values("membership_probability", ascending=False).drop_duplicates("cell_id")
    pred_hard = pred_membership.sort_values("membership_probability", ascending=False).drop_duplicates("unit_id")
    assignment = true_hard.merge(pred_hard, left_on="cell_id", right_on="unit_id", how="left", suffixes=("_true", "_pred"))
    assignment_coverage = float(assignment["root_id_pred"].notna().mean())
    assignment_accuracy = float(
        (assignment.loc[assignment["root_id_pred"].notna(), "root_id_true"] ==
         assignment.loc[assignment["root_id_pred"].notna(), "root_id_pred"]).mean()
    ) if assignment["root_id_pred"].notna().any() else float("nan")

    true_branch = pd.read_csv(truth_dir / "TRUE_CELL_BRANCH.csv")
    true_branch = true_branch[true_branch["stage"] == "T1"].copy()
    if metacell_truth_dir is not None:
        branch_comp = pd.read_csv(metacell_truth_dir / "METACELL_BRANCH_COMPOSITION.csv")
        true_branch = branch_comp[branch_comp["stage"].astype(str).eq("T1")].copy()
        true_branch = true_branch.sort_values("branch_probability", ascending=False).drop_duplicates("metacell_id")
        true_branch = true_branch[true_branch["branch_probability"].astype(float).ge(0.70)].copy()
        true_branch = true_branch.rename(columns={"metacell_id": "cell_id"})
    pred_branch = pd.read_csv(prediction_dir / "PRED_BRANCH.csv")
    branches = true_branch.merge(pred_branch, left_on=["cell_id", "root_id"], right_on=["unit_id", "root_id"], how="inner", suffixes=("_true", "_pred"))
    branch_coverage = safe_ratio(len(branches), len(true_branch))
    branch_ari = float(adjusted_rand_score(branches["branch_id_true"], branches["branch_id_pred"])) if len(branches) >= 2 else float("nan")

    true_pt = pd.read_csv(truth_dir / "TRUE_PSEUDOTIME.csv")
    if metacell_truth_dir is not None:
        meta_summary = pd.read_csv(metacell_truth_dir / "METACELL_TRUTH_SUMMARY.csv")
        true_pt = meta_summary[["metacell_id", "dominant_root", "median_true_pseudotime"]].rename(
            columns={"metacell_id": "cell_id", "dominant_root": "root_id", "median_true_pseudotime": "pseudotime"}
        )
    pred_pt = pd.read_csv(prediction_dir / "PRED_PSEUDOTIME.csv")
    pt = true_pt.merge(pred_pt, left_on=["cell_id", "root_id"], right_on=["unit_id", "root_id"], how="inner", suffixes=("_true", "_pred"))
    pt_scores = []
    for _, group in pt.groupby("root_id"):
        if len(group) >= 3 and group["pseudotime_true"].nunique() > 1 and group["pseudotime_pred"].nunique() > 1:
            pt_scores.append(float(spearmanr(group["pseudotime_true"], group["pseudotime_pred"]).statistic))
    pt_sr = float(np.mean(pt_scores)) if pt_scores else float("nan")
    pt_coverage = safe_ratio(len(pt), len(true_pt))

    true_events_all = pd.read_csv(truth_dir / "TRUE_EVENTS.csv")
    true_events = true_events_all[as_bool(true_events_all["causal"])].copy()
    pred_events_all = pd.read_csv(prediction_dir / "PRED_EVENTS.csv")
    pred_events = pred_events_all[as_bool(pred_events_all["is_called"])].copy()
    event_p, event_r, event_f1 = prf(event_set(true_events), event_set(pred_events))
    component_accuracy = {}
    true_by_key = {event_key(row): row for row in true_events.itertuples(index=False)}
    pred_by_key = {event_key(row): row for row in pred_events.itertuples(index=False)}
    matched_keys = sorted(set(true_by_key) & set(pred_by_key))
    for column in ("event_type", "source_root", "receiver_root"):
        component_accuracy[column] = (
            float(np.mean([
                str(getattr(true_by_key[key], column)) == str(getattr(pred_by_key[key], column))
                for key in matched_keys
            ])) if matched_keys else float("nan")
        )

    branch_localization_scores = []
    for key in matched_keys:
        true_event = true_by_key[key]
        pred_event = pred_by_key[key]
        true_units = set(
            true_branch.loc[
                true_branch["root_id"].astype(str).eq(str(true_event.receiver_root))
                & true_branch["branch_id"].astype(str).eq(str(true_event.affected_branch)),
                "cell_id",
            ].astype(str)
        )
        pred_units = set(
            pred_branch.loc[
                pred_branch["root_id"].astype(str).eq(str(pred_event.receiver_root))
                & pred_branch["branch_id"].astype(str).eq(str(pred_event.affected_branch)),
                "unit_id",
            ].astype(str)
        )
        branch_localization_scores.append(prf(true_units, pred_units))
    if branch_localization_scores:
        affected_branch_precision = float(np.mean([value[0] for value in branch_localization_scores]))
        affected_branch_recall = float(np.mean([value[1] for value in branch_localization_scores]))
        affected_branch_f1 = float(np.mean([value[2] for value in branch_localization_scores]))
    else:
        affected_branch_precision = affected_branch_recall = affected_branch_f1 = float("nan")

    true_t0_df = pd.read_csv(truth_dir / "TRUE_T0_EDGES.csv")
    true_t1_df = pd.read_csv(truth_dir / "TRUE_T1_EDGES.csv")
    pred_t0_df = pd.read_csv(prediction_dir / "PRED_T0_EDGES.csv")
    pred_t1_df = pd.read_csv(prediction_dir / "PRED_T1_EDGES.csv")
    pred_t1_native_df = pred_t1_df.copy()
    if metacell_truth_dir is not None:
        state_comp = pd.read_csv(metacell_truth_dir / "METACELL_STATE_COMPOSITION.csv")
        # Use one fixed T1 state projection for both predicted stages so that
        # delta_d measures structural rewiring, not a truth-label change.
        pred_t0_df = map_metacell_edges_to_states(pred_t0_df, state_comp, "T1")
        pred_t1_df = map_metacell_edges_to_states(pred_t1_df, state_comp, "T1")
    true_t0, true_t1 = edge_set(true_t0_df), edge_set(true_t1_df)
    pred_t0, pred_t1 = edge_set(pred_t0_df), edge_set(pred_t1_df)
    true_removed, true_added = true_t0 - true_t1, true_t1 - true_t0
    pred_removed, pred_added = pred_t0 - pred_t1, pred_t1 - pred_t0
    removed_p, removed_r, removed_f1 = prf(true_removed, pred_removed)
    added_p, added_r, added_f1 = prf(true_added, pred_added)
    topology_p, topology_r, topology_f1 = prf(true_t1, pred_t1)
    d_t0 = len(pred_t0 ^ true_t1)
    d_t1 = len(pred_t1 ^ true_t1)
    delta_d = d_t0 - d_t1
    normalized_delta_d = safe_ratio(delta_d, d_t0)
    safety = graph_safety(pred_t1_native_df)
    no_causal_truth = len(true_events) == 0
    false_rewiring_under_null = int(no_causal_truth and pred_t0 != pred_t1)
    null_called = pred_events["event_type"].astype(str).eq("null_contact").sum() if len(pred_events) else 0
    called_events_under_global_null = int(len(pred_events)) if no_causal_truth else 0

    summary = {
        "method": method,
        "evaluation_unit": "metacell" if metacell_truth_dir is not None else "cell_or_native",
        "forest_root_precision": root_p,
        "forest_root_recall": root_r,
        "forest_root_F1": root_f1,
        "assignment_accuracy": assignment_accuracy,
        "assignment_coverage": assignment_coverage,
        "forest_branch_ARI": branch_ari,
        "branch_coverage": branch_coverage,
        "forest_pt_SR": pt_sr,
        "pseudotime_coverage": pt_coverage,
        "event_precision": event_p,
        "event_recall": event_r,
        "event_F1": event_f1,
        "event_type_accuracy": component_accuracy["event_type"],
        "sender_accuracy": component_accuracy["source_root"],
        "receiver_accuracy": component_accuracy["receiver_root"],
        "affected_branch_precision": affected_branch_precision,
        "affected_branch_recall": affected_branch_recall,
        "affected_branch_cell_F1": affected_branch_f1,
        "removed_edge_precision": removed_p,
        "removed_edge_recall": removed_r,
        "removed_edge_F1": removed_f1,
        "added_edge_precision": added_p,
        "added_edge_recall": added_r,
        "added_edge_F1": added_f1,
        "final_topology_precision": topology_p,
        "final_topology_recall": topology_r,
        "final_topology_F1": topology_f1,
        "distance_T0_to_true_T1": d_t0,
        "distance_T1_to_true_T1": d_t1,
        "rewiring_gain_delta_d": delta_d,
        "rewiring_gain_normalized": normalized_delta_d,
        "cycle_violating_roots": safety["cycle_violating_roots"],
        "disconnected_roots": safety["disconnected_roots"],
        "false_rewiring_under_global_null": false_rewiring_under_null,
        "called_null_contacts": int(null_called),
        "called_events_under_global_null": called_events_under_global_null,
    }
    pd.DataFrame([summary]).to_csv(output_dir / "SCORE_SUMMARY.csv", index=False)
    pd.DataFrame([{
        "method": method,
        "n_true_causal_events": len(true_events),
        "n_called_events": len(pred_events),
        "n_true_removed_edges": len(true_removed),
        "n_pred_removed_edges": len(pred_removed),
        "n_true_added_edges": len(true_added),
        "n_pred_added_edges": len(pred_added),
        **safety,
        "false_rewiring_under_global_null": false_rewiring_under_null,
    }]).to_csv(output_dir / "SAFETY_AUDIT.csv", index=False)
    (output_dir / "SCORE_SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--metacell-truth-dir", type=Path)
    args = parser.parse_args()
    score(
        Path(args.truth_dir),
        Path(args.prediction_dir),
        Path(args.output_dir),
        args.method,
        args.metacell_truth_dir,
    )


if __name__ == "__main__":
    main()
