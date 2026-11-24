#!/usr/bin/env python3
"""Independently aggregate the frozen Phase E locked validation grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


EXPECTED_STAGES = {f"M{i}" for i in range(7)}
BOOTSTRAP_SEED = 8122026


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("No expected path exists: " + ", ".join(map(str, paths)))


def replicate_from(run: Path) -> int:
    match = re.search(r"replicate_(\d+)", run.name)
    if not match:
        raise ValueError(f"No replicate token in {run}")
    return int(match.group(1))


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def hash_signature(source_hashes: dict[str, str]) -> str:
    payload = json.dumps(source_hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def localized_cells(call: pd.Series) -> set[str]:
    text = str(call.get("localized_support_cell_ids", ""))
    return {item for item in text.split(";") if item and item.lower() != "nan"}


def event_region_metrics(
    call: pd.Series, truth: pd.Series, obs: pd.DataFrame
) -> tuple[float, float, float, str, float]:
    predicted = localized_cells(call)
    labels = obs.get("causal_event_id", pd.Series("", index=obs.index)).astype(str)
    active = set(obs.index[labels.str.contains(str(truth.event_id), regex=False)])
    overlap = len(predicted & active)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(active) if active else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mapped_branch = "unmapped"
    branch_correct = 0.0
    if predicted and "truth_branch_T0" in obs:
        present = list(predicted & set(obs.index.astype(str)))
        if present:
            mapped_branch = str(obs.loc[present, "truth_branch_T0"].astype(str).mode().iloc[0])
            branch_correct = float(mapped_branch == str(truth.affected_branch))
    return precision, recall, f1, mapped_branch, branch_correct


def greedy_matches(calls: pd.DataFrame, truth: pd.DataFrame) -> list[tuple[int, int]]:
    candidates: list[tuple[float, float, int, int]] = []
    for ci, call in calls.iterrows():
        call_center = 0.5 * (
            float(call.receiver_window_start) + float(call.receiver_window_end)
        )
        for ti, event in truth.iterrows():
            exact = (
                str(call.source_root) == str(event.source_root)
                and str(call.receiver_root) == str(event.receiver_root)
                and str(call.event_type) == str(event.event_type)
            )
            if not exact:
                continue
            truth_center = 0.5 * (
                float(event.receiver_tau_start) + float(event.receiver_tau_end)
            )
            candidates.append((
                abs(call_center - truth_center),
                -float(call.get("call_strength", 0.0)),
                int(ci),
                int(ti),
            ))
    matches: list[tuple[int, int]] = []
    used_calls: set[int] = set()
    used_truth: set[int] = set()
    for _, _, ci, ti in sorted(candidates):
        if ci not in used_calls and ti not in used_truth:
            matches.append((ci, ti))
            used_calls.add(ci)
            used_truth.add(ti)
    return matches


def score_calls(
    evidence: pd.DataFrame,
    truth: pd.DataFrame,
    obs: pd.DataFrame,
    call_column: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    causal = truth[truth.causal.map(as_bool)].reset_index(drop=True)
    if call_column not in evidence:
        calls = evidence.iloc[0:0].copy()
    else:
        calls = evidence[evidence[call_column].map(as_bool)].reset_index(drop=True)
    matches = greedy_matches(calls, causal)
    tp = len(matches)
    fp = len(calls) - tp
    fn = len(causal) - tp
    precision, recall, f1 = prf(tp, fp, fn)
    details: list[dict[str, object]] = []
    for ci, ti in matches:
        call = calls.iloc[ci]
        event = causal.iloc[ti]
        region_p, region_r, region_f1, mapped_branch, branch_ok = event_region_metrics(
            call, event, obs
        )
        pred_center = 0.5 * (
            float(call.receiver_window_start) + float(call.receiver_window_end)
        )
        truth_center = 0.5 * (
            float(event.receiver_tau_start) + float(event.receiver_tau_end)
        )
        details.append({
            "call_column": call_column,
            "event_id": str(event.event_id),
            "truth_event_type": str(event.event_type),
            "source_root": str(event.source_root),
            "receiver_root": str(event.receiver_root),
            "truth_affected_branch": str(event.affected_branch),
            "predicted_affected_branch": str(call.affected_branch),
            "mapped_truth_branch": mapped_branch,
            "affected_branch_choice_correct": branch_ok,
            "event_region_precision": region_p,
            "event_region_recall": region_r,
            "event_region_F1": region_f1,
            "event_location_error": abs(pred_center - truth_center),
            "call_strength": float(call.get("call_strength", np.nan)),
            "joint_dependency_statistic": float(
                call.get("joint_dependency_statistic", np.nan)
            ),
        })
    return ({
        "n_true_events": int(len(causal)),
        "n_calls": int(len(calls)),
        "event_tp": int(tp),
        "event_fp": int(fp),
        "event_fn": int(fn),
        "event_precision": precision,
        "event_recall": recall,
        "event_F1": f1,
    }, details)


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def paired_seed_bootstrap(run_frame: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    replicates = np.array(sorted(run_frame.replicate.unique()), dtype=int)
    records: list[dict[str, float]] = []
    for _ in range(n_boot):
        sampled = rng.choice(replicates, size=len(replicates), replace=True)
        frame = pd.concat(
            [run_frame[run_frame.replicate.eq(int(rep))] for rep in sampled],
            ignore_index=True,
        )
        causal = frame[frame.causal_scenario]
        p, r, f1 = prf(
            int(frame.event_tp.sum()),
            int(frame.event_fp.sum()),
            int(frame.event_fn.sum()),
        )
        records.append({
            "event_precision": p,
            "event_recall": r,
            "event_F1": f1,
            "mean_causal_t1_gain": float(causal.inferred_topology_gain.mean()),
            "mean_causal_tfinal_gain": float(causal.tfinal_topology_gain.mean()),
            "null_false_call_rate": float(
                frame.loc[~frame.causal_scenario, "n_calls"].gt(0).mean()
            ),
            "null_false_rewire_rate": float(
                frame.loc[~frame.causal_scenario, "n_accepted_relation_events"].gt(0).mean()
            ),
        })
    draws = pd.DataFrame(records)
    rows: list[dict[str, object]] = []
    for column in draws:
        low, high = percentile_ci(draws[column].to_numpy(float))
        rows.append({
            "metric": column,
            "estimate": float(draws[column].mean()),
            "ci_lower": low,
            "ci_upper": high,
            "bootstrap_unit": "replicate_seed_across_complete_scenario_grid",
            "n_bootstrap": n_boot,
            "bootstrap_seed": BOOTSTRAP_SEED,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--relation-subdir", default="phase_b_v3_8_frozen")
    parser.add_argument("--phase-d-subdir", default="phase_d_v5_frozen")
    parser.add_argument("--ablation-subdir", default="phase_d_m0_m6_frozen")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    matched_rows: list[dict[str, object]] = []
    stage_rows: list[pd.DataFrame] = []
    hash_rows: list[dict[str, object]] = []
    statuses = sorted(args.locked_root.rglob("LOCKED_CASE_STATUS.json"))
    for status_file in statuses:
        run = status_file.parent
        status = json.loads(status_file.read_text(encoding="utf-8"))
        scenario = json.loads((run / "SCENARIO_CONFIG.json").read_text(encoding="utf-8"))
        replicate = replicate_from(run)
        setting = str(status["setting"])
        truth_file = first_existing([
            run / "event" / "truth" / "TRUE_EVENTS.csv",
            run / "event" / "ground_truth_events.csv",
        ])
        h5ad_file = first_existing([
            run / "event" / "data" / "synthetic.h5ad",
            run / "event" / "synthetic.h5ad",
        ])
        truth = pd.read_csv(truth_file)
        evidence = pd.read_csv(
            run / args.relation_subdir / "PHASE_B_CMI_E_EVIDENCE.csv"
        )
        obs = ad.read_h5ad(h5ad_file, backed="r").obs.copy()
        full, details = score_calls(evidence, truth, obs, "is_called")
        minus, _ = score_calls(evidence, truth, obs, "raw_is_called_without_cmi_e")
        for row in details:
            row.update({"setting": setting, "replicate": replicate})
            matched_rows.append(row)
        d_metrics = pd.read_csv(
            run / args.phase_d_subdir / "PHASE_D_INFERRED_REFINEMENT_METRICS.csv"
        ).iloc[0]
        stage = pd.read_csv(
            run / args.ablation_subdir / "PHASE_D_STAGE_ABLATION.csv"
        )
        stage.insert(1, "replicate", replicate)
        stage["family"] = str(scenario["family"])
        stage["event_type_scenario"] = str(scenario["event_type"])
        stage["effect_level"] = str(scenario["effect_level"])
        stage_rows.append(stage)
        signature = hash_signature(status["source_hashes"])
        hash_rows.append({
            "setting": setting,
            "replicate": replicate,
            "hash_signature": signature,
            "complete": bool(status.get("complete", False)),
            "method_version": str(status["refinement_operator"]),
            "relation_detector": str(status["relation_detector"]),
            "truth_generator": str(status["truth_generator"]),
            "benchmark_specification": str(status["benchmark_specification"]),
        })
        causal_scenario = bool(truth.causal.map(as_bool).any())
        intended_pairs = {
            (str(row.source_root), str(row.receiver_root))
            for row in truth.itertuples(index=False)
        }
        intended_pair_scores = evidence[
            evidence.apply(
                lambda row: (str(row.source_root), str(row.receiver_root)) in intended_pairs,
                axis=1,
            )
        ].joint_dependency_statistic.astype(float)
        run_rows.append({
            "setting": setting,
            "replicate": replicate,
            "seed": int(status["seed"]),
            "family": str(scenario["family"]),
            "scenario_event_type": str(scenario["event_type"]),
            "effect_level": str(scenario["effect_level"]),
            "effect_strength": float(scenario["effect_strength"]),
            "n_roots": int(scenario["n_roots"]),
            "causal_scenario": causal_scenario,
            **full,
            "minus_cmi_e_n_calls": int(minus["n_calls"]),
            "minus_cmi_e_event_tp": int(minus["event_tp"]),
            "minus_cmi_e_event_fp": int(minus["event_fp"]),
            "minus_cmi_e_event_fn": int(minus["event_fn"]),
            "max_true_pair_dependency": (
                float(intended_pair_scores.max()) if len(intended_pair_scores) else np.nan
            ),
            "n_relation_calls_d": int(d_metrics.n_relation_calls),
            "n_accepted_relation_events": int(d_metrics.n_accepted_relation_events),
            "significant_but_no_rewire": int(d_metrics.significant_but_no_rewire),
            "topology_distance_before": float(d_metrics.topology_distance_before),
            "topology_distance_after": float(d_metrics.topology_distance_after),
            "inferred_topology_gain": float(d_metrics.inferred_topology_gain),
            "tfinal_topology_distance": float(d_metrics.tfinal_topology_distance),
            "tfinal_topology_gain": float(d_metrics.tfinal_topology_gain),
            "rewiring_precision": float(d_metrics.rewiring_precision),
            "rewiring_recall": float(d_metrics.rewiring_recall),
            "rewiring_F1": float(d_metrics.rewiring_F1),
            "candidate_legal": as_bool(d_metrics.candidate_legal),
            "connected": as_bool(d_metrics.connected),
            "acyclic": as_bool(d_metrics.acyclic),
            "upstream_violation": as_bool(d_metrics.upstream_violation),
            "tfinal_candidate_legal": as_bool(d_metrics.tfinal_candidate_legal),
            "tfinal_connected": as_bool(d_metrics.tfinal_connected),
            "tfinal_acyclic": as_bool(d_metrics.tfinal_acyclic),
            "event_added_edges": int(d_metrics.event_added_edges),
            "event_added_edges_retained_after_pruning": int(
                d_metrics.event_added_edges_retained_after_pruning
            ),
            "event_boundary_retained": as_bool(d_metrics.event_boundary_retained),
            "hash_signature": signature,
        })

    runs = pd.DataFrame(run_rows).sort_values(["setting", "replicate"])
    matches = pd.DataFrame(matched_rows)
    stages = pd.concat(stage_rows, ignore_index=True)
    hashes = pd.DataFrame(hash_rows).sort_values(["setting", "replicate"])
    runs.to_csv(args.output / "PHASE_E_RUN_LEVEL_RESULTS.csv", index=False)
    matches.to_csv(args.output / "PHASE_E_EVENT_DETECTION.csv", index=False)
    stages.to_csv(args.output / "PHASE_E_STAGE_ABLATION.csv", index=False)
    hashes.to_csv(args.output / "PHASE_E_REPRODUCIBILITY_AUDIT.csv", index=False)

    scenario_summary = runs.groupby(
        ["setting", "family", "scenario_event_type", "effect_level", "causal_scenario"],
        sort=True,
    ).agg(
        n_runs=("replicate", "size"),
        event_precision=("event_precision", "mean"),
        event_recall=("event_recall", "mean"),
        event_F1=("event_F1", "mean"),
        call_rate=("n_calls", lambda values: float(np.mean(np.asarray(values) > 0))),
        rewire_rate=(
            "n_accepted_relation_events",
            lambda values: float(np.mean(np.asarray(values) > 0)),
        ),
        mean_t1_gain=("inferred_topology_gain", "mean"),
        median_t1_gain=("inferred_topology_gain", "median"),
        positive_t1_gain_rate=(
            "inferred_topology_gain", lambda values: float(np.mean(np.asarray(values) > 0))
        ),
        negative_t1_gain_rate=(
            "inferred_topology_gain", lambda values: float(np.mean(np.asarray(values) < 0))
        ),
        mean_tfinal_gain=("tfinal_topology_gain", "mean"),
    ).reset_index()
    scenario_summary.to_csv(args.output / "PHASE_E_SCENARIO_SUMMARY.csv", index=False)

    stage_summary = stages.groupby("stage", sort=False).agg(
        n_runs=("dataset", "size"),
        root_assignment_accuracy=("root_assignment_accuracy", "mean"),
        root_assignment_brier=("root_assignment_brier", "mean"),
        fine_edge_F1=("fine_edge_F1", "mean"),
        branch_ARI=("branch_ARI_rootwise", "mean"),
        pseudotime_SR=("pseudotime_spearman", "mean"),
        topology_F1=("topology_F1", "mean"),
        topology_distance=("topology_distance_to_true_T1", "mean"),
        event_F1=("event_F1", "mean"),
        rewiring_gain=("rewiring_gain", "mean"),
        false_rewire=("false_rewire", "sum"),
    ).reset_index()
    stage_summary.to_csv(args.output / "PHASE_E_STAGE_ABLATION_SUMMARY.csv", index=False)
    m4 = stages[stages.stage.eq("M4")].set_index(["dataset", "replicate"])
    m5 = stages[stages.stage.eq("M5")].set_index(["dataset", "replicate"])
    paired = pd.DataFrame({
        "topology_F1_M4": m4.topology_F1,
        "topology_F1_M5": m5.topology_F1,
        "topology_distance_M4": m4.topology_distance_to_true_T1,
        "topology_distance_M5": m5.topology_distance_to_true_T1,
    }).reset_index()
    paired["topology_F1_delta_M5_minus_M4"] = (
        paired.topology_F1_M5 - paired.topology_F1_M4
    )
    paired["topology_distance_gain_M4_minus_M5"] = (
        paired.topology_distance_M4 - paired.topology_distance_M5
    )
    paired.to_csv(args.output / "PHASE_E_M4_M5_PAIRED.csv", index=False)

    nulls = runs[~runs.causal_scenario].copy()
    nulls["false_call"] = nulls.n_calls.gt(0)
    nulls["false_rewire"] = nulls.n_accepted_relation_events.gt(0)
    nulls.to_csv(args.output / "PHASE_E_NULL_SAFETY.csv", index=False)
    bootstrap = paired_seed_bootstrap(runs, args.bootstrap)
    bootstrap.to_csv(args.output / "PHASE_E_BOOTSTRAP_CI.csv", index=False)

    tp, fp, fn = int(runs.event_tp.sum()), int(runs.event_fp.sum()), int(runs.event_fn.sum())
    event_precision, event_recall, event_f1 = prf(tp, fp, fn)
    minus_p, minus_r, minus_f1 = prf(
        int(runs.minus_cmi_e_event_tp.sum()),
        int(runs.minus_cmi_e_event_fp.sum()),
        int(runs.minus_cmi_e_event_fn.sum()),
    )
    causal = runs[runs.causal_scenario]
    null_false_call_rate = float(nulls.false_call.mean())
    null_false_rewire_rate = float(nulls.false_rewire.mean())
    gain_ci = bootstrap.set_index("metric").loc["mean_causal_t1_gain"]
    strength = runs.groupby("effect_level").max_true_pair_dependency.mean().to_dict()
    required_strengths = ["null", "weak", "medium", "strong"]
    strength_ordered = all(
        key in strength and np.isfinite(strength[key]) for key in required_strengths
    ) and all(
        float(strength[left]) < float(strength[right])
        for left, right in zip(required_strengths[:-1], required_strengths[1:])
    )
    stage_complete = (
        len(stages) == args.expected_runs * 7
        and set(stages.stage.astype(str)) == EXPECTED_STAGES
    )
    all_legal = bool((
        runs.tfinal_candidate_legal & runs.tfinal_connected & runs.tfinal_acyclic
    ).all())
    checks = {
        "all_locked_runs_complete": len(runs) == args.expected_runs,
        "single_frozen_source_hash_signature": runs.hash_signature.nunique() == 1,
        "m0_m6_complete": stage_complete,
        "event_precision_at_least_0_70": event_precision >= 0.70,
        "event_recall_at_least_0_50": event_recall >= 0.50,
        "event_f1_at_least_0_58": event_f1 >= 0.58,
        "null_false_call_rate_at_most_0_10": null_false_call_rate <= 0.10,
        "effect_strength_strictly_ordered": strength_ordered,
        "mean_inferred_causal_topology_gain_positive": float(
            causal.inferred_topology_gain.mean()
        ) > 0,
        "no_systematic_negative_inferred_gain": int(
            causal.inferred_topology_gain.lt(0).sum()
        ) <= int(causal.inferred_topology_gain.gt(0).sum()),
        "null_false_rewire_rate_zero": np.isclose(null_false_rewire_rate, 0.0),
        "upstream_invariant": not bool(runs.upstream_violation.any()),
        "candidate_legal_connected_acyclic": all_legal,
        "event_edges_retained_after_pruning": bool(
            (runs.event_added_edges == runs.event_added_edges_retained_after_pruning).all()
            and runs.event_boundary_retained.all()
        ),
        "significant_relations_can_abstain": int(runs.significant_but_no_rewire.sum()) > 0,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    core_innovation = bool(
        checks["mean_inferred_causal_topology_gain_positive"]
        and float(gain_ci.ci_lower) > 0
        and int(causal.inferred_topology_gain.gt(0).sum())
        > int(causal.inferred_topology_gain.lt(0).sum())
    )
    decision = {
        "phase": "E",
        "evaluation_split": "LOCKED_VALIDATION",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "n_locked_runs": int(len(runs)),
        "n_causal_runs": int(len(causal)),
        "n_null_runs": int(len(nulls)),
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_F1": event_f1,
        "minus_cmi_e_event_precision": minus_p,
        "minus_cmi_e_event_recall": minus_r,
        "minus_cmi_e_event_F1": minus_f1,
        "full_null_false_call_rate": null_false_call_rate,
        "minus_cmi_e_null_false_call_rate": float(
            nulls.minus_cmi_e_n_calls.gt(0).mean()
        ),
        "mean_causal_t1_gain": float(causal.inferred_topology_gain.mean()),
        "median_causal_t1_gain": float(causal.inferred_topology_gain.median()),
        "causal_positive_zero_negative_runs": [
            int(causal.inferred_topology_gain.gt(0).sum()),
            int(causal.inferred_topology_gain.eq(0).sum()),
            int(causal.inferred_topology_gain.lt(0).sum()),
        ],
        "mean_causal_t1_gain_ci95": [
            float(gain_ci.ci_lower), float(gain_ci.ci_upper)
        ],
        "null_false_rewire_rate": null_false_rewire_rate,
        "effect_strength_mean_dependency": {
            str(key): float(value) for key, value in strength.items()
        },
        "relation_refinement_core_innovation": core_innovation,
        "relation_refinement_position": (
            "core_innovation" if core_innovation else "optional_conservative_local_correction"
        ),
        "truth_used_for_model_selection": False,
        "truth_used_for_locked_scoring_only": True,
        "bootstrap_unit": "replicate_seed_across_complete_scenario_grid",
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    (args.output / "PHASE_E_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
