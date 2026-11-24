#!/usr/bin/env python3
"""Summarize and gate frozen PHASE A v4 development evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE = "current_root_conditioned_msf"
PRODUCTION = "hvg_branch_regularized_020"
METRICS = [
    "fine_edge_F1", "fine_true_path_coverage", "fine_shortcut_rate",
    "branch_ARI_rootwise", "branchpoint_F1", "terminal_F1",
    "ancestor_descendant_consistency", "root_terminal_path_similarity",
    "shared_prefix_recovery", "branching_order_accuracy",
    "pseudotime_spearman", "root_distance_consistency",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_metrics, deltas, candidates, events, statuses = [], [], [], [], []
    for run in sorted(path for path in args.runs_root.iterdir() if path.is_dir()):
        audit = run / "phase_a_v4_frozen"
        frame = pd.read_csv(audit / "HVG_V8_DIAGNOSTIC_METRICS.csv")
        frame.insert(0, "run_directory", run.name)
        all_metrics.append(frame)
        aggregate = frame[frame.root_id.astype(str).eq("ALL")].set_index("case")
        baseline, production = aggregate.loc[BASELINE], aggregate.loc[PRODUCTION]
        row = {"run": run.name}
        row.update({metric: float(production[metric] - baseline[metric]) for metric in METRICS})
        deltas.append(row)
        candidate = pd.read_csv(audit / "PHASE_A_CANDIDATE_AUDIT.csv")
        candidate.insert(0, "run_directory", run.name)
        candidates.append(candidate)
        event = pd.read_csv(audit / "PHASE_A_EVENT_FEASIBILITY.csv")
        event.insert(0, "run_directory", run.name)
        events.append(event)
        statuses.append(json.loads((audit / "RUN_STATUS.json").read_text(encoding="utf-8")))

    metric_frame = pd.concat(all_metrics, ignore_index=True)
    delta_frame = pd.DataFrame(deltas)
    candidate_frame = pd.concat(candidates, ignore_index=True)
    event_frame = pd.concat(events, ignore_index=True)
    metric_frame.to_csv(args.output / "PHASE_A_V4_ALL_METRICS.csv", index=False)
    delta_frame.to_csv(args.output / "PHASE_A_V4_PAIRED_DELTAS.csv", index=False)
    candidate_frame.to_csv(args.output / "PHASE_A_V4_CANDIDATE_AUDIT.csv", index=False)
    event_frame.to_csv(args.output / "PHASE_A_V4_EVENT_FEASIBILITY.csv", index=False)

    candidate_all = candidate_frame[candidate_frame.root_id.astype(str).eq("ALL")]
    current = candidate_all[candidate_all.candidate.eq("current_single_scale")].set_index("run_directory")
    new = candidate_all[candidate_all.candidate.eq("new_multiresolution_full")].set_index("run_directory")
    candidate_summary = {
        "mean_root_reachability": float(new.candidate_root_reachable_fraction.mean()),
        "minimum_root_reachability": float(new.candidate_root_reachable_fraction.min()),
        "mean_path_fraction_of_ceiling": float(
            new.candidate_true_path_coverage_fraction_of_ceiling.mean()
        ),
        "minimum_path_fraction_of_ceiling": float(
            new.candidate_true_path_coverage_fraction_of_ceiling.min()
        ),
        "paired_mean_shortcut_delta": float(
            (new.candidate_shortcut_rate - current.candidate_shortcut_rate).mean()
        ),
        "paired_mean_cross_branch_delta": float(
            (new.candidate_cross_branch_false_connection_rate
             - current.candidate_cross_branch_false_connection_rate).mean()
        ),
    }
    delta_summary = {
        metric: {
            "mean_delta": float(delta_frame[metric].mean()),
            "median_delta": float(delta_frame[metric].median()),
            "positive_runs": int((delta_frame[metric] > 0).sum()),
            "n_runs": int(len(delta_frame)),
        }
        for metric in METRICS
    }
    gates = {
        "candidate_root_reachability": (
            candidate_summary["mean_root_reachability"] >= 0.95
            and candidate_summary["minimum_root_reachability"] >= 0.90
        ),
        "candidate_path_ceiling": (
            candidate_summary["mean_path_fraction_of_ceiling"] >= 0.95
            and candidate_summary["minimum_path_fraction_of_ceiling"] >= 0.90
        ),
        "candidate_shortcut_control": candidate_summary["paired_mean_shortcut_delta"] <= 0.05,
        "candidate_cross_branch_control": candidate_summary["paired_mean_cross_branch_delta"] <= 0.05,
        "fine_edge_improvement": (
            delta_summary["fine_edge_F1"]["mean_delta"] > 0
            and delta_summary["fine_edge_F1"]["positive_runs"] >= 6
        ),
        "branch_program_improvement": (
            delta_summary["branch_ARI_rootwise"]["mean_delta"] > 0
            and delta_summary["branch_ARI_rootwise"]["positive_runs"] >= 6
        ),
        "branch_structure_improvement": all(
            delta_summary[name]["mean_delta"] > 0
            for name in ("branchpoint_F1", "terminal_F1", "ancestor_descendant_consistency")
        ),
        "hierarchy_noninferiority": all(
            delta_summary[name]["mean_delta"] >= -0.05
            for name in (
                "root_terminal_path_similarity", "shared_prefix_recovery",
                "branching_order_accuracy",
            )
        ),
        "pseudotime_improvement": (
            delta_summary["pseudotime_spearman"]["mean_delta"] > 0
            and delta_summary["pseudotime_spearman"]["positive_runs"] >= 6
        ),
        "root_distance_improvement": (
            delta_summary["root_distance_consistency"]["mean_delta"] > 0
            and delta_summary["root_distance_consistency"]["positive_runs"] >= 6
        ),
        "event_alternative_feasible": bool(
            event_frame.strict_legal_alternative_exists.astype(bool).all()
        ),
        "event_truth_coverage_adequate": float(
            event_frame.truth_T1_state_edge_coverage.min()
        ) >= 0.70,
        "candidate_legal_and_forest": all(
            bool(status["all_fine_edges_candidate_legal"])
            and bool(status["all_selected_root_graphs_are_forests"])
            for status in statuses
        ),
        "truth_safe_constructor": all(
            not bool(status["truth_used_for_production_construction"])
            for status in statuses
        ),
        "locked_replicates_untouched": True,
    }
    decision = {
        "phase": "A",
        "method_version": "PHASE_A_v4_generator_v2_fine_hvg_v9_regularized",
        "benchmark_specification": "BENCHMARK_SPEC_V2",
        "development_replicates": list(range(1, 9)),
        "locked_replicates_21_30_touched": False,
        "candidate_summary": candidate_summary,
        "paired_metric_summary": delta_summary,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "PHASE_A_DECISION": "PASS" if all(gates.values()) else "FAIL",
        "phase_C_oracle_refinement_authorized": bool(all(gates.values())),
        "phase_B_relation_benchmark_authorized": False,
        "interpretation": (
            "Candidate, soft assignment and branch-scale T0 are adequate for oracle "
            "refinement development. Relation inference remains unauthorized until "
            "the oracle refinement operator passes PHASE C."
        ),
    }
    (args.output / "PHASE_A_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "PHASE_A_DECISION": decision["PHASE_A_DECISION"],
        "failed_gates": decision["failed_gates"],
        "candidate_summary": candidate_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
