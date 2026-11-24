#!/usr/bin/env python3
"""Create the frozen PHASE A v3 audit bundle and PASS/FAIL decision."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PHASE_ROOT = REPO / "outputs" / "final_tree_method_development" / "phase_a_v3"
OUTPUT = PHASE_ROOT / "final_audit"

BRANCH_FAMILIES = {
    "branch_program": ["branch_ARI_rootwise"],
    "branchpoint": ["branchpoint_F1"],
    "terminal": ["terminal_F1"],
    "ancestor": ["ancestor_descendant_consistency"],
    "root_terminal": ["root_terminal_path_similarity"],
    "prefix_order": ["shared_prefix_recovery", "branching_order_accuracy"],
}


def collect(filename: str) -> pd.DataFrame:
    frames = []
    for split in ("development", "confirmation"):
        for path in sorted((PHASE_ROOT / split).glob(f"*/{filename}")):
            frame = pd.read_csv(path)
            frame.insert(0, "evaluation_split", split)
            frame.insert(1, "source_run_directory", path.parent.name)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(filename)
    return pd.concat(frames, ignore_index=True)


def mean_value(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def bool_values(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def candidate_gate(candidate: pd.DataFrame, split: str) -> dict:
    local = candidate[candidate.evaluation_split.eq(split)]
    aggregate = local[local.root_id.astype(str).eq("ALL")]
    roots = local[~local.root_id.astype(str).eq("ALL")]
    current = aggregate[aggregate.candidate.eq("current_single_scale")]
    new = aggregate[aggregate.candidate.eq("new_multiresolution_full")]
    ceiling = aggregate[aggregate.candidate.eq("complete_graph_ceiling")]
    current_path = mean_value(current, "candidate_true_path_coverage")
    new_path = mean_value(new, "candidate_true_path_coverage")
    ceiling_path = mean_value(ceiling, "candidate_true_path_coverage")
    shortcut_delta = mean_value(new, "candidate_shortcut_rate") - mean_value(
        current, "candidate_shortcut_rate"
    )
    cross_delta = mean_value(
        new, "candidate_cross_branch_false_connection_rate"
    ) - mean_value(current, "candidate_cross_branch_false_connection_rate")
    new_roots = roots[roots.candidate.eq("new_multiresolution_full")]
    result = {
        "split": split,
        "mean_root_reachability": mean_value(new, "candidate_root_reachable_fraction"),
        "minimum_root_reachability": float(
            pd.to_numeric(new_roots.candidate_root_reachable_fraction).min()
        ),
        "current_true_path_coverage": current_path,
        "new_true_path_coverage": new_path,
        "complete_graph_path_ceiling": ceiling_path,
        "fraction_of_complete_graph_path_ceiling": new_path / ceiling_path,
        "paired_mean_shortcut_delta": shortcut_delta,
        "paired_mean_cross_branch_false_connection_delta": cross_delta,
    }
    result["root_reachability_pass"] = (
        result["mean_root_reachability"] >= 0.90
        and result["minimum_root_reachability"] >= 0.75
    )
    result["path_coverage_pass"] = (
        new_path >= 0.70 and result["fraction_of_complete_graph_path_ceiling"] >= 0.90
    )
    result["shortcut_control_pass"] = shortcut_delta <= 0.05 + 1e-12
    result["cross_branch_control_pass"] = cross_delta <= 0.05 + 1e-12
    return result


def t0_gate(metrics: pd.DataFrame, split: str) -> dict:
    local = metrics[
        metrics.evaluation_split.eq(split) & metrics.root_id.astype(str).eq("ALL")
    ]
    by_run = []
    for run, frame in local.groupby("run", sort=True):
        current = frame[frame.case.eq("B_inferred_membership_current_T0")].iloc[0]
        new = frame[frame.case.eq("D_inferred_membership_new_T0")].iloc[0]
        deltas = {}
        for family, columns in BRANCH_FAMILIES.items():
            deltas[family] = float(np.mean([
                float(new[column]) - float(current[column]) for column in columns
            ]))
        by_run.append({
            "run": run,
            **{f"{family}_delta": value for family, value in deltas.items()},
            "n_improved_branch_families": sum(value > 0 for value in deltas.values()),
            "pseudotime_delta": float(new.pseudotime_spearman) - float(current.pseudotime_spearman),
            "root_distance_delta": float(new.root_distance_consistency) - float(current.root_distance_consistency),
            "fine_edge_F1_delta": float(new.fine_edge_F1) - float(current.fine_edge_F1),
            "fine_true_path_coverage_delta": (
                float(new.fine_true_path_coverage) - float(current.fine_true_path_coverage)
            ),
        })
    run_frame = pd.DataFrame(by_run)
    family_means = {
        family: float(run_frame[f"{family}_delta"].mean()) for family in BRANCH_FAMILIES
    }
    n_family_improved = sum(value > 0 for value in family_means.values())
    consistent_fraction = float((run_frame.n_improved_branch_families >= 4).mean())
    result = {
        "split": split,
        "family_mean_deltas": family_means,
        "n_branch_families_improved": n_family_improved,
        "fraction_runs_with_at_least_four_improved_families": consistent_fraction,
        "mean_pseudotime_delta": float(run_frame.pseudotime_delta.mean()),
        "mean_root_distance_delta": float(run_frame.root_distance_delta.mean()),
        "mean_fine_edge_F1_delta": float(run_frame.fine_edge_F1_delta.mean()),
        "mean_fine_true_path_coverage_delta": float(
            run_frame.fine_true_path_coverage_delta.mean()
        ),
        "run_level": by_run,
    }
    result["branch_scale_pass"] = (
        n_family_improved >= 4
        and consistent_fraction >= 2 / 3
        and result["mean_pseudotime_delta"] >= -0.05
    )
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidate = collect("PHASE_A_CANDIDATE_AUDIT.csv")
    metrics = collect("PHASE_A_T0_METRICS.csv")
    oracle = collect("PHASE_A_ORACLE_DECOMPOSITION.csv")
    events = collect("PHASE_A_EVENT_FEASIBILITY.csv")
    candidate.to_csv(OUTPUT / "PHASE_A_CANDIDATE_AUDIT.csv", index=False)
    metrics.to_csv(OUTPUT / "PHASE_A_T0_METRICS.csv", index=False)
    oracle.to_csv(OUTPUT / "PHASE_A_ORACLE_DECOMPOSITION.csv", index=False)
    events.to_csv(OUTPUT / "PHASE_A_EVENT_FEASIBILITY.csv", index=False)
    events.to_csv(OUTPUT / "ORACLE_EVENT_FEASIBILITY_ON_T0.csv", index=False)
    shutil.copy2(HERE / "C3_T0_DUALCHANNEL_V7_DESIGN.md", OUTPUT / "PHASE_A_DESIGN.md")

    candidate_results = {
        split: candidate_gate(candidate, split) for split in ("development", "confirmation")
    }
    t0_results = {
        split: t0_gate(metrics, split) for split in ("development", "confirmation")
    }
    statuses = [
        json.loads(path.read_text(encoding="utf-8"))
        for split in ("development", "confirmation")
        for path in sorted((PHASE_ROOT / split).glob("*/RUN_STATUS.json"))
    ]
    truth_safe = all(
        not status["relation_detector_called"]
        and not status["refinement_operator_called"]
        and not status["truth_used_for_construction_D"]
        and status["all_fine_edges_candidate_legal"]
        and status["all_cases_are_forests"]
        for status in statuses
    ) and len(statuses) == 8
    strict = bool_values(events.strict_legal_alternative_exists)
    representable = bool_values(events.all_truth_T1_state_edges_representable)
    event_result = {
        "n_events": int(len(events)),
        "n_strict_legal_alternatives": int(strict.sum()),
        "all_events_have_strict_legal_alternative": bool(strict.all()),
        "n_runs_with_complete_truth_T1_representability": int(representable.sum()),
        "all_truth_T1_state_edges_representable": bool(representable.all()),
        "mean_truth_T1_state_edge_coverage": mean_value(events, "truth_T1_state_edge_coverage"),
    }
    gates = {
        "candidate_root_reachability": all(
            value["root_reachability_pass"] for value in candidate_results.values()
        ),
        "candidate_path_coverage": all(
            value["path_coverage_pass"] for value in candidate_results.values()
        ),
        "branch_scale_T0": all(value["branch_scale_pass"] for value in t0_results.values()),
        "truth_safe_constructor": truth_safe,
        "confirmation_reproduced": t0_results["confirmation"]["branch_scale_pass"],
        "shortcut_and_cross_branch_control": all(
            value["shortcut_control_pass"] and value["cross_branch_control_pass"]
            for value in candidate_results.values()
        ),
        "oracle_event_feasibility": (
            event_result["all_events_have_strict_legal_alternative"]
            and event_result["all_truth_T1_state_edges_representable"]
        ),
    }
    failures = [name for name, passed in gates.items() if not passed]
    decision = {
        "phase": "A",
        "method": "PHASE_A_v3_C3_dualchannel_v7",
        "PHASE_A_DECISION": "PASS" if not failures else "FAIL",
        "phase_B_authorized": not failures,
        "gates": gates,
        "failed_gates": failures,
        "candidate_results": candidate_results,
        "t0_results": t0_results,
        "event_feasibility": event_result,
        "locked_final_replicates_21_30_touched": False,
        "interpretation": (
            "PHASE A is not paper-ready; do not optimize relation/refinement."
            if failures else "PHASE A passed; PHASE B may begin."
        ),
    }
    (OUTPUT / "PHASE_A_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
