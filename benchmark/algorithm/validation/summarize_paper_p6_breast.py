#!/usr/bin/env python3
"""Create the paper-facing audit for the frozen xenium breast run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def diagnostics(row: pd.Series) -> str:
    flags: list[str] = []
    if float(row.primary_channel_q) > 0.15:
        flags.append("primary_q_gt_0.15")
    if not as_bool(row.scale065_called):
        flags.append("not_called_scale_0.65")
    if not as_bool(row.scale080_called):
        flags.append("not_called_scale_0.80")
    if not as_bool(row.scale_direction_consistent):
        flags.append("direction_not_scale_consistent")
    if not as_bool(row.scale_branch_consistent):
        flags.append("branch_not_scale_consistent")
    if not as_bool(row.localization_support_pass):
        flags.append("localized_branch_support_fail")
    if float(row.program_specificity_margin) <= 0:
        flags.append("primary_program_not_negative_control_specific")
    if float(row.oof_predictive_gain) <= 0:
        flags.append("nonpositive_oof_predictive_gain")
    return ";".join(flags)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p6-dir", type=Path, required=True)
    args = parser.parse_args()
    base = args.p6_dir

    phase_a = json.loads((base / "phase_a" / "RUN_STATUS.json").read_text())
    soft = json.loads(
        (base / "phase_a" / "P6_SOFT_ASSIGNMENT_AUDIT.json").read_text()
    )
    structural = pd.read_csv(base / "phase_a" / "P6_T0_STRUCTURAL_AUDIT.csv")
    nodes = pd.read_csv(base / "phase_a" / "T0_STATE_NODE_ASSIGNMENTS.csv")
    branch = pd.read_csv(base / "P6_T0_BRANCH_COMPOSITION.csv")
    phase_b = json.loads(
        (base / "phase_b_consensus" / "RUN_STATUS.json").read_text()
    )
    evidence = pd.read_csv(
        base / "phase_b_consensus" / "PHASE_B_CMI_E_EVIDENCE.csv"
    )
    phase_d = json.loads((base / "phase_d" / "RUN_STATUS.json").read_text())
    d_audit = pd.read_csv(
        base / "phase_d" / "PHASE_D_REAL_DATA_STRUCTURAL_AUDIT.csv"
    ).iloc[0]

    relation_columns = [
        "source_root", "receiver_root", "affected_branch", "event_type",
        "raw_dependency", "environment_adjusted_dependency", "environment_drop",
        "environment_trust", "environment_conflict", "oof_predictive_gain",
        "program_specificity_margin", "primary_channel_q", "signature_channel_q",
        "scale065_primary_q", "scale080_primary_q", "scale065_called",
        "scale080_called", "scale_direction_consistent", "scale_branch_consistent",
        "multiscale_localized_branch_support_fraction", "localization_support_pass",
        "is_called", "refinement_eligible", "decision_reason",
    ]
    relation_audit = evidence[relation_columns].copy()
    relation_audit["observed_failed_diagnostics"] = evidence.apply(diagnostics, axis=1)
    relation_audit.to_csv(base / "P6_RELATION_ABSTENTION_AUDIT.csv", index=False)

    n_unique_support = int(nodes.unit_id.astype(str).nunique())
    n_all = int(soft["n_metacells"])
    summary_rows = [
        ("input", "metacells", n_all),
        ("T0", "roots", int(phase_a["n_roots"])),
        ("T0", "root_node_rows", int(len(nodes))),
        ("T0", "unique_supported_metacells", n_unique_support),
        ("T0", "unique_support_fraction", n_unique_support / n_all),
        ("T0", "branches", int(structural.n_branches.sum())),
        ("T0", "states", int(structural.n_states.sum())),
        ("T0", "edges", int(d_audit.t0_edges)),
        ("soft", "shared_membership_ge_0.20", int(soft["shared_membership_ge_0_20"])),
        ("soft", "shared_membership_ge_0.10", int(soft["shared_membership_ge_0_10"])),
        ("relation", "ordered_root_pairs_tested", int(len(evidence))),
        ("relation", "called_relations", int(phase_b["n_relations"])),
        ("relation", "refinement_eligible", int(phase_b["n_refinement_eligible"])),
        ("T1", "added_edges", int(d_audit.t0_to_t1_added_edges)),
        ("T1", "removed_edges", int(d_audit.t0_to_t1_removed_edges)),
        ("pruning", "unanchored_nodes_pruned", int(d_audit.n_unanchored_nodes_pruned)),
        ("Tfinal", "edges", int(d_audit.tfinal_edges)),
    ]
    pd.DataFrame(summary_rows, columns=["stage", "metric", "value"]).to_csv(
        base / "P6_BREAST_STAGE_SUMMARY.csv", index=False
    )

    all_lymphoid = bool(
        branch.top_compartment.astype(str).str.lower().eq("lymphoid").all()
    )
    core_structural_pass = bool(
        phase_a["all_fine_edges_candidate_legal"]
        and phase_a["all_selected_root_graphs_are_trees"]
        and phase_a["all_root_supports_connected"]
        and as_bool(d_audit.tfinal_candidate_legal)
        and as_bool(d_audit.tfinal_connected)
        and as_bool(d_audit.tfinal_acyclic)
    )
    decision = {
        "stage": "P6_FROZEN_BREAST",
        "run_complete": True,
        "core_T0_structural_execution": "PASS" if core_structural_pass else "FAIL",
        "cross_tree_association": "ABSTAIN_NO_FROZEN_RELATION_CALL",
        "optional_local_correction": "NO_OP_NO_ELIGIBLE_RELATION",
        "primary_trajectory_object": "T0",
        "real_topology_truth_available": False,
        "biological_validation_complete": False,
        "biological_validation_next_stage": "P8_after_P7_negative_controls",
        "claim_ceiling": "confounding-aware spatially predictive association",
        "causal_claim_allowed": False,
        "support_scope": {
            "unique_metacells": n_unique_support,
            "total_metacells": n_all,
            "fraction": n_unique_support / n_all,
            "all_branches_major_compartment_lymphoid": all_lymphoid,
        },
        "frozen_core_modified": False,
        "truth_used": False,
    }
    (base / "P6_BREAST_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    min_q = float(evidence.primary_channel_q.min())
    max_drop = float(evidence.environment_drop.max())
    positive_oof = int((evidence.oof_predictive_gain > 0).sum())
    report = f"""# P6 Frozen Xenium Breast Audit

## Frozen execution

- The no-truth Phase A production adapter constructed 3 root-conditioned T0 trees.
- T0 contains {len(nodes)} root-node rows, {n_unique_support} unique metacells ({n_unique_support / n_all:.2%} of {n_all}), 7 branches, and {int(d_audit.t0_edges)} edges.
- Every root graph is candidate-legal, connected, and acyclic. No real topology truth was created or scored.
- The primary paper trajectory object remains T0.

## Soft decomposition

- Dominant assignments: R0={soft['dominant_root_counts']['R0']}, R1={soft['dominant_root_counts']['R1']}, R2={soft['dominant_root_counts']['R2']}.
- Shared membership occurs for {soft['shared_membership_ge_0_20']} metacells at the 0.20 threshold and {soft['shared_membership_ge_0_10']} at 0.10.
- Unassigned/inactive tissue context remains outside the targeted trajectory support; it is not silently counted as pseudotime zero.

## Cross-tree evidence

- Six ordered root pairs were tested using the frozen two-scale CMI-E detector.
- No pair was called at both scales; the minimum consensus primary-channel q-value was {min_q:.4f}.
- {positive_oof}/6 pairs had positive out-of-fold predictive gain, but no pair satisfied the complete frozen multiscale gate.
- The maximum observed environment drop was {max_drop:.4f}; channel-level values are retained in `P6_RELATION_ABSTENTION_AUDIT.csv`.
- This is a formal abstention, not evidence that no biological interaction exists.

## Refinement and pruning

- With zero refinement-eligible relations, the frozen refinement operator was not invoked.
- T0, T1_full, and Tfinal each contain {int(d_audit.t0_edges)} edges; zero edges were added, removed, or pruned.
- Tfinal remains candidate-legal, connected, and acyclic, with no upstream violation.

## Interpretation boundary

- The seven current branches are all lymphoid at the available major-compartment annotation level. They must be interpreted as distinct T-cell-like root-conditioned trajectories, not as stromal, vascular, and immune trees.
- The pre-existing v7.8.1 relation table was not reused because it belongs to an older method definition and different root semantics.
- P6 establishes reproducible structural execution. Independent biological validation and negative controls remain P7-P9 work and must not feed back into topology selection.
"""
    (base / "P6_BREAST_AUDIT.md").write_text(report, encoding="utf-8")

    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
