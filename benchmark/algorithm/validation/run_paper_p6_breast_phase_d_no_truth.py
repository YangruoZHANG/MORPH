#!/usr/bin/env python3
"""Finalize frozen breast T0 after relation abstention without topology truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


EXPERIMENTAL_DIR = (
    Path(__file__).resolve().parents[1]
    / "experimental"
    / "final_tree_method_development"
)
sys.path.insert(0, str(EXPERIMENTAL_DIR))

from run_phase_d_inferred_refinement_v3 import (  # noqa: E402
    as_bool,
    legality,
    prune_unanchored_components,
    recompute_pseudotime,
)


CSV_FLOAT_FORMAT = "%.15g"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def edge_keys(frame: pd.DataFrame) -> set[tuple[str, int, int]]:
    return {
        (str(row.root_id), int(row.source), int(row.target))
        for row in frame.itertuples(index=False)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-a-dir", type=Path, required=True)
    parser.add_argument("--relation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    t0_path = args.phase_a_dir / "T0_FINE_TRAJECTORY_EDGES.csv"
    nodes_path = args.phase_a_dir / "T0_STATE_NODE_ASSIGNMENTS.csv"
    evidence_path = args.relation_dir / "PHASE_B_CMI_E_EVIDENCE.csv"
    t0 = pd.read_csv(t0_path)
    nodes = pd.read_csv(nodes_path)
    evidence = pd.read_csv(evidence_path)

    relation_calls = evidence[evidence.is_called.map(as_bool)].copy()
    eligible = relation_calls[
        relation_calls.refinement_eligible.map(as_bool)
    ].copy() if "refinement_eligible" in relation_calls else relation_calls.copy()
    if len(eligible):
        raise ValueError(
            "This no-truth finalizer is valid only after complete relation abstention; "
            f"found {len(eligible)} refinement-eligible relation(s)."
        )

    # With no eligible event, the frozen refinement operator must abstain exactly.
    t1 = t0.copy()
    t1_nodes = recompute_pseudotime(t1, nodes, drop_unreachable=False)
    tfinal, pruning = prune_unanchored_components(t1, nodes, protected_names=set())
    tfinal_nodes = recompute_pseudotime(tfinal, nodes, drop_unreachable=True)

    t0_keys = edge_keys(t0)
    t1_keys = edge_keys(t1)
    final_keys = edge_keys(tfinal)
    t1_legal, t1_connected, t1_acyclic = legality(t1)
    final_legal, final_connected, final_acyclic = legality(tfinal)

    audit_columns = [
        "accepted", "reason", "n_selected_swaps", "upstream_violation",
        "source_root", "receiver_root", "event_type", "primary_channel_q",
        "call_strength",
    ]
    event_audit = pd.DataFrame(columns=audit_columns)
    result = {
        "dataset": "xenium_breast",
        "primary_trajectory_object": "T0",
        "n_ordered_root_pairs_tested": int(len(evidence)),
        "n_relation_calls": int(len(relation_calls)),
        "n_refinement_eligible_relations": int(len(eligible)),
        "n_accepted_relation_events": 0,
        "significant_but_no_rewire": int(len(relation_calls)),
        "relation_abstention": True,
        "t0_edges": int(len(t0)),
        "t1_full_edges": int(len(t1)),
        "tfinal_edges": int(len(tfinal)),
        "t0_to_t1_removed_edges": int(len(t0_keys - t1_keys)),
        "t0_to_t1_added_edges": int(len(t1_keys - t0_keys)),
        "t1_to_tfinal_removed_edges": int(len(t1_keys - final_keys)),
        "t1_to_tfinal_added_edges": int(len(final_keys - t1_keys)),
        "event_added_edges": 0,
        "event_added_edges_retained_after_pruning": 0,
        "event_boundary_retained": True,
        "n_unanchored_nodes_pruned": int(pruning.n_unanchored_nodes_pruned.sum()),
        "t1_candidate_legal": bool(t1_legal),
        "t1_connected": bool(t1_connected),
        "t1_acyclic": bool(t1_acyclic),
        "tfinal_candidate_legal": bool(final_legal),
        "tfinal_connected": bool(final_connected),
        "tfinal_acyclic": bool(final_acyclic),
        "upstream_violation": False,
        "topology_truth_available": False,
        "topology_accuracy_scored": False,
        "truth_used_for_selection": False,
    }

    t0.to_csv(
        args.output / "M4_T0_RELATION_INFERENCE_NO_TOPOLOGY_CHANGE.csv",
        index=False, float_format=CSV_FLOAT_FORMAT,
    )
    t1.to_csv(
        args.output / "M5_T1_FULL_INFERRED_RELATION_REFINED.csv",
        index=False, float_format=CSV_FLOAT_FORMAT,
    )
    t1_nodes.to_csv(
        args.output / "M5_TAU1_NODE_ASSIGNMENTS.csv",
        index=False, float_format=CSV_FLOAT_FORMAT,
    )
    tfinal.to_csv(
        args.output / "M6_TFINAL_PRUNED.csv",
        index=False, float_format=CSV_FLOAT_FORMAT,
    )
    tfinal_nodes.to_csv(
        args.output / "M6_TFINAL_NODE_ASSIGNMENTS.csv",
        index=False, float_format=CSV_FLOAT_FORMAT,
    )
    pruning.to_csv(
        args.output / "PHASE_D_PRUNING_RETENTION.csv",
        index=False, float_format=CSV_FLOAT_FORMAT,
    )
    event_audit.to_csv(
        args.output / "PHASE_D_EVENT_REFINEMENT_AUDIT.csv", index=False
    )
    pd.DataFrame([result]).to_csv(
        args.output / "PHASE_D_REAL_DATA_STRUCTURAL_AUDIT.csv",
        index=False, float_format=CSV_FLOAT_FORMAT,
    )

    status = {
        "phase": "P6_REAL_DATA_D",
        "operator_version": "INFERRED_REFINEMENT_V10_STRUCTURAL_TRANSITION_SAFETY_GATE",
        "pruning_version": "ROOT_CONNECTED_EVENT_PROTECTED_V1",
        "relation_detector": "PHASE_B_V7_1_MULTISCALE_LOCALIZED_STABILITY_SELECTION",
        "relation_significance_recomputed_after_t1": False,
        "refinement_operator_invoked": False,
        "refinement_operator_reason": "no_refinement_eligible_relation",
        "t1_equals_t0": bool(t1_keys == t0_keys),
        "tfinal_equals_t1": bool(final_keys == t1_keys),
        "truth_used": False,
        "complete": True,
        "input_sha256": {
            "t0": sha256(t0_path),
            "nodes": sha256(nodes_path),
            "relation_evidence": sha256(evidence_path),
        },
    }
    (args.output / "RUN_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
