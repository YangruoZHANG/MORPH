#!/usr/bin/env python3
"""PHASE A v3: dual-channel T0 with frozen branch programs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import anndata as ad
import pandas as pd


HERE = Path(__file__).resolve().parent


def load_version(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = load_version("run_phase_a_case_v2.py", "phase_a_v2_for_v3")
V21 = load_version("run_phase_a_case_v2_1.py", "phase_a_v21_for_v3")
DUAL = load_version(
    "diagnose_phase_a_dualchannel_v7.py", "phase_a_dualchannel_constructor_v7"
)


def full_candidate_union(full_candidate: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Serialize all candidate roles with one provenance record per pair."""
    full = full_candidate.copy()
    existing = existing.copy()
    role_priority = {
        "trajectory_local_reciprocal": 0,
        "trajectory_mesoscale_reciprocal": 0,
        "trajectory_crossscale_mst_bridge": 0,
        "trajectory_current_single_scale": 1,
        "trajectory_local_union": 2,
        "interaction_only_broad_neighbour": 3,
    }
    full["candidate_priority"] = full.edge_graph_role.astype(str).map(role_priority).fillna(4)
    existing["candidate_priority"] = 1
    candidate = pd.concat([full, existing], ignore_index=True, sort=False)
    candidate["pair_key"] = [
        V2.P.V1.edge_key(int(row.source), int(row.target))
        for row in candidate.itertuples(index=False)
    ]
    candidate = candidate.sort_values(
        ["candidate_priority", "D_joint", "source", "target"], kind="mergesort"
    ).drop_duplicates("pair_key", keep="first")
    return candidate.drop(columns=["candidate_priority", "pair_key"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="CORE_05")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir, output = args.run_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    method_dir = run_dir / "generic_upstream_v1_exact_root" / args.dataset
    meta = ad.read_h5ad(method_dir / "metacell_adata.h5ad")
    geometry = V2.P.V1.prepare_geometry(meta)
    inferred_a, roots, inferred_supports = V2.P.inferred_membership(method_dir, meta.n_obs)
    current_candidate = V2.P.current_candidate(method_dir)
    multires_candidate, _, build_audit = V2.P.new_candidates(meta, geometry)
    candidate = full_candidate_union(multires_candidate, current_candidate)

    # Both production T0 variants are constructed before any synthetic truth is loaded.
    current_inferred = V2.topological_state_upgrade(
        V2.P.construct_t0(
            meta, geometry, current_candidate, inferred_a, roots, inferred_supports, False
        )
    )
    new_inferred = DUAL.construct_variant(
        meta, geometry, multires_candidate, current_candidate,
        inferred_a, roots, inferred_supports, "frozen_branch_quotient",
    )

    truth = V2.Q.add_soft_state_truth(run_dir, V2.P.load_truth_context(run_dir))
    true_a, true_supports = V2.P.true_membership(meta, roots, truth)
    current_true = V2.topological_state_upgrade(
        V2.P.construct_t0(
            meta, geometry, current_candidate, true_a, roots, true_supports, False
        )
    )
    new_true = DUAL.construct_variant(
        meta, geometry, multires_candidate, current_candidate,
        true_a, roots, true_supports, "frozen_branch_quotient",
    )
    V2.mapped_state_backbone = V21.exact_anchor_mapped_state_backbone
    cases = {
        "A_true_membership_current_T0": current_true,
        "B_inferred_membership_current_T0": current_inferred,
        "C_true_membership_new_T0": new_true,
        "D_inferred_membership_new_T0": new_inferred,
    }
    for name, result in cases.items():
        V2.P.write_case(output, name, result)

    new_inferred["backbone"].to_csv(output / "T0_STATE_BACKBONE_EDGES.csv", index=False)
    new_inferred["edges"].to_csv(output / "T0_FINE_TRAJECTORY_EDGES.csv", index=False)
    new_inferred["nodes"].to_csv(output / "T0_STATE_NODE_ASSIGNMENTS.csv", index=False)
    new_inferred["components"].to_csv(output / "T0_COMPONENT_AUDIT.csv", index=False)
    new_inferred["branch_models"].to_csv(
        output / "T0_BRANCH_PROGRAM_MODEL_AUDIT.csv", index=False
    )
    candidate.to_csv(output / "C1_MULTIRESOLUTION_CANDIDATE_EDGES.csv", index=False)

    candidate_frame = pd.concat([
        V2.R.directional_candidate_audit(
            args.run_label, "current_single_scale", current_candidate, truth
        ),
        V2.R.directional_candidate_audit(
            args.run_label, "new_multiresolution_full", candidate, truth
        ),
        V2.R.directional_candidate_audit(
            args.run_label, "complete_graph_ceiling", V2.R.complete_candidate(meta), truth
        ),
    ], ignore_index=True)
    V2.R.add_ceiling_columns(candidate_frame).to_csv(
        output / "PHASE_A_CANDIDATE_AUDIT.csv", index=False
    )
    t0_frame = pd.concat([
        V2.dual_scale_metrics(args.run_label, name, result, truth)
        for name, result in cases.items()
    ], ignore_index=True)
    t0_frame.to_csv(output / "PHASE_A_T0_METRICS.csv", index=False)
    t0_frame[t0_frame.root_id.astype(str).eq("ALL")].to_csv(
        output / "PHASE_A_ORACLE_DECOMPOSITION.csv", index=False
    )
    feasibility = V2.Q.event_feasibility(
        args.run_label, run_dir, candidate, new_inferred, truth
    )
    feasibility.to_csv(output / "PHASE_A_EVENT_FEASIBILITY.csv", index=False)
    feasibility.to_csv(output / "ORACLE_EVENT_FEASIBILITY_ON_T0.csv", index=False)
    V2.Q.threshold_sensitivity(
        args.run_label, run_dir, current_candidate, candidate
    ).to_csv(output / "STATE_SUPPORT_THRESHOLD_SENSITIVITY.csv", index=False)
    pd.DataFrame([{**build_audit,
        "n_existing_k6_edges_in_union": int(len(current_candidate)),
        "n_full_union_edges": int(len(candidate)),
    }]).to_csv(output / "CANDIDATE_BUILD_AUDIT.csv", index=False)

    status = {
        "phase": "A",
        "version": "PHASE_A_v3_C3_dualchannel_v7",
        "parent": "PHASE_A_v2_1_C3_dualscale_v6",
        "construction_change": (
            "freeze high-confidence expression branch programs; reconstruct topology "
            "from their quotient over multiresolution plus existing eligible local edges"
        ),
        "run": args.run_label,
        "dataset": args.dataset,
        "relation_detector_called": False,
        "refinement_operator_called": False,
        "production_case": "D_inferred_membership_new_T0",
        "truth_used_for_construction_D": False,
        "branch_program_channel": sorted(V2.P.T0_ACTIVE_ROLES),
        "topology_additional_channel": "trajectory_current_single_scale",
        "fallback_connector_channel": "trajectory_local_union",
        "n_union_connectors_used": int(new_inferred["connector_total"]),
        "all_fine_edges_candidate_legal": bool(new_inferred["edges"].candidate_legal.all()),
        "all_cases_are_forests": bool(t0_frame.is_forest.all()),
        "complete": True,
    }
    (output / "RUN_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
