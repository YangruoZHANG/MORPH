#!/usr/bin/env python3
"""Selected C1/C3 design: broad typed candidate, high-confidence T0."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import anndata as ad
import pandas as pd


HERE = Path(__file__).resolve().parent
T0_ROLES = {
    "trajectory_local_reciprocal",
    "trajectory_mesoscale_reciprocal",
    "trajectory_crossscale_mst_bridge",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=HERE.parents[2])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="CORE_05")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project, run_dir, output = args.project_root.resolve(), args.run_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    v1 = load_module(HERE / "run_c1_c3_development_v1.py", "nicheforest_v1_for_v5")
    c3v2 = load_module(HERE / "run_c1_c3_development_v2.py", "nicheforest_c3v2_for_v5")
    c1v2 = load_module(HERE / "run_c1_v2_c3_v2.py", "nicheforest_c1v2_for_v5")
    v1.infer_branch_communities = c3v2.raw_expression_diagonal_gmm
    method_dir = run_dir / "generic_upstream_v1_exact_root" / args.dataset
    meta = ad.read_h5ad(method_dir / "metacell_adata.h5ad")
    geometry = v1.prepare_geometry(meta)
    candidate, candidate_audit = c1v2.candidate_v2(v1, meta, geometry)
    candidate["T0_role_active"] = candidate.edge_graph_role.astype(str).isin(T0_ROLES)
    candidate["T1_reserve_eligible"] = candidate.trajectory_edge_eligible & ~candidate.T0_role_active
    candidate.to_csv(output / "C1_MULTIRESOLUTION_CANDIDATE_EDGES.csv", index=False)
    candidate_audit = dict(candidate_audit)
    candidate_audit.update({
        "T0_active_roles": "|".join(sorted(T0_ROLES)),
        "n_T0_active_edges": int(candidate.T0_role_active.sum()),
        "n_T1_reserve_edges": int(candidate.T1_reserve_eligible.sum()),
        "reserve_policy": "legal search space; inactive in T0 absent downstream event constraints",
    })
    pd.DataFrame([candidate_audit]).to_csv(output / "C1_CANDIDATE_CONSTRUCTION_AUDIT.csv", index=False)
    v1.candidate_truth_audit(candidate, run_dir / "formal_metacell", run_dir / "event" / "truth").to_csv(output / "C1_TRUTH_ONLY_RECOVERABILITY_AUDIT.csv", index=False)
    t0_candidate = candidate[candidate.T0_role_active].copy()
    t0 = v1.construct_t0(method_dir, meta, t0_candidate, geometry)
    t0["edges"].to_csv(output / "C3_BRANCH_SCALE_T0_EDGES.csv", index=False)
    t0["nodes"].to_csv(output / "C3_BRANCH_SCALE_NODE_ASSIGNMENTS.csv", index=False)
    t0["components"].to_csv(output / "C3_COMPONENT_RETENTION_AUDIT.csv", index=False)
    t0["branch_audit"].to_csv(output / "C3_BRANCH_MODEL_SELECTION_AUDIT.csv", index=False)
    prediction = output / "prediction"
    v1.write_prediction(t0, prediction)
    score_dir = output / "score"
    v1.evaluate_only(project, run_dir, prediction, score_dir, "C1_v2_C3_highconf_T0_v5")
    summary = pd.read_csv(score_dir / "SCORE_SUMMARY.csv").iloc[0].to_dict()
    integrity = {
        "method_version": "C1_candidate_multires_v2+C3_T0_highconf_v5",
        "parent": "C1_candidate_multires_v2+C3_T0_frozen_internal_v4",
        "truth_available_to_constructor": False, "relation_module_called": False,
        "T0_uses_reserve_edges": False,
        "multiple_components_per_root_allowed": True,
        "all_T0_edges_candidate_legal": bool(t0["edges"]["candidate_legal"].all()),
        "score": summary,
    }
    (output / "RUN_DECISION_INPUT.json").write_text(json.dumps(integrity, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(integrity, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
