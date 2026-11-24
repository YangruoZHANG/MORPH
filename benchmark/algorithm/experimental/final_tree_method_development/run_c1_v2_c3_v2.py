#!/usr/bin/env python3
"""C1_v2 + C3_v2: symmetric local kNN union with typed confidence roles."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import anndata as ad
import pandas as pd


HERE = Path(__file__).resolve().parent


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_v2(v1, adata, geometry):
    frame, audit = v1.build_multiresolution_candidate(adata, geometry)
    k_local = max(3, int(math.ceil(math.log2(max(adata.n_obs, 2)))))
    local_union = v1.directed_union_pairs(geometry["d_joint"], k_local)
    keys = [v1.edge_key(int(r.source), int(r.target)) for r in frame.itertuples(index=False)]
    key_to_index = {key: idx for idx, key in enumerate(keys)}
    promoted = 0
    for key in sorted(local_union):
        idx = key_to_index.get(key)
        if idx is None:
            raise RuntimeError(f"local union edge absent from broad edge universe: {key}")
        if not bool(frame.at[idx, "trajectory_edge_eligible"]):
            frame.at[idx, "trajectory_edge_eligible"] = True
            frame.at[idx, "edge_graph_role"] = "trajectory_local_union"
            frame.at[idx, "candidate_scale_rank"] = 1
            promoted += 1
    eligible = frame[frame["trajectory_edge_eligible"]]
    ranks = dict(zip(
        [v1.edge_key(int(r.source), int(r.target)) for r in eligible.itertuples(index=False)],
        v1.empirical_rank(eligible["D_joint"].to_numpy(float)),
    ))
    frame["joint_distance_ecdf"] = [
        ranks.get(v1.edge_key(int(r.source), int(r.target)), 1.0)
        for r in frame.itertuples(index=False)
    ]
    audit = dict(audit)
    audit.update({
        "candidate_version": "C1_candidate_multires_v2",
        "local_symmetrization": "union",
        "n_local_union": len(local_union),
        "n_promoted_from_interaction_only": promoted,
        "n_trajectory_eligible": int(frame["trajectory_edge_eligible"].sum()),
        "rationale": "retain boundary nodes that are one-sided local neighbours while preserving the fixed local k",
    })
    return frame, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=HERE.parents[2])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="CORE_05")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project, run_dir, output = args.project_root.resolve(), args.run_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    v1 = load_module(HERE / "run_c1_c3_development_v1.py", "nicheforest_c1_v1")
    v2 = load_module(HERE / "run_c1_c3_development_v2.py", "nicheforest_c3_v2")
    v1.infer_branch_communities = v2.raw_expression_diagonal_gmm
    method_dir = run_dir / "generic_upstream_v1_exact_root" / args.dataset
    meta = ad.read_h5ad(method_dir / "metacell_adata.h5ad")
    geometry = v1.prepare_geometry(meta)
    candidate, candidate_audit = candidate_v2(v1, meta, geometry)
    candidate.to_csv(output / "C1_MULTIRESOLUTION_CANDIDATE_EDGES.csv", index=False)
    pd.DataFrame([candidate_audit]).to_csv(output / "C1_CANDIDATE_CONSTRUCTION_AUDIT.csv", index=False)
    v1.candidate_truth_audit(candidate, run_dir / "formal_metacell", run_dir / "event" / "truth").to_csv(
        output / "C1_TRUTH_ONLY_RECOVERABILITY_AUDIT.csv", index=False
    )
    t0 = v1.construct_t0(method_dir, meta, candidate, geometry)
    t0["edges"].to_csv(output / "C3_BRANCH_SCALE_T0_EDGES.csv", index=False)
    t0["nodes"].to_csv(output / "C3_BRANCH_SCALE_NODE_ASSIGNMENTS.csv", index=False)
    t0["components"].to_csv(output / "C3_COMPONENT_RETENTION_AUDIT.csv", index=False)
    t0["branch_audit"].to_csv(output / "C3_BRANCH_MODEL_SELECTION_AUDIT.csv", index=False)
    prediction = output / "prediction"
    v1.write_prediction(t0, prediction)
    score_dir = output / "score"
    v1.evaluate_only(project, run_dir, prediction, score_dir, "C1_v2_C3_v2")
    summary = pd.read_csv(score_dir / "SCORE_SUMMARY.csv").iloc[0].to_dict()
    integrity = {
        "method_version": "C1_candidate_multires_v2+C3_T0_branchscale_v2",
        "parent": "C1_candidate_multires_v1+C3_T0_branchscale_v2",
        "truth_available_to_constructor": False,
        "truth_read_only_after_method_outputs_written": True,
        "relation_module_called": False,
        "all_T0_edges_candidate_legal": bool(t0["edges"]["candidate_legal"].all()),
        "cycle_violating_roots": int(summary["cycle_violating_roots"]),
        "disconnected_roots": int(summary["disconnected_roots"]),
        "score": summary,
    }
    (output / "RUN_DECISION_INPUT.json").write_text(json.dumps(integrity, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(integrity, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
