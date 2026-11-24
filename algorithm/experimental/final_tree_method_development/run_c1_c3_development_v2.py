#!/usr/bin/env python3
"""C3_v2: small-sample branch programs on the frozen C1_v1 candidate graph."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "run_c1_c3_development_v1.py"


def load_v1():
    spec = importlib.util.spec_from_file_location("nicheforest_c1_c3_v1", V1_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(V1_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_expression_diagonal_gmm(expression: np.ndarray, forest: nx.Graph, tau: dict[int, float]):
    """Infer branch programs with a covariance model suitable for n << p.

    Tau is accepted for the common constructor interface but deliberately not
    residualized: the v1 diagnostic showed that removing continuous progression
    split the dominant branch into several artificial clusters.
    """
    labels: dict[int, str] = {}
    audit: list[dict] = []
    for cid, nodes_set in enumerate(sorted(nx.connected_components(forest), key=lambda x: min(x))):
        nodes = sorted(map(int, nodes_set))
        if len(nodes) < 8:
            for node in nodes:
                labels[node] = f"c{cid}_b0"
            audit.append({
                "component_id": cid, "n_nodes": len(nodes), "selected_k": 1,
                "selection": "small_component", "covariance_type": "diagonal",
            })
            continue
        features = expression[nodes]
        k_max = min(6, max(1, len(nodes) // 4))
        candidates = []
        for k in range(1, k_max + 1):
            reg = max(float(np.median(np.var(features, axis=0))) * 1e-6, 1e-8)
            model = GaussianMixture(
                n_components=k, covariance_type="diag", reg_covar=reg,
                random_state=20260809, n_init=20,
            )
            prediction = model.fit_predict(features)
            counts = np.bincount(prediction, minlength=k)
            if k > 1 and counts.min() < 2:
                continue
            candidates.append((float(model.bic(features)), k, prediction, counts))
        bic, selected_k, prediction, counts = min(candidates, key=lambda x: (x[0], x[1]))
        cluster_order = sorted(
            range(selected_k),
            key=lambda k: (
                np.median([tau[nodes[i]] for i in np.flatnonzero(prediction == k)]),
                min(nodes[i] for i in np.flatnonzero(prediction == k)),
            ),
        )
        remap = {old: new for new, old in enumerate(cluster_order)}
        for node, raw in zip(nodes, prediction):
            labels[node] = f"c{cid}_b{remap[int(raw)]}"
        audit.append({
            "component_id": cid, "n_nodes": len(nodes), "selected_k": selected_k,
            "bic": bic, "k_max": k_max,
            "cluster_sizes": "|".join(map(str, sorted(counts.tolist(), reverse=True))),
            "selection": "minimum_diagonal_GMM_BIC_on_root_expression",
            "covariance_type": "diagonal",
        })
    return labels, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=HERE.parents[2])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="CORE_05")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    method_dir = run_dir / "generic_upstream_v1_exact_root" / args.dataset

    v1 = load_v1()
    v1.infer_branch_communities = raw_expression_diagonal_gmm
    meta = ad.read_h5ad(method_dir / "metacell_adata.h5ad")
    geometry = v1.prepare_geometry(meta)
    candidate, candidate_audit = v1.build_multiresolution_candidate(meta, geometry)
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
    v1.evaluate_only(project, run_dir, prediction, score_dir, "C1_v1_C3_branchscale_v2")
    summary = pd.read_csv(score_dir / "SCORE_SUMMARY.csv").iloc[0].to_dict()
    integrity = {
        "method_version": "C1_candidate_multires_v1+C3_T0_branchscale_v2",
        "parent": "C1_candidate_multires_v1+C3_T0_branchscale_v1",
        "truth_available_to_constructor": False,
        "truth_read_only_after_method_outputs_written": True,
        "relation_module_called": False,
        "all_T0_edges_candidate_legal": bool(t0["edges"]["candidate_legal"].all()),
        "cycle_violating_roots": int(summary["cycle_violating_roots"]),
        "disconnected_roots": int(summary["disconnected_roots"]),
        "score": summary,
    }
    (output / "RUN_DECISION_INPUT.json").write_text(
        json.dumps(integrity, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(integrity, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
