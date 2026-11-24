#!/usr/bin/env python3
"""C1_v2 + C3_v3: edge-role hierarchical branch-scale T0."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def role_hierarchical_graph(candidate: pd.DataFrame, support: set[int], membership: np.ndarray) -> nx.Graph:
    """Give union edges connector-only semantics under Kruskal ordering.

    High-confidence edge costs occupy [0, 2]. Local-union edges occupy [3, 5],
    so they can connect high-confidence components but cannot replace an edge
    inside an already connected high-confidence component.
    """
    eligible = candidate[candidate["trajectory_edge_eligible"]].copy()
    graph = nx.Graph()
    graph.add_nodes_from(sorted(support))
    rows = []
    for row in eligible.itertuples(index=False):
        i, j = int(row.source), int(row.target)
        if i not in support or j not in support:
            continue
        soft_penalty = -math.log(max(float(membership[i] * membership[j]), 1e-12))
        rows.append((i, j, float(row.joint_distance_ecdf), soft_penalty, str(row.edge_graph_role)))
    if not rows:
        return graph
    soft_rank = pd.Series([row[3] for row in rows]).rank(method="average", pct=True).to_numpy(float)
    for (i, j, distance_rank, _, role), membership_rank in zip(rows, soft_rank):
        connector_tier = 3.0 if role == "trajectory_local_union" else 0.0
        graph.add_edge(
            i, j, weight=float(connector_tier + distance_rank + membership_rank),
            edge_graph_role=role, connector_only=bool(connector_tier),
        )
    return graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=HERE.parents[2])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="CORE_05")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project, run_dir, output = args.project_root.resolve(), args.run_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    v1 = load_module(HERE / "run_c1_c3_development_v1.py", "nicheforest_c1_base")
    c3v2 = load_module(HERE / "run_c1_c3_development_v2.py", "nicheforest_c3_branch_v2")
    c1v2 = load_module(HERE / "run_c1_v2_c3_v2.py", "nicheforest_candidate_v2")
    v1.infer_branch_communities = c3v2.raw_expression_diagonal_gmm
    v1.graph_for_support = role_hierarchical_graph
    method_dir = run_dir / "generic_upstream_v1_exact_root" / args.dataset
    meta = ad.read_h5ad(method_dir / "metacell_adata.h5ad")
    geometry = v1.prepare_geometry(meta)
    candidate, candidate_audit = c1v2.candidate_v2(v1, meta, geometry)
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
    connector_edges = t0["edges"][t0["edges"]["edge_graph_role"].astype(str).eq("trajectory_local_union")]
    connector_edges.to_csv(output / "C3_CONNECTOR_ONLY_EDGES_USED.csv", index=False)
    prediction = output / "prediction"
    v1.write_prediction(t0, prediction)
    score_dir = output / "score"
    v1.evaluate_only(project, run_dir, prediction, score_dir, "C1_v2_C3_role_hierarchical_v3")
    summary = pd.read_csv(score_dir / "SCORE_SUMMARY.csv").iloc[0].to_dict()
    integrity = {
        "method_version": "C1_candidate_multires_v2+C3_T0_role_hierarchical_v3",
        "parent": "C1_candidate_multires_v2+C3_T0_branchscale_v2",
        "truth_available_to_constructor": False,
        "relation_module_called": False,
        "union_edge_semantics": "connector_only_after_high_confidence_components",
        "n_union_connectors_used": int(len(connector_edges)),
        "all_T0_edges_candidate_legal": bool(t0["edges"]["candidate_legal"].all()),
        "cycle_violating_roots": int(summary["cycle_violating_roots"]),
        "disconnected_roots": int(summary["disconnected_roots"]),
        "score": summary,
    }
    (output / "RUN_DECISION_INPUT.json").write_text(json.dumps(integrity, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(integrity, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
