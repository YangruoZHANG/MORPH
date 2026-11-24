#!/usr/bin/env python3
"""Run frozen Phase A v7.1 on breast without constructing real-data truth."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "algorithm" / "experimental" / "final_tree_method_development"
DEFAULT_METHOD_DIR = (
    REPO / "outputs" / "tme_v7_4_3_3_anchor_bridge_all7" / "xenium_breast"
)
DEFAULT_OUTPUT = (
    REPO
    / "outputs"
    / "NicheForest_paper_frozen_candidate"
    / "P6_frozen_breast"
    / "phase_a"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_audit(nodes: pd.DataFrame, edges: pd.DataFrame, backbone: pd.DataFrame) -> pd.DataFrame:
    source_column = "source_unit" if "source_unit" in edges else "source_name"
    target_column = "target_unit" if "target_unit" in edges else "target_name"
    rows = []
    for root_id, local_nodes in nodes.groupby("root_id"):
        root = str(root_id)
        local_edges = edges[edges.root_id.astype(str).eq(root)]
        fine = nx.Graph()
        fine.add_nodes_from(local_nodes.unit_id.astype(str).unique())
        fine.add_edges_from(
            local_edges[[source_column, target_column]].astype(str).itertuples(
                index=False, name=None
            )
        )
        local_backbone = backbone[backbone.root_id.astype(str).eq(root)]
        state = nx.Graph()
        state.add_nodes_from(local_nodes.state_id.astype(str).unique())
        state.add_edges_from(
            local_backbone[["source_state", "target_state"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        root_nodes = local_nodes[local_nodes.is_root.map(bool)]
        rows.append(
            {
                "root_id": root,
                "n_supported_nodes": int(local_nodes.unit_id.astype(str).nunique()),
                "n_fine_edges": int(len(local_edges)),
                "n_branches": int(local_nodes.branch_id.astype(str).nunique()),
                "n_states": int(local_nodes.state_id.astype(str).nunique()),
                "n_root_nodes": int(len(root_nodes)),
                "fine_connected": nx.is_connected(fine) if fine.number_of_nodes() else False,
                "fine_acyclic": nx.is_forest(fine),
                "fine_is_tree": nx.is_tree(fine) if fine.number_of_nodes() else False,
                "state_is_tree": nx.is_tree(state) if state.number_of_nodes() else False,
                "all_edges_candidate_legal": bool(local_edges.candidate_legal.map(bool).all()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-dir", type=Path, default=DEFAULT_METHOD_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    method_dir = args.method_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    module = load_module(
        DEV / "diagnose_phase_a_hierarchical_v10.py",
        "p6_phase_a_v7_1_frozen",
    )
    module.METHOD_VERSION = "PHASE_A_v7_1_state_constrained_fine_all_eligible_attach"
    module.BRANCH_ESTIMATOR_LABEL = "stable_HVG_PCA_KMeans_v9_complexity_0.10"
    module.PRODUCTION_CASE = "hierarchical_branch_program_state_fine"

    meta = ad.read_h5ad(method_dir / "metacell_adata.h5ad")
    membership, roots, supports = module.V2.P.inferred_membership(method_dir, meta.n_obs)
    if args.audit_only:
        selected = {
            "edges": pd.read_csv(output / "T0_FINE_TRAJECTORY_EDGES.csv"),
            "backbone": pd.read_csv(output / "T0_STATE_BACKBONE_EDGES.csv"),
            "nodes": pd.read_csv(output / "T0_STATE_NODE_ASSIGNMENTS.csv"),
        }
    else:
        geometry = module.V2.P.V1.prepare_geometry(meta)
        full_candidate, _, build_audit = module.V2.P.new_candidates(meta, geometry)
        existing_candidate = module.V2.P.current_candidate(method_dir)
        counts = module.dense_counts(meta)
        branch_cache = {
            str(root["root_id"]): module.BRANCH.stable_hvg_kmeans(
                counts, sorted(map(int, supports[str(root["root_id"])]))
            )
            for root in roots
        }
        baseline = module.V2.topological_state_upgrade(
            module.V2.P.construct_t0(
                meta, geometry, existing_candidate, membership, roots, supports, False
            )
        )
        baseline.setdefault("components", pd.DataFrame())
        baseline.setdefault("branch_models", pd.DataFrame())
        selected = module.construct_variant(
            meta,
            full_candidate,
            existing_candidate,
            membership,
            roots,
            supports,
            branch_cache,
            (
                float(build_audit["bridge_expression_guard_q95"]),
                float(build_audit["bridge_spatial_guard_q975"]),
            ),
            module.PRODUCTION_CASE,
        )
        for name, result in (
            ("current_root_conditioned_msf", baseline),
            (module.PRODUCTION_CASE, selected),
        ):
            destination = output / name
            destination.mkdir(exist_ok=True)
            result["edges"].to_csv(destination / "T0_FINE_TRAJECTORY_EDGES.csv", index=False)
            result["backbone"].to_csv(destination / "T0_STATE_BACKBONE_EDGES.csv", index=False)
            result["nodes"].to_csv(destination / "T0_NODE_ASSIGNMENTS.csv", index=False)
            if len(result["components"].columns):
                result["components"].to_csv(destination / "T0_COMPONENT_AUDIT.csv", index=False)
            if len(result["branch_models"].columns):
                result["branch_models"].to_csv(
                    destination / "T0_BRANCH_PROGRAM_AUDIT.csv", index=False
                )
        selected["edges"].to_csv(output / "T0_FINE_TRAJECTORY_EDGES.csv", index=False)
        selected["backbone"].to_csv(output / "T0_STATE_BACKBONE_EDGES.csv", index=False)
        selected["nodes"].to_csv(output / "T0_STATE_NODE_ASSIGNMENTS.csv", index=False)
        selected["components"].to_csv(output / "T0_COMPONENT_AUDIT.csv", index=False)
        selected["branch_models"].to_csv(
            output / "T0_BRANCH_PROGRAM_MODEL_AUDIT.csv", index=False
        )
        selected["root_connectors"].to_csv(
            output / "ROOT_SPECIFIC_CONNECTOR_EDGES.csv", index=False
        )
        selected["excluded_support"].to_csv(
            output / "EXCLUDED_ROOT_SUPPORT_NODES.csv", index=False
        )
        full_candidate.to_csv(output / "C1_MULTIRESOLUTION_CANDIDATE_EDGES.csv", index=False)
        pd.DataFrame([build_audit]).to_csv(output / "CANDIDATE_BUILD_AUDIT.csv", index=False)
    audit = tree_audit(selected["nodes"], selected["edges"], selected["backbone"])
    audit.to_csv(output / "P6_T0_STRUCTURAL_AUDIT.csv", index=False)

    primary = selected["nodes"].sort_values("membership", ascending=False).drop_duplicates("unit_id")
    dominant_counts = primary.groupby("root_id").size().to_dict()
    membership_frame = pd.DataFrame(membership, columns=[str(root["root_id"]) for root in roots])
    membership_frame.insert(0, "unit_id", meta.obs_names.astype(str))
    membership_frame.to_csv(output / "P6_FROZEN_SOFT_MEMBERSHIP.csv", index=False)
    sorted_membership = np.sort(membership, axis=1)
    second = sorted_membership[:, -2] if membership.shape[1] > 1 else np.zeros(meta.n_obs)
    assignment_audit = {
        "n_metacells": int(meta.n_obs),
        "n_roots": int(len(roots)),
        "root_ids": [str(root["root_id"]) for root in roots],
        "dominant_root_counts": {str(k): int(v) for k, v in dominant_counts.items()},
        "shared_membership_ge_0_20": int((second >= 0.20).sum()),
        "shared_membership_ge_0_10": int((second >= 0.10).sum()),
        "median_max_membership": float(pd.Series(membership.max(axis=1)).median()),
    }
    (output / "P6_SOFT_ASSIGNMENT_AUDIT.json").write_text(
        json.dumps(assignment_audit, indent=2), encoding="utf-8"
    )
    status = {
        "phase": "P6_BREAST_PHASE_A",
        "dataset": "xenium_breast",
        "method_version": module.METHOD_VERSION,
        "production_case": module.PRODUCTION_CASE,
        "truth_used_for_construction": False,
        "truth_used_for_scoring": False,
        "real_topology_truth_created": False,
        "input_method_dir": str(method_dir),
        "n_roots": int(len(roots)),
        "all_fine_edges_candidate_legal": bool(audit.all_edges_candidate_legal.all()),
        "all_selected_root_graphs_are_trees": bool(audit.fine_is_tree.all()),
        "all_state_backbones_are_trees": bool(audit.state_is_tree.all()),
        "all_root_supports_connected": bool(audit.fine_connected.all()),
        "complete": True,
    }
    (output / "RUN_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    manifest = pd.DataFrame(
        [
            {"path": str(path), "sha256": sha256(path)}
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "P6_PHASE_A_SHA256_MANIFEST.csv"
        ]
    )
    manifest.to_csv(output / "P6_PHASE_A_SHA256_MANIFEST.csv", index=False)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
