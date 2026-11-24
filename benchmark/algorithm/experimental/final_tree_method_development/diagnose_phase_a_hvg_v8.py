#!/usr/bin/env python3
"""Truth-blind PHASE A diagnostic using stable root-global branch programs.

The candidate graph and soft assignment are inherited unchanged from PHASE A
v3.  Two fixed constructions test whether learned branch programs should be an
annotation channel only or should also constrain the fine-scale forest.
Synthetic truth is loaded only after both variants have been constructed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse


HERE = Path(__file__).resolve().parent


def load_version(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = load_version("run_phase_a_case_v2.py", "phase_a_v2_for_hvg_v8_diagnostic")
BRANCH = load_version("branch_program_estimator_v8.py", "branch_program_estimator_v8_diagnostic")
METHOD_VERSION = "PHASE_A_v4_experimental_hvg_v8"
BRANCH_ESTIMATOR_LABEL = "stable_HVG_PCA_KMeans_v8"
PRODUCTION_CASE = "hvg_quotient"


def topology_candidate(high: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Union the frozen multiresolution and local topology evidence channels."""
    high = high.copy()
    existing = existing.copy()
    high["channel_priority"] = 0
    existing["channel_priority"] = 1
    combined = pd.concat([high, existing], ignore_index=True, sort=False)
    combined["edge_key"] = [
        V2.P.V1.edge_key(int(row.source), int(row.target))
        for row in combined.itertuples(index=False)
    ]
    combined = combined.sort_values(
        ["channel_priority", "D_joint", "source", "target"], kind="mergesort"
    ).drop_duplicates("edge_key", keep="first")
    combined["joint_distance_ecdf"] = V2.P.V1.empirical_rank(
        combined.D_joint.to_numpy(float)
    )
    return combined.drop(columns=["edge_key", "channel_priority"])


def dense_counts(meta) -> np.ndarray:
    return meta.X.toarray() if sparse.issparse(meta.X) else np.asarray(meta.X)


def load_truth_for_metacells(
    truth_run_dir: Path, metacell_truth_dir: Path, fixed_root_input: Path
):
    """Build scorer-only truth for an arbitrary metacell resolution."""
    truth_dir = truth_run_dir / "event" / "truth"
    fixed = pd.read_csv(fixed_root_input)
    roots = fixed["truth_root_id"].astype(str).tolist()
    anchors = dict(
        zip(fixed["truth_root_id"].astype(str), fixed["root_anchor_id"].astype(str))
    )
    state_frame = pd.read_csv(metacell_truth_dir / "METACELL_STATE_COMPOSITION.csv")
    state_frame = state_frame[state_frame["stage"].astype(str).eq("T0")]
    dominant = state_frame.sort_values(
        ["metacell_id", "root_id", "state_probability", "state_id"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["metacell_id", "root_id"])
    state = {
        (str(row.metacell_id), str(row.root_id)): str(row.state_id)
        for row in dominant.itertuples(index=False)
        if float(row.state_probability) > 0
    }
    root_frame = pd.read_csv(metacell_truth_dir / "METACELL_ROOT_COMPOSITION.csv")
    root_probability = {
        (str(row.metacell_id), str(row.root_id)): float(row.root_probability)
        for row in root_frame.itertuples(index=False)
    }
    mapping = pd.read_csv(metacell_truth_dir / "cell_to_metacell_mapping.csv")[
        ["cell_name", "metacell_id"]
    ]
    pt = pd.read_csv(truth_dir / "TRUE_PSEUDOTIME.csv").rename(
        columns={"cell_id": "cell_name"}
    )
    medians = mapping.merge(pt, on="cell_name", how="inner").groupby(
        ["metacell_id", "root_id"], as_index=False
    )["pseudotime"].median()
    pseudotime = {
        (str(row.metacell_id), str(row.root_id)): float(row.pseudotime)
        for row in medians.itertuples(index=False)
    }
    branch_cells = pd.read_csv(truth_dir / "TRUE_CELL_BRANCH.csv")
    branch_cells = branch_cells[
        branch_cells["stage"].astype(str).eq("T0")
    ].rename(columns={"cell_id": "cell_name"})
    branch_cells = mapping.merge(
        branch_cells[["cell_name", "root_id", "branch_id"]],
        on="cell_name", how="inner",
    )
    branch = {
        (str(metacell), str(root)): str(V2.P.mode(group["branch_id"].astype(str).tolist()))
        for (metacell, root), group in branch_cells.groupby(["metacell_id", "root_id"])
    }
    truth_edges = pd.read_csv(truth_dir / "TRUE_T0_EDGES.csv")
    graphs = {}
    for root in roots:
        graph = nx.DiGraph()
        local = truth_edges[truth_edges["root_id"].astype(str).eq(root)]
        graph.add_edges_from(
            (str(row.source), str(row.target)) for row in local.itertuples(index=False)
        )
        graphs[root] = graph
    truth = V2.P.TruthContext(
        roots, anchors, state, branch, root_probability, pseudotime, graphs
    )
    truth.state_probability = {
        (str(row.metacell_id), str(row.root_id), str(row.state_id)): float(row.state_probability)
        for row in state_frame.itertuples(index=False)
    }
    truth.state_support = {}
    threshold = V2.Q.STATE_SUPPORT_THRESHOLD
    for (metacell, root), group in state_frame.groupby(["metacell_id", "root_id"]):
        supported = {
            str(row.state_id)
            for row in group.itertuples(index=False)
            if float(row.state_probability) >= threshold
        }
        if supported:
            truth.state_support[(str(metacell), str(root))] = supported
    truth.state_support_threshold = threshold
    return truth


def construct_variant(
    meta,
    full_candidate,
    existing_candidate,
    membership,
    roots,
    supports,
    branch_cache,
    variant: str,
    branch_penalty: float = 0.0,
):
    high = full_candidate[
        full_candidate.edge_graph_role.astype(str).isin(V2.P.T0_ACTIVE_ROLES)
    ].copy()
    high["joint_distance_ecdf"] = V2.P.V1.empirical_rank(high.D_joint.to_numpy(float))
    topology = topology_candidate(high, existing_candidate)
    union = full_candidate[
        full_candidate.edge_graph_role.astype(str).eq("trajectory_local_union")
    ].copy()
    if len(union):
        union["joint_distance_ecdf"] = V2.P.V1.empirical_rank(union.D_joint.to_numpy(float))
    legal = {
        V2.P.V1.edge_key(int(row.source), int(row.target))
        for row in pd.concat(
            [full_candidate, existing_candidate], ignore_index=True, sort=False
        ).itertuples(index=False)
        if bool(row.trajectory_edge_eligible)
    }
    names = [str(name) for name in meta.obs_names]
    edge_rows, node_rows, backbone_frames, component_rows, model_rows = [], [], [], [], []
    connector_total = 0

    for root in roots:
        root_id = str(root["root_id"])
        root_pos, root_index = int(root["root_pos"]), int(root["root_index"])
        support = sorted(map(int, supports[root_id]))
        branches, audits = branch_cache[root_id]

        topology_graph = V2.P.V1.graph_for_support(
            topology, support, membership[:, root_pos]
        )
        if variant == "hvg_labels_combined_mst":
            frozen_topology = V2.P.V1.component_forest(topology_graph)
        elif variant == "hvg_quotient":
            frozen_topology = V2.P.V1.quotient_branch_forest(topology_graph, branches)
        elif variant == "hvg_branch_regularized_mst":
            regularized = topology_graph.copy()
            for left, right, data in regularized.edges(data=True):
                crosses_programs = branches[int(left)] != branches[int(right)]
                data["base_weight"] = float(data.get("weight", 0.0))
                data["branch_boundary_penalty"] = (
                    float(branch_penalty) if crosses_programs else 0.0
                )
                data["weight"] = data["base_weight"] * (
                    1.0 + data["branch_boundary_penalty"]
                )
            frozen_topology = V2.P.V1.component_forest(regularized)
        else:
            raise ValueError(f"Unknown diagnostic variant: {variant}")

        union_graph = (
            V2.P.V1.graph_for_support(union, support, membership[:, root_pos])
            if len(union)
            else nx.Graph()
        )
        union_graph.add_nodes_from(support)
        final, connectors = V2.V4.attach_components_with_union(
            V2.P.V1, frozen_topology, union_graph
        )
        connector_keys = set(connectors)
        connector_total += len(connectors)
        for left, right in final.edges():
            final[left][right]["connector_only"] = (
                V2.P.V1.edge_key(int(left), int(right)) in connector_keys
            )

        tau, component = V2.P.V1.rooted_tau(final, root_index, membership[:, root_pos])
        states = V2.topology_state_labels(final, root_index, root_id)
        backbone_frames.append(V2.backbone_from_states(root_id, final, states, tau, names))
        for audit in audits:
            model_rows.append({"root_id": root_id, "variant": variant, **audit})
        for left, right, data in final.edges(data=True):
            edge_rows.append({
                "root_id": root_id,
                "source": int(left),
                "target": int(right),
                "source_name": names[int(left)],
                "target_name": names[int(right)],
                "edge_cost": float(data.get("weight", 0.0)),
                "edge_graph_role": str(data.get("edge_graph_role", "")),
                "candidate_legal": V2.P.V1.edge_key(int(left), int(right)) in legal,
                "connector_only": bool(data.get("connector_only", False)),
            })
        for node in support:
            node_rows.append({
                "root_id": root_id,
                "node_id": int(node),
                "unit_id": names[int(node)],
                "membership": float(membership[int(node), root_pos]),
                "pseudotime": float(tau.get(int(node), 0.0)),
                "branch_id": f"{root_id}_{branches[int(node)]}",
                "state_id": states[int(node)],
                "component_id": int(component.get(int(node), -1)),
                "is_root": int(node) == root_index,
            })
        for cid, nodes in enumerate(sorted(nx.connected_components(final), key=min)):
            component_rows.append({
                "root_id": root_id,
                "component_id": cid,
                "n_nodes": len(nodes),
                "contains_root": root_index in nodes,
                "n_fine_edges": final.subgraph(nodes).number_of_edges(),
                "n_states": len({states[int(node)] for node in nodes}),
                "n_connectors": sum(
                    V2.P.V1.edge_key(int(left), int(right)) in connector_keys
                    for left, right in final.subgraph(nodes).edges()
                ),
            })
    return {
        "edges": pd.DataFrame(edge_rows),
        "nodes": pd.DataFrame(node_rows),
        "backbone": pd.concat(backbone_frames, ignore_index=True),
        "components": pd.DataFrame(component_rows),
        "branch_models": pd.DataFrame(model_rows),
        "connector_total": connector_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="CORE_05")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method-dir", type=Path)
    parser.add_argument("--truth-run-dir", type=Path)
    parser.add_argument("--metacell-truth-dir", type=Path)
    parser.add_argument("--fixed-root-input", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    method_dir = (
        args.method_dir.resolve()
        if args.method_dir
        else args.run_dir.resolve() / "generic_upstream_v1_exact_root" / args.dataset
    )
    truth_run_dir = (
        args.truth_run_dir.resolve() if args.truth_run_dir else args.run_dir.resolve()
    )
    metacell_truth_dir = (
        args.metacell_truth_dir.resolve()
        if args.metacell_truth_dir
        else truth_run_dir / "formal_metacell"
    )
    fixed_root_input = (
        args.fixed_root_input.resolve()
        if args.fixed_root_input
        else truth_run_dir / "generic_upstream_v1_exact_root" / "FIXED_ROOT_INPUT.csv"
    )
    meta = ad.read_h5ad(method_dir / "metacell_adata.h5ad")
    membership, roots, supports = V2.P.inferred_membership(method_dir, meta.n_obs)
    geometry = V2.P.V1.prepare_geometry(meta)
    full_candidate, _, build_audit = V2.P.new_candidates(meta, geometry)
    existing_candidate = V2.P.current_candidate(method_dir)
    counts = dense_counts(meta)
    branch_cache = {
        str(root["root_id"]): BRANCH.stable_hvg_kmeans(
            counts, sorted(map(int, supports[str(root["root_id"])]))
        )
        for root in roots
    }

    current = V2.topological_state_upgrade(
        V2.P.construct_t0(
            meta, geometry, existing_candidate, membership, roots, supports, False
        )
    )
    current.setdefault("components", pd.DataFrame())
    current.setdefault("branch_models", pd.DataFrame())
    variants = {
        "current_root_conditioned_msf": current,
        **{
        name: construct_variant(
            meta, full_candidate, existing_candidate, membership, roots, supports,
            branch_cache, name,
        )
        for name in ("hvg_labels_combined_mst", "hvg_quotient")
        },
        **{
            f"hvg_branch_regularized_{int(round(penalty * 100)):03d}": construct_variant(
                meta, full_candidate, existing_candidate, membership, roots, supports,
                branch_cache, "hvg_branch_regularized_mst", penalty,
            )
            for penalty in (0.10, 0.20, 0.35, 0.50)
        },
    }

    # Truth remains outside both production constructors.
    truth = load_truth_for_metacells(
        truth_run_dir, metacell_truth_dir, fixed_root_input
    )
    metric_frames = []
    for name, result in variants.items():
        metric_frames.append(V2.dual_scale_metrics(args.run_label, name, result, truth))
        case_dir = output / name
        case_dir.mkdir(exist_ok=True)
        result["edges"].to_csv(case_dir / "T0_FINE_TRAJECTORY_EDGES.csv", index=False)
        result["backbone"].to_csv(case_dir / "T0_STATE_BACKBONE_EDGES.csv", index=False)
        result["nodes"].to_csv(case_dir / "T0_NODE_ASSIGNMENTS.csv", index=False)
        if len(result["components"].columns):
            result["components"].to_csv(case_dir / "T0_COMPONENT_AUDIT.csv", index=False)
        if len(result["branch_models"].columns):
            result["branch_models"].to_csv(case_dir / "T0_BRANCH_PROGRAM_AUDIT.csv", index=False)
    pd.concat(metric_frames, ignore_index=True).to_csv(
        output / "HVG_V8_DIAGNOSTIC_METRICS.csv", index=False
    )
    selected = variants[PRODUCTION_CASE]
    selected["edges"].to_csv(output / "T0_FINE_TRAJECTORY_EDGES.csv", index=False)
    selected["backbone"].to_csv(output / "T0_STATE_BACKBONE_EDGES.csv", index=False)
    selected["nodes"].to_csv(output / "T0_STATE_NODE_ASSIGNMENTS.csv", index=False)
    selected["components"].to_csv(output / "T0_COMPONENT_AUDIT.csv", index=False)
    selected["branch_models"].to_csv(
        output / "T0_BRANCH_PROGRAM_MODEL_AUDIT.csv", index=False
    )
    full_candidate.to_csv(output / "C1_MULTIRESOLUTION_CANDIDATE_EDGES.csv", index=False)
    pd.DataFrame([build_audit]).to_csv(output / "CANDIDATE_BUILD_AUDIT.csv", index=False)
    candidate_frame = pd.concat([
        V2.R.directional_candidate_audit(
            args.run_label, "current_single_scale", existing_candidate, truth
        ),
        V2.R.directional_candidate_audit(
            args.run_label, "new_multiresolution_full", full_candidate, truth
        ),
        V2.R.directional_candidate_audit(
            args.run_label, "complete_graph_ceiling", V2.R.complete_candidate(meta), truth
        ),
    ], ignore_index=True)
    V2.R.add_ceiling_columns(candidate_frame).to_csv(
        output / "PHASE_A_CANDIDATE_AUDIT.csv", index=False
    )
    feasibility = V2.Q.event_feasibility(
        args.run_label, truth_run_dir, full_candidate, selected, truth
    )
    feasibility.to_csv(output / "PHASE_A_EVENT_FEASIBILITY.csv", index=False)
    feasibility.to_csv(output / "ORACLE_EVENT_FEASIBILITY_ON_T0.csv", index=False)
    status = {
        "phase": "A",
        "version": METHOD_VERSION,
        "production_case": PRODUCTION_CASE,
        "candidate_definition": "frozen_multiresolution_plus_existing_eligible_local",
        "soft_assignment_definition": "inherited_frozen_root_conditioned_soft_assignment",
        "branch_program_estimator": BRANCH_ESTIMATOR_LABEL,
        "truth_used_for_production_construction": False,
        "truth_used_for_audit_only": True,
        "all_fine_edges_candidate_legal": bool(selected["edges"].candidate_legal.all()),
        "all_selected_root_graphs_are_forests": bool(
            pd.concat(metric_frames).query("case == @PRODUCTION_CASE").is_forest.all()
        ),
        "relation_detector_called": False,
        "refinement_operator_called": False,
        "complete": True,
    }
    (output / "RUN_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
