#!/usr/bin/env python3
"""Truth-blind PHASE A diagnostic using a hierarchical branch-state T0.

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
from sklearn.metrics import adjusted_rand_score


HERE = Path(__file__).resolve().parent


def load_version(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = load_version("run_phase_a_case_v2.py", "phase_a_v2_for_hvg_v8_diagnostic")
BRANCH = load_version("branch_program_estimator_v9.py", "branch_program_estimator_v9_production")
RESIDUAL = load_version("branch_program_estimator_v10.py", "branch_program_estimator_v10")
METHOD_VERSION = "PHASE_A_v5_3_guarded_hierarchical_graph_partition_state"
BRANCH_ESTIMATOR_LABEL = "stable_HVG_PCA_KMeans_v9_complexity_0.10"
PRODUCTION_CASE = "hierarchical_graph_partition_state"
TRUNK_TAU_BOUNDARY = 0.30
MIN_STATE_SIZE = 4
MAX_BRANCH_SEGMENTS = 3
INVALID_STATE_TRANSITION_PENALTY = 2.0
STATE_RESOLUTIONS = (0.50, 0.75, 1.00, 1.25, 1.50)
STATE_COMPLEXITY_PENALTY = 0.05
STATE_BOOTSTRAPS = 10


def hierarchical_state_labels(
    nodes: list[int], root: int, branches: dict[int, str], tau: dict[int, float],
) -> dict[int, str]:
    """Create a shared trunk followed by ordered states within each program."""
    labels: dict[int, str] = {int(root): "root"}
    trunk = [
        int(node) for node in nodes
        if int(node) != int(root) and float(tau.get(int(node), 0.0)) <= TRUNK_TAU_BOUNDARY
    ]
    for node in trunk:
        labels[node] = "trunk"
    post = [node for node in map(int, nodes) if node not in labels]
    by_branch: dict[str, list[int]] = {}
    for node in post:
        by_branch.setdefault(str(branches[node]), []).append(node)
    for branch, local in sorted(by_branch.items()):
        ordered = sorted(local, key=lambda node: (float(tau[node]), int(node)))
        n_segments = min(
            MAX_BRANCH_SEGMENTS,
            max(1, int(round(len(ordered) / max(MIN_STATE_SIZE, 1)))),
        )
        for position, node in enumerate(ordered):
            segment = min(n_segments - 1, position * n_segments // len(ordered))
            labels[node] = f"{branch}_s{segment}"
    return labels


def connected_state_labels(
    graph: nx.Graph, labels: dict[int, str]
) -> dict[int, str]:
    """Split repeated disconnected state regions before graph contraction."""
    result: dict[int, str] = {}
    for label in sorted(set(labels.values())):
        local_nodes = {node for node, value in labels.items() if value == label}
        induced = graph.subgraph(local_nodes)
        pieces = sorted(nx.connected_components(induced), key=min)
        for index, piece in enumerate(pieces):
            suffix = f"_p{index}" if len(pieces) > 1 else ""
            for node in piece:
                result[int(node)] = f"{label}{suffix}"
    return result


def logical_state(label: str) -> str:
    return str(label).rsplit("_p", 1)[0]


def state_transition_allowed(left: str, right: str) -> bool:
    left, right = logical_state(left), logical_state(right)
    if left == right:
        return True
    if {left, right} == {"root", "trunk"}:
        return True
    if "trunk" in {left, right}:
        other = right if left == "trunk" else left
        return other.endswith("_s0")
    if "root" in {left, right}:
        return False
    try:
        left_branch, left_segment = left.rsplit("_s", 1)
        right_branch, right_segment = right.rsplit("_s", 1)
        return left_branch == right_branch and abs(int(left_segment) - int(right_segment)) <= 1
    except (ValueError, TypeError):
        return False


def regularize_state_transitions(
    graph: nx.Graph, labels: dict[int, str]
) -> nx.Graph:
    result = graph.copy()
    for left, right, data in result.edges(data=True):
        allowed = state_transition_allowed(labels[int(left)], labels[int(right)])
        base = float(data.get("weight", 0.0))
        data["base_weight"] = base
        data["state_transition_allowed"] = allowed
        data["state_transition_penalty"] = 0.0 if allowed else INVALID_STATE_TRANSITION_PENALTY
        data["weight"] = base * (1.0 + data["state_transition_penalty"])
    return result


def state_backbone_support_tree(
    graph: nx.Graph, labels: dict[int, str]
) -> nx.Graph:
    """Return fine support edges for the MST of the contracted state graph."""
    quotient = nx.Graph()
    quotient.add_nodes_from(sorted(set(labels.values())))
    for left, right, data in graph.edges(data=True):
        source_state, target_state = labels[int(left)], labels[int(right)]
        if source_state == target_state:
            continue
        key = tuple(sorted((source_state, target_state)))
        trial = (
            float(data.get("weight", 0.0)), min(int(left), int(right)),
            max(int(left), int(right)), int(left), int(right), dict(data),
        )
        current = quotient.get_edge_data(*key)
        if current is None or trial[:3] < current["trial"][:3]:
            quotient.add_edge(*key, weight=trial[0], trial=trial)
    support = nx.Graph()
    for component in nx.connected_components(quotient):
        local = quotient.subgraph(component)
        for left_state, right_state, qdata in nx.minimum_spanning_tree(
            local, weight="weight"
        ).edges(data=True):
            _, _, _, left, right, data = qdata["trial"]
            support.add_edge(left, right, **data)
    return support


def state_constrained_fine_forest(
    graph: nx.Graph, labels: dict[int, str]
) -> tuple[nx.Graph, dict[int, str]]:
    """Build fine trees within connected states and an admissible state MST.

    The input labels come only from the learned expression program and a
    preliminary root-conditioned pseudotime.  Splitting disconnected copies
    prevents state contraction from creating non-local shortcuts.
    """
    connected = connected_state_labels(graph, labels)
    result = nx.Graph()
    result.add_nodes_from(graph.nodes())
    for state in sorted(set(connected.values())):
        state_nodes = {node for node, value in connected.items() if value == state}
        local = graph.subgraph(state_nodes)
        for component in nx.connected_components(local):
            induced = local.subgraph(component)
            if induced.number_of_edges():
                result.add_edges_from(
                    nx.minimum_spanning_tree(induced, weight="weight").edges(data=True)
                )

    quotient = nx.Graph()
    quotient.add_nodes_from(sorted(set(connected.values())))
    for left, right, data in graph.edges(data=True):
        source_state, target_state = connected[int(left)], connected[int(right)]
        if source_state == target_state:
            continue
        if not state_transition_allowed(source_state, target_state):
            continue
        key = tuple(sorted((source_state, target_state)))
        trial = (
            float(data.get("weight", 0.0)), min(int(left), int(right)),
            max(int(left), int(right)), int(left), int(right), dict(data),
        )
        current = quotient.get_edge_data(*key)
        if current is None or trial[:3] < current["trial"][:3]:
            quotient.add_edge(*key, weight=trial[0], trial=trial)
    for component in nx.connected_components(quotient):
        local = quotient.subgraph(component)
        for _, _, qdata in nx.minimum_spanning_tree(local, weight="weight").edges(data=True):
            _, _, _, left, right, data = qdata["trial"]
            result.add_edge(left, right, **data)
    return result, connected


def attach_components_with_all_eligible(
    forest: nx.Graph, candidate_graph: nx.Graph
) -> tuple[nx.Graph, list[tuple[int, int]]]:
    """Connect forest components using any already-eligible candidate edge."""
    result = forest.copy()
    components = list(nx.connected_components(result))
    node_component = {
        int(node): index for index, component in enumerate(components)
        for node in component
    }
    quotient = nx.Graph()
    quotient.add_nodes_from(range(len(components)))
    best = {}
    for left, right, data in candidate_graph.edges(data=True):
        source_component = node_component[int(left)]
        target_component = node_component[int(right)]
        if source_component == target_component:
            continue
        key = tuple(sorted((source_component, target_component)))
        trial = (
            float(data.get("weight", 0.0)), min(int(left), int(right)),
            max(int(left), int(right)), int(left), int(right), dict(data),
        )
        if key not in best or trial[:3] < best[key][:3]:
            best[key] = trial
    for (source_component, target_component), trial in best.items():
        quotient.add_edge(
            source_component, target_component, weight=trial[0], trial=trial
        )
    used = []
    for component in nx.connected_components(quotient):
        local = quotient.subgraph(component)
        for _, _, data in nx.minimum_spanning_tree(local, weight="weight").edges(data=True):
            _, _, _, left, right, edge_data = data["trial"]
            result.add_edge(left, right, **edge_data)
            used.append(V2.P.V1.edge_key(left, right))
    return result, used


def _partition_labels(nodes: list[int], communities: list[set[int]]) -> np.ndarray:
    lookup = {
        int(node): index for index, community in enumerate(communities)
        for node in community
    }
    return np.asarray([lookup[int(node)] for node in nodes], dtype=int)


def stable_graph_partitions(
    graph: nx.Graph, tau: dict[int, float], minimum_states: int,
):
    """Select a connected graph abstraction by stability and complexity."""
    weighted = graph.copy()
    for left, right, data in weighted.edges(data=True):
        data["similarity"] = float(np.exp(-float(data.get("weight", 0.0))))
    nodes = sorted(map(int, weighted.nodes()))
    candidates = []
    for resolution in STATE_RESOLUTIONS:
        communities = [
            set(map(int, community)) for community in nx.community.louvain_communities(
                weighted, weight="similarity", resolution=resolution, seed=20260811
            )
        ]
        communities = sorted(communities, key=min)
        labels = _partition_labels(nodes, communities)
        modularity = float(nx.community.modularity(
            weighted, communities, weight="similarity", resolution=resolution
        ))
        stability_values = []
        edges = sorted(weighted.edges())
        rng = np.random.default_rng(20260811 + int(round(100 * resolution)))
        for iteration in range(STATE_BOOTSTRAPS):
            trial = weighted.copy()
            n_drop = int(round(0.10 * len(edges)))
            if n_drop:
                drop = rng.choice(len(edges), size=n_drop, replace=False)
                trial.remove_edges_from([edges[index] for index in drop])
            trial_communities = [
                set(map(int, community)) for community in nx.community.louvain_communities(
                    trial, weight="similarity", resolution=resolution,
                    seed=20260811 + iteration + 1,
                )
            ]
            stability_values.append(adjusted_rand_score(
                labels, _partition_labels(nodes, trial_communities)
            ))
        stability = float(np.mean(stability_values))
        k = len(communities)
        score = (
            max(stability, 0.0) + 0.25 * modularity
            - STATE_COMPLEXITY_PENALTY
            * max(k - minimum_states, 0) / np.sqrt(max(len(nodes), 1))
        )
        candidates.append({
            "resolution": resolution, "communities": communities,
            "labels": labels, "selected_k": k, "modularity": modularity,
            "edge_bootstrap_stability": stability, "selection_score": score,
            "cluster_sizes": ";".join(map(str, sorted(map(len, communities)))),
        })
    eligible = [
        candidate for candidate in candidates
        if candidate["selected_k"] >= int(minimum_states)
    ]
    selected = max(
        eligible or candidates,
        key=lambda item: (item["selection_score"], -item["selected_k"], -item["resolution"]),
    )
    order = sorted(
        range(selected["selected_k"]),
        key=lambda label: (
            float(np.median([
                tau[nodes[index]] for index in np.flatnonzero(selected["labels"] == label)
            ])),
            min(nodes[index] for index in np.flatnonzero(selected["labels"] == label)),
        ),
    )
    remap = {old: new for new, old in enumerate(order)}
    labels = {
        node: f"state_{remap[int(label)]}"
        for node, label in zip(nodes, selected["labels"])
    }
    audit = [{
        "model_channel": "candidate_graph_state_partition",
        "candidate_resolution": candidate["resolution"],
        "candidate_k": candidate["selected_k"],
        "selected": candidate is selected,
        "selected_resolution": selected["resolution"],
        "selected_k": selected["selected_k"],
        "cluster_sizes": candidate["cluster_sizes"],
        "modularity": candidate["modularity"],
        "edge_bootstrap_stability": candidate["edge_bootstrap_stability"],
        "selection_score": candidate["selection_score"],
        "complexity_penalty": STATE_COMPLEXITY_PENALTY,
        "minimum_states_from_branch_programs": int(minimum_states),
    } for candidate in candidates]
    return labels, audit


def guarded_root_connectors(
    full_candidate: pd.DataFrame,
    reserve: pd.DataFrame,
    support: list[int],
    root_index: int,
    membership: np.ndarray,
    expression_guard: float,
    spatial_guard: float,
):
    """Promote only a component-MST of doubly guarded reserve edges."""
    base = V2.P.V1.graph_for_support(reserve, set(support), membership)
    components = [set(map(int, part)) for part in nx.connected_components(base)]
    node_component = {
        node: index for index, component in enumerate(components) for node in component
    }
    anchor_component = node_component[int(root_index)]
    guarded = full_candidate[
        ~full_candidate.trajectory_edge_eligible.astype(bool)
        & full_candidate.D_expression.astype(float).le(float(expression_guard))
        & full_candidate.D_spatial.astype(float).le(float(spatial_guard))
        & full_candidate.source.astype(int).isin(support)
        & full_candidate.target.astype(int).isin(support)
    ].copy()
    best: dict[tuple[int, int], tuple[tuple[float, int, int], pd.Series]] = {}
    for row in guarded.itertuples(index=False):
        left_component = node_component[int(row.source)]
        right_component = node_component[int(row.target)]
        if left_component == right_component:
            continue
        key = tuple(sorted((left_component, right_component)))
        rank = (float(row.D_joint), min(int(row.source), int(row.target)), max(int(row.source), int(row.target)))
        series = guarded.loc[
            (guarded.source.astype(int) == int(row.source))
            & (guarded.target.astype(int) == int(row.target))
        ].iloc[0].copy()
        if key not in best or rank < best[key][0]:
            best[key] = (rank, series)
    component_graph = nx.Graph()
    component_graph.add_nodes_from(range(len(components)))
    for (left, right), (rank, _) in best.items():
        component_graph.add_edge(left, right, weight=rank[0])
    reachable_components = set(nx.node_connected_component(
        component_graph, anchor_component
    ))
    accepted_support = sorted({
        node for component_index in reachable_components
        for node in components[component_index]
    })
    excluded_support = sorted(set(support) - set(accepted_support))
    selected_rows = []
    local_component_graph = component_graph.subgraph(reachable_components)
    if local_component_graph.number_of_edges():
        tree = nx.minimum_spanning_tree(local_component_graph, weight="weight")
        for left, right in tree.edges():
            row = best[tuple(sorted((int(left), int(right))))][1].copy()
            row["trajectory_edge_eligible"] = True
            # The legacy attachment helper consumes the local-union role.
            # Provenance is restored in the emitted edge/audit tables below.
            row["edge_graph_role"] = "trajectory_local_union"
            row["root_specific_connector"] = True
            selected_rows.append(row)
    connectors = pd.DataFrame(selected_rows)
    root_reserve = reserve.copy()
    root_reserve["root_specific_connector"] = False
    if len(connectors):
        root_reserve = pd.concat([root_reserve, connectors], ignore_index=True, sort=False)
        root_reserve["joint_distance_ecdf"] = V2.P.V1.empirical_rank(
            root_reserve.D_joint.to_numpy(float)
        )
    return root_reserve, accepted_support, excluded_support, connectors


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
    connector_guards,
    variant: str,
    branch_penalty: float = 0.0,
):
    high = full_candidate[
        full_candidate.edge_graph_role.astype(str).isin(V2.P.T0_ACTIVE_ROLES)
    ].copy()
    high["joint_distance_ecdf"] = V2.P.V1.empirical_rank(high.D_joint.to_numpy(float))
    topology = topology_candidate(high, existing_candidate)
    reserve = topology_candidate(
        full_candidate[full_candidate.trajectory_edge_eligible.astype(bool)].copy(),
        existing_candidate,
    )
    legal = {
        V2.P.V1.edge_key(int(row.source), int(row.target))
        for row in pd.concat(
            [full_candidate, existing_candidate], ignore_index=True, sort=False
        ).itertuples(index=False)
        if bool(row.trajectory_edge_eligible)
    }
    names = [str(name) for name in meta.obs_names]
    edge_rows, node_rows, backbone_frames, component_rows, model_rows = [], [], [], [], []
    connector_audit_rows, excluded_rows = [], []
    connector_total = 0

    for root in roots:
        root_id = str(root["root_id"])
        root_pos, root_index = int(root["root_pos"]), int(root["root_index"])
        original_support = sorted(map(int, supports[root_id]))
        branches, audits = branch_cache[root_id]

        root_reserve, support, excluded, promoted = guarded_root_connectors(
            full_candidate, reserve, original_support, root_index,
            membership[:, root_pos], connector_guards[0], connector_guards[1],
        )
        if excluded:
            for node in excluded:
                excluded_rows.append({
                    "root_id": root_id, "node_id": int(node),
                    "unit_id": names[int(node)],
                    "membership": float(membership[int(node), root_pos]),
                    "reason": "no_double_guard_path_to_root_component",
                })
        if len(promoted):
            for row in promoted.itertuples(index=False):
                connector_audit_rows.append({
                    "root_id": root_id, "source": int(row.source), "target": int(row.target),
                    "source_name": str(row.source_name), "target_name": str(row.target_name),
                    "D_expression": float(row.D_expression),
                    "D_spatial": float(row.D_spatial), "D_joint": float(row.D_joint),
                    "expression_guard": float(connector_guards[0]),
                    "spatial_guard": float(connector_guards[1]),
                    "connector_role": "component_MST_double_guard",
                })
        root_legal = set(legal) | {
            V2.P.V1.edge_key(int(row.source), int(row.target))
            for row in promoted.itertuples(index=False)
        }
        promoted_keys = {
            V2.P.V1.edge_key(int(row.source), int(row.target))
            for row in promoted.itertuples(index=False)
        }
        branches = {node: label for node, label in branches.items() if node in set(support)}

        topology_graph = V2.P.V1.graph_for_support(
            topology, support, membership[:, root_pos]
        )
        state_labels = None
        if variant in {
            "hierarchical_graph_partition_state",
            "hierarchical_branch_program_state",
            "hierarchical_branch_program_state_fine",
        }:
            reserve_graph = V2.P.V1.graph_for_support(
                root_reserve, set(support), membership[:, root_pos]
            )
            frozen_topology = V2.P.V1.quotient_branch_forest(
                topology_graph, branches
            )
        elif variant == "hvg_labels_combined_mst":
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
            V2.P.V1.graph_for_support(root_reserve, set(support), membership[:, root_pos])
            if len(root_reserve)
            else nx.Graph()
        )
        union_graph.add_nodes_from(support)
        if variant == "hierarchical_branch_program_state_fine":
            final, connectors = attach_components_with_all_eligible(
                frozen_topology, union_graph
            )
        else:
            final, connectors = V2.V4.attach_components_with_union(
                V2.P.V1, frozen_topology, union_graph
            )
        connector_keys = set(connectors)
        tau, component = V2.P.V1.rooted_tau(final, root_index, membership[:, root_pos])
        if variant == "hierarchical_graph_partition_state":
            state_labels, state_audits = stable_graph_partitions(
                union_graph, tau, minimum_states=len(set(branches.values())) + 1
            )
            audits = list(audits) + state_audits
            states = state_labels
            backbone_graph = state_backbone_support_tree(union_graph, state_labels)
        elif variant == "hierarchical_branch_program_state":
            states = hierarchical_state_labels(
                support, root_index, branches, tau
            )
            # States are biological program-by-time bins. Their contraction is
            # allowed to pool disconnected fine-tree representatives because
            # the backbone is a coarse abstraction, and is reconstructed from
            # the full candidate graph rather than inherited from fine-MST
            # accidents.
            backbone_graph = state_backbone_support_tree(union_graph, states)
            audits = list(audits) + [{
                "model_channel": "branch_program_pseudotime_state_partition",
                "candidate_resolution": np.nan,
                "candidate_k": len(set(states.values())),
                "selected": True,
                "selected_resolution": np.nan,
                "selected_k": len(set(states.values())),
                "cluster_sizes": ";".join(map(str, sorted(
                    pd.Series(states).value_counts().astype(int).tolist()
                ))),
                "modularity": np.nan,
                "edge_bootstrap_stability": np.nan,
                "selection_score": np.nan,
                "complexity_penalty": 0.0,
                "minimum_states_from_branch_programs": len(set(branches.values())) + 1,
            }]
        elif variant == "hierarchical_branch_program_state_fine":
            preliminary_states = hierarchical_state_labels(
                support, root_index, branches, tau
            )
            frozen_topology, connected_states = state_constrained_fine_forest(
                union_graph, preliminary_states
            )
            final, connectors = attach_components_with_all_eligible(
                frozen_topology, union_graph
            )
            connector_keys = set(connectors)
            tau, component = V2.P.V1.rooted_tau(
                final, root_index, membership[:, root_pos]
            )
            states = preliminary_states
            backbone_graph = state_backbone_support_tree(union_graph, states)
            invalid_connectors = sum(
                not state_transition_allowed(
                    connected_states[int(left)], connected_states[int(right)]
                )
                for left, right in connectors
            )
            audits = list(audits) + [{
                "model_channel": "state_constrained_fine_tree",
                "candidate_resolution": np.nan,
                "candidate_k": len(set(states.values())),
                "selected": True,
                "selected_resolution": np.nan,
                "selected_k": len(set(states.values())),
                "cluster_sizes": ";".join(map(str, sorted(
                    pd.Series(states).value_counts().astype(int).tolist()
                ))),
                "modularity": np.nan,
                "edge_bootstrap_stability": np.nan,
                "selection_score": np.nan,
                "complexity_penalty": 0.0,
                "minimum_states_from_branch_programs": len(set(branches.values())) + 1,
                "state_disconnected_pieces": len(set(connected_states.values())),
                "fallback_component_connectors": len(connectors),
                "invalid_state_fallback_connectors": int(invalid_connectors),
            }]
        else:
            states = V2.topology_state_labels(final, root_index, root_id)
            backbone_graph = final
        connector_total += len(connectors)
        for left, right in final.edges():
            final[left][right]["connector_only"] = (
                V2.P.V1.edge_key(int(left), int(right)) in connector_keys
            )
        backbone_frames.append(V2.backbone_from_states(
            root_id, backbone_graph, states, tau, names
        ))
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
                "edge_graph_role": (
                    "trajectory_guarded_component_connector"
                    if V2.P.V1.edge_key(int(left), int(right)) in promoted_keys
                    else str(data.get("edge_graph_role", ""))
                ),
                "candidate_legal": V2.P.V1.edge_key(int(left), int(right)) in root_legal,
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
        "root_connectors": pd.DataFrame(connector_audit_rows),
        "excluded_support": pd.DataFrame(excluded_rows),
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
    parser.add_argument("--production-only", action="store_true")
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
    if args.production_only:
        variants = {
            "current_root_conditioned_msf": current,
            PRODUCTION_CASE: construct_variant(
                meta, full_candidate, existing_candidate, membership, roots, supports,
                branch_cache,
                (float(build_audit["bridge_expression_guard_q95"]),
                 float(build_audit["bridge_spatial_guard_q975"])),
                PRODUCTION_CASE,
            ),
        }
    else:
        variants = {
            "current_root_conditioned_msf": current,
            **{
        name: construct_variant(
            meta, full_candidate, existing_candidate, membership, roots, supports,
            branch_cache,
            (float(build_audit["bridge_expression_guard_q95"]),
             float(build_audit["bridge_spatial_guard_q975"])),
            name,
        )
        for name in (
            "hvg_labels_combined_mst",
            "hvg_quotient",
            "hierarchical_graph_partition_state",
            "hierarchical_branch_program_state",
        )
        },
        **{
            f"hvg_branch_regularized_{int(round(penalty * 100)):03d}": construct_variant(
                meta, full_candidate, existing_candidate, membership, roots, supports,
                branch_cache,
                (float(build_audit["bridge_expression_guard_q95"]),
                 float(build_audit["bridge_spatial_guard_q975"])),
                "hvg_branch_regularized_mst", penalty,
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
    selected["root_connectors"].to_csv(
        output / "ROOT_SPECIFIC_CONNECTOR_EDGES.csv", index=False
    )
    selected["excluded_support"].to_csv(
        output / "EXCLUDED_ROOT_SUPPORT_NODES.csv", index=False
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
    backbone_tree_checks = []
    for root_id, local_nodes in selected["nodes"].groupby("root_id"):
        local_edges = selected["backbone"][
            selected["backbone"].root_id.astype(str).eq(str(root_id))
        ]
        graph = nx.Graph()
        graph.add_nodes_from(local_nodes.state_id.astype(str).unique())
        graph.add_edges_from(local_edges[["source_state", "target_state"]].itertuples(
            index=False, name=None
        ))
        backbone_tree_checks.append(nx.is_tree(graph))
    status = {
        "phase": "A",
        "version": METHOD_VERSION,
        "production_case": PRODUCTION_CASE,
        "candidate_definition": "multiresolution_plus_existing_eligible_local_with_typed_connectors",
        "soft_assignment_definition": "inherited_frozen_root_conditioned_soft_assignment",
        "branch_program_estimator": BRANCH_ESTIMATOR_LABEL,
        "state_abstraction": {
            "partition": (
                "branch_program_by_preliminary_pseudotime"
                if PRODUCTION_CASE in {
                    "hierarchical_branch_program_state",
                    "hierarchical_branch_program_state_fine",
                }
                else "Louvain_on_root_conditioned_candidate_graph"
            ),
            "resolution_grid": STATE_RESOLUTIONS,
            "edge_bootstraps": STATE_BOOTSTRAPS,
            "complexity_penalty": STATE_COMPLEXITY_PENALTY,
        },
        "truth_used_for_production_construction": False,
        "truth_used_for_audit_only": True,
        "all_fine_edges_candidate_legal": bool(selected["edges"].candidate_legal.all()),
        "all_selected_root_graphs_are_forests": bool(
            pd.concat(metric_frames).query("case == @PRODUCTION_CASE").is_forest.all()
        ),
        "all_state_backbones_are_trees": bool(all(backbone_tree_checks)),
        "all_root_supports_connected": bool(
            (selected["components"].groupby("root_id").size() == 1).all()
            and selected["components"].contains_root.map(bool).all()
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
