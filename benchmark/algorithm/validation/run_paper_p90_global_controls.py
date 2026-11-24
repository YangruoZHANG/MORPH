#!/usr/bin/env python3
"""Construct and audit P9 global controls before biological scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def choose_center(tree: nx.Graph) -> int:
    candidates = []
    for node, distances in nx.all_pairs_dijkstra_path_length(tree, weight="D_joint"):
        values = list(distances.values())
        candidates.append((max(values), sum(values), int(node)))
    return min(candidates)[2]


def root_arms(tree: nx.Graph, root: int) -> dict[int, str]:
    graph = tree.copy()
    neighbors = sorted(graph.neighbors(root))
    graph.remove_node(root)
    labels = {}
    sizes = []
    for index, neighbor in enumerate(neighbors):
        component = nx.node_connected_component(graph, neighbor)
        label = f"native_arm_{index}"
        labels.update({int(node): label for node in component})
        sizes.append((len(component), label))
    labels[root] = max(sizes)[1] if sizes else "native_arm_0"
    return labels


def most_valuable_edge(graph: nx.Graph) -> tuple[int, int]:
    """Choose the unweighted betweenness edge with a deterministic tie break."""
    centrality = nx.edge_betweenness_centrality(graph, weight=None)
    ranked = sorted(
        centrality.items(),
        key=lambda item: (-item[1], min(item[0]), max(item[0])),
    )
    return ranked[0][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p6-dir", type=Path, required=True)
    parser.add_argument("--p9-dir", type=Path, required=True)
    args = parser.parse_args()
    args.p9_dir.mkdir(parents=True, exist_ok=True)

    phase_a = args.p6_dir / "phase_a"
    node_path = phase_a / "T0_STATE_NODE_ASSIGNMENTS.csv"
    candidate_path = phase_a / "C1_MULTIRESOLUTION_CANDIDATE_EDGES.csv"
    soft_path = phase_a / "P6_FROZEN_SOFT_MEMBERSHIP.csv"
    fine_edge_path = phase_a / "T0_FINE_TRAJECTORY_EDGES.csv"
    nodes = pd.read_csv(node_path, usecols=["root_id", "node_id", "unit_id", "branch_id", "pseudotime"])
    fine_edges = pd.read_csv(fine_edge_path, usecols=["root_id", "source", "target"])
    soft = pd.read_csv(soft_path).set_index("unit_id")
    universe = sorted(nodes.unit_id.astype(str).unique())
    node_lookup = nodes.drop_duplicates("unit_id").set_index("unit_id").node_id.astype(int).to_dict()
    node_ids = set(node_lookup.values())

    chunks = []
    for chunk in pd.read_csv(candidate_path, chunksize=200000):
        eligible = as_bool(chunk.trajectory_edge_eligible)
        keep = eligible & chunk.source.astype(int).isin(node_ids) & chunk.target.astype(int).isin(node_ids)
        if keep.any():
            chunks.append(chunk.loc[keep].copy())
    candidate = pd.concat(chunks, ignore_index=True)
    candidate["_pair_source"] = np.minimum(candidate.source.astype(int), candidate.target.astype(int))
    candidate["_pair_target"] = np.maximum(candidate.source.astype(int), candidate.target.astype(int))
    candidate = candidate.sort_values(
        ["D_joint", "_pair_source", "_pair_target", "source", "target"], kind="mergesort"
    )
    candidate = candidate.drop_duplicates(["_pair_source", "_pair_target"])

    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    for row in candidate.itertuples(index=False):
        graph.add_edge(int(row.source), int(row.target), D_joint=float(row.D_joint))
    induced_connected = nx.is_connected(graph)
    if not induced_connected:
        raise ValueError("Frozen eligible candidate subgraph is not connected on 883 units")
    tree = nx.minimum_spanning_tree(graph, weight="D_joint", algorithm="kruskal")
    root = choose_center(tree)
    distance = nx.single_source_dijkstra_path_length(tree, root, weight="D_joint")
    scale = max(distance.values())
    native_label = root_arms(tree, root)

    communities = next(
        partition
        for partition in nx.community.girvan_newman(tree, most_valuable_edge=most_valuable_edge)
        if len(partition) == 7
    )
    ordered_communities = sorted(
        [sorted(map(int, community)) for community in communities],
        key=lambda values: (-len(values), min(values)),
    )
    matched_label = {
        node: f"matched_program_{index}"
        for index, community in enumerate(ordered_communities)
        for node in community
    }

    edge_rows = []
    pair_to_row = {
        tuple(sorted((int(row.source), int(row.target)))): row
        for row in candidate.itertuples(index=False)
    }
    for source, target in sorted(tree.edges()):
        row = pair_to_row[tuple(sorted((int(source), int(target))))]
        edge_rows.append({
            "source": int(source), "target": int(target),
            "source_name": str(row.source_name), "target_name": str(row.target_name),
            "D_joint": float(row.D_joint), "candidate_legal": True,
        })
    edge_frame = pd.DataFrame(edge_rows)
    edge_frame.to_csv(args.p9_dir / "P9_GLOBAL_SINGLE_TREE.csv", index=False)

    id_to_unit = {value: key for key, value in node_lookup.items()}
    native = pd.DataFrame({
        "node_id": sorted(node_ids),
        "unit_id": [id_to_unit[node] for node in sorted(node_ids)],
        "global_root_node_id": root,
        "global_pseudotime": [distance[node] / max(scale, 1e-12) for node in sorted(node_ids)],
        "native_program": [native_label[node] for node in sorted(node_ids)],
    })
    native.to_csv(args.p9_dir / "P9_GLOBAL_NATIVE_ASSIGNMENTS.csv", index=False)
    matched = native.copy()
    matched["matched_program"] = [matched_label[node] for node in matched.node_id]
    matched.to_csv(args.p9_dir / "P9_COMPLEXITY_MATCHED_CONTROL.csv", index=False)

    # Collapse structurally shared multi-tree nodes to one frozen root without biology.
    chosen_rows = []
    for unit_id, part in nodes.groupby("unit_id", sort=True):
        ranked = part.copy()
        ranked["selection_membership"] = [
            float(soft.loc[str(unit_id), str(root_id)]) for root_id in ranked.root_id
        ]
        chosen = ranked.sort_values(
            ["selection_membership", "root_id"], ascending=[False, True], kind="mergesort"
        ).iloc[0]
        chosen_rows.append({
            "unit_id": str(unit_id), "root_id": str(chosen.root_id),
            "program_id": str(chosen.branch_id), "pseudotime": float(chosen.pseudotime),
            "selection_membership": float(chosen.selection_membership),
            "n_structural_roots": int(part.root_id.nunique()),
            "selection_rule": "max_frozen_soft_membership_then_root_id",
        })
    multi = pd.DataFrame(chosen_rows)
    multi.to_csv(args.p9_dir / "P9_FROZEN_MULTITREE_SINGLE_ROW_ASSIGNMENTS.csv", index=False)

    terminal_count = sum(1 for node in tree if tree.degree(node) == 1)
    multi_terminal_counts = {}
    for root_id, part in fine_edges.groupby("root_id", sort=True):
        local_tree = nx.from_pandas_edgelist(part, source="source", target="target")
        multi_terminal_counts[str(root_id)] = int(sum(degree == 1 for _, degree in local_tree.degree()))
    multi_terminal_total = int(sum(multi_terminal_counts.values()))
    audit = {
        "stage": "P9.0_CONTROL_CONSTRUCTION_AUDIT",
        "same_metacell_universe": set(native.unit_id) == set(multi.unit_id),
        "n_metacells": len(native),
        "coverage": len(native) / len(universe),
        "induced_candidate_connected": induced_connected,
        "global_connected": nx.is_connected(tree),
        "global_acyclic": nx.is_tree(tree),
        "global_candidate_legal": True,
        "global_root_node_id": root,
        "global_root_unit_id": id_to_unit[root],
        "root_rule": "weighted_tree_center_min_eccentricity_then_sum_then_node_id",
        "native_program_count": int(native.native_program.nunique()),
        "complexity_matched_program_count": int(matched.matched_program.nunique()),
        "target_program_count": 7,
        "global_terminal_count": terminal_count,
        "multi_tree_terminal_counts_by_root": multi_terminal_counts,
        "multi_tree_terminal_count_row_sum": multi_terminal_total,
        "global_to_multi_terminal_count_ratio": terminal_count / max(multi_terminal_total, 1),
        "terminal_count_exactly_matched": terminal_count == multi_terminal_total,
        "multi_program_count_after_single_row_assignment": int(multi.program_id.nunique()),
        "native_program_sizes": native.native_program.value_counts().sort_index().to_dict(),
        "matched_program_sizes": matched.matched_program.value_counts().sort_index().to_dict(),
        "multi_program_sizes": multi.program_id.value_counts().sort_index().to_dict(),
        "biology_used_in_construction": False,
        "P8_scores_used_in_construction": False,
        "P8_labels_used_in_control_construction": False,
        "multi_tree_size_distribution_used_for_matching": False,
        "complexity_matching_rule": "topology_only_Girvan_Newman_to_seven_connected_programs; terminal count audited but not optimized",
        "coverage_documented": True,
        "complete": True,
    }
    audit["decision"] = "PASS" if all([
        audit["same_metacell_universe"], audit["n_metacells"] == 883,
        audit["global_connected"], audit["global_acyclic"],
        audit["global_candidate_legal"], audit["coverage"] == 1.0,
        audit["complexity_matched_program_count"] == 7,
        not audit["biology_used_in_construction"], not audit["P8_scores_used_in_construction"],
    ]) else "FAIL"
    (args.p9_dir / "P90_CONTROL_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    files = [
        args.p9_dir / "P9_GLOBAL_SINGLE_TREE.csv",
        args.p9_dir / "P9_GLOBAL_NATIVE_ASSIGNMENTS.csv",
        args.p9_dir / "P9_COMPLEXITY_MATCHED_CONTROL.csv",
        args.p9_dir / "P9_FROZEN_MULTITREE_SINGLE_ROW_ASSIGNMENTS.csv",
    ]
    pd.DataFrame([{"file": path.name, "sha256": sha256(path)} for path in files]).to_csv(
        args.p9_dir / "P90_CONTROL_SHA256.csv", index=False
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
