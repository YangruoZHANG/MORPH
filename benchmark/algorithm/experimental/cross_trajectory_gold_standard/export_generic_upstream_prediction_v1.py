#!/usr/bin/env python3
"""Export generic NicheForest upstream stages to prediction contract v1."""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def branch_rows(edges: pd.DataFrame, root_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for anchor, part in edges.groupby(edges["root_id"].astype(str)):
        logical = root_map[str(anchor)]
        graph = nx.Graph()
        graph.add_edges_from(part[["source_name", "target_name"]].astype(str).itertuples(index=False, name=None))
        if anchor not in graph:
            rows.append({"unit_id": str(anchor), "root_id": logical, "branch_id": "ROOT"})
            continue
        rows.append({"unit_id": str(anchor), "root_id": logical, "branch_id": "ROOT"})
        for node in graph.nodes:
            if str(node) == str(anchor):
                continue
            path = nx.shortest_path(graph, str(anchor), str(node))
            first = path[1] if len(path) > 1 else str(anchor)
            rows.append({"unit_id": str(node), "root_id": logical, "branch_id": f"BR_{logical}_{first}"})
    return pd.DataFrame(rows).drop_duplicates(["unit_id", "root_id"])


def edge_rows(edges: pd.DataFrame, root_map: dict[str, str]) -> pd.DataFrame:
    out = edges[["root_id", "source_name", "target_name"]].copy()
    out["root_id"] = out["root_id"].astype(str).map(root_map)
    out = out.rename(columns={"source_name": "source", "target_name": "target"})
    return out[["root_id", "source", "target"]].dropna().drop_duplicates()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--fixed-root-input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=["M2", "M3", "M4", "M5"], required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fixed = pd.read_csv(args.fixed_root_input)
    root_map = dict(zip(fixed["root_anchor_id"].astype(str), fixed["truth_root_id"].astype(str)))
    pos_map = dict(enumerate(fixed["truth_root_id"].astype(str)))
    pd.DataFrame({"root_id": list(root_map.values())}).to_csv(args.output_dir / "PRED_ROOTS.csv", index=False)

    membership_name = "membership_A_sharp.csv" if args.stage in {"M2", "M3", "M4"} else "membership_A_sharp_final.csv"
    membership = pd.read_csv(args.upstream_dir / membership_name)
    membership["root_id"] = membership["root_name"].astype(str).map(root_map)
    membership.rename(columns={"node_name": "unit_id", "A_sharp": "membership_probability"})[
        ["unit_id", "root_id", "membership_probability"]
    ].dropna().to_csv(args.output_dir / "PRED_CELL_OR_METACELL_ASSIGNMENT.csv", index=False)

    edge_name = "initial_root_specific_tree_edges.csv" if args.stage == "M2" else "progression_tree_edges.csv"
    t0 = pd.read_csv(args.upstream_dir / edge_name)
    t1 = t0 if args.stage in {"M2", "M3", "M4"} else pd.read_csv(args.upstream_dir / "final_selected_tree_edges.csv")
    edge_rows(t0, root_map).to_csv(args.output_dir / "PRED_T0_EDGES.csv", index=False)
    edge_rows(t1, root_map).to_csv(args.output_dir / "PRED_T1_EDGES.csv", index=False)
    branch_rows(t1, root_map).to_csv(args.output_dir / "PRED_BRANCH.csv", index=False)

    if args.stage == "M5":
        pt = pd.read_csv(args.upstream_dir / "final_constrained_pseudotime_by_root.csv")
        value_column = "constrained_pseudotime"
    else:
        label = "base" if args.stage == "M2" else "progression"
        pt = pd.read_csv(args.upstream_dir / "pseudotime_table.csv")
        pt = pt[pt["tree_label"].astype(str).eq(label)].copy()
        node_names = membership.drop_duplicates("node_id").set_index("node_id")["node_name"].astype(str)
        pt["node_name"] = pt["node_id"].astype(int).map(node_names)
        value_column = "selected_pseudotime"
    pt["root_id"] = pt["root_pos"].astype(int).map(pos_map)
    pt.rename(columns={"node_name": "unit_id", value_column: "pseudotime"})[
        ["unit_id", "root_id", "pseudotime"]
    ].dropna().to_csv(args.output_dir / "PRED_PSEUDOTIME.csv", index=False)

    event_rows = []
    relation_path = args.upstream_dir / "tree_pair_relation_matrix_detailed.csv"
    if args.stage in {"M4", "M5"} and relation_path.exists():
        relations = pd.read_csv(relation_path)
        for index, row in relations.iterrows():
            called = as_bool(row.get("relation_active", False))
            relation = str(row.get("dominant_tree_pair_relation", "weak_or_independent"))
            if "conflict" in relation or "inhibition" in relation:
                event_type = "inhibition"
            elif "convergence" in relation or "promotion" in relation:
                event_type = "promotion"
            else:
                event_type = "collision_convergence"
            score = max(
                float(row.get("convergence_relation_score", 0.0) or 0.0),
                float(row.get("conflict_relation_score", 0.0) or 0.0),
            )
            event_rows.append({
                "event_id": f"UPSTREAM_REL_{int(row.tree_k)}_{int(row.tree_l)}",
                "event_type": event_type,
                "source_root": pos_map[int(row.tree_k)],
                "receiver_root": pos_map[int(row.tree_l)],
                "affected_branch": "none",
                "score": score,
                "is_called": called,
            })
    pd.DataFrame(event_rows, columns=[
        "event_id", "event_type", "source_root", "receiver_root",
        "affected_branch", "score", "is_called",
    ]).to_csv(args.output_dir / "PRED_EVENTS.csv", index=False)
    (args.output_dir / "METHOD_STAGE.txt").write_text(
        f"{args.stage}: generic production upstream export\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
