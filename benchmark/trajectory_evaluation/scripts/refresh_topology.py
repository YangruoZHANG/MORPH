"""Refresh topology columns/results without recomputing feature-importance forests."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from morph_eval.evaluator import derived_fate_star, read_graph, write_result_tables
from morph_eval.metrics import quotient_to_truth, topology_one_minus_him


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    registry = pd.read_csv(source / "data" / "partitions.csv")
    part_lookup = {(x.case, int(x.root_index)): x.part for x in registry.itertuples(index=False)}
    path = ROOT / "results" / "by_root.csv"
    scores = pd.read_csv(path)

    for i, row in scores.iterrows():
        case, root_index, method = row["case"], int(row["root_index"]), row["method_id"]
        part = part_lookup[(case, root_index)]
        case_dir = source / "data" / case
        truth = pd.read_csv(case_dir / "truth.csv", index_col=0, dtype={"origin": str})
        truth = truth.loc[truth["origin"].eq(str(root_index))]
        edge_table = pd.read_csv(case_dir / "truth_edges.csv", dtype={"source": str, "target": str})
        edge_table = edge_table.loc[edge_table["source"].str.startswith(f"O{root_index}_")]
        truth_graph = nx.Graph()
        for edge in edge_table.itertuples(index=False):
            truth_graph.add_edge(str(edge.source), str(edge.target), length=float(edge.length))

        pred_dir = source / "results" / "predictions" / part / method
        cells = pd.read_csv(pred_dir / "cells.csv", dtype={"cell_id": str, "node": str, "native_branch": str}).set_index("cell_id").reindex(truth.index)
        pseudotime = pd.to_numeric(cells["pseudotime"], errors="coerce").to_numpy(float)
        graph = read_graph(pred_dir / "edges.csv")
        nodes = cells["node"].to_numpy(object) if "node" in cells else np.full(len(cells), None, object)
        mode = "native_graph"
        if not graph.number_of_edges():
            graph, nodes = derived_fate_star(cells, pseudotime)
            mode = "derived_fate_star" if graph.number_of_edges() else "unavailable"
        quotient = quotient_to_truth(truth, graph, nodes) if graph.number_of_edges() else nx.Graph()
        scores.loc[i, "topo_1_him"] = topology_one_minus_him(truth_graph, quotient)
        scores.loc[i, "topology_available"] = bool(np.isfinite(scores.loc[i, "topo_1_him"]))
        scores.loc[i, "topology_mode"] = mode
        print(f"[{i + 1:03d}/{len(scores)}] {case} R{root_index} {row['method']}: {mode}", flush=True)

    scores.to_csv(path, index=False, encoding="utf-8-sig")
    write_result_tables(scores, ROOT / "results")


if __name__ == "__main__":
    main()

