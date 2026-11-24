#!/usr/bin/env python3
"""PHASE A v2.1: exact-anchor-aware coarse topology audit."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import networkx as nx


HERE = Path(__file__).resolve().parent


def load_version(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = load_version("run_phase_a_case_v2.py", "phase_a_v2_for_v21")


def exact_anchor_mapped_state_backbone(root_id: str, result, truth):
    nodes = result["nodes"]
    nodes = nodes[nodes.root_id.astype(str).eq(root_id)]
    mapping = {}
    for state_id, group in nodes.groupby("state_id"):
        weights = {}
        for unit in group.unit_id.astype(str):
            for truth_state in truth.state_support.get((unit, root_id), set()):
                category = V2.R.coarse_category(truth_state)
                if category:
                    weights[category] = weights.get(category, 0.0) + V2.Q.state_probability(
                        truth, unit, root_id, truth_state
                    )
        mapping[str(state_id)] = min(weights, key=lambda value: (-weights[value], value)) if weights else None
    root_rows = nodes[nodes.is_root.astype(bool)]
    root_state = str(root_rows.iloc[0].state_id) if len(root_rows) else None
    if root_state is not None:
        mapping[root_state] = "TRUNK"
    graph = nx.DiGraph()
    backbone = result["backbone"]
    backbone = backbone[backbone.root_id.astype(str).eq(root_id)]
    for row in backbone.itertuples(index=False):
        source = mapping.get(str(row.source_state))
        target = mapping.get(str(row.target_state))
        if source and target and source != target:
            graph.add_edge(source, target)
    return graph, mapping.get(root_state)


def main() -> None:
    V2.mapped_state_backbone = exact_anchor_mapped_state_backbone
    V2.main()
    output = None
    for index, value in enumerate(sys.argv):
        if value == "--output" and index + 1 < len(sys.argv):
            output = Path(sys.argv[index + 1]).resolve()
            break
    if output is not None:
        status_path = output / "RUN_STATUS.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.update({
            "version": "PHASE_A_v2_1_C3_dualscale_v6",
            "parent": "PHASE_A_v2_C3_dualscale_v6",
            "evaluation_correction": "exact_anchor_state_maps_to_TRUNK",
            "construction_changed_from_v2": False,
        })
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
