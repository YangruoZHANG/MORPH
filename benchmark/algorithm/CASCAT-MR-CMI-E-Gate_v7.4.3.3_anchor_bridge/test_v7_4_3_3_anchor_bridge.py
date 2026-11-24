#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit and optional integration tests for v7.4.3.3 anchor bridge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import cascat_v7_4_3_3_truth_first as truth


def cfg(**updates):
    values = dict(
        core_edge_confidence_quantile=0.60,
        core_edge_confidence_floor=0.45,
        truth_min_edge_stability=0.60,
        truth_rescue_min_stability=0.80,
        truth_rescue_min_confidence=0.55,
        truth_min_s_global_floor=0.03,
        truth_s_global_quantile_floor=0.05,
        truth_min_edges_for_adaptive_threshold=5,
        truth_max_cost_median_factor=8.0,
        truth_max_cmi_conflict=0.25,
        truth_min_trajectory_nodes=4,
        truth_min_trajectory_edges=3,
        truth_enable_component_bridge_rescue=True,
        truth_bridge_min_root_quality=0.65,
        truth_bridge_max_edges=2,
        truth_bridge_min_confidence=0.75,
        truth_bridge_max_confidence_gap=0.05,
        truth_bridge_min_stability=0.40,
        truth_bridge_min_sglobal_ratio=0.85,
        truth_bridge_min_component_nodes=8,
        truth_bridge_min_component_edges=7,
        truth_bridge_require_anchor_support=True,
    )
    values.update(updates)
    return SimpleNamespace(**values)


def make_tree(
    bridge_edges: int = 1,
    bridge_sglobal: float = 0.86,
    bridge_confidence: float = 0.80,
    bridge_stability: float = 0.40,
    bridge_cost: float = 1.0,
    target_nodes: int = 9,
):
    rows = []
    root = 0
    current = root
    for step in range(bridge_edges):
        nxt = current + 1
        rows.append((current, nxt, bridge_sglobal, bridge_confidence, bridge_stability, bridge_cost, "root_path_rescue"))
        current = nxt
    start = current
    for node in range(start, start + target_nodes - 1):
        rows.append((node, node + 1, 1.0, 0.95, 1.0, 1.0, "confidence_core"))
    return pd.DataFrame(
        [
            {
                "root_id": "MC0000",
                "root_pos": 0,
                "root_index": root,
                "source": u,
                "target": v,
                "source_name": f"MC{u:04d}",
                "target_name": f"MC{v:04d}",
                "S_global": sg,
                "edge_confidence": conf,
                "core_threshold": 0.82,
                "edge_stability_recurrence": stab,
                "cost_final": cost,
                "edge_cost": cost,
                "cmi_env_conflict": 0.0,
                "core_selected_reason": reason,
            }
            for u, v, sg, conf, stab, cost, reason in rows
        ]
    )


def anchors(root: int, anchor: int):
    return pd.DataFrame([
        {"root_pos": 0, "anchor_node": root, "is_mapped_root": True},
        {"root_pos": 0, "anchor_node": anchor, "is_mapped_root": False},
    ])


def validity(status="strong", quality=0.85):
    return pd.DataFrame([
        {"root_pos": 0, "root_index": 0, "root_name": "MC0000", "root_validity_status": status, "root_quality_score": quality}
    ])


def run_case(frame, anchor_node, config=None, rv=None):
    return truth.apply_edge_truth_gate(
        frame,
        [0],
        config or cfg(),
        root_anchor_sets=anchors(0, anchor_node),
        root_validity=rv if rv is not None else validity(),
    )


def test_one_edge_bridge_rescue():
    frame = make_tree(bridge_edges=1)
    passed, audit, conn = run_case(frame, anchor_node=2)
    assert int(conn.iloc[0].bridge_edge_count) == 1
    assert conn.iloc[0].connectivity_status == "COMPONENT_SUPPORTED_BRIDGE_RESCUED"
    assert len(passed) >= 8
    assert int(audit.edge_truth_gate_bridge_rescued.sum()) == 1


def test_two_edge_bridge_rescue():
    frame = make_tree(bridge_edges=2)
    passed, _, conn = run_case(frame, anchor_node=3)
    assert int(conn.iloc[0].bridge_edge_count) == 2
    assert len(passed) >= 9


def test_three_edge_bridge_rejected():
    frame = make_tree(bridge_edges=3)
    passed, _, conn = run_case(frame, anchor_node=4)
    assert int(conn.iloc[0].bridge_edge_count) == 0
    assert passed.empty


def test_extreme_cost_rejected():
    frame = make_tree(bridge_edges=1, bridge_cost=100.0)
    passed, _, conn = run_case(frame, anchor_node=2)
    assert int(conn.iloc[0].bridge_edge_count) == 0
    assert passed.empty


def test_low_similarity_rejected():
    frame = make_tree(bridge_edges=1, bridge_sglobal=0.20)
    passed, _, conn = run_case(frame, anchor_node=2)
    assert int(conn.iloc[0].bridge_edge_count) == 0
    assert passed.empty


def test_no_nonroot_anchor_rejected():
    frame = make_tree(bridge_edges=1)
    passed, _, conn = truth.apply_edge_truth_gate(
        frame, [0], cfg(), root_anchor_sets=anchors(0, 999), root_validity=validity()
    )
    assert int(conn.iloc[0].bridge_edge_count) == 0
    assert passed.empty


def test_weak_root_rejected():
    frame = make_tree(bridge_edges=1)
    passed, _, conn = run_case(frame, anchor_node=2, rv=validity(status="weak", quality=0.90))
    assert int(conn.iloc[0].bridge_edge_count) == 0
    assert passed.empty


def test_bridge_disabled_reproduces_strict_behavior():
    frame = make_tree(bridge_edges=1)
    passed, _, conn = run_case(frame, anchor_node=2, config=cfg(truth_enable_component_bridge_rescue=False))
    assert int(conn.iloc[0].bridge_edge_count) == 0
    assert passed.empty


def integration(project_root: Path) -> None:
    # Optional exact regression against the user's audited v7.4.3.2 outputs.
    from audit_v7_4_3_3_anchor_bridge_on_existing_outputs import (
        DATASET_LOCATIONS, PATCH_DEFAULTS, compare_raw_gate, config_from_run_summary,
        read_csv, roots_from_output,
    )
    locations = {
        "GSE": project_root / "outputs/tme_v7_4_3_2_full_run/GSE",
        "GSE193460": project_root / "outputs/tme_v7_4_3_2_full_run/GSE193460",
        "GSE245582": project_root / "outputs/tme_v7_4_3_2_full_run/GSE245582",
        "xenium_lung": project_root / "outputs/tme_v7_4_3_2_full_run/xenium_lung",
        "xenium_melanoma": project_root / "outputs/tme_v7_4_3_2_full_run/xenium_melanoma",
        "xenium_pancreatic": project_root / "outputs/tme_v7_4_3_2_pancreatic_rerun/xenium_pancreatic",
        "xenium_breast": project_root / "outputs/tme_v7_4_3_2/xenium_breast",
    }
    rescued = []
    for dataset, ds_dir in locations.items():
        if not (ds_dir / "pre_truth_gate_backbone_tree_edges.csv").exists():
            continue
        config = config_from_run_summary(ds_dir)
        roots = roots_from_output(ds_dir)
        passed, audit, conn = truth.apply_edge_truth_gate(
            read_csv(ds_dir / "pre_truth_gate_backbone_tree_edges.csv"),
            roots,
            config,
            read_csv(ds_dir / "root_anchor_sets.csv"),
            read_csv(ds_dir / "root_validity_audit.csv"),
        )
        mismatch, schema = compare_raw_gate(read_csv(ds_dir / "edge_truth_gate_audit.csv"), audit)
        assert mismatch == 0 and schema == 0, f"strict gate regression in {dataset}: {mismatch}/{schema}"
        for r in conn.loc[pd.to_numeric(conn.bridge_edge_count, errors="coerce").fillna(0) > 0].itertuples(index=False):
            rescued.append((dataset, str(r.root_name), int(r.bridge_edge_count), int(r.final_root_connected_edges)))
    expected = {
        ("GSE", "MC3052", 2, 158),
        ("xenium_pancreatic", "MC2237", 1, 162),
    }
    assert set(rescued) == expected, f"unexpected rescued roots: {rescued}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None, help="Optional NicheForest root for integration regression")
    args = parser.parse_args()
    tests = [
        test_one_edge_bridge_rescue,
        test_two_edge_bridge_rescue,
        test_three_edge_bridge_rejected,
        test_extreme_cost_rejected,
        test_low_similarity_rejected,
        test_no_nonroot_anchor_rejected,
        test_weak_root_rejected,
        test_bridge_disabled_reproduces_strict_behavior,
    ]
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    if args.project_root:
        integration(Path(args.project_root).resolve())
        print("[PASS] integration_regression")
    print(f"[PASS] {len(tests) + (1 if args.project_root else 0)} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
