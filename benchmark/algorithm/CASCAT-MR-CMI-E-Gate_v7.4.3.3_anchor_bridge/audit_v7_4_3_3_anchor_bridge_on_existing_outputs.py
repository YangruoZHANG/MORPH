#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only counterfactual audit of the v7.4.3.3 anchor-bridge gate.

This script never modifies v7.4.3.2 outputs. It re-applies the strict gate and
then the bounded anchor-supported bridge rule to existing pre-Truth-First
backbones. It verifies that every original raw hard-veto decision is reproduced
before the bridge is considered.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import cascat_v7_4_3_3_truth_first as truth

DATASET_LOCATIONS = {
    "GSE": ("full", "GSE"),
    "GSE193460": ("full", "GSE193460"),
    "GSE245582": ("full", "GSE245582"),
    "xenium_lung": ("full", "xenium_lung"),
    "xenium_melanoma": ("full", "xenium_melanoma"),
    "xenium_pancreatic": ("pancreatic", "xenium_pancreatic"),
    "xenium_breast": ("breast", "xenium_breast"),
}

PATCH_DEFAULTS: dict[str, Any] = {
    "truth_enable_component_bridge_rescue": True,
    "truth_bridge_min_root_quality": 0.65,
    "truth_bridge_max_edges": 2,
    "truth_bridge_min_confidence": 0.75,
    "truth_bridge_max_confidence_gap": 0.05,
    "truth_bridge_min_stability": 0.40,
    "truth_bridge_min_sglobal_ratio": 0.85,
    "truth_bridge_min_component_nodes": 8,
    "truth_bridge_min_component_edges": 7,
    "truth_bridge_require_anchor_support": True,
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def resolve_dataset_dir(args: argparse.Namespace, dataset: str) -> Path:
    kind, leaf = DATASET_LOCATIONS[dataset]
    if kind == "full":
        return args.full_run_root / leaf
    if kind == "pancreatic":
        return args.pancreatic_dir
    return args.breast_dir


def config_from_run_summary(dataset_dir: Path) -> SimpleNamespace:
    cfg: dict[str, Any] = {}
    summary = dataset_dir / "run_summary.json"
    if summary.exists():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        cfg.update(payload.get("config", {}))
    cfg.update(PATCH_DEFAULTS)
    return SimpleNamespace(**cfg)


def roots_from_output(dataset_dir: Path) -> list[int]:
    resolved = read_csv(dataset_dir / "resolved_roots.csv")
    for col in ("resolved_index", "subset_index", "root_index"):
        if col in resolved.columns:
            return pd.to_numeric(resolved[col], errors="coerce").dropna().astype(int).tolist()
    raise KeyError(f"No resolved root index column in {dataset_dir / 'resolved_roots.csv'}")


def compare_raw_gate(old: pd.DataFrame, new: pd.DataFrame) -> tuple[int, int]:
    keys = ["root_pos", "source", "target"]
    if old.empty or not set(keys + ["edge_truth_gate_pass_raw"]).issubset(old.columns):
        return -1, 0
    merged = old[keys + ["edge_truth_gate_pass_raw"]].merge(
        new[keys + ["edge_truth_gate_pass_raw"]],
        on=keys,
        how="outer",
        suffixes=("_v7_4_3_2", "_v7_4_3_3"),
        indicator=True,
    )
    schema_mismatch = int((merged["_merge"] != "both").sum())
    both = merged.loc[merged["_merge"] == "both"].copy()
    decision_mismatch = int(
        (
            both["edge_truth_gate_pass_raw_v7_4_3_2"].astype(bool)
            != both["edge_truth_gate_pass_raw_v7_4_3_3"].astype(bool)
        ).sum()
    )
    return decision_mismatch, schema_mismatch


def structural_gate(passed: pd.DataFrame, roots: list[int], root_validity: pd.DataFrame, cfg: SimpleNamespace) -> pd.DataFrame:
    # This intentionally assesses Level-1 structural readiness only. Program and
    # biological evidence must be recomputed from a real v7.4.3.3 rerun.
    program_placeholder = pd.DataFrame({"root_pos": list(range(len(roots)))})
    return truth.build_trajectory_truth_gate(
        final_edges=passed,
        roots=roots,
        root_validity=root_validity,
        program_summary=program_placeholder,
        cfg=cfg,
    )


def audit_one(dataset: str, dataset_dir: Path, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required = [
        "pre_truth_gate_backbone_tree_edges.csv",
        "resolved_roots.csv",
        "root_validity_audit.csv",
        "root_anchor_sets.csv",
    ]
    missing = [name for name in required if not (dataset_dir / name).exists()]
    if missing:
        return [], {
            "dataset": dataset,
            "status": "MISSING_REQUIRED_OUTPUT",
            "dataset_dir": str(dataset_dir),
            "missing": "|".join(missing),
        }

    pre = read_csv(dataset_dir / "pre_truth_gate_backbone_tree_edges.csv")
    roots = roots_from_output(dataset_dir)
    anchors = read_csv(dataset_dir / "root_anchor_sets.csv")
    root_validity = read_csv(dataset_dir / "root_validity_audit.csv")
    old_audit = read_csv(dataset_dir / "edge_truth_gate_audit.csv")
    cfg = config_from_run_summary(dataset_dir)

    passed, audit, connectivity = truth.apply_edge_truth_gate(
        pre_truth_backbone_edges=pre,
        roots=roots,
        cfg=cfg,
        root_anchor_sets=anchors,
        root_validity=root_validity,
    )
    gate = structural_gate(passed, roots, root_validity, cfg)
    raw_mismatch, schema_mismatch = compare_raw_gate(old_audit, audit)

    ds_out = output_dir / dataset
    ds_out.mkdir(parents=True, exist_ok=True)
    passed.to_csv(ds_out / "edge_truth_gate_passed_backbone_edges_v7_4_3_3_counterfactual.csv", index=False)
    audit.to_csv(ds_out / "edge_truth_gate_audit_v7_4_3_3_counterfactual.csv", index=False)
    connectivity.to_csv(ds_out / "root_connectivity_audit_v7_4_3_3_counterfactual.csv", index=False)
    gate.to_csv(ds_out / "trajectory_truth_gate_v7_4_3_3_structural_counterfactual.csv", index=False)

    records: list[dict[str, Any]] = []
    for row in connectivity.to_dict(orient="records"):
        root_pos = int(row["root_pos"])
        g = gate.loc[pd.to_numeric(gate["root_pos"], errors="coerce").fillna(-1).astype(int) == root_pos]
        g0 = g.iloc[0].to_dict() if not g.empty else {}
        records.append({
            "dataset": dataset,
            **row,
            "counterfactual_truth_level": int(g0.get("truth_level", 0)),
            "counterfactual_trajectory_status": str(g0.get("trajectory_truth_status", "")),
            "strict_raw_gate_decision_mismatch": raw_mismatch,
            "edge_schema_mismatch": schema_mismatch,
        })

    summary = {
        "dataset": dataset,
        "status": "PASS" if raw_mismatch == 0 and schema_mismatch == 0 else "STRICT_GATE_REGRESSION",
        "dataset_dir": str(dataset_dir),
        "candidate_roots": len(roots),
        "strict_raw_gate_decision_mismatch": raw_mismatch,
        "edge_schema_mismatch": schema_mismatch,
        "rescued_roots": int((connectivity.get("bridge_edge_count", pd.Series(dtype=int)) > 0).sum()),
        "rescued_bridge_edges": int(pd.to_numeric(connectivity.get("bridge_edge_count", pd.Series(dtype=int)), errors="coerce").fillna(0).sum()),
        "counterfactual_level1_roots": int((pd.to_numeric(gate.get("truth_level", pd.Series(dtype=int)), errors="coerce").fillna(0) >= 1).sum()),
        "counterfactual_final_edges": int(len(passed.loc[passed["root_pos"].isin(gate.loc[gate["truth_level"] >= 1, "root_pos"])]) if not passed.empty else 0),
    }
    return records, summary


def write_report(output_dir: Path, root_frame: pd.DataFrame, summary_frame: pd.DataFrame) -> None:
    lines = [
        "# v7.4.3.3 anchor-bridge counterfactual audit",
        "",
        "This is a read-only audit of existing v7.4.3.2 pre-Truth-First backbones. It does not rebuild roots, memberships, trees, pseudotime or biological annotations.",
        "",
        "## Invariants",
        "",
        "- Original strict raw hard-veto decisions must be reproduced exactly before bridge evaluation.",
        "- A bridge is limited to at most two failed attachment edges.",
        "- The target must be a large strict-passing component containing a non-root anchor.",
        "- Extreme cost, low confidence and CMI-E conflict are never rescued.",
        "",
        "## Dataset summary",
        "",
        "| Dataset | Audit | Candidate roots | Rescued roots | Bridge edges | Counterfactual Level-1 roots | Final edges |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_frame.itertuples(index=False):
        lines.append(
            f"| {r.dataset} | {r.status} | {r.candidate_roots} | {r.rescued_roots} | {r.rescued_bridge_edges} | {r.counterfactual_level1_roots} | {r.counterfactual_final_edges} |"
        )
    lines += ["", "## Rescued root attachments", ""]
    rescued = root_frame.loc[pd.to_numeric(root_frame.get("bridge_edge_count", pd.Series(dtype=int)), errors="coerce").fillna(0) > 0]
    if rescued.empty:
        lines.append("No root attachment met the bounded rescue requirements.")
    else:
        for r in rescued.itertuples(index=False):
            lines.append(
                f"- `{r.dataset}/{r.root_name}`: bridge `{r.bridge_edges}`; path `{r.bridge_path_nodes}`; strict target component {r.target_component_nodes} nodes / {r.target_component_edges} edges; final root-connected edges {r.final_root_connected_edges}."
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "A rescued root is only a Level-1 structural candidate. This counterfactual audit does not establish a biological origin, differentiation trajectory or independent biological time. A complete v7.4.3.3 rerun is required before annotation or figure generation.",
    ]
    (output_dir / "COUNTERFACTUAL_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--full-run-root", default="outputs/tme_v7_4_3_2_full_run")
    parser.add_argument("--pancreatic-dir", default="outputs/tme_v7_4_3_2_pancreatic_rerun/xenium_pancreatic")
    parser.add_argument("--breast-dir", default="outputs/tme_v7_4_3_2/xenium_breast")
    parser.add_argument("--output-dir", default="outputs/tme_v7_4_3_3_anchor_bridge_counterfactual_audit")
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_LOCATIONS))
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    args.full_run_root = (project / args.full_run_root).resolve()
    args.pancreatic_dir = (project / args.pancreatic_dir).resolve()
    args.breast_dir = (project / args.breast_dir).resolve()
    output_dir = (project / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    root_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for dataset in args.datasets:
        if dataset not in DATASET_LOCATIONS:
            raise ValueError(f"Unknown dataset: {dataset}")
        records, summary = audit_one(dataset, resolve_dataset_dir(args, dataset), output_dir)
        root_records.extend(records)
        summaries.append(summary)
        print(
            f"[{summary['status']}] {dataset}: rescued_roots={summary.get('rescued_roots', 0)} "
            f"level1_roots={summary.get('counterfactual_level1_roots', 0)}",
            flush=True,
        )

    root_frame = pd.DataFrame(root_records)
    summary_frame = pd.DataFrame(summaries)
    root_frame.to_csv(output_dir / "MASTER_ROOT_CONNECTIVITY_COUNTERFACTUAL.csv", index=False, encoding="utf-8-sig")
    summary_frame.to_csv(output_dir / "MASTER_DATASET_COUNTERFACTUAL.csv", index=False, encoding="utf-8-sig")
    write_report(output_dir, root_frame, summary_frame)
    ok = bool((summary_frame["status"] == "PASS").all()) if not summary_frame.empty else False
    print(f"[REPORT] {output_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
