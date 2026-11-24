#!/usr/bin/env python3
"""Run the frozen P4 root-conditioned benchmark and primary T0 ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import anndata as ad
import numpy as np
import pandas as pd

from run_paper_baseline_adapter_smoke import (
    DEV,
    GOLD,
    prepare_identity_metacells,
    roots_from_truth,
)
from run_paper_root_conditioned_smoke import ADAPTERS, _score_root, _timecourse


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS = Path(
    r"D:\桌面\FYP\BASE_CASCAT\algorithms\unified_eval_kit\datasets_multiroot"
)
CURRENT_CASE = "hierarchical_branch_program_state_fine"
BASELINE_CASE = "current_root_conditioned_msf"
PREEXECUTION_GATE = {
    "version": "P4_ROOT_CONDITIONED_GATE_V1",
    "frozen_before_execution": True,
    "primary_comparison": "current branch-scale T0 versus root-conditioned MSF",
    "primary_metrics": [
        "branch_ARI_rootwise",
        "topology_F1",
        "pseudotime_spearman",
    ],
    "metric_mapping": {
        "topology_F1": "Phase A scorer fine_edge_F1",
        "branch_ARI_rootwise": "Phase A scorer branch_ARI_rootwise",
        "pseudotime_spearman": "Phase A rootwise pseudotime_spearman",
    },
    "go_to_P4_5_rule": {
        "branch": "mean and median paired delta > 0 and positive on at least 5/7 datasets",
        "topology_no_harm": "mean paired topology_F1 delta >= -0.05",
        "pseudotime_no_harm": "mean paired pseudotime delta >= -0.05",
        "integrity": "current T0 candidate legality, forest, state-tree, and root-support connectivity all 100%",
    },
    "external_methods_role": "contextual; cannot determine the primary core-method gate",
    "root_f1": "N/A because all roots are supplied",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_core(adata: ad.AnnData, dataset: Path, output: Path) -> tuple[list[dict], dict]:
    roots = roots_from_truth(adata)
    meta, mapping, truth_roots, truth_meta = prepare_identity_metacells(
        adata, roots, output
    )
    run_dir = output / "nicheforest_run"
    upstream = run_dir / "generic_upstream_v1_exact_root"
    phase_a = run_dir / "phase_a"
    commands: list[list[str]] = []
    if not (upstream / "GENERIC_UPSTREAM_STATUS.json").exists():
        commands.append(
            [
                sys.executable,
                str(GOLD / "run_generic_upstream_v1.py"),
                "--metacell-h5ad",
                str(meta),
                "--true-roots-csv",
                str(truth_roots),
                "--cell-to-metacell-csv",
                str(mapping),
                "--dataset",
                dataset.stem,
                "--output-dir",
                str(upstream),
            ]
        )
    if not (phase_a / "RUN_STATUS.json").exists():
        commands.append(
            [
                sys.executable,
                str(DEV / "run_phase_a_case_v7_state_constrained_fine.py"),
                "--run-dir",
                str(run_dir),
                "--dataset",
                dataset.stem,
                "--run-label",
                f"{dataset.stem}_P4_locked",
                "--output",
                str(phase_a),
                "--truth-run-dir",
                str(run_dir),
                "--metacell-truth-dir",
                str(truth_meta),
                "--fixed-root-input",
                str(upstream / "FIXED_ROOT_INPUT.csv"),
                "--production-only",
            ]
        )
    logs: list[str] = []
    for command in commands:
        process = subprocess.run(
            command,
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=900,
        )
        logs.extend([subprocess.list2cmdline(command), process.stdout, process.stderr])
        if process.returncode:
            raise RuntimeError(process.stderr[-6000:])
    (output / "P4_CORE_COMMANDS.log").write_text("\n".join(logs), encoding="utf-8")

    metrics = pd.read_csv(phase_a / "HVG_V8_DIAGNOSTIC_METRICS.csv")
    status = json.loads((phase_a / "RUN_STATUS.json").read_text(encoding="utf-8"))
    core_rows = []
    for case, method in (
        (BASELINE_CASE, "root-conditioned MSF"),
        (CURRENT_CASE, "current branch-scale T0"),
    ):
        selected = metrics[
            metrics["case"].astype(str).eq(case)
            & metrics["root_id"].astype(str).eq("ALL")
        ]
        if len(selected) != 1:
            raise ValueError(f"Missing unique ALL metrics for {dataset.stem}/{case}")
        row = selected.iloc[0]
        nodes_path = (
            phase_a / "T0_STATE_NODE_ASSIGNMENTS.csv"
            if case == CURRENT_CASE
            else phase_a / BASELINE_CASE / "T0_NODE_ASSIGNMENTS.csv"
        )
        nodes = pd.read_csv(nodes_path)
        core_rows.append(
            {
                "dataset": dataset.stem,
                "comparison_tier": "PRIMARY",
                "method": method,
                "trajectory_object": "T0",
                "root_contract": "all fixed roots supplied",
                "root_F1": "N/A_ROOTS_SUPPLIED",
                "coverage": nodes["unit_id"].astype(str).nunique() / adata.n_obs,
                "branch_ARI_rootwise": float(row["branch_ARI_rootwise"]),
                "topology_F1": float(row["fine_edge_F1"]),
                "pseudotime_spearman": float(row["pseudotime_spearman"]),
                "fine_shortcut_rate": float(row["fine_shortcut_rate"]),
                "n_components_mean": float(row["n_components"]),
                "is_forest": bool(row["is_forest"]),
                "candidate_legal": bool(status["all_fine_edges_candidate_legal"]),
                "all_state_backbones_are_trees": bool(
                    status["all_state_backbones_are_trees"]
                ),
                "all_root_supports_connected": bool(
                    status["all_root_supports_connected"]
                )
                if case == CURRENT_CASE
                else "N/A_BASELINE",
                "truth_use": str(row["truth_use"]),
            }
        )
    return core_rows, status


def _run_external(
    adata: ad.AnnData, dataset: Path, output: Path
) -> tuple[list[dict], list[dict]]:
    roots = roots_from_truth(adata)
    true_time = _timecourse(adata)
    prediction_dir = output / "external_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    root_rows: list[dict] = []
    context_rows: list[dict] = []
    for method, adapter in ADAPTERS.items():
        completed = 0
        for root_index, root_cell in enumerate(roots):
            started = time.perf_counter()
            try:
                prediction = adapter(adata, root_index, root_cell)
                row, exported = _score_root(adata, prediction, true_time)
                exported.insert(0, "dataset", dataset.stem)
                exported.to_csv(
                    prediction_dir
                    / f"{method.replace('+', '_')}__R{root_index}.csv",
                    index=False,
                )
                completed += int(row["status"] == "PASS")
            except Exception as exc:
                row = {
                    "method": method,
                    "root_index": root_index,
                    "root_cell": root_cell,
                    "status": "FAIL",
                    "root_coverage": 0.0,
                    "conditional_pt_spearman": np.nan,
                    "conditional_branch_ari": np.nan,
                    "branch_output_available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                (output / f"{method.replace('+', '_')}__R{root_index}_ERROR.log").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            row = {
                "dataset": dataset.stem,
                "seconds": round(time.perf_counter() - started, 3),
                **row,
            }
            root_rows.append(row)
        selected = [row for row in root_rows if row["dataset"] == dataset.stem and row["method"] == method]
        context_rows.append(
            {
                "dataset": dataset.stem,
                "comparison_tier": "CONTEXTUAL",
                "method": method,
                "status": "PASS" if completed == len(roots) else "FAIL",
                "n_roots": len(roots),
                "n_root_runs_completed": completed,
                "coverage": float(np.mean([float(row["root_coverage"]) for row in selected])),
                "pseudotime_spearman": float(
                    np.nanmean([float(row["conditional_pt_spearman"]) for row in selected])
                ),
                "branch_ARI_rootwise": float(
                    np.nanmean([float(row["conditional_branch_ari"]) for row in selected])
                ),
                "branch_output_rate": float(
                    np.mean([str(row["branch_output_available"]).lower() == "true" for row in selected])
                ),
                "topology_F1": "N/A_CAPABILITY_OR_LABEL_MAPPING",
                "root_F1": "N/A_ROOTS_SUPPLIED",
                "quality_denominator": "covered cells within each conditioned root",
                "uncovered_policy": "reported separately; never imputed",
                "primary_core_decision": "NOT_USED",
            }
        )
    return root_rows, context_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO
        / "outputs"
        / "NicheForest_paper_frozen_candidate"
        / "P4_root_conditioned_benchmark",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    runs = output / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (output / "P4_PREEXECUTION_GATE.json").write_text(
        json.dumps(PREEXECUTION_GATE, indent=2), encoding="utf-8"
    )
    core_rows: list[dict] = []
    external_rows: list[dict] = []
    root_rows: list[dict] = []
    statuses: list[dict] = []
    datasets = sorted(args.datasets.resolve().glob("*.h5ad"))
    for dataset in datasets:
        adata = ad.read_h5ad(dataset)
        dataset_output = runs / dataset.stem
        dataset_output.mkdir(parents=True, exist_ok=True)
        current_core, status = _run_core(adata, dataset, dataset_output)
        current_root, current_external = _run_external(
            adata, dataset, dataset_output
        )
        core_rows.extend(current_core)
        root_rows.extend(current_root)
        external_rows.extend(current_external)
        statuses.append({"dataset": dataset.stem, **status})
        pd.DataFrame(core_rows).to_csv(output / "P4_CORE_COMPARISON.csv", index=False)
        pd.DataFrame(root_rows).to_csv(output / "P4_EXTERNAL_ROOT_RUNS.csv", index=False)
        pd.DataFrame(external_rows).to_csv(output / "P4_EXTERNAL_CONTEXT.csv", index=False)

    core = pd.DataFrame(core_rows)
    wide = core.pivot(index="dataset", columns="method")
    delta_rows = []
    for dataset in sorted(core["dataset"].unique()):
        selected = core[core["dataset"].eq(dataset)].set_index("method")
        current = selected.loc["current branch-scale T0"]
        baseline = selected.loc["root-conditioned MSF"]
        delta_rows.append(
            {
                "dataset": dataset,
                "delta_branch_ARI_rootwise": float(current["branch_ARI_rootwise"])
                - float(baseline["branch_ARI_rootwise"]),
                "delta_topology_F1": float(current["topology_F1"])
                - float(baseline["topology_F1"]),
                "delta_pseudotime_spearman": float(current["pseudotime_spearman"])
                - float(baseline["pseudotime_spearman"]),
                "delta_shortcut_rate": float(current["fine_shortcut_rate"])
                - float(baseline["fine_shortcut_rate"]),
            }
        )
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(output / "P4_CORE_PAIRED_DELTA.csv", index=False)
    current_rows = core[core["method"].eq("current branch-scale T0")]
    n_positive = int(np.sum(deltas["delta_branch_ARI_rootwise"] > 0))
    checks = {
        "branch_mean_positive": bool(deltas["delta_branch_ARI_rootwise"].mean() > 0),
        "branch_median_positive": bool(deltas["delta_branch_ARI_rootwise"].median() > 0),
        "branch_positive_at_least_5_of_7": bool(n_positive >= 5),
        "topology_mean_no_harm": bool(deltas["delta_topology_F1"].mean() >= -0.05),
        "pseudotime_mean_no_harm": bool(
            deltas["delta_pseudotime_spearman"].mean() >= -0.05
        ),
        "candidate_legality_100pct": bool(current_rows["candidate_legal"].eq(True).all()),
        "forest_100pct": bool(current_rows["is_forest"].eq(True).all()),
        "state_tree_100pct": bool(
            current_rows["all_state_backbones_are_trees"].eq(True).all()
        ),
        "root_support_connected_100pct": bool(
            current_rows["all_root_supports_connected"].eq(True).all()
        ),
    }
    external_failures = pd.DataFrame(external_rows).query("status == 'FAIL'")
    decision = {
        "phase": "P4_ROOT_CONDITIONED_BENCHMARK",
        "method_frozen": True,
        "n_datasets": len(datasets),
        "primary_comparison": PREEXECUTION_GATE["primary_comparison"],
        "checks": checks,
        "paired_delta_summary": {
            "mean_branch_ARI": float(deltas["delta_branch_ARI_rootwise"].mean()),
            "median_branch_ARI": float(deltas["delta_branch_ARI_rootwise"].median()),
            "n_positive_branch_ARI": n_positive,
            "mean_topology_F1": float(deltas["delta_topology_F1"].mean()),
            "mean_pseudotime": float(deltas["delta_pseudotime_spearman"].mean()),
        },
        "external_context_failures": int(len(external_failures)),
        "primary_decision": "PASS" if all(checks.values()) else "FAIL",
        "P4_5_recommendation": "START_LOCKED_FIVE_SEED_CONFIRMATION"
        if all(checks.values())
        else "DO_NOT_START; AUDIT FROZEN RESULTS WITHOUT METHOD CHANGES",
    }
    (output / "P4_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    pd.DataFrame(statuses).to_csv(output / "P4_CORE_STATUS_AUDIT.csv", index=False)
    manifest = pd.DataFrame(
        [
            {"path": str(path), "sha256": _sha256(path)}
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "P4_SHA256_MANIFEST.csv"
        ]
    )
    manifest.to_csv(output / "P4_SHA256_MANIFEST.csv", index=False)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
