#!/usr/bin/env python3
"""Run predeclared C1/C3 development replicates without relation inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GOLD = REPO / "algorithm" / "experimental" / "cross_trajectory_gold_standard"
REGISTRY = HERE / "DEVELOPMENT_REPLICATE_REGISTRY.csv"


def execute(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs" / "final_tree_method_development" / "independent_seed_protocol_runs")
    args = parser.parse_args()
    registry = pd.read_csv(REGISTRY)
    row = registry[
        registry.replicate.astype(int).eq(args.replicate)
        & registry.role.astype(str).eq("development_replication")
    ]
    if len(row) != 1:
        raise ValueError("replicate is not predeclared for C1/C3 development")
    setting = str(row.iloc[0].setting_id)
    seed = int(row.iloc[0].seed)
    root = args.output_root.resolve()
    run = root / "core" / setting / f"replicate_{args.replicate:02d}_seed_{seed}"
    status_path = run / "C1_C3_REPLICATION_STATUS.json"
    status = {"setting_id": setting, "replicate": args.replicate, "seed": seed, "relation_called": False, "steps": {}}

    def checkpoint(name: str) -> None:
        status["steps"][name] = "complete"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    if not (run / "formal_metacell" / "metacell_adata.h5ad").exists():
        execute([sys.executable, str(GOLD / "run_benchmark_scenario_v1.py"), "--tier", "core", "--setting-id", setting, "--replicate", str(args.replicate), "--output-root", str(root)])
    checkpoint("generation_and_truth_safe_metacells")
    upstream_root = run / "generic_upstream_v1_exact_root"
    if not (upstream_root / "GENERIC_UPSTREAM_STATUS.json").exists():
        execute([
            sys.executable, str(GOLD / "run_generic_upstream_v1.py"),
            "--metacell-h5ad", str(run / "formal_metacell" / "metacell_adata.h5ad"),
            "--true-roots-csv", str(run / "event" / "truth" / "TRUE_ROOTS.csv"),
            "--cell-to-metacell-csv", str(run / "formal_metacell" / "cell_to_metacell_mapping.csv"),
            "--dataset", setting, "--output-dir", str(upstream_root),
        ])
    checkpoint("root_and_soft_assignment_upstream")
    baseline = run / "pred_M3"
    if not (baseline / "PRED_T1_EDGES.csv").exists():
        execute([
            sys.executable, str(GOLD / "export_generic_upstream_prediction_v1.py"),
            "--upstream-dir", str(upstream_root / setting),
            "--fixed-root-input", str(upstream_root / "FIXED_ROOT_INPUT.csv"),
            "--output-dir", str(baseline), "--stage", "M3",
        ])
    baseline_score = run / "score_M3"
    if not (baseline_score / "SCORE_SUMMARY.csv").exists():
        execute([
            sys.executable, str(GOLD / "score_benchmark_prediction_v1.py"),
            "--truth-dir", str(run / "event" / "truth"), "--prediction-dir", str(baseline),
            "--output-dir", str(baseline_score), "--method", "NicheForest_M3",
            "--metacell-truth-dir", str(run / "formal_metacell"),
        ])
    checkpoint("baseline_export_and_score")
    method_output = run / "C1_v1_C3_v2"
    if not (method_output / "RUN_DECISION_INPUT.json").exists():
        execute([
            sys.executable, str(HERE / "run_c1_c3_development_v2.py"),
            "--project-root", str(REPO), "--run-dir", str(run), "--dataset", setting,
            "--output", str(method_output),
        ])
    checkpoint("frozen_C1_v1_C3_v2")
    status["complete"] = True
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
