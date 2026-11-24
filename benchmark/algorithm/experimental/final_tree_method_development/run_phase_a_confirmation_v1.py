#!/usr/bin/env python3
"""Run predeclared PHASE A confirmation replicates with the frozen v3 method."""

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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "outputs" / "final_tree_method_development" / "phase_a_confirmation_runs",
    )
    args = parser.parse_args()
    registry = pd.read_csv(REGISTRY)
    row = registry[
        registry.replicate.astype(int).eq(args.replicate)
        & registry.role.astype(str).eq("phase_a_confirmation")
        & registry.status.astype(str).eq("registered_before_run")
    ]
    if len(row) != 1:
        raise ValueError("replicate is not an untouched predeclared PHASE A confirmation run")
    setting = str(row.iloc[0].setting_id)
    seed = int(row.iloc[0].seed)
    root = args.output_root.resolve()
    run = root / "core" / setting / f"replicate_{args.replicate:02d}_seed_{seed}"
    phase_a_output = (
        REPO / "outputs" / "final_tree_method_development" / "phase_a_v3"
        / "confirmation" / f"{setting}_rep{args.replicate:02d}_seed{seed}"
    )
    status_path = run / "PHASE_A_CONFIRMATION_STATUS.json"
    status = {
        "setting_id": setting,
        "replicate": args.replicate,
        "seed": seed,
        "method_frozen_before_run": "PHASE_A_v3_C3_dualchannel_v7",
        "relation_called": False,
        "refinement_called": False,
        "steps": {},
    }

    def checkpoint(name: str) -> None:
        status["steps"][name] = "complete"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    if not (run / "formal_metacell" / "metacell_adata.h5ad").exists():
        execute([
            sys.executable, str(GOLD / "run_benchmark_scenario_v1.py"),
            "--tier", "core", "--setting-id", setting,
            "--replicate", str(args.replicate), "--output-root", str(root),
        ])
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
    if not (phase_a_output / "RUN_STATUS.json").exists():
        execute([
            sys.executable, str(HERE / "run_phase_a_case_v3.py"),
            "--run-dir", str(run), "--dataset", setting,
            "--run-label", f"{setting}_rep{args.replicate:02d}_seed{seed}",
            "--output", str(phase_a_output),
        ])
    checkpoint("frozen_phase_a_v3")
    status["complete"] = True
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
