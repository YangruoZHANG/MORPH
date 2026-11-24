#!/usr/bin/env python3
"""Rebuild consumed DEV replicates with generator v2 and frozen fine metacells."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GOLD = REPO / "algorithm" / "experimental" / "cross_trajectory_gold_standard"
REGISTRY = HERE / "DEVELOPMENT_REPLICATE_REGISTRY.csv"
METHOD_VERSION = "PHASE_A_v4_generator_v2_fine_hvg_v8"
FINE_PROFILE = {
    "target_size": 25,
    "min_size": 5,
    "max_size": 75,
    "birch_multiplier": 1.0,
    "tiny_merge_size": 3,
}


def execute(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO, check=True)


def source_run(replicate: int, seed: int) -> Path:
    suffix = f"replicate_{replicate:02d}_seed_{seed}"
    if replicate == 1:
        return (
            REPO / "outputs" / "cross_trajectory_gold_standard" /
            "protocol_runs_v1" / "core" / "CORE_05" / suffix
        )
    family = (
        "independent_seed_protocol_runs" if replicate <= 4
        else "phase_a_confirmation_runs"
    )
    return (
        REPO / "outputs" / "final_tree_method_development" /
        family / "core" / "CORE_05" / suffix
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument(
        "--output-root", type=Path,
        default=(
            REPO / "outputs" / "final_tree_method_development" /
            "phase_a_v4_generator_v2_fine_runs"
        ),
    )
    args = parser.parse_args()
    registry = pd.read_csv(REGISTRY)
    row = registry[
        registry.replicate.astype(int).eq(args.replicate)
        & registry.setting_id.astype(str).eq("CORE_05")
        & registry.replicate.astype(int).between(1, 8)
    ]
    if len(row) != 1:
        raise ValueError("PHASE A v4 development is restricted to consumed CORE_05 reps 1-8")
    seed = int(row.iloc[0].seed)
    source = source_run(args.replicate, seed)
    required_base = {
        "scmultisim_cell_metadata.csv", "scmultisim_counts.csv.gz",
        "scmultisim_manifest.csv", "scmultisim_spatial.csv",
        "scmultisim_within_root_cci.csv",
    }
    observed_base = {path.name for path in (source / "base").glob("*")}
    if not required_base.issubset(observed_base):
        missing = sorted(required_base - observed_base)
        raise FileNotFoundError(f"Missing frozen scMultiSim inputs: {missing}")
    output_root = args.output_root.resolve()
    run = output_root / "core" / "CORE_05" / f"replicate_{args.replicate:02d}_seed_{seed}"
    event = run / "event"
    metacell = run / "formal_metacell"
    upstream_root = run / "generic_upstream_v1_exact_root"
    phase_a = run / "phase_a_v4"
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "SCENARIO_CONFIG.json", run / "SCENARIO_CONFIG.json")

    status = {
        "method_version": METHOD_VERSION,
        "benchmark_specification": "BENCHMARK_SPEC_V2",
        "replicate": args.replicate,
        "seed": seed,
        "role": "development_only_consumed_seed",
        "locked_replicates_21_30_touched": False,
        "fine_metacell_profile": FINE_PROFILE,
        "steps": {},
    }
    status_path = run / "PHASE_A_V4_STATUS.json"

    def checkpoint(name: str) -> None:
        status["steps"][name] = "complete"
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    if not (event / "GENERATOR_VERSION.json").exists():
        execute([
            sys.executable, str(GOLD / "build_event_conditioned_benchmark_v2.py"),
            "--base-dir", str(source / "base"),
            "--output-dir", str(event),
            "--seed", str(seed),
            "--scenario-json", str(run / "SCENARIO_CONFIG.json"),
        ])
    checkpoint("truth_generator_v2")
    if not (metacell / "METACELL_TRUTH_INTEGRITY.json").exists():
        execute([
            sys.executable, str(GOLD / "construct_metacell_truth_v1.py"),
            "--input-h5ad", str(event / "synthetic.h5ad"),
            "--truth-dir", str(event / "truth"),
            "--output-dir", str(metacell),
            "--dataset", "CORE_05",
            "--target-size", str(FINE_PROFILE["target_size"]),
            "--min-size", str(FINE_PROFILE["min_size"]),
            "--max-size", str(FINE_PROFILE["max_size"]),
            "--birch-multiplier", str(FINE_PROFILE["birch_multiplier"]),
            "--tiny-merge-size", str(FINE_PROFILE["tiny_merge_size"]),
        ])
    checkpoint("fine_metacell_truth_projection")
    if not (upstream_root / "GENERIC_UPSTREAM_STATUS.json").exists():
        execute([
            sys.executable, str(GOLD / "run_generic_upstream_v1.py"),
            "--metacell-h5ad", str(metacell / "metacell_adata.h5ad"),
            "--true-roots-csv", str(event / "truth" / "TRUE_ROOTS.csv"),
            "--cell-to-metacell-csv", str(metacell / "cell_to_metacell_mapping.csv"),
            "--dataset", "CORE_05", "--output-dir", str(upstream_root),
        ])
    checkpoint("root_and_soft_assignment_upstream")
    if not (phase_a / "HVG_V8_DIAGNOSTIC_METRICS.csv").exists():
        execute([
            sys.executable, str(HERE / "diagnose_phase_a_hvg_v8.py"),
            "--run-dir", str(run), "--dataset", "CORE_05",
            "--run-label", f"CORE_05_rep{args.replicate:02d}_seed{seed}",
            "--output", str(phase_a),
        ])
    checkpoint("phase_a_v4_three_way_audit")
    status["complete"] = True
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
