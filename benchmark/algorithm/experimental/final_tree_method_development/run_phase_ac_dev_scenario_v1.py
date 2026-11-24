#!/usr/bin/env python3
"""Build non-locked scenario DEV runs for frozen PHASE A and oracle PHASE C."""

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
PROTOCOL = GOLD / "protocol_v1"
FINE = {
    "target_size": 25, "min_size": 5, "max_size": 75,
    "birch_multiplier": 1.0, "tiny_merge_size": 3,
}


def execute(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument(
        "--source-root", type=Path,
        default=REPO / "outputs" / "cross_trajectory_gold_standard" / "protocol_runs_v1" / "core",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO / "outputs" / "final_tree_method_development" / "phase_ac_scenario_dev_v1",
    )
    args = parser.parse_args()
    seeds = pd.read_csv(PROTOCOL / "CORE_SEEDS_V1.csv")
    selected = seeds[seeds.replicate.astype(int).eq(args.replicate)]
    if len(selected) != 1 or args.replicate >= 21:
        raise ValueError("Scenario DEV runner cannot access locked replicate 21 or later")
    seed = int(selected.iloc[0].seed)
    suffix = f"replicate_{args.replicate:02d}_seed_{seed}"
    source = args.source_root.resolve() / args.setting / suffix
    scenario = source / "SCENARIO_CONFIG.json"
    if not scenario.exists():
        raise FileNotFoundError(scenario)
    run = args.output_root.resolve() / "core" / args.setting / suffix
    event = run / "event"
    metacell = run / "formal_metacell"
    upstream_root = run / "generic_upstream_v1_exact_root"
    phase_a = run / "phase_a_v4_frozen"
    phase_c = run / "phase_c_oracle_v1"
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scenario, run / "SCENARIO_CONFIG.json")
    if not (event / "GENERATOR_VERSION.json").exists():
        execute([
            sys.executable, str(GOLD / "build_event_conditioned_benchmark_v2.py"),
            "--base-dir", str(source / "base"), "--output-dir", str(event),
            "--seed", str(seed), "--scenario-json", str(run / "SCENARIO_CONFIG.json"),
        ])
    if not (metacell / "METACELL_TRUTH_INTEGRITY.json").exists():
        execute([
            sys.executable, str(GOLD / "construct_metacell_truth_v1.py"),
            "--input-h5ad", str(event / "synthetic.h5ad"),
            "--truth-dir", str(event / "truth"), "--output-dir", str(metacell),
            "--dataset", args.setting,
            "--target-size", str(FINE["target_size"]),
            "--min-size", str(FINE["min_size"]),
            "--max-size", str(FINE["max_size"]),
            "--birch-multiplier", str(FINE["birch_multiplier"]),
            "--tiny-merge-size", str(FINE["tiny_merge_size"]),
        ])
    if not (upstream_root / "GENERIC_UPSTREAM_STATUS.json").exists():
        execute([
            sys.executable, str(GOLD / "run_generic_upstream_v1.py"),
            "--metacell-h5ad", str(metacell / "metacell_adata.h5ad"),
            "--true-roots-csv", str(event / "truth" / "TRUE_ROOTS.csv"),
            "--cell-to-metacell-csv", str(metacell / "cell_to_metacell_mapping.csv"),
            "--dataset", args.setting, "--output-dir", str(upstream_root),
        ])
    if not (phase_a / "RUN_STATUS.json").exists():
        execute([
            sys.executable, str(HERE / "run_phase_a_case_v4.py"),
            "--run-dir", str(run), "--dataset", args.setting,
            "--run-label", f"{args.setting}_rep{args.replicate:02d}_seed{seed}",
            "--output", str(phase_a),
        ])
    if not (phase_c / "RUN_STATUS.json").exists():
        execute([
            sys.executable, str(HERE / "run_phase_c_oracle_refinement_v1.py"),
            "--run-dir", str(run), "--dataset", args.setting,
            "--phase-a-dir", str(phase_a), "--output", str(phase_c),
        ])
    status = {
        "setting": args.setting, "replicate": args.replicate, "seed": seed,
        "role": "development_only", "locked_replicates_21_30_touched": False,
        "benchmark_specification": "BENCHMARK_SPEC_V2",
        "phase_a_method": "PHASE_A_v4_generator_v2_fine_hvg_v9_regularized",
        "phase_c_operator": "ORACLE_REFINEMENT_V1",
        "complete": True,
    }
    (run / "PHASE_AC_DEV_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
