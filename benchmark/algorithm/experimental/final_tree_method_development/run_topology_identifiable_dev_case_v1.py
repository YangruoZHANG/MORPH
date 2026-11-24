#!/usr/bin/env python3
"""Build one topology-identifiable development case through frozen Phase A."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GOLD = REPO / "algorithm" / "experimental" / "cross_trajectory_gold_standard"


def load_locked_helpers():
    path = HERE / "run_phase_e_locked_case_v1.py"
    spec = importlib.util.spec_from_file_location("phase_e_case_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HELPERS = load_locked_helpers()


def execute(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    args = parser.parse_args()
    config, seed = HELPERS.config_for(args.setting, args.replicate)
    suffix = f"replicate_{args.replicate:02d}_seed_{seed}"
    run = args.output_root.resolve() / "core" / args.setting / suffix
    base = HELPERS.base_path(
        args.base_cache.resolve(), config, args.replicate, seed
    )
    if not (base / "scmultisim_manifest.csv").exists():
        raise FileNotFoundError(f"Prepared base missing: {base}")
    event = run / "event"
    metacell = run / "formal_metacell"
    upstream = run / "generic_upstream_v1_exact_root"
    phase_a = run / "phase_a_v6_branch_state_dev"
    run.mkdir(parents=True, exist_ok=True)
    scenario_json = run / "SCENARIO_CONFIG.json"
    scenario_json.write_text(json.dumps(config, indent=2), encoding="utf-8")
    execute([
        sys.executable, str(GOLD / "build_event_conditioned_benchmark_v4.py"),
        "--base-dir", str(base), "--output-dir", str(event),
        "--seed", str(seed), "--scenario-json", str(scenario_json),
    ])
    execute([
        sys.executable, str(GOLD / "construct_metacell_truth_v1.py"),
        "--input-h5ad", str(event / "synthetic.h5ad"),
        "--truth-dir", str(event / "truth"), "--output-dir", str(metacell),
        "--dataset", args.setting, "--target-size", "25", "--min-size", "5",
        "--max-size", "75", "--birch-multiplier", "1.0", "--tiny-merge-size", "3",
    ])
    execute([
        sys.executable, str(GOLD / "run_generic_upstream_v1.py"),
        "--metacell-h5ad", str(metacell / "metacell_adata.h5ad"),
        "--true-roots-csv", str(event / "truth" / "TRUE_ROOTS.csv"),
        "--cell-to-metacell-csv", str(metacell / "cell_to_metacell_mapping.csv"),
        "--dataset", args.setting, "--output-dir", str(upstream),
    ])
    execute([
        sys.executable, str(HERE / "run_phase_a_case_v6_branch_state.py"),
        "--run-dir", str(run), "--dataset", args.setting,
        "--run-label", f"{args.setting}_rep{args.replicate:02d}_topology_identifiable_dev",
        "--output", str(phase_a), "--production-only",
    ])
    status = {
        "setting": args.setting, "replicate": args.replicate, "seed": seed,
        "evaluation_split": "DEVELOPMENT_TOPOLOGY_IDENTIFIABILITY",
        "benchmark_specification": "BENCHMARK_SPEC_V4_TOPOLOGY_IDENTIFIABLE",
        "truth_generator": "BENCHMARK_TRUTH_GENERATOR_V6_TOPOLOGY_IDENTIFIABLE",
        "phase_a_method": "PHASE_A_v6_branch_program_by_pseudotime_state",
        "complete": True,
    }
    (run / "TOPOLOGY_DEV_CASE_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
