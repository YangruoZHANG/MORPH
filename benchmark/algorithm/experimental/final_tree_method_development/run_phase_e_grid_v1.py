#!/usr/bin/env python3
"""Checkpointed launcher for the frozen 900-task Phase E benchmark."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from run_phase_e_locked_case_v1 import base_path, config_for


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GOLD = REPO / "algorithm" / "experimental" / "cross_trajectory_gold_standard"
PROTOCOL = GOLD / "protocol_v1"
R_SCRIPT = Path(r"D:\Program Files\R\R-4.5.2\bin\Rscript.exe")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--settings", nargs="*")
    parser.add_argument("--replicates", nargs="*", type=int)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--base-workers", type=int, default=3)
    parser.add_argument("--permutations", type=int, default=399)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.base_cache.mkdir(parents=True, exist_ok=True)
    scenarios = pd.read_csv(PROTOCOL / "CORE_SCENARIOS_V1.csv", keep_default_na=False)
    seeds = pd.read_csv(PROTOCOL / "CORE_SEEDS_V1.csv")
    settings = args.settings or scenarios.setting_id.astype(str).tolist()
    replicates = args.replicates or seeds.replicate.astype(int).tolist()
    tasks = []
    bases: dict[Path, tuple[dict[str, object], int, int]] = {}
    for setting in settings:
        for replicate in replicates:
            config, seed = config_for(setting, replicate)
            base = base_path(args.base_cache.resolve(), config, replicate, seed)
            bases[base] = (config, replicate, seed)
            run = args.output_root.resolve() / "core" / setting / f"replicate_{replicate:02d}_seed_{seed}"
            tasks.append((setting, replicate, seed, run))
    plan = pd.DataFrame([{
        "setting_id": setting, "replicate": replicate, "seed": seed,
        "evaluation_split": "LOCKED_VALIDATION" if replicate >= 21 else "DEV_GRID",
        "run_dir": str(run), "complete": (run / "COMPLETE.marker").exists(),
    } for setting, replicate, seed, run in tasks])
    plan.to_csv(args.output_root / "PHASE_E_GRID_PLAN.csv", index=False)
    freeze_files = sorted(HERE.glob("*.py")) + [
        HERE / "SYNTHETIC_RELATION_PROGRAM_CATALOG_V1.csv",
        GOLD / "build_event_conditioned_benchmark.py",
        GOLD / "build_event_conditioned_benchmark_v2.py",
        GOLD / "build_event_conditioned_benchmark_v3.py",
        GOLD / "construct_metacell_truth_v1.py",
        GOLD / "run_generic_upstream_v1.py",
        PROTOCOL / "CORE_SCENARIOS_V1.csv",
        PROTOCOL / "CORE_SEEDS_V1.csv",
    ]
    freeze = {
        "benchmark_specification": "BENCHMARK_SPEC_V1_FROZEN_GRID_METHOD_V5",
        "planned_tasks": len(tasks), "planned_unique_bases": len(bases),
        "permutations": args.permutations,
        "method_frozen_before_locked_results": True,
        "files": {str(path.relative_to(REPO)): sha256(path) for path in freeze_files},
    }
    (args.output_root / "PHASE_E_METHOD_FREEZE.json").write_text(
        json.dumps(freeze, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps({
            "tasks": len(tasks), "unique_bases": len(bases),
            "already_complete": int(plan.complete.sum()), "dry_run": True,
        }, indent=2))
        return

    def prepare_base(item):
        base, (config, replicate, seed) = item
        manifest = base / "scmultisim_manifest.csv"
        if manifest.exists():
            return {"base": str(base), "status": "cached", "seconds": 0.0}
        base.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["R_LIBS_USER"] = str(REPO / ".r_libs")
        started = time.time()
        command = [
            str(R_SCRIPT), str(GOLD / "generate_scmultisim_base.R"),
            str(base), str(seed), str(config["cells_per_root"]),
            str(config["n_roots"]), str(config["n_genes"]),
            str(config["expression_noise"]), str(config["root_imbalance"]),
        ]
        completed = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True)
        (base.parent / "BASE_GENERATION.log").write_text(
            completed.stdout + "\nSTDERR\n" + completed.stderr, encoding="utf-8"
        )
        return {
            "base": str(base),
            "status": "complete" if completed.returncode == 0 and manifest.exists() else "failed",
            "return_code": completed.returncode, "seconds": time.time() - started,
        }

    base_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.base_workers, 1)) as pool:
        for result in pool.map(prepare_base, sorted(bases.items(), key=lambda item: str(item[0]))):
            base_results.append(result)
            print(json.dumps({"stage": "base", **result}), flush=True)
    pd.DataFrame(base_results).to_csv(args.output_root / "PHASE_E_BASE_RUN_SUMMARY.csv", index=False)
    if any(result["status"] == "failed" for result in base_results):
        raise SystemExit("Base preparation failed")

    def execute_case(task):
        setting, replicate, seed, run = task
        marker = run / "COMPLETE.marker"
        if marker.exists():
            return {"setting_id": setting, "replicate": replicate, "status": "cached", "seconds": 0.0}
        command = [
            sys.executable, str(HERE / "run_phase_e_locked_case_v1.py"),
            "--setting", setting, "--replicate", str(replicate),
            "--output-root", str(args.output_root.resolve()),
            "--base-cache", str(args.base_cache.resolve()),
            "--permutations", str(args.permutations),
        ]
        started = time.time()
        completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
        run.mkdir(parents=True, exist_ok=True)
        (run / "PHASE_E_WORKER.log").write_text(
            completed.stdout + "\nSTDERR\n" + completed.stderr, encoding="utf-8"
        )
        return {
            "setting_id": setting, "replicate": replicate,
            "evaluation_split": "LOCKED_VALIDATION" if replicate >= 21 else "DEV_GRID",
            "status": "complete" if completed.returncode == 0 and marker.exists() else "failed",
            "return_code": completed.returncode, "seconds": time.time() - started,
            "log": str(run / "PHASE_E_WORKER.log"),
        }

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.max_workers, 1)) as pool:
        for result in pool.map(execute_case, tasks):
            results.append(result)
            print(json.dumps({"stage": "case", **result}), flush=True)
    frame = pd.DataFrame(results)
    frame.to_csv(args.output_root / "PHASE_E_GRID_RUN_SUMMARY.csv", index=False)
    if (frame.status == "failed").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
