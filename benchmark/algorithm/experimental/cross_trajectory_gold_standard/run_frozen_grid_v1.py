#!/usr/bin/env python3
"""Parallel, checkpoint-aware launcher for the frozen benchmark grid."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol_v1"


def registry(tier: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = "CORE" if tier == "core" else "STRESS"
    return (
        pd.read_csv(PROTOCOL / f"{prefix}_SCENARIOS_V1.csv", keep_default_na=False),
        pd.read_csv(PROTOCOL / f"{prefix}_SEEDS_V1.csv", keep_default_na=False),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["core", "stress"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--settings", nargs="*")
    parser.add_argument("--replicates", nargs="*", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenarios, seeds = registry(args.tier)
    settings = args.settings or scenarios["setting_id"].astype(str).tolist()
    replicates = args.replicates or seeds["replicate"].astype(int).tolist()
    seed_lookup = dict(zip(seeds["replicate"].astype(int), seeds["seed"].astype(int)))
    tasks = []
    for setting in settings:
        if setting not in set(scenarios["setting_id"].astype(str)):
            raise ValueError(f"Unknown frozen setting: {setting}")
        for replicate in replicates:
            seed = seed_lookup[replicate]
            run_dir = args.output_root.resolve() / args.tier / setting / f"replicate_{replicate:02d}_seed_{seed}"
            tasks.append((setting, replicate, run_dir))

    planned = pd.DataFrame([
        {
            "setting_id": setting,
            "replicate": replicate,
            "run_dir": str(run_dir),
            "checkpoint_complete": (run_dir / "COMPLETE.marker").exists(),
        }
        for setting, replicate, run_dir in tasks
    ])
    args.output_root.mkdir(parents=True, exist_ok=True)
    planned.to_csv(args.output_root / f"{args.tier.upper()}_GRID_PLAN.csv", index=False)
    if args.dry_run:
        print(json.dumps({
            "tier": args.tier,
            "tasks": len(tasks),
            "already_complete": int(planned["checkpoint_complete"].sum()),
            "remaining": int((~planned["checkpoint_complete"]).sum()),
            "dry_run": True,
        }, indent=2))
        return

    def execute(task):
        setting, replicate, run_dir = task
        if (run_dir / "COMPLETE.marker").exists():
            return {"setting_id": setting, "replicate": replicate, "status": "skipped_complete", "seconds": 0.0}
        command = [
            sys.executable, str(HERE / "run_full_benchmark_pipeline_v1.py"),
            "--tier", args.tier, "--setting-id", setting, "--replicate", str(replicate),
            "--output-root", str(args.output_root.resolve()),
            "--permutations", str(args.permutations),
        ]
        started = time.time()
        completed = subprocess.run(command, capture_output=True, text=True)
        log = run_dir / "GRID_WORKER.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout + "\nSTDERR\n" + completed.stderr, encoding="utf-8")
        return {
            "setting_id": setting,
            "replicate": replicate,
            "status": "complete" if completed.returncode == 0 else "failed",
            "return_code": completed.returncode,
            "seconds": time.time() - started,
            "log": str(log),
        }

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        for result in pool.map(execute, tasks):
            results.append(result)
            print(json.dumps(result), flush=True)
    frame = pd.DataFrame(results)
    frame.to_csv(args.output_root / f"{args.tier.upper()}_GRID_RUN_SUMMARY.csv", index=False)
    if (frame["status"] == "failed").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
