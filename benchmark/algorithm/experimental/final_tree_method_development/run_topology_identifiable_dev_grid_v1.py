#!/usr/bin/env python3
"""Run topology-identifiable development generation over the core scenarios."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
CASE = HERE / "run_topology_identifiable_dev_case_v1.py"
PROTOCOL = HERE.parent / "cross_trajectory_gold_standard" / "protocol_v1"


def run_one(setting: str, replicate: int, output_root: Path, base_cache: Path) -> None:
    suffix = None
    seeds = pd.read_csv(PROTOCOL / "CORE_SEEDS_V1.csv")
    seed = int(seeds.loc[seeds.replicate.astype(int).eq(replicate), "seed"].iloc[0])
    suffix = f"replicate_{replicate:02d}_seed_{seed}"
    complete = output_root / "core" / setting / suffix / "TOPOLOGY_DEV_CASE_STATUS.json"
    if complete.exists():
        return
    process = subprocess.run([
        sys.executable, str(CASE), "--setting", setting,
        "--replicate", str(replicate), "--output-root", str(output_root),
        "--base-cache", str(base_cache),
    ], capture_output=True, text=True)
    run = complete.parent
    run.mkdir(parents=True, exist_ok=True)
    (run / "TOPOLOGY_DEV_WORKER.log").write_text(
        process.stdout + "\nSTDERR\n" + process.stderr, encoding="utf-8"
    )
    if process.returncode:
        raise RuntimeError(f"{setting} replicate {replicate}: exit {process.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--replicate", type=int, default=21)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    settings = pd.read_csv(PROTOCOL / "CORE_SCENARIOS_V1.csv").setting_id.astype(str).tolist()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one, setting, args.replicate,
                args.output_root.resolve(), args.base_cache.resolve(),
            ): setting for setting in settings
        }
        for count, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
            except Exception as error:
                failures.append(f"{futures[future]}: {error}")
            if count % 5 == 0 or count == len(futures):
                print(json.dumps({
                    "finished": count, "total": len(futures), "failures": len(failures)
                }))
    result = {
        "role": "TOPOLOGY_IDENTIFIABLE_DEVELOPMENT_ONLY",
        "replicate": args.replicate, "n_tasks": len(settings),
        "n_failures": len(failures), "failures": failures,
    }
    (args.output_root / "TOPOLOGY_IDENTIFIABLE_GRID_STATUS.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
