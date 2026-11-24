#!/usr/bin/env python3
"""Merge frozen Phase B contact-scale evidence over a benchmark grid."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
MERGER = HERE / "run_phase_b_multiscale_consensus_v1.py"


def run_one(status_file: Path, scale065: str, scale080: str, output_subdir: str) -> None:
    run = status_file.parent
    output = run / output_subdir
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(MERGER),
        "--scale065-dir", str(run / scale065),
        "--scale080-dir", str(run / scale080),
        "--output", str(output),
    ], check=True, capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--scale065-subdir", required=True)
    parser.add_argument("--scale080-subdir", required=True)
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    statuses = sorted(args.grid_root.rglob("TOPOLOGY_DEV_CASE_STATUS.json"))
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one, status, args.scale065_subdir,
                args.scale080_subdir, args.output_subdir,
            ): status for status in statuses
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(f"{futures[future]}: {error}")
    status = {
        "method": "PHASE_B_V7_1_MULTISCALE_LOCALIZED_STABILITY_SELECTION",
        "n_tasks": len(statuses), "n_failures": len(failures),
        "failures": failures, "output_subdir": args.output_subdir,
    }
    (args.grid_root / "PHASE_B_MULTISCALE_GRID_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
