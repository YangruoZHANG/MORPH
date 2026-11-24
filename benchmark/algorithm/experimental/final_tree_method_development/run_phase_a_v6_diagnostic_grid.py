#!/usr/bin/env python3
"""Run the Phase A v6 program-by-time state constructor on a diagnostic grid."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONSTRUCTOR = HERE / "run_phase_a_case_v6_branch_state.py"
VERSION = "PHASE_A_v6_branch_program_by_pseudotime_state"


def run_one(status_file: Path, output_subdir: str) -> str:
    run = status_file.parent
    status = json.loads(status_file.read_text(encoding="utf-8"))
    output = run / output_subdir
    complete = output / "RUN_STATUS.json"
    if complete.exists():
        recorded = json.loads(complete.read_text(encoding="utf-8"))
        if recorded.get("version") == VERSION:
            return "skipped_complete"
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "LOKY_MAX_CPU_COUNT": "1"})
    process = subprocess.run([
        sys.executable, str(CONSTRUCTOR), "--run-dir", str(run),
        "--dataset", str(status["setting"]),
        "--run-label", f"{status['setting']}_rep{int(status['replicate']):02d}_phase_a_v6_dev",
        "--output", str(output), "--production-only",
    ], capture_output=True, text=True, env=env)
    output.mkdir(parents=True, exist_ok=True)
    (output / "GRID_WORKER.log").write_text(
        process.stdout + "\nSTDERR\n" + process.stderr, encoding="utf-8"
    )
    if process.returncode:
        raise RuntimeError(f"{run}: exit {process.returncode}")
    return "completed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--output-subdir", default="phase_a_v6_branch_state_dev")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--replicate", type=int)
    args = parser.parse_args()
    statuses = sorted(args.grid_root.rglob("LOCKED_CASE_STATUS.json"))
    if args.replicate is not None:
        statuses = [
            path for path in statuses
            if int(json.loads(path.read_text(encoding="utf-8"))["replicate"]) == args.replicate
        ]
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, path, args.output_subdir): path for path in statuses}
        for count, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
            except Exception as error:
                failures.append(f"{futures[future]}: {error}")
            if count % 10 == 0 or count == len(futures):
                print(json.dumps({"finished": count, "total": len(futures), "failures": len(failures)}))
    result = {
        "role": "POST_LOCK_PHASE_A_REDEVELOPMENT_DIAGNOSTIC",
        "method_version": VERSION, "n_tasks": len(statuses),
        "n_failures": len(failures), "failures": failures,
        "output_subdir": args.output_subdir,
    }
    (args.grid_root / "PHASE_A_V6_GRID_STATUS.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
