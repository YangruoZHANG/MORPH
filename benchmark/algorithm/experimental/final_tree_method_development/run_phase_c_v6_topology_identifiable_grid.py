#!/usr/bin/env python3
"""Run Phase C v6 on topology-identifiable development cases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
ORACLE = HERE / "run_phase_c_oracle_refinement_v6.py"


def run_one(status_file: Path, output_subdir: str, phase_a_subdir: str) -> str:
    run = status_file.parent
    status = json.loads(status_file.read_text(encoding="utf-8"))
    output = run / output_subdir
    complete = output / "RUN_STATUS.json"
    if complete.exists():
        return "skipped_complete"
    process = subprocess.run([
        sys.executable, str(ORACLE), "--run-dir", str(run),
        "--dataset", str(status["setting"]),
        "--phase-a-dir", str(run / phase_a_subdir),
        "--output", str(output),
    ], capture_output=True, text=True)
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
    parser.add_argument("--output-subdir", default="phase_c_oracle_fine_v5_topology_dev")
    parser.add_argument("--phase-a-subdir", default="phase_a_v6_branch_state_dev")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    statuses = sorted(args.grid_root.rglob("TOPOLOGY_DEV_CASE_STATUS.json"))
    if not statuses:
        raise RuntimeError(f"No TOPOLOGY_DEV_CASE_STATUS.json files under {args.grid_root}")
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one, path, args.output_subdir, args.phase_a_subdir
            ): path for path in statuses
        }
        for count, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
            except Exception as error:
                failures.append(f"{futures[future]}: {error}")
            if count % 25 == 0 or count == len(futures):
                print(json.dumps({
                    "finished": count, "total": len(futures), "failures": len(failures)
                }))
    result = {
        "role": "TOPOLOGY_IDENTIFIABLE_ORACLE_DEVELOPMENT",
        "n_tasks": len(statuses), "n_failures": len(failures),
        "failures": failures, "output_subdir": args.output_subdir,
        "phase_a_subdir": args.phase_a_subdir,
    }
    (args.grid_root / "PHASE_C_FULLGRID_STATUS.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
