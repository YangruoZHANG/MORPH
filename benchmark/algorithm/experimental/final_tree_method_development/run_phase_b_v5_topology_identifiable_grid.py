#!/usr/bin/env python3
"""Run joint-aligned Phase B v5 evidence on a topology-identifiable DEV grid."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
DETECTOR = HERE / "run_phase_b_multichannel_detector_v5.py"
CATALOG = HERE / "SYNTHETIC_RELATION_PROGRAM_CATALOG_V1.csv"


def run_one(
    status_file: Path, output_subdir: str, phase_a_subdir: str, permutations: int
) -> dict[str, object]:
    run = status_file.parent
    status = json.loads(status_file.read_text(encoding="utf-8"))
    output = run / output_subdir
    complete = output / "RUN_STATUS.json"
    if complete.exists():
        payload = json.loads(complete.read_text(encoding="utf-8"))
        if payload.get("complete"):
            return {"run": str(run), "status": "skipped_complete"}
    command = [
        sys.executable, str(DETECTOR),
        "--run-dir", str(run),
        "--dataset", str(status["setting"]),
        "--phase-a-dir", str(run / phase_a_subdir),
        "--program-catalog", str(CATALOG),
        "--output", str(output),
        "--permutations", str(permutations),
        "--seed", str(int(status["seed"]) + 2_000_003),
        "--primary-significance", "conjunctive",
        "--multiplicity", "BH",
        "--collision-environment-trust-min", "0.40",
        "--competition-scope", "event-local",
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    (output / "GRID_WORKER.log").write_text(
        process.stdout + "\nSTDERR\n" + process.stderr, encoding="utf-8"
    )
    if process.returncode:
        raise RuntimeError(f"{run}: detector exit {process.returncode}")
    return {"run": str(run), "status": "completed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--output-subdir", default="phase_b_v4_0_allbranch_dev")
    parser.add_argument("--phase-a-subdir", default="phase_a_v7_1_state_constrained_fine_dev")
    parser.add_argument("--permutations", type=int, default=399)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    statuses = sorted(args.grid_root.rglob("TOPOLOGY_DEV_CASE_STATUS.json"))
    if not statuses:
        raise RuntimeError(f"No topology DEV cases under {args.grid_root}")
    results: list[dict[str, object]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one, path, args.output_subdir,
                args.phase_a_subdir, args.permutations,
            ): path
            for path in statuses
        }
        for count, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(f"{path}: {error}")
            if count % 25 == 0 or count == len(futures):
                print(json.dumps({
                    "finished": count,
                    "total": len(futures),
                    "failures": len(failures),
                }))
    manifest = {
        "role": "TOPOLOGY_IDENTIFIABLE_RELATION_DEVELOPMENT",
        "n_tasks": len(statuses),
        "n_completed_or_skipped": len(results),
        "n_failures": len(failures),
        "failures": failures,
        "output_subdir": args.output_subdir,
        "phase_a_subdir": args.phase_a_subdir,
        "permutations": args.permutations,
        "workers": args.workers,
    }
    (args.grid_root / "PHASE_B_V4_DIAGNOSTIC_GRID_STATUS.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
