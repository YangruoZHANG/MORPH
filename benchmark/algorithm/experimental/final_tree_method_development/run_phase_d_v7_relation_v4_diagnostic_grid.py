#!/usr/bin/env python3
"""Run event-support-invariant Phase D v7 over diagnostic cases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
FINALIZER = HERE / "finalize_phase_b_v4_selected_dev.py"
REFINER = HERE / "run_phase_d_inferred_refinement_v3.py"
ABLATION = HERE / "run_phase_d_stage_ablation_v1.py"
CATALOG = HERE / "SYNTHETIC_RELATION_PROGRAM_CATALOG_V1.csv"


def execute(command: list[str], log: list[str]) -> None:
    process = subprocess.run(command, capture_output=True, text=True)
    log.extend(["COMMAND " + " ".join(command), process.stdout, process.stderr])
    if process.returncode:
        raise RuntimeError(f"exit={process.returncode}: {' '.join(command)}")


def run_one(status_file: Path) -> dict[str, str]:
    run = status_file.parent
    status = json.loads(status_file.read_text(encoding="utf-8"))
    setting = str(status["setting"])
    relation_source = run / "phase_b_v4_0_allbranch_dev"
    relation = run / "phase_b_v4_1_selected_dev"
    phase_d = run / "phase_d_v7_relation_v4_dev"
    ablation = run / "phase_d_m0_m6_v7_relation_v4_dev"
    complete = ablation / "PHASE_D_STAGE_ABLATION.csv"
    if complete.exists():
        return {"run": str(run), "status": "skipped_complete"}
    log: list[str] = []
    execute([
        sys.executable, str(FINALIZER), "--input", str(relation_source),
        "--output", str(relation),
    ], log)
    execute([
        sys.executable, str(REFINER), "--run-dir", str(run), "--dataset", setting,
        "--phase-a-dir", str(run / "phase_a_v4_frozen"),
        "--relation-dir", str(relation), "--program-catalog", str(CATALOG),
        "--output", str(phase_d),
    ], log)
    execute([
        sys.executable, str(ABLATION), "--run-dir", str(run), "--dataset", setting,
        "--phase-a-dir", str(run / "phase_a_v4_frozen"),
        "--phase-d-dir", str(phase_d), "--relation-dir", str(relation),
        "--output", str(ablation),
    ], log)
    (run / "PHASE_D_RELATION_V4_WORKER.log").write_text("\n".join(log), encoding="utf-8")
    return {"run": str(run), "status": "completed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    statuses = sorted(args.grid_root.rglob("LOCKED_CASE_STATUS.json"))
    failures: list[str] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, path): path for path in statuses}
        for count, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
                completed += 1
            except Exception as error:
                failures.append(f"{futures[future]}: {error}")
            if count % 25 == 0 or count == len(futures):
                print(json.dumps({
                    "finished": count, "total": len(futures), "failures": len(failures)
                }))
    result = {
        "role": "POST_LOCK_FAILURE_DIAGNOSTIC",
        "n_tasks": len(statuses), "n_completed": completed,
        "n_failures": len(failures), "failures": failures,
        "relation_version": "BRANCH_LOCAL_MULTICHANNEL_RELATION_V4_1_DEV",
        "refinement_version": "INFERRED_REFINEMENT_V7_EVENT_SUPPORT_INVARIANT_DEV",
    }
    (args.grid_root / "PHASE_D_RELATION_V4_DIAGNOSTIC_STATUS.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
