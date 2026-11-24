#!/usr/bin/env python3
"""Run Phase D v10 inferred refinement and M0-M6 ablation on one grid."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
REFINER = HERE / "run_phase_d_inferred_refinement_v3.py"
ABLATION = HERE / "run_phase_d_stage_ablation_v1.py"
CATALOG = HERE / "SYNTHETIC_RELATION_PROGRAM_CATALOG_V1.csv"


def execute(command: list[str], log: list[str]) -> None:
    process = subprocess.run(command, capture_output=True, text=True)
    log.extend(["COMMAND " + " ".join(command), process.stdout, process.stderr])
    if process.returncode:
        raise RuntimeError(process.stderr[-2000:])


def run_one(
    status_file: Path, phase_a_subdir: str, relation_subdir: str,
    phase_d_subdir: str, ablation_subdir: str,
    min_support: float, skip_ablation: bool,
) -> None:
    run = status_file.parent
    status = json.loads(status_file.read_text(encoding="utf-8"))
    setting = str(status["setting"])
    phase_d = run / phase_d_subdir
    ablation = run / ablation_subdir
    if (ablation / "PHASE_D_STAGE_ABLATION.csv").exists():
        return
    phase_d.mkdir(parents=True, exist_ok=True)
    ablation.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    execute([
        sys.executable, str(REFINER), "--run-dir", str(run),
        "--dataset", setting, "--phase-a-dir", str(run / phase_a_subdir),
        "--relation-dir", str(run / relation_subdir),
        "--program-catalog", str(CATALOG), "--output", str(phase_d),
        "--min-downstream-support-fraction", str(min_support),
    ], log)
    if not skip_ablation:
        execute([
            sys.executable, str(ABLATION), "--run-dir", str(run),
            "--dataset", setting, "--phase-a-dir", str(run / phase_a_subdir),
            "--phase-d-dir", str(phase_d), "--relation-dir", str(run / relation_subdir),
            "--output", str(ablation),
        ], log)
    (run / f"{phase_d_subdir}_WORKER.log").write_text(
        "\n".join(log), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--phase-a-subdir", required=True)
    parser.add_argument("--relation-subdir", required=True)
    parser.add_argument("--phase-d-subdir", required=True)
    parser.add_argument("--ablation-subdir", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--min-support", type=float, default=0.20)
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()
    statuses = sorted(args.grid_root.rglob("TOPOLOGY_DEV_CASE_STATUS.json"))
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one, status, args.phase_a_subdir, args.relation_subdir,
                args.phase_d_subdir, args.ablation_subdir,
                args.min_support, args.skip_ablation,
            ): status for status in statuses
        }
        for count, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
            except Exception as error:
                failures.append(f"{futures[future]}: {error}")
            if count % 10 == 0 or count == len(futures):
                print(json.dumps({
                    "finished": count, "total": len(futures),
                    "failures": len(failures),
                }))
    result = {
        "phase": "D", "method": "INFERRED_REFINEMENT_V10_STRUCTURAL_TRANSITION_SAFETY_GATE",
        "n_tasks": len(statuses), "n_failures": len(failures),
        "failures": failures, "phase_d_subdir": args.phase_d_subdir,
        "ablation_subdir": args.ablation_subdir,
        "min_support": args.min_support,
        "skip_ablation": args.skip_ablation,
    }
    (args.grid_root / "PHASE_D_V10_GRID_STATUS.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
