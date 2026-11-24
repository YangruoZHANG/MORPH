#!/usr/bin/env python3
"""Checkpointed execution of one frozen NicheForest benchmark replicate."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PROTOCOL = HERE / "protocol_v1"


def run(command: list[str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def frozen_seed(tier: str, replicate: int) -> int:
    name = "CORE_SEEDS_V1.csv" if tier == "core" else "STRESS_SEEDS_V1.csv"
    seeds = pd.read_csv(PROTOCOL / name, keep_default_na=False)
    row = seeds[seeds["replicate"].astype(int).eq(replicate)]
    if len(row) != 1:
        raise ValueError(f"Replicate {replicate} is not frozen in {name}")
    return int(row.iloc[0]["seed"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["core", "stress"], required=True)
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seed = frozen_seed(args.tier, args.replicate)
    run_dir = args.output_root.resolve() / args.tier / args.setting_id / f"replicate_{args.replicate:02d}_seed_{seed}"
    event_dir = run_dir / "event"
    truth_dir = event_dir / "truth"
    metacell_dir = run_dir / "formal_metacell"
    upstream_root = run_dir / "generic_upstream_v1_exact_root"
    upstream_dir = upstream_root / args.setting_id
    relation_dir = run_dir / "generic_relation_refinement_v1"
    status_path = run_dir / "PIPELINE_STATUS.json"
    status = {
        "tier": args.tier,
        "setting_id": args.setting_id,
        "replicate": args.replicate,
        "seed": seed,
        "started_at_unix": time.time(),
        "steps": {},
    }

    def checkpoint(step: str, state: str) -> None:
        status["steps"][step] = state
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    try:
        generated = (truth_dir / "TRUE_EVENTS.csv").exists() and (metacell_dir / "metacell_adata.h5ad").exists()
        if args.force or not generated:
            checkpoint("generation", "running")
            run([
                sys.executable, str(HERE / "run_benchmark_scenario_v1.py"),
                "--tier", args.tier, "--setting-id", args.setting_id,
                "--replicate", str(args.replicate), "--output-root", str(args.output_root.resolve()),
            ])
        checkpoint("generation", "complete")

        upstream_complete = (upstream_root / "GENERIC_UPSTREAM_STATUS.json").exists()
        if args.force or not upstream_complete:
            checkpoint("upstream", "running")
            run([
                sys.executable, str(HERE / "run_generic_upstream_v1.py"),
                "--metacell-h5ad", str(metacell_dir / "metacell_adata.h5ad"),
                "--true-roots-csv", str(truth_dir / "TRUE_ROOTS.csv"),
                "--cell-to-metacell-csv", str(metacell_dir / "cell_to_metacell_mapping.csv"),
                "--dataset", args.setting_id, "--output-dir", str(upstream_root),
            ])
        checkpoint("upstream", "complete")

        relation_complete = (relation_dir / "RUN_STATUS.json").exists()
        if args.force or not relation_complete:
            checkpoint("relation_refinement", "running")
            run([
                sys.executable, str(HERE / "run_generic_relation_refinement_v1.py"),
                "--upstream-dir", str(upstream_dir),
                "--fixed-root-input", str(upstream_root / "FIXED_ROOT_INPUT.csv"),
                "--cell-h5ad", str(event_dir / "synthetic.h5ad"),
                "--cell-to-metacell", str(metacell_dir / "cell_to_metacell_mapping.csv"),
                "--output-dir", str(relation_dir), "--seed", str(seed),
                "--permutations", str(args.permutations),
            ])
        checkpoint("relation_refinement", "complete")

        for stage in ("M0", "M1"):
            prediction = run_dir / f"pred_{stage}"
            if args.force or not (prediction / "PRED_T1_EDGES.csv").exists():
                run([
                    sys.executable, str(HERE / "export_m0_m1_prediction_v1.py"),
                    "--upstream-dir", str(upstream_dir),
                    "--fixed-root-input", str(upstream_root / "FIXED_ROOT_INPUT.csv"),
                    "--output-dir", str(prediction), "--stage", stage,
                ])
        for stage in ("M2", "M3", "M4"):
            prediction = run_dir / f"pred_{stage}"
            if args.force or not (prediction / "PRED_T1_EDGES.csv").exists():
                run([
                    sys.executable, str(HERE / "export_generic_upstream_prediction_v1.py"),
                    "--upstream-dir", str(upstream_dir),
                    "--fixed-root-input", str(upstream_root / "FIXED_ROOT_INPUT.csv"),
                    "--output-dir", str(prediction), "--stage", stage,
                ])
        shutil.copyfile(
            relation_dir / "prediction" / "PRED_EVENTS.csv",
            run_dir / "pred_M4" / "PRED_EVENTS.csv",
        )
        m6_prediction = run_dir / "pred_M6"
        if args.force or not (m6_prediction / "PRED_T1_EDGES.csv").exists():
            run([
                sys.executable, str(HERE / "export_m6_pruned_prediction_v1.py"),
                "--m5-prediction-dir", str(relation_dir / "prediction"),
                "--output-dir", str(m6_prediction),
            ])
        checkpoint("stage_export", "complete")

        predictions = {
            "M0": run_dir / "pred_M0",
            "M1": run_dir / "pred_M1",
            "M2": run_dir / "pred_M2",
            "M3": run_dir / "pred_M3",
            "M4": run_dir / "pred_M4",
            "M5": relation_dir / "prediction",
            "M6": run_dir / "pred_M6",
        }
        score_rows = []
        for stage, prediction in predictions.items():
            score_dir = run_dir / f"score_{stage}_generic_v1"
            if args.force or not (score_dir / "SCORE_SUMMARY.csv").exists():
                run([
                    sys.executable, str(HERE / "score_benchmark_prediction_v1.py"),
                    "--truth-dir", str(truth_dir), "--prediction-dir", str(prediction),
                    "--output-dir", str(score_dir), "--method", f"NicheForest_{stage}",
                    "--metacell-truth-dir", str(metacell_dir),
                ])
            row = pd.read_csv(score_dir / "SCORE_SUMMARY.csv").iloc[0].to_dict()
            row.update({"stage": stage, "setting_id": args.setting_id, "replicate": args.replicate, "seed": seed})
            score_rows.append(row)
        pd.DataFrame(score_rows).to_csv(run_dir / "STAGE_SCORE_SUMMARY.csv", index=False)
        checkpoint("scoring", "complete")
        status["status"] = "complete"
        status["finished_at_unix"] = time.time()
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        (run_dir / "COMPLETE.marker").write_text("complete\n", encoding="ascii")
        print(json.dumps(status, indent=2))
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["finished_at_unix"] = time.time()
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
