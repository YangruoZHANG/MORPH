#!/usr/bin/env python3
"""Run predeclared metacell-resolution profiles on one diagnostic replicate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

PROFILES = {
    "default": {"target_size": 50, "min_size": 10, "max_size": 150, "birch_multiplier": 1.50, "tiny_merge_size": 5},
    "fine": {"target_size": 25, "min_size": 5, "max_size": 75, "birch_multiplier": 1.00, "tiny_merge_size": 3},
    "finer": {"target_size": 15, "min_size": 3, "max_size": 45, "birch_multiplier": 0.75, "tiny_merge_size": 2},
}


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profiles", nargs="+", default=["fine", "finer"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth_dir = args.run_dir / "event" / "truth"
    h5ad = args.run_dir / "event" / "synthetic.h5ad"
    rows = []
    for profile_name in args.profiles:
        profile = PROFILES[profile_name]
        profile_dir = args.output_dir / profile_name
        metacell_dir = profile_dir / "metacell"
        upstream_root = profile_dir / "upstream"
        upstream = upstream_root / args.setting
        run([
            sys.executable, str(HERE / "construct_metacell_truth_v1.py"),
            "--input-h5ad", str(h5ad), "--truth-dir", str(truth_dir),
            "--output-dir", str(metacell_dir), "--dataset", args.setting,
            "--target-size", str(profile["target_size"]),
            "--min-size", str(profile["min_size"]),
            "--max-size", str(profile["max_size"]),
            "--birch-multiplier", str(profile["birch_multiplier"]),
            "--tiny-merge-size", str(profile["tiny_merge_size"]),
        ])
        run([
            sys.executable, str(HERE / "run_generic_upstream_v1.py"),
            "--metacell-h5ad", str(metacell_dir / "metacell_adata.h5ad"),
            "--true-roots-csv", str(truth_dir / "TRUE_ROOTS.csv"),
            "--cell-to-metacell-csv", str(metacell_dir / "cell_to_metacell_mapping.csv"),
            "--dataset", args.setting, "--output-dir", str(upstream_root),
        ])
        prediction = profile_dir / "pred_M3"
        run([
            sys.executable, str(HERE / "export_generic_upstream_prediction_v1.py"),
            "--upstream-dir", str(upstream),
            "--fixed-root-input", str(upstream_root / "FIXED_ROOT_INPUT.csv"),
            "--output-dir", str(prediction), "--stage", "M3",
        ])
        score_dir = profile_dir / "score_M3"
        run([
            sys.executable, str(HERE / "score_benchmark_prediction_v1.py"),
            "--truth-dir", str(truth_dir), "--prediction-dir", str(prediction),
            "--output-dir", str(score_dir), "--method", f"M3_resolution_{profile_name}",
            "--metacell-truth-dir", str(metacell_dir),
        ])
        score_row = pd.read_csv(score_dir / "SCORE_SUMMARY.csv").iloc[0].to_dict()
        truth_summary = pd.read_csv(metacell_dir / "METACELL_TRUTH_SUMMARY.csv")
        rows.append({
            "profile": profile_name,
            **profile,
            "n_metacells": len(truth_summary),
            "median_metacell_size": float(truth_summary["n_cells"].median()),
            "mean_branch_T0_purity": float(truth_summary["branch_T0_purity"].mean()),
            "hard_truth_eligible_fraction": float(truth_summary["hard_truth_eligible"].mean()),
            **score_row,
            "truth_used_for_evaluation_only": True,
        })
    pd.DataFrame(rows).to_csv(args.output_dir / "METACELL_RESOLUTION_SENSITIVITY.csv", index=False)
    status = {
        "protocol": "ORACLE_DECOMPOSITION_SPEC_V1",
        "profiles": {name: PROFILES[name] for name in args.profiles},
        "one_factor_interpretation": "profiles jointly reduce target, minimum, maximum, BIRCH threshold and tiny-merge sizes; no single parameter effect is claimed",
        "completed": True,
    }
    (args.output_dir / "RUN_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
