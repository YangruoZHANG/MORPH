#!/usr/bin/env python3
"""Execute one frozen benchmark setting/replicate from the protocol registry."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PROTOCOL = HERE / "protocol_v1"


def value(row: pd.Series, name: str, default):
    candidate = row.get(name, default)
    return default if pd.isna(candidate) or candidate == "" else candidate


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["core", "stress"], required=True)
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rscript", default=r"D:\Program Files\R\R-4.5.2\bin\Rscript.exe")
    parser.add_argument("--skip-metacell", action="store_true")
    args = parser.parse_args()

    scenario_file = PROTOCOL / ("CORE_SCENARIOS_V1.csv" if args.tier == "core" else "STRESS_SCENARIOS_V1.csv")
    seed_file = PROTOCOL / ("CORE_SEEDS_V1.csv" if args.tier == "core" else "STRESS_SEEDS_V1.csv")
    scenarios = pd.read_csv(scenario_file, keep_default_na=False)
    selected = scenarios.loc[scenarios["setting_id"].astype(str).eq(args.setting_id)]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one registry row for {args.setting_id}, found {len(selected)}")
    seeds = pd.read_csv(seed_file, keep_default_na=False)
    seed_row = seeds.loc[seeds["replicate"].astype(int).eq(args.replicate)]
    if len(seed_row) != 1:
        raise ValueError(f"Replicate {args.replicate} is not frozen in {seed_file.name}")
    row = selected.iloc[0]
    seed = int(seed_row.iloc[0]["seed"])

    n_roots = int(value(row, "n_roots", 3))
    scenario = {
        "benchmark_spec": "BENCHMARK_SPEC_V1",
        "tier": args.tier,
        "setting_id": args.setting_id,
        "family": str(value(row, "family", "one_factor_stress")),
        "seed": seed,
        "n_roots": n_roots,
        "event_type": str(value(row, "event_type", "promotion")),
        "effect_level": str(value(row, "effect_level", "medium")),
        "effect_strength": float(value(row, "effect_strength", 0.7)),
        "confounding": str(value(row, "confounding", "low")),
        "exposure_overlap": str(value(row, "exposure_overlap", "adequate")),
        "spatial_overlap": str(value(row, "spatial_overlap", "adequate")),
        "spatial_noise": float(value(row, "spatial_noise", 0.0)),
        "expression_noise": float(value(row, "expression_noise", 0.45)),
        "event_window_width": float(value(row, "event_window_width", 0.50)),
        "concurrent_events": int(value(row, "concurrent_events", 1)),
        "membership_ambiguity": float(value(row, "membership_ambiguity", 0.0)),
        "root_imbalance": str(value(row, "root_imbalance", ":".join(["1"] * n_roots))),
        "cells_per_root": int(value(row, "cells_per_root", 400)),
        "n_genes": int(value(row, "n_genes", 300)),
    }
    run_dir = args.output_root.resolve() / args.tier / args.setting_id / f"replicate_{args.replicate:02d}_seed_{seed}"
    base_dir = run_dir / "base"
    event_dir = run_dir / "event"
    metacell_dir = run_dir / "formal_metacell"
    run_dir.mkdir(parents=True, exist_ok=True)
    scenario_json = run_dir / "SCENARIO_CONFIG.json"
    scenario_json.write_text(json.dumps(scenario, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["R_LIBS_USER"] = str(REPO / ".r_libs")
    run([
        args.rscript,
        str(HERE / "generate_scmultisim_base.R"),
        str(base_dir),
        str(seed),
        str(scenario["cells_per_root"]),
        str(n_roots),
        str(scenario["n_genes"]),
        str(scenario["expression_noise"]),
        str(scenario["root_imbalance"]),
    ], env=env)
    run([
        sys.executable,
        str(HERE / "build_event_conditioned_benchmark.py"),
        "--base-dir", str(base_dir),
        "--output-dir", str(event_dir),
        "--seed", str(seed),
        "--scenario-json", str(scenario_json),
    ])
    if not args.skip_metacell:
        run([
            sys.executable,
            str(HERE / "construct_metacell_truth_v1.py"),
            "--input-h5ad", str(event_dir / "synthetic.h5ad"),
            "--truth-dir", str(event_dir / "truth"),
            "--output-dir", str(metacell_dir),
            "--dataset", args.setting_id,
        ])
    print(json.dumps({"status": "complete", "run_dir": str(run_dir), **scenario}, indent=2))


if __name__ == "__main__":
    main()
