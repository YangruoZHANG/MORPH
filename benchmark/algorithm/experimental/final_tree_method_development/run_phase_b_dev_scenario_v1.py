#!/usr/bin/env python3
"""Build generator-v3 DEV cases with the frozen PHASE A method."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GOLD = REPO / "algorithm" / "experimental" / "cross_trajectory_gold_standard"
PROTOCOL = GOLD / "protocol_v1"
SOURCE = REPO / "outputs" / "cross_trajectory_gold_standard" / "protocol_runs_v1" / "core"
FINE = {
    "target_size": 25, "min_size": 5, "max_size": 75,
    "birch_multiplier": 1.0, "tiny_merge_size": 3,
}


def execute(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO, check=True)


def scenario(setting: str, replicate: int) -> tuple[dict, int]:
    settings = pd.read_csv(PROTOCOL / "CORE_SCENARIOS_V1.csv", keep_default_na=False)
    selected = settings[settings.setting_id.astype(str).eq(setting)]
    if len(selected) != 1:
        raise ValueError(f"Unknown setting {setting}")
    seeds = pd.read_csv(PROTOCOL / "CORE_SEEDS_V1.csv")
    seed_row = seeds[seeds.replicate.astype(int).eq(replicate)]
    if len(seed_row) != 1 or replicate >= 21:
        raise ValueError("DEV runner cannot access locked replicate 21 or later")
    row = selected.iloc[0]
    seed = int(seed_row.iloc[0].seed)
    n_roots = int(row.n_roots)
    result = {
        "benchmark_spec": "BENCHMARK_SPEC_V3",
        "tier": "core", "setting_id": setting, "family": str(row.family),
        "seed": seed, "n_roots": n_roots, "event_type": str(row.event_type),
        "effect_level": str(row.effect_level), "effect_strength": float(row.effect_strength),
        "confounding": str(row.confounding), "exposure_overlap": str(row.exposure_overlap),
        "spatial_overlap": str(row.spatial_overlap),
        "spatial_noise": float(row.get("spatial_noise", 0.0) or 0.0),
        "expression_noise": float(row.get("expression_noise", 0.45) or 0.45),
        "event_window_width": float(row.get("event_window_width", 0.50) or 0.50),
        "concurrent_events": int(row.concurrent_events),
        "membership_ambiguity": float(row.membership_ambiguity),
        "root_imbalance": str(row.get("root_imbalance", ":".join(["1"] * n_roots)) or ":".join(["1"] * n_roots)),
        "cells_per_root": int(row.cells_per_root), "n_genes": int(row.n_genes),
    }
    return result, seed


def find_base(
    setting: str, replicate: int, seed: int, config: dict, output_root: Path
) -> Path:
    n_roots = int(config["n_roots"])
    suffix = f"replicate_{replicate:02d}_seed_{seed}"
    direct = SOURCE / setting / suffix / "base"
    if direct.exists():
        return direct
    fallback = {2: "CORE_11", 3: "CORE_05", 4: "CORE_24"}[n_roots]
    candidate = SOURCE / fallback / suffix / "base"
    if candidate.exists():
        return candidate
    signature = (
        f"roots{n_roots}_cells{config['cells_per_root']}_genes{config['n_genes']}"
        f"_noise{config['expression_noise']}_imbalance{config['root_imbalance']}"
    ).replace(":", "-").replace(".", "p")
    return (
        output_root.resolve().parent / "scmultisim_base_dev_cache_v1"
        / signature / suffix / "base"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO / "outputs" / "final_tree_method_development" / "phase_b_scenario_dev_v16",
    )
    args = parser.parse_args()
    config, seed = scenario(args.setting, args.replicate)
    suffix = f"replicate_{args.replicate:02d}_seed_{seed}"
    run = args.output_root.resolve() / "core" / args.setting / suffix
    event = run / "event"
    metacell = run / "formal_metacell"
    upstream_root = run / "generic_upstream_v1_exact_root"
    phase_a = run / "phase_a_v4_frozen"
    run.mkdir(parents=True, exist_ok=True)
    scenario_json = run / "SCENARIO_CONFIG.json"
    scenario_json.write_text(json.dumps(config, indent=2), encoding="utf-8")
    base = find_base(
        args.setting, args.replicate, seed, config, args.output_root
    )
    if not (base / "scmultisim_manifest.csv").exists():
        env = os.environ.copy()
        env["R_LIBS_USER"] = str(REPO / ".r_libs")
        subprocess.run([
            r"D:\Program Files\R\R-4.5.2\bin\Rscript.exe",
            str(GOLD / "generate_scmultisim_base.R"), str(base), str(seed),
            str(config["cells_per_root"]), str(config["n_roots"]),
            str(config["n_genes"]), str(config["expression_noise"]),
            str(config["root_imbalance"]),
        ], cwd=REPO, env=env, check=True)
    generator_current = False
    version_file = event / "GENERATOR_VERSION.json"
    if version_file.exists():
        try:
            generator_current = (
                json.loads(version_file.read_text(encoding="utf-8"))
                .get("truth_generator_version") == "BENCHMARK_TRUTH_GENERATOR_V5_4_2_MULTI_SOURCE_ASSERTION_REPAIR"
            )
        except (json.JSONDecodeError, OSError):
            generator_current = False
    if not generator_current:
        execute([
            sys.executable, str(GOLD / "build_event_conditioned_benchmark_v3.py"),
            "--base-dir", str(base), "--output-dir", str(event),
            "--seed", str(seed), "--scenario-json", str(scenario_json),
        ])
    if not (metacell / "METACELL_TRUTH_INTEGRITY.json").exists():
        execute([
            sys.executable, str(GOLD / "construct_metacell_truth_v1.py"),
            "--input-h5ad", str(event / "synthetic.h5ad"),
            "--truth-dir", str(event / "truth"), "--output-dir", str(metacell),
            "--dataset", args.setting,
            "--target-size", str(FINE["target_size"]),
            "--min-size", str(FINE["min_size"]), "--max-size", str(FINE["max_size"]),
            "--birch-multiplier", str(FINE["birch_multiplier"]),
            "--tiny-merge-size", str(FINE["tiny_merge_size"]),
        ])
    if not (upstream_root / "GENERIC_UPSTREAM_STATUS.json").exists():
        execute([
            sys.executable, str(GOLD / "run_generic_upstream_v1.py"),
            "--metacell-h5ad", str(metacell / "metacell_adata.h5ad"),
            "--true-roots-csv", str(event / "truth" / "TRUE_ROOTS.csv"),
            "--cell-to-metacell-csv", str(metacell / "cell_to_metacell_mapping.csv"),
            "--dataset", args.setting, "--output-dir", str(upstream_root),
        ])
    if not (phase_a / "RUN_STATUS.json").exists():
        execute([
            sys.executable, str(HERE / "run_phase_a_case_v4.py"),
            "--run-dir", str(run), "--dataset", args.setting,
            "--run-label", f"{args.setting}_rep{args.replicate:02d}_seed{seed}_generator_v3",
            "--output", str(phase_a),
        ])
    status = {
        "setting": args.setting, "replicate": args.replicate, "seed": seed,
        "role": "development_only", "locked_replicates_21_30_touched": False,
        "benchmark_specification": "BENCHMARK_SPEC_V3",
        "truth_generator": "BENCHMARK_TRUTH_GENERATOR_V5_4_2_MULTI_SOURCE_ASSERTION_REPAIR",
        "phase_a_method": "PHASE_A_v4_generator_v2_fine_hvg_v9_regularized",
        "frozen_candidate_t0_modified": False, "complete": True,
    }
    (run / "PHASE_B_DEV_INPUT_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
