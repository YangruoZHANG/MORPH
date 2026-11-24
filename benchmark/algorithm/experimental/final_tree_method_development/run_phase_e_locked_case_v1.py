#!/usr/bin/env python3
"""Run one frozen NicheForest paper-benchmark case from a prepared base."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
CATALOG = HERE / "SYNTHETIC_RELATION_PROGRAM_CATALOG_V1.csv"
GENERATOR_VERSION = "BENCHMARK_TRUTH_GENERATOR_V5_4_2_MULTI_SOURCE_ASSERTION_REPAIR"
PHASE_A_VERSION = "PHASE_A_v4_generator_v2_fine_hvg_v9_regularized"
RELATION_VERSION = "MULTICHANNEL_RELATION_V3_8"
REFINEMENT_VERSION = "INFERRED_REFINEMENT_V5_SUPPORTED_PROGRAM_BOUNDARY"
FINE = {
    "target_size": 25, "min_size": 5, "max_size": 75,
    "birch_multiplier": 1.0, "tiny_merge_size": 3,
}


def execute(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_for(setting: str, replicate: int) -> tuple[dict[str, object], int]:
    settings = pd.read_csv(PROTOCOL / "CORE_SCENARIOS_V1.csv", keep_default_na=False)
    selected = settings[settings.setting_id.astype(str).eq(setting)]
    seeds = pd.read_csv(PROTOCOL / "CORE_SEEDS_V1.csv")
    seed_row = seeds[seeds.replicate.astype(int).eq(replicate)]
    if len(selected) != 1 or len(seed_row) != 1:
        raise ValueError(f"Unknown frozen task {setting} replicate {replicate}")
    row = selected.iloc[0]
    seed = int(seed_row.iloc[0].seed)
    n_roots = int(row.n_roots)
    value = {
        "benchmark_spec": "BENCHMARK_SPEC_V1_FROZEN_GRID_METHOD_V5",
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
        "root_imbalance": str(
            row.get("root_imbalance", ":".join(["1"] * n_roots))
            or ":".join(["1"] * n_roots)
        ),
        "cells_per_root": int(row.cells_per_root), "n_genes": int(row.n_genes),
    }
    return value, seed


def base_path(base_cache: Path, config: dict[str, object], replicate: int, seed: int) -> Path:
    signature = (
        f"roots{config['n_roots']}_cells{config['cells_per_root']}_genes{config['n_genes']}"
        f"_noise{config['expression_noise']}_imbalance{config['root_imbalance']}"
    ).replace(":", "-").replace(".", "p")
    return base_cache / signature / f"replicate_{replicate:02d}_seed_{seed}" / "base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=399)
    args = parser.parse_args()
    config, seed = config_for(args.setting, args.replicate)
    suffix = f"replicate_{args.replicate:02d}_seed_{seed}"
    run = args.output_root.resolve() / "core" / args.setting / suffix
    base = base_path(args.base_cache.resolve(), config, args.replicate, seed)
    if not (base / "scmultisim_manifest.csv").exists():
        raise FileNotFoundError(f"Prepared base missing: {base}")
    event = run / "event"
    metacell = run / "formal_metacell"
    upstream = run / "generic_upstream_v1_exact_root"
    phase_a = run / "phase_a_v4_frozen"
    relation = run / "phase_b_v3_8_frozen"
    phase_d = run / "phase_d_v5_frozen"
    ablation = run / "phase_d_m0_m6_frozen"
    run.mkdir(parents=True, exist_ok=True)
    scenario_json = run / "SCENARIO_CONFIG.json"
    scenario_json.write_text(json.dumps(config, indent=2), encoding="utf-8")
    execute([
        sys.executable, str(GOLD / "build_event_conditioned_benchmark_v3.py"),
        "--base-dir", str(base), "--output-dir", str(event),
        "--seed", str(seed), "--scenario-json", str(scenario_json),
    ])
    execute([
        sys.executable, str(GOLD / "construct_metacell_truth_v1.py"),
        "--input-h5ad", str(event / "synthetic.h5ad"),
        "--truth-dir", str(event / "truth"), "--output-dir", str(metacell),
        "--dataset", args.setting, "--target-size", str(FINE["target_size"]),
        "--min-size", str(FINE["min_size"]), "--max-size", str(FINE["max_size"]),
        "--birch-multiplier", str(FINE["birch_multiplier"]),
        "--tiny-merge-size", str(FINE["tiny_merge_size"]),
    ])
    execute([
        sys.executable, str(GOLD / "run_generic_upstream_v1.py"),
        "--metacell-h5ad", str(metacell / "metacell_adata.h5ad"),
        "--true-roots-csv", str(event / "truth" / "TRUE_ROOTS.csv"),
        "--cell-to-metacell-csv", str(metacell / "cell_to_metacell_mapping.csv"),
        "--dataset", args.setting, "--output-dir", str(upstream),
    ])
    execute([
        sys.executable, str(HERE / "run_phase_a_case_v4.py"),
        "--run-dir", str(run), "--dataset", args.setting,
        "--run-label", f"{args.setting}_rep{args.replicate:02d}_seed{seed}_locked_v5",
        "--output", str(phase_a),
    ])
    execute([
        sys.executable, str(HERE / "run_phase_b_multichannel_detector_v3.py"),
        "--run-dir", str(run), "--dataset", args.setting,
        "--phase-a-dir", str(phase_a), "--program-catalog", str(CATALOG),
        "--output", str(relation), "--permutations", str(args.permutations),
        "--seed", str(seed + 19001),
    ])
    execute([
        sys.executable, str(HERE / "run_phase_d_inferred_refinement_v1.py"),
        "--run-dir", str(run), "--dataset", args.setting,
        "--phase-a-dir", str(phase_a), "--relation-dir", str(relation),
        "--program-catalog", str(CATALOG), "--output", str(phase_d),
    ])
    execute([
        sys.executable, str(HERE / "run_phase_d_stage_ablation_v1.py"),
        "--run-dir", str(run), "--dataset", args.setting,
        "--phase-a-dir", str(phase_a), "--phase-d-dir", str(phase_d),
        "--relation-dir", str(relation), "--output", str(ablation),
    ])
    provenance_files = [
        HERE / "run_phase_a_case_v4.py",
        HERE / "run_phase_b_relation_detector_v2.py",
        HERE / "run_phase_b_multichannel_detector_v3.py",
        HERE / "run_phase_d_inferred_refinement_v1.py",
        HERE / "run_phase_d_stage_ablation_v1.py",
        GOLD / "build_event_conditioned_benchmark_v3.py",
        PROTOCOL / "CORE_SCENARIOS_V1.csv",
        PROTOCOL / "CORE_SEEDS_V1.csv",
        CATALOG,
    ]
    status = {
        "setting": args.setting, "replicate": args.replicate, "seed": seed,
        "evaluation_split": "LOCKED_VALIDATION" if args.replicate >= 21 else "DEV_GRID",
        "benchmark_specification": "BENCHMARK_SPEC_V1_FROZEN_GRID_METHOD_V5",
        "truth_generator": GENERATOR_VERSION, "phase_a_method": PHASE_A_VERSION,
        "relation_detector": RELATION_VERSION, "refinement_operator": REFINEMENT_VERSION,
        "permutations": args.permutations,
        "source_hashes": {str(path.relative_to(REPO)): sha256(path) for path in provenance_files},
        "complete": True,
    }
    (run / "LOCKED_CASE_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    (run / "COMPLETE.marker").write_text("complete\n", encoding="ascii")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
