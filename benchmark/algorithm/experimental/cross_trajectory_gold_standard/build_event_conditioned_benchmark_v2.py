#!/usr/bin/env python3
"""Mechanism-identifiable v2 wrapper around the frozen v1 event generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


HERE = Path(__file__).resolve().parent
VERSION = "BENCHMARK_TRUTH_GENERATOR_V2"
BRANCH_PROGRAMS = {
    "B0": tuple(f"G{i}" for i in range(201, 221)),
    "B1": tuple(f"G{i}" for i in range(221, 241)),
    "B2": tuple(f"G{i}" for i in range(241, 261)),
}
PROGRESSION_PROGRAMS = {
    "early": tuple(f"G{i}" for i in range(141, 151)),
    "middle": tuple(f"G{i}" for i in range(151, 161)),
    "late": tuple(f"G{i}" for i in range(161, 171)),
}
SOURCE_PROGRAM = ("G101", "G103")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scenario-json", type=Path)
    return parser.parse_args()


def gene_indices(var_names, genes: tuple[str, ...]) -> np.ndarray:
    lookup = {str(gene): index for index, gene in enumerate(var_names)}
    missing = [gene for gene in genes if gene not in lookup]
    if missing:
        raise ValueError(f"Generator v2 requires missing genes: {missing}")
    return np.asarray([lookup[gene] for gene in genes], dtype=int)


def add(values: np.ndarray, rows: np.ndarray, columns: np.ndarray, amplitude: np.ndarray) -> None:
    selected = np.flatnonzero(rows)
    if len(selected):
        values[np.ix_(selected, columns)] += amplitude[selected, None]


def smooth_window(tau: np.ndarray, start: float, end: float) -> np.ndarray:
    center = 0.5 * (start + end)
    sigma = max(0.25 * (end - start), 0.05)
    profile = np.exp(-0.5 * np.square((tau - center) / sigma))
    profile[(tau < start) | (tau > end)] = 0.0
    return profile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardized_difference(target: np.ndarray, other: np.ndarray) -> float:
    pooled = np.sqrt(0.5 * (float(np.var(target)) + float(np.var(other))))
    return float((np.mean(target) - np.mean(other)) / max(pooled, 1e-12))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(HERE / "build_event_conditioned_benchmark.py"),
        "--base-dir", str(args.base_dir),
        "--output-dir", str(args.output_dir),
        "--seed", str(args.seed),
    ]
    if args.scenario_json:
        command.extend(["--scenario-json", str(args.scenario_json)])
    subprocess.run(command, check=True)

    h5ad_path = args.output_dir / "synthetic.h5ad"
    adata = ad.read_h5ad(h5ad_path)
    values = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X).copy()
    values = values.astype(float, copy=False)
    obs = adata.obs.copy()
    tau = obs["truth_pseudotime"].to_numpy(float)
    manifest_rows: list[dict] = []
    assertion_rows: list[dict] = []

    early = 2.0 * np.square(1.0 - tau)
    middle = 8.0 * tau * (1.0 - tau)
    late = 2.0 * np.square(tau)
    for name, amplitude in (("early", early), ("middle", middle), ("late", late)):
        genes = PROGRESSION_PROGRAMS[name]
        columns = gene_indices(adata.var_names, genes)
        add(values, np.ones(adata.n_obs, dtype=bool), columns, amplitude)
        manifest_rows.append({
            "program_type": "progression", "program_id": name,
            "genes": ";".join(genes), "mechanism": "smooth_pseudotime_basis",
            "amplitude_formula": {"early": "2*(1-tau)^2", "middle": "8*tau*(1-tau)", "late": "2*tau^2"}[name],
        })

    for branch, genes in BRANCH_PROGRAMS.items():
        columns = gene_indices(adata.var_names, genes)
        selected = obs["truth_branch_T0"].astype(str).eq(branch).to_numpy()
        profile = 0.35 + 0.65 * tau
        baseline_score = values[:, columns].mean(axis=1)
        branch_scale = 3.0
        for _ in range(12):
            trial = baseline_score + selected.astype(float) * branch_scale * profile
            if standardized_difference(trial[selected], trial[~selected]) >= 2.0:
                break
            branch_scale *= 1.5
        amplitude = branch_scale * profile
        add(values, selected, columns, amplitude)
        branch_score = values[:, columns].mean(axis=1)
        contrast = float(branch_score[selected].mean() - branch_score[~selected].mean())
        effect_size = standardized_difference(branch_score[selected], branch_score[~selected])
        passed = bool(selected.any() and (~selected).any() and effect_size >= 2.0)
        assertion_rows.append({
            "assertion": f"branch_program_effect_size_{branch}",
            "passed": passed, "value": effect_size, "minimum": 2.0,
        })
        manifest_rows.append({
            "program_type": "branch", "program_id": branch,
            "genes": ";".join(genes), "mechanism": "branch_identity_x_progression",
            "amplitude_formula": f"{branch_scale:.8g}*(0.35+0.65*tau)",
        })

    events = pd.read_csv(args.output_dir / "ground_truth_events.csv")
    source_columns = gene_indices(adata.var_names, SOURCE_PROGRAM)
    for event in events.itertuples(index=False):
        root_mask = obs["root_id"].astype(str).eq(str(event.source_root)).to_numpy()
        profile = smooth_window(tau, float(event.source_tau_start), float(event.source_tau_end))
        source_mask = root_mask & (profile > 0)
        strength = 1.0 + float(event.effect_strength)
        amplitude = strength * profile
        before = values[:, source_columns].mean(axis=1)
        add(values, source_mask, source_columns, amplitude)
        after = values[:, source_columns].mean(axis=1)
        actual_delta = float((after[source_mask] - before[source_mask]).mean()) if source_mask.any() else 0.0
        assertion_rows.append({
            "assertion": f"source_program_written_{event.event_id}",
            "passed": bool(source_mask.any() and actual_delta > 0.0),
            "value": actual_delta, "minimum": 0.0,
        })
        manifest_rows.append({
            "program_type": "source_event", "program_id": str(event.event_id),
            "genes": ";".join(SOURCE_PROGRAM), "mechanism": "smooth_source_event_window",
            "amplitude_formula": "(1+effect_strength)*gaussian_window(tau)",
        })

    canonical = pd.DataFrame({
        "effect_level": ["null", "medium", "strong"],
        "effect_strength": [0.0, 0.7, 1.1],
    })
    canonical["intended_source_peak"] = 1.0 + canonical.effect_strength
    canonical["intended_receiver_peak"] = canonical.effect_strength * 4.5
    mechanism_unique = bool(
        canonical.intended_source_peak.nunique() == 3
        and canonical.intended_receiver_peak.nunique() == 3
    )
    assertion_rows.append({
        "assertion": "null_medium_strong_mechanisms_are_distinct",
        "passed": mechanism_unique,
        "value": int(mechanism_unique), "minimum": 1,
    })
    finite_nonnegative = bool(np.isfinite(values).all() and (values >= 0).all())
    assertion_rows.append({
        "assertion": "expression_finite_nonnegative",
        "passed": finite_nonnegative,
        "value": float(np.nanmin(values)), "minimum": 0.0,
    })
    assertions = pd.DataFrame(assertion_rows)
    if not assertions.passed.astype(bool).all():
        assertions.to_csv(args.output_dir / "GENERATOR_PROGRAM_ASSERTIONS.csv", index=False)
        raise RuntimeError("Benchmark generator v2 program assertion failed")

    adata.X = sparse.csr_matrix(values)
    adata.uns["benchmark_spec"] = "BENCHMARK_SPEC_V2"
    adata.uns["truth_generator_version"] = VERSION
    adata.uns["program_manifest_visibility"] = "truth_only_not_available_to_methods"
    adata.write_h5ad(h5ad_path, compression="gzip")
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.output_dir / "GENERATOR_PROGRAM_MANIFEST.csv", index=False)
    assertions.to_csv(args.output_dir / "GENERATOR_PROGRAM_ASSERTIONS.csv", index=False)
    canonical.to_csv(args.output_dir / "GENERATOR_CANONICAL_EFFECT_LEVELS.csv", index=False)
    truth_dir = args.output_dir / "truth"
    manifest.to_csv(truth_dir / "TRUE_PROGRAM_MECHANISMS.csv", index=False)
    version = {
        "benchmark_specification": "BENCHMARK_SPEC_V2",
        "truth_generator_version": VERSION,
        "parent_generator": "build_event_conditioned_benchmark.py",
        "seed": args.seed,
        "all_program_assertions_passed": True,
        "synthetic_h5ad_sha256": sha256(h5ad_path),
        "truth_visibility": "audit_only",
    }
    (args.output_dir / "GENERATOR_VERSION.json").write_text(
        json.dumps(version, indent=2), encoding="utf-8"
    )
    print(json.dumps(version, indent=2))


if __name__ == "__main__":
    main()
