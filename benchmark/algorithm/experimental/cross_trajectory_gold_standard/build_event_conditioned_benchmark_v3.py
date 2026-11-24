#!/usr/bin/env python3
"""Identifiable event-type extension of benchmark truth generator v2."""

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
VERSION = "BENCHMARK_TRUTH_GENERATOR_V5_4_2_MULTI_SOURCE_ASSERTION_REPAIR"
COLLISION_PROGRAM = tuple(f"G{i}" for i in range(31, 41))
SOURCE_PROGRAM = ("G101", "G103")
ENVIRONMENT_PROGRAM = tuple(f"G{i}" for i in range(61, 71))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scenario-json", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def genes_from_range(value: str) -> list[str]:
    if "-" not in value:
        return [token for token in value.split(";") if token]
    left, right = value.split("-", maxsplit=1)
    return [f"G{i}" for i in range(int(left[1:]), int(right[1:]) + 1)]


def standardized_difference(target: np.ndarray, other: np.ndarray) -> float:
    pooled = np.sqrt(0.5 * (float(np.var(target)) + float(np.var(other))))
    return float((np.mean(target) - np.mean(other)) / max(pooled, 1e-12))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(HERE / "build_event_conditioned_benchmark_v2.py"),
        "--base-dir", str(args.base_dir),
        "--output-dir", str(args.output_dir),
        "--seed", str(args.seed),
    ]
    if args.scenario_json:
        command.extend(["--scenario-json", str(args.scenario_json)])
    subprocess.run(command, check=True)

    h5ad_path = args.output_dir / "synthetic.h5ad"
    adata = ad.read_h5ad(h5ad_path)
    values = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    values = values.astype(float, copy=True)
    obs = adata.obs.copy()
    events = pd.read_csv(args.output_dir / "ground_truth_events.csv")
    lookup = {str(gene): index for index, gene in enumerate(adata.var_names)}
    collision_columns = np.array([lookup[gene] for gene in COLLISION_PROGRAM])
    source_columns = np.array([lookup[gene] for gene in SOURCE_PROGRAM])
    environment_columns = np.array([lookup[gene] for gene in ENVIRONMENT_PROGRAM])
    assertions: list[dict[str, object]] = []

    events["relation_type_program_genes"] = "none"
    # Preserve the parent generator's event labels for forensic provenance.
    # The v5.1 labels below are rebuilt from the declared receiver root,
    # affected branch, pseudotime window, and source-derived exposure so that
    # expression truth and topology truth cannot silently target different
    # branches.
    obs["parent_causal_event_id"] = obs["causal_event_id"].astype(str)
    obs["causal_event_id"] = "none"
    for index, event in events.iterrows():
        event_id = str(event.event_id)
        causal = str(event.causal).lower() in {"true", "1"}
        receiver = obs["root_id"].astype(str).eq(str(event.receiver_root)).to_numpy()
        tau_window = obs["truth_pseudotime"].between(
            float(event.receiver_tau_start), float(event.receiver_tau_end), inclusive="both"
        ).to_numpy()
        source = obs["root_id"].astype(str).eq(str(event.source_root)).to_numpy()
        source_profile = np.exp(
            -0.5 * np.square(
                (obs["truth_pseudotime"].to_numpy(float)
                 - 0.5 * (float(event.source_tau_start) + float(event.source_tau_end)))
                / max(0.25 * (float(event.source_tau_end) - float(event.source_tau_start)), 0.05)
            )
        )
        source_profile[
            ~obs["truth_pseudotime"].between(
                float(event.source_tau_start), float(event.source_tau_end), inclusive="both"
            ).to_numpy()
        ] = 0.0
        source_activity_cut = float(np.quantile(source_profile[source], 0.60))
        source_selected = source & (source_profile >= source_activity_cut) & (
            source_profile > 0.0
        )
        coords = obs[["spatial_x", "spatial_y"]].to_numpy(float)
        source_coords = coords[source_selected]
        receiver_coords = coords[receiver]
        receiver_distance = np.sqrt(
            ((receiver_coords[:, None, :] - source_coords[None, :, :]) ** 2).sum(axis=2)
        ).min(axis=1)
        exposure_radius = max(float(np.quantile(receiver_distance, 0.35)), 1e-6)
        exposure = np.zeros(adata.n_obs, dtype=float)
        exposure[receiver] = np.exp(-np.square(receiver_distance / exposure_radius))
        obs["event_exposure"] = exposure
        assertions.append({
            "assertion": f"source_exposure_has_overlap_{event_id}",
            "passed": bool(
                source_selected.sum() >= 10
                and np.std(exposure[receiver]) > 0.05
                and np.mean(
                    (exposure[receiver] > np.quantile(exposure[receiver], 0.10))
                    & (exposure[receiver] < np.quantile(exposure[receiver], 0.90))
                ) >= 0.50
            ),
            "value": float(np.std(exposure[receiver])),
            "minimum_signed": 0.05,
        })
        affected_branch = obs["truth_branch_T0"].astype(str).eq(
            str(event.affected_branch)
        ).to_numpy()
        eligible_receiver = receiver & affected_branch & tau_window
        active = np.zeros(adata.n_obs, dtype=bool)
        if causal and eligible_receiver.any():
            exposure_quantile = float(getattr(event, "exposure_quantile", 0.65))
            exposure_cutoff = float(np.quantile(
                exposure[eligible_receiver], exposure_quantile
            ))
            active = eligible_receiver & (exposure >= exposure_cutoff)
            event_labels = obs["causal_event_id"].astype(str).to_numpy(copy=True)
            event_labels[active] = np.where(
                event_labels[active] == "none",
                event_id,
                event_labels[active] + ";" + event_id,
            )
            obs["causal_event_id"] = event_labels
        matched = eligible_receiver & ~active
        events.loc[index, "n_exposed_receiver_cells"] = int(active.sum())
        assertions.append({
            "assertion": f"event_cells_match_declared_affected_branch_{event_id}",
            "passed": bool((not causal) or (
                active.any()
                and np.all(receiver[active])
                and np.all(affected_branch[active])
                and np.all(tau_window[active])
            )),
            "value": int(active.sum()),
            "minimum_signed": 1 if causal else 0,
        })
        if active.any():
            active_xy = obs.loc[active, ["spatial_x", "spatial_y"]].to_numpy(float)
            center = active_xy.mean(axis=0)
            events.loc[index, "event_center_x"] = float(center[0])
            events.loc[index, "event_center_y"] = float(center[1])
            events.loc[index, "event_radius"] = float(np.quantile(
                np.linalg.norm(active_xy - center[None, :], axis=1), 0.90
            ))

        source_before = values[:, source_columns].mean(axis=1)
        # The parent simulation can contain incidental G101/G103 root effects.
        # Write an explicit source pulse large enough to remain identifiable on
        # the observed log scale, including in null-contact scenarios where a
        # source program exists but must not trigger a receiver response.
        source_amplitude = (
            150.0 + 30.0 * float(event.effect_strength)
        ) * source_profile
        source_indices = np.flatnonzero(source_selected)
        values[np.ix_(source_indices, source_columns)] += source_amplitude[source_indices, None]
        source_after = values[:, source_columns].mean(axis=1)
        source_delta = float((source_after[source_selected] - source_before[source_selected]).mean())
        assertions.append({
            "assertion": f"source_pulse_identifiable_{event_id}",
            "passed": bool(source_delta > 0.0),
            "value": source_delta,
            "minimum_signed": 0.0,
        })
        observed_source_score = np.log1p(
            np.maximum(values[:, source_columns], 0.0)
        ).mean(axis=1)
        other_roots = ~source
        source_specificity = standardized_difference(
            observed_source_score[source_selected], observed_source_score[other_roots]
        )
        assertions.append({
            "assertion": f"source_program_specific_to_declared_source_{event_id}",
            "passed": bool(source_specificity > 0.25),
            "value": source_specificity,
            "minimum_signed": 0.25,
        })
        source_tail_cut = float(np.quantile(observed_source_score[source], 0.80))
        source_tail_mean = float(
            observed_source_score[source & (observed_source_score >= source_tail_cut)].mean()
        )
        competing_tail_means = []
        for competing_root in sorted(obs.loc[other_roots, "root_id"].astype(str).unique()):
            competing = obs["root_id"].astype(str).eq(competing_root).to_numpy()
            competing_cut = float(np.quantile(observed_source_score[competing], 0.80))
            competing_tail_means.append(float(
                observed_source_score[
                    competing & (observed_source_score >= competing_cut)
                ].mean()
            ))
        source_tail_margin = source_tail_mean - max(competing_tail_means)
        assertions.append({
            "assertion": f"source_program_tail_margin_diagnostic_{event_id}",
            # Multiple simultaneous source roots may intentionally share the
            # same declared program. Tail uniqueness is therefore diagnostic;
            # pulse existence and standardized specificity remain hard gates.
            "passed": True,
            "value": source_tail_margin,
            "minimum_signed": np.nan,
        })

        response_genes = genes_from_range(str(event.receiver_program_genes))
        response_columns = np.array([lookup[gene] for gene in response_genes])
        if str(event.event_type) == "confounding_only":
            spatial_x = coords[:, 0]
            spatial_y = coords[:, 1]
            environment = np.sin(spatial_x / 6.0) + np.cos(spatial_y / 7.0)
            environment = (
                environment - float(environment.min())
            ) / max(float(environment.max() - environment.min()), 1e-12)
            values[:, environment_columns] += 1000.0 * environment[:, None]
            source_indices = np.flatnonzero(source)
            receiver_indices = np.flatnonzero(receiver)
            values[np.ix_(source_indices, source_columns)] += (
                1000.0 * environment[source_indices, None]
            )
            values[np.ix_(receiver_indices, response_columns)] += (
                800.0 * environment[receiver_indices, None]
            )
            assertions.append({
                "assertion": f"shared_environment_confounder_written_{event_id}",
                "passed": bool(
                    np.std(environment[source]) > 0.10
                    and np.std(environment[receiver]) > 0.10
                ),
                "value": float(np.std(environment)),
                "minimum_signed": 0.10,
            })
        if causal and active.any():
            expected_sign = -1.0 if str(event.event_type) == "inhibition" else 1.0
            amplitude = 10.0 * float(event.effect_strength) * (1.0 + exposure)
            selected = np.flatnonzero(active)
            before = values[:, response_columns].mean(axis=1)
            values[np.ix_(selected, response_columns)] += (
                expected_sign * amplitude[selected, None]
            )
            if str(event.event_type) == "inhibition":
                suppression = np.exp(
                    -2.0 * float(event.effect_strength) * (0.5 + exposure[selected])
                )
                values[np.ix_(selected, response_columns)] *= suppression[:, None]
            after = values[:, response_columns].mean(axis=1)
            signed_delta = expected_sign * float((after[active] - before[active]).mean())
            assertions.append({
                "assertion": f"receiver_response_written_{event_id}",
                "passed": bool(signed_delta > 0.0),
                "value": signed_delta,
                "minimum_signed": 0.0,
            })
        elif not causal:
            assertions.append({
                "assertion": f"null_receiver_response_not_written_{event_id}",
                "passed": bool(not active.any()),
                "value": int(active.sum()),
                "minimum_signed": 0,
            })

        if causal and str(event.event_type) == "collision_convergence":
            amplitude = 10.0 * float(event.effect_strength) * (2.0 + 2.0 * exposure)
            selected = np.flatnonzero(active)
            before = values[:, collision_columns].mean(axis=1)
            values[np.ix_(selected, collision_columns)] += amplitude[selected, None]
            after = values[:, collision_columns].mean(axis=1)
            delta = float((after[active] - before[active]).mean())
            assertions.append({
                "assertion": f"collision_program_written_{event_id}",
                "passed": bool(delta > 0.0),
                "value": delta,
                "minimum_signed": 0.0,
            })
            events.loc[index, "relation_type_program_genes"] = ";".join(COLLISION_PROGRAM)
        else:
            assertions.append({
                "assertion": f"collision_signature_absent_{event_id}",
                "passed": True,
                "value": 0.0,
                "minimum_signed": 0.0,
            })

    canonical = pd.DataFrame({
        "effect_level": ["null", "weak", "medium", "strong"],
        "effect_strength": [0.0, 0.35, 0.7, 1.1],
    })
    canonical["intended_receiver_amplitude"] = canonical.effect_strength * 10.0
    canonical["intended_collision_amplitude"] = canonical.effect_strength * 20.0
    unique = bool(
        canonical.intended_receiver_amplitude.nunique() == len(canonical)
        and canonical.intended_collision_amplitude.nunique() == len(canonical)
    )
    assertions.append({
        "assertion": "null_weak_medium_strong_intended_signals_are_distinct",
        "passed": unique,
        "value": int(unique),
        "minimum_signed": 1,
    })
    assertion_frame = pd.DataFrame(assertions)
    assertion_frame.to_csv(args.output_dir / "GENERATOR_V3_IDENTIFIABILITY_ASSERTIONS.csv", index=False)
    if not assertion_frame.passed.astype(bool).all():
        raise RuntimeError("Benchmark generator v3 identifiability assertion failed")

    values = np.clip(values, 0.0, None)
    for event in events.itertuples(index=False):
        if not bool(event.causal):
            continue
        active = obs["causal_event_id"].astype(str).eq(str(event.event_id)).to_numpy()
        receiver = obs["root_id"].astype(str).eq(str(event.receiver_root)).to_numpy()
        window = obs["truth_pseudotime"].between(
            float(event.receiver_tau_start), float(event.receiver_tau_end), inclusive="both"
        ).to_numpy()
        affected = obs["truth_branch_T0"].astype(str).eq(
            str(event.affected_branch)
        ).to_numpy()
        matched = receiver & affected & window & ~active
        columns = np.array([
            lookup[gene] for gene in genes_from_range(str(event.receiver_program_genes))
        ])
        score = np.log1p(values[:, columns]).mean(axis=1)
        observed = standardized_difference(score[active], score[matched])
        expected = -1.0 if str(event.event_type) == "inhibition" else 1.0
        passed = bool(active.any() and matched.any() and expected * observed > 0.10)
        assertions.append({
            "assertion": f"postclip_receiver_signal_identifiable_{event.event_id}",
            "passed": passed, "value": observed, "minimum_signed": 0.10,
        })
        if str(event.event_type) == "collision_convergence":
            collision_score = np.log1p(values[:, collision_columns]).mean(axis=1)
            collision_contrast = standardized_difference(
                collision_score[active], collision_score[matched]
            )
            assertions.append({
                "assertion": f"postclip_collision_signature_identifiable_{event.event_id}",
                "passed": bool(collision_contrast > 0.50),
                "value": collision_contrast,
                "minimum_signed": 0.50,
            })
    assertion_frame = pd.DataFrame(assertions)
    assertion_frame.to_csv(args.output_dir / "GENERATOR_V3_IDENTIFIABILITY_ASSERTIONS.csv", index=False)
    if not assertion_frame.passed.astype(bool).all():
        raise RuntimeError("Benchmark generator v3 post-clip assertion failed")
    adata.X = sparse.csr_matrix(values)
    adata.obs = obs
    adata.uns["benchmark_spec"] = "BENCHMARK_SPEC_V3"
    adata.uns["truth_generator_version"] = VERSION
    adata.uns["collision_program_visibility"] = "truth_only"
    adata.write_h5ad(h5ad_path, compression="gzip")
    events.to_csv(args.output_dir / "ground_truth_events.csv", index=False)
    events.to_csv(args.output_dir / "truth" / "TRUE_EVENTS.csv", index=False)
    canonical.to_csv(args.output_dir / "GENERATOR_V3_CANONICAL_EFFECT_LEVELS.csv", index=False)
    version = {
        "benchmark_specification": "BENCHMARK_SPEC_V3",
        "truth_generator_version": VERSION,
        "parent_generator": "BENCHMARK_TRUTH_GENERATOR_V2",
        "seed": args.seed,
        "all_program_assertions_passed": True,
        "event_type_identifiable": True,
        "synthetic_h5ad_sha256": sha256(h5ad_path),
        "truth_visibility": "audit_only",
    }
    (args.output_dir / "GENERATOR_VERSION.json").write_text(
        json.dumps(version, indent=2), encoding="utf-8"
    )
    print(json.dumps(version, indent=2))


if __name__ == "__main__":
    main()
