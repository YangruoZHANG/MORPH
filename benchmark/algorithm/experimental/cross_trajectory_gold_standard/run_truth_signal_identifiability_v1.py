#!/usr/bin/env python3
"""Truth-fed signal-identifiability audit for paired frozen benchmark settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression


DEFAULT_SETTINGS = ("CORE_19", "CORE_05", "CORE_06")


def parse_gene_range(value: str) -> list[str]:
    value = str(value)
    if ";" in value:
        return [item.strip() for item in value.split(";") if item.strip()]
    if "-" in value:
        left, right = value.split("-", 1)
        prefix = "".join(ch for ch in left if not ch.isdigit())
        start = int("".join(ch for ch in left if ch.isdigit()))
        stop = int("".join(ch for ch in right if ch.isdigit()))
        return [f"{prefix}{index}" for index in range(start, stop + 1)]
    return [value]


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(np.std(values, ddof=0))
    return (values - np.mean(values)) / scale if scale > 1e-12 else np.zeros_like(values)


def residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    covariates = np.asarray(covariates, dtype=float)
    scales = np.std(covariates, axis=0, ddof=0)
    scaled = (covariates - np.mean(covariates, axis=0)) / np.where(scales > 1e-12, scales, 1.0)
    design = np.column_stack([np.ones(len(values)), scaled])
    return np.asarray(values, dtype=float) - design @ (np.linalg.pinv(design) @ values)


def standardized_shift(response: np.ndarray, exposed: np.ndarray) -> float:
    a, b = response[exposed], response[~exposed]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 1e-12 else 0.0


def spatial_blocks(coords: np.ndarray, bins: int = 3) -> np.ndarray:
    labels = []
    for axis in range(coords.shape[1]):
        ranked = pd.Series(coords[:, axis]).rank(method="first")
        labels.append(pd.qcut(ranked, q=bins, labels=False, duplicates="drop").to_numpy(int))
    return labels[0] * bins + labels[1]


def permute_within_blocks(values: np.ndarray, blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = np.asarray(values).copy()
    for block in np.unique(blocks):
        positions = np.flatnonzero(blocks == block)
        result[positions] = values[rng.permutation(positions)]
    return result


def summarize_statistic(contact: np.ndarray, response: np.ndarray, branch: np.ndarray, covariates: np.ndarray, quantile: float) -> dict[str, float]:
    exposed = contact >= np.quantile(contact, quantile)
    contact_residual = residualize(contact, covariates)
    response_residual = residualize(response, covariates)
    adjusted_correlation = float(np.corrcoef(contact_residual, response_residual)[0, 1])
    raw_mi = float(mutual_info_regression(contact.reshape(-1, 1), response, random_state=7112101)[0])
    residual_mi = float(mutual_info_regression(contact_residual.reshape(-1, 1), response_residual, random_state=7112101)[0])
    return {
        "n_eligible": int(len(contact)),
        "n_exposed": int(exposed.sum()),
        "mean_contact_exposed": float(np.mean(contact[exposed])),
        "mean_contact_unexposed": float(np.mean(contact[~exposed])),
        "program_mean_shift": float(np.mean(response[exposed]) - np.mean(response[~exposed])),
        "program_standardized_shift": standardized_shift(response, exposed),
        "branch_choice_shift": float(np.mean(branch[exposed]) - np.mean(branch[~exposed])),
        "raw_spearman": float(spearmanr(contact, response).statistic),
        "environment_adjusted_correlation": adjusted_correlation,
        "raw_mutual_information": raw_mi,
        "residual_mutual_information": residual_mi,
    }


def bootstrap_intervals(contact: np.ndarray, response: np.ndarray, branch: np.ndarray, covariates: np.ndarray, quantile: float, replicates: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    exposed = contact >= np.quantile(contact, quantile)
    groups = [np.flatnonzero(exposed), np.flatnonzero(~exposed)]
    metrics = {name: [] for name in ("program_mean_shift", "program_standardized_shift", "branch_choice_shift", "environment_adjusted_correlation")}
    for _ in range(replicates):
        sample = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        sample_exposed = exposed[sample]
        sample_contact = contact[sample]
        sample_response = response[sample]
        sample_branch = branch[sample]
        sample_covariates = covariates[sample]
        metrics["program_mean_shift"].append(float(
            np.mean(sample_response[sample_exposed]) - np.mean(sample_response[~sample_exposed])
        ))
        metrics["program_standardized_shift"].append(standardized_shift(sample_response, sample_exposed))
        metrics["branch_choice_shift"].append(float(
            np.mean(sample_branch[sample_exposed]) - np.mean(sample_branch[~sample_exposed])
        ))
        metrics["environment_adjusted_correlation"].append(float(np.corrcoef(
            residualize(sample_contact, sample_covariates),
            residualize(sample_response, sample_covariates),
        )[0, 1]))
    intervals: dict[str, float] = {}
    for name, values in metrics.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        intervals[f"{name}_ci_low"] = float(np.quantile(finite, 0.025))
        intervals[f"{name}_ci_high"] = float(np.quantile(finite, 0.975))
    return intervals


def permutation_pvalue(contact: np.ndarray, response: np.ndarray, covariates: np.ndarray, coords: np.ndarray, permutations: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    blocks = spatial_blocks(coords)
    observed = abs(float(np.corrcoef(residualize(contact, covariates), residualize(response, covariates))[0, 1]))
    null = []
    for _ in range(permutations):
        permuted = permute_within_blocks(contact, blocks, rng)
        null.append(abs(float(np.corrcoef(residualize(permuted, covariates), residualize(response, covariates))[0, 1])))
    return float((1 + np.sum(np.asarray(null) >= observed)) / (permutations + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--settings", nargs="+", default=list(DEFAULT_SETTINGS))
    parser.add_argument("--positive-template", default="CORE_05")
    parser.add_argument("--replicate-dir", default="replicate_01_seed_7112101")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=7112101)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    template_dir = args.protocol_run_root / args.positive_template / args.replicate_dir
    template = pd.read_csv(template_dir / "event" / "truth" / "TRUE_EVENTS.csv").iloc[0]
    receiver_genes = parse_gene_range(template.receiver_program_genes)
    source_genes = parse_gene_range(template.source_program_genes)
    quantile = float(template.exposure_quantile)
    rows = []
    source_program_rows = []

    for setting_index, setting in enumerate(args.settings):
        run_dir = args.protocol_run_root / setting / args.replicate_dir
        adata = ad.read_h5ad(run_dir / "event" / "synthetic.h5ad")
        contacts = pd.read_csv(run_dir / "event" / "truth" / "TRUE_SPATIAL_CONTACTS.csv")
        contacts = contacts.set_index("receiver_cell_id")["contact_score"]
        obs = adata.obs
        eligible = (
            obs["root_id"].astype(str).eq(str(template.receiver_root))
            & obs["truth_pseudotime"].astype(float).between(float(template.receiver_tau_start), float(template.receiver_tau_end), inclusive="both")
            & obs["truth_branch_T0"].astype(str).eq(str(template.decision_from))
        )
        cells = obs.index[eligible]
        contact = contacts.reindex(cells).to_numpy(float)
        matrix = adata[cells, receiver_genes].X
        matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        response = np.log1p(matrix).mean(axis=1)
        branch = obs.loc[cells, "truth_branch_T1"].astype(str).eq(str(template.decision_to)).to_numpy(float)
        covariates = obs.loc[cells, ["truth_pseudotime", "spatial_x", "spatial_y", "confounded_exposure"]].astype(float).to_numpy()
        coords = obs.loc[cells, ["spatial_x", "spatial_y"]].astype(float).to_numpy()
        config = json.loads((run_dir / "SCENARIO_CONFIG.json").read_text(encoding="utf-8"))
        values = summarize_statistic(contact, response, branch, covariates, quantile)
        values.update(bootstrap_intervals(contact, response, branch, covariates, quantile, args.bootstrap, args.seed + setting_index))
        values["adjusted_correlation_spatial_permutation_p"] = permutation_pvalue(
            contact, response, covariates, coords, args.permutations, args.seed + setting_index
        )
        rows.append({
            "setting_id": setting,
            "effect_level": "null" if float(config["effect_strength"]) == 0 else str(config["effect_level"]),
            "effect_strength": float(config["effect_strength"]),
            "source_root": str(template.source_root),
            "receiver_root": str(template.receiver_root),
            "receiver_branch_T0": str(template.decision_from),
            "receiver_branch_T1": str(template.decision_to),
            "receiver_tau_start": float(template.receiver_tau_start),
            "receiver_tau_end": float(template.receiver_tau_end),
            "exposure_quantile": quantile,
            "receiver_program_genes": ";".join(receiver_genes),
            **values,
            "truth_used": True,
        })

        source_mask = (
            obs["root_id"].astype(str).eq(str(template.source_root))
            & obs["truth_pseudotime"].astype(float).between(float(template.source_tau_start), float(template.source_tau_end), inclusive="both")
        )
        source_matrix = adata[obs.index[source_mask], source_genes].X
        source_matrix = source_matrix.toarray() if hasattr(source_matrix, "toarray") else np.asarray(source_matrix)
        source_program_rows.append({
            "setting_id": setting,
            "effect_strength": float(config["effect_strength"]),
            "n_source_window_cells": int(source_mask.sum()),
            "source_program_mean_log1p": float(np.log1p(source_matrix).mean()),
            "source_program_genes": ";".join(source_genes),
            "truth_used": True,
        })

    result = pd.DataFrame(rows).sort_values("effect_strength")
    result.to_csv(args.output_dir / "TRUTH_SIGNAL_IDENTIFIABILITY.csv", index=False)
    source_result = pd.DataFrame(source_program_rows).sort_values("effect_strength")
    reference = float(source_result.iloc[0]["source_program_mean_log1p"])
    source_result["delta_from_null"] = source_result["source_program_mean_log1p"] - reference
    source_result.to_csv(args.output_dir / "DECLARED_SOURCE_PROGRAM_AUDIT.csv", index=False)

    ordered = result.sort_values("effect_strength")
    monotonic = {}
    for metric in ("program_mean_shift", "program_standardized_shift", "environment_adjusted_correlation", "residual_mutual_information"):
        values = ordered[metric].to_numpy(float)
        monotonic[metric] = bool(np.all(np.diff(values) > 0))
    decision = {
        "protocol": "ORACLE_DECOMPOSITION_SPEC_V1",
        "truth_used": True,
        "paired_settings": args.settings,
        "all_primary_signal_metrics_strictly_monotonic": bool(all(monotonic.values())),
        "metric_monotonicity": monotonic,
        "source_program_declared_but_condition_invariant": bool(np.allclose(source_result["delta_from_null"], 0.0)),
        "interpretation": "diagnostic identifiability only; not fitted-method performance and not a causal estimate",
    }
    (args.output_dir / "TRUTH_SIGNAL_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
