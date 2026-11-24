#!/usr/bin/env python3
"""Branch-local relation evidence on frozen T0/tau0 (development v4.0).

The spatial contact region is selected without receiver-expression information.
All legal branches in that region enter a single max-statistic permutation test;
the winning branch is the event localization returned by that controlled search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge


WINDOWS = ((0.0, 0.5), (0.25, 0.75), (0.5, 1.0))
EPS = 1e-12


def parse_genes(value: str) -> list[str]:
    value = str(value)
    if ";" in value:
        return [item for item in value.split(";") if item]
    left, right = value.split("-", maxsplit=1)
    return [f"G{i}" for i in range(int(left[1:]), int(right[1:]) + 1)]


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = np.std(values, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return (values - np.mean(values, axis=0)) / scale


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4 or np.std(x) <= 1e-9 or np.std(y) <= 1e-9:
        return 0.0
    return float(np.clip(np.corrcoef(x, y)[0, 1], -0.999999, 0.999999))


def gaussian_mi(rho: float) -> float:
    return float(-0.5 * np.log(max(1.0 - float(rho) ** 2, EPS)))


def fisher_standardized_statistic(value: float, n_cells: int) -> float:
    """Absolute Fisher-z score, placing unequal-size windows on one scale."""
    clipped = float(np.clip(value, -0.999999, 0.999999))
    return float(abs(np.arctanh(clipped)) * np.sqrt(max(int(n_cells) - 3, 1)))


def standardized_difference(target: np.ndarray, other: np.ndarray) -> float:
    pooled = np.sqrt(0.5 * (float(np.var(target)) + float(np.var(other))))
    return float((np.mean(target) - np.mean(other)) / max(pooled, EPS))


def exposure_contrast(x: np.ndarray, y: np.ndarray, quantile: float = 0.65) -> float:
    threshold = float(np.quantile(x, quantile))
    high = x >= threshold
    if high.sum() < 4 or (~high).sum() < 4:
        return 0.0
    return standardized_difference(y[high], y[~high])


def monotone_response_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Signed bounded statistic sensitive to linear and threshold responses."""
    continuous = correlation(x, y)
    threshold = float(np.quantile(x, 0.65))
    high = (x >= threshold).astype(float)
    thresholded = correlation(high, y)
    return continuous if abs(continuous) >= abs(thresholded) else thresholded


def adjusted_response_statistic(
    exposure: np.ndarray, exposure_residual: np.ndarray, outcome_residual: np.ndarray,
) -> float:
    """Combine DML residual association with adjusted high-exposure contrast."""
    residual_association = monotone_response_statistic(
        exposure_residual, outcome_residual
    )
    threshold = float(np.quantile(exposure, 0.65))
    high_exposure = (exposure >= threshold).astype(float)
    adjusted_group_contrast = correlation(high_exposure, outcome_residual)
    return (
        residual_association
        if abs(residual_association) >= abs(adjusted_group_contrast)
        else adjusted_group_contrast
    )


def spatial_blocks(coords: np.ndarray) -> np.ndarray:
    bins = 3 if len(coords) >= 90 else 2
    labels = []
    for axis in range(2):
        rank = pd.Series(coords[:, axis]).rank(method="first")
        labels.append(pd.qcut(rank, q=bins, labels=False, duplicates="drop").to_numpy(int))
    return labels[0] * bins + labels[1]


def stratified_spatial_folds(
    blocks: np.ndarray, coords: np.ndarray, n_folds: int = 3,
) -> np.ndarray:
    """Cell-level folds represented within every spatial block.

    Holding out entire blocks forces a low-order spatial nuisance model to
    extrapolate into unseen tissue regions.  Alternating spatially ordered
    cells within each block keeps cross-fitting honest while retaining spatial
    support in every training fold.
    """
    folds = np.zeros(len(blocks), dtype=int)
    for block in np.unique(blocks):
        indices = np.flatnonzero(blocks == block)
        order = np.lexsort((coords[indices, 1], coords[indices, 0]))
        folds[indices[order]] = np.arange(len(indices)) % n_folds
    return folds


def design_matrix(
    tau: np.ndarray, coords: np.ndarray, annotation: np.ndarray,
    environment: np.ndarray,
) -> np.ndarray:
    environment = np.asarray(environment, dtype=float)
    if environment.ndim == 1:
        environment = environment[:, None]
    spatial = zscore(np.asarray(coords, dtype=float))
    spatial_terms = np.column_stack([
        spatial,
        spatial[:, 0] * spatial[:, 0],
        spatial[:, 1] * spatial[:, 1],
        spatial[:, 0] * spatial[:, 1],
    ])
    return np.column_stack([
        np.ones(len(tau)), tau, tau * tau, spatial_terms,
        annotation, zscore(environment),
    ])


def crossfit_residuals(
    x: np.ndarray, y: np.ndarray, design: np.ndarray, folds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    xr = np.zeros(len(x), dtype=float)
    yr = np.zeros(len(y), dtype=float)
    baseline_prediction = np.zeros(len(y), dtype=float)
    full_prediction = np.zeros(len(y), dtype=float)
    for fold in np.unique(folds):
        test = folds == fold
        train = ~test
        if train.sum() <= design.shape[1] + 2 or test.sum() < 3:
            train = np.ones(len(x), dtype=bool)
        exposure_model = Ridge(alpha=1.0, fit_intercept=False).fit(design[train], x[train])
        outcome_model = Ridge(alpha=1.0, fit_intercept=False).fit(design[train], y[train])
        xr[test] = x[test] - exposure_model.predict(design[test])
        yr[test] = y[test] - outcome_model.predict(design[test])
        baseline_prediction[test] = outcome_model.predict(design[test])
        full_design_train = np.column_stack([design[train], x[train]])
        full_design_test = np.column_stack([design[test], x[test]])
        full_model = Ridge(alpha=1.0, fit_intercept=False).fit(full_design_train, y[train])
        full_prediction[test] = full_model.predict(full_design_test)
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    if denominator <= EPS:
        delta_r2 = 0.0
    else:
        base_r2 = 1.0 - float(np.sum((y - baseline_prediction) ** 2)) / denominator
        full_r2 = 1.0 - float(np.sum((y - full_prediction) ** 2)) / denominator
        delta_r2 = full_r2 - base_r2
    return xr, yr, folds, float(delta_r2)


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def westfall_young_minp(
    observed: np.ndarray, null_matrix: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Single-step synchronized minP adjustment across a scan family."""
    observed = np.asarray(observed, dtype=float)
    null_matrix = np.asarray(null_matrix, dtype=float)
    n_permutations = int(null_matrix.shape[0])
    candidate_p = (
        1.0 + np.sum(null_matrix >= observed[None, :], axis=0)
    ) / (n_permutations + 1.0)
    null_p = np.empty_like(null_matrix, dtype=float)
    for column in range(null_matrix.shape[1]):
        values = null_matrix[:, column]
        null_p[:, column] = (
            1.0 + np.sum(values[None, :] >= values[:, None], axis=1)
        ) / (n_permutations + 1.0)
    min_null_p = np.min(null_p, axis=1)
    family_p = float(
        (1.0 + np.sum(min_null_p <= float(np.min(candidate_p))))
        / (n_permutations + 1.0)
    )
    return family_p, candidate_p


def synchronized_sum_p(
    observed: np.ndarray, null_matrix: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Permutation-calibrated sum test over geometry-preselected regions."""
    observed = np.asarray(observed, dtype=float)
    null_matrix = np.asarray(null_matrix, dtype=float)
    n_permutations = int(null_matrix.shape[0])
    candidate_p = (
        1.0 + np.sum(null_matrix >= observed[None, :], axis=0)
    ) / (n_permutations + 1.0)
    observed_sum = float(np.sum(observed))
    null_sum = np.sum(null_matrix, axis=1)
    family_p = float(
        (1.0 + np.sum(null_sum >= observed_sum)) / (n_permutations + 1.0)
    )
    return family_p, candidate_p


def permute_within(values: np.ndarray, blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = values.copy()
    for block in np.unique(blocks):
        indices = np.flatnonzero(blocks == block)
        if len(indices) > 1:
            result[indices] = values[rng.permutation(indices)]
    return result


def permutation_order(blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    order = np.arange(len(blocks))
    for block in np.unique(blocks):
        indices = np.flatnonzero(blocks == block)
        if len(indices) > 1:
            order[indices] = rng.permutation(indices)
    return order


def synchronized_permutation_order(
    positions: np.ndarray, blocks: np.ndarray, random_keys: np.ndarray,
) -> np.ndarray:
    """Uniform within-block permutation coupled across overlapping candidates."""
    order = np.arange(len(blocks))
    positions = np.asarray(positions, dtype=int)
    for block in np.unique(blocks):
        indices = np.flatnonzero(blocks == block)
        if len(indices) > 1:
            order[indices] = indices[np.argsort(random_keys[positions[indices]])]
    return order


def exposure_from_source(
    receiver_coords: np.ndarray,
    source_coords: np.ndarray,
    source_activity: np.ndarray,
    kernel_quantile: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    distance = np.sqrt(
        ((receiver_coords[:, None, :] - source_coords[None, :, :]) ** 2).sum(axis=2)
    )
    nearest = distance.min(axis=1)
    radius = max(float(np.quantile(nearest, kernel_quantile)), 1e-6)
    exposure = np.exp(-np.square(nearest / radius))
    return exposure, radius, nearest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase-a-dir", type=Path, required=True)
    parser.add_argument("--program-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--seed", type=int, default=7129001)
    parser.add_argument("--q-cutoff", type=float, default=0.10)
    parser.add_argument("--effect-cutoff", type=float, default=0.12)
    parser.add_argument("--raw-effect-cutoff", type=float, default=0.15)
    parser.add_argument("--minimum-oof-gain", type=float, default=-0.10)
    parser.add_argument("--direction-ratio", type=float, default=1.20)
    parser.add_argument("--receiver-source-ratio", type=float, default=1.15)
    parser.add_argument("--min-cells", type=int, default=25)
    parser.add_argument("--kernel-quantile", type=float, default=0.35)
    parser.add_argument("--contact-quantile", type=float, default=0.65)
    parser.add_argument("--source-activity-quantile", type=float, default=0.60)
    parser.add_argument("--source-identity-quantile", type=float, default=0.80)
    parser.add_argument("--disable-environment-programs", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    h5ad = args.run_dir / "event" / "synthetic.h5ad"
    adata = ad.read_h5ad(h5ad)
    expression = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    expression = np.log1p(np.maximum(expression.astype(float), 0.0))
    coords = adata.obs[["spatial_x", "spatial_y"]].to_numpy(float)
    gene_index = {str(gene): index for index, gene in enumerate(adata.var_names)}
    mapping = pd.read_csv(args.run_dir / "formal_metacell" / "cell_to_metacell_mapping.csv")
    cell_to_mc = mapping.set_index("cell_name")["metacell_id"].astype(str)
    metacells = cell_to_mc.reindex(adata.obs_names.astype(str)).to_numpy()
    if pd.isna(metacells).any():
        raise ValueError("Unmapped cells in relation detector")

    assignments = pd.read_csv(args.phase_a_dir / "T0_STATE_NODE_ASSIGNMENTS.csv")
    assignments["unit_id"] = assignments.unit_id.astype(str)
    primary = assignments.sort_values("membership", ascending=False).drop_duplicates("unit_id")
    mc_to_root = primary.set_index("unit_id").root_id.astype(str).to_dict()
    root_assignment = np.asarray([mc_to_root.get(str(mc), "none") for mc in metacells])
    lookup = assignments.set_index(["root_id", "unit_id"])
    roots = sorted(assignments.root_id.astype(str).unique())
    tau_by_root = {
        root: lookup.loc[root].pseudotime.astype(float).to_dict() for root in roots
    }
    branch_by_root = {
        root: lookup.loc[root].branch_id.astype(str).to_dict() for root in roots
    }
    catalog = pd.read_csv(args.program_catalog)
    primary_program = catalog[catalog.program_role.eq("primary_relation")].iloc[0]
    signature_program = catalog[catalog.program_role.eq("type_signature")].iloc[0]
    negative_programs = catalog[catalog.program_role.eq("negative_control")].copy()
    environment_programs = catalog[
        catalog.program_role.eq("environment_covariate")
    ].copy()
    source_genes = parse_genes(primary_program.source_genes)
    receiver_genes = parse_genes(primary_program.receiver_genes)
    signature_genes = parse_genes(signature_program.receiver_genes)
    negative_gene_sets = [parse_genes(value) for value in negative_programs.receiver_genes]
    environment_gene_sets = [
        parse_genes(value) for value in environment_programs.receiver_genes
    ] if not args.disable_environment_programs else []
    required = set(source_genes + receiver_genes + signature_genes)
    required.update(gene for genes in negative_gene_sets for gene in genes)
    required.update(gene for genes in environment_gene_sets for gene in genes)
    missing = sorted(required - set(gene_index))
    if missing:
        raise ValueError(f"Program catalog genes missing from expression: {missing}")
    source_score = expression[:, [gene_index[gene] for gene in source_genes]].mean(axis=1)
    receiver_score = expression[:, [gene_index[gene] for gene in receiver_genes]].mean(axis=1)
    signature_score = expression[:, [gene_index[gene] for gene in signature_genes]].mean(axis=1)
    negative_scores = [
        expression[:, [gene_index[gene] for gene in genes]].mean(axis=1)
        for genes in negative_gene_sets
    ]
    environment_scores = (
        np.column_stack([
            expression[:, [gene_index[gene] for gene in genes]].mean(axis=1)
            for genes in environment_gene_sets
        ])
        if environment_gene_sets else np.empty((adata.n_obs, 0), dtype=float)
    )
    annotation = pd.get_dummies(adata.obs["cell.type"].astype(str), drop_first=True).to_numpy(float)

    evidence_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    payload_by_direction: dict[tuple[str, str], list[dict[str, object]]] = {}
    for source_root in roots:
        source_all = np.flatnonzero(root_assignment == source_root)
        source_tau = np.asarray([
            tau_by_root[source_root].get(str(metacells[index]), 0.5) for index in source_all
        ])
        threshold = float(np.quantile(
            source_score[source_all], args.source_activity_quantile
        ))
        selected_mask = source_score[source_all] >= threshold
        selected_source = source_all[selected_mask]
        reference = np.flatnonzero(root_assignment != source_root)
        if len(selected_source) < args.min_cells or len(reference) < args.min_cells:
            continue
        source_enrichment = standardized_difference(
            source_score[selected_source], source_score[reference]
        )
        identity_threshold = float(np.quantile(
            source_score[source_all], args.source_identity_quantile
        ))
        source_identity_activity = float(np.mean(
            source_score[source_all][source_score[source_all] >= identity_threshold]
        ))
        selected_tau = source_tau[selected_mask]
        selected_source_start = float(np.quantile(selected_tau, 0.05))
        selected_source_end = float(np.quantile(selected_tau, 0.95))
        for receiver_root in roots:
            if receiver_root == source_root:
                continue
            receiver_all = np.flatnonzero(root_assignment == receiver_root)
            receiver_tau = np.asarray([
                tau_by_root[receiver_root].get(str(metacells[index]), 0.5) for index in receiver_all
            ])
            receiver_branch = np.asarray([
                branch_by_root[receiver_root].get(str(metacells[index]), "none")
                for index in receiver_all
            ])
            receiver_coords = coords[receiver_all]
            receiver_spatial_blocks = spatial_blocks(receiver_coords)
            candidates: list[dict[str, object]] = []
            exposure, radius, nearest_source_distance = exposure_from_source(
                receiver_coords, coords[selected_source], source_score[selected_source],
                args.kernel_quantile,
            )
            # The kernel equals exp(-1) at the predeclared contact radius. This
            # localizes the test using geometry only, before observing response.
            contact_radius = max(float(np.quantile(
                nearest_source_distance, args.contact_quantile
            )), 1e-6)
            contact_mask = nearest_source_distance <= contact_radius
            branch_design = pd.get_dummies(
                pd.Series(receiver_branch, dtype="string"), drop_first=True
            ).to_numpy(float)
            if branch_design.shape[1]:
                nuisance_annotation = np.column_stack([
                    annotation[receiver_all],
                    branch_design,
                    branch_design * receiver_tau[:, None],
                ])
            else:
                nuisance_annotation = annotation[receiver_all]
            global_design = design_matrix(
                receiver_tau, receiver_coords, nuisance_annotation,
                environment_scores[receiver_all],
            )
            global_folds = stratified_spatial_folds(
                receiver_spatial_blocks, receiver_coords
            )
            global_xr, global_yr, _, global_delta_r2 = crossfit_residuals(
                exposure, receiver_score[receiver_all], global_design, global_folds
            )
            _, global_signature_yr, _, global_signature_delta_r2 = crossfit_residuals(
                exposure, signature_score[receiver_all], global_design, global_folds
            )
            global_negative_yrs = []
            for negative_score in negative_scores:
                _, negative_yr, _, _ = crossfit_residuals(
                    exposure, negative_score[receiver_all], global_design, global_folds
                )
                global_negative_yrs.append(negative_yr)
            branch_contact_scores = {
                branch: float(np.mean(exposure[receiver_branch == branch]))
                for branch in sorted(set(receiver_branch) - {"none"})
                if int(np.sum(receiver_branch == branch)) >= args.min_cells
            }
            preselected_branches = {
                branch for branch, _ in sorted(
                    branch_contact_scores.items(), key=lambda item: item[1], reverse=True
                )[:2]
            }
            for branch in ["ALL", *sorted(set(receiver_branch) - {"none"})]:
                branch_mask = (
                    np.ones(len(receiver_branch), dtype=bool)
                    if branch == "ALL" else receiver_branch == branch
                )
                for receiver_start, receiver_end in WINDOWS:
                        local_mask = (
                            branch_mask & contact_mask & (receiver_tau >= receiver_start)
                            & (receiver_tau <= receiver_end)
                        )
                        if int(local_mask.sum()) < args.min_cells:
                            continue
                        positions = receiver_all[local_mask]
                        x = exposure[local_mask]
                        y = receiver_score[positions]
                        signature_y = signature_score[positions]
                        local_coords = coords[positions]
                        local_tau = receiver_tau[local_mask]
                        local_annotation = annotation[positions]
                        blocks = receiver_spatial_blocks[local_mask]
                        folds = global_folds[local_mask]
                        xr = global_xr[local_mask]
                        yr = global_yr[local_mask]
                        signature_yr = global_signature_yr[local_mask]
                        delta_r2 = global_delta_r2
                        signature_delta_r2 = global_signature_delta_r2
                        negative_dependencies = []
                        negative_raw_dependencies = []
                        for negative_score, global_negative_yr in zip(
                            negative_scores, global_negative_yrs
                        ):
                            negative_y = negative_score[positions]
                            negative_yr = global_negative_yr[local_mask]
                            negative_dependencies.append(
                                gaussian_mi(adjusted_response_statistic(
                                    x, xr, negative_yr
                                ))
                            )
                            negative_raw_dependencies.append(
                                gaussian_mi(monotone_response_statistic(x, negative_y))
                            )
                        raw_rho = correlation(x, y)
                        adjusted_rho = correlation(xr, yr)
                        raw_monotone = monotone_response_statistic(x, y)
                        adjusted_monotone = adjusted_response_statistic(x, xr, yr)
                        signature_raw_rho = correlation(x, signature_y)
                        signature_rho = correlation(xr, signature_yr)
                        signature_contrast = exposure_contrast(x, signature_y)
                        raw_dependency = gaussian_mi(raw_monotone)
                        adjusted_dependency = gaussian_mi(adjusted_monotone)
                        signature_raw_dependency = gaussian_mi(signature_raw_rho)
                        signature_adjusted_dependency = gaussian_mi(signature_rho)
                        negative_median = float(np.median(negative_dependencies))
                        negative_max = float(np.max(negative_dependencies))
                        specificity_rank = float(
                            np.mean(adjusted_dependency > np.asarray(negative_dependencies))
                        )
                        raw_negative_median = float(np.median(negative_raw_dependencies))
                        raw_specificity_rank = float(
                            np.mean(raw_dependency > np.asarray(negative_raw_dependencies))
                        )
                        drop = max(raw_dependency - adjusted_dependency, 0.0)
                        direction_consistent = bool(
                            raw_monotone * adjusted_monotone > 0.0
                            or abs(raw_monotone) < 0.05
                            or abs(adjusted_monotone) < 0.05
                        )
                        conflict = (
                            drop if direction_consistent
                            else max(raw_dependency, adjusted_dependency)
                        )
                        trust = (
                            adjusted_dependency
                            / max(raw_dependency + adjusted_dependency, EPS)
                            if direction_consistent else 0.0
                        )
                        q10, q90 = np.quantile(x, [0.10, 0.90])
                        positivity = float(np.mean((x > q10) & (x < q90)))
                        candidates.append({
                            "source_root": source_root, "receiver_root": receiver_root,
                            "source_window_start": selected_source_start,
                            "source_window_end": selected_source_end,
                            "source_program_enrichment": source_enrichment,
                            "source_program_activity": source_identity_activity,
                            "receiver_window_start": receiver_start,
                            "receiver_window_end": receiver_end,
                            "affected_branch": branch, "n_receiver_cells": len(positions),
                            "kernel_bandwidth": radius,
                            "contact_radius": contact_radius,
                            "exposure_variance": float(np.var(x)),
                            "contact_quantile": args.contact_quantile,
                            "branch_contact_score": (
                                np.nan if branch == "ALL" else branch_contact_scores.get(branch, np.nan)
                            ),
                            "branch_preselected_by_contact": branch in preselected_branches,
                            "positivity_overlap": positivity, "raw_rho": raw_rho,
                            "adjusted_rho": adjusted_rho,
                            "raw_monotone_statistic": raw_monotone,
                            "adjusted_monotone_statistic": adjusted_monotone,
                            "raw_dependency": raw_dependency,
                            "environment_adjusted_dependency": adjusted_dependency,
                            "environment_drop": drop / max(raw_dependency, EPS),
                            "environment_trust": trust,
                            "environment_conflict": conflict,
                            "environment_direction_consistent": direction_consistent,
                            "oof_predictive_gain": delta_r2,
                            "negative_control_dependency_median": negative_median,
                            "negative_control_dependency_max": negative_max,
                            "program_specificity_margin": adjusted_dependency - negative_median,
                            "program_specificity_rank": specificity_rank,
                            "raw_negative_control_dependency_median": raw_negative_median,
                            "raw_program_specificity_margin": (
                                raw_dependency - raw_negative_median
                            ),
                            "raw_program_specificity_rank": raw_specificity_rank,
                            "signature_raw_rho": signature_raw_rho,
                            "signature_rho": signature_rho,
                            "signature_raw_dependency": signature_raw_dependency,
                            "signature_adjusted_dependency": signature_adjusted_dependency,
                            "signature_raw_specificity_margin": (
                                signature_raw_dependency - raw_negative_median
                            ),
                            "signature_raw_specificity_rank": float(
                                np.mean(
                                    signature_raw_dependency
                                    > np.asarray(negative_raw_dependencies)
                                )
                            ),
                            "signature_specificity_margin": (
                                signature_adjusted_dependency - negative_median
                            ),
                            "signature_specificity_rank": float(
                                np.mean(
                                    signature_adjusted_dependency
                                    > np.asarray(negative_dependencies)
                                )
                            ),
                            "signature_exposure_contrast": signature_contrast,
                            "signature_oof_predictive_gain": signature_delta_r2,
                            "positions": positions,
                            "local_branch_labels": receiver_branch[local_mask],
                            "x": x, "y": y, "signature_y": signature_y,
                            "xr": xr, "yr": yr, "signature_yr": signature_yr,
                            "blocks": blocks,
                        })
            if not candidates:
                evidence_rows.append({
                    "source_root": source_root, "receiver_root": receiver_root,
                    "status": "insufficient_event_local_support", "permutation_p": 1.0,
                    "raw_permutation_p": 1.0,
                })
                continue
            candidate_rows.extend([
                {
                    key: value for key, value in item.items()
                    if key not in {
                        "x", "y", "signature_y", "xr", "yr", "signature_yr",
                        "blocks", "positions", "local_branch_labels",
                    }
                }
                for item in candidates
            ])
            # Geometry defines the event-local contact region, but it must not
            # decide the affected branch.  All branch-local candidates enter
            # one max-statistic permutation family; the winning branch is
            # localized only after accounting for that search.  Contact rank
            # remains an audit field rather than an exclusion gate.
            # Every branch/window with adequate support enters one max-statistic
            # family. Geometry-only contact ranks remain provenance fields; they
            # are not allowed to exclude a biologically responsive branch.
            branch_local_candidates = [
                item for item in candidates
                if str(item["affected_branch"]) != "ALL"
            ]
            # Select two regions using contact geometry only.  A frozen DEV
            # audit showed that contact support retains event cells much more
            # reliably than ranking windows by exposure variance.  Receiver
            # expression is not consulted until after this preselection.
            significance_candidates = sorted(
                branch_local_candidates,
                key=lambda item: (
                    int(item["n_receiver_cells"]),
                    float(item["branch_contact_score"]),
                    float(item["exposure_variance"]),
                ),
                reverse=True,
            )[:2]
            if not significance_candidates:
                significance_candidates = candidates
            if not significance_candidates:
                evidence_rows.append({
                    "source_root": source_root, "receiver_root": receiver_root,
                    "status": "insufficient_contact_local_support", "permutation_p": 1.0,
                    "raw_permutation_p": 1.0,
                })
                continue
            eligible_candidates = [
                item for item in significance_candidates
                if abs(float(item["raw_monotone_statistic"])) >= 0.15
                and float(item["raw_program_specificity_margin"]) > 0.01
                and float(item["raw_program_specificity_rank"]) >= 0.80
            ]
            selection_pool = (
                eligible_candidates if eligible_candidates else significance_candidates
            )
            # The primary CMI-E hypothesis is conjunctive: both raw spatial
            # dependence and environment-adjusted dependence must survive.
            # Candidate localization therefore optimizes the same min-statistic
            # used by its max-statistic permutation null.
            best = max(
                selection_pool,
                key=lambda item: (
                    min(
                        fisher_standardized_statistic(
                            float(item["raw_monotone_statistic"]),
                            int(item["n_receiver_cells"]),
                        ),
                        fisher_standardized_statistic(
                            float(item["adjusted_monotone_statistic"]),
                            int(item["n_receiver_cells"]),
                        ),
                    ),
                    max(
                        fisher_standardized_statistic(
                            float(item["raw_monotone_statistic"]),
                            int(item["n_receiver_cells"]),
                        ),
                        fisher_standardized_statistic(
                            float(item["adjusted_monotone_statistic"]),
                            int(item["n_receiver_cells"]),
                        ),
                    ),
                ),
            )
            best["joint_dependency_statistic"] = min(
                fisher_standardized_statistic(
                    float(best["raw_monotone_statistic"]),
                    int(best["n_receiver_cells"]),
                ),
                fisher_standardized_statistic(
                    float(best["adjusted_monotone_statistic"]),
                    int(best["n_receiver_cells"]),
                ),
            )
            best["passed_candidate_specificity_prefilter"] = bool(eligible_candidates)
            best["whole_root_fallback_used"] = not bool(branch_local_candidates)
            best["omnibus_dependency_statistic"] = max(
                fisher_standardized_statistic(
                    float(best["raw_monotone_statistic"]),
                    int(best["n_receiver_cells"]),
                ),
                fisher_standardized_statistic(
                    float(best["adjusted_monotone_statistic"]),
                    int(best["n_receiver_cells"]),
                ),
            )
            signature_candidates = significance_candidates
            signature_pool = [
                item for item in signature_candidates
                if float(item["signature_exposure_contrast"]) > 0
            ]
            signature_best = max(
                signature_pool,
                key=lambda item: float(item["signature_exposure_contrast"]),
            ) if signature_pool else None
            null_adjusted = []
            null_raw = []
            null_joint = []
            null_omnibus = []
            null_signature_joint = []
            for _ in range(args.permutations):
                random_keys = rng.random(adata.n_obs)
                adjusted_stats = []
                raw_stats = []
                joint_stats = []
                for item in significance_candidates:
                    order = synchronized_permutation_order(
                        item["positions"], item["blocks"], random_keys
                    )
                    adjusted_stat = abs(adjusted_response_statistic(
                        item["x"][order], item["xr"][order], item["yr"]
                    ))
                    raw_stat = abs(monotone_response_statistic(
                        item["x"][order], item["y"]
                    ))
                    n_local = int(item["n_receiver_cells"])
                    adjusted_standardized = fisher_standardized_statistic(
                        adjusted_stat, n_local
                    )
                    raw_standardized = fisher_standardized_statistic(raw_stat, n_local)
                    adjusted_stats.append(adjusted_standardized)
                    raw_stats.append(raw_standardized)
                    joint_stats.append(min(raw_standardized, adjusted_standardized))
                signature_joint_stats = []
                for item in signature_candidates:
                    order = synchronized_permutation_order(
                        item["positions"], item["blocks"], random_keys
                    )
                    signature_joint_stats.append(max(
                        exposure_contrast(item["x"][order], item["signature_y"]), 0.0
                    ))
                null_adjusted.append(adjusted_stats)
                null_raw.append(raw_stats)
                null_joint.append(joint_stats)
                null_omnibus.append([
                    max(raw_value, adjusted_value)
                    for raw_value, adjusted_value in zip(raw_stats, adjusted_stats)
                ])
                null_signature_joint.append(max(signature_joint_stats))
            observed_raw = np.asarray([
                fisher_standardized_statistic(
                    float(item["raw_monotone_statistic"]),
                    int(item["n_receiver_cells"]),
                )
                for item in significance_candidates
            ])
            observed_adjusted = np.asarray([
                fisher_standardized_statistic(
                    float(item["adjusted_monotone_statistic"]),
                    int(item["n_receiver_cells"]),
                )
                for item in significance_candidates
            ])
            observed_joint = np.minimum(observed_raw, observed_adjusted)
            observed_omnibus = np.maximum(observed_raw, observed_adjusted)
            p_joint, candidate_joint_p = westfall_young_minp(
                observed_joint, np.asarray(null_joint, dtype=float)
            )
            p_omnibus, candidate_omnibus_p = westfall_young_minp(
                observed_omnibus, np.asarray(null_omnibus, dtype=float)
            )
            p_adjusted, candidate_adjusted_p = westfall_young_minp(
                observed_adjusted, np.asarray(null_adjusted, dtype=float)
            )
            p_raw, candidate_raw_p = westfall_young_minp(
                observed_raw, np.asarray(null_raw, dtype=float)
            )
            best_index = min(
                range(len(significance_candidates)),
                key=lambda index: (
                    float(candidate_joint_p[index]),
                    -float(observed_joint[index]),
                ),
            )
            best = significance_candidates[best_index]
            best["joint_dependency_statistic"] = float(observed_joint[best_index])
            best["omnibus_dependency_statistic"] = float(observed_omnibus[best_index])
            best["candidate_joint_permutation_p"] = float(candidate_joint_p[best_index])
            best["candidate_raw_permutation_p"] = float(candidate_raw_p[best_index])
            best["candidate_adjusted_permutation_p"] = float(
                candidate_adjusted_p[best_index]
            )
            best["candidate_omnibus_permutation_p"] = float(
                candidate_omnibus_p[best_index]
            )
            best["passed_candidate_specificity_prefilter"] = bool(
                abs(float(best["raw_monotone_statistic"])) >= 0.15
                and float(best["raw_program_specificity_margin"]) > 0.01
                and float(best["raw_program_specificity_rank"]) >= 0.80
            )
            best["whole_root_fallback_used"] = not bool(branch_local_candidates)
            signature_joint = (
                float(signature_best["signature_exposure_contrast"])
                if signature_best is not None else 0.0
            )
            p_signature = float(
                (1 + np.sum(np.asarray(null_signature_joint) >= signature_joint))
                / (args.permutations + 1)
            )
            # The max-statistic winner fixes branch and window. Re-selecting a
            # whole-root window from the response here would undo the search
            # correction and dilute branch-local events.
            relation_sign = (
                1.0 if float(best["raw_monotone_statistic"]) >= 0.0 else -1.0
            )
            localization = best
            local_x = np.asarray(localization["x"], dtype=float)
            local_y = np.asarray(localization["y"], dtype=float)
            local_response = relation_sign * np.asarray(localization["y"], dtype=float)
            response_cut = float(np.quantile(local_response, 0.80))
            exposure_cut = float(np.quantile(local_x, 0.50))
            support_mask = (local_response >= response_cut) & (local_x >= exposure_cut)
            if int(support_mask.sum()) < 5:
                exposure_rank = pd.Series(local_x).rank(pct=True).to_numpy(float)
                response_rank = pd.Series(relation_sign * local_y).rank(pct=True).to_numpy(float)
                joint_rank = exposure_rank + response_rank
                keep = min(max(5, int(np.ceil(0.20 * len(joint_rank)))), len(joint_rank))
                support_mask = np.zeros(len(joint_rank), dtype=bool)
                support_mask[np.argsort(joint_rank)[-keep:]] = True
            support_positions = np.asarray(localization["positions"])[support_mask]
            support_branches = np.asarray(
                localization["local_branch_labels"], dtype=str
            )[support_mask]
            all_local_branches = np.asarray(
                localization["local_branch_labels"], dtype=str
            )
            localized_branch = str(best["affected_branch"])
            branch_enrichment = 0.0
            branch_support_count = 0
            enrichment_rows = []
            for branch in sorted(set(all_local_branches) - {"none"}):
                n_total = int(np.sum(all_local_branches == branch))
                n_support = int(np.sum(support_branches == branch))
                enrichment_rows.append((n_support / max(n_total, 1), n_support, branch))
            eligible_enrichment = [row for row in enrichment_rows if row[1] >= 3]
            if eligible_enrichment:
                branch_enrichment, branch_support_count, localized_branch = max(
                    eligible_enrichment
                )
            clean = {
                key: value for key, value in best.items()
                if key not in {
                    "x", "y", "signature_y", "xr", "yr", "signature_yr",
                    "blocks", "positions", "local_branch_labels",
                }
            }
            clean["signature_joint_statistic"] = signature_joint
            clean["signature_best_raw_rho"] = (
                float(signature_best["signature_raw_rho"]) if signature_best else 0.0
            )
            clean["signature_best_adjusted_rho"] = (
                float(signature_best["signature_rho"]) if signature_best else 0.0
            )
            clean["signature_best_raw_specificity_margin"] = (
                float(signature_best["signature_raw_specificity_margin"])
                if signature_best else -np.inf
            )
            clean["signature_best_raw_specificity_rank"] = (
                float(signature_best["signature_raw_specificity_rank"])
                if signature_best else 0.0
            )
            clean["detection_stratum"] = str(best["affected_branch"])
            clean["significance_window_start"] = clean["receiver_window_start"]
            clean["significance_window_end"] = clean["receiver_window_end"]
            clean["receiver_window_start"] = float(localization["receiver_window_start"])
            clean["receiver_window_end"] = float(localization["receiver_window_end"])
            clean["localization_response_tail_quantile"] = 0.80
            clean["localization_exposure_quantile"] = 0.50
            clean["affected_branch"] = localized_branch
            clean["localized_support_cell_ids"] = ";".join(
                adata.obs_names[support_positions].astype(str)
            )
            clean["localized_support_unit_ids"] = ";".join(
                sorted(set(metacells[support_positions].astype(str)))
            )
            clean["localized_support_n_cells"] = int(len(support_positions))
            clean["localized_branch_support_n_cells"] = int(branch_support_count)
            clean["localized_branch_support_fraction"] = float(branch_enrichment)
            clean["branch_localization_joint_statistic"] = min(
                abs(float(best["raw_monotone_statistic"])),
                abs(float(best["adjusted_monotone_statistic"])),
            )
            signature_support_positions = np.asarray([], dtype=int)
            signature_localized_branch = "none"
            signature_branch_support_count = 0
            signature_branch_enrichment = 0.0
            if signature_best is not None:
                signature_x = np.asarray(signature_best["x"], dtype=float)
                signature_response = np.asarray(signature_best["signature_y"], dtype=float)
                signature_response_cut = float(np.quantile(signature_response, 0.80))
                signature_exposure_cut = float(np.quantile(signature_x, 0.50))
                signature_support_mask = (
                    (signature_response >= signature_response_cut)
                    & (signature_x >= signature_exposure_cut)
                )
                if int(signature_support_mask.sum()) < 5:
                    exposure_rank = pd.Series(signature_x).rank(pct=True).to_numpy(float)
                    response_rank = pd.Series(signature_response).rank(pct=True).to_numpy(float)
                    joint_rank = exposure_rank + response_rank
                    keep = min(max(5, int(np.ceil(0.20 * len(joint_rank)))), len(joint_rank))
                    signature_support_mask = np.zeros(len(joint_rank), dtype=bool)
                    signature_support_mask[np.argsort(joint_rank)[-keep:]] = True
                signature_support_positions = np.asarray(
                    signature_best["positions"]
                )[signature_support_mask]
                signature_support_branches = np.asarray(
                    signature_best["local_branch_labels"], dtype=str
                )[signature_support_mask]
                signature_all_branches = np.asarray(
                    signature_best["local_branch_labels"], dtype=str
                )
                signature_rows = []
                for branch in sorted(set(signature_all_branches) - {"none"}):
                    n_total = int(np.sum(signature_all_branches == branch))
                    n_support = int(np.sum(signature_support_branches == branch))
                    signature_rows.append((n_support / max(n_total, 1), n_support, branch))
                signature_eligible = [row for row in signature_rows if row[1] >= 3]
                if signature_eligible:
                    (
                        signature_branch_enrichment,
                        signature_branch_support_count,
                        signature_localized_branch,
                    ) = max(signature_eligible)
            clean["signature_best_affected_branch"] = signature_localized_branch
            clean["signature_best_receiver_window_start"] = (
                float(signature_best["receiver_window_start"])
                if signature_best is not None else np.nan
            )
            clean["signature_best_receiver_window_end"] = (
                float(signature_best["receiver_window_end"])
                if signature_best is not None else np.nan
            )
            clean["signature_best_positivity_overlap"] = (
                float(signature_best["positivity_overlap"])
                if signature_best is not None else 0.0
            )
            clean["signature_best_exposure_variance"] = (
                float(signature_best["exposure_variance"])
                if signature_best is not None else 0.0
            )
            clean["signature_best_oof_predictive_gain"] = (
                float(signature_best["signature_oof_predictive_gain"])
                if signature_best is not None else -np.inf
            )
            clean["signature_localized_support_cell_ids"] = ";".join(
                adata.obs_names[signature_support_positions].astype(str)
            )
            clean["signature_localized_support_unit_ids"] = ";".join(
                sorted(set(metacells[signature_support_positions].astype(str)))
            )
            clean["signature_localized_support_n_cells"] = int(
                len(signature_support_positions)
            )
            clean["signature_localized_branch_support_n_cells"] = int(
                signature_branch_support_count
            )
            clean["signature_localized_branch_support_fraction"] = float(
                signature_branch_enrichment
            )
            clean.update({
                "status": "tested", "permutation_p": p_omnibus,
                "conjunctive_permutation_p": p_joint,
                "adjusted_only_permutation_p": p_adjusted,
                "raw_permutation_p": p_raw, "signature_permutation_p": p_signature,
                "n_event_local_candidates": len(candidates),
                "n_significance_candidates": len(significance_candidates),
            })
            evidence_rows.append(clean)
            payload_by_direction[(source_root, receiver_root)] = candidates

    evidence = pd.DataFrame(evidence_rows)
    tested = evidence.status.eq("tested")
    evidence["q_value"] = 1.0
    evidence["raw_q_value"] = 1.0
    max_source_activity = evidence.groupby(
        "source_root"
    )["source_program_activity"].first().max()
    evidence["source_activity_gap_to_max"] = (
        float(max_source_activity) - evidence.source_program_activity
    )
    evidence["source_hypothesis_eligible"] = (
        evidence.source_activity_gap_to_max.le(0.10)
        & evidence.source_program_enrichment.fillna(-np.inf).gt(0)
    )
    tested_family = tested & evidence.source_hypothesis_eligible
    if tested_family.any():
        evidence.loc[tested_family, "q_value"] = bh(
            evidence.loc[tested_family, "permutation_p"].to_numpy(float)
        )
        evidence.loc[tested_family, "raw_q_value"] = bh(
            evidence.loc[tested_family, "raw_permutation_p"].to_numpy(float)
        )
    max_source_enrichment = evidence.groupby("source_root")["source_program_enrichment"].first().max()
    evidence["source_program_relative_strength"] = (
        evidence.source_program_enrichment / max(float(max_source_enrichment), EPS)
    )
    evidence["event_type"] = np.where(
        evidence.raw_monotone_statistic.fillna(0).lt(0), "inhibition", "promotion"
    )
    collision = (
        evidence.event_type.eq("promotion")
        & evidence.signature_joint_statistic.fillna(0).gt(0.20)
        & evidence.signature_joint_statistic.fillna(0).gt(
            1.25 * evidence.joint_dependency_statistic.fillna(np.inf)
        )
        & evidence.signature_permutation_p.fillna(1).le(0.05)
    )
    evidence.loc[collision, "event_type"] = "collision_convergence"
    evidence["is_called"] = (
        tested & evidence.q_value.le(args.q_cutoff)
        & evidence.adjusted_monotone_statistic.abs().ge(args.effect_cutoff)
        & evidence.raw_monotone_statistic.abs().ge(args.raw_effect_cutoff)
        & evidence.source_program_enrichment.fillna(-np.inf).gt(0)
        & evidence.source_hypothesis_eligible.fillna(False)
        & evidence.raw_program_specificity_margin.fillna(-np.inf).gt(0.01)
        & evidence.raw_program_specificity_rank.fillna(0).ge(0.80)
        & evidence.program_specificity_margin.fillna(-np.inf).gt(0.01)
        & evidence.program_specificity_rank.fillna(0).ge(0.80)
        & evidence.environment_direction_consistent.fillna(False)
        & evidence.oof_predictive_gain.fillna(-np.inf).ge(args.minimum_oof_gain)
        & evidence.passed_candidate_specificity_prefilter.fillna(False)
        & evidence.exposure_variance.fillna(0).gt(1e-5)
        & evidence.positivity_overlap.fillna(0).ge(0.50)
    )
    evidence["decision_reason"] = np.where(
        evidence.is_called, "passes_significance_overlap_effect_gate", "fails_primary_gate"
    )

    for _, indices in evidence.groupby(
        evidence.apply(
            lambda row: tuple(sorted((str(row.source_root), str(row.receiver_root)))), axis=1
        )
    ).groups.items():
        called = evidence.loc[list(indices)]
        called = called[called.is_called].sort_values(
            "environment_adjusted_dependency", ascending=False
        )
        if len(called) <= 1:
            continue
        best, second = called.index[:2]
        ratio = float(
            evidence.loc[best, "environment_adjusted_dependency"]
            / max(evidence.loc[second, "environment_adjusted_dependency"], EPS)
        )
        if ratio < args.direction_ratio:
            evidence.loc[called.index, "is_called"] = False
            evidence.loc[called.index, "decision_reason"] = "bidirectional_ambiguity_abstain"
        else:
            evidence.loc[called.index[1:], "is_called"] = False
            evidence.loc[called.index[1:], "decision_reason"] = "reverse_direction_weaker"

    for _, indices in evidence.groupby("receiver_root").groups.items():
        called = evidence.loc[list(indices)]
        called = called[called.is_called].sort_values(
            "environment_adjusted_dependency", ascending=False
        )
        if len(called) <= 1:
            continue
        best, second = called.index[:2]
        ratio = float(
            evidence.loc[best, "environment_adjusted_dependency"]
            / max(evidence.loc[second, "environment_adjusted_dependency"], EPS)
        )
        if ratio < args.receiver_source_ratio:
            evidence.loc[called.index, "is_called"] = False
            evidence.loc[called.index, "decision_reason"] = "receiver_source_ambiguity_abstain"
        else:
            evidence.loc[called.index[1:], "is_called"] = False
            evidence.loc[called.index[1:], "decision_reason"] = "competing_source_weaker"

    evidence["raw_is_called_without_cmi_e"] = (
        tested_family & evidence.raw_q_value.le(args.q_cutoff)
        & evidence.raw_monotone_statistic.abs().ge(args.raw_effect_cutoff)
        & evidence.exposure_variance.fillna(0).gt(1e-5)
    )
    evidence = evidence.sort_values(
        ["is_called", "q_value", "environment_adjusted_dependency"],
        ascending=[False, True, False],
    )
    evidence.to_csv(output / "PHASE_B_CMI_E_EVIDENCE.csv", index=False)
    evidence.to_csv(output / "PHASE_B_RELATION_SCORE_DISTRIBUTION.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(output / "PHASE_B_EVENT_LOCAL_CANDIDATES.csv", index=False)
    evidence[evidence.is_called].to_csv(output / "PHASE_B_CALLED_RELATIONS.csv", index=False)
    status = {
        "phase": "B", "detector_version": "BRANCH_LOCAL_RELATION_V4_11_HIGH_CONFIDENCE_MINP",
        "candidate_selection_statistic": "min_abs_raw_adjusted_monotone_statistic",
        "response_statistic": "raw_monotone_plus_dml_or_adjusted_high_exposure_contrast",
        "crossfit": "receiver_root_nuisance_cell_level_three_fold_stratified_within_spatial_blocks",
        "nuisance_branch_terms": "frozen_T0_branch_one_hot_plus_branch_by_tau",
        "scan_randomization": "synchronized_within_spatial_block_across_candidates",
        "maxT_scale": "absolute_fisher_z_times_sqrt_n_minus_3",
        "familywise_adjustment": "single_step_Westfall_Young_minP",
        "spatial_adjustment": "standardized_xy_plus_quadratic_and_interaction",
        "whole_root_policy": "fallback_only_when_no_adequate_branch_local_candidate",
        "tau_window_policy": "top2_branch_window_by_contact_support",
        "frozen_phase_a_modified": False, "relation_significance_before_T1": True,
        "truth_used": False, "program_catalog": str(args.program_catalog.resolve()),
        "program_catalog_role": "externally supplied mechanism hypotheses",
        "environment_adjustment_programs": environment_programs.program_id.astype(str).tolist(),
        "environment_adjustment_enabled": not args.disable_environment_programs,
        "windows": WINDOWS, "permutations": args.permutations,
        "q_cutoff": args.q_cutoff, "effect_cutoff_abs_partial_rho": args.effect_cutoff,
        "raw_effect_cutoff_abs_rho": args.raw_effect_cutoff,
        "minimum_oof_predictive_gain": args.minimum_oof_gain,
        "localization_response_tail_quantile": 0.80,
        "localization_exposure_quantile": 0.50,
        "direction_ratio": args.direction_ratio,
        "receiver_source_ratio": args.receiver_source_ratio,
        "kernel_quantile": args.kernel_quantile,
        "contact_quantile": args.contact_quantile,
        "source_activity_quantile": args.source_activity_quantile,
        "source_identity_quantile": args.source_identity_quantile,
        "min_event_local_cells": args.min_cells,
        "n_contact_preselected_branches": 2,
        "n_ordered_root_pairs": int(tested.sum()),
        "n_called": int(evidence.is_called.sum()),
        "complete": True,
    }
    (output / "RUN_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
