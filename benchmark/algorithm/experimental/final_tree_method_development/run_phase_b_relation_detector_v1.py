#!/usr/bin/env python3
"""Event-local, confounding-aware relation evidence on frozen T0/tau0."""

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


def standardized_difference(target: np.ndarray, other: np.ndarray) -> float:
    pooled = np.sqrt(0.5 * (float(np.var(target)) + float(np.var(other))))
    return float((np.mean(target) - np.mean(other)) / max(pooled, EPS))


def spatial_blocks(coords: np.ndarray) -> np.ndarray:
    bins = 3 if len(coords) >= 90 else 2
    labels = []
    for axis in range(2):
        rank = pd.Series(coords[:, axis]).rank(method="first")
        labels.append(pd.qcut(rank, q=bins, labels=False, duplicates="drop").to_numpy(int))
    return labels[0] * bins + labels[1]


def design_matrix(tau: np.ndarray, coords: np.ndarray, annotation: np.ndarray) -> np.ndarray:
    scaled = zscore(coords)
    spatial_domains = []
    for projection in (
        scaled[:, 0], scaled[:, 1], scaled[:, 0] + scaled[:, 1],
        scaled[:, 0] - scaled[:, 1],
    ):
        rank = pd.Series(projection).rank(method="first")
        code = pd.qcut(rank, q=4, labels=False, duplicates="drop")
        spatial_domains.append(pd.get_dummies(code, drop_first=True).to_numpy(float))
    return np.column_stack([
        np.ones(len(tau)), tau, tau * tau, scaled[:, 0], scaled[:, 1],
        scaled[:, 0] ** 2, scaled[:, 1] ** 2, scaled[:, 0] * scaled[:, 1],
        *spatial_domains, annotation,
    ])


def crossfit_residuals(
    x: np.ndarray, y: np.ndarray, design: np.ndarray, blocks: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    unique = np.unique(blocks)
    fold_by_block = {block: index % 3 for index, block in enumerate(unique)}
    folds = np.asarray([fold_by_block[block] for block in blocks], dtype=int)
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


def exposure_from_source(
    receiver_coords: np.ndarray,
    source_coords: np.ndarray,
    source_activity: np.ndarray,
) -> tuple[np.ndarray, float]:
    distance = np.sqrt(
        ((receiver_coords[:, None, :] - source_coords[None, :, :]) ** 2).sum(axis=2)
    )
    nearest = distance.min(axis=1)
    radius = max(float(np.quantile(nearest, 0.35)), 1e-6)
    exposure = np.exp(-np.square(nearest / radius))
    return exposure, radius


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
    parser.add_argument("--direction-ratio", type=float, default=1.20)
    parser.add_argument("--receiver-source-ratio", type=float, default=1.15)
    parser.add_argument("--min-cells", type=int, default=25)
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
    source_genes = parse_genes(primary_program.source_genes)
    receiver_genes = parse_genes(primary_program.receiver_genes)
    signature_genes = parse_genes(signature_program.receiver_genes)
    negative_gene_sets = [parse_genes(value) for value in negative_programs.receiver_genes]
    required = set(source_genes + receiver_genes + signature_genes)
    required.update(gene for genes in negative_gene_sets for gene in genes)
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
    annotation = pd.get_dummies(adata.obs["cell.type"].astype(str), drop_first=True).to_numpy(float)

    evidence_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    payload_by_direction: dict[tuple[str, str], list[dict[str, object]]] = {}
    for source_root in roots:
        source_all = np.flatnonzero(root_assignment == source_root)
        source_tau = np.asarray([
            tau_by_root[source_root].get(str(metacells[index]), 0.5) for index in source_all
        ])
        threshold = float(np.quantile(source_score[source_all], 0.40))
        selected_mask = source_score[source_all] >= threshold
        selected_source = source_all[selected_mask]
        reference = np.flatnonzero(root_assignment != source_root)
        if len(selected_source) < args.min_cells or len(reference) < args.min_cells:
            continue
        source_enrichment = standardized_difference(
            source_score[selected_source], source_score[reference]
        )
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
            candidates: list[dict[str, object]] = []
            exposure, radius = exposure_from_source(
                coords[receiver_all], coords[selected_source], source_score[selected_source]
            )
            for branch in ["ALL", *sorted(set(receiver_branch) - {"none"})]:
                branch_mask = (
                    np.ones(len(receiver_branch), dtype=bool)
                    if branch == "ALL" else receiver_branch == branch
                )
                for receiver_start, receiver_end in WINDOWS:
                        local_mask = (
                            branch_mask & (receiver_tau >= receiver_start)
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
                        design = design_matrix(local_tau, local_coords, local_annotation)
                        blocks = spatial_blocks(local_coords)
                        xr, yr, folds, delta_r2 = crossfit_residuals(x, y, design, blocks)
                        _, signature_yr, _, signature_delta_r2 = crossfit_residuals(
                            x, signature_y, design, blocks
                        )
                        negative_dependencies = []
                        negative_raw_dependencies = []
                        for negative_score in negative_scores:
                            negative_y = negative_score[positions]
                            _, negative_yr, _, _ = crossfit_residuals(
                                x, negative_y, design, blocks
                            )
                            negative_dependencies.append(
                                gaussian_mi(correlation(xr, negative_yr))
                            )
                            negative_raw_dependencies.append(
                                gaussian_mi(correlation(x, negative_y))
                            )
                        raw_rho = correlation(x, y)
                        adjusted_rho = correlation(xr, yr)
                        signature_raw_rho = correlation(x, signature_y)
                        signature_rho = correlation(xr, signature_yr)
                        raw_dependency = gaussian_mi(raw_rho)
                        adjusted_dependency = gaussian_mi(adjusted_rho)
                        negative_median = float(np.median(negative_dependencies))
                        negative_max = float(np.max(negative_dependencies))
                        specificity_rank = float(
                            np.mean(adjusted_dependency > np.asarray(negative_dependencies))
                        )
                        drop = max(raw_dependency - adjusted_dependency, 0.0)
                        q10, q90 = np.quantile(x, [0.10, 0.90])
                        positivity = float(np.mean((x > q10) & (x < q90)))
                        candidates.append({
                            "source_root": source_root, "receiver_root": receiver_root,
                            "source_window_start": selected_source_start,
                            "source_window_end": selected_source_end,
                            "source_program_enrichment": source_enrichment,
                            "receiver_window_start": receiver_start,
                            "receiver_window_end": receiver_end,
                            "affected_branch": branch, "n_receiver_cells": len(positions),
                            "contact_radius": radius, "exposure_variance": float(np.var(x)),
                            "positivity_overlap": positivity, "raw_rho": raw_rho,
                            "adjusted_rho": adjusted_rho,
                            "raw_dependency": raw_dependency,
                            "environment_adjusted_dependency": adjusted_dependency,
                            "environment_drop": drop / max(raw_dependency, EPS),
                            "environment_trust": adjusted_dependency / max(raw_dependency + adjusted_dependency, EPS),
                            "environment_conflict": drop,
                            "oof_predictive_gain": delta_r2,
                            "negative_control_dependency_median": negative_median,
                            "negative_control_dependency_max": negative_max,
                            "program_specificity_margin": adjusted_dependency - negative_median,
                            "program_specificity_rank": specificity_rank,
                            "raw_negative_control_dependency_median": float(
                                np.median(negative_raw_dependencies)
                            ),
                            "signature_raw_rho": signature_raw_rho,
                            "signature_rho": signature_rho,
                            "signature_oof_predictive_gain": signature_delta_r2,
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
                    if key not in {"x", "y", "signature_y", "xr", "yr", "signature_yr", "blocks"}
                }
                for item in candidates
            ])
            eligible_candidates = [
                item for item in candidates
                if abs(float(item["raw_rho"])) >= 0.15
                and float(item["program_specificity_margin"]) > 0.01
                and float(item["program_specificity_rank"]) >= 0.80
            ]
            selection_pool = eligible_candidates if eligible_candidates else candidates
            best = max(
                selection_pool,
                key=lambda item: min(
                    abs(float(item["raw_rho"])), abs(float(item["adjusted_rho"]))
                ),
            )
            best["joint_dependency_statistic"] = min(
                abs(float(best["raw_rho"])), abs(float(best["adjusted_rho"]))
            )
            best["passed_candidate_specificity_prefilter"] = bool(eligible_candidates)
            signature_pool = [
                item for item in candidates
                if float(item["signature_raw_rho"]) > 0
                and float(item["signature_rho"]) > 0
            ]
            signature_best = max(
                signature_pool,
                key=lambda item: min(
                    float(item["signature_raw_rho"]), float(item["signature_rho"])
                ),
            ) if signature_pool else None
            null_adjusted = []
            null_raw = []
            null_joint = []
            null_signature_joint = []
            for _ in range(args.permutations):
                adjusted_stats = []
                raw_stats = []
                joint_stats = []
                signature_joint_stats = []
                for item in candidates:
                    order = permutation_order(item["blocks"], rng)
                    adjusted_stat = abs(correlation(item["xr"][order], item["yr"]))
                    raw_stat = abs(correlation(item["x"][order], item["y"]))
                    signature_adjusted = correlation(
                        item["xr"][order], item["signature_yr"]
                    )
                    signature_raw = correlation(
                        item["x"][order], item["signature_y"]
                    )
                    adjusted_stats.append(adjusted_stat)
                    raw_stats.append(raw_stat)
                    joint_stats.append(min(raw_stat, adjusted_stat))
                    signature_joint_stats.append(
                        min(signature_raw, signature_adjusted)
                        if signature_raw > 0 and signature_adjusted > 0 else 0.0
                    )
                null_adjusted.append(max(adjusted_stats))
                null_raw.append(max(raw_stats))
                null_joint.append(max(joint_stats))
                null_signature_joint.append(max(signature_joint_stats))
            p_joint = float(
                (1 + np.sum(
                    np.asarray(null_joint) >= float(best["joint_dependency_statistic"])
                )) / (args.permutations + 1)
            )
            p_adjusted = float(
                (1 + np.sum(np.asarray(null_adjusted) >= abs(float(best["adjusted_rho"]))))
                / (args.permutations + 1)
            )
            p_raw = float(
                (1 + np.sum(np.asarray(null_raw) >= abs(float(best["raw_rho"]))))
                / (args.permutations + 1)
            )
            signature_joint = (
                min(
                    float(signature_best["signature_raw_rho"]),
                    float(signature_best["signature_rho"]),
                ) if signature_best is not None else 0.0
            )
            p_signature = float(
                (1 + np.sum(np.asarray(null_signature_joint) >= signature_joint))
                / (args.permutations + 1)
            )
            clean = {
                key: value for key, value in best.items()
                if key not in {"x", "y", "signature_y", "xr", "yr", "signature_yr", "blocks"}
            }
            clean["signature_joint_statistic"] = signature_joint
            clean["signature_best_raw_rho"] = (
                float(signature_best["signature_raw_rho"]) if signature_best else 0.0
            )
            clean["signature_best_adjusted_rho"] = (
                float(signature_best["signature_rho"]) if signature_best else 0.0
            )
            clean["detection_stratum"] = str(best["affected_branch"])
            if str(best["affected_branch"]) == "ALL":
                localized = [
                    item for item in candidates
                    if item["affected_branch"] != "ALL"
                    and item["receiver_window_start"] == best["receiver_window_start"]
                    and item["receiver_window_end"] == best["receiver_window_end"]
                ]
                if localized:
                    clean["affected_branch"] = max(
                        localized, key=lambda item: abs(float(item["adjusted_rho"]))
                    )["affected_branch"]
            clean.update({
                "status": "tested", "permutation_p": p_joint,
                "adjusted_only_permutation_p": p_adjusted,
                "raw_permutation_p": p_raw, "signature_permutation_p": p_signature,
                "n_event_local_candidates": len(candidates),
            })
            evidence_rows.append(clean)
            payload_by_direction[(source_root, receiver_root)] = candidates

    evidence = pd.DataFrame(evidence_rows)
    tested = evidence.status.eq("tested")
    evidence["q_value"] = 1.0
    evidence["raw_q_value"] = 1.0
    if tested.any():
        evidence.loc[tested, "q_value"] = bh(evidence.loc[tested, "permutation_p"].to_numpy(float))
        evidence.loc[tested, "raw_q_value"] = bh(evidence.loc[tested, "raw_permutation_p"].to_numpy(float))
    max_source_enrichment = evidence.groupby("source_root")["source_program_enrichment"].first().max()
    evidence["source_program_relative_strength"] = (
        evidence.source_program_enrichment / max(float(max_source_enrichment), EPS)
    )
    evidence["event_type"] = np.where(
        evidence.raw_rho.fillna(0).lt(0), "inhibition", "promotion"
    )
    collision = (
        evidence.event_type.eq("promotion")
        & evidence.signature_joint_statistic.fillna(0).gt(0.20)
        & evidence.signature_permutation_p.fillna(1).le(0.05)
    )
    evidence.loc[collision, "event_type"] = "collision_convergence"
    evidence["is_called"] = (
        tested & evidence.q_value.le(args.q_cutoff)
        & evidence.adjusted_rho.abs().ge(args.effect_cutoff)
        & evidence.raw_rho.abs().ge(0.15)
        & evidence.source_program_enrichment.fillna(-np.inf).gt(0)
        & evidence.source_program_relative_strength.fillna(0).ge(0.80)
        & evidence.program_specificity_margin.fillna(-np.inf).gt(0.01)
        & evidence.program_specificity_rank.fillna(0).ge(0.80)
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
        tested & evidence.raw_q_value.le(args.q_cutoff)
        & evidence.raw_rho.abs().ge(args.effect_cutoff)
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
        "phase": "B", "detector_version": "EVENT_LOCAL_RELATION_V1",
        "frozen_phase_a_modified": False, "relation_significance_before_T1": True,
        "truth_used": False, "program_catalog": str(args.program_catalog.resolve()),
        "program_catalog_role": "externally supplied mechanism hypotheses",
        "windows": WINDOWS, "permutations": args.permutations,
        "q_cutoff": args.q_cutoff, "effect_cutoff_abs_partial_rho": args.effect_cutoff,
        "direction_ratio": args.direction_ratio,
        "receiver_source_ratio": args.receiver_source_ratio,
        "min_event_local_cells": args.min_cells,
        "n_ordered_root_pairs": int(tested.sum()),
        "n_called": int(evidence.is_called.sum()),
        "complete": True,
    }
    (output / "RUN_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
