#!/usr/bin/env python3
"""Frozen P8 biological validation for the xenium breast T0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import multipletests


SEED = 20260812
N_PERM = 499
N_BOOT = 199
MIN_GENES = 3
PRIMARY_FAMILIES = {"external_curated", "Hallmark_2020"}
EFFECT_THRESHOLD = 0.5
RHO_THRESHOLD = 0.20
Q_THRESHOLD = 0.10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh(values: pd.Series) -> np.ndarray:
    data = values.to_numpy(float)
    result = np.full(len(data), np.nan)
    valid = np.isfinite(data)
    if valid.any():
        result[valid] = multipletests(data[valid], method="fdr_bh")[1]
    return result


def dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def parse_gmt(path: Path, family: str) -> list[dict[str, object]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.rstrip().split("\t")
        if len(fields) < 3:
            continue
        genes = [gene.strip().upper() for gene in fields[2:] if gene.strip()]
        records.append({"gene_set": fields[0].strip(), "family": family, "genes": genes})
    return records


def construction_genes(adata: ad.AnnData, nodes: pd.DataFrame) -> pd.DataFrame:
    matrix = dense(adata.X)
    rows = []
    for root_id, part in nodes.groupby("root_id", sort=True):
        indices = np.sort(part.node_id.astype(int).unique())
        local = np.log1p(np.maximum(matrix[indices], 0.0))
        n_hvg = min(local.shape[1], max(10, int(math.ceil(2.0 * math.sqrt(local.shape[1])))))
        selected = np.argsort(np.var(local, axis=0), kind="mergesort")[-n_hvg:]
        for rank, index in enumerate(selected[::-1], start=1):
            rows.append({
                "root_id": str(root_id),
                "gene": str(adata.var_names[index]).upper(),
                "variance_rank_desc": rank,
                "construction_role": "branch_program_estimator_v9_HVG",
            })
    return pd.DataFrame(rows)


def spatial_blocks(obs: pd.DataFrame, q: int = 4) -> np.ndarray:
    x = pd.qcut(obs.spatial_x.rank(method="first"), q=q, labels=False).to_numpy(int)
    y = pd.qcut(obs.spatial_y.rank(method="first"), q=q, labels=False).to_numpy(int)
    return x * q + y


def standardized_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.sqrt(np.mean((a.mean(axis=0) - b.mean(axis=0)) ** 2))
    scale = np.sqrt(np.nanmean(np.r_[np.var(a, axis=0, ddof=1), np.var(b, axis=0, ddof=1)]))
    return float(diff / max(scale, 1e-12))


def spatial_label_permutation(labels: np.ndarray, blocks: np.ndarray, rng) -> np.ndarray:
    result = labels.copy()
    for block in np.unique(blocks):
        positions = np.flatnonzero(blocks == block)
        result[positions] = rng.permutation(result[positions])
    return result


def pooled_effect(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    pooled = math.sqrt(
        max(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1))
            / max(len(x) + len(y) - 2, 1), 1e-12)
    )
    d = (float(np.mean(x)) - float(np.mean(y))) / pooled
    correction = 1.0 - 3.0 / max(4.0 * (len(x) + len(y)) - 9.0, 1.0)
    return float(d * correction)


def block_bootstrap_rho(score: np.ndarray, tau: np.ndarray, blocks: np.ndarray, rng) -> tuple[float, float, float]:
    unique = np.unique(blocks)
    values = []
    for _ in range(N_BOOT):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([np.flatnonzero(blocks == block) for block in chosen])
        if len(np.unique(tau[positions])) < 3:
            continue
        values.append(float(stats.spearmanr(tau[positions], score[positions]).statistic))
    if not values:
        return float("nan"), float("nan"), float("nan")
    observed = float(stats.spearmanr(tau, score).statistic)
    same = np.mean(np.sign(values) == np.sign(observed))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), float(same)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p6-dir", type=Path, required=True)
    parser.add_argument("--method-dir", type=Path, required=True)
    parser.add_argument("--raw-h5ad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    phase_a = args.p6_dir / "phase_a"
    nodes = pd.read_csv(phase_a / "T0_STATE_NODE_ASSIGNMENTS.csv")
    edges = pd.read_csv(phase_a / "T0_FINE_TRAJECTORY_EDGES.csv")
    soft = pd.read_csv(phase_a / "P6_FROZEN_SOFT_MEMBERSHIP.csv").set_index("unit_id")
    adata = ad.read_h5ad(args.method_dir / "metacell_adata.h5ad")
    adata.obs_names = adata.obs.metacell_id.astype(str)
    covered = sorted(nodes.unit_id.astype(str).unique())
    obs = adata.obs.loc[covered].copy()
    obs["spatial_block"] = spatial_blocks(obs)

    hvg = construction_genes(adata, nodes)
    hvg.to_csv(args.output / "P8_CONSTRUCTION_GENE_MANIFEST.csv", index=False)
    hvg_union = set(hvg.gene)

    records = []
    curated = pd.read_csv(args.output / "P8_CURATED_IMMUNE_SIGNATURES.csv")
    for row in curated.itertuples(index=False):
        records.append({
            "gene_set": str(row.signature), "family": str(row.evidence_role),
            "genes": [gene.strip().upper() for gene in str(row.genes).split(";") if gene.strip()],
        })
    gene_dir = args.output / "gene_sets"
    records += parse_gmt(gene_dir / "Hallmark_2020.gmt", "Hallmark_2020")
    records += parse_gmt(gene_dir / "Reactome_2022.gmt", "Reactome_2022")
    records += parse_gmt(gene_dir / "GO_Biological_Process_2023.gmt", "GO_Biological_Process_2023")

    panel = {str(gene).upper() for gene in adata.var_names}
    metadata = []
    usable = []
    seen = set()
    for record in records:
        key = (record["family"], record["gene_set"])
        if key in seen:
            continue
        seen.add(key)
        full_genes = sorted(set(record["genes"]) & panel)
        independent = sorted(set(full_genes) - hvg_union)
        metadata.append({
            "family": record["family"], "gene_set": record["gene_set"],
            "n_source_genes": len(set(record["genes"])),
            "n_panel_genes": len(full_genes), "panel_genes": ";".join(full_genes),
            "n_independent_genes": len(independent),
            "independent_genes": ";".join(independent),
            "primary_family": record["family"] in PRIMARY_FAMILIES,
            "independent_usable": len(independent) >= MIN_GENES,
            "full_usable": len(full_genes) >= MIN_GENES,
            "evidence_role": "independent_leave_branch_HVG_out" if len(independent) >= MIN_GENES else "post_hoc_or_unavailable",
        })
        if len(full_genes) >= MIN_GENES or len(independent) >= MIN_GENES:
            usable.append((record["family"], record["gene_set"], full_genes, independent))
    meta_sets = pd.DataFrame(metadata)
    meta_sets.to_csv(args.output / "P8_GENE_SET_AVAILABILITY.csv", index=False)

    all_index = {str(gene).upper(): i for i, gene in enumerate(adata.var_names)}
    covered_idx = np.array([adata.obs_names.get_loc(unit) for unit in covered])
    expression = np.log1p(np.maximum(dense(adata.X)[covered_idx], 0.0))
    z = stats.zscore(expression, axis=0, nan_policy="omit")
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    score_columns = {}
    score_meta = []
    full_columns = {}
    for family, name, full_genes, independent in usable:
        token = f"{family}::{name}"
        if len(independent) >= MIN_GENES:
            score_columns[token] = z[:, [all_index[g] for g in independent]].mean(axis=1)
            score_meta.append({"score_id": token, "family": family, "gene_set": name, "n_genes": len(independent), "evidence_role": "independent"})
        if len(full_genes) >= MIN_GENES:
            full_columns[token] = z[:, [all_index[g] for g in full_genes]].mean(axis=1)
    scores = pd.DataFrame(score_columns, index=covered)
    full_scores = pd.DataFrame(full_columns, index=covered)
    pd.DataFrame(score_meta).to_csv(args.output / "P8_INDEPENDENT_SIGNATURE_INDEX.csv", index=False)
    score_long = scores.reset_index(names="unit_id").melt(id_vars="unit_id", var_name="score_id", value_name="score")
    score_long = score_long.merge(pd.DataFrame(score_meta), on="score_id", how="left")
    score_long.to_csv(args.output / "P8_INDEPENDENT_SIGNATURE_SCORES.csv", index=False)

    primary_ids = [row["score_id"] for row in score_meta if row["family"] in PRIMARY_FAMILIES]
    primary = scores[primary_ids]
    structural_count = nodes.groupby("unit_id").root_id.nunique()
    unique_units = set(structural_count[structural_count.eq(1)].index.astype(str))
    primary_nodes = nodes[nodes.unit_id.astype(str).isin(unique_units)].drop_duplicates("unit_id").copy()
    primary_nodes = primary_nodes.set_index("unit_id").loc[primary.index.intersection(unique_units)]
    primary_matrix = primary.loc[primary_nodes.index].to_numpy(float)
    labels = primary_nodes.root_id.astype(str).to_numpy()
    blocks = obs.loc[primary_nodes.index, "spatial_block"].to_numpy(int)

    root_pairs = []
    roots = sorted(np.unique(labels))
    for i, left in enumerate(roots):
        for right in roots[i + 1:]:
            mask = np.isin(labels, [left, right])
            local_x, local_y, local_b = primary_matrix[mask], labels[mask], blocks[mask]
            observed = standardized_distance(local_x[local_y == left], local_x[local_y == right])
            null = []
            for _ in range(N_PERM):
                permuted = spatial_label_permutation(local_y, local_b, rng)
                null.append(standardized_distance(local_x[permuted == left], local_x[permuted == right]))
            p = (1 + np.sum(np.asarray(null) >= observed)) / (N_PERM + 1)
            root_pairs.append({
                "root_a": left, "root_b": right,
                "n_a": int(np.sum(local_y == left)), "n_b": int(np.sum(local_y == right)),
                "n_primary_independent_scores": len(primary_ids),
                "standardized_centroid_distance": observed,
                "spatial_block_permutation_p": p,
                "null_median": float(np.median(null)),
            })
    root_profile = pd.DataFrame(root_pairs)
    root_profile["spatial_block_permutation_q"] = bh(root_profile.spatial_block_permutation_p)
    root_profile["passes_primary_gate"] = (
        root_profile.standardized_centroid_distance.ge(EFFECT_THRESHOLD)
        & root_profile.spatial_block_permutation_q.le(Q_THRESHOLD)
    )
    root_profile.to_csv(args.output / "P8_ROOT_IDENTITY_AUDIT.csv", index=False)

    # One-vs-other branch specialization within each frozen root.
    branch_tests = []
    for root_id, part in primary_nodes.groupby("root_id", sort=True):
        local_units = part.index
        local_scores = primary.loc[local_units]
        local_blocks = obs.loc[local_units, "spatial_block"].to_numpy(int)
        branch_labels = part.branch_id.astype(str).to_numpy()
        for branch_id in sorted(np.unique(branch_labels)):
            target = branch_labels == branch_id
            if target.sum() < 5 or (~target).sum() < 5:
                continue
            for score_id in primary_ids:
                values = local_scores[score_id].to_numpy(float)
                effect = pooled_effect(values[target], values[~target])
                null = []
                for _ in range(N_PERM):
                    permuted = spatial_label_permutation(branch_labels, local_blocks, rng)
                    trial = permuted == branch_id
                    null.append(abs(pooled_effect(values[trial], values[~trial])))
                p = (1 + np.sum(np.asarray(null) >= abs(effect))) / (N_PERM + 1)
                branch_tests.append({
                    "root_id": str(root_id), "branch_id": branch_id,
                    "score_id": score_id, "n_branch": int(target.sum()),
                    "n_other": int((~target).sum()), "standardized_effect": effect,
                    "spatial_block_permutation_p": p,
                })
    branch_tests = pd.DataFrame(branch_tests)
    branch_tests["spatial_block_permutation_q"] = bh(branch_tests.spatial_block_permutation_p)
    branch_tests["qualifying_specialization"] = (
        branch_tests.standardized_effect.abs().ge(EFFECT_THRESHOLD)
        & branch_tests.spatial_block_permutation_q.le(Q_THRESHOLD)
    )
    branch_tests.to_csv(args.output / "P8_BRANCH_SIGNATURE_CONTRASTS.csv", index=False)
    branch_summary = []
    for (root_id, branch_id), part in branch_tests.groupby(["root_id", "branch_id"], sort=True):
        qualifying = part[part.qualifying_specialization]
        selection = qualifying if len(qualifying) else part
        best = selection.assign(abs_effect=selection.standardized_effect.abs()).sort_values(
            ["spatial_block_permutation_q", "abs_effect"], ascending=[True, False]
        ).iloc[0]
        branch_summary.append({
            "root_id": root_id, "branch_id": branch_id,
            "n_primary_scores_tested": len(part),
            "n_qualifying_specializations": int(part.qualifying_specialization.sum()),
            "best_score_id": best.score_id,
            "best_effect": float(best.standardized_effect),
            "best_q": float(best.spatial_block_permutation_q),
            "passes_specialization_gate": bool(part.qualifying_specialization.any()),
        })
    pd.DataFrame(branch_summary).to_csv(args.output / "P8_BRANCH_IDENTITY_AUDIT.csv", index=False)

    # Frozen program and state/step tables.
    program_table = (
        nodes.groupby(["root_id", "branch_id"], sort=True)
        .agg(n_node_rows=("unit_id", "size"), n_unique_metacells=("unit_id", "nunique"),
             tau_min=("pseudotime", "min"), tau_max=("pseudotime", "max"),
             n_frozen_states=("state_id", "nunique"))
        .reset_index()
    )
    program_table["program_definition"] = "frozen_T0_branch"
    program_table.to_csv(args.output / "P8_PROGRAM_TABLE.csv", index=False)
    step_stats = nodes.groupby(["root_id", "branch_id", "state_id"], sort=True).pseudotime.agg(["median", "min", "max", "size"]).reset_index()
    step_stats["step_order"] = step_stats.groupby(["root_id", "branch_id"])["median"].rank(method="first").astype(int) - 1
    steps = nodes.merge(step_stats[["root_id", "branch_id", "state_id", "step_order"]], on=["root_id", "branch_id", "state_id"], how="left")
    steps.to_csv(args.output / "P8_STEP_ASSIGNMENTS.csv", index=False)

    step_scores = steps[["root_id", "branch_id", "state_id", "step_order", "unit_id", "pseudotime"]].merge(
        score_long, on="unit_id", how="left"
    )
    step_function = (
        step_scores.groupby(["root_id", "branch_id", "state_id", "step_order", "score_id", "family", "gene_set", "evidence_role"], sort=True)
        .agg(n=("score", "size"), mean_score=("score", "mean"), sd_score=("score", "std"), median_tau=("pseudotime", "median"))
        .reset_index()
    )
    step_function.to_csv(args.output / "P8_STEP_FUNCTIONAL_SCORES.csv", index=False)

    # Continuous pseudotime analysis on primary independent scores.
    progression = []
    for (root_id, branch_id), part in nodes.groupby(["root_id", "branch_id"], sort=True):
        local = part.drop_duplicates("unit_id").set_index("unit_id")
        local_scores = primary.loc[local.index]
        tau = local.pseudotime.to_numpy(float)
        local_blocks = obs.loc[local.index, "spatial_block"].to_numpy(int)
        if len(local) < 15 or len(np.unique(tau)) < 3:
            continue
        for score_id in primary_ids:
            values = local_scores[score_id].to_numpy(float)
            rho = float(stats.spearmanr(tau, values).statistic)
            null = []
            for _ in range(N_PERM):
                perm_tau = tau.copy()
                for block in np.unique(local_blocks):
                    ix = np.flatnonzero(local_blocks == block)
                    perm_tau[ix] = rng.permutation(perm_tau[ix])
                null.append(abs(float(stats.spearmanr(perm_tau, values).statistic)))
            p = (1 + np.sum(np.asarray(null) >= abs(rho))) / (N_PERM + 1)
            low, high, stability = block_bootstrap_rho(values, tau, local_blocks, rng)
            progression.append({
                "root_id": str(root_id), "branch_id": str(branch_id),
                "score_id": score_id, "n_metacells": len(local),
                "spearman_rho": rho, "spatial_block_permutation_p": p,
                "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                "sign_stability": stability,
            })
    progression = pd.DataFrame(progression)
    progression["spatial_block_permutation_q"] = bh(progression.spatial_block_permutation_p)
    progression["qualifying_progression"] = (
        progression.spearman_rho.abs().ge(RHO_THRESHOLD)
        & progression.spatial_block_permutation_q.le(Q_THRESHOLD)
        & progression.sign_stability.ge(0.70)
    )
    progression.to_csv(args.output / "P8_CONTINUOUS_PSEUDOTIME_FUNCTIONS.csv", index=False)

    # Full-score versus leave-branch-HVG-out sensitivity using the same gates.
    full_progression = []
    sensitivity = []
    full_primary = [score for score in primary_ids if score in full_scores]
    for (root_id, branch_id), part in nodes.groupby(["root_id", "branch_id"], sort=True):
        local = part.drop_duplicates("unit_id").set_index("unit_id")
        tau = local.pseudotime.to_numpy(float)
        local_blocks = obs.loc[local.index, "spatial_block"].to_numpy(int)
        for score_id in full_primary:
            full_values = full_scores.loc[local.index, score_id].to_numpy(float)
            full_rho = float(stats.spearmanr(tau, full_values).statistic)
            null = []
            for _ in range(N_PERM):
                perm_tau = tau.copy()
                for block in np.unique(local_blocks):
                    ix = np.flatnonzero(local_blocks == block)
                    perm_tau[ix] = rng.permutation(perm_tau[ix])
                null.append(abs(float(stats.spearmanr(perm_tau, full_values).statistic)))
            full_p = (1 + np.sum(np.asarray(null) >= abs(full_rho))) / (N_PERM + 1)
            full_low, full_high, full_stability = block_bootstrap_rho(
                full_values, tau, local_blocks, rng
            )
            full_progression.append({
                "root_id": str(root_id), "branch_id": str(branch_id),
                "score_id": score_id, "full_score_rho": full_rho,
                "full_spatial_block_permutation_p": full_p,
                "full_bootstrap_ci_low": full_low,
                "full_bootstrap_ci_high": full_high,
                "full_sign_stability": full_stability,
            })
            independent_rho = float(stats.spearmanr(tau, scores.loc[local.index, score_id]).statistic)
            sensitivity.append({
                "root_id": str(root_id), "branch_id": str(branch_id), "score_id": score_id,
                "full_score_rho": full_rho, "leave_HVG_out_rho": independent_rho,
                "direction_retained": bool(np.sign(full_rho) == np.sign(independent_rho)),
                "effect_fraction_retained": abs(independent_rho) / max(abs(full_rho), 1e-12),
                "topology_rebuilt": False,
            })
    full_progression = pd.DataFrame(full_progression)
    full_progression["full_spatial_block_permutation_q"] = bh(
        full_progression.full_spatial_block_permutation_p
    )
    full_progression["full_effect_qualifies"] = (
        full_progression.full_score_rho.abs().ge(RHO_THRESHOLD)
        & full_progression.full_spatial_block_permutation_q.le(Q_THRESHOLD)
        & full_progression.full_sign_stability.ge(0.70)
    )
    sensitivity = pd.DataFrame(sensitivity).drop(columns="full_score_rho").merge(
        full_progression, on=["root_id", "branch_id", "score_id"], how="left"
    )
    sensitivity["sensitivity_survives"] = (
        sensitivity.full_effect_qualifies & sensitivity.direction_retained
        & sensitivity.effect_fraction_retained.ge(0.50)
    )
    sensitivity.to_csv(args.output / "P8_LEAVE_CONSTRUCTION_GENES_OUT.csv", index=False)

    # Secondary Reactome/GO tests are descriptive and do not unlock P9.
    secondary_rows = []
    secondary_ids = [row["score_id"] for row in score_meta if row["family"] not in PRIMARY_FAMILIES]
    for (root_id, branch_id), part in nodes.groupby(["root_id", "branch_id"], sort=True):
        local = part.drop_duplicates("unit_id").set_index("unit_id")
        tau = local.pseudotime.to_numpy(float)
        for score_id in secondary_ids:
            rho, p = stats.spearmanr(tau, scores.loc[local.index, score_id].to_numpy(float))
            secondary_rows.append({
                "analysis": "continuous_pseudotime_secondary",
                "root_id": str(root_id), "branch_id": str(branch_id),
                "score_id": score_id, "effect": float(rho),
                "nominal_p": float(p), "n_metacells": len(local),
            })
    secondary = pd.DataFrame(secondary_rows)
    secondary["family"] = secondary.score_id.str.split("::", n=1).str[0]
    secondary["BH_q_within_family"] = secondary.groupby("family", group_keys=False).nominal_p.transform(
        lambda values: pd.Series(bh(values), index=values.index)
    )
    secondary["inference_role"] = "secondary_descriptive_single_section"
    secondary.to_csv(args.output / "P8_SECONDARY_PATHWAY_TESTS.csv", index=False)

    functional = pd.concat([
        root_profile.assign(analysis="root_profile", score_id="MULTIVARIATE_PRIMARY_PROFILE"),
        branch_tests.assign(analysis="branch_specialization"),
        progression.assign(analysis="continuous_pseudotime"),
        secondary,
    ], ignore_index=True, sort=False)
    functional.to_csv(args.output / "P8_FUNCTIONAL_ENRICHMENT.csv", index=False)
    evidence_roles = meta_sets[["family", "gene_set", "evidence_role", "n_panel_genes", "n_independent_genes", "panel_genes", "independent_genes"]].copy()
    evidence_roles["strict_external_assay_independent"] = False
    evidence_roles["limitation"] = "all panel genes contributed to candidate PCA geometry"
    evidence_roles.to_csv(args.output / "P8_INDEPENDENT_VS_POSTHOC_EVIDENCE.csv", index=False)

    # Cell composition and spatial provenance are inherited from the no-refit P6.5 audit.
    root_comp = pd.read_csv(args.p6_dir / "P65_ROOT_CELL_TYPE_COMPOSITION.csv")
    branch_comp = pd.read_csv(args.p6_dir / "P65_BRANCH_CELL_TYPE_COMPOSITION.csv")
    root_comp["scope"] = "root"
    branch_comp["scope"] = "branch"
    pd.concat([root_comp, branch_comp], ignore_index=True, sort=False).to_csv(args.output / "P8_CELL_COMPOSITION.csv", index=False)

    spatial_rows = []
    for (root_id, branch_id), part in nodes.groupby(["root_id", "branch_id"], sort=True):
        units = part.unit_id.astype(str).unique()
        xy = obs.loc[units, ["spatial_x", "spatial_y"]].to_numpy(float)
        spatial_rows.append({
            "root_id": str(root_id), "branch_id": str(branch_id), "n_metacells": len(units),
            "centroid_x": float(xy[:, 0].mean()), "centroid_y": float(xy[:, 1].mean()),
            "x_sd": float(xy[:, 0].std()), "y_sd": float(xy[:, 1].std()),
            "n_spatial_blocks": int(obs.loc[units, "spatial_block"].nunique()),
            "sample_id": "xenium_breast", "section_id": "xenium_breast_single_section",
        })
    pd.DataFrame(spatial_rows).to_csv(args.output / "P8_SPATIAL_PROVENANCE.csv", index=False)

    # Soft-sharing biology, keeping the predeclared 0.20 threshold.
    memberships = soft.loc[covered, ["R0", "R1", "R2"]].to_numpy(float)
    row_sum = memberships.sum(axis=1, keepdims=True)
    probability = np.divide(memberships, row_sum, out=np.zeros_like(memberships), where=row_sum > 0)
    entropy = -np.sum(np.where(probability > 0, probability * np.log(probability + 1e-12), 0.0), axis=1) / np.log(3)
    active = (memberships >= 0.20).sum(axis=1)
    structural = structural_count.reindex(covered).fillna(0).to_numpy(int)
    sharing_group = np.where(structural >= 2, "structurally_multi_tree", np.where(active >= 2, "soft_shared_only", "root_specific"))

    raw = ad.read_h5ad(args.raw_h5ad, backed="r")
    raw_type = raw.obs[["cell_type_coarse"]].copy()
    raw_type.index = raw_type.index.astype(str)
    raw.file.close()
    mapping = pd.read_csv(args.method_dir / "cell_to_metacell_mapping.csv", usecols=["cell_name", "metacell_id"], dtype=str)
    mapping = mapping.join(raw_type, on="cell_name", how="left")
    subtype_entropy = {}
    for unit, part in mapping[mapping.metacell_id.isin(covered)].groupby("metacell_id"):
        p = part.cell_type_coarse.astype(object).fillna("unknown").value_counts(normalize=True).to_numpy(float)
        subtype_entropy[str(unit)] = float(-np.sum(p * np.log(p + 1e-12)) / np.log(max(len(p), 2)))

    node_degree = {}
    for root_id, part in nodes.groupby("root_id"):
        graph = nx.Graph(edges[edges.root_id.astype(str).eq(str(root_id))][["source", "target"]].astype(int).itertuples(index=False, name=None))
        for row in part.itertuples(index=False):
            node_degree.setdefault(str(row.unit_id), []).append(graph.degree[int(row.node_id)])
    boundary = np.array([any(degree != 2 for degree in node_degree.get(unit, [])) for unit in covered])
    soft_bio = pd.DataFrame({
        "unit_id": covered, "sharing_group": sharing_group,
        "soft_membership_entropy": entropy,
        "n_soft_roots_ge_0.20": active,
        "n_structural_T0_roots": structural,
        "raw_classifier_subtype_entropy": [subtype_entropy.get(unit, np.nan) for unit in covered],
        "topological_boundary_node": boundary,
        "spatial_block": obs.loc[covered, "spatial_block"].to_numpy(int),
    })
    root_centroids = {root: obs.loc[nodes[nodes.root_id.astype(str).eq(root)].unit_id.astype(str).unique(), ["spatial_x", "spatial_y"]].mean().to_numpy() for root in roots}
    xy = obs.loc[covered, ["spatial_x", "spatial_y"]].to_numpy(float)
    centroid_dist = np.column_stack([np.linalg.norm(xy - root_centroids[root], axis=1) for root in roots])
    soft_bio["spatial_root_centroid_distance_gap"] = np.sort(centroid_dist, axis=1)[:, 1] - np.sort(centroid_dist, axis=1)[:, 0]
    soft_bio.to_csv(args.output / "P8_SOFT_SHARING_BIOLOGY.csv", index=False)
    soft_summary = soft_bio.groupby("sharing_group", sort=True).agg(
        n_metacells=("unit_id", "size"),
        membership_entropy_mean=("soft_membership_entropy", "mean"),
        subtype_entropy_mean=("raw_classifier_subtype_entropy", "mean"),
        boundary_fraction=("topological_boundary_node", "mean"),
        spatial_centroid_gap_mean=("spatial_root_centroid_distance_gap", "mean"),
    ).reset_index()
    soft_summary.to_csv(args.output / "P8_SOFT_SHARING_SUMMARY.csv", index=False)

    branches_specialized = int(pd.DataFrame(branch_summary).passes_specialization_gate.sum())
    progression_by_branch = progression.groupby(["root_id", "branch_id"]).qualifying_progression.any()
    branches_progressing = int(progression_by_branch.sum())
    root_pairs_pass = int(root_profile.passes_primary_gate.sum())
    qualifying_full = sensitivity[sensitivity.full_effect_qualifies]
    lco_fraction = float(qualifying_full.sensitivity_survives.mean()) if len(qualifying_full) else 0.0
    gates = {
        "root_profile_distinction": root_pairs_pass >= 2,
        "branch_specialization": branches_specialized >= 4,
        "ordered_progression": branches_progressing >= 4,
        "leave_construction_genes_out": lco_fraction >= 0.70,
    }
    decision = {
        "stage": "P8_MULTI_TREE_BIOLOGICAL_VALIDATION",
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "root_pairs_passing": root_pairs_pass,
        "root_pairs_required": 2,
        "branches_with_specialization": branches_specialized,
        "branches_with_specialization_required": 4,
        "branches_with_ordered_progression": branches_progressing,
        "branches_with_ordered_progression_required": 4,
        "leave_construction_genes_out_survival_fraction": lco_fraction,
        "leave_construction_genes_out_required": 0.70,
        "n_primary_independent_scores": len(primary_ids),
        "strict_external_assay_independence": False,
        "strict_independence_limitation": "candidate PCA geometry used all panel genes",
        "fine_subtype_names_authorized": False,
        "P9_unlocked": bool(all(gates.values())),
        "algorithm_modified": False,
    }
    (args.output / "P8_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    report = f"""# P8 Biological Validation Report

## Hard-gate result

**{decision['decision']}**. Root profile pairs passing: {root_pairs_pass}/3; specialized branches: {branches_specialized}/7; branches with ordered independent progression: {branches_progressing}/7; leave-HVG-out survival: {lco_fraction:.1%}.

## Evidence boundary

All roots, branches, state steps, and tau0 values were frozen before scoring. Primary evidence uses external curated and Hallmark definitions after removing the 50 exact branch-assignment HVGs. Reactome and GO BP are secondary. Because all 313 panel genes contributed to candidate PCA geometry, these expression results reduce circularity but are not a fully independent assay.

Raw-cell subtype annotations remain mixed and are not used to name the trees. This analysis concerns three root-conditioned trajectories within the lymphoid construction scope.

## P9 rule

P9 is {'unlocked' if decision['P9_unlocked'] else 'not unlocked'}. No algorithm, threshold, branch, or biological label may be changed to alter this decision.
"""
    (args.output / "P8_BIOLOGICAL_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")

    manifest_paths = [
        phase_a / "T0_STATE_NODE_ASSIGNMENTS.csv", phase_a / "T0_FINE_TRAJECTORY_EDGES.csv",
        phase_a / "P6_FROZEN_SOFT_MEMBERSHIP.csv", args.output / "P8_PREREGISTRATION.json",
        args.output / "P8_CURATED_IMMUNE_SIGNATURES.csv",
        gene_dir / "Hallmark_2020.gmt", gene_dir / "Reactome_2022.gmt",
        gene_dir / "GO_Biological_Process_2023.gmt",
    ]
    pd.DataFrame([{"file": str(path.resolve()), "sha256": sha256(path)} for path in manifest_paths]).to_csv(args.output / "P8_INPUT_MANIFEST.csv", index=False)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
