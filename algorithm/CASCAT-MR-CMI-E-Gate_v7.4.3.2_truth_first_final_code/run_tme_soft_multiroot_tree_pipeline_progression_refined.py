#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CASCAT-MR-CMI-E-Gate v7.4.3.2 Truth-First corrected patch.

The structural base builds TME-aware metacells, process-constrained absorbing
Markov memberships, root-local support trees, progression-aware refinements, and
CMI-E reliability evidence.  v7.4 then applies explicit root-validity, edge
hard-veto, threshold-stability, trajectory-existence, molecular-program, and
process-distinctness gates.  A root is allowed to end as ``insufficient_support``;
no tree is forced merely to satisfy one-root-one-tree coverage.

Synthetic truth is used only when explicitly supplied in benchmark mode.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

import run_tme_soft_multiroot_tree_pipeline as legacy
import cascat_v7_3_biological_markov as v71bio
import generate_v7_4_biological_interpretation as v72viz
import cascat_v7_4_2_truth_first as v74truth


def load_root_table_any_format(path: str | Path) -> pd.DataFrame:
    """Load root tables without changing their contents.

    Supported: CSV/TXT, TSV, XLSX, XLS. The legacy CLI argument remains
    ``--roots-csv`` for backward compatibility.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".csv", ".txt", ""}:
        return pd.read_csv(p)
    if suffix == ".tsv":
        return pd.read_csv(p, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    raise ValueError(f"Unsupported root-table format: {p}")


@dataclass
class ProgressionConfig(legacy.Config):
    # Selected by the 7-dataset OFAT sensitivity analysis. A sparse
    # metacell-level candidate graph (k=6) improved overall and topology while
    # preserving root recovery; theta_A=0.30 keeps shared-state prevalence
    # interpretable without changing benchmark accuracy.
    k_neighbors: int = 6
    theta_A: float = 0.30
    enable_progression_aware_tree: bool = True
    progression_source: str = "auto"
    lambda_prog: float = 0.20
    theta_progression_jump: Optional[float] = None
    theta_membership_jump: float = 0.25
    w_jump: float = 1.0
    w_smooth: float = 0.5
    w_membership: float = 0.5
    enable_interaction_v2: bool = True
    interaction_refinement_mode: str = "conservative"
    lambda_dir_guard: float = 0.15
    lambda_stability: float = 0.10
    theta_promotion_high: float = 0.15
    theta_inhibition_high: float = 0.20
    theta_conflict_high: float = 0.20
    theta_merge_high: float = 0.15
    enable_acceptance_rule: bool = True
    selected_pseudotime_method: str = "hybrid"
    hybrid_alpha: float = 0.50
    tolerance_direction: float = 0.01
    tolerance_pseudotime: float = 0.02
    tolerance_topology: float = 0.01
    allow_oracle_construction: bool = False
    enable_backbone_pruning: bool = True
    enable_confidence_core_backbone: bool = True
    core_edge_confidence_quantile: float = 0.60
    core_edge_confidence_floor: float = 0.45
    core_min_nodes: int = 3
    core_w_global: float = 0.30
    core_w_rootprog: float = 0.20
    core_w_membership: float = 0.20
    core_w_progression: float = 0.15
    core_w_inverse_cost: float = 0.15
    min_branch_nodes: int = 1
    min_branch_pseudotime_span: float = 0.0
    collapse_degree2: bool = True
    score_backbone_not_full: bool = True
    enable_segment_split: bool = True
    segment_split_min_nodes: int = 4
    segment_split_pseudotime_span: float = 0.25
    enable_constrained_pseudotime: bool = True
    constrained_mu: float = 0.60
    constrained_eta: float = 0.30
    constrained_laplacian: float = 0.35
    constrained_root_anchor: float = 25.0
    constrained_monotone_eps: float = 1e-4
    enable_cmi_e: bool = True
    # v6 CMI-E is a confounding gate, not a naive edge-attraction term.
    # lambda_cmi_e is retained for backward compatibility but is not used by v6 cost.
    lambda_cmi_e: float = 0.12
    lambda_cmi_e_drop: float = 0.08
    lambda_cmi_e_trust: float = 0.02
    cmi_e_n_bins: int = 3
    cmi_e_env_method: str = "pca"
    # Zero means all available expression features and all valid edges.
    cmi_e_top_genes: int = 0
    cmi_e_min_valid_edges: int = 0
    cmi_e_use_signature_env: bool = True

    # ------------------------------------------------------------------
    # v7.4.2 Truth-First evidence gates. These parameters never force a tree.
    # A root/edge/tree that fails remains unsupported rather than rescued.
    # ------------------------------------------------------------------
    truth_min_root_quality: float = 0.50
    truth_root_max_projection_nn_factor: float = 6.0
    truth_min_trajectory_edges: int = 3
    truth_min_trajectory_nodes: int = 4
    truth_rescue_min_confidence: float = 0.55
    truth_rescue_min_stability: float = 0.80
    truth_min_edge_stability: float = 0.60
    truth_min_s_global_floor: float = 0.03
    truth_s_global_quantile_floor: float = 0.05
    # v7.4.2: tiny trees must use dataset-level reference floors instead of
    # self-defined adaptive thresholds.
    truth_min_edges_for_adaptive_threshold: int = 5
    truth_max_cost_median_factor: float = 8.0
    truth_max_cmi_conflict: float = 0.25
    truth_program_min_genes: int = 2
    truth_min_program_nodes: int = 4
    truth_min_program_spearman: float = 0.40
    truth_max_program_q: float = 0.10
    truth_interface_min_mass: float = 0.10
    truth_min_interface_support_rate: float = 0.10
    truth_distinct_program_distance: float = 0.25
    truth_distinct_max_program_diff: float = 0.50
    # v7.4.2: internal signature correlations are coherence evidence, not
    # independent process validation.  Level 2 requires an optional external or
    # orthogonal evidence CSV unless explicitly disabled.
    truth_require_external_process_evidence_for_level2: bool = True
    truth_external_program_evidence_csv: str = ""


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    lo, hi = float(np.nanmin(values[finite])), float(np.nanmax(values[finite]))
    out = np.zeros_like(values)
    if hi > lo:
        out[finite] = (values[finite] - lo) / (hi - lo)
    out[~finite] = 1.0
    return out




# -----------------------------------------------------------------------------
# Environment-adjusted CMI support (CMI-E) for multi-root metacell backbones
# -----------------------------------------------------------------------------

_TME_GENE_SIGNATURES: dict[str, list[str]] = {
    "Hypoxia": ["HIF1A", "VEGFA", "LDHA", "PGK1", "BNIP3", "CA9", "SLC2A1", "ENO1", "ALDOA", "PDK1"],
    "Glycolysis": ["ALDOA", "ENO1", "GAPDH", "GPI", "HK1", "HK2", "LDHA", "PFKL", "PFKP", "PGK1", "PKM", "TPI1"],
    "Oxidative_Stress": ["SOD1", "SOD2", "CAT", "GPX1", "GPX4", "PRDX1", "TXN", "NQO1", "HMOX1", "GSR"],
    "Effector_T": ["IFNG", "GZMA", "GZMB", "PRF1", "CD8A", "CD8B", "CXCL9", "CXCL10", "TBX21", "STAT1"],
    "Immune_Checkpoint": ["PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "TOX", "ENTPD1"],
    "M2_Macrophage": ["CD163", "MRC1", "MSR1", "CD68", "TGFB1", "IL10", "ARG1", "VEGFA", "CCL2", "CSF1R"],
    "CAF": ["ACTA2", "FAP", "PDGFRA", "PDGFRB", "COL1A1", "COL1A2", "FN1", "THY1", "VIM", "POSTN"],
    "EMT": ["VIM", "CDH2", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "FN1", "MMP2", "MMP9"],
    "ECM_Remodeling": ["COL1A1", "COL1A2", "COL3A1", "FN1", "MMP2", "MMP9", "MMP14", "LOX", "LOXL2", "TNC"],
    "VEGF_Signaling": ["VEGFA", "VEGFB", "VEGFC", "FLT1", "KDR", "FLT4", "NRP1", "NRP2", "PGF", "ANGPT2"],
}

_TME_SIGNATURE_DIMENSIONS: dict[str, list[str]] = {
    "Metabolic_Stress": ["Hypoxia", "Glycolysis", "Oxidative_Stress"],
    "Immune": ["Effector_T", "Immune_Checkpoint", "M2_Macrophage"],
    "Mesenchymal": ["CAF", "EMT", "ECM_Remodeling"],
    "Angiogenesis": ["VEGF_Signaling"],
}


def _as_dense_float(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=float)


def _standardize_columns(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)


def _first_pc_score(X: np.ndarray) -> np.ndarray:
    Xs = _standardize_columns(X)
    if Xs.ndim != 2 or min(Xs.shape) == 0:
        return np.zeros(Xs.shape[0] if Xs.ndim == 2 else 0, dtype=float)
    try:
        u, svals, _ = np.linalg.svd(Xs, full_matrices=False)
        if len(svals) == 0:
            return np.zeros(Xs.shape[0], dtype=float)
        return _normalise(u[:, 0] * svals[0])
    except np.linalg.LinAlgError:
        return _normalise(np.nanmean(Xs, axis=1))


def _feature_matrix_for_cmi_e(adata, emb: Optional[np.ndarray], cfg: ProgressionConfig) -> tuple[np.ndarray, str]:
    """Return node-by-feature matrix for CMI-E support estimation.

    We prefer expression features because CASCAT CMI is expression-profile based.
    To keep the estimator stable on small metacell graphs, only the top-variable
    genes are retained. If expression is unavailable, fall back to the embedding.
    """
    X = None
    source = "expression"
    try:
        X = _as_dense_float(adata.X)
    except Exception:
        X = None
    if X is None or X.ndim != 2 or X.shape[0] != adata.n_obs or X.shape[1] < 2:
        X = np.asarray(emb, dtype=float) if emb is not None else np.zeros((adata.n_obs, 1), dtype=float)
        source = "embedding_fallback"
    requested_top = int(getattr(cfg, "cmi_e_top_genes", 0) or 0)
    top = X.shape[1] if requested_top <= 0 else max(2, requested_top)
    if requested_top > 0 and X.shape[1] > top:
        var = np.nanvar(X, axis=0)
        keep = np.argsort(-var)[:top]
        X = X[:, keep]
        source += f"_top{top}"
    return _standardize_columns(X), source


def _environment_from_tme_signatures(adata, X: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str, dict[str, int]]:
    """Construct TME environment score from built-in signatures when genes exist."""
    if not hasattr(adata, "var_names"):
        return None, None, "no_var_names", {}
    var_lookup = {str(g).upper(): i for i, g in enumerate(adata.var_names)}
    sig_scores: dict[str, np.ndarray] = {}
    hit_counts: dict[str, int] = {}
    for name, genes in _TME_GENE_SIGNATURES.items():
        idx = [var_lookup[g.upper()] for g in genes if g.upper() in var_lookup and var_lookup[g.upper()] < X.shape[1]]
        hit_counts[name] = len(idx)
        if len(idx) >= 2:
            sig_scores[name] = np.nanmean(X[:, idx], axis=1)
    dim_scores = []
    for _, sigs in _TME_SIGNATURE_DIMENSIONS.items():
        available = [sig_scores[s] for s in sigs if s in sig_scores]
        if available:
            dim_scores.append(np.nanmean(np.vstack(available), axis=0))
    if len(dim_scores) == 0:
        return None, None, "signature_genes_insufficient", hit_counts
    D = np.vstack(dim_scores).T
    score = _first_pc_score(D)
    return score, None, "tme_signature_pca", hit_counts


def _environment_covariates_for_cmi_e(adata, X: np.ndarray, emb: Optional[np.ndarray], cfg: ProgressionConfig) -> tuple[np.ndarray, np.ndarray, str, dict[str, object]]:
    """Build the environmental variable E used by CMI-E.

    Priority:
    1) pre-existing adata.obs['env_score']/['env_bin'];
    2) TME gene-signature score when marker genes are present;
    3) first PC of expression/embedding as a benchmark-safe fallback.
    """
    info: dict[str, object] = {}
    if "env_score" in adata.obs:
        score = pd.to_numeric(adata.obs["env_score"], errors="coerce").to_numpy(dtype=float)
        source = "obs_env_score"
    elif bool(getattr(cfg, "cmi_e_use_signature_env", True)):
        try:
            X_sig = _as_dense_float(adata.X)
        except Exception:
            X_sig = X
        score, _, source, hits = _environment_from_tme_signatures(adata, X_sig)
        info["signature_hits"] = hits
        if score is None:
            base = X if X.shape[1] >= 2 else np.asarray(emb if emb is not None else X, dtype=float)
            score = _first_pc_score(base)
            source = "fallback_feature_pc1"
    else:
        base = X if X.shape[1] >= 2 else np.asarray(emb if emb is not None else X, dtype=float)
        score = _first_pc_score(base)
        source = "feature_pc1"
    score = _normalise(np.asarray(score, dtype=float))
    if "env_bin" in adata.obs:
        bins = pd.to_numeric(adata.obs["env_bin"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        q = max(2, int(getattr(cfg, "cmi_e_n_bins", 3)))
        try:
            score_series = pd.Series(score, dtype=float)
            bins = (
                pd.qcut(score_series, q=q, labels=False, duplicates="drop")
                .fillna(0)
                .astype(int)
                .to_numpy()
            )
        except Exception:
            bins = np.zeros(len(score), dtype=int)
    bins = np.asarray(bins, dtype=int)
    info["env_source"] = source
    info["n_env_bins_actual"] = int(len(set(bins.tolist()))) if len(bins) else 0
    return score, bins, source, info


def _one_hot_bins(bins: np.ndarray) -> np.ndarray:
    vals = sorted(set(np.asarray(bins, dtype=int).tolist()))
    if len(vals) <= 1:
        return np.zeros((len(bins), 0), dtype=float)
    # Drop the first level to avoid exact collinearity with the intercept.
    cols = [(bins == v).astype(float) for v in vals[1:]]
    return np.vstack(cols).T if cols else np.zeros((len(bins), 0), dtype=float)


def _residualize_features(X: np.ndarray, design: np.ndarray) -> np.ndarray:
    design = np.asarray(design, dtype=float)
    X = np.asarray(X, dtype=float)
    try:
        beta = np.linalg.lstsq(design, X, rcond=None)[0]
        R = X - design @ beta
    except np.linalg.LinAlgError:
        R = X - np.nanmean(X, axis=0, keepdims=True)
    return _standardize_columns(R)


def _row_correlation_to_mi(row_i: np.ndarray, row_j: np.ndarray) -> tuple[float, float]:
    xi = np.asarray(row_i, dtype=float)
    xj = np.asarray(row_j, dtype=float)
    xi = xi - np.nanmean(xi)
    xj = xj - np.nanmean(xj)
    denom = float(np.linalg.norm(xi) * np.linalg.norm(xj))
    if denom <= 1e-12:
        return 0.0, 0.0
    r = float(np.clip(np.dot(xi, xj) / denom, -0.999999, 0.999999))
    mi = float(max(0.0, -0.5 * math.log(max(1.0 - r * r, 1e-12))))
    return mi, r


def compute_cmi_e_edge_support(
    adata,
    edges_df: pd.DataFrame,
    A: np.ndarray,
    roots: list[int],
    emb: Optional[np.ndarray],
    cfg: ProgressionConfig,
) -> pd.DataFrame:
    """CMI-E evidence for multi-root metacell edge reliability.

    v6 treats CMI-E as a confounding-aware gate rather than a naive attraction
    score.  For each root k and candidate edge (i, j), it estimates two
    Gaussian-CMI-like dependencies from expression/embedding profiles:

      raw CMI: residualise profiles against root membership R_k only.
      env CMI: residualise profiles against R_k and environmental state E.

    A large raw-to-env drop indicates that the edge dependency is likely driven
    by the shared microenvironment rather than a stable root-specific trajectory
    relation.  A trustworthy edge should have high env-CMI and small drop.

    The returned table contains raw/env supports, env_drop, env_trust, and edge
    annotations.  It does not use benchmark truth.
    """
    if not bool(getattr(cfg, "enable_cmi_e", True)):
        return pd.DataFrame()
    if edges_df is None or edges_df.empty:
        return pd.DataFrame()

    X, feature_source = _feature_matrix_for_cmi_e(adata, emb, cfg)
    env_score, env_bin, env_source, env_info = _environment_covariates_for_cmi_e(adata, X, emb, cfg)
    bin_oh = _one_hot_bins(env_bin)
    primary = np.argmax(A, axis=1) if A.size else np.zeros(adata.n_obs, dtype=int)

    rows = []
    for k, root in enumerate(roots):
        root_membership = A[:, k] if k < A.shape[1] else np.zeros(adata.n_obs, dtype=float)

        raw_design = np.vstack([np.ones(adata.n_obs), _normalise(root_membership)]).T
        env_cols = [np.ones(adata.n_obs), _normalise(root_membership), _normalise(env_score)]
        if bin_oh.shape[1] > 0:
            env_cols.extend([bin_oh[:, j] for j in range(bin_oh.shape[1])])
        env_design = np.vstack(env_cols).T

        R_raw = _residualize_features(X, raw_design)
        R_env = _residualize_features(X, env_design)
        sub_rows = []

        for row in edges_df.itertuples(index=False):
            i, j = int(row.source), int(row.target)
            if i >= R_env.shape[0] or j >= R_env.shape[0]:
                continue
            cmi_raw, corr_raw = _row_correlation_to_mi(R_raw[i], R_raw[j])
            cmi_env, corr_env = _row_correlation_to_mi(R_env[i], R_env[j])
            if cmi_raw > cfg.eps:
                env_drop = float(np.clip((cmi_raw - cmi_env) / (cmi_raw + cfg.eps), 0.0, 1.0))
            else:
                env_drop = 0.0
            cross_root = bool(primary[i] != primary[j])
            prog_gap = 0.0
            try:
                prog_gap = abs(float(root_membership[i]) - float(root_membership[j]))
            except Exception:
                prog_gap = 0.0
            sub_rows.append(
                {
                    "root_pos": int(k),
                    "root_index": int(root),
                    "source": int(i),
                    "target": int(j),
                    "source_name": str(adata.obs_names[i]),
                    "target_name": str(adata.obs_names[j]),
                    "cmi_raw": float(cmi_raw),
                    "cmi_env": float(cmi_env),
                    "cmi_raw_corr": float(corr_raw),
                    "cmi_env_corr": float(corr_env),
                    # Backward-compatible aliases used by earlier v5 diagnostics.
                    "cmi_e_raw": float(cmi_env),
                    "cmi_e_corr": float(corr_env),
                    "cmi_env_drop": float(env_drop),
                    "cross_root_edge": cross_root,
                    "root_membership_gap": float(prog_gap),
                    "env_source": env_source,
                    "feature_source": feature_source,
                    "n_env_bins_actual": int(env_info.get("n_env_bins_actual", 0)),
                }
            )

        if not sub_rows:
            continue

        raw_vals = np.asarray([r["cmi_raw"] for r in sub_rows], dtype=float)
        env_vals = np.asarray([r["cmi_env"] for r in sub_rows], dtype=float)
        raw_support = _normalise(raw_vals)
        env_support = _normalise(env_vals)

        for r, sr, se in zip(sub_rows, raw_support, env_support):
            drop = float(r["cmi_env_drop"])
            trust = float(se) * (1.0 - drop)
            conflict = float(se) * drop
            if bool(r["cross_root_edge"]) and conflict > 0.25:
                label = "cross_root_environment_conflict"
            elif trust > 0.50 and drop < 0.35:
                label = "environment_stable_support"
            elif drop >= 0.50:
                label = "environment_confounded"
            else:
                label = "weak_or_ambiguous"
            r["cmi_raw_support"] = float(sr)
            r["cmi_env_support"] = float(se)
            r["cmi_e_support"] = float(se)
            r["cmi_env_trust"] = trust
            r["cmi_env_conflict"] = conflict
            r["cmi_e_edge_label"] = label
            rows.append(r)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out

def _natural_time(values: Iterable[object]) -> np.ndarray:
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return _normalise(numeric.to_numpy(dtype=float))
    order = sorted(series.astype(str).unique(), key=lambda x: (len(x), x))
    lookup = {value: pos for pos, value in enumerate(order)}
    return _normalise(series.astype(str).map(lookup).to_numpy(dtype=float))


def _truth_vector(path: Optional[Path], adata) -> Optional[np.ndarray]:
    if path is None:
        return None
    table = pd.read_csv(path)
    id_col = next((c for c in ("node_name", "cell_id", "obs_name", "id") if c in table), None)
    value_col = next(
        (c for c in ("pseudotime", "true_pseudotime", "latent_time", "time") if c in table),
        None,
    )
    if value_col is None:
        raise ValueError(f"{path}: no pseudotime column found")
    if id_col is None:
        if len(table) != adata.n_obs:
            raise ValueError(f"{path}: row count does not match adata")
        return _normalise(pd.to_numeric(table[value_col], errors="coerce").to_numpy())
    mapped = table.set_index(table[id_col].astype(str))[value_col]
    return _normalise(pd.to_numeric(mapped.reindex(adata.obs_names.astype(str)), errors="coerce").to_numpy())


def _graph_distance_matrix(edges_df: pd.DataFrame, roots: list[int], n_nodes: int, eps: float) -> np.ndarray:
    rows, cols, vals = [], [], []
    for row in edges_df.itertuples(index=False):
        i, j = int(row.source), int(row.target)
        d = max(float(getattr(row, "d_global", 1.0)), eps)
        rows.extend((i, j))
        cols.extend((j, i))
        vals.extend((d, d))
    graph = csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    distances = np.asarray(shortest_path(graph, directed=False, indices=roots)).T
    return np.column_stack([_normalise(distances[:, k]) for k in range(len(roots))])


def _proxy_smoothness(values: np.ndarray, edges_df: pd.DataFrame, membership: np.ndarray) -> float:
    deltas = []
    for row in edges_df.itertuples(index=False):
        i, j = int(row.source), int(row.target)
        if max(membership[i], membership[j]) < 0.05:
            continue
        deltas.append(abs(float(values[i]) - float(values[j])))
    return 1.0 - float(np.mean(deltas)) if deltas else -math.inf


def compute_progression_scores(
    adata,
    A: np.ndarray,
    roots: list[int],
    emb: np.ndarray,
    edges_df: pd.DataFrame,
    cfg: ProgressionConfig,
    truth_pseudotime: Optional[np.ndarray] = None,
    time_key: Optional[str] = None,
    mode: str = "auto",
    allow_oracle_construction: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, np.ndarray], dict[int, str]]:
    """Generate progression proxies and select one per root.

    Truth pseudotime is never allowed to influence construction unless
    allow_oracle_construction=True and mode requests truth explicitly.  When truth
    is supplied but not allowed, it is used only for diagnostic columns in the
    source summary.
    """
    n, k_count = A.shape
    graph_distance = _graph_distance_matrix(edges_df, roots, n, cfg.eps)
    time_values = _natural_time(adata.obs[time_key]) if time_key and time_key in adata.obs else None
    primary = np.argmax(A, axis=1)
    candidate_rows: list[dict[str, object]] = []
    selected: dict[int, np.ndarray] = {}
    selected_sources: dict[int, str] = {}
    summaries: list[dict[str, object]] = []

    for k, root in enumerate(roots):
        signature = legacy.root_signature_compatibility(emb, A[:, k], cfg)
        candidates: dict[str, np.ndarray] = {
            "membership_distance": _normalise(1.0 - A[:, k]),
            "root_signature_proxy_forward": _normalise(1.0 - signature),
            "root_signature_proxy_reverse": _normalise(signature),
            "graph_distance": graph_distance[:, k],
        }
        if time_values is not None:
            candidates["timepoint_proxy"] = time_values
        if truth_pseudotime is not None and allow_oracle_construction:
            candidates["truth_pseudotime_oracle"] = _normalise(truth_pseudotime)

        requested = mode
        aliases = {
            "truth": "truth_pseudotime_oracle",
            "timepoint": "timepoint_proxy",
            "membership": "membership_distance",
            "root_signature": "root_signature_proxy_forward",
        }
        requested = aliases.get(requested, requested)
        if requested == "auto":
            if truth_pseudotime is not None and allow_oracle_construction:
                choice = "truth_pseudotime_oracle"
            elif time_values is not None:
                choice = "timepoint_proxy"
            else:
                internal = {
                    name: _proxy_smoothness(values, edges_df, A[:, k])
                    + 0.15 * float(np.corrcoef(values, graph_distance[:, k])[0, 1])
                    for name, values in candidates.items()
                }
                choice = max(internal, key=lambda name: np.nan_to_num(internal[name], nan=-math.inf))
        elif requested in candidates:
            choice = requested
        elif requested == "truth_pseudotime_oracle" and truth_pseudotime is not None and not allow_oracle_construction:
            warnings.warn("truth pseudotime was supplied but oracle construction is disabled; using graph_distance")
            choice = "graph_distance"
        else:
            warnings.warn(f"progression source {mode!r} unavailable; using graph_distance")
            choice = "graph_distance"

        selected[k] = candidates[choice]
        selected_sources[k] = choice
        for source, values in candidates.items():
            smoothness = _proxy_smoothness(values, edges_df, A[:, k])
            summaries.append(
                {
                    "root_pos": k,
                    "root_id": str(adata.obs_names[root]),
                    "progression_source": source,
                    "selected": source == choice,
                    "internal_smoothness": smoothness,
                    "truth_spearman_diagnostic": (
                        float(spearmanr(values, truth_pseudotime, nan_policy="omit").statistic)
                        if truth_pseudotime is not None
                        else np.nan
                    ),
                }
            )
        ranks = pd.Series(selected[k]).rank(method="average", pct=True).to_numpy()
        for i in range(n):
            candidate_rows.append(
                {
                    "node_id": i,
                    "node_name": str(adata.obs_names[i]),
                    "root_pos": k,
                    "root_id": str(adata.obs_names[root]),
                    "g_ik": float(selected[k][i]),
                    "progression_source": choice,
                    "progression_rank": float(ranks[i]),
                    "A_ik": float(A[i, k]),
                    "primary_root": int(primary[i]),
                    "timepoint": str(adata.obs.iloc[i][time_key]) if time_values is not None else "",
                }
            )
    return pd.DataFrame(candidate_rows), pd.DataFrame(summaries), selected, selected_sources


def _basin_nodes(
    A: np.ndarray,
    root_pos: int,
    root_idx: int,
    cfg: ProgressionConfig,
    active_assignment: Optional[np.ndarray] = None,
    compatibility_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if active_assignment is not None:
        return v71bio.basin_nodes_from_active(active_assignment, root_pos, root_idx, compatibility_mask)
    primary = v71bio.safe_primary_roots(A, cfg.eps)
    nodes = set(map(int, np.flatnonzero(primary == root_pos)))
    nodes.add(int(root_idx))
    return np.asarray(sorted(nodes), dtype=int)

def _progression_threshold(
    edges_df: pd.DataFrame, basin: set[int], progression: np.ndarray, configured: Optional[float]
) -> float:
    if configured is not None:
        return float(configured)
    delta = [
        abs(float(progression[int(r.source)]) - float(progression[int(r.target)]))
        for r in edges_df.itertuples(index=False)
        if int(r.source) in basin and int(r.target) in basin
    ]
    return float(np.quantile(delta, 0.75)) if delta else 1.0


def compute_direction_penalty_for_edge(
    i: int,
    j: int,
    root_pos: int,
    progression_scores: dict[int, np.ndarray],
    A: np.ndarray,
    cfg: ProgressionConfig,
    theta_progression_jump: float,
) -> dict[str, float]:
    if root_pos not in progression_scores:
        return {"jump": 0.0, "smooth": 0.0, "membership": 0.0, "total": 0.0}
    delta_g = abs(float(progression_scores[root_pos][i]) - float(progression_scores[root_pos][j]))
    delta_a = abs(float(A[i, root_pos]) - float(A[j, root_pos]))
    jump = max(0.0, delta_g - theta_progression_jump)
    membership = max(0.0, delta_a - cfg.theta_membership_jump)
    total = cfg.w_jump * jump + cfg.w_smooth * delta_g + cfg.w_membership * membership
    return {"jump": jump, "smooth": delta_g, "membership": membership, "total": total}


def _edge_key(i: int, j: int) -> tuple[int, int]:
    return tuple(sorted((int(i), int(j))))


def compute_multiple_pseudotimes(
    tree: nx.Graph,
    root_idx: int,
    root_pos: int,
    A: np.ndarray,
    progression_scores: dict[int, np.ndarray],
    cfg: ProgressionConfig,
    tree_label: str,
    truth_pseudotime: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    distance = nx.single_source_dijkstra_path_length(tree, root_idx, weight="weight") if root_idx in tree else {}
    nodes = sorted(tree.nodes())
    raw = np.asarray([distance.get(i, np.nan) for i in nodes], dtype=float)
    norm_tree = _normalise(raw)
    membership = _normalise(1.0 - A[nodes, root_pos])
    progression = _normalise(progression_scores.get(root_pos, membership)[nodes])
    hybrid = cfg.hybrid_alpha * norm_tree + (1.0 - cfg.hybrid_alpha) * progression
    methods = {
        "tree_distance": norm_tree,
        "membership": membership,
        "progression": progression,
        "hybrid": hybrid,
    }
    method = cfg.selected_pseudotime_method
    if method == "auto":
        if truth_pseudotime is not None:
            method = max(
                methods,
                key=lambda name: np.nan_to_num(
                    spearmanr(methods[name], truth_pseudotime[nodes], nan_policy="omit").statistic,
                    nan=-math.inf,
                ),
            )
        else:
            method = "hybrid"
    selected = methods.get(method, hybrid)
    return pd.DataFrame(
        {
            "root_pos": root_pos,
            "node_id": nodes,
            "tree_label": tree_label,
            "tree_distance_pseudotime": raw,
            "normalized_tree_distance": norm_tree,
            "membership_pseudotime": membership,
            "progression_score_pseudotime": progression,
            "hybrid_pseudotime": hybrid,
            "selected_pseudotime": selected,
            "selected_pseudotime_method": method,
        }
    )


def build_one_tree(
    adata,
    edges_df: pd.DataFrame,
    A: np.ndarray,
    root_idx: int,
    root_pos: int,
    emb: np.ndarray,
    domains,
    cfg: ProgressionConfig,
    nmf_edge_lookup: Optional[dict[tuple[int, int, int], float]] = None,
    progression_scores: Optional[dict[int, np.ndarray]] = None,
    progression_sources: Optional[dict[int, str]] = None,
    interaction_modifiers: Optional[pd.DataFrame] = None,
    cost_mode: str = "base",
    tree_label: str = "base",
    reference_edges: Optional[set[tuple[int, int]]] = None,
    interaction_scale: float = 1.0,
    truth_pseudotime: Optional[np.ndarray] = None,
    cmi_e_support: Optional[pd.DataFrame] = None,
    active_assignment: Optional[np.ndarray] = None,
    compatibility_mask: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, dict[str, object], nx.Graph, pd.DataFrame]:
    progression_scores = progression_scores or {}
    progression_sources = progression_sources or {}
    basin_arr = _basin_nodes(
        A, root_pos, root_idx, cfg,
        active_assignment=active_assignment,
        compatibility_mask=compatibility_mask,
    )
    basin = set(map(int, basin_arr))
    membership = A[:, root_pos]
    signature = legacy.root_signature_compatibility(emb, membership, cfg)
    theta = _progression_threshold(
        edges_df, basin, progression_scores.get(root_pos, np.zeros(A.shape[0])), cfg.theta_progression_jump
    )
    modifier_lookup: dict[tuple[int, int], object] = {}
    if interaction_modifiers is not None and not interaction_modifiers.empty:
        for row in interaction_modifiers.loc[interaction_modifiers["root_pos"] == root_pos].itertuples(index=False):
            modifier_lookup[_edge_key(row.source, row.target)] = row

    cmi_lookup: dict[tuple[int, int], object] = {}
    if cmi_e_support is not None and not cmi_e_support.empty and bool(getattr(cfg, "enable_cmi_e", True)):
        for row in cmi_e_support.loc[cmi_e_support["root_pos"].astype(int) == int(root_pos)].itertuples(index=False):
            cmi_lookup[_edge_key(row.source, row.target)] = row

    graph = nx.Graph()
    graph.add_nodes_from(basin_arr.tolist())
    for row in edges_df.itertuples(index=False):
        i, j = int(row.source), int(row.target)
        if i not in basin or j not in basin:
            continue
        if hasattr(row, "trajectory_edge_eligible") and not bool(row.trajectory_edge_eligible):
            continue
        kappa = float(getattr(row, "trajectory_compatibility", 1.0))
        if kappa <= 0.0:
            continue
        if nmf_edge_lookup and (root_pos, i, j) in nmf_edge_lookup:
            rootprog = float(nmf_edge_lookup[(root_pos, i, j)])
            rootprog_source = "root_specific_NMF"
        else:
            rootprog = math.sqrt(max(signature[i], cfg.eps) * max(signature[j], cfg.eps))
            rootprog_source = "embedding_signature_fallback"
        domain_penalty = float(domains is not None and domains[i] != domains[j])
        biological_penalty = 0.75 * (1.0 - kappa) if kappa < 1.0 else 0.0
        cost_base = (
            float(row.d_global)
            - cfg.lambda_A * math.log(float(membership[i] * membership[j]) + cfg.eps)
            - cfg.lambda_P * math.log(rootprog + cfg.eps)
            + cfg.lambda_D * domain_penalty
            + biological_penalty
        )
        penalty = compute_direction_penalty_for_edge(i, j, root_pos, progression_scores, A, cfg, theta)
        progression_penalty = (
            cfg.lambda_prog * penalty["total"]
            if cost_mode in {"progression", "interaction_v2", "oracle_direction"}
            and cfg.enable_progression_aware_tree
            else 0.0
        )

        p = h = f = m = b = 0.0
        direction_guard = stability = 0.0
        modifier_reason = "none"
        modifier = modifier_lookup.get(_edge_key(i, j))
        if cost_mode == "interaction_v2" and modifier is not None:
            p = float(getattr(modifier, "P_ij_v2", 0.0))
            h = float(getattr(modifier, "H_ij_v2", 0.0))
            f = float(getattr(modifier, "conflict_edge_score", 0.0))
            m = float(getattr(modifier, "merge_edge_score", 0.0))
            b = float(getattr(modifier, "collision_edge_score", 0.0))
            high_conf = (
                p >= cfg.theta_promotion_high
                or h >= cfg.theta_inhibition_high
                or f >= cfg.theta_conflict_high
                or m >= cfg.theta_merge_high
            )
            if cfg.interaction_refinement_mode == "aggressive":
                high_conf = True
            if not high_conf:
                p = h = f = m = b = 0.0
                modifier_reason = "low_confidence_no_change"
            else:
                modifier_reason = str(getattr(modifier, "modifier_reason", "high_confidence"))
                direction_guard = cfg.lambda_dir_guard * max(0.0, penalty["smooth"] - theta)
                if reference_edges is not None:
                    stability = -0.25 * cfg.lambda_stability if _edge_key(i, j) in reference_edges else cfg.lambda_stability

        modifier_total = interaction_scale * (
            cfg.lambda_inh * h
            - cfg.lambda_prom * p
            + cfg.lambda_conflict * f
            - cfg.lambda_merge * m
            + cfg.lambda_boundary * b
        )
        cmi_row = cmi_lookup.get(_edge_key(i, j))
        cmi_e_support_val = float(getattr(cmi_row, "cmi_e_support", 0.0)) if cmi_row is not None else 0.0
        cmi_e_raw_val = float(getattr(cmi_row, "cmi_e_raw", 0.0)) if cmi_row is not None else 0.0
        cmi_e_corr_val = float(getattr(cmi_row, "cmi_e_corr", 0.0)) if cmi_row is not None else 0.0
        cmi_raw_val = float(getattr(cmi_row, "cmi_raw", cmi_e_raw_val)) if cmi_row is not None else 0.0
        cmi_env_val = float(getattr(cmi_row, "cmi_env", cmi_e_raw_val)) if cmi_row is not None else 0.0
        cmi_env_drop_val = float(getattr(cmi_row, "cmi_env_drop", 0.0)) if cmi_row is not None else 0.0
        cmi_env_trust_val = float(getattr(cmi_row, "cmi_env_trust", cmi_e_support_val)) if cmi_row is not None else 0.0
        cmi_env_conflict_val = float(getattr(cmi_row, "cmi_env_conflict", 0.0)) if cmi_row is not None else 0.0
        cmi_edge_label = str(getattr(cmi_row, "cmi_e_edge_label", "none")) if cmi_row is not None else "none"
        cmi_e_penalty = 0.0
        if bool(getattr(cfg, "enable_cmi_e", True)) and cost_mode in {"progression", "interaction_v2", "oracle_direction"}:
            cmi_e_penalty = (
                float(getattr(cfg, "lambda_cmi_e_drop", 0.08)) * cmi_env_drop_val
                - float(getattr(cfg, "lambda_cmi_e_trust", 0.02)) * cmi_env_trust_val
            )
        cost_final = max(
            cfg.min_edge_cost,
            cost_base + progression_penalty + cmi_e_penalty + modifier_total + direction_guard + stability,
        )
        graph.add_edge(
            i,
            j,
            weight=cost_final,
            cost_base=cost_base,
            biological_penalty=biological_penalty,
            trajectory_compatibility=kappa,
            progression_penalty=progression_penalty,
            cmi_e_penalty=cmi_e_penalty,
            cmi_e_support=cmi_e_support_val,
            cmi_e_raw=cmi_e_raw_val,
            cmi_e_corr=cmi_e_corr_val,
            cmi_raw=cmi_raw_val,
            cmi_env=cmi_env_val,
            cmi_env_drop=cmi_env_drop_val,
            cmi_env_trust=cmi_env_trust_val,
            cmi_env_conflict=cmi_env_conflict_val,
            cmi_e_edge_label=cmi_edge_label,
            interaction_inhibition=h,
            interaction_promotion=p,
            interaction_conflict=f,
            interaction_merge=m,
            interaction_boundary=b,
            direction_guard_penalty=direction_guard,
            stability_penalty=stability,
            S_global=float(row.S_global),
            S_rootprog=rootprog,
            rootprog_source=rootprog_source,
            modifier_reason=modifier_reason,
        )

    disconnected_audit = {"n_components_before": 0, "dropped_nonroot_components": 0, "dropped_nonroot_nodes": 0}
    if bool(getattr(cfg, "keep_only_root_component", True)):
        graph, disconnected_audit = v71bio.keep_root_component_only(graph, root_idx)
    tree = nx.minimum_spanning_tree(graph, weight="weight") if graph.number_of_edges() else graph.copy()

    pseudotime = compute_multiple_pseudotimes(
        tree, root_idx, root_pos, A, progression_scores, cfg, tree_label, truth_pseudotime
    )
    pt = pseudotime.set_index("node_id")["selected_pseudotime"].to_dict() if not pseudotime.empty else {}
    rows = []
    for u, v, data in tree.edges(data=True):
        if pt.get(u, math.inf) <= pt.get(v, math.inf):
            source, target = u, v
        else:
            source, target = v, u
        rows.append(
            {
                "root_id": str(adata.obs_names[root_idx]),
                "root_pos": root_pos,
                "root_index": int(root_idx),
                "source": int(source),
                "target": int(target),
                "source_name": str(adata.obs_names[source]),
                "target_name": str(adata.obs_names[target]),
                "cost_base": float(data["cost_base"]),
                "base_cost": float(data["cost_base"]),
                "biological_penalty": float(data.get("biological_penalty", 0.0)),
                "trajectory_compatibility": float(data.get("trajectory_compatibility", 1.0)),
                "progression_penalty": float(data["progression_penalty"]),
                "cmi_e_penalty": float(data.get("cmi_e_penalty", 0.0)),
                "cmi_e_support": float(data.get("cmi_e_support", 0.0)),
                "cmi_e_raw": float(data.get("cmi_e_raw", 0.0)),
                "cmi_e_corr": float(data.get("cmi_e_corr", 0.0)),
                "cmi_raw": float(data.get("cmi_raw", data.get("cmi_e_raw", 0.0))),
                "cmi_env": float(data.get("cmi_env", data.get("cmi_e_raw", 0.0))),
                "cmi_env_drop": float(data.get("cmi_env_drop", 0.0)),
                "cmi_env_trust": float(data.get("cmi_env_trust", data.get("cmi_e_support", 0.0))),
                "cmi_env_conflict": float(data.get("cmi_env_conflict", 0.0)),
                "cmi_e_edge_label": str(data.get("cmi_e_edge_label", "none")),
                "interaction_inhibition": float(data["interaction_inhibition"]),
                "interaction_promotion": float(data["interaction_promotion"]),
                "interaction_conflict": float(data["interaction_conflict"]),
                "interaction_merge": float(data["interaction_merge"]),
                "interaction_boundary": float(data["interaction_boundary"]),
                "direction_guard_penalty": float(data["direction_guard_penalty"]),
                "stability_penalty": float(data["stability_penalty"]),
                "cost_final": float(data["weight"]),
                "edge_cost": float(data["weight"]),
                "modifier_total": float(data["weight"] - data["cost_base"] - data["progression_penalty"] - data.get("cmi_e_penalty", 0.0)),
                "S_global": float(data["S_global"]),
                "S_rootprog": float(data["S_rootprog"]),
                "rootprog_source": str(data["rootprog_source"]),
                "A_source_root": float(membership[source]),
                "A_target_root": float(membership[target]),
                "pseudotime_source": float(pt.get(source, np.nan)),
                "pseudotime_target": float(pt.get(target, np.nan)),
                "direction": "selected_progression_outward",
                "cost_mode": cost_mode,
                "tree_label": tree_label,
                "progression_source": progression_sources.get(root_pos, "missing"),
                "modifier_reason": str(data["modifier_reason"]),
            }
        )
    summary = {
        "root_id": str(adata.obs_names[root_idx]),
        "root_pos": root_pos,
        "root_index": int(root_idx),
        "basin_size": len(basin),
        "tree_nodes": tree.number_of_nodes(),
        "tree_edges": tree.number_of_edges(),
        "n_components": nx.number_connected_components(tree) if tree.number_of_nodes() else 0,
        "tree_label": tree_label,
        "cost_mode": cost_mode,
        "progression_source": progression_sources.get(root_pos, "missing"),
        "theta_progression_jump_used": theta,
        "direction_method": "selected_progression_outward",
        **disconnected_audit,
    }
    edge_df = pd.DataFrame(rows)
    if edge_df.empty:
        edge_df = pd.DataFrame(columns=[
            "root_id", "root_pos", "root_index", "source", "target",
            "source_name", "target_name", "cost_base", "base_cost",
            "biological_penalty", "trajectory_compatibility",
            "progression_penalty", "cmi_e_penalty", "cmi_e_support",
            "cmi_env_drop", "cmi_env_trust", "cmi_env_conflict",
            "interaction_inhibition", "interaction_promotion",
            "interaction_conflict", "interaction_merge", "interaction_boundary",
            "direction_guard_penalty", "stability_penalty", "cost_final",
            "edge_cost", "modifier_total", "S_global", "S_rootprog",
            "A_source_root", "A_target_root", "pseudotime_source",
            "pseudotime_target", "direction", "cost_mode", "tree_label",
            "progression_source", "modifier_reason",
        ])
    return edge_df, summary, tree, pseudotime

def _metadata_column(adata, names: tuple[str, ...]) -> Optional[str]:
    lower = {str(col).lower(): str(col) for col in adata.obs.columns}
    return next((lower[name] for name in names if name in lower), None)


def compute_interaction_scores_v2(
    adata,
    A: np.ndarray,
    roots: list[int],
    progression_scores: dict[int, np.ndarray],
    progression_trees: dict[int, nx.Graph],
    cfg: ProgressionConfig,
    active_assignment: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Compute auditable node-level v2 interaction scores and collision decisions."""
    n, k_count = A.shape
    cell_key = _metadata_column(adata, ("cell_type", "celltype", "annotation"))
    domain_key = _metadata_column(adata, ("domain", "region", "spatial_domain"))
    sample_key = _metadata_column(adata, ("sample", "slice", "sample_id"))
    P = np.zeros((n, k_count))
    H = np.zeros((n, k_count))
    F = np.zeros((n, k_count))
    M = np.zeros((n, k_count))
    B = np.zeros((n, k_count))
    rows, decisions = [], []

    for i in range(n):
        for k in range(k_count):
            for ell in range(k + 1, k_count):
                # v7.3: node-level tree-tree modifiers are computed only for
                # validated dual-root active assignments. Diffuse Markov ambiguity
                # alone cannot create interaction modifiers.
                if active_assignment is not None and not (
                    bool(active_assignment[i, k]) and bool(active_assignment[i, ell])
                ):
                    continue
                ak, al = float(A[i, k]), float(A[i, ell])
                coll_v1 = min(ak, al) * (1.0 - abs(ak - al))
                prog_consistency = 1.0 - abs(
                    float(progression_scores[k][i]) - float(progression_scores[ell][i])
                )
                prog_consistency = float(np.clip(prog_consistency, 0.0, 1.0))
                state_compat = 1.0 - abs(ak - al)
                spatial_compat = 1.0
                cell_compat = 1.0 if cell_key is None else 0.85
                domain_compat = 1.0 if domain_key is None else 0.85
                promotion_v1 = ak * al * state_compat
                promotion_v2 = promotion_v1 * prog_consistency * spatial_compat * cell_compat
                exclusivity = abs(ak - al)
                inhibition_v1 = max(al * (1.0 - ak), ak * (1.0 - al)) * exclusivity
                inhibition_v2 = inhibition_v1 * (1.0 - prog_consistency) * (1.0 - cell_compat + 0.25)
                collision_v2 = coll_v1 * max(spatial_compat, state_compat)
                merge = collision_v2 * prog_consistency * cell_compat * domain_compat
                conflict = collision_v2 * max(1.0 - prog_consistency, exclusivity)
                if collision_v2 < cfg.theta_collision:
                    decision = "weak_shared"
                elif merge >= max(conflict, cfg.theta_merge_high):
                    decision = "convergence_merge"
                elif conflict >= max(merge, cfg.theta_conflict_high):
                    decision = "conflict_boundary"
                else:
                    decision = "shared_boundary"
                confidence = float(max(merge, conflict, collision_v2))
                reason = f"collision={collision_v2:.3f}; merge={merge:.3f}; conflict={conflict:.3f}"
                for root_pos in (k, ell):
                    P[i, root_pos] = max(P[i, root_pos], promotion_v2)
                    H[i, root_pos] = max(H[i, root_pos], inhibition_v2)
                    if decision == "convergence_merge":
                        M[i, root_pos] = max(M[i, root_pos], merge)
                    elif decision == "conflict_boundary":
                        F[i, root_pos] = max(F[i, root_pos], conflict)
                    else:
                        B[i, root_pos] = max(B[i, root_pos], collision_v2)
                common = {
                    "node_id": i,
                    "node_name": str(adata.obs_names[i]),
                    "tree_k": k,
                    "tree_l": ell,
                    "collision_score_v1": coll_v1,
                    "collision_score_v2": collision_v2,
                    "promotion_score_v1": promotion_v1,
                    "promotion_score_v2": promotion_v2,
                    "inhibition_score_v1": inhibition_v1,
                    "inhibition_score_v2": inhibition_v2,
                    "merge_score": merge,
                    "conflict_score": conflict,
                    "progression_consistency": prog_consistency,
                    "spatial_compatibility": spatial_compat,
                    "celltype_compatibility": cell_compat,
                    "domain_compatibility": domain_compat,
                    "sample": str(adata.obs.iloc[i][sample_key]) if sample_key else "",
                    "cell_type": str(adata.obs.iloc[i][cell_key]) if cell_key else "",
                }
                rows.append(common)
                decisions.append(
                    {
                        **common,
                        "collision_decision": decision,
                        "collision_decision_confidence": confidence,
                        "reason": reason,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(decisions), {"P": P, "H": H, "F": F, "M": M, "B": B}


def build_interaction_edge_modifiers_v2(
    adata,
    edges_df: pd.DataFrame,
    A: np.ndarray,
    progression_scores: dict[int, np.ndarray],
    node_modifiers: dict[str, np.ndarray],
    domains,
    cfg: ProgressionConfig,
    sample_key: Optional[str] = None,
) -> pd.DataFrame:
    coords = legacy.get_spatial_coords(adata)
    samples = legacy.get_sample_labels(adata, sample_key)
    spatial_lengths = np.asarray(
        [np.linalg.norm(coords[int(r.source)] - coords[int(r.target)]) for r in edges_df.itertuples(index=False)]
    )
    length_scale = max(float(np.median(spatial_lengths[spatial_lengths > 0])) if np.any(spatial_lengths > 0) else 1.0, cfg.eps)
    rows = []
    for edge_pos, row in enumerate(edges_df.itertuples(index=False)):
        i, j = int(row.source), int(row.target)
        for k in range(A.shape[1]):
            dg = abs(float(progression_scores[k][i]) - float(progression_scores[k][j]))
            da = abs(float(A[i, k]) - float(A[j, k]))
            e_prog = math.exp(-dg / 0.20)
            e_membership = math.exp(-da)
            p_v1 = 0.5 * (node_modifiers["P"][i, k] + node_modifiers["P"][j, k])
            h_v1 = 0.5 * (node_modifiers["H"][i, k] + node_modifiers["H"][j, k])
            collision = 0.5 * (node_modifiers["B"][i, k] + node_modifiers["B"][j, k])
            merge = 0.5 * (node_modifiers["M"][i, k] + node_modifiers["M"][j, k])
            conflict = 0.5 * (node_modifiers["F"][i, k] + node_modifiers["F"][j, k])
            cross_slice = bool(samples is not None and samples[i] != samples[j])
            cross_domain = bool(domains is not None and domains[i] != domains[j])
            p_v2 = p_v1 * e_prog * e_membership
            h_v2 = h_v1 + max(0.0, dg - 0.25) + max(0.0, da - cfg.theta_membership_jump)
            h_v2 += 0.10 * float(cross_slice or cross_domain)
            evidence = {
                "promotion": p_v2,
                "inhibition": h_v2,
                "conflict": conflict,
                "merge": merge,
            }
            dominant = max(evidence, key=evidence.get)
            rows.append(
                {
                    "root_pos": k,
                    "source": i,
                    "target": j,
                    "source_name": str(adata.obs_names[i]),
                    "target_name": str(adata.obs_names[j]),
                    "P_ij_v1": p_v1,
                    "P_ij_v2": p_v2,
                    "H_ij_v1": h_v1,
                    "H_ij_v2": h_v2,
                    "collision_edge_score": collision,
                    "merge_edge_score": merge,
                    "conflict_edge_score": conflict,
                    "edge_progression_smoothness": e_prog,
                    "edge_membership_continuity": e_membership,
                    "edge_spatial_length": float(spatial_lengths[edge_pos] / length_scale),
                    "cross_slice": cross_slice,
                    "cross_domain": cross_domain,
                    "modifier_reason": dominant if evidence[dominant] > cfg.eps else "none",
                }
            )
    return pd.DataFrame(rows)


def _branch_table(
    adata,
    trees: dict[int, nx.Graph],
    pseudotimes: pd.DataFrame,
    edge_modifiers: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for root_pos, tree in trees.items():
        root_pt = pseudotimes.loc[pseudotimes["root_pos"] == root_pos].set_index("node_id")
        leaves = [node for node, degree in tree.degree() if degree == 1]
        root_candidates = root_pt.sort_values("selected_pseudotime").index.tolist()
        if not root_candidates:
            continue
        root = int(root_candidates[0])
        for branch_id, leaf in enumerate(leaves):
            try:
                nodes = nx.shortest_path(tree, root, leaf)
            except nx.NetworkXNoPath:
                continue
            keys = {_edge_key(a, b) for a, b in zip(nodes[:-1], nodes[1:])}
            edge_rows = edge_modifiers.loc[
                (edge_modifiers["root_pos"] == root_pos)
                & edge_modifiers.apply(lambda r: _edge_key(r.source, r.target) in keys, axis=1)
            ]
            mean_p = float(edge_rows["P_ij_v2"].mean()) if not edge_rows.empty else 0.0
            mean_h = float(edge_rows["H_ij_v2"].mean()) if not edge_rows.empty else 0.0
            mean_c = float(edge_rows["collision_edge_score"].mean()) if not edge_rows.empty else 0.0
            mean_s = float(edge_rows["edge_progression_smoothness"].mean()) if not edge_rows.empty else 1.0
            mean_a = float(edge_rows["edge_membership_continuity"].mean()) if not edge_rows.empty else 1.0
            if mean_p > mean_h and mean_s > 0.75:
                action = "preserve_or_strengthen"
            elif mean_h > mean_p and mean_a < 0.75:
                action = "penalize_or_reroute"
            elif mean_c > 0.20:
                action = "inspect_shared_branch"
            else:
                action = "no_change"
            pt_values = root_pt.reindex(nodes)["selected_pseudotime"]
            rows.append(
                {
                    "root_pos": root_pos,
                    "branch_id": branch_id,
                    "n_nodes": len(nodes),
                    "n_edges": max(0, len(nodes) - 1),
                    "pseudotime_min": float(pt_values.min()),
                    "pseudotime_max": float(pt_values.max()),
                    "mean_promotion": mean_p,
                    "mean_inhibition": mean_h,
                    "mean_collision": mean_c,
                    "mean_progression_smoothness": mean_s,
                    "mean_membership_continuity": mean_a,
                    "dominant_cell_type": "",
                    "dominant_slice": "",
                    "suggested_action": action,
                }
            )
    return pd.DataFrame(rows)


def _tree_edge_set(table: pd.DataFrame, root_pos: int) -> set[tuple[int, int]]:
    subset = table.loc[table["root_pos"] == root_pos]
    return {_edge_key(r.source, r.target) for r in subset.itertuples(index=False)}


def _edge_change_fraction(before: set[tuple[int, int]], after: set[tuple[int, int]]) -> float:
    return (len(before - after) + len(after - before)) / max(2 * len(before), 1)


def apply_conservative_interaction_refinement(
    adata,
    edges_df: pd.DataFrame,
    A: np.ndarray,
    roots: list[int],
    emb: np.ndarray,
    domains,
    cfg: ProgressionConfig,
    progression_scores: dict[int, np.ndarray],
    progression_sources: dict[int, str],
    progression_edges: pd.DataFrame,
    edge_modifiers: pd.DataFrame,
    truth_pseudotime: Optional[np.ndarray],
    cmi_e_support: Optional[pd.DataFrame] = None,
    active_assignment: Optional[np.ndarray] = None,
    compatibility_mask: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, nx.Graph], pd.DataFrame, pd.DataFrame]:
    edge_tables, summaries, trees, pseudotimes, audits = [], [], {}, [], []
    for k, root in enumerate(roots):
        reference = _tree_edge_set(progression_edges, k)
        accepted = False
        best = None
        # Apply the complete evidence-supported interaction modifier.  Earlier
        # versions progressively shrank it until an edge-change quota was met.
        # That structural quota is intentionally removed.
        scales = [1.0]
        for attempt, scale in enumerate(scales):
            result = build_one_tree(
                adata,
                edges_df,
                A,
                root,
                k,
                emb,
                domains,
                cfg,
                progression_scores=progression_scores,
                progression_sources=progression_sources,
                interaction_modifiers=edge_modifiers,
                cost_mode="interaction_v2",
                tree_label="interaction_refined",
                reference_edges=reference,
                interaction_scale=scale,
                truth_pseudotime=truth_pseudotime,
                cmi_e_support=cmi_e_support,
                active_assignment=active_assignment,
                compatibility_mask=compatibility_mask,
            )
            current_edges = _tree_edge_set(result[0], k)
            fraction = _edge_change_fraction(reference, current_edges)
            best = (*result, fraction, scale, attempt)
            accepted = True
            break
        assert best is not None
        edge_table, summary, tree, pseudo, fraction, scale, attempt = best
        if not accepted:
            edge_table = progression_edges.loc[progression_edges["root_pos"] == k].copy()
            edge_table["tree_label"] = "interaction_refined"
            edge_table["cost_mode"] = "interaction_v2_fallback_progression"
            summary = {
                "root_id": str(adata.obs_names[root]),
                "root_pos": k,
                "root_index": int(root),
                "tree_nodes": int(len(set(edge_table["source"]) | set(edge_table["target"]))),
                "tree_edges": len(edge_table),
                "n_components": 1,
                "tree_label": "interaction_refined",
                "cost_mode": "interaction_v2_fallback_progression",
                "progression_source": progression_sources[k],
            }
            tree = nx.Graph()
            for row in edge_table.itertuples(index=False):
                tree.add_edge(int(row.source), int(row.target), weight=float(row.cost_final))
            pseudo = compute_multiple_pseudotimes(
                tree, root, k, A, progression_scores, cfg, "interaction_refined", truth_pseudotime
            )
            fraction = 0.0
        current = _tree_edge_set(edge_table, k)
        union = sorted(reference | current)
        modifier_lookup = {
            _edge_key(r.source, r.target): r
            for r in edge_modifiers.loc[edge_modifiers["root_pos"] == k].itertuples(index=False)
        }
        prog_lookup = {
            _edge_key(r.source, r.target): r
            for r in progression_edges.loc[progression_edges["root_pos"] == k].itertuples(index=False)
        }
        refined_lookup = {_edge_key(r.source, r.target): r for r in edge_table.itertuples(index=False)}
        for edge in union:
            if edge in reference and edge in current:
                action = "kept"
            elif edge in current:
                action = "added"
            else:
                action = "removed"
            mod = modifier_lookup.get(edge)
            evidence = {}
            if mod is not None:
                evidence = {
                    "promotion": float(mod.P_ij_v2),
                    "inhibition": float(mod.H_ij_v2),
                    "conflict": float(mod.conflict_edge_score),
                    "merge": float(mod.merge_edge_score),
                }
            modifier_type = max(evidence, key=evidence.get) if evidence else "none"
            modifier_score = evidence.get(modifier_type, 0.0)
            prog_row, ref_row = prog_lookup.get(edge), refined_lookup.get(edge)
            cost_prog = float(getattr(prog_row, "cost_final", np.nan))
            cost_ref = float(getattr(ref_row, "cost_final", np.nan))
            audits.append(
                {
                    "root_pos": k,
                    "edge_source": edge[0],
                    "edge_target": edge[1],
                    "action": action,
                    "modifier_type": modifier_type,
                    "modifier_score": modifier_score,
                    "cost_prog": cost_prog,
                    "cost_interaction_v2": cost_ref,
                    "delta_cost": cost_ref - cost_prog if np.isfinite(cost_prog) and np.isfinite(cost_ref) else np.nan,
                    "evidence_confidence": modifier_score,
                    "reason": f"scale={scale}; attempt={attempt}",
                    "accepted": accepted,
                }
            )
        summary["edge_change_fraction"] = fraction
        summary["interaction_scale"] = scale
        summary["conservative_refinement_accepted"] = accepted
        edge_tables.append(edge_table)
        summaries.append(summary)
        trees[k] = tree
        pseudotimes.append(pseudo)
    return (
        pd.concat(edge_tables, ignore_index=True),
        pd.DataFrame(summaries),
        trees,
        pd.concat(pseudotimes, ignore_index=True),
        pd.DataFrame(audits),
    )


def compute_tree_internal_qc(
    tree_edges: pd.DataFrame,
    A: np.ndarray,
    progression_scores: dict[int, np.ndarray],
    edges_df: pd.DataFrame,
    root_pos: int,
    root_idx: int,
    tree_label: str,
    reference_edges: Optional[set[tuple[int, int]]] = None,
) -> dict[str, object]:
    table = tree_edges.loc[tree_edges["root_pos"] == root_pos].copy()
    graph = nx.Graph()
    graph.add_nodes_from([root_idx])
    for row in table.itertuples(index=False):
        graph.add_edge(int(row.source), int(row.target))
    component = nx.node_connected_component(graph, root_idx) if root_idx in graph else set()
    dg = [
        abs(float(progression_scores[root_pos][int(r.source)]) - float(progression_scores[root_pos][int(r.target)]))
        for r in table.itertuples(index=False)
    ]
    da = [abs(float(A[int(r.source), root_pos]) - float(A[int(r.target), root_pos])) for r in table.itertuples(index=False)]
    current = _tree_edge_set(table, root_pos)
    reference_edges = reference_edges or current
    return {
        "root_pos": root_pos,
        "root_index": root_idx,
        "tree_label": tree_label,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_components": nx.number_connected_components(graph) if graph.number_of_nodes() else 0,
        "root_component_size": len(component),
        "fraction_reachable_from_root": len(component) / max(graph.number_of_nodes(), 1),
        "mean_edge_cost": float(table["cost_final"].mean()) if not table.empty else np.nan,
        "mean_base_cost": float(table["cost_base"].mean()) if not table.empty else np.nan,
        "mean_membership_product": float(
            np.mean([A[int(r.source), root_pos] * A[int(r.target), root_pos] for r in table.itertuples(index=False)])
        ) if not table.empty else np.nan,
        "mean_abs_membership_delta": float(np.mean(da)) if da else np.nan,
        "mean_progression_delta": float(np.mean(dg)) if dg else np.nan,
        "mean_progression_smoothness": float(np.mean(np.exp(-np.asarray(dg) / 0.20))) if dg else np.nan,
        "mean_spatial_edge_length": np.nan,
        "mean_rootprog_score": float(table["S_rootprog"].mean()) if not table.empty else np.nan,
        "cross_slice_edges": 0,
        "cross_sample_edges": 0,
        "n_added_edges": len(current - reference_edges),
        "n_removed_edges": len(reference_edges - current),
        "n_kept_edges": len(current & reference_edges),
        "edge_change_fraction": _edge_change_fraction(reference_edges, current),
    }



# -----------------------------------------------------------------------------
# Backbone pruning / topology simplification
# -----------------------------------------------------------------------------


def _tree_from_edge_table(table: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in table.itertuples(index=False):
        graph.add_edge(int(row.source), int(row.target), weight=float(getattr(row, "cost_final", getattr(row, "edge_cost", 1.0))))
    return graph


def _edge_lookup(table: pd.DataFrame) -> dict[tuple[int, int], object]:
    return {_edge_key(r.source, r.target): r for r in table.itertuples(index=False)}


def _leaf_anchor_path(graph: nx.Graph, root_idx: int, leaf: int) -> list[int]:
    """Return terminal path from a leaf to its nearest branchpoint or root."""
    if leaf not in graph or graph.degree(leaf) != 1:
        return []
    path = [int(leaf)]
    prev = None
    cur = int(leaf)
    while True:
        neigh = [int(n) for n in graph.neighbors(cur) if n != prev]
        if not neigh:
            break
        nxt = neigh[0]
        path.append(nxt)
        prev, cur = cur, nxt
        if cur == root_idx or graph.degree(cur) != 2:
            break
    return path


def _terminal_branch_score(
    path: list[int],
    edge_table: pd.DataFrame,
    A: np.ndarray,
    progression_scores: dict[int, np.ndarray],
    root_pos: int,
    pt: dict[int, float],
) -> float:
    if len(path) < 2:
        return -math.inf
    keys = {_edge_key(a, b) for a, b in zip(path[:-1], path[1:])}
    lookup = _edge_lookup(edge_table)
    rows = [lookup[k] for k in keys if k in lookup]
    span = abs(float(pt.get(path[0], 0.0)) - float(pt.get(path[-1], 0.0)))
    mean_a = float(np.mean([A[n, root_pos] for n in path if n < A.shape[0]])) if path else 0.0
    if root_pos in progression_scores:
        vals = progression_scores[root_pos]
        dg = [abs(float(vals[a]) - float(vals[b])) for a, b in zip(path[:-1], path[1:])]
        smooth = float(np.mean(np.exp(-np.asarray(dg) / 0.20))) if dg else 0.0
    else:
        smooth = 0.0
    rootprog = float(np.mean([getattr(r, "S_rootprog", 0.0) for r in rows])) if rows else 0.0
    return span + mean_a + smooth + rootprog


def extract_backbone_tree(
    tree: nx.Graph,
    root_idx: int,
    edge_table: pd.DataFrame,
    pseudotime_table: pd.DataFrame,
    A: np.ndarray,
    progression_scores: dict[int, np.ndarray],
    root_pos: int,
    cfg: ProgressionConfig,
    n_roots: int,
) -> tuple[pd.DataFrame, nx.Graph, dict[str, object]]:
    """Extract a high-confidence root-connected trajectory core.

    v7.3 no longer treats the full root-specific MST as the biological backbone.
    Each edge receives an internal confidence score from graph similarity,
    root-program continuity, root membership, progression smoothness and inverse
    construction cost.  Only an adaptive high-confidence core is retained; low
    support nodes remain off-backbone and are not forced into a biological tree.
    """
    if edge_table is None or edge_table.empty or tree.number_of_edges() == 0:
        empty = edge_table.copy() if edge_table is not None else pd.DataFrame()
        if not empty.empty:
            empty["tree_form"] = "confidence_core_empty"
        return empty, nx.Graph([(root_idx, root_idx)]) if False else nx.Graph(), {
            "pruned_twig_count": 0,
            "collapsed_degree2_count": 0,
            "target_max_leaves": 0,
            "core_threshold": np.nan,
            "core_edge_count": 0,
            "core_node_count": 1,
            "full_edge_count": int(len(edge_table)) if edge_table is not None else 0,
        }

    if not getattr(cfg, "enable_backbone_pruning", True) or not getattr(cfg, "enable_confidence_core_backbone", True):
        table = edge_table.copy()
        table["tree_form"] = "full_no_confidence_core"
        table["edge_confidence"] = np.nan
        table["core_threshold"] = np.nan
        table["core_selected_reason"] = "confidence_core_disabled"
        return table, tree.copy(), {
            "pruned_twig_count": 0,
            "collapsed_degree2_count": 0,
            "target_max_leaves": 0,
            "core_threshold": np.nan,
            "core_edge_count": int(len(table)),
            "core_node_count": int(tree.number_of_nodes()),
            "full_edge_count": int(len(table)),
        }

    full = tree.copy()
    if root_idx in full and not nx.is_connected(full):
        comp = nx.node_connected_component(full, root_idx)
        full = full.subgraph(comp).copy()
    full_keys = {_edge_key(a, b) for a, b in full.edges()}
    table = edge_table.loc[edge_table.apply(lambda r: _edge_key(r.source, r.target) in full_keys, axis=1)].copy()
    if table.empty:
        return table, full, {
            "pruned_twig_count": 0,
            "collapsed_degree2_count": 0,
            "target_max_leaves": 0,
            "core_threshold": np.nan,
            "core_edge_count": 0,
            "core_node_count": 1,
            "full_edge_count": 0,
        }

    def norm_col(values, invert=False):
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(arr)
        out = np.zeros(len(arr), dtype=float)
        if finite.any():
            lo, hi = float(np.nanmin(arr[finite])), float(np.nanmax(arr[finite]))
            if hi > lo:
                out[finite] = (arr[finite] - lo) / (hi - lo)
            else:
                out[finite] = 1.0
        if invert:
            out = 1.0 - out
        return np.clip(out, 0.0, 1.0)

    s_global = norm_col(table["S_global"] if "S_global" in table else np.ones(len(table)))
    s_rootprog = norm_col(table["S_rootprog"] if "S_rootprog" in table else np.ones(len(table)))
    membership = np.asarray([
        math.sqrt(max(float(A[int(r.source), root_pos]), 0.0) * max(float(A[int(r.target), root_pos]), 0.0))
        for r in table.itertuples(index=False)
    ], dtype=float)
    membership = np.clip(membership, 0.0, 1.0)
    if root_pos in progression_scores:
        prog = progression_scores[root_pos]
        progression_smooth = np.asarray([
            math.exp(-abs(float(prog[int(r.source)]) - float(prog[int(r.target)])) / 0.20)
            for r in table.itertuples(index=False)
        ], dtype=float)
    else:
        progression_smooth = np.full(len(table), 0.5, dtype=float)
    cost_col = "cost_final" if "cost_final" in table else ("edge_cost" if "edge_cost" in table else None)
    inverse_cost = norm_col(table[cost_col] if cost_col else np.ones(len(table)), invert=True)

    wg = float(getattr(cfg, "core_w_global", 0.30))
    wr = float(getattr(cfg, "core_w_rootprog", 0.20))
    wm = float(getattr(cfg, "core_w_membership", 0.20))
    wp = float(getattr(cfg, "core_w_progression", 0.15))
    wc = float(getattr(cfg, "core_w_inverse_cost", 0.15))
    total_w = max(wg + wr + wm + wp + wc, 1e-9)
    confidence = (wg*s_global + wr*s_rootprog + wm*membership + wp*progression_smooth + wc*inverse_cost) / total_w
    table["edge_confidence"] = confidence

    q = float(np.clip(getattr(cfg, "core_edge_confidence_quantile", 0.60), 0.0, 1.0))
    floor = float(np.clip(getattr(cfg, "core_edge_confidence_floor", 0.45), 0.0, 1.0))
    threshold = max(floor, float(np.quantile(confidence, q)) if len(confidence) else floor)
    table["core_threshold"] = threshold

    key_to_conf = {_edge_key(r.source, r.target): float(c) for r, c in zip(table.itertuples(index=False), confidence)}
    selected_keys = {k for k, c in key_to_conf.items() if c >= threshold}

    core = nx.Graph()
    core.add_node(int(root_idx))
    for u, v in full.edges():
        if _edge_key(u, v) in selected_keys:
            core.add_edge(int(u), int(v), weight=float(full[u][v].get("weight", 1.0)))

    # Keep only the directly root-connected high-confidence component.
    if root_idx in core and core.number_of_edges() > 0:
        root_comp = set(nx.node_connected_component(core, root_idx))
        core = core.subgraph(root_comp).copy()
    else:
        core = nx.Graph()
        core.add_node(int(root_idx))

    # v7.4.2 Truth-First: a rescue path is optional, never mandatory. Every edge
    # on a proposed rescue path must independently pass hard confidence,
    # similarity, and cost gates. A disconnected root is preferable to a fake
    # biological trajectory supported by one extreme long/low-similarity edge.
    min_nodes = max(1, int(getattr(cfg, "core_min_nodes", 3)))
    if core.number_of_nodes() < min_nodes and full.number_of_edges() > 0:
        cost_col_local = "cost_final" if "cost_final" in table.columns else ("edge_cost" if "edge_cost" in table.columns else None)
        local_costs = pd.to_numeric(table[cost_col_local], errors="coerce") if cost_col_local else pd.Series(1.0, index=table.index)
        finite_local_costs = local_costs[np.isfinite(local_costs) & (local_costs > 0)]
        median_local_cost = float(np.median(finite_local_costs)) if len(finite_local_costs) else 1.0
        max_rescue_cost = median_local_cost * float(getattr(cfg, "truth_max_cost_median_factor", 8.0))
        sg = pd.to_numeric(table.get("S_global", pd.Series(0.0, index=table.index)), errors="coerce").fillna(0.0)
        q_sg = float(np.clip(getattr(cfg, "truth_s_global_quantile_floor", 0.05), 0.0, 1.0))
        rescue_sg_floor = max(
            float(getattr(cfg, "truth_min_s_global_floor", 0.03)),
            float(np.quantile(sg.to_numpy(dtype=float), q_sg)) if len(sg) else 0.0,
        )
        rescue_conf_floor = float(getattr(cfg, "truth_rescue_min_confidence", 0.55))
        row_lookup = {_edge_key(r.source, r.target): r for r in table.itertuples(index=False)}

        def _rescue_edge_pass(a: int, b: int) -> bool:
            key = _edge_key(a, b)
            row = row_lookup.get(key)
            if row is None:
                return False
            conf = key_to_conf.get(key, 0.0)
            sgv = float(getattr(row, "S_global", 0.0))
            cv = float(getattr(row, cost_col_local, 1.0)) if cost_col_local else 1.0
            return bool(
                conf >= rescue_conf_floor
                and np.isfinite(sgv) and sgv >= rescue_sg_floor
                and np.isfinite(cv) and cv <= max_rescue_cost
            )

        ranked_edges = sorted(full.edges(), key=lambda e: -key_to_conf.get(_edge_key(e[0], e[1]), 0.0))
        for u, v in ranked_edges:
            try:
                path_u = nx.shortest_path(full, root_idx, u)
                path_v = nx.shortest_path(full, root_idx, v)
            except nx.NetworkXNoPath:
                continue
            path = path_u if len(path_u) <= len(path_v) else path_v
            path = path + ([v] if path[-1] == u else [u])
            path_edges = list(zip(path[:-1], path[1:]))
            if not path_edges or not all(_rescue_edge_pass(a, b) for a, b in path_edges):
                continue
            for a, b in path_edges:
                core.add_edge(int(a), int(b), weight=float(full[a][b].get("weight", 1.0)))
            if core.number_of_nodes() >= min_nodes:
                break

    # Add high-confidence edges adjacent to the current root component when they
    # naturally extend it; do not connect distant components merely for coverage.
    changed = True
    while changed:
        changed = False
        current_nodes = set(core.nodes())
        candidates = []
        for u, v in full.edges():
            key = _edge_key(u, v)
            conf = key_to_conf.get(key, 0.0)
            if conf < threshold:
                continue
            if (u in current_nodes) ^ (v in current_nodes):
                candidates.append((conf, int(u), int(v)))
        if candidates:
            candidates.sort(reverse=True)
            conf, u, v = candidates[0]
            core.add_edge(u, v, weight=float(full[u][v].get("weight", 1.0)))
            changed = True

    core_keys = {_edge_key(a, b) for a, b in core.edges()}
    out = table.loc[table.apply(lambda r: _edge_key(r.source, r.target) in core_keys, axis=1)].copy()
    out["tree_form"] = "confidence_core_backbone"
    out["core_selected_reason"] = np.where(out["edge_confidence"] >= threshold, "above_adaptive_confidence_threshold", "root_path_rescue")

    full_nodes = set(full.nodes())
    core_nodes = set(core.nodes())
    pruned_nodes = len(full_nodes - core_nodes)
    deg2 = sum(1 for n, d in core.degree() if d == 2 and n != root_idx)
    n_leaves = sum(1 for n, d in core.degree() if d == 1 and n != root_idx)
    n_branch = sum(1 for _, d in core.degree() if d >= 3)
    stats = {
        "pruned_twig_count": int(pruned_nodes),
        "collapsed_degree2_count": int(deg2 if getattr(cfg, "collapse_degree2", True) else 0),
        "target_max_leaves": 0,
        "n_leaves": int(n_leaves),
        "n_branchpoints": int(n_branch),
        "overbranching_ratio": np.nan,
        "core_threshold": float(threshold),
        "core_edge_count": int(len(out)),
        "core_node_count": int(core.number_of_nodes()),
        "full_edge_count": int(len(table)),
        "off_backbone_node_count": int(pruned_nodes),
    }
    return out, core, stats


def compute_topology_qc(
    table: pd.DataFrame,
    root_pos: int,
    root_idx: int,
    tree_label: str,
    full_or_backbone: str,
    backbone_stats: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    graph = _tree_from_edge_table(table.loc[table["root_pos"] == root_pos] if "root_pos" in table else table)
    graph.add_node(root_idx)
    n_leaves = sum(1 for n, d in graph.degree() if d == 1 and n != root_idx)
    n_branch = sum(1 for _, d in graph.degree() if d >= 3)
    stats = backbone_stats or {}
    target = 0
    return {
        "tree_label": tree_label,
        "root_pos": root_pos,
        "full_or_backbone": full_or_backbone,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_leaves": n_leaves,
        "n_branchpoints": n_branch,
        "leaf_to_root_distance_mean": np.nan,
        "terminal_twig_count": n_leaves,
        "pruned_twig_count": int(stats.get("pruned_twig_count", 0)),
        "collapsed_degree2_count": int(stats.get("collapsed_degree2_count", 0)),
        "target_max_leaves": target,
        "overbranching_ratio": np.nan,
    }


def _summary_from_edges(table: pd.DataFrame, roots: list[int], label: str, form: str) -> pd.DataFrame:
    rows = []
    for k, root in enumerate(roots):
        sub = table.loc[table["root_pos"] == k]
        nodes = set(sub["source"].astype(int)) | set(sub["target"].astype(int)) if not sub.empty else {root}
        g = _tree_from_edge_table(sub)
        g.add_node(root)
        rows.append({
            "root_pos": k,
            "root_index": root,
            "tree_nodes": len(nodes),
            "tree_edges": len(sub),
            "n_components": nx.number_connected_components(g) if g.number_of_nodes() else 0,
            "tree_label": label,
            "tree_form": form,
        })
    return pd.DataFrame(rows)


def build_backbone_tables_for_label(
    labels: dict[str, pd.DataFrame],
    tree_sets: dict[str, dict[int, nx.Graph]],
    pseudotime_table: pd.DataFrame,
    roots: list[int],
    A: np.ndarray,
    progression_scores: dict[int, np.ndarray],
    cfg: ProgressionConfig,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts, qc_rows = [], []
    for k, root in enumerate(roots):
        full_sub = labels[label].loc[labels[label]["root_pos"] == k].copy()
        pseudo_sub = pseudotime_table.loc[(pseudotime_table.root_pos == k) & (pseudotime_table.tree_label == label)]
        bb, bb_graph, stats = extract_backbone_tree(
            tree_sets[label][k], root, full_sub, pseudo_sub, A, progression_scores, k, cfg, len(roots)
        )
        bb["tree_label"] = label
        bb["tree_form"] = "backbone"
        parts.append(bb)
        qc_rows.append(compute_topology_qc(full_sub, k, root, label, "full"))
        qc_rows.append(compute_topology_qc(bb, k, root, label, "backbone", stats))
    backbone = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    summaries = _summary_from_edges(backbone, roots, label, "backbone")
    return backbone, summaries, pd.DataFrame(qc_rows)


def select_final_tree(
    dataset: str,
    roots: list[int],
    labels: dict[str, pd.DataFrame],
    qc_table: pd.DataFrame,
    cfg: ProgressionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_tables, decisions, summaries = [], [], []
    for k, root in enumerate(roots):
        qc = {
            label: qc_table.loc[(qc_table.root_pos == k) & (qc_table.tree_label == label)].iloc[0]
            for label in ("base", "progression", "interaction_refined")
        }
        progression_ok = (
            qc["progression"].fraction_reachable_from_root >= qc["base"].fraction_reachable_from_root
            and qc["progression"].mean_progression_smoothness + 1e-9
            >= qc["base"].mean_progression_smoothness - 0.02
        )
        refined_ok = (
            qc["interaction_refined"].n_components <= qc["progression"].n_components
            and qc["interaction_refined"].fraction_reachable_from_root + 1e-9
            >= qc["progression"].fraction_reachable_from_root
            and qc["interaction_refined"].mean_progression_smoothness + 1e-9
            >= qc["progression"].mean_progression_smoothness - 0.01
            and qc["interaction_refined"].mean_abs_membership_delta
            <= qc["progression"].mean_abs_membership_delta + 0.01
            and qc["interaction_refined"].mean_rootprog_score + 1e-9
            >= qc["progression"].mean_rootprog_score - 0.02
        )
        if not cfg.enable_acceptance_rule:
            selected = "interaction_refined" if cfg.enable_interaction_v2 else "progression"
            reason = "acceptance_rule_disabled"
        elif progression_ok and refined_ok and cfg.enable_interaction_v2:
            selected, reason = "interaction_refined", "internal_qc_passed"
        elif progression_ok:
            selected, reason = "progression", "refined_rejected_by_internal_qc"
        else:
            selected, reason = "base", "progression_rejected_by_internal_qc"
        table = labels[selected].loc[labels[selected]["root_pos"] == k].copy()
        table["selected_from_tree_label"] = selected
        table["tree_label"] = "final_selected"
        final_tables.append(table)
        decisions.append(
            {
                "dataset": dataset,
                "root_pos": k,
                "selected_tree_label": selected,
                "accepted_progression": progression_ok,
                "accepted_refined": refined_ok,
                "fallback_to": selected if selected != "interaction_refined" else "",
                "reason": reason,
                "edge_change_fraction": qc["interaction_refined"].edge_change_fraction,
                "base_direction": np.nan,
                "progression_direction": np.nan,
                "refined_direction": np.nan,
                "base_pseudotime_sr": np.nan,
                "progression_pseudotime_sr": np.nan,
                "refined_pseudotime_sr": np.nan,
                "base_internal_qc": float(qc["base"].mean_progression_smoothness),
                "progression_internal_qc": float(qc["progression"].mean_progression_smoothness),
                "refined_internal_qc": float(qc["interaction_refined"].mean_progression_smoothness),
            }
        )
        summaries.append(
            {
                "root_pos": k,
                "root_index": root,
                "selected_tree_label": selected,
                "tree_nodes": int(qc[selected].n_nodes),
                "tree_edges": int(qc[selected].n_edges),
                "n_components": int(qc[selected].n_components),
                "selection_reason": reason,
            }
        )
    return pd.concat(final_tables, ignore_index=True), pd.DataFrame(summaries), pd.DataFrame(decisions)


# -----------------------------------------------------------------------------
# Formal method-level prediction export
# -----------------------------------------------------------------------------


def _root_oriented_children(graph: nx.Graph, root_idx: int) -> tuple[dict[int, list[int]], dict[int, Optional[int]], dict[int, int]]:
    """Orient an undirected root-specific backbone outward from the root."""
    children: dict[int, list[int]] = {}
    parent: dict[int, Optional[int]] = {}
    depth: dict[int, int] = {}
    if graph.number_of_nodes() == 0:
        return children, parent, depth
    if root_idx not in graph:
        # Fallback: choose the node with minimum graph eccentricity proxy.
        root_idx = sorted(graph.nodes())[0]
    component = nx.node_connected_component(graph, root_idx)
    sub = graph.subgraph(component).copy()
    parent[int(root_idx)] = None
    depth[int(root_idx)] = 0
    queue = [int(root_idx)]
    seen = {int(root_idx)}
    while queue:
        u = queue.pop(0)
        nbrs = sorted(int(v) for v in sub.neighbors(u) if int(v) not in seen)
        children[u] = nbrs
        for v in nbrs:
            seen.add(v)
            parent[v] = u
            depth[v] = depth[u] + 1
            queue.append(v)
    return children, parent, depth


def _collapse_backbone_to_segments_for_root(
    table: pd.DataFrame,
    root_idx: int,
    root_pos: int,
    adata,
    final_pseudo: pd.DataFrame,
    cfg: ProgressionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """Collapse a root-specific metacell backbone into branch/state segments.

    This is now part of the method output rather than an external adapter hack.
    The segment labels are stable, root-specific, and can be used directly as
    ``branch`` labels in a scorer JSON.
    """
    sub = table.loc[table["root_pos"] == root_pos].copy() if "root_pos" in table else table.copy()
    graph = _tree_from_edge_table(sub)
    graph.add_node(int(root_idx))
    if graph.number_of_edges() == 0:
        seg = f"root{root_pos}_seg0000"
        node_to_segment = {int(root_idx): seg}
        return (
            pd.DataFrame(
                [{
                    "root_pos": root_pos,
                    "segment_label": seg,
                    "parent_segment_label": "",
                    "start_node": int(root_idx),
                    "end_node": int(root_idx),
                    "n_nodes": 1,
                    "n_edges": 0,
                    "is_root_segment": True,
                    "is_leaf_segment": True,
                    "pseudotime_min": 0.0,
                    "pseudotime_max": 0.0,
                    "node_names": str(adata.obs_names[int(root_idx)]),
                }]
            ),
            pd.DataFrame(columns=["root_pos", "source_segment", "target_segment"]),
            node_to_segment,
        )
    children, parent, depth = _root_oriented_children(graph, int(root_idx))
    pt_lookup = {}
    if final_pseudo is not None and not final_pseudo.empty:
        pt_sub = final_pseudo.loc[final_pseudo["root_pos"] == root_pos]
        if "selected_pseudotime" in pt_sub:
            pt_lookup = pt_sub.set_index("node_id")["selected_pseudotime"].to_dict()
    branchpoints = {n for n, ch in children.items() if len(ch) >= 2}
    segment_rows = []
    segment_edges: set[tuple[str, str]] = set()
    node_to_segment: dict[int, str] = {}
    seg_count = 0

    def new_label() -> str:
        nonlocal seg_count
        label = f"root{root_pos}_seg{seg_count:04d}"
        seg_count += 1
        return label

    def split_segment_nodes(nodes: list[int]) -> list[list[int]]:
        """Split an overly long path segment by tree order when its pseudotime span is too large."""
        nodes = [int(n) for n in nodes]
        if not getattr(cfg, "enable_segment_split", True):
            return [nodes]
        min_nodes = max(2, int(getattr(cfg, "segment_split_min_nodes", 4)))
        span_cut = float(getattr(cfg, "segment_split_pseudotime_span", 0.25))
        vals = [float(pt_lookup.get(n, np.nan)) for n in nodes]
        finite = [v for v in vals if np.isfinite(v)]
        if len(nodes) < min_nodes or len(finite) < 2:
            return [nodes]
        span = float(np.nanmax(finite) - np.nanmin(finite))
        if not np.isfinite(span) or span <= span_cut:
            return [nodes]
        n_parts = int(math.ceil(span / max(span_cut, 1e-12)))
        n_parts = max(2, min(n_parts, len(nodes)))
        raw_chunks = [list(map(int, chunk)) for chunk in np.array_split(np.asarray(nodes, dtype=int), n_parts)]
        return [chunk for chunk in raw_chunks if chunk]

    def append_segment_row(label: str, parent_segment: str, nodes: list[int], is_root_segment: bool, is_leaf_segment: bool) -> None:
        vals = [float(pt_lookup.get(n, np.nan)) for n in nodes]
        finite_vals = [v for v in vals if np.isfinite(v)]
        start_node = int(nodes[0])
        end_node = int(nodes[-1])
        segment_rows.append(
            {
                "root_pos": root_pos,
                "root_index": int(root_idx),
                "root_id": str(adata.obs_names[int(root_idx)]),
                "segment_label": label,
                "parent_segment_label": parent_segment,
                "start_node": start_node,
                "start_node_name": str(adata.obs_names[start_node]),
                "end_node": end_node,
                "end_node_name": str(adata.obs_names[end_node]),
                "n_nodes": len(nodes),
                "n_edges": max(0, len(nodes) - 1),
                "is_root_segment": bool(is_root_segment),
                "is_leaf_segment": bool(is_leaf_segment),
                "pseudotime_min": float(np.nanmin(finite_vals)) if finite_vals else np.nan,
                "pseudotime_max": float(np.nanmax(finite_vals)) if finite_vals else np.nan,
                "node_ids": ";".join(map(str, nodes)),
                "node_names": ";".join(str(adata.obs_names[n]) for n in nodes),
                "segment_split_enabled": bool(getattr(cfg, "enable_segment_split", True)),
            }
        )

    def emit_segment_chain(nodes: list[int], parent_segment: str) -> str:
        """Create one or more ordered segment labels for a traced tree path."""
        chunks = split_segment_nodes(nodes)
        prev_parent = parent_segment
        last_label = ""
        for idx, chunk in enumerate(chunks):
            label = new_label()
            if prev_parent:
                segment_edges.add((prev_parent, label))
            for n in chunk:
                node_to_segment[int(n)] = label
            is_root_segment = int(chunk[0]) == int(root_idx)
            is_leaf_segment = (idx == len(chunks) - 1 and len(children.get(int(chunk[-1]), [])) == 0)
            append_segment_row(label, prev_parent, chunk, is_root_segment, is_leaf_segment)
            prev_parent = label
            last_label = label
        return last_label

    def trace_segment(start: int, parent_segment: str) -> None:
        nodes = [int(start)]
        cur = int(start)
        while True:
            ch = children.get(cur, [])
            if len(ch) == 0:
                break
            if cur != start and len(ch) >= 2:
                break
            if len(ch) >= 2:
                break
            nxt = int(ch[0])
            # Stop before entering a merge/reused node; root-oriented trees should not merge,
            # but this guard keeps the export stable for imperfect forests.
            if nxt in node_to_segment:
                break
            nodes.append(nxt)
            cur = nxt
        end = int(nodes[-1])
        last_label = emit_segment_chain(nodes, parent_segment)
        # Spawn child segments after a branchpoint or after the traced chain end.
        for c in children.get(end, []):
            if c not in node_to_segment:
                trace_segment(int(c), last_label)

    trace_segment(int(root_idx), "")
    # Any node still unassigned becomes a singleton attached by its parent segment if possible.
    for n in sorted(graph.nodes()):
        n = int(n)
        if n in node_to_segment:
            continue
        parent_node = parent.get(n)
        parent_segment = node_to_segment.get(parent_node, "") if parent_node is not None else ""
        label = new_label()
        node_to_segment[n] = label
        if parent_segment:
            segment_edges.add((parent_segment, label))
        val = float(pt_lookup.get(n, np.nan))
        segment_rows.append(
            {
                "root_pos": root_pos,
                "root_index": int(root_idx),
                "root_id": str(adata.obs_names[int(root_idx)]),
                "segment_label": label,
                "parent_segment_label": parent_segment,
                "start_node": n,
                "start_node_name": str(adata.obs_names[n]),
                "end_node": n,
                "end_node_name": str(adata.obs_names[n]),
                "n_nodes": 1,
                "n_edges": 0,
                "is_root_segment": n == int(root_idx),
                "is_leaf_segment": len(children.get(n, [])) == 0,
                "pseudotime_min": val,
                "pseudotime_max": val,
                "node_ids": str(n),
                "node_names": str(adata.obs_names[n]),
            }
        )
    edge_rows = [
        {"root_pos": root_pos, "source_segment": u, "target_segment": v}
        for u, v in sorted(segment_edges)
        if u != v
    ]
    return pd.DataFrame(segment_rows), pd.DataFrame(edge_rows), node_to_segment



def _root_specific_tree_distance(
    graph: nx.Graph,
    root_idx: int,
    nodes: list[int],
    fallback: np.ndarray,
) -> np.ndarray:
    """Root-anchored distance proxy for constrained pseudotime."""
    if root_idx in graph and graph.number_of_nodes() > 0:
        dist = nx.single_source_dijkstra_path_length(graph, int(root_idx), weight="weight")
    else:
        dist = {}
    vals = np.asarray([float(dist.get(int(n), np.nan)) for n in nodes], dtype=float)
    finite = np.isfinite(vals)
    if finite.any():
        vals[finite] = _normalise(vals[finite])
    fb = _normalise(np.asarray(fallback, dtype=float))
    vals[~finite] = fb[~finite]
    return _normalise(vals)


def _enforce_tree_monotonicity(
    tau: np.ndarray,
    nodes: list[int],
    graph: nx.Graph,
    root_idx: int,
    tree_distance: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Project pseudotime to be nondecreasing along root-outward tree edges."""
    out = np.asarray(tau, dtype=float).copy()
    idx = {int(n): i for i, n in enumerate(nodes)}
    if root_idx not in idx or graph.number_of_edges() == 0:
        return _normalise(out)
    # Process nodes from root outward using the tree-distance proxy.
    order = sorted(nodes, key=lambda n: (tree_distance[idx[int(n)]], str(n)))
    for u in order:
        if int(u) not in idx:
            continue
        iu = idx[int(u)]
        for v in graph.neighbors(int(u)) if int(u) in graph else []:
            if int(v) not in idx:
                continue
            iv = idx[int(v)]
            # Orient locally by root-distance. Only constrain outward edges.
            if tree_distance[iv] + 1e-12 >= tree_distance[iu]:
                out[iv] = max(out[iv], out[iu] + float(eps))
    return _normalise(out)


def compute_constrained_root_pseudotimes(
    final_backbone_edges: pd.DataFrame,
    final_pseudo: pd.DataFrame,
    roots: list[int],
    A: np.ndarray,
    cfg: ProgressionConfig,
) -> dict[tuple[int, int], float]:
    """Estimate final root-specific pseudotime by a root-anchored graph-regularized objective.

    For each root k, this solves the unconstrained quadratic core

        min_tau  L * sum_(i,j in E_k) w_ij (tau_i - tau_j)^2
               + mu * sum_i (tau_i - g_i^k)^2
               + eta * sum_i (tau_i - d_i^k)^2
               + anchor * tau_root^2,

    then applies a deterministic tree-monotone projection along root-outward
    backbone edges. This directly improves the method-level pseudotime output;
    it does not read benchmark truth.
    """
    if final_pseudo is None or final_pseudo.empty or "selected_pseudotime" not in final_pseudo:
        return {}

    out: dict[tuple[int, int], float] = {}
    mu = float(getattr(cfg, "constrained_mu", 0.60))
    eta = float(getattr(cfg, "constrained_eta", 0.30))
    lap_weight = float(getattr(cfg, "constrained_laplacian", 0.35))
    anchor = float(getattr(cfg, "constrained_root_anchor", 25.0))
    eps = float(getattr(cfg, "constrained_monotone_eps", 1e-4))

    for k, root in enumerate(roots):
        sub_p = final_pseudo.loc[final_pseudo["root_pos"].astype(int) == int(k)].copy()
        if sub_p.empty:
            continue
        nodes = [int(x) for x in sub_p["node_id"].tolist()]
        idx = {n: i for i, n in enumerate(nodes)}
        m = len(nodes)
        g = _normalise(pd.to_numeric(sub_p["selected_pseudotime"], errors="coerce").to_numpy(dtype=float))

        sub_e = final_backbone_edges.loc[final_backbone_edges["root_pos"].astype(int) == int(k)].copy()
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for row in sub_e.itertuples(index=False):
            u, v = int(row.source), int(row.target)
            if u in idx and v in idx:
                w = float(getattr(row, "cost_final", getattr(row, "edge_cost", 1.0)))
                graph.add_edge(u, v, weight=max(w, 1e-8))

        # d_i^k is a root-anchored tree-distance target. If a node was pruned
        # from the backbone, fall back to g_i^k rather than inserting truth.
        d = _root_specific_tree_distance(graph, int(root), nodes, fallback=g)

        # Weighted graph Laplacian. Low-cost/high-confidence edges receive larger smoothness weight.
        L = np.zeros((m, m), dtype=float)
        for u, v, data in graph.edges(data=True):
            iu, iv = idx[int(u)], idx[int(v)]
            ew = 1.0 / max(float(data.get("weight", 1.0)), 1e-8)
            L[iu, iu] += ew
            L[iv, iv] += ew
            L[iu, iv] -= ew
            L[iv, iu] -= ew

        M = lap_weight * L + (mu + eta + 1e-8) * np.eye(m)
        b = mu * g + eta * d
        if int(root) in idx:
            ir = idx[int(root)]
            M[ir, ir] += anchor
            b[ir] += 0.0

        try:
            tau = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            tau = np.linalg.lstsq(M, b, rcond=None)[0]
        tau = _normalise(tau)
        tau = _enforce_tree_monotonicity(tau, nodes, graph, int(root), d, eps)

        for n, val in zip(nodes, tau):
            out[(int(k), int(n))] = float(val)
    return out

def _nearest_segment_label_for_root(segments: pd.DataFrame, root_pos: int, pseudotime: float) -> Optional[str]:
    """Assign pruned/off-backbone nodes to the nearest exported segment of the same root."""
    if segments is None or segments.empty:
        return None
    sub = segments.loc[segments["root_pos"].astype(int) == int(root_pos)].copy()
    if sub.empty:
        return None
    pt = float(pseudotime) if np.isfinite(pseudotime) else np.nan
    candidates = []
    for row in sub.itertuples(index=False):
        label = str(row.segment_label)
        lo = float(getattr(row, "pseudotime_min", np.nan))
        hi = float(getattr(row, "pseudotime_max", np.nan))
        n_nodes = int(getattr(row, "n_nodes", 1))
        is_root = bool(getattr(row, "is_root_segment", False))
        if not np.isfinite(pt) or not np.isfinite(lo) or not np.isfinite(hi):
            dist = 1.0
        elif lo <= pt <= hi:
            dist = 0.0
        else:
            dist = min(abs(pt - lo), abs(pt - hi))
        center = 0.5 * (lo + hi) if np.isfinite(lo) and np.isfinite(hi) else 0.5
        center_dist = abs(pt - center) if np.isfinite(pt) else 1.0
        # Prefer non-root, larger/supporting segments when distances tie.
        candidates.append((dist, center_dist, 1 if is_root else 0, -n_nodes, label))
    return min(candidates, key=lambda x: x[:4])[-1] if candidates else None


def build_final_method_outputs(
    adata,
    final_backbone_edges: pd.DataFrame,
    final_pseudo: pd.DataFrame,
    roots: list[int],
    A: np.ndarray,
    outdir: Path,
    dataset: str,
    cfg: ProgressionConfig,
    method_name: str = "cascat_backbone_final",
    active_assignment: Optional[np.ndarray] = None,
) -> dict[str, object]:
    """Write formal v7.3 outputs without forcing full backbone coverage.

    Nodes with no active root remain unassigned. Nodes with a valid active root but
    absent from the selected backbone are labelled trajectory_off_backbone rather
    than being silently snapped onto the nearest branch segment.
    """
    n = adata.n_obs
    if active_assignment is None:
        active_assignment = np.zeros_like(A, dtype=np.int8)
        primary = v71bio.safe_primary_roots(A, cfg.eps)
        for i, p in enumerate(primary):
            if p >= 0:
                active_assignment[i, p] = 1

    segment_tables = []
    segment_edge_tables = []
    node_segment_by_root: dict[int, dict[int, str]] = {}
    for k, root in enumerate(roots):
        seg, seg_edges, mapping = _collapse_backbone_to_segments_for_root(
            final_backbone_edges, int(root), k, adata, final_pseudo, cfg
        )
        segment_tables.append(seg)
        segment_edge_tables.append(seg_edges)
        node_segment_by_root[k] = mapping

    segments = pd.concat(segment_tables, ignore_index=True) if segment_tables else pd.DataFrame()
    segment_edges = pd.concat(segment_edge_tables, ignore_index=True) if segment_edge_tables else pd.DataFrame()

    pseudo_lookup: dict[tuple[int, int], float] = {}
    if final_pseudo is not None and not final_pseudo.empty and "selected_pseudotime" in final_pseudo:
        if getattr(cfg, "enable_constrained_pseudotime", True):
            pseudo_lookup = compute_constrained_root_pseudotimes(final_backbone_edges, final_pseudo, roots, A, cfg)
        if not pseudo_lookup:
            for row in final_pseudo.itertuples(index=False):
                pseudo_lookup[(int(row.root_pos), int(row.node_id))] = float(row.selected_pseudotime)

    constrained_rows = []
    for (root_pos, node_id), value in sorted(pseudo_lookup.items()):
        constrained_rows.append(
            {
                "root_pos": int(root_pos),
                "root_index": int(roots[int(root_pos)]),
                "root_id": str(adata.obs_names[int(roots[int(root_pos)])]),
                "node_id": int(node_id),
                "node_name": str(adata.obs_names[int(node_id)]),
                "constrained_pseudotime": float(value),
            }
        )
    pd.DataFrame(constrained_rows).to_csv(
        outdir / "final_constrained_pseudotime_by_root.csv", index=False
    )

    rows = []
    primary = v71bio.safe_primary_roots(A, cfg.eps)
    for node in range(n):
        active_roots = [int(k) for k in np.flatnonzero(active_assignment[node] > 0)]
        if not active_roots:
            rows.append(
                {
                    "dataset": dataset,
                    "node_id": node,
                    "node_name": str(adata.obs_names[node]),
                    "predicted_origin_root_pos": -1,
                    "predicted_origin_root_index": -1,
                    "predicted_origin_root_id": "",
                    "origin_membership": 0.0,
                    "n_active_roots": 0,
                    "active_roots": "",
                    "node_in_selected_backbone": False,
                    "final_pseudotime": np.nan,
                    "branch_segment_label": "unassigned",
                    "trajectory_status": "unassigned",
                    "primary_membership_root_pos": int(primary[node]),
                    "assignment_method": "v7.3_sparse_active_root_assignment",
                }
            )
            continue

        chosen_root = max(active_roots, key=lambda k: float(A[node, k]))
        membership = float(A[node, chosen_root])
        in_tree = node in node_segment_by_root.get(chosen_root, {})
        chosen_pt = pseudo_lookup.get((chosen_root, node), np.nan)
        if in_tree:
            seg_label = node_segment_by_root[chosen_root][node]
            status = "trajectory_backbone"
        else:
            seg_label = f"root{chosen_root}_off_backbone"
            status = "trajectory_off_backbone"

        rows.append(
            {
                "dataset": dataset,
                "node_id": node,
                "node_name": str(adata.obs_names[node]),
                "predicted_origin_root_pos": int(chosen_root),
                "predicted_origin_root_index": int(roots[chosen_root]),
                "predicted_origin_root_id": str(adata.obs_names[int(roots[chosen_root])]),
                "origin_membership": membership,
                "n_active_roots": int(len(active_roots)),
                "active_roots": "|".join(f"T{k}" for k in active_roots),
                "node_in_selected_backbone": bool(in_tree),
                "final_pseudotime": float(chosen_pt) if np.isfinite(chosen_pt) else np.nan,
                "branch_segment_label": seg_label,
                "trajectory_status": status,
                "primary_membership_root_pos": int(primary[node]),
                "assignment_method": "v7.3_sparse_active_root_assignment",
            }
        )

    node_assignments = pd.DataFrame(rows)
    norm_parts = []
    for root_pos, part in node_assignments.groupby("predicted_origin_root_pos", sort=False):
        tmp = part.copy()
        vals = tmp["final_pseudotime"].to_numpy(dtype=float)
        finite = np.isfinite(vals)
        norm = np.full(len(vals), np.nan, dtype=float)
        if finite.any():
            norm[finite] = _normalise(vals[finite])
        tmp["final_pseudotime_origin_normalized"] = norm
        norm_parts.append(tmp)
    node_assignments = (
        pd.concat(norm_parts, ignore_index=True).sort_values("node_id")
        if norm_parts else node_assignments
    )

    node_assignments.to_csv(outdir / "final_node_assignments.csv", index=False)
    segments.to_csv(outdir / "final_branch_segments.csv", index=False)
    segment_edges.to_csv(outdir / "final_segment_tree_edges.csv", index=False)

    tree_edges = (
        segment_edges[["source_segment", "target_segment"]]
        .drop_duplicates()
        .rename(columns={"source_segment": "source", "target_segment": "target"})
        if not segment_edges.empty
        else pd.DataFrame(columns=["source", "target"])
    )
    ordered = node_assignments.sort_values("node_id")
    branch = ordered["branch_segment_label"].astype(str).tolist()
    pseudo_vals = ordered["final_pseudotime_origin_normalized"].to_numpy(dtype=float)
    # Keep the legacy JSON schema numeric. Missing/off-backbone pseudotimes are 0.0
    # and remain explicitly identifiable through branch/status in the CSV output.
    pseudotime = [float(v) if np.isfinite(v) else 0.0 for v in pseudo_vals]
    roots_segments = []
    if not segments.empty and "is_root_segment" in segments.columns:
        roots_segments = (
            segments.loc[segments["is_root_segment"].astype(bool), "segment_label"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
    if not roots_segments and branch:
        finite_idx = np.flatnonzero(np.isfinite(pseudo_vals))
        if len(finite_idx):
            roots_segments = [branch[int(finite_idx[np.argmin(pseudo_vals[finite_idx])])]]

    prediction = {
        "method": method_name,
        "dataset": dataset.removesuffix("_metacell"),
        "cell_ids": [str(x) for x in adata.obs_names],
        "pseudotime": pseudotime,
        "branch": branch,
        "tree_edges": tree_edges[["source", "target"]].astype(str).values.tolist(),
        "roots": roots_segments,
        "note": "v7.3 does not force unassigned/off-backbone nodes onto trajectory segments; see final_node_assignments.csv",
    }
    (outdir / "final_prediction_metacell.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "final_node_assignments": "final_node_assignments.csv",
        "final_branch_segments": "final_branch_segments.csv",
        "final_segment_tree_edges": "final_segment_tree_edges.csv",
        "final_constrained_pseudotime_by_root": "final_constrained_pseudotime_by_root.csv",
        "final_prediction_metacell_json": "final_prediction_metacell.json",
        "n_branch_segments": int(len(set(branch))),
        "n_segment_edges": int(len(prediction["tree_edges"])),
        "n_unassigned_nodes": int((node_assignments["trajectory_status"] == "unassigned").sum()),
        "n_off_backbone_nodes": int((node_assignments["trajectory_status"] == "trajectory_off_backbone").sum()),
        "enable_segment_split": bool(cfg.enable_segment_split),
        "segment_split_min_nodes": int(cfg.segment_split_min_nodes),
        "segment_split_pseudotime_span": float(cfg.segment_split_pseudotime_span),
        "segment_split_max_parts": "unlimited",
        "enable_constrained_pseudotime": bool(getattr(cfg, "enable_constrained_pseudotime", True)),
        "constrained_mu": float(getattr(cfg, "constrained_mu", 0.60)),
        "constrained_eta": float(getattr(cfg, "constrained_eta", 0.30)),
        "constrained_laplacian": float(getattr(cfg, "constrained_laplacian", 0.35)),
        "constrained_root_anchor": float(getattr(cfg, "constrained_root_anchor", 25.0)),
        "enable_cmi_e": bool(getattr(cfg, "enable_cmi_e", True)),
        "lambda_cmi_e_drop": float(getattr(cfg, "lambda_cmi_e_drop", 0.08)),
        "lambda_cmi_e_trust": float(getattr(cfg, "lambda_cmi_e_trust", 0.02)),
        "cmi_e_n_bins": int(getattr(cfg, "cmi_e_n_bins", 3)),
    }

def _load_membership_and_roots(outdir: Path) -> tuple[np.ndarray, list[int]]:
    """Prefer the v7.3 A_sharp matrix; fall back to legacy soft_membership.csv."""
    sharp_path = outdir / "membership_A_sharp.csv"
    if sharp_path.exists():
        table = pd.read_csv(sharp_path)
        n = int(table["node_id"].max()) + 1
        k_count = int(table["root_pos"].max()) + 1
        A = np.zeros((n, k_count), dtype=float)
        value_col = "A_sharp" if "A_sharp" in table.columns else "A_ik"
        for row in table.itertuples(index=False):
            A[int(row.node_id), int(row.root_pos)] = float(getattr(row, value_col))
        roots = (
            table[["root_pos", "root_index"]]
            .drop_duplicates()
            .sort_values("root_pos")["root_index"]
            .astype(int)
            .tolist()
        )
        return A, roots

    membership = pd.read_csv(outdir / "soft_membership.csv")
    n = int(membership["node_id"].max()) + 1
    k_count = int(membership["root_pos"].max()) + 1
    A = np.zeros((n, k_count))
    for row in membership.itertuples(index=False):
        A[int(row.node_id), int(row.root_pos)] = float(row.A_ik)
    roots = (
        membership[["root_pos", "root_index"]]
        .drop_duplicates()
        .sort_values("root_pos")["root_index"]
        .astype(int)
        .tolist()
    )
    return A, roots


def _load_active_assignment(outdir: Path, n_nodes: int, n_roots: int) -> np.ndarray:
    path = outdir / "active_root_assignment.csv"
    if path.exists():
        return v71bio.load_active_assignment(path, n_nodes, n_roots)
    # Conservative fallback: primary root only.
    A, _ = _load_membership_and_roots(outdir)
    primary = v71bio.safe_primary_roots(A, 1e-9)
    Z = np.zeros((n_nodes, n_roots), dtype=np.int8)
    for i, p in enumerate(primary):
        if p >= 0:
            Z[i, p] = 1
    return Z

def compute_multi_root_membership_diagnostics(
    adata,
    A: np.ndarray,
    roots: list[int],
    cfg: ProgressionConfig,
    active_assignment: Optional[np.ndarray] = None,
    assignment_evidence: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export v7.3 uncertainty diagnostics without equating ambiguity to multi-tree inclusion."""
    n, k_count = A.shape
    if n == 0 or k_count == 0:
        empty = pd.DataFrame()
        return empty, empty
    if active_assignment is None:
        active_assignment = np.zeros_like(A, dtype=np.int8)
        primary_fallback = v71bio.safe_primary_roots(A, cfg.eps)
        for i, p in enumerate(primary_fallback):
            if p >= 0:
                active_assignment[i, p] = 1

    primary = v71bio.safe_primary_roots(A, cfg.eps)
    order = np.argsort(-A, axis=1)
    secondary = order[:, 1] if k_count > 1 else order[:, 0]
    rows: list[dict[str, object]] = []
    evidence_lookup = {}
    if assignment_evidence is not None and not assignment_evidence.empty:
        evidence_lookup = assignment_evidence.set_index("node_id").to_dict("index")

    for i in range(n):
        p = int(primary[i])
        s = int(secondary[i]) if p >= 0 else -1
        p1 = float(A[i, p]) if p >= 0 else 0.0
        p2 = float(A[i, s]) if s >= 0 else 0.0
        margin = p1 - p2
        shared_score = p2 / (p1 + cfg.eps) if p >= 0 else 0.0
        positive = A[i] > cfg.eps
        if int(positive.sum()) > 1:
            vals = A[i, positive]
            entropy = -float(np.sum(vals * np.log(vals + cfg.eps)))
            entropy_norm = entropy / math.log(int(positive.sum()))
        else:
            entropy = 0.0
            entropy_norm = 0.0
        active_roots = [int(k) for k in np.flatnonzero(active_assignment[i] > 0)]
        active_count = len(active_roots)
        ev = evidence_lookup.get(i, {})
        if p < 0:
            label = "unassigned"
        elif active_count >= 2:
            label = "validated_dual_root_shared"
        elif int((A[i] >= float(cfg.theta_secondary)).sum()) >= 3:
            label = "high_order_ambiguity"
        elif entropy_norm >= getattr(cfg, "theta_entropy", 0.70) or margin <= getattr(cfg, "shared_margin_threshold", 0.20):
            label = "soft_ambiguity_not_multi_tree"
        else:
            label = "root_specific"
        rows.append({
            "node_id": int(i),
            "node_name": str(adata.obs_names[i]),
            "primary_root": p,
            "primary_root_index": int(roots[p]) if 0 <= p < len(roots) else -1,
            "primary_root_name": str(adata.obs_names[int(roots[p])]) if 0 <= p < len(roots) else "",
            "primary_membership": p1,
            "secondary_root": s,
            "secondary_root_index": int(roots[s]) if 0 <= s < len(roots) else -1,
            "secondary_root_name": str(adata.obs_names[int(roots[s])]) if 0 <= s < len(roots) else "",
            "secondary_membership": p2,
            "membership_margin": margin,
            "root_entropy": entropy,
            "root_entropy_norm": entropy_norm,
            "shared_score": shared_score,
            "n_active_roots": active_count,
            "active_roots": "|".join(f"T{k}" for k in active_roots),
            "is_multi_root": bool(active_count >= 2),
            "multi_root_label": label,
            "shared_evidence_channel_count": int(ev.get("shared_evidence_channel_count", 0) or 0),
            "evidence_channels": str(ev.get("evidence_channels", "")),
            "soft_memberships": "|".join(f"T{k}:{A[i, k]:.6g}" for k in range(k_count)),
        })
    diag = pd.DataFrame(rows)
    summary = (
        diag.groupby("multi_root_label", as_index=False)
        .agg(
            n_nodes=("node_id", "count"),
            mean_entropy=("root_entropy_norm", "mean"),
            mean_shared_score=("shared_score", "mean"),
            mean_active_roots=("n_active_roots", "mean"),
        )
        .sort_values("n_nodes", ascending=False)
        .reset_index(drop=True)
    )
    return diag, summary

def write_cmi_e_diagnostics(
    cmi_e_support: Optional[pd.DataFrame],
    outdir: Path,
    cfg: ProgressionConfig,
) -> dict[str, object]:
    """Write compact CMI-E environment and label diagnostics."""
    if cmi_e_support is None or cmi_e_support.empty:
        pd.DataFrame().to_csv(outdir / "cmi_e_environment_diagnostics.csv", index=False)
        pd.DataFrame().to_csv(outdir / "cmi_e_label_summary.csv", index=False)
        return {"cmi_e_available": False, "n_env_bins_min": 0, "n_env_bins_max": 0}

    diag_cols = [c for c in ["root_pos", "env_source", "feature_source", "n_env_bins_actual"] if c in cmi_e_support.columns]
    if diag_cols:
        env_diag = (
            cmi_e_support.groupby(diag_cols, as_index=False)
            .agg(
                n_edges=("source", "count"),
                mean_cmi_raw=("cmi_raw", "mean"),
                mean_cmi_env=("cmi_env", "mean"),
                mean_env_drop=("cmi_env_drop", "mean"),
                mean_env_trust=("cmi_env_trust", "mean"),
                mean_env_conflict=("cmi_env_conflict", "mean"),
            )
        )
    else:
        env_diag = pd.DataFrame()
    env_diag["env_bin_status"] = env_diag.get("n_env_bins_actual", pd.Series(dtype=float)).apply(
        lambda x: "ok" if pd.notna(x) and int(x) >= 2 else "degenerate_single_bin"
    ) if not env_diag.empty else pd.Series(dtype=str)
    env_diag.to_csv(outdir / "cmi_e_environment_diagnostics.csv", index=False)

    if "cmi_e_edge_label" in cmi_e_support.columns:
        label_summary = (
            cmi_e_support.groupby(["root_pos", "cmi_e_edge_label"], as_index=False)
            .agg(
                n_edges=("source", "count"),
                mean_env_drop=("cmi_env_drop", "mean"),
                mean_env_trust=("cmi_env_trust", "mean"),
                mean_env_conflict=("cmi_env_conflict", "mean"),
            )
        )
    else:
        label_summary = pd.DataFrame()
    label_summary.to_csv(outdir / "cmi_e_label_summary.csv", index=False)

    bins = pd.to_numeric(cmi_e_support.get("n_env_bins_actual", pd.Series([0])), errors="coerce").fillna(0).astype(int)
    return {
        "cmi_e_available": True,
        "n_env_bins_min": int(bins.min()) if len(bins) else 0,
        "n_env_bins_max": int(bins.max()) if len(bins) else 0,
        "env_sources": sorted(map(str, cmi_e_support.get("env_source", pd.Series(dtype=str)).dropna().unique().tolist())),
        "feature_sources": sorted(map(str, cmi_e_support.get("feature_source", pd.Series(dtype=str)).dropna().unique().tolist())),
        "n_environment_degenerate_rows": int((bins <= 1).sum()) if len(bins) else 0,
    }


def compute_tree_pair_relation_matrix_detailed(
    adata,
    A: np.ndarray,
    roots: list[int],
    final_edges: pd.DataFrame,
    progression_scores: dict[int, np.ndarray],
    cmi_e_support: Optional[pd.DataFrame],
    cfg: ProgressionConfig,
    active_assignment: Optional[np.ndarray] = None,
    node_domains: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Sparse v7.3 tree-pair relation matrix.

    A tree pair remains ``weak_or_independent`` unless spatial contact passes and
    at least one additional independent evidence channel passes. Diffuse Markov
    overlap alone is never sufficient.
    """
    K = A.shape[1]
    if K <= 1:
        return pd.DataFrame()
    if active_assignment is None:
        active_assignment = np.zeros_like(A, dtype=np.int8)
        primary_fallback = v71bio.safe_primary_roots(A, cfg.eps)
        for i, p in enumerate(primary_fallback):
            if p >= 0:
                active_assignment[i, p] = 1

    coords = legacy.get_spatial_coords(adata)
    try:
        coords_z = (coords - np.nanmean(coords, axis=0, keepdims=True)) / (
            np.nanstd(coords, axis=0, keepdims=True) + cfg.eps
        )
    except Exception:
        coords_z = coords

    rows: list[dict[str, object]] = []
    node_sets: dict[int, set[int]] = {}
    for k in range(K):
        subset = (
            final_edges.loc[final_edges["root_pos"].astype(int) == int(k)]
            if final_edges is not None and not final_edges.empty
            else pd.DataFrame()
        )
        nodes: set[int] = set()
        if not subset.empty:
            nodes.update(map(int, subset["source"].dropna().astype(int).tolist()))
            nodes.update(map(int, subset["target"].dropna().astype(int).tolist()))
        if not nodes and k < len(roots):
            nodes.add(int(roots[k]))
        node_sets[k] = nodes

    all_d = []
    for i in range(len(coords_z)):
        for j in range(i + 1, len(coords_z)):
            d = float(np.linalg.norm(coords_z[i] - coords_z[j]))
            if d > 0:
                all_d.append(d)
    scale = max(float(np.median(all_d)) if all_d else 1.0, cfg.eps)

    cmi = cmi_e_support if cmi_e_support is not None else pd.DataFrame()
    primary = v71bio.safe_primary_roots(A, cfg.eps)
    scope_lookup = {}
    if node_domains is not None and not node_domains.empty:
        scope_lookup = node_domains.set_index("node_id")["process_scope"].to_dict()
    root_scopes = [scope_lookup.get(int(r), "unknown_process") for r in roots]

    for k in range(K):
        for ell in range(k + 1, K):
            nk_set, nl_set = node_sets.get(k, set()), node_sets.get(ell, set())
            nk, nl = sorted(nk_set), sorted(nl_set)
            if nk and nl:
                vals = [
                    math.exp(
                        -(float(np.linalg.norm(coords_z[i] - coords_z[j])) ** 2)
                        / (2.0 * scale * scale + cfg.eps)
                    )
                    for i in nk
                    for j in nl
                ]
                contact = float(np.mean(vals)) if vals else 0.0
            else:
                contact = 0.0

            raw_overlap = float(np.mean(np.minimum(A[:, k], A[:, ell])))
            active_shared_mask = np.logical_and(active_assignment[:, k] > 0, active_assignment[:, ell] > 0)
            active_overlap = v71bio.active_overlap_score(
                active_assignment, k, ell, nk_set, nl_set, cfg.eps
            )
            n_active_shared = int(active_shared_mask.sum())

            if np.any(active_shared_mask):
                gk = progression_scores.get(k, np.zeros(A.shape[0]))
                gl = progression_scores.get(ell, np.zeros(A.shape[0]))
                x = np.asarray(gk[active_shared_mask], dtype=float)
                y = np.asarray(gl[active_shared_mask], dtype=float)
                if len(x) >= 3 and np.std(x) > cfg.eps and np.std(y) > cfg.eps:
                    prog_compat = float(np.corrcoef(x, y)[0, 1])
                    if not np.isfinite(prog_compat):
                        prog_compat = 0.0
                else:
                    # For very small validated shared sets, use a bounded similarity
                    # rather than claiming correlation from insufficient samples.
                    prog_compat = float(
                        np.clip(1.0 - np.mean(np.abs(x - y)), -1.0, 1.0)
                    ) if len(x) else 0.0
            else:
                prog_compat = 0.0

            cmi_conflict = 0.0
            cmi_trust = 0.0
            if not cmi.empty and {"source", "target", "root_pos"}.issubset(cmi.columns):
                tmp = cmi.loc[cmi["root_pos"].astype(int).isin([k, ell])].copy()
                if not tmp.empty:
                    src = tmp["source"].astype(int).to_numpy()
                    tgt = tmp["target"].astype(int).to_numpy()
                    valid = (
                        (src >= 0) & (src < len(primary)) & (tgt >= 0) & (tgt < len(primary))
                    )
                    tmp = tmp.loc[valid].copy()
                    if not tmp.empty:
                        src = tmp["source"].astype(int).to_numpy()
                        tgt = tmp["target"].astype(int).to_numpy()
                        cross = primary[src] != primary[tgt]
                        tmp = tmp.loc[cross]
                        if not tmp.empty:
                            cmi_conflict = float(
                                pd.to_numeric(tmp.get("cmi_env_conflict", 0.0), errors="coerce")
                                .fillna(0).mean()
                            )
                            cmi_trust = float(
                                pd.to_numeric(tmp.get("cmi_env_trust", 0.0), errors="coerce")
                                .fillna(0).mean()
                            )

            cross_domain = bool(
                k < len(root_scopes)
                and ell < len(root_scopes)
                and root_scopes[k] != root_scopes[ell]
            )
            relation_active, evidence_count, relation, channels = v71bio.sparse_tree_relation_gate(
                contact=contact,
                active_overlap=active_overlap,
                progression_compatibility=prog_compat,
                cmi_trust=cmi_trust,
                cross_domain=cross_domain,
                cfg=cfg,
            )
            convergence_score = contact * active_overlap * max(prog_compat, 0.0) * (1.0 - cmi_conflict)
            conflict_score = contact * active_overlap * max(-prog_compat, 0.0 + cmi_conflict)
            rows.append(
                {
                    "tree_k": int(k),
                    "tree_l": int(ell),
                    "root_k_index": int(roots[k]) if k < len(roots) else -1,
                    "root_l_index": int(roots[ell]) if ell < len(roots) else -1,
                    "root_k_name": str(adata.obs_names[int(roots[k])]) if k < len(roots) else "",
                    "root_l_name": str(adata.obs_names[int(roots[ell])]) if ell < len(roots) else "",
                    "n_nodes_tree_k": int(len(nk)),
                    "n_nodes_tree_l": int(len(nl)),
                    "n_validated_shared_nodes": n_active_shared,
                    "active_assignment_overlap": float(active_overlap),
                    "raw_membership_overlap_diagnostic": raw_overlap,
                    "spatial_contact_score": contact,
                    "progression_compatibility": prog_compat,
                    "mean_cross_root_cmi_env_conflict": cmi_conflict,
                    "mean_cross_root_cmi_env_trust": cmi_trust,
                    "convergence_relation_score": float(convergence_score),
                    "conflict_relation_score": float(conflict_score),
                    "relation_scope": "cross_domain_interaction" if cross_domain else "within_process_trajectory",
                    "relation_active": bool(relation_active),
                    "evidence_channel_count": int(evidence_count),
                    "can_modify_trajectory_cost": bool(relation_active and not cross_domain),
                    "can_create_trajectory_edge": False,
                    "dominant_tree_pair_relation": relation,
                    **channels,
                    "evidence_source": "active_assignment+final_tree_contact+progression+CMI-E",
                    "evidence_status": "model_inferred_association_only",
                }
            )
    return pd.DataFrame(rows)

def write_engineering_output_manifest(outdir: Path) -> pd.DataFrame:
    """Write a stable manifest of major method outputs."""
    descriptions = {
        "global_evidence_graph_edges.csv": "Global metacell evidence graph before biological trajectory eligibility filtering.",
        "trajectory_candidate_edges.csv": "Biologically eligible trajectory graph used for process-constrained Markov and MST construction.",
        "interaction_edges.csv": "Interaction-only or conditional cross-process evidence graph; never used to create trajectory topology by itself.",
        "biological_node_domains.csv": "Node annotation, major compartment and conservative process scope.",
        "root_anchor_sets.csv": "Disjoint local absorbing super-root anchors selected from process, expression, spatial and graph evidence.",
        "root_validity_audit.csv": "Truth-First audit of root projection, type compatibility, anchor support, graph degree and local scope.",
        "process_program_scores.csv": "Per-metacell molecular-program scores; unavailable signatures remain missing rather than guessed.",
        "process_program_availability.csv": "Observed-gene coverage and availability of each internal TME program signature.",
        "pre_truth_gate_backbone_tree_edges.csv": "Structural candidate backbone preserved before Truth-First rejection; audit only, never final biological claim.",
        "edge_truth_gate_audit.csv": "Per-edge hard-veto, root-connectivity and threshold-stability audit; rescue cannot override failure.",
        "edge_truth_gate_passed_backbone_edges.csv": "Backbone edges that pass Truth-First edge evidence gates before root-level trajectory existence gating.",
        "tree_program_progression_evidence.csv": "Multiple-testing-controlled program-versus-pseudotime evidence by root and program.",
        "tree_program_evidence_summary.csv": "Root-level program coherence and interface-composition support summary.",
        "trajectory_truth_gate_by_root.csv": "Level 0-3 trajectory claim status; unsupported roots explicitly make no trajectory claim.",
        "process_distinctness_matrix.csv": "Program-based distinction of root trees; spatial separation alone never establishes a distinct biological process.",
        "truth_first_validation_overview.csv": "Dataset-level Truth-First validation overview and counts of rejected rescue edges and supported trajectories.",
        "root_local_affinity.csv": "Per-node per-root local affinity restricting same-process Markov competition to biologically local domains.",
        "membership_A_raw.csv": "Raw root-local process-constrained absorbing-Markov probabilities.",
        "membership_A_bio.csv": "Raw probabilities after biological root eligibility masking and renormalization.",
        "membership_A_sharp.csv": "Optional globally sharpened root probabilities used for root competition diagnostics.",
        "active_root_assignment.csv": "Final sparse active-root matrix: primary-only by default, at most one validated secondary root.",
        "assignment_uncertainty_diagnostics.csv": "Top-2 probabilities, entropy, evidence channels and secondary-root activation decision.",
        "soft_membership.csv": "Backward-compatible A_sharp table; active roots come from evidence-gated Z_active.",
        "multi_root_membership_diagnostics.csv": "Node-level primary/secondary root, entropy, shared score and multi-root labels.",
        "multi_root_membership_summary.csv": "Counts and mean uncertainty by multi-root label.",
        "candidate_graph_edges.csv": "Backward-compatible alias of trajectory_candidate_edges.csv inherited from the structural base stage.",
        "biological_compatibility_audit_final.csv": "Hard audit requiring zero forbidden trajectory edges in the final selected forest.",
        "cmi_e_edge_evidence.csv": "Root-specific CMI-E edge evidence, raw/env dependency, env drop, trust and conflict labels.",
        "cmi_e_environment_diagnostics.csv": "Environment-source and bin-count diagnostics for CMI-E; checks single-bin degeneration.",
        "cmi_e_label_summary.csv": "CMI-E edge label counts and average gate statistics by root.",
        "interaction_node_scores_v2.csv": "Node-level promotion, inhibition, collision, merge and conflict evidence.",
        "interaction_edge_scores_v2.csv": "Edge-level interaction modifiers entering the refined MST cost.",
        "tree_pair_relation_matrix_detailed.csv": "Pairwise root-tree contact, overlap, progression compatibility and CMI-E conflict summary.",
        "tree_selection_decisions.csv": "Acceptance-rule decision for base/progression/interaction-refined tree per root.",
        "final_selected_backbone_tree_edges.csv": "Final Truth-First validated trajectory backbone; Level-0 roots and failed rescue edges are excluded.",
        "final_node_assignments.csv": "Method-level node assignment to origin, branch segment and final pseudotime.",
        "final_constrained_pseudotime_by_root.csv": "Root-conditioned constrained pseudotime for every node retained by each selected root system.",
        "final_branch_segments.csv": "Branch segment table after long-segment split.",
        "final_prediction_metacell.json": "Frozen metacell-level prediction JSON consumed by external scoring adapter.",
        "run_summary.json": "Dataset-level configuration and output summary.",
    }
    rows = []
    for filename, desc in descriptions.items():
        path = outdir / filename
        rows.append(
            {
                "filename": filename,
                "exists": bool(path.exists()),
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
                "description": desc,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(outdir / "method_output_manifest.csv", index=False)
    return manifest


def _write_report(outdir: Path, dataset: str, decisions: pd.DataFrame, cfg: ProgressionConfig) -> None:
    counts = decisions["selected_tree_label"].value_counts().to_dict() if not decisions.empty else {}
    report = f"""# CASCAT-MR-CMI-E-Gate v7.4.2 Truth-First report: {dataset}

## Core scientific rule

The method does not require one tree per root.  A root may end as
`insufficient_support`, and no trajectory is claimed in that case.  A final
trajectory edge must survive biological compatibility, confidence, extreme-cost,
CMI-E conflict, threshold-stability, and root-connectivity gates.  A
`root_path_rescue` edge cannot override a hard veto.

## Claim hierarchy

- Level 0: insufficient support; no trajectory claim.
- Level 1: structurally plausible trajectory.
- Level 2: process-supported trajectory with multiple-testing-controlled molecular-program evidence and, for interface processes, composition support.
- Level 3: externally biologically supported; never assigned without explicit external evidence.

Spatial separation alone is not used to call two trees distinct biological
processes.  Same-scope trees are labelled as different spatial branches of the
same process family unless molecular-program separation is independently
supported.

## Tree-selection summary

- Selected structural labels before Truth-First gating: `{json.dumps(counts, ensure_ascii=False)}`
- Progression source: `{cfg.progression_source}`
- Markov sharpening q: `{getattr(cfg, 'markov_sharpen_q', 1.0)}`
- Minimum trajectory edges/nodes: `{getattr(cfg, 'truth_min_trajectory_edges', 3)}` / `{getattr(cfg, 'truth_min_trajectory_nodes', 4)}`
- Minimum edge stability: `{getattr(cfg, 'truth_min_edge_stability', 0.60)}`
- Rescue minimum confidence/stability: `{getattr(cfg, 'truth_rescue_min_confidence', 0.55)}` / `{getattr(cfg, 'truth_rescue_min_stability', 0.80)}`

## Interpretation limitation

Internal molecular programs and CMI-E are evidence channels, not experimental
truth.  Program signatures that are absent from the assay remain unavailable;
the code does not replace missing evidence with an embedding-derived biological
label.  Synthetic truth, when supplied in benchmark mode, is diagnostic unless
oracle construction is explicitly enabled.
"""
    (outdir / "progression_refined_tree_report.md").write_text(report, encoding="utf-8")


def upgrade_dataset(
    adata_path: Path,
    dataset: str,
    base_outdir: Path,
    cfg: ProgressionConfig,
    truth_pseudotime_csv: Optional[Path],
    time_key: Optional[str],
    sample_key: Optional[str],
    benchmark_mode: bool,
) -> dict[str, object]:
    """Upgrade the v7.3 structural base with v7.4.2 Truth-First evidence gates."""
    outdir = base_outdir / dataset
    # v7.4 progression operates on the exact metacell AnnData created by the base
    # stage. Falling back to raw cells would silently invalidate root mapping.
    metacell_path = outdir / "metacell_adata.h5ad"
    if not metacell_path.exists():
        raise FileNotFoundError(
            f"v7.4 requires {metacell_path}; the base metacell stage did not complete successfully"
        )
    adata = legacy.read_h5ad(metacell_path)

    A, roots = _load_membership_and_roots(outdir)
    Z_initial = _load_active_assignment(outdir, A.shape[0], A.shape[1])
    node_domains_path = outdir / "biological_node_domains.csv"
    node_domains = (
        pd.read_csv(node_domains_path)
        if node_domains_path.exists()
        else v71bio.infer_node_biological_domains(adata, getattr(cfg, "annotation_key", None))
    )
    node_domains.to_csv(node_domains_path, index=False)
    mask_path = outdir / "biological_root_eligibility_mask.csv"
    if mask_path.exists():
        mask_df = pd.read_csv(mask_path)
        mask_cols = [f"root_{k}" for k in range(A.shape[1])]
        if all(c in mask_df.columns for c in mask_cols) and len(mask_df) == A.shape[0]:
            compatibility_mask = mask_df[mask_cols].to_numpy(dtype=bool)
        else:
            raise ValueError("biological_root_eligibility_mask.csv has incompatible shape/columns")
    else:
        compatibility_mask = np.ones_like(A, dtype=bool)
        for k, root in enumerate(roots):
            compatibility_mask[int(root), k] = True

    trajectory_path = outdir / "trajectory_candidate_edges.csv"
    edges_df = pd.read_csv(trajectory_path if trajectory_path.exists() else outdir / "candidate_graph_edges.csv")
    interaction_edges_path = outdir / "interaction_edges.csv"
    interaction_graph_edges = pd.read_csv(interaction_edges_path) if interaction_edges_path.exists() else pd.DataFrame()
    emb = legacy.get_expression_embedding(adata, cfg)
    domains = legacy.get_domain_labels(adata)

    truth = None
    if truth_pseudotime_csv is not None:
        if not benchmark_mode:
            warnings.warn("--truth-pseudotime-csv ignored without --benchmark-mode")
        else:
            truth = _truth_vector(truth_pseudotime_csv, adata)
    construction_truth_allowed = bool(
        benchmark_mode
        and truth is not None
        and cfg.progression_source == "truth"
        and cfg.allow_oracle_construction
    )
    construction_truth = truth if construction_truth_allowed else None
    if truth is not None and not construction_truth_allowed:
        warnings.warn("truth pseudotime is available for diagnostics, but is not used for tree construction")

    progression_long, source_summary, progression_scores, progression_sources = compute_progression_scores(
        adata,
        A,
        roots,
        emb,
        edges_df,
        cfg,
        construction_truth,
        time_key,
        cfg.progression_source,
        cfg.allow_oracle_construction,
    )
    progression_long.to_csv(outdir / "progression_scores_long.csv", index=False)
    source_summary.to_csv(outdir / "progression_score_sources_summary.csv", index=False)

    cmi_e_support = compute_cmi_e_edge_support(adata, edges_df, A, roots, emb, cfg)
    if cmi_e_support is not None and not cmi_e_support.empty:
        cmi_e_support.to_csv(outdir / "cmi_e_edge_support.csv", index=False)
        cmi_e_support.to_csv(outdir / "cmi_e_edge_evidence.csv", index=False)
    else:
        cmi_e_support = pd.DataFrame()
        pd.DataFrame().to_csv(outdir / "cmi_e_edge_support.csv", index=False)
        pd.DataFrame().to_csv(outdir / "cmi_e_edge_evidence.csv", index=False)
    cmi_e_diagnostic_summary = write_cmi_e_diagnostics(cmi_e_support, outdir, cfg)

    # v7.4 final active assignment: primary-only by default; one secondary root can
    # be activated only with probability + uncertainty + biological + independent evidence.
    Z_final, assignment_evidence = v71bio.refine_sparse_active_assignment(
        A_sharp=A,
        roots=roots,
        node_domains=node_domains,
        trajectory_edges=edges_df,
        cfg=cfg,
        progression_scores=progression_scores,
        cmi_e_support=cmi_e_support,
    )
    A_raw_path = outdir / "membership_A_raw.csv"
    A_bio_path = outdir / "membership_A_bio.csv"
    A_raw = (
        v71bio.load_membership_matrix(A_raw_path, "A_raw", A.shape[0], A.shape[1])
        if A_raw_path.exists()
        else A.copy()
    )
    A_bio = (
        v71bio.load_membership_matrix(A_bio_path, "A_bio", A.shape[0], A.shape[1])
        if A_bio_path.exists()
        else A.copy()
    )
    v71bio.write_assignment_outputs(
        outdir,
        adata,
        roots,
        A_raw,
        A_bio,
        A,
        Z_final,
        assignment_evidence,
        stage="final",
    )
    # Stable final aliases used by downstream code and visualisation.
    pd.read_csv(outdir / "active_root_assignment_final.csv").to_csv(
        outdir / "active_root_assignment.csv", index=False
    )
    assignment_evidence.to_csv(outdir / "assignment_uncertainty_diagnostics.csv", index=False)

    membership_diag, membership_summary = compute_multi_root_membership_diagnostics(
        adata, A, roots, cfg, active_assignment=Z_final, assignment_evidence=assignment_evidence
    )
    membership_diag.to_csv(outdir / "multi_root_membership_diagnostics.csv", index=False)
    membership_summary.to_csv(outdir / "multi_root_membership_summary.csv", index=False)

    labels: dict[str, pd.DataFrame] = {}
    tree_sets: dict[str, dict[int, nx.Graph]] = {}
    pseudo_tables: list[pd.DataFrame] = []
    summary_tables: dict[str, pd.DataFrame] = {}
    for label, mode in (("base", "base"), ("progression", "progression")):
        edge_parts, summaries, trees = [], [], {}
        for k, root in enumerate(roots):
            table, summary, tree, pseudo = build_one_tree(
                adata,
                edges_df,
                A,
                root,
                k,
                emb,
                domains,
                cfg,
                progression_scores=progression_scores,
                progression_sources=progression_sources,
                cost_mode=mode,
                tree_label=label,
                truth_pseudotime=construction_truth,
                cmi_e_support=cmi_e_support,
                active_assignment=Z_final,
                compatibility_mask=compatibility_mask,
            )
            edge_parts.append(table)
            summaries.append(summary)
            trees[k] = tree
            pseudo_tables.append(pseudo)
        labels[label] = pd.concat(edge_parts, ignore_index=True) if edge_parts else pd.DataFrame()
        tree_sets[label] = trees
        summary_tables[label] = pd.DataFrame(summaries)
        labels[label].to_csv(outdir / f"{label}_full_tree_edges.csv", index=False)
        labels[label].to_csv(outdir / f"{label}_tree_edges.csv", index=False)
        summary_tables[label].to_csv(outdir / f"{label}_tree_summaries.csv", index=False)

    topology_qc_parts = []
    backbone_labels: dict[str, pd.DataFrame] = {}
    backbone_summaries: dict[str, pd.DataFrame] = {}
    current_pseudotime_table = pd.concat(pseudo_tables, ignore_index=True) if pseudo_tables else pd.DataFrame()
    for _label in ("base", "progression"):
        bb_edges, bb_sum, bb_qc = build_backbone_tables_for_label(
            labels, tree_sets, current_pseudotime_table, roots, A, progression_scores, cfg, _label
        )
        backbone_labels[_label] = bb_edges
        backbone_summaries[_label] = bb_sum
        bb_edges.to_csv(outdir / f"{_label}_backbone_tree_edges.csv", index=False)
        bb_sum.to_csv(outdir / f"{_label}_backbone_tree_summaries.csv", index=False)
        topology_qc_parts.append(bb_qc)

    # Interaction modifiers are sparse: only validated dual-root nodes can produce
    # node-level modifiers, and only trajectory-eligible edges can be reweighted.
    node_scores, collision_decisions, node_modifiers = compute_interaction_scores_v2(
        adata, A, roots, progression_scores, tree_sets["progression"], cfg,
        active_assignment=Z_final,
    )
    edge_modifiers = build_interaction_edge_modifiers_v2(
        adata, edges_df, A, progression_scores, node_modifiers, domains, cfg, sample_key
    )
    (
        interaction_edges,
        interaction_summaries,
        interaction_trees,
        interaction_pseudo,
        refinement_audit,
    ) = apply_conservative_interaction_refinement(
        adata,
        edges_df,
        A,
        roots,
        emb,
        domains,
        cfg,
        progression_scores,
        progression_sources,
        labels["progression"],
        edge_modifiers,
        truth,
        cmi_e_support=cmi_e_support,
        active_assignment=Z_final,
        compatibility_mask=compatibility_mask,
    )
    labels["interaction_refined"] = interaction_edges
    tree_sets["interaction_refined"] = interaction_trees
    pseudo_tables.append(interaction_pseudo)
    interaction_edges.to_csv(outdir / "interaction_v2_full_tree_edges.csv", index=False)
    interaction_edges.to_csv(outdir / "interaction_v2_tree_edges.csv", index=False)
    interaction_summaries.to_csv(outdir / "interaction_v2_tree_summaries.csv", index=False)

    tmp_pseudo_for_interaction = interaction_pseudo.copy()
    backbone_labels["interaction_refined"], backbone_summaries["interaction_refined"], bb_qc_interaction = build_backbone_tables_for_label(
        {"interaction_refined": interaction_edges},
        {"interaction_refined": interaction_trees},
        tmp_pseudo_for_interaction,
        roots,
        A,
        progression_scores,
        cfg,
        "interaction_refined",
    )
    backbone_labels["interaction_refined"].to_csv(outdir / "interaction_v2_backbone_tree_edges.csv", index=False)
    backbone_summaries["interaction_refined"].to_csv(outdir / "interaction_v2_backbone_tree_summaries.csv", index=False)
    topology_qc_parts.append(bb_qc_interaction)
    node_scores.to_csv(outdir / "interaction_node_scores_v2.csv", index=False)
    edge_modifiers.to_csv(outdir / "interaction_edge_scores_v2.csv", index=False)
    collision_decisions.to_csv(outdir / "collision_decision_v2.csv", index=False)
    refinement_audit.to_csv(outdir / "interaction_refinement_audit_v2.csv", index=False)

    pseudotime_table = pd.concat(pseudo_tables, ignore_index=True) if pseudo_tables else pd.DataFrame()
    branches = _branch_table(adata, interaction_trees, interaction_pseudo, edge_modifiers)
    branches.to_csv(outdir / "interaction_branch_scores_v2.csv", index=False)
    if not node_scores.empty and {"tree_k", "tree_l"}.issubset(node_scores.columns):
        pair_summary = (
            node_scores.groupby(["tree_k", "tree_l"], as_index=False)
            .agg(
                mean_promotion_v2=("promotion_score_v2", "mean"),
                mean_inhibition_v2=("inhibition_score_v2", "mean"),
                mean_collision_v2=("collision_score_v2", "mean"),
                mean_merge=("merge_score", "mean"),
                mean_conflict=("conflict_score", "mean"),
            )
        )
    else:
        pair_summary = pd.DataFrame(columns=[
            "tree_k", "tree_l", "mean_promotion_v2", "mean_inhibition_v2",
            "mean_collision_v2", "mean_merge", "mean_conflict"
        ])
    pair_summary.to_csv(outdir / "tree_pair_interaction_summary_v2.csv", index=False)

    qc_rows = []
    for k, root in enumerate(roots):
        base_reference = _tree_edge_set(labels["base"], k)
        for label in ("base", "progression", "interaction_refined"):
            reference = base_reference if label != "interaction_refined" else _tree_edge_set(labels["progression"], k)
            qc_rows.append(
                compute_tree_internal_qc(labels[label], A, progression_scores, edges_df, k, root, label, reference)
            )
    qc_table = pd.DataFrame(qc_rows)
    qc_table.to_csv(outdir / "tree_internal_qc.csv", index=False)
    final_edges, final_summaries, decisions = select_final_tree(dataset, roots, labels, qc_table, cfg)
    final_edges.to_csv(outdir / "final_selected_full_tree_edges.csv", index=False)
    final_edges.to_csv(outdir / "final_selected_tree_edges.csv", index=False)
    final_summaries.to_csv(outdir / "final_selected_full_tree_summaries.csv", index=False)
    final_summaries.to_csv(outdir / "final_selected_tree_summaries.csv", index=False)
    decisions.to_csv(outdir / "tree_selection_decisions.csv", index=False)
    decisions.to_csv(outdir / "interaction_acceptance_decisions_v2.csv", index=False)

    final_pseudo_parts = []
    for k, root in enumerate(roots):
        selected_rows = decisions.loc[decisions.root_pos == k, "selected_tree_label"]
        selected = selected_rows.iloc[0] if not selected_rows.empty else "progression"
        chosen = pseudotime_table.loc[
            (pseudotime_table.root_pos == k) & (pseudotime_table.tree_label == selected)
        ].copy() if not pseudotime_table.empty else pd.DataFrame()
        if not chosen.empty:
            chosen["tree_label"] = "final_selected"
            chosen["selected_from_tree_label"] = selected
            final_pseudo_parts.append(chosen)
    final_pseudo = pd.concat(final_pseudo_parts, ignore_index=True) if final_pseudo_parts else pd.DataFrame()
    final_pseudo.to_csv(outdir / "final_selected_pseudotime.csv", index=False)
    all_pseudotime = pd.concat([pseudotime_table, final_pseudo], ignore_index=True) if not pseudotime_table.empty else final_pseudo.copy()
    all_pseudotime.to_csv(outdir / "pseudotime_table.csv", index=False)

    final_tree_sets = {"final_selected": {}}
    for k, root in enumerate(roots):
        subset = final_edges.loc[final_edges["root_pos"] == k] if not final_edges.empty and "root_pos" in final_edges.columns else pd.DataFrame()
        final_tree_sets["final_selected"][k] = _tree_from_edge_table(subset)
    final_backbone_edges, final_backbone_summaries, final_backbone_qc = build_backbone_tables_for_label(
        {"final_selected": final_edges},
        final_tree_sets,
        final_pseudo,
        roots,
        A,
        progression_scores,
        cfg,
        "final_selected",
    )
    # ------------------------------------------------------------------
    # v7.4.2 Truth-First layer.  Preserve the structural candidate backbone for
    # audit, then apply root validity, hard edge veto, stability, trajectory
    # existence, molecular-program coherence, and process-distinctness gates.
    # No failed rescue edge can be reintroduced for connectivity.
    # ------------------------------------------------------------------
    pre_truth_backbone_edges = final_backbone_edges.copy()
    pre_truth_backbone_summaries = final_backbone_summaries.copy()
    pre_truth_backbone_edges.to_csv(outdir / "pre_truth_gate_backbone_tree_edges.csv", index=False)
    pre_truth_backbone_summaries.to_csv(outdir / "pre_truth_gate_backbone_tree_summaries.csv", index=False)

    root_validity_audit = v74truth.build_root_validity_audit(
        adata=adata,
        roots=roots,
        results_dir=outdir,
        trajectory_edges=edges_df,
        cfg=cfg,
    )
    root_validity_audit.to_csv(outdir / "root_validity_audit.csv", index=False)

    process_program_scores, process_program_availability = v74truth.compute_process_program_scores(
        adata,
        min_genes=int(getattr(cfg, "truth_program_min_genes", 2)),
    )
    process_program_scores.to_csv(outdir / "process_program_scores.csv", index=False)
    process_program_availability.to_csv(outdir / "process_program_availability.csv", index=False)

    edge_truth_filtered, edge_truth_gate_audit = v74truth.apply_edge_truth_gate(
        pre_truth_backbone_edges=pre_truth_backbone_edges,
        roots=roots,
        cfg=cfg,
    )
    edge_truth_gate_audit.to_csv(outdir / "edge_truth_gate_audit.csv", index=False)
    edge_truth_filtered.to_csv(outdir / "edge_truth_gate_passed_backbone_edges.csv", index=False)

    program_detail, program_summary = v74truth.build_tree_program_evidence(
        program_scores=process_program_scores,
        program_availability=process_program_availability,
        final_edges=edge_truth_filtered,
        final_pseudo=final_pseudo,
        roots=roots,
        results_dir=outdir,
        cfg=cfg,
    )
    program_detail.to_csv(outdir / "tree_program_progression_evidence.csv", index=False)
    program_summary.to_csv(outdir / "tree_program_evidence_summary.csv", index=False)

    trajectory_truth_gate = v74truth.build_trajectory_truth_gate(
        final_edges=edge_truth_filtered,
        roots=roots,
        root_validity=root_validity_audit,
        program_summary=program_summary,
        cfg=cfg,
    )
    trajectory_truth_gate.to_csv(outdir / "trajectory_truth_gate_by_root.csv", index=False)

    # Level 0 roots make no trajectory claim and contribute no final backbone
    # edges.  Their structural candidates remain in pre_truth_gate_* for audit.
    final_backbone_edges = v74truth.filter_edges_by_trajectory_truth_level(
        edge_truth_filtered,
        trajectory_truth_gate,
        min_level=1,
    )
    final_backbone_summaries = v74truth.summarise_backbone_edges(final_backbone_edges, roots, adata)
    final_backbone_edges.to_csv(outdir / "final_selected_backbone_tree_edges.csv", index=False)
    final_backbone_summaries.to_csv(outdir / "final_selected_backbone_tree_summaries.csv", index=False)

    process_distinctness = v74truth.build_process_distinctness_matrix(
        final_edges=final_backbone_edges,
        roots=roots,
        trajectory_gate=trajectory_truth_gate,
        program_scores=process_program_scores,
        program_availability=process_program_availability,
        adata=adata,
        results_dir=outdir,
        cfg=cfg,
    )
    process_distinctness.to_csv(outdir / "process_distinctness_matrix.csv", index=False)

    truth_first_overview = v74truth.build_truth_first_overview(
        dataset=dataset,
        roots=roots,
        root_validity=root_validity_audit,
        edge_audit=edge_truth_gate_audit,
        trajectory_gate=trajectory_truth_gate,
        distinctness=process_distinctness,
    )
    truth_first_overview.to_csv(outdir / "truth_first_validation_overview.csv", index=False)

    topology_qc_parts.append(final_backbone_qc)
    tree_topology_qc = pd.concat(topology_qc_parts, ignore_index=True) if topology_qc_parts else pd.DataFrame()
    tree_topology_qc.to_csv(outdir / "tree_topology_qc.csv", index=False)

    tree_pair_relation_detailed = compute_tree_pair_relation_matrix_detailed(
        adata=adata,
        A=A,
        roots=roots,
        final_edges=final_backbone_edges,
        progression_scores=progression_scores,
        cmi_e_support=cmi_e_support,
        cfg=cfg,
        active_assignment=Z_final,
        node_domains=node_domains,
    )
    if not process_distinctness.empty and not tree_pair_relation_detailed.empty:
        tree_pair_relation_detailed = tree_pair_relation_detailed.merge(
            process_distinctness[[
                "tree_k", "tree_l", "distinct_process_supported",
                "process_relation_classification", "program_centroid_distance",
                "top_differential_program",
            ]],
            on=["tree_k", "tree_l"],
            how="left",
        )
    tree_pair_relation_detailed.to_csv(outdir / "tree_pair_relation_matrix_detailed.csv", index=False)

    # Hard biological audit. A final Truth-First trajectory forest may not
    # contain a forbidden cross-process trajectory edge.
    biological_audit = v71bio.audit_tree_biological_validity(final_backbone_edges, node_domains)
    biological_audit.to_csv(outdir / "biological_compatibility_audit_final.csv", index=False)
    n_forbidden = int(biological_audit["is_forbidden_trajectory_edge"].sum()) if not biological_audit.empty else 0
    if n_forbidden != 0:
        raise RuntimeError(f"v7.4 hard constraint failed: {n_forbidden} forbidden final trajectory edges")

    # Truth-First exports are always the evidence-gated backbone. Unsupported
    # full-tree candidates are audit artifacts, never final biological claims.
    export_edges = final_backbone_edges
    export_summaries = final_backbone_summaries
    export_edges.to_csv(outdir / "root_specific_tree_edges.csv", index=False)
    export_edges.to_csv(outdir / "refined_root_specific_tree_edges.csv", index=False)
    labels["base"].to_csv(outdir / "initial_root_specific_tree_edges.csv", index=False)
    backbone_labels.get("base", labels["base"]).to_csv(outdir / "initial_backbone_root_specific_tree_edges.csv", index=False)
    export_summaries.to_csv(outdir / "tree_summaries.csv", index=False)
    export_summaries.to_csv(outdir / "refined_tree_summaries.csv", index=False)

    method_output_summary = build_final_method_outputs(
        adata=adata,
        final_backbone_edges=final_backbone_edges,
        final_pseudo=final_pseudo,
        roots=roots,
        A=A,
        outdir=outdir,
        dataset=dataset,
        cfg=cfg,
        method_name="cascat_mr_cmi_e_v7_4_3_2_truth_first_forest",
        active_assignment=Z_final,
    )

    collision_decisions.assign(
        classification=collision_decisions.get("collision_decision", pd.Series(dtype=str))
    ).to_csv(outdir / "convergence_nodes.csv", index=False)
    pair_summary.to_csv(outdir / "tree_interactions.csv", index=False)
    pair_summary.to_csv(outdir / "tree_pair_interaction_summary.csv", index=False)

    benchmark_placeholder = pd.DataFrame(
        {
            "dataset": dataset,
            "tree_label": ["base", "progression", "interaction_refined", "final_selected"],
            "Direction_accuracy": np.nan,
            "Pseudotime_Spearman": np.nan,
            "Pseudotime_Kendall": np.nan,
            "Topology_score": np.nan,
            "scoring_status": "run_external_teammate_score.py_after_prediction_export",
        }
    )
    benchmark_placeholder.to_csv(outdir / "benchmark_scores_by_tree_label.csv", index=False)
    benchmark_placeholder.iloc[0:0].to_csv(outdir / "benchmark_score_delta.csv", index=False)
    output_manifest = write_engineering_output_manifest(outdir)
    _write_report(outdir, dataset, decisions, cfg)

    # v7.4.2 Truth-First biological interpretation is part of the end-to-end contract.
    biological_interpretation_manifest = v72viz.generate_biological_interpretation(
        adata=adata,
        results_dir=outdir,
        dataset_name=dataset,
        annotation_key=getattr(cfg, "annotation_key", None),
        make_figures=True,
    )

    active_counts = Z_final.sum(axis=1)
    if not tree_pair_relation_detailed.empty:
        active_relation_rate = float(tree_pair_relation_detailed["relation_active"].astype(bool).mean())
        relation_counts = tree_pair_relation_detailed["dominant_tree_pair_relation"].value_counts().to_dict()
    else:
        active_relation_rate = 0.0
        relation_counts = {}
    summary = {
        "dataset": dataset,
        "method": "CASCAT-MR-CMI-E-Gate_v7.4.3.2_Truth-First",
        "n_obs": adata.n_obs,
        "n_roots": len(roots),
        "progression_sources": progression_sources,
        "selected_tree_counts": decisions["selected_tree_label"].value_counts().to_dict() if not decisions.empty else {},
        "visualizations_generated": True,
        "biological_interpretation_manifest": biological_interpretation_manifest,
        "truth_supplied_for_diagnostics": bool(truth is not None),
        "truth_used_for_construction": bool(construction_truth_allowed),
        "oracle_construction_allowed": bool(cfg.allow_oracle_construction),
        "scoring_export_form": "backbone" if cfg.score_backbone_not_full else "full",
        "validated_dual_root_nodes": int((active_counts == 2).sum()),
        "validated_dual_root_rate": float((active_counts == 2).mean()) if len(active_counts) else 0.0,
        "high_order_ambiguity_nodes": int((assignment_evidence.get("assignment_status", pd.Series(dtype=str)) == "high_order_ambiguity").sum()),
        "active_tree_pair_relation_rate": active_relation_rate,
        "forbidden_final_trajectory_edges": n_forbidden,
        "global_interaction_graph_edges": int(len(interaction_graph_edges)),
        "method_output_files": method_output_summary,
        "engineering_output_files": output_manifest.loc[output_manifest["exists"], "filename"].astype(str).tolist(),
        "cmi_e_diagnostic_summary": cmi_e_diagnostic_summary,
        "membership_diagnostic_counts": membership_diag["multi_root_label"].value_counts().to_dict() if not membership_diag.empty else {},
        "tree_pair_relation_counts": relation_counts,
        "truth_first_status_counts": trajectory_truth_gate["trajectory_truth_status"].value_counts().to_dict() if not trajectory_truth_gate.empty else {},
        "truth_level_counts": trajectory_truth_gate["truth_level"].value_counts().sort_index().to_dict() if not trajectory_truth_gate.empty else {},
        "distinct_process_pairs_supported": int(process_distinctness.get("distinct_process_supported", pd.Series(dtype=bool)).astype(bool).sum()) if not process_distinctness.empty else 0,
        "truth_first_overview": truth_first_overview.iloc[0].to_dict() if not truth_first_overview.empty else {},
        "config": asdict(cfg),
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

def _paths_from_args(args) -> list[tuple[Optional[str], Path]]:
    if args.adata:
        return [(args.dataset_name, Path(args.adata))]
    if args.adata_glob:
        return [(None, Path(path)) for path in sorted(glob.glob(args.adata_glob))]
    return legacy.default_dataset_paths()


def _bool_switch(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--enable-{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--disable-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adata")
    source.add_argument("--adata-glob")
    source.add_argument("--run-defaults", action="store_true")
    parser.add_argument("--roots-csv", required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-col")
    parser.add_argument("--root-col")
    parser.add_argument("--root-specific-nmf-dir")
    parser.add_argument("--outdir", default="outputs/tme_progression_refined")
    parser.add_argument("--k-neighbors", type=int, default=6)
    parser.add_argument("--n-pca", type=int, default=30)
    parser.add_argument("--theta-a", type=float, default=0.30)
    parser.add_argument("--beta-transition", type=float, default=4.0)
    parser.add_argument("--theta-secondary", type=float, default=0.25)
    parser.add_argument("--theta-entropy", type=float, default=0.70)
    parser.add_argument("--annotation-key", default=None)
    parser.add_argument("--input-level", choices=("auto", "cell", "metacell"), default="auto")
    parser.add_argument("--metacell-key", default=None)
    parser.add_argument("--metacell-target-size", type=int, default=50)
    parser.add_argument("--metacell-min-size", type=int, default=10)
    parser.add_argument("--metacell-max-size", type=int, default=150)
    parser.add_argument("--metacell-birch-threshold-multiplier", type=float, default=1.5)
    parser.add_argument("--metacell-spatial-weight", type=float, default=0.35)
    parser.add_argument("--metacell-tiny-merge-size", type=int, default=5)
    parser.add_argument("--tiny-merge-min-comp-similarity", type=float, default=0.55)
    parser.add_argument("--tiny-merge-max-spatial-nn-factor", type=float, default=4.0)
    parser.add_argument("--root-projection-top-k", type=int, default=25)
    parser.add_argument("--root-projection-type-weight", type=float, default=0.35)
    parser.add_argument("--root-node-compatibility-threshold", type=float, default=0.20)
    parser.add_argument("--root-competition-jaccard-threshold", type=float, default=0.50)
    parser.add_argument("--min-assigned-fraction", type=float, default=0.05)
    parser.add_argument("--composition-similarity-allowed", type=float, default=0.55)
    parser.add_argument("--mixed-bridge-min-mass", type=float, default=0.20)
    parser.add_argument("--pure-compartment-threshold", type=float, default=0.80)
    parser.add_argument("--process-active-fraction", type=float, default=0.10)
    parser.add_argument("--process-bridge-min-composition-similarity", type=float, default=0.35)
    parser.add_argument("--process-bridge-min-shared-mass", type=float, default=0.10)
    parser.add_argument("--root-anchor-max-nodes", type=int, default=7)
    parser.add_argument("--root-anchor-max-hops", type=int, default=3)
    parser.add_argument("--root-anchor-min-process-score", type=float, default=0.25)
    parser.add_argument("--root-local-affinity-threshold", type=float, default=0.42)
    parser.add_argument("--root-local-affinity-power", type=float, default=1.5)
    parser.add_argument("--root-local-min-nodes", type=int, default=5)
    parser.add_argument("--allow-conditional-trajectory-edges", action="store_true")
    parser.add_argument("--markov-sharpen-q", type=float, default=1.5)
    parser.add_argument("--shared-ratio-threshold", type=float, default=0.65)
    parser.add_argument("--shared-margin-threshold", type=float, default=0.20)
    parser.add_argument("--min-shared-evidence-channels", type=int, default=1)
    parser.add_argument("--max-active-roots-per-node", type=int, default=2)
    parser.add_argument("--shared-progression-max-gap", type=float, default=0.20)
    parser.add_argument("--shared-cmi-trust-threshold", type=float, default=0.50)
    parser.add_argument("--relation-contact-threshold", type=float, default=0.25)
    parser.add_argument("--relation-active-overlap-threshold", type=float, default=0.01)
    parser.add_argument("--relation-progression-abs-threshold", type=float, default=0.30)
    parser.add_argument("--relation-cmi-trust-threshold", type=float, default=0.35)
    parser.add_argument("--min-relation-evidence-channels", type=int, default=1)
    parser.add_argument("--lambda-a", type=float, default=0.30)
    parser.add_argument("--lambda-p", type=float, default=0.20)
    parser.add_argument("--lambda-d", type=float, default=0.10)
    parser.add_argument("--graph-mode", choices=("product", "step6"), default="product")
    parser.add_argument("--lambda-s", type=float, default=1.0)
    parser.add_argument("--lambda-x", type=float, default=1.0)
    parser.add_argument("--sample-key")
    parser.add_argument("--time-key")
    parser.add_argument("--truth-pseudotime-csv")
    parser.add_argument("--truth-dir")
    parser.add_argument("--benchmark-mode", action="store_true")
    parser.add_argument("--allow-oracle-construction", action="store_true")
    parser.add_argument(
        "--progression-source",
        choices=("auto", "truth", "timepoint", "membership", "root_signature", "graph_distance"),
        default="auto",
    )
    parser.add_argument("--lambda-prog", type=float, default=0.20)
    parser.add_argument("--theta-progression-jump", type=float)
    parser.add_argument("--theta-membership-jump", type=float, default=0.25)
    parser.add_argument("--w-jump", type=float, default=1.0)
    parser.add_argument("--w-smooth", type=float, default=0.5)
    parser.add_argument("--w-membership", type=float, default=0.5)
    parser.add_argument("--interaction-refinement-mode", choices=("conservative", "aggressive"), default="conservative")
    parser.add_argument("--lambda-inh", type=float, default=0.50)
    parser.add_argument("--lambda-prom", type=float, default=0.30)
    parser.add_argument("--lambda-conflict", type=float, default=0.70)
    parser.add_argument("--lambda-merge", type=float, default=0.30)
    parser.add_argument("--lambda-boundary", type=float, default=0.0)
    parser.add_argument("--lambda-dir-guard", type=float, default=0.15)
    parser.add_argument("--lambda-stability", type=float, default=0.10)
    parser.add_argument("--theta-promotion-high", type=float, default=0.15)
    parser.add_argument("--theta-inhibition-high", type=float, default=0.20)
    parser.add_argument("--theta-conflict-high", type=float, default=0.20)
    parser.add_argument("--theta-merge-high", type=float, default=0.15)
    parser.add_argument(
        "--selected-pseudotime-method",
        choices=("auto", "tree_distance", "membership", "progression", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--hybrid-alpha", type=float, default=0.50)
    parser.add_argument("--min-branch-nodes", type=int, default=1)
    parser.add_argument("--min-branch-pseudotime-span", type=float, default=0.0)
    parser.add_argument("--core-edge-confidence-quantile", type=float, default=0.60)
    parser.add_argument("--core-edge-confidence-floor", type=float, default=0.45)
    parser.add_argument("--core-min-nodes", type=int, default=3)
    # v7.4.2 Truth-First evidence gates
    parser.add_argument("--truth-min-root-quality", type=float, default=0.50)
    parser.add_argument("--truth-root-max-projection-nn-factor", type=float, default=6.0)
    parser.add_argument("--truth-min-trajectory-edges", type=int, default=3)
    parser.add_argument("--truth-min-trajectory-nodes", type=int, default=4)
    parser.add_argument("--truth-rescue-min-confidence", type=float, default=0.55)
    parser.add_argument("--truth-rescue-min-stability", type=float, default=0.80)
    parser.add_argument("--truth-min-edge-stability", type=float, default=0.60)
    parser.add_argument("--truth-min-s-global-floor", type=float, default=0.03)
    parser.add_argument("--truth-s-global-quantile-floor", type=float, default=0.05)
    parser.add_argument("--truth-min-edges-for-adaptive-threshold", type=int, default=5)
    parser.add_argument("--truth-max-cost-median-factor", type=float, default=8.0)
    parser.add_argument("--truth-max-cmi-conflict", type=float, default=0.25)
    parser.add_argument("--truth-program-min-genes", type=int, default=2)
    parser.add_argument("--truth-min-program-nodes", type=int, default=4)
    parser.add_argument("--truth-min-program-spearman", type=float, default=0.40)
    parser.add_argument("--truth-max-program-q", type=float, default=0.10)
    parser.add_argument("--truth-interface-min-mass", type=float, default=0.10)
    parser.add_argument("--truth-min-interface-support-rate", type=float, default=0.10)
    parser.add_argument("--truth-distinct-program-distance", type=float, default=0.25)
    parser.add_argument("--truth-distinct-max-program-diff", type=float, default=0.50)
    parser.add_argument("--truth-external-program-evidence-csv", default="")
    group_ext = parser.add_mutually_exclusive_group()
    group_ext.add_argument("--truth-require-external-process-evidence-for-level2", dest="truth_require_external_process_evidence_for_level2", action="store_true")
    group_ext.add_argument("--truth-allow-internal-program-evidence-for-level2", dest="truth_require_external_process_evidence_for_level2", action="store_false")
    parser.set_defaults(truth_require_external_process_evidence_for_level2=True)
    parser.add_argument("--segment-split-min-nodes", type=int, default=4)
    parser.add_argument("--segment-split-pseudotime-span", type=float, default=0.25)
    parser.add_argument("--constrained-mu", type=float, default=0.60)
    parser.add_argument("--constrained-eta", type=float, default=0.30)
    parser.add_argument("--constrained-laplacian", type=float, default=0.35)
    parser.add_argument("--constrained-root-anchor", type=float, default=25.0)
    parser.add_argument("--constrained-monotone-eps", type=float, default=1e-4)
    parser.add_argument("--lambda-cmi-e", type=float, default=0.12, help="Legacy v5 naive CMI-E attraction weight; retained for compatibility but not used by v6 gated cost")
    parser.add_argument("--lambda-cmi-e-drop", type=float, default=0.08, help="Penalty weight for raw-to-environment CMI drop, interpreted as microenvironment confounding")
    parser.add_argument("--lambda-cmi-e-trust", type=float, default=0.02, help="Reward weight for environment-stable CMI trust")
    parser.add_argument("--cmi-e-n-bins", type=int, default=3)
    parser.add_argument("--cmi-e-env-method", choices=("pca", "mean"), default="pca")
    parser.add_argument("--cmi-e-top-genes", type=int, default=0, help="0 uses all available genes")
    parser.add_argument("--cmi-e-min-valid-edges", type=int, default=0)
    _bool_switch(parser, "cmi-e", True, "Use environment-adjusted CMI edge support in root-specific tree cost")
    _bool_switch(parser, "cmi-e-use-signature-env", True, "Construct E from TME signatures when genes are available; otherwise use feature PC fallback")
    _bool_switch(parser, "segment-split", True, "Split long high-span branch segments in method-level export")
    _bool_switch(parser, "constrained-pseudotime", True, "Use root-anchored graph-regularized pseudotime in final method export")
    _bool_switch(parser, "backbone-pruning", True, "Enable confidence-core backbone extraction")
    _bool_switch(parser, "confidence-core-backbone", True, "Use adaptive high-confidence root-connected core instead of full-MST coverage")
    _bool_switch(parser, "collapse-degree2", True, "Record degree-2 chain collapse statistics")
    _bool_switch(parser, "score-backbone-not-full", True, "Export backbone tree under legacy scoring filenames")
    _bool_switch(parser, "progression-aware-tree", True, "Enable progression-aware cost")
    _bool_switch(parser, "interaction-v2", True, "Enable interaction-v2 refinement")
    _bool_switch(parser, "acceptance-rule", True, "Enable final-tree acceptance")
    args = parser.parse_args()

    cfg = ProgressionConfig(
        k_neighbors=args.k_neighbors,
        n_pca=args.n_pca,
        theta_A=args.theta_a,
        beta_transition=args.beta_transition,
        theta_secondary=args.theta_secondary,
        theta_entropy=args.theta_entropy,
        annotation_key=args.annotation_key,
        input_level=args.input_level,
        metacell_key=args.metacell_key,
        metacell_target_size=args.metacell_target_size,
        metacell_min_size=args.metacell_min_size,
        metacell_max_size=args.metacell_max_size,
        metacell_birch_threshold_multiplier=args.metacell_birch_threshold_multiplier,
        metacell_spatial_weight=args.metacell_spatial_weight,
        metacell_tiny_merge_size=args.metacell_tiny_merge_size,
        tiny_merge_min_comp_similarity=args.tiny_merge_min_comp_similarity,
        tiny_merge_max_spatial_nn_factor=args.tiny_merge_max_spatial_nn_factor,
        root_projection_top_k=args.root_projection_top_k,
        root_projection_type_weight=args.root_projection_type_weight,
        root_node_compatibility_threshold=args.root_node_compatibility_threshold,
        root_competition_jaccard_threshold=args.root_competition_jaccard_threshold,
        min_assigned_fraction=args.min_assigned_fraction,
        composition_similarity_allowed=args.composition_similarity_allowed,
        mixed_bridge_min_mass=args.mixed_bridge_min_mass,
        pure_compartment_threshold=args.pure_compartment_threshold,
        process_active_fraction=args.process_active_fraction,
        process_bridge_min_composition_similarity=args.process_bridge_min_composition_similarity,
        process_bridge_min_shared_mass=args.process_bridge_min_shared_mass,
        root_anchor_max_nodes=args.root_anchor_max_nodes,
        root_anchor_max_hops=args.root_anchor_max_hops,
        root_anchor_min_process_score=args.root_anchor_min_process_score,
        root_local_affinity_threshold=args.root_local_affinity_threshold,
        root_local_affinity_power=args.root_local_affinity_power,
        root_local_min_nodes=args.root_local_min_nodes,
        allow_conditional_trajectory_edges=args.allow_conditional_trajectory_edges,
        markov_sharpen_q=args.markov_sharpen_q,
        shared_ratio_threshold=args.shared_ratio_threshold,
        shared_margin_threshold=args.shared_margin_threshold,
        min_shared_evidence_channels=args.min_shared_evidence_channels,
        max_active_roots_per_node=args.max_active_roots_per_node,
        shared_progression_max_gap=args.shared_progression_max_gap,
        shared_cmi_trust_threshold=args.shared_cmi_trust_threshold,
        relation_contact_threshold=args.relation_contact_threshold,
        relation_active_overlap_threshold=args.relation_active_overlap_threshold,
        relation_progression_abs_threshold=args.relation_progression_abs_threshold,
        relation_cmi_trust_threshold=args.relation_cmi_trust_threshold,
        min_relation_evidence_channels=args.min_relation_evidence_channels,
        lambda_A=args.lambda_a,
        lambda_P=args.lambda_p,
        lambda_D=args.lambda_d,
        disable_interaction_refinement=True,
        enable_progression_aware_tree=args.progression_aware_tree,
        progression_source=args.progression_source,
        lambda_prog=args.lambda_prog,
        theta_progression_jump=args.theta_progression_jump,
        theta_membership_jump=args.theta_membership_jump,
        w_jump=args.w_jump,
        w_smooth=args.w_smooth,
        w_membership=args.w_membership,
        enable_interaction_v2=args.interaction_v2,
        interaction_refinement_mode=args.interaction_refinement_mode,
        lambda_inh=args.lambda_inh,
        lambda_prom=args.lambda_prom,
        lambda_conflict=args.lambda_conflict,
        lambda_merge=args.lambda_merge,
        lambda_boundary=args.lambda_boundary,
        lambda_dir_guard=args.lambda_dir_guard,
        lambda_stability=args.lambda_stability,
        theta_promotion_high=args.theta_promotion_high,
        theta_inhibition_high=args.theta_inhibition_high,
        theta_conflict_high=args.theta_conflict_high,
        theta_merge_high=args.theta_merge_high,
        enable_acceptance_rule=args.acceptance_rule,
        selected_pseudotime_method=args.selected_pseudotime_method,
        hybrid_alpha=args.hybrid_alpha,
        allow_oracle_construction=args.allow_oracle_construction,
        enable_backbone_pruning=args.backbone_pruning,
        enable_confidence_core_backbone=args.confidence_core_backbone,
        core_edge_confidence_quantile=args.core_edge_confidence_quantile,
        core_edge_confidence_floor=args.core_edge_confidence_floor,
        core_min_nodes=args.core_min_nodes,
        truth_min_root_quality=args.truth_min_root_quality,
        truth_root_max_projection_nn_factor=args.truth_root_max_projection_nn_factor,
        truth_min_trajectory_edges=args.truth_min_trajectory_edges,
        truth_min_trajectory_nodes=args.truth_min_trajectory_nodes,
        truth_rescue_min_confidence=args.truth_rescue_min_confidence,
        truth_rescue_min_stability=args.truth_rescue_min_stability,
        truth_min_edge_stability=args.truth_min_edge_stability,
        truth_min_s_global_floor=args.truth_min_s_global_floor,
        truth_s_global_quantile_floor=args.truth_s_global_quantile_floor,
        truth_min_edges_for_adaptive_threshold=args.truth_min_edges_for_adaptive_threshold,
        truth_max_cost_median_factor=args.truth_max_cost_median_factor,
        truth_max_cmi_conflict=args.truth_max_cmi_conflict,
        truth_program_min_genes=args.truth_program_min_genes,
        truth_min_program_nodes=args.truth_min_program_nodes,
        truth_min_program_spearman=args.truth_min_program_spearman,
        truth_max_program_q=args.truth_max_program_q,
        truth_interface_min_mass=args.truth_interface_min_mass,
        truth_min_interface_support_rate=args.truth_min_interface_support_rate,
        truth_distinct_program_distance=args.truth_distinct_program_distance,
        truth_distinct_max_program_diff=args.truth_distinct_max_program_diff,
        truth_require_external_process_evidence_for_level2=args.truth_require_external_process_evidence_for_level2,
        truth_external_program_evidence_csv=args.truth_external_program_evidence_csv,
        min_branch_nodes=args.min_branch_nodes,
        min_branch_pseudotime_span=args.min_branch_pseudotime_span,
        collapse_degree2=args.collapse_degree2,
        score_backbone_not_full=args.score_backbone_not_full,
        enable_segment_split=args.segment_split,
        segment_split_min_nodes=args.segment_split_min_nodes,
        segment_split_pseudotime_span=args.segment_split_pseudotime_span,
        enable_constrained_pseudotime=args.constrained_pseudotime,
        constrained_mu=args.constrained_mu,
        constrained_eta=args.constrained_eta,
        constrained_laplacian=args.constrained_laplacian,
        constrained_root_anchor=args.constrained_root_anchor,
        constrained_monotone_eps=args.constrained_monotone_eps,
        enable_cmi_e=args.cmi_e,
        lambda_cmi_e=args.lambda_cmi_e,
        lambda_cmi_e_drop=args.lambda_cmi_e_drop,
        lambda_cmi_e_trust=args.lambda_cmi_e_trust,
        cmi_e_n_bins=args.cmi_e_n_bins,
        cmi_e_env_method=args.cmi_e_env_method,
        cmi_e_top_genes=args.cmi_e_top_genes,
        cmi_e_min_valid_edges=args.cmi_e_min_valid_edges,
        cmi_e_use_signature_env=args.cmi_e_use_signature_env,
    )
    roots_df = load_root_table_any_format(args.roots_csv)
    outdir = legacy.ensure_dir(Path(args.outdir))
    summaries, failures = [], []
    for requested_name, adata_path in _paths_from_args(args):
        dataset = args.dataset_name or requested_name or legacy.dataset_name_from_path(adata_path)
        try:
            legacy.run_one_dataset(
                adata_path=adata_path,
                roots_df=roots_df,
                base_outdir=outdir,
                cfg=cfg,
                dataset_name=dataset,
                dataset_col=args.dataset_col,
                root_col=args.root_col,
                root_specific_nmf_dir=Path(args.root_specific_nmf_dir) if args.root_specific_nmf_dir else None,
                sample_key=args.sample_key,
                graph_mode=args.graph_mode,
                lambda_s=args.lambda_s,
                lambda_x=args.lambda_x,
                skip_spatial_figures=True,
            )
            truth_path = Path(args.truth_pseudotime_csv) if args.truth_pseudotime_csv else None
            if truth_path is None and args.truth_dir:
                truth_dir = Path(args.truth_dir)
                candidates = [
                    truth_dir / f"{dataset}_truth_pseudotime.csv",
                    truth_dir / f"{dataset.removesuffix('_metacell')}_truth_pseudotime.csv",
                ]
                truth_path = next((path for path in candidates if path.exists()), None)
            summary = upgrade_dataset(
                adata_path,
                dataset,
                outdir,
                cfg,
                truth_path,
                args.time_key,
                args.sample_key,
                args.benchmark_mode,
            )
            summaries.append(summary)
            print(f"[DONE] {dataset}: selected={summary['selected_tree_counts']}")
        except Exception as exc:
            failures.append(
                {
                    "dataset": dataset,
                    "adata_path": str(adata_path),
                    "error_message": str(exc),
                    "traceback_short": traceback.format_exc(limit=8),
                }
            )
            print(f"[FAIL] {dataset}: {exc}")
    if summaries:
        pd.DataFrame(
            [
                {
                    "dataset": item["dataset"],
                    "n_obs": item["n_obs"],
                    "n_roots": item["n_roots"],
                    "selected_tree_counts": json.dumps(item["selected_tree_counts"], ensure_ascii=False),
                }
                for item in summaries
            ]
        ).to_csv(outdir / "run_summary_table.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(outdir / "run_failures.csv", index=False)
    elif (outdir / "run_failures.csv").exists():
        (outdir / "run_failures.csv").unlink()
    return 0 if summaries and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
