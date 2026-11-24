#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program-aware Soft Basin-wise Multi-root Trajectory Graph pipeline.

Main method:
1. Candidate graph construction.
2. Absorbing Markov soft root membership.
3. Soft basin-wise weighted local tree construction.
4. Parent-rule/root-distance direction refinement.
5. Shared / convergence / collision handling.
6. Tree-level interaction summary and before/after figures.

Notes:
- Standard MSF is not used as the main method.
- CMI/LR are optional evidence; if absent, their factors are neutral.
- TODO: replace embedding fallback S_rootprog with root-specific NMF program similarity.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, identity
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import cascat_markov_programmes as v71bio
import cascat_metacell_process as v72meta

try:
    import anndata as ad
except Exception:  # pragma: no cover
    ad = None


DEFAULT_DATASETS = {
    "GSE": "data/原始数据/GSE.h5ad",
    "GSE193460": "data/原始数据/GSE193460.h5ad",
    "GSE245582": "data/原始数据/GSE245582.h5ad",
    "xenium_breast": "data/原始数据/xenium_breast.h5ad",
    "xenium_lung": "data/原始数据/xenium_lung.h5ad",
    "xenium_melanoma": "data/原始数据/xenium_melanoma.h5ad",
    "xenium_pancreatic": "data/原始数据/xenium_pancreatic.h5ad",
}


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
class Config:
    # Final sensitivity analysis selected a sparse metacell-level graph.
    k_neighbors: int = 6
    n_pca: int = 30
    # Final shared/root-basin threshold retained as a stable platform value.
    theta_A: float = 0.30
    theta_entropy: float = 0.70
    theta_margin: float = 0.20
    theta_secondary: float = 0.25
    theta_low_membership: float = 0.10
    theta_conv: float = 0.08
    theta_conflict: float = 0.08
    theta_collision: float = 0.20
    theta_promotion: float = 0.05
    theta_inhibition: float = 0.20
    beta_transition: float = 4.0
    eps: float = 1e-9
    alpha_spatial: float = 0.35
    beta_expr: float = 0.35
    gamma_program: float = 0.20
    delta_cmi: float = 0.10
    rho_lr: float = 0.00
    lambda_A: float = 0.30
    lambda_P: float = 0.20
    lambda_D: float = 0.10
    # Interaction-aware refinement weights. These are applied after the initial
    # root-specific MSTs are built and tree-tree interaction scores are computed.
    lambda_inh: float = 0.50
    lambda_prom: float = 0.30
    lambda_conflict: float = 0.70
    lambda_merge: float = 0.30
    lambda_boundary: float = 0.00
    min_edge_cost: float = 1e-6
    # v7.3 structural controls.  Markov probabilities are preserved, but actual
    # multi-tree inclusion is sparse and evidence-gated.
    enable_biological_gate: bool = True
    annotation_key: Optional[str] = None
    allow_conditional_trajectory_edges: bool = False
    markov_sharpen_q: float = 1.5
    shared_ratio_threshold: float = 0.65
    shared_margin_threshold: float = 0.20
    min_shared_evidence_channels: int = 1
    max_active_roots_per_node: int = 2
    shared_progression_max_gap: float = 0.20
    shared_cmi_trust_threshold: float = 0.50
    relation_contact_threshold: float = 0.25
    relation_active_overlap_threshold: float = 0.01
    relation_progression_abs_threshold: float = 0.30
    relation_cmi_trust_threshold: float = 0.35
    min_relation_evidence_channels: int = 1
    keep_only_root_component: bool = True
    min_trajectory_edge_confidence: float = 0.0
    # Legacy base-pipeline interaction refinement is disabled by default in v7.3;
    # the progression-refined script performs compatibility-preserving refinement.
    disable_interaction_refinement: bool = True
    # v7.3 input/metacell contract. Existing root-compatible cluster columns are
    # preferred; otherwise TME-aware metacells are constructed adaptively.
    input_level: str = "auto"
    metacell_key: Optional[str] = None
    metacell_target_size: int = 50
    metacell_min_size: int = 10
    metacell_max_size: int = 150
    metacell_birch_threshold_multiplier: float = 1.5
    metacell_spatial_weight: float = 0.35
    # v7.3 tiny-metacell post-merge: reduce singleton/doubleton noise without fixing K.
    metacell_tiny_merge_size: int = 5
    tiny_merge_min_comp_similarity: float = 0.55
    tiny_merge_max_spatial_nn_factor: float = 4.0
    root_projection_top_k: int = 25
    root_projection_type_weight: float = 0.35
    root_node_compatibility_threshold: float = 0.20
    root_competition_jaccard_threshold: float = 0.50
    min_assigned_fraction: float = 0.05
    composition_similarity_allowed: float = 0.55
    mixed_bridge_min_mass: float = 0.20
    pure_compartment_threshold: float = 0.80
    # v7.3 real process-aware bridge eligibility.
    process_active_fraction: float = 0.10
    process_bridge_min_composition_similarity: float = 0.35
    process_bridge_min_shared_mass: float = 0.10
    # v7.3 local super-root anchor sets.
    root_anchor_max_nodes: int = 7
    root_anchor_max_hops: int = 3
    root_anchor_min_process_score: float = 0.25
    root_anchor_w_process: float = 0.40
    root_anchor_w_expr: float = 0.25
    root_anchor_w_spatial: float = 0.20
    root_anchor_w_graph: float = 0.15
    # v7.3 local process competition: same-process roots do not compete globally.
    root_local_affinity_threshold: float = 0.42
    root_local_affinity_power: float = 1.5
    root_local_min_nodes: int = 5
    root_affinity_w_process: float = 0.35
    root_affinity_w_expr: float = 0.25
    root_affinity_w_spatial: float = 0.20
    root_affinity_w_graph: float = 0.20
    random_state: int = 7


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_h5ad(path: Path):
    if ad is None:
        raise ImportError("anndata is required to read .h5ad files")
    return ad.read_h5ad(str(path))


def infer_col(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def dataset_name_from_path(path: Path, override: Optional[str] = None) -> str:
    if override:
        return override
    stem = path.stem
    aliases = {"GSE193460(1)": "GSE193460"}
    return aliases.get(stem, stem)


def to_dense(X) -> np.ndarray:
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def get_spatial_coords(adata) -> np.ndarray:
    if "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"])
        if xy.ndim == 2 and xy.shape[1] >= 2:
            return xy[:, :2].astype(float)
    for a, b in [("x", "y"), ("X", "Y"), ("spatial_x", "spatial_y"), ("array_row", "array_col")]:
        if a in adata.obs.columns and b in adata.obs.columns:
            return adata.obs[[a, b]].to_numpy(dtype=float)
    if "X_pca" in adata.obsm:
        return np.asarray(adata.obsm["X_pca"][:, :2], dtype=float)
    raise ValueError("No spatial coordinates found: need obsm['spatial'] or obs x/y-like columns")


def get_expression_embedding(adata, cfg: Config) -> np.ndarray:
    if "X_pca" in adata.obsm:
        Xp = np.asarray(adata.obsm["X_pca"])
        return Xp[:, : min(cfg.n_pca, Xp.shape[1])].astype(float)
    X = to_dense(adata.X).astype(float)
    n_comp = min(cfg.n_pca, X.shape[0] - 1, X.shape[1] - 1)
    if n_comp < 2:
        return StandardScaler().fit_transform(X)
    Xz = StandardScaler(with_mean=not sparse.issparse(adata.X)).fit_transform(X)
    return PCA(n_components=n_comp, random_state=cfg.random_state).fit_transform(Xz)


def get_domain_labels(adata) -> Optional[np.ndarray]:
    for key in ["domain", "spatial_domain", "eco_niche", "region", "banksy_domain", "leiden_domain"]:
        if key in adata.obs.columns:
            return adata.obs[key].astype(str).to_numpy()
    return None


def get_sample_labels(adata, sample_key: Optional[str]) -> Optional[np.ndarray]:
    """Per-node sample/slice label, used to avoid mixing samples spatially.

    Returns None when no sample key is requested or the column is absent, in
    which case the graph is built exactly as before (combined spatial+expr).
    """
    if not sample_key:
        return None
    if sample_key in adata.obs.columns:
        return adata.obs[sample_key].astype(str).to_numpy()
    return None


def choose_root_col(roots_df: pd.DataFrame, root_col: Optional[str]) -> str:
    if root_col and root_col in roots_df.columns:
        return root_col
    inferred = infer_col(
        roots_df.columns,
        ["root_id", "root", "root_node", "mapped_root_nodes", "node", "node_id", "cell_id", "obs_name", "cluster_id"],
    )
    if inferred is None:
        raise ValueError(f"Cannot infer root column from columns={list(roots_df.columns)}")
    return inferred


def choose_dataset_col(roots_df: pd.DataFrame, dataset_col: Optional[str]) -> Optional[str]:
    if dataset_col and dataset_col in roots_df.columns:
        return dataset_col
    return infer_col(roots_df.columns, ["dataset", "dataset_name", "sample", "adata"])


def split_root_values(value: object) -> list[str]:
    text = str(value)
    for sep in ["|", ";", ","]:
        if sep in text:
            return [v.strip() for v in text.split(sep) if v.strip()]
    return [text.strip()]


def nearest_by_xy(coords: np.ndarray, x: float, y: float) -> int:
    target = np.asarray([x, y], dtype=float)
    return int(np.argmin(np.linalg.norm(coords - target[None, :], axis=1)))


def resolve_one_root(
    adata,
    value: str,
    row: pd.Series,
    root_col: str,
    coords: np.ndarray,
) -> tuple[int, str]:
    obs_names = pd.Index(adata.obs_names.astype(str))
    if value in obs_names:
        return int(obs_names.get_loc(value)), "obs_name"

    try:
        iv = int(float(value))
        if 0 <= iv < adata.n_obs:
            return iv, "row_index"
    except Exception:
        pass

    cluster_cols = [
        c
        for c in [
            "cluster",
            "clusters",
            "leiden",
            "louvain",
            "state",
            "global_state",
            "cell_type",
            "celltype",
            "annotation",
        ]
        if c in adata.obs.columns
    ]
    for ccol in cluster_cols:
        hit = np.flatnonzero(adata.obs[ccol].astype(str).to_numpy() == value)
        if len(hit):
            center = coords[hit].mean(axis=0)
            idx = int(hit[np.argmin(np.linalg.norm(coords[hit] - center[None, :], axis=1))])
            return idx, f"cluster:{ccol}"

    if "x" in row.index and "y" in row.index and pd.notna(row["x"]) and pd.notna(row["y"]):
        return nearest_by_xy(coords, float(row["x"]), float(row["y"])), "xy"

    raise ValueError(f"Cannot resolve root {value!r} from column {root_col!r}")


def resolve_roots(
    adata,
    roots_df: pd.DataFrame,
    dataset_name: str,
    dataset_col: Optional[str],
    root_col: Optional[str],
) -> tuple[list[int], pd.DataFrame]:
    dcol = choose_dataset_col(roots_df, dataset_col)
    df = roots_df.copy()
    if dcol:
        df = df[df[dcol].astype(str) == str(dataset_name)].copy()
    if df.empty:
        raise ValueError(f"No roots found for dataset={dataset_name}")

    rcol = choose_root_col(df, root_col)
    coords = get_spatial_coords(adata)
    roots: list[int] = []
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        for raw in split_root_values(row[rcol]):
            idx, mode = resolve_one_root(adata, raw, row, rcol, coords)
            if idx not in roots:
                roots.append(idx)
            rec = row.to_dict()
            rec["root_col_used"] = rcol
            rec["original_root_value"] = raw
            rec["resolved_index"] = idx
            rec["resolved_obs_name"] = str(adata.obs_names[idx])
            rec["resolution_mode"] = mode
            rec["resolved_x"] = float(coords[idx, 0])
            rec["resolved_y"] = float(coords[idx, 1])
            rows.append(rec)
    if not roots:
        raise ValueError(f"No roots resolved for dataset={dataset_name}")
    return roots, pd.DataFrame(rows)


def subset_adata_keep_roots(adata, roots: list[int], cfg: Config) -> tuple[object, list[int], pd.DataFrame]:
    """Retain every input node.

    Earlier versions randomly capped the analysis at ``max_cells``.  That made
    valid states disappear solely because a dataset was large; the cap and its
    configuration field have been removed.
    """
    n = adata.n_obs
    rows = [
        {
            "original_index": i,
            "subset_index": i,
            "obs_name": str(adata.obs_names[i]),
            "kept_reason": "all_unbounded",
        }
        for i in range(n)
    ]
    return adata, roots, pd.DataFrame(rows)


def build_candidate_graph(adata, cfg: Config, outdir: Path, sample_key: Optional[str] = None):
    coords = get_spatial_coords(adata)
    emb = get_expression_embedding(adata, cfg)
    domains = get_domain_labels(adata)
    samples = get_sample_labels(adata, sample_key)

    coords_z = StandardScaler().fit_transform(coords)
    emb_z = StandardScaler().fit_transform(emb)
    k = min(cfg.k_neighbors + 1, adata.n_obs)
    # When samples are present, their spatial frames overlap and are NOT
    # comparable across samples, so neighbours are selected in expression space
    # only (cross-sample links reflect expression/state similarity). Spatial
    # structure is reintroduced as a within-sample-only similarity factor below.
    feat = emb_z if samples is not None else np.hstack([coords_z, emb_z])
    nn = NearestNeighbors(n_neighbors=k).fit(feat)
    _, inds = nn.kneighbors(feat)

    spatial_d, expr_d = [], []
    for i in range(adata.n_obs):
        for j in inds[i, 1:]:
            j = int(j)
            # Estimate the spatial bandwidth from within-sample pairs only.
            if samples is None or samples[i] == samples[j]:
                spatial_d.append(np.linalg.norm(coords_z[i] - coords_z[j]))
            expr_d.append(np.linalg.norm(emb_z[i] - emb_z[j]))
    sig_s = max(float(np.median(spatial_d)) if spatial_d else 1.0, cfg.eps)
    sig_x = max(float(np.median(expr_d)) if expr_d else 1.0, cfg.eps)

    conn = adata.obsp["connectivities"].tocsr() if "connectivities" in adata.obsp else None
    records = []
    row_idx: list[int] = []
    col_idx: list[int] = []
    vals: list[float] = []
    seen: set[tuple[int, int]] = set()
    for i in range(adata.n_obs):
        for j0 in inds[i, 1:]:
            j = int(j0)
            a, b = sorted((i, j))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            ds = np.linalg.norm(coords_z[a] - coords_z[b])
            dx = np.linalg.norm(emb_z[a] - emb_z[b])
            # Cross-sample pairs have incomparable spatial coordinates: make the
            # spatial factor neutral (1.0) so only expression drives the edge.
            cross_sample = samples is not None and samples[a] != samples[b]
            S_sp = 1.0 if cross_sample else math.exp(-(ds * ds) / (sig_s * sig_s + cfg.eps))
            S_ex = math.exp(-(dx * dx) / (sig_x * sig_x + cfg.eps))
            S_prog = 1.0
            S_cmi = 1.0
            S_lr = 1.0
            S_conn = float(conn[a, b]) if conn is not None else 0.0
            S_global = (
                (S_sp + cfg.eps) ** cfg.alpha_spatial
                * (S_ex + cfg.eps) ** cfg.beta_expr
                * (S_prog + cfg.eps) ** cfg.gamma_program
                * (S_cmi + cfg.eps) ** cfg.delta_cmi
                * (S_lr + cfg.eps) ** cfg.rho_lr
            )
            if S_conn > 0:
                S_global = 0.5 * S_global + 0.5 * min(1.0, S_conn)
            d_global = 1.0 / (S_global + cfg.eps)
            row_idx.extend([a, b])
            col_idx.extend([b, a])
            vals.extend([S_global, S_global])
            records.append(
                {
                    "source": a,
                    "target": b,
                    "source_name": str(adata.obs_names[a]),
                    "target_name": str(adata.obs_names[b]),
                    "S_spatial": S_sp,
                    "S_expr": S_ex,
                    "S_program": S_prog,
                    "S_CMI": S_cmi,
                    "S_LR": S_lr,
                    "S_connectivities": S_conn,
                    "S_global": S_global,
                    "d_global": d_global,
                    "cross_sample": bool(cross_sample),
                }
            )
    S = csr_matrix((vals, (row_idx, col_idx)), shape=(adata.n_obs, adata.n_obs))
    edges_df = pd.DataFrame(records)
    edges_df.to_csv(outdir / "candidate_graph_edges.csv", index=False)
    return S, edges_df, coords, emb_z, domains


def build_candidate_graph_step6(
    adata,
    cfg: Config,
    outdir: Path,
    sample_key: Optional[str] = None,
    lambda_s: float = 1.0,
    lambda_x: float = 1.0,
):
    """Metacell-level graph using the explicit Step-6 distance.

        D_ab = lambda_s * D_spatial(c_a, c_b) + lambda_x * (1 - corr(x_a, x_b))

    Distances are recomputed between metacells (not inherited from cell edges),
    each median-normalised so the two terms are comparable, then a kNN graph is
    built on D_ab. Spatial is only used within a sample (cross-sample frames are
    incomparable); across samples the edge is expression-only.
    """
    from scipy.spatial.distance import cdist

    coords = get_spatial_coords(adata)
    emb = get_expression_embedding(adata, cfg)
    domains = get_domain_labels(adata)
    samples = get_sample_labels(adata, sample_key)
    n = adata.n_obs
    coords_z = StandardScaler().fit_transform(coords)
    emb_z = StandardScaler().fit_transform(emb)

    # Expression distance: 1 - Pearson correlation between metacell embeddings.
    C = np.nan_to_num(np.corrcoef(emb), nan=0.0, posinf=0.0, neginf=0.0)
    D_expr = 1.0 - C
    # Spatial distance: z-scored Euclidean.
    D_spat = cdist(coords_z, coords_z)

    off = ~np.eye(n, dtype=bool)
    med_s = float(np.median(D_spat[off & (D_spat > 0)])) if np.any(D_spat[off] > 0) else 1.0
    med_x = float(np.median(D_expr[off & (D_expr > 0)])) if np.any(D_expr[off] > 0) else 1.0
    D_spat_n = D_spat / (med_s + cfg.eps)
    D_expr_n = D_expr / (med_x + cfg.eps)

    if samples is not None:
        same = samples[:, None] == samples[None, :]
        spat_term = np.where(same, D_spat_n, 0.0)
    else:
        spat_term = D_spat_n

    D_ab = lambda_s * spat_term + lambda_x * D_expr_n
    np.fill_diagonal(D_ab, np.inf)

    k = min(cfg.k_neighbors, n - 1)
    knn_idx = np.argsort(D_ab, axis=1)[:, :k]

    records, row_idx, col_idx, vals = [], [], [], []
    seen: set[tuple[int, int]] = set()
    for i in range(n):
        for j0 in knn_idx[i]:
            j = int(j0)
            a, b = sorted((i, j))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            d = float(D_ab[a, b])
            S_global = math.exp(-d)
            cross_sample = samples is not None and samples[a] != samples[b]
            row_idx.extend([a, b])
            col_idx.extend([b, a])
            vals.extend([S_global, S_global])
            records.append(
                {
                    "source": a,
                    "target": b,
                    "source_name": str(adata.obs_names[a]),
                    "target_name": str(adata.obs_names[b]),
                    "S_spatial": math.exp(-float(spat_term[a, b])),
                    "S_expr": float(C[a, b]),
                    "S_program": 1.0,
                    "S_CMI": 1.0,
                    "S_LR": 1.0,
                    "S_connectivities": 0.0,
                    "S_global": S_global,
                    "d_global": d,
                    "D_spatial": float(D_spat_n[a, b]),
                    "D_expr": float(D_expr_n[a, b]),
                    "cross_sample": bool(cross_sample),
                }
            )
    S = csr_matrix((vals, (row_idx, col_idx)), shape=(n, n))
    edges_df = pd.DataFrame(records)
    edges_df.to_csv(outdir / "candidate_graph_edges.csv", index=False)
    return S, edges_df, coords, emb_z, domains


def shortest_path_softmax(S: csr_matrix, roots: list[int], cfg: Config) -> np.ndarray:
    D = S.copy()
    D.data = 1.0 / (D.data + cfg.eps)
    dist = np.asarray(shortest_path(D, directed=False, indices=roots)).T
    finite = np.isfinite(dist)
    fill = float(np.nanmax(dist[finite])) if finite.any() else 1e6
    dist[~finite] = fill
    logits = -cfg.beta_transition * dist
    logits -= logits.max(axis=1, keepdims=True)
    A = np.exp(logits)
    A /= A.sum(axis=1, keepdims=True) + cfg.eps
    for k, r in enumerate(roots):
        A[r, :] = 0.0
        A[r, k] = 1.0
    return A


def absorbing_markov_membership(S: csr_matrix, roots: list[int], cfg: Config) -> tuple[np.ndarray, str]:
    n, K = S.shape[0], len(roots)
    coo = S.tocoo()
    Praw = csr_matrix((np.exp(cfg.beta_transition * coo.data), (coo.row, coo.col)), shape=S.shape)
    row_sum = np.asarray(Praw.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    P = sparse.diags(1.0 / row_sum) @ Praw

    roots_arr = np.asarray(roots, dtype=int)
    is_root = np.zeros(n, dtype=bool)
    is_root[roots_arr] = True
    transient = np.flatnonzero(~is_root)
    A = np.zeros((n, K), dtype=float)
    for k, r in enumerate(roots):
        A[r, k] = 1.0
    if len(transient) == 0:
        return A, "absorbing_markov"

    PTT = P[transient][:, transient].tocsr()
    PTR = P[transient][:, roots_arr].toarray()
    M = identity(len(transient), format="csr") - PTT
    try:
        B = np.asarray(spsolve(M, PTR))
        if B.ndim == 1:
            B = B[:, None]
        B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
        B[B < 0] = 0.0
        B /= B.sum(axis=1, keepdims=True) + cfg.eps
        A[transient] = B
        return A, "absorbing_markov"
    except Exception:
        return shortest_path_softmax(S, roots, cfg), "shortest_path_softmax_solver_fallback"


def normalized_entropy(A: np.ndarray, eps: float) -> np.ndarray:
    if A.shape[1] <= 1:
        return np.zeros(A.shape[0])
    H = -np.sum(A * np.log(A + eps), axis=1)
    return H / math.log(A.shape[1])


def root_signature_compatibility(emb: np.ndarray, membership: np.ndarray, cfg: Config) -> np.ndarray:
    sig = np.average(emb, axis=0, weights=membership + cfg.eps)
    dist = np.linalg.norm(emb - sig[None, :], axis=1)
    scale = max(float(np.median(dist[dist > 0])) if np.any(dist > 0) else 1.0, cfg.eps)
    return np.exp(-(dist**2) / (scale * scale + cfg.eps))


def load_root_specific_nmf_edges(path: Optional[Path]) -> dict[tuple[int, int, int], float]:
    if path is None:
        return {}
    edge_path = path / "root_specific_graph_edges.csv"
    if not edge_path.exists():
        return {}
    df = pd.read_csv(edge_path)
    required = {"root_pos", "source", "target", "S_rootprog_nmf"}
    if not required.issubset(df.columns):
        return {}
    lookup: dict[tuple[int, int, int], float] = {}
    for row in df.itertuples(index=False):
        root_pos = int(row.root_pos)
        i = int(row.source)
        j = int(row.target)
        val = float(row.S_rootprog_nmf)
        lookup[(root_pos, i, j)] = val
        lookup[(root_pos, j, i)] = val
    return lookup


def build_one_tree(
    adata,
    edges_df: pd.DataFrame,
    A: np.ndarray,
    root_idx: int,
    root_pos: int,
    emb: np.ndarray,
    domains,
    cfg: Config,
    nmf_edge_lookup: Optional[dict[tuple[int, int, int], float]] = None,
    interaction_modifiers: Optional[dict[str, np.ndarray]] = None,
    basin_override: Optional[set[int]] = None,
    active_assignment: Optional[np.ndarray] = None,
    compatibility_mask: Optional[np.ndarray] = None,
):
    """Build one root-specific v7.3 trajectory tree.

    The basin is primary-only by default and may include a validated secondary
    assignment through ``active_assignment``.  Only trajectory-eligible edges are
    expected in ``edges_df``.  Disconnected non-root components are not repaired
    with weak bridges and are excluded from this root-specific backbone.
    """
    membership = A[:, root_pos]
    if basin_override is not None:
        basin_arr = np.asarray(sorted(set(map(int, basin_override)) | {int(root_idx)}), dtype=int)
    elif active_assignment is not None:
        basin_arr = v71bio.basin_nodes_from_active(
            active_assignment, root_pos, root_idx, compatibility_mask
        )
    else:
        primary = v71bio.safe_primary_roots(A, cfg.eps)
        keep = primary == root_pos
        basin_arr = np.asarray(sorted(set(map(int, np.flatnonzero(keep))) | {int(root_idx)}), dtype=int)

    basin_set = set(map(int, basin_arr))
    node_prog = root_signature_compatibility(emb, membership, cfg)

    G = nx.Graph()
    G.add_nodes_from(map(int, basin_arr))
    for row in edges_df.itertuples(index=False):
        i, j = int(row.source), int(row.target)
        if i not in basin_set or j not in basin_set:
            continue
        if hasattr(row, "trajectory_edge_eligible") and not bool(row.trajectory_edge_eligible):
            continue
        if hasattr(row, "trajectory_compatibility") and float(row.trajectory_compatibility) <= 0.0:
            continue
        if nmf_edge_lookup and (root_pos, i, j) in nmf_edge_lookup:
            S_rootprog = float(nmf_edge_lookup[(root_pos, i, j)])
            rootprog_source = "root_specific_NMF"
        else:
            S_rootprog = math.sqrt(max(float(node_prog[i]), cfg.eps) * max(float(node_prog[j]), cfg.eps))
            rootprog_source = "embedding_signature_fallback"
        domain_penalty = 1.0 if domains is not None and domains[i] != domains[j] else 0.0
        bio_penalty = 0.0
        if hasattr(row, "trajectory_compatibility") and float(row.trajectory_compatibility) < 1.0:
            # Conditional edges are permitted only when explicitly enabled upstream.
            bio_penalty = 0.75 * (1.0 - float(row.trajectory_compatibility))
        base_cost = (
            float(row.d_global)
            - cfg.lambda_A * math.log(float(membership[i] * membership[j]) + cfg.eps)
            - cfg.lambda_P * math.log(S_rootprog + cfg.eps)
            + cfg.lambda_D * domain_penalty
            + bio_penalty
        )

        P_ij = H_ij = F_ij = M_ij = B_ij = 0.0
        if interaction_modifiers is not None:
            P = interaction_modifiers.get("P")
            Hm = interaction_modifiers.get("H")
            F = interaction_modifiers.get("F")
            M = interaction_modifiers.get("M")
            B = interaction_modifiers.get("B")
            if P is not None:
                P_ij = 0.5 * (float(P[i, root_pos]) + float(P[j, root_pos]))
            if Hm is not None:
                H_ij = 0.5 * (float(Hm[i, root_pos]) + float(Hm[j, root_pos]))
            if F is not None:
                F_ij = 0.5 * (float(F[i, root_pos]) + float(F[j, root_pos]))
            if M is not None:
                M_ij = 0.5 * (float(M[i, root_pos]) + float(M[j, root_pos]))
            if B is not None:
                B_ij = 0.5 * (float(B[i, root_pos]) + float(B[j, root_pos]))

        modifier_total = (
            cfg.lambda_inh * H_ij
            - cfg.lambda_prom * P_ij
            + cfg.lambda_conflict * F_ij
            - cfg.lambda_merge * M_ij
            + cfg.lambda_boundary * B_ij
        )
        cost = max(cfg.min_edge_cost, base_cost + modifier_total)
        G.add_edge(
            i,
            j,
            weight=cost,
            base_cost=float(base_cost),
            biological_penalty=float(bio_penalty),
            modifier_total=float(modifier_total),
            P_ij=float(P_ij),
            H_ij=float(H_ij),
            F_ij=float(F_ij),
            M_ij=float(M_ij),
            B_ij=float(B_ij),
            S_global=float(row.S_global),
            S_rootprog=S_rootprog,
            rootprog_source=rootprog_source,
            A_source=float(membership[i]),
            A_target=float(membership[j]),
            trajectory_compatibility=float(getattr(row, "trajectory_compatibility", 1.0)),
        )

    disconnected_audit = {
        "n_components_before": 0,
        "dropped_nonroot_components": 0,
        "dropped_nonroot_nodes": 0,
    }
    if bool(getattr(cfg, "keep_only_root_component", True)):
        G, disconnected_audit = v71bio.keep_root_component_only(G, root_idx)

    if G.number_of_edges() > 0:
        tree = nx.minimum_spanning_tree(G, weight="weight")
    else:
        tree = G.copy()

    dist = nx.single_source_dijkstra_path_length(tree, root_idx, weight="weight") if root_idx in tree else {}
    rows = []
    for u, v, data in tree.edges(data=True):
        du = dist.get(u, math.inf)
        dv = dist.get(v, math.inf)
        if du <= dv:
            src, tgt, ps, pt = u, v, du, dv
        else:
            src, tgt, ps, pt = v, u, dv, du
        rows.append(
            {
                "root_id": str(adata.obs_names[root_idx]),
                "root_pos": root_pos,
                "root_index": int(root_idx),
                "source": int(src),
                "target": int(tgt),
                "source_name": str(adata.obs_names[src]),
                "target_name": str(adata.obs_names[tgt]),
                "edge_cost": float(data["weight"]),
                "base_cost": float(data.get("base_cost", data["weight"])),
                "biological_penalty": float(data.get("biological_penalty", 0.0)),
                "modifier_total": float(data.get("modifier_total", 0.0)),
                "P_ij": float(data.get("P_ij", 0.0)),
                "H_ij": float(data.get("H_ij", 0.0)),
                "F_ij": float(data.get("F_ij", 0.0)),
                "M_ij": float(data.get("M_ij", 0.0)),
                "B_ij": float(data.get("B_ij", 0.0)),
                "S_global": float(data["S_global"]),
                "S_rootprog": float(data["S_rootprog"]),
                "rootprog_source": str(data.get("rootprog_source", "embedding_signature_fallback")),
                "trajectory_compatibility": float(data.get("trajectory_compatibility", 1.0)),
                "A_source_root": float(membership[src]),
                "A_target_root": float(membership[tgt]),
                "pseudotime_source": float(ps) if math.isfinite(ps) else np.nan,
                "pseudotime_target": float(pt) if math.isfinite(pt) else np.nan,
                "direction": "root_distance_outward",
            }
        )
    summary = {
        "root_id": str(adata.obs_names[root_idx]),
        "root_pos": root_pos,
        "root_index": int(root_idx),
        "basin_size": int(len(basin_arr)),
        "tree_nodes": int(tree.number_of_nodes()),
        "tree_edges": int(tree.number_of_edges()),
        "n_components": int(nx.number_connected_components(tree)) if tree.number_of_nodes() else 0,
        "direction_method": "root_distance_outward",
        "rootprog_source": "root_specific_NMF" if nmf_edge_lookup else "embedding_signature_fallback",
        **disconnected_audit,
    }
    edge_df = pd.DataFrame(rows)
    if edge_df.empty:
        edge_df = pd.DataFrame(columns=[
            "root_id", "root_pos", "root_index", "source", "target",
            "source_name", "target_name", "edge_cost", "base_cost",
            "biological_penalty", "modifier_total", "S_global", "S_rootprog",
            "trajectory_compatibility", "A_source_root", "A_target_root",
            "pseudotime_source", "pseudotime_target", "direction",
        ])
    return edge_df, summary, tree

def _safe_cosine(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= eps:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _tree_distances(tree: nx.Graph, root: int) -> dict[int, float]:
    if root not in tree:
        return {}
    return nx.single_source_dijkstra_path_length(tree, root, weight="weight")


def _local_tree_direction(tree: nx.Graph, dist: dict[int, float], node: int, emb: np.ndarray, eps: float) -> np.ndarray:
    if node not in tree or node not in dist:
        return np.zeros(emb.shape[1], dtype=float)
    downstream = []
    upstream = []
    for nb in tree.neighbors(node):
        if nb not in dist:
            continue
        if dist[nb] > dist[node] + eps:
            downstream.append(emb[int(nb)] - emb[node])
        elif dist[nb] < dist[node] - eps:
            upstream.append(emb[node] - emb[int(nb)])
    vecs = downstream if downstream else upstream
    if not vecs:
        return np.zeros(emb.shape[1], dtype=float)
    return np.mean(np.vstack(vecs), axis=0)


def _downstream_nodes(tree: nx.Graph, dist: dict[int, float], node: int, eps: float) -> set[int]:
    if node not in tree or node not in dist:
        return set()
    seen: set[int] = set()
    stack = [int(nb) for nb in tree.neighbors(node) if nb in dist and dist[nb] > dist[node] + eps]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nb in tree.neighbors(cur):
            nb = int(nb)
            if nb not in seen and nb in dist and dist[nb] > dist[cur] + eps:
                stack.append(nb)
    return seen


def detect_events(adata, A: np.ndarray, emb: np.ndarray, trees: dict[int, nx.Graph], roots: list[int], cfg: Config):
    """Score shared nodes and tree-pair interactions after initial root-specific MSTs."""
    H = normalized_entropy(A, cfg.eps)
    K = A.shape[1]
    sigs = np.vstack([np.average(emb, axis=0, weights=A[:, k] + cfg.eps) for k in range(K)])
    scale = max(float(np.median(np.linalg.norm(emb - emb.mean(axis=0), axis=1))), cfg.eps)
    primary = np.argmax(A, axis=1)
    dist_maps = {k: _tree_distances(trees.get(k, nx.Graph()), int(roots[k])) for k in range(K)}
    direction_maps = {
        k: {i: _local_tree_direction(trees.get(k, nx.Graph()), dist_maps[k], i, emb, cfg.eps) for i in trees.get(k, nx.Graph()).nodes()}
        for k in range(K)
    }
    downstream_cache: dict[tuple[int, int], set[int]] = {}

    rows = []
    pair_rows = []
    promotion_rows = []
    inhibition_rows = []
    collision_rows = []
    type_rows = []

    for k in range(K):
        for ell in range(k + 1, K):
            min_a = np.minimum(A[:, k], A[:, ell])
            margin = np.abs(A[:, k] - A[:, ell])
            collision_vector = min_a * (1.0 - margin)
            candidate_mask = (
                ((A[:, k] > cfg.theta_A) & (A[:, ell] > cfg.theta_A))
                | (collision_vector > cfg.theta_collision)
                | ((H > cfg.theta_entropy) & (A[:, k] > cfg.theta_secondary) & (A[:, ell] > cfg.theta_secondary))
            )
            candidates = np.flatnonzero(candidate_mask)
            class_counts = {
                "shared": 0,
                "shared_boundary": 0,
                "convergence_candidate": 0,
                "collision_boundary": 0,
                "ambiguous": 0,
            }
            conv_scores: list[float] = []
            conflict_scores: list[float] = []
            collision_scores: list[float] = []
            promotion_scores: list[float] = []
            inhibition_scores: list[float] = []

            for i0 in candidates:
                i = int(i0)
                aik = float(A[i, k])
                ail = float(A[i, ell])
                dk = np.linalg.norm(emb[i] - sigs[k])
                dl = np.linalg.norm(emb[i] - sigs[ell])
                state_plus = float(math.exp(-((dk * dk + dl * dl) / (scale * scale + cfg.eps))))
                cos_dir = _safe_cosine(
                    direction_maps.get(k, {}).get(i, np.zeros(emb.shape[1], dtype=float)),
                    direction_maps.get(ell, {}).get(i, np.zeros(emb.shape[1], dtype=float)),
                    cfg.eps,
                )
                dir_plus = 0.5 * (cos_dir + 1.0)
                dir_minus = 1.0 - dir_plus
                exclusivity = abs(aik - ail)

                downstream_cache.setdefault((k, i), _downstream_nodes(trees.get(k, nx.Graph()), dist_maps[k], i, cfg.eps))
                downstream_cache.setdefault((ell, i), _downstream_nodes(trees.get(ell, nx.Graph()), dist_maps[ell], i, cfg.eps))
                dk_nodes = downstream_cache[(k, i)]
                dl_nodes = downstream_cache[(ell, i)]
                union = dk_nodes | dl_nodes
                downstream_overlap = float(len(dk_nodes & dl_nodes) / len(union)) if union else 0.0

                collision_score = float(min(aik, ail) * (1.0 - exclusivity))
                merge_score = float(dir_plus * downstream_overlap)
                conflict_score = float(dir_minus * exclusivity)
                convergence_score = float(collision_score * merge_score * state_plus)
                promotion_score = float(aik * ail * dir_plus * state_plus)
                inhibition_k_given_l = float(ail * (1.0 - aik) * dir_minus * exclusivity)
                inhibition_l_given_k = float(aik * (1.0 - ail) * dir_minus * exclusivity)

                if collision_score > cfg.theta_collision and merge_score > cfg.theta_conv and merge_score >= conflict_score:
                    collision_type = "convergence_merge"
                    cls = "convergence_candidate"
                elif collision_score > cfg.theta_collision and conflict_score > cfg.theta_conflict and conflict_score > merge_score:
                    collision_type = "conflict_boundary"
                    cls = "collision_boundary"
                elif collision_score > cfg.theta_collision:
                    collision_type = "shared_boundary"
                    cls = "shared_boundary"
                elif H[i] > cfg.theta_entropy:
                    collision_type = "ambiguous_shared"
                    cls = "ambiguous"
                else:
                    collision_type = "weak_shared"
                    cls = "shared"

                class_counts[cls] += 1
                conv_scores.append(convergence_score)
                conflict_scores.append(conflict_score)
                collision_scores.append(collision_score)
                promotion_scores.append(promotion_score)
                inhibition_scores.extend([inhibition_k_given_l, inhibition_l_given_k])
                active_roots = [f"T{r}" for r in range(K) if A[i, r] > cfg.theta_A]
                secondary_roots = [f"T{r}" for r in range(K) if r != int(primary[i]) and A[i, r] > cfg.theta_secondary]

                base_row = {
                    "node_id": i,
                    "node_name": str(adata.obs_names[i]),
                    "tree_k": k,
                    "tree_l": ell,
                    "A_ik": aik,
                    "A_il": ail,
                    "primary_root": int(primary[i]),
                    "secondary_roots": "|".join(secondary_roots),
                    "active_roots": "|".join(active_roots),
                    "entropy": float(H[i]),
                    "state_compatibility": state_plus,
                    "direction_cosine": cos_dir,
                    "direction_consistency": dir_plus,
                    "direction_conflict": dir_minus,
                    "membership_exclusivity": exclusivity,
                    "downstream_overlap": downstream_overlap,
                    "collision_score": collision_score,
                    "merge_score": merge_score,
                    "convergence_score": convergence_score,
                    "conflict_score": conflict_score,
                    "promotion_score": promotion_score,
                    "collision_type": collision_type,
                    "classification": cls,
                }
                rows.append(base_row)
                collision_rows.append(base_row)
                type_rows.append(
                    {
                        "node_id": i,
                        "node_name": str(adata.obs_names[i]),
                        "tree_k": k,
                        "tree_l": ell,
                        "primary_root": int(primary[i]),
                        "secondary_roots": "|".join(secondary_roots),
                        "soft_memberships": "|".join(f"T{r}:{A[i, r]:.6g}" for r in range(K)),
                        "entropy": float(H[i]),
                        "collision_score": collision_score,
                        "merge_score": merge_score,
                        "conflict_score": conflict_score,
                        "collision_type": collision_type,
                    }
                )
                promotion_rows.append(
                    {
                        "tree_k": k,
                        "tree_l": ell,
                        "node_id": i,
                        "node_name": str(adata.obs_names[i]),
                        "A_ik": aik,
                        "A_il": ail,
                        "direction_consistency": dir_plus,
                        "state_compatibility": state_plus,
                        "promotion_score": promotion_score,
                        "passes_threshold": bool(promotion_score > cfg.theta_promotion),
                    }
                )
                inhibition_rows.extend(
                    [
                        {
                            "target_tree": k,
                            "inhibiting_tree": ell,
                            "node_id": i,
                            "node_name": str(adata.obs_names[i]),
                            "A_target": aik,
                            "A_inhibiting": ail,
                            "direction_conflict": dir_minus,
                            "membership_exclusivity": exclusivity,
                            "inhibition_score": inhibition_k_given_l,
                            "passes_threshold": bool(inhibition_k_given_l > cfg.theta_inhibition),
                        },
                        {
                            "target_tree": ell,
                            "inhibiting_tree": k,
                            "node_id": i,
                            "node_name": str(adata.obs_names[i]),
                            "A_target": ail,
                            "A_inhibiting": aik,
                            "direction_conflict": dir_minus,
                            "membership_exclusivity": exclusivity,
                            "inhibition_score": inhibition_l_given_k,
                            "passes_threshold": bool(inhibition_l_given_k > cfg.theta_inhibition),
                        },
                    ]
                )

            mean_prom = float(np.mean(promotion_scores)) if promotion_scores else 0.0
            mean_inh = float(np.mean(inhibition_scores)) if inhibition_scores else 0.0
            mean_coll = float(np.mean(collision_scores)) if collision_scores else 0.0
            mean_merge = float(np.mean(conv_scores)) if conv_scores else 0.0
            mean_conflict = float(np.mean(conflict_scores)) if conflict_scores else 0.0
            dominant_scores = {
                "promotion_like": mean_prom,
                "inhibition_like": mean_inh,
                "collision_like": mean_coll,
                "convergence_merge": mean_merge,
                "conflict_boundary": mean_conflict,
            }
            dominant_relationship = max(dominant_scores, key=dominant_scores.get) if candidates.size else "none"
            pair_rows.append(
                {
                    "tree_k": k,
                    "tree_l": ell,
                    "n_shared": int(((A[:, k] > cfg.theta_A) & (A[:, ell] > cfg.theta_A)).sum()),
                    "n_scored_nodes": int(len(candidates)),
                    "n_shared_boundary": class_counts["shared_boundary"],
                    "n_convergence": class_counts["convergence_candidate"],
                    "n_collision": class_counts["collision_boundary"],
                    "n_ambiguous": class_counts["ambiguous"],
                    "mean_convergence_score": mean_merge,
                    "mean_collision_score": mean_coll,
                    "mean_conflict_score": mean_conflict,
                    "max_convergence_score": float(np.max(conv_scores)) if conv_scores else 0.0,
                    "max_collision_score": float(np.max(collision_scores)) if collision_scores else 0.0,
                    "max_conflict_score": float(np.max(conflict_scores)) if conflict_scores else 0.0,
                    "promotion_like_score": mean_prom,
                    "inhibition_like_score": mean_inh,
                    "dominant_relationship": dominant_relationship,
                    "evidence_source": "soft_membership+tree_direction+embedding_state",
                    "evidence_status": "model_inferred_association_only",
                }
            )

    event_cols = [
        "node_id",
        "node_name",
        "tree_k",
        "tree_l",
        "A_ik",
        "A_il",
        "primary_root",
        "secondary_roots",
        "active_roots",
        "entropy",
        "state_compatibility",
        "direction_cosine",
        "direction_consistency",
        "direction_conflict",
        "membership_exclusivity",
        "downstream_overlap",
        "collision_score",
        "merge_score",
        "convergence_score",
        "conflict_score",
        "promotion_score",
        "collision_type",
        "classification",
    ]
    pair_cols = [
        "tree_k",
        "tree_l",
        "n_shared",
        "n_scored_nodes",
        "n_shared_boundary",
        "n_convergence",
        "n_collision",
        "n_ambiguous",
        "mean_convergence_score",
        "mean_collision_score",
        "mean_conflict_score",
        "max_convergence_score",
        "max_collision_score",
        "max_conflict_score",
        "promotion_like_score",
        "inhibition_like_score",
        "dominant_relationship",
        "evidence_source",
        "evidence_status",
    ]
    tables = {
        "promotion_scores.csv": pd.DataFrame(promotion_rows),
        "inhibition_scores.csv": pd.DataFrame(inhibition_rows),
        "collision_scores.csv": pd.DataFrame(collision_rows, columns=event_cols),
        "collision_type_table.csv": pd.DataFrame(type_rows),
    }
    return pd.DataFrame(rows, columns=event_cols), pd.DataFrame(pair_rows, columns=pair_cols), tables



def build_interaction_node_modifiers(
    events: pd.DataFrame,
    promotion_df: pd.DataFrame,
    inhibition_df: pd.DataFrame,
    n_nodes: int,
    n_roots: int,
    cfg: Config,
) -> dict[str, np.ndarray]:
    """Convert node-level tree interaction evidence into root-specific modifiers.

    P: promotion support. Higher P lowers edge cost for that root.
    H: inhibition penalty. Higher H increases edge cost for that root.
    F: conflict-boundary penalty. Higher F increases crossing cost.
    M: merge/convergence support. Higher M lowers shared downstream branch cost.
    B: shared-boundary flag/modifier. Usually lambda_boundary is 0 by default;
       B is still exported for labeling and future tuning.
    """
    P = np.zeros((n_nodes, n_roots), dtype=float)
    Hm = np.zeros((n_nodes, n_roots), dtype=float)
    F = np.zeros((n_nodes, n_roots), dtype=float)
    M = np.zeros((n_nodes, n_roots), dtype=float)
    B = np.zeros((n_nodes, n_roots), dtype=float)

    if promotion_df is not None and not promotion_df.empty:
        for row in promotion_df.itertuples(index=False):
            score = float(getattr(row, "promotion_score", 0.0))
            if score <= 0:
                continue
            i = int(row.node_id)
            k = int(row.tree_k)
            ell = int(row.tree_l)
            P[i, k] = max(P[i, k], score)
            P[i, ell] = max(P[i, ell], score)

    if inhibition_df is not None and not inhibition_df.empty:
        for row in inhibition_df.itertuples(index=False):
            score = float(getattr(row, "inhibition_score", 0.0))
            if score <= 0:
                continue
            i = int(row.node_id)
            k = int(row.target_tree)
            Hm[i, k] = max(Hm[i, k], score)

    if events is not None and not events.empty:
        for row in events.itertuples(index=False):
            i = int(row.node_id)
            k = int(row.tree_k)
            ell = int(row.tree_l)
            ctype = str(getattr(row, "collision_type", ""))
            collision_score = float(getattr(row, "collision_score", 0.0))
            merge_score = float(getattr(row, "merge_score", 0.0))
            conflict_score = float(getattr(row, "conflict_score", 0.0))
            if ctype == "convergence_merge":
                val = max(collision_score, merge_score)
                M[i, k] = max(M[i, k], val)
                M[i, ell] = max(M[i, ell], val)
            elif ctype == "conflict_boundary":
                val = max(collision_score, conflict_score)
                F[i, k] = max(F[i, k], val)
                F[i, ell] = max(F[i, ell], val)
            elif ctype in {"shared_boundary", "ambiguous_shared", "weak_shared"}:
                val = collision_score
                B[i, k] = max(B[i, k], val)
                B[i, ell] = max(B[i, ell], val)

    return {"P": P, "H": Hm, "F": F, "M": M, "B": B}


def build_refined_basin_overrides(
    A: np.ndarray,
    roots: list[int],
    modifiers: dict[str, np.ndarray],
    cfg: Config,
) -> tuple[dict[int, set[int]], pd.DataFrame]:
    """Update each root basin using promotion, inhibition and collision evidence."""
    primary = np.argmax(A, axis=1)
    P = modifiers["P"]
    Hm = modifiers["H"]
    F = modifiers["F"]
    M = modifiers["M"]
    B = modifiers["B"]
    overrides: dict[int, set[int]] = {}
    audit_rows: list[dict[str, object]] = []

    for k, root_idx in enumerate(roots):
        initial = set(map(int, np.flatnonzero((A[:, k] > cfg.theta_A) | (primary == k))))
        promoted = set(map(int, np.flatnonzero(P[:, k] > cfg.theta_promotion)))
        collision = set(map(int, np.flatnonzero((F[:, k] > cfg.eps) | (M[:, k] > cfg.eps) | (B[:, k] > cfg.eps))))
        inhibited = set(
            map(
                int,
                np.flatnonzero(
                    (Hm[:, k] > cfg.theta_inhibition)
                    & (A[:, k] < cfg.theta_low_membership)
                    & (primary != k)
                ),
            )
        )
        refined = (initial | promoted | collision) - inhibited
        refined.add(int(root_idx))
        overrides[k] = refined

        for node in sorted(initial | refined):
            if node in initial and node in refined:
                action = "kept"
            elif node in refined:
                action = "added"
            else:
                action = "removed"
            audit_rows.append(
                {
                    "root_pos": k,
                    "root_index": int(root_idx),
                    "node_id": node,
                    "action": action,
                    "membership": float(A[node, k]),
                    "is_primary": bool(primary[node] == k),
                    "promotion": float(P[node, k]),
                    "inhibition": float(Hm[node, k]),
                    "conflict": float(F[node, k]),
                    "merge": float(M[node, k]),
                    "boundary": float(B[node, k]),
                }
            )
    return overrides, pd.DataFrame(audit_rows)


def write_edge_modifier_table(
    edges_df: pd.DataFrame,
    A: np.ndarray,
    modifiers: dict[str, np.ndarray],
    outdir: Path,
    cfg: Config,
) -> pd.DataFrame:
    """Export non-zero interaction-aware edge modifiers for auditing."""
    rows = []
    K = A.shape[1]
    P = modifiers.get("P")
    Hm = modifiers.get("H")
    F = modifiers.get("F")
    M = modifiers.get("M")
    B = modifiers.get("B")
    for root_pos in range(K):
        membership = A[:, root_pos]
        primary = np.argmax(A, axis=1)
        basin_set = set(map(int, np.flatnonzero((membership > cfg.theta_A) | (primary == root_pos))))
        for row in edges_df.itertuples(index=False):
            i, j = int(row.source), int(row.target)
            if i not in basin_set or j not in basin_set:
                continue
            P_ij = 0.5 * (float(P[i, root_pos]) + float(P[j, root_pos])) if P is not None else 0.0
            H_ij = 0.5 * (float(Hm[i, root_pos]) + float(Hm[j, root_pos])) if Hm is not None else 0.0
            F_ij = 0.5 * (float(F[i, root_pos]) + float(F[j, root_pos])) if F is not None else 0.0
            M_ij = 0.5 * (float(M[i, root_pos]) + float(M[j, root_pos])) if M is not None else 0.0
            B_ij = 0.5 * (float(B[i, root_pos]) + float(B[j, root_pos])) if B is not None else 0.0
            modifier_total = (
                cfg.lambda_inh * H_ij
                - cfg.lambda_prom * P_ij
                + cfg.lambda_conflict * F_ij
                - cfg.lambda_merge * M_ij
                + cfg.lambda_boundary * B_ij
            )
            if abs(modifier_total) <= cfg.eps and P_ij <= cfg.eps and H_ij <= cfg.eps and F_ij <= cfg.eps and M_ij <= cfg.eps and B_ij <= cfg.eps:
                continue
            rows.append(
                {
                    "root_pos": root_pos,
                    "source": i,
                    "target": j,
                    "source_name": getattr(row, "source_name", str(i)),
                    "target_name": getattr(row, "target_name", str(j)),
                    "P_ij": P_ij,
                    "H_ij": H_ij,
                    "F_ij": F_ij,
                    "M_ij": M_ij,
                    "B_ij": B_ij,
                    "modifier_total": modifier_total,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "edge_cost_modifiers.csv", index=False)
    return df


def compare_initial_refined_trees(initial_edges: pd.DataFrame, refined_edges: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Compare undirected initial/refined tree edges and export an audit log."""
    def edge_key(row) -> tuple[int, int, int]:
        u, v = sorted((int(row.source), int(row.target)))
        return (int(row.root_pos), u, v)

    init_map = {edge_key(row): row for row in initial_edges.itertuples(index=False)} if initial_edges is not None and not initial_edges.empty else {}
    ref_map = {edge_key(row): row for row in refined_edges.itertuples(index=False)} if refined_edges is not None and not refined_edges.empty else {}
    rows = []
    for key in sorted(set(init_map) | set(ref_map)):
        irow = init_map.get(key)
        rrow = ref_map.get(key)
        if irow is not None and rrow is not None:
            action = "kept"
        elif irow is not None:
            action = "removed_after_refinement"
        else:
            action = "added_after_refinement"
        src = int(getattr(rrow or irow, "source"))
        tgt = int(getattr(rrow or irow, "target"))
        rows.append(
            {
                "root_pos": key[0],
                "source": src,
                "target": tgt,
                "action": action,
                "initial_edge_cost": float(getattr(irow, "edge_cost", np.nan)) if irow is not None else np.nan,
                "refined_edge_cost": float(getattr(rrow, "edge_cost", np.nan)) if rrow is not None else np.nan,
                "refined_modifier_total": float(getattr(rrow, "modifier_total", np.nan)) if rrow is not None else np.nan,
                "P_ij": float(getattr(rrow, "P_ij", np.nan)) if rrow is not None else np.nan,
                "H_ij": float(getattr(rrow, "H_ij", np.nan)) if rrow is not None else np.nan,
                "F_ij": float(getattr(rrow, "F_ij", np.nan)) if rrow is not None else np.nan,
                "M_ij": float(getattr(rrow, "M_ij", np.nan)) if rrow is not None else np.nan,
                "B_ij": float(getattr(rrow, "B_ij", np.nan)) if rrow is not None else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "tree_refinement_log.csv", index=False)
    return df


def plot_before(coords, A, roots, out: Path):
    plt.figure(figsize=(7, 6))
    primary = np.argmax(A, axis=1)
    plt.scatter(coords[:, 0], coords[:, 1], c=primary, s=8, cmap="tab20", alpha=0.75, linewidths=0)
    plt.scatter(coords[roots, 0], coords[roots, 1], marker="*", s=180, c="black", edgecolor="white", linewidths=0.7)
    plt.title("Before: hard primary-root assignment")
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_entropy(coords, A, roots, out: Path, cfg: Config):
    H = normalized_entropy(A, cfg.eps)
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=H, s=8, cmap="viridis", alpha=0.82, linewidths=0)
    plt.colorbar(sc, label="Normalized entropy")
    plt.scatter(coords[roots, 0], coords[roots, 1], marker="*", s=180, c="red", edgecolor="white", linewidths=0.7)
    plt.title("Soft assignment entropy")
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_after(coords, roots, tree_edges: pd.DataFrame, events: pd.DataFrame, out: Path):
    plt.figure(figsize=(8, 7))
    plt.scatter(coords[:, 0], coords[:, 1], s=5, c="#d1d5db", alpha=0.55, linewidths=0)
    if not tree_edges.empty:
        for row in tree_edges.itertuples(index=False):
            u, v = int(row.source), int(row.target)
            plt.plot([coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]], lw=0.55, alpha=0.42, color="#111827")
    if not events.empty:
        plot_specs = [
            ("convergence_candidate", "o", "#2563eb", 34),
            ("collision_boundary", "x", "#dc2626", 48),
            ("ambiguous", "s", "#9333ea", 24),
        ]
        for cls, marker, color, size in plot_specs:
            sub = events[events["classification"] == cls]
            if not sub.empty:
                idx = sub["node_id"].astype(int).to_numpy()
                plt.scatter(coords[idx, 0], coords[idx, 1], marker=marker, s=size, c=color, label=cls, alpha=0.9)
    plt.scatter(coords[roots, 0], coords[roots, 1], marker="*", s=190, c="black", edgecolor="white", linewidths=0.7, label="roots")
    plt.legend(fontsize=8)
    plt.title("After: soft trees + convergence/collision")
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_interactions(pair_df: pd.DataFrame, K: int, out: Path):
    G = nx.Graph()
    G.add_nodes_from(range(K))
    if not pair_df.empty:
        for row in pair_df.itertuples(index=False):
            w = max(float(row.mean_convergence_score), float(row.mean_collision_score))
            if w > 0:
                G.add_edge(int(row.tree_k), int(row.tree_l), weight=w)
    pos = nx.circular_layout(G)
    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_size=850, node_color="#bfdbfe", edgecolors="#111827")
    nx.draw_networkx_labels(G, pos, {k: f"T{k}" for k in G.nodes()}, font_size=10)
    widths = [1.0 + 8.0 * float(G[u][v]["weight"]) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.65, edge_color="#374151")
    plt.title("Tree-level interaction graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def write_soft_membership(
    adata,
    A: np.ndarray,
    roots: list[int],
    outdir: Path,
    cfg: Config,
    active_assignment: Optional[np.ndarray] = None,
):
    """Write backward-compatible soft membership with v7.3 active-root semantics."""
    H = normalized_entropy(A, cfg.eps)
    primary = v71bio.safe_primary_roots(A, cfg.eps)
    rows = []
    for i in range(A.shape[0]):
        active = (
            np.flatnonzero(active_assignment[i] > 0)
            if active_assignment is not None
            else np.asarray([primary[i]], dtype=int) if primary[i] >= 0 else np.asarray([], dtype=int)
        )
        for k, root in enumerate(roots):
            p = int(primary[i])
            rows.append(
                {
                    "node_id": i,
                    "node_name": str(adata.obs_names[i]),
                    "root_pos": k,
                    "root_index": int(root),
                    "root_id": str(adata.obs_names[root]),
                    "A_ik": float(A[i, k]),
                    "primary_root": p,
                    "primary_root_id": str(adata.obs_names[roots[p]]) if p >= 0 else "",
                    "is_primary": bool(p == k),
                    "is_secondary": bool(k in active and p != k),
                    "entropy": float(H[i]),
                    "n_active_roots": int(len(active)),
                    "shared_status": "validated_shared" if len(active) >= 2 else ("single" if len(active) == 1 else "unassigned"),
                    "membership_semantics": "A_sharp_probability; active roots come from evidence-gated Z_active",
                }
            )
    pd.DataFrame(rows).to_csv(outdir / "soft_membership.csv", index=False)
    return H

def quality_checks(outdir: Path, A: np.ndarray, roots: list[int], tree_edges: pd.DataFrame, tree_summaries: pd.DataFrame, events: pd.DataFrame, cfg: Config, skip_spatial_figures: bool = False):
    figdir = outdir / "figures"
    checks = {
        "each_root_has_tree_summary": bool(len(tree_summaries) == len(roots)),
        "each_tree_edges_positive": bool((tree_summaries["tree_edges"].fillna(0) > 0).all()) if not tree_summaries.empty else False,
        "soft_membership_rows_sum_to_0_or_1": bool(np.all(np.isclose(A.sum(axis=1), 1.0, atol=1e-4) | np.isclose(A.sum(axis=1), 0.0, atol=1e-4))),
        "entropy_in_0_1": bool(((normalized_entropy(A, cfg.eps) >= -1e-8) & (normalized_entropy(A, cfg.eps) <= 1 + 1e-8)).all()),
        "tree_edges_has_source_target": bool({"source", "target"}.issubset(tree_edges.columns)),
        "convergence_has_classification": bool("classification" in events.columns),
        "figures_saved": True if skip_spatial_figures else all(
            (figdir / name).exists()
            for name in [
                "01_before_hard_primary_assignment.png",
                "02_soft_assignment_entropy.png",
                "03_after_soft_trees_convergence_collision.png",
                "04_tree_level_interaction_graph.png",
            ]
        ),
    }
    return checks


def write_final_graphml(adata, A: np.ndarray, tree_edges: pd.DataFrame, events: pd.DataFrame, outdir: Path, cfg: Config, active_assignment: Optional[np.ndarray] = None) -> None:
    H = normalized_entropy(A, cfg.eps)
    primary = v71bio.safe_primary_roots(A, cfg.eps)
    active_counts = active_assignment.sum(axis=1) if active_assignment is not None else (A > cfg.theta_A).sum(axis=1)
    event_lookup: dict[int, str] = {}
    if not events.empty:
        priority = {"collision_boundary": 5, "convergence_candidate": 4, "shared_boundary": 3, "ambiguous": 2, "shared": 1}
        for row in events.itertuples(index=False):
            node = int(row.node_id)
            cls = str(row.classification)
            if priority.get(cls, 0) > priority.get(event_lookup.get(node, ""), 0):
                event_lookup[node] = cls

    G = nx.DiGraph()
    for i, name in enumerate(adata.obs_names.astype(str)):
        G.add_node(
            str(i),
            obs_name=name,
            primary_root=int(primary[i]),
            entropy=float(H[i]),
            active_root_count=int(active_counts[i]),
            node_class=event_lookup.get(i, "shared" if active_counts[i] >= 2 else "single"),
        )
    if not tree_edges.empty:
        for row in tree_edges.itertuples(index=False):
            src = str(int(row.source))
            tgt = str(int(row.target))
            key = f"r{int(row.root_pos)}_{src}_{tgt}"
            G.add_edge(
                src,
                tgt,
                key=key,
                root_pos=int(row.root_pos),
                edge_cost=float(row.edge_cost),
                S_global=float(row.S_global),
                S_rootprog=float(row.S_rootprog),
                direction=str(row.direction),
            )
    nx.write_graphml(G, outdir / "final_multiroot_graph.graphml")


def run_one_dataset(
    adata_path: Path,
    roots_df: pd.DataFrame,
    base_outdir: Path,
    cfg: Config,
    dataset_name: Optional[str],
    dataset_col: Optional[str],
    root_col: Optional[str],
    root_specific_nmf_dir: Optional[Path] = None,
    sample_key: Optional[str] = None,
    graph_mode: str = "product",
    lambda_s: float = 1.0,
    lambda_x: float = 1.0,
    skip_spatial_figures: bool = False,
):
    """Run the v7.3 base layer: dual graph + sparse process-constrained Markov."""
    ds = dataset_name_from_path(adata_path, dataset_name)
    outdir = ensure_dir(base_outdir / ds)
    figdir = outdir / "figures"
    if not skip_spatial_figures:
        ensure_dir(figdir)
    adata_raw = read_h5ad(adata_path)

    # v7.3: construct/resolve metacells before root mapping. Cluster-level roots
    # such as C41/C00/C08 must never be silently projected to arbitrary raw cells.
    metacell_result = v72meta.build_or_use_metacells(
        adata_raw=adata_raw,
        roots_df=roots_df,
        dataset_name=ds,
        cfg=cfg,
        dataset_col=dataset_col,
        root_col=root_col,
        sample_key=sample_key,
    )
    v72meta.write_metacell_outputs(metacell_result, outdir)
    adata = metacell_result.adata
    roots, resolved = v72meta.resolve_roots_to_metacells(
        adata_meta=adata,
        roots_df=roots_df,
        dataset_name=ds,
        dataset_col=dataset_col,
        root_col=root_col,
        cfg=cfg,
    )
    resolved["subset_index"] = resolved["resolved_index"].astype(int)
    resolved["subset_obs_name"] = resolved["resolved_obs_name"].astype(str)
    resolved.to_csv(outdir / "resolved_roots.csv", index=False)
    root_profiles = v72meta.build_root_process_profiles(adata, roots, resolved)
    root_profiles.to_csv(outdir / "root_process_profiles.csv", index=False)

    # Backward-compatible identity mapping now refers to metacell nodes, not raw cells.
    subset_map = pd.DataFrame({
        "original_index": np.arange(adata.n_obs, dtype=int),
        "subset_index": np.arange(adata.n_obs, dtype=int),
        "obs_name": adata.obs_names.astype(str),
        "kept_reason": "v7.3_metacell_node",
    })
    subset_map.to_csv(outdir / "cell_subset_mapping.csv", index=False)

    # Step 1: build a global evidence graph with the legacy similarity machinery.
    if graph_mode == "step6":
        S_global, global_edges, coords, emb, domains = build_candidate_graph_step6(
            adata, cfg, outdir, sample_key, lambda_s, lambda_x
        )
    else:
        S_global, global_edges, coords, emb, domains = build_candidate_graph(
            adata, cfg, outdir, sample_key
        )
    global_edges.to_csv(outdir / "global_evidence_graph_edges.csv", index=False)

    # Step 2: infer broad biological domains and split the graph.
    node_domains = v71bio.infer_node_biological_domains(adata, cfg.annotation_key)
    node_domains = v71bio.apply_root_profile_overrides(node_domains, root_profiles)
    node_domains.to_csv(outdir / "biological_node_domains.csv", index=False)
    annotated_edges = v71bio.annotate_edge_biological_compatibility(global_edges, node_domains, cfg)
    trajectory_edges, interaction_edges = v71bio.split_edges(annotated_edges)
    annotated_edges.to_csv(outdir / "biological_edge_compatibility.csv", index=False)
    trajectory_edges.to_csv(outdir / "trajectory_candidate_edges.csv", index=False)
    interaction_edges.to_csv(outdir / "interaction_edges.csv", index=False)
    # Backward-compatible filename now explicitly means the graph used by Markov/tree construction.
    trajectory_edges.to_csv(outdir / "candidate_graph_edges.csv", index=False)

    # Step 3: v7.3 local super-root anchors and root-local process affinity.
    root_anchor_sets = v71bio.build_root_anchor_sets(
        adata=adata,
        roots=roots,
        resolved_roots=resolved,
        root_profiles=root_profiles,
        node_domains=node_domains,
        trajectory_edges=trajectory_edges,
        cfg=cfg,
        embedding=emb,
    )
    root_anchor_sets.to_csv(outdir / "root_anchor_sets.csv", index=False)
    root_local_affinity, root_affinity_df = v71bio.compute_root_local_affinity(
        adata=adata,
        roots=roots,
        root_profiles=root_profiles,
        node_domains=node_domains,
        trajectory_edges=trajectory_edges,
        root_anchor_sets=root_anchor_sets,
        cfg=cfg,
        embedding=emb,
    )
    root_affinity_df.to_csv(outdir / "root_local_affinity.csv", index=False)

    # Step 4: root-local, process-constrained absorbing Markov assignment.
    A_raw, root_mask, component_audit, root_scope = v71bio.process_constrained_absorbing_markov(
        adata.n_obs,
        trajectory_edges,
        roots,
        node_domains,
        cfg,
        root_profiles=root_profiles,
        root_anchor_sets=root_anchor_sets,
        root_local_affinity=root_local_affinity,
    )
    A_bio = v71bio.apply_biological_root_mask(A_raw, root_mask, roots, cfg.eps)
    A_sharp = v71bio.sharpen_membership(A_bio, cfg.markov_sharpen_q, cfg.eps)
    Z_active, assignment_diag = v71bio.build_sparse_active_assignment(
        A_sharp, roots, node_domains, trajectory_edges, cfg, root_mask=root_mask
    )
    component_audit.to_csv(outdir / "disconnected_component_audit.csv", index=False)
    root_scope.to_csv(outdir / "root_process_scope.csv", index=False)
    pd.DataFrame(root_mask.astype(int), columns=[f"root_{k}" for k in range(len(roots))]).assign(
        node_id=np.arange(adata.n_obs, dtype=int)
    ).to_csv(outdir / "biological_root_eligibility_mask.csv", index=False)
    v71bio.write_assignment_outputs(
        outdir, adata, roots, A_raw, A_bio, A_sharp, Z_active, assignment_diag, stage="initial"
    )
    H = write_soft_membership(adata, A_sharp, roots, outdir, cfg, Z_active)
    assignment_method = "v7.3_local_anchor_process_constrained_absorbing_markov"

    nmf_dir = None
    if root_specific_nmf_dir is not None:
        if (root_specific_nmf_dir / "root_specific_graph_edges.csv").exists():
            nmf_dir = root_specific_nmf_dir
        elif (root_specific_nmf_dir / ds / "root_specific_graph_edges.csv").exists():
            nmf_dir = root_specific_nmf_dir / ds
        elif (root_specific_nmf_dir / ds / "root_specific_nmf" / "root_specific_graph_edges.csv").exists():
            nmf_dir = root_specific_nmf_dir / ds / "root_specific_nmf"
    nmf_edge_lookup = load_root_specific_nmf_edges(nmf_dir)

    if not skip_spatial_figures:
        plot_before(coords, A_sharp, roots, figdir / "01_before_hard_primary_assignment.png")
        plot_entropy(coords, A_sharp, roots, figdir / "02_soft_assignment_entropy.png", cfg)

    tree_dfs = []
    summaries = []
    trees: dict[int, nx.Graph] = {}
    for k, root in enumerate(roots):
        tdf, summary, tree = build_one_tree(
            adata,
            trajectory_edges,
            A_sharp,
            root,
            k,
            emb,
            domains,
            cfg,
            nmf_edge_lookup,
            active_assignment=Z_active,
            compatibility_mask=root_mask,
        )
        tree_dfs.append(tdf)
        summaries.append(summary)
        trees[k] = tree
    initial_tree_edges = pd.concat(tree_dfs, ignore_index=True) if tree_dfs else pd.DataFrame()
    initial_tree_edges.to_csv(outdir / "initial_root_specific_tree_edges.csv", index=False)
    initial_tree_summaries = pd.DataFrame(summaries)
    initial_tree_summaries.to_csv(outdir / "initial_tree_summaries.csv", index=False)
    if not initial_tree_summaries.empty:
        initial_tree_summaries.to_csv(outdir / "root_component_retention_audit.csv", index=False)

    # Old event scoring is retained as a diagnostic only.  It no longer controls basin inclusion.
    events, interactions, interaction_tables = detect_events(adata, A_sharp, emb, trees, roots, cfg)
    events.to_csv(outdir / "initial_convergence_nodes.csv", index=False)
    interactions.to_csv(outdir / "initial_tree_interactions.csv", index=False)
    interactions.to_csv(outdir / "initial_tree_pair_interaction_summary.csv", index=False)
    for filename, table in interaction_tables.items():
        table.to_csv(outdir / filename, index=False)

    # v7.3 does not use the legacy basin-expanding interaction refinement by default.
    refined_tree_edges = initial_tree_edges.copy()
    refined_tree_summaries = initial_tree_summaries.copy()
    refined_trees = trees
    refined_events = events
    refined_interactions = interactions
    legacy_refinement_ran = False
    if not cfg.enable_biological_gate and not cfg.disable_interaction_refinement:
        legacy_refinement_ran = True
        modifiers = build_interaction_node_modifiers(
            events=events,
            promotion_df=interaction_tables.get("promotion_scores.csv", pd.DataFrame()),
            inhibition_df=interaction_tables.get("inhibition_scores.csv", pd.DataFrame()),
            n_nodes=adata.n_obs,
            n_roots=len(roots),
            cfg=cfg,
        )
        write_edge_modifier_table(trajectory_edges, A_sharp, modifiers, outdir, cfg)

    refined_tree_edges.to_csv(outdir / "root_specific_tree_edges.csv", index=False)
    refined_tree_edges.to_csv(outdir / "refined_root_specific_tree_edges.csv", index=False)
    refined_tree_summaries.to_csv(outdir / "tree_summaries.csv", index=False)
    refined_tree_summaries.to_csv(outdir / "refined_tree_summaries.csv", index=False)
    refined_events.to_csv(outdir / "convergence_nodes.csv", index=False)
    refined_interactions.to_csv(outdir / "tree_interactions.csv", index=False)
    refined_interactions.to_csv(outdir / "tree_pair_interaction_summary.csv", index=False)
    write_final_graphml(adata, A_sharp, refined_tree_edges, refined_events, outdir, cfg, Z_active)

    bio_audit = v71bio.audit_tree_biological_validity(refined_tree_edges, node_domains)
    bio_audit.to_csv(outdir / "biological_compatibility_audit.csv", index=False)
    n_forbidden = int(bio_audit["is_forbidden_trajectory_edge"].sum()) if not bio_audit.empty else 0
    if n_forbidden != 0:
        raise RuntimeError(f"v7.3 hard constraint failed: {n_forbidden} forbidden trajectory edges selected")

    if not skip_spatial_figures:
        plot_after(coords, roots, refined_tree_edges, refined_events, figdir / "03_after_soft_trees_convergence_collision.png")
        plot_interactions(refined_interactions, len(roots), figdir / "04_tree_level_interaction_graph.png")

    checks = quality_checks(
        outdir, A_sharp, roots, refined_tree_edges, refined_tree_summaries, refined_events, cfg, skip_spatial_figures
    )
    active_counts = Z_active.sum(axis=1)
    summary = {
        "dataset": ds,
        "adata_path": str(adata_path),
        "method": "CASCAT-MR-CMI-E-Gate",
        "n_obs_original": int(adata_raw.n_obs),
        "n_metacells": int(adata.n_obs),
        "n_obs_used": int(adata.n_obs),
        "n_roots": int(len(roots)),
        "root_ids": [str(adata.obs_names[r]) for r in roots],
        "root_indices": [int(r) for r in roots],
        "assignment_method": assignment_method,
        "sample_key": sample_key,
        "graph_mode": graph_mode,
        "lambda_s": lambda_s if graph_mode == "step6" else None,
        "lambda_x": lambda_x if graph_mode == "step6" else None,
        "global_evidence_edges": int(len(global_edges)),
        "trajectory_candidate_edges": int(len(trajectory_edges)),
        "interaction_edges": int(len(interaction_edges)),
        "cross_sample_edges": int(global_edges["cross_sample"].sum()) if "cross_sample" in global_edges.columns else 0,
        "initial_tree_edge_count": int(len(initial_tree_edges)),
        "tree_edge_count": int(len(refined_tree_edges)),
        "legacy_interaction_refinement_ran": legacy_refinement_ran,
        "validated_dual_root_nodes": int((active_counts == 2).sum()),
        "validated_dual_root_rate": float((active_counts == 2).mean()) if len(active_counts) else 0.0,
        "unassigned_nodes": int((active_counts == 0).sum()),
        "high_order_ambiguity_nodes": int((assignment_diag.get("assignment_status", pd.Series(dtype=str)) == "high_order_ambiguity").sum()),
        "forbidden_trajectory_edges_selected": n_forbidden,
        "ambiguous_high_entropy_nodes": int((H > cfg.theta_entropy).sum()),
        "quality_checks": checks,
        "config": asdict(cfg),
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

def default_dataset_paths() -> list[tuple[str, Path]]:
    out = []
    for name, rel in DEFAULT_DATASETS.items():
        path = Path(rel)
        if path.exists():
            out.append((name, path))
    return out


def paths_from_args(args) -> list[tuple[Optional[str], Path]]:
    if args.run_defaults:
        return default_dataset_paths()
    if args.adata:
        return [(args.dataset_name, Path(args.adata))]
    paths = [Path(p) for p in glob.glob(args.adata_glob, recursive=True)]
    return [(None, p) for p in sorted(paths)]


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--adata")
    group.add_argument("--adata-glob")
    group.add_argument("--run-defaults", action="store_true")
    parser.add_argument("--roots-csv", required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-col")
    parser.add_argument("--root-col")
    parser.add_argument("--root-specific-nmf-dir")
    parser.add_argument("--outdir", default="outputs/tme_soft_multiroot_tree")
    parser.add_argument("--k-neighbors", type=int, default=6)
    parser.add_argument("--theta-a", type=float, default=0.30)
    parser.add_argument("--beta-transition", type=float, default=4.0)
    parser.add_argument("--theta-entropy", type=float, default=0.70)
    parser.add_argument("--theta-margin", type=float, default=0.20)
    parser.add_argument("--theta-secondary", type=float, default=0.25)
    parser.add_argument("--theta-low-membership", type=float, default=0.10)
    parser.add_argument("--theta-collision", type=float, default=0.20)
    parser.add_argument("--theta-promotion", type=float, default=0.05)
    parser.add_argument("--theta-inhibition", type=float, default=0.20)
    parser.add_argument("--theta-conv", type=float, default=0.08)
    parser.add_argument("--theta-conflict", type=float, default=0.08)
    parser.add_argument("--lambda-a", type=float, default=0.30)
    parser.add_argument("--lambda-p", type=float, default=0.20)
    parser.add_argument("--lambda-d", type=float, default=0.10)
    parser.add_argument("--lambda-inh", type=float, default=0.50)
    parser.add_argument("--lambda-prom", type=float, default=0.30)
    parser.add_argument("--lambda-conflict", type=float, default=0.70)
    parser.add_argument("--lambda-merge", type=float, default=0.30)
    parser.add_argument("--lambda-boundary", type=float, default=0.00)
    parser.add_argument("--min-edge-cost", type=float, default=1e-6)
    parser.add_argument("--disable-interaction-refinement", action="store_true")
    parser.add_argument("--annotation-key", default=None, help="obs annotation column used for v7.3 metacell composition and biological process scopes")
    parser.add_argument("--input-level", choices=["auto", "cell", "metacell"], default="auto")
    parser.add_argument("--metacell-key", default=None, help="Existing obs grouping column to aggregate as metacells. Auto-detected from root IDs when omitted.")
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
    parser.add_argument("--allow-conditional-trajectory-edges", action="store_true", help="Allow conditional cross-process trajectory candidates; default is conservative interaction-only")
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
    parser.add_argument(
        "--sample-key",
        default=None,
        help="obs column with sample/slice labels. When set, spatial coordinates are only "
        "used within a sample; cross-sample edges are driven by expression only.",
    )
    parser.add_argument("--alpha-spatial", type=float, default=0.35, help="Spatial similarity exponent in S_global (product mode)")
    parser.add_argument("--beta-expr", type=float, default=0.35, help="Expression similarity exponent in S_global (product mode)")
    parser.add_argument(
        "--graph-mode",
        choices=["product", "step6"],
        default="product",
        help="product: gaussian similarity product (default). step6: D=lambda_s*spatial + lambda_x*(1-corr), then kNN.",
    )
    parser.add_argument("--lambda-s", type=float, default=1.0, help="Spatial distance weight in step6 graph mode")
    parser.add_argument("--lambda-x", type=float, default=1.0, help="Expression (1-corr) distance weight in step6 graph mode")
    parser.add_argument(
        "--skip-spatial-figures",
        action="store_true",
        help="Skip the physical-coordinate figures (01-04). Recommended for multi-sample data "
        "where pooled coordinates overlap; use plot_metacell_trees_layout.py instead.",
    )
    args = parser.parse_args()

    cfg = Config(
        k_neighbors=args.k_neighbors,
        theta_A=args.theta_a,
        beta_transition=args.beta_transition,
        theta_entropy=args.theta_entropy,
        theta_margin=args.theta_margin,
        theta_secondary=args.theta_secondary,
        theta_low_membership=args.theta_low_membership,
        theta_collision=args.theta_collision,
        theta_promotion=args.theta_promotion,
        theta_inhibition=args.theta_inhibition,
        theta_conv=args.theta_conv,
        theta_conflict=args.theta_conflict,
        lambda_A=args.lambda_a,
        lambda_P=args.lambda_p,
        lambda_D=args.lambda_d,
        lambda_inh=args.lambda_inh,
        lambda_prom=args.lambda_prom,
        lambda_conflict=args.lambda_conflict,
        lambda_merge=args.lambda_merge,
        lambda_boundary=args.lambda_boundary,
        min_edge_cost=args.min_edge_cost,
        disable_interaction_refinement=True,
        alpha_spatial=args.alpha_spatial,
        beta_expr=args.beta_expr,
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
    )
    roots_df = load_root_table_any_format(args.roots_csv)
    outdir = ensure_dir(Path(args.outdir))
    summaries = []
    failures = []
    for ds_name, adata_path in paths_from_args(args):
        try:
            if not adata_path.exists():
                raise FileNotFoundError(str(adata_path))
            summary = run_one_dataset(
                adata_path=adata_path,
                roots_df=roots_df,
                base_outdir=outdir,
                cfg=cfg,
                dataset_name=args.dataset_name or ds_name,
                dataset_col=args.dataset_col,
                root_col=args.root_col,
                root_specific_nmf_dir=Path(args.root_specific_nmf_dir) if args.root_specific_nmf_dir else None,
                sample_key=args.sample_key,
                graph_mode=args.graph_mode,
                lambda_s=args.lambda_s,
                lambda_x=args.lambda_x,
                skip_spatial_figures=args.skip_spatial_figures,
            )
            summaries.append(summary)
            print(f"[DONE] {summary['dataset']}: roots={summary['n_roots']} tree_edges={summary['tree_edge_count']}")
        except Exception as exc:
            fail_ds = ds_name or dataset_name_from_path(adata_path)
            failures.append(
                {
                    "dataset": fail_ds,
                    "adata_path": str(adata_path),
                    "error_message": str(exc),
                    "traceback_short": traceback.format_exc(limit=6),
                }
            )
            print(f"[FAIL] {fail_ds}: {exc}")
    if summaries:
        pd.DataFrame(summaries).drop(columns=["config", "quality_checks", "fallbacks"], errors="ignore").to_csv(
            outdir / "run_summary_table.csv", index=False
        )
    if failures:
        pd.DataFrame(failures).to_csv(outdir / "run_failures.csv", index=False)
    elif (outdir / "run_failures.csv").exists():
        (outdir / "run_failures.csv").unlink()
    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
