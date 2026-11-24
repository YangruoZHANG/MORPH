#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CASCAT-MR-CMI-E-Gate v7.3 metacell construction and strict root projection.

The v7.3 contract is deliberately explicit:

1. Never interpret raw cells as trajectory-tree nodes unless ``--input-level metacell``
   is supplied or the input is objectively detected as metacell-level.
2. Prefer an existing cluster/metacell column that actually contains the supplied root IDs.
3. Otherwise construct TME-aware metacells with adaptive BIRCH clustering inside broad
   biological compartments.
4. Project roots to metacells, never to arbitrary raw cells. Exact cluster/metacell matching
   is preferred; spatial projection is only a metacell-level fallback and is audited.
5. Use root_type plus metacell composition to define root process profiles.

This module intentionally works with broad process compartments. It is not a replacement for
expert cell-type annotation and does not claim experimental lineage truth.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.cluster import Birch, MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import anndata as ad
except Exception:  # pragma: no cover
    ad = None


COMPARTMENTS = [
    "tumor_epithelial",
    "lymphoid",
    "myeloid",
    "endothelial_vascular",
    "stromal_fibroblast",
    "perivascular_smooth_muscle",
]

_ANNOTATION_CANDIDATES = [
    "v7_label", "v6.3_label", "v6_label", "final_label", "major_cell_type",
    "cell_type", "celltype", "annotation", "anno", "label", "labels",
    "subtype", "state", "cluster_label", "cell_state",
]

_SAMPLE_CANDIDATES = ["sample", "sample_id", "batch", "slice", "section", "library_id", "library"]

_MARKERS: dict[str, list[str]] = {
    "tumor_epithelial": [
        "EPCAM", "TACSTD2", "KRT7", "KRT8", "KRT18", "KRT19", "KRT14",
        "KRT17", "MSLN", "MUC1", "KRT5",
    ],
    "lymphoid": [
        "CD3D", "CD3E", "CD247", "TRBC1", "TRBC2", "CD4", "CD8A", "CD8B",
        "NKG7", "GNLY", "KLRD1", "IL7R",
    ],
    "myeloid": [
        "LST1", "TYROBP", "FCER1G", "CTSS", "AIF1", "CD68", "LILRB1",
        "S100A8", "S100A9", "FCGR3A", "CSF1R",
    ],
    "endothelial_vascular": [
        "PECAM1", "VWF", "EMCN", "KDR", "ENG", "RAMP2", "PLVAP", "FLT1",
        "LYVE1", "FLT4", "PDPN",
    ],
    "stromal_fibroblast": [
        "COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "COL6A1", "FAP",
        "PDGFRA", "POSTN", "THY1",
    ],
    "perivascular_smooth_muscle": [
        "RGS5", "CSPG4", "MCAM", "ACTA2", "TAGLN", "MYH11", "PDGFRB", "DES",
    ],
}

_LABEL_PATTERNS: list[tuple[str, list[str]]] = [
    ("tumor_epithelial", ["proliferative tumor", "cycling tumor", "tumor", "malignant", "cancer", "carcinoma", "epithelial"]),
    ("lymphoid", ["t/nk", "tnk", "regulatory t", "treg", "cd4", "cd8", "t cell", "t-cell", "nk", "lymphocyte", "lymphoid"]),
    ("myeloid", ["macrophage", "macro", "monocyte", "myeloid", "dendritic", "neutrophil", "mast"]),
    ("endothelial_vascular", ["lymphatic endothelial", "endothelial", "panvascular", "vascular", "angiogenic"]),
    ("stromal_fibroblast", ["fibroblast", "myofibroblast", "caf", "stromal", "matrix"]),
    ("perivascular_smooth_muscle", ["pericyte", "smooth muscle", "mural", "contractile"]),
]

_ROOT_TYPE_PATTERNS = {
    "tumor_epithelial": ["epithelial", "tumor", "malignant", "cancer", "carcinoma"],
    "lymphoid": ["t_cell", "t cell", "t-cell", "t/nk", "tnk", "lymph", "nk"],
    "myeloid": ["myeloid", "macro", "macrophage", "monocyte"],
    "endothelial_vascular": ["endothelial", "vascular", "angiogenic"],
    "stromal_fibroblast": ["fibroblast", "stromal", "caf"],
    "perivascular_smooth_muscle": ["pericyte", "smooth muscle", "mural"],
}


@dataclass
class MetacellBuildResult:
    adata: Any
    cell_to_metacell: pd.DataFrame
    metacell_summary: pd.DataFrame
    audit: pd.DataFrame
    source_mode: str
    source_group_key: str


def _cfg(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def infer_col(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    cols = list(columns)
    lower = {str(c).lower(): str(c) for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def find_annotation_column(adata, preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in adata.obs.columns:
        return preferred
    return infer_col(adata.obs.columns, _ANNOTATION_CANDIDATES)


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
    raise ValueError("No spatial coordinates found in obsm['spatial'] or x/y-like columns")


def infer_major_compartment_from_label(label: object) -> str:
    text = re.sub(r"[_\-]+", " ", str(label).strip().lower())
    if text in {"", "nan", "none", "background", "unknown", "unassigned", "ambiguous"}:
        return "unknown"
    for compartment, patterns in _LABEL_PATTERNS:
        if any(p in text for p in patterns):
            return compartment
    return "unknown"


def parse_root_type_compartments(root_type: object) -> list[str]:
    text = re.sub(r"[-_/]", " ", str(root_type).strip().lower())
    text = text.replace("+", " + ")
    if text in {"", "nan", "none", "unknown", "unassigned", "ambiguous", "na", "n a"}:
        return []
    found: list[str] = []
    for compartment, patterns in _ROOT_TYPE_PATTERNS.items():
        if any(p in text for p in patterns):
            found.append(compartment)
    return sorted(set(found))


def process_scope_from_compartments(compartments: Iterable[str]) -> str:
    comps = sorted({str(c) for c in compartments if str(c) in COMPARTMENTS})
    if not comps:
        return "unknown_process"
    if len(comps) == 1:
        return f"{comps[0]}_process"
    return "+".join(comps) + "_interface_process"


def _expression_embedding(adata, n_components: int, random_state: int) -> np.ndarray:
    if "X_pca" in adata.obsm:
        Xp = np.asarray(adata.obsm["X_pca"], dtype=float)
        return Xp[:, : min(n_components, Xp.shape[1])]
    X = adata.X
    max_comp = max(2, min(int(n_components), adata.n_obs - 1, adata.n_vars - 1))
    if sparse.issparse(X):
        return TruncatedSVD(n_components=max_comp, random_state=random_state).fit_transform(X)
    Xd = np.asarray(X, dtype=float)
    Xz = StandardScaler().fit_transform(Xd)
    return PCA(n_components=max_comp, random_state=random_state).fit_transform(Xz)


def _marker_scores(adata) -> tuple[np.ndarray, dict[str, int]]:
    n = adata.n_obs
    scores = np.zeros((n, len(COMPARTMENTS)), dtype=float)
    lookup = {str(g).upper(): i for i, g in enumerate(adata.var_names)}
    hits: dict[str, int] = {}
    X = adata.X
    for j, compartment in enumerate(COMPARTMENTS):
        idx = [lookup[g.upper()] for g in _MARKERS.get(compartment, []) if g.upper() in lookup]
        hits[compartment] = len(idx)
        if not idx:
            continue
        block = X[:, idx]
        vals = np.asarray(block.mean(axis=1)).ravel() if sparse.issparse(block) else np.nanmean(np.asarray(block), axis=1)
        scores[:, j] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    # Standardise each signature across cells to compare marker enrichment.
    for j in range(scores.shape[1]):
        col = scores[:, j]
        med = float(np.nanmedian(col))
        mad = float(np.nanmedian(np.abs(col - med)))
        scale = 1.4826 * mad
        if scale <= 1e-8:
            scale = float(np.nanstd(col))
        if scale <= 1e-8:
            scores[:, j] = 0.0
        else:
            scores[:, j] = (col - med) / scale
    return scores, hits


def infer_cell_compartments(adata, annotation_key: Optional[str] = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    key = find_annotation_column(adata, annotation_key)
    labels = adata.obs[key].astype(str).to_numpy() if key else np.asarray(["unknown"] * adata.n_obs, dtype=object)
    label_comp = np.asarray([infer_major_compartment_from_label(x) for x in labels], dtype=object)
    marker_scores, marker_hits = _marker_scores(adata)
    top = np.argmax(marker_scores, axis=1)
    sorted_scores = np.sort(marker_scores, axis=1)
    top_score = sorted_scores[:, -1]
    second = sorted_scores[:, -2] if marker_scores.shape[1] >= 2 else np.zeros(adata.n_obs)
    margin = top_score - second
    marker_comp = np.asarray([COMPARTMENTS[int(i)] for i in top], dtype=object)

    marker_min_z = 0.50
    marker_min_margin = 0.15
    inferred = label_comp.copy()
    source = np.asarray(["annotation"] * adata.n_obs, dtype=object)
    unknown = label_comp == "unknown"
    confident = unknown & (top_score >= marker_min_z) & (margin >= marker_min_margin)
    inferred[confident] = marker_comp[confident]
    source[confident] = "marker_fallback"
    source[unknown & ~confident] = "unresolved_background"

    out = pd.DataFrame({
        "cell_index": np.arange(adata.n_obs, dtype=int),
        "cell_name": adata.obs_names.astype(str),
        "annotation_key": key or "",
        "annotation_label": labels,
        "annotation_compartment": label_comp,
        "inferred_compartment": inferred,
        "compartment_source": source,
        "marker_top_compartment": marker_comp,
        "marker_top_score": top_score,
        "marker_margin": margin,
    })
    for j, comp in enumerate(COMPARTMENTS):
        out[f"marker_score_{comp}"] = marker_scores[:, j]
    info = {
        "annotation_key": key or "",
        "marker_hits": marker_hits,
        "n_annotation_resolved": int((label_comp != "unknown").sum()),
        "n_marker_resolved": int(confident.sum()),
        "n_unresolved": int((inferred == "unknown").sum()),
    }
    return out, info


def _filter_roots_df(roots_df: pd.DataFrame, dataset_name: str, dataset_col: Optional[str]) -> pd.DataFrame:
    df = roots_df.copy()
    dcol = dataset_col if dataset_col and dataset_col in df.columns else infer_col(df.columns, ["dataset", "dataset_name", "sample", "adata"])
    if dcol:
        df = df.loc[df[dcol].astype(str) == str(dataset_name)].copy()
    if df.empty:
        raise ValueError(f"No roots found for dataset={dataset_name}")
    return df


def choose_root_col(roots_df: pd.DataFrame, root_col: Optional[str]) -> str:
    if root_col and root_col in roots_df.columns:
        return root_col
    inferred = infer_col(roots_df.columns, ["root_id", "root", "root_node", "mapped_root_nodes", "node", "node_id", "cluster_id"])
    if inferred is None:
        raise ValueError(f"Cannot infer root column from columns={list(roots_df.columns)}")
    return inferred


def detect_existing_group_key(
    adata,
    root_ids: list[str],
    preferred: Optional[str] = None,
) -> tuple[Optional[str], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    if preferred:
        if preferred not in adata.obs.columns:
            raise ValueError(f"Requested metacell key {preferred!r} is absent from adata.obs")
        return preferred, pd.DataFrame([{"column": preferred, "root_matches": len(root_ids), "preferred": True}])

    priority_tokens = ["metacell", "meta_cell", "cluster", "niche", "domain", "state", "community", "leiden", "louvain"]
    root_set = {str(x) for x in root_ids}
    n = adata.n_obs
    for col in adata.obs.columns:
        s = adata.obs[col]
        try:
            vals = s.astype(str)
        except Exception:
            continue
        nunique = int(vals.nunique(dropna=True))
        if nunique < 2 or nunique >= max(5000, n):
            continue
        value_set = set(vals.unique().tolist())
        matches = len(root_set & value_set)
        name = str(col).lower()
        priority = sum(1 for token in priority_tokens if token in name)
        rows.append({
            "column": str(col),
            "n_unique": nunique,
            "root_matches": matches,
            "all_roots_matched": matches == len(root_set) and len(root_set) > 0,
            "name_priority": priority,
        })
    audit = pd.DataFrame(rows)
    if audit.empty:
        return None, audit
    audit = audit.sort_values(["all_roots_matched", "root_matches", "name_priority", "n_unique"], ascending=[False, False, False, True])
    best = audit.iloc[0]
    # Only trust an existing grouping when it contains every supplied root ID.
    if bool(best["all_roots_matched"]):
        return str(best["column"]), audit
    return None, audit


def _adaptive_birch_labels(features: np.ndarray, cfg: Any, random_state: int) -> np.ndarray:
    n = len(features)
    if n <= 1:
        return np.zeros(n, dtype=int)
    target = max(2, int(_cfg(cfg, "metacell_target_size", 50)))
    max_size = max(target, int(_cfg(cfg, "metacell_max_size", 150)))
    multiplier = float(_cfg(cfg, "metacell_birch_threshold_multiplier", 1.5))
    k = min(5, n)
    nn = NearestNeighbors(n_neighbors=k).fit(features)
    d, _ = nn.kneighbors(features)
    kth = d[:, -1]
    positive = kth[np.isfinite(kth) & (kth > 0)]
    base = float(np.median(positive)) if len(positive) else 1.0
    threshold = max(base * multiplier, 1e-3)
    labels = Birch(threshold=threshold, n_clusters=None, branching_factor=50).fit_predict(features)

    # Safety split only for pathologically large microclusters.
    next_label = int(labels.max()) + 1 if len(labels) else 0
    result = labels.copy()
    for lab in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == lab)
        if len(idx) <= max_size:
            continue
        ksplit = int(math.ceil(len(idx) / max_size))
        km = MiniBatchKMeans(n_clusters=ksplit, random_state=random_state, n_init=5, batch_size=min(1024, len(idx)))
        sub = km.fit_predict(features[idx])
        for s in sorted(set(sub.tolist())):
            members = idx[sub == s]
            result[members] = next_label
            next_label += 1
    # Relabel contiguously.
    uniq = {lab: i for i, lab in enumerate(sorted(set(result.tolist())))}
    return np.asarray([uniq[int(x)] for x in result], dtype=int)


def _construct_group_labels(
    adata,
    cell_domains: pd.DataFrame,
    cfg: Any,
    sample_key: Optional[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    emb = _expression_embedding(adata, int(_cfg(cfg, "n_pca", 30)), int(_cfg(cfg, "random_state", 7)))
    coords = get_spatial_coords(adata)
    emb_z = StandardScaler().fit_transform(emb)
    coords_z = StandardScaler().fit_transform(coords)
    spatial_weight = float(_cfg(cfg, "metacell_spatial_weight", 0.35))
    feat = np.hstack([emb_z, spatial_weight * coords_z])

    if sample_key and sample_key in adata.obs.columns:
        samples = adata.obs[sample_key].astype(str).to_numpy()
    else:
        samples = np.asarray(["__single_sample__"] * adata.n_obs, dtype=object)
    compartments = cell_domains["inferred_compartment"].astype(str).to_numpy()

    group_keys = np.asarray([f"{s}||{c}" for s, c in zip(samples, compartments)], dtype=object)
    labels = np.empty(adata.n_obs, dtype=object)
    audit_rows: list[dict[str, Any]] = []
    global_counter = 0
    min_size = max(1, int(_cfg(cfg, "metacell_min_size", 10)))
    for key in sorted(set(group_keys.tolist())):
        idx = np.flatnonzero(group_keys == key)
        if len(idx) <= min_size:
            sublabels = np.zeros(len(idx), dtype=int)
        else:
            sublabels = _adaptive_birch_labels(feat[idx], cfg, int(_cfg(cfg, "random_state", 7)))
        for local_lab in sorted(set(sublabels.tolist())):
            members = idx[sublabels == local_lab]
            mc_id = f"MC{global_counter:04d}"
            labels[members] = mc_id
            audit_rows.append({
                "construction_group": key,
                "metacell_id": mc_id,
                "n_cells": int(len(members)),
                "source": "adaptive_birch",
            })
            global_counter += 1
    return labels.astype(str), pd.DataFrame(audit_rows)



def _merge_tiny_metacells(
    group_labels: np.ndarray,
    adata,
    cell_domains: pd.DataFrame,
    cfg: Any,
    sample_key: Optional[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Merge tiny adaptive metacells into local composition-compatible neighbours.

    The implementation is group-level and kNN-accelerated.  It does not scan all
    cells for every candidate pair, so it remains practical for tens of thousands
    of raw cells and a few thousand preliminary metacells.
    """
    labels = np.asarray(group_labels, dtype=object).copy()
    tiny_size = max(1, int(_cfg(cfg, "metacell_tiny_merge_size", 5)))
    if tiny_size <= 1 or len(labels) == 0:
        return labels.astype(str), pd.DataFrame()

    max_size = max(tiny_size, int(_cfg(cfg, "metacell_max_size", 150)))
    min_comp_sim = float(_cfg(cfg, "tiny_merge_min_comp_similarity", 0.55))
    spatial_factor = float(_cfg(cfg, "tiny_merge_max_spatial_nn_factor", 4.0))
    spatial_weight = float(_cfg(cfg, "metacell_spatial_weight", 0.35))
    random_state = int(_cfg(cfg, "random_state", 7))
    neighbour_pool = max(8, int(_cfg(cfg, "tiny_merge_candidate_neighbors", 32)))

    emb = _expression_embedding(adata, int(_cfg(cfg, "n_pca", 30)), random_state)
    coords = get_spatial_coords(adata)
    emb_z = StandardScaler().fit_transform(emb)
    coords_z = StandardScaler().fit_transform(coords)
    if sample_key and sample_key in adata.obs.columns:
        samples = adata.obs[sample_key].astype(str).to_numpy()
    else:
        samples = np.asarray(["__single_sample__"] * adata.n_obs, dtype=object)
    comps = cell_domains["inferred_compartment"].astype(str).to_numpy()

    audit_rows: list[dict[str, Any]] = []

    for pass_id in range(2):
        unique = pd.Index(pd.unique(labels.astype(str)))
        if len(unique) <= 1:
            break
        lookup = {g: i for i, g in enumerate(unique)}
        codes = np.asarray([lookup[str(g)] for g in labels], dtype=int)
        g = len(unique)
        counts = np.bincount(codes, minlength=g).astype(int)
        tiny_ids = np.flatnonzero(counts < tiny_size)
        if len(tiny_ids) == 0:
            break

        # Group means via sparse aggregation matrix.
        weights = 1.0 / np.maximum(counts[codes], 1)
        G = csr_matrix((weights, (codes, np.arange(len(codes)))), shape=(g, len(codes)))
        emb_cent = np.asarray(G @ emb_z, dtype=float)
        xy_cent = np.asarray(G @ coords_z, dtype=float)

        sample_group = np.empty(g, dtype=object)
        frac = np.zeros((g, len(COMPARTMENTS)), dtype=float)
        dominant = np.empty(g, dtype=object)
        for gi in range(g):
            idx = np.flatnonzero(codes == gi)
            sample_group[gi] = str(samples[idx[0]]) if len(idx) else ""
            for cj, comp in enumerate(COMPARTMENTS):
                frac[gi, cj] = float(np.mean(comps[idx] == comp)) if len(idx) else 0.0
            dominant[gi] = COMPARTMENTS[int(np.argmax(frac[gi]))] if frac[gi].max(initial=0.0) > 0 else "unknown"

        # Spatial nearest-neighbour scale among preliminary metacells.
        xy_nn = NearestNeighbors(n_neighbors=min(2, g)).fit(xy_cent)
        dxy, _ = xy_nn.kneighbors(xy_cent)
        positive = dxy[:, -1][dxy[:, -1] > 0]
        median_nn = float(np.median(positive)) if len(positive) else 1.0

        feat = np.hstack([emb_cent, spatial_weight * xy_cent])
        nn = NearestNeighbors(n_neighbors=min(neighbour_pool, g)).fit(feat)
        _, neigh = nn.kneighbors(feat)

        parent = np.arange(g, dtype=int)
        alive = np.ones(g, dtype=bool)
        dyn_count = counts.astype(int).copy()
        dyn_emb = emb_cent.copy()
        dyn_xy = xy_cent.copy()
        dyn_frac = frac.copy()

        def root_of(x: int) -> int:
            y = int(x)
            while parent[y] != y:
                parent[y] = parent[parent[y]]
                y = int(parent[y])
            return y

        changed = False
        order = sorted(tiny_ids.tolist(), key=lambda gi: (int(counts[gi]), str(unique[gi])))
        for src0 in order:
            src = root_of(int(src0))
            if not alive[src] or dyn_count[src] >= tiny_size:
                continue

            # Start with local feature neighbours.  If none survive, use a vectorized
            # all-group fallback rather than a Python all-pairs scan.
            candidate_ids = [root_of(int(x)) for x in neigh[src] if int(x) != src]
            candidate_ids = list(dict.fromkeys(candidate_ids))
            if not candidate_ids:
                candidate_ids = [i for i in range(g) if i != src]

            def valid_candidates(ids: list[int]):
                out = []
                for tgt0 in ids:
                    tgt = root_of(int(tgt0))
                    if tgt == src or not alive[tgt]:
                        continue
                    if sample_group[tgt] != sample_group[src]:
                        continue
                    if dyn_count[tgt] + dyn_count[src] > max_size:
                        continue
                    if dyn_count[tgt] < tiny_size:
                        if dyn_count[tgt] < dyn_count[src]:
                            continue
                        if dyn_count[tgt] == dyn_count[src] and str(unique[tgt]) > str(unique[src]):
                            continue
                    a, b = dyn_frac[src], dyn_frac[tgt]
                    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                    comp_sim = float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0
                    same_dom = dominant[src] == dominant[tgt]
                    if not same_dom and comp_sim < min_comp_sim:
                        continue
                    d_sp = float(np.linalg.norm(dyn_xy[src] - dyn_xy[tgt]))
                    if d_sp > spatial_factor * max(median_nn, 1e-9):
                        continue
                    d_ex = float(np.linalg.norm(dyn_emb[src] - dyn_emb[tgt])) / math.sqrt(max(dyn_emb.shape[1], 1))
                    score = d_ex + spatial_weight * d_sp + 0.75 * (1.0 - comp_sim)
                    out.append((score, -int(dyn_count[tgt]), int(tgt), comp_sim, d_sp, d_ex))
                return out

            candidates = valid_candidates(candidate_ids)
            if not candidates and len(candidate_ids) < g - 1:
                candidates = valid_candidates([i for i in range(g) if i != src])
            if not candidates:
                continue

            candidates.sort(key=lambda x: (x[0], x[1], str(unique[x[2]])))
            score, _, tgt, comp_sim, d_sp, d_ex = candidates[0]
            src_n, tgt_n = int(dyn_count[src]), int(dyn_count[tgt])
            new_n = src_n + tgt_n
            dyn_emb[tgt] = (src_n * dyn_emb[src] + tgt_n * dyn_emb[tgt]) / max(new_n, 1)
            dyn_xy[tgt] = (src_n * dyn_xy[src] + tgt_n * dyn_xy[tgt]) / max(new_n, 1)
            dyn_frac[tgt] = (src_n * dyn_frac[src] + tgt_n * dyn_frac[tgt]) / max(new_n, 1)
            dyn_count[tgt] = new_n
            dominant[tgt] = COMPARTMENTS[int(np.argmax(dyn_frac[tgt]))] if dyn_frac[tgt].max(initial=0.0) > 0 else "unknown"
            parent[src] = tgt
            alive[src] = False
            changed = True
            audit_rows.append({
                "pass_id": int(pass_id),
                "source_group": str(unique[src]),
                "target_group": str(unique[tgt]),
                "source_n_cells": src_n,
                "target_n_cells_before": tgt_n,
                "composition_similarity": float(comp_sim),
                "spatial_distance_z": float(d_sp),
                "expression_distance_scaled": float(d_ex),
                "merge_score": float(score),
                "reason": "tiny_metacell_compatible_local_merge",
            })

        if not changed:
            break
        final_group = np.asarray([root_of(int(code)) for code in codes], dtype=int)
        final_ids = pd.unique(final_group)
        relabel = {int(old): f"MC{i:04d}" for i, old in enumerate(final_ids)}
        labels = np.asarray([relabel[int(x)] for x in final_group], dtype=str)

    return labels.astype(str), pd.DataFrame(audit_rows)


def _group_mean_matrix(X, group_codes: np.ndarray, n_groups: int):
    counts = np.bincount(group_codes, minlength=n_groups).astype(float)
    rows = group_codes
    cols = np.arange(len(group_codes), dtype=int)
    vals = 1.0 / np.maximum(counts[group_codes], 1.0)
    G = csr_matrix((vals, (rows, cols)), shape=(n_groups, len(group_codes)))
    return G @ X, G, counts


def build_or_use_metacells(
    adata_raw,
    roots_df: pd.DataFrame,
    dataset_name: str,
    cfg: Any,
    dataset_col: Optional[str] = None,
    root_col: Optional[str] = None,
    sample_key: Optional[str] = None,
) -> MetacellBuildResult:
    if ad is None:
        raise ImportError("anndata is required for v7.3 metacell construction")
    roots_sub = _filter_roots_df(roots_df, dataset_name, dataset_col)
    rcol = choose_root_col(roots_sub, root_col)
    root_ids = roots_sub[rcol].astype(str).tolist()
    input_level = str(_cfg(cfg, "input_level", "auto")).lower()
    preferred_key = _cfg(cfg, "metacell_key", None)

    cell_domains, domain_info = infer_cell_compartments(adata_raw, _cfg(cfg, "annotation_key", None))
    existing_key, group_key_audit = detect_existing_group_key(adata_raw, root_ids, preferred_key)
    roots_match_obs_names = set(map(str, root_ids)).issubset(set(adata_raw.obs_names.astype(str)))

    if input_level == "metacell" or (input_level == "auto" and roots_match_obs_names):
        group_labels = adata_raw.obs_names.astype(str).to_numpy()
        source_mode = "input_declared_metacell" if input_level == "metacell" else "input_root_ids_match_obs_names_metacell"
        source_group_key = "obs_names"
        construction_audit = pd.DataFrame({"metacell_id": group_labels, "n_cells": 1, "source": source_mode})
    elif existing_key is not None:
        group_labels = adata_raw.obs[existing_key].astype(str).to_numpy()
        source_mode = "existing_root_compatible_grouping"
        source_group_key = existing_key
        construction_audit = pd.DataFrame()
    else:
        group_labels, construction_audit = _construct_group_labels(adata_raw, cell_domains, cfg, sample_key)
        group_labels, tiny_merge_audit = _merge_tiny_metacells(group_labels, adata_raw, cell_domains, cfg, sample_key)
        if tiny_merge_audit is not None and not tiny_merge_audit.empty:
            tiny_merge_audit = tiny_merge_audit.assign(source="tiny_metacell_postmerge")
            construction_audit = pd.concat([construction_audit, tiny_merge_audit], ignore_index=True, sort=False)
        source_mode = "adaptive_tme_metacell_construction_with_tiny_postmerge"
        source_group_key = "adaptive_birch"

    # If a group key repeats across explicit samples, preserve sample identity.
    if sample_key and sample_key in adata_raw.obs.columns and source_mode == "existing_root_compatible_grouping":
        samples = adata_raw.obs[sample_key].astype(str).to_numpy()
        group_labels = np.asarray([f"{s}::{g}" for s, g in zip(samples, group_labels)], dtype=object)

    unique_groups = pd.Index(pd.unique(group_labels.astype(str)))
    group_to_code = {g: i for i, g in enumerate(unique_groups)}
    codes = np.asarray([group_to_code[str(g)] for g in group_labels], dtype=int)
    n_groups = len(unique_groups)
    X_meta, G, counts = _group_mean_matrix(adata_raw.X, codes, n_groups)
    coords_raw = get_spatial_coords(adata_raw)
    coords_meta = np.asarray(G @ coords_raw, dtype=float)

    # Aggregate any available PCA embedding so downstream code does not re-PCA large matrices.
    obsm: dict[str, np.ndarray] = {"spatial": coords_meta}
    if "X_pca" in adata_raw.obsm:
        obsm["X_pca"] = np.asarray(G @ np.asarray(adata_raw.obsm["X_pca"], dtype=float), dtype=float)

    annotation_key = find_annotation_column(adata_raw, _cfg(cfg, "annotation_key", None))
    raw_labels = adata_raw.obs[annotation_key].astype(str).to_numpy() if annotation_key else np.asarray(["unknown"] * adata_raw.n_obs, dtype=object)
    inferred = cell_domains["inferred_compartment"].astype(str).to_numpy()

    obs_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for code, group_name in enumerate(unique_groups.astype(str)):
        idx = np.flatnonzero(codes == code)
        label_counts = pd.Series(raw_labels[idx]).value_counts(dropna=False)
        dominant_label = str(label_counts.index[0]) if len(label_counts) else "unknown"
        annotation_purity = float(label_counts.iloc[0] / len(idx)) if len(idx) else 0.0
        frac: dict[str, float] = {}
        for comp in COMPARTMENTS:
            frac[comp] = float(np.mean(inferred[idx] == comp)) if len(idx) else 0.0
        unresolved_fraction = float(np.mean(inferred[idx] == "unknown")) if len(idx) else 1.0
        dominant_comp = max(frac, key=frac.get) if frac and max(frac.values()) > 0 else "unknown"
        comp_purity = max(frac.values()) if frac else 0.0
        active_comps = [c for c, v in frac.items() if v >= 0.20]
        process_scope = process_scope_from_compartments(active_comps or ([dominant_comp] if dominant_comp != "unknown" else []))
        pvals = np.asarray([v for v in frac.values() if v > 0], dtype=float)
        entropy = float(-np.sum(pvals * np.log(pvals + 1e-12)) / math.log(len(COMPARTMENTS))) if len(pvals) > 1 else 0.0

        # Existing source cluster ID is kept separately when samples were prefixed.
        source_cluster_id = group_name.split("::", 1)[1] if "::" in group_name else group_name
        obs_row: dict[str, Any] = {
            "metacell_id": group_name,
            "source_cluster_id": source_cluster_id,
            "source_group_key": source_group_key,
            "n_cells": int(len(idx)),
            "annotation_label": dominant_label,
            "annotation_purity": annotation_purity,
            "major_compartment": dominant_comp,
            "compartment_purity": comp_purity,
            "compartment_entropy": entropy,
            "unresolved_cell_fraction": unresolved_fraction,
            "process_scope": process_scope,
            "spatial_x": float(coords_meta[code, 0]),
            "spatial_y": float(coords_meta[code, 1]),
        }
        for comp in COMPARTMENTS:
            obs_row[f"frac_{comp}"] = frac[comp]
        obs_rows.append(obs_row)
        for cell_idx in idx:
            mapping_rows.append({
                "cell_index": int(cell_idx),
                "cell_name": str(adata_raw.obs_names[cell_idx]),
                "metacell_index": int(code),
                "metacell_id": group_name,
                "source_group_key": source_group_key,
                "cell_annotation_label": str(raw_labels[cell_idx]),
                "cell_inferred_compartment": str(inferred[cell_idx]),
            })

    obs = pd.DataFrame(obs_rows).set_index("metacell_id", drop=False)
    meta = ad.AnnData(X=X_meta, obs=obs, var=adata_raw.var.copy())
    for key, value in obsm.items():
        meta.obsm[key] = value
    meta.uns["cascat_metacell_build"] = {
        "source_mode": source_mode,
        "source_group_key": source_group_key,
        "n_raw_cells": int(adata_raw.n_obs),
        "n_metacells": int(meta.n_obs),
        "domain_inference": domain_info,
    }

    audit_rows = [
        {"metric": "source_mode", "value": source_mode},
        {"metric": "source_group_key", "value": source_group_key},
        {"metric": "n_raw_cells", "value": int(adata_raw.n_obs)},
        {"metric": "n_metacells", "value": int(meta.n_obs)},
        {"metric": "median_cells_per_metacell", "value": float(np.median(counts)) if len(counts) else 0.0},
        {"metric": "max_cells_per_metacell", "value": int(np.max(counts)) if len(counts) else 0},
        {"metric": "singleton_metacell_fraction", "value": float(np.mean(counts == 1)) if len(counts) else 0.0},
        {"metric": "tiny_metacell_fraction_lt_merge_size", "value": float(np.mean(counts < max(1, int(_cfg(cfg, 'metacell_tiny_merge_size', 5))))) if len(counts) else 0.0},
        {"metric": "root_compatible_group_key_detected", "value": bool(existing_key is not None)},
        {"metric": "annotation_key", "value": domain_info.get("annotation_key", "")},
        {"metric": "marker_resolved_cells", "value": int(domain_info.get("n_marker_resolved", 0))},
        {"metric": "unresolved_cells", "value": int(domain_info.get("n_unresolved", 0))},
    ]
    if not group_key_audit.empty:
        best = group_key_audit.iloc[0].to_dict()
        audit_rows.append({"metric": "best_group_key_candidate", "value": json.dumps(best, ensure_ascii=False)})
    if not construction_audit.empty:
        audit_rows.append({"metric": "adaptive_construction_groups", "value": int(len(construction_audit))})

    return MetacellBuildResult(
        adata=meta,
        cell_to_metacell=pd.DataFrame(mapping_rows),
        metacell_summary=obs.reset_index(drop=True),
        audit=pd.DataFrame(audit_rows),
        source_mode=source_mode,
        source_group_key=source_group_key,
    )


def _metacell_type_compatibility(row: pd.Series, allowed: list[str]) -> float:
    if not allowed:
        return 0.0
    vals = [float(row.get(f"frac_{c}", 0.0)) for c in allowed]
    score = float(sum(vals))
    if score <= 0 and str(row.get("major_compartment", "unknown")) in allowed:
        score = 1.0
    return float(np.clip(score, 0.0, 1.0))


def resolve_roots_to_metacells(
    adata_meta,
    roots_df: pd.DataFrame,
    dataset_name: str,
    dataset_col: Optional[str] = None,
    root_col: Optional[str] = None,
    cfg: Any = None,
) -> tuple[list[int], pd.DataFrame]:
    df = _filter_roots_df(roots_df, dataset_name, dataset_col)
    rcol = choose_root_col(df, root_col)
    root_type_col = infer_col(df.columns, ["root_type", "root_label", "type", "process", "annotation"])
    coords = get_spatial_coords(adata_meta)
    obs_names = pd.Index(adata_meta.obs_names.astype(str))
    meta_obs = adata_meta.obs.reset_index(drop=True).copy()
    roots: list[int] = []
    rows: list[dict[str, Any]] = []
    top_k = max(1, int(_cfg(cfg, "root_projection_top_k", 25))) if cfg is not None else 25
    type_weight = float(_cfg(cfg, "root_projection_type_weight", 0.35)) if cfg is not None else 0.35

    for _, row in df.iterrows():
        value = str(row[rcol])
        root_type = str(row[root_type_col]) if root_type_col and pd.notna(row[root_type_col]) else ""
        allowed = parse_root_type_compartments(root_type)
        # v7.4.3.2: an unrecognized ontology label means that type evidence is
        # unavailable, not that the root is biologically incompatible.  When no
        # broad compartment can be inferred, root projection is spatial-only and
        # the Truth-First layer receives an explicit availability flag.
        type_evidence_available = bool(allowed)
        effective_type_weight = type_weight if type_evidence_available else 0.0
        candidates: list[int] = []
        mode = ""
        if value in obs_names:
            candidates = [int(obs_names.get_loc(value))]
            mode = "metacell_obs_name_exact"
        elif "metacell_id" in adata_meta.obs.columns:
            hit = np.flatnonzero(adata_meta.obs["metacell_id"].astype(str).to_numpy() == value)
            if len(hit):
                candidates = list(map(int, hit))
                mode = "metacell_id_exact"
        if not candidates and "source_cluster_id" in adata_meta.obs.columns:
            hit = np.flatnonzero(adata_meta.obs["source_cluster_id"].astype(str).to_numpy() == value)
            if len(hit):
                candidates = list(map(int, hit))
                mode = "source_cluster_id_exact"

        target_xy = None
        if "x" in row.index and "y" in row.index and pd.notna(row["x"]) and pd.notna(row["y"]):
            target_xy = np.asarray([float(row["x"]), float(row["y"])], dtype=float)

        if candidates:
            if len(candidates) == 1:
                idx = candidates[0]
                dist = float(np.linalg.norm(coords[idx] - target_xy)) if target_xy is not None else 0.0
                type_score = _metacell_type_compatibility(meta_obs.iloc[idx], allowed)
            else:
                # Disambiguate repeated cluster IDs by metacell-level spatial/type evidence.
                dists = np.asarray([np.linalg.norm(coords[i] - target_xy) for i in candidates]) if target_xy is not None else np.zeros(len(candidates))
                scale = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0
                spatial_score = np.exp(-dists / (scale + 1e-9))
                type_scores = np.asarray([_metacell_type_compatibility(meta_obs.iloc[i], allowed) for i in candidates])
                combined = (1.0 - effective_type_weight) * spatial_score + effective_type_weight * type_scores
                best = int(np.argmax(combined))
                idx = candidates[best]
                dist = float(dists[best])
                type_score = float(type_scores[best])
                mode += "+metacell_disambiguation"
        else:
            if target_xy is None:
                raise ValueError(f"Root {value!r} cannot be mapped to a metacell: no exact ID and no x/y fallback")
            dists = np.linalg.norm(coords - target_xy[None, :], axis=1)
            nearest = np.argsort(dists)[: min(top_k, len(dists))]
            scale = float(np.median(dists[nearest][dists[nearest] > 0])) if np.any(dists[nearest] > 0) else 1.0
            spatial_score = np.exp(-dists[nearest] / (scale + 1e-9))
            type_scores = np.asarray([_metacell_type_compatibility(meta_obs.iloc[int(i)], allowed) for i in nearest])
            combined = (1.0 - effective_type_weight) * spatial_score + effective_type_weight * type_scores
            best = int(np.argmax(combined))
            idx = int(nearest[best])
            dist = float(dists[idx])
            type_score = float(type_scores[best])
            mode = "metacell_xy_root_type_projection"

        if idx in roots:
            raise ValueError(
                f"Two supplied roots resolve to the same metacell index={idx} ({adata_meta.obs_names[idx]}). "
                "This is rejected because distinct biological processes cannot share an identical root anchor."
            )
        roots.append(idx)
        rec = row.to_dict()
        rec.update({
            "root_col_used": rcol,
            "original_root_value": value,
            "root_type_used": root_type,
            "root_allowed_compartments": "|".join(allowed),
            "root_type_evidence_available": bool(type_evidence_available),
            "root_type_evidence_status": "recognized_compartment_evidence" if type_evidence_available else "unrecognized_type_evidence_unavailable",
            "root_projection_type_weight_effective": float(effective_type_weight),
            "resolved_index": int(idx),
            "resolved_obs_name": str(adata_meta.obs_names[idx]),
            "resolved_metacell_id": str(adata_meta.obs.iloc[idx].get("metacell_id", adata_meta.obs_names[idx])),
            "resolved_source_cluster_id": str(adata_meta.obs.iloc[idx].get("source_cluster_id", "")),
            "resolution_mode": mode,
            "projection_distance": float(dist),
            "root_type_compatibility": float(type_score),
            "resolved_x": float(coords[idx, 0]),
            "resolved_y": float(coords[idx, 1]),
        })
        rows.append(rec)

    return roots, pd.DataFrame(rows)


def build_root_process_profiles(adata_meta, roots: list[int], resolved_roots: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k, root in enumerate(roots):
        rr = resolved_roots.iloc[k] if k < len(resolved_roots) else pd.Series(dtype=object)
        root_type = str(rr.get("root_type_used", rr.get("root_type", "")))
        allowed = parse_root_type_compartments(root_type)
        mapped = adata_meta.obs.iloc[int(root)]
        if not allowed:
            # Use metacell composition if the root table carries no biological type.
            allowed = [c for c in COMPARTMENTS if float(mapped.get(f"frac_{c}", 0.0)) >= 0.20]
        if not allowed:
            major = str(mapped.get("major_compartment", "unknown"))
            if major in COMPARTMENTS:
                allowed = [major]
        scope = process_scope_from_compartments(allowed)
        rec: dict[str, Any] = {
            "root_pos": int(k),
            "root_index": int(root),
            "root_name": str(adata_meta.obs_names[int(root)]),
            "original_root_id": str(rr.get("original_root_value", "")),
            "root_type": root_type,
            "root_allowed_compartments": "|".join(allowed),
            "root_process_scope": scope,
            "mapped_metacell_major_compartment": str(mapped.get("major_compartment", "unknown")),
            "mapped_metacell_annotation": str(mapped.get("annotation_label", "unknown")),
            "root_type_compatibility": float(rr.get("root_type_compatibility", 0.0)),
            "resolution_mode": str(rr.get("resolution_mode", "")),
        }
        for comp in COMPARTMENTS:
            rec[f"root_weight_{comp}"] = 1.0 / len(allowed) if comp in allowed and allowed else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def write_metacell_outputs(result: MetacellBuildResult, outdir: Path) -> None:
    outdir = Path(outdir)
    result.cell_to_metacell.to_csv(outdir / "cell_to_metacell_mapping.csv", index=False)
    result.metacell_summary.to_csv(outdir / "metacell_summary.csv", index=False)
    result.audit.to_csv(outdir / "metacell_build_audit.csv", index=False)
    result.adata.write_h5ad(outdir / "metacell_adata.h5ad")
