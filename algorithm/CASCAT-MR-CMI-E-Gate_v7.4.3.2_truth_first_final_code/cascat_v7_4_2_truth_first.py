#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Truth-first evidence gates for CASCAT-MR-CMI-E-Gate v7.4.3.2 corrected patch.

The module is deliberately conservative.  It never fabricates a trajectory when
support is absent.  It separates four claims that are often conflated:

1. a root is a usable anchor;
2. an edge is structurally reliable enough for a backbone;
3. a root-specific connected structure is large and stable enough to call a
   trajectory;
4. a trajectory has molecular-program evidence supporting a biological-process
   interpretation.

Levels used by the module:
    Level 0 -- insufficient support
    Level 1 -- structurally plausible
    Level 2 -- process supported
    Level 3 -- externally biologically supported (reserved for explicit external evidence)

The module does not use synthetic truth or external labels unless they are
explicitly supplied by the user.  Internal program scores are evidence, not
experimental validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Optional
import re

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


# A compact, auditable TME program panel.  These are heuristic marker sets used
# only for internal process evidence.  Absence of genes yields "unavailable";
# the code never replaces missing molecular evidence with a guessed label.
TME_PROGRAM_SIGNATURES: dict[str, tuple[str, ...]] = {
    "Hypoxia": ("HIF1A", "VEGFA", "CA9", "BNIP3", "LDHA", "SLC2A1", "PDK1", "ENO1"),
    "Glycolysis": ("HK1", "HK2", "PFKP", "GPI", "ALDOA", "GAPDH", "PGK1", "PKM", "LDHA"),
    "Oxidative_Stress": ("SOD1", "SOD2", "CAT", "GPX1", "GPX4", "PRDX1", "HMOX1", "NQO1"),
    "Proliferation": ("MKI67", "TOP2A", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "TYMS"),
    "Interferon_Response": ("STAT1", "IRF1", "ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1", "OAS1", "CXCL10"),
    "Antigen_Presentation": ("B2M", "HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "HLA-DRB1", "TAP1", "TAP2"),
    "Effector_T": ("CD3D", "CD3E", "CD8A", "CD8B", "IFNG", "GZMA", "GZMB", "PRF1", "NKG7"),
    "Immune_Checkpoint": ("PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "TOX", "ENTPD1"),
    "Myeloid_Inflammatory": ("IL1B", "TNF", "CXCL8", "S100A8", "S100A9", "NLRP3", "FCN1", "LYZ"),
    "M2_Macrophage": ("CD163", "MRC1", "MSR1", "C1QA", "C1QB", "C1QC", "IL10", "TGFB1", "CSF1R"),
    "EMT": ("VIM", "CDH2", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "FN1", "MMP2", "MMP9"),
    "CAF": ("FAP", "ACTA2", "PDGFRA", "PDGFRB", "COL1A1", "COL1A2", "POSTN", "THY1", "FN1"),
    "ECM_Remodeling": ("COL1A1", "COL1A2", "COL3A1", "FN1", "MMP2", "MMP9", "MMP14", "LOX", "LOXL2", "TNC"),
    "Angiogenesis": ("VEGFA", "VEGFB", "VEGFC", "FLT1", "KDR", "FLT4", "NRP1", "NRP2", "ANGPT2", "PGF"),
    "Apoptosis_Stress": ("BAX", "BAK1", "BBC3", "PMAIP1", "DDIT3", "ATF4", "JUN", "FOS", "HSPA1A", "HSP90AA1"),
}

COMPARTMENT_FRACTION_COLUMNS: dict[str, str] = {
    "tumor_epithelial": "frac_tumor_epithelial",
    "lymphoid": "frac_lymphoid",
    "myeloid": "frac_myeloid",
    "endothelial_vascular": "frac_endothelial_vascular",
    "stromal_fibroblast": "frac_stromal_fibroblast",
    "perivascular_smooth_muscle": "frac_perivascular_smooth_muscle",
}


def _dense_float(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=float)


def _safe_numeric(series: pd.Series, default: float = np.nan) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(default):
        arr[~np.isfinite(arr)] = float(default)
    return arr


def _edge_key(a: int, b: int) -> tuple[int, int]:
    a, b = int(a), int(b)
    return (a, b) if a <= b else (b, a)


def _normalise_01(values: Iterable[float], default: float = 0.0) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    finite = np.isfinite(arr)
    out = np.full(arr.shape, float(default), dtype=float)
    if not finite.any():
        return out
    lo, hi = float(np.nanmin(arr[finite])), float(np.nanmax(arr[finite]))
    if hi > lo:
        out[finite] = (arr[finite] - lo) / (hi - lo)
    else:
        out[finite] = 1.0
    return np.clip(out, 0.0, 1.0)


def _bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return q


def _spatial_coords(adata) -> Optional[np.ndarray]:
    for key in ("spatial", "X_spatial"):
        if key in getattr(adata, "obsm", {}):
            arr = np.asarray(adata.obsm[key], dtype=float)
            if arr.ndim == 2 and arr.shape[0] == adata.n_obs and arr.shape[1] >= 2:
                return arr[:, :2]
    cols = [c for c in ("spatial_x", "spatial_y") if c in adata.obs.columns]
    if len(cols) == 2:
        return adata.obs[cols].to_numpy(dtype=float)
    return None


def _median_nn_scale(coords: Optional[np.ndarray]) -> float:
    if coords is None or len(coords) < 2:
        return 1.0
    valid = np.all(np.isfinite(coords), axis=1)
    pts = np.asarray(coords[valid], dtype=float)
    if len(pts) < 2:
        return 1.0
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    nn = d[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    return float(np.median(nn)) if len(nn) else 1.0



_GENE_ALIAS_TABLE: dict[str, tuple[str, ...]] = {
    # Human/mouse and common marker aliases used in TME panels.  Matching is
    # deliberately case-insensitive and punctuation-insensitive; this avoids
    # false unavailability when an AnnData object mixes symbols such as Mki67,
    # Ki67, H2-K1, HLA-A, IFIT1, Ifit1, or ENSG|SYMBOL annotations.
    "MKI67": ("KI67", "MKI67", "MKI-67", "MKI_67"),
    "TOP2A": ("TOP2A", "TOPIIA"),
    "TYMS": ("TYMS", "TS"),
    "HLA-A": ("HLA-A", "HLAA", "H2-K1", "H2-D1", "H2-Q7"),
    "HLA-B": ("HLA-B", "HLAB", "H2-K1", "H2-D1"),
    "HLA-C": ("HLA-C", "HLAC", "H2-K1", "H2-D1"),
    "HLA-DRA": ("HLA-DRA", "HLADRA", "H2-AA", "H2-AB1", "H2-EB1", "H2-EA"),
    "HLA-DRB1": ("HLA-DRB1", "HLADRB1", "H2-AB1", "H2-EB1"),
    "TAP1": ("TAP1",),
    "TAP2": ("TAP2",),
    "CD8A": ("CD8A",),
    "CD8B": ("CD8B", "CD8B1"),
    "IFNG": ("IFNG", "IFN-G", "IFNγ"),
    "GZMA": ("GZMA",),
    "GZMB": ("GZMB",),
    "PRF1": ("PRF1", "PERFORIN1"),
    "NKG7": ("NKG7",),
    "CXCL8": ("CXCL8", "IL8", "IL-8", "Cxcl15"),
    "LYZ": ("LYZ", "LYZ1", "LYZ2"),
    "VEGFA": ("VEGFA", "VEGF-A"),
    "LDHA": ("LDHA", "LDH-A"),
    "HSPA1A": ("HSPA1A", "HSP70", "HSPA1B"),
    "HSP90AA1": ("HSP90AA1", "HSP90A", "HSP90"),
}


def _canonical_gene_token(name: object) -> str:
    """Return a punctuation-insensitive uppercase token for gene matching."""
    text = str(name).strip()
    # Common feature names may look like "ENSG...|MKI67" or "MKI67-1". Keep
    # several decomposed aliases downstream, but define a deterministic primary
    # token here.
    text = text.split("|")[-1].split(";")[-1].strip()
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _gene_name_candidates(adata) -> list[str]:
    names: list[str] = [str(v) for v in adata.var_names]
    var = getattr(adata, "var", None)
    if var is not None:
        for col in (
            "gene_symbol", "gene_symbols", "symbol", "symbols", "gene_name", "gene_names",
            "feature_name", "feature_names", "external_gene_name", "Gene", "gene", "SYMBOL",
        ):
            if col in var.columns:
                names.extend(str(v) for v in var[col].tolist())
    return names


def _build_gene_index(adata) -> tuple[np.ndarray, dict[str, int]]:
    var_names = np.asarray([str(v) for v in adata.var_names], dtype=object)
    token_to_idx: dict[str, int] = {}
    # Index var_names and useful annotation columns.  Preserve first occurrence
    # for deterministic slicing.
    for i, raw in enumerate(var_names):
        for piece in re.split(r"[|;,/\\s]+", str(raw)) + [str(raw)]:
            tok = _canonical_gene_token(piece)
            if tok:
                token_to_idx.setdefault(tok, int(i))
    var = getattr(adata, "var", None)
    if var is not None:
        for col in (
            "gene_symbol", "gene_symbols", "symbol", "symbols", "gene_name", "gene_names",
            "feature_name", "feature_names", "external_gene_name", "Gene", "gene", "SYMBOL",
        ):
            if col in var.columns:
                for i, raw in enumerate(var[col].tolist()):
                    for piece in re.split(r"[|;,/\\s]+", str(raw)) + [str(raw)]:
                        tok = _canonical_gene_token(piece)
                        if tok and i < len(var_names):
                            token_to_idx.setdefault(tok, int(i))
    return var_names, token_to_idx


def _signature_gene_tokens(gene: str) -> list[str]:
    tokens = {_canonical_gene_token(gene)}
    for alias in _GENE_ALIAS_TABLE.get(str(gene).upper(), ()):  # explicit aliases
        tokens.add(_canonical_gene_token(alias))
    # Generic mouse/human style is already handled by case-insensitive matching;
    # adding punctuation-stripped tokens catches HLA-A vs HLAA, IFN-G vs IFNG.
    return [t for t in tokens if t]

def compute_process_program_scores(
    adata,
    *,
    min_genes: int = 2,
    signatures: Optional[dict[str, Iterable[str]]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute internal molecular-program evidence per metacell.

    Scores are means of gene-wise z-scores.  A program is marked unavailable when
    fewer than ``min_genes`` signature genes are observed.  Missing programs are
    not imputed from geometry, cluster labels, or embeddings.
    """
    signatures = signatures or TME_PROGRAM_SIGNATURES
    var_names, token_to_idx = _build_gene_index(adata)

    base = pd.DataFrame({
        "node_id": np.arange(adata.n_obs, dtype=int),
        "node_name": np.asarray(adata.obs_names, dtype=str),
    })
    availability_rows = []
    for program, genes in signatures.items():
        genes = tuple(str(g) for g in genes)
        idx: list[int] = []
        matched_markers: list[str] = []
        for gene in genes:
            hit = None
            for tok in _signature_gene_tokens(gene):
                if tok in token_to_idx:
                    hit = token_to_idx[tok]
                    break
            if hit is not None and hit not in idx:
                idx.append(int(hit))
                matched_markers.append(str(gene))
        availability_rows.append({
            "program": program,
            "n_signature_genes": len(genes),
            "n_observed_genes": len(idx),
            "observed_genes": "|".join(var_names[idx].astype(str).tolist()) if idx else "",
            "matched_signature_markers": "|".join(matched_markers),
            "gene_matching_mode": "species_aware_case_and_punctuation_insensitive_alias_matching",
            "program_available": bool(len(idx) >= int(min_genes)),
            "min_genes_required": int(min_genes),
        })
        col = f"program__{program}"
        if len(idx) < int(min_genes):
            base[col] = np.nan
            continue
        # Slice only the required signature genes before densifying. This avoids
        # materialising the full metacell-by-gene matrix for large assays.
        sub = _dense_float(adata.X[:, idx])
        mu = np.nanmean(sub, axis=0, keepdims=True)
        sd = np.nanstd(sub, axis=0, keepdims=True)
        sd[sd < 1e-12] = 1.0
        z = np.nan_to_num((sub - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        base[col] = np.nanmean(z, axis=1)
    return base, pd.DataFrame(availability_rows)


def build_root_validity_audit(
    adata,
    roots: list[int],
    results_dir: Path,
    trajectory_edges: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """Evaluate whether each root is a trustworthy anchor before trajectory claims."""
    results_dir = Path(results_dir)
    resolved = pd.read_csv(results_dir / "resolved_roots.csv") if (results_dir / "resolved_roots.csv").exists() else pd.DataFrame()
    scope = pd.read_csv(results_dir / "root_process_scope.csv") if (results_dir / "root_process_scope.csv").exists() else pd.DataFrame()
    anchors = pd.read_csv(results_dir / "root_anchor_sets.csv") if (results_dir / "root_anchor_sets.csv").exists() else pd.DataFrame()

    coords = _spatial_coords(adata)
    nn_scale = _median_nn_scale(coords)
    g = nx.Graph()
    g.add_nodes_from(range(adata.n_obs))
    if trajectory_edges is not None and not trajectory_edges.empty:
        for r in trajectory_edges.itertuples(index=False):
            g.add_edge(int(r.source), int(r.target))
    positive_degrees = [d for _, d in g.degree() if d > 0]
    median_degree = float(np.median(positive_degrees)) if positive_degrees else 1.0

    max_projection_factor = float(getattr(cfg, "truth_root_max_projection_nn_factor", 6.0))
    min_root_quality = float(getattr(cfg, "truth_min_root_quality", 0.50))
    rows = []
    for k, root in enumerate(roots):
        rr = resolved.iloc[k] if len(resolved) > k else pd.Series(dtype=object)
        rs = scope.loc[scope.get("root_pos", pd.Series(dtype=int)).astype(int) == k]
        rs = rs.iloc[0] if not rs.empty else pd.Series(dtype=object)
        ra = anchors.loc[anchors.get("root_pos", pd.Series(dtype=int)).astype(int) == k]

        projection_distance = float(pd.to_numeric(pd.Series([rr.get("projection_distance", np.nan)]), errors="coerce").iloc[0])
        projection_factor = projection_distance / max(nn_scale, 1e-9) if np.isfinite(projection_distance) else np.nan
        projection_score = math.exp(-projection_factor / 3.0) if np.isfinite(projection_factor) else 0.0
        type_compat = float(pd.to_numeric(pd.Series([rr.get("root_type_compatibility", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        raw_type_available = rr.get("root_type_evidence_available", np.nan)
        if pd.isna(raw_type_available):
            # Backward-compatible fallback for older outputs that did not write the
            # availability flag. A non-empty allowed-compartment list means type
            # evidence was recognized; empty means unavailable, not incompatible.
            allowed_text = str(rr.get("root_allowed_compartments", "")).strip()
            type_evidence_available = bool(allowed_text and allowed_text.lower() not in {"nan", "none"})
        elif isinstance(raw_type_available, (bool, np.bool_)):
            type_evidence_available = bool(raw_type_available)
        else:
            type_evidence_available = str(raw_type_available).strip().lower() in {"true", "1", "yes", "y", "t"}
        anchor_count = int(len(ra))
        mean_anchor_score = float(pd.to_numeric(ra.get("anchor_score", pd.Series(dtype=float)), errors="coerce").mean()) if anchor_count else 0.0
        anchor_support = min(anchor_count / 3.0, 1.0) * np.clip(mean_anchor_score, 0.0, 1.0)
        degree = int(g.degree(int(root))) if int(root) in g else 0
        degree_support = min(degree / max(median_degree, 1.0), 1.0)
        scope_size = int(pd.to_numeric(pd.Series([rs.get("root_scope_size", 0)]), errors="coerce").fillna(0).iloc[0])
        scope_support = min(scope_size / 10.0, 1.0)
        # Keep the exact v7.4.2 score when type evidence is available.  When the
        # ontology label is unrecognized, omit that unavailable channel and
        # renormalize the remaining evidence weights instead of injecting a false 0.
        quality_terms = [
            (0.25, projection_score),
            (0.20, anchor_support),
            (0.15, degree_support),
            (0.15, scope_support),
        ]
        if type_evidence_available:
            quality_terms.insert(1, (0.25, float(np.clip(type_compat, 0.0, 1.0))))
        quality_weight_sum = float(sum(w for w, _ in quality_terms))
        quality = float(sum(w * float(v) for w, v in quality_terms) / max(quality_weight_sum, 1e-12))

        hard_reasons = []
        validity_notes = []
        if type_evidence_available and type_compat < 0.50:
            hard_reasons.append("root_type_incompatible")
        elif not type_evidence_available:
            validity_notes.append("root_type_unrecognized_type_evidence_unavailable")
        if anchor_count < 1:
            hard_reasons.append("no_anchor_support")
        if np.isfinite(projection_factor) and projection_factor > max_projection_factor:
            hard_reasons.append("projection_too_far_from_root_prior")

        if hard_reasons:
            status = "invalid"
        elif quality < min_root_quality:
            status = "weak"
        elif quality < 0.70:
            status = "usable_with_review"
        else:
            status = "strong"

        rows.append({
            "root_pos": int(k),
            "root_index": int(root),
            "root_name": str(adata.obs_names[int(root)]),
            "original_root_id": str(rr.get("root_id", rr.get("original_root_value", ""))),
            "root_type": str(rr.get("root_type_used", rr.get("root_type", ""))),
            "root_process_scope": str(rs.get("root_process_scope", "")),
            "root_allowed_compartments": str(rs.get("root_allowed_compartments", "")),
            "projection_distance": projection_distance,
            "dataset_median_spatial_nn_distance": float(nn_scale),
            "projection_distance_nn_factor": projection_factor,
            "projection_score": projection_score,
            "root_type_compatibility": type_compat,
            "root_type_evidence_available": bool(type_evidence_available),
            "root_type_evidence_status": str(rr.get("root_type_evidence_status", "recognized_compartment_evidence" if type_evidence_available else "unrecognized_type_evidence_unavailable")),
            "root_quality_weight_sum": quality_weight_sum,
            "anchor_count": anchor_count,
            "mean_anchor_score": mean_anchor_score,
            "root_graph_degree": degree,
            "root_isolated_in_trajectory_graph": bool(degree <= 0),
            "median_positive_graph_degree": median_degree,
            "root_scope_size": scope_size,
            "root_quality_score": float(quality),
            "root_validity_status": status,
            "root_hard_fail_reasons": "|".join(hard_reasons),
            "root_validity_notes": "|".join(validity_notes),
        })
    return pd.DataFrame(rows)


def _stability_recurrence_by_threshold(
    sub: pd.DataFrame,
    root_idx: int,
    cfg,
) -> dict[tuple[int, int], float]:
    """Return intrinsic edge recurrence under nearby confidence thresholds.

    The baseline is the threshold recorded when the core was extracted. Reusing
    the original threshold avoids the pathological inflation that occurs when a
    quantile is recomputed on the already-pruned core itself. Root connectivity
    is audited separately after hard vetoes.
    """
    if sub.empty or "edge_confidence" not in sub.columns:
        return {}
    conf = pd.to_numeric(sub["edge_confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if "core_threshold" in sub.columns:
        base_vals = pd.to_numeric(sub["core_threshold"], errors="coerce")
        base_vals = base_vals[np.isfinite(base_vals)]
        baseline = float(np.median(base_vals)) if len(base_vals) else np.nan
    else:
        baseline = np.nan
    if not np.isfinite(baseline):
        q0 = float(np.clip(getattr(cfg, "core_edge_confidence_quantile", 0.60), 0.0, 1.0))
        floor0 = float(np.clip(getattr(cfg, "core_edge_confidence_floor", 0.45), 0.0, 1.0))
        baseline = max(floor0, float(np.quantile(conf, q0)) if len(conf) else floor0)

    offsets = (-0.08, -0.04, 0.0, 0.04, 0.08)
    thresholds = [float(np.clip(baseline + delta, 0.0, 1.0)) for delta in offsets]
    counts = {_edge_key(r.source, r.target): 0 for r in sub.itertuples(index=False)}
    for threshold in thresholds:
        for r, c in zip(sub.itertuples(index=False), conf):
            if float(c) >= threshold:
                counts[_edge_key(r.source, r.target)] += 1
    return {k: v / max(len(thresholds), 1) for k, v in counts.items()}


def apply_edge_truth_gate(
    pre_truth_backbone_edges: pd.DataFrame,
    roots: list[int],
    cfg,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply hard-veto, stability and root-connectivity gates to backbone edges.

    Crucially, ``root_path_rescue`` may never override a failed hard veto.
    """
    if pre_truth_backbone_edges is None or pre_truth_backbone_edges.empty:
        return pd.DataFrame(), pd.DataFrame()

    # v7.4.2: compute dataset-level reference floors before per-root filtering.
    # Very small candidate trees must not define their own weak adaptive floors.
    all_sglobal = pd.to_numeric(
        pre_truth_backbone_edges.get("S_global", pd.Series(0.0, index=pre_truth_backbone_edges.index)),
        errors="coerce",
    ).fillna(0.0)
    global_s_q = float(np.clip(getattr(cfg, "truth_s_global_quantile_floor", 0.05), 0.0, 1.0))
    global_sglobal_floor = max(
        float(getattr(cfg, "truth_min_s_global_floor", 0.03)),
        float(np.quantile(all_sglobal.to_numpy(dtype=float), global_s_q)) if len(all_sglobal) else 0.0,
    )
    global_cost_col = "cost_final" if "cost_final" in pre_truth_backbone_edges.columns else (
        "edge_cost" if "edge_cost" in pre_truth_backbone_edges.columns else None
    )
    if global_cost_col:
        all_costs = pd.to_numeric(pre_truth_backbone_edges[global_cost_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite_global_costs = all_costs[np.isfinite(all_costs) & (all_costs > 0)]
        global_median_cost = float(np.median(finite_global_costs)) if len(finite_global_costs) else 1.0
    else:
        global_median_cost = 1.0

    passed_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    for k, root in enumerate(roots):
        sub = pre_truth_backbone_edges.loc[
            pd.to_numeric(pre_truth_backbone_edges.get("root_pos", pd.Series(dtype=int)), errors="coerce").fillna(-1).astype(int) == int(k)
        ].copy()
        if sub.empty:
            continue
        stability = _stability_recurrence_by_threshold(sub, int(root), cfg)
        sub["edge_stability_recurrence"] = [stability.get(_edge_key(r.source, r.target), 0.0) for r in sub.itertuples(index=False)]

        conf = pd.to_numeric(sub.get("edge_confidence", pd.Series(0.0, index=sub.index)), errors="coerce").fillna(0.0)
        s_global = pd.to_numeric(sub.get("S_global", pd.Series(0.0, index=sub.index)), errors="coerce").fillna(0.0)
        cost_col = "cost_final" if "cost_final" in sub.columns else ("edge_cost" if "edge_cost" in sub.columns else None)
        costs = pd.to_numeric(sub[cost_col], errors="coerce").replace([np.inf, -np.inf], np.nan) if cost_col else pd.Series(1.0, index=sub.index)
        finite_costs = costs[np.isfinite(costs) & (costs > 0)]
        median_cost = float(np.median(finite_costs)) if len(finite_costs) else 1.0
        min_reference_edges = int(getattr(cfg, "truth_min_edges_for_adaptive_threshold", 5))
        use_dataset_reference = len(sub) < min_reference_edges
        sglobal_q = float(np.clip(getattr(cfg, "truth_s_global_quantile_floor", 0.05), 0.0, 1.0))
        if use_dataset_reference:
            adaptive_sglobal_floor = global_sglobal_floor
            reference_median_cost = global_median_cost
            threshold_reference_scope = "dataset_level_due_to_small_tree"
        else:
            adaptive_sglobal_floor = max(
                float(getattr(cfg, "truth_min_s_global_floor", 0.03)),
                float(np.quantile(s_global.to_numpy(dtype=float), sglobal_q)) if len(s_global) else 0.0,
            )
            reference_median_cost = median_cost
            threshold_reference_scope = "root_tree_level"
        max_cost = reference_median_cost * float(getattr(cfg, "truth_max_cost_median_factor", 8.0))
        min_stability = float(getattr(cfg, "truth_min_edge_stability", 0.60))
        rescue_min_stability = float(getattr(cfg, "truth_rescue_min_stability", 0.80))
        rescue_min_conf = float(getattr(cfg, "truth_rescue_min_confidence", 0.55))
        max_cmi_conflict = float(getattr(cfg, "truth_max_cmi_conflict", 0.25))

        pass_flags = []
        reasons_all = []
        for idx, row in sub.iterrows():
            reasons = []
            is_rescue = str(row.get("core_selected_reason", "")) == "root_path_rescue"
            row_conf = float(conf.loc[idx])
            row_sg = float(s_global.loc[idx])
            row_cost = float(costs.loc[idx]) if np.isfinite(costs.loc[idx]) else math.inf
            row_stability = float(row.get("edge_stability_recurrence", 0.0))
            cmi_conflict = float(pd.to_numeric(pd.Series([row.get("cmi_env_conflict", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            if row_stability < min_stability:
                reasons.append("unstable_under_threshold_perturbation")
            if cmi_conflict > max_cmi_conflict:
                reasons.append("extreme_environment_conflict")

            # v7.4.2 universal Truth-First hard vetoes.
            # These constraints apply to every candidate backbone edge, not only
            # to root_path_rescue edges.  A threshold that is merely exported but
            # does not affect pass/fail is not a real truth gate.
            if row_sg < adaptive_sglobal_floor:
                reasons.append("global_similarity_below_hard_floor")
            if row_cost > max_cost:
                reasons.append("cost_extreme_vs_reference")

            # Rescue edges are optional and face additional stricter gates.
            # They can never override any universal hard veto above.
            if is_rescue:
                if row_conf < rescue_min_conf:
                    reasons.append("rescue_confidence_below_hard_floor")
                if row_stability < rescue_min_stability:
                    reasons.append("rescue_stability_below_hard_floor")
            pass_flags.append(len(reasons) == 0)
            reasons_all.append("|".join(reasons))

        sub["edge_truth_gate_pass_raw"] = pass_flags
        sub["edge_truth_gate_reasons"] = reasons_all
        sub["truth_adaptive_sglobal_floor"] = adaptive_sglobal_floor
        sub["truth_tree_median_cost"] = median_cost
        sub["truth_reference_median_cost"] = reference_median_cost
        sub["truth_threshold_reference_scope"] = threshold_reference_scope
        sub["truth_min_edges_for_adaptive_threshold"] = min_reference_edges
        sub["truth_max_allowed_cost"] = max_cost
        sub["truth_universal_sglobal_hard_veto_enabled"] = True
        sub["truth_universal_cost_hard_veto_enabled"] = True

        # v7.4.2: first remove every edge that fails any hard veto, then retain
        # only the component that remains connected to the declared root.  No
        # edge is re-added to preserve topology or visual continuity.
        g = nx.Graph()
        g.add_node(int(root))
        for r in sub.loc[sub["edge_truth_gate_pass_raw"]].itertuples(index=False):
            g.add_edge(int(r.source), int(r.target))
        connected_keys: set[tuple[int, int]] = set()
        if int(root) in g and g.number_of_edges() > 0:
            comp = set(nx.node_connected_component(g, int(root)))
            connected_keys = {_edge_key(u, v) for u, v in g.edges() if u in comp and v in comp}
        sub["edge_truth_gate_root_connected"] = [
            _edge_key(r.source, r.target) in connected_keys for r in sub.itertuples(index=False)
        ]
        sub["edge_truth_gate_pass"] = sub["edge_truth_gate_pass_raw"] & sub["edge_truth_gate_root_connected"]
        disconnected_mask = sub["edge_truth_gate_pass_raw"] & ~sub["edge_truth_gate_root_connected"]
        existing_reason = sub["edge_truth_gate_reasons"].astype(str)
        empty_reason_mask = disconnected_mask & existing_reason.eq("")
        nonempty_reason_mask = disconnected_mask & ~existing_reason.eq("")
        sub.loc[empty_reason_mask, "edge_truth_gate_reasons"] = "not_root_connected_after_hard_veto"
        sub.loc[nonempty_reason_mask, "edge_truth_gate_reasons"] = (
            sub.loc[nonempty_reason_mask, "edge_truth_gate_reasons"].astype(str)
            + "|not_root_connected_after_hard_veto"
        )

        audit_parts.append(sub.copy())
        passed_parts.append(sub.loc[sub["edge_truth_gate_pass"]].copy())

    passed = pd.concat(passed_parts, ignore_index=True) if passed_parts else pd.DataFrame(columns=pre_truth_backbone_edges.columns)
    audit = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame()
    return passed, audit


def _node_pseudotime_lookup(final_pseudo: pd.DataFrame, edge_sub: pd.DataFrame, root_pos: int) -> dict[int, float]:
    lookup: dict[int, float] = {}
    if final_pseudo is not None and not final_pseudo.empty:
        sub = final_pseudo.loc[pd.to_numeric(final_pseudo.get("root_pos", pd.Series(dtype=int)), errors="coerce").fillna(-1).astype(int) == int(root_pos)]
        node_candidates = ["node_id", "node", "index", "node_index"]
        pt_candidates = ["constrained_pseudotime", "pseudotime", "selected_pseudotime", "final_pseudotime"]
        node_col = next((c for c in node_candidates if c in sub.columns), None)
        pt_col = next((c for c in pt_candidates if c in sub.columns), None)
        if node_col and pt_col:
            for r in sub[[node_col, pt_col]].dropna().itertuples(index=False):
                lookup[int(r[0])] = float(r[1])
    # Edge tables carry source/target pseudotime in current CASCAT exports.
    if edge_sub is not None and not edge_sub.empty:
        for r in edge_sub.itertuples(index=False):
            if hasattr(r, "pseudotime_source") and np.isfinite(float(getattr(r, "pseudotime_source"))):
                lookup.setdefault(int(r.source), float(getattr(r, "pseudotime_source")))
            if hasattr(r, "pseudotime_target") and np.isfinite(float(getattr(r, "pseudotime_target"))):
                lookup.setdefault(int(r.target), float(getattr(r, "pseudotime_target")))
    return lookup


def _tree_nodes(edge_sub: pd.DataFrame, root_idx: int) -> list[int]:
    nodes: set[int] = {int(root_idx)}
    if edge_sub is not None and not edge_sub.empty:
        nodes.update(pd.to_numeric(edge_sub["source"], errors="coerce").dropna().astype(int).tolist())
        nodes.update(pd.to_numeric(edge_sub["target"], errors="coerce").dropna().astype(int).tolist())
    return sorted(nodes)


def build_tree_program_evidence(
    program_scores: pd.DataFrame,
    program_availability: pd.DataFrame,
    final_edges: pd.DataFrame,
    final_pseudo: pd.DataFrame,
    roots: list[int],
    results_dir: Path,
    cfg,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify process support without inventing a process when genes are absent."""
    results_dir = Path(results_dir)
    meta_path = results_dir / "metacell_summary.csv"
    meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
    scope_path = results_dir / "root_process_scope.csv"
    scope = pd.read_csv(scope_path) if scope_path.exists() else pd.DataFrame()
    available_programs = program_availability.loc[program_availability["program_available"].astype(bool), "program"].astype(str).tolist()
    program_cols = [f"program__{p}" for p in available_programs if f"program__{p}" in program_scores.columns]
    external = pd.DataFrame()
    external_path = str(getattr(cfg, "truth_external_program_evidence_csv", "") or "")
    if external_path:
        try:
            p_ext = Path(external_path)
            if p_ext.exists():
                external = pd.read_csv(p_ext)
        except Exception:
            external = pd.DataFrame()

    detail_rows = []
    summary_rows = []
    for k, root in enumerate(roots):
        edge_sub = final_edges.loc[pd.to_numeric(final_edges.get("root_pos", pd.Series(dtype=int)), errors="coerce").fillna(-1).astype(int) == int(k)].copy() if final_edges is not None and not final_edges.empty else pd.DataFrame()
        nodes = _tree_nodes(edge_sub, int(root))
        pt = _node_pseudotime_lookup(final_pseudo, edge_sub, k)
        scope_row = scope.loc[pd.to_numeric(scope.get("root_pos", pd.Series(dtype=int)), errors="coerce").fillna(-1).astype(int) == int(k)]
        scope_row = scope_row.iloc[0] if not scope_row.empty else pd.Series(dtype=object)
        allowed = [x for x in str(scope_row.get("root_allowed_compartments", "")).split("|") if x]
        process_scope = str(scope_row.get("root_process_scope", ""))

        pvals = []
        per_program = []
        for col in program_cols:
            program = col.replace("program__", "", 1)
            vals = []
            pts = []
            for n in nodes:
                if n >= len(program_scores) or n not in pt:
                    continue
                v = float(pd.to_numeric(pd.Series([program_scores.iloc[n][col]]), errors="coerce").iloc[0])
                if np.isfinite(v) and np.isfinite(pt[n]):
                    vals.append(v)
                    pts.append(pt[n])
            rho = pval = np.nan
            if len(vals) >= max(4, int(getattr(cfg, "truth_min_program_nodes", 4))) and np.nanstd(vals) > 1e-12 and np.nanstd(pts) > 1e-12:
                rho, pval = spearmanr(pts, vals)
                rho = float(rho) if np.isfinite(rho) else np.nan
                pval = float(pval) if np.isfinite(pval) else np.nan
            per_program.append({
                "root_pos": k,
                "root_index": int(root),
                "program": program,
                "n_nodes_with_program_and_pseudotime": len(vals),
                "spearman_rho_vs_pseudotime": rho,
                "p_value": pval,
                "mean_program_score": float(np.nanmean(vals)) if vals else np.nan,
                "std_program_score": float(np.nanstd(vals)) if vals else np.nan,
            })
            pvals.append(pval)
        qvals = _bh_adjust(pvals)
        for rec, q in zip(per_program, qvals):
            rec["bh_q_value"] = q
            detail_rows.append(rec)

        # Interface support is based on actual metacell composition, not merely
        # membership in an allowed broad compartment.
        interface_support_rate = np.nan
        interface_node_count = 0
        if len(allowed) >= 2 and not meta.empty:
            masses = []
            for n in nodes:
                if n >= len(meta):
                    continue
                row = meta.iloc[n]
                vals = []
                for comp in allowed[:2]:
                    col = COMPARTMENT_FRACTION_COLUMNS.get(comp)
                    vals.append(float(pd.to_numeric(pd.Series([row.get(col, 0.0)]), errors="coerce").fillna(0.0).iloc[0]) if col else 0.0)
                masses.append(min(vals) if vals else 0.0)
            threshold = float(getattr(cfg, "truth_interface_min_mass", 0.10))
            interface_node_count = int(sum(v >= threshold for v in masses))
            interface_support_rate = interface_node_count / len(masses) if masses else np.nan

        detail_df = pd.DataFrame(per_program)
        if not detail_df.empty:
            valid = detail_df.loc[np.isfinite(pd.to_numeric(detail_df["spearman_rho_vs_pseudotime"], errors="coerce"))].copy()
        else:
            valid = pd.DataFrame()
        if not valid.empty:
            valid["abs_rho"] = pd.to_numeric(valid["spearman_rho_vs_pseudotime"], errors="coerce").abs()
            best = valid.sort_values(["abs_rho", "bh_q_value"], ascending=[False, True]).iloc[0]
            best_program = str(best["program"])
            best_abs_rho = float(best["abs_rho"])
            best_q = float(best["bh_q_value"]) if np.isfinite(best["bh_q_value"]) else np.nan
        else:
            best_program, best_abs_rho, best_q = "", np.nan, np.nan
        min_rho = float(getattr(cfg, "truth_min_program_spearman", 0.40))
        max_q = float(getattr(cfg, "truth_max_program_q", 0.10))
        coherence_pass = bool(
            np.isfinite(best_abs_rho)
            and best_abs_rho >= min_rho
            and np.isfinite(best_q)
            and best_q <= max_q
        )
        evidence_status = "available" if program_cols else "unavailable_no_signature_coverage"
        ext_sub = pd.DataFrame()
        if not external.empty:
            if "root_pos" in external.columns:
                ext_sub = external.loc[pd.to_numeric(external["root_pos"], errors="coerce").fillna(-1).astype(int) == int(k)]
            elif "root_index" in external.columns:
                ext_sub = external.loc[pd.to_numeric(external["root_index"], errors="coerce").fillna(-1).astype(int) == int(root)]
        external_supported = False
        external_note = "not_supplied"
        external_label = ""
        if not ext_sub.empty:
            row_ext = ext_sub.iloc[0]
            external_supported = str(row_ext.get("external_process_supported", row_ext.get("process_supported", False))).lower() in {"1", "true", "yes", "y"}
            external_note = str(row_ext.get("external_evidence_note", row_ext.get("note", "supplied")))
            external_label = str(row_ext.get("external_process_label", row_ext.get("process_label", "")))
        program_evidence_independence = (
            "external_process_evidence_supplied" if external_supported else "internal_signature_coherence_only_not_independent_validation"
        )
        summary_rows.append({
            "root_pos": int(k),
            "root_index": int(root),
            "root_process_scope": process_scope,
            "n_tree_nodes": len(nodes),
            "n_tree_edges": int(len(edge_sub)),
            "n_available_programs": len(program_cols),
            "program_evidence_status": evidence_status,
            "best_progression_program": best_program,
            "best_program_abs_spearman": best_abs_rho,
            "best_program_bh_q": best_q,
            "program_coherence_pass": coherence_pass,
            "interface_support_node_count": interface_node_count,
            "interface_support_rate": interface_support_rate,
            "interface_support_pass": bool(
                np.isfinite(interface_support_rate)
                and interface_support_rate >= float(getattr(cfg, "truth_min_interface_support_rate", 0.10))
            ) if len(allowed) >= 2 else True,
            "program_evidence_independence": program_evidence_independence,
            "external_process_supported": bool(external_supported),
            "external_process_label": external_label,
            "external_evidence_note": external_note,
        })
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def build_trajectory_truth_gate(
    final_edges: pd.DataFrame,
    roots: list[int],
    root_validity: pd.DataFrame,
    program_summary: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    rows = []
    min_edges = int(getattr(cfg, "truth_min_trajectory_edges", 3))
    min_nodes = int(getattr(cfg, "truth_min_trajectory_nodes", 4))
    for k, root in enumerate(roots):
        sub = final_edges.loc[pd.to_numeric(final_edges.get("root_pos", pd.Series(dtype=int)), errors="coerce").fillna(-1).astype(int) == int(k)].copy() if final_edges is not None and not final_edges.empty else pd.DataFrame()
        nodes = _tree_nodes(sub, int(root))
        rv = root_validity.loc[pd.to_numeric(root_validity.get("root_pos", pd.Series(dtype=int)), errors="coerce").fillna(-1).astype(int) == int(k)]
        rv = rv.iloc[0] if not rv.empty else pd.Series(dtype=object)
        ps = program_summary.loc[pd.to_numeric(program_summary.get("root_pos", pd.Series(dtype=int)), errors="coerce").fillna(-1).astype(int) == int(k)]
        ps = ps.iloc[0] if not ps.empty else pd.Series(dtype=object)
        root_status = str(rv.get("root_validity_status", "invalid"))
        root_hard_fail_reasons = str(rv.get("root_hard_fail_reasons", ""))
        mapping_fail_reasons = [
            reason for reason in root_hard_fail_reasons.split("|")
            if reason and reason != "root_isolated_in_trajectory_graph"
        ]
        root_graph_degree = int(
            pd.to_numeric(pd.Series([rv.get("root_graph_degree", 0)]), errors="coerce")
            .fillna(0).iloc[0]
        )
        root_isolated = bool(rv.get("root_isolated_in_trajectory_graph", root_graph_degree <= 0))

        reasons = []
        invalid_mapping = bool(root_status == "invalid" and mapping_fail_reasons)
        if invalid_mapping:
            reasons.append("invalid_root_mapping=" + ",".join(mapping_fail_reasons))
        elif root_status in {"invalid", "weak"}:
            reasons.append(f"root_validity={root_status}")
        if len(sub) < min_edges:
            reasons.append(f"trajectory_edges={len(sub)}<{min_edges}")
        if len(nodes) < min_nodes:
            reasons.append(f"trajectory_nodes={len(nodes)}<{min_nodes}")

        if reasons:
            level = 0
            trajectory_status = "insufficient_support"
            if invalid_mapping:
                root_truth_status = "INVALID_ROOT"
            elif len(sub) >= 1 and len(nodes) >= 2:
                # A local, truth-passed connected state structure exists, but it is
                # too small to support a directional trajectory claim.
                root_truth_status = "LOCAL_STATE_SYSTEM"
            else:
                # Zero truth-passed edges must never be promoted to LOCAL_STATE_SYSTEM
                # merely because the candidate graph degree is positive.
                root_truth_status = "INSUFFICIENT_SUPPORT"
        else:
            level = 1
            trajectory_status = "structurally_plausible"
            root_truth_status = "STRUCTURALLY_PLAUSIBLE_TRAJECTORY_CANDIDATE"

        program_available = str(ps.get("program_evidence_status", "")).startswith("available")
        coherence_pass = bool(ps.get("program_coherence_pass", False))
        interface_pass = bool(ps.get("interface_support_pass", False))
        require_external = bool(getattr(cfg, "truth_require_external_process_evidence_for_level2", True))
        external_supported = bool(ps.get("external_process_supported", False))
        if level >= 1 and program_available and coherence_pass and interface_pass:
            if require_external and not external_supported:
                reasons.append("external_or_orthogonal_process_evidence_not_supplied")
            else:
                level = 2
                trajectory_status = "process_supported"
                root_truth_status = "PROCESS_SUPPORTED"
        elif level >= 1 and not program_available:
            reasons.append("molecular_program_evidence_unavailable")
        elif level >= 1 and not coherence_pass:
            reasons.append("no_multiple-testing-controlled_program_progression_support")
        if level >= 1 and not interface_pass:
            reasons.append("declared_interface_process_not_supported_by_metacell_composition")

        process_scope = str(ps.get("root_process_scope", rv.get("root_process_scope", "")))
        interface_rate = float(pd.to_numeric(pd.Series([ps.get("interface_support_rate", np.nan)]), errors="coerce").iloc[0])
        if "+" in process_scope and (not np.isfinite(interface_rate) or not interface_pass):
            conservative_interpretation = "local_branch_from_mixed_process_root_not_validated_as_interface_continuum"
        elif level >= 2:
            conservative_interpretation = process_scope or "process_supported_trajectory"
        elif level >= 1:
            conservative_interpretation = "structurally_plausible_trajectory_candidate_not_biologically_validated"
        elif root_truth_status == "INVALID_ROOT":
            conservative_interpretation = "invalid_root_mapping_no_trajectory_claim"
        elif root_truth_status == "LOCAL_STATE_SYSTEM":
            conservative_interpretation = "local_state_system_no_directional_trajectory_claim"
        else:
            conservative_interpretation = "insufficient_support_no_trajectory_claim"

        rows.append({
            "root_pos": int(k),
            "root_index": int(root),
            "root_name": str(rv.get("root_name", "")),
            "original_root_id": str(rv.get("original_root_id", "")),
            "root_process_scope": process_scope,
            "root_validity_status": root_status,
            "root_truth_status": root_truth_status,
            "root_quality_score": float(pd.to_numeric(pd.Series([rv.get("root_quality_score", np.nan)]), errors="coerce").iloc[0]),
            "root_graph_degree": root_graph_degree,
            "root_isolated_in_trajectory_graph": root_isolated,
            "n_truth_gate_nodes": len(nodes),
            "n_truth_gate_edges": int(len(sub)),
            "truth_level": int(level),
            "truth_level_label": {
                0: "Level 0 - insufficient support",
                1: "Level 1 - structurally plausible trajectory candidate",
                2: "Level 2 - process supported",
                3: "Level 3 - externally biologically supported",
            }[level],
            "trajectory_truth_status": trajectory_status,
            "program_coherence_pass": coherence_pass,
            "interface_support_pass": interface_pass,
            "conservative_process_interpretation": conservative_interpretation,
            "truth_gate_reasons": "|".join(reasons),
            "program_evidence_independence": str(ps.get("program_evidence_independence", "internal_signature_coherence_only_not_independent_validation")),
            "external_biological_support": bool(external_supported),
            "external_process_label": str(ps.get("external_process_label", "")),
            "external_evidence_note": str(ps.get("external_evidence_note", "not supplied; Level 3 cannot be assigned")),
        })
    return pd.DataFrame(rows)


def filter_edges_by_trajectory_truth_level(
    edges: pd.DataFrame,
    trajectory_gate: pd.DataFrame,
    min_level: int = 1,
) -> pd.DataFrame:
    if edges is None or edges.empty:
        return pd.DataFrame(columns=edges.columns if edges is not None else None)
    allowed = set(
        trajectory_gate.loc[pd.to_numeric(trajectory_gate["truth_level"], errors="coerce").fillna(0).astype(int) >= int(min_level), "root_pos"]
        .astype(int)
        .tolist()
    )
    return edges.loc[pd.to_numeric(edges["root_pos"], errors="coerce").fillna(-1).astype(int).isin(allowed)].copy()


def _tree_program_centroid(program_scores: pd.DataFrame, nodes: list[int], program_cols: list[str]) -> np.ndarray:
    if not nodes or not program_cols:
        return np.array([], dtype=float)
    valid_nodes = [n for n in nodes if 0 <= n < len(program_scores)]
    if not valid_nodes:
        return np.array([], dtype=float)
    X = program_scores.iloc[valid_nodes][program_cols].to_numpy(dtype=float)
    return np.nanmean(X, axis=0)


def build_process_distinctness_matrix(
    final_edges: pd.DataFrame,
    roots: list[int],
    trajectory_gate: pd.DataFrame,
    program_scores: pd.DataFrame,
    program_availability: pd.DataFrame,
    adata,
    results_dir: Path,
    cfg,
) -> pd.DataFrame:
    available = program_availability.loc[program_availability["program_available"].astype(bool), "program"].astype(str).tolist()
    program_cols = [f"program__{p}" for p in available if f"program__{p}" in program_scores.columns]
    scope_path = Path(results_dir) / "root_process_scope.csv"
    scope = pd.read_csv(scope_path) if scope_path.exists() else pd.DataFrame()
    coords = _spatial_coords(adata)
    nn_scale = _median_nn_scale(coords)
    rows = []
    for k in range(len(roots)):
        for l in range(k + 1, len(roots)):
            gk = trajectory_gate.loc[trajectory_gate["root_pos"].astype(int) == k]
            gl = trajectory_gate.loc[trajectory_gate["root_pos"].astype(int) == l]
            gk = gk.iloc[0] if not gk.empty else pd.Series(dtype=object)
            gl = gl.iloc[0] if not gl.empty else pd.Series(dtype=object)
            ek = final_edges.loc[final_edges["root_pos"].astype(int) == k] if final_edges is not None and not final_edges.empty else pd.DataFrame()
            el = final_edges.loc[final_edges["root_pos"].astype(int) == l] if final_edges is not None and not final_edges.empty else pd.DataFrame()
            nk = _tree_nodes(ek, roots[k])
            nl = _tree_nodes(el, roots[l])
            ck = _tree_program_centroid(program_scores, nk, program_cols)
            cl = _tree_program_centroid(program_scores, nl, program_cols)
            if len(ck) and len(cl) and np.isfinite(ck).any() and np.isfinite(cl).any():
                ck2 = np.nan_to_num(ck, nan=0.0)
                cl2 = np.nan_to_num(cl, nan=0.0)
                denom = np.linalg.norm(ck2) * np.linalg.norm(cl2)
                cosine_similarity = float(np.dot(ck2, cl2) / denom) if denom > 1e-12 else np.nan
                program_distance = float(1.0 - cosine_similarity) if np.isfinite(cosine_similarity) else np.nan
                diff = np.abs(ck2 - cl2)
                max_diff = float(np.max(diff)) if len(diff) else np.nan
                top_diff_program = available[int(np.argmax(diff))] if len(diff) and len(available) == len(diff) else ""
            else:
                cosine_similarity = program_distance = max_diff = np.nan
                top_diff_program = ""

            spatial_centroid_distance = np.nan
            if coords is not None:
                vk = np.asarray([n for n in nk if 0 <= n < len(coords)], dtype=int)
                vl = np.asarray([n for n in nl if 0 <= n < len(coords)], dtype=int)
                if len(vk) and len(vl):
                    spatial_centroid_distance = float(np.linalg.norm(np.nanmean(coords[vk], axis=0) - np.nanmean(coords[vl], axis=0)))
            spatial_nn_factor = spatial_centroid_distance / max(nn_scale, 1e-9) if np.isfinite(spatial_centroid_distance) else np.nan

            sk = scope.loc[pd.to_numeric(scope.get("root_pos", pd.Series(dtype=int)), errors="coerce").fillna(-1).astype(int) == k]
            sl = scope.loc[pd.to_numeric(scope.get("root_pos", pd.Series(dtype=int)), errors="coerce").fillna(-1).astype(int) == l]
            scope_k = str(sk.iloc[0].get("root_process_scope", "")) if not sk.empty else ""
            scope_l = str(sl.iloc[0].get("root_process_scope", "")) if not sl.empty else ""
            same_broad_scope = bool(scope_k == scope_l and scope_k != "")

            levels_ok = int(gk.get("truth_level", 0)) >= 2 and int(gl.get("truth_level", 0)) >= 2
            program_evidence_available = len(program_cols) > 0 and np.isfinite(program_distance)
            distinct = bool(
                levels_ok
                and program_evidence_available
                and program_distance >= float(getattr(cfg, "truth_distinct_program_distance", 0.25))
                and max_diff >= float(getattr(cfg, "truth_distinct_max_program_diff", 0.50))
            )
            if distinct:
                classification = "distinct_biological_processes_supported"
            elif same_broad_scope and int(gk.get("truth_level", 0)) >= 1 and int(gl.get("truth_level", 0)) >= 1:
                classification = "same_process_family_different_spatial_branch"
            elif int(gk.get("truth_level", 0)) == 0 or int(gl.get("truth_level", 0)) == 0:
                classification = "not_comparable_insufficient_trajectory_support"
            elif not program_evidence_available:
                classification = "process_distinctness_unresolved_program_evidence_unavailable"
            else:
                classification = "process_distinctness_not_supported"

            rows.append({
                "tree_k": k,
                "tree_l": l,
                "root_k_index": int(roots[k]),
                "root_l_index": int(roots[l]),
                "root_k_name": str(adata.obs_names[int(roots[k])]),
                "root_l_name": str(adata.obs_names[int(roots[l])]),
                "root_k_process_scope": scope_k,
                "root_l_process_scope": scope_l,
                "same_broad_process_scope": same_broad_scope,
                "root_k_truth_level": int(gk.get("truth_level", 0)),
                "root_l_truth_level": int(gl.get("truth_level", 0)),
                "n_programs_available": len(program_cols),
                "program_centroid_cosine_similarity": cosine_similarity,
                "program_centroid_distance": program_distance,
                "max_absolute_program_mean_difference": max_diff,
                "top_differential_program": top_diff_program,
                "spatial_centroid_distance": spatial_centroid_distance,
                "spatial_centroid_distance_nn_factor": spatial_nn_factor,
                "distinct_process_supported": distinct,
                "process_relation_classification": classification,
                "interpretation_rule": "spatial separation alone never establishes a distinct biological process",
            })
    return pd.DataFrame(rows)


def build_truth_first_overview(
    dataset: str,
    roots: list[int],
    root_validity: pd.DataFrame,
    edge_audit: pd.DataFrame,
    trajectory_gate: pd.DataFrame,
    distinctness: pd.DataFrame,
) -> pd.DataFrame:
    levels = pd.to_numeric(trajectory_gate.get("truth_level", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int)
    status_counts = trajectory_gate.get("trajectory_truth_status", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
    n_failed_edges = int((~edge_audit.get("edge_truth_gate_pass", pd.Series(dtype=bool)).astype(bool)).sum()) if not edge_audit.empty else 0
    n_rescue_failed = 0
    if not edge_audit.empty and "core_selected_reason" in edge_audit.columns:
        mask = edge_audit["core_selected_reason"].astype(str).eq("root_path_rescue")
        n_rescue_failed = int((mask & ~edge_audit["edge_truth_gate_pass"].astype(bool)).sum())
    n_distinct = int(distinctness.get("distinct_process_supported", pd.Series(dtype=bool)).astype(bool).sum()) if not distinctness.empty else 0
    overall = "TRUTH_FIRST_REVIEW"
    if int((levels >= 1).sum()) == 0:
        overall = "INSUFFICIENT_SUPPORT_NO_VALID_TRAJECTORY"
    elif int((levels >= 2).sum()) >= 1:
        overall = "AT_LEAST_ONE_PROCESS_SUPPORTED_TRAJECTORY"
    return pd.DataFrame([{
        "dataset": dataset,
        "method": "CASCAT-MR-CMI-E-Gate_v7.4.3.2_Truth-First",
        "n_roots": len(roots),
        "n_strong_or_usable_roots": int(root_validity["root_validity_status"].isin(["strong", "usable_with_review"]).sum()) if not root_validity.empty else 0,
        "n_level0_insufficient_support": int((levels == 0).sum()),
        "n_level1_structurally_plausible": int((levels == 1).sum()),
        "n_level2_process_supported": int((levels == 2).sum()),
        "n_level3_external_supported": int((levels >= 3).sum()),
        "trajectory_status_counts": str(status_counts),
        "n_edges_rejected_by_truth_gate": n_failed_edges,
        "n_root_path_rescue_edges_rejected": n_rescue_failed,
        "n_distinct_process_pairs_supported": n_distinct,
        "overall_truth_first_status": overall,
        "core_principle": "no trajectory claim without sufficient root, edge, stability, and molecular-process evidence",
        "validation_scope": "internal_truth_first_evidence_gates_not_experimental_validation",
    }])


def summarise_backbone_edges(edges: pd.DataFrame, roots: list[int], adata) -> pd.DataFrame:
    rows = []
    for k, root in enumerate(roots):
        sub = edges.loc[pd.to_numeric(edges.get("root_pos", pd.Series(dtype=int)), errors="coerce").fillna(-1).astype(int) == k] if edges is not None and not edges.empty else pd.DataFrame()
        nodes = _tree_nodes(sub, int(root))
        rows.append({
            "root_pos": int(k),
            "root_index": int(root),
            "root_id": str(adata.obs_names[int(root)]),
            "tree_nodes": len(nodes),
            "tree_edges": int(len(sub)),
            "n_components": 1 if len(sub) > 0 else 0,
            "tree_form": "truth_first_validated_backbone" if len(sub) > 0 else "insufficient_support_no_backbone",
        })
    return pd.DataFrame(rows)
