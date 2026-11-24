#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CASCAT-MR-CMI-E-Gate v7.1 biological compatibility and sparse Markov layer.

This module implements the v7.1 structural contract used by the updated base and
progression-refined pipelines:

1. Split a global evidence graph into a trajectory graph and an interaction graph.
2. Run absorbing Markov competition only inside trajectory-compatible process components.
3. Preserve A_raw, apply a biological root mask (A_bio), optionally sharpen (A_sharp).
4. Keep primary-root assignment by default and activate at most one secondary root only
   after a strict, evidence-gated shared-state test.
5. Build basins from the sparse active-root matrix Z_active rather than from all
   memberships above a loose threshold.
6. Never repair disconnected trajectory components with automatic weak bridges.

The implementation is intentionally conservative. It relies on existing annotations when
available, writes explicit audits, and never uses benchmark truth for construction.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve


@dataclass
class V71Config:
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
    eps: float = 1e-9


# -----------------------------------------------------------------------------
# Annotation and process-domain inference
# -----------------------------------------------------------------------------

_ANNOTATION_CANDIDATES = [
    "v7_label", "v6.3_label", "v6_label", "final_label", "major_cell_type",
    "cell_type", "celltype", "annotation", "anno", "label", "labels",
    "subtype", "state", "cluster_label", "cell_state",
]

_COMPARTMENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("tumor_epithelial", [
        "proliferative tumor", "cycling tumor", "tumor", "malignant", "cancer",
        "carcinoma", "epithelial", "epithelial-like",
    ]),
    ("lymphoid", [
        "t/nk", "tnk", "regulatory t", "treg", "cd4", "cd8", "t cell",
        "t-cell", "nk cell", "natural killer", "lymphocyte", "lymphoid",
    ]),
    ("myeloid", [
        "macrophage", "macro", "monocyte", "myeloid", "dendritic", "dc",
        "neutrophil", "mast cell",
    ]),
    ("endothelial_vascular", [
        "lymphatic endothelial", "endothelial", "panvascular", "vascular",
        "blood vessel", "angiogenic",
    ]),
    ("stromal_fibroblast", [
        "fibroblast", "myofibroblast", "caf", "stromal", "matrix",
    ]),
    ("perivascular_smooth_muscle", [
        "pericyte", "smooth muscle", "mural", "contractile",
    ]),
    ("microenvironment_ambiguous", [
        "microenvironment-associated", "microenvironment associated", "mixed",
        "ambiguous", "unknown", "unassigned", "uncertain",
    ]),
]


def _cfg(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def find_annotation_column(adata, preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in adata.obs.columns:
        return preferred
    lower = {str(c).lower(): str(c) for c in adata.obs.columns}
    for candidate in _ANNOTATION_CANDIDATES:
        if candidate in adata.obs.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def infer_major_compartment_from_label(label: object) -> str:
    text = str(label).strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    for compartment, patterns in _COMPARTMENT_PATTERNS:
        if any(pattern in text for pattern in patterns):
            return compartment
    return "unknown"


def process_scope_from_compartment(compartment: str) -> str:
    """Return the conservative biological process scope used for root competition.

    v7.1 deliberately keeps major processes separate. Conditional cross-process
    transitions can be re-enabled only through explicit trajectory evidence; they are
    not part of default Markov competition.
    """
    mapping = {
        "tumor_epithelial": "tumor_epithelial_process",
        "lymphoid": "lymphoid_process",
        "myeloid": "myeloid_process",
        "endothelial_vascular": "endothelial_vascular_process",
        "stromal_fibroblast": "stromal_fibroblast_process",
        "perivascular_smooth_muscle": "perivascular_smooth_muscle_process",
        "microenvironment_ambiguous": "ambiguous_process",
        "unknown": "unknown_process",
    }
    return mapping.get(str(compartment), "unknown_process")


def infer_node_biological_domains(adata, annotation_key: Optional[str] = None) -> pd.DataFrame:
    key = find_annotation_column(adata, annotation_key)
    if key is None:
        labels = np.asarray(["unknown"] * adata.n_obs, dtype=object)
    else:
        labels = adata.obs[key].astype(str).to_numpy()
    compartments = [infer_major_compartment_from_label(x) for x in labels]
    scopes = [process_scope_from_compartment(x) for x in compartments]
    return pd.DataFrame({
        "node_id": np.arange(adata.n_obs, dtype=int),
        "node_name": adata.obs_names.astype(str),
        "annotation_key": key or "",
        "annotation_label": labels,
        "major_compartment": compartments,
        "process_scope": scopes,
    })


# -----------------------------------------------------------------------------
# Biological edge eligibility and dual-graph split
# -----------------------------------------------------------------------------

_ALLOWED_SAME = {
    "tumor_epithelial", "lymphoid", "myeloid", "endothelial_vascular",
    "stromal_fibroblast", "perivascular_smooth_muscle",
}

_CONDITIONAL_PAIRS = {
    tuple(sorted(("endothelial_vascular", "perivascular_smooth_muscle"))),
    tuple(sorted(("tumor_epithelial", "stromal_fibroblast"))),
}


def trajectory_compatibility_score(source_domain: str, target_domain: str) -> tuple[float, str]:
    a = str(source_domain)
    b = str(target_domain)
    if a == b and a in _ALLOWED_SAME:
        return 1.0, "allowed_same_process_domain"
    if a in {"unknown", "microenvironment_ambiguous"} or b in {"unknown", "microenvironment_ambiguous"}:
        return 0.5, "conditional_annotation_uncertainty"
    if tuple(sorted((a, b))) in _CONDITIONAL_PAIRS:
        return 0.5, "conditional_transition_program_required"
    return 0.0, "forbidden_cross_process_trajectory"


def annotate_edge_biological_compatibility(
    edges_df: pd.DataFrame,
    node_domains: pd.DataFrame,
    cfg: Any,
) -> pd.DataFrame:
    if edges_df is None or edges_df.empty:
        return pd.DataFrame()
    domain_lookup = node_domains.set_index("node_id")["major_compartment"].to_dict()
    scope_lookup = node_domains.set_index("node_id")["process_scope"].to_dict()
    label_lookup = node_domains.set_index("node_id")["annotation_label"].to_dict()
    usable_domains = {d for d in domain_lookup.values() if d not in {"unknown", "microenvironment_ambiguous"}}
    annotation_fallback = len(usable_domains) == 0
    allow_conditional = bool(_cfg(cfg, "allow_conditional_trajectory_edges", False))
    min_conf = float(_cfg(cfg, "min_trajectory_edge_confidence", 0.0))
    rows: list[dict[str, Any]] = []
    for row in edges_df.itertuples(index=False):
        i, j = int(row.source), int(row.target)
        di, dj = domain_lookup.get(i, "unknown"), domain_lookup.get(j, "unknown")
        si, sj = scope_lookup.get(i, "unknown_process"), scope_lookup.get(j, "unknown_process")
        if annotation_fallback:
            kappa, reason = 1.0, "allowed_annotation_unavailable_fallback"
        else:
            kappa, reason = trajectory_compatibility_score(di, dj)
        base_conf = float(getattr(row, "S_global", 1.0))
        if kappa >= 1.0:
            eligible = base_conf >= min_conf
            role = "trajectory_allowed" if eligible else "interaction_only_low_confidence"
        elif kappa > 0.0:
            eligible = allow_conditional and base_conf >= min_conf
            role = "trajectory_conditional" if eligible else "interaction_only_conditional"
        else:
            eligible = False
            role = "interaction_only_forbidden_as_trajectory"
        rec = row._asdict()
        rec.update({
            "source_compartment": di,
            "target_compartment": dj,
            "source_process_scope": si,
            "target_process_scope": sj,
            "source_annotation_label": label_lookup.get(i, ""),
            "target_annotation_label": label_lookup.get(j, ""),
            "trajectory_compatibility": float(kappa),
            "trajectory_edge_eligible": bool(eligible),
            "bio_compatibility_reason": reason,
            "edge_graph_role": role,
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def split_edges(annotated_edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if annotated_edges is None or annotated_edges.empty:
        empty = pd.DataFrame()
        return empty, empty
    traj = annotated_edges.loc[annotated_edges["trajectory_edge_eligible"].astype(bool)].copy()
    # Interaction graph keeps every edge that is not a fully allowed trajectory edge;
    # conditional edges are retained here even if explicitly allowed in trajectory.
    inter = annotated_edges.loc[
        (~annotated_edges["trajectory_edge_eligible"].astype(bool))
        | (annotated_edges["trajectory_compatibility"].astype(float) < 1.0)
    ].copy()
    return traj.reset_index(drop=True), inter.reset_index(drop=True)


def sparse_similarity_from_edges(
    n_nodes: int,
    edges_df: pd.DataFrame,
    value_col: str = "S_global",
) -> csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    if edges_df is None or edges_df.empty:
        return csr_matrix((n_nodes, n_nodes), dtype=float)
    for row in edges_df.itertuples(index=False):
        i, j = int(row.source), int(row.target)
        value = float(getattr(row, value_col, 1.0))
        if not np.isfinite(value) or value <= 0:
            continue
        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([value, value])
    return csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))


# -----------------------------------------------------------------------------
# Process-constrained absorbing Markov assignment
# -----------------------------------------------------------------------------


def _absorbing_membership_on_component(
    S_sub: csr_matrix,
    local_roots: list[int],
    beta_transition: float,
    eps: float,
) -> np.ndarray:
    n = S_sub.shape[0]
    K = len(local_roots)
    A = np.zeros((n, K), dtype=float)
    if K == 0:
        return A
    if K == 1:
        A[:, 0] = 1.0
        return A

    coo = S_sub.tocoo()
    # Guard exponentials from overflow while preserving ranking.
    scaled = np.clip(beta_transition * coo.data, -50.0, 50.0)
    Praw = csr_matrix((np.exp(scaled), (coo.row, coo.col)), shape=S_sub.shape)
    row_sum = np.asarray(Praw.sum(axis=1)).ravel()
    row_sum[row_sum <= eps] = 1.0
    P = sparse.diags(1.0 / row_sum) @ Praw

    roots_arr = np.asarray(local_roots, dtype=int)
    is_root = np.zeros(n, dtype=bool)
    is_root[roots_arr] = True
    transient = np.flatnonzero(~is_root)
    for k, r in enumerate(local_roots):
        A[r, k] = 1.0
    if len(transient) == 0:
        return A

    PTT = P[transient][:, transient].tocsr()
    PTR = P[transient][:, roots_arr].toarray()
    M = identity(len(transient), format="csr") - PTT
    try:
        B = np.asarray(spsolve(M, PTR))
        if B.ndim == 1:
            B = B[:, None]
        B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
        B[B < 0] = 0.0
        denom = B.sum(axis=1, keepdims=True)
        valid = denom[:, 0] > eps
        B[valid] /= denom[valid]
        # If the linear solve leaves a zero row, use inverse shortest-path softmax
        # inside the component rather than silently assigning root 0.
        if np.any(~valid):
            D = S_sub.copy()
            D.data = 1.0 / (D.data + eps)
            from scipy.sparse.csgraph import shortest_path
            dist = np.asarray(shortest_path(D, directed=False, indices=roots_arr)).T
            logits = -beta_transition * dist
            finite = np.isfinite(logits)
            if finite.any():
                floor = float(np.nanmin(logits[finite])) - 50.0
                logits[~finite] = floor
            logits -= np.max(logits, axis=1, keepdims=True)
            soft = np.exp(np.clip(logits, -50.0, 50.0))
            soft /= soft.sum(axis=1, keepdims=True) + eps
            B[~valid] = soft[transient][~valid]
        A[transient] = B
        return A
    except Exception:
        from scipy.sparse.csgraph import shortest_path
        D = S_sub.copy()
        D.data = 1.0 / (D.data + eps)
        dist = np.asarray(shortest_path(D, directed=False, indices=roots_arr)).T
        logits = -beta_transition * dist
        finite = np.isfinite(logits)
        if finite.any():
            floor = float(np.nanmin(logits[finite])) - 50.0
            logits[~finite] = floor
        logits -= np.max(logits, axis=1, keepdims=True)
        soft = np.exp(np.clip(logits, -50.0, 50.0))
        soft /= soft.sum(axis=1, keepdims=True) + eps
        for k, r in enumerate(local_roots):
            soft[r, :] = 0.0
            soft[r, k] = 1.0
        return soft


def process_constrained_absorbing_markov(
    n_nodes: int,
    trajectory_edges: pd.DataFrame,
    roots: list[int],
    node_domains: pd.DataFrame,
    cfg: Any,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Return A_raw, biological root mask, component audit, and root scope table."""
    K = len(roots)
    eps = float(_cfg(cfg, "eps", 1e-9))
    beta = float(_cfg(cfg, "beta_transition", 4.0))
    A_raw = np.zeros((n_nodes, K), dtype=float)
    root_mask = np.zeros((n_nodes, K), dtype=bool)

    scope_by_node = node_domains.set_index("node_id")["process_scope"].to_dict()
    comp_by_node = node_domains.set_index("node_id")["major_compartment"].to_dict()
    root_scope_rows = []
    for k, root in enumerate(roots):
        scope = scope_by_node.get(int(root), "unknown_process")
        compartment = comp_by_node.get(int(root), "unknown")
        for i in range(n_nodes):
            root_mask[i, k] = scope_by_node.get(i, "unknown_process") == scope
        root_mask[int(root), k] = True
        root_scope_rows.append({
            "root_pos": int(k),
            "root_index": int(root),
            "root_process_scope": scope,
            "root_compartment": compartment,
            "root_scope_size": int(root_mask[:, k].sum()),
        })

    # Build separate graph components inside each process scope. This is stricter
    # than global connected components and prevents roots from different biological
    # processes from competing even when a conditional/global evidence edge exists.
    edges = trajectory_edges if trajectory_edges is not None else pd.DataFrame()
    component_rows: list[dict[str, Any]] = []
    processed_nodes: set[int] = set()
    all_scopes = sorted(set(scope_by_node.values()))
    S_global = sparse_similarity_from_edges(n_nodes, edges, "S_global")

    for scope in all_scopes:
        scope_nodes = sorted(i for i in range(n_nodes) if scope_by_node.get(i) == scope)
        if not scope_nodes:
            continue
        G = nx.Graph()
        G.add_nodes_from(scope_nodes)
        if not edges.empty:
            sub_edges = edges.loc[
                edges["source"].astype(int).isin(scope_nodes)
                & edges["target"].astype(int).isin(scope_nodes)
            ]
            for r in sub_edges.itertuples(index=False):
                G.add_edge(int(r.source), int(r.target))
        for comp_idx, comp in enumerate(nx.connected_components(G)):
            nodes = sorted(map(int, comp))
            processed_nodes.update(nodes)
            comp_roots = [(k, int(r)) for k, r in enumerate(roots) if int(r) in comp]
            status = "no_root_unassigned"
            if len(comp_roots) == 1:
                k, _ = comp_roots[0]
                A_raw[nodes, k] = 1.0
                status = "single_root_component"
            elif len(comp_roots) >= 2:
                idx_map = {node: pos for pos, node in enumerate(nodes)}
                local_roots = [idx_map[r] for _, r in comp_roots]
                S_sub = S_global[nodes][:, nodes].tocsr()
                local_A = _absorbing_membership_on_component(S_sub, local_roots, beta, eps)
                for local_k, (global_k, _) in enumerate(comp_roots):
                    A_raw[nodes, global_k] = local_A[:, local_k]
                status = "multi_root_markov_component"
            component_rows.append({
                "process_scope": scope,
                "component_id": f"{scope}__C{comp_idx}",
                "n_nodes": int(len(nodes)),
                "n_roots": int(len(comp_roots)),
                "root_positions": "|".join(str(k) for k, _ in comp_roots),
                "root_indices": "|".join(str(r) for _, r in comp_roots),
                "assignment_status": status,
            })

    # Root anchors are exact even if annotation is uncertain.
    for k, root in enumerate(roots):
        A_raw[int(root), :] = 0.0
        A_raw[int(root), k] = 1.0

    return A_raw, root_mask, pd.DataFrame(component_rows), pd.DataFrame(root_scope_rows)


def apply_biological_root_mask(
    A_raw: np.ndarray,
    root_mask: np.ndarray,
    roots: list[int],
    eps: float = 1e-9,
) -> np.ndarray:
    A = np.asarray(A_raw, dtype=float).copy()
    if A.shape != root_mask.shape:
        raise ValueError(f"A_raw shape {A.shape} != root_mask shape {root_mask.shape}")
    A[~root_mask] = 0.0
    denom = A.sum(axis=1, keepdims=True)
    valid = denom[:, 0] > eps
    A[valid] /= denom[valid]
    A[~valid] = 0.0
    for k, root in enumerate(roots):
        A[int(root), :] = 0.0
        A[int(root), k] = 1.0
    return A


def sharpen_membership(A_bio: np.ndarray, q: float, eps: float = 1e-9) -> np.ndarray:
    q = max(1.0, float(q))
    A = np.asarray(A_bio, dtype=float)
    if q == 1.0:
        return A.copy()
    powered = np.power(np.clip(A, 0.0, 1.0), q)
    denom = powered.sum(axis=1, keepdims=True)
    valid = denom[:, 0] > eps
    out = np.zeros_like(powered)
    out[valid] = powered[valid] / denom[valid]
    return out


def safe_primary_roots(A: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    if A.size == 0:
        return np.asarray([], dtype=int)
    primary = np.argmax(A, axis=1).astype(int)
    primary[np.sum(A, axis=1) <= eps] = -1
    return primary


def _normalized_entropy_row(row: np.ndarray, eps: float) -> float:
    positive = row > eps
    k = int(positive.sum())
    if k <= 1:
        return 0.0
    vals = row[positive]
    h = -float(np.sum(vals * np.log(vals + eps)))
    return h / math.log(k)


def _trajectory_adjacency(edges_df: pd.DataFrame, n_nodes: int) -> list[set[int]]:
    adj = [set() for _ in range(n_nodes)]
    if edges_df is None or edges_df.empty:
        return adj
    for row in edges_df.itertuples(index=False):
        i, j = int(row.source), int(row.target)
        if 0 <= i < n_nodes and 0 <= j < n_nodes:
            adj[i].add(j)
            adj[j].add(i)
    return adj


def _initial_shared_evidence(
    i: int,
    secondary: int,
    primary: np.ndarray,
    adjacency: list[set[int]],
) -> dict[str, bool]:
    local_contact = any(primary[nb] == secondary for nb in adjacency[i])
    return {
        "local_cross_root_trajectory_contact": bool(local_contact),
    }


def build_sparse_active_assignment(
    A_sharp: np.ndarray,
    roots: list[int],
    node_domains: pd.DataFrame,
    trajectory_edges: pd.DataFrame,
    cfg: Any,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Primary-only default assignment with a strict secondary activation gate."""
    n, K = A_sharp.shape
    eps = float(_cfg(cfg, "eps", 1e-9))
    theta_secondary = float(_cfg(cfg, "theta_secondary", 0.25))
    theta_entropy = float(_cfg(cfg, "theta_entropy", 0.70))
    ratio_threshold = float(_cfg(cfg, "shared_ratio_threshold", 0.65))
    margin_threshold = float(_cfg(cfg, "shared_margin_threshold", 0.20))
    min_evidence = int(_cfg(cfg, "min_shared_evidence_channels", 1))
    max_active = max(1, int(_cfg(cfg, "max_active_roots_per_node", 2)))

    primary = safe_primary_roots(A_sharp, eps)
    adjacency = _trajectory_adjacency(trajectory_edges, n)
    scope_by_node = node_domains.set_index("node_id")["process_scope"].to_dict()
    root_scopes = [scope_by_node.get(int(r), "unknown_process") for r in roots]
    Z = np.zeros((n, K), dtype=np.int8)
    rows: list[dict[str, Any]] = []

    for i in range(n):
        p = int(primary[i])
        if p < 0:
            rows.append({
                "node_id": i,
                "primary_root": -1,
                "secondary_root": -1,
                "primary_probability": 0.0,
                "secondary_probability": 0.0,
                "top2_margin": np.nan,
                "shared_ratio": 0.0,
                "entropy_norm": 0.0,
                "n_competing_roots": 0,
                "shared_evidence_channel_count": 0,
                "secondary_activated": False,
                "assignment_status": "unassigned",
                "evidence_channels": "",
            })
            continue

        order = np.argsort(-A_sharp[i])
        s = int(order[1]) if K > 1 else -1
        p1 = float(A_sharp[i, p])
        p2 = float(A_sharp[i, s]) if s >= 0 else 0.0
        margin = p1 - p2
        ratio = p2 / (p1 + eps)
        entropy = _normalized_entropy_row(A_sharp[i], eps)
        n_competing = int((A_sharp[i] >= theta_secondary).sum())
        Z[i, p] = 1

        evidence = _initial_shared_evidence(i, s, primary, adjacency) if s >= 0 else {}
        evidence_count = int(sum(bool(v) for v in evidence.values()))
        prob_pass = bool(s >= 0 and p2 >= theta_secondary and ratio >= ratio_threshold)
        uncertainty_pass = bool(entropy >= theta_entropy or margin <= margin_threshold)
        bio_pass = bool(
            s >= 0
            and root_scopes[p] == root_scopes[s]
            and scope_by_node.get(i, "unknown_process") == root_scopes[p]
        )
        evidence_pass = evidence_count >= min_evidence
        activate_secondary = bool(
            max_active >= 2 and prob_pass and uncertainty_pass and bio_pass and evidence_pass
        )
        if activate_secondary:
            Z[i, s] = 1

        if n_competing >= 3 and not activate_secondary:
            status = "high_order_ambiguity"
        elif activate_secondary:
            status = "validated_dual_root_shared"
        else:
            status = "root_specific"

        rows.append({
            "node_id": i,
            "primary_root": p,
            "secondary_root": s,
            "primary_probability": p1,
            "secondary_probability": p2,
            "top2_margin": margin,
            "shared_ratio": ratio,
            "entropy_norm": entropy,
            "n_competing_roots": n_competing,
            "probability_pass": prob_pass,
            "uncertainty_pass": uncertainty_pass,
            "biological_process_pass": bio_pass,
            "shared_evidence_channel_count": evidence_count,
            "secondary_activated": activate_secondary,
            "assignment_status": status,
            "evidence_channels": "|".join(k for k, v in evidence.items() if v),
            **evidence,
        })

    for k, root in enumerate(roots):
        Z[int(root), :] = 0
        Z[int(root), k] = 1
    return Z, pd.DataFrame(rows)


def refine_sparse_active_assignment(
    A_sharp: np.ndarray,
    roots: list[int],
    node_domains: pd.DataFrame,
    trajectory_edges: pd.DataFrame,
    cfg: Any,
    progression_scores: Optional[dict[int, np.ndarray]] = None,
    cmi_e_support: Optional[pd.DataFrame] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Re-evaluate secondary activation with progression and CMI-E evidence.

    This function still defaults to primary-only and activates at most one secondary root.
    Markov ambiguity alone is never sufficient.
    """
    n, K = A_sharp.shape
    eps = float(_cfg(cfg, "eps", 1e-9))
    theta_secondary = float(_cfg(cfg, "theta_secondary", 0.25))
    theta_entropy = float(_cfg(cfg, "theta_entropy", 0.70))
    ratio_threshold = float(_cfg(cfg, "shared_ratio_threshold", 0.65))
    margin_threshold = float(_cfg(cfg, "shared_margin_threshold", 0.20))
    min_evidence = int(_cfg(cfg, "min_shared_evidence_channels", 1))
    max_active = max(1, int(_cfg(cfg, "max_active_roots_per_node", 2)))
    prog_gap_thr = float(_cfg(cfg, "shared_progression_max_gap", 0.20))
    cmi_thr = float(_cfg(cfg, "shared_cmi_trust_threshold", 0.50))

    progression_scores = progression_scores or {}
    primary = safe_primary_roots(A_sharp, eps)
    adjacency = _trajectory_adjacency(trajectory_edges, n)
    scope_by_node = node_domains.set_index("node_id")["process_scope"].to_dict()
    root_scopes = [scope_by_node.get(int(r), "unknown_process") for r in roots]

    cmi_incident: dict[tuple[int, int], float] = {}
    if cmi_e_support is not None and not cmi_e_support.empty:
        for row in cmi_e_support.itertuples(index=False):
            rp = int(getattr(row, "root_pos", -1))
            i, j = int(row.source), int(row.target)
            trust = float(getattr(row, "cmi_env_trust", getattr(row, "cmi_e_support", 0.0)))
            cmi_incident[(i, rp)] = max(cmi_incident.get((i, rp), 0.0), trust)
            cmi_incident[(j, rp)] = max(cmi_incident.get((j, rp), 0.0), trust)

    Z = np.zeros((n, K), dtype=np.int8)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        p = int(primary[i])
        if p < 0:
            rows.append({
                "node_id": i, "primary_root": -1, "secondary_root": -1,
                "secondary_activated": False, "assignment_status": "unassigned",
                "shared_evidence_channel_count": 0, "evidence_channels": "",
            })
            continue
        order = np.argsort(-A_sharp[i])
        s = int(order[1]) if K > 1 else -1
        p1 = float(A_sharp[i, p])
        p2 = float(A_sharp[i, s]) if s >= 0 else 0.0
        margin = p1 - p2
        ratio = p2 / (p1 + eps)
        entropy = _normalized_entropy_row(A_sharp[i], eps)
        n_competing = int((A_sharp[i] >= theta_secondary).sum())
        Z[i, p] = 1

        evidence: dict[str, bool] = {}
        if s >= 0:
            evidence.update(_initial_shared_evidence(i, s, primary, adjacency))
            if p in progression_scores and s in progression_scores:
                gap = abs(float(progression_scores[p][i]) - float(progression_scores[s][i]))
                evidence["progression_compatibility"] = bool(np.isfinite(gap) and gap <= prog_gap_thr)
            else:
                gap = np.nan
                evidence["progression_compatibility"] = False
            cmi_value = max(cmi_incident.get((i, p), 0.0), cmi_incident.get((i, s), 0.0))
            evidence["environment_stable_cmi_e"] = bool(cmi_value >= cmi_thr)
        else:
            gap = np.nan
            cmi_value = 0.0

        evidence_count = int(sum(bool(v) for v in evidence.values()))
        prob_pass = bool(s >= 0 and p2 >= theta_secondary and ratio >= ratio_threshold)
        uncertainty_pass = bool(entropy >= theta_entropy or margin <= margin_threshold)
        bio_pass = bool(
            s >= 0
            and root_scopes[p] == root_scopes[s]
            and scope_by_node.get(i, "unknown_process") == root_scopes[p]
        )
        evidence_pass = evidence_count >= min_evidence
        activate_secondary = bool(
            max_active >= 2 and prob_pass and uncertainty_pass and bio_pass and evidence_pass
        )
        if activate_secondary:
            Z[i, s] = 1

        if n_competing >= 3 and not activate_secondary:
            status = "high_order_ambiguity"
        elif activate_secondary:
            status = "validated_dual_root_shared"
        else:
            status = "root_specific"
        rows.append({
            "node_id": i,
            "primary_root": p,
            "secondary_root": s,
            "primary_probability": p1,
            "secondary_probability": p2,
            "top2_margin": margin,
            "shared_ratio": ratio,
            "entropy_norm": entropy,
            "n_competing_roots": n_competing,
            "progression_gap_primary_secondary": gap,
            "max_incident_cmi_env_trust": cmi_value,
            "probability_pass": prob_pass,
            "uncertainty_pass": uncertainty_pass,
            "biological_process_pass": bio_pass,
            "shared_evidence_channel_count": evidence_count,
            "secondary_activated": activate_secondary,
            "assignment_status": status,
            "evidence_channels": "|".join(k for k, v in evidence.items() if v),
            **evidence,
        })

    for k, root in enumerate(roots):
        Z[int(root), :] = 0
        Z[int(root), k] = 1
    return Z, pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Outputs and basin helpers
# -----------------------------------------------------------------------------


def _membership_long(A: np.ndarray, roots: list[int], adata, value_name: str) -> pd.DataFrame:
    rows = []
    for i in range(A.shape[0]):
        for k, root in enumerate(roots):
            rows.append({
                "node_id": i,
                "node_name": str(adata.obs_names[i]),
                "root_pos": k,
                "root_index": int(root),
                "root_name": str(adata.obs_names[int(root)]),
                value_name: float(A[i, k]),
            })
    return pd.DataFrame(rows)


def write_assignment_outputs(
    outdir: Path,
    adata,
    roots: list[int],
    A_raw: np.ndarray,
    A_bio: np.ndarray,
    A_sharp: np.ndarray,
    Z_active: np.ndarray,
    diagnostics: pd.DataFrame,
    stage: str = "initial",
) -> dict[str, str]:
    outdir = Path(outdir)
    suffix = "" if stage == "initial" else f"_{stage}"
    files = {
        "A_raw": f"membership_A_raw{suffix}.csv",
        "A_bio": f"membership_A_bio{suffix}.csv",
        "A_sharp": f"membership_A_sharp{suffix}.csv",
        "active": f"active_root_assignment{suffix}.csv",
        "diagnostics": f"assignment_uncertainty_diagnostics{suffix}.csv",
        "summary": f"assignment_sparsity_summary{suffix}.csv",
    }
    _membership_long(A_raw, roots, adata, "A_raw").to_csv(outdir / files["A_raw"], index=False)
    _membership_long(A_bio, roots, adata, "A_bio").to_csv(outdir / files["A_bio"], index=False)
    _membership_long(A_sharp, roots, adata, "A_sharp").to_csv(outdir / files["A_sharp"], index=False)

    active_rows = []
    for i in range(Z_active.shape[0]):
        for k, root in enumerate(roots):
            active_rows.append({
                "node_id": i,
                "node_name": str(adata.obs_names[i]),
                "root_pos": k,
                "root_index": int(root),
                "root_name": str(adata.obs_names[int(root)]),
                "is_active": bool(Z_active[i, k]),
            })
    pd.DataFrame(active_rows).to_csv(outdir / files["active"], index=False)
    diagnostics.to_csv(outdir / files["diagnostics"], index=False)

    active_count = Z_active.sum(axis=1)
    summary = pd.DataFrame([{ 
        "stage": stage,
        "n_nodes": int(Z_active.shape[0]),
        "n_roots": int(Z_active.shape[1]),
        "n_unassigned": int((active_count == 0).sum()),
        "n_single_root": int((active_count == 1).sum()),
        "n_dual_root": int((active_count == 2).sum()),
        "n_gt2_active": int((active_count > 2).sum()),
        "dual_root_rate": float((active_count == 2).mean()) if len(active_count) else 0.0,
        "mean_active_roots": float(active_count.mean()) if len(active_count) else 0.0,
    }])
    summary.to_csv(outdir / files["summary"], index=False)
    return files


def load_membership_matrix(path: Path, value_col: str, n_nodes: int, n_roots: int) -> np.ndarray:
    table = pd.read_csv(path)
    A = np.zeros((n_nodes, n_roots), dtype=float)
    for row in table.itertuples(index=False):
        A[int(row.node_id), int(row.root_pos)] = float(getattr(row, value_col))
    return A


def load_active_assignment(path: Path, n_nodes: int, n_roots: int) -> np.ndarray:
    table = pd.read_csv(path)
    Z = np.zeros((n_nodes, n_roots), dtype=np.int8)
    for row in table.itertuples(index=False):
        if bool(row.is_active):
            Z[int(row.node_id), int(row.root_pos)] = 1
    return Z


def basin_nodes_from_active(
    Z_active: np.ndarray,
    root_pos: int,
    root_idx: int,
    compatibility_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    keep = Z_active[:, root_pos].astype(bool)
    if compatibility_mask is not None:
        keep &= compatibility_mask[:, root_pos].astype(bool)
    nodes = set(map(int, np.flatnonzero(keep)))
    nodes.add(int(root_idx))
    return np.asarray(sorted(nodes), dtype=int)


def keep_root_component_only(graph: nx.Graph, root_idx: int) -> tuple[nx.Graph, dict[str, int]]:
    if graph.number_of_nodes() == 0:
        return graph.copy(), {
            "n_components_before": 0,
            "dropped_nonroot_components": 0,
            "dropped_nonroot_nodes": 0,
        }
    n_components = nx.number_connected_components(graph)
    if root_idx not in graph:
        return nx.Graph(), {
            "n_components_before": int(n_components),
            "dropped_nonroot_components": int(n_components),
            "dropped_nonroot_nodes": int(graph.number_of_nodes()),
        }
    root_comp = set(nx.node_connected_component(graph, int(root_idx)))
    return graph.subgraph(root_comp).copy(), {
        "n_components_before": int(n_components),
        "dropped_nonroot_components": int(max(0, n_components - 1)),
        "dropped_nonroot_nodes": int(graph.number_of_nodes() - len(root_comp)),
    }


def audit_tree_biological_validity(tree_edges: pd.DataFrame, node_domains: pd.DataFrame) -> pd.DataFrame:
    if tree_edges is None or tree_edges.empty:
        return pd.DataFrame()
    domain_lookup = node_domains.set_index("node_id")["major_compartment"].to_dict()
    rows = []
    for row in tree_edges.itertuples(index=False):
        i, j = int(row.source), int(row.target)
        di, dj = domain_lookup.get(i, "unknown"), domain_lookup.get(j, "unknown")
        kappa, reason = trajectory_compatibility_score(di, dj)
        rows.append({
            "root_pos": int(getattr(row, "root_pos", -1)),
            "source": i,
            "target": j,
            "source_compartment": di,
            "target_compartment": dj,
            "trajectory_compatibility": float(kappa),
            "is_forbidden_trajectory_edge": bool(kappa <= 0.0),
            "bio_compatibility_reason": reason,
        })
    return pd.DataFrame(rows)


def active_overlap_score(Z: np.ndarray, k: int, ell: int, tree_nodes_k: set[int], tree_nodes_l: set[int], eps: float = 1e-9) -> float:
    shared = int(np.logical_and(Z[:, k] > 0, Z[:, ell] > 0).sum())
    union = len(tree_nodes_k | tree_nodes_l)
    return float(shared / (union + eps))


def sparse_tree_relation_gate(
    contact: float,
    active_overlap: float,
    progression_compatibility: float,
    cmi_trust: float,
    cross_domain: bool,
    cfg: Any,
) -> tuple[bool, int, str, dict[str, bool]]:
    channels = {
        "E_contact": bool(contact >= float(_cfg(cfg, "relation_contact_threshold", 0.25))),
        "E_active_overlap": bool(active_overlap >= float(_cfg(cfg, "relation_active_overlap_threshold", 0.01))),
        "E_progression": bool(abs(progression_compatibility) >= float(_cfg(cfg, "relation_progression_abs_threshold", 0.30))),
        "E_CMI_E": bool(cmi_trust >= float(_cfg(cfg, "relation_cmi_trust_threshold", 0.35))),
    }
    extra_count = int(sum(v for k, v in channels.items() if k != "E_contact"))
    active = bool(channels["E_contact"] and extra_count >= int(_cfg(cfg, "min_relation_evidence_channels", 1)))
    if not active:
        label = "weak_or_independent"
    elif cross_domain:
        label = "cross_domain_ecological_interaction"
    elif progression_compatibility >= 0.30:
        label = "convergence_compatible"
    elif progression_compatibility <= -0.30:
        label = "conflict_or_collision_boundary"
    elif active_overlap > 0:
        label = "validated_shared_boundary"
    else:
        label = "promotion_like"
    return active, extra_count, label, channels
