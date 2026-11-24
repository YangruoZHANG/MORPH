#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CASCAT-MR-CMI-E-Gate v7.3 process-aware biological gate and sparse Markov layer.

v7.3 keeps the tested v7.1 output contract but changes the semantics of the
biological gate:

* nodes are metacells, not raw cells;
* compatibility uses metacell composition and root-process profiles;
* a mixed/interface root can connect to biologically compatible states on either
  side of the interface without allowing arbitrary pure cross-lineage edges;
* roots that are isolated in the trajectory graph are audited and left for the
  Truth-First layer to classify, without fabricating bridge edges;
* raw Markov probabilities remain diagnostics, while actual multi-tree inclusion
  stays sparse and evidence-gated.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

import cascat_v7_1_biological_markov as legacy_bio
from cascat_v7_1_biological_markov import *  # re-export stable helpers/output writers
from cascat_v7_3_metacell_process import COMPARTMENTS, process_scope_from_compartments


def _cfg(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def _fraction_columns(df: pd.DataFrame) -> list[str]:
    return [f"frac_{c}" for c in COMPARTMENTS if f"frac_{c}" in df.columns]


def _row_vector(row: pd.Series) -> np.ndarray:
    return np.asarray([float(row.get(f"frac_{c}", 0.0)) for c in COMPARTMENTS], dtype=float)


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= eps:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def infer_node_biological_domains(adata, annotation_key: Optional[str] = None) -> pd.DataFrame:
    """Prefer explicit metacell process profiles produced by the v7.3 metacell stage."""
    required = {"major_compartment", "process_scope"}
    if required.issubset(adata.obs.columns):
        key = annotation_key if annotation_key and annotation_key in adata.obs.columns else (
            "annotation_label" if "annotation_label" in adata.obs.columns else ""
        )
        labels = adata.obs[key].astype(str).to_numpy() if key else np.asarray(["unknown"] * adata.n_obs, dtype=object)
        out = pd.DataFrame({
            "node_id": np.arange(adata.n_obs, dtype=int),
            "node_name": adata.obs_names.astype(str),
            "annotation_key": key,
            "annotation_label": labels,
            "major_compartment": adata.obs["major_compartment"].astype(str).to_numpy(),
            "process_scope": adata.obs["process_scope"].astype(str).to_numpy(),
        })
        for col in [
            "annotation_purity", "compartment_purity", "compartment_entropy",
            "unresolved_cell_fraction", "n_cells", "source_cluster_id",
        ] + [f"frac_{c}" for c in COMPARTMENTS]:
            if col in adata.obs.columns:
                out[col] = adata.obs[col].to_numpy()
        return out
    return legacy_bio.infer_node_biological_domains(adata, annotation_key)


def apply_root_profile_overrides(
    node_domains: pd.DataFrame,
    root_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Inject root-table process evidence into mapped root metacells.

    The override only affects the root anchor rows used by the structural gate. It
    does not rewrite cell-level annotations or claim that every member cell has the
    root_type identity.
    """
    out = node_domains.copy()
    if out.empty or root_profiles is None or root_profiles.empty:
        return out
    for rr in root_profiles.itertuples(index=False):
        idx = int(rr.root_index)
        hit = out["node_id"].astype(int) == idx
        if not hit.any():
            continue
        allowed = [c for c in str(getattr(rr, "root_allowed_compartments", "")).split("|") if c]
        if not allowed:
            continue
        # Blend rather than erase existing composition. Root-type evidence has a
        # conservative 0.60 floor distributed over allowed compartments.
        existing = np.asarray([
            float(out.loc[hit, f"frac_{c}"].iloc[0]) if f"frac_{c}" in out.columns else 0.0
            for c in COMPARTMENTS
        ])
        prior = np.asarray([1.0 / len(allowed) if c in allowed else 0.0 for c in COMPARTMENTS])
        blended = 0.40 * existing + 0.60 * prior
        if blended.sum() > 0:
            blended /= blended.sum()
        for c, value in zip(COMPARTMENTS, blended):
            col = f"frac_{c}"
            if col not in out.columns:
                out[col] = 0.0
            out.loc[hit, col] = float(value)
        active = [c for c, value in zip(COMPARTMENTS, blended) if value >= 0.20]
        major = active[0] if len(active) == 1 else ("mixed_process" if len(active) >= 2 else "unknown")
        out.loc[hit, "major_compartment"] = major
        out.loc[hit, "process_scope"] = str(getattr(rr, "root_process_scope", process_scope_from_compartments(active)))
        out.loc[hit, "annotation_label"] = str(getattr(rr, "root_type", out.loc[hit, "annotation_label"].iloc[0]))
        out.loc[hit, "root_profile_override"] = True
    if "root_profile_override" not in out.columns:
        out["root_profile_override"] = False
    out["root_profile_override"] = out["root_profile_override"].map(lambda x: bool(x) if pd.notna(x) else False)
    return out



def _process_support_set(row: pd.Series, threshold: float = 0.10) -> set[str]:
    return {c for c in COMPARTMENTS if float(row.get(f"frac_{c}", 0.0)) >= threshold}


def _allowed_compartments_from_profile(root_profiles: pd.DataFrame, root_pos: int, fallback: str = "unknown") -> list[str]:
    if root_profiles is not None and not root_profiles.empty:
        hit = root_profiles.loc[root_profiles["root_pos"].astype(int) == int(root_pos)]
        if not hit.empty:
            vals = [x for x in str(hit.iloc[0].get("root_allowed_compartments", "")).split("|") if x in COMPARTMENTS]
            if vals:
                return vals
    return [fallback] if fallback in COMPARTMENTS else []


def build_root_anchor_sets(
    adata,
    roots: list[int],
    resolved_roots: pd.DataFrame,
    root_profiles: pd.DataFrame,
    node_domains: pd.DataFrame,
    trajectory_edges: pd.DataFrame,
    cfg: Any,
    embedding: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Construct disjoint local absorbing anchor sets for every biological root.

    A single mapped metacell can be unstable when it contains only one or two raw
    cells.  v7.3 therefore expands each mapped root to a small local set of
    metacells supported by process composition, expression continuity, spatial
    locality and graph reachability.  Anchors are disjoint across roots by
    default; the original mapped root is always retained.
    """
    n = int(adata.n_obs)
    if n == 0 or not roots:
        return pd.DataFrame()
    max_anchors = max(1, int(_cfg(cfg, "root_anchor_max_nodes", 7)))
    max_hops = max(1, int(_cfg(cfg, "root_anchor_max_hops", 3)))
    min_process = float(_cfg(cfg, "root_anchor_min_process_score", 0.25))
    w_process = float(_cfg(cfg, "root_anchor_w_process", 0.40))
    w_expr = float(_cfg(cfg, "root_anchor_w_expr", 0.25))
    w_spatial = float(_cfg(cfg, "root_anchor_w_spatial", 0.20))
    w_graph = float(_cfg(cfg, "root_anchor_w_graph", 0.15))

    lookup = node_domains.set_index("node_id")
    coords = np.asarray(adata.obsm["spatial"], dtype=float) if "spatial" in adata.obsm else np.zeros((n, 2))
    if embedding is None:
        if "X_pca" in adata.obsm:
            embedding = np.asarray(adata.obsm["X_pca"], dtype=float)
        else:
            X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X, dtype=float)
            embedding = X[:, : min(30, X.shape[1])]
    emb = np.asarray(embedding, dtype=float)
    emb_mu = np.nanmean(emb, axis=0, keepdims=True)
    emb_sd = np.nanstd(emb, axis=0, keepdims=True)
    emb_sd[emb_sd < 1e-9] = 1.0
    emb_z = np.nan_to_num((emb - emb_mu) / emb_sd, nan=0.0, posinf=0.0, neginf=0.0)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    if trajectory_edges is not None and not trajectory_edges.empty:
        for r in trajectory_edges.itertuples(index=False):
            G.add_edge(int(r.source), int(r.target), weight=max(float(getattr(r, "d_global", 1.0)), 1e-9))

    raw_candidates: dict[int, list[dict[str, Any]]] = {}
    for k, root in enumerate(roots):
        root = int(root)
        fallback = str(lookup.loc[root].get("major_compartment", "unknown")) if root in lookup.index else "unknown"
        allowed = _allowed_compartments_from_profile(root_profiles, k, fallback)
        # Graph-local candidates within a small hop radius.
        hop = nx.single_source_shortest_path_length(G, root, cutoff=max_hops) if root in G else {root: 0}
        candidate_nodes = sorted(hop)
        if root not in candidate_nodes:
            candidate_nodes.insert(0, root)

        d_sp = np.linalg.norm(coords[candidate_nodes] - coords[root][None, :], axis=1) if len(candidate_nodes) else np.asarray([])
        pos = d_sp[d_sp > 0]
        sp_scale = float(np.median(pos)) if len(pos) else 1.0
        root_vec = emb_z[root]
        rows = []
        for node, ds in zip(candidate_nodes, d_sp):
            if node not in lookup.index:
                continue
            row = lookup.loc[node]
            vec = _row_vector(row)
            process_score = float(sum(vec[COMPARTMENTS.index(c)] for c in allowed if c in COMPARTMENTS))
            if node == root:
                process_score = max(process_score, 1.0)
            if node != root and process_score < min_process:
                continue
            denom = float(np.linalg.norm(root_vec) * np.linalg.norm(emb_z[node]))
            expr_sim = float(max(np.dot(root_vec, emb_z[node]) / denom, 0.0)) if denom > 1e-12 else 0.0
            spatial_sim = float(math.exp(-float(ds) / (sp_scale + 1e-9)))
            graph_sim = 1.0 / (1.0 + float(hop.get(node, max_hops + 1)))
            score = w_process * process_score + w_expr * expr_sim + w_spatial * spatial_sim + w_graph * graph_sim
            if node == root:
                score = max(score, 1.0)
            rows.append({
                "root_pos": int(k),
                "root_index": int(root),
                "root_name": str(adata.obs_names[root]),
                "anchor_node": int(node),
                "anchor_name": str(adata.obs_names[int(node)]),
                "anchor_score": float(score),
                "process_score": float(process_score),
                "expression_similarity": float(expr_sim),
                "spatial_similarity": float(spatial_sim),
                "graph_hops": int(hop.get(node, max_hops + 1)),
                "is_mapped_root": bool(node == root),
                "allowed_compartments": "|".join(allowed),
            })
        rows.sort(key=lambda r: (-float(r["anchor_score"]), int(r["graph_hops"]), int(r["anchor_node"])))
        root_row = next((r for r in rows if r["is_mapped_root"]), None)
        chosen = []
        if root_row is not None:
            chosen.append(root_row)
        for r in rows:
            if r["is_mapped_root"]:
                continue
            if len(chosen) >= max_anchors:
                break
            chosen.append(r)
        raw_candidates[k] = chosen

    # Resolve anchor collisions by strongest support, while mapped roots stay fixed.
    mapped_roots = {int(r): k for k, r in enumerate(roots)}
    ownership: dict[int, tuple[int, float]] = {}
    for k, rows in raw_candidates.items():
        for r in rows:
            node = int(r["anchor_node"])
            if node in mapped_roots:
                owner = mapped_roots[node]
                if owner == k:
                    ownership[node] = (k, 1e9)
                continue
            score = float(r["anchor_score"])
            if node not in ownership or score > ownership[node][1]:
                ownership[node] = (k, score)

    out = []
    for k, rows in raw_candidates.items():
        kept = [r for r in rows if ownership.get(int(r["anchor_node"]), (k, 0.0))[0] == k or bool(r["is_mapped_root"])]
        kept.sort(key=lambda r: (not bool(r["is_mapped_root"]), -float(r["anchor_score"]), int(r["anchor_node"])))
        for rank, r in enumerate(kept[:max_anchors]):
            rec = dict(r)
            rec["anchor_rank"] = int(rank)
            rec["anchor_set_size"] = int(min(len(kept), max_anchors))
            out.append(rec)
    return pd.DataFrame(out)


def compute_root_local_affinity(
    adata,
    roots: list[int],
    root_profiles: pd.DataFrame,
    node_domains: pd.DataFrame,
    trajectory_edges: pd.DataFrame,
    root_anchor_sets: pd.DataFrame,
    cfg: Any,
    embedding: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute root-local process affinity used to restrict Markov competition.

    The score combines process composition, expression continuity, spatial locality
    and graph proximity to the root anchor set.  It is not a biological truth
    score; it is a conservative structural prior preventing same-process roots in a
    giant connected component from globally competing everywhere.
    """
    n, K = int(adata.n_obs), len(roots)
    lookup = node_domains.set_index("node_id")
    coords = np.asarray(adata.obsm["spatial"], dtype=float) if "spatial" in adata.obsm else np.zeros((n, 2))
    if embedding is None:
        embedding = np.asarray(adata.obsm["X_pca"], dtype=float) if "X_pca" in adata.obsm else (adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X, dtype=float))
    emb = np.asarray(embedding, dtype=float)
    emb_mu = np.nanmean(emb, axis=0, keepdims=True)
    emb_sd = np.nanstd(emb, axis=0, keepdims=True)
    emb_sd[emb_sd < 1e-9] = 1.0
    emb_z = np.nan_to_num((emb - emb_mu) / emb_sd, nan=0.0, posinf=0.0, neginf=0.0)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    if trajectory_edges is not None and not trajectory_edges.empty:
        for r in trajectory_edges.itertuples(index=False):
            G.add_edge(int(r.source), int(r.target), weight=max(float(getattr(r, "d_global", 1.0)), 1e-9))

    w_process = float(_cfg(cfg, "root_affinity_w_process", 0.35))
    w_expr = float(_cfg(cfg, "root_affinity_w_expr", 0.25))
    w_spatial = float(_cfg(cfg, "root_affinity_w_spatial", 0.20))
    w_graph = float(_cfg(cfg, "root_affinity_w_graph", 0.20))
    A = np.zeros((n, K), dtype=float)
    rows = []

    for k, root in enumerate(roots):
        root = int(root)
        anchors = []
        if root_anchor_sets is not None and not root_anchor_sets.empty:
            sub = root_anchor_sets.loc[root_anchor_sets["root_pos"].astype(int) == int(k)]
            anchors = sub["anchor_node"].astype(int).tolist()
        if not anchors:
            anchors = [root]
        fallback = str(lookup.loc[root].get("major_compartment", "unknown")) if root in lookup.index else "unknown"
        allowed = _allowed_compartments_from_profile(root_profiles, k, fallback)

        anchor_emb = emb_z[anchors].mean(axis=0)
        anchor_xy = coords[anchors]
        # Spatial scale from nearby metacells to the mapped root.
        all_sp = np.min(np.linalg.norm(coords[:, None, :] - anchor_xy[None, :, :], axis=2), axis=1)
        positive_sp = np.sort(all_sp[all_sp > 0])
        sp_scale = float(np.median(positive_sp[: min(50, len(positive_sp))])) if len(positive_sp) else 1.0

        dist = nx.multi_source_dijkstra_path_length(G, anchors, weight="weight") if G.number_of_edges() else {a: 0.0 for a in anchors}
        finite_graph = np.asarray([d for d in dist.values() if np.isfinite(d) and d > 0], dtype=float)
        graph_scale = float(np.quantile(finite_graph, 0.20)) if len(finite_graph) else 1.0

        for i in range(n):
            row = lookup.loc[i] if i in lookup.index else pd.Series(dtype=object)
            vec = _row_vector(row)
            process_score = float(sum(vec[COMPARTMENTS.index(c)] for c in allowed if c in COMPARTMENTS))
            denom = float(np.linalg.norm(anchor_emb) * np.linalg.norm(emb_z[i]))
            expr_sim = float(max(np.dot(anchor_emb, emb_z[i]) / denom, 0.0)) if denom > 1e-12 else 0.0
            spatial_sim = float(math.exp(-float(all_sp[i]) / (sp_scale + 1e-9)))
            gd = float(dist.get(i, math.inf))
            graph_sim = float(math.exp(-gd / (graph_scale + 1e-9))) if np.isfinite(gd) else 0.0
            affinity = w_process * process_score + w_expr * expr_sim + w_spatial * spatial_sim + w_graph * graph_sim
            if i in anchors:
                affinity = max(affinity, 1.0)
            A[i, k] = float(np.clip(affinity, 0.0, 1.0))
            rows.append({
                "node_id": int(i),
                "node_name": str(adata.obs_names[i]),
                "root_pos": int(k),
                "root_index": int(root),
                "root_name": str(adata.obs_names[root]),
                "root_local_affinity": float(A[i, k]),
                "process_score": float(process_score),
                "expression_similarity": float(expr_sim),
                "spatial_similarity": float(spatial_sim),
                "graph_similarity": float(graph_sim),
                "is_anchor": bool(i in anchors),
            })
    return A, pd.DataFrame(rows)

def _edge_compatibility(
    row_i: pd.Series,
    row_j: pd.Series,
    cfg: Any,
) -> tuple[float, str, float, float, float]:
    vi, vj = _row_vector(row_i), _row_vector(row_j)
    di, dj = str(row_i.get("major_compartment", "unknown")), str(row_j.get("major_compartment", "unknown"))
    cosine = _cosine(vi, vj)
    shared_mass = float(np.minimum(vi, vj).sum())
    max_i = float(vi.max()) if len(vi) else 0.0
    max_j = float(vj.max()) if len(vj) else 0.0
    active_fraction = float(_cfg(cfg, "process_active_fraction", 0.10))
    support_i = _process_support_set(row_i, active_fraction)
    support_j = _process_support_set(row_j, active_fraction)
    mixed_i = len(support_i) >= 2 or di == "mixed_process" or bool(row_i.get("root_profile_override", False))
    mixed_j = len(support_j) >= 2 or dj == "mixed_process" or bool(row_j.get("root_profile_override", False))

    allowed_cos = float(_cfg(cfg, "composition_similarity_allowed", 0.55))
    process_cos = float(_cfg(cfg, "process_bridge_min_composition_similarity", 0.35))
    bridge_mass = float(_cfg(cfg, "mixed_bridge_min_mass", 0.20))
    process_shared_mass = float(_cfg(cfg, "process_bridge_min_shared_mass", 0.10))
    pure_threshold = float(_cfg(cfg, "pure_compartment_threshold", 0.80))

    # Same-compartment continuity remains eligible, but it is no longer the only
    # biologically meaningful path into the trajectory graph.
    if di == dj and di in COMPARTMENTS:
        return 1.0, "allowed_same_dominant_compartment", cosine, shared_mass, max(max_i, max_j)
    if cosine >= allowed_cos:
        return 1.0, "allowed_composition_continuity", cosine, shared_mass, max(max_i, max_j)

    # Real process-aware bridge: endpoints share at least one active process
    # component and have moderate composition continuity.  This admits, for
    # example, lymphoid->mixed lymphoid/myeloid->myeloid interface states while
    # still rejecting arbitrary pure cross-lineage shortcuts.
    shared_process = support_i & support_j
    if shared_process and (cosine >= process_cos or shared_mass >= process_shared_mass):
        return 1.0, "allowed_shared_process_support", cosine, shared_mass, max(max_i, max_j)

    if mixed_i or mixed_j:
        if shared_mass >= process_shared_mass or cosine >= process_cos:
            return 1.0, "allowed_mixed_interface_process_bridge", cosine, shared_mass, max(max_i, max_j)
        return 0.5, "conditional_mixed_interface_low_overlap", cosine, shared_mass, max(max_i, max_j)

    if di in COMPARTMENTS and dj in COMPARTMENTS and di != dj and max_i >= pure_threshold and max_j >= pure_threshold:
        return 0.0, "forbidden_pure_cross_lineage_trajectory", cosine, shared_mass, max(max_i, max_j)
    if di == "unknown" or dj == "unknown":
        return 0.5, "conditional_unresolved_metacell_annotation", cosine, shared_mass, max(max_i, max_j)
    return 0.0, "forbidden_cross_process_without_bridge_evidence", cosine, shared_mass, max(max_i, max_j)



def trajectory_compatibility_score(source_domain: str, target_domain: str) -> tuple[float, str]:
    """Broad label-only fallback used by downstream reports.

    Construction itself uses composition-aware profiles via ``_edge_compatibility``.
    """
    a, b = str(source_domain), str(target_domain)
    if a == b and a in COMPARTMENTS:
        return 1.0, "same_dominant_compartment"
    if a == "mixed_process" or b == "mixed_process":
        return 0.5, "mixed_process_requires_profile_check"
    if a == "unknown" or b == "unknown":
        return 0.5, "unresolved_metacell_requires_profile_check"
    return 0.0, "cross_process_without_profile_evidence"


def edge_compatibility_from_profiles(row_i: pd.Series, row_j: pd.Series, cfg: Any = None):
    return _edge_compatibility(row_i, row_j, cfg or {})

def annotate_edge_biological_compatibility(
    edges_df: pd.DataFrame,
    node_domains: pd.DataFrame,
    cfg: Any,
) -> pd.DataFrame:
    if edges_df is None or edges_df.empty:
        return pd.DataFrame()
    lookup = node_domains.set_index("node_id")
    allow_conditional = bool(_cfg(cfg, "allow_conditional_trajectory_edges", False))
    min_conf = float(_cfg(cfg, "min_trajectory_edge_confidence", 0.0))
    rows: list[dict[str, Any]] = []
    for row in edges_df.itertuples(index=False):
        i, j = int(row.source), int(row.target)
        if i not in lookup.index or j not in lookup.index:
            continue
        ri, rj = lookup.loc[i], lookup.loc[j]
        kappa, reason, comp_sim, shared_mass, purity = _edge_compatibility(ri, rj, cfg)
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
            "source_compartment": str(ri.get("major_compartment", "unknown")),
            "target_compartment": str(rj.get("major_compartment", "unknown")),
            "source_process_scope": str(ri.get("process_scope", "unknown_process")),
            "target_process_scope": str(rj.get("process_scope", "unknown_process")),
            "source_annotation_label": str(ri.get("annotation_label", "")),
            "target_annotation_label": str(rj.get("annotation_label", "")),
            "composition_similarity": float(comp_sim),
            "composition_shared_mass": float(shared_mass),
            "edge_max_compartment_purity": float(purity),
            "trajectory_compatibility": float(kappa),
            "trajectory_edge_eligible": bool(eligible),
            "bio_compatibility_reason": reason,
            "edge_graph_role": role,
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def split_v7_edges(annotated_edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return legacy_bio.split_v7_edges(annotated_edges)


def _root_allowed_set(root_profiles: pd.DataFrame, k: int, fallback_compartment: str) -> list[str]:
    if root_profiles is not None and not root_profiles.empty:
        hit = root_profiles.loc[root_profiles["root_pos"].astype(int) == int(k)]
        if not hit.empty:
            vals = [x for x in str(hit.iloc[0].get("root_allowed_compartments", "")).split("|") if x in COMPARTMENTS]
            if vals:
                return vals
    return [fallback_compartment] if fallback_compartment in COMPARTMENTS else []


def process_constrained_absorbing_markov(
    n_nodes: int,
    trajectory_edges: pd.DataFrame,
    roots: list[int],
    node_domains: pd.DataFrame,
    cfg: Any,
    root_profiles: Optional[pd.DataFrame] = None,
    root_anchor_sets: Optional[pd.DataFrame] = None,
    root_local_affinity: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Run root-local, process-constrained absorbing Markov assignment.

    v7.3 differs from v7.2 in two important ways:

    1. Each biological root is represented by a small local absorbing anchor set,
       not by a single potentially noisy metacell.
    2. Same-process roots compete only inside their local affinity domains.  A
       giant connected tumor component therefore does not imply global root
       competition across the whole component.
    """
    K = len(roots)
    eps = float(_cfg(cfg, "eps", 1e-9))
    beta = float(_cfg(cfg, "beta_transition", 4.0))
    compat_threshold = float(_cfg(cfg, "root_node_compatibility_threshold", 0.20))
    competition_threshold = float(_cfg(cfg, "root_competition_jaccard_threshold", 0.50))
    affinity_threshold = float(_cfg(cfg, "root_local_affinity_threshold", 0.42))
    affinity_power = float(_cfg(cfg, "root_local_affinity_power", 1.5))
    min_local_nodes = max(1, int(_cfg(cfg, "root_local_min_nodes", 5)))

    A_raw = np.zeros((n_nodes, K), dtype=float)
    root_mask = np.zeros((n_nodes, K), dtype=bool)
    lookup = node_domains.set_index("node_id")
    if root_local_affinity is None:
        root_local_affinity = np.ones((n_nodes, K), dtype=float)
    affinity = np.asarray(root_local_affinity, dtype=float)
    if affinity.shape != (n_nodes, K):
        raise ValueError(f"root_local_affinity shape={affinity.shape} != {(n_nodes, K)}")

    anchors_by_root: dict[int, list[int]] = {}
    for k, root in enumerate(roots):
        anchors = [int(root)]
        if root_anchor_sets is not None and not root_anchor_sets.empty:
            sub = root_anchor_sets.loc[root_anchor_sets["root_pos"].astype(int) == int(k)]
            vals = sub["anchor_node"].astype(int).tolist() if "anchor_node" in sub.columns else []
            anchors = sorted(set([int(root)] + vals))
        anchors_by_root[k] = anchors

    root_allowed: list[set[str]] = []
    root_scope_rows: list[dict[str, Any]] = []
    for k, root in enumerate(roots):
        fallback = str(lookup.loc[int(root)].get("major_compartment", "unknown")) if int(root) in lookup.index else "unknown"
        allowed = _root_allowed_set(root_profiles, k, fallback)
        allowed_set = set(allowed)
        root_allowed.append(allowed_set)
        process_scores = np.zeros(n_nodes, dtype=float)
        for i in range(n_nodes):
            if i not in lookup.index:
                continue
            vec = _row_vector(lookup.loc[i])
            score = float(sum(vec[COMPARTMENTS.index(c)] for c in allowed if c in COMPARTMENTS))
            if score <= 0 and str(lookup.loc[i].get("major_compartment", "unknown")) in allowed:
                score = 1.0
            process_scores[i] = score
        # Missing root-process evidence is unavailable evidence, not evidence
        # that every non-root node is biologically incompatible. In that case
        # the root-local expression/spatial affinity remains the active gate.
        biological = (
            process_scores >= compat_threshold
            if allowed_set
            else np.ones(n_nodes, dtype=bool)
        )
        local = (
            affinity[:, k] >= affinity_threshold
            if allowed_set
            else np.ones(n_nodes, dtype=bool)
        )
        root_mask[:, k] = biological & local
        # If the local affinity gate is very restrictive, retain the strongest
        # biologically compatible nodes rather than manufacturing cross-process links.
        if int(root_mask[:, k].sum()) < min_local_nodes:
            candidates = np.flatnonzero(biological)
            if len(candidates):
                order = candidates[np.argsort(-affinity[candidates, k])]
                root_mask[order[: min(min_local_nodes, len(order))], k] = True
        for a in anchors_by_root[k]:
            if 0 <= int(a) < n_nodes:
                root_mask[int(a), k] = True
        root_mask[int(root), k] = True

        scope = (
            str(root_profiles.loc[root_profiles["root_pos"].astype(int) == k, "root_process_scope"].iloc[0])
            if root_profiles is not None and not root_profiles.empty and (root_profiles["root_pos"].astype(int) == k).any()
            else process_scope_from_compartments(allowed)
        )
        root_scope_rows.append({
            "root_pos": int(k),
            "root_index": int(root),
            "root_process_scope": scope,
            "root_allowed_compartments": "|".join(allowed),
            "root_scope_size": int(root_mask[:, k].sum()),
            "anchor_count": int(len(anchors_by_root[k])),
            "root_local_affinity_threshold": float(affinity_threshold),
        })

    edges = trajectory_edges if trajectory_edges is not None else pd.DataFrame()
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    if not edges.empty:
        G.add_edges_from((int(r.source), int(r.target)) for r in edges.itertuples(index=False))

    isolated = [(k, int(r)) for k, r in enumerate(roots) if G.degree(int(r)) == 0]
    isolated_by_pos = {int(k): int(r) for k, r in isolated}
    for row in root_scope_rows:
        root_pos = int(row["root_pos"])
        row["root_isolated_in_trajectory_graph"] = bool(root_pos in isolated_by_pos)
        row["root_isolation_status"] = (
            "ISOLATED_ROOT_CONTINUED_FOR_TRUTH_FIRST_STATUS"
            if root_pos in isolated_by_pos
            else "CONNECTED_OR_HAS_CANDIDATE_TRAJECTORY_EDGE"
        )

    RG = nx.Graph()
    RG.add_nodes_from(range(K))
    for k in range(K):
        for ell in range(k + 1, K):
            union = root_allowed[k] | root_allowed[ell]
            inter = root_allowed[k] & root_allowed[ell]
            # Two roots with unavailable process ontologies must still compete
            # in the same Markov component. Treating empty/empty as disjoint
            # duplicates the entire component into independent memberships.
            jaccard = len(inter) / len(union) if union else 1.0
            if jaccard >= competition_threshold:
                RG.add_edge(k, ell, jaccard=jaccard)
    competition_groups = [sorted(map(int, comp)) for comp in nx.connected_components(RG)]
    root_to_group = {k: gi for gi, group in enumerate(competition_groups) for k in group}
    for row in root_scope_rows:
        row["competition_group"] = int(root_to_group.get(int(row["root_pos"]), -1))

    S_global = legacy_bio.sparse_similarity_from_edges(n_nodes, edges, "S_global")
    component_rows: list[dict[str, Any]] = []

    for comp_idx, comp in enumerate(nx.connected_components(G)):
        comp_nodes = sorted(map(int, comp))
        comp_node_set = set(comp_nodes)
        comp_root_positions = [k for k in range(K) if any(a in comp_node_set for a in anchors_by_root[k])]
        if not comp_root_positions:
            component_rows.append({
                "component_id": f"trajectory_component_{comp_idx}",
                "competition_group": -1,
                "n_nodes": int(len(comp_nodes)),
                "n_roots": 0,
                "root_positions": "",
                "root_indices": "",
                "assignment_status": "no_root_unassigned",
            })
            continue

        groups_here: dict[int, list[int]] = {}
        for k in comp_root_positions:
            groups_here.setdefault(root_to_group[k], []).append(k)

        for group_id, group_roots in sorted(groups_here.items()):
            eligible_nodes = [i for i in comp_nodes if any(root_mask[i, k] for k in group_roots)]
            if not eligible_nodes:
                continue
            subG = G.subgraph(eligible_nodes).copy()
            for local_comp_idx, local_comp in enumerate(nx.connected_components(subG)):
                nodes = sorted(map(int, local_comp))
                node_set = set(nodes)
                roots_here = [k for k in group_roots if any(a in node_set for a in anchors_by_root[k])]
                if not roots_here:
                    continue
                if len(roots_here) == 1:
                    k = roots_here[0]
                    assign_nodes = [i for i in nodes if root_mask[i, k]]
                    A_raw[assign_nodes, k] = np.maximum(A_raw[assign_nodes, k], 1.0)
                    status = (
                        "isolated_root_local_state_no_trajectory_edges"
                        if k in isolated_by_pos
                        else "single_root_local_process_component"
                    )
                else:
                    idx_map = {node: pos for pos, node in enumerate(nodes)}
                    all_anchor_nodes: list[int] = []
                    anchor_owner: list[int] = []
                    for k in roots_here:
                        for a in anchors_by_root[k]:
                            if a in node_set:
                                all_anchor_nodes.append(int(a))
                                anchor_owner.append(int(k))
                    # Deduplicate anchors defensively.
                    dedup_nodes, dedup_owner, seen = [], [], set()
                    for a, owner in zip(all_anchor_nodes, anchor_owner):
                        if a in seen:
                            continue
                        seen.add(a)
                        dedup_nodes.append(a)
                        dedup_owner.append(owner)
                    local_roots = [idx_map[a] for a in dedup_nodes]
                    S_sub = S_global[nodes][:, nodes].tocsr()
                    local_anchor_A = legacy_bio._absorbing_membership_on_component(S_sub, local_roots, beta, eps)
                    root_scores = np.zeros((len(nodes), len(roots_here)), dtype=float)
                    for a_col, owner in enumerate(dedup_owner):
                        root_scores[:, roots_here.index(owner)] += local_anchor_A[:, a_col]
                    for local_k, global_k in enumerate(roots_here):
                        vals = root_scores[:, local_k] * np.power(np.clip(affinity[nodes, global_k], 0.0, 1.0), affinity_power)
                        vals = np.where(root_mask[nodes, global_k], vals, 0.0)
                        A_raw[nodes, global_k] = np.maximum(A_raw[nodes, global_k], vals)
                    status = "multi_root_local_anchor_markov_component"

                component_rows.append({
                    "component_id": f"trajectory_component_{comp_idx}__group_{group_id}__sub_{local_comp_idx}",
                    "competition_group": int(group_id),
                    "n_nodes": int(len(nodes)),
                    "n_roots": int(len(roots_here)),
                    "root_positions": "|".join(str(k) for k in roots_here),
                    "root_indices": "|".join(str(int(roots[k])) for k in roots_here),
                    "assignment_status": status,
                })

    denom = A_raw.sum(axis=1, keepdims=True)
    valid = denom[:, 0] > eps
    A_raw[valid] /= denom[valid]
    A_raw[~valid] = 0.0

    # Anchor sets are absorbing super-roots.
    for k, root in enumerate(roots):
        for a in anchors_by_root[k]:
            A_raw[int(a), :] = 0.0
            A_raw[int(a), k] = 1.0
        A_raw[int(root), :] = 0.0
        A_raw[int(root), k] = 1.0

    assigned_fraction = float((A_raw.sum(axis=1) > eps).mean()) if n_nodes else 0.0
    min_assigned = float(_cfg(cfg, "min_assigned_fraction", 0.05))
    if assigned_fraction < min_assigned:
        raise RuntimeError(
            f"v7.3 assignment hard failure: only {assigned_fraction:.3%} of metacells receive any root membership; "
            f"minimum required={min_assigned:.3%}."
        )

    return A_raw, root_mask, pd.DataFrame(component_rows), pd.DataFrame(root_scope_rows)


def apply_biological_root_mask(A_raw: np.ndarray, root_mask: np.ndarray, roots: list[int], eps: float = 1e-9) -> np.ndarray:
    return legacy_bio.apply_biological_root_mask(A_raw, root_mask, roots, eps)


def sharpen_membership(A_bio: np.ndarray, q: float, eps: float = 1e-9) -> np.ndarray:
    return legacy_bio.sharpen_membership(A_bio, q, eps)


def safe_primary_roots(A: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return legacy_bio.safe_primary_roots(A, eps)


def build_sparse_active_assignment(
    A_sharp: np.ndarray,
    roots: list[int],
    node_domains: pd.DataFrame,
    trajectory_edges: pd.DataFrame,
    cfg: Any,
    root_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Primary-only default with one optional evidence-gated secondary root."""
    n, K = A_sharp.shape
    eps = float(_cfg(cfg, "eps", 1e-9))
    theta_secondary = float(_cfg(cfg, "theta_secondary", 0.25))
    theta_entropy = float(_cfg(cfg, "theta_entropy", 0.70))
    ratio_threshold = float(_cfg(cfg, "shared_ratio_threshold", 0.65))
    margin_threshold = float(_cfg(cfg, "shared_margin_threshold", 0.20))
    min_evidence = int(_cfg(cfg, "min_shared_evidence_channels", 1))
    max_active = max(1, int(_cfg(cfg, "max_active_roots_per_node", 2)))

    primary = safe_primary_roots(A_sharp, eps)
    adjacency = legacy_bio._trajectory_adjacency(trajectory_edges, n)
    Z = np.zeros((n, K), dtype=np.int8)
    rows: list[dict[str, Any]] = []

    for i in range(n):
        p = int(primary[i])
        if p < 0:
            rows.append({
                "node_id": i, "primary_root": -1, "secondary_root": -1,
                "primary_probability": 0.0, "secondary_probability": 0.0,
                "top2_margin": np.nan, "shared_ratio": 0.0, "entropy_norm": 0.0,
                "n_competing_roots": 0, "shared_evidence_channel_count": 0,
                "secondary_activated": False, "assignment_status": "unassigned",
                "evidence_channels": "",
            })
            continue
        order = np.argsort(-A_sharp[i])
        s = int(order[1]) if K > 1 else -1
        p1 = float(A_sharp[i, p])
        p2 = float(A_sharp[i, s]) if s >= 0 else 0.0
        margin = p1 - p2
        ratio = p2 / (p1 + eps)
        entropy = legacy_bio._normalized_entropy_row(A_sharp[i], eps)
        n_competing = int((A_sharp[i] >= theta_secondary).sum())
        Z[i, p] = 1

        evidence = legacy_bio._initial_shared_evidence(i, s, primary, adjacency) if s >= 0 else {}
        evidence_count = int(sum(bool(v) for v in evidence.values()))
        prob_pass = bool(s >= 0 and p2 >= theta_secondary and ratio >= ratio_threshold)
        uncertainty_pass = bool(entropy >= theta_entropy or margin <= margin_threshold)
        bio_pass = bool(s >= 0 and (root_mask is None or (root_mask[i, p] and root_mask[i, s])))
        evidence_pass = evidence_count >= min_evidence
        activate_secondary = bool(max_active >= 2 and prob_pass and uncertainty_pass and bio_pass and evidence_pass)
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


def audit_tree_biological_validity(tree_edges: pd.DataFrame, node_domains: pd.DataFrame) -> pd.DataFrame:
    if tree_edges is None or tree_edges.empty:
        return pd.DataFrame()
    lookup = node_domains.set_index("node_id")
    rows = []
    for row in tree_edges.itertuples(index=False):
        i, j = int(row.source), int(row.target)
        if i not in lookup.index or j not in lookup.index:
            continue
        kappa, reason, comp_sim, shared_mass, _ = _edge_compatibility(lookup.loc[i], lookup.loc[j], cfg={})
        rows.append({
            "root_pos": int(getattr(row, "root_pos", -1)),
            "source": i,
            "target": j,
            "source_compartment": str(lookup.loc[i].get("major_compartment", "unknown")),
            "target_compartment": str(lookup.loc[j].get("major_compartment", "unknown")),
            "composition_similarity": float(comp_sim),
            "composition_shared_mass": float(shared_mass),
            "trajectory_compatibility": float(kappa),
            "is_forbidden_trajectory_edge": bool(kappa <= 0.0),
            "bio_compatibility_reason": reason,
        })
    return pd.DataFrame(rows)
