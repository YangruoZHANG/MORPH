#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final biological interpretation, audit CSVs, and labelled visualisations for CASCAT v7.4 Truth-First.

This module is intentionally downstream-only: it does not modify tree topology,
root assignment, pseudotime, CMI-E, or direction. It reads the final v7.4 Truth-First outputs,
adds cell-type/compartment labels, summarizes each root-specific biological process,
and creates labelled UMAP/spatial figures for manual biological review.

Important:
    The generated status is an internal structural plausibility audit, not proof of
    biological truth or experimental validation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

try:
    import anndata as ad
except Exception:  # pragma: no cover
    ad = None

try:
    import cascat_v7_3_biological_markov as v71bio
except Exception:  # pragma: no cover
    v71bio = None


PROCESS_NAMES = {
    "tumor_epithelial": "Tumor / epithelial-state progression",
    "lymphoid": "Lymphoid / T-NK-state process",
    "myeloid": "Myeloid / macrophage-state process",
    "endothelial_vascular": "Endothelial / vascular-state process",
    "stromal_fibroblast": "Fibroblast / stromal-remodelling process",
    "perivascular_smooth_muscle": "Perivascular / contractile-state process",
    "microenvironment_ambiguous": "Microenvironment-associated ambiguous process",
    "unknown": "Unresolved biological process",
}

# Restrained palette for publication-style root-specific trees.
TREE_COLORS = [
    "#9C4A4A", "#4F6D8A", "#6B7D5A", "#8A6A88", "#A0714F",
    "#4D7A76", "#7A6F52", "#6A6785", "#8A5D65", "#5A7685",
]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _series_or_default(df: pd.DataFrame, column: str, default):
    """Return a Series aligned to ``df.index`` even when ``column`` is absent.

    pandas.DataFrame.get(..., scalar_default) returns the scalar itself when a
    column is missing. Chaining ``fillna``/``astype`` on that scalar then fails.
    This helper guarantees a Series in both cases.
    """
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def _normalise_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})


def _sanitize_filename(text: object, max_len: int = 80) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return value[:max_len] or "unnamed"


def _counter_string(values: Iterable[object], max_items: int = 12) -> str:
    counts = Counter(str(v) for v in values if str(v) not in {"", "nan", "None"})
    ordered = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max_items]
    return " | ".join(f"{k}:{v}" for k, v in ordered)


def _find_annotation_key(adata, preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in adata.obs.columns:
        return preferred
    if v71bio is not None:
        try:
            return v71bio.find_annotation_column(adata, preferred)
        except Exception:
            pass
    candidates = [
        "v6.3_label", "v6_label", "label", "labels", "cell_type", "celltype",
        "annotation", "anno", "major_cell_type", "subtype", "cluster_label", "state",
    ]
    lower = {str(c).lower(): str(c) for c in adata.obs.columns}
    for c in candidates:
        if c in adata.obs.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _infer_domain(label: object) -> str:
    if v71bio is not None:
        try:
            return str(v71bio.infer_major_compartment_from_label(label))
        except Exception:
            pass
    text = str(label).lower()
    rules = [
        ("tumor_epithelial", ["tumor", "epithelial", "malignant", "carcinoma"]),
        ("lymphoid", ["t/nk", "tnk", "t cell", "cd4", "cd8", "treg", "regulatory t", " nk"]),
        ("myeloid", ["macrophage", "monocyte", "myeloid", "dendritic", "neutrophil"]),
        ("endothelial_vascular", ["endothelial", "vascular", "panvascular", "lymphatic"]),
        ("stromal_fibroblast", ["fibroblast", "caf", "stromal", "myofibroblast"]),
        ("perivascular_smooth_muscle", ["pericyte", "smooth muscle", "contractile", "mural"]),
        ("microenvironment_ambiguous", ["microenvironment", "ambiguous", "mixed", "unknown"]),
    ]
    for domain, terms in rules:
        if any(t in text for t in terms):
            return domain
    return "unknown"


def _compatibility(source_domain: str, target_domain: str) -> tuple[float, str]:
    if v71bio is not None:
        try:
            return v71bio.trajectory_compatibility_score(source_domain, target_domain)
        except Exception:
            pass
    return (1.0, "same_domain") if source_domain == target_domain else (0.0, "cross_domain_unresolved")


def _extract_coords(adata, kind: str) -> tuple[Optional[np.ndarray], str]:
    if kind == "umap":
        for key in ("X_umap", "umap", "X_tsne", "X_pca"):
            if key in adata.obsm:
                arr = np.asarray(adata.obsm[key], dtype=float)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    return arr[:, :2], f"obsm[{key}]"
        return None, "unavailable"
    if kind == "spatial":
        if "spatial" in adata.obsm:
            arr = np.asarray(adata.obsm["spatial"], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2], "obsm[spatial]"
        for a, b in (("x", "y"), ("X", "Y"), ("spatial_x", "spatial_y"), ("array_col", "array_row")):
            if a in adata.obs.columns and b in adata.obs.columns:
                return adata.obs[[a, b]].to_numpy(dtype=float), f"obs[{a},{b}]"
        return None, "unavailable"
    raise ValueError(f"unknown coordinate kind: {kind}")


def _root_table(results_dir: Path, membership: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    profiles = _safe_read_csv(results_dir / "root_process_profiles.csv")
    if not membership.empty and {"root_pos", "root_index", "root_name"}.issubset(membership.columns):
        roots = (
            membership[["root_pos", "root_index", "root_name"]]
            .drop_duplicates()
            .sort_values("root_pos")
            .reset_index(drop=True)
        )
        if not profiles.empty and "root_pos" in profiles.columns:
            extra = [c for c in profiles.columns if c not in {"root_index", "root_name"}]
            roots = roots.merge(profiles[extra], on="root_pos", how="left")
        return roots
    resolved = _safe_read_csv(results_dir / "resolved_roots.csv")
    if not resolved.empty:
        resolved = resolved.reset_index(drop=True)
        rows = []
        for pos, row in resolved.iterrows():
            idx = row.get("subset_index", row.get("resolved_index", np.nan))
            name = row.get("subset_obs_name", row.get("resolved_obs_name", ""))
            if pd.isna(idx):
                continue
            rows.append({"root_pos": int(pos), "root_index": int(idx), "root_name": str(name)})
        if rows:
            return pd.DataFrame(rows)
    root_positions = sorted(edges["root_pos"].astype(int).unique()) if not edges.empty and "root_pos" in edges else []
    return pd.DataFrame({"root_pos": root_positions, "root_index": [-1] * len(root_positions), "root_name": [""] * len(root_positions)})


def _load_node_domains(adata, results_dir: Path, annotation_key: Optional[str]) -> pd.DataFrame:
    domains = _safe_read_csv(results_dir / "biological_node_domains.csv")
    if not domains.empty and {"node_id", "annotation_label", "major_compartment"}.issubset(domains.columns):
        return domains.copy()
    key = _find_annotation_key(adata, annotation_key)
    labels = adata.obs[key].astype(str).to_numpy() if key else np.asarray(["unknown"] * adata.n_obs, dtype=object)
    return pd.DataFrame({
        "node_id": np.arange(adata.n_obs, dtype=int),
        "node_name": adata.obs_names.astype(str),
        "annotation_key": key or "",
        "annotation_label": labels,
        "major_compartment": [_infer_domain(x) for x in labels],
    })


def _load_membership(results_dir: Path) -> pd.DataFrame:
    for name in ("membership_A_sharp_final.csv", "membership_A_sharp.csv", "soft_membership.csv"):
        table = _safe_read_csv(results_dir / name)
        if not table.empty:
            if "A_ik" in table.columns and "A_sharp" not in table.columns:
                table = table.rename(columns={"A_ik": "A_sharp"})
            return table
    return pd.DataFrame()


def _load_active(results_dir: Path) -> pd.DataFrame:
    for name in ("active_root_assignment_final.csv", "active_root_assignment.csv"):
        table = _safe_read_csv(results_dir / name)
        if not table.empty:
            if "is_active" in table.columns:
                table["is_active"] = _normalise_bool(table["is_active"])
            return table
    return pd.DataFrame()


def _load_final_edges(results_dir: Path) -> pd.DataFrame:
    """Load the authoritative final trajectory backbone without fabricating fallback edges.

    A present-but-empty ``final_selected_backbone_tree_edges.csv`` is a valid
    Truth-First conclusion: no trajectory survived the gates.  It must therefore
    be returned as empty and must not fall back to pre-truth/root-specific trees.
    Legacy fallback is used only when the authoritative file does not exist at all.
    """
    preferred = results_dir / "final_selected_backbone_tree_edges.csv"
    if preferred.exists():
        return _safe_read_csv(preferred)

    for name in (
        "root_specific_tree_edges.csv",
        "final_selected_tree_edges.csv",
    ):
        path = results_dir / name
        if path.exists():
            return _safe_read_csv(path)
    return pd.DataFrame()


def build_node_interpretation(
    adata,
    dataset_name: str,
    domains: pd.DataFrame,
    membership: pd.DataFrame,
    active: pd.DataFrame,
    final_edges: pd.DataFrame,
    final_assignments: pd.DataFrame,
) -> pd.DataFrame:
    base = domains.copy()
    if "node_name" not in base.columns:
        base["node_name"] = [str(adata.obs_names[int(i)]) for i in base["node_id"]]

    if not membership.empty and {"node_id", "root_pos", "A_sharp"}.issubset(membership.columns):
        m = membership.copy()
        m["A_sharp"] = pd.to_numeric(m["A_sharp"], errors="coerce").fillna(0.0)
        rows = []
        for node, group in m.groupby("node_id", sort=False):
            g = group.sort_values("A_sharp", ascending=False)
            primary = g.iloc[0]
            secondary = g.iloc[1] if len(g) > 1 else None
            rows.append({
                "node_id": int(node),
                "primary_root_pos": int(primary.root_pos),
                "primary_root_name": str(getattr(primary, "root_name", "")),
                "primary_membership": float(primary.A_sharp),
                "secondary_root_pos": int(secondary.root_pos) if secondary is not None else -1,
                "secondary_root_name": str(getattr(secondary, "root_name", "")) if secondary is not None else "",
                "secondary_membership": float(secondary.A_sharp) if secondary is not None else 0.0,
                "top1_top2_margin": float(primary.A_sharp - (secondary.A_sharp if secondary is not None else 0.0)),
            })
        base = base.merge(pd.DataFrame(rows), on="node_id", how="left")

    if not active.empty and {"node_id", "root_pos", "is_active"}.issubset(active.columns):
        arows = []
        for node, group in active.groupby("node_id", sort=False):
            act = group.loc[group["is_active"]]
            positions = sorted(act["root_pos"].astype(int).tolist())
            names = act["root_name"].astype(str).tolist() if "root_name" in act else []
            arows.append({
                "node_id": int(node),
                "n_active_roots": len(positions),
                "active_root_positions": "|".join(map(str, positions)),
                "active_root_names": "|".join(names),
                "assignment_class": "dual_root_validated" if len(positions) == 2 else ("single_root" if len(positions) == 1 else ("unassigned" if len(positions) == 0 else "high_order_ambiguity")),
            })
        base = base.merge(pd.DataFrame(arows), on="node_id", how="left")

    if not final_edges.empty and {"source", "target", "root_pos"}.issubset(final_edges.columns):
        node_roots: dict[int, set[int]] = {}
        for row in final_edges.itertuples(index=False):
            k = int(row.root_pos)
            node_roots.setdefault(int(row.source), set()).add(k)
            node_roots.setdefault(int(row.target), set()).add(k)
        edge_nodes = pd.DataFrame([
            {
                "node_id": node,
                "in_final_backbone": True,
                "backbone_root_positions": "|".join(map(str, sorted(root_set))),
                "n_backbone_roots": len(root_set),
            }
            for node, root_set in node_roots.items()
        ])
        base = base.merge(edge_nodes, on="node_id", how="left")

    if not final_assignments.empty and "node_id" in final_assignments.columns:
        keep = [
            c for c in (
                "node_id", "predicted_origin_root_pos", "predicted_origin_root_id", "origin_membership",
                "final_pseudotime", "final_pseudotime_origin_normalized", "branch_segment_label",
                "trajectory_status", "n_active_roots", "active_roots",
            ) if c in final_assignments.columns
        ]
        fa = final_assignments[keep].drop_duplicates("node_id")
        rename = {c: f"final_{c}" for c in fa.columns if c != "node_id" and c in base.columns}
        fa = fa.rename(columns=rename)
        base = base.merge(fa, on="node_id", how="left")

    base["dataset"] = dataset_name
    base["in_final_backbone"] = base.get("in_final_backbone", False)
    if isinstance(base["in_final_backbone"], pd.Series):
        base["in_final_backbone"] = base["in_final_backbone"].map(lambda x: bool(x) if pd.notna(x) else False)
    base["n_active_roots"] = pd.to_numeric(_series_or_default(base, "n_active_roots", 0), errors="coerce").fillna(0).astype(int)
    base["n_backbone_roots"] = pd.to_numeric(_series_or_default(base, "n_backbone_roots", 0), errors="coerce").fillna(0).astype(int)
    base["assignment_class"] = _series_or_default(base, "assignment_class", "unassigned").fillna("unassigned")
    base["biological_label"] = base.apply(
        lambda r: f"N{int(r.node_id)} {r.annotation_label}", axis=1
    )
    return base.sort_values("node_id").reset_index(drop=True)


def build_edge_interpretation(
    dataset_name: str,
    final_edges: pd.DataFrame,
    node_summary: pd.DataFrame,
    roots: pd.DataFrame,
) -> pd.DataFrame:
    if final_edges.empty:
        return pd.DataFrame()
    lookup = node_summary.set_index("node_id").to_dict(orient="index")
    root_lookup = roots.set_index("root_pos").to_dict(orient="index") if not roots.empty else {}
    rows = []
    for row in final_edges.itertuples(index=False):
        rec = row._asdict()
        i, j = int(rec["source"]), int(rec["target"])
        k = int(rec.get("root_pos", -1))
        si, tj = lookup.get(i, {}), lookup.get(j, {})
        di = str(si.get("major_compartment", "unknown"))
        dj = str(tj.get("major_compartment", "unknown"))
        if v71bio is not None and hasattr(v71bio, "edge_compatibility_from_profiles"):
            try:
                kappa, reason, comp_sim, shared_mass, _ = v71bio.edge_compatibility_from_profiles(
                    pd.Series(si), pd.Series(tj), {}
                )
            except Exception:
                kappa, reason = _compatibility(di, dj)
                comp_sim, shared_mass = np.nan, np.nan
        else:
            kappa, reason = _compatibility(di, dj)
            comp_sim, shared_mass = np.nan, np.nan
        rec.update({
            "dataset": dataset_name,
            "root_name": str(root_lookup.get(k, {}).get("root_name", "")),
            "source_label": str(si.get("annotation_label", "unknown")),
            "target_label": str(tj.get("annotation_label", "unknown")),
            "source_compartment": di,
            "target_compartment": dj,
            "trajectory_compatibility": float(kappa),
            "composition_similarity": float(comp_sim) if np.isfinite(comp_sim) else np.nan,
            "composition_shared_mass": float(shared_mass) if np.isfinite(shared_mass) else np.nan,
            "is_forbidden_trajectory_edge": bool(kappa <= 0.0),
            "is_conditional_trajectory_edge": bool(0.0 < kappa < 1.0),
            "bio_compatibility_reason": reason,
            "label_transition": f"{si.get('annotation_label', 'unknown')} -> {tj.get('annotation_label', 'unknown')}",
            "compartment_transition": f"{di} -> {dj}",
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def build_tree_summary(
    dataset_name: str,
    edge_summary: pd.DataFrame,
    node_summary: pd.DataFrame,
    roots: pd.DataFrame,
    active_relations: pd.DataFrame,
    review_min_root_process_purity: float,
    review_max_dual_root_rate: float,
) -> pd.DataFrame:
    rows = []
    root_lookup = roots.set_index("root_pos").to_dict(orient="index") if not roots.empty else {}
    root_positions = sorted(set(roots["root_pos"].astype(int).tolist()) | set(edge_summary.get("root_pos", pd.Series(dtype=int)).astype(int).tolist()))
    node_lookup = node_summary.set_index("node_id").to_dict(orient="index")

    for k in root_positions:
        sub = edge_summary.loc[edge_summary["root_pos"].astype(int) == int(k)].copy() if not edge_summary.empty else pd.DataFrame()
        nodes = sorted(set(sub.get("source", pd.Series(dtype=int)).astype(int)) | set(sub.get("target", pd.Series(dtype=int)).astype(int)))
        root_idx = int(root_lookup.get(k, {}).get("root_index", -1))
        if root_idx >= 0 and root_idx not in nodes:
            nodes.append(root_idx)
        node_rows = node_summary.loc[node_summary["node_id"].isin(nodes)].copy()
        root_info = node_lookup.get(root_idx, {})
        root_label = str(root_info.get("annotation_label", root_lookup.get(k, {}).get("root_name", "unknown")))
        root_compartment = str(root_info.get("major_compartment", "unknown"))
        compat_flags = []
        root_series = pd.Series(root_info) if root_info else pd.Series({"major_compartment": root_compartment})
        for _, nr in node_rows.iterrows():
            if v71bio is not None and hasattr(v71bio, "edge_compatibility_from_profiles"):
                try:
                    kappa, *_ = v71bio.edge_compatibility_from_profiles(root_series, nr, {})
                except Exception:
                    kappa, _ = _compatibility(str(nr.get("major_compartment", "unknown")), root_compartment)
            else:
                kappa, _ = _compatibility(str(nr.get("major_compartment", "unknown")), root_compartment)
            compat_flags.append(kappa > 0.0)
        compatible_rate = float(np.mean(compat_flags)) if compat_flags else np.nan
        exact_rate = float((node_rows.get("major_compartment", pd.Series(dtype=str)).astype(str) == root_compartment).mean()) if len(node_rows) else np.nan
        dual_rate = float((pd.to_numeric(_series_or_default(node_rows, "n_active_roots", 0), errors="coerce").fillna(0) == 2).mean()) if len(node_rows) else 0.0
        forbidden = int(sub.get("is_forbidden_trajectory_edge", pd.Series(dtype=bool)).astype(bool).sum()) if not sub.empty else 0
        conditional = int(sub.get("is_conditional_trajectory_edge", pd.Series(dtype=bool)).astype(bool).sum()) if not sub.empty else 0
        n_rel = 0
        if not active_relations.empty and {"tree_k", "tree_l", "relation_active"}.issubset(active_relations.columns):
            active_mask = _normalise_bool(active_relations["relation_active"])
            n_rel = int(((active_relations["tree_k"].astype(int) == k) | (active_relations["tree_l"].astype(int) == k))[active_mask].sum())

        notes = []
        if forbidden > 0:
            status = "HARD_FAIL"
            notes.append(f"{forbidden} forbidden trajectory edge(s)")
        elif len(sub) == 0:
            status = "HARD_FAIL"
            notes.append("root-specific biological process has no trajectory edge")
        elif len(sub) < 2:
            status = "REVIEW"
            notes.append("root-specific tree is trivial (<2 edges)")
        else:
            status = "STRUCTURALLY_PLAUSIBLE"
        if np.isfinite(compatible_rate) and compatible_rate < review_min_root_process_purity:
            if status != "HARD_FAIL":
                status = "REVIEW"
            notes.append(f"root-process compatible node rate {compatible_rate:.3f} below {review_min_root_process_purity:.3f}")
        if dual_rate > review_max_dual_root_rate:
            if status != "HARD_FAIL":
                status = "REVIEW"
            notes.append(f"dual-root rate {dual_rate:.3f} above {review_max_dual_root_rate:.3f}")
        if conditional > 0:
            notes.append(f"{conditional} conditional trajectory edge(s) require biological review")
        if not notes:
            notes.append("No internal structural biology warning triggered")

        rows.append({
            "dataset": dataset_name,
            "root_pos": int(k),
            "root_index": root_idx,
            "root_name": str(root_lookup.get(k, {}).get("root_name", "")),
            "root_annotation_label": root_label,
            "root_major_compartment": root_compartment,
            "proposed_process_interpretation": str(
                root_lookup.get(k, {}).get("root_process_scope", "")
                or PROCESS_NAMES.get(root_compartment, PROCESS_NAMES["unknown"])
            ),
            "n_backbone_nodes": int(len(nodes)),
            "n_backbone_edges": int(len(sub)),
            "annotation_composition": _counter_string(node_rows.get("annotation_label", pd.Series(dtype=str))),
            "compartment_composition": _counter_string(node_rows.get("major_compartment", pd.Series(dtype=str))),
            "root_process_compatible_node_rate": compatible_rate,
            "exact_root_compartment_node_rate": exact_rate,
            "forbidden_edge_count": forbidden,
            "conditional_edge_count": conditional,
            "dual_root_nodes_in_tree": int((pd.to_numeric(_series_or_default(node_rows, "n_active_roots", 0), errors="coerce").fillna(0) == 2).sum()) if len(node_rows) else 0,
            "dual_root_rate_in_tree": dual_rate,
            "active_tree_relations": n_rel,
            "internal_biological_status": status,
            "review_notes": "; ".join(notes),
            "validation_scope": "internal_truth_first_evidence_gates_not_experimental_validation",
        })
    return pd.DataFrame(rows)


def build_transition_summary(edge_summary: pd.DataFrame) -> pd.DataFrame:
    if edge_summary.empty:
        return pd.DataFrame()
    group_cols = [
        "dataset", "root_pos", "root_name", "source_label", "target_label",
        "source_compartment", "target_compartment", "trajectory_compatibility",
        "is_forbidden_trajectory_edge", "is_conditional_trajectory_edge",
    ]
    existing = [c for c in group_cols if c in edge_summary.columns]
    out = edge_summary.groupby(existing, dropna=False, as_index=False).size().rename(columns={"size": "edge_count"})
    return out.sort_values(["root_pos", "edge_count"], ascending=[True, False])


def build_validation_overview(
    dataset_name: str,
    node_summary: pd.DataFrame,
    edge_summary: pd.DataFrame,
    tree_summary: pd.DataFrame,
    relations: pd.DataFrame,
    review_max_dual_root_rate: float,
    review_max_active_relation_rate: float,
) -> pd.DataFrame:
    n_nodes = len(node_summary)
    n_dual = int((pd.to_numeric(_series_or_default(node_summary, "n_active_roots", 0), errors="coerce").fillna(0) == 2).sum()) if n_nodes else 0
    n_gt2 = int((pd.to_numeric(_series_or_default(node_summary, "n_active_roots", 0), errors="coerce").fillna(0) > 2).sum()) if n_nodes else 0
    dual_rate = n_dual / n_nodes if n_nodes else 0.0
    forbidden = int(edge_summary.get("is_forbidden_trajectory_edge", pd.Series(dtype=bool)).astype(bool).sum()) if not edge_summary.empty else 0
    conditional = int(edge_summary.get("is_conditional_trajectory_edge", pd.Series(dtype=bool)).astype(bool).sum()) if not edge_summary.empty else 0
    if not relations.empty and "relation_active" in relations.columns:
        active_relation_rate = float(_normalise_bool(relations["relation_active"]).mean())
        n_active_relations = int(_normalise_bool(relations["relation_active"]).sum())
        n_relation_pairs = int(len(relations))
    else:
        active_relation_rate = 0.0
        n_active_relations = 0
        n_relation_pairs = 0
    mean_purity = float(pd.to_numeric(tree_summary.get("root_process_compatible_node_rate", pd.Series(dtype=float)), errors="coerce").mean()) if not tree_summary.empty else np.nan

    status = "STRUCTURALLY_PLAUSIBLE"
    notes = []
    n_backbone_nodes = int(_normalise_bool(_series_or_default(node_summary, "in_final_backbone", False)).sum()) if n_nodes else 0
    n_backbone_edges = int(len(edge_summary))
    unassigned_rate = float((pd.to_numeric(_series_or_default(node_summary, "n_active_roots", 0), errors="coerce").fillna(0) == 0).mean()) if n_nodes else 1.0
    if n_backbone_edges == 0 or n_backbone_nodes <= 1:
        status = "HARD_FAIL"
        notes.append(f"empty or trivial final backbone: nodes={n_backbone_nodes}, edges={n_backbone_edges}")
    if unassigned_rate > 0.95:
        status = "HARD_FAIL"
        notes.append(f"unassigned node rate={unassigned_rate:.3f} exceeds hard-fail threshold=0.950")
    if forbidden > 0 or n_gt2 > 0:
        status = "HARD_FAIL"
        if forbidden > 0:
            notes.append(f"forbidden trajectory edges={forbidden}")
        if n_gt2 > 0:
            notes.append(f"nodes with >2 active roots={n_gt2}")
    if dual_rate > review_max_dual_root_rate:
        if status != "HARD_FAIL":
            status = "REVIEW"
        notes.append(f"dual-root rate={dual_rate:.3f} exceeds review threshold={review_max_dual_root_rate:.3f}")
    if active_relation_rate > review_max_active_relation_rate:
        if status != "HARD_FAIL":
            status = "REVIEW"
        notes.append(f"active tree-relation rate={active_relation_rate:.3f} exceeds review threshold={review_max_active_relation_rate:.3f}")
    if not tree_summary.empty and (tree_summary["internal_biological_status"] == "HARD_FAIL").any():
        status = "HARD_FAIL"
        notes.append("at least one root-specific tree has HARD_FAIL status")
    elif not tree_summary.empty and (tree_summary["internal_biological_status"] == "REVIEW").any() and status != "HARD_FAIL":
        status = "REVIEW"
        notes.append("at least one root-specific tree requires review")
    if not tree_summary.empty and "n_backbone_nodes" in tree_summary.columns:
        sizes = pd.to_numeric(tree_summary["n_backbone_nodes"], errors="coerce").fillna(0).to_numpy(dtype=float)
        total = float(sizes.sum())
        dominance = float(sizes.max() / total) if total > 0 else 0.0
        if dominance > 0.90 and len(sizes) > 1:
            if status != "HARD_FAIL":
                status = "REVIEW"
            notes.append(f"one root-specific tree dominates {dominance:.1%} of summed backbone nodes")
    else:
        dominance = np.nan
    if not notes:
        notes.append("No internal structural biology warning triggered")

    return pd.DataFrame([{
        "dataset": dataset_name,
        "method": "CASCAT-MR-CMI-E-Gate_v7.4_Truth-First",
        "n_nodes": n_nodes,
        "n_roots": int(len(tree_summary)),
        "n_final_backbone_nodes": n_backbone_nodes,
        "n_final_backbone_edges": n_backbone_edges,
        "unassigned_node_rate": unassigned_rate,
        "forbidden_final_trajectory_edges": forbidden,
        "conditional_final_trajectory_edges": conditional,
        "validated_dual_root_nodes": n_dual,
        "validated_dual_root_rate": dual_rate,
        "nodes_with_gt2_active_roots": n_gt2,
        "active_tree_relation_pairs": n_active_relations,
        "total_tree_relation_pairs": n_relation_pairs,
        "active_tree_relation_rate": active_relation_rate,
        "mean_root_process_compatible_node_rate": mean_purity,
        "max_tree_node_dominance_fraction": dominance,
        "overall_internal_biological_status": status,
        "review_notes": "; ".join(notes),
        "review_max_dual_root_rate": review_max_dual_root_rate,
        "review_max_active_relation_rate": review_max_active_relation_rate,
        "validation_scope": "internal_truth_first_evidence_gates_not_experimental_validation",
    }])


def _tree_nodes(edges: pd.DataFrame, root_pos: int, root_idx: int) -> list[int]:
    sub = edges.loc[edges["root_pos"].astype(int) == int(root_pos)] if not edges.empty else pd.DataFrame()
    nodes = sorted(set(sub.get("source", pd.Series(dtype=int)).astype(int)) | set(sub.get("target", pd.Series(dtype=int)).astype(int)))
    if root_idx >= 0 and root_idx not in nodes:
        nodes.append(root_idx)
    return sorted(nodes)


def _draw_tree_edges(ax, coords: np.ndarray, edges: pd.DataFrame, color: str, alpha: float = 0.88) -> None:
    if edges.empty:
        return
    for row in edges.itertuples(index=False):
        i, j = int(row.source), int(row.target)
        if i >= len(coords) or j >= len(coords):
            continue
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        patch = FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
            linewidth=1.25, color=color, alpha=alpha,
            shrinkA=2.0, shrinkB=2.0, zorder=3,
        )
        ax.add_patch(patch)


def _annotate_nodes(ax, coords: np.ndarray, node_rows: pd.DataFrame, fontsize: float = 7.0) -> None:
    for row in node_rows.itertuples(index=False):
        i = int(row.node_id)
        if i >= len(coords):
            continue
        label = f"N{i} {str(row.annotation_label)}"
        ax.annotate(
            label,
            (coords[i, 0], coords[i, 1]),
            xytext=(3, 3), textcoords="offset points",
            fontsize=fontsize, ha="left", va="bottom", zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.72),
        )


def plot_root_tree_labeled(
    coords: np.ndarray,
    coord_name: str,
    root_pos: int,
    root_idx: int,
    root_label: str,
    edges: pd.DataFrame,
    node_summary: pd.DataFrame,
    outpath: Path,
    color: str,
    label_all_tree_nodes: bool = True,
) -> None:
    sub = edges.loc[edges["root_pos"].astype(int) == int(root_pos)].copy() if not edges.empty else pd.DataFrame()
    nodes = _tree_nodes(edges, root_pos, root_idx)
    node_rows = node_summary.loc[node_summary["node_id"].isin(nodes)].copy()

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(coords[:, 0], coords[:, 1], s=12, c="#D8D8D8", alpha=0.42, linewidths=0, zorder=1)
    if nodes:
        ax.scatter(coords[nodes, 0], coords[nodes, 1], s=38, c=color, alpha=0.88, edgecolors="white", linewidths=0.4, zorder=4)
    _draw_tree_edges(ax, coords, sub, color)
    if 0 <= root_idx < len(coords):
        ax.scatter(coords[root_idx, 0], coords[root_idx, 1], marker="*", s=320, c=color, edgecolors="black", linewidths=0.8, zorder=7)
    if label_all_tree_nodes:
        _annotate_nodes(ax, coords, node_rows, fontsize=7.2)
    title = f"R{root_pos}: {root_label}\nFinal biological trajectory on {coord_name}"
    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_roots_labeled(
    coords: np.ndarray,
    coord_name: str,
    roots: pd.DataFrame,
    edges: pd.DataFrame,
    node_summary: pd.DataFrame,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=10, c="#D8D8D8", alpha=0.36, linewidths=0, zorder=1)
    union_nodes: set[int] = set()
    for pos_idx, root in roots.sort_values("root_pos").reset_index(drop=True).iterrows():
        k = int(root.root_pos)
        ridx = int(root.root_index)
        color = TREE_COLORS[pos_idx % len(TREE_COLORS)]
        sub = edges.loc[edges["root_pos"].astype(int) == k].copy() if not edges.empty else pd.DataFrame()
        nodes = _tree_nodes(edges, k, ridx)
        union_nodes.update(nodes)
        if nodes:
            ax.scatter(coords[nodes, 0], coords[nodes, 1], s=32, c=color, alpha=0.78, edgecolors="white", linewidths=0.35, zorder=3, label=f"R{k} {root.root_name}")
        _draw_tree_edges(ax, coords, sub, color, alpha=0.78)
        if 0 <= ridx < len(coords):
            ax.scatter(coords[ridx, 0], coords[ridx, 1], marker="*", s=280, c=color, edgecolors="black", linewidths=0.8, zorder=7)
    label_rows = node_summary.loc[node_summary["node_id"].isin(sorted(union_nodes))]
    _annotate_nodes(ax, coords, label_rows, fontsize=6.8)
    ax.set_title(f"Final CASCAT v7.4 Truth-First biological trajectories on {coord_name}\nLabels show N(node_id) + annotation", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if len(roots) <= 12:
        ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_markdown_report(
    outpath: Path,
    overview: pd.DataFrame,
    tree_summary: pd.DataFrame,
    transition_summary: pd.DataFrame,
) -> None:
    overview_row = overview.iloc[0].to_dict() if not overview.empty else {}
    lines = [
        "# CASCAT v7.4 Truth-First final biological interpretation report",
        "",
        "> This is an internal structural plausibility audit. It does not establish experimental biological truth.",
        "",
        "## Dataset-level overview",
        "",
        f"- Dataset: `{overview_row.get('dataset', '')}`",
        f"- Overall internal status: **{overview_row.get('overall_internal_biological_status', '')}**",
        f"- Final backbone edges: `{overview_row.get('n_final_backbone_edges', 0)}`",
        f"- Forbidden final trajectory edges: `{overview_row.get('forbidden_final_trajectory_edges', 0)}`",
        f"- Validated dual-root rate: `{overview_row.get('validated_dual_root_rate', 0.0):.4f}`",
        f"- Active tree-relation rate: `{overview_row.get('active_tree_relation_rate', 0.0):.4f}`",
        f"- Mean root-process compatible node rate: `{overview_row.get('mean_root_process_compatible_node_rate', float('nan')):.4f}`",
        f"- Review notes: {overview_row.get('review_notes', '')}",
        "",
        "## Root-specific biological processes",
        "",
    ]
    if tree_summary.empty:
        lines.append("No final root-specific tree was available.")
    else:
        for row in tree_summary.itertuples(index=False):
            lines.extend([
                f"### R{int(row.root_pos)} — {row.root_annotation_label}",
                "",
                f"- Proposed interpretation: **{row.proposed_process_interpretation}**",
                f"- Root compartment: `{row.root_major_compartment}`",
                f"- Backbone nodes / edges: `{int(row.n_backbone_nodes)}` / `{int(row.n_backbone_edges)}`",
                f"- Root-process compatible node rate: `{float(row.root_process_compatible_node_rate):.4f}`" if pd.notna(row.root_process_compatible_node_rate) else "- Root-process compatible node rate: `NA`",
                f"- Forbidden edges: `{int(row.forbidden_edge_count)}`",
                f"- Conditional edges: `{int(row.conditional_edge_count)}`",
                f"- Dual-root rate within tree: `{float(row.dual_root_rate_in_tree):.4f}`",
                f"- Internal status: **{row.internal_biological_status}**",
                f"- Annotation composition: {row.annotation_composition}",
                f"- Review notes: {row.review_notes}",
                "",
            ])
    lines.extend(["## Dominant labelled transitions", ""])
    if transition_summary.empty:
        lines.append("No final trajectory transitions were available.")
    else:
        for k, group in transition_summary.groupby("root_pos", sort=True):
            lines.append(f"### R{int(k)}")
            lines.append("")
            for row in group.head(12).itertuples(index=False):
                lines.append(f"- `{row.source_label} -> {row.target_label}`: {int(row.edge_count)} edge(s)")
            lines.append("")
    outpath.write_text("\n".join(lines), encoding="utf-8")


def generate_biological_interpretation(
    adata,
    results_dir: Path,
    dataset_name: Optional[str] = None,
    annotation_key: Optional[str] = None,
    review_min_root_process_purity: float = 0.70,
    review_max_dual_root_rate: float = 0.20,
    review_max_active_relation_rate: float = 0.50,
    make_figures: bool = True,
) -> dict[str, object]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = dataset_name or results_dir.name

    final_edges = _load_final_edges(results_dir)
    membership = _load_membership(results_dir)
    active = _load_active(results_dir)
    domains = _load_node_domains(adata, results_dir, annotation_key)
    final_assignments = _safe_read_csv(results_dir / "final_node_assignments.csv")
    relations = _safe_read_csv(results_dir / "tree_pair_relation_matrix_detailed.csv")
    roots = _root_table(results_dir, membership, final_edges)

    node_summary = build_node_interpretation(
        adata, dataset_name, domains, membership, active, final_edges, final_assignments
    )
    edge_summary = build_edge_interpretation(dataset_name, final_edges, node_summary, roots)
    tree_summary = build_tree_summary(
        dataset_name,
        edge_summary,
        node_summary,
        roots,
        relations,
        review_min_root_process_purity,
        review_max_dual_root_rate,
    )
    # Truth-First status overrides purely structural labels.  A Level-0 root is
    # not a failed biological trajectory; it is an explicit refusal to make a
    # trajectory claim because support is insufficient.
    truth_gate = _safe_read_csv(results_dir / "trajectory_truth_gate_by_root.csv")
    if not truth_gate.empty and "root_pos" in truth_gate.columns:
        keep_cols = [c for c in [
            "root_pos", "truth_level", "truth_level_label", "trajectory_truth_status",
            "root_truth_status",
            "root_validity_status", "root_quality_score", "program_coherence_pass",
            "interface_support_pass", "conservative_process_interpretation",
            "truth_gate_reasons", "external_biological_support",
        ] if c in truth_gate.columns]
        tree_summary = tree_summary.merge(truth_gate[keep_cols], on="root_pos", how="left")
        level_map = {
            0: "INSUFFICIENT_SUPPORT",
            1: "STRUCTURALLY_PLAUSIBLE",
            2: "PROCESS_SUPPORTED",
            3: "EXTERNALLY_SUPPORTED",
        }
        levels = pd.to_numeric(tree_summary.get("truth_level", pd.Series(np.nan, index=tree_summary.index)), errors="coerce")
        for idx, level in levels.items():
            if np.isfinite(level):
                truth_status = str(
                    tree_summary.loc[idx].get("root_truth_status", "")
                    or tree_summary.loc[idx].get("trajectory_truth_status", "")
                )
                if not truth_status or truth_status.lower() == "nan":
                    truth_status = level_map.get(int(level), "REVIEW")
                tree_summary.loc[idx, "internal_biological_status"] = truth_status
                if int(level) == 0:
                    tree_summary.loc[idx, "review_notes"] = str(tree_summary.loc[idx].get("truth_gate_reasons", "insufficient support"))
                    tree_summary.loc[idx, "proposed_process_interpretation"] = str(
                        tree_summary.loc[idx].get("conservative_process_interpretation", "no_trajectory_claim")
                    )
                elif int(level) >= 2:
                    tree_summary.loc[idx, "proposed_process_interpretation"] = str(
                        tree_summary.loc[idx].get("conservative_process_interpretation", tree_summary.loc[idx].get("proposed_process_interpretation", ""))
                    )

    transition_summary = build_transition_summary(edge_summary)
    overview = build_validation_overview(
        dataset_name,
        node_summary,
        edge_summary,
        tree_summary,
        relations,
        review_max_dual_root_rate,
        review_max_active_relation_rate,
    )
    truth_overview = _safe_read_csv(results_dir / "truth_first_validation_overview.csv")
    if not truth_overview.empty:
        for col in truth_overview.columns:
            if col not in overview.columns:
                overview[col] = truth_overview.iloc[0][col]
        if "overall_truth_first_status" in truth_overview.columns:
            overview["overall_internal_biological_status"] = truth_overview.iloc[0]["overall_truth_first_status"]

    outputs = {
        "node_summary": "final_biological_node_interpretation.csv",
        "edge_summary": "final_biological_edge_interpretation.csv",
        "tree_summary": "final_biological_process_tree_summary.csv",
        "transition_summary": "final_biological_transition_summary.csv",
        "overview": "final_biological_validation_overview.csv",
        "report": "final_biological_interpretation_report.md",
    }
    node_summary.to_csv(results_dir / outputs["node_summary"], index=False)
    edge_summary.to_csv(results_dir / outputs["edge_summary"], index=False)
    tree_summary.to_csv(results_dir / outputs["tree_summary"], index=False)
    transition_summary.to_csv(results_dir / outputs["transition_summary"], index=False)
    overview.to_csv(results_dir / outputs["overview"], index=False)
    _write_markdown_report(results_dir / outputs["report"], overview, tree_summary, transition_summary)

    figure_files: list[str] = []
    coord_sources: dict[str, str] = {}
    if make_figures and not roots.empty:
        figdir = results_dir / "figures_biological_interpretation"
        figdir.mkdir(parents=True, exist_ok=True)
        for kind in ("umap", "spatial"):
            coords, source = _extract_coords(adata, kind)
            coord_sources[kind] = source
            if coords is None or len(coords) != adata.n_obs:
                continue
            all_name = f"final_all_roots_{kind}_labeled.png"
            plot_all_roots_labeled(coords, kind.upper(), roots, final_edges, node_summary, figdir / all_name)
            figure_files.append(str(Path("figures_biological_interpretation") / all_name))
            for idx, root in roots.sort_values("root_pos").reset_index(drop=True).iterrows():
                k = int(root.root_pos)
                ridx = int(root.root_index)
                root_node = node_summary.loc[node_summary["node_id"] == ridx]
                root_label = str(root_node["annotation_label"].iloc[0]) if not root_node.empty else str(root.root_name)
                name = f"R{k}_{_sanitize_filename(root_label)}_{kind}_labeled.png"
                plot_root_tree_labeled(
                    coords=coords,
                    coord_name=kind.upper(),
                    root_pos=k,
                    root_idx=ridx,
                    root_label=root_label,
                    edges=final_edges,
                    node_summary=node_summary,
                    outpath=figdir / name,
                    color=TREE_COLORS[idx % len(TREE_COLORS)],
                    label_all_tree_nodes=True,
                )
                figure_files.append(str(Path("figures_biological_interpretation") / name))

    manifest = {
        "dataset": dataset_name,
        "status": str(overview.iloc[0]["overall_internal_biological_status"]) if not overview.empty else "UNAVAILABLE",
        "csv_outputs": outputs,
        "figure_files": figure_files,
        "coordinate_sources": coord_sources,
        "validation_scope": "internal_truth_first_evidence_gates_not_experimental_validation",
    }
    (results_dir / "final_biological_interpretation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate labelled final biological interpretation for CASCAT v7.4 Truth-First")
    parser.add_argument("--adata", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--annotation-key", default=None)
    parser.add_argument("--review-min-root-process-purity", type=float, default=0.70)
    parser.add_argument("--review-max-dual-root-rate", type=float, default=0.20)
    parser.add_argument("--review-max-active-relation-rate", type=float, default=0.50)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    if ad is None:
        raise ImportError("anndata is required to read --adata")
    results_dir = Path(args.results_dir)
    metacell_path = results_dir / "metacell_adata.h5ad"
    adata_path = metacell_path if metacell_path.exists() else Path(args.adata)
    adata = ad.read_h5ad(adata_path)
    manifest = generate_biological_interpretation(
        adata=adata,
        results_dir=results_dir,
        dataset_name=args.dataset_name,
        annotation_key=args.annotation_key,
        review_min_root_process_purity=args.review_min_root_process_purity,
        review_max_dual_root_rate=args.review_max_dual_root_rate,
        review_max_active_relation_rate=args.review_max_active_relation_rate,
        make_figures=not args.skip_figures,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
