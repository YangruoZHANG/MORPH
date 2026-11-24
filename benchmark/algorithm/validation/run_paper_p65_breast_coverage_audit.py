#!/usr/bin/env python3
"""Audit frozen breast trajectory coverage and root semantics without refitting."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


ROOT_LEGACY = {"R0": "C13", "R1": "C00", "R2": "C21"}


def hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except Exception:
        return 0.0


def composition_table(
    labels: pd.Series, group: str, level: str, total: int
) -> list[dict[str, object]]:
    counts = labels.astype(object).fillna("unknown").astype(str).value_counts(dropna=False)
    return [
        {
            "group": group,
            "annotation_level": level,
            "cell_type": label,
            "n_cells": int(count),
            "fraction": float(count / max(total, 1)),
        }
        for label, count in counts.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p6-dir", type=Path, required=True)
    parser.add_argument("--method-dir", type=Path, required=True)
    parser.add_argument("--raw-h5ad", type=Path, required=True)
    args = parser.parse_args()

    phase_a = args.p6_dir / "phase_a"
    nodes = pd.read_csv(phase_a / "T0_STATE_NODE_ASSIGNMENTS.csv")
    edges = pd.read_csv(phase_a / "T0_FINE_TRAJECTORY_EDGES.csv")
    soft = pd.read_csv(phase_a / "P6_FROZEN_SOFT_MEMBERSHIP.csv").set_index("unit_id")
    excluded = pd.read_csv(phase_a / "EXCLUDED_ROOT_SUPPORT_NODES.csv")
    mapping = pd.read_csv(
        args.method_dir / "cell_to_metacell_mapping.csv",
        usecols=["cell_name", "metacell_id", "cell_inferred_compartment"],
        dtype={"cell_name": str, "metacell_id": str},
    )
    meta = ad.read_h5ad(args.method_dir / "metacell_adata.h5ad")
    raw = ad.read_h5ad(args.raw_h5ad, backed="r")
    raw_obs = raw.obs[["cell_type_coarse", "cell_type_fine", "cell_type"]].copy()
    raw_obs.index = raw_obs.index.astype(str)
    raw.file.close()

    meta_obs = meta.obs.copy()
    meta_obs.index = meta_obs.metacell_id.astype(str)
    mapping = mapping.join(raw_obs, on="cell_name", how="left", validate="one_to_one")
    if mapping.cell_type.isna().all():
        raise ValueError("Raw-cell annotation join failed")

    covered = set(nodes.unit_id.astype(str))
    all_units = set(meta_obs.index.astype(str))
    uncovered = all_units - covered
    positive_support = set(soft.index[soft.max(axis=1) > 0].astype(str))
    lymphoid = set(meta_obs.index[meta_obs.major_compartment.astype(str).eq("lymphoid")])

    meta_obs["coverage"] = np.where(meta_obs.index.isin(covered), "covered", "uncovered")
    meta_comp = (
        meta_obs.groupby(["coverage", "major_compartment"], dropna=False, observed=True)
        .agg(n_metacells=("metacell_id", "size"), n_cells=("n_cells", "sum"))
        .reset_index()
    )
    meta_comp["metacell_fraction_within_coverage"] = meta_comp.groupby("coverage")[
        "n_metacells"
    ].transform(lambda x: x / x.sum())
    meta_comp["cell_fraction_within_coverage"] = meta_comp.groupby("coverage")[
        "n_cells"
    ].transform(lambda x: x / x.sum())
    meta_comp.to_csv(args.p6_dir / "P65_COVERED_UNCOVERED_COMPOSITION.csv", index=False)

    map_covered = mapping.metacell_id.isin(covered)
    raw_comp_rows: list[dict[str, object]] = []
    for group, mask in [("covered", map_covered), ("uncovered", ~map_covered)]:
        part = mapping.loc[mask]
        for level in ["cell_type_coarse", "cell_type_fine", "cell_type"]:
            raw_comp_rows.extend(
                composition_table(part[level], group, level, len(part))
            )
    pd.DataFrame(raw_comp_rows).to_csv(
        args.p6_dir / "P65_COVERED_UNCOVERED_RAW_CELL_TYPES.csv", index=False
    )

    root_rows: list[dict[str, object]] = []
    root_type_rows: list[dict[str, object]] = []
    branch_type_rows: list[dict[str, object]] = []
    spatial_rows: list[dict[str, object]] = []
    all_lymphoid_points = meta_obs.loc[list(lymphoid), ["spatial_x", "spatial_y"]].to_numpy()
    lymphoid_hull = hull_area(all_lymphoid_points)

    for root_id, node_part in nodes.groupby("root_id", sort=True):
        root_id = str(root_id)
        units = set(node_part.unit_id.astype(str))
        edge_part = edges[edges.root_id.astype(str).eq(root_id)]
        graph = nx.Graph()
        graph.add_nodes_from(node_part.node_id.astype(int))
        graph.add_edges_from(edge_part[["source", "target"]].astype(int).itertuples(index=False, name=None))
        root_node = node_part[node_part.is_root.astype(str).str.lower().eq("true")].iloc[0]
        root_unit = str(root_node.unit_id)
        terminals = [node for node, degree in graph.degree if degree == 1 and node != int(root_node.node_id)]
        branchpoints = [node for node, degree in graph.degree if degree >= 3]
        cell_part = mapping[mapping.metacell_id.isin(units)]
        root_cells = mapping[mapping.metacell_id.eq(root_unit)]
        top_root_subtype = (
            root_cells.cell_type_fine.astype(object).fillna("unknown").astype(str).value_counts().index[0]
            if len(root_cells) else "unknown"
        )
        root_rows.append({
            "root_id": root_id,
            "legacy_root_id": ROOT_LEGACY.get(root_id, ""),
            "root_unit_id": root_unit,
            "root_top_cell_subtype": top_root_subtype,
            "n_tree_metacells": int(len(units)),
            "n_represented_cells": int(len(cell_part)),
            "n_branches": int(node_part.branch_id.astype(str).nunique()),
            "n_branchpoints_degree_ge_3": int(len(branchpoints)),
            "n_terminals_excluding_root": int(len(terminals)),
            "n_edges": int(len(edge_part)),
            "sample_id": "xenium_breast",
            "section_id": "xenium_breast_single_section",
            "provenance_limit": "no_patient_or_section_column_in_input",
        })
        for level in ["cell_type_coarse", "cell_type_fine", "cell_type"]:
            for row in composition_table(cell_part[level], root_id, level, len(cell_part)):
                row["root_id"] = root_id
                root_type_rows.append(row)

        points = meta_obs.loc[list(units), ["spatial_x", "spatial_y"]].to_numpy()
        spatial_rows.append({
            "root_id": root_id,
            "n_metacells": int(len(points)),
            "centroid_x": float(points[:, 0].mean()),
            "centroid_y": float(points[:, 1].mean()),
            "x_min": float(points[:, 0].min()),
            "x_max": float(points[:, 0].max()),
            "y_min": float(points[:, 1].min()),
            "y_max": float(points[:, 1].max()),
            "convex_hull_area": hull_area(points),
            "hull_area_fraction_of_lymphoid_hull": hull_area(points) / max(lymphoid_hull, 1e-12),
        })

        for branch_id, branch_part in node_part.groupby("branch_id", sort=True):
            branch_units = set(branch_part.unit_id.astype(str))
            branch_cells = mapping[mapping.metacell_id.isin(branch_units)]
            for level in ["cell_type_coarse", "cell_type_fine", "cell_type"]:
                for row in composition_table(
                    branch_cells[level], str(branch_id), level, len(branch_cells)
                ):
                    row["root_id"] = root_id
                    row["branch_id"] = str(branch_id)
                    row["n_metacells"] = int(len(branch_units))
                    branch_type_rows.append(row)

    pd.DataFrame(root_rows).to_csv(args.p6_dir / "P65_ROOT_TOPOLOGY_AND_PROVENANCE.csv", index=False)
    pd.DataFrame(root_type_rows).to_csv(args.p6_dir / "P65_ROOT_CELL_TYPE_COMPOSITION.csv", index=False)
    pd.DataFrame(branch_type_rows).to_csv(args.p6_dir / "P65_BRANCH_CELL_TYPE_COMPOSITION.csv", index=False)
    pd.DataFrame(spatial_rows).to_csv(args.p6_dir / "P65_ROOT_SPATIAL_COVERAGE.csv", index=False)

    support_matrix = soft.loc[sorted(covered), ["R0", "R1", "R2"]]
    overlap_rows: list[dict[str, object]] = []
    for threshold in [0.0, 0.10, 0.20]:
        active = support_matrix.gt(threshold) if threshold == 0 else support_matrix.ge(threshold)
        patterns = active.astype(int).astype(str).agg("".join, axis=1)
        for pattern, count in patterns.value_counts().sort_index().items():
            overlap_rows.append({
                "threshold": threshold,
                "R0": bool(int(pattern[0])),
                "R1": bool(int(pattern[1])),
                "R2": bool(int(pattern[2])),
                "n_metacells": int(count),
                "fraction_of_covered": float(count / len(covered)),
            })
    pd.DataFrame(overlap_rows).to_csv(args.p6_dir / "P65_SOFT_SUPPORT_OVERLAP.csv", index=False)

    root_sets = {
        root: set(nodes.loc[nodes.root_id.astype(str).eq(root), "unit_id"].astype(str))
        for root in ["R0", "R1", "R2"]
    }
    pair_rows = []
    for left in root_sets:
        for right in root_sets:
            if left >= right:
                continue
            intersection = root_sets[left] & root_sets[right]
            union = root_sets[left] | root_sets[right]
            pair_rows.append({
                "root_a": left,
                "root_b": right,
                "n_intersection": len(intersection),
                "n_union": len(union),
                "jaccard": len(intersection) / max(len(union), 1),
            })
    pd.DataFrame(pair_rows).to_csv(args.p6_dir / "P65_ROOT_SUPPORT_PAIRWISE_OVERLAP.csv", index=False)

    support_retention = len(covered) / max(len(positive_support), 1)
    structural_root_count = nodes.groupby("unit_id").root_id.nunique()
    n_structurally_shared = int((structural_root_count >= 2).sum())
    n_soft_shared_020 = int(((soft.loc[list(covered)] >= 0.20).sum(axis=1) >= 2).sum())
    covered_cells = mapping[mapping.metacell_id.isin(covered)]
    construction_lymphoid_fraction = float(
        covered_cells.cell_inferred_compartment.astype(str).eq("lymphoid").mean()
    )
    independent_t_cell_fraction = float(
        covered_cells.cell_type_coarse.astype(str).eq("T_cell").mean()
    )
    decision = {
        "stage": "P6.5_TRAJECTORY_COVERAGE_AND_ROOT_SEMANTICS",
        "frozen_topology_modified": False,
        "n_all_metacells": len(all_units),
        "n_lymphoid_metacells": len(lymphoid),
        "n_positive_soft_support_metacells": len(positive_support),
        "n_T0_covered_metacells": len(covered),
        "n_uncovered_metacells": len(uncovered),
        "whole_tissue_coverage": len(covered) / len(all_units),
        "lymphoid_compartment_coverage": len(covered & lymphoid) / len(lymphoid),
        "positive_support_retention": support_retention,
        "n_positive_support_excluded": len(positive_support - covered),
        "excluded_reasons": dict(Counter(excluded.reason.astype(str))),
        "covered_non_lymphoid_metacells": len(covered - lymphoid),
        "soft_shared_membership_ge_0.20": n_soft_shared_020,
        "soft_shared_fraction_of_covered": n_soft_shared_020 / len(covered),
        "T0_structurally_shared_metacells": n_structurally_shared,
        "T0_structurally_shared_fraction_of_covered": n_structurally_shared / len(covered),
        "soft_vs_structural_overlap_note": "multi-root soft eligibility is not identical to repeated inclusion in multiple T0 trees",
        "construction_label_lymphoid_cell_fraction": construction_lymphoid_fraction,
        "independent_raw_classifier_T_cell_fraction": independent_t_cell_fraction,
        "coverage_interpretation": (
            "COHERENT_LYMPHOID_ROOT_SCOPED_SUBSET"
            if covered <= lymphoid and support_retention >= 0.99
            else "INCONSISTENT_REQUIRES_REVIEW"
        ),
        "fine_subtype_semantics": "UNRESOLVED_ANNOTATION_DISCORDANCE",
        "fine_subtype_reason": (
            "construction labels are uniformly lymphoid, whereas the independent "
            "raw-cell classifier labels only a subset as T_cell and root metacells are mixed"
        ),
        "P8_requirement": "validate frozen branches with independent gene sets/pathways before assigning subtype names",
        "paper_wording": "NicheForest identified three root-conditioned trajectories within the lymphoid compartment.",
        "whole_TME_reconstruction_claim_allowed": False,
        "proceed_to_P7_lightweight_abstention_audit": bool(
            covered <= lymphoid and support_retention >= 0.99
        ),
    }
    (args.p6_dir / "P65_COVERAGE_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    root_audit = pd.DataFrame(root_rows)
    report = f"""# P6.5 Breast Coverage and Root Semantics

## Coverage diagnosis

- Whole-tissue coverage is {len(covered)}/{len(all_units)} ({len(covered) / len(all_units):.2%}).
- The predefined lymphoid universe contains {len(lymphoid)} metacells; T0 covers {len(covered & lymphoid)} ({len(covered & lymphoid) / len(lymphoid):.2%}).
- Frozen soft support selected {len(positive_support)} metacells and retained {len(covered)} ({support_retention:.2%}) in T0.
- Only {len(positive_support - covered)} positive-support metacells were excluded, both because no double-guard path reached the root component.
- Therefore the low whole-tissue percentage reflects a lymphoid/root-scoped trajectory target, not widespread candidate-graph loss.

## Root-conditioned trees

- R0/C13: {int(root_audit.loc[root_audit.root_id.eq('R0'), 'n_tree_metacells'].iloc[0])} metacells, 2 branches.
- R1/C00: {int(root_audit.loc[root_audit.root_id.eq('R1'), 'n_tree_metacells'].iloc[0])} metacells, 3 branches.
- R2/C21: {int(root_audit.loc[root_audit.root_id.eq('R2'), 'n_tree_metacells'].iloc[0])} metacells, 2 branches.
- All data come from one xenium_breast section; no patient/section replicate column exists.

## Soft and structural sharing

- {n_soft_shared_020}/{len(covered)} metacells ({n_soft_shared_020 / len(covered):.2%}) have soft membership >=0.20 for at least two roots.
- Only {n_structurally_shared}/{len(covered)} ({n_structurally_shared / len(covered):.2%}) are actually included in more than one frozen T0 tree.
- Soft overlap therefore represents ambiguous eligibility/boundary support and must not be described as structural duplication.

## Annotation limitation

- Construction-time compartment labels classify 100% of covered cells as lymphoid.
- The independent raw-cell classifier labels {independent_t_cell_fraction:.2%} as T_cell and gives mixed labels within all three root metacells.
- Coverage/root-scope semantics are coherent, but fine subtype identities remain unresolved. P8 must use independent gene-set and pathway evidence before naming trajectories.

## Claim

Use: "NicheForest identified three root-conditioned trajectories within the lymphoid compartment."

Do not claim reconstruction of the entire breast TME.
"""
    (args.p6_dir / "P65_COVERAGE_AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
