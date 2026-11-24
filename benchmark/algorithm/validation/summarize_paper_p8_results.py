#!/usr/bin/env python3
"""Add root-level drivers and a detailed report to the frozen P8 result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


SEED = 20260812
N_PERM = 499


def pooled_effect(x: np.ndarray, y: np.ndarray) -> float:
    pooled = np.sqrt(max(
        ((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1))
        / max(len(x) + len(y) - 2, 1), 1e-12,
    ))
    correction = 1.0 - 3.0 / max(4.0 * (len(x) + len(y)) - 9.0, 1.0)
    return float((np.mean(x) - np.mean(y)) / pooled * correction)


def permute(labels: np.ndarray, blocks: np.ndarray, rng) -> np.ndarray:
    result = labels.copy()
    for block in np.unique(blocks):
        positions = np.flatnonzero(blocks == block)
        result[positions] = rng.permutation(result[positions])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p6-dir", type=Path, required=True)
    parser.add_argument("--method-dir", type=Path, required=True)
    parser.add_argument("--p8-dir", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)

    nodes = pd.read_csv(args.p6_dir / "phase_a" / "T0_STATE_NODE_ASSIGNMENTS.csv")
    counts = nodes.groupby("unit_id").root_id.nunique()
    unique = set(counts[counts.eq(1)].index.astype(str))
    assignment = nodes[nodes.unit_id.astype(str).isin(unique)].drop_duplicates("unit_id").set_index("unit_id")
    long = pd.read_csv(args.p8_dir / "P8_INDEPENDENT_SIGNATURE_SCORES.csv")
    long = long[long.family.isin(["external_curated", "Hallmark_2020"])]
    scores = long.pivot(index="unit_id", columns="score_id", values="score").loc[assignment.index]

    adata = ad.read_h5ad(args.method_dir / "metacell_adata.h5ad", backed="r")
    obs = adata.obs[["metacell_id", "spatial_x", "spatial_y"]].copy()
    adata.file.close()
    obs.index = obs.metacell_id.astype(str)
    x = pd.qcut(obs.spatial_x.rank(method="first"), 4, labels=False).astype(int)
    y = pd.qcut(obs.spatial_y.rank(method="first"), 4, labels=False).astype(int)
    block_map = x * 4 + y

    rows = []
    roots = sorted(assignment.root_id.astype(str).unique())
    for i, left in enumerate(roots):
        for right in roots[i + 1:]:
            units = assignment.index[assignment.root_id.astype(str).isin([left, right])]
            labels = assignment.loc[units, "root_id"].astype(str).to_numpy()
            blocks = block_map.loc[units].to_numpy(int)
            local = scores.loc[units]
            for score_id in scores.columns:
                values = local[score_id].to_numpy(float)
                effect = pooled_effect(values[labels == left], values[labels == right])
                null = []
                for _ in range(N_PERM):
                    trial_labels = permute(labels, blocks, rng)
                    null.append(abs(pooled_effect(
                        values[trial_labels == left], values[trial_labels == right]
                    )))
                p = (1 + np.sum(np.asarray(null) >= abs(effect))) / (N_PERM + 1)
                rows.append({
                    "root_a": left, "root_b": right, "score_id": score_id,
                    "standardized_effect_a_minus_b": effect,
                    "spatial_block_permutation_p": p,
                })
    contrasts = pd.DataFrame(rows)
    contrasts["spatial_block_permutation_q"] = multipletests(
        contrasts.spatial_block_permutation_p, method="fdr_bh"
    )[1]
    contrasts["qualifying_driver"] = (
        contrasts.standardized_effect_a_minus_b.abs().ge(0.5)
        & contrasts.spatial_block_permutation_q.le(0.10)
    )
    contrasts.to_csv(args.p8_dir / "P8_ROOT_SIGNATURE_CONTRASTS.csv", index=False)

    decision = json.loads((args.p8_dir / "P8_DECISION.json").read_text())
    root = pd.read_csv(args.p8_dir / "P8_ROOT_IDENTITY_AUDIT.csv")
    branch = pd.read_csv(args.p8_dir / "P8_BRANCH_IDENTITY_AUDIT.csv")
    progression = pd.read_csv(args.p8_dir / "P8_CONTINUOUS_PSEUDOTIME_FUNCTIONS.csv")
    lco = pd.read_csv(args.p8_dir / "P8_LEAVE_CONSTRUCTION_GENES_OUT.csv")
    soft = pd.read_csv(args.p8_dir / "P8_SOFT_SHARING_SUMMARY.csv")

    pair_lines = []
    for row in root.itertuples(index=False):
        pair_lines.append(
            f"- {row.root_a}-{row.root_b}: standardized distance {row.standardized_centroid_distance:.3f}, "
            f"spatial-block q={row.spatial_block_permutation_q:.3g}, gate={'PASS' if row.passes_primary_gate else 'FAIL'}."
        )
    branch_lines = []
    for row in branch.itertuples(index=False):
        branch_lines.append(
            f"- {row.branch_id}: {row.n_qualifying_specializations} qualifying primary scores; "
            f"strongest audited driver `{row.best_score_id}` (effect={row.best_effect:.3f}, q={row.best_q:.3g})."
        )
    progress_count = progression[progression.qualifying_progression].groupby(["root_id", "branch_id"]).size()
    progress_lines = [f"- {branch_id}: {int(count)} qualifying continuous scores." for (_, branch_id), count in progress_count.items()]
    lco_qualifying = lco[lco.full_effect_qualifies]

    soft_lookup = soft.set_index("sharing_group")
    report = f"""# P8 Biological Validation Report

## Hard-Gate Decision

**{decision['decision']}**. P9 is {'unlocked' if decision['P9_unlocked'] else 'not unlocked'} under the preregistered rules.

- Root-profile distinction: {decision['root_pairs_passing']}/3 pairs passed; 2 were required.
- Branch specialization: {decision['branches_with_specialization']}/7 branches passed; 4 were required.
- Ordered functional progression: {decision['branches_with_ordered_progression']}/7 branches passed; 4 were required.
- Leave-construction-HVG-out: {int(lco_qualifying.sensitivity_survives.sum())}/{len(lco_qualifying)} qualifying full-score effects survived ({decision['leave_construction_genes_out_survival_fraction']:.1%}); 70% was required.

## Root Profiles

{chr(10).join(pair_lines)}

R0-R1 was statistically distinguishable under spatial permutation but did not reach the preregistered effect-size threshold. The result therefore supports partial, not complete, three-way molecular separation. Root names remain R0/R1/R2.

## Frozen Branch Programs

{chr(10).join(branch_lines)}

These labels identify statistical drivers, not authorized biological branch names. Branches were defined before pathway scoring.

## Continuous Progression

{chr(10).join(progress_lines)}

The continuous analyses accompany the inherited frozen state steps, so the conclusion does not depend only on early/middle/late binning.

## Soft Sharing

- Root-specific: {int(soft_lookup.loc['root_specific', 'n_metacells'])} metacells; mean subtype entropy {soft_lookup.loc['root_specific', 'subtype_entropy_mean']:.3f}; boundary fraction {soft_lookup.loc['root_specific', 'boundary_fraction']:.3f}.
- Soft-shared only: {int(soft_lookup.loc['soft_shared_only', 'n_metacells'])}; subtype entropy {soft_lookup.loc['soft_shared_only', 'subtype_entropy_mean']:.3f}; boundary fraction {soft_lookup.loc['soft_shared_only', 'boundary_fraction']:.3f}.
- Structurally multi-tree: {int(soft_lookup.loc['structurally_multi_tree', 'n_metacells'])}; subtype entropy {soft_lookup.loc['structurally_multi_tree', 'subtype_entropy_mean']:.3f}; boundary fraction {soft_lookup.loc['structurally_multi_tree', 'boundary_fraction']:.3f}.

The descriptive pattern supports greater annotation mixture and boundary localization among shared cells. The centroid-gap statistic did not support a simple "midway between root centroids" interface model, so no such spatial claim is made.

## Evidence Boundary

All roots, branches, state steps, and tau0 values were frozen before scoring. Primary evidence uses external curated and Hallmark definitions after removing the 50 exact branch-assignment HVGs. Reactome and GO BP are secondary. Hallmark gene sets are designed to summarize coherent biological states and processes; Reactome and GO provide curated pathway and process vocabularies.

This is not a fully independent assay: all 313 panel genes contributed to candidate PCA geometry. Raw-cell subtype labels also remain discordant with construction-time compartment labels. Therefore the current result supports distinct root-conditioned lymphoid progression programs, but does not authorize T-cell subtype names or population-level inference from this single section.

## P9 Contract

P9 may now compare the frozen multi-tree representation with native and complexity-matched global trees on exactly the same 883 metacells. P8 results cannot be used to alter the global-tree construction rule or comparison metrics.
"""
    (args.p8_dir / "P8_BIOLOGICAL_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "root_signature_contrasts": len(contrasts),
        "qualifying_root_signature_drivers": int(contrasts.qualifying_driver.sum()),
        "report_updated": True,
    }, indent=2))


if __name__ == "__main__":
    main()
