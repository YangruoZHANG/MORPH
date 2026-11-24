#!/usr/bin/env python3
"""Program-catalog source-activity relation detector for the v2 diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from run_generic_relation_refinement_v1 import bh_qvalues, branch_assignment, residualize, spatial_blocks, zscore


WINDOWS = ((0.0, 0.5), (0.25, 0.75), (0.5, 1.0))


def parse_genes(value: str) -> list[str]:
    value = str(value)
    if ";" in value:
        return [item.strip() for item in value.split(";") if item.strip()]
    left, right = value.split("-", 1)
    prefix = "".join(ch for ch in left if not ch.isdigit())
    start = int("".join(ch for ch in left if ch.isdigit()))
    stop = int("".join(ch for ch in right if ch.isdigit()))
    return [f"{prefix}{index}" for index in range(start, stop + 1)]


def corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True, type=Path)
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--fixed-root-input", required=True, type=Path)
    parser.add_argument("--cell-to-metacell", required=True, type=Path)
    parser.add_argument("--program-catalog", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--q-cutoff", type=float, default=0.05)
    parser.add_argument("--min-stratum-cells", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7112101)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    adata = ad.read_h5ad(args.h5ad)
    expression = np.log1p(adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X))
    gene_index = {str(gene): index for index, gene in enumerate(adata.var_names.astype(str))}
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    mapping = pd.read_csv(args.cell_to_metacell).set_index("cell_name")["metacell_id"].astype(str)
    metacells = mapping.reindex(adata.obs_names.astype(str)).to_numpy()
    membership = pd.read_csv(args.upstream_dir / "membership_A_sharp.csv")
    primary = membership.pivot(index="node_name", columns="root_pos", values="A_sharp").fillna(0.0).idxmax(axis=1).astype(int).to_dict()
    root_assignment = np.asarray([primary.get(str(node), -1) for node in metacells], dtype=int)
    fixed = pd.read_csv(args.fixed_root_input)
    pos_to_root = dict(enumerate(fixed["truth_root_id"].astype(str)))
    t0 = pd.read_csv(args.upstream_dir / "progression_tree_edges.csv")
    branch_maps = {
        position: branch_assignment(t0[t0["root_pos"].astype(int).eq(position)], str(fixed.iloc[position].root_anchor_id))
        for position in pos_to_root
    }
    pseudotime = pd.read_csv(args.upstream_dir / "pseudotime_table.csv")
    pseudotime = pseudotime[pseudotime["tree_label"].astype(str).eq("progression")].copy()
    id_to_name = membership.drop_duplicates("node_id").set_index("node_id")["node_name"].astype(str)
    pseudotime["metacell"] = pseudotime["node_id"].astype(int).map(id_to_name)
    pt_lookup = {
        (int(row.root_pos), str(row.metacell)): float(row.selected_pseudotime)
        for row in pseudotime.itertuples(index=False)
    }
    inferred_pt = np.asarray([pt_lookup.get((int(root), str(node)), 0.5) for root, node in zip(root_assignment, metacells)])
    catalog = pd.read_csv(args.program_catalog)
    annotation_column = "cell.type" if "cell.type" in adata.obs.columns else None
    annotation = (
        pd.get_dummies(adata.obs[annotation_column].astype(str), drop_first=True).to_numpy(float)
        if annotation_column else np.empty((adata.n_obs, 0), dtype=float)
    )
    rows = []
    null_distributions: dict[int, list[float]] = {}

    for source_root in sorted(pos_to_root):
        source_all = np.flatnonzero(root_assignment == source_root)
        for receiver_root in sorted(pos_to_root):
            if source_root == receiver_root:
                continue
            receiver = np.flatnonzero(root_assignment == receiver_root)
            receiver_branches = np.asarray([branch_maps[receiver_root].get(str(metacells[index]), "") for index in receiver])
            base_strata = [("ALL", np.ones(len(receiver), dtype=bool))]
            for branch in sorted(set(receiver_branches) - {"", "ROOT"}):
                base_strata.append((branch, receiver_branches == branch))
            strata = []
            seen_masks: set[bytes] = set()
            for base_name, base_mask in base_strata:
                variants = [(base_name, base_mask)]
                for receiver_start, receiver_end in WINDOWS:
                    variants.append((
                        f"{base_name}|PT_{receiver_start:.2f}_{receiver_end:.2f}",
                        base_mask & (inferred_pt[receiver] >= receiver_start) & (inferred_pt[receiver] <= receiver_end),
                    ))
                for stratum_name, mask in variants:
                    if int(mask.sum()) < args.min_stratum_cells or mask.tobytes() in seen_masks:
                        continue
                    seen_masks.add(mask.tobytes())
                    strata.append((stratum_name, mask))
            for program in catalog.itertuples(index=False):
                expected_sign = int(getattr(program, "expected_sign", 1))
                if expected_sign not in {-1, 1}:
                    raise ValueError(f"expected_sign must be -1 or 1 for {program.program_pair_id}")
                source_genes = parse_genes(program.source_genes)
                receiver_genes = parse_genes(program.receiver_genes)
                if not set(source_genes + receiver_genes).issubset(gene_index):
                    continue
                source_score_all = expression[:, [gene_index[gene] for gene in source_genes]].mean(axis=1)
                response_all = expression[:, [gene_index[gene] for gene in receiver_genes]].mean(axis=1)
                candidate_stats = []
                candidate_payloads = []
                for start, end in WINDOWS:
                    source = source_all[(inferred_pt[source_all] >= start) & (inferred_pt[source_all] <= end)]
                    if len(source) < args.min_stratum_cells:
                        continue
                    distance = np.sqrt(((coords[receiver, None, :] - coords[source][None, :, :]) ** 2).sum(axis=2))
                    nearest_distance = distance.min(axis=1)
                    radius = max(float(np.quantile(nearest_distance, 0.50)), 1e-6)
                    source_activity = zscore(source_score_all[source].reshape(-1, 1)).ravel()
                    weights = np.exp(-np.square(distance / radius))
                    exposure = (weights @ source_activity) / np.maximum(weights.sum(axis=1), 1e-12)
                    for stratum_name, stratum_mask in strata:
                        if int(stratum_mask.sum()) < args.min_stratum_cells:
                            continue
                        positions = receiver[stratum_mask]
                        x = exposure[stratum_mask]
                        y = response_all[positions]
                        xy = coords[positions]
                        covariates = np.column_stack([
                            inferred_pt[positions], xy[:, 0], xy[:, 1],
                            xy[:, 0] ** 2, xy[:, 1] ** 2, xy[:, 0] * xy[:, 1],
                            annotation[positions],
                        ])
                        xr = residualize(x[:, None], covariates).ravel()
                        yr = residualize(y[:, None], covariates).ravel()
                        statistic = corr(xr, yr)
                        candidate_stats.append(expected_sign * statistic)
                        candidate_payloads.append((
                            statistic, start, end, stratum_name, len(positions), xr, yr, spatial_blocks(xy)
                        ))
                if not candidate_stats:
                    continue
                best_index = int(np.argmax(candidate_stats))
                signed, start, end, stratum_name, n_cells, _, _, _ = candidate_payloads[best_index]
                null_max = []
                for _ in range(args.permutations):
                    permuted_stats = []
                    for _, _, _, _, _, xr, yr, blocks in candidate_payloads:
                        permuted = xr.copy()
                        for block in np.unique(blocks):
                            indices = np.flatnonzero(blocks == block)
                            permuted[indices] = xr[rng.permutation(indices)]
                        permuted_stats.append(expected_sign * corr(permuted, yr))
                    null_max.append(max(permuted_stats))
                directed_effect = expected_sign * float(signed)
                pvalue = float((1 + np.sum(np.asarray(null_max) >= directed_effect)) / (args.permutations + 1))
                rows.append({
                    "source_root_pos": source_root,
                    "receiver_root_pos": receiver_root,
                    "source_root": pos_to_root[source_root],
                    "receiver_root": pos_to_root[receiver_root],
                    "program_pair_id": str(program.program_pair_id),
                    "effect_size": directed_effect,
                    "signed_effect": float(signed),
                    "expected_sign": expected_sign,
                    "event_type": "promotion" if expected_sign > 0 else "inhibition",
                    "permutation_p": pvalue,
                    "selected_source_pt_start": start,
                    "selected_source_pt_end": end,
                    "selected_receiver_stratum": stratum_name,
                    "n_receiver_cells": n_cells,
                })

    result = pd.DataFrame(rows)
    result["q_value"] = bh_qvalues(result["permutation_p"].to_numpy(float))
    result["is_called"] = result["q_value"].le(args.q_cutoff)
    result = result.sort_values(["q_value", "effect_size"], ascending=[True, False])
    result.to_csv(args.output_dir / "PROGRAM_RELATION_DIAGNOSTICS.csv", index=False)
    result[result["is_called"]].to_csv(args.output_dir / "CALLED_PROGRAM_RELATIONS.csv", index=False)
    status = {
        "method": "MECHANISTIC_RELATION_V2_3_DIAGNOSTIC",
        "truth_used": False,
        "program_catalog": str(args.program_catalog.resolve()),
        "program_catalog_has_truth_label": False,
        "n_hypotheses": len(result),
        "n_called": int(result["is_called"].sum()),
        "permutations": args.permutations,
        "q_cutoff": args.q_cutoff,
        "source_windows": WINDOWS,
        "min_stratum_cells": args.min_stratum_cells,
        "annotation_covariate": annotation_column,
    }
    (args.output_dir / "RUN_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
