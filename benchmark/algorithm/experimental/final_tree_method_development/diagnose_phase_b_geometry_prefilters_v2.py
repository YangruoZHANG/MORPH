#!/usr/bin/env python3
"""Audit response-free branch/window prefilters for Phase B development.

Truth is used only to score how well a predeclared geometric ranking recovers
the synthetic event region.  It is never used to construct a detector input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


WINDOWS = ((0.0, 0.5), (0.25, 0.75), (0.5, 1.0))
EPS = 1e-12


def parse_genes(value: str) -> list[str]:
    value = str(value)
    if ";" in value:
        return [item for item in value.split(";") if item]
    left, right = value.split("-", maxsplit=1)
    return [f"G{i}" for i in range(int(left[1:]), int(right[1:]) + 1)]


def prf(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum(predicted & truth))
    precision = tp / max(int(np.sum(predicted)), 1)
    recall = tp / max(int(np.sum(truth)), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    return float(precision), float(recall), float(f1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--phase-a-subdir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-cells", type=int, default=25)
    parser.add_argument("--source-activity-quantile", type=float, default=0.60)
    parser.add_argument("--kernel-quantile", type=float, default=0.35)
    parser.add_argument("--contact-quantile", type=float, default=0.65)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    strategies = {
        "exposure_variance": lambda row: row["exposure_variance"],
        "mean_exposure": lambda row: row["mean_exposure"],
        "contact_density": lambda row: row["contact_density"],
        "upper_exposure_mean": lambda row: row["upper_exposure_mean"],
        "exposure_dynamic_range": lambda row: row["exposure_dynamic_range"],
        "contact_support": lambda row: row["n_cells"],
    }

    statuses = sorted(args.grid_root.rglob("TOPOLOGY_DEV_CASE_STATUS.json"))
    for status_file in statuses:
        run = status_file.parent
        status = json.loads(status_file.read_text(encoding="utf-8"))
        truth = pd.read_csv(run / "event" / "truth" / "TRUE_EVENTS.csv")
        truth = truth[truth.causal.astype(str).str.lower().isin({"true", "1"})]
        if truth.empty:
            continue
        adata = ad.read_h5ad(run / "event" / "synthetic.h5ad")
        values = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
        values = np.log1p(np.maximum(values.astype(float), 0.0))
        gene_index = {str(gene): index for index, gene in enumerate(adata.var_names)}
        mapping = pd.read_csv(run / "formal_metacell" / "cell_to_metacell_mapping.csv")
        cell_to_mc = mapping.set_index("cell_name").metacell_id.astype(str)
        metacells = cell_to_mc.reindex(adata.obs_names.astype(str)).to_numpy()
        assignments = pd.read_csv(run / args.phase_a_subdir / "T0_STATE_NODE_ASSIGNMENTS.csv")
        assignments["unit_id"] = assignments.unit_id.astype(str)
        primary = assignments.sort_values("membership", ascending=False).drop_duplicates("unit_id")
        root_lookup = primary.set_index("unit_id").root_id.astype(str).to_dict()
        inferred_root = np.asarray([root_lookup.get(str(mc), "none") for mc in metacells])
        indexed = assignments.set_index(["root_id", "unit_id"])

        for event in truth.itertuples(index=False):
            source_genes = parse_genes(str(event.source_program_genes))
            source_score = values[:, [gene_index[gene] for gene in source_genes]].mean(axis=1)
            source_root = str(event.source_root)
            receiver_root = str(event.receiver_root)
            source_all = np.flatnonzero(inferred_root == source_root)
            receiver_all = np.flatnonzero(inferred_root == receiver_root)
            if len(source_all) < args.min_cells or len(receiver_all) < args.min_cells:
                continue
            threshold = float(np.quantile(source_score[source_all], args.source_activity_quantile))
            selected_source = source_all[source_score[source_all] >= threshold]
            receiver_coords = adata.obs.iloc[receiver_all][["spatial_x", "spatial_y"]].to_numpy(float)
            source_coords = adata.obs.iloc[selected_source][["spatial_x", "spatial_y"]].to_numpy(float)
            distance = np.sqrt(
                ((receiver_coords[:, None, :] - source_coords[None, :, :]) ** 2).sum(axis=2)
            )
            nearest = distance.min(axis=1)
            radius = max(float(np.quantile(nearest, args.kernel_quantile)), EPS)
            exposure = np.exp(-np.square(nearest / radius))
            contact_radius = max(float(np.quantile(nearest, args.contact_quantile)), EPS)
            contact = nearest <= contact_radius
            tau_lookup = indexed.loc[receiver_root].pseudotime.astype(float).to_dict()
            branch_lookup = indexed.loc[receiver_root].branch_id.astype(str).to_dict()
            tau = np.asarray([tau_lookup.get(str(metacells[index]), 0.5) for index in receiver_all])
            branch = np.asarray([branch_lookup.get(str(metacells[index]), "none") for index in receiver_all])
            true_event = adata.obs.causal_event_id.astype(str).str.split(";").map(
                lambda ids: str(event.event_id) in ids
            ).to_numpy(bool)
            truth_local = true_event[receiver_all]
            local_candidates: list[dict[str, object]] = []
            for branch_id in sorted(set(branch) - {"none"}):
                branch_mask = branch == branch_id
                for start, end in WINDOWS:
                    window_mask = (tau >= start) & (tau <= end)
                    eligible = branch_mask & window_mask
                    selected = eligible & contact
                    if int(np.sum(selected)) < args.min_cells:
                        continue
                    local_x = exposure[selected]
                    q65 = float(np.quantile(local_x, 0.65))
                    upper = local_x[local_x >= q65]
                    precision, recall, f1 = prf(selected, truth_local)
                    row = {
                        "setting": str(status["setting"]),
                        "replicate": int(status["replicate"]),
                        "event_id": str(event.event_id),
                        "event_type": str(event.event_type),
                        "effect_strength": float(event.effect_strength),
                        "source_root": source_root,
                        "receiver_root": receiver_root,
                        "branch_id": branch_id,
                        "window_start": start,
                        "window_end": end,
                        "n_cells": int(np.sum(selected)),
                        "eligible_n_cells": int(np.sum(eligible)),
                        "mean_exposure": float(np.mean(local_x)),
                        "exposure_variance": float(np.var(local_x)),
                        "upper_exposure_mean": float(np.mean(upper)),
                        "exposure_dynamic_range": float(np.quantile(local_x, 0.90) - np.quantile(local_x, 0.10)),
                        "contact_density": float(np.mean(contact[eligible])),
                        "event_region_precision": precision,
                        "event_region_recall": recall,
                        "event_region_F1": f1,
                        "truth_used_for_scoring_only": True,
                        "selected_mask": selected,
                    }
                    local_candidates.append(row)
                    candidate_rows.append({key: value for key, value in row.items() if key != "selected_mask"})

            for strategy, key in strategies.items():
                ranked = sorted(local_candidates, key=key, reverse=True)
                for top_k in (1, 2, 3):
                    chosen = ranked[:top_k]
                    predicted = np.zeros(len(receiver_all), dtype=bool)
                    for row in chosen:
                        predicted |= np.asarray(row["selected_mask"], dtype=bool)
                    precision, recall, f1 = prf(predicted, truth_local)
                    selection_rows.append({
                        "setting": str(status["setting"]),
                        "event_id": str(event.event_id),
                        "event_type": str(event.event_type),
                        "effect_strength": float(event.effect_strength),
                        "strategy": strategy,
                        "top_k": top_k,
                        "n_candidates": len(local_candidates),
                        "event_region_precision": precision,
                        "event_region_recall": recall,
                        "event_region_F1": f1,
                        "selected_branches": ";".join(str(row["branch_id"]) for row in chosen),
                        "selected_windows": ";".join(
                            f"{row['window_start']:.2f}-{row['window_end']:.2f}" for row in chosen
                        ),
                        "truth_used_for_scoring_only": True,
                    })

    candidates = pd.DataFrame(candidate_rows)
    selections = pd.DataFrame(selection_rows)
    summary = selections.groupby(["strategy", "top_k"], as_index=False).agg(
        n_events=("event_id", "count"),
        mean_precision=("event_region_precision", "mean"),
        mean_recall=("event_region_recall", "mean"),
        mean_F1=("event_region_F1", "mean"),
        recall_ge_0_50=("event_region_recall", lambda values: float(np.mean(values >= 0.50))),
        recall_ge_0_75=("event_region_recall", lambda values: float(np.mean(values >= 0.75))),
    )
    candidates.to_csv(args.output / "GEOMETRY_CANDIDATES.csv", index=False)
    selections.to_csv(args.output / "GEOMETRY_SELECTION_EVENT_METRICS.csv", index=False)
    summary.to_csv(args.output / "GEOMETRY_SELECTION_SUMMARY.csv", index=False)
    print(summary.sort_values(["top_k", "mean_F1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
