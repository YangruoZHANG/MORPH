#!/usr/bin/env python3
"""Truth-only diagnostic for PHASE B exposure localization failures."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-9 and np.std(y) > 1e-9 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for run in args.root.glob("core/*/replicate_01_seed_7112101"):
        setting = run.parent.name
        truth = pd.read_csv(run / "event" / "truth" / "TRUE_EVENTS.csv").iloc[0]
        adata = ad.read_h5ad(run / "event" / "synthetic.h5ad")
        values = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
        values = np.log1p(np.maximum(values, 0.0))
        gene = {str(name): i for i, name in enumerate(adata.var_names)}
        source_score = values[:, [gene["G101"], gene["G103"]]].mean(axis=1)
        receiver_score = values[:, [gene[f"G{i}"] for i in range(11, 21)]].mean(axis=1)
        mapping = pd.read_csv(run / "formal_metacell" / "cell_to_metacell_mapping.csv")
        cell_to_mc = mapping.set_index("cell_name").metacell_id.astype(str)
        metacells = cell_to_mc.reindex(adata.obs_names.astype(str)).to_numpy()
        assignment = pd.read_csv(run / "phase_a_v4_frozen" / "T0_STATE_NODE_ASSIGNMENTS.csv")
        primary = assignment.sort_values("membership", ascending=False).drop_duplicates("unit_id")
        root_lookup = primary.set_index(primary.unit_id.astype(str)).root_id.astype(str).to_dict()
        inferred_root = np.asarray([root_lookup.get(str(mc), "none") for mc in metacells])
        source_all = np.flatnonzero(inferred_root == str(truth.source_root))
        receiver_all = np.flatnonzero(inferred_root == str(truth.receiver_root))
        true_source = (
            adata.obs.root_id.astype(str).eq(str(truth.source_root)).to_numpy()
            & adata.obs.truth_pseudotime.between(
                float(truth.source_tau_start), float(truth.source_tau_end), inclusive="both"
            ).to_numpy()
        )
        true_exposure = adata.obs.event_exposure.to_numpy(float)[receiver_all]
        active = adata.obs.causal_event_id.astype(str).eq(str(truth.event_id)).to_numpy()[receiver_all]
        for activity_quantile in (0.40, 0.50, 0.60, 0.65, 0.70):
            threshold = np.quantile(source_score[source_all], activity_quantile)
            selected = source_all[source_score[source_all] >= threshold]
            selected_mask = np.zeros(adata.n_obs, dtype=bool)
            selected_mask[selected] = True
            precision = float((selected_mask & true_source).sum() / max(selected_mask.sum(), 1))
            recall = float((selected_mask & true_source).sum() / max(true_source.sum(), 1))
            source_coords = adata.obs.iloc[selected][["spatial_x", "spatial_y"]].to_numpy(float)
            receiver_coords = adata.obs.iloc[receiver_all][["spatial_x", "spatial_y"]].to_numpy(float)
            distance = np.sqrt(((receiver_coords[:, None] - source_coords[None]) ** 2).sum(axis=2))
            nearest = distance.min(axis=1)
            radius = max(float(np.quantile(nearest, 0.35)), 1e-6)
            inferred_exposure = np.exp(-np.square(nearest / radius))
            rows.append({
            "setting": setting, "event_type": truth.event_type,
            "effect_strength": truth.effect_strength,
            "source_activity_quantile": activity_quantile,
            "source_window_precision": precision, "source_window_recall": recall,
            "exposure_truth_correlation": corr(inferred_exposure, true_exposure),
            "exposure_active_auc_proxy": corr(inferred_exposure, active.astype(float)),
            "receiver_response_vs_inferred_exposure": corr(receiver_score[receiver_all], inferred_exposure),
            "receiver_response_vs_true_exposure": corr(receiver_score[receiver_all], true_exposure),
            "receiver_response_active_contrast": float(
                receiver_score[receiver_all][active].mean() - receiver_score[receiver_all][~active].mean()
            ) if active.any() else 0.0,
                "truth_only_diagnostic": True,
            })
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("setting").to_csv(output, index=False)


if __name__ == "__main__":
    main()
