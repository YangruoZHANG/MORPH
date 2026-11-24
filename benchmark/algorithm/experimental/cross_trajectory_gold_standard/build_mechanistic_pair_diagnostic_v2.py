#!/usr/bin/env python3
"""Build a paired mechanistic source-program diagnostic without altering benchmark v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from build_event_conditioned_benchmark import state_labels


CONDITIONS = {"null": 0.0, "medium": 0.7, "strong": 1.1}


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-dir", required=True, type=Path)
    parser.add_argument("--positive-template-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7112101)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    baseline = ad.read_h5ad(args.baseline_run_dir / "event" / "synthetic.h5ad")
    baseline_truth = args.baseline_run_dir / "event" / "truth"
    template_truth = args.positive_template_run_dir / "event" / "truth"
    template = pd.read_csv(template_truth / "TRUE_EVENTS.csv").iloc[0]
    contacts = pd.read_csv(baseline_truth / "TRUE_SPATIAL_CONTACTS.csv")
    t0 = pd.read_csv(baseline_truth / "TRUE_T0_EDGES.csv")
    template_t1 = pd.read_csv(template_truth / "TRUE_T1_EDGES.csv")
    template_removed = pd.read_csv(template_truth / "TRUE_REMOVED_EDGES.csv")
    template_added = pd.read_csv(template_truth / "TRUE_ADDED_EDGES.csv")

    source_mask = (
        baseline.obs["root_id"].astype(str).eq(str(template.source_root))
        & baseline.obs["truth_pseudotime"].astype(float).between(
            float(template.source_tau_start), float(template.source_tau_end), inclusive="both"
        )
    )
    source_cells = baseline.obs.index[source_mask]
    tau = baseline.obs.loc[source_cells, "truth_pseudotime"].astype(float).to_numpy()
    latent = (tau - tau.min()) / max(float(tau.max() - tau.min()), 1e-12)
    latent = np.clip(latent + rng.normal(0.0, 0.12, len(latent)), 0.0, 1.0)
    source_activity = pd.Series(latent, index=source_cells)
    contact = contacts.set_index("receiver_cell_id").copy()
    contact["nearest_source_activity"] = contact["nearest_source_cell_id"].map(source_activity).fillna(0.0)
    contact["activity_weighted_exposure"] = contact["contact_score"].astype(float) * contact["nearest_source_activity"].astype(float)

    catalog_rows = [
        {"program_pair_id": "PP01", "source_genes": "G101;G103", "receiver_genes": "G11-G20"},
        {"program_pair_id": "PP02", "source_genes": "G111;G113", "receiver_genes": "G21-G30"},
        {"program_pair_id": "PP03", "source_genes": "G121;G123", "receiver_genes": "G31-G40"},
        {"program_pair_id": "PP04", "source_genes": "G131;G133", "receiver_genes": "G41-G50"},
        {"program_pair_id": "PP05", "source_genes": "G141;G143", "receiver_genes": "G71-G80"},
        {"program_pair_id": "PP06", "source_genes": "G151;G153", "receiver_genes": "G81-G90"},
        {"program_pair_id": "PP07", "source_genes": "G161;G163", "receiver_genes": "G91-G100"},
        {"program_pair_id": "PP08", "source_genes": "G171;G173", "receiver_genes": "G181-G190"},
    ]
    catalog = pd.DataFrame(catalog_rows)
    catalog["expected_sign"] = 1
    catalog.to_csv(args.output_dir / "PROGRAM_PAIR_CATALOG.csv", index=False)
    source_indices = [baseline.var_names.get_loc(gene) for gene in ("G101", "G103")]
    receiver_indices = [baseline.var_names.get_loc(f"G{index}") for index in range(11, 21)]
    baseline_values = dense(baseline.X).astype(float, copy=False)
    source_program_baseline = baseline_values[np.ix_(np.flatnonzero(baseline.obs["root_id"].astype(str).eq(str(template.source_root))), source_indices)].mean(axis=1)
    source_increment_scale = float(np.std(source_program_baseline, ddof=0))

    manifest_rows = []
    for condition, beta in CONDITIONS.items():
        output = args.output_dir / condition
        truth_dir = output / "truth"
        truth_dir.mkdir(parents=True, exist_ok=True)
        adata = baseline.copy()
        values = dense(adata.X).astype(float, copy=True)
        source_positions = adata.obs_names.get_indexer(source_cells)
        values[np.ix_(source_positions, source_indices)] += source_increment_scale * latent[:, None]

        obs = adata.obs.copy()
        obs["source_program_activity"] = 0.0
        obs.loc[source_cells, "source_program_activity"] = latent
        obs["activity_weighted_exposure"] = 0.0
        obs.loc[contact.index, "activity_weighted_exposure"] = contact["activity_weighted_exposure"]
        eligible = (
            obs["root_id"].astype(str).eq(str(template.receiver_root))
            & obs["truth_pseudotime"].astype(float).between(
                float(template.receiver_tau_start), float(template.receiver_tau_end), inclusive="both"
            )
            & obs["truth_branch_T0"].astype(str).eq(str(template.decision_from))
        )
        exposure = obs["activity_weighted_exposure"].astype(float).to_numpy()
        eligible_values = exposure[eligible.to_numpy()]
        active = eligible.to_numpy() & (exposure >= np.quantile(eligible_values, float(template.exposure_quantile)))
        # Preserve the frozen v1 response scale; v2 changes the exposure from
        # geometry alone to activity-weighted source exposure.
        normalized_exposure = np.zeros(len(obs), dtype=float)
        eligible_max = max(float(np.max(eligible_values)), 1e-12)
        normalized_exposure[eligible.to_numpy()] = exposure[eligible.to_numpy()] / eligible_max
        response_increment = beta * normalized_exposure * (2.5 + 2.0 * normalized_exposure)
        response_positions = np.flatnonzero(eligible.to_numpy())
        values[np.ix_(response_positions, receiver_indices)] += response_increment[response_positions, None]
        obs["mechanistic_event_active"] = active
        obs["event_exposure"] = normalized_exposure
        obs["causal_event_id"] = "none"
        obs["truth_branch_T1"] = obs["truth_branch_T0"].astype(str)
        if beta > 0:
            obs.loc[active, "causal_event_id"] = "E_MECHANISTIC_PP01_R0_R1"
            obs.loc[active, "truth_branch_T1"] = str(template.decision_to)
            t1 = template_t1.copy()
        else:
            t1 = t0.copy()
        obs["truth_state_T1"] = state_labels(obs["root_id"], obs["truth_branch_T1"], obs["truth_pseudotime"])
        adata.X = values
        adata.obs = obs
        adata.uns["benchmark_name"] = "NicheForest mechanistic paired diagnostic v2"
        adata.uns["mechanistic_equation"] = "eligible receiver increment = beta * normalized exposure * (2.5 + 2*normalized exposure)"
        adata.write_h5ad(output / "synthetic.h5ad", compression="gzip")

        event = pd.DataFrame([{
            **template.to_dict(),
            "event_id": "E_MECHANISTIC_PP01_R0_R1",
            "event_type": "promotion" if beta > 0 else "null_contact",
            "causal": beta > 0,
            "topology_changed": beta > 0,
            "effect_strength": beta,
            "source_program_genes": "G101;G103",
            "receiver_program_genes": "G11-G20",
            "n_exposed_receiver_cells": int(active.sum()),
            "is_true_event": beta > 0,
            "program_pair_id": "PP01",
        }])
        event.to_csv(truth_dir / "TRUE_EVENTS.csv", index=False)
        t0.to_csv(truth_dir / "TRUE_T0_EDGES.csv", index=False)
        t1.to_csv(truth_dir / "TRUE_T1_EDGES.csv", index=False)
        if beta > 0:
            template_removed.assign(event_id="E_MECHANISTIC_PP01_R0_R1").to_csv(truth_dir / "TRUE_REMOVED_EDGES.csv", index=False)
            template_added.assign(event_id="E_MECHANISTIC_PP01_R0_R1").to_csv(truth_dir / "TRUE_ADDED_EDGES.csv", index=False)
        else:
            template_removed.iloc[0:0].to_csv(truth_dir / "TRUE_REMOVED_EDGES.csv", index=False)
            template_added.iloc[0:0].to_csv(truth_dir / "TRUE_ADDED_EDGES.csv", index=False)
        contact_out = contact.reset_index()
        contact_out["is_event_exposed"] = contact_out["receiver_cell_id"].isin(obs.index[active])
        contact_out["causal"] = beta > 0
        contact_out.to_csv(truth_dir / "TRUE_SPATIAL_CONTACTS.csv", index=False)
        manifest_rows.append({
            "condition": condition,
            "effect_strength": beta,
            "n_source_program_cells": len(source_cells),
            "n_active_receiver_cells": int(active.sum()),
            "source_program_increment_scale": source_increment_scale,
            "source_program_scale_definition": "one baseline source-root program standard deviation",
            "receiver_equation": "eligibility * beta * normalized exposure * (2.5 + 2*normalized exposure)",
            "same_exogenous_noise_and_geometry": True,
        })

    pd.DataFrame(manifest_rows).to_csv(args.output_dir / "V2_DGP_MANIFEST.csv", index=False)
    status = {
        "protocol": "MECHANISTIC_PAIR_DIAGNOSTIC_V2",
        "derived_from_frozen_v1_null_seed": args.seed,
        "v1_outputs_modified": False,
        "program_catalog_contains_truth_label": False,
        "truth_program_pair": "PP01",
        "conditions": CONDITIONS,
    }
    (args.output_dir / "RUN_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
