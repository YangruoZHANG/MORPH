#!/usr/bin/env python3
"""Lesion-level Perturb-map silver-standard validation for NicheForest claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


SEED = 8122026
RIDGE = 1.0
BOOTSTRAPS = 5000
MOUSE_BY_SAMPLE = {
    "KP_1": "mouse_4", "KP_2": "mouse_4",
    "KP_3": "mouse_5", "KP_4": "mouse_7",
}
PROGRAMS = {
    "Tgfbr2_KO": {
        "program": "TGF_STROMAL_RESPONSE",
        "positive": [
            "Col1a1", "Col6a1", "Col6a2", "Tnc",
            "Timp1", "Arg1", "Tgfb1", "Tgfb3",
        ],
        "negative": ["Sftpb"],
        "expected_direction": 1,
        "evidence_tier": "primary",
    },
    "Ifngr2_KO": {
        "program": "IFNG_RESPONSE",
        "positive": [
            "Nlrc5", "Cxcl9", "Cxcl10", "Gbp2", "Gbp3", "Socs1",
            "Isg15", "Ifit1", "Ifit2", "Ifit3", "Oas2", "Ccl5",
        ],
        "negative": [],
        "expected_direction": -1,
        "evidence_tier": "primary",
    },
    "Jak2_KO": {
        "program": "IFNG_RESPONSE",
        "positive": [
            "Nlrc5", "Cxcl9", "Cxcl10", "Gbp2", "Gbp3", "Socs1",
            "Isg15", "Ifit1", "Ifit2", "Ifit3", "Oas2", "Ccl5",
        ],
        "negative": [],
        "expected_direction": -1,
        "evidence_tier": "exploratory_low_power",
    },
}
PAPER_URL = "https://doi.org/10.1016/j.cell.2022.02.015"


def design_matrix(obs: pd.DataFrame, coords: np.ndarray) -> np.ndarray:
    samples = pd.get_dummies(obs["sample"].astype(str), dtype=float)
    scaled = coords.astype(float).copy()
    for sample in samples.columns:
        mask = obs["sample"].astype(str).eq(sample).to_numpy()
        for column in range(2):
            values = scaled[mask, column]
            scale = float(np.std(values))
            scaled[mask, column] = (
                values - float(np.mean(values))
            ) / max(scale, 1e-12)
    spatial_terms = np.column_stack([
        scaled[:, 0], scaled[:, 1],
        scaled[:, 0] ** 2, scaled[:, 1] ** 2,
        scaled[:, 0] * scaled[:, 1],
    ])
    return np.column_stack([np.ones(len(obs)), samples.to_numpy(), spatial_terms])


def ridge_predict(
    design: np.ndarray, outcome: np.ndarray,
    train: np.ndarray, test: np.ndarray,
) -> np.ndarray:
    left = design[train].T @ design[train]
    penalty = np.eye(left.shape[0]) * RIDGE
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        left + penalty, design[train].T @ outcome[train]
    )
    return design[test] @ coefficients


def context_residuals(
    obs: pd.DataFrame, design: np.ndarray, outcome: np.ndarray,
) -> np.ndarray:
    perturbation = obs["perturbation"].astype(str).to_numpy()
    lesions = obs["lesion_id"].astype(str).to_numpy()
    control = perturbation == "Control"
    residual = np.full(len(obs), np.nan, dtype=float)
    for lesion in sorted(set(lesions[control])):
        test = control & (lesions == lesion)
        train = control & (lesions != lesion)
        residual[test] = outcome[test] - ridge_predict(
            design, outcome, train, test
        )
    treated = ~control
    residual[treated] = outcome[treated] - ridge_predict(
        design, outcome, control, treated
    )
    return residual


def hierarchical_bootstrap(
    frame: pd.DataFrame, rng: np.random.Generator,
) -> tuple[float, float]:
    mice = sorted(frame.biological_mouse.unique())
    estimates = []
    for _ in range(BOOTSTRAPS):
        sampled_mice = rng.choice(mice, size=len(mice), replace=True)
        values = []
        for mouse in sampled_mice:
            local = frame[frame.biological_mouse.eq(mouse)].contrast.to_numpy(float)
            values.extend(rng.choice(local, size=len(local), replace=True))
        estimates.append(float(np.mean(values)))
    return tuple(np.quantile(estimates, [0.025, 0.975]).astype(float))


def lesion_contrasts(
    obs: pd.DataFrame, residual: np.ndarray, perturbation: str,
) -> pd.DataFrame:
    frame = obs[[
        "lesion_id", "perturbation", "sample", "biological_mouse"
    ]].copy()
    frame["residual"] = residual
    lesion = frame.groupby([
        "lesion_id", "perturbation", "sample", "biological_mouse"
    ], observed=True).agg(
        residual=("residual", "mean"), n_spots=("residual", "size")
    ).reset_index()
    controls = lesion[lesion.perturbation.eq("Control")].groupby(
        "sample", observed=True
    ).residual.median().astype(float)
    treated = lesion[lesion.perturbation.eq(perturbation)].copy()
    treated["residual"] = treated.residual.astype(float)
    treated["control_residual_same_section"] = (
        treated["sample"].map(controls).astype(float)
    )
    treated["contrast"] = (
        treated.residual - treated.control_residual_same_section
    )
    return treated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.input)
    matrix = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    matrix = np.log1p(np.maximum(matrix.astype(float), 0.0))
    obs = adata.obs.copy()
    obs["perturbation"] = obs.perturbation.astype("string").fillna("Unannotated")
    obs["lesion_id"] = obs.phenotypes.astype("string").fillna("Unannotated")
    obs["biological_mouse"] = obs["sample"].astype(str).map(MOUSE_BY_SAMPLE)
    included = obs.perturbation.isin(["Control", *PROGRAMS]).to_numpy()
    obs = obs.loc[included].copy()
    matrix = matrix[included]
    coords = np.asarray(adata.obsm["spatial"])[included]
    design = design_matrix(obs, coords)
    gene_index = {str(gene): index for index, gene in enumerate(adata.var_names)}
    control = obs.perturbation.astype(str).eq("Control").to_numpy()
    rng = np.random.default_rng(SEED)
    lesion_frames: list[pd.DataFrame] = []
    gene_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for perturbation, specification in PROGRAMS.items():
        signed_genes = [
            (gene, 1.0) for gene in specification["positive"] if gene in gene_index
        ] + [
            (gene, -1.0) for gene in specification["negative"] if gene in gene_index
        ]
        standardized = []
        for gene, sign in signed_genes:
            values = matrix[:, gene_index[gene]]
            center = float(np.mean(values[control]))
            scale = float(np.std(values[control]))
            standardized.append(sign * (values - center) / max(scale, 1e-12))
        program_score = np.mean(np.column_stack(standardized), axis=1)
        residual = context_residuals(obs, design, program_score)
        contrasts = lesion_contrasts(obs, residual, perturbation)
        contrasts.insert(0, "program", specification["program"])
        contrasts.insert(0, "target_perturbation", perturbation)
        lesion_frames.append(contrasts)
        effect = float(contrasts.contrast.mean())
        lower, upper = hierarchical_bootstrap(contrasts, rng)
        expected = int(specification["expected_direction"])
        gene_matches = []
        for gene, sign in signed_genes:
            values = matrix[:, gene_index[gene]] * sign
            gene_residual = context_residuals(obs, design, values)
            local = lesion_contrasts(obs, gene_residual, perturbation)
            gene_effect = float(local.contrast.mean())
            match = bool(np.sign(gene_effect) == expected)
            gene_matches.append(match)
            gene_rows.append({
                "perturbation": perturbation,
                "program": specification["program"],
                "gene": gene,
                "program_sign": sign,
                "context_adjusted_lesion_effect": gene_effect,
                "expected_direction": expected,
                "direction_concordant": match,
            })
        summary_rows.append({
            "perturbation": perturbation,
            "program": specification["program"],
            "evidence_tier": specification["evidence_tier"],
            "n_spots": int(obs.perturbation.astype(str).eq(perturbation).sum()),
            "n_lesions": int(len(contrasts)),
            "n_sections": int(contrasts["sample"].nunique()),
            "n_independent_mice": int(contrasts.biological_mouse.nunique()),
            "context_adjusted_lesion_effect": effect,
            "hierarchical_bootstrap_ci_low": lower,
            "hierarchical_bootstrap_ci_high": upper,
            "expected_direction": expected,
            "direction_concordant": bool(np.sign(effect) == expected),
            "gene_direction_concordance": float(np.mean(gene_matches)),
            "n_program_genes_measured": len(signed_genes),
            "analysis_unit": "lesion nested within section and biological mouse",
            "claim_scope": "perturbational silver standard; not topology truth",
            "source": PAPER_URL,
        })
    pd.concat(lesion_frames, ignore_index=True).to_csv(
        args.output / "PHASE_F_PERTURBMAP_LESION_EFFECTS.csv", index=False
    )
    pd.DataFrame(gene_rows).to_csv(
        args.output / "PHASE_F_PERTURBMAP_GENE_CONCORDANCE.csv", index=False
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        args.output / "PHASE_F_PERTURBMAP_SUMMARY.csv", index=False
    )
    decision = {
        "dataset": "GSE193460 Perturb-map",
        "statistical_unit": "lesion nested within section and mouse",
        "context_model": "control-only ridge on section and within-section spatial terms",
        "control_prediction": "leave-one-control-lesion-out",
        "treated_prediction": "fit on all control lesions only",
        "ridge": RIDGE,
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "primary_direction_concordant": bool(summary[
            summary.evidence_tier.eq("primary")
        ].direction_concordant.all()),
        "complete_topology_truth_available": False,
        "allowed_claim": "directional perturbational program concordance",
        "complete": True,
    }
    (args.output / "PHASE_F_PERTURBMAP_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
