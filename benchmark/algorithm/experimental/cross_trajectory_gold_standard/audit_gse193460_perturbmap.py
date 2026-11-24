#!/usr/bin/env python
"""Audit the local GSE193460 AnnData for perturbation-aware validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd


SAMPLE_PROVENANCE = pd.DataFrame(
    [
        {"sample": "KP_1", "gsm": "GSM5808054", "mouse_section": "4.1", "biological_mouse": "mouse_4"},
        {"sample": "KP_2", "gsm": "GSM5808055", "mouse_section": "4.3", "biological_mouse": "mouse_4"},
        {"sample": "KP_3", "gsm": "GSM5808056", "mouse_section": "5.1", "biological_mouse": "mouse_5"},
        {"sample": "KP_4", "gsm": "GSM5808057", "mouse_section": "7.2", "biological_mouse": "mouse_7"},
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.input, backed="r")
    obs = adata.obs.copy()
    required = {"sample", "perturbation", "cell_type"}
    missing = sorted(required - set(obs.columns))
    if missing:
        raise ValueError(f"Missing required GSE193460 fields: {missing}")

    obs["perturbation"] = obs["perturbation"].astype("string").fillna("Unannotated")
    obs["sample"] = obs["sample"].astype("string").fillna("Unknown")
    obs["cell_type"] = obs["cell_type"].astype("string").fillna("Unknown")
    if "phenotypes" in obs:
        obs["lesion_id"] = obs["phenotypes"].astype("string").fillna("Unannotated")
    else:
        obs["lesion_id"] = "Unavailable"
    mouse_map = SAMPLE_PROVENANCE.set_index("sample")["biological_mouse"]
    obs["biological_mouse"] = obs["sample"].map(mouse_map).fillna("Unknown")

    perturbation_counts = obs["perturbation"].value_counts(dropna=False).rename_axis("perturbation").reset_index(name="n_spots")
    sample_counts = pd.crosstab(obs["sample"], obs["perturbation"]).reset_index()
    cell_type_counts = pd.crosstab([obs["perturbation"], obs["sample"]], obs["cell_type"]).reset_index()
    perturbation_counts.to_csv(args.output_dir / "01_PERTURBATION_COUNTS.csv", index=False)
    sample_counts.to_csv(args.output_dir / "02_PERTURBATION_BY_SAMPLE.csv", index=False)
    cell_type_counts.to_csv(args.output_dir / "03_CELL_TYPE_BY_PERTURBATION_SAMPLE.csv", index=False)
    SAMPLE_PROVENANCE.to_csv(args.output_dir / "05_SAMPLE_PROVENANCE.csv", index=False)
    lesions = (
        obs.loc[obs["perturbation"].ne("Unannotated")]
        .groupby(["lesion_id", "perturbation", "sample", "biological_mouse"], observed=True)
        .size()
        .rename("n_spots")
        .reset_index()
    )
    lesions.to_csv(args.output_dir / "06_LESION_PROVENANCE.csv", index=False)

    contrast_rows = []
    for perturbation in sorted(set(obs["perturbation"]) - {"Control", "Unannotated"}):
        treated = obs["perturbation"].eq(perturbation)
        control = obs["perturbation"].eq("Control")
        treated_samples = set(obs.loc[treated, "sample"])
        control_samples = set(obs.loc[control, "sample"])
        shared_samples = sorted(treated_samples & control_samples)
        treated_mice = sorted(set(obs.loc[treated, "biological_mouse"]) - {"Unknown"})
        treated_lesions = sorted(set(obs.loc[treated, "lesion_id"]) - {"Unavailable", "Unannotated"})
        n_treated = int(treated.sum())
        contrast_rows.append(
            {
                "contrast": f"{perturbation}_vs_Control",
                "n_treated": n_treated,
                "n_control": int(control.sum()),
                "n_treated_samples": len(treated_samples),
                "n_shared_samples_with_control": len(shared_samples),
                "shared_samples": ";".join(shared_samples),
                "n_independent_treated_mice": len(treated_mice),
                "independent_treated_mice": ";".join(treated_mice),
                "n_treated_lesions": len(treated_lesions),
                "treated_lesions": ";".join(treated_lesions),
                "minimum_power_label": "primary" if n_treated >= 100 and len(treated_mice) >= 2 else "exploratory_low_power",
                "can_validate_expression_response": bool(n_treated >= 20 and shared_samples),
                "can_validate_complete_topology_truth": False,
            }
        )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(args.output_dir / "04_USABLE_PERTURBATION_CONTRASTS.csv", index=False)

    decision = {
        "accession": "GSE193460",
        "input": str(args.input.resolve()),
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "has_spatial_coordinates": "spatial" in adata.obsm,
        "n_sections": int(obs["sample"].nunique()),
        "n_independent_mice": int(obs.loc[obs["biological_mouse"].ne("Unknown"), "biological_mouse"].nunique()),
        "n_annotated_lesions": int(len(lesions)),
        "perturbation_annotation_scope": "tumor lesion spots; immune/background spots are unannotated",
        "independent_analysis_unit": "lesion nested within section and biological mouse, not individual spot",
        "perturbations": perturbation_counts.set_index("perturbation")["n_spots"].astype(int).to_dict(),
        "allowed_claim": "real-perturbation silver-standard validation of direction and downstream spatial response",
        "disallowed_claim": "complete multi-root trajectory topology ground truth",
        "primary_contrasts": contrasts.loc[contrasts["minimum_power_label"].eq("primary"), "contrast"].tolist(),
        "exploratory_contrasts": contrasts.loc[contrasts["minimum_power_label"].ne("primary"), "contrast"].tolist(),
    }
    (args.output_dir / "PERTURBMAP_AUDIT_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    adata.file.close()
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
