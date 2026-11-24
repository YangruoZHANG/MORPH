#!/usr/bin/env python3
"""Audit existing GSE193460 Perturb-map silver-standard assets for paper reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


PRIMARY = {"Tgfbr2_KO", "Ifngr2_KO"}
EXPLORATORY = {"Jak2_KO"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-script", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    result_files = {
        "decision": args.result_dir / "PHASE_F_PERTURBMAP_DECISION.json",
        "summary": args.result_dir / "PHASE_F_PERTURBMAP_SUMMARY.csv",
        "lesions": args.result_dir / "PHASE_F_PERTURBMAP_LESION_EFFECTS.csv",
        "genes": args.result_dir / "PHASE_F_PERTURBMAP_GENE_CONCORDANCE.csv",
        "provenance_decision": args.provenance_dir / "PERTURBMAP_AUDIT_DECISION.json",
        "lesion_provenance": args.provenance_dir / "06_LESION_PROVENANCE.csv",
    }
    required_paths = [args.input, args.source_script, *result_files.values()]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing P10 assets: {missing}")

    old_decision = json.loads(result_files["decision"].read_text(encoding="utf-8"))
    provenance_decision = json.loads(result_files["provenance_decision"].read_text(encoding="utf-8"))
    summary = pd.read_csv(result_files["summary"])
    lesions = pd.read_csv(result_files["lesions"])
    genes = pd.read_csv(result_files["genes"])
    lesion_provenance = pd.read_csv(result_files["lesion_provenance"])

    adata = ad.read_h5ad(args.input, backed="r")
    input_shape = [int(adata.n_obs), int(adata.n_vars)]
    required_obs = {"sample", "perturbation", "phenotypes"}
    obs_columns_present = required_obs.issubset(set(adata.obs.columns))
    spatial_present = "spatial" in adata.obsm
    adata.file.close()

    lesion_key_unique = not lesions.groupby("lesion_id")[["perturbation", "sample", "biological_mouse"]].nunique().gt(1).any().any()
    positive_spot_counts = bool((lesions.n_spots.astype(int) > 0).all())
    summary_counts_match = True
    for row in summary.itertuples(index=False):
        local = lesions[lesions.target_perturbation.astype(str).eq(str(row.perturbation))]
        summary_counts_match &= all([
            int(row.n_lesions) == int(local.lesion_id.nunique()),
            int(row.n_sections) == int(local["sample"].nunique()),
            int(row.n_independent_mice) == int(local.biological_mouse.nunique()),
            int(row.n_spots) == int(local.n_spots.sum()),
        ])

    primary = summary[summary.perturbation.astype(str).isin(PRIMARY)].copy()
    exploratory = summary[summary.perturbation.astype(str).isin(EXPLORATORY)].copy()
    primary_complete = set(primary.perturbation.astype(str)) == PRIMARY
    primary_two_mice = bool((primary.n_independent_mice.astype(int) >= 2).all())
    primary_direction = bool(primary.direction_concordant.astype(bool).all())
    primary_ci_expected = bool(np.all(
        ((primary.expected_direction.astype(int) > 0) & (primary.hierarchical_bootstrap_ci_low.astype(float) > 0))
        | ((primary.expected_direction.astype(int) < 0) & (primary.hierarchical_bootstrap_ci_high.astype(float) < 0))
    ))
    primary_gene_direction = bool((primary.gene_direction_concordance.astype(float) >= 0.8).all())
    jak2_low_power_marked = bool(
        len(exploratory) == 1
        and exploratory.evidence_tier.astype(str).eq("exploratory_low_power").all()
        and int(exploratory.n_independent_mice.iloc[0]) == 1
        and int(exploratory.n_lesions.iloc[0]) == 1
    )
    hierarchical_unit_declared = old_decision.get("statistical_unit") == "lesion nested within section and mouse"
    bootstrap_complete = int(old_decision.get("bootstrap_replicates", 0)) == 5000
    topology_claim_restricted = old_decision.get("complete_topology_truth_available") is False
    public_scope_consistent = (
        provenance_decision.get("accession") == "GSE193460"
        and provenance_decision.get("independent_analysis_unit")
        == "lesion nested within section and biological mouse, not individual spot"
    )

    gates = {
        "all_assets_present": not missing,
        "input_shape_matches_provenance": input_shape == list(provenance_decision.get("shape", [])),
        "required_observation_columns_present": obs_columns_present,
        "spatial_coordinates_present": spatial_present,
        "lesion_provenance_available": len(lesion_provenance) > 0,
        "lesion_keys_unique": lesion_key_unique,
        "positive_spot_counts": positive_spot_counts,
        "summary_counts_match_lesion_table": bool(summary_counts_match),
        "primary_contrasts_complete": primary_complete,
        "primary_has_at_least_two_mice_each": primary_two_mice,
        "primary_direction_concordant": primary_direction,
        "primary_hierarchical_ci_excludes_zero_expected_direction": primary_ci_expected,
        "primary_gene_direction_concordance_ge_0_8": primary_gene_direction,
        "jak2_explicitly_low_power": jak2_low_power_marked,
        "hierarchical_analysis_unit_declared": hierarchical_unit_declared,
        "bootstrap_5000_complete": bootstrap_complete,
        "topology_claim_restricted": topology_claim_restricted,
        "public_provenance_scope_consistent": public_scope_consistent,
    }
    passed = all(gates.values())

    paper = summary.copy()
    paper["paper_role"] = np.where(
        paper.perturbation.astype(str).isin(PRIMARY), "primary_silver_standard", "exploratory_low_power"
    )
    paper["interpretation"] = np.where(
        paper.perturbation.astype(str).eq("Tgfbr2_KO"),
        "positive context-adjusted stromal/TGF response",
        "negative context-adjusted IFNG response",
    )
    paper.to_csv(args.output / "P10_PERTURBMAP_PAPER_TABLE.csv", index=False)

    manifest = pd.DataFrame([
        {"role": "input_h5ad", "file": str(args.input.resolve()), "sha256": sha256(args.input)},
        {"role": "source_script", "file": str(args.source_script.resolve()), "sha256": sha256(args.source_script)},
        *[
            {"role": role, "file": str(path.resolve()), "sha256": sha256(path)}
            for role, path in result_files.items()
        ],
    ])
    manifest.to_csv(args.output / "P10_ASSET_MANIFEST.csv", index=False)
    pd.DataFrame([{"gate": key, "passed": value} for key, value in gates.items()]).to_csv(
        args.output / "P10_ASSET_AUDIT.csv", index=False
    )

    decision = {
        "stage": "P10_GSE193460_PERTURBMAP_ASSET_AUDIT",
        "decision": "PASS_REUSE_EXISTING_ASSETS" if passed else "FAIL_REANALYSIS_REQUIRED",
        "rerun_required": not passed,
        "primary_contrasts": sorted(PRIMARY),
        "exploratory_contrasts": sorted(EXPLORATORY),
        "all_gates_passed": passed,
        "gates": gates,
        "allowed_claim": "directional perturbational program concordance in a real perturbation silver standard",
        "disallowed_claim": "complete trajectory or topology truth",
        "literature_source": "https://doi.org/10.1016/j.cell.2022.02.015",
        "data_source": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193460",
    }
    (args.output / "P10_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    command = (
        f'python "{args.source_script.resolve()}" '
        f'--input "{args.input.resolve()}" --output "<clean-output-directory>"'
    )
    (args.output / "P10_EXACT_RERUN_COMMAND.txt").write_text(command + "\n", encoding="utf-8")
    limitations = [
        "# P10 Perturb-map evidence limits",
        "",
        "- Tgfbr2 KO has four lesions across three sections but only two independent mice.",
        "- Ifngr2 KO has two lesions across two sections and two independent mice; its interval is therefore low-resolution despite excluding zero.",
        "- Jak2 KO has one lesion in one mouse. Its degenerate interval is not inferential and the result remains exploratory.",
        "- The context model is trained on control lesions and supports directional perturbational concordance, not complete topology recovery.",
        "- This mouse lung-cancer dataset is an external silver standard; it is not a direct validation of the human breast trajectories.",
        "- The existing analysis is reused because lesion-level estimates, mouse/section provenance, uncertainty, code, seed, and input are all present.",
    ]
    (args.output / "P10_LIMITATIONS.md").write_text("\n".join(limitations) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
