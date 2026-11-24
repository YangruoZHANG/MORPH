#!/usr/bin/env python3
"""Create one predeclared spatial-misalignment control for frozen breast P7."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7137701)
    args = parser.parse_args()

    source_h5ad = args.source_run_dir / "event" / "synthetic.h5ad"
    source_mapping = (
        args.source_run_dir / "formal_metacell" / "cell_to_metacell_mapping.csv"
    )
    output_h5ad = args.output_run_dir / "event" / "synthetic.h5ad"
    output_mapping = (
        args.output_run_dir / "formal_metacell" / "cell_to_metacell_mapping.csv"
    )
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    output_mapping.parent.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(source_h5ad)
    original = np.asarray(adata.obsm["spatial"], dtype=float).copy()
    if not {"spatial_x", "spatial_y"}.issubset(adata.obs.columns):
        raise ValueError("Detector-facing obs spatial_x/spatial_y columns are missing")
    original_obs = adata.obs[["spatial_x", "spatial_y"]].to_numpy(float)
    if not np.allclose(original, original_obs):
        raise ValueError("obsm and detector-facing obs coordinates disagree before control")
    permutation = np.random.default_rng(args.seed).permutation(adata.n_obs)
    adata.obsm["spatial"] = original[permutation]
    adata.obs["spatial_x"] = original_obs[permutation, 0]
    adata.obs["spatial_y"] = original_obs[permutation, 1]
    adata.uns["p7_negative_control"] = {
        "control": "global_cell_coordinate_permutation",
        "seed": int(args.seed),
        "expression_unchanged": True,
        "cell_identity_unchanged": True,
        "metacell_mapping_unchanged": True,
        "t0_unchanged": True,
        "purpose": "break_expression_trajectory_to_spatial_position_alignment",
    }
    adata.write_h5ad(output_h5ad, compression="gzip")
    shutil.copy2(source_mapping, output_mapping)

    status = {
        "stage": "P7_SPATIAL_MISALIGNMENT_PREPARATION",
        "seed": int(args.seed),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "coordinate_rows_changed": int(np.sum(np.any(original != original[permutation], axis=1))),
        "detector_facing_obs_coordinates_permuted": True,
        "obsm_coordinates_permuted": True,
        "expression_unchanged": True,
        "mapping_unchanged": sha256(source_mapping) == sha256(output_mapping),
        "source_h5ad_sha256": sha256(source_h5ad),
        "output_h5ad_sha256": sha256(output_h5ad),
        "complete": True,
    }
    (args.output_run_dir / "RUN_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
