#!/usr/bin/env python3
"""Prepare the real breast compatibility input for the frozen relation detector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPO / "data" / "raw_cell_dense_embedding_bundle" / "xenium_breast" / "raw_cells.h5ad"
DEFAULT_METHOD = REPO / "outputs" / "tme_v7_4_3_3_anchor_bridge_all7" / "xenium_breast"
DEFAULT_OUTPUT = REPO / "outputs" / "NicheForest_paper_frozen_candidate" / "P6_frozen_breast" / "relation_input"


CATALOG = [
    {
        "program_id": "EFF_T_TO_CHECKPOINT",
        "program_role": "primary_relation",
        "source_genes": "CD3D;CD3E;CD8A;CD8B;GZMA;GZMB;PRF1;NKG7",
        "receiver_genes": "PDCD1;CD274;CTLA4;LAG3;HAVCR2;TIGIT",
        "preexisting_basis": "PROCESS_PROGRAMS from v7.4.3.3, defined before P6",
    },
    {
        "program_id": "PROLIFERATION_SIGNATURE",
        "program_role": "type_signature",
        "source_genes": "CD3D;CD3E;CD8A;CD8B;GZMA;GZMB;PRF1;NKG7",
        "receiver_genes": "CENPF;MKI67;TOP2A;CCND1",
        "preexisting_basis": "PROCESS_PROGRAMS from v7.4.3.3, defined before P6",
    },
    {
        "program_id": "CAF_ENV",
        "program_role": "environment_covariate",
        "source_genes": "CD3D;CD3E;CD8A;CD8B;GZMA;GZMB;PRF1;NKG7",
        "receiver_genes": "ACTA2;PDGFRA;PDGFRB;POSTN;MMP2",
        "preexisting_basis": "CAF program from v7.4.3.3, panel-observed genes only",
    },
    {
        "program_id": "MYELOID_ENV",
        "program_role": "environment_covariate",
        "source_genes": "CD3D;CD3E;CD8A;CD8B;GZMA;GZMB;PRF1;NKG7",
        "receiver_genes": "AIF1;LYZ;CD68;CD14",
        "preexisting_basis": "myeloid program from v7.4.3.3, panel-observed genes only",
    },
    {
        "program_id": "VASCULAR_ENV",
        "program_role": "environment_covariate",
        "source_genes": "CD3D;CD3E;CD8A;CD8B;GZMA;GZMB;PRF1;NKG7",
        "receiver_genes": "PECAM1;VWF;KDR;CLDN5;ANGPT2",
        "preexisting_basis": "angiogenesis/endothelial program from v7.4.3.3",
    },
    {
        "program_id": "EPITHELIAL_NC",
        "program_role": "negative_control",
        "source_genes": "CD3D;CD3E;CD8A;CD8B;GZMA;GZMB;PRF1;NKG7",
        "receiver_genes": "EPCAM;KRT8;ERBB2",
        "preexisting_basis": "tumor-epithelial compartment markers; specificity control",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--method-dir", type=Path, default=DEFAULT_METHOD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw_path = args.raw.resolve()
    method_dir = args.method_dir.resolve()
    output = args.output.resolve()
    event = output / "event"
    formal = output / "formal_metacell"
    event.mkdir(parents=True, exist_ok=True)
    formal.mkdir(parents=True, exist_ok=True)

    catalog = pd.DataFrame(CATALOG)
    raw = ad.read_h5ad(raw_path)
    genes = set(raw.var_names.astype(str))
    required = set()
    for row in catalog.itertuples(index=False):
        required.update(str(row.source_genes).split(";"))
        required.update(str(row.receiver_genes).split(";"))
    missing = sorted(required - genes)
    if missing:
        raise ValueError(f"Catalog genes missing from breast panel: {missing}")

    prepared = raw.copy()
    original_ids = prepared.obs_names.astype(str)
    prepared.obs_names = pd.Index([f"CELL_{value}" for value in original_ids])
    spatial = np.asarray(prepared.obsm["spatial"], dtype=float)
    prepared.obs["spatial_x"] = spatial[:, 0]
    prepared.obs["spatial_y"] = spatial[:, 1]
    prepared.obs["cell.type"] = prepared.obs["cell_type"].astype(str)
    prepared.uns["p6_input_contract"] = {
        "dataset": "xenium_breast_real_data",
        "compatibility_filename": "synthetic.h5ad",
        "is_synthetic": False,
        "reason": "frozen detector expects run_dir/event/synthetic.h5ad",
        "truth_available": False,
    }
    prepared_path = event / "synthetic.h5ad"
    prepared.write_h5ad(prepared_path, compression="gzip")

    mapping = pd.read_csv(method_dir / "cell_to_metacell_mapping.csv")
    mapping["cell_name"] = "CELL_" + mapping["cell_name"].astype(str)
    mapping.to_csv(formal / "cell_to_metacell_mapping.csv", index=False)
    catalog_path = output / "P6_BREAST_RELATION_PROGRAM_CATALOG.csv"
    catalog.to_csv(catalog_path, index=False)
    contract = {
        "dataset": "xenium_breast",
        "input_is_real": True,
        "topology_truth_available": False,
        "catalog_frozen_before_relation_run": True,
        "catalog_selection": "preexisting T-cell/TME programs; not selected from P6 relation outcomes",
        "primary_hypothesis": "Effector-T spatial exposure predicts local downstream checkpoint response after adjustment",
        "claim_ceiling": "confounding-aware spatially predictive cross-trajectory association",
        "causal_claim_allowed": False,
        "source_raw_h5ad": str(raw_path),
        "source_method_dir": str(method_dir),
        "n_cells": int(prepared.n_obs),
        "n_genes": int(prepared.n_vars),
        "prepared_sha256": sha256(prepared_path),
        "catalog_sha256": sha256(catalog_path),
    }
    (output / "P6_RELATION_INPUT_CONTRACT.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
