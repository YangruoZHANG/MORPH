#!/usr/bin/env python3
"""Run the production generic NicheForest upstream on a fixed-root benchmark replicate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PIPELINE = (
    REPO
    / "algorithm"
    / "CASCAT-MR-CMI-E-Gate_v7.4.3.3_anchor_bridge"
    / "run_tme_soft_multiroot_tree_pipeline_progression_refined.py"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metacell-h5ad", required=True, type=Path)
    parser.add_argument("--true-roots-csv", required=True, type=Path)
    parser.add_argument("--cell-to-metacell-csv", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task", choices=["fixed_root"], default="fixed_root")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots = pd.read_csv(args.true_roots_csv)
    mapping = pd.read_csv(args.cell_to_metacell_csv)
    required = {"root_id", "spatial_x", "spatial_y"}
    if not required.issubset(roots.columns):
        raise ValueError(f"TRUE_ROOTS is missing {sorted(required - set(roots.columns))}")
    root_mapping = roots[["root_id", "representative_cell_id"]].merge(
        mapping[["cell_name", "metacell_id"]],
        left_on="representative_cell_id",
        right_on="cell_name",
        how="left",
        validate="one_to_one",
    )
    if root_mapping["metacell_id"].isna().any() or root_mapping["metacell_id"].duplicated().any():
        raise ValueError(f"Fixed roots do not map one-to-one to metacells:\n{root_mapping}")
    fixed = pd.DataFrame({
        "dataset": args.dataset,
        "truth_root_id": root_mapping["root_id"].astype(str),
        "root_anchor_id": root_mapping["metacell_id"].astype(str),
        "root_type": "",
    })
    root_input = args.output_dir / "FIXED_ROOT_INPUT.csv"
    fixed.to_csv(root_input, index=False)
    command = [
        sys.executable,
        str(PIPELINE),
        "--adata", str(args.metacell_h5ad.resolve()),
        "--roots-csv", str(root_input.resolve()),
        "--dataset-name", args.dataset,
        "--dataset-col", "dataset",
        "--root-col", "root_anchor_id",
        "--outdir", str(args.output_dir.resolve()),
        "--input-level", "metacell",
        "--progression-source", "auto",
        "--selected-pseudotime-method", "hybrid",
        "--allow-conditional-trajectory-edges",
        "--truth-allow-internal-program-evidence-for-level2",
    ]
    subprocess.run(command, cwd=REPO, check=True)
    dataset_dir = args.output_dir / args.dataset
    status = {
        "status": "complete",
        "task": args.task,
        "dataset": args.dataset,
        "metacell_h5ad": str(args.metacell_h5ad.resolve()),
        "fixed_root_input": str(root_input.resolve()),
        "pipeline": str(PIPELINE),
        "dataset_output": str(dataset_dir.resolve()),
        "truth_fields_supplied": ["root_id", "representative_cell_id"],
        "non_root_truth_supplied": False,
    }
    (args.output_dir / "GENERIC_UPSTREAM_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
