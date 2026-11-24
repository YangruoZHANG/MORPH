#!/usr/bin/env python3
"""Audit the frozen v7.8.1 A_raw root-column and first-argmax contract.

This script is read-only with respect to historical outputs.  It records the
exact root-column order, quantifies all-zero/no-response nodes and cells, and
hashes the as-run and corrected root-label vectors.  The audit does not change
the frozen A0 trajectory layer or the historical v7.8.1 relation outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = (
    "GSE",
    "GSE193460",
    "GSE245582",
    "xenium_breast",
    "xenium_lung",
    "xenium_melanoma",
    "xenium_pancreatic",
)
ZERO_TOLERANCE = 1e-12


def vector_hash(values) -> str:
    text = "\n".join(map(str, values)) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def float64_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def audit_dataset(project: Path, dataset: str):
    a_path = (
        project
        / "outputs"
        / "v7_4_7_0_all7_relation_aware_FINAL"
        / "trees"
        / dataset
        / "A_RAW_BY_NODE.csv"
    )
    map_path = (
        project
        / "NicheForest_v7_4_6_all7_reproduction_bundle"
        / "data"
        / dataset
        / "cell_to_metacell.csv"
    )
    raw_path = (
        project
        / "data"
        / "raw_cell_dense_embedding_bundle"
        / dataset
        / "raw_cell_dense_coordinates.csv.gz"
    )
    a_raw = pd.read_csv(a_path).sort_values("node_id", kind="stable").reset_index(drop=True)
    root_columns = [column for column in a_raw.columns if column.startswith("A_raw_")]
    roots = [column.removeprefix("A_raw_") for column in root_columns]
    weights = a_raw[root_columns].to_numpy(float)
    row_sum = weights.sum(axis=1)
    zero_nodes = np.isclose(row_sum, 0.0, atol=ZERO_TOLERANCE, rtol=0.0)
    maxima = weights.max(axis=1)
    tied = np.isclose(weights, maxima[:, None], atol=ZERO_TOLERANCE, rtol=0.0).sum(axis=1) > 1
    nonzero_tied = tied & ~zero_nodes

    mapping = pd.read_csv(map_path, usecols=["cell_name", "metacell_index"])
    if mapping["cell_name"].duplicated().any():
        conflicts = mapping.groupby("cell_name")["metacell_index"].nunique().gt(1)
        if conflicts.any():
            raise RuntimeError(f"{dataset}: conflicting duplicate cell mappings")
        mapping = mapping.drop_duplicates("cell_name", keep="first")
    raw = pd.read_csv(raw_path, usecols=["cell_id"])
    lookup = mapping.set_index(mapping["cell_name"].astype(str))["metacell_index"]
    node_id = raw["cell_id"].astype(str).map(lookup)
    missing = int(node_id.isna().sum())
    if missing:
        raise RuntimeError(f"{dataset}: {missing} raw cells lack formal node mapping")
    node_id = node_id.astype(int).to_numpy()
    if node_id.min(initial=0) < 0 or node_id.max(initial=-1) >= len(weights):
        raise RuntimeError(f"{dataset}: mapped node index outside A_raw universe")

    cell_weights = weights[node_id]
    cell_sum = cell_weights.sum(axis=1)
    cell_zero = np.isclose(cell_sum, 0.0, atol=ZERO_TOLERANCE, rtol=0.0)
    cell_maxima = cell_weights.max(axis=1)
    cell_tied = (
        np.isclose(cell_weights, cell_maxima[:, None], atol=ZERO_TOLERANCE, rtol=0.0)
        .sum(axis=1)
        > 1
    )
    cell_nonzero_tied = cell_tied & ~cell_zero
    first_index = np.argmax(cell_weights, axis=1)
    as_run_labels = np.asarray([roots[index] for index in first_index], dtype=object)
    corrected_labels = as_run_labels.copy()
    corrected_labels[cell_zero] = "NO_RESPONSE"
    corrected_labels[cell_nonzero_tied] = "AMBIGUOUS"
    primary_support_labels = corrected_labels.copy()

    reversed_roots = roots[::-1]
    reversed_weights = cell_weights[:, ::-1]
    reversed_index = np.argmax(reversed_weights, axis=1)
    reversed_labels = np.asarray([reversed_roots[index] for index in reversed_index], dtype=object)

    order_rows = []
    order_hash = vector_hash(roots)
    for index, (root, column) in enumerate(zip(roots, root_columns)):
        order_rows.append(
            {
                "dataset": dataset,
                "root_column_index_zero_based": index,
                "root_cluster": root,
                "a_raw_column": column,
                "n_formal_nodes": len(weights),
                "a_raw_column_sha256_float64_le": float64_hash(weights[:, index]),
                "frozen_root_order_sha256": order_hash,
                "firstargmax_priority": index,
            }
        )

    no_response_row = {
        "dataset": dataset,
        "n_formal_nodes": len(weights),
        "n_zero_response_nodes": int(zero_nodes.sum()),
        "n_tied_max_nodes": int(tied.sum()),
        "n_nonzero_tied_max_nodes": int(nonzero_tied.sum()),
        "n_mapped_cells": len(raw),
        "n_zero_response_cells": int(cell_zero.sum()),
        "zero_response_cell_fraction": float(cell_zero.mean()),
        "n_nonzero_tied_max_cells": int(cell_nonzero_tied.sum()),
        "first_assigned_root": roots[0],
        "frozen_root_order": "|".join(roots),
        "frozen_root_order_sha256": order_hash,
        "as_run_root_label_vector_sha256": vector_hash(as_run_labels),
    }

    assignment_row = {
        "dataset": dataset,
        "n_raw_cells": len(raw),
        "mapping_coverage": 1.0,
        "n_unique_nonzero_assignments": int((~cell_zero & ~cell_nonzero_tied).sum()),
        "n_no_response_assignments": int(cell_zero.sum()),
        "n_ambiguous_nonzero_assignments": int(cell_nonzero_tied.sum()),
        "as_run_firstargmax_root_label_vector_sha256": vector_hash(as_run_labels),
        "corrected_label_vector_sha256": vector_hash(corrected_labels),
        "primary_support_only_label_vector_sha256": vector_hash(primary_support_labels),
        "reversed_column_firstargmax_label_vector_sha256": vector_hash(reversed_labels),
        "as_run_equals_corrected": bool(np.array_equal(as_run_labels, corrected_labels)),
        "as_run_equals_reversed_order": bool(np.array_equal(as_run_labels, reversed_labels)),
        "root_column_order_invariant_for_all_cells": bool(np.array_equal(as_run_labels, reversed_labels)),
        "historical_relation_output_modified": False,
    }
    return order_rows, no_response_row, assignment_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/v7_8_1_no_response_label_sensitivity_audit/00_root_label_contract"),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output if args.output.is_absolute() else project / args.output
    output.mkdir(parents=True, exist_ok=True)

    order_rows, no_response_rows, assignment_rows = [], [], []
    for dataset in DATASETS:
        order, no_response, assignment = audit_dataset(project, dataset)
        order_rows.extend(order)
        no_response_rows.append(no_response)
        assignment_rows.append(assignment)

    order_frame = pd.DataFrame(order_rows)
    no_response_frame = pd.DataFrame(no_response_rows)
    assignment_frame = pd.DataFrame(assignment_rows)
    order_frame.to_csv(output / "FROZEN_ROOT_COLUMN_ORDER.csv", index=False, encoding="utf-8-sig")
    no_response_frame.to_csv(output / "NO_RESPONSE_FIRSTARGMAX_AUDIT.csv", index=False, encoding="utf-8-sig")
    assignment_frame.to_csv(output / "A_RAW_ROOT_LABEL_ASSIGNMENT_AUDIT.csv", index=False, encoding="utf-8-sig")

    summary = {
        "audit_id": "v7.8.1_root_label_contract_audit_v1",
        "datasets": list(DATASETS),
        "zero_tolerance": ZERO_TOLERANCE,
        "root_order_contract_pass": bool(len(order_frame) == order_frame["root_cluster"].notna().sum()),
        "all_mapping_coverage_one": bool(np.allclose(assignment_frame["mapping_coverage"], 1.0)),
        "all_nonzero_ties_absent": bool((no_response_frame["n_nonzero_tied_max_nodes"] == 0).all()),
        "any_no_response_cells": bool((no_response_frame["n_zero_response_cells"] > 0).any()),
        "all_root_order_invariant": bool(
            assignment_frame["root_column_order_invariant_for_all_cells"].all()
        ),
        "interpretation": (
            "Cell-to-formal-node mapping is exact, but the frozen first-argmax relation labels "
            "are not invariant to arbitrary root-column permutations when no-response cells exist."
        ),
    }
    (output / "ROOT_LABEL_CONTRACT_AUDIT.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
