#!/usr/bin/env python3
"""Topology-identifiable extension of the event-conditioned benchmark.

The parent generator writes event-local response programs and T1 truth, but
does not make the selected T1 parent identifiable from observed expression.
This extension adds a declared transition program: exposed receiver cells
partially adopt the program of the state to which their branch is rerouted.
The truth tables remain unavailable to the inference method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse


HERE = Path(__file__).resolve().parent
VERSION = "BENCHMARK_TRUTH_GENERATOR_V6_TOPOLOGY_IDENTIFIABLE"
def genes_from_range(value: object) -> set[str]:
    text = str(value)
    if "-" not in text:
        return {token for token in text.split(";") if token}
    left, right = text.split("-", maxsplit=1)
    return {f"G{i}" for i in range(int(left[1:]), int(right[1:]) + 1)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scenario-json", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(HERE / "build_event_conditioned_benchmark_v3.py"),
        "--base-dir", str(args.base_dir), "--output-dir", str(args.output_dir),
        "--seed", str(args.seed),
    ]
    if args.scenario_json:
        command.extend(["--scenario-json", str(args.scenario_json)])
    subprocess.run(command, check=True)

    h5ad_path = args.output_dir / "synthetic.h5ad"
    adata = ad.read_h5ad(h5ad_path)
    values = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    values = values.astype(float, copy=True)
    obs = adata.obs.copy()
    events = pd.read_csv(args.output_dir / "truth" / "TRUE_EVENTS.csv")
    t0 = pd.read_csv(args.output_dir / "truth" / "TRUE_T0_EDGES.csv")
    t1 = pd.read_csv(args.output_dir / "truth" / "TRUE_T1_EDGES.csv")
    contact_path = args.output_dir / "truth" / "TRUE_SPATIAL_CONTACTS.csv"
    contacts = pd.read_csv(contact_path)
    lookup = {str(gene): index for index, gene in enumerate(adata.var_names)}
    assertions: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []

    for event_index, event in enumerate(events.itertuples(index=False)):
        event_id = str(event.event_id)
        causal = str(event.causal).strip().lower() in {"true", "1", "yes"}
        labels = obs.causal_event_id.astype(str).str.split(";")
        active = labels.map(lambda parts: event_id in parts).to_numpy(bool)
        active_cells = set(obs.index[active].astype(str))
        contact_mask = contacts.event_id.astype(str).eq(event_id)
        contacts.loc[contact_mask, "is_event_exposed"] = contacts.loc[
            contact_mask, "receiver_cell_id"
        ].astype(str).isin(active_cells)
        contact_active = set(
            contacts.loc[
                contact_mask & contacts.is_event_exposed.astype(bool),
                "receiver_cell_id",
            ].astype(str)
        )
        assertions.append({
            "assertion": f"contact_exposure_matches_event_cells_{event_id}",
            "passed": bool(contact_active == active_cells),
            "value": len(contact_active.symmetric_difference(active_cells)),
            "minimum_signed": 0,
        })
        if not causal:
            assertions.append({
                "assertion": f"null_has_no_topology_transition_program_{event_id}",
                "passed": bool(not active.any()), "value": int(active.sum()),
                "minimum_signed": 0,
            })
            continue
        event_edge = t1[
            t1.root_id.astype(str).eq(str(event.receiver_root))
            & t1.edge_role.astype(str).eq(f"event:{event_id}")
        ]
        removed = pd.merge(
            t0, t1, on=["root_id", "source", "target"], how="left", indicator=True
        )
        removed = removed[
            removed.root_id.astype(str).eq(str(event.receiver_root))
            & removed._merge.eq("left_only")
        ]
        removed = removed[
            removed.target.astype(str).eq(str(event_edge.iloc[0].target))
        ]
        if len(event_edge) != 1 or len(removed) < 1 or not active.any():
            raise RuntimeError(f"Incomplete topology truth for {event_id}")
        new_parent = str(event_edge.iloc[0].source)
        event_child = str(event_edge.iloc[0].target)
        old_parent = str(removed.iloc[0].source)
        receiver = obs.root_id.astype(str).eq(str(event.receiver_root)).to_numpy()
        truth_graph = nx.DiGraph()
        local_truth = t1[t1.root_id.astype(str).eq(str(event.receiver_root))]
        truth_graph.add_edges_from(local_truth[["source", "target"]].astype(str).itertuples(
            index=False, name=None
        ))
        downstream_states = {event_child} | set(nx.descendants(truth_graph, event_child))
        transition = receiver & obs.truth_state_T0.astype(str).isin(downstream_states).to_numpy()
        donor = receiver & obs.truth_state_T0.astype(str).eq(new_parent).to_numpy()
        old_donor = receiver & obs.truth_state_T0.astype(str).eq(old_parent).to_numpy()
        if donor.sum() < 5 or old_donor.sum() < 5:
            raise RuntimeError(
                f"Insufficient parent-state donors for {event_id}: "
                f"new={donor.sum()} old={old_donor.sum()}"
            )
        gene_start = 271 + 10 * event_index
        coupling_genes = tuple(f"G{i}" for i in range(gene_start, gene_start + 10))
        if any(gene not in lookup for gene in coupling_genes):
            raise RuntimeError(f"Missing topology transition genes for {event_id}")
        columns = np.asarray([lookup[gene] for gene in coupling_genes], dtype=int)
        transition_index = np.flatnonzero(transition)
        donor_index = np.flatnonzero(donor)
        transition_before = float(values[np.ix_(transition, columns)].mean())
        donor_before = float(values[np.ix_(donor, columns)].mean())
        old_before = float(values[np.ix_(old_donor, columns)].mean())
        amplitude = float(50.0 + 30.0 * float(event.effect_strength))
        values[np.ix_(transition_index, columns)] += amplitude
        values[np.ix_(donor_index, columns)] += amplitude
        transition_after = float(values[np.ix_(transition, columns)].mean())
        donor_after = float(values[np.ix_(donor, columns)].mean())
        old_after = float(values[np.ix_(old_donor, columns)].mean())
        transition_delta = transition_after - transition_before
        donor_delta = donor_after - donor_before
        shared_parent_margin = min(transition_after, donor_after) - old_after
        assertions.extend([
            {
                "assertion": f"topology_parent_program_written_{event_id}",
                "passed": bool(transition_delta > 10.0 and donor_delta > 10.0),
                "value": min(transition_delta, donor_delta),
                "minimum_signed": 10.0,
            },
            {
                "assertion": f"new_parent_preferred_over_old_parent_{event_id}",
                "passed": bool(shared_parent_margin > 10.0),
                "value": shared_parent_margin,
                "minimum_signed": 10.0,
            },
        ])
        manifest.append({
            "event_id": event_id, "receiver_root": str(event.receiver_root),
            "event_type": str(event.event_type), "old_parent_state": old_parent,
            "new_parent_state": new_parent, "event_child_state": event_child,
            "n_contact_exposed_cells": int(active.sum()),
            "n_downstream_transition_cells": int(transition.sum()),
            "coupling_genes": ";".join(coupling_genes),
            "coupling_amplitude": amplitude,
            "mechanism": "event_triggered_downstream_program_shared_with_new_parent_state",
            "transition_program_delta": transition_delta,
            "new_parent_program_delta": donor_delta,
            "shared_parent_margin_over_old_parent": shared_parent_margin,
        })

    canonical = pd.DataFrame({
        "effect_level": ["null", "weak", "medium", "strong"],
        "effect_strength": [0.0, 0.35, 0.7, 1.1],
    })
    canonical["topology_coupling_amplitude"] = np.where(
        canonical.effect_strength.eq(0.0), 0.0,
        50.0 + 30.0 * canonical.effect_strength,
    )
    assertions.append({
        "assertion": "causal_effect_levels_have_ordered_topology_coupling",
        "passed": bool(np.all(np.diff(canonical.topology_coupling_amplitude) > 0)),
        "value": int(np.all(np.diff(canonical.topology_coupling_amplitude) > 0)),
        "minimum_signed": 1,
    })
    assertion_frame = pd.DataFrame(assertions)
    assertion_frame.to_csv(
        args.output_dir / "TOPOLOGY_COUPLING_ASSERTIONS.csv", index=False
    )
    if not assertion_frame.passed.astype(bool).all():
        raise RuntimeError("Topology-identifiable generator assertion failed")
    contacts.to_csv(contact_path, index=False)
    pd.DataFrame(manifest).to_csv(
        args.output_dir / "TOPOLOGY_COUPLING_MANIFEST.csv", index=False
    )
    canonical.to_csv(
        args.output_dir / "GENERATOR_V4_CANONICAL_EFFECT_LEVELS.csv", index=False
    )
    adata.X = sparse.csr_matrix(np.clip(values, 0.0, None))
    adata.uns["benchmark_spec"] = "BENCHMARK_SPEC_V4_TOPOLOGY_IDENTIFIABLE"
    adata.uns["truth_generator_version"] = VERSION
    adata.uns["topology_coupling_truth_visibility"] = "generator_audit_only"
    adata.write_h5ad(h5ad_path, compression="gzip")
    version = {
        "benchmark_specification": "BENCHMARK_SPEC_V4_TOPOLOGY_IDENTIFIABLE",
        "truth_generator_version": VERSION,
        "parent_generator": "BENCHMARK_TRUTH_GENERATOR_V5_4_2_MULTI_SOURCE_ASSERTION_REPAIR",
        "seed": args.seed, "all_program_assertions_passed": True,
        "topology_parent_identifiable_from_observed_transition_program": True,
        "synthetic_h5ad_sha256": sha256(h5ad_path),
        "truth_visibility": "audit_only",
    }
    (args.output_dir / "GENERATOR_VERSION.json").write_text(
        json.dumps(version, indent=2), encoding="utf-8"
    )
    print(json.dumps(version, indent=2))


if __name__ == "__main__":
    main()
