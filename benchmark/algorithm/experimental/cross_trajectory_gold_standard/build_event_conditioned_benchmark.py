#!/usr/bin/env python
"""Build a multi-root, event-conditioned trajectory gold standard.

scMultiSim provides expression, within-root differentiation, spatial location,
and CCI structure. This layer adds explicit cross-root interventions and exact
T0/T1/counterfactual topology truth for NicheForest evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse


BRANCHES = ("B0", "B1", "B2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71121)
    parser.add_argument("--scenario-json", type=Path)
    return parser.parse_args()


def default_events() -> list[dict[str, object]]:
    return [
        {
            "event_id": "E_PROMOTE_R0_R1", "event_type": "promotion", "source_root": "R0",
            "receiver_root": "R1", "affected_branch": "B1", "decision_from": "B0",
            "decision_to": "B1", "removed": ("R1_T1", "R1_B1_S1"),
            "added": ("R1_ROOT", "R1_B1_S1"), "causal": True, "topology_changed": True,
            "source_tau_start": 0.55, "source_tau_end": 0.90, "receiver_tau_start": 0.20,
            "receiver_tau_end": 0.60, "effect_strength": 1.0, "exposure_quantile": 0.65,
        },
        {
            "event_id": "E_INHIBIT_R1_R2", "event_type": "inhibition", "source_root": "R1",
            "receiver_root": "R2", "affected_branch": "B2", "decision_from": "B2",
            "decision_to": "B0", "removed": ("R2_B2_S1", "R2_B2_S2"),
            "added": ("R2_B0_S1", "R2_B2_S2"), "causal": True, "topology_changed": True,
            "source_tau_start": 0.50, "source_tau_end": 0.90, "receiver_tau_start": 0.35,
            "receiver_tau_end": 0.75, "effect_strength": 0.75, "exposure_quantile": 0.65,
        },
        {
            "event_id": "E_NULL_R0_R2", "event_type": "null_contact", "source_root": "R0",
            "receiver_root": "R2", "affected_branch": "none", "decision_from": "none",
            "decision_to": "none", "removed": ("", ""), "added": ("", ""),
            "causal": False, "topology_changed": False, "source_tau_start": 0.45,
            "source_tau_end": 0.90, "receiver_tau_start": 0.20, "receiver_tau_end": 0.80,
            "effect_strength": 0.0, "exposure_quantile": 0.65,
        },
    ]


def adjust_spatial_overlap(meta: pd.DataFrame, level: str) -> pd.DataFrame:
    factors = {"absent": 2.5, "poor": 1.5, "adequate": 1.0, "excessive": 0.35}
    factor = factors.get(str(level), 1.0)
    if factor == 1.0:
        return meta
    out = meta.copy()
    grand = out[["spatial_x", "spatial_y"]].mean().to_numpy(float)
    for _, index in out.groupby("root_id").groups.items():
        centroid = out.loc[index, ["spatial_x", "spatial_y"]].mean().to_numpy(float)
        shift = (factor - 1.0) * (centroid - grand)
        out.loc[index, ["spatial_x", "spatial_y"]] = (
            out.loc[index, ["spatial_x", "spatial_y"]].to_numpy(float) + shift
        )
    return out


def perturb_spatial_geometry(
    meta: pd.DataFrame, spatial_noise: float, membership_ambiguity: float, rng: np.random.Generator
) -> pd.DataFrame:
    out = meta.copy()
    coords = out[["spatial_x", "spatial_y"]].to_numpy(float)
    scale = float(np.mean(np.std(coords, axis=0)))
    if spatial_noise > 0:
        coords += rng.normal(0.0, spatial_noise * max(scale, 1e-6), size=coords.shape)
        out[["spatial_x", "spatial_y"]] = coords
    fraction = float(np.clip(membership_ambiguity, 0.0, 0.8))
    if fraction <= 0:
        return out
    roots = sorted(out["root_id"].astype(str).unique())
    centroids = out.groupby("root_id")[["spatial_x", "spatial_y"]].mean()
    for index, root in enumerate(roots):
        candidates = out.index[out["root_id"].astype(str).eq(root)].to_numpy()
        n_move = int(round(len(candidates) * fraction))
        if n_move <= 0:
            continue
        selected = rng.choice(candidates, size=n_move, replace=False)
        receiver = roots[(index + 1) % len(roots)]
        target = centroids.loc[receiver].to_numpy(float)
        current = out.loc[selected, ["spatial_x", "spatial_y"]].to_numpy(float)
        out.loc[selected, ["spatial_x", "spatial_y"]] = 0.30 * current + 0.70 * target
    return out


def scenario_events(roots: tuple[str, ...], scenario: dict[str, object]) -> list[dict[str, object]]:
    scenario_type = str(scenario.get("event_type", "promotion"))
    concurrent = max(1, min(int(scenario.get("concurrent_events", 1)), len(roots)))
    effect_strength = float(scenario.get("effect_strength", 0.7))
    exposure_quantile = 0.90 if str(scenario.get("exposure_overlap", "adequate")) == "sparse" else 0.65
    window_width = float(scenario.get("event_window_width", 0.50))
    source_start, source_end = max(0.0, 0.70 - window_width / 2), min(1.0, 0.70 + window_width / 2)
    receiver_start, receiver_end = max(0.0, 0.50 - window_width / 2), min(1.0, 0.50 + window_width / 2)
    null_types = {"null_contact", "confounding_only", "overlap_unrelated"}
    events: list[dict[str, object]] = []
    for event_index in range(concurrent):
        source = roots[event_index % len(roots)]
        receiver = roots[(event_index + 1) % len(roots)]
        event_type = scenario_type
        if scenario_type in {"overlapping_events", "ambiguous_membership", "wrong_soft_membership", "sparse_exposure"}:
            event_type = "promotion" if event_index % 2 == 0 else "inhibition"
        causal = scenario_type not in null_types
        if event_type in {"promotion", "collision_convergence"}:
            affected, decision_from, decision_to = "B1", "B0", "B1"
            removed = (f"{receiver}_T1", f"{receiver}_B1_S1")
            added_parent = f"{receiver}_B0_S1" if event_type == "collision_convergence" else f"{receiver}_ROOT"
            added = (added_parent, f"{receiver}_B1_S1")
        else:
            affected, decision_from, decision_to = "B2", "B2", "B0"
            removed = (f"{receiver}_B2_S1", f"{receiver}_B2_S2")
            added = (f"{receiver}_B0_S1", f"{receiver}_B2_S2")
        if not causal:
            affected, decision_from, decision_to = "none", "none", "none"
            removed, added = ("", ""), ("", "")
        events.append({
            "event_id": f"E_{scenario_type.upper()}_{event_index + 1}_{source}_{receiver}",
            "event_type": scenario_type if scenario_type in null_types | {"collision_convergence"} else event_type,
            "source_root": source,
            "receiver_root": receiver,
            "affected_branch": affected,
            "decision_from": decision_from,
            "decision_to": decision_to,
            "removed": removed,
            "added": added,
            "causal": causal,
            "topology_changed": causal,
            "source_tau_start": source_start,
            "source_tau_end": source_end,
            "receiver_tau_start": receiver_start,
            "receiver_tau_end": receiver_end,
            "effect_strength": effect_strength,
            "exposure_quantile": exposure_quantile,
        })
    return events


def make_t0_edges(roots: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for root in roots:
        root_node = f"{root}_ROOT"
        trunk = f"{root}_T1"
        rows.append({"root_id": root, "source": root_node, "target": trunk, "edge_role": "trunk"})
        for branch in BRANCHES:
            previous = trunk
            for segment in range(1, 4):
                node = f"{root}_{branch}_S{segment}"
                rows.append(
                    {
                        "root_id": root,
                        "source": previous,
                        "target": node,
                        "edge_role": "baseline_branch",
                    }
                )
                previous = node
    return pd.DataFrame(rows)


def replace_edge(edges: pd.DataFrame, root: str, old: tuple[str, str], new: tuple[str, str], event_id: str) -> pd.DataFrame:
    out = edges.copy()
    mask = (out["root_id"] == root) & (out["source"] == old[0]) & (out["target"] == old[1])
    if int(mask.sum()) != 1:
        raise ValueError(f"Expected one editable edge for {event_id}: {old}")
    out = out.loc[~mask].copy()
    out = pd.concat(
        [
            out,
            pd.DataFrame(
                [{"root_id": root, "source": new[0], "target": new[1], "edge_role": f"event:{event_id}"}]
            ),
        ],
        ignore_index=True,
    )
    return out


def assert_forest(edges: pd.DataFrame) -> None:
    for root, part in edges.groupby("root_id"):
        graph = nx.DiGraph()
        graph.add_edges_from(part[["source", "target"]].itertuples(index=False, name=None))
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(f"Cycle in {root}")
        root_nodes = [node for node, indegree in graph.in_degree() if indegree == 0]
        if len(root_nodes) != 1 or not nx.is_weakly_connected(graph):
            raise ValueError(f"Invalid tree structure in {root}: roots={root_nodes}")


def minmax(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    span = float(np.nanmax(array) - np.nanmin(array))
    if span <= 0:
        return np.zeros_like(array)
    return (array - np.nanmin(array)) / span


def choose_branch(pop: pd.Series) -> pd.Series:
    labels = sorted(pop.astype(str).unique())
    mapping = {label: BRANCHES[index % len(BRANCHES)] for index, label in enumerate(labels)}
    return pop.astype(str).map(mapping)


def spatial_contact(
    meta: pd.DataFrame,
    source_root: str,
    receiver_root: str,
    source_tau_start: float = 0.0,
    source_tau_end: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, pd.DataFrame]:
    source_window = (
        meta["root_id"].eq(source_root)
        & meta["truth_pseudotime"].between(source_tau_start, source_tau_end, inclusive="both")
    )
    source_frame = meta.loc[source_window, ["cell_id", "spatial_x", "spatial_y"]]
    if source_frame.empty:
        raise ValueError(f"No source cells in tau window for {source_root}: {source_tau_start}-{source_tau_end}")
    source = source_frame[["spatial_x", "spatial_y"]].to_numpy(float)
    receiver_mask = meta["root_id"].eq(receiver_root).to_numpy()
    receiver_frame = meta.loc[receiver_mask, ["cell_id", "spatial_x", "spatial_y"]]
    receiver = receiver_frame[["spatial_x", "spatial_y"]].to_numpy(float)
    distances = np.sqrt(((receiver[:, None, :] - source[None, :, :]) ** 2).sum(axis=2))
    nearest_index = distances.argmin(axis=1)
    nearest = distances[np.arange(len(receiver)), nearest_index]
    radius = float(np.quantile(nearest, 0.35))
    contact = np.exp(-np.square(nearest / max(radius, 1e-6)))
    full = np.zeros(len(meta), dtype=float)
    full[receiver_mask] = contact
    event_center = receiver[np.argmax(contact)]
    contacts = pd.DataFrame(
        {
            "source_root": source_root,
            "receiver_root": receiver_root,
            "receiver_cell_id": receiver_frame["cell_id"].to_numpy(str),
            "nearest_source_cell_id": source_frame["cell_id"].to_numpy(str)[nearest_index],
            "distance": nearest,
            "contact_score": contact,
        }
    )
    return full, event_center, radius, contacts


def state_labels(root: pd.Series, branch: pd.Series, pseudotime: pd.Series) -> pd.Series:
    labels = []
    for r, b, tau in zip(root.astype(str), branch.astype(str), pseudotime.astype(float)):
        if tau <= 0.15:
            labels.append(f"{r}_ROOT")
        elif tau <= 0.35:
            labels.append(f"{r}_T1")
        else:
            segment = min(3, max(1, int(np.ceil((tau - 0.35) / 0.65 * 3))))
            labels.append(f"{r}_{b}_S{segment}")
    return pd.Series(np.asarray(labels, dtype=object), index=root.index)


def add_program(counts: np.ndarray, rows: np.ndarray, genes: slice, strength: np.ndarray) -> None:
    selected = np.flatnonzero(rows)
    if not len(selected):
        return
    gene_indices = np.arange(counts.shape[1])[genes]
    counts[np.ix_(selected, gene_indices)] += strength[selected, None]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = pd.read_csv(args.base_dir / "scmultisim_counts.csv.gz", index_col=0)
    meta = pd.read_csv(args.base_dir / "scmultisim_cell_metadata.csv")
    spatial = pd.read_csv(args.base_dir / "scmultisim_spatial.csv")
    meta = meta.merge(spatial, on="cell_id", validate="one_to_one")
    counts = counts.loc[meta["cell_id"]]
    roots = tuple(sorted(meta["root_id"].astype(str).unique()))
    scenario = json.loads(args.scenario_json.read_text(encoding="utf-8")) if args.scenario_json else None
    if scenario:
        requested_roots = int(scenario.get("n_roots", len(roots)))
        if requested_roots != len(roots):
            raise ValueError(f"Scenario requests {requested_roots} roots but base data contain {len(roots)}")
        meta = adjust_spatial_overlap(meta, str(scenario.get("spatial_overlap", "adequate")))
        meta = perturb_spatial_geometry(
            meta,
            float(scenario.get("spatial_noise", 0.0)),
            float(scenario.get("membership_ambiguity", 0.0)),
            rng,
        )

    meta["truth_pseudotime"] = meta.groupby("root_id", group_keys=False)["depth"].transform(minmax)
    meta["truth_branch_T0"] = meta.groupby("root_id", group_keys=False)["pop"].apply(choose_branch)
    meta["truth_branch_T1"] = meta["truth_branch_T0"]
    meta["event_exposure"] = 0.0
    meta["causal_event_id"] = "none"
    meta["null_contact"] = False
    meta["confounded_exposure"] = False

    t0 = make_t0_edges(roots)
    events = scenario_events(roots, scenario) if scenario else default_events()

    t1 = t0.copy()
    event_rows: list[dict[str, object]] = []
    contact_rows: list[pd.DataFrame] = []
    values = counts.to_numpy(dtype=float)
    for event_index, event in enumerate(events):
        exposure, center, radius, contacts = spatial_contact(
            meta,
            event["source_root"],
            event["receiver_root"],
            float(event["source_tau_start"]),
            float(event["source_tau_end"]),
        )
        receiver = meta["root_id"].eq(event["receiver_root"]).to_numpy()
        receiver_window = meta["truth_pseudotime"].between(
            float(event["receiver_tau_start"]), float(event["receiver_tau_end"]), inclusive="both"
        ).to_numpy()
        if event["decision_from"] == "none":
            eligible = receiver & receiver_window
        else:
            eligible = receiver & receiver_window & meta["truth_branch_T0"].eq(event["decision_from"]).to_numpy()
        if not eligible.any():
            raise ValueError(f"No eligible receiver cells for {event['event_id']}")
        active = eligible & (exposure >= np.quantile(exposure[eligible], float(event["exposure_quantile"])))
        contacts.insert(0, "event_id", event["event_id"])
        contacts["is_event_exposed"] = contacts["receiver_cell_id"].isin(meta.loc[active, "cell_id"])
        contacts["is_true_event"] = bool(event["causal"])
        contact_rows.append(contacts)
        meta.loc[active, "event_exposure"] = np.maximum(meta.loc[active, "event_exposure"], exposure[active])
        if event["causal"]:
            meta.loc[active, "causal_event_id"] = event["event_id"]
            meta.loc[active, "truth_branch_T1"] = event["decision_to"]
            program = slice(10 + 10 * event_index, 20 + 10 * event_index)
            direction = 1.0 if event["event_type"] in {"promotion", "collision_convergence"} else -0.75
            add_program(values, active, program, event["effect_strength"] * direction * (2.5 + 2.0 * exposure))
            t1 = replace_edge(t1, event["receiver_root"], event["removed"], event["added"], event["event_id"])
        else:
            meta.loc[active, "null_contact"] = True
        event_rows.append(
            {
                **{key: value for key, value in event.items() if key not in {"removed", "added"}},
                "source_program_genes": "G101;G103",
                "receiver_program_genes": f"G{11 + 10 * event_index}-G{20 + 10 * event_index}",
                "event_center_x": float(center[0]),
                "event_center_y": float(center[1]),
                "event_radius": radius,
                "removed_edge": "->".join(event["removed"]) if event["topology_changed"] else "",
                "added_edge": "->".join(event["added"]) if event["topology_changed"] else "",
                "decision_transition": f"{event['decision_from']}->{event['decision_to']}",
                "n_exposed_receiver_cells": int(active.sum()),
            }
        )

    # A measured spatial domain drives both programs but is not an intervention.
    confounder = (meta["spatial_x"] + meta["spatial_y"] > np.quantile(meta["spatial_x"] + meta["spatial_y"], 0.72)).to_numpy()
    meta.loc[confounder, "confounded_exposure"] = True
    confounder_strength = (
        1.5 if not scenario else (2.5 if str(scenario.get("confounding", "low")) == "high" else 0.5)
    )
    add_program(values, confounder, slice(60, 70), confounder_strength + rng.normal(0, 0.1, len(meta)))
    values = np.clip(values, 0, None)

    assert_forest(t0)
    assert_forest(t1)
    event_table = pd.DataFrame(event_rows)
    spatial_contacts = pd.concat(contact_rows, ignore_index=True)
    meta["truth_state_T0"] = state_labels(meta["root_id"], meta["truth_branch_T0"], meta["truth_pseudotime"])
    meta["truth_state_T1"] = state_labels(meta["root_id"], meta["truth_branch_T1"], meta["truth_pseudotime"])

    obs = meta.set_index("cell_id")
    var = pd.DataFrame(index=counts.columns)
    adata = ad.AnnData(X=sparse.csr_matrix(values), obs=obs, var=var)
    adata.obsm["spatial"] = obs[["spatial_x", "spatial_y"]].to_numpy(float)
    adata.uns["benchmark_name"] = "NicheForest cross-trajectory event gold standard v1"
    adata.uns["truth_contract"] = "T0 is pre-event; T1 is event-intervened; Tcf disables causal events."

    outputs = {
        "synthetic.h5ad": lambda path: adata.write_h5ad(path, compression="gzip"),
        "ground_truth_edges_T0.csv": lambda path: t0.to_csv(path, index=False),
        "ground_truth_edges_T1.csv": lambda path: t1.to_csv(path, index=False),
        "ground_truth_events.csv": lambda path: event_table.to_csv(path, index=False),
        "ground_truth_roots.csv": lambda path: pd.DataFrame(
            {"root_id": roots, "root_node": [f"{root}_ROOT" for root in roots]}
        ).to_csv(path, index=False),
    }
    for filename, writer in outputs.items():
        writer(args.output_dir / filename)

    counterfactual = pd.concat(
        [t0.assign(intervention="events_off"), t1.assign(intervention="events_on")], ignore_index=True
    )
    counterfactual.to_csv(args.output_dir / "ground_truth_counterfactual_edges.csv", index=False)

    truth_dir = args.output_dir / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    representative = (
        meta.sort_values("truth_pseudotime")
        .groupby("root_id", observed=True)
        .first()
        .reset_index()[["root_id", "cell_id", "spatial_x", "spatial_y"]]
        .rename(columns={"cell_id": "representative_cell_id"})
    )
    representative["root_node"] = representative["root_id"].map(lambda root: f"{root}_ROOT")
    representative.to_csv(truth_dir / "TRUE_ROOTS.csv", index=False)
    membership = pd.MultiIndex.from_product([meta["cell_id"], roots], names=["cell_id", "root_id"]).to_frame(index=False)
    true_root = meta.set_index("cell_id")["root_id"].astype(str)
    membership["membership_probability"] = [float(true_root[cell] == root) for cell, root in membership[["cell_id", "root_id"]].itertuples(index=False)]
    membership.to_csv(truth_dir / "TRUE_CELL_ROOT_MEMBERSHIP.csv", index=False)
    pd.concat(
        [
            meta[["cell_id", "root_id", "truth_branch_T0", "truth_state_T0"]]
            .rename(columns={"truth_branch_T0": "branch_id", "truth_state_T0": "state_id"})
            .assign(stage="T0"),
            meta[["cell_id", "root_id", "truth_branch_T1", "truth_state_T1"]]
            .rename(columns={"truth_branch_T1": "branch_id", "truth_state_T1": "state_id"})
            .assign(stage="T1"),
        ],
        ignore_index=True,
    ).to_csv(truth_dir / "TRUE_CELL_BRANCH.csv", index=False)
    meta[["cell_id", "root_id", "truth_pseudotime"]].rename(columns={"truth_pseudotime": "pseudotime"}).to_csv(
        truth_dir / "TRUE_PSEUDOTIME.csv", index=False
    )
    t0.to_csv(truth_dir / "TRUE_T0_EDGES.csv", index=False)
    t1.to_csv(truth_dir / "TRUE_T1_EDGES.csv", index=False)
    event_table.assign(is_true_event=event_table["causal"]).to_csv(truth_dir / "TRUE_EVENTS.csv", index=False)
    event_table[["event_id", "source_tau_start", "source_tau_end", "receiver_tau_start", "receiver_tau_end"]].to_csv(
        truth_dir / "TRUE_EVENT_WINDOWS.csv", index=False
    )
    event_table[["event_id", "receiver_root", "affected_branch", "decision_from", "decision_to"]].to_csv(
        truth_dir / "TRUE_EVENT_AFFECTED_BRANCHES.csv", index=False
    )
    changed_events = event_table[event_table["topology_changed"]].copy()
    changed_events.assign(
        root_id=changed_events["receiver_root"],
        source=changed_events["removed_edge"].str.split("->").str[0],
        target=changed_events["removed_edge"].str.split("->").str[1],
    )[["event_id", "root_id", "source", "target"]].to_csv(truth_dir / "TRUE_REMOVED_EDGES.csv", index=False)
    changed_events.assign(
        root_id=changed_events["receiver_root"],
        source=changed_events["added_edge"].str.split("->").str[0],
        target=changed_events["added_edge"].str.split("->").str[1],
    )[["event_id", "root_id", "source", "target"]].to_csv(truth_dir / "TRUE_ADDED_EDGES.csv", index=False)
    event_table.loc[~event_table["causal"]].to_csv(truth_dir / "TRUE_NULL_CONTACTS.csv", index=False)
    meta.loc[meta["confounded_exposure"], ["cell_id", "root_id", "spatial_x", "spatial_y", "truth_pseudotime"]].to_csv(
        truth_dir / "TRUE_CONFOUNDED_CONTACTS.csv", index=False
    )
    spatial_contacts.to_csv(truth_dir / "TRUE_SPATIAL_CONTACTS.csv", index=False)
    cci = pd.read_csv(args.base_dir / "scmultisim_within_root_cci.csv")
    cci.insert(0, "cci_id", [f"CCI_WITHIN_{i:05d}" for i in range(len(cci))])
    cci["cci_scope"] = "within_root_scmultisim"
    cross_cci = event_table[["event_id", "source_root", "receiver_root", "source_program_genes", "receiver_program_genes", "causal"]].copy()
    cross_cci = cross_cci.rename(columns={"event_id": "cci_id"})
    cross_cci["cci_scope"] = "cross_root_event_layer"
    pd.concat([cci, cross_cci], ignore_index=True, sort=False).to_csv(truth_dir / "TRUE_CCI_EDGES.csv", index=False)

    changed_cells = meta["truth_branch_T0"].astype(str).ne(meta["truth_branch_T1"].astype(str))
    causal_exposed = event_table.loc[event_table["causal"], "n_exposed_receiver_cells"]
    has_causal_events = bool(event_table["causal"].any())
    integrity = pd.DataFrame(
        [
            {"check": "T0_connected_acyclic_forest", "passed": True, "value": len(t0)},
            {"check": "T1_connected_acyclic_forest", "passed": True, "value": len(t1)},
            {
                "check": "all_causal_events_have_exposed_cells",
                "passed": bool((causal_exposed > 0).all()),
                "value": int(causal_exposed.min()) if len(causal_exposed) else 0,
            },
            {
                "check": "causal_events_change_branch_choice",
                "passed": bool(changed_cells.any()) if has_causal_events else True,
                "value": int(changed_cells.sum()),
            },
            {
                "check": "null_contact_has_no_topology_change",
                "passed": bool((~event_table.loc[~event_table["causal"], "topology_changed"]).all()),
                "value": int((~event_table["causal"]).sum()),
            },
            {
                "check": "expression_is_finite_nonnegative",
                "passed": bool(np.isfinite(values).all() and (values >= 0).all()),
                "value": float(values.min()),
            },
        ]
    )
    integrity.to_csv(args.output_dir / "BENCHMARK_INTEGRITY_AUDIT.csv", index=False)
    if not integrity["passed"].all():
        raise RuntimeError(f"Benchmark integrity failed:\n{integrity.loc[~integrity['passed']]}")

    manifest = {
        "benchmark": "cross_trajectory_event_gold_standard_v1",
        "seed": args.seed,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_roots": len(roots),
        "n_events": len(events),
        "n_causal_events": int(event_table["causal"].sum()),
        "n_null_events": int((~event_table["causal"]).sum()),
        "t0_edges": len(t0),
        "t1_edges": len(t1),
        "changed_edge_pairs": int(len(set(map(tuple, t0[["source", "target"]].to_numpy())) ^ set(map(tuple, t1[["source", "target"]].to_numpy()))) / 2),
        "changed_branch_choice_cells": int(changed_cells.sum()),
        "integrity_checks_passed": int(integrity["passed"].sum()),
        "integrity_checks_total": len(integrity),
    }
    manifest_path = args.output_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    hashes = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "FILE_HASHES.csv":
            hashes.append({"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    pd.DataFrame(hashes).to_csv(args.output_dir / "FILE_HASHES.csv", index=False)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
