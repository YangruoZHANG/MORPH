#!/usr/bin/env python3
"""Truth-free ordered relation detection and constrained receiver-tree rewiring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = np.std(values, axis=0, ddof=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return (values - np.mean(values, axis=0)) / scale


def residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(covariates)), zscore(covariates)])
    return values - design @ (np.linalg.pinv(design) @ values)


def correlation_vector(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    xz = zscore(np.asarray(x, dtype=float).reshape(-1, 1)).ravel()
    yz = zscore(np.asarray(matrix, dtype=float))
    return np.nan_to_num((xz[:, None] * yz).mean(axis=0), nan=0.0)


def spatial_blocks(coords: np.ndarray) -> np.ndarray:
    n_bins = 3 if len(coords) >= 90 else 2
    labels = []
    for axis in range(coords.shape[1]):
        ranked = pd.Series(coords[:, axis]).rank(method="first")
        labels.append(pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop").to_numpy(int))
    return labels[0] * n_bins + labels[1]


def permute_within_blocks(values: np.ndarray, blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(values).copy()
    for block in np.unique(blocks):
        indices = np.flatnonzero(blocks == block)
        out[indices] = values[rng.permutation(indices)]
    return out


def crossfit_effect(
    contact: np.ndarray,
    expression: np.ndarray,
    blocks: np.ndarray,
    rng: np.random.Generator,
    permutations: int,
) -> tuple[float, np.ndarray, list[float]]:
    folds = np.zeros(len(contact), dtype=int)
    for block in np.unique(blocks):
        indices = np.flatnonzero(blocks == block)
        shuffled = rng.permutation(indices)
        folds[shuffled] = np.arange(len(indices)) % 2
    variable = np.std(expression, axis=0) > 1e-8
    eligible = np.flatnonzero(variable)
    top_k = min(10, len(eligible))

    def evaluate(candidate_contact: np.ndarray) -> float:
        fold_effects = []
        for test_fold in (0, 1):
            train = folds != test_fold
            test = folds == test_fold
            if train.sum() < 6 or test.sum() < 6 or top_k == 0:
                continue
            train_corr = correlation_vector(candidate_contact[train], expression[train])
            selected = eligible[np.argsort(-np.abs(train_corr[eligible]))[:top_k]]
            test_corr = correlation_vector(candidate_contact[test], expression[test][:, selected])
            fold_effects.append(float(np.mean(np.abs(test_corr))))
        return float(np.mean(fold_effects)) if fold_effects else 0.0

    effect = evaluate(contact)
    full_corr = correlation_vector(contact, expression)
    top = eligible[np.argsort(-np.abs(full_corr[eligible]))[:top_k]] if top_k else np.asarray([], dtype=int)
    null_effects = [
        evaluate(permute_within_blocks(contact, blocks, rng))
        for _ in range(permutations)
    ]
    return effect, top, null_effects


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def branch_assignment(part: pd.DataFrame, anchor: str) -> dict[str, str]:
    graph = nx.Graph()
    graph.add_edges_from(part[["source_name", "target_name"]].astype(str).itertuples(index=False, name=None))
    assignment = {str(anchor): "ROOT"}
    if str(anchor) not in graph:
        return assignment
    for node in graph.nodes:
        if str(node) == str(anchor):
            continue
        path = nx.shortest_path(graph, str(anchor), str(node))
        assignment[str(node)] = f"BR_{path[1]}"
    return assignment


def relation_scan(
    adata,
    membership: pd.DataFrame,
    t0: pd.DataFrame,
    pseudotime: pd.DataFrame,
    fixed: pd.DataFrame,
    permutations: int,
    seed: int,
    cell_adata=None,
    cell_mapping: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[tuple[int, int], dict[str, float]]]:
    rng = np.random.default_rng(seed)
    pivot = membership.pivot(index="node_id", columns="root_pos", values="A_sharp").fillna(0.0)
    primary = pivot.idxmax(axis=1).astype(int)
    node_id_to_name = membership.drop_duplicates("node_id").set_index("node_id")["node_name"].astype(str)
    pt_lookup = {
        int(root): pd.Series(
            group["selected_pseudotime"].astype(float).to_numpy(),
            index=group["node_id"].astype(int).map(node_id_to_name).astype(str),
        )
        for root, group in pseudotime.groupby("root_pos")
    }
    primary_by_name = dict(zip(
        membership.drop_duplicates("node_id")["node_name"].astype(str),
        membership.drop_duplicates("node_id")["node_id"].astype(int).map(primary),
    ))
    if cell_adata is not None and cell_mapping is not None:
        analysis_adata = cell_adata
        mapping = cell_mapping.set_index("cell_name")["metacell_id"].astype(str)
        metacell_names = mapping.reindex(analysis_adata.obs_names.astype(str)).to_numpy()
        if pd.isna(metacell_names).any():
            raise ValueError("Cell-level relation scan has unmapped cells")
        primary_array = np.asarray([primary_by_name.get(str(name), -1) for name in metacell_names], dtype=int)
    else:
        analysis_adata = adata
        metacell_names = np.asarray(adata.obs_names.astype(str))
        primary_array = np.asarray([primary_by_name.get(str(name), -1) for name in metacell_names], dtype=int)
    coords = np.asarray(analysis_adata.obsm["spatial"], dtype=float)
    expression = np.log1p(
        analysis_adata.X.toarray() if hasattr(analysis_adata.X, "toarray") else np.asarray(analysis_adata.X)
    )
    anchors = dict(enumerate(fixed["root_anchor_id"].astype(str)))
    branch_maps = {
        root: branch_assignment(t0[t0["root_pos"].astype(int).eq(root)], anchor)
        for root, anchor in anchors.items()
    }
    node_names = np.asarray(analysis_adata.obs_names.astype(str))
    diagnostics: list[dict[str, object]] = []
    node_scores: dict[tuple[int, int], dict[str, float]] = {}

    for source_root in sorted(pivot.columns.astype(int)):
        source_nodes = np.flatnonzero(primary_array == source_root)
        for receiver_root in sorted(pivot.columns.astype(int)):
            if source_root == receiver_root:
                continue
            receiver_nodes = np.flatnonzero(primary_array == receiver_root)
            if len(source_nodes) < 6 or len(receiver_nodes) < 6:
                diagnostics.append({
                    "source_root_pos": source_root, "receiver_root_pos": receiver_root,
                    "n_source_nodes": len(source_nodes), "n_receiver_nodes": len(receiver_nodes),
                    "effect_size": 0.0, "permutation_p": 1.0, "affected_branch": "none",
                    "branch_score": 0.0, "event_type": "none", "status": "insufficient_nodes",
                })
                continue
            distance = np.sqrt(((coords[receiver_nodes, None, :] - coords[source_nodes][None, :, :]) ** 2).sum(axis=2))
            nearest = distance.min(axis=1)
            radius = max(float(np.quantile(nearest, 0.50)), 1e-6)
            contact = np.exp(-np.square(nearest / radius))
            receiver_pt = pt_lookup[receiver_root].reindex(metacell_names[receiver_nodes]).fillna(0.5).to_numpy(float)
            competing_contacts = []
            for other_source in sorted(pivot.columns.astype(int)):
                if other_source in {source_root, receiver_root}:
                    continue
                other_nodes = np.flatnonzero(primary_array == other_source)
                if len(other_nodes) < 3:
                    continue
                other_distance = np.sqrt(
                    ((coords[receiver_nodes, None, :] - coords[other_nodes][None, :, :]) ** 2).sum(axis=2)
                ).min(axis=1)
                other_radius = max(float(np.quantile(other_distance, 0.50)), 1e-6)
                competing_contacts.append(np.exp(-np.square(other_distance / other_radius)))
            covariate_columns = [receiver_pt, coords[receiver_nodes, 0], coords[receiver_nodes, 1], *competing_contacts]
            covariates = np.column_stack(covariate_columns)
            expression_residual = residualize(expression[receiver_nodes], covariates)
            contact_residual = residualize(contact[:, None], covariates).ravel()
            blocks = spatial_blocks(coords[receiver_nodes])
            effect, top, null_effects = crossfit_effect(
                contact_residual, expression_residual, blocks, rng, permutations
            )
            pvalue = float((1 + np.sum(np.asarray(null_effects) >= effect)) / (permutations + 1))
            response = (
                np.mean(zscore(expression_residual[:, top]), axis=1)
                if len(top) else np.zeros(len(receiver_nodes))
            )
            response = zscore(response.reshape(-1, 1)).ravel() if np.std(response) > 1e-9 else response
            local_scores = np.abs(response) * zscore(contact.reshape(-1, 1)).ravel()
            score_frame = pd.DataFrame({"metacell": metacell_names[receiver_nodes], "score": local_scores})
            node_scores[(source_root, receiver_root)] = score_frame.groupby("metacell")["score"].mean().to_dict()
            branch_values: list[tuple[str, float, int]] = []
            receiver_branch = branch_maps[receiver_root]
            for branch in sorted(set(receiver_branch.values()) - {"ROOT"}):
                positions = [
                    i for i, node in enumerate(receiver_nodes)
                    if receiver_branch.get(str(metacell_names[node])) == branch
                ]
                if len(positions) < 3:
                    continue
                branch_contact = contact_residual[positions]
                branch_response = response[positions]
                score = abs(float(np.corrcoef(branch_contact, branch_response)[0, 1])) if np.std(branch_contact) > 0 and np.std(branch_response) > 0 else 0.0
                branch_values.append((branch, score, len(positions)))
            affected_branch, branch_score, branch_n = max(branch_values, key=lambda item: item[1]) if branch_values else ("none", 0.0, 0)
            signed_relation = float(np.corrcoef(contact_residual, response)[0, 1]) if np.std(response) > 0 else 0.0
            event_type = "promotion" if signed_relation >= 0 else "inhibition"
            diagnostics.append({
                "source_root_pos": source_root,
                "receiver_root_pos": receiver_root,
                "n_source_nodes": len(source_nodes),
                "n_receiver_nodes": len(receiver_nodes),
                "contact_radius": radius,
                "contact_variance": float(np.var(contact)),
                "effect_size": effect,
                "permutation_p": pvalue,
                "affected_branch": affected_branch,
                "affected_branch_nodes": branch_n,
                "branch_score": branch_score,
                "signed_relation": signed_relation,
                "event_type": event_type,
                "top_genes": ";".join(map(str, top.tolist())),
                "status": "tested",
            })
    frame = pd.DataFrame(diagnostics)
    tested = frame["status"].eq("tested")
    frame["q_value"] = 1.0
    if tested.any():
        frame.loc[tested, "q_value"] = bh_qvalues(frame.loc[tested, "permutation_p"].to_numpy(float))
    return frame, node_scores


def choose_called_events(
    frame: pd.DataFrame,
    q_cutoff: float,
    effect_cutoff: float,
    direction_ratio: float,
    receiver_ratio: float,
    max_sources_per_receiver: int,
) -> pd.DataFrame:
    out = frame.copy()
    out["is_called"] = (
        out["status"].eq("tested")
        & out["q_value"].le(q_cutoff)
        & out["effect_size"].ge(effect_cutoff)
        & out["affected_branch"].ne("none")
    )
    out["decision_reason"] = np.where(out["is_called"], "passes_independent_gate", "fails_independent_gate")
    for _, indices in out.groupby(out.apply(lambda row: tuple(sorted((int(row.source_root_pos), int(row.receiver_root_pos)))), axis=1)).groups.items():
        called = out.loc[list(indices)]
        called = called[called["is_called"]]
        if len(called) <= 1:
            continue
        ordered = called.sort_values("effect_size", ascending=False)
        best, second = ordered.index[0], ordered.index[1]
        ratio = float(out.loc[best, "effect_size"] / max(out.loc[second, "effect_size"], 1e-9))
        if ratio < direction_ratio:
            out.loc[ordered.index, "is_called"] = False
            out.loc[ordered.index, "decision_reason"] = "bidirectional_ambiguity_abstain"
        else:
            out.loc[ordered.index[1:], "is_called"] = False
            out.loc[ordered.index[1:], "decision_reason"] = "reverse_direction_weaker"

    # A receiver-specific sparse competition prevents spatially adjacent roots
    # from becoming interchangeable causes of the same receiver response.
    for _, indices in out.groupby("receiver_root_pos").groups.items():
        called = out.loc[list(indices)]
        called = called[called["is_called"]].sort_values("effect_size", ascending=False)
        if len(called) <= max_sources_per_receiver:
            continue
        boundary = float(called.iloc[max_sources_per_receiver - 1]["effect_size"])
        challenger = float(called.iloc[max_sources_per_receiver]["effect_size"])
        if boundary / max(challenger, 1e-9) < receiver_ratio:
            out.loc[called.index, "is_called"] = False
            out.loc[called.index, "decision_reason"] = "receiver_source_ambiguity_abstain"
        else:
            rejected = called.index[max_sources_per_receiver:]
            out.loc[rejected, "is_called"] = False
            out.loc[rejected, "decision_reason"] = "receiver_source_competition_weaker"
    return out


def refine_trees(
    t0: pd.DataFrame,
    candidates: pd.DataFrame,
    called: pd.DataFrame,
    node_scores: dict[tuple[int, int], dict[str, float]],
    fixed: pd.DataFrame,
    pseudotime: pd.DataFrame,
    clip_fraction: float,
    evidence_scaling: str = "legacy_divide_by_3",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    refined = t0.copy()
    changes: list[dict[str, object]] = []
    anchor_by_pos = dict(enumerate(fixed["root_anchor_id"].astype(str)))
    pt_by_pos = {
        int(root): dict(zip(group["node_id"].astype(int), group["selected_pseudotime"].astype(float)))
        for root, group in pseudotime.groupby("root_pos")
    }
    candidate_cost = {}
    candidate_names = {}
    for row in candidates.itertuples(index=False):
        key = tuple(sorted((int(row.source), int(row.target))))
        candidate_cost[key] = float(row.d_global)
        candidate_names[key] = (str(row.source_name), str(row.target_name))

    for event_index, event in called[called["is_called"]].iterrows():
        receiver = int(event.receiver_root_pos)
        source = int(event.source_root_pos)
        local_mask = refined["root_pos"].astype(int).eq(receiver)
        part = refined.loc[local_mask].copy()
        graph = nx.Graph()
        graph.add_edges_from(part[["source", "target"]].astype(int).itertuples(index=False, name=None))
        anchor_name = anchor_by_pos[receiver]
        name_to_id = dict(zip(part["source_name"].astype(str), part["source"].astype(int)))
        name_to_id.update(dict(zip(part["target_name"].astype(str), part["target"].astype(int))))
        if anchor_name not in name_to_id:
            continue
        anchor = name_to_id[anchor_name]
        branches = branch_assignment(part, anchor_name)
        branch_nodes = [name for name, branch in branches.items() if branch == str(event.affected_branch)]
        scores = node_scores.get((source, receiver), {})
        child_candidates = sorted(branch_nodes, key=lambda name: abs(scores.get(name, 0.0)), reverse=True)
        branch_absolute_scores = pd.Series(
            {name: abs(float(scores.get(name, 0.0))) for name in branch_nodes}, dtype=float
        )
        branch_percentiles = branch_absolute_scores.rank(method="average", pct=True).to_dict()
        accepted = False
        for child_name in child_candidates:
            child = name_to_id.get(child_name)
            if child is None or child == anchor or child not in graph:
                continue
            path = nx.shortest_path(graph, anchor, child)
            if len(path) < 2:
                continue
            current_parent = path[-2]
            current_key = tuple(sorted((current_parent, child)))
            current_cost = candidate_cost.get(current_key, 1.0)
            graph_without = graph.copy()
            graph_without.remove_edge(current_parent, child)
            subtree = nx.node_connected_component(graph_without, child)
            alternatives = []
            for key, base_cost in candidate_cost.items():
                if child not in key:
                    continue
                parent = key[0] if key[1] == child else key[1]
                if parent in subtree or parent == current_parent or parent not in graph:
                    continue
                if pt_by_pos[receiver].get(parent, 1.0) >= pt_by_pos[receiver].get(child, 1.0):
                    continue
                if evidence_scaling == "branch_percentile":
                    evidence = float(branch_percentiles.get(child_name, 0.0))
                elif evidence_scaling == "legacy_divide_by_3":
                    evidence = float(np.clip(abs(scores.get(child_name, 0.0)) / 3.0, 0.0, 1.0))
                else:
                    raise ValueError(f"Unknown evidence scaling: {evidence_scaling}")
                modifier = min(clip_fraction, clip_fraction * float(event.effect_size) * evidence)
                adjusted = base_cost * (1.0 - modifier)
                alternatives.append((adjusted, base_cost, parent, modifier))
            if not alternatives:
                continue
            adjusted, new_cost, new_parent, modifier = min(alternatives)
            if adjusted >= current_cost:
                continue
            graph_without.add_edge(new_parent, child)
            if not nx.is_tree(graph_without):
                continue
            remove_mask = local_mask & (
                ((refined["source"].astype(int) == current_parent) & (refined["target"].astype(int) == child))
                | ((refined["source"].astype(int) == child) & (refined["target"].astype(int) == current_parent))
            )
            template = candidates[
                ((candidates["source"].astype(int) == new_parent) & (candidates["target"].astype(int) == child))
                | ((candidates["source"].astype(int) == child) & (candidates["target"].astype(int) == new_parent))
            ].iloc[0]
            new_row = refined.loc[local_mask].iloc[0].copy()
            for column in refined.columns:
                if column in template.index:
                    new_row[column] = template[column]
            new_row["root_id"] = part.iloc[0]["root_id"]
            new_row["root_pos"] = receiver
            new_row["root_index"] = part.iloc[0]["root_index"]
            new_row["source"] = new_parent
            new_row["target"] = child
            new_row["source_name"] = str(template.source_name if int(template.source) == new_parent else template.target_name)
            new_row["target_name"] = child_name
            refined = pd.concat([refined.loc[~remove_mask], pd.DataFrame([new_row])], ignore_index=True)
            changes.append({
                "event_index": int(event_index), "source_root_pos": source, "receiver_root_pos": receiver,
                "affected_branch": event.affected_branch, "event_node": child_name,
                "removed_source": current_parent, "removed_target": child,
                "added_source": new_parent, "added_target": child,
                "base_cost_removed": current_cost, "base_cost_added": new_cost,
                "adjusted_cost_added": adjusted, "modifier_fraction": modifier,
                "evidence_scaling": evidence_scaling, "event_evidence": evidence,
                "accepted": True,
            })
            accepted = True
            break
        if not accepted:
            changes.append({
                "event_index": int(event_index), "source_root_pos": source, "receiver_root_pos": receiver,
                "affected_branch": event.affected_branch, "accepted": False,
            })
    return refined, pd.DataFrame(changes)


def export_prediction(
    output: Path,
    adata,
    fixed: pd.DataFrame,
    membership: pd.DataFrame,
    t0: pd.DataFrame,
    t1: pd.DataFrame,
    called: pd.DataFrame,
    candidates: pd.DataFrame,
    pseudotime: pd.DataFrame,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    root_map = dict(zip(fixed["root_anchor_id"].astype(str), fixed["truth_root_id"].astype(str)))
    pos_map = dict(enumerate(fixed["truth_root_id"].astype(str)))
    pd.DataFrame({"root_id": list(pos_map.values())}).to_csv(output / "PRED_ROOTS.csv", index=False)
    assign = membership.copy()
    assign["root_id"] = assign["root_pos"].astype(int).map(pos_map)
    assign.rename(columns={"node_name": "unit_id", "A_sharp": "membership_probability"})[
        ["unit_id", "root_id", "membership_probability"]
    ].to_csv(output / "PRED_CELL_OR_METACELL_ASSIGNMENT.csv", index=False)

    def edges_for_contract(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[["root_pos", "source_name", "target_name"]].copy()
        out["root_id"] = out["root_pos"].astype(int).map(pos_map)
        return out.rename(columns={"source_name": "source", "target_name": "target"})[
            ["root_id", "source", "target"]
        ].drop_duplicates()

    edges_for_contract(t0).to_csv(output / "PRED_T0_EDGES.csv", index=False)
    edges_for_contract(t1).to_csv(output / "PRED_T1_EDGES.csv", index=False)
    branch_records = []
    pt_records = []
    for root_pos, part in t1.groupby(t1["root_pos"].astype(int)):
        anchor = str(fixed.iloc[int(root_pos)]["root_anchor_id"])
        branches = branch_assignment(part, anchor)
        for unit, branch in branches.items():
            branch_records.append({"unit_id": unit, "root_id": pos_map[int(root_pos)], "branch_id": branch})
    for row in pseudotime.itertuples(index=False):
        node_id = int(row.node_id)
        unit_id = str(getattr(row, "node_name", adata.obs_names[node_id] if 0 <= node_id < adata.n_obs else ""))
        value = float(
            row.constrained_pseudotime
            if hasattr(row, "constrained_pseudotime")
            else row.selected_pseudotime
        )
        if unit_id:
            pt_records.append({
                "unit_id": unit_id,
                "root_id": pos_map[int(row.root_pos)],
                "pseudotime": value,
            })
    pd.DataFrame(branch_records).to_csv(output / "PRED_BRANCH.csv", index=False)
    pd.DataFrame(pt_records).to_csv(output / "PRED_PSEUDOTIME.csv", index=False)
    event_records = []
    for index, event in called.iterrows():
        event_records.append({
            "event_id": f"REL_{int(event.source_root_pos)}_{int(event.receiver_root_pos)}",
            "event_type": event.event_type,
            "source_root": pos_map[int(event.source_root_pos)],
            "receiver_root": pos_map[int(event.receiver_root_pos)],
            "affected_branch": event.affected_branch,
            "score": float(event.effect_size),
            "is_called": bool(event.is_called),
        })
    pd.DataFrame(event_records, columns=[
        "event_id", "event_type", "source_root", "receiver_root", "affected_branch", "score", "is_called",
    ]).to_csv(output / "PRED_EVENTS.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", required=True, type=Path)
    parser.add_argument("--fixed-root-input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cell-h5ad", type=Path)
    parser.add_argument("--cell-to-metacell", type=Path)
    parser.add_argument("--seed", type=int, default=7112101)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--q-cutoff", type=float, default=0.10)
    parser.add_argument("--effect-cutoff", type=float, default=0.30)
    parser.add_argument("--direction-ratio", type=float, default=1.05)
    parser.add_argument("--receiver-ratio", type=float, default=1.05)
    parser.add_argument("--max-sources-per-receiver", type=int, default=1)
    parser.add_argument("--clip-fraction", type=float, default=0.35)
    parser.add_argument(
        "--evidence-scaling", choices=["legacy_divide_by_3", "branch_percentile"],
        default="legacy_divide_by_3",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.upstream_dir / "metacell_adata.h5ad")
    fixed = pd.read_csv(args.fixed_root_input)
    membership = pd.read_csv(args.upstream_dir / "membership_A_sharp.csv")
    t0_path = args.upstream_dir / "progression_tree_edges.csv"
    if not t0_path.exists():
        t0_path = args.upstream_dir / "initial_root_specific_tree_edges.csv"
    t0 = pd.read_csv(t0_path)
    pseudotime = pd.read_csv(args.upstream_dir / "pseudotime_table.csv")
    pseudotime = pseudotime[pseudotime["tree_label"].astype(str).eq("progression")]
    export_pseudotime = pseudotime.copy()
    candidates = pd.read_csv(args.upstream_dir / "trajectory_candidate_edges.csv")

    cell_adata = ad.read_h5ad(args.cell_h5ad) if args.cell_h5ad else None
    cell_mapping = pd.read_csv(args.cell_to_metacell) if args.cell_to_metacell else None
    if (cell_adata is None) != (cell_mapping is None):
        raise ValueError("--cell-h5ad and --cell-to-metacell must be supplied together")
    diagnostics, node_scores = relation_scan(
        adata, membership, t0, pseudotime, fixed, args.permutations, args.seed,
        cell_adata=cell_adata, cell_mapping=cell_mapping,
    )
    called = choose_called_events(
        diagnostics,
        args.q_cutoff,
        args.effect_cutoff,
        args.direction_ratio,
        args.receiver_ratio,
        args.max_sources_per_receiver,
    )
    refined, changes = refine_trees(
        t0, candidates, called, node_scores, fixed, pseudotime, args.clip_fraction,
        evidence_scaling=args.evidence_scaling,
    )
    diagnostics = called
    diagnostics.to_csv(args.output_dir / "RELATION_DIAGNOSTICS.csv", index=False)
    diagnostics[diagnostics["is_called"]].to_csv(args.output_dir / "CALLED_EVENTS.csv", index=False)
    refined.to_csv(args.output_dir / "REFINED_T1_EDGES.csv", index=False)
    changes.to_csv(args.output_dir / "EDGE_CHANGES.csv", index=False)
    export_prediction(
        args.output_dir / "prediction", adata, fixed, membership, t0, refined,
        diagnostics, candidates, export_pseudotime,
    )
    status = {
        "n_tested_ordered_pairs": int(diagnostics["status"].eq("tested").sum()),
        "n_called_events": int(diagnostics["is_called"].sum()),
        "n_accepted_rewires": int(changes.get("accepted", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "all_refined_roots_are_trees": bool(all(
            nx.is_tree(nx.from_pandas_edgelist(part, "source", "target"))
            for _, part in refined.groupby("root_pos") if len(part)
        )),
        "truth_used": False,
        "inference_unit": "cell" if cell_adata is not None else "metacell",
        "topology_unit": "metacell",
        "crossfit_folds": 2,
        "spatial_block_permutation": True,
        "receiver_source_competition": True,
        "max_sources_per_receiver": args.max_sources_per_receiver,
        "q_cutoff": args.q_cutoff,
        "effect_cutoff": args.effect_cutoff,
        "clip_fraction": args.clip_fraction,
        "evidence_scaling": args.evidence_scaling,
    }
    (args.output_dir / "RUN_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
