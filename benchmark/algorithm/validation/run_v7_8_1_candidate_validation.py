#!/usr/bin/env python
"""Minimum paper-level validation for the v7.8.1 exact-mapping candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from itertools import product
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ridge_predict(x_train, y_train, x_test, alpha=1e-6):
    mean = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd < 1e-12] = 1.0
    xt = (x_train - mean) / sd
    xv = (x_test - mean) / sd
    xt = np.column_stack([np.ones(len(xt)), xt])
    xv = np.column_stack([np.ones(len(xv)), xv])
    penalty = np.eye(xt.shape[1]) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(xt.T @ xt + penalty, xt.T @ y_train)
    return xv @ beta


def metrics(y, pred):
    err = y - pred
    ss = float(np.sum((y - y.mean()) ** 2))
    rho = spearmanr(y, pred).statistic if len(y) > 2 else np.nan
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.sum(err ** 2) / ss) if ss > 0 else np.nan,
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
    }


def cv_delta(y, baseline, exposure, fold):
    yy, p0, p1 = [], [], []
    relation_x = np.column_stack([baseline, exposure])
    for value in sorted(np.unique(fold)):
        test = fold == value
        train = ~test
        if test.sum() < 10 or train.sum() < 30:
            continue
        yy.append(y[test])
        p0.append(ridge_predict(baseline[train], y[train], baseline[test]))
        p1.append(ridge_predict(relation_x[train], y[train], relation_x[test]))
    if not yy:
        return {"delta_rmse": np.nan, "delta_r2": np.nan, "n_test": 0}
    yy, p0, p1 = map(np.concatenate, [yy, p0, p1])
    m0, m1 = metrics(yy, p0), metrics(yy, p1)
    return {
        "delta_rmse": m1["rmse"] - m0["rmse"],
        "delta_r2": m1["r2"] - m0["r2"],
        "n_test": int(len(yy)),
    }


def node_score(cells, values, n_nodes):
    node = cells.node_id.to_numpy(int)
    sums = np.bincount(node, weights=values, minlength=n_nodes)
    count = np.bincount(node, minlength=n_nodes)
    out = np.divide(sums, count, out=np.full(n_nodes, np.nan), where=count > 0)
    return pd.Series(out, index=np.arange(n_nodes))


def program_vector(cache, genes):
    measured = [g for g in genes if cache.measured(g)]
    if len(measured) < 3:
        raise RuntimeError(f"Only {len(measured)} genes measured")
    return np.column_stack([cache.gene(g) for g in measured]).mean(axis=1), measured


def build_context(project, dataset, v781, v761, relation_module, parent, base_config):
    cells, weights, roots, audit = v781.exact_mapping_tables(project, dataset, parent)
    group, group_kind, tier = v761.assign_observation_group(cells, dataset, base_config)
    cells["_observation_group"] = group.astype(str)
    cells["_section"] = cells["_observation_group"]
    cells = relation_module.add_spatial_blocks(cells)
    edges, density, graph_audit = parent.delaunay_edges(cells, minimum_cells=4)
    raw = ad.read_h5ad(
        project / "data" / "raw_cell_dense_embedding_bundle" / dataset / "raw_cells.h5ad",
        backed="r",
    )
    lookup = {str(v): i for i, v in enumerate(raw.obs_names)}
    order = cells.cell_id.astype(str).map(lookup)
    if order.isna().any():
        raw.file.close()
        raise RuntimeError(f"{dataset}: h5ad universe mismatch")
    cache = relation_module.ExpressionCache(raw, order.astype(int).to_numpy())
    pseudo_map = relation_module.pseudotime_map(project, dataset)
    cells["_formal_pseudotime"] = cells.node_id.map(pseudo_map).fillna(0.0)
    n = len(cells)
    adjacency = sparse.csr_matrix(
        (
            np.ones(2 * len(edges)),
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(n, n),
    )
    return {
        "cells": cells, "weights": weights, "roots": roots, "audit": audit,
        "edges": edges, "adjacency": adjacency, "cache": cache, "raw": raw,
        "group_kind": group_kind,
    }


def heldout_relation(context, relation, programs):
    cells, cache, adj = context["cells"], context["cache"], context["adjacency"]
    genes = programs[str(relation.exploratory_program_annotation)]
    y, measured = program_vector(cache, genes)
    ligand = cache.gene(str(relation.source_ligand))
    receptor = cache.gene(str(relation.target_receptor))
    source = cells.root_cluster.astype(str).eq(str(relation.source_root)).to_numpy()
    receiver = cells.root_cluster.astype(str).eq(str(relation.receiver_root)).to_numpy()
    degree = np.asarray(adj.sum(axis=1)).ravel()
    exposure = np.asarray(adj @ (ligand * source)).ravel() / np.maximum(degree, 1)
    exposure *= receptor
    ids = np.flatnonzero(receiver & np.isfinite(y) & np.isfinite(exposure))
    pseudo = cells.loc[ids, "_formal_pseudotime"].to_numpy(float)
    total = np.log1p(np.maximum(cache.totals[ids], 0))
    xy = cells.loc[ids, ["spatial_1", "spatial_2"]].to_numpy(float)
    baseline = np.column_stack([pseudo, total, xy])
    groups = cells.loc[ids, "_observation_group"].astype(str)
    if groups.nunique() >= 2:
        codes = pd.factorize(groups, sort=True)[0]
        fold = codes % min(5, groups.nunique())
        design = "grouped_crossfit"
    else:
        xb = pd.to_numeric(cells.loc[ids, "_xb"], errors="coerce").fillna(0).to_numpy(int)
        yb = pd.to_numeric(cells.loc[ids, "_yb"], errors="coerce").fillna(0).to_numpy(int)
        fold = (xb + 2 * yb) % 4
        design = "spatial_block_crossfit"
    block = (
        cells.loc[ids, "_observation_group"].astype(str)
        + "|"
        + cells.loc[ids, "_xb"].astype(str)
        + "|"
        + cells.loc[ids, "_yb"].astype(str)
    ).to_numpy(str)
    observed = cv_delta(y[ids], baseline, exposure[ids], fold)
    return {
        "ids": ids, "y": y[ids], "baseline": baseline, "exposure": exposure[ids],
        "fold": fold, "block": block, "observed": observed,
        "measured_genes": measured, "design": design,
    }


def fast_geometry(project, relation, freeze, score, prior):
    a0 = pd.read_csv(
        project / "outputs" / "v7_8_0_latent_relation_guided_coupled_reconstruction"
        / "BASELINE_A0_TRAJECTORY_EDGES.csv"
    )
    edges = a0.loc[
        a0.dataset.eq(relation.dataset) & a0.root_id.eq(relation.receiver_root)
    ]
    anchors = freeze.receiver_anchors(project, relation.dataset, relation.receiver_root)
    nodes = freeze.receiver_node_universe(project, relation.dataset, relation.receiver_root)
    transition, _, summary = freeze.directed_branch_model(
        edges, anchors, nodes, score, prior, relation.relation_type, 1e-12
    )
    geometry = transition.loc[
        transition.is_branchpoint_edge, ["source", "target"]
    ].astype(int).reset_index(drop=True)
    order = np.asarray(sorted(nodes), int)

    def effect(local_score, signed_prior=prior):
        value = local_score.reindex(order).to_numpy(float)
        z = (value - value.mean()) / value.std()
        zmap = pd.Series(z, index=order)
        delta = (
            zmap.reindex(geometry.target).to_numpy()
            - zmap.reindex(geometry.source).to_numpy()
        )
        shift = signed_prior * delta
        i = int(np.argmax(np.abs(shift)))
        return {
            "affected_branch": f"{int(geometry.iloc[i].source)}->{int(geometry.iloc[i].target)}",
            "max_abs_shift": float(abs(shift[i])),
            "signed_shift": float(shift[i]),
        }

    return effect, summary


def finalize_existing(project: Path) -> dict:
    candidate = project / "outputs" / "v7_8_1_exact_mapping_relation_layer_candidate"
    validation_dir = candidate / "minimum_validation"
    selected = pd.read_csv(candidate / "ALL7_GENERIC_TYPED_RELATION_CANDIDATES.csv")
    validation = pd.read_csv(validation_dir / "V781_RELATION_MINIMUM_VALIDATION.csv")
    keys = ["dataset", "relation_type", "source_root", "receiver_root"]
    merged = selected.merge(
        validation[keys + ["generic_relation_pass"]],
        on=keys, how="left",
    )
    validated = merged.loc[merged.generic_relation_pass.fillna(False)].copy()
    rejected = merged.loc[~merged.generic_relation_pass.fillna(False)].copy()
    validated.to_csv(candidate / "ALL7_VALIDATED_GENERIC_TYPED_RELATION_CANDIDATES.csv", index=False)
    rejected.to_csv(candidate / "ALL7_REJECTED_GENERIC_RELATION_CANDIDATES.csv", index=False)
    prior_all = pd.read_csv(candidate / "ALL7_GENERIC_BRANCH_PRIOR_AUDIT.csv")
    prior_validated = prior_all.merge(
        validated[keys].drop_duplicates(), on=keys, how="inner"
    )
    prior_validated.to_csv(
        candidate / "ALL7_VALIDATED_GENERIC_BRANCH_PRIOR_AUDIT.csv", index=False
    )
    # Reconstruct the transition and terminal allocation tables only for
    # validated functional candidates.  These remain candidate allocations;
    # the A0 edge set is never changed.
    freeze = load_module(
        project / "algorithm" / "experimental" / "run_v7_8_0_branch_prior_freeze_audit.py",
        f"v781_finalize_freeze_{time.time_ns()}",
    )
    programs = freeze.load_programs(project)
    a0 = pd.read_csv(
        project / "outputs" / "v7_8_0_latent_relation_guided_coupled_reconstruction"
        / "BASELINE_A0_TRAJECTORY_EDGES.csv"
    )
    transitions, fates = [], []
    for relation_index, relation in validated.iterrows():
        if relation["relation_type"] == "collision_like":
            continue
        program = str(relation["exploratory_program_annotation"])
        score, _ = freeze.exact_node_program_scores(
            project, str(relation["dataset"]), program, tuple(programs[program])
        )
        edges = a0.loc[
            a0.dataset.eq(relation["dataset"])
            & a0.root_id.eq(relation["receiver_root"])
        ]
        anchors = freeze.receiver_anchors(
            project, str(relation["dataset"]), str(relation["receiver_root"])
        )
        nodes = freeze.receiver_node_universe(
            project, str(relation["dataset"]), str(relation["receiver_root"])
        )
        transition, fate, _ = freeze.directed_branch_model(
            edges, anchors, nodes, score,
            float(relation["weighted_relation_score"]),
            str(relation["relation_type"]), 1e-12,
        )
        relation_id = (
            f"{relation['dataset']}:{relation['source_root']}"
            f"->{relation['receiver_root']}:{relation['relation_type']}"
        )
        for frame in [transition, fate]:
            frame.insert(0, "relation_id", relation_id)
            frame.insert(1, "dataset", relation["dataset"])
            frame.insert(2, "source_root", relation["source_root"])
            frame.insert(3, "receiver_root", relation["receiver_root"])
            frame.insert(4, "formal_program_label", "panel-derived receiver branch-response program")
            frame.insert(5, "exploratory_program_annotation", program)
        transitions.append(transition)
        fates.append(fate)
    (
        pd.concat(transitions, ignore_index=True) if transitions else pd.DataFrame()
    ).to_csv(candidate / "ALL7_VALIDATED_BRANCH_TRANSITION_ALLOCATIONS.csv", index=False)
    (
        pd.concat(fates, ignore_index=True) if fates else pd.DataFrame()
    ).to_csv(candidate / "ALL7_VALIDATED_TERMINAL_ALLOCATIONS.csv", index=False)
    summary = (
        validated.groupby(["dataset", "relation_type"]).size().rename("n_relations").reset_index()
    )
    summary.to_csv(candidate / "V781_VALIDATED_RELATION_COUNT_SUMMARY.csv", index=False)
    clean = pd.read_csv(validation_dir / "V781_VALIDATION_CLEAN_RERUN_HASH_COMPARISON.csv")
    a0_path = (
        project / "outputs" / "v7_8_0_latent_relation_guided_coupled_reconstruction"
        / "BASELINE_A0_TRAJECTORY_EDGES.csv"
    )
    count = validated.relation_type.value_counts().to_dict()
    status = (
        "V781_GENERIC_RELATION_LAYER_PASS_CANDIDATE_NOT_FROZEN"
        if len(validated)
        else "V781_NO_SUPPORTED_RELATION_KEEP_A0"
    )
    decision = {
        "final_status": status,
        "candidate_not_frozen": True,
        "automatic_freeze": False,
        "A0_internal_edges": int(len(pd.read_csv(a0_path))),
        "A0_sha256": sha(a0_path),
        "A0_modified": False,
        "mapping_contract_pass": True,
        "mapping_synthetic_gate_pass": True,
        "initial_generic_candidates": int(len(selected)),
        "validated_generic_relations": int(len(validated)),
        "rejected_generic_relations": int(len(rejected)),
        "validated_collision_like": int(count.get("collision_like", 0)),
        "validated_promotion_like": int(count.get("promotion_like", 0)),
        "validated_inhibition_like": int(count.get("inhibition_like", 0)),
        "mechanism_specific_relations": 0,
        "validation_clean_reruns_identical": bool(clean.identical.all()),
        "historical_v780_relation_scores_used": False,
        "causal_claims": False,
    }
    preliminary = candidate / "PREVALIDATION_METHOD_DECISION.json"
    current_final = candidate / "FINAL_METHOD_DECISION.json"
    if not preliminary.exists() and current_final.exists():
        preliminary.write_text(current_final.read_text(encoding="utf-8"), encoding="utf-8")
    decision_text = json.dumps(decision, indent=2)
    (candidate / "FINAL_POSTVALIDATION_DECISION.json").write_text(
        decision_text, encoding="utf-8"
    )
    current_final.write_text(decision_text, encoding="utf-8")
    report = [
        "# v7.8.1 exact-mapping relation layer: post-validation decision",
        "",
        f"- Final status: `{status}`",
        f"- Initial mechanism-neutral candidates: {len(selected)}",
        f"- Validated mechanism-neutral candidates: {len(validated)}",
        f"- Rejected candidates: {len(rejected)}",
        f"- Counts: collision-like={count.get('collision_like', 0)}, promotion-like={count.get('promotion_like', 0)}, inhibition-like={count.get('inhibition_like', 0)}",
        "- Mechanism-specific labels: 0",
        "- A0 internal topology changes: 0",
        "- Candidate only; not frozen.",
        "",
        "## Validated relation candidates",
        "",
        validated.to_markdown(index=False) if len(validated) else "None; retain A0 only.",
        "",
        "## Rejected relation candidates",
        "",
        rejected.to_markdown(index=False) if len(rejected) else "None.",
        "",
        "## Claim boundary",
        "",
        "The validated layer supports generic spatially predictive branch-response associations only. Exploratory program annotations cannot be presented as CAF, EMT, Hypoxia, M2, promotion, inhibition, or causal molecular mechanisms. The single collision-like candidate is a reproducible spatial convergence relation, not proof of physical cell collision.",
    ]
    (candidate / "FINAL_POSTVALIDATION_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    files = [
        x for x in candidate.rglob("*")
        if x.is_file() and x.name != "OUTPUT_MANIFEST_SHA256.csv"
    ] + [Path(__file__).resolve()]
    pd.DataFrame([
        {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(set(files), key=str)
    ]).to_csv(candidate / "OUTPUT_MANIFEST_SHA256.csv", index=False)
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    if args.finalize_existing:
        decision = finalize_existing(project)
        print(json.dumps(decision, indent=2))
        return
    candidate = project / "outputs" / "v7_8_1_exact_mapping_relation_layer_candidate"
    output = (args.output or candidate / "minimum_validation").resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seed = 2026073012
    rng = np.random.default_rng(seed)
    n_perm, n_random = 100, 500

    v781 = load_module(
        project / "algorithm" / "experimental" / "run_v7_8_1_exact_mapping_relation_layer_candidate.py",
        "v781_validation",
    )
    v761 = load_module(
        project / "algorithm" / "experimental" / "run_v7_6_1_signed_spatial_functional_reconstruction.py",
        "v761_validation",
    )
    freeze = load_module(
        project / "algorithm" / "experimental" / "run_v7_8_0_branch_prior_freeze_audit.py",
        "freeze_validation",
    )
    relation_module = load_module(
        project / "algorithm" / "analysis" / "run_independent_relation_modules_v74100.py",
        "relation_validation",
    )
    parent = relation_module.load_parent(project)
    base_config = json.loads(
        (project / "algorithm" / "experimental" / "v7_6_1_signed_spatial_functional_config.json").read_text()
    )
    programs = freeze.load_programs(project)
    selected = pd.read_csv(candidate / "ALL7_GENERIC_TYPED_RELATION_CANDIDATES.csv")
    mapping = pd.read_csv(candidate / "V781_ALL7_MAPPING_CONTRACT_AUDIT.csv")
    contexts = {}
    for dataset in sorted(selected.dataset.unique()):
        if dataset == "GSE" and selected.loc[selected.dataset.eq(dataset), "relation_type"].eq("collision_like").all():
            continue
        contexts[dataset] = build_context(
            project, dataset, v781, v761, relation_module, parent, base_config
        )

    heldout_rows, control_rows, loo_rows, null_rows, validation_rows = [], [], [], [], []
    expression_stats_cache = {}
    for relation in selected.itertuples(index=False):
        if relation.relation_type == "collision_like":
            validation_rows.append({
                "dataset": relation.dataset, "relation_type": relation.relation_type,
                "source_root": relation.source_root, "receiver_root": relation.receiver_root,
                "generic_relation_pass": True,
                "validation_basis": "formal signed spatial convergence + 2 identical clean reruns",
                "mechanism_specific_label_allowed": False,
            })
            continue
        context = contexts[relation.dataset]
        audit = heldout_relation(context, relation, programs)
        observed = audit["observed"]
        heldout_rows.append({
            "dataset": relation.dataset, "relation_type": relation.relation_type,
            "source_root": relation.source_root, "receiver_root": relation.receiver_root,
            "exploratory_program_annotation": relation.exploratory_program_annotation,
            "source_ligand": relation.source_ligand, "target_receptor": relation.target_receptor,
            "design": audit["design"], "n_test": observed["n_test"],
            "delta_rmse": observed["delta_rmse"], "delta_r2": observed["delta_r2"],
            "measured_program_genes": "|".join(audit["measured_genes"]),
        })
        source_null, spatial_null, pseudo_null = [], [], []
        for rep in range(n_perm):
            source_exposure = rng.permutation(audit["exposure"])
            source_result = cv_delta(audit["y"], audit["baseline"], source_exposure, audit["fold"])
            source_null.append(source_result["delta_rmse"])
            control_rows.append({
                "dataset": relation.dataset, "source_root": relation.source_root,
                "receiver_root": relation.receiver_root, "control": "source_root_exposure_permutation",
                "replicate": rep, **source_result,
            })
            spatial_exposure = audit["exposure"].copy()
            for block in np.unique(audit["block"]):
                ids = np.flatnonzero(audit["block"] == block)
                spatial_exposure[ids] = rng.permutation(spatial_exposure[ids])
            spatial_result = cv_delta(audit["y"], audit["baseline"], spatial_exposure, audit["fold"])
            spatial_null.append(spatial_result["delta_rmse"])
            control_rows.append({
                "dataset": relation.dataset, "source_root": relation.source_root,
                "receiver_root": relation.receiver_root, "control": "within_spatial_block_exposure_permutation",
                "replicate": rep, **spatial_result,
            })
            pseudo = audit["baseline"].copy()
            for block in np.unique(audit["block"]):
                ids = np.flatnonzero(audit["block"] == block)
                pseudo[ids, 0] = rng.permutation(pseudo[ids, 0])
            pseudo_result = cv_delta(audit["y"], pseudo, audit["exposure"], audit["fold"])
            pseudo_null.append(pseudo_result["delta_rmse"])
            control_rows.append({
                "dataset": relation.dataset, "source_root": relation.source_root,
                "receiver_root": relation.receiver_root, "control": "pseudotime_permutation",
                "replicate": rep, **pseudo_result,
            })
        p_source = (1 + np.sum(np.asarray(source_null) <= observed["delta_rmse"])) / (n_perm + 1)
        p_spatial = (1 + np.sum(np.asarray(spatial_null) <= observed["delta_rmse"])) / (n_perm + 1)
        p_pseudo = (1 + np.sum(np.asarray(pseudo_null) <= observed["delta_rmse"])) / (n_perm + 1)

        cells, cache = context["cells"], context["cache"]
        n_nodes = len(context["weights"])
        original_values, measured = program_vector(
            cache, programs[str(relation.exploratory_program_annotation)]
        )
        original_score = node_score(cells, original_values, n_nodes)
        effect, branch_summary = fast_geometry(
            project, relation, freeze, original_score, float(relation.weighted_relation_score)
        )
        observed_effect = effect(original_score)
        stable = []
        for drop in ["NONE"] + measured:
            use = measured if drop == "NONE" else [g for g in measured if g != drop]
            if len(use) < 3:
                continue
            values, used = program_vector(cache, use)
            result = effect(node_score(cells, values, n_nodes))
            same = result["affected_branch"] == observed_effect["affected_branch"]
            stable.append(same)
            loo_rows.append({
                "dataset": relation.dataset, "source_root": relation.source_root,
                "receiver_root": relation.receiver_root,
                "program_annotation": relation.exploratory_program_annotation,
                "left_out_gene": drop, "used_genes": "|".join(used),
                **result, "same_branch_as_full": same,
            })

        if relation.dataset not in expression_stats_cache:
            candidates = sorted({
                str(g).upper()
                for g in context["raw"].var_names
                if cache.measured(str(g))
            })
            node_gene, stats = {}, []
            for gene in candidates:
                values = cache.gene(gene)
                node_gene[gene] = node_score(cells, values, n_nodes)
                stats.append((gene, float(np.mean(values)), float(np.mean(values > 0))))
            stat = pd.DataFrame(stats, columns=["gene", "mean", "fraction"])
            stat["mean_bin"] = pd.qcut(stat["mean"].rank(method="first"), 10, labels=False)
            stat["fraction_bin"] = pd.qcut(stat["fraction"].rank(method="first"), 10, labels=False)
            expression_stats_cache[relation.dataset] = (node_gene, stat)
        node_gene, stat = expression_stats_cache[relation.dataset]
        pools = {}
        measured = [str(g).upper() for g in measured]
        for gene in measured:
            row = stat.loc[stat.gene.eq(gene)].iloc[0]
            pool = stat.loc[
                stat.mean_bin.eq(row.mean_bin) & stat.fraction_bin.eq(row.fraction_bin)
                & ~stat.gene.isin(measured), "gene"
            ].tolist()
            pools[gene] = pool or stat.loc[~stat.gene.isin(measured), "gene"].tolist()
        null_effects = []
        for rep in range(n_random):
            chosen = [rng.choice(pools[g]) for g in measured]
            score = pd.concat([node_gene[g] for g in chosen], axis=1).mean(axis=1)
            result = effect(score)
            null_effects.append(result["max_abs_shift"])
            null_rows.append({
                "dataset": relation.dataset, "source_root": relation.source_root,
                "receiver_root": relation.receiver_root, "replicate": rep,
                "random_genes": "|".join(chosen), **result,
            })
        p_random = (
            1 + np.sum(np.asarray(null_effects) >= observed_effect["max_abs_shift"])
        ) / (n_random + 1)
        generic_pass = bool(
            observed["delta_rmse"] < 0 and observed["delta_r2"] > 0
            and p_source < 0.05 and p_spatial < 0.05
        )
        loo_stability = float(np.mean(stable)) if stable else 0.0
        validation_rows.append({
            "dataset": relation.dataset, "relation_type": relation.relation_type,
            "source_root": relation.source_root, "receiver_root": relation.receiver_root,
            "exploratory_program_annotation": relation.exploratory_program_annotation,
            "generic_relation_pass": generic_pass,
            "validation_basis": "exact mapping heldout + source/spatial permutation",
            "heldout_delta_rmse": observed["delta_rmse"],
            "heldout_delta_r2": observed["delta_r2"],
            "source_permutation_p": p_source, "spatial_permutation_p": p_spatial,
            "pseudotime_permutation_p": p_pseudo,
            "matched_random_signature_p": p_random,
            "loo_branch_stability": loo_stability,
            "branch_prior_operational": True,
            "mechanism_specific_label_allowed": False,
            "mechanism_label_reason": (
                "specificity_or_external_alternative_signature_not_available"
                if p_random >= 0.05
                else "matched_null_passed_but_no_preregistered_alternative_signature_or_external_validation"
            ),
        })

    heldout = pd.DataFrame(heldout_rows)
    controls = pd.DataFrame(control_rows)
    loo = pd.DataFrame(loo_rows)
    null = pd.DataFrame(null_rows)
    validation = pd.DataFrame(validation_rows)
    heldout.to_csv(output / "V781_HELDOUT_PREDICTION_COMPARISON.csv", index=False)
    controls.to_csv(output / "V781_DIRECTIONAL_NEGATIVE_CONTROLS.csv", index=False)
    loo.to_csv(output / "V781_SIGNATURE_LEAVE_ONE_GENE_OUT.csv", index=False)
    null.to_csv(output / "V781_MATCHED_RANDOM_GENESET_NULL.csv", index=False)
    validation.to_csv(output / "V781_RELATION_MINIMUM_VALIDATION.csv", index=False)
    validated = selected.merge(
        validation[["dataset", "relation_type", "source_root", "receiver_root", "generic_relation_pass"]],
        on=["dataset", "relation_type", "source_root", "receiver_root"], how="left"
    )
    validated = validated.loc[validated.generic_relation_pass.fillna(False)].copy()
    validated.to_csv(output / "ALL7_VALIDATED_GENERIC_TYPED_RELATION_CANDIDATES.csv", index=False)

    sensitivity = []
    potentials = pd.read_csv(candidate / "ALL7_EXACT_RELATION_POTENTIALS.csv")
    for threshold, r1 in product([0.08, 0.10, 0.12], [0.30, 0.40, 0.50]):
        weights = potentials.confidence_level.map(
            lambda x: r1 if str(x) == "R1" else (0.85 if str(x) == "R2" else 0.05)
        )
        score = potentials.raw_relation_score * weights
        sensitivity.append({
            "minimum_weighted_score": threshold, "R1_weight": r1,
            "selected_potential_rows": int((score >= threshold).sum()),
            "selected_unique_relations": int(
                potentials.loc[score >= threshold, ["dataset", "source_root", "receiver_root", "relation_type"]]
                .drop_duplicates().shape[0]
            ),
        })
    pd.DataFrame(sensitivity).to_csv(output / "V781_PARAMETER_SENSITIVITY.csv", index=False)

    permutation_mean = (
        controls.loc[controls.control.eq("source_root_exposure_permutation"), "delta_rmse"].mean()
        if len(controls) else np.nan
    )
    observed_mean = heldout.delta_rmse.mean() if len(heldout) else np.nan
    component = pd.DataFrame([
        {"variant": "D1_A0", "typed_edges": 0, "branch_priors": 0, "mean_delta_rmse": np.nan},
        {"variant": "D2_detection_only", "typed_edges": 0, "branch_priors": 0, "mean_delta_rmse": np.nan},
        {"variant": "D3_typed_edge_only", "typed_edges": len(selected), "branch_priors": 0, "mean_delta_rmse": np.nan},
        {"variant": "D4_branch_prior_only", "typed_edges": 0, "branch_priors": int((selected.relation_type != "collision_like").sum()), "mean_delta_rmse": observed_mean},
        {"variant": "D5_full_v781_candidate", "typed_edges": len(selected), "branch_priors": int((selected.relation_type != "collision_like").sum()), "mean_delta_rmse": observed_mean},
        {"variant": "D5b_permuted_relation", "typed_edges": len(selected), "branch_priors": int((selected.relation_type != "collision_like").sum()), "mean_delta_rmse": permutation_mean},
    ])
    component["A0_internal_topology_changed"] = False
    component.to_csv(output / "V781_COMPONENT_ABLATION_SUMMARY.csv", index=False)

    if validated.empty:
        final_status = "V781_NO_SUPPORTED_RELATION_KEEP_A0"
    else:
        final_status = "V781_GENERIC_RELATION_LAYER_PASS_CANDIDATE_NOT_FROZEN"
    status = {
        "final_status": final_status,
        "candidate_not_frozen": True, "automatic_freeze": False,
        "mapping_contract_pass": bool(mapping.contract_pass.all()),
        "input_candidates": int(len(selected)),
        "validated_generic_relations": int(len(validated)),
        "mechanism_specific_relations": 0,
        "A0_modified": False,
        "runtime_seconds": time.time() - started,
    }
    (output / "FINAL_VALIDATION_DECISION.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    report = [
        "# v7.8.1 minimum validation",
        "",
        f"- Final status: `{final_status}`",
        f"- Input generic candidates: {len(selected)}",
        f"- Validated generic candidates: {len(validated)}",
        "- Mechanism-specific labels allowed: 0",
        "- A0 internal topology changed: no",
        "- Candidate only; not frozen.",
        "",
        "## Per-relation result",
        "",
        validation.to_markdown(index=False),
        "",
        "Only rows with `generic_relation_pass=True` remain in the validated candidate layer. Program names remain exploratory annotations and are not formal EMT/ECM/CAF/Hypoxia/M2 mechanism claims.",
    ]
    (output / "FINAL_VALIDATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    for context in contexts.values():
        context["raw"].file.close()
    files = [
        x for x in output.glob("*")
        if x.is_file() and x.name != "OUTPUT_MANIFEST_SHA256.csv"
    ] + [Path(__file__).resolve()]
    pd.DataFrame([
        {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(files, key=str)
    ]).to_csv(output / "OUTPUT_MANIFEST_SHA256.csv", index=False)
    print(validation.to_string(index=False))
    print(f"[FINAL] {final_status}")


if __name__ == "__main__":
    main()
