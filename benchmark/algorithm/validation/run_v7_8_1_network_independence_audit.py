#!/usr/bin/env python
"""Network-level conditional-independence audit of validated v7.8.1 relations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


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


def design_covariates(context, ids, roots):
    cells, cache, adj = context["cells"], context["cache"], context["adjacency"]
    degree = np.asarray(adj.sum(axis=1)).ravel()
    columns = [
        cells.loc[ids, "_formal_pseudotime"].to_numpy(float),
        np.log1p(np.maximum(cache.totals[ids], 0)),
        cells.loc[ids, "spatial_1"].to_numpy(float),
        cells.loc[ids, "spatial_2"].to_numpy(float),
        np.log1p(degree[ids]),
    ]
    for root in roots:
        mask = cells.root_cluster.astype(str).eq(root).to_numpy(float)
        local = np.asarray(adj @ mask).ravel() / np.maximum(degree, 1)
        columns.append(local[ids])
    categorical = pd.DataFrame({
        "group": cells.loc[ids, "_observation_group"].astype(str).to_numpy(),
        "block": (
            cells.loc[ids, "_observation_group"].astype(str)
            + "|" + cells.loc[ids, "_xb"].astype(str)
            + "|" + cells.loc[ids, "_yb"].astype(str)
        ).to_numpy(),
        "compartment": cells.loc[ids, "_compartment"].astype(str).to_numpy(),
    })
    onehot = pd.get_dummies(categorical, drop_first=True, dtype=float)
    if len(onehot.columns):
        columns.extend([onehot[c].to_numpy(float) for c in onehot])
    return np.column_stack(columns)


def fold_and_block(context, ids):
    cells = context["cells"]
    groups = cells.loc[ids, "_observation_group"].astype(str)
    if groups.nunique() >= 2:
        codes = pd.factorize(groups, sort=True)[0]
        fold = codes % min(5, groups.nunique())
    else:
        xb = pd.to_numeric(cells.loc[ids, "_xb"], errors="coerce").fillna(0).to_numpy(int)
        yb = pd.to_numeric(cells.loc[ids, "_yb"], errors="coerce").fillna(0).to_numpy(int)
        fold = (xb + 2 * yb) % 4
    block = (
        groups + "|" + cells.loc[ids, "_xb"].astype(str)
        + "|" + cells.loc[ids, "_yb"].astype(str)
    ).to_numpy(str)
    return fold, block


def exposure_and_response(context, relation, programs, validator):
    cells, cache, adj = context["cells"], context["cache"], context["adjacency"]
    source = cells.root_cluster.astype(str).eq(str(relation.source_root)).to_numpy()
    receiver = cells.root_cluster.astype(str).eq(str(relation.receiver_root)).to_numpy()
    ligand = cache.gene(str(relation.source_ligand))
    receptor = cache.gene(str(relation.target_receptor))
    degree = np.asarray(adj.sum(axis=1)).ravel()
    exposure = np.asarray(adj @ (ligand * source)).ravel() / np.maximum(degree, 1)
    exposure *= receptor
    y, measured = validator.program_vector(
        cache, programs[str(relation.exploratory_program_annotation)]
    )
    ids = np.flatnonzero(receiver & np.isfinite(exposure) & np.isfinite(y))
    return ids, exposure[ids], y[ids], measured


def cv_predictions(validator, y, reduced, full, fold):
    yy, pr, pf = [], [], []
    for value in sorted(np.unique(fold)):
        test = fold == value
        train = ~test
        if test.sum() < 10 or train.sum() < 30:
            continue
        yy.append(y[test])
        pr.append(validator.ridge_predict(reduced[train], y[train], reduced[test]))
        pf.append(validator.ridge_predict(full[train], y[train], full[test]))
    if not yy:
        return {"rmse_gain": np.nan, "r2_gain": np.nan, "n_test": 0}
    yy, pr, pf = map(np.concatenate, [yy, pr, pf])
    mr = validator.metrics(yy, pr)
    mf = validator.metrics(yy, pf)
    return {
        "rmse_gain": mr["rmse"] - mf["rmse"],
        "r2_gain": mf["r2"] - mr["r2"],
        "n_test": int(len(yy)),
    }


def conditional_test(validator, y, base, focal, other, fold, block, rng, permutations):
    reduced = np.column_stack([base, other]) if other.size else base
    full = np.column_stack([reduced, focal])
    observed = cv_predictions(validator, y, reduced, full, fold)
    null = []
    for _ in range(permutations):
        perm = focal.copy()
        for value in np.unique(block):
            ids = np.flatnonzero(block == value)
            perm[ids] = rng.permutation(perm[ids])
        result = cv_predictions(
            validator, y, reduced, np.column_stack([reduced, perm]), fold
        )
        null.append(result["rmse_gain"])
    p = (1 + np.sum(np.asarray(null) >= observed["rmse_gain"])) / (permutations + 1)
    return {
        **observed,
        "empirical_p": float(p),
        "null_mean_rmse_gain": float(np.nanmean(null)),
        "null_q95_rmse_gain": float(np.nanquantile(null, 0.95)),
    }


def align_exposure(context, relation, target_ids):
    cells, cache, adj = context["cells"], context["cache"], context["adjacency"]
    source = cells.root_cluster.astype(str).eq(str(relation.source_root)).to_numpy()
    ligand = cache.gene(str(relation.source_ligand))
    receptor = cache.gene(str(relation.target_receptor))
    degree = np.asarray(adj.sum(axis=1)).ravel()
    value = np.asarray(adj @ (ligand * source)).ravel() / np.maximum(degree, 1)
    return (value * receptor)[target_ids]


def multiplicity_audit(candidate, v761):
    functional = pd.read_csv(candidate / "ALL7_EXACT_SIGNED_FUNCTIONAL_RELATIONS.csv")
    spatial = pd.read_csv(candidate / "ALL7_EXACT_SIGNED_SPATIAL_RELATIONS.csv")
    recomputed_functional = v761.bh(functional.best_lr_within_family_bonferroni_p)
    recomputed_sign = v761.bh(functional.sign_concordance_pvalue)
    recomputed_spatial = v761.bh(spatial.p_two_sided)
    return pd.DataFrame([
        {
            "family": "functional_relation_family",
            "n_tests": len(functional),
            "scope": "all eligible root pairs x directions x receiver programs across all7",
            "maximum_q_absolute_difference": float(
                np.nanmax(np.abs(recomputed_functional - functional.relation_family_q_all7))
            ),
            "selection_aware_complete_family": True,
            "audit_pass": bool(np.allclose(
                recomputed_functional, functional.relation_family_q_all7, equal_nan=True
            )),
        },
        {
            "family": "functional_sign_concordance",
            "n_tests": len(functional),
            "scope": "all eligible functional relation families across all7",
            "maximum_q_absolute_difference": float(
                np.nanmax(np.abs(recomputed_sign - functional.sign_concordance_q_all7))
            ),
            "selection_aware_complete_family": True,
            "audit_pass": bool(np.allclose(
                recomputed_sign, functional.sign_concordance_q_all7, equal_nan=True
            )),
        },
        {
            "family": "signed_spatial_root_pairs",
            "n_tests": len(spatial),
            "scope": "all eligible unordered root pairs across all7",
            "maximum_q_absolute_difference": float(
                np.nanmax(np.abs(recomputed_spatial - spatial.q_two_sided_all_pairs))
            ),
            "selection_aware_complete_family": True,
            "audit_pass": bool(np.allclose(
                recomputed_spatial, spatial.q_two_sided_all_pairs, equal_nan=True
            )),
        },
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    candidate = project / "outputs" / "v7_8_1_exact_mapping_relation_layer_candidate"
    output = (args.output or candidate / "network_independence_audit").resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = project / "algorithm" / "validation" / "v7_8_1_network_independence_audit_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    (output / "NETWORK_AUDIT_CONFIG_LOCK.json").write_text(
        json.dumps({"locked_before_results": True, "config": config}, indent=2),
        encoding="utf-8",
    )
    rng = np.random.default_rng(config["random_seed"])
    validator = load_module(
        project / "algorithm" / "validation" / "run_v7_8_1_candidate_validation.py",
        "v781_validator_network",
    )
    v781 = load_module(
        project / "algorithm" / "experimental" / "run_v7_8_1_exact_mapping_relation_layer_candidate.py",
        "v781_network",
    )
    v761 = load_module(
        project / "algorithm" / "experimental" / "run_v7_6_1_signed_spatial_functional_reconstruction.py",
        "v761_network",
    )
    relation_module = load_module(
        project / "algorithm" / "analysis" / "run_independent_relation_modules_v74100.py",
        "relation_network",
    )
    parent = relation_module.load_parent(project)
    base_config = json.loads(
        (project / "algorithm" / "experimental" / "v7_6_1_signed_spatial_functional_config.json").read_text()
    )
    programs = load_module(
        project / "algorithm" / "experimental" / "run_v7_8_0_branch_prior_freeze_audit.py",
        "freeze_network",
    ).load_programs(project)
    relations = pd.read_csv(candidate / "ALL7_VALIDATED_GENERIC_TYPED_RELATION_CANDIDATES.csv")
    contexts = {}
    for dataset in ["xenium_melanoma", "xenium_breast", "GSE245582"]:
        contexts[dataset] = validator.build_context(
            project, dataset, v781, v761, relation_module, parent, base_config
        )

    rows, edge_status = [], []
    permutations = int(config["permutations"])

    # Melanoma reciprocal-edge audit.
    melanoma = relations.loc[relations.dataset.eq("xenium_melanoma")]
    melanoma_results = {}
    for relation in melanoma.itertuples(index=False):
        context = contexts["xenium_melanoma"]
        ids, focal, y, _ = exposure_and_response(context, relation, programs, validator)
        other_relation = melanoma.loc[
            ~(
                melanoma.source_root.eq(relation.source_root)
                & melanoma.receiver_root.eq(relation.receiver_root)
            )
        ].iloc[0]
        other = align_exposure(context, other_relation, ids)
        base = design_covariates(context, ids, ["C03", "C15"])
        fold, block = fold_and_block(context, ids)
        total = conditional_test(
            validator, y, base, focal, np.empty((len(ids), 0)), fold, block,
            rng, permutations,
        )
        conditional = conditional_test(
            validator, y, base, focal, other[:, None], fold, block,
            rng, permutations,
        )
        retained = conditional["rmse_gain"] > 0 and conditional["empirical_p"] < 0.05
        explained = (
            1 - conditional["rmse_gain"] / total["rmse_gain"]
            if total["rmse_gain"] > 0 else np.nan
        )
        key = f"{relation.source_root}_to_{relation.receiver_root}"
        melanoma_results[key] = retained
        rows.append({
            "audit": "reciprocal_edge", "dataset": relation.dataset,
            "source_root": relation.source_root, "receiver_root": relation.receiver_root,
            "total_rmse_gain": total["rmse_gain"],
            "conditional_rmse_gain": conditional["rmse_gain"],
            "conditional_empirical_p": conditional["empirical_p"],
            "shared_covariate_explained_fraction": explained,
            "edge_retained": retained,
        })
    n_mel = sum(melanoma_results.values())
    reciprocal_status = (
        "bidirectional_independent" if n_mel == 2
        else "asymmetric_after_conditioning" if n_mel == 1
        else "shared_context_association"
    )
    for row in rows:
        if row["audit"] == "reciprocal_edge":
            row["reciprocal_status"] = reciprocal_status

    # Breast mediation/transitive audit.
    breast = relations.loc[relations.dataset.eq("xenium_breast")]
    e0013 = breast.loc[breast.source_root.eq("C00") & breast.receiver_root.eq("C13")].iloc[0]
    e0021 = breast.loc[breast.source_root.eq("C00") & breast.receiver_root.eq("C21")].iloc[0]
    e2113 = breast.loc[breast.source_root.eq("C21") & breast.receiver_root.eq("C13")].iloc[0]
    context = contexts["xenium_breast"]
    ids, focal, y, _ = exposure_and_response(context, e0013, programs, validator)
    mediator = align_exposure(context, e2113, ids)
    base = design_covariates(context, ids, ["C00", "C21", "C13"])
    fold, block = fold_and_block(context, ids)
    total = conditional_test(
        validator, y, base, focal, np.empty((len(ids), 0)), fold, block, rng, permutations
    )
    direct = conditional_test(
        validator, y, base, focal, mediator[:, None], fold, block, rng, permutations
    )
    mediator_effect = conditional_test(
        validator, y, base, mediator, focal[:, None], fold, block, rng, permutations
    )
    retention = direct["rmse_gain"] / total["rmse_gain"] if total["rmse_gain"] > 0 else np.nan
    c0013_retained = bool(
        direct["rmse_gain"] > 0 and direct["empirical_p"] < 0.05
        and retention >= config["mediation"]["direct_effect_retention_fraction"]
    )
    c2113_retained = bool(
        mediator_effect["rmse_gain"] > 0 and mediator_effect["empirical_p"] < 0.05
    )
    mediation_status = (
        "three_edges_independent" if c0013_retained and c2113_retained
        else "C00_to_C13_latent_transitive" if (not c0013_retained and c2113_retained)
        else "C21_not_supported_as_mediator"
    )
    rows.append({
        "audit": "mediation_transitive", "dataset": "xenium_breast",
        "source_root": "C00", "receiver_root": "C13",
        "total_rmse_gain": total["rmse_gain"],
        "conditional_rmse_gain": direct["rmse_gain"],
        "conditional_empirical_p": direct["empirical_p"],
        "mediator_conditional_rmse_gain": mediator_effect["rmse_gain"],
        "mediator_empirical_p": mediator_effect["empirical_p"],
        "direct_effect_retention_fraction": retention,
        "mediation_status": mediation_status,
        "edge_retained": c0013_retained,
    })

    # GSE245582 shared-source audit.
    gse = relations.loc[relations.dataset.eq("GSE245582")]
    gse_retained = {}
    for relation in gse.itertuples(index=False):
        context = contexts["GSE245582"]
        ids, focal, y, _ = exposure_and_response(context, relation, programs, validator)
        other_relation = gse.loc[~gse.receiver_root.eq(relation.receiver_root)].iloc[0]
        other = align_exposure(context, other_relation, ids)
        base = design_covariates(context, ids, ["C41", "C08", "C00"])
        fold, block = fold_and_block(context, ids)
        total = conditional_test(
            validator, y, base, focal, np.empty((len(ids), 0)), fold, block,
            rng, permutations,
        )
        conditional = conditional_test(
            validator, y, base, focal, other[:, None], fold, block,
            rng, permutations,
        )
        retained = conditional["rmse_gain"] > 0 and conditional["empirical_p"] < 0.05
        gse_retained[relation.receiver_root] = retained
        rows.append({
            "audit": "shared_source", "dataset": relation.dataset,
            "source_root": relation.source_root, "receiver_root": relation.receiver_root,
            "total_rmse_gain": total["rmse_gain"],
            "conditional_rmse_gain": conditional["rmse_gain"],
            "conditional_empirical_p": conditional["empirical_p"],
            "shared_covariate_explained_fraction": (
                1 - conditional["rmse_gain"] / total["rmse_gain"]
                if total["rmse_gain"] > 0 else np.nan
            ),
            "edge_retained": retained,
        })
    n_gse = sum(gse_retained.values())
    shared_status = (
        "multi_receiver_independent" if n_gse == 2
        else "receiver_specific_only" if n_gse == 1
        else "shared_global_source_effect"
    )
    for row in rows:
        if row["audit"] == "shared_source":
            row["shared_source_status"] = shared_status

    audit = pd.DataFrame(rows)
    audit.to_csv(output / "V781_NETWORK_CONDITIONAL_INDEPENDENCE_AUDIT.csv", index=False)
    multiplicity = multiplicity_audit(candidate, v761)
    multiplicity.to_csv(output / "V781_RELATION_MULTIPLICITY_AUDIT.csv", index=False)

    # Apply only status downgrades; never add a relation.
    final = relations.copy()
    final["network_status"] = "supported_independent"
    final["network_reason"] = "single_or_nonredundant_relation"
    for index, relation in final.iterrows():
        if relation.dataset == "xenium_melanoma":
            retained = melanoma_results[
                f"{relation.source_root}_to_{relation.receiver_root}"
            ]
            if not retained:
                final.loc[index, ["network_status", "network_reason"]] = [
                    "latent_reciprocal_context", reciprocal_status
                ]
        elif relation.dataset == "xenium_breast":
            if relation.source_root == "C00" and relation.receiver_root == "C13" and not c0013_retained:
                final.loc[index, ["network_status", "network_reason"]] = [
                    "latent_transitive_association", mediation_status
                ]
            elif relation.source_root == "C21" and relation.receiver_root == "C13" and not c2113_retained:
                final.loc[index, ["network_status", "network_reason"]] = [
                    "latent_mediator_unresolved", mediation_status
                ]
        elif relation.dataset == "GSE245582":
            if not gse_retained[str(relation.receiver_root)]:
                final.loc[index, ["network_status", "network_reason"]] = [
                    "latent_shared_source_context", shared_status
                ]

    prior = pd.read_csv(candidate / "ALL7_VALIDATED_GENERIC_BRANCH_PRIOR_AUDIT.csv")
    mapping_root = (
        project / "NicheForest_v7_4_6_all7_reproduction_bundle" / "data"
    )
    operational_rows = []
    tolerance = float(config["prior_operation"]["minimum_nonzero_change"])
    for relation in final.itertuples(index=False):
        local = prior.loc[
            prior.dataset.eq(relation.dataset)
            & prior.relation_type.eq(relation.relation_type)
            & prior.source_root.eq(relation.source_root)
            & prior.receiver_root.eq(relation.receiver_root)
        ]
        if relation.relation_type == "collision_like":
            relation_operation_class = (
                "typed_relation_edge_only"
                if relation.network_status == "supported_independent"
                else "latent_relation_no_operational_application"
            )
            operational_rows.append({
                "dataset": relation.dataset, "source_root": relation.source_root,
                "receiver_root": relation.receiver_root,
                "relation_operation_class": relation_operation_class,
                "numeric_prior_candidate": False,
                "prior_operational": False, "affected_cells": 0,
            })
            continue
        row = local.iloc[0]
        affected = {
            int(x) for x in str(row.affected_metacells).split("|") if x
        }
        mapping = pd.read_csv(
            mapping_root / relation.dataset / "cell_to_metacell.csv",
            usecols=["metacell_index"],
        )
        affected_cells = int(mapping.metacell_index.isin(affected).sum())
        transition_probability_change = abs(
            float(row.transition_probability_after)
            - float(row.transition_probability_before)
        )
        numeric_prior_candidate = bool(
            transition_probability_change > tolerance
            and float(row.terminal_fate_L1_change) > tolerance
            and int(row.n_affected_metacells) > 0 and affected_cells > 0
        )
        operational = bool(
            numeric_prior_candidate
            and relation.network_status == "supported_independent"
        )
        operational_rows.append({
            "dataset": relation.dataset, "source_root": relation.source_root,
            "receiver_root": relation.receiver_root,
            "relation_operation_class": (
                "typed_relation_edge_plus_operational_prior"
                if operational else "typed_relation_edge_only"
            ) if relation.network_status == "supported_independent"
            else "latent_relation_no_operational_application",
            "numeric_prior_candidate": numeric_prior_candidate,
            "prior_operational": operational,
            "affected_branch": row.affected_branch,
            "affected_metacells": int(row.n_affected_metacells),
            "affected_cells": affected_cells,
            "transition_probability_change": transition_probability_change,
            "terminal_fate_L1_change": row.terminal_fate_L1_change,
        })
    operational = pd.DataFrame(operational_rows)
    operational.to_csv(output / "V781_RELATION_OPERATION_CLASS_AUDIT.csv", index=False)
    final = final.merge(
        operational[
            ["dataset", "source_root", "receiver_root", "relation_operation_class"]
        ],
        on=["dataset", "source_root", "receiver_root"], how="left",
    )
    final.to_csv(output / "ALL7_NETWORK_AUDITED_RELATION_LAYER.csv", index=False)
    supported = final.loc[final.network_status.eq("supported_independent")].copy()
    latent = final.loc[~final.network_status.eq("supported_independent")].copy()
    supported.to_csv(output / "ALL7_SUPPORTED_INDEPENDENT_RELATION_EDGES.csv", index=False)
    latent.to_csv(output / "ALL7_LATENT_OR_REDUNDANT_RELATION_EDGES.csv", index=False)

    all_pass = bool(multiplicity.audit_pass.all())
    final_status = (
        "V781_NETWORK_AUDIT_PASS_FREEZE_CANDIDATE_NOT_FROZEN"
        if all_pass and len(supported)
        else "V781_NETWORK_AUDIT_NO_SUPPORTED_RELATION_KEEP_A0"
    )
    decision = {
        "final_status": final_status,
        "candidate_not_frozen": True,
        "automatic_freeze": False,
        "input_validated_edges": int(len(final)),
        "supported_independent_edges": int(len(supported)),
        "latent_or_redundant_edges": int(len(latent)),
        "supported_collision_like": int(supported.relation_type.eq("collision_like").sum()),
        "supported_promotion_like": int(supported.relation_type.eq("promotion_like").sum()),
        "supported_inhibition_like": int(supported.relation_type.eq("inhibition_like").sum()),
        "edge_plus_operational_prior": int(
            supported.relation_operation_class.eq(
                "typed_relation_edge_plus_operational_prior"
            ).sum()
        ),
        "edge_only": int(
            supported.relation_operation_class.eq("typed_relation_edge_only").sum()
        ),
        "multiplicity_complete_family_pass": all_pass,
        "A0_modified": False,
        "mechanism_specific_relations": 0,
    }
    (output / "FINAL_NETWORK_AUDIT_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    report = [
        "# v7.8.1 network-level conditional-independence audit",
        "",
        f"- Final status: `{final_status}`",
        f"- Input edges: {len(final)}",
        f"- Supported independent edges: {len(supported)}",
        f"- Latent/redundant edges: {len(latent)}",
        "- Candidate only; not frozen.",
        "",
        "## Conditional network audit",
        "",
        audit.to_markdown(index=False),
        "",
        "## Final network-audited layer",
        "",
        final.to_markdown(index=False),
        "",
        "Program annotations remain exploratory. Conditional independence is predictive/statistical and does not establish molecular causality.",
    ]
    (output / "FINAL_NETWORK_AUDIT_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    for context in contexts.values():
        context["raw"].file.close()
    files = [
        x for x in output.glob("*")
        if x.is_file() and x.name != "OUTPUT_MANIFEST_SHA256.csv"
    ] + [Path(__file__).resolve(), config_path]
    pd.DataFrame([
        {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(files, key=str)
    ]).to_csv(output / "OUTPUT_MANIFEST_SHA256.csv", index=False)
    print(audit.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
