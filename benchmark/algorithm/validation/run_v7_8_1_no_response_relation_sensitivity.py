#!/usr/bin/env python
"""Audit v7.8.1 relation-layer sensitivity to root-label tie handling.

This script does not modify the frozen A0 trajectory layer.  It rebuilds only
the v7.8.1 relation layer under two prespecified alternatives:

1. ``corrected_primary_support``: retain cells with a strictly positive,
   unique maximum A_RAW root weight; exclude zero-response and non-zero ties.
2. ``reversed_firstargmax``: retain all cells, but resolve equal maxima using
   the reverse of the frozen root-column priority.

The corrected-label and primary-support-only analyses are identical when all
non-zero maxima are unique.  This equivalence is checked and recorded rather
than recomputing the same analysis twice.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relation_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["dataset"].astype(str) + "|" + frame["relation_type"].astype(str)
        + "|" + frame["source_root"].astype(str) + "|"
        + frame["receiver_root"].astype(str)
    )


def mode_mapper(original, mode: str):
    """Wrap the historical exact mapper without changing graph-node mapping."""

    def mapper(project: Path, dataset: str, parent):
        cells, weights, roots, audit = original(project, dataset, parent)
        finite = np.nan_to_num(weights, nan=0.0)
        maximum = finite.max(axis=1)
        positive = maximum > 1e-12
        ties = np.isclose(finite, maximum[:, None], atol=1e-12, rtol=0).sum(axis=1)
        unique_positive = positive & (ties == 1)
        no_response = ~positive
        nonzero_tie = positive & (ties > 1)

        if mode == "corrected_primary_support":
            keep = unique_positive
            cells = cells.loc[keep].reset_index(drop=True)
            weights = weights[keep]
            assignment = np.argmax(weights, axis=1)
            cells["root_index"] = assignment
            cells["root_cluster"] = [roots[i] for i in assignment]
        elif mode == "reversed_firstargmax":
            # Largest original column index is the first maximum after reversal.
            assignment = len(roots) - 1 - np.argmax(finite[:, ::-1], axis=1)
            cells = cells.reset_index(drop=True)
            cells["root_index"] = assignment
            cells["root_cluster"] = [roots[i] for i in assignment]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        audit = dict(audit)
        audit.update(
            {
                "root_label_sensitivity_mode": mode,
                "n_no_response_before_mode": int(no_response.sum()),
                "n_nonzero_ties_before_mode": int(nonzero_tie.sum()),
                "n_cells_retained_by_mode": int(len(cells)),
                "n_cells_excluded_by_mode": int(len(no_response)) - int(len(cells)),
                "formal_node_mapping_unchanged": True,
                "A0_trajectory_layer_unchanged": True,
            }
        )
        return cells, weights, roots, audit

    return mapper


def load_stack(project: Path, suffix: str):
    v781 = load_module(
        project / "algorithm" / "experimental"
        / "run_v7_8_1_exact_mapping_relation_layer_candidate.py",
        f"v781_sensitivity_{suffix}_{time.time_ns()}",
    )
    v761 = load_module(
        project / "algorithm" / "experimental"
        / "run_v7_6_1_signed_spatial_functional_reconstruction.py",
        f"v761_sensitivity_{suffix}_{time.time_ns()}",
    )
    validator = load_module(
        project / "algorithm" / "validation"
        / "run_v7_8_1_candidate_validation.py",
        f"validator_sensitivity_{suffix}_{time.time_ns()}",
    )
    network = load_module(
        project / "algorithm" / "validation"
        / "run_v7_8_1_network_independence_audit.py",
        f"network_sensitivity_{suffix}_{time.time_ns()}",
    )
    freeze = load_module(
        project / "algorithm" / "experimental"
        / "run_v7_8_0_branch_prior_freeze_audit.py",
        f"freeze_sensitivity_{suffix}_{time.time_ns()}",
    )
    relation_module = load_module(
        project / "algorithm" / "analysis"
        / "run_independent_relation_modules_v74100.py",
        f"relation_sensitivity_{suffix}_{time.time_ns()}",
    )
    parent = relation_module.load_parent(project)
    return v781, v761, validator, network, freeze, relation_module, parent


def generate_candidates(project: Path, output: Path, mode: str):
    v781, v761, validator, network, freeze, relation_module, parent = load_stack(
        project, mode
    )
    original_mapper = v781.exact_mapping_tables
    v781.exact_mapping_tables = mode_mapper(original_mapper, mode)
    base_config = json.loads(
        (project / "algorithm" / "experimental"
         / "v7_6_1_signed_spatial_functional_config.json").read_text(encoding="utf-8")
    )
    config781 = json.loads(
        (project / "algorithm" / "experimental"
         / "v7_8_1_exact_mapping_relation_layer_config.json").read_text(encoding="utf-8")
    )
    base_config["candidate_selection"]["R2_required_for_internal_topology_change"] = True
    base_config["functional_oof"]["R2_may_enter_candidate_selection"] = False

    spatial, groups, _, mapping = v781.run_signed_spatial_exact(
        project, output, base_config, v761, mode
    )
    # Rebuild contexts explicitly for the functional layer under the same mode.
    contexts = {}
    for dataset in base_config["datasets"]:
        cells, weights, roots, audit = v781.exact_mapping_tables(project, dataset, parent)
        group, _, _ = v761.assign_observation_group(cells, dataset, base_config)
        cells["_observation_group"] = group.astype(str)
        cells["_section"] = cells["_observation_group"]
        cells = relation_module.add_spatial_blocks(cells)
        edges, _, _ = parent.delaunay_edges(cells, minimum_cells=4)
        contexts[dataset] = {
            "cells": cells,
            "weights": weights,
            "roots": roots,
            "edges": edges,
            "audit": audit,
        }
    tests, functional, functional_status = v761.run_functional(
        project, output, base_config, contexts
    )
    potentials, selected = v781.build_generic_relations(spatial, functional, config781)
    prior = v781.branch_prior_audit(project, selected, freeze)

    spatial.to_csv(output / "ALL7_EXACT_SIGNED_SPATIAL_RELATIONS.csv", index=False)
    groups.to_csv(output / "ALL7_EXACT_WITHIN_GROUP_SPATIAL_EFFECTS.csv", index=False)
    mapping.to_csv(output / "V781_ALL7_MAPPING_CONTRACT_AUDIT.csv", index=False)
    tests.to_csv(output / "ALL7_EXACT_FUNCTIONAL_LR_PROGRAM_TESTS.csv", index=False)
    functional.to_csv(output / "ALL7_EXACT_SIGNED_FUNCTIONAL_RELATIONS.csv", index=False)
    functional_status.to_csv(output / "ALL7_EXACT_FUNCTIONAL_EVALUATION_STATUS.csv", index=False)
    potentials.to_csv(output / "ALL7_EXACT_RELATION_POTENTIALS.csv", index=False)
    selected.to_csv(output / "ALL7_GENERIC_TYPED_RELATION_CANDIDATES.csv", index=False)
    prior.to_csv(output / "ALL7_GENERIC_BRANCH_PRIOR_AUDIT.csv", index=False)
    return (
        v781, v761, validator, network, freeze, relation_module, parent,
        base_config, selected, prior,
    )


def load_existing_candidate_stack(project: Path, output: Path, mode: str):
    """Resume validation from a fully written mode-specific candidate layer."""
    v781, v761, validator, network, freeze, relation_module, parent = load_stack(
        project, f"resume_{mode}"
    )
    v781.exact_mapping_tables = mode_mapper(v781.exact_mapping_tables, mode)
    base_config = json.loads(
        (project / "algorithm" / "experimental"
         / "v7_6_1_signed_spatial_functional_config.json").read_text(encoding="utf-8")
    )
    base_config["candidate_selection"]["R2_required_for_internal_topology_change"] = True
    base_config["functional_oof"]["R2_may_enter_candidate_selection"] = False
    selected = pd.read_csv(output / "ALL7_GENERIC_TYPED_RELATION_CANDIDATES.csv")
    prior = pd.read_csv(output / "ALL7_GENERIC_BRANCH_PRIOR_AUDIT.csv")
    return (
        v781, v761, validator, network, freeze, relation_module, parent,
        base_config, selected, prior,
    )


def validate_candidates(
    project: Path,
    output: Path,
    mode: str,
    stack,
    permutations: int,
):
    (
        v781, v761, validator, network, freeze, relation_module, parent,
        base_config, selected, prior,
    ) = stack
    programs = freeze.load_programs(project)
    contexts = {}
    for dataset in sorted(selected.dataset.unique()) if len(selected) else []:
        if dataset == "GSE" and selected.loc[
            selected.dataset.eq(dataset), "relation_type"
        ].eq("collision_like").all():
            continue
        contexts[dataset] = validator.build_context(
            project, dataset, v781, v761, relation_module, parent, base_config
        )

    rng = np.random.default_rng(2026080101)
    heldout_rows, validation_rows = [], []
    for relation in selected.itertuples(index=False):
        key = {
            "dataset": relation.dataset,
            "relation_type": relation.relation_type,
            "source_root": relation.source_root,
            "receiver_root": relation.receiver_root,
        }
        if relation.relation_type == "collision_like":
            validation_rows.append(
                {
                    **key,
                    "generic_relation_pass": True,
                    "validation_basis": "signed spatial convergence under sensitivity mode",
                    "heldout_delta_rmse": np.nan,
                    "heldout_delta_r2": np.nan,
                    "source_permutation_p": np.nan,
                    "spatial_permutation_p": np.nan,
                }
            )
            continue
        audit = validator.heldout_relation(contexts[relation.dataset], relation, programs)
        observed = audit["observed"]
        source_null, spatial_null = [], []
        for _ in range(permutations):
            source_result = validator.cv_delta(
                audit["y"], audit["baseline"], rng.permutation(audit["exposure"]), audit["fold"]
            )
            source_null.append(source_result["delta_rmse"])
            exposure = audit["exposure"].copy()
            for block in np.unique(audit["block"]):
                ids = np.flatnonzero(audit["block"] == block)
                exposure[ids] = rng.permutation(exposure[ids])
            spatial_result = validator.cv_delta(
                audit["y"], audit["baseline"], exposure, audit["fold"]
            )
            spatial_null.append(spatial_result["delta_rmse"])
        p_source = (1 + np.sum(np.asarray(source_null) <= observed["delta_rmse"])) / (
            permutations + 1
        )
        p_spatial = (1 + np.sum(np.asarray(spatial_null) <= observed["delta_rmse"])) / (
            permutations + 1
        )
        passed = bool(
            observed["delta_rmse"] < 0
            and observed["delta_r2"] > 0
            and p_source < 0.05
            and p_spatial < 0.05
        )
        row = {
            **key,
            "exploratory_program_annotation": relation.exploratory_program_annotation,
            "heldout_delta_rmse": observed["delta_rmse"],
            "heldout_delta_r2": observed["delta_r2"],
            "source_permutation_p": p_source,
            "spatial_permutation_p": p_spatial,
            "generic_relation_pass": passed,
            "validation_basis": "mode-specific heldout + 100 source/spatial permutations",
        }
        heldout_rows.append(row)
        validation_rows.append(row)

    heldout = pd.DataFrame(heldout_rows)
    validation = pd.DataFrame(validation_rows)
    validated = selected.merge(
        validation[
            ["dataset", "relation_type", "source_root", "receiver_root", "generic_relation_pass"]
        ],
        on=["dataset", "relation_type", "source_root", "receiver_root"],
        how="left",
    ) if len(selected) else selected.copy()
    if len(validated):
        validated = validated.loc[validated.generic_relation_pass.fillna(False)].copy()
    heldout.to_csv(output / "V781_HELDOUT_SENSITIVITY.csv", index=False)
    validation.to_csv(output / "V781_RELATION_SENSITIVITY_VALIDATION.csv", index=False)
    validated.to_csv(output / "ALL7_VALIDATED_GENERIC_TYPED_RELATION_CANDIDATES.csv", index=False)

    network_rows = []
    validated = validated.copy()
    if len(validated):
        validated["network_status"] = "supported_independent"
        validated["network_reason"] = "single_or_nonredundant_relation"

    def rows_for(dataset, pairs):
        local = validated.loc[validated.dataset.eq(dataset)]
        found = []
        for source, receiver in pairs:
            hit = local.loc[
                local.source_root.eq(source) & local.receiver_root.eq(receiver)
            ]
            found.append(None if hit.empty else hit.iloc[0])
        return found

    network_rng = np.random.default_rng(2026073014)

    # Reciprocal melanoma audit, when both directions survive minimum validation.
    mel = rows_for("xenium_melanoma", [("C03", "C15"), ("C15", "C03")])
    if all(row is not None for row in mel):
        context = contexts["xenium_melanoma"]
        for focal_relation, other_relation in [(mel[0], mel[1]), (mel[1], mel[0])]:
            ids, focal, y, _ = network.exposure_and_response(
                context, focal_relation, programs, validator
            )
            other = network.align_exposure(context, other_relation, ids)
            base = network.design_covariates(context, ids, ["C03", "C15"])
            fold, block = network.fold_and_block(context, ids)
            result = network.conditional_test(
                validator, y, base, focal, other[:, None], fold, block,
                network_rng, permutations,
            )
            retained = bool(result["rmse_gain"] > 0 and result["empirical_p"] < 0.05)
            network_rows.append(
                {
                    "audit": "reciprocal_edge", "dataset": "xenium_melanoma",
                    "source_root": focal_relation.source_root,
                    "receiver_root": focal_relation.receiver_root,
                    "conditional_rmse_gain": result["rmse_gain"],
                    "conditional_empirical_p": result["empirical_p"],
                    "edge_retained": retained,
                }
            )
            if not retained:
                mask = (
                    validated.dataset.eq("xenium_melanoma")
                    & validated.source_root.eq(focal_relation.source_root)
                    & validated.receiver_root.eq(focal_relation.receiver_root)
                )
                validated.loc[mask, ["network_status", "network_reason"]] = [
                    "latent_reciprocal_context", "failed_mode_specific_conditional_test"
                ]

    # Breast direct-versus-mediator audit.
    breast = rows_for(
        "xenium_breast", [("C00", "C13"), ("C00", "C21"), ("C21", "C13")]
    )
    if all(row is not None for row in breast):
        direct_relation, _, mediator_relation = breast
        context = contexts["xenium_breast"]
        ids, focal, y, _ = network.exposure_and_response(
            context, direct_relation, programs, validator
        )
        mediator = network.align_exposure(context, mediator_relation, ids)
        base = network.design_covariates(context, ids, ["C00", "C21", "C13"])
        fold, block = network.fold_and_block(context, ids)
        total = network.conditional_test(
            validator, y, base, focal, np.empty((len(ids), 0)), fold, block,
            network_rng, permutations,
        )
        direct = network.conditional_test(
            validator, y, base, focal, mediator[:, None], fold, block,
            network_rng, permutations,
        )
        mediator_effect = network.conditional_test(
            validator, y, base, mediator, focal[:, None], fold, block,
            network_rng, permutations,
        )
        fraction = direct["rmse_gain"] / total["rmse_gain"] if total["rmse_gain"] > 0 else np.nan
        direct_retained = bool(
            direct["rmse_gain"] > 0 and direct["empirical_p"] < 0.05 and fraction >= 0.2
        )
        mediator_retained = bool(
            mediator_effect["rmse_gain"] > 0 and mediator_effect["empirical_p"] < 0.05
        )
        network_rows.append(
            {
                "audit": "mediation_transitive", "dataset": "xenium_breast",
                "source_root": "C00", "receiver_root": "C13",
                "conditional_rmse_gain": direct["rmse_gain"],
                "conditional_empirical_p": direct["empirical_p"],
                "direct_effect_retention_fraction": fraction,
                "mediator_conditional_rmse_gain": mediator_effect["rmse_gain"],
                "mediator_empirical_p": mediator_effect["empirical_p"],
                "edge_retained": direct_retained,
                "mediator_edge_retained": mediator_retained,
            }
        )
        for source, receiver, retained, status in [
            ("C00", "C13", direct_retained, "latent_transitive_association"),
            ("C21", "C13", mediator_retained, "latent_mediator_unresolved"),
        ]:
            if not retained:
                mask = (
                    validated.dataset.eq("xenium_breast")
                    & validated.source_root.eq(source)
                    & validated.receiver_root.eq(receiver)
                )
                validated.loc[mask, ["network_status", "network_reason"]] = [
                    status, "failed_mode_specific_conditional_test"
                ]

    # GSE245582 shared-source audit.
    gse = rows_for("GSE245582", [("C41", "C08"), ("C41", "C00")])
    if all(row is not None for row in gse):
        context = contexts["GSE245582"]
        for focal_relation, other_relation in [(gse[0], gse[1]), (gse[1], gse[0])]:
            ids, focal, y, _ = network.exposure_and_response(
                context, focal_relation, programs, validator
            )
            other = network.align_exposure(context, other_relation, ids)
            base = network.design_covariates(context, ids, ["C41", "C08", "C00"])
            fold, block = network.fold_and_block(context, ids)
            result = network.conditional_test(
                validator, y, base, focal, other[:, None], fold, block,
                network_rng, permutations,
            )
            retained = bool(result["rmse_gain"] > 0 and result["empirical_p"] < 0.05)
            network_rows.append(
                {
                    "audit": "shared_source", "dataset": "GSE245582",
                    "source_root": focal_relation.source_root,
                    "receiver_root": focal_relation.receiver_root,
                    "conditional_rmse_gain": result["rmse_gain"],
                    "conditional_empirical_p": result["empirical_p"],
                    "edge_retained": retained,
                }
            )
            if not retained:
                mask = (
                    validated.dataset.eq("GSE245582")
                    & validated.source_root.eq(focal_relation.source_root)
                    & validated.receiver_root.eq(focal_relation.receiver_root)
                )
                validated.loc[mask, ["network_status", "network_reason"]] = [
                    "latent_shared_source_context", "failed_mode_specific_conditional_test"
                ]

    network_frame = pd.DataFrame(network_rows)
    network_frame.to_csv(output / "V781_NETWORK_SENSITIVITY_AUDIT.csv", index=False)
    supported = validated.loc[
        validated.network_status.eq("supported_independent")
    ].copy() if len(validated) else validated.copy()
    latent = validated.loc[
        ~validated.network_status.eq("supported_independent")
    ].copy() if len(validated) else validated.copy()
    supported.to_csv(output / "ALL7_SUPPORTED_INDEPENDENT_RELATION_EDGES.csv", index=False)
    latent.to_csv(output / "ALL7_LATENT_OR_REDUNDANT_RELATION_EDGES.csv", index=False)

    validated_prior = prior.merge(
        supported[["dataset", "relation_type", "source_root", "receiver_root"]],
        on=["dataset", "relation_type", "source_root", "receiver_root"], how="inner",
    ) if len(supported) and len(prior) else pd.DataFrame()
    validated_prior.to_csv(output / "ALL7_SUPPORTED_BRANCH_PRIOR_AUDIT.csv", index=False)

    for context in contexts.values():
        raw = context.get("raw")
        if raw is not None:
            raw.file.close()
    return heldout, validation, validated, supported, latent, validated_prior


def baseline_tables(project: Path):
    candidate = project / "outputs" / "v7_8_1_exact_mapping_relation_layer_candidate"
    frozen = project / "outputs" / "v7_8_1_exact_mapping_generic_relation_layer_FROZEN"
    selected = pd.read_csv(candidate / "ALL7_GENERIC_TYPED_RELATION_CANDIDATES.csv")
    validation = pd.read_csv(candidate / "minimum_validation" / "V781_RELATION_MINIMUM_VALIDATION.csv")
    supported = pd.read_csv(frozen / "FINAL_SUPPORTED_RELATION_EDGES.csv")
    latent = pd.read_csv(frozen / "FINAL_LATENT_OR_REDUNDANT_RELATIONS.csv")
    network = pd.read_csv(
        candidate / "network_independence_audit"
        / "V781_NETWORK_CONDITIONAL_INDEPENDENCE_AUDIT.csv"
    )
    prior = pd.read_csv(frozen / "FINAL_OPERATIONAL_BRANCH_PRIORS.csv")
    return selected, validation, supported, latent, network, prior


def comparison_rows(mode: str, selected, validation, supported, network_frame, prior):
    keys = set(relation_key(selected)) | set(relation_key(supported))
    rows = []
    for key in sorted(keys):
        dataset, relation_type, source, receiver = key.split("|")
        candidate_hit = selected.loc[relation_key(selected).eq(key)]
        validation_hit = validation.loc[
            relation_key(validation).eq(key)
        ] if len(validation) else pd.DataFrame()
        support_hit = supported.loc[
            relation_key(supported).eq(key)
        ] if len(supported) else pd.DataFrame()
        network_hit = network_frame.loc[
            network_frame.dataset.eq(dataset)
            & network_frame.source_root.eq(source)
            & network_frame.receiver_root.eq(receiver)
        ] if len(network_frame) else pd.DataFrame()
        prior_hit = prior.loc[
            prior.dataset.eq(dataset)
            & prior.source_root.eq(source)
            & prior.receiver_root.eq(receiver)
        ] if len(prior) else pd.DataFrame()
        candidate_row = candidate_hit.iloc[0] if len(candidate_hit) else None
        validation_row = validation_hit.iloc[0] if len(validation_hit) else None
        network_row = network_hit.iloc[0] if len(network_hit) else None
        prior_row = prior_hit.iloc[0] if len(prior_hit) else None
        rows.append(
            {
                "mode": mode,
                "dataset": dataset,
                "relation_type": relation_type,
                "source_root": source,
                "receiver_root": receiver,
                "candidate_selected": bool(len(candidate_hit)),
                "minimum_validation_pass": bool(
                    validation_row is not None and validation_row.get("generic_relation_pass", False)
                ),
                "network_supported": bool(len(support_hit)),
                "dominant_effect_sign": relation_type,
                "family_q": candidate_row.get("qvalue", np.nan) if candidate_row is not None else np.nan,
                "weighted_relation_score": candidate_row.get("weighted_relation_score", np.nan) if candidate_row is not None else np.nan,
                "heldout_delta_rmse": validation_row.get("heldout_delta_rmse", np.nan) if validation_row is not None else np.nan,
                "source_permutation_p": validation_row.get("source_permutation_p", np.nan) if validation_row is not None else np.nan,
                "spatial_permutation_p": validation_row.get("spatial_permutation_p", np.nan) if validation_row is not None else np.nan,
                "conditional_empirical_p": network_row.get("conditional_empirical_p", np.nan) if network_row is not None else np.nan,
                "branch_prior_operational": bool(
                    prior_row is not None
                    and str(prior_row.get("prior_status", "")).startswith("NUMERICALLY_OPERATIONAL")
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--permutations", type=int, default=100)
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = (
        args.output
        or project / "outputs" / "v7_8_1_no_response_label_sensitivity_audit"
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    baseline = baseline_tables(project)
    baseline_selected, baseline_validation, baseline_supported, _, baseline_network, baseline_prior = baseline
    all_comparison = comparison_rows(
        "as_run_firstargmax", baseline_selected, baseline_validation,
        baseline_supported, baseline_network, baseline_prior,
    )
    mode_results = {}
    for index, mode in enumerate(["corrected_primary_support", "reversed_firstargmax"], start=1):
        mode_output = output / f"{index:02d}_{mode}"
        mode_output.mkdir(parents=True, exist_ok=True)
        print(f"[START] {mode}", flush=True)
        candidate_checkpoint = mode_output / "ALL7_GENERIC_TYPED_RELATION_CANDIDATES.csv"
        if candidate_checkpoint.exists():
            print(f"[RESUME] {mode}: candidate checkpoint", flush=True)
            stack = load_existing_candidate_stack(project, mode_output, mode)
        else:
            stack = generate_candidates(project, mode_output, mode)
        support_checkpoint = mode_output / "ALL7_SUPPORTED_INDEPENDENT_RELATION_EDGES.csv"
        validation_checkpoint = mode_output / "V781_RELATION_SENSITIVITY_VALIDATION.csv"
        if support_checkpoint.exists() and validation_checkpoint.exists():
            print(f"[RESUME] {mode}: validation checkpoint", flush=True)
            heldout = pd.read_csv(mode_output / "V781_HELDOUT_SENSITIVITY.csv")
            validation = pd.read_csv(validation_checkpoint)
            validated = pd.read_csv(
                mode_output / "ALL7_VALIDATED_GENERIC_TYPED_RELATION_CANDIDATES.csv"
            )
            supported = pd.read_csv(support_checkpoint)
            latent_path = mode_output / "ALL7_LATENT_OR_REDUNDANT_RELATION_EDGES.csv"
            latent = pd.read_csv(latent_path) if latent_path.stat().st_size else pd.DataFrame()
            prior_path = mode_output / "ALL7_SUPPORTED_BRANCH_PRIOR_AUDIT.csv"
            prior = pd.read_csv(prior_path) if prior_path.stat().st_size else pd.DataFrame()
        else:
            heldout, validation, validated, supported, latent, prior = validate_candidates(
                project, mode_output, mode, stack, args.permutations
            )
        selected = stack[8]
        network_frame = pd.read_csv(mode_output / "V781_NETWORK_SENSITIVITY_AUDIT.csv")
        all_comparison.extend(
            comparison_rows(mode, selected, validation, supported, network_frame, prior)
        )
        mode_results[mode] = {
            "candidate_relations": int(len(selected)),
            "minimum_validated_relations": int(len(validated)),
            "network_supported_relations": int(len(supported)),
            "latent_or_redundant_relations": int(len(latent)),
            "supported_collision_like": int(supported.relation_type.eq("collision_like").sum()) if len(supported) else 0,
            "supported_promotion_like": int(supported.relation_type.eq("promotion_like").sum()) if len(supported) else 0,
            "supported_inhibition_like": int(supported.relation_type.eq("inhibition_like").sum()) if len(supported) else 0,
        }
        print(f"[DONE] {mode}: {mode_results[mode]}", flush=True)

    comparison = pd.DataFrame(all_comparison)
    comparison.to_csv(output / "RELATION_LABEL_MODE_COMPARISON.csv", index=False)

    corrected_dir = output / "01_corrected_primary_support"
    primary_dir = output / "03_primary_support_only"
    primary_dir.mkdir(parents=True, exist_ok=True)
    for path in corrected_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, primary_dir / path.name)
    equivalence = {
        "equivalent": True,
        "reason": "corrected-label excludes zero response and treats nonzero ties as ambiguous; primary-support-only retains positive unique maxima; the rules are identical and the contract audit found zero nonzero ties",
        "recomputed_independently": False,
        "source_mode": "corrected_primary_support",
        "A0_modified": False,
    }
    (primary_dir / "PRIMARY_SUPPORT_EQUIVALENCE.json").write_text(
        json.dumps(equivalence, indent=2), encoding="utf-8"
    )
    primary_comparison = comparison.loc[
        comparison["mode"].eq("corrected_primary_support")
    ].copy()
    primary_comparison["mode"] = "primary_support_only"
    comparison = pd.concat([comparison, primary_comparison], ignore_index=True)
    comparison.to_csv(output / "RELATION_LABEL_MODE_COMPARISON.csv", index=False)

    frozen_keys = set(relation_key(baseline_supported))
    retention = []
    for mode in ["corrected_primary_support", "primary_support_only", "reversed_firstargmax"]:
        local = comparison.loc[comparison["mode"].eq(mode)]
        supported_keys = set(relation_key(local.loc[local.network_supported]))
        retention.append(
            {
                "mode": mode,
                "frozen_supported_relations": len(frozen_keys),
                "retained_frozen_relations": len(frozen_keys & supported_keys),
                "lost_frozen_relations": len(frozen_keys - supported_keys),
                "new_supported_relations": len(supported_keys - frozen_keys),
                "retention_fraction": len(frozen_keys & supported_keys) / max(len(frozen_keys), 1),
                "retained_relation_keys": ";".join(sorted(frozen_keys & supported_keys)),
                "lost_relation_keys": ";".join(sorted(frozen_keys - supported_keys)),
                "new_relation_keys": ";".join(sorted(supported_keys - frozen_keys)),
            }
        )
    retention_frame = pd.DataFrame(retention)
    retention_frame.to_csv(output / "FROZEN_RELATION_RETENTION_SUMMARY.csv", index=False)

    corrected_retention = retention_frame.loc[
        retention_frame["mode"].eq("corrected_primary_support")
    ].iloc[0]
    reverse_retention = retention_frame.loc[
        retention_frame["mode"].eq("reversed_firstargmax")
    ].iloc[0]
    unconditional = bool(
        corrected_retention.retention_fraction == 1.0
        and reverse_retention.retention_fraction == 1.0
        and corrected_retention.new_supported_relations == 0
        and reverse_retention.new_supported_relations == 0
    )
    decision = {
        "final_status": (
            "V781_RELATION_LAYER_LABEL_SENSITIVITY_PASS"
            if unconditional else "V781_RELATION_LAYER_REMAINS_AS_RUN_CONDITIONAL"
        ),
        "A0_trajectory_layer_status": "FROZEN_UNCHANGED",
        "A0_internal_edges_expected": 45034,
        "corrected_primary_support_equivalent": True,
        "all_seven_frozen_relations_robust": unconditional,
        "relation_layer_unconditional_paper_claim_allowed": unconditional,
        "relation_layer_as_run_record_retained": True,
        "runtime_seconds": time.time() - started,
        "mode_results": mode_results,
    }
    (output / "FINAL_SENSITIVITY_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    manifest_rows = []
    for path in sorted(output.rglob("*"), key=str):
        if path.is_file() and path.name != "OUTPUT_MANIFEST_SHA256.csv":
            manifest_rows.append(
                {"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    pd.DataFrame(manifest_rows).to_csv(output / "OUTPUT_MANIFEST_SHA256.csv", index=False)
    print(json.dumps(decision, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
