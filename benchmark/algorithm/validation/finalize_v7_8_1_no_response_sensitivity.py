#!/usr/bin/env python
"""Finalize the paper-facing v7.8.1 no-response sensitivity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["dataset", "relation_type", "source_root", "receiver_root"]


def key(frame: pd.DataFrame) -> pd.Series:
    return frame[KEYS].astype(str).agg("|".join, axis=1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_mode(project: Path, audit: Path, mode: str):
    if mode == "as_run_firstargmax":
        candidate = project / "outputs" / "v7_8_1_exact_mapping_relation_layer_candidate"
        frozen = project / "outputs" / "v7_8_1_exact_mapping_generic_relation_layer_FROZEN"
        return {
            "selected": pd.read_csv(candidate / "ALL7_GENERIC_TYPED_RELATION_CANDIDATES.csv"),
            "functional": pd.read_csv(candidate / "ALL7_EXACT_SIGNED_FUNCTIONAL_RELATIONS.csv"),
            "validation": pd.read_csv(candidate / "minimum_validation" / "V781_RELATION_MINIMUM_VALIDATION.csv"),
            "supported": pd.read_csv(frozen / "FINAL_SUPPORTED_RELATION_EDGES.csv"),
            "prior": pd.read_csv(frozen / "FINAL_OPERATIONAL_BRANCH_PRIORS.csv"),
        }
    folder = {
        "corrected_primary_support": "01_corrected_primary_support",
        "reversed_firstargmax": "02_reversed_firstargmax",
        "primary_support_only": "03_primary_support_only",
    }[mode]
    base = audit / folder
    return {
        "selected": pd.read_csv(base / "ALL7_GENERIC_TYPED_RELATION_CANDIDATES.csv"),
        "functional": pd.read_csv(base / "ALL7_EXACT_SIGNED_FUNCTIONAL_RELATIONS.csv"),
        "validation": pd.read_csv(base / "V781_RELATION_SENSITIVITY_VALIDATION.csv"),
        "supported": pd.read_csv(base / "ALL7_SUPPORTED_INDEPENDENT_RELATION_EDGES.csv"),
        "prior": pd.read_csv(base / "ALL7_SUPPORTED_BRANCH_PRIOR_AUDIT.csv"),
    }


def detailed_rows(mode: str, tables: dict) -> list[dict]:
    selected = tables["selected"]
    functional = tables["functional"]
    validation = tables["validation"]
    supported = tables["supported"]
    prior = tables["prior"]
    rows = []
    universe = sorted(set(key(selected)) | set(key(supported)))
    for relation_key in universe:
        dataset, relation_type, source, receiver = relation_key.split("|")
        hit = selected.loc[key(selected).eq(relation_key)]
        candidate = hit.iloc[0] if len(hit) else None
        val_hit = validation.loc[key(validation).eq(relation_key)] if len(validation) else pd.DataFrame()
        val = val_hit.iloc[0] if len(val_hit) else None
        sup_hit = supported.loc[key(supported).eq(relation_key)] if len(supported) else pd.DataFrame()
        func = None
        if candidate is not None and relation_type != "collision_like":
            f = functional.loc[
                functional.dataset.eq(dataset)
                & functional.source_root.eq(source)
                & functional.receiver_root.eq(receiver)
                & functional.dominant_effect_sign.eq(relation_type)
                & functional.receiver_target_program.eq(
                    str(candidate.exploratory_program_annotation)
                )
            ]
            if len(f):
                func = f.iloc[0]
        prior_hit = prior.loc[
            prior.dataset.eq(dataset)
            & prior.source_root.eq(source)
            & prior.receiver_root.eq(receiver)
        ] if len(prior) else pd.DataFrame()
        prior_operational = False
        transition_change = np.nan
        fate_change = np.nan
        if len(prior_hit):
            p = prior_hit.iloc[0]
            if "prior_operational" in p.index:
                prior_operational = bool(p.prior_operational)
            else:
                prior_operational = str(p.get("prior_status", "")).startswith(
                    "NUMERICALLY_OPERATIONAL"
                )
            transition_change = p.get("transition_probability_change", np.nan)
            if not np.isfinite(pd.to_numeric(transition_change, errors="coerce")):
                before = pd.to_numeric(p.get("transition_probability_before", np.nan), errors="coerce")
                after = pd.to_numeric(p.get("transition_probability_after", np.nan), errors="coerce")
                transition_change = abs(after - before)
            fate_change = p.get("terminal_fate_L1_change", np.nan)
        rows.append(
            {
                "mode": mode,
                "dataset": dataset,
                "relation_type": relation_type,
                "source_root": source,
                "receiver_root": receiver,
                "candidate_selected": bool(len(hit)),
                "minimum_validation_pass": bool(
                    val is not None and str(val.get("generic_relation_pass", "")).lower() == "true"
                ),
                "network_supported": bool(len(sup_hit)),
                "exploratory_program_annotation": (
                    candidate.get("exploratory_program_annotation", "") if candidate is not None else ""
                ),
                "source_ligand": candidate.get("source_ligand", "") if candidate is not None else "",
                "target_receptor": candidate.get("target_receptor", "") if candidate is not None else "",
                "family_q": candidate.get("qvalue", np.nan) if candidate is not None else np.nan,
                "sign_concordance_q": func.get("sign_concordance_q_all7", np.nan) if func is not None else np.nan,
                "weighted_relation_score": candidate.get("weighted_relation_score", np.nan) if candidate is not None else np.nan,
                "heldout_delta_rmse": val.get("heldout_delta_rmse", np.nan) if val is not None else np.nan,
                "source_permutation_p": val.get("source_permutation_p", np.nan) if val is not None else np.nan,
                "spatial_permutation_p": val.get("spatial_permutation_p", np.nan) if val is not None else np.nan,
                "branch_prior_operational": prior_operational,
                "transition_probability_change": transition_change,
                "terminal_fate_L1_change": fate_change,
            }
        )
    return rows


def coupled_graph(a0: pd.DataFrame, relations: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Build a typed coupled-graph edge table without modifying A0."""
    a0_rows = a0.copy()
    a0_rows["edge_layer"] = "A0_internal_trajectory"
    a0_rows["relation_type"] = ""
    a0_rows["source_root"] = a0_rows["root_id"]
    a0_rows["receiver_root"] = a0_rows["root_id"]
    a0_rows["direction"] = ""
    a0_rows["formal_label"] = ""
    a0_rows["relation_operation_class"] = ""
    a0_rows["weighted_relation_score"] = np.nan
    a0_rows["confidence_level"] = ""
    a0_rows["qvalue"] = np.nan
    a0_rows["release_mode"] = mode
    a0_rows["paper_canonical"] = mode == "corrected_paper"

    relation_rows = []
    for relation in relations.itertuples(index=False):
        relation_rows.append(
            {
                "dataset": relation.dataset,
                "root_id": "",
                "candidate_tree_id": "",
                "source": relation.source_root,
                "target": relation.receiver_root,
                "base_edge_cost": np.nan,
                "edge_generation_recurrence": np.nan,
                "edge_layer": "supported_typed_cross_tree_relation",
                "relation_type": relation.relation_type,
                "source_root": relation.source_root,
                "receiver_root": relation.receiver_root,
                "direction": getattr(relation, "direction", ""),
                "formal_label": relation.formal_label,
                "relation_operation_class": (
                    "typed_relation_edge_plus_mode_specific_operational_prior"
                    if relation.relation_type != "collision_like"
                    else "typed_relation_edge_only"
                ),
                "weighted_relation_score": relation.weighted_relation_score,
                "confidence_level": relation.confidence_level,
                "qvalue": relation.qvalue,
                "release_mode": mode,
                "paper_canonical": mode == "corrected_paper",
            }
        )
    relation_frame = pd.DataFrame(relation_rows)
    columns = [
        "dataset", "edge_layer", "root_id", "candidate_tree_id", "source",
        "target", "base_edge_cost", "edge_generation_recurrence",
        "relation_type", "source_root", "receiver_root", "direction",
        "formal_label", "relation_operation_class", "weighted_relation_score",
        "confidence_level", "qvalue", "release_mode", "paper_canonical",
    ]
    return pd.concat(
        [a0_rows.reindex(columns=columns), relation_frame.reindex(columns=columns)],
        ignore_index=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    project = args.project_root.resolve()
    audit = project / "outputs" / "v7_8_1_no_response_label_sensitivity_audit"
    modes = [
        "as_run_firstargmax", "corrected_primary_support",
        "primary_support_only", "reversed_firstargmax",
    ]
    tables = {mode: read_mode(project, audit, mode) for mode in modes}
    detailed = pd.DataFrame(
        [row for mode in modes for row in detailed_rows(mode, tables[mode])]
    )
    detailed.to_csv(audit / "RELATION_LABEL_MODE_DETAILED_COMPARISON.csv", index=False)

    supported_keys = {
        mode: set(key(tables[mode]["supported"])) for mode in modes
    }
    robust_keys = set.intersection(*supported_keys.values())
    robust_rows = []
    for relation_key in sorted(robust_keys):
        dataset, relation_type, source, receiver = relation_key.split("|")
        local = detailed.loc[
            detailed.dataset.eq(dataset)
            & detailed.relation_type.eq(relation_type)
            & detailed.source_root.eq(source)
            & detailed.receiver_root.eq(receiver)
            & detailed.network_supported
        ]
        programs = sorted(
            set(x for x in local.exploratory_program_annotation.astype(str) if x and x != "nan")
        )
        robust_rows.append(
            {
                "dataset": dataset,
                "relation_type": relation_type,
                "source_root": source,
                "receiver_root": receiver,
                "supported_in_all_label_modes": True,
                "relation_type_stable": True,
                "exploratory_program_stable": len(programs) == 1,
                "exploratory_programs_across_modes": "|".join(programs),
                "paper_label": "sensitivity-robust generic association",
                "causal_claim_allowed": False,
                "formal_mechanism_label_allowed": False,
            }
        )
    robust = pd.DataFrame(robust_rows)
    robust.to_csv(audit / "SENSITIVITY_ROBUST_RELATION_CORE.csv", index=False)

    # Publish unambiguous mode-specific artifacts.  The corrected mode is the
    # canonical paper relation layer because it excludes no-response and
    # ambiguous labels; its five supported keys equal the all-mode intersection.
    frozen = project / "outputs" / "v7_8_1_exact_mapping_generic_relation_layer_FROZEN"
    a0 = pd.read_csv(frozen / "A0_TRAJECTORY_EDGES_READONLY.csv")
    as_run_edges = tables["as_run_firstargmax"]["supported"].copy()
    corrected_edges = tables["corrected_primary_support"]["supported"].copy()
    reversed_edges = tables["reversed_firstargmax"]["supported"].copy()
    robust.to_csv(audit / "PAPER_ROBUST_RELATION_KEYS.csv", index=False)
    as_run_edges.to_csv(audit / "AS_RUN_SUPPORTED_RELATION_EDGES.csv", index=False)
    corrected_edges.to_csv(audit / "CORRECTED_SUPPORTED_RELATION_EDGES.csv", index=False)
    reversed_edges.to_csv(audit / "REVERSED_SUPPORTED_RELATION_EDGES.csv", index=False)
    tables["as_run_firstargmax"]["prior"].to_csv(
        audit / "AS_RUN_OPERATIONAL_BRANCH_PRIORS.csv", index=False
    )
    tables["corrected_primary_support"]["prior"].to_csv(
        audit / "CORRECTED_OPERATIONAL_BRANCH_PRIORS.csv", index=False
    )
    tables["reversed_firstargmax"]["prior"].to_csv(
        audit / "REVERSED_OPERATIONAL_BRANCH_PRIORS.csv", index=False
    )
    shutil.copy2(
        frozen / "FINAL_COUPLED_GRAPH_EDGES.csv",
        audit / "AS_RUN_COUPLED_GRAPH_EDGES.csv",
    )
    corrected_coupled = coupled_graph(a0, corrected_edges, "corrected_paper")
    corrected_coupled.to_csv(audit / "CORRECTED_COUPLED_GRAPH_EDGES.csv", index=False)
    corrected_edges.to_csv(audit / "PAPER_CORRECTED_RELATION_EDGES.csv", index=False)
    tables["corrected_primary_support"]["prior"].to_csv(
        audit / "PAPER_CORRECTED_OPERATIONAL_BRANCH_PRIORS.csv", index=False
    )
    corrected_coupled.to_csv(audit / "PAPER_COUPLED_GRAPH_EDGES.csv", index=False)
    paper_decision = {
        "paper_graph": "G_paper = G_corrected",
        "paper_relation_label_mode": "corrected_primary_support",
        "paper_A0_internal_edges": int(len(a0)),
        "paper_relation_edges": int(len(corrected_edges)),
        "paper_coupled_graph_edge_rows": int(len(corrected_coupled)),
        "paper_relation_keys_equal_all_mode_intersection": bool(
            set(key(corrected_edges)) == robust_keys
        ),
        "paper_operational_priors_are_mode_specific": True,
        "operational_prior_parameter_robustness_claim_allowed": False,
        "historical_as_run_graph_edge_rows": int(len(a0) + len(as_run_edges)),
        "reverse_stress_test_relation_edges": int(len(reversed_edges)),
        "A0_modified": False,
    }
    (audit / "PAPER_METHOD_DECISION.json").write_text(
        json.dumps(paper_decision, indent=2), encoding="utf-8"
    )

    original = supported_keys["as_run_firstargmax"]
    corrected = supported_keys["corrected_primary_support"]
    reversed_set = supported_keys["reversed_firstargmax"]
    report = [
        "# v7.8.1 no-response and root-column-order sensitivity audit",
        "",
        "## Decision",
        "",
        "`V781_RELATION_LAYER_REMAINS_AS_RUN_CONDITIONAL`",
        "",
        "The A0 trajectory layer remains frozen and unchanged (45,034 internal edges). "
        "The seven-relation v7.8.1 as-run release is not eligible for an unconditional paper claim because relation support changes under prespecified no-response and root-column-order analyses.",
        "",
        "## Prespecified analyses",
        "",
        "- As-run: deterministic first-argmax in the locked root-column order.",
        "- Corrected/primary-support: exclude all-zero no-response cells and any non-zero ties from root-exclusive contact and exposure analyses.",
        "- Reversed first-argmax: retain the historical rule but reverse tie priority to test arbitrary root-column-order dependence.",
        "- Corrected-label and primary-support-only are exactly equivalent here because no non-zero ties were observed.",
        "",
        "## Relation support",
        "",
        f"- As-run supported: {len(original)} relations (1 collision-like, 6 promotion-like).",
        f"- Corrected/primary-support supported: {len(corrected)} relations (0 collision-like, 5 promotion-like).",
        f"- Reversed first-argmax supported: {len(reversed_set)} relations (0 collision-like, 8 promotion-like, 1 inhibition-like).",
        f"- Supported in every tested label mode: {len(robust_keys)} generic relations.",
        "",
        "## Sensitivity-robust generic core",
        "",
        robust.to_markdown(index=False),
        "",
        "The five retained relation keys are structurally robust at the generic source-root/receiver-root/type level. "
        "However, the selected exploratory receiver program changed across label modes for some keys; therefore program-specific biological interpretations must be reported per mode and cannot be promoted to formal mechanisms.",
        "",
        "## Relations invalidated as unconditional v7.8.1 findings",
        "",
        "- GSE C47-C29 collision-like: absent after excluding no-response cells (family q=0.1556) and absent after reversed tie priority (family q=0.6114; signed z changed from positive to negative).",
        "- Melanoma C15->C03 promotion-like: passed minimum validation after correction but failed the reciprocal conditional test (p=0.4752); it reappeared under reversed first-argmax, demonstrating order sensitivity.",
        "- Three additional GSE245582 relations appeared only under reversed first-argmax, further demonstrating that the historical seven-edge set is not root-column-order invariant.",
        "",
        "## Paper-use boundary",
        "",
        "The paper may report the seven v7.8.1 edges only as an explicitly conditional as-run historical analysis. "
        "For a robustness-focused main result, the defensible descriptive subset is the five sensitivity-robust generic associations, with no collision-like or inhibition-like relation and no formal mechanism label. "
        "This audit does not alter A0 roots, memberships, branches, pseudotime, directions, components, or internal edges.",
        "",
        "## Canonical paper graph",
        "",
        "The canonical paper relation layer is the corrected/primary-support result, whose five supported keys exactly equal the all-mode intersection. The paper coupled graph therefore contains 45,034 A0 edges plus 5 corrected relation edges (45,039 rows). Historical as-run and reversed-priority graphs remain separate audit objects. Operational priors are reported from the corrected mode and are not claimed to be parameter-level robust across label modes.",
    ]
    (audit / "PAPER_RELATION_SENSITIVITY_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    decision_path = audit / "FINAL_SENSITIVITY_DECISION.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision.update(
        {
            "sensitivity_robust_generic_relations": len(robust_keys),
            "sensitivity_robust_collision_like": int(
                robust.relation_type.eq("collision_like").sum()
            ),
            "sensitivity_robust_promotion_like": int(
                robust.relation_type.eq("promotion_like").sum()
            ),
            "sensitivity_robust_inhibition_like": int(
                robust.relation_type.eq("inhibition_like").sum()
            ),
            "v781_as_run_seven_edge_release_unconditional_claim_deprecated": True,
        }
    )
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    manifest = []
    for path in sorted(audit.rglob("*"), key=str):
        if path.is_file() and path.name != "OUTPUT_MANIFEST_SHA256.csv":
            manifest.append(
                {"path": str(path.relative_to(audit)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    pd.DataFrame(manifest).to_csv(audit / "OUTPUT_MANIFEST_SHA256.csv", index=False)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
