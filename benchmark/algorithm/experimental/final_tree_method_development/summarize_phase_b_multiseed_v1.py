#!/usr/bin/env python3
"""Truth audit for frozen multichannel PHASE B development runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


POSITIVE_TYPES = {"promotion", "collision_convergence", "positive_association"}


def as_bool(value) -> bool:
    return str(value).lower() in {"true", "1"}


def f1(predicted: set[str], truth: set[str]) -> float:
    if not predicted and not truth:
        return 1.0
    if not predicted or not truth:
        return 0.0
    overlap = len(predicted & truth)
    precision = overlap / len(predicted)
    recall = overlap / len(truth)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def root_alignment(run: Path, adata: ad.AnnData, assignments: pd.DataFrame):
    mapping = pd.read_csv(run / "formal_metacell" / "cell_to_metacell_mapping.csv")
    cell_to_mc = mapping.set_index("cell_name").metacell_id.astype(str).to_dict()
    primary = assignments.sort_values("membership", ascending=False).drop_duplicates("unit_id")
    mc_to_root = primary.set_index(primary.unit_id.astype(str)).root_id.astype(str).to_dict()
    inferred = np.asarray([
        mc_to_root.get(cell_to_mc.get(str(cell), ""), "none") for cell in adata.obs_names
    ])
    truth = adata.obs.root_id.astype(str).to_numpy()
    pred_roots = sorted(set(inferred) - {"none"})
    truth_roots = sorted(set(truth))
    matrix = np.asarray([
        [int(np.sum((inferred == pred) & (truth == target))) for target in truth_roots]
        for pred in pred_roots
    ])
    row, column = linear_sum_assignment(-matrix)
    alignment = {pred_roots[i]: truth_roots[j] for i, j in zip(row, column)}
    return cell_to_mc, primary, inferred, alignment


def branch_cell_set(
    adata: ad.AnnData, cell_to_mc: dict[str, str], primary: pd.DataFrame,
    inferred: np.ndarray, pred_root: str, pred_branch: str,
    window: tuple[float, float] | None = None,
) -> set[str]:
    state = primary.set_index(primary.unit_id.astype(str))
    result = set()
    for index in np.flatnonzero(inferred == pred_root):
        cell = str(adata.obs_names[index])
        mc = cell_to_mc.get(cell)
        if (
            mc in state.index
            and str(state.loc[mc, "branch_id"]) == pred_branch
            and (
                window is None
                or window[0] <= float(state.loc[mc, "pseudotime"]) <= window[1]
            )
        ):
            result.add(cell)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relation-dir", default="phase_b_multichannel_v3_frozen_dev")
    parser.add_argument("--development-replicates", default="1,2")
    parser.add_argument("--confirmation-replicates", default="3,4")
    args = parser.parse_args()
    development_replicates = {
        int(value) for value in args.development_replicates.split(",") if value
    }
    confirmation_replicates = {
        int(value) for value in args.confirmation_replicates.split(",") if value
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    event_rows = []
    evidence_frames = []
    score_rows = []
    ablation_rows = []
    null_rows = []

    runs = sorted(args.root.glob("core/*/replicate_*_seed_*"))
    for run in runs:
        relation = run / args.relation_dir
        if not (relation / "RUN_STATUS.json").exists():
            continue
        setting = run.parent.name
        replicate = int(run.name.split("_")[1])
        if replicate not in development_replicates | confirmation_replicates:
            continue
        split = (
            "DEV" if replicate in development_replicates
            else "HELD_OUT_CONFIRMATION"
        )
        truth = pd.read_csv(run / "event" / "truth" / "TRUE_EVENTS.csv").iloc[0]
        causal = as_bool(truth.causal)
        expected_coarse = "inhibition" if str(truth.event_type) == "inhibition" else "positive"
        evidence = pd.read_csv(relation / "PHASE_B_CMI_E_EVIDENCE.csv")
        evidence.insert(0, "setting", setting)
        evidence.insert(1, "replicate", replicate)
        evidence.insert(2, "evaluation_split", split)
        evidence_frames.append(evidence)

        adata = ad.read_h5ad(run / "event" / "synthetic.h5ad", backed="r")
        assignments = pd.read_csv(run / "phase_a_v4_frozen" / "T0_STATE_NODE_ASSIGNMENTS.csv")
        cell_to_mc, primary, inferred, alignment = root_alignment(run, adata, assignments)
        inverse_alignment = {truth_root: pred_root for pred_root, truth_root in alignment.items()}
        expected_source_pred = inverse_alignment.get(str(truth.source_root))
        expected_receiver_pred = inverse_alignment.get(str(truth.receiver_root))
        true_pair = evidence[
            evidence.source_root.astype(str).eq(str(expected_source_pred))
            & evidence.receiver_root.astype(str).eq(str(expected_receiver_pred))
        ]
        true_evidence = true_pair.iloc[0] if len(true_pair) else None
        called = evidence[evidence.is_called.map(as_bool)].copy()
        correct_calls = called[
            called.source_root.astype(str).map(alignment).eq(str(truth.source_root))
            & called.receiver_root.astype(str).map(alignment).eq(str(truth.receiver_root))
        ]
        event_tp = int(causal and len(correct_calls) > 0)
        event_fp = int(len(called) - event_tp)
        event_fn = int(causal and not event_tp)
        selected = correct_calls.iloc[0] if len(correct_calls) else None
        source_accuracy = float(
            selected is not None
            and alignment.get(str(selected.source_root)) == str(truth.source_root)
        )
        receiver_accuracy = float(
            selected is not None
            and alignment.get(str(selected.receiver_root)) == str(truth.receiver_root)
        )
        fine_type_accuracy = float(
            selected is not None and str(selected.event_type) == str(truth.event_type)
        )
        predicted_coarse = None
        if selected is not None:
            predicted_coarse = (
                "inhibition" if str(selected.event_type) == "inhibition" else "positive"
            )
        coarse_type_accuracy = float(
            selected is not None and predicted_coarse == expected_coarse
        )
        branch_f1 = 0.0
        local_branch_f1 = 0.0
        branch_choice_accuracy = 0.0
        active_response_recall = 0.0
        affected_event_region_f1 = 0.0
        affected_event_region_precision = 0.0
        affected_event_region_recall = 0.0
        location_error = 1.0
        mapped_branch = "abstain"
        if selected is not None:
            predicted_cells = branch_cell_set(
                adata, cell_to_mc, primary, inferred,
                str(selected.receiver_root), str(selected.affected_branch),
            )
            truth_mask = (
                adata.obs.root_id.astype(str).eq(str(truth.receiver_root)).to_numpy()
                & adata.obs.truth_branch_T0.astype(str).eq(str(truth.affected_branch)).to_numpy()
            )
            truth_cells = set(adata.obs_names[truth_mask].astype(str))
            branch_f1 = f1(predicted_cells, truth_cells)
            predicted_window = (
                float(selected.receiver_window_start),
                float(selected.receiver_window_end),
            )
            predicted_local_cells = branch_cell_set(
                adata, cell_to_mc, primary, inferred,
                str(selected.receiver_root), str(selected.affected_branch),
                predicted_window,
            )
            truth_local_mask = truth_mask & adata.obs.truth_pseudotime.between(
                float(truth.receiver_tau_start), float(truth.receiver_tau_end),
                inclusive="both",
            ).to_numpy()
            truth_local_cells = set(adata.obs_names[truth_local_mask].astype(str))
            local_branch_f1 = f1(predicted_local_cells, truth_local_cells)
            if predicted_local_cells:
                mapped_branch = adata.obs.loc[
                    list(predicted_local_cells), "truth_branch_T0"
                ].astype(str).value_counts().index[0]
                branch_choice_accuracy = float(
                    mapped_branch == str(truth.affected_branch)
                )
            event_labels = adata.obs.causal_event_id.astype(str)
            active_cells = set(adata.obs_names[
                event_labels.str.contains(str(truth.event_id), regex=False).to_numpy()
            ].astype(str))
            localized_value = str(selected.get("localized_support_cell_ids", ""))
            localized_cells = {
                cell for cell in localized_value.split(";")
                if cell and cell.lower() != "nan"
            }
            if localized_cells:
                mapped_branch = adata.obs.loc[
                    list(localized_cells), "truth_branch_T0"
                ].astype(str).value_counts().index[0]
                branch_choice_accuracy = float(
                    mapped_branch == str(truth.affected_branch)
                )
            if active_cells:
                active_response_recall = len(
                    localized_cells & active_cells
                ) / len(active_cells)
                affected_event_region_recall = active_response_recall
            if localized_cells:
                affected_event_region_precision = len(
                    localized_cells & active_cells
                ) / len(localized_cells)
            affected_event_region_f1 = f1(localized_cells, active_cells)
            predicted_center = 0.5 * (
                float(selected.receiver_window_start) + float(selected.receiver_window_end)
            )
            truth_center = 0.5 * (
                float(truth.receiver_tau_start) + float(truth.receiver_tau_end)
            )
            location_error = abs(predicted_center - truth_center)

        event_rows.append({
            "setting": setting, "replicate": replicate, "evaluation_split": split,
            "truth_event_type": truth.event_type, "truth_effect_strength": truth.effect_strength,
            "causal": causal, "n_called": len(called), "event_tp": event_tp,
            "event_fp": event_fp, "event_fn": event_fn,
            "source_root_accuracy": source_accuracy,
            "receiver_root_accuracy": receiver_accuracy,
            "coarse_relation_type_accuracy": coarse_type_accuracy,
            "fine_relation_type_accuracy": fine_type_accuracy,
            "affected_branch_cell_f1": branch_f1,
            "event_local_affected_branch_f1": local_branch_f1,
            "affected_branch_choice_accuracy": branch_choice_accuracy,
            "active_response_cell_recall": active_response_recall,
            "affected_event_region_precision": affected_event_region_precision,
            "affected_event_region_recall": affected_event_region_recall,
            "affected_event_region_f1": affected_event_region_f1,
            "mapped_truth_branch": mapped_branch,
            "truth_affected_branch": truth.affected_branch,
            "event_location_error": location_error,
            "abstained": int(len(called) == 0),
        })
        if true_evidence is not None:
            score_rows.append({
                "setting": setting, "replicate": replicate, "evaluation_split": split,
                "truth_event_type": truth.event_type,
                "effect_strength": float(truth.effect_strength), "causal": causal,
                "raw_dependency": float(true_evidence.raw_dependency),
                "environment_adjusted_dependency": float(
                    true_evidence.environment_adjusted_dependency
                ),
                "environment_drop": float(true_evidence.environment_drop),
                "environment_trust": float(true_evidence.environment_trust),
                "environment_conflict": float(true_evidence.environment_conflict),
                "raw_rho": float(true_evidence.raw_rho),
                "adjusted_rho": float(true_evidence.adjusted_rho),
                "joint_dependency_statistic": float(
                    true_evidence.joint_dependency_statistic
                ),
                "omnibus_statistic": float(true_evidence.omnibus_dependency_statistic),
                "primary_channel_q": float(true_evidence.primary_channel_q),
                "signature_channel_q": float(true_evidence.signature_channel_q),
                "called": as_bool(true_evidence.is_called),
            })
        primary = pd.read_csv(relation / "channel_primary" / "PHASE_B_CMI_E_EVIDENCE.csv")
        raw_calls = primary[
            primary.raw_is_called_without_cmi_e.map(as_bool)
            & primary.source_hypothesis_eligible.map(as_bool)
        ]
        ablation_rows.append({
            "setting": setting, "replicate": replicate, "evaluation_split": split,
            "causal": causal, "full_called": int(len(called) > 0),
            "raw_only_called": int(len(raw_calls) > 0),
            "full_false_call": int((not causal) and len(called) > 0),
            "raw_only_false_call": int((not causal) and len(raw_calls) > 0),
        })
        if not causal:
            null_rows.append({
                "setting": setting, "replicate": replicate, "evaluation_split": split,
                "null_type": truth.event_type, "n_called": len(called),
                "false_call": int(len(called) > 0),
                "raw_only_false_call": int(len(raw_calls) > 0),
                "upstream_relation_inference_only": True,
            })
        adata.file.close()

    events = pd.DataFrame(event_rows)
    scores = pd.DataFrame(score_rows)
    ablation = pd.DataFrame(ablation_rows)
    nulls = pd.DataFrame(null_rows)
    all_evidence = pd.concat(evidence_frames, ignore_index=True)
    all_evidence.to_csv(output / "PHASE_B_CMI_E_EVIDENCE.csv", index=False)
    scores.to_csv(output / "PHASE_B_RELATION_SCORE_DISTRIBUTION.csv", index=False)
    events.to_csv(output / "PHASE_B_EVENT_DETECTION_METRICS.csv", index=False)
    ablation.to_csv(output / "PHASE_B_CMI_E_ABLATION.csv", index=False)
    nulls.to_csv(output / "PHASE_B_NULL_CONTROL.csv", index=False)

    summary = {}
    for split, frame in events.groupby("evaluation_split"):
        tp, fp, fn = frame.event_tp.sum(), frame.event_fp.sum(), frame.event_fn.sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        summary[split] = {
            "n_runs": len(frame), "event_precision": precision, "event_recall": recall,
            "event_f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "null_false_call_rate": float(frame.loc[~frame.causal, "event_fp"].gt(0).mean()),
            "coarse_type_accuracy_called_true": float(
                frame.loc[frame.event_tp.eq(1), "coarse_relation_type_accuracy"].mean()
            ),
            "fine_type_accuracy_called_true": float(
                frame.loc[frame.event_tp.eq(1), "fine_relation_type_accuracy"].mean()
            ),
            "affected_branch_f1_mean_causal": float(
                frame.loc[frame.causal, "affected_branch_cell_f1"].mean()
            ),
            "event_local_branch_f1_mean_called_true": float(
                frame.loc[frame.event_tp.eq(1), "event_local_affected_branch_f1"].mean()
            ),
            "affected_branch_choice_accuracy_called_true": float(
                frame.loc[frame.event_tp.eq(1), "affected_branch_choice_accuracy"].mean()
            ),
            "active_response_cell_recall_called_true": float(
                frame.loc[frame.event_tp.eq(1), "active_response_cell_recall"].mean()
            ),
            "affected_event_region_f1_mean_called_true": float(
                frame.loc[frame.event_tp.eq(1), "affected_event_region_f1"].mean()
            ),
            "event_location_error_mean_causal": float(
                frame.loc[frame.causal, "event_location_error"].mean()
            ),
            "event_location_error_mean_called_true": float(
                frame.loc[frame.event_tp.eq(1), "event_location_error"].mean()
            ),
        }
    strength_response = {}
    for split, frame in scores.groupby("evaluation_split"):
        null_values = frame.loc[~frame.causal, "joint_dependency_statistic"]
        weak_values = frame.loc[frame.setting.eq("CORE_01"), "joint_dependency_statistic"]
        medium_values = frame.loc[frame.setting.eq("CORE_02"), "joint_dependency_statistic"]
        strong_values = frame.loc[frame.setting.eq("CORE_03"), "joint_dependency_statistic"]
        medians = {
            "null": float(null_values.median()),
            "weak": float(weak_values.median()),
            "medium": float(medium_values.median()),
            "strong": float(strong_values.median()),
        }
        medians["strictly_ordered"] = bool(
            medians["null"] < medians["weak"]
            < medians["medium"] < medians["strong"]
        )
        strength_response[split] = medians
    cmi_e_ablation = {}
    for split, frame in ablation.groupby("evaluation_split"):
        causal_frame = frame[frame.causal]
        null_frame = frame[~frame.causal]
        cmi_e_ablation[split] = {
            "full_true_event_call_rate": float(causal_frame.full_called.mean()),
            "minus_cmi_e_true_event_call_rate": float(causal_frame.raw_only_called.mean()),
            "full_null_false_call_rate": float(null_frame.full_false_call.mean()),
            "minus_cmi_e_null_false_call_rate": float(
                null_frame.raw_only_false_call.mean()
            ),
        }
    held = summary.get("HELD_OUT_CONFIRMATION", {})
    held_strength = strength_response.get("HELD_OUT_CONFIRMATION", {})
    held_ablation = cmi_e_ablation.get("HELD_OUT_CONFIRMATION", {})
    core_pass = bool(
        held.get("null_false_call_rate", 1.0) <= 0.10
        and held.get("event_precision", 0.0) >= 0.70
        and held.get("event_recall", 0.0) >= 0.50
        and held.get("coarse_type_accuracy_called_true", 0.0) >= 0.80
        and held.get("affected_branch_choice_accuracy_called_true", 0.0) >= 0.70
        and held.get("affected_event_region_f1_mean_called_true", 0.0) >= 0.40
        and held.get("event_location_error_mean_called_true", 1.0) <= 0.30
        and held_strength.get("strictly_ordered", False)
        and held_ablation.get("full_null_false_call_rate", 1.0)
        < held_ablation.get("minus_cmi_e_null_false_call_rate", 0.0)
        and held_ablation.get("full_true_event_call_rate", 0.0)
        >= held_ablation.get("minus_cmi_e_true_event_call_rate", 1.0) - 0.15
    )
    decision = {
        "phase": "B", "decision": "PASS" if core_pass else "FAIL",
        "relation_type_gate": "coarse positive_vs_inhibition",
        "fine_collision_label_core_claim": False,
        "summary": summary,
        "effect_strength_response": strength_response,
        "cmi_e_ablation": cmi_e_ablation,
        "locked_replicates_21_30_touched": False,
        "reason": (
            "Held-out relation power, precision, coarse direction/type, and null safety passed."
            if core_pass else
            "At least one held-out relation power, precision, coarse type, or null-safety gate failed."
        ),
    }
    (output / "PHASE_B_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    failures = events[(events.causal & events.event_tp.eq(0)) | events.event_fp.gt(0)]
    lines = [
        "# PHASE B Failure Attribution", "",
        "The detector was frozen before held-out confirmation replicates 3-4.",
        "Fine collision/convergence typing is not a core claim; positive versus inhibitory relation is the frozen type gate.",
        "",
        f"Failed or false-call runs: {len(failures)} / {len(events)}.", "",
    ]
    for row in failures.itertuples(index=False):
        lines.append(
            f"- {row.setting} replicate {row.replicate}: causal={row.causal}, "
            f"TP={row.event_tp}, FP={row.event_fp}, abstained={row.abstained}."
        )
    (output / "PHASE_B_FAILURE_ATTRIBUTION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
