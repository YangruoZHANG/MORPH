#!/usr/bin/env python3
"""Merge two predeclared contact scales into stable Phase B evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "PHASE_B_V7_1_MULTISCALE_LOCALIZED_STABILITY_SELECTION"
MIN_LOCALIZED_BRANCH_SUPPORT_FRACTION = 0.15


def as_bool(value: object) -> bool:
    return str(value).lower() in {"true", "1"}


def collision_supported(row: pd.Series, suffix: str) -> bool:
    def value(name: str) -> object:
        return row[f"{name}_{suffix}"]

    return bool(
        float(value("signature_channel_q")) <= 0.15
        and float(value("signature_joint_statistic")) >= 0.50
        and float(value("signature_support_raw_rho")) >= 0.15
        and float(value("signature_support_raw_specificity_margin")) > 0.01
        and float(value("signature_support_raw_specificity_rank")) >= 0.80
        and as_bool(value("signature_environment_direction_consistent"))
        and float(value("signature_environment_trust")) >= 0.30
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale065-dir", type=Path, required=True)
    parser.add_argument("--scale080-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    low = pd.read_csv(args.scale065_dir / "PHASE_B_CMI_E_EVIDENCE.csv")
    high = pd.read_csv(args.scale080_dir / "PHASE_B_CMI_E_EVIDENCE.csv")
    keys = ["source_root", "receiver_root"]
    merged = low.merge(high, on=keys, how="outer", suffixes=("_65", "_80"))
    rows: list[dict[str, object]] = []
    for _, pair in merged.iterrows():
        called65 = as_bool(pair.get("is_called_65", False))
        called80 = as_bool(pair.get("is_called_80", False))
        sign65 = np.sign(float(pair.get("raw_monotone_statistic_65", 0.0)))
        sign80 = np.sign(float(pair.get("raw_monotone_statistic_80", 0.0)))
        direction_stable = bool(sign65 != 0 and sign65 == sign80)
        branch65 = str(pair.get("affected_branch_65", "none"))
        branch80 = str(pair.get("affected_branch_80", "none"))
        branch_stable = bool(branch65 == branch80 and branch65 != "none")
        evidence_stable = bool(called65 and called80 and direction_stable)
        q65 = float(pair.get("primary_channel_q_65", 1.0))
        q80 = float(pair.get("primary_channel_q_80", 1.0))
        representative_suffix = "65" if q65 <= q80 else "80"
        representative = {
            column[:-3]: value
            for column, value in pair.items()
            if column.endswith(f"_{representative_suffix}")
        }
        representative.update({key: pair[key] for key in keys})
        collision65 = collision_supported(pair, "65") if called65 else False
        collision80 = collision_supported(pair, "80") if called80 else False
        collision_stable = bool(collision65 and collision80)
        event_type = (
            "collision_convergence" if collision_stable
            else ("inhibition" if sign65 < 0 else "promotion")
        )
        if collision_stable:
            signature_branch65 = str(pair.get("signature_support_affected_branch_65", "none"))
            signature_branch80 = str(pair.get("signature_support_affected_branch_80", "none"))
            branch_stable = bool(
                signature_branch65 == signature_branch80
                and signature_branch65 != "none"
            )
            representative["affected_branch"] = (
                signature_branch65 if representative_suffix == "65"
                else signature_branch80
            )
            for target, source in (
                ("receiver_window_start", "signature_support_receiver_window_start"),
                ("receiver_window_end", "signature_support_receiver_window_end"),
                ("localized_support_cell_ids", "signature_support_localized_cell_ids"),
                ("localized_support_unit_ids", "signature_support_localized_unit_ids"),
                ("localized_support_n_cells", "signature_support_localized_n_cells"),
                ("localized_branch_support_n_cells", "signature_support_branch_n_cells"),
                ("localized_branch_support_fraction", "signature_support_branch_fraction"),
            ):
                representative[target] = pair.get(
                    f"{source}_{representative_suffix}", representative.get(target)
                )
            support65 = float(pair.get("signature_support_branch_fraction_65", 0.0))
            support80 = float(pair.get("signature_support_branch_fraction_80", 0.0))
        else:
            support65 = float(pair.get("localized_branch_support_fraction_65", 0.0))
            support80 = float(pair.get("localized_branch_support_fraction_80", 0.0))
        localization_supported = bool(
            min(support65, support80) >= MIN_LOCALIZED_BRANCH_SUPPORT_FRACTION
        )
        relation_stable = bool(evidence_stable and localization_supported)
        representative.update({
            "event_type": event_type,
            "is_called": relation_stable,
            "refinement_eligible": bool(relation_stable and branch_stable),
            "scale065_called": called65,
            "scale080_called": called80,
            "scale_direction_consistent": direction_stable,
            "scale_branch_consistent": branch_stable,
            "scale_collision_consistent": collision_stable,
            "scale065_localized_branch_support_fraction": support65,
            "scale080_localized_branch_support_fraction": support80,
            "multiscale_localized_branch_support_fraction": min(support65, support80),
            "localization_support_pass": localization_supported,
            "scale065_primary_q": q65,
            "scale080_primary_q": q80,
            "multiscale_primary_q": max(q65, q80),
            "primary_channel_q": max(q65, q80),
            "call_strength": min(
                float(pair.get("environment_adjusted_dependency_65", 0.0)),
                float(pair.get("environment_adjusted_dependency_80", 0.0)),
            ),
            "decision_reason": (
                "multiscale_relation_and_branch_stable"
                if relation_stable and branch_stable
                else (
                    "multiscale_relation_stable_branch_abstain"
                    if relation_stable else (
                        "multiscale_localization_support_abstain"
                        if evidence_stable and not localization_supported
                        else "multiscale_relation_abstain"
                    )
                )
            ),
        })
        rows.append(representative)

    evidence = pd.DataFrame(rows).sort_values(
        ["is_called", "multiscale_primary_q"], ascending=[False, True]
    )
    evidence.to_csv(args.output / "PHASE_B_CMI_E_EVIDENCE.csv", index=False)
    evidence.to_csv(args.output / "PHASE_B_RELATION_SCORE_DISTRIBUTION.csv", index=False)
    evidence[evidence.is_called.map(as_bool)].to_csv(
        args.output / "PHASE_B_CALLED_RELATIONS.csv", index=False
    )
    status = {
        "phase": "B",
        "detector_version": VERSION,
        "contact_scales": [0.65, 0.80],
        "relation_call_rule": (
            "called_at_both_scales_with_same_raw_direction_and_"
            "localized_branch_support_ge_0.15_at_both_scales"
        ),
        "minimum_localized_branch_support_fraction": (
            MIN_LOCALIZED_BRANCH_SUPPORT_FRACTION
        ),
        "refinement_gate": "relation_call_plus_same_affected_branch_at_both_scales",
        "collision_subtype_rule": "supported_at_both_scales_environment_trust_ge_0.30",
        "n_relations": int(evidence.is_called.map(as_bool).sum()),
        "n_refinement_eligible": int(evidence.refinement_eligible.map(as_bool).sum()),
        "truth_used": False,
        "relation_significance_before_T1": True,
        "complete": True,
    }
    (args.output / "RUN_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
