#!/usr/bin/env python3
"""Materialize the selected post-failure Phase B v4 development rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from diagnose_phase_e_relation_gate_v4 import gate


VERSION = "BRANCH_LOCAL_MULTICHANNEL_RELATION_V4_1_DEV"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = pd.read_csv(args.input / "PHASE_B_CMI_E_EVIDENCE.csv")
    selected = gate(
        evidence,
        primary_p="omnibus",
        correction="MIXED",
        collision_trust=0.50,
        competition_scope="event_local",
        source_eligibility="max_gap",
    )
    selected["decision_reason"] = selected.is_called.map(
        {True: "passes_v4_1_branch_local_multichannel_gate", False: "abstain_v4_1_gate"}
    )
    selected.to_csv(args.output / "PHASE_B_CMI_E_EVIDENCE.csv", index=False)
    selected.to_csv(args.output / "PHASE_B_RELATION_SCORE_DISTRIBUTION.csv", index=False)
    selected[selected.is_called].to_csv(
        args.output / "PHASE_B_CALLED_RELATIONS.csv", index=False
    )
    status = {
        "phase": "B",
        "detector_version": VERSION,
        "evidence_generator": "BRANCH_LOCAL_MAXSTAT_RELATION_V2_9",
        "primary_significance": "omnibus_maxstat",
        "primary_multiplicity": "weighted_Holm",
        "signature_multiplicity": "weighted_BH",
        "q_cutoff": 0.10,
        "collision_environment_trust_min": 0.50,
        "competition_scope": "event_local",
        "source_eligibility": "max_activity_gap_0.10_and_positive_enrichment",
        "truth_used": False,
        "post_lock_failure_development": True,
        "n_called": int(selected.is_called.sum()),
        "complete": True,
    }
    (args.output / "RUN_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
