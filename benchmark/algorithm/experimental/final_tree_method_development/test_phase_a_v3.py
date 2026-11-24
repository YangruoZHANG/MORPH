from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import unittest
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PHASE_ROOT = REPO / "outputs" / "final_tree_method_development" / "phase_a_v3"


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V3 = load_module("run_phase_a_case_v3.py", "phase_a_v3_tested")
DUAL = load_module("diagnose_phase_a_dualchannel_v7.py", "phase_a_dualchannel_tested")


class PhaseAV3Tests(unittest.TestCase):
    def test_production_constructor_has_no_truth_argument(self):
        names = set(inspect.signature(DUAL.construct_variant).parameters)
        self.assertFalse(any("truth" in name.lower() for name in names))

    def test_eligible_existing_edge_beats_interaction_only_duplicate(self):
        full = pd.DataFrame([{
            "source": 0, "target": 1, "D_joint": 0.1,
            "edge_graph_role": "interaction_only_broad_neighbour",
            "trajectory_edge_eligible": False,
        }])
        existing = pd.DataFrame([{
            "source": 0, "target": 1, "D_joint": 0.2,
            "edge_graph_role": "trajectory_current_single_scale",
            "trajectory_edge_eligible": True,
        }])
        result = V3.full_candidate_union(full, existing)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].edge_graph_role, "trajectory_current_single_scale")
        self.assertTrue(bool(result.iloc[0].trajectory_edge_eligible))

    def test_frozen_outputs_keep_branch_and_state_fields_separate(self):
        paths = sorted((PHASE_ROOT / "development").glob("*/T0_STATE_NODE_ASSIGNMENTS.csv"))
        self.assertEqual(len(paths), 4)
        for path in paths:
            frame = pd.read_csv(path)
            self.assertIn("branch_id", frame)
            self.assertIn("state_id", frame)
            self.assertTrue((frame.branch_id.astype(str) != frame.state_id.astype(str)).all())

    def test_all_eight_runs_are_legal_forests_without_relation(self):
        statuses = []
        for split in ("development", "confirmation"):
            statuses.extend(
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((PHASE_ROOT / split).glob("*/RUN_STATUS.json"))
            )
        self.assertEqual(len(statuses), 8)
        for status in statuses:
            self.assertFalse(status["relation_detector_called"])
            self.assertFalse(status["refinement_operator_called"])
            self.assertFalse(status["truth_used_for_construction_D"])
            self.assertTrue(status["all_fine_edges_candidate_legal"])
            self.assertTrue(status["all_cases_are_forests"])

    def test_failed_gate_does_not_authorize_phase_b(self):
        path = PHASE_ROOT / "final_audit" / "PHASE_A_DECISION.json"
        decision = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(decision["PHASE_A_DECISION"], "FAIL")
        self.assertFalse(decision["phase_B_authorized"])
        self.assertFalse(decision["locked_final_replicates_21_30_touched"])


if __name__ == "__main__":
    unittest.main()

