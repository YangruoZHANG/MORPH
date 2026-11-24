#!/usr/bin/env python3
"""PHASE A v7 development constructor with state-constrained fine T0."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DIAGNOSTIC = load(
    "diagnose_phase_a_hierarchical_v10.py", "phase_a_v7_state_constrained_fine"
)
DIAGNOSTIC.METHOD_VERSION = "PHASE_A_v7_1_state_constrained_fine_all_eligible_attach"
DIAGNOSTIC.BRANCH_ESTIMATOR_LABEL = "stable_HVG_PCA_KMeans_v9_complexity_0.10"
DIAGNOSTIC.PRODUCTION_CASE = "hierarchical_branch_program_state_fine"


if __name__ == "__main__":
    DIAGNOSTIC.main()
