#!/usr/bin/env python3
"""PHASE A v5 hierarchical branch-state constructor and audit runner."""

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
    "diagnose_phase_a_hierarchical_v10.py", "phase_a_v5_hierarchical_constructor"
)
DIAGNOSTIC.METHOD_VERSION = "PHASE_A_v5_3_guarded_hierarchical_graph_partition_state"
DIAGNOSTIC.BRANCH_ESTIMATOR_LABEL = "stable_HVG_PCA_KMeans_v9_complexity_0.10"
DIAGNOSTIC.PRODUCTION_CASE = "hierarchical_graph_partition_state"


if __name__ == "__main__":
    DIAGNOSTIC.main()
