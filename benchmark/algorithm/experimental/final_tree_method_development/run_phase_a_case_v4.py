#!/usr/bin/env python3
"""Frozen PHASE A v4 production constructor and audit runner."""

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


DIAGNOSTIC = load("diagnose_phase_a_hvg_v8.py", "phase_a_v4_frozen_constructor")
V9 = load("branch_program_estimator_v9.py", "phase_a_v4_branch_estimator_v9")


class FrozenBranchEstimator:
    @staticmethod
    def stable_hvg_kmeans(counts, nodes):
        return V9.stable_hvg_kmeans(counts, nodes, complexity_penalty=0.10)


DIAGNOSTIC.BRANCH = FrozenBranchEstimator
DIAGNOSTIC.METHOD_VERSION = "PHASE_A_v4_generator_v2_fine_hvg_v9_regularized"
DIAGNOSTIC.BRANCH_ESTIMATOR_LABEL = (
    "stable_HVG_PCA_KMeans_v9_complexity_0.10"
)
DIAGNOSTIC.PRODUCTION_CASE = "hvg_branch_regularized_020"


if __name__ == "__main__":
    DIAGNOSTIC.main()
