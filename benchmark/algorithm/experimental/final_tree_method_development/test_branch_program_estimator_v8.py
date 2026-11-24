from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("branch_v8_tested", HERE / "branch_program_estimator_v8.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["branch_v8_tested"] = MODULE
SPEC.loader.exec_module(MODULE)


class StableBranchProgramTests(unittest.TestCase):
    def test_deterministic_and_recovers_explicit_programs(self):
        rng = np.random.default_rng(7)
        counts = rng.poisson(2, size=(30, 120)).astype(float)
        counts[:10, :20] += 8
        counts[10:20, 20:40] += 8
        counts[20:, 40:60] += 8
        first, audit = MODULE.stable_hvg_kmeans(counts, list(range(30)))
        second, _ = MODULE.stable_hvg_kmeans(counts, list(range(30)))
        self.assertEqual(first, second)
        self.assertEqual(audit[0]["selected_k"], 3)
        self.assertEqual(len({first[index] for index in range(10)}), 1)
        self.assertEqual(len({first[index] for index in range(10, 20)}), 1)
        self.assertEqual(len({first[index] for index in range(20, 30)}), 1)

    def test_small_support_abstains_to_one_program(self):
        labels, audit = MODULE.stable_hvg_kmeans(np.ones((5, 20)), list(range(5)))
        self.assertEqual(set(labels.values()), {"b0"})
        self.assertEqual(audit[0]["selection"], "small_root_support")


if __name__ == "__main__":
    unittest.main()
