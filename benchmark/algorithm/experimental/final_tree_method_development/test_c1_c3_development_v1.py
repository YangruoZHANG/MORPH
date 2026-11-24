from __future__ import annotations

import importlib.util
import inspect
import unittest
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("c1_c3_v1_tested", HERE / "run_c1_c3_development_v1.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C1_V2 = load_module("run_c1_v2_c3_v2.py", "c1_v2_tested")
V5_1 = load_module("run_c1_v2_c3_v5_1.py", "c3_v5_1_tested")


class CandidateAndT0Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.adata = ad.AnnData(rng.poisson(2.0, size=(18, 12)).astype(float))
        self.adata.obs_names = [f"MC{i:04d}" for i in range(self.adata.n_obs)]
        self.adata.obsm["spatial"] = rng.normal(size=(self.adata.n_obs, 2))

    def test_truth_is_not_a_constructor_argument(self):
        for function in (MODULE.build_multiresolution_candidate, MODULE.construct_t0):
            names = set(inspect.signature(function).parameters)
            self.assertFalse(any("truth" in name.lower() for name in names))

    def test_edge_roles_are_exclusive(self):
        geometry = MODULE.prepare_geometry(self.adata)
        candidate, _ = MODULE.build_multiresolution_candidate(self.adata, geometry)
        allowed = {
            "trajectory_local_reciprocal", "trajectory_mesoscale_reciprocal",
            "trajectory_crossscale_mst_bridge", "interaction_only_broad_neighbour",
        }
        self.assertTrue(set(candidate.edge_graph_role).issubset(allowed))
        interaction = candidate.edge_graph_role.eq("interaction_only_broad_neighbour")
        self.assertTrue((~candidate.loc[interaction, "trajectory_edge_eligible"]).all())

    def test_interaction_only_edges_cannot_enter_support_graph(self):
        geometry = MODULE.prepare_geometry(self.adata)
        candidate, _ = MODULE.build_multiresolution_candidate(self.adata, geometry)
        graph = MODULE.graph_for_support(candidate, set(range(self.adata.n_obs)), np.ones(self.adata.n_obs))
        legal = {
            MODULE.edge_key(int(row.source), int(row.target))
            for row in candidate[candidate.trajectory_edge_eligible].itertuples(index=False)
        }
        self.assertTrue(all(MODULE.edge_key(int(i), int(j)) in legal for i, j in graph.edges()))

    def test_quotient_constructor_returns_a_forest(self):
        graph = nx.Graph()
        graph.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (0, 2, 2), (2, 3, 1), (1, 3, 3)])
        for i, j in graph.edges():
            graph[i][j]["edge_graph_role"] = "trajectory_local_reciprocal"
        labels = {0: "a", 1: "a", 2: "b", 3: "b"}
        result = MODULE.quotient_branch_forest(graph, labels)
        self.assertTrue(nx.is_forest(result))
        self.assertEqual(result.number_of_edges(), result.number_of_nodes() - 1)

    def test_reserve_roles_are_inactive_in_t0(self):
        geometry = MODULE.prepare_geometry(self.adata)
        candidate, _ = C1_V2.candidate_v2(MODULE, self.adata, geometry)
        active = candidate.edge_graph_role.astype(str).isin(V5_1.T0_ROLES)
        reserve = candidate.trajectory_edge_eligible & ~active
        self.assertTrue((candidate.loc[reserve, "edge_graph_role"] == "trajectory_local_union").all())
        self.assertFalse((active & reserve).any())

    def test_active_edge_ranks_do_not_depend_on_reserve_edges(self):
        geometry = MODULE.prepare_geometry(self.adata)
        candidate, _ = C1_V2.candidate_v2(MODULE, self.adata, geometry)
        active = candidate[candidate.edge_graph_role.astype(str).isin(V5_1.T0_ROLES)].copy()
        reference = MODULE.empirical_rank(active["D_joint"].to_numpy(float))
        augmented = pd.concat([
            active,
            candidate[candidate.edge_graph_role.eq("trajectory_local_union")],
        ], ignore_index=True)
        recomputed = MODULE.empirical_rank(
            augmented[augmented.edge_graph_role.astype(str).isin(V5_1.T0_ROLES)]["D_joint"].to_numpy(float)
        )
        np.testing.assert_allclose(reference, recomputed)


if __name__ == "__main__":
    unittest.main()
