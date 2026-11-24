import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from morph_eval.metrics import (
    branch_f1,
    set_mapping_f1,
    maximal_segments,
    simplify_graph,
    topology_one_minus_him,
)
from morph_eval.evaluator import derived_fate_star


def test_branch_f1_is_label_invariant():
    truth = np.array(["a", "a", "b", "b", "c", "c"])
    pred = np.array(["x", "x", "z", "z", "y", "y"])
    score, recovery, relevance, coverage = branch_f1(truth, pred)
    assert score == recovery == relevance == coverage == 1.0


def test_branch_fragmentation_is_penalized_but_not_destroyed():
    truth = np.array(["a"] * 6 + ["b"] * 6)
    pred = np.array(["x"] * 3 + ["y"] * 3 + ["z"] * 6)
    score, _, _, _ = branch_f1(truth, pred)
    assert 0.6 < score < 1.0


def test_degree_two_simplification_preserves_length():
    graph = nx.Graph()
    graph.add_edge("a", "b", length=2.0)
    graph.add_edge("b", "c", length=3.0)
    simple = simplify_graph(graph)
    assert set(simple.nodes) == {"a", "c"}
    assert simple["a"]["c"]["length"] == 5.0


def test_him_identity_and_missing_edge():
    truth = nx.Graph([("r", "b"), ("b", "x"), ("b", "y")])
    assert topology_one_minus_him(truth, truth) > 0.999
    pred = nx.Graph([("r", "b"), ("b", "x")])
    score = topology_one_minus_him(truth, pred)
    assert 0.0 < score < 1.0


def test_cycle_has_no_tree_segments():
    assert maximal_segments(nx.cycle_graph(4), 0) == {}


def test_overlapping_lineage_mapping_accepts_shared_trunk():
    truth = [{0, 1, 2}, {0, 1, 3}]
    pred = [{0, 1, 3}, {0, 1, 2}]
    score, recovery, relevance, coverage = set_mapping_f1(truth, pred, 4)
    assert score == recovery == relevance == coverage == 1.0


def test_derived_fate_star_is_explicit_and_truth_free():
    import pandas as pd
    cells = pd.DataFrame({"native_branch": [None, "A", "A", "B", "B"]})
    graph, nodes = derived_fate_star(cells, np.array([0.0, 0.2, 0.8, 0.3, 0.9]))
    assert set(graph.edges) == {("__trunk__", "fate::A"), ("__trunk__", "fate::B")}
    assert nodes[0] == "__trunk__"

