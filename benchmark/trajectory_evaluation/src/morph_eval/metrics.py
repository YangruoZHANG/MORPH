"""Metric primitives with no file-system or method-specific dependencies."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import brentq
from scipy.stats import rankdata, spearmanr


def safe_spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.ptp(x[ok]) <= 1e-12 or np.ptp(y[ok]) <= 1e-12:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).statistic)


def simplify_graph(graph: nx.Graph) -> nx.Graph:
    """Remove degree-two vertices while preserving path lengths."""
    g = nx.Graph()
    for u, v, d in graph.edges(data=True):
        w = float(d.get("length", d.get("weight", 1.0)))
        if u != v:
            g.add_edge(str(u), str(v), length=max(w, 1e-12))
    changed = True
    while changed and g.number_of_nodes() > 2:
        changed = False
        for v in list(g.nodes):
            if g.degree(v) != 2:
                continue
            a, b = list(g.neighbors(v))
            length = g[v][a]["length"] + g[v][b]["length"]
            g.remove_node(v)
            if a != b:
                old = g.get_edge_data(a, b, {}).get("length", np.inf)
                g.add_edge(a, b, length=min(old, length))
            changed = True
            break
    return g


def maximal_segments(graph: nx.Graph, root: str | None = None) -> dict[str, str]:
    """Assign graph nodes to maximal non-branching segments.

    Junction nodes are assigned deterministically to the incident segment nearest the
    supplied root. This changes only a measure-zero boundary in a continuous tree.
    """
    g = nx.Graph(graph)
    if not g or not nx.is_forest(g):
        return {}
    root = str(root) if root is not None else None
    specials = {v for v, d in g.degree if d != 2}
    if root in g:
        specials.add(root)
        root_dist = nx.single_source_shortest_path_length(g, root)
    else:
        root_dist = {}
    seen: set[frozenset] = set()
    memberships: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    seg_i = 0
    for start in sorted(specials, key=str):
        for neighbor in sorted(g.neighbors(start), key=str):
            if frozenset((start, neighbor)) in seen:
                continue
            path = [start, neighbor]
            seen.add(frozenset((start, neighbor)))
            prev, cur = start, neighbor
            while cur not in specials:
                nxt = next(x for x in g.neighbors(cur) if x != prev)
                seen.add(frozenset((cur, nxt)))
                path.append(nxt)
                prev, cur = cur, nxt
            label = f"S{seg_i}"
            seg_i += 1
            for node in path:
                memberships[str(node)].append((label, [str(x) for x in path]))
    out = {}
    for node, options in memberships.items():
        out[node] = min(
            options,
            key=lambda z: (
                min(root_dist.get(z[1][0], 10**9), root_dist.get(z[1][-1], 10**9)),
                z[0],
            ),
        )[0]
    return out


def truth_segment_labels(truth: pd.DataFrame, truth_graph: nx.Graph, root_state: str) -> np.ndarray:
    edge_to_segment: dict[frozenset, str] = {}
    specials = {v for v, d in truth_graph.degree if d != 2} | {str(root_state)}
    seen: set[frozenset] = set()
    seg_i = 0
    for start in sorted(specials, key=str):
        for neighbor in sorted(truth_graph.neighbors(start), key=str):
            edge = frozenset((start, neighbor))
            if edge in seen:
                continue
            prev, cur = start, neighbor
            while True:
                edge = frozenset((prev, cur))
                seen.add(edge)
                edge_to_segment[edge] = f"T{seg_i}"
                if cur in specials:
                    break
                nxt = next(x for x in truth_graph.neighbors(cur) if x != prev)
                prev, cur = cur, nxt
            seg_i += 1
    return np.asarray(
        [edge_to_segment[frozenset((str(a), str(b)))] for a, b in zip(truth["from"], truth["to"])],
        dtype=object,
    )


def branch_f1(reference, prediction, valid=None) -> tuple[float, float, float, float]:
    """Dynbenchmark-style, label-invariant symmetric best-match branch F1.

    Each reference branch is matched to the prediction with maximal cell-set
    Jaccard (recovery); prediction-to-reference gives relevance. Their harmonic
    mean is reported. Missing predictions are excluded from branch quality and
    disclosed separately as branch coverage.
    """
    ref, pred = np.asarray(reference, object), np.asarray(prediction, object)
    if valid is None:
        valid = pd.notna(pred)
    valid = np.asarray(valid, bool) & pd.notna(ref) & pd.notna(pred)
    coverage = float(valid.mean()) if len(valid) else np.nan
    if valid.sum() < 5:
        return np.nan, np.nan, np.nan, coverage
    ref, pred = ref[valid].astype(str), pred[valid].astype(str)
    ref_sets = {x: set(np.flatnonzero(ref == x)) for x in np.unique(ref)}
    pred_sets = {x: set(np.flatnonzero(pred == x)) for x in np.unique(pred)}

    def jaccard(a, b):
        return len(a & b) / len(a | b) if a or b else 0.0

    recovery = float(np.mean([max(jaccard(a, b) for b in pred_sets.values()) for a in ref_sets.values()]))
    relevance = float(np.mean([max(jaccard(a, b) for a in ref_sets.values()) for b in pred_sets.values()]))
    score = 2 * recovery * relevance / (recovery + relevance) if recovery + relevance else 0.0
    return score, recovery, relevance, coverage


def set_mapping_f1(reference_sets, prediction_sets, n_cells: int) -> tuple[float, float, float, float]:
    """Symmetric Jaccard mapping for possibly overlapping root-to-leaf lineages."""
    reference_sets = [set(x) for x in reference_sets if len(x)]
    prediction_sets = [set(x) for x in prediction_sets if len(x)]
    covered = set().union(*prediction_sets) if prediction_sets else set()
    coverage = len(covered) / n_cells if n_cells else np.nan
    if not reference_sets or not prediction_sets:
        return np.nan, np.nan, np.nan, coverage

    def jaccard(a, b):
        return len(a & b) / len(a | b) if a or b else 0.0

    recovery = float(np.mean([max(jaccard(a, b) for b in prediction_sets) for a in reference_sets]))
    relevance = float(np.mean([max(jaccard(a, b) for a in reference_sets) for b in prediction_sets]))
    score = 2 * recovery * relevance / (recovery + relevance) if recovery + relevance else 0.0
    return score, recovery, relevance, coverage


def truth_lineage_sets(truth: pd.DataFrame, graph: nx.Graph, root: str) -> list[set[int]]:
    """Cell sets for every root-to-leaf truth lineage; shared trunk cells overlap."""
    root = str(root)
    if root not in graph or not nx.is_tree(graph):
        return []
    leaves = [str(v) for v, degree in graph.degree if degree == 1 and str(v) != root]
    edge_paths = [
        {frozenset((str(a), str(b))) for a, b in nx.utils.pairwise(nx.shortest_path(graph, root, leaf))}
        for leaf in leaves
    ]
    cell_edges = [frozenset((str(a), str(b))) for a, b in zip(truth["from"], truth["to"])]
    return [{i for i, edge in enumerate(cell_edges) if edge in path} for path in edge_paths]


def graph_lineage_sets(graph: nx.Graph, root: str | None, cell_nodes) -> list[set[int]]:
    """Cell sets for native root-to-leaf graph paths."""
    if root is None or str(root) not in graph or not nx.is_forest(graph):
        return []
    root = str(root)
    component = graph.subgraph(nx.node_connected_component(graph, root)).copy()
    leaves = [str(v) for v, degree in component.degree if degree == 1 and str(v) != root]
    nodes = np.asarray(cell_nodes, object)
    out = []
    for leaf in leaves:
        path = {str(v) for v in nx.shortest_path(component, root, leaf)}
        out.append({i for i, node in enumerate(nodes) if pd.notna(node) and str(node) in path})
    return out


def fate_lineage_sets(native_branch) -> list[set[int]]:
    """Interpret fate labels as terminal lineages with one shared unlabeled trunk."""
    raw = pd.Series(np.asarray(native_branch, object))
    missing = raw.isna() | raw.astype(str).str.lower().isin(["", "nan", "none", "na"])
    labels = sorted(raw.loc[~missing].astype(str).unique())
    trunk = set(np.flatnonzero(missing.to_numpy()))
    return [trunk | set(np.flatnonzero(raw.astype(str).to_numpy() == label)) for label in labels]


def truth_pair_distances(truth: pd.DataFrame, graph: nx.Graph, i, j) -> np.ndarray:
    """Exact tree-geodesic distance between cells embedded on truth edges."""
    lengths = {(str(a), str(b)): float(d.get("length", 1.0)) for a, b, d in graph.edges(data=True)}
    lengths.update({(b, a): v for (a, b), v in list(lengths.items())})
    milestone_dist = dict(nx.all_pairs_dijkstra_path_length(graph, weight="length"))
    rows = truth.reset_index(drop=True)
    out = np.empty(len(i), float)
    for q, (ia, ib) in enumerate(zip(i, j)):
        a, b = rows.iloc[int(ia)], rows.iloc[int(ib)]
        au, av, af = str(a["from"]), str(a["to"]), float(a["fraction"])
        bu, bv, bf = str(b["from"]), str(b["to"]), float(b["fraction"])
        al, bl = lengths[(au, av)], lengths[(bu, bv)]
        if frozenset((au, av)) == frozenset((bu, bv)):
            # Orient b consistently if its stored edge direction differs.
            bf2 = bf if (au, av) == (bu, bv) else 1.0 - bf
            out[q] = abs(af - bf2) * al
            continue
        aends = [(au, af * al), (av, (1.0 - af) * al)]
        bends = [(bu, bf * bl), (bv, (1.0 - bf) * bl)]
        out[q] = min(xoff + milestone_dist[x][y] + yoff for (x, xoff), (y, yoff) in product(aends, bends))
    return out


def sampled_pairs(n: int, max_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = np.triu_indices(n, 1)
    if len(ii) > max_pairs:
        chosen = np.random.default_rng(seed).choice(len(ii), max_pairs, replace=False)
        ii, jj = ii[chosen], jj[chosen]
    return ii, jj


def cordist_score(
    truth: pd.DataFrame,
    truth_graph: nx.Graph,
    pseudotime: np.ndarray,
    pred_graph: nx.Graph,
    pred_nodes: np.ndarray,
    seed: int,
    max_pairs: int = 10000,
) -> tuple[float, float, str]:
    valid_pt = np.isfinite(pseudotime)
    nodes = np.asarray(pred_nodes, object)
    usable_nodes = np.asarray([str(x) in pred_graph for x in nodes]) & pd.notna(nodes)
    use_graph = pred_graph.number_of_edges() > 0 and usable_nodes[valid_pt].mean() >= 0.5
    valid = valid_pt & usable_nodes if use_graph else valid_pt
    ids = np.flatnonzero(valid)
    if len(ids) < 5:
        return np.nan, 0.0, "unavailable"
    li, lj = sampled_pairs(len(ids), max_pairs, seed)
    i, j = ids[li], ids[lj]
    true_d = truth_pair_distances(truth, truth_graph, i, j)
    if use_graph:
        distance = dict(nx.all_pairs_dijkstra_path_length(pred_graph, weight="length"))
        pred_d = np.asarray([distance.get(str(nodes[a]), {}).get(str(nodes[b]), np.nan) for a, b in zip(i, j)])
        mode = "native_graph_geodesic"
    else:
        pred_d = np.abs(pseudotime[i] - pseudotime[j])
        mode = "pseudotime_distance_fallback"
    ok = np.isfinite(pred_d)
    return safe_spearman(true_d[ok], pred_d[ok]), float(ok.sum() / len(ok)), mode


def quotient_to_truth(
    truth: pd.DataFrame, pred_graph: nx.Graph, pred_nodes: np.ndarray
) -> nx.Graph:
    """Post-hoc cell-vote projection of a predicted graph onto truth milestones."""
    nodes = np.asarray(pred_nodes, object)
    cell_state = np.where(truth["fraction"].to_numpy() <= 0.5, truth["from"], truth["to"]).astype(str)
    votes: dict[str, str] = {}
    for node in sorted({str(x) for x in nodes if pd.notna(x) and str(x) in pred_graph}):
        states = cell_state[np.asarray([pd.notna(x) and str(x) == node for x in nodes])]
        if len(states):
            values, counts = np.unique(states, return_counts=True)
            votes[node] = str(values[np.argmax(counts)])
    # Internal graph vertices without cells inherit the nearest occupied graph vertex.
    for node in pred_graph:
        node = str(node)
        if node in votes or not votes:
            continue
        d = nx.single_source_shortest_path_length(pred_graph, node)
        candidates = sorted((dist, mapped) for mapped, dist in d.items() if str(mapped) in votes)
        if candidates:
            votes[node] = votes[str(candidates[0][1])]
    q = nx.Graph()
    for u, v in pred_graph.edges:
        a, b = votes.get(str(u)), votes.get(str(v))
        if a is not None and b is not None and a != b:
            q.add_edge(a, b, length=1.0)
    return q


def _spectral_density(graph: nx.Graph, nodes: list[str], gamma: float, grid: np.ndarray) -> np.ndarray:
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight=None)
    lap = np.diag(adjacency.sum(axis=1)) - adjacency
    omega = np.sqrt(np.clip(np.linalg.eigvalsh(lap), 0, None))
    omega = omega[omega > 1e-8]
    if not len(omega):
        return np.zeros_like(grid)
    rho = (gamma / ((grid[:, None] - omega[None, :]) ** 2 + gamma**2)).sum(axis=1)
    area = trapezoid(rho, grid)
    return rho / area if area > 0 else rho


@lru_cache(maxsize=None)
def _im_gamma(n: int) -> float:
    nodes = [str(i) for i in range(n)]
    complete, empty = nx.complete_graph(nodes), nx.empty_graph(nodes)
    hi = np.sqrt(max(n, 2)) + 4.0
    grid = np.linspace(0, hi, 2048)

    def empty_complete_distance(gamma):
        ra = _spectral_density(empty, nodes, gamma, grid)
        rb = _spectral_density(complete, nodes, gamma, grid)
        return np.sqrt(trapezoid((ra - rb) ** 2, grid))

    try:
        return float(brentq(lambda x: empty_complete_distance(x) - 1.0, 1e-3, 10.0))
    except ValueError:
        return 0.08


def _im_distance(a: nx.Graph, b: nx.Graph, nodes: list[str]) -> float:
    n = len(nodes)
    if n < 2:
        return 0.0
    hi = np.sqrt(max(n, 2)) + 4.0
    grid = np.linspace(0, hi, 2048)
    gamma = _im_gamma(n)
    ra = _spectral_density(a, nodes, gamma, grid)
    rb = _spectral_density(b, nodes, gamma, grid)
    return float(min(1.0, np.sqrt(trapezoid((ra - rb) ** 2, grid))))


def topology_one_minus_him(reference: nx.Graph, prediction: nx.Graph) -> float:
    """HIM similarity = 1 - combined normalized Hamming/IM distance."""
    nodes = sorted({str(x) for x in reference.nodes} | {str(x) for x in prediction.nodes})
    if len(nodes) < 2 or prediction.number_of_edges() == 0:
        return np.nan
    ref = nx.Graph(reference)
    pred = nx.Graph(prediction)
    ref.add_nodes_from(nodes)
    pred.add_nodes_from(nodes)
    ar = nx.to_numpy_array(ref, nodelist=nodes, weight=None)
    ap = nx.to_numpy_array(pred, nodelist=nodes, weight=None)
    hamming = float(np.abs(ar - ap).sum() / max(len(nodes) * (len(nodes) - 1), 1))
    im = _im_distance(ref, pred, nodes)
    return float(np.clip(1.0 - np.sqrt((hamming**2 + im**2) / 2.0), 0.0, 1.0))


def weighted_rank_correlation(reference_importance, predicted_importance) -> float:
    """Reference-importance-weighted Pearson correlation of feature ranks."""
    a = np.asarray(reference_importance, float)
    b = np.asarray(predicted_importance, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan
    a, b = a[ok], b[ok]
    ra, rb = rankdata(a), rankdata(b)
    weights = np.maximum(a, 0)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights /= weights.sum()
    ma, mb = np.sum(weights * ra), np.sum(weights * rb)
    cov = np.sum(weights * (ra - ma) * (rb - mb))
    den = np.sqrt(np.sum(weights * (ra - ma) ** 2) * np.sum(weights * (rb - mb) ** 2))
    return float(np.clip(cov / den, 0.0, 1.0)) if den > 0 else np.nan

