#!/usr/bin/env python3
"""Stable branch programs after removing smooth within-root progression."""

from __future__ import annotations

import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


SEED = 20260811


def _features(counts: np.ndarray, nodes: list[int], tau: dict[int, float]):
    matrix = np.log1p(np.maximum(np.asarray(counts, dtype=float)[nodes], 0.0))
    n_hvg = min(matrix.shape[1], max(10, int(math.ceil(2.0 * math.sqrt(matrix.shape[1])))))
    genes = np.argsort(np.var(matrix, axis=0), kind="mergesort")[-n_hvg:]
    selected = matrix[:, genes]
    t = np.asarray([tau[int(node)] for node in nodes], dtype=float)
    design = np.column_stack([np.ones(len(t)), t, t * t])
    residual = selected - design @ np.linalg.lstsq(design, selected, rcond=None)[0]
    residual = StandardScaler().fit_transform(residual)
    n_pca = min(10, residual.shape[0] - 1, residual.shape[1])
    return PCA(n_components=n_pca, random_state=SEED).fit_transform(residual), n_hvg, n_pca


def stable_residual_hvg_kmeans(
    counts: np.ndarray,
    nodes: list[int],
    tau: dict[int, float],
    complexity_penalty: float = 0.10,
):
    nodes = sorted(map(int, nodes))
    if len(nodes) < 6:
        return {node: "b0" for node in nodes}, [{
            "n_nodes": len(nodes), "selected_k": 1,
            "selection": "small_support", "complexity_penalty": complexity_penalty,
            "candidate_k": 1, "selected": True,
        }]
    features, n_hvg, n_pca = _features(counts, nodes, tau)
    candidates = []
    for k in range(2, min(6, len(nodes) // 3) + 1):
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=50).fit_predict(features)
        sizes = np.bincount(labels)
        if sizes.min() < 2:
            continue
        silhouette = float(silhouette_score(features, labels))
        stability = []
        rng = np.random.default_rng(SEED + k)
        for iteration in range(20):
            keep = np.sort(rng.choice(len(nodes), size=max(4, int(math.ceil(0.8 * len(nodes)))), replace=False))
            trial = KMeans(
                n_clusters=k, random_state=SEED + iteration, n_init=20
            ).fit_predict(features[keep])
            stability.append(adjusted_rand_score(labels[keep], trial))
        stable = float(np.mean(stability))
        score = silhouette * max(stable, 0.0) - complexity_penalty * (k - 1) / math.sqrt(len(nodes))
        candidates.append({
            "k": k, "labels": labels, "silhouette": silhouette,
            "feature_bootstrap_stability": stable, "selection_score": score,
            "cluster_sizes": ";".join(map(str, sorted(sizes.tolist()))),
            "n_hvg": n_hvg, "n_pca": n_pca,
        })
    if not candidates:
        return {node: "b0" for node in nodes}, [{
            "n_nodes": len(nodes), "selected_k": 1,
            "selection": "no_stable_partition", "complexity_penalty": complexity_penalty,
            "candidate_k": 1, "selected": True,
        }]
    selected = max(candidates, key=lambda item: (item["selection_score"], -item["k"]))
    order = sorted(
        range(int(selected["k"])),
        key=lambda label: (
            float(np.median([tau[nodes[i]] for i in np.flatnonzero(selected["labels"] == label)])),
            min(nodes[i] for i in np.flatnonzero(selected["labels"] == label)),
        ),
    )
    remap = {old: new for new, old in enumerate(order)}
    labels = {
        node: f"b{remap[int(label)]}"
        for node, label in zip(nodes, selected["labels"])
    }
    audit = [{
        "n_nodes": len(nodes), "selected_k": int(selected["k"]),
        "selection": "residual_HVG_silhouette_x_stability_minus_complexity",
        "complexity_penalty": complexity_penalty,
        "candidate_k": int(candidate["k"]),
        "selected": int(candidate["k"]) == int(selected["k"]),
        "cluster_sizes": candidate["cluster_sizes"], "n_hvg": candidate["n_hvg"],
        "n_pca": candidate["n_pca"], "silhouette": candidate["silhouette"],
        "feature_bootstrap_stability": candidate["feature_bootstrap_stability"],
        "selection_score": candidate["selection_score"],
    } for candidate in candidates]
    return labels, audit
