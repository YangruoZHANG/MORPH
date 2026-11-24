#!/usr/bin/env python3
"""Truth-free, root-global stable branch-program estimator."""

from __future__ import annotations

import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


SEED = 20260810


def stable_hvg_kmeans(counts: np.ndarray, nodes: list[int]):
    nodes = sorted(map(int, nodes))
    if len(nodes) < 6:
        return {node: "b0" for node in nodes}, [{
            "n_nodes": len(nodes), "selected_k": 1,
            "selection": "small_root_support", "n_hvg": 0,
        }]
    matrix = np.log1p(np.maximum(np.asarray(counts, dtype=float)[nodes], 0.0))
    n_hvg = min(matrix.shape[1], max(10, int(math.ceil(2.0 * math.sqrt(matrix.shape[1])))))
    variances = np.var(matrix, axis=0)
    genes = np.argsort(variances, kind="mergesort")[-n_hvg:]
    selected = StandardScaler().fit_transform(matrix[:, genes])
    n_pca = min(10, selected.shape[0] - 1, selected.shape[1])
    features = PCA(n_components=n_pca, random_state=SEED).fit_transform(selected)
    k_max = min(6, len(nodes) // 3)
    candidates = []
    for k in range(2, k_max + 1):
        base = KMeans(n_clusters=k, random_state=SEED, n_init=50).fit_predict(features)
        if np.bincount(base).min() < 2:
            continue
        silhouette = float(silhouette_score(features, base))
        stability = []
        rng = np.random.default_rng(SEED + k)
        for iteration in range(20):
            keep = np.sort(rng.choice(n_hvg, size=max(2, int(math.ceil(0.8 * n_hvg))), replace=False))
            trial_input = StandardScaler().fit_transform(matrix[:, genes[keep]])
            trial_pca = PCA(
                n_components=min(n_pca, trial_input.shape[0] - 1, trial_input.shape[1]),
                random_state=SEED + iteration,
            ).fit_transform(trial_input)
            trial = KMeans(
                n_clusters=k, random_state=SEED + iteration, n_init=20
            ).fit_predict(trial_pca)
            stability.append(adjusted_rand_score(base, trial))
        mean_stability = float(np.mean(stability))
        candidates.append((silhouette * max(mean_stability, 0.0), k, base, silhouette, mean_stability))
    if not candidates:
        return {node: "b0" for node in nodes}, [{
            "n_nodes": len(nodes), "selected_k": 1,
            "selection": "no_stable_partition", "n_hvg": n_hvg,
        }]
    score, selected_k, prediction, silhouette, stability = max(
        candidates, key=lambda item: (item[0], -item[1])
    )
    order = sorted(
        range(selected_k),
        key=lambda label: (
            float(np.mean(features[prediction == label, 0])),
            min(nodes[index] for index in np.flatnonzero(prediction == label)),
        ),
    )
    remap = {old: new for new, old in enumerate(order)}
    labels = {
        node: f"b{remap[int(label)]}" for node, label in zip(nodes, prediction)
    }
    audit = [{
        "n_nodes": len(nodes), "selected_k": selected_k,
        "selection": "max_silhouette_x_feature_bootstrap_stability",
        "n_hvg": n_hvg, "n_pca": n_pca,
        "silhouette": silhouette, "feature_bootstrap_stability": stability,
        "selection_score": score,
    }]
    return labels, audit

