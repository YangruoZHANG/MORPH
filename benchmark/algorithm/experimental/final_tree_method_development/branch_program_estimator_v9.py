#!/usr/bin/env python3
"""Stable branch-program estimator with explicit complexity control."""

from __future__ import annotations

import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


SEED = 20260810


def candidate_partitions(counts: np.ndarray, nodes: list[int]) -> tuple[list[dict], np.ndarray]:
    nodes = sorted(map(int, nodes))
    if len(nodes) < 6:
        return [], np.empty((len(nodes), 0), dtype=float)
    matrix = np.log1p(np.maximum(np.asarray(counts, dtype=float)[nodes], 0.0))
    n_hvg = min(matrix.shape[1], max(10, int(math.ceil(2.0 * math.sqrt(matrix.shape[1])))))
    variances = np.var(matrix, axis=0)
    genes = np.argsort(variances, kind="mergesort")[-n_hvg:]
    selected = StandardScaler().fit_transform(matrix[:, genes])
    n_pca = min(10, selected.shape[0] - 1, selected.shape[1])
    features = PCA(n_components=n_pca, random_state=SEED).fit_transform(selected)
    candidates = []
    for k in range(2, min(6, len(nodes) // 3) + 1):
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=50).fit_predict(features)
        cluster_sizes = np.bincount(labels)
        if cluster_sizes.min() < 2:
            continue
        silhouette = float(silhouette_score(features, labels))
        stability_values = []
        rng = np.random.default_rng(SEED + k)
        for iteration in range(20):
            keep = np.sort(
                rng.choice(
                    n_hvg, size=max(2, int(math.ceil(0.8 * n_hvg))), replace=False
                )
            )
            trial_input = StandardScaler().fit_transform(matrix[:, genes[keep]])
            trial_features = PCA(
                n_components=min(n_pca, trial_input.shape[0] - 1, trial_input.shape[1]),
                random_state=SEED + iteration,
            ).fit_transform(trial_input)
            trial = KMeans(
                n_clusters=k, random_state=SEED + iteration, n_init=20
            ).fit_predict(trial_features)
            stability_values.append(adjusted_rand_score(labels, trial))
        stability = float(np.mean(stability_values))
        candidates.append({
            "k": k,
            "labels": labels,
            "silhouette": silhouette,
            "feature_bootstrap_stability": stability,
            "unpenalized_score": silhouette * max(stability, 0.0),
            "cluster_sizes": ";".join(map(str, sorted(cluster_sizes.tolist()))),
            "n_hvg": n_hvg,
            "n_pca": n_pca,
        })
    return candidates, features


def stable_hvg_kmeans(
    counts: np.ndarray, nodes: list[int], complexity_penalty: float = 0.25
):
    nodes = sorted(map(int, nodes))
    candidates, features = candidate_partitions(counts, nodes)
    if not candidates:
        return {node: "b0" for node in nodes}, [{
            "n_nodes": len(nodes), "selected_k": 1,
            "selection": "small_support_or_no_stable_partition",
            "complexity_penalty": float(complexity_penalty),
            "candidate_k": 1, "selected": True,
        }]
    for candidate in candidates:
        candidate["selection_score"] = float(
            candidate["unpenalized_score"]
            - float(complexity_penalty) * (candidate["k"] - 1) / math.sqrt(len(nodes))
        )
    selected = max(candidates, key=lambda item: (item["selection_score"], -item["k"]))
    selected_k = int(selected["k"])
    prediction = selected["labels"]
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
    audit = []
    for candidate in candidates:
        audit.append({
            "n_nodes": len(nodes),
            "selected_k": selected_k,
            "selection": "silhouette_x_stability_minus_complexity",
            "complexity_penalty": float(complexity_penalty),
            "candidate_k": int(candidate["k"]),
            "selected": int(candidate["k"]) == selected_k,
            "cluster_sizes": candidate["cluster_sizes"],
            "n_hvg": candidate["n_hvg"],
            "n_pca": candidate["n_pca"],
            "silhouette": candidate["silhouette"],
            "feature_bootstrap_stability": candidate["feature_bootstrap_stability"],
            "unpenalized_score": candidate["unpenalized_score"],
            "selection_score": candidate["selection_score"],
        })
    return labels, audit
