#!/usr/bin/env python3
"""DEV-only sensitivity audit for branch-program model complexity."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score


HERE = Path(__file__).resolve().parent


def load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = load("run_phase_a_case_v2.py", "phase_a_v2_for_branch_complexity_v9")
V9 = load("branch_program_estimator_v9.py", "branch_program_estimator_v9_audit")
PENALTIES = (0.0, 0.10, 0.20, 0.25, 0.35, 0.50, 0.75)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in sorted(path for path in args.runs_root.iterdir() if path.is_dir()):
        method_dir = run / "generic_upstream_v1_exact_root" / "CORE_05"
        meta = ad.read_h5ad(method_dir / "metacell_adata.h5ad")
        membership, roots, supports = V2.P.inferred_membership(method_dir, meta.n_obs)
        del membership
        truth = V2.P.load_truth_context(run)
        counts = meta.X.toarray() if sparse.issparse(meta.X) else np.asarray(meta.X)
        for root in roots:
            root_id = str(root["root_id"])
            nodes = sorted(map(int, supports[root_id]))
            candidates, _ = V9.candidate_partitions(counts, nodes)
            names = [str(meta.obs_names[node]) for node in nodes]
            eligible = [
                index for index, name in enumerate(names)
                if (name, root_id) in truth.branch
            ]
            y_true = [truth.branch[(names[index], root_id)] for index in eligible]
            for penalty in PENALTIES:
                scored = []
                for candidate in candidates:
                    score = (
                        candidate["unpenalized_score"]
                        - penalty * (candidate["k"] - 1) / np.sqrt(len(nodes))
                    )
                    scored.append((score, -candidate["k"], candidate))
                if not scored:
                    rows.append({
                        "run": run.name, "root_id": root_id, "n_nodes": len(nodes),
                        "complexity_penalty": penalty, "selected_k": 1,
                        "branch_ARI": 0.0, "selection_score": np.nan,
                    })
                    continue
                score, _, selected = max(scored, key=lambda item: (item[0], item[1]))
                ari = adjusted_rand_score(
                    y_true, selected["labels"][eligible]
                ) if len(eligible) >= 2 else np.nan
                rows.append({
                    "run": run.name, "root_id": root_id, "n_nodes": len(nodes),
                    "complexity_penalty": penalty,
                    "selected_k": selected["k"], "branch_ARI": ari,
                    "selection_score": score,
                    "silhouette": selected["silhouette"],
                    "feature_bootstrap_stability": selected["feature_bootstrap_stability"],
                    "cluster_sizes": selected["cluster_sizes"],
                    "truth_used_for_selection": False,
                    "truth_used_for_audit_only": True,
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "BRANCH_COMPLEXITY_SENSITIVITY.csv", index=False)
    summary = frame.groupby("complexity_penalty", as_index=False).agg(
        mean_selected_k=("selected_k", "mean"),
        median_selected_k=("selected_k", "median"),
        mean_branch_ARI=("branch_ARI", "mean"),
        median_branch_ARI=("branch_ARI", "median"),
        minimum_branch_ARI=("branch_ARI", "min"),
    )
    summary.to_csv(args.output / "BRANCH_COMPLEXITY_SUMMARY.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
