#!/usr/bin/env python3
"""Score frozen breast multi-tree and global single-tree representations for P9."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score, silhouette_score


PRIMARY_METRICS = ["functional_continuity", "program_separation", "program_purity"]
REPRESENTATIONS = ["multi_tree", "complexity_matched_global", "native_global"]
BOOTSTRAP_SEED = 20260812
N_BOOTSTRAP = 499


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_primary_scores(p8_dir: Path) -> pd.DataFrame:
    index = pd.read_csv(p8_dir / "P8_INDEPENDENT_SIGNATURE_INDEX.csv")
    primary = index[index.family.isin(["external_curated", "Hallmark_2020"])].copy()
    score_ids = sorted(primary.score_id.astype(str).unique())
    if len(score_ids) != 43:
        raise ValueError(f"Expected 43 frozen primary signatures, found {len(score_ids)}")

    selected = []
    for chunk in pd.read_csv(
        p8_dir / "P8_INDEPENDENT_SIGNATURE_SCORES.csv",
        usecols=["unit_id", "score_id", "score"],
        chunksize=200000,
    ):
        keep = chunk.score_id.astype(str).isin(score_ids)
        if keep.any():
            selected.append(chunk.loc[keep].copy())
    long = pd.concat(selected, ignore_index=True)
    matrix = long.pivot(index="unit_id", columns="score_id", values="score").sort_index()
    matrix = matrix.reindex(columns=score_ids)
    if matrix.shape != (883, 43) or matrix.isna().any().any():
        raise ValueError(f"Frozen primary score matrix is incomplete: {matrix.shape}")
    return matrix


def standardized(matrix: pd.DataFrame) -> pd.DataFrame:
    values = matrix.to_numpy(float)
    scale = values.std(axis=0, ddof=0)
    if np.any(scale <= 0):
        bad = matrix.columns[scale <= 0].tolist()
        raise ValueError(f"Constant primary scores cannot be standardized: {bad}")
    z = (values - values.mean(axis=0)) / scale
    return pd.DataFrame(z, index=matrix.index, columns=matrix.columns)


def continuity(labels: np.ndarray, tau: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, int]:
    squared_program = []
    absolute_program = []
    transition_program = []
    for label in sorted(set(labels.astype(str))):
        use = np.flatnonzero(labels.astype(str) == label)
        if len(use) < 5 or np.unique(tau[use]).size < 2:
            continue
        ranked_tau = stats.rankdata(tau[use]).astype(float)
        ranked_scores = stats.rankdata(scores[use], axis=0).astype(float)
        ranked_tau -= ranked_tau.mean()
        ranked_scores -= ranked_scores.mean(axis=0, keepdims=True)
        denominator = np.sqrt(np.sum(ranked_tau ** 2) * np.sum(ranked_scores ** 2, axis=0))
        rho_array = np.divide(
            ranked_tau @ ranked_scores,
            denominator,
            out=np.full(scores.shape[1], np.nan),
            where=denominator > 0,
        )
        rho_array = rho_array[np.isfinite(rho_array)]
        if rho_array.size:
            squared_program.append(float(np.mean(rho_array ** 2)))
            absolute_program.append(float(np.mean(np.abs(rho_array))))
            transition_program.append(float(np.mean(np.abs(rho_array) >= 0.20)))
    if not squared_program:
        return np.nan, np.nan, np.nan, 0
    return (
        float(np.median(squared_program)),
        float(np.median(absolute_program)),
        float(np.median(transition_program)),
        len(squared_program),
    )


def variance_explained(labels: np.ndarray, values: np.ndarray) -> float:
    total = float(np.sum((values - values.mean(axis=0, keepdims=True)) ** 2))
    within = 0.0
    for label in sorted(set(labels.astype(str))):
        local = values[labels.astype(str) == label]
        within += float(np.sum((local - local.mean(axis=0, keepdims=True)) ** 2))
    return float(1.0 - within / total) if total > 0 else np.nan


def program_audit_rows(
    representation: str, labels: np.ndarray, tau: np.ndarray, scores: np.ndarray
) -> list[dict[str, float | int | str]]:
    rows = []
    for label in sorted(set(labels.astype(str))):
        use = np.flatnonzero(labels.astype(str) == label)
        ranked_tau = stats.rankdata(tau[use]).astype(float)
        ranked_scores = stats.rankdata(scores[use], axis=0).astype(float)
        ranked_tau -= ranked_tau.mean()
        ranked_scores -= ranked_scores.mean(axis=0, keepdims=True)
        denominator = np.sqrt(np.sum(ranked_tau ** 2) * np.sum(ranked_scores ** 2, axis=0))
        rho = np.divide(
            ranked_tau @ ranked_scores,
            denominator,
            out=np.full(scores.shape[1], np.nan),
            where=denominator > 0,
        )
        rho = rho[np.isfinite(rho)]
        rows.append({
            "representation": representation,
            "program": str(label),
            "n_metacells": int(len(use)),
            "tau_min": float(np.min(tau[use])),
            "tau_max": float(np.max(tau[use])),
            "n_scores_valid": int(len(rho)),
            "mean_squared_spearman": float(np.mean(rho ** 2)) if len(rho) else np.nan,
            "mean_absolute_spearman": float(np.mean(np.abs(rho))) if len(rho) else np.nan,
            "fraction_abs_rho_ge_0_20": float(np.mean(np.abs(rho) >= 0.20)) if len(rho) else np.nan,
        })
    return rows


def metrics(
    labels: np.ndarray,
    tau: np.ndarray,
    scores: np.ndarray,
    spatial_blocks: np.ndarray,
    subtype_entropy: np.ndarray,
) -> dict[str, float]:
    functional, mean_abs_rho, transition_fraction, n_programs_used = continuity(labels, tau, scores)
    counts = pd.Series(labels.astype(str)).value_counts()
    can_silhouette = counts.size >= 2 and counts.min() >= 2 and counts.size < len(labels)
    separation = float(silhouette_score(scores, labels, metric="euclidean")) if can_silhouette else np.nan
    return {
        "functional_continuity": functional,
        "program_separation": separation,
        "program_purity": variance_explained(labels, scores),
        "secondary_mean_absolute_rho": mean_abs_rho,
        "secondary_transition_fraction_abs_rho_ge_0_20": transition_fraction,
        "secondary_spatial_block_NMI": float(normalized_mutual_info_score(spatial_blocks, labels)),
        "secondary_subtype_entropy_purity": variance_explained(labels, subtype_entropy[:, None]),
        "n_programs": int(counts.size),
        "n_programs_used_for_continuity": int(n_programs_used),
        "min_program_size": int(counts.min()),
        "max_program_size": int(counts.max()),
    }


def load_representations(p9_dir: Path, units: pd.Index) -> dict[str, pd.DataFrame]:
    multi = pd.read_csv(p9_dir / "P9_FROZEN_MULTITREE_SINGLE_ROW_ASSIGNMENTS.csv").set_index("unit_id")
    matched = pd.read_csv(p9_dir / "P9_COMPLEXITY_MATCHED_CONTROL.csv").set_index("unit_id")
    native = pd.read_csv(p9_dir / "P9_GLOBAL_NATIVE_ASSIGNMENTS.csv").set_index("unit_id")
    frames = {
        "multi_tree": multi[["program_id", "pseudotime"]].rename(
            columns={"program_id": "program", "pseudotime": "tau"}
        ),
        "complexity_matched_global": matched[["matched_program", "global_pseudotime"]].rename(
            columns={"matched_program": "program", "global_pseudotime": "tau"}
        ),
        "native_global": native[["native_program", "global_pseudotime"]].rename(
            columns={"native_program": "program", "global_pseudotime": "tau"}
        ),
    }
    expected = set(units.astype(str))
    for name, frame in frames.items():
        if set(frame.index.astype(str)) != expected:
            raise ValueError(f"{name} does not use the exact frozen 883-metacell universe")
        frames[name] = frame.reindex(units)
    return frames


def comparison_rows(distribution: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for control in ["complexity_matched_global", "native_global"]:
        comparison = f"multi_tree_minus_{control}"
        for metric in PRIMARY_METRICS:
            delta = float(
                points.loc[points.representation.eq("multi_tree"), metric].iloc[0]
                - points.loc[points.representation.eq(control), metric].iloc[0]
            )
            boot = distribution[
                distribution.comparison.eq(comparison) & distribution.metric.eq(metric)
            ].delta.dropna().to_numpy(float)
            low, high = np.quantile(boot, [0.025, 0.975])
            rows.append({
                "comparison": comparison,
                "metric": metric,
                "multi_tree_score": float(points.loc[points.representation.eq("multi_tree"), metric].iloc[0]),
                "control_score": float(points.loc[points.representation.eq(control), metric].iloc[0]),
                "point_delta": delta,
                "bootstrap_delta_mean": float(np.mean(boot)),
                "bootstrap_delta_median": float(np.median(boot)),
                "paired_ci_low": float(low),
                "paired_ci_high": float(high),
                "direction_favors_multi_tree": bool(delta > 0),
                "ci_favors_multi_tree": bool(low > 0),
                "ci_favors_control": bool(high < 0),
                "n_bootstrap_valid": int(len(boot)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p8-dir", type=Path, required=True)
    parser.add_argument("--p9-dir", type=Path, required=True, help="Frozen P9.0 control directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    audit = json.loads((args.p9_dir / "P90_CONTROL_AUDIT.json").read_text(encoding="utf-8"))
    if audit.get("decision") != "PASS":
        raise ValueError("P9.0 did not PASS; biological scoring is prohibited")

    raw_scores = load_primary_scores(args.p8_dir)
    scores = standardized(raw_scores)
    representations = load_representations(args.p9_dir, scores.index)
    context = pd.read_csv(args.p8_dir / "P8_SOFT_SHARING_BIOLOGY.csv").set_index("unit_id").reindex(scores.index)
    if context[["spatial_block", "raw_classifier_subtype_entropy"]].isna().any().any():
        raise ValueError("P8 spatial block or subtype-entropy context is incomplete")
    blocks = context.spatial_block.to_numpy(int)
    entropy = context.raw_classifier_subtype_entropy.to_numpy(float)
    score_values = scores.to_numpy(float)

    point_rows = []
    program_rows = []
    arrays = {}
    for name in REPRESENTATIONS:
        frame = representations[name]
        labels = frame.program.astype(str).to_numpy()
        tau = frame.tau.to_numpy(float)
        arrays[name] = (labels, tau)
        point_rows.append({"representation": name, **metrics(labels, tau, score_values, blocks, entropy)})
        program_rows.extend(program_audit_rows(name, labels, tau, score_values))
    points = pd.DataFrame(point_rows)
    points.to_csv(args.output / "P9_MULTITREE_VS_SINGLETREE.csv", index=False)
    pd.DataFrame(program_rows).to_csv(args.output / "P9_PROGRAM_CONTINUITY_AUDIT.csv", index=False)

    unique_blocks = np.sort(np.unique(blocks))
    block_indices = {block: np.flatnonzero(blocks == block) for block in unique_blocks}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_rows = []
    for replicate in range(N_BOOTSTRAP):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        use = np.concatenate([block_indices[block] for block in sampled_blocks])
        replicate_metrics = {}
        for name in REPRESENTATIONS:
            labels, tau = arrays[name]
            replicate_metrics[name] = metrics(
                labels[use], tau[use], score_values[use], blocks[use], entropy[use]
            )
        for control in ["complexity_matched_global", "native_global"]:
            comparison = f"multi_tree_minus_{control}"
            for metric in PRIMARY_METRICS:
                bootstrap_rows.append({
                    "bootstrap_replicate": replicate,
                    "comparison": comparison,
                    "metric": metric,
                    "multi_tree_score": replicate_metrics["multi_tree"][metric],
                    "control_score": replicate_metrics[control][metric],
                    "delta": replicate_metrics["multi_tree"][metric] - replicate_metrics[control][metric],
                })
    distribution = pd.DataFrame(bootstrap_rows)
    distribution.to_csv(args.output / "P9_BOOTSTRAP_DISTRIBUTION.csv", index=False)
    effects = comparison_rows(distribution, points)
    effects.to_csv(args.output / "P9_BIOLOGICAL_EFFECT_SIZES.csv", index=False)

    primary = effects[effects.comparison.eq("multi_tree_minus_complexity_matched_global")]
    contextual = effects[effects.comparison.eq("multi_tree_minus_native_global")]
    n_favorable = int(primary.direction_favors_multi_tree.sum())
    n_favorable_ci = int(primary.ci_favors_multi_tree.sum())
    n_adverse_ci = int(primary.ci_favors_control.sum())
    n_contextual_favorable = int(contextual.direction_favors_multi_tree.sum())
    if n_favorable >= 2 and n_favorable_ci >= 1 and n_adverse_ci == 0 and n_contextual_favorable >= 2:
        decision_value = "PASS"
    elif n_favorable >= 2 and n_adverse_ci == 0:
        decision_value = "QUALIFIED_PASS"
    else:
        decision_value = "FAIL"

    decision = {
        "stage": "P9_MULTI_TREE_VS_SINGLE_TREE",
        "decision": decision_value,
        "n_primary_favor_multi_vs_complexity_matched": n_favorable,
        "n_primary_ci_favor_multi_vs_complexity_matched": n_favorable_ci,
        "n_primary_ci_favor_complexity_matched": n_adverse_ci,
        "n_primary_favor_multi_vs_native": n_contextual_favorable,
        "n_metacells": int(len(scores)),
        "n_primary_signatures": int(scores.shape[1]),
        "n_spatial_blocks": int(len(unique_blocks)),
        "n_paired_spatial_block_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "construction_genes_removed_from_primary_scores": True,
        "strict_external_assay_independence": False,
        "strict_independence_limitation": "candidate PCA geometry used all panel genes",
        "terminal_count_exactly_matched": bool(audit.get("terminal_count_exactly_matched", False)),
        "terminal_count_limitation": "program count was matched topologically; the native global MST was not rewired to match multi-tree leaf counts",
        "trees_modified_after_P8": False,
    }
    (args.output / "P9_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    report_lines = [
        "# P9 Multi-tree versus single-tree biological control",
        "",
        f"Decision: **{decision_value}**.",
        "",
        "The comparison uses the same 883 frozen metacells and the same 43 leave-construction-HVG-out curated/Hallmark scores for every representation.",
        "Uncertainty is estimated by 499 paired spatial-block bootstrap replicates; metacells are not treated as biological replicates.",
        "",
        "## Primary results",
        "",
        "| Comparison | Metric | Multi-tree | Control | Delta | Paired 95% CI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in effects.itertuples(index=False):
        report_lines.append(
            f"| {row.comparison} | {row.metric} | {row.multi_tree_score:.4f} | "
            f"{row.control_score:.4f} | {row.point_delta:+.4f} | "
            f"[{row.paired_ci_low:+.4f}, {row.paired_ci_high:+.4f}] |"
        )
    report_lines += [
        "",
        "## Interpretation boundary",
        "",
        "The complexity-matched control matches coverage and seven-program granularity using topology alone. It does not exactly match the multi-tree leaf count or program-size distribution.",
        "Primary scores exclude the 50 frozen branch-construction HVGs, but they are not a fully independent assay because candidate PCA used the full panel.",
        "The result tests representational biological structure, not lineage truth or causal topology.",
    ]
    (args.output / "P9_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest_paths = [
        args.p9_dir / "P90_CONTROL_AUDIT.json",
        args.p9_dir / "P90_CONTROL_SHA256.csv",
        args.p9_dir / "P9_FROZEN_MULTITREE_SINGLE_ROW_ASSIGNMENTS.csv",
        args.p9_dir / "P9_GLOBAL_NATIVE_ASSIGNMENTS.csv",
        args.p9_dir / "P9_COMPLEXITY_MATCHED_CONTROL.csv",
        args.p8_dir / "P8_INDEPENDENT_SIGNATURE_INDEX.csv",
        args.p8_dir / "P8_INDEPENDENT_SIGNATURE_SCORES.csv",
        args.p8_dir / "P8_SOFT_SHARING_BIOLOGY.csv",
        args.p9_dir / "P9_PREREGISTRATION.json",
    ]
    pd.DataFrame([{"file": str(path.resolve()), "sha256": sha256(path)} for path in manifest_paths]).to_csv(
        args.output / "P9_INPUT_MANIFEST.csv", index=False
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
