#!/usr/bin/env python3
"""Smoke-test paper baseline adapters without starting the formal benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path(
    r"D:\桌面\FYP\BASE_CASCAT\algorithms\unified_eval_kit\datasets_multiroot\sim_K2_sep.h5ad"
)
GOLD = REPO / "algorithm" / "experimental" / "cross_trajectory_gold_standard"
DEV = REPO / "algorithm" / "experimental" / "final_tree_method_development"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Empty smoke table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not_installed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def roots_from_truth(adata: ad.AnnData) -> list[str]:
    percentages = adata.uns["milestone_percentages"]
    roots = [str(value) for value in adata.uns["roots"]]
    result = []
    for root in roots:
        candidates = percentages[percentages["milestone_id"].astype(str).eq(root)]
        selected = candidates.sort_values(
            ["percentage", "cell_id"], ascending=[False, True]
        ).iloc[0]
        result.append(str(selected["cell_id"]))
    return result


def prepare_scanpy(adata: ad.AnnData, seed: int) -> ad.AnnData:
    result = adata.copy()
    result.obs_names_make_unique()
    matrix = result.X.toarray() if sparse.issparse(result.X) else np.asarray(result.X)
    matrix = np.log1p(np.maximum(matrix.astype(float), 0.0))
    matrix = StandardScaler().fit_transform(matrix)
    n_components = min(30, result.n_vars - 1, result.n_obs - 1)
    embedding = PCA(n_components=n_components, random_state=seed).fit_transform(matrix)
    result.obsm["X_pca"] = embedding
    n_neighbors = min(15, result.n_obs - 1)
    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embedding)
    distances, indices = neighbors.kneighbors(embedding)
    rows = np.repeat(np.arange(result.n_obs), n_neighbors)
    columns = indices[:, 1:].reshape(-1)
    values = distances[:, 1:].reshape(-1)
    sigma = float(np.median(values[values > 0])) if np.any(values > 0) else 1.0
    weights = np.exp(-(values**2) / (2.0 * sigma**2 + 1e-12))
    distance_graph = sparse.csr_matrix(
        (values, (rows, columns)), shape=(result.n_obs, result.n_obs)
    )
    connectivity_graph = sparse.csr_matrix(
        (weights, (rows, columns)), shape=(result.n_obs, result.n_obs)
    )
    result.obsp["distances"] = distance_graph.maximum(distance_graph.T)
    result.obsp["connectivities"] = connectivity_graph.maximum(connectivity_graph.T)
    result.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {
            "n_neighbors": n_neighbors,
            "method": "fixed_sklearn_knn",
            "metric": "euclidean",
        },
    }
    n_clusters = min(16, max(6, int(round(np.sqrt(result.n_obs / 2.0)))))
    labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(
        embedding
    )
    result.obs["smoke_cluster"] = pd.Categorical(labels.astype(str))
    return result


def run_paga_dpt(adata: ad.AnnData, root_cell: str) -> dict[str, object]:
    result = prepare_scanpy(adata, 1101)
    sc.tl.paga(result, groups="smoke_cluster")
    result.uns["iroot"] = int(result.obs_names.get_loc(root_cell))
    sc.tl.diffmap(result)
    sc.tl.dpt(result)
    pseudotime = result.obs["dpt_pseudotime"].to_numpy(float)
    tree = result.uns["paga"]["connectivities_tree"]
    return {
        "finite_pseudotime_fraction": float(np.isfinite(pseudotime).mean()),
        "n_output_states": int(result.obs["smoke_cluster"].nunique()),
        "n_output_edges": int(tree.nnz // 2),
        "root_interface": "iroot supplied",
        "method_api": "scanpy.tl.paga + scanpy.tl.dpt",
        "shared_preprocessing": "fixed sklearn PCA+kNN+KMeans; truth-blind",
    }


def run_cellrank_on_dpt(adata: ad.AnnData, root_cell: str) -> dict[str, object]:
    from cellrank.estimators import GPCCA
    from cellrank.kernels import PseudotimeKernel

    full = prepare_scanpy(adata, 1102)
    adjacency = full.obsp["connectivities"].copy()
    graph = nx.from_scipy_sparse_array(adjacency)
    root_index = int(full.obs_names.get_loc(root_cell))
    component = sorted(nx.node_connected_component(graph, root_index))
    result = full[component].copy()
    coverage = len(component) / full.n_obs
    result.uns["iroot"] = int(result.obs_names.get_loc(root_cell))
    sc.tl.diffmap(result)
    sc.tl.dpt(result)
    kernel = PseudotimeKernel(result, time_key="dpt_pseudotime")
    kernel.compute_transition_matrix(show_progress_bar=False, n_jobs=1)
    estimator = GPCCA(kernel)
    n_states = min(6, max(2, int(result.obs["smoke_cluster"].nunique())))
    estimator.compute_schur(n_components=max(n_states + 1, 3))
    estimator.compute_macrostates(n_states=n_states, n_cells=3)
    transition = kernel.transition_matrix
    return {
        "finite_pseudotime_fraction": float(
            coverage * np.isfinite(result.obs["dpt_pseudotime"].to_numpy(float)).mean()
        ),
        "n_output_states": int(len(estimator.macrostates.cat.categories)),
        "n_output_edges": int(transition.nnz),
        "root_interface": "DPT iroot supplied",
        "method_api": "CellRank PseudotimeKernel + GPCCA; label is CellRank-on-DPT",
        "shared_preprocessing": "fixed sklearn PCA+kNN+KMeans; truth-blind",
    }


def run_palantir(adata: ad.AnnData, root_cell: str) -> dict[str, object]:
    import palantir

    result = adata.copy()
    sc.pp.normalize_total(result, target_sum=1e4)
    sc.pp.log1p(result)
    sc.pp.pca(result, n_comps=min(20, result.n_vars - 1, result.n_obs - 1))
    palantir.utils.run_diffusion_maps(result, n_components=10, knn=15)
    palantir.utils.determine_multiscale_space(result)
    multiscale = pd.DataFrame(
        result.obsm["DM_EigenVectors_multiscaled"], index=result.obs_names
    )
    palantir_result = palantir.core.run_palantir(
        multiscale,
        early_cell=root_cell,
        num_waypoints=min(250, result.n_obs),
        knn=15,
        n_jobs=1,
        seed=1103,
    )
    probabilities = palantir_result.branch_probs
    return {
        "finite_pseudotime_fraction": float(
            np.isfinite(palantir_result.pseudotime.to_numpy(float)).mean()
        ),
        "n_output_states": int(probabilities.shape[1]),
        "n_output_edges": 0,
        "root_interface": "early_cell supplied",
        "method_api": "palantir.core.run_palantir",
        "shared_preprocessing": "Palantir native diffusion map after fixed normalization/PCA",
    }


def run_stlearn(adata: ad.AnnData, root_cell: str) -> dict[str, object]:
    import stlearn

    result = prepare_scanpy(adata, 1104)
    result.obs["imagerow"] = np.asarray(result.obsm["spatial"])[:, 0]
    result.obs["imagecol"] = np.asarray(result.obsm["spatial"])[:, 1]
    root_cluster = str(result.obs.loc[root_cell, "smoke_cluster"])
    stlearn.spatial.trajectory.set_root(
        result, use_label="smoke_cluster", cluster=root_cluster
    )
    result.uns["iroot"] = int(result.obs_names.get_loc(root_cell))
    stlearn.spatial.trajectory.pseudotime(
        result,
        use_label="smoke_cluster",
        n_neighbors=15,
        eps=20,
        use_rep="X_pca",
        threshold=0.01,
        radius=50,
        use_sme=False,
        pseudotime_key="dpt_pseudotime",
        max_nodes=4,
        run_knn=False,
    )
    pseudotime = result.obs["dpt_pseudotime"].to_numpy(float)
    graph = result.uns.get("global_graph", nx.Graph())
    return {
        "finite_pseudotime_fraction": float(np.isfinite(pseudotime).mean()),
        "n_output_states": int(result.obs["smoke_cluster"].nunique()),
        "n_output_edges": int(graph.number_of_edges()) if hasattr(graph, "number_of_edges") else 0,
        "root_interface": "root cluster and iroot supplied",
        "method_api": "stlearn.spatial.trajectory.pseudotime",
        "shared_preprocessing": "fixed sklearn PCA+kNN+KMeans; truth-blind",
    }


def prepare_identity_metacells(
    adata: ad.AnnData, roots: list[str], output: Path
) -> tuple[Path, Path, Path, Path]:
    inputs = output / "nicheforest_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    meta = adata.copy()
    meta.obs_names = pd.Index([f"MC{i:04d}" for i in range(meta.n_obs)])
    meta.obs["source_cell_id"] = [str(value) for value in adata.obs_names]
    meta_path = inputs / "metacell_adata.h5ad"
    meta.write_h5ad(meta_path)
    mapping = pd.DataFrame(
        {
            "cell_index": np.arange(meta.n_obs),
            "cell_name": [str(value) for value in adata.obs_names],
            "metacell_index": np.arange(meta.n_obs),
            "metacell_id": meta.obs_names.astype(str),
            "source_group_key": "identity_smoke_only",
            "cell_annotation_label": "unknown",
            "cell_inferred_compartment": "unknown",
        }
    )
    mapping_path = inputs / "cell_to_metacell_mapping.csv"
    mapping.to_csv(mapping_path, index=False)
    coordinates = np.asarray(adata.obsm["spatial"], float)
    root_rows = []
    for index, cell in enumerate(roots):
        cell_index = int(adata.obs_names.get_loc(cell))
        root_rows.append(
            {
                "root_id": f"R{index}",
                "representative_cell_id": cell,
                "spatial_x": float(coordinates[cell_index, 0]),
                "spatial_y": float(coordinates[cell_index, 1]),
                "root_node": f"R{index}_ROOT",
            }
        )
    roots_path = inputs / "TRUE_ROOTS.csv"
    pd.DataFrame(root_rows).to_csv(roots_path, index=False)
    truth_root = output / "nicheforest_run"
    truth_dir = truth_root / "event" / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    root_ids = [f"R{index}" for index in range(len(roots))]
    origin_to_root = {
        str(origin): root_ids[index]
        for index, origin in enumerate(sorted(adata.obs["origin"].astype(str).unique()))
    }
    percentages = adata.uns["milestone_percentages"].copy()
    percentages["origin"] = percentages["milestone_id"].astype(str).str.extract(
        r"^O([^_]+)_", expand=False
    )
    percentages["root_id"] = percentages["origin"].map(origin_to_root)
    percentages = percentages.merge(
        mapping[["cell_name", "metacell_id"]],
        left_on="cell_id",
        right_on="cell_name",
        how="left",
        validate="many_to_one",
    )
    state_composition = percentages.rename(
        columns={"milestone_id": "state_id", "percentage": "state_probability"}
    )[["metacell_id", "root_id", "state_id", "state_probability"]]
    state_composition.insert(0, "stage", "T0")
    state_composition.to_csv(inputs / "METACELL_STATE_COMPOSITION.csv", index=False)
    root_composition_rows = []
    for row in mapping.itertuples(index=False):
        origin = str(adata.obs.iloc[int(row.cell_index)]["origin"])
        true_root = origin_to_root[origin]
        for root_id in root_ids:
            root_composition_rows.append(
                {
                    "metacell_id": str(row.metacell_id),
                    "root_id": root_id,
                    "root_probability": float(root_id == true_root),
                }
            )
    pd.DataFrame(root_composition_rows).to_csv(
        inputs / "METACELL_ROOT_COMPOSITION.csv", index=False
    )
    timecourse = np.asarray(adata.uns["timecourse"], float)
    pt_rows = []
    branch_rows = []
    for index, cell in enumerate(adata.obs_names.astype(str)):
        origin = str(adata.obs.iloc[index]["origin"])
        root_id = origin_to_root[origin]
        pt_rows.append(
            {"cell_id": cell, "root_id": root_id, "pseudotime": float(timecourse[index])}
        )
        branch_rows.append(
            {
                "stage": "T0",
                "cell_id": cell,
                "root_id": root_id,
                "branch_id": str(adata.obs.iloc[index]["branch"]),
            }
        )
    pd.DataFrame(pt_rows).to_csv(truth_dir / "TRUE_PSEUDOTIME.csv", index=False)
    pd.DataFrame(branch_rows).to_csv(truth_dir / "TRUE_CELL_BRANCH.csv", index=False)
    milestone = adata.uns["milestone_network"].copy()
    edge_rows = []
    for row in milestone.itertuples(index=False):
        origin = str(row[0]).split("_", maxsplit=1)[0][1:]
        edge_rows.append(
            {
                "root_id": origin_to_root[origin],
                "source": str(row[0]),
                "target": str(row[1]),
            }
        )
    pd.DataFrame(edge_rows).to_csv(truth_dir / "TRUE_T0_EDGES.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(truth_dir / "TRUE_T1_EDGES.csv", index=False)
    pd.DataFrame(
        columns=["event_id", "source_root", "receiver_root", "event_type", "topology_changed"]
    ).to_csv(truth_dir / "TRUE_EVENTS.csv", index=False)
    pd.DataFrame(columns=["event_id", "root_id", "source", "target"]).to_csv(
        truth_dir / "TRUE_ADDED_EDGES.csv", index=False
    )
    pd.DataFrame(columns=["event_id", "root_id", "source", "target"]).to_csv(
        truth_dir / "TRUE_REMOVED_EDGES.csv", index=False
    )
    return meta_path, mapping_path, roots_path, inputs


def run_nicheforest_t0_smoke(
    adata: ad.AnnData, roots: list[str], output: Path
) -> dict[str, object]:
    meta, mapping, truth_roots, truth_meta = prepare_identity_metacells(
        adata, roots, output
    )
    run = output / "nicheforest_run"
    upstream = run / "generic_upstream_v1_exact_root"
    phase_a = run / "phase_a"
    commands = []
    if not (upstream / "GENERIC_UPSTREAM_STATUS.json").exists():
        commands.append([
            sys.executable,
            str(GOLD / "run_generic_upstream_v1.py"),
            "--metacell-h5ad",
            str(meta),
            "--true-roots-csv",
            str(truth_roots),
            "--cell-to-metacell-csv",
            str(mapping),
            "--dataset",
            "sim_K2_sep",
            "--output-dir",
            str(upstream),
        ])
    commands.append([
            sys.executable,
            str(DEV / "run_phase_a_case_v7_state_constrained_fine.py"),
            "--run-dir",
            str(run),
            "--dataset",
            "sim_K2_sep",
            "--run-label",
            "sim_K2_sep_P2_smoke",
            "--output",
            str(phase_a),
            "--truth-run-dir",
            str(run),
            "--metacell-truth-dir",
            str(truth_meta),
            "--fixed-root-input",
            str(upstream / "FIXED_ROOT_INPUT.csv"),
            "--production-only",
        ])
    logs = []
    for command in commands:
        process = subprocess.run(
            command, cwd=REPO, capture_output=True, text=True, timeout=600
        )
        logs.extend([subprocess.list2cmdline(command), process.stdout, process.stderr])
        if process.returncode:
            raise RuntimeError(process.stderr[-5000:])
    (output / "NICHEFOREST_SMOKE_COMMANDS.log").write_text(
        "\n".join(logs), encoding="utf-8"
    )
    primary = pd.read_csv(phase_a / "T0_FINE_TRAJECTORY_EDGES.csv")
    nodes = pd.read_csv(phase_a / "T0_STATE_NODE_ASSIGNMENTS.csv")
    baseline = pd.read_csv(
        phase_a / "current_root_conditioned_msf" / "T0_FINE_TRAJECTORY_EDGES.csv"
    )
    return {
        "finite_pseudotime_fraction": float(
            np.isfinite(nodes["pseudotime"].to_numpy(float)).mean()
        ),
        "n_output_states": int(nodes["state_id"].nunique()),
        "n_output_edges": int(len(primary)),
        "root_interface": "fixed roots supplied equally to T0 and root-conditioned MSF",
        "method_api": (
            f"frozen Phase A v7.1; root-conditioned MSF edges={len(baseline)}"
        ),
        "shared_preprocessing": "NicheForest native candidate/soft/T0 pipeline",
    }


def validate_result(method: str, value: dict[str, object]) -> None:
    minimum_coverage = (
        0.95 if method == "current_T0_and_root_conditioned_MSF" else 0.25
    )
    if float(value["finite_pseudotime_fraction"]) < minimum_coverage:
        raise ValueError(f"Insufficient finite pseudotime: {value}")
    if int(value["n_output_states"]) < 1:
        raise ValueError(f"No output states: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--methods",
        default="current_T0_and_root_conditioned_MSF,PAGA+DPT,CellRank-on-DPT,Palantir,stLearn",
        help="Comma-separated subset of adapters to run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "outputs" / "NicheForest_paper_frozen_candidate" / "P2_adapter_smoke",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.dataset.resolve())
    roots = roots_from_truth(adata)
    adapters = [
        ("current_T0_and_root_conditioned_MSF", lambda: run_nicheforest_t0_smoke(adata, roots, output)),
        ("PAGA+DPT", lambda: run_paga_dpt(adata, roots[0])),
        ("CellRank-on-DPT", lambda: run_cellrank_on_dpt(adata, roots[0])),
        ("Palantir", lambda: run_palantir(adata, roots[0])),
        ("stLearn", lambda: run_stlearn(adata, roots[0])),
    ]
    selected = {value.strip() for value in args.methods.split(",") if value.strip()}
    unknown = selected - {name for name, _ in adapters}
    if unknown:
        raise ValueError(f"Unknown adapter names: {sorted(unknown)}")
    table_path = output / "P2_BASELINE_ADAPTER_SMOKE.csv"
    existing: dict[str, dict[str, object]] = {}
    if table_path.exists():
        existing = {
            str(row["method"]): dict(row)
            for row in pd.read_csv(table_path, keep_default_na=False).to_dict("records")
            if str(row["method"]) != "Slingshot"
        }
        for row in existing.values():
            row.setdefault("shared_preprocessing", "")
    for name, function in adapters:
        if name not in selected:
            continue
        started = time.perf_counter()
        try:
            result = function()
            validate_result(name, result)
            existing[name] = {
                    "method": name,
                    "dataset": args.dataset.stem,
                    "status": "PASS",
                    "seconds": round(time.perf_counter() - started, 3),
                    **result,
                    "error": "",
                }
        except Exception as exc:
            existing[name] = {
                    "method": name,
                    "dataset": args.dataset.stem,
                    "status": "FAIL",
                    "seconds": round(time.perf_counter() - started, 3),
                    "finite_pseudotime_fraction": "",
                    "n_output_states": "",
                    "n_output_edges": "",
                    "root_interface": "",
                    "method_api": "",
                    "shared_preprocessing": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            (output / f"{name.replace('+', '_').replace(' ', '_')}_ERROR.log").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        write_csv(table_path, list(existing.values()))
    rows = list(existing.values())
    rows.append(
        {
            "method": "Slingshot",
            "dataset": args.dataset.stem,
            "status": "NOT_RUN_NO_R_RUNTIME",
            "seconds": 0,
            "finite_pseudotime_fraction": "",
            "n_output_states": "",
            "n_output_edges": "",
            "root_interface": "native start.clus only when R is available",
            "method_api": "Bioconductor slingshot",
            "shared_preprocessing": "not_run",
            "error": "Rscript is not available; no surrogate is permitted",
        }
    )
    write_csv(table_path, rows)
    included = {
        "current_T0_and_root_conditioned_MSF",
        "PAGA+DPT",
        "CellRank-on-DPT",
        "Palantir",
        "stLearn",
    }
    passed = [row for row in rows if row["status"] == "PASS"]
    failed = [row for row in rows if row["status"] == "FAIL"]
    observed = {str(row["method"]) for row in rows}
    missing = sorted(included - observed)
    decision = {
        "scope": "P2_SMOKE_ONLY",
        "dataset": str(args.dataset.resolve()),
        "formal_benchmark_started": False,
        "n_pass": len(passed),
        "n_fail": len(failed),
        "missing_included_adapters": missing,
        "slingshot_excluded_no_runtime": True,
        "decision": "PASS" if not failed and not missing else "FAIL",
        "rule": "P3/P4 remain blocked until every included adapter passes",
        "package_versions": {
            package: version(package)
            for package in ("scanpy", "cellrank", "palantir", "stlearn")
        },
    }
    (output / "P2_DECISION.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    provenance_paths = [
        Path(__file__).resolve(),
        args.dataset.resolve(),
        REPO
        / "outputs"
        / "NicheForest_paper_frozen_candidate"
        / "BASELINE_COMPARISON_CONTRACT.csv",
        table_path,
        output / "P2_DECISION.json",
    ]
    write_csv(
        output / "P2_SHA256_MANIFEST.csv",
        [
            {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in provenance_paths
        ],
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
