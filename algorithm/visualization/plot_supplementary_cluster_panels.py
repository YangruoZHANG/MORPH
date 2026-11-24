#!/usr/bin/env python
"""Draw current formal all7 clusters with the approved legacy three-panel style.

Visual coordinates come from the historical dense plotting bundle only because
that bundle contains the approved spatial/UMAP presentation coordinates.
Cluster labels come exclusively from the frozen NicheForest reproduction bundle:
data/<dataset>/cell_cluster_mapping.csv.

The plotting coordinates never change the formal cluster assignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, to_rgb
from scipy.ndimage import gaussian_filter


DATASETS = [
    "GSE",
    "GSE193460",
    "GSE245582",
    "xenium_breast",
    "xenium_lung",
    "xenium_melanoma",
    "xenium_pancreatic",
]

DISPLAY_NAMES = {
    "GSE": "Pooled GSE",
    "GSE193460": "GSE193460",
    "GSE245582": "GSE245582",
    "xenium_breast": "Xenium breast",
    "xenium_lung": "Xenium lung",
    "xenium_melanoma": "Xenium melanoma",
    "xenium_pancreatic": "Xenium pancreatic",
}

# Copied unchanged from the approved auto-k detailed legacy script.
CLUSTER_COLORS = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CC3311",
    "#AA3377",
    "#66CCEE",
    "#EE7733",
    "#0077BB",
    "#BBBB22",
    "#33BBEE",
    "#EE3377",
    "#009988",
    "#997700",
    "#CC6677",
    "#882255",
    "#44AA99",
    "#DDCC77",
]

COMPARTMENT_ORDER = [
    "Tumor / epithelial",
    "Lymphoid",
    "Myeloid",
    "Endothelial / vascular",
    "Stromal / fibroblast",
    "Perivascular / smooth muscle",
    "Unknown",
]

COMPARTMENT_LABELS = {
    "tumor_epithelial": "Tumor / epithelial",
    "lymphoid": "Lymphoid",
    "myeloid": "Myeloid",
    "endothelial_vascular": "Endothelial / vascular",
    "stromal_fibroblast": "Stromal / fibroblast",
    "perivascular_smooth_muscle": "Perivascular / smooth muscle",
    "unknown": "Unknown",
}

INK = "#202124"
MUTED = "#6C737B"
LIGHT_BORDER = "#C8CDD2"


def get_cluster_colors(n_clusters: int) -> list[str]:
    if n_clusters <= len(CLUSTER_COLORS):
        return CLUSTER_COLORS[:n_clusters]
    colors = list(CLUSTER_COLORS)
    seen = {x.lower() for x in colors}
    for cmap_name in ["tab20", "tab20b", "tab20c", "Set3", "Dark2", "Paired"]:
        cmap = plt.get_cmap(cmap_name)
        for index in range(getattr(cmap, "N", 20)):
            rgba = cmap(index)
            color = "#{:02x}{:02x}{:02x}".format(
                *[int(round(channel * 255)) for channel in rgba[:3]]
            )
            if color.lower() not in seen:
                colors.append(color)
                seen.add(color.lower())
            if len(colors) >= n_clusters:
                return colors[:n_clusters]
    cmap = plt.get_cmap("hsv")
    while len(colors) < n_clusters:
        rgba = cmap(((len(colors) - len(CLUSTER_COLORS)) * 0.61803398875) % 1)
        color = "#{:02x}{:02x}{:02x}".format(
            *[int(round(channel * 255)) for channel in rgba[:3]]
        )
        if color.lower() not in seen:
            colors.append(color)
            seen.add(color.lower())
    return colors[:n_clusters]


def cluster_number(label: str) -> int:
    digits = "".join(x for x in str(label) if x.isdigit())
    return int(digits) if digits else 10**9


def normalize_cell_ids(values: pd.Series) -> np.ndarray:
    return (
        values.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .to_numpy()
    )


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "axes.edgecolor": LIGHT_BORDER,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def spatial_limits(xy: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    finite = xy[np.isfinite(xy).all(axis=1)]
    low = np.nanpercentile(finite, 0.25, axis=0)
    high = np.nanpercentile(finite, 99.75, axis=0)
    span = np.maximum(high - low, 1e-6)
    return (
        (float(low[0] - 0.02 * span[0]), float(high[0] + 0.02 * span[0])),
        (float(low[1] - 0.02 * span[1]), float(high[1] + 0.02 * span[1])),
    )


def embedding_limits(xy: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    finite = xy[np.isfinite(xy).all(axis=1)]
    low = np.nanpercentile(finite, 0.4, axis=0)
    high = np.nanpercentile(finite, 99.6, axis=0)
    span = np.maximum(high - low, 1e-6)
    return (
        (float(low[0] - 0.08 * span[0]), float(high[0] + 0.08 * span[0])),
        (float(low[1] - 0.08 * span[1]), float(high[1] + 0.08 * span[1])),
    )


def set_axis_frame(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(LIGHT_BORDER)
        spine.set_linewidth(0.7)


def add_scale_bar(ax: plt.Axes, xy: np.ndarray, dataset: str) -> None:
    finite = xy[np.isfinite(xy).all(axis=1)]
    low = np.nanpercentile(finite, 0.5, axis=0)
    high = np.nanpercentile(finite, 99.5, axis=0)
    width, height = high - low
    length = 0.065 * width
    x0 = low[0] + 0.055 * width
    y0 = low[1] + 0.055 * height
    ax.plot(
        [x0, x0 + length],
        [y0, y0],
        color="#252525",
        lw=1.6,
        solid_capstyle="butt",
        zorder=30,
    )
    label = "500 µm" if dataset.startswith("xenium_") else "spatial scale"
    ax.text(
        x0 + length / 2,
        y0 + 0.012 * height,
        label,
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#353535",
        zorder=30,
    )


def categorical_raster(
    xy: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int,
    smooth: float,
    alpha_max: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    valid = np.isfinite(xy).all(axis=1) & (labels >= 0) & (labels < n_clusters)
    width = max(xlim[1] - xlim[0], 1e-9)
    height = max(ylim[1] - ylim[0], 1e-9)
    nx = grid_size
    ny = max(200, int(round(grid_size * height / width)))
    counts = np.zeros((n_clusters, ny, nx), dtype=np.float32)
    ix = np.clip(((xy[valid, 0] - xlim[0]) / width * (nx - 1)).astype(int), 0, nx - 1)
    iy = np.clip(((xy[valid, 1] - ylim[0]) / height * (ny - 1)).astype(int), 0, ny - 1)
    np.add.at(counts, (labels[valid].astype(int), iy, ix), 1)
    if smooth > 0:
        for cluster in range(n_clusters):
            counts[cluster] = gaussian_filter(counts[cluster], sigma=smooth)
    total = counts.sum(axis=0)
    dominant = np.argmax(counts, axis=0)
    rgb = np.asarray([to_rgb(x) for x in get_cluster_colors(n_clusters)])
    rgba = np.zeros((ny, nx, 4), float)
    occupied = total > 1e-6
    rgba[occupied, :3] = rgb[dominant[occupied]]
    if occupied.any():
        ref = max(float(np.percentile(total[occupied], 98.5)), 1e-6)
        rgba[..., 3] = np.clip((total / ref) ** 0.55, 0, alpha_max)
    return rgba, (xlim[0], xlim[1], ylim[0], ylim[1])


def load_dataset(
    bundle: Path,
    dense_root: Path,
    dataset: str,
) -> dict[str, Any]:
    current = pd.read_csv(bundle / "data" / dataset / "cell_cluster_mapping.csv")
    mapping = pd.read_csv(bundle / "data" / dataset / "cell_to_metacell.csv")
    dense = pd.read_csv(dense_root / dataset / "raw_cell_dense_coordinates.csv.gz")

    current_ids = normalize_cell_ids(current["cell"])
    dense_ids = normalize_cell_ids(dense["cell_id"])
    mapping = mapping.sort_values("cell_index").reset_index(drop=True)
    mapping_ids = normalize_cell_ids(mapping["cell_name"])
    if not np.array_equal(current_ids, dense_ids):
        raise ValueError(f"{dataset}: current clusters and dense plotting table are not row-aligned")
    if not np.array_equal(current_ids, mapping_ids):
        raise ValueError(f"{dataset}: current clusters and exact cell-to-metacell mapping are not aligned")

    cluster_labels = sorted(current["cluster"].astype(str).unique(), key=cluster_number)
    cluster_to_index = {label: i for i, label in enumerate(cluster_labels)}
    labels = current["cluster"].astype(str).map(cluster_to_index).to_numpy(int)
    dense["formal_cluster"] = current["cluster"].astype(str).to_numpy()
    dense["cluster_index"] = labels
    dense["metacell_id"] = mapping["metacell_id"].astype(str).to_numpy()
    dense["formal_compartment"] = (
        mapping["cell_inferred_compartment"].fillna("unknown").astype(str).to_numpy()
    )

    cell_xy = dense[["spatial_1", "spatial_2"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    umap_xy = dense[["umap_1", "umap_2"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    meta = (
        dense.groupby("metacell_id", sort=False)
        .agg(
            x=("spatial_1", "mean"),
            y=("spatial_2", "mean"),
            n_cells=("cell_id", "size"),
            cluster_index=("cluster_index", lambda x: int(pd.Series(x).mode().iloc[0])),
        )
        .reset_index()
    )
    return {
        "dataset": dataset,
        "frame": dense,
        "cell_xy": cell_xy,
        "umap_xy": umap_xy,
        "labels": labels,
        "cluster_labels": cluster_labels,
        "meta": meta,
    }


def cluster_positions(xy: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    out = np.full((n_clusters, 2), np.nan)
    for cluster in range(n_clusters):
        selected = xy[(labels == cluster) & np.isfinite(xy).all(axis=1)]
        if len(selected):
            out[cluster] = np.nanmedian(selected, axis=0)
    return out


def draw_spatial_panel(ax: plt.Axes, data: dict[str, Any]) -> None:
    xy = data["cell_xy"]
    labels = data["labels"]
    cluster_labels = data["cluster_labels"]
    n_clusters = len(cluster_labels)
    colors = get_cluster_colors(n_clusters)
    xlim, ylim = spatial_limits(xy)
    finite = np.isfinite(xy).all(axis=1)
    ax.scatter(
        xy[finite, 0],
        xy[finite, 1],
        s=0.018,
        c="#C7CDD2",
        alpha=0.04,
        linewidths=0,
        rasterized=True,
        zorder=0,
    )
    raster, extent = categorical_raster(
        xy, labels, n_clusters, xlim, ylim, 650, 0, 0.82
    )
    ax.imshow(raster, extent=extent, origin="lower", interpolation="nearest", aspect="auto")

    rng = np.random.default_rng(20260703)
    parts = []
    for cluster in range(n_clusters):
        idx = np.flatnonzero(finite & (labels == cluster))
        if len(idx) > 4500:
            idx = rng.choice(idx, 4500, replace=False)
        parts.append(idx)
    overlay = np.concatenate(parts)
    ax.scatter(
        xy[overlay, 0],
        xy[overlay, 1],
        s=0.055,
        c=[colors[x] for x in labels[overlay]],
        alpha=0.56,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    meta = data["meta"]
    transformed = np.sqrt(np.maximum(meta["n_cells"].to_numpy(float), 1))
    sizes = 4.5 + 8.0 * (transformed - transformed.min()) / max(np.ptp(transformed), 1e-9)
    ax.scatter(
        meta["x"],
        meta["y"],
        s=sizes,
        c="#222222",
        edgecolors="white",
        linewidths=0.25,
        alpha=0.48,
        zorder=5,
    )
    for index, pos in enumerate(cluster_positions(xy, labels, n_clusters)):
        if np.isfinite(pos).all():
            ax.text(
                pos[0],
                pos[1],
                cluster_labels[index],
                ha="center",
                va="center",
                fontsize=5.4,
                fontweight="bold",
                color=INK,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": colors[index],
                    "linewidth": 0.75,
                    "alpha": 0.94,
                },
                zorder=8,
            )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    set_axis_frame(ax)
    add_scale_bar(ax, xy, data["dataset"])
    ax.text(
        0.985,
        0.018,
        "colors: current formal clusters · black dots: current metacell centroids",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.2,
        color=MUTED,
    )


def draw_umap_panel(ax: plt.Axes, data: dict[str, Any]) -> None:
    xy = data["umap_xy"]
    labels = data["labels"]
    cluster_labels = data["cluster_labels"]
    n_clusters = len(cluster_labels)
    colors = get_cluster_colors(n_clusters)
    xlim, ylim = embedding_limits(xy)
    raster, extent = categorical_raster(
        xy, labels, n_clusters, xlim, ylim, 520, 2.2, 0.64
    )
    ax.imshow(raster, extent=extent, origin="lower", interpolation="bilinear", aspect="auto")
    rng = np.random.default_rng(20260703)
    for cluster in range(n_clusters):
        idx = np.flatnonzero(labels == cluster)
        if len(idx) > 9000:
            idx = rng.choice(idx, 9000, replace=False)
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=0.18,
            c=colors[cluster],
            alpha=0.36,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
    for index, pos in enumerate(cluster_positions(xy, labels, n_clusters)):
        if np.isfinite(pos).all():
            ax.text(
                pos[0],
                pos[1],
                cluster_labels[index],
                ha="center",
                va="center",
                fontsize=5.4,
                fontweight="bold",
                color=INK,
                bbox={
                    "boxstyle": "round,pad=0.11",
                    "facecolor": "white",
                    "edgecolor": colors[index],
                    "linewidth": 0.72,
                    "alpha": 0.93,
                },
                zorder=8,
            )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1", fontsize=6.4, labelpad=1)
    ax.set_ylabel("UMAP 2", fontsize=6.4, labelpad=1)
    for spine in ax.spines.values():
        spine.set_color(LIGHT_BORDER)
        spine.set_linewidth(0.7)
    ax.text(
        0.985,
        0.018,
        "raw cells; legacy UMAP coordinates used for display only",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=MUTED,
    )


def composition_tables(data: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    frame = data["frame"]
    labels = data["labels"]
    cluster_labels = data["cluster_labels"]
    matrix = np.zeros((len(COMPARTMENT_ORDER), len(cluster_labels)), float)
    summary = []
    for cluster, cluster_label in enumerate(cluster_labels):
        mask = labels == cluster
        compartments = (
            frame.loc[mask, "formal_compartment"]
            .map(lambda x: COMPARTMENT_LABELS.get(str(x), "Unknown"))
        )
        counts = compartments.value_counts()
        total = int(mask.sum())
        for row, label in enumerate(COMPARTMENT_ORDER):
            matrix[row, cluster] = float(counts.get(label, 0)) / max(total, 1)
        dominant = int(np.argmax(matrix[:, cluster]))
        summary.append(
            {
                "dataset": data["dataset"],
                "cluster": cluster_label,
                "n_cells": total,
                "n_metacells": int(
                    data["meta"].loc[data["meta"]["cluster_index"] == cluster].shape[0]
                ),
                "dominant_formal_compartment": COMPARTMENT_ORDER[dominant],
                "purity": float(matrix[dominant, cluster]),
            }
        )
    keep = (matrix.sum(axis=1) > 0.01) | (
        np.asarray(COMPARTMENT_ORDER) == "Unknown"
    )
    composition = pd.DataFrame(
        matrix[keep],
        index=np.asarray(COMPARTMENT_ORDER)[keep],
        columns=cluster_labels,
    )
    return composition, matrix.max(axis=0), pd.DataFrame(summary)


def draw_color_strip(ax: plt.Axes, labels: list[str]) -> None:
    n = len(labels)
    ax.imshow(
        np.arange(n, dtype=float)[None, :],
        cmap=ListedColormap(get_cluster_colors(n)),
        vmin=-0.5,
        vmax=n - 0.5,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_yticks([])
    ax.set_xticks(range(n), labels, fontsize=5.0, rotation=90 if n > 55 else 0)
    ax.tick_params(axis="x", length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_purity_bar(ax: plt.Axes, purity: np.ndarray) -> None:
    n = len(purity)
    ax.bar(range(n), purity, width=0.78, color="#597DA3", edgecolor="none")
    ax.axhline(0.65, color="#B23A48", lw=0.8, ls="--")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([0, 0.5, 1], ["0", ".5", "1"], fontsize=5.8)
    ax.set_ylabel("Purity", fontsize=6.5, labelpad=2)
    ax.tick_params(axis="y", length=2, width=0.6, color="#777777")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B9B9B9")
    ax.spines[["left", "bottom"]].set_linewidth(0.6)
    ax.text(n - 0.55, 0.67, "0.65", ha="right", va="bottom", fontsize=5.4, color="#A02D3C")


def draw_composition_heatmap(
    ax: plt.Axes,
    composition: pd.DataFrame,
    figure: plt.Figure,
) -> None:
    matrix = composition.to_numpy(float)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks(range(len(composition.index)), composition.index, fontsize=5.8)
    ax.tick_params(length=0, axis="y", pad=1)
    for spine in ax.spines.values():
        spine.set_color("#C8C8C8")
        spine.set_linewidth(0.6)
    for col in range(matrix.shape[1]):
        row = int(np.argmax(matrix[:, col]))
        value = float(matrix[row, col])
        if value >= 0.30:
            ax.text(
                col,
                row,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=4.6,
                color="white" if value >= 0.52 else INK,
            )
    colorbar = figure.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Cell fraction", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=5.8, length=2)
    colorbar.outline.set_linewidth(0.5)


def panel_title(ax: plt.Axes, letter: str, title: str, y: float = 1.025) -> None:
    ax.text(
        -0.02,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.052,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.4,
        fontweight="semibold",
        color=INK,
    )


def make_figure(
    bundle: Path,
    dense_root: Path,
    output_root: Path,
    dataset: str,
    dpi: int,
) -> dict[str, Any]:
    data = load_dataset(bundle, dense_root, dataset)
    composition, purity, summary = composition_tables(data)
    n_clusters = len(data["cluster_labels"])
    out = output_root / dataset / "stages"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "02A_current_formal_clusters_three_panel"
    summary.to_csv(out / "02A_current_formal_cluster_summary.csv", index=False)

    fig = plt.figure(figsize=(18.6, 5.8), facecolor="white")
    outer = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.58, 1.28, 0.10, 1.27],
        left=0.030,
        right=0.984,
        top=0.89,
        bottom=0.17,
        wspace=0.06,
    )
    ax_spatial = fig.add_subplot(outer[0, 0])
    ax_umap = fig.add_subplot(outer[0, 1])
    comp = outer[0, 3].subgridspec(3, 1, height_ratios=[0.10, 0.17, 0.73], hspace=0.20)
    ax_strip = fig.add_subplot(comp[0, 0])
    ax_purity = fig.add_subplot(comp[1, 0])
    ax_heatmap = fig.add_subplot(comp[2, 0])

    draw_spatial_panel(ax_spatial, data)
    draw_umap_panel(ax_umap, data)
    draw_color_strip(ax_strip, data["cluster_labels"])
    draw_purity_bar(ax_purity, purity)
    draw_composition_heatmap(ax_heatmap, composition, fig)

    display = DISPLAY_NAMES[dataset]
    panel_title(
        ax_spatial,
        "A",
        f"{display}: spatial projection of current formal clusters (k={n_clusters})",
    )
    panel_title(ax_umap, "B", "2-D expression-space projection")
    ax_strip.text(
        -0.11,
        1.72,
        "C",
        transform=ax_strip.transAxes,
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    ax_strip.text(
        0,
        1.72,
        "Post-hoc formal-compartment composition",
        transform=ax_strip.transAxes,
        fontsize=9.4,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.030,
        0.050,
        "Cluster assignments are read unchanged from the frozen v7.4.6 cell_cluster_mapping.csv. "
        "Spatial and UMAP coordinates are visualization coordinates only and do not alter assignments. "
        "Cell labels are used only for post-hoc compartment composition.",
        fontsize=6.4,
        color=MUTED,
        ha="left",
        va="bottom",
    )

    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=max(300, dpi), bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "dataset": dataset,
        "status": "complete",
        "n_cells": len(data["frame"]),
        "n_clusters": n_clusters,
        "cell_id_alignment": "exact",
        "png": str(png.relative_to(output_root)),
        "pdf": str(pdf.relative_to(output_root)),
        "svg": str(svg.relative_to(output_root)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("NicheForest_all7_reproduction_bundle"),
    )
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/CB_figure_legacy_style_rebuild/01_clustering"),
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_publication_style()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for dataset in args.datasets:
        try:
            print(f"[START] {dataset}", flush=True)
            result = make_figure(
                args.bundle.resolve(),
                args.dense_root.resolve(),
                args.output.resolve(),
                dataset,
                args.dpi,
            )
        except Exception as exc:
            result = {
                "dataset": dataset,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        manifest.append(result)
        print(f"[{result['status'].upper()}] {dataset}", flush=True)
        pd.DataFrame(manifest).to_csv(args.output / "run_manifest.csv", index=False)
    final = {"status": "complete" if all(x["status"] == "complete" for x in manifest) else "partial"}
    (args.output / "STATUS.json").write_text(
        json.dumps(final | {"datasets": manifest}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0 if final["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
