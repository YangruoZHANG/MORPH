#!/usr/bin/env python3
"""Two-panel angular trajectory preview for all frozen xenium_breast roots.

The same formal representative MST edges are drawn in tissue and UMAP space.
Cell colors use the formal cell-to-metacell map and component pseudotime.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as path_effects
import networkx as nx
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
DATASET = "xenium_breast"
ROOTS = ["C13", "C00", "C21"]
ROOT_LABELS = {"C13": "R0 · C13", "C00": "R1 · C00", "C21": "R2 · C21"}
LINE = {"C13": "#c62838", "C00": "#1656c0", "C21": "#087a48"}
STOPS = {
    "C13": ("#7b001a", "#efa5ae"),
    "C00": ("#07357c", "#a8c7fa"),
    "C21": ("#064a2d", "#a5d8b6"),
}
BASE = PROJECT / "outputs" / "CB_figure_v7_8_1_FROZEN_trajectory_embedding_connected_rebuild"
AUDIT_PATH = BASE / "05_tree_refinement_before_after_v6_2" / DATASET / "connected_representative_display_audit.csv"
EMBED = PROJECT / "outputs" / "v7_8_1_frozen_A0_trajectory_aware_embedding" / DATASET
DIRECTION = PROJECT / "outputs" / "v7_8_1_frozen_direction" / DATASET
TREE = PROJECT / "outputs" / "v7_8_1_frozen_formal_tree_inputs" / DATASET
FORMAL_DATA = PROJECT / "NicheForest_v7_4_6_all7_reproduction_bundle" / "data" / DATASET
OUTPUT = PROJECT / "outputs" / "trajectory_visualization_preview_xenium_breast" / "angular_all_roots"


def key(u: int, v: int) -> tuple[int, int]:
    return (min(int(u), int(v)), max(int(u), int(v)))


def load_cells() -> pd.DataFrame:
    cells = pd.read_csv(
        EMBED / "raw_cell_dense_coordinates.csv.gz",
        usecols=["cell_id", "spatial_1", "spatial_2", "umap_1", "umap_2"],
    )
    mapping = pd.read_csv(
        FORMAL_DATA / "cell_to_metacell.csv",
        usecols=["cell_name", "metacell_id"],
    )
    nodes = pd.read_csv(
        FORMAL_DATA / "biological_node_domains.csv",
        usecols=["node_name", "node_id"],
    )
    cells["cell_id"] = cells.cell_id.astype(str)
    mapping["cell_name"] = mapping.cell_name.astype(str)
    cells = cells.merge(mapping, left_on="cell_id", right_on="cell_name",
                        how="left", validate="one_to_one")
    cells = cells.merge(nodes, left_on="metacell_id", right_on="node_name",
                        how="left", validate="many_to_one")
    if cells.node_id.isna().any() or len(cells) != 167780:
        raise ValueError("The raw cells do not align completely to formal metacells")
    cells["node_id"] = cells.node_id.astype(int)
    return cells.drop(columns=["cell_name", "metacell_id", "node_name"])


def load_formal_paths():
    selected = pd.read_csv(AUDIT_PATH)
    formal_edges = pd.read_csv(DIRECTION / "DIRECTED_COMPONENT_MST_EDGES.csv")
    formal_pt = pd.read_csv(DIRECTION / "NODE_PSEUDOTIME_BY_ROOT_COMPONENT.csv")
    u = pd.read_csv(TREE / "U_BY_NODE.csv").set_index("node_id")
    paths = {}
    pt_parts = []
    for root in ROOTS:
        row = selected.loc[selected.root_id.eq(root)].iloc[0]
        component = int(row.component_id)
        hub = int(row.display_hub_node)
        terminals = [int(x) for x in str(row.requested_terminals).split("|")]
        edges = formal_edges.loc[
            formal_edges.root_cluster.eq(root)
            & formal_edges.component_id.eq(component)
            & formal_edges.component_hash.eq(row.component_hash)
        ].copy()
        graph = nx.Graph()
        graph.add_edges_from(zip(edges.node_u.astype(int), edges.node_v.astype(int)))
        if not nx.is_tree(graph) or hub not in graph:
            raise ValueError(f"{root}: invalid selected formal tree")
        display = nx.Graph()
        for terminal in terminals:
            display.add_edges_from(nx.utils.pairwise(nx.shortest_path(graph, hub, terminal)))
        lookup = {key(int(e.node_u), int(e.node_v)): e for e in edges.itertuples()}
        if not all(key(a, b) in lookup for a, b in display.edges):
            raise AssertionError(f"{root}: display edge missing from formal tree")
        paths[root] = dict(hub=hub, terminals=terminals, graph=display,
                           lookup=lookup, component=component)

        pt = formal_pt.loc[
            formal_pt.root_cluster.eq(root)
            & formal_pt.component_id.eq(component)
            & formal_pt.component_hash.eq(row.component_hash),
            ["node_id", "constrained_pseudotime"],
        ].copy()
        if not set(display.nodes).issubset(set(pt.node_id)):
            raise AssertionError(f"{root}: missing formal pseudotime")
        pt["root"] = root
        pt["U"] = pt.node_id.map(u[f"U_{root}"])
        values = pt.constrained_pseudotime.astype(float)
        pt["pt_display"] = 0.0
        positive = values.gt(0)
        pt.loc[positive, "pt_display"] = (
            0.07 + 0.93 * values.loc[positive].rank(method="average", pct=True)
        )
        pt_parts.append(pt[["node_id", "root", "U", "pt_display"]])
    # Some formal components overlap. Give a cell the root with the highest
    # membership among the three selected components; other cells stay grey.
    all_root_pt = pd.concat(pt_parts, ignore_index=True)
    node_colors = (all_root_pt.sort_values(["node_id", "U", "root"],
                                          ascending=[True, False, True])
                  .drop_duplicates("node_id"))
    return paths, node_colors, all_root_pt


def cell_rgba(cells: pd.DataFrame) -> np.ndarray:
    rgba = np.empty((len(cells), 4), dtype=float)
    rgba[:] = to_rgba("#c9ced4", alpha=0.24)
    for root in ROOTS:
        mask = cells.root.eq(root).to_numpy()
        if not mask.any():
            continue
        t = cells.loc[mask, "pt_display"].to_numpy(float) ** 0.76
        early = np.array(to_rgba(STOPS[root][0])[:3])
        late = np.array(to_rgba(STOPS[root][1])[:3])
        rgba[mask, :3] = early[None, :] * (1 - t[:, None]) + late[None, :] * t[:, None]
        rgba[mask, 3] = 0.74
    return rgba


def draw_edges(ax, paths, positions: pd.DataFrame, xcol: str, ycol: str,
               roots_to_draw=None, arrow_limit=7, umap_spacing=0.042):
    arrow_rows = []
    extent = np.ptp(positions[[xcol, ycol]].to_numpy(float), axis=0)
    diagonal = float(np.linalg.norm(extent))
    for root in (roots_to_draw if roots_to_draw is not None else ROOTS):
        item = paths[root]
        for u, v in item["graph"].edges:
            p = positions.loc[[u, v], [xcol, ycol]].to_numpy(float)
            formal = item["lookup"][key(u, v)]
            inferred = formal.direction_status == "exact_anchor_outward"
            ax.plot(p[:, 0], p[:, 1], color=LINE[root],
                    lw=2.05 if inferred else 1.45,
                    alpha=0.90 if inferred else 0.53,
                    solid_capstyle="round", zorder=5 if inferred else 4)

        # Select arrow locations from inferred formal edges only. Spatial and
        # UMAP panels may choose different visible edges, but both use exactly
        # the same frozen orientation table.
        candidates = []
        for u, v in item["graph"].edges:
            formal = item["lookup"][key(u, v)]
            if formal.direction_status != "exact_anchor_outward":
                continue
            a = positions.loc[int(formal.oriented_source), [xcol, ycol]].to_numpy(float)
            b = positions.loc[int(formal.oriented_target), [xcol, ycol]].to_numpy(float)
            length = float(np.linalg.norm(b - a))
            if length > 0.012 * diagonal:
                candidates.append((length, int(formal.oriented_source),
                                   int(formal.oriented_target), a, b))
        candidates.sort(reverse=True)
        shown = []
        for length, source, target, a, b in candidates:
            middle = (a + b) / 2
            min_spacing = 0.070 if xcol == "spatial_1" else umap_spacing
            if any(np.linalg.norm(middle - other) < min_spacing * diagonal for other in shown):
                continue
            p0 = a + 0.35 * (b - a)
            p1 = a + 0.67 * (b - a)
            arrow = FancyArrowPatch(
                tuple(p0), tuple(p1), arrowstyle="-|>",
                mutation_scale=11.5, lw=1.35,
                facecolor=LINE[root], edgecolor=LINE[root], zorder=8,
            )
            arrow.set_path_effects([
                path_effects.Stroke(linewidth=3.4, foreground="white"),
                path_effects.Normal(),
            ])
            ax.add_patch(arrow)
            shown.append(middle)
            arrow_rows.append({"panel": xcol, "root": root,
                               "source": source, "target": target,
                               "direction_status": "exact_anchor_outward"})
            if len(shown) >= arrow_limit:
                break
        hub = item["hub"]
        x, y = positions.loc[hub, [xcol, ycol]]
        ax.scatter(x, y, s=95, marker="*", c=LINE[root], edgecolors="white",
                   linewidths=0.95, zorder=10)
    return arrow_rows


def add_gradients(fig):
    grad = np.linspace(0, 1, 256)
    for i, root in enumerate(ROOTS):
        ax = fig.add_axes([0.09 + i * 0.302, 0.075, 0.23, 0.018])
        a = np.array(to_rgba(STOPS[root][0])[:3])
        b = np.array(to_rgba(STOPS[root][1])[:3])
        rgb = a[None, :] * (1 - grad[:, None]) + b[None, :] * grad[:, None]
        ax.imshow(rgb[None, :, :], aspect="auto", extent=(0, 1, 0, 1))
        ax.set_xticks([0, 1], ["early", "late"], fontsize=7)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.text(0.09 + i * 0.302, 0.105, ROOT_LABELS[root], color=LINE[root],
                 fontsize=8.5, fontweight="bold")


def add_single_gradient(fig, root):
    grad = np.linspace(0, 1, 256)
    a = np.array(to_rgba(STOPS[root][0])[:3])
    b = np.array(to_rgba(STOPS[root][1])[:3])
    rgb = a[None, :] * (1 - grad[:, None]) + b[None, :] * grad[:, None]
    ax = fig.add_axes([0.20, 0.105, 0.60, 0.024])
    ax.imshow(rgb[None, :, :], aspect="auto", extent=(0, 1, 0, 1))
    ax.set_xticks([0, 1], ["early", "late"], fontsize=8)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.20, 0.145, f"{ROOT_LABELS[root]} pseudotime", color=LINE[root],
             fontsize=9, fontweight="bold")


def main():
    mpl.use("Agg")
    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_cells = load_cells()
    paths, node_colors, all_root_pt = load_formal_paths()
    cells = raw_cells.merge(node_colors[["node_id", "root", "pt_display"]],
                        on="node_id", how="left", validate="many_to_one")
    colors = cell_rgba(cells)
    positions = cells.groupby("node_id").agg(
        spatial_1=("spatial_1", "mean"), spatial_2=("spatial_2", "mean"),
    )
    selected_umap = pd.read_csv(EMBED / f"{DATASET}_metacells_selected_umap.csv")
    positions = positions.join(selected_umap.set_index("node_id")[["umap_1", "umap_2"]],
                               validate="one_to_one")
    if positions[["spatial_1", "spatial_2", "umap_1", "umap_2"]].isna().any().any():
        raise ValueError("Missing spatial or embedding node position")

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 7.3), facecolor="white")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.91, bottom=0.16, wspace=0.055)
    panel_specs = [
        (axes[0], "spatial_1", "spatial_2", "a  Original tissue section"),
        (axes[1], "umap_1", "umap_2", "b  Trajectory-aware UMAP"),
    ]
    arrow_rows = []
    for ax, xcol, ycol, title in panel_specs:
        ax.scatter(cells[xcol], cells[ycol], s=1.25, c=colors,
                   linewidths=0, rasterized=True, zorder=1)
        arrow_rows.extend(draw_edges(ax, paths, positions, xcol, ycol))
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_xlim(0, 7600)
    axes[0].set_ylim(5550, 0)  # Image row increases downward.
    axes[1].set_xlim(0.1, 9.85)
    axes[1].set_ylim(6.55, 16.45)
    fig.suptitle("xenium_breast  |  all-root angular trajectories", x=0.025,
                 ha="left", fontsize=15, fontweight="bold")
    add_gradients(fig)
    fig.text(0.025, 0.018,
             "Cells outside the displayed components are grey. Star = exact anchor; arrows = formally inferred direction; lines without arrows may cross unresolved multi-anchor boundaries.",
             fontsize=8, color="#48515d")
    stem = OUTPUT / "xenium_breast_all_roots_spatial_umap_angular"
    fig.savefig(stem.with_suffix(".png"), dpi=270)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)

    individual_rows = []
    for root in ROOTS:
        root_pt = all_root_pt.loc[all_root_pt.root.eq(root),
                                  ["node_id", "pt_display"]]
        one = raw_cells.merge(root_pt, on="node_id", how="left",
                              validate="many_to_one")
        one["root"] = np.where(one.pt_display.notna(), root, None)
        one_colors = cell_rgba(one)
        root_fig, root_ax = plt.subplots(figsize=(6.8, 7.4), facecolor="white")
        root_fig.subplots_adjust(left=0.035, right=0.965, top=0.91, bottom=0.20)
        root_ax.scatter(one.umap_1, one.umap_2, s=2.0, c=one_colors,
                        linewidths=0, rasterized=True, zorder=1)
        one_arrows = draw_edges(root_ax, paths, positions, "umap_1", "umap_2",
                                roots_to_draw=[root], arrow_limit=11,
                                umap_spacing=0.027)
        root_ax.set_aspect("equal", adjustable="box")
        root_ax.set_xlim(0.1, 9.85)
        root_ax.set_ylim(6.55, 16.45)
        root_ax.set_xticks([])
        root_ax.set_yticks([])
        for spine in root_ax.spines.values():
            spine.set_visible(False)
        root_fig.suptitle(f"xenium_breast  |  {ROOT_LABELS[root]} trajectory",
                          x=0.035, ha="left", fontsize=13, fontweight="bold")
        add_single_gradient(root_fig, root)
        root_fig.text(0.035, 0.035,
                      "Grey: cells outside this root's displayed component. Star: exact anchor. Arrows: formally inferred direction.",
                      fontsize=8, color="#48515d")
        root_stem = OUTPUT / f"xenium_breast_{root}_umap_angular"
        root_fig.savefig(root_stem.with_suffix(".png"), dpi=300)
        root_fig.savefig(root_stem.with_suffix(".pdf"))
        root_fig.savefig(root_stem.with_suffix(".svg"))
        plt.close(root_fig)

        node_xy = positions.loc[list(paths[root]["graph"].nodes),
                                ["umap_1", "umap_2"]].to_numpy(float)
        lower = node_xy.min(axis=0)
        upper = node_xy.max(axis=0)
        pad = np.maximum(0.38, 0.10 * (upper - lower))
        zoom_fig, zoom_ax = plt.subplots(figsize=(7.2, 6.4), facecolor="white")
        zoom_fig.subplots_adjust(left=0.045, right=0.955, top=0.90, bottom=0.21)
        zoom_ax.scatter(one.umap_1, one.umap_2, s=2.0, c=one_colors,
                        linewidths=0, rasterized=True, zorder=1)
        draw_edges(zoom_ax, paths, positions, "umap_1", "umap_2",
                   roots_to_draw=[root], arrow_limit=11, umap_spacing=0.027)
        zoom_ax.set_aspect("equal", adjustable="box")
        zoom_ax.set_xlim(lower[0] - pad[0], upper[0] + pad[0])
        zoom_ax.set_ylim(lower[1] - pad[1], upper[1] + pad[1])
        zoom_ax.set_xticks([])
        zoom_ax.set_yticks([])
        for spine in zoom_ax.spines.values():
            spine.set_visible(False)
        zoom_fig.suptitle(f"xenium_breast  |  {ROOT_LABELS[root]} trajectory detail",
                          x=0.045, ha="left", fontsize=13, fontweight="bold")
        add_single_gradient(zoom_fig, root)
        zoom_fig.text(0.045, 0.035,
                      "Grey: other cells. Star: exact anchor. Arrows: formally inferred direction.",
                      fontsize=8, color="#48515d")
        zoom_stem = OUTPUT / f"xenium_breast_{root}_umap_angular_zoom"
        zoom_fig.savefig(zoom_stem.with_suffix(".png"), dpi=300)
        zoom_fig.savefig(zoom_stem.with_suffix(".pdf"))
        zoom_fig.savefig(zoom_stem.with_suffix(".svg"))
        plt.close(zoom_fig)
        individual_rows.append({"root": root,
                                "colored_cells": int(one.root.notna().sum()),
                                "grey_cells": int(one.root.isna().sum()),
                                "display_edges": paths[root]["graph"].number_of_edges(),
                                "arrows_umap": len(one_arrows)})

    pd.DataFrame(individual_rows).to_csv(OUTPUT / "individual_root_summary.csv",
                                          index=False)

    pd.DataFrame(arrow_rows).to_csv(OUTPUT / "display_arrows_audit.csv", index=False)
    cell_counts = cells.root.fillna("grey").value_counts().rename_axis("color_class")
    cell_counts.to_csv(OUTPUT / "background_cell_color_counts.csv", header=["n_cells"])
    summary = pd.DataFrame([{
        "root": root, "component_id": item["component"],
        "display_edges": item["graph"].number_of_edges(),
        "inferred_direction_display_edges": sum(
            item["lookup"][key(a, b)].direction_status == "exact_anchor_outward"
            for a, b in item["graph"].edges
        ),
        "arrows_spatial": sum(r["root"] == root and r["panel"] == "spatial_1" for r in arrow_rows),
        "arrows_umap": sum(r["root"] == root and r["panel"] == "umap_1" for r in arrow_rows),
    } for root, item in paths.items()])
    summary.to_csv(OUTPUT / "display_summary.csv", index=False)
    (OUTPUT / "figure_note.json").write_text(json.dumps({
        "dataset": DATASET,
        "cell_count": len(cells),
        "cell_mapping": "formal cell_to_metacell joined by cell ID, then biological node ID",
        "active_cell_rule": "node is in one of the three selected formal components; overlaps assigned to largest formal root U",
        "pseudotime": "rank of positive formal constrained_pseudotime within selected root component; zero anchors stay at darkest color",
        "trajectory": "identical selected frozen MST edges in spatial and UMAP panels, drawn as angular segments",
        "direction": "sparse arrow cues only on selected edges with exact_anchor_outward status",
        "scope": "one highest-U-mass representative formal component per root, not every formal component",
        "individual_umap": "three separate root-specific panels, each colored by that root's formal pseudotime; other cells grey",
        "individual_zoom": "additional per-root UMAP crops around the displayed path nodes; same source cells, edges and directions",
        "line_width": "2.05 pt for inferred edges and 1.45 pt for unresolved edges",
        "visualization_only": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(stem)
    print(summary.to_string(index=False))
    print(cell_counts.to_string())


if __name__ == "__main__":
    main()
