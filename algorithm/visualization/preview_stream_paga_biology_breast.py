#!/usr/bin/env python3
"""Biology-first, linked-view preview of frozen breast trajectories.

The categorical cell background is a *marker-program call*, not a validated
cell-type annotation. The right panel retains the frozen root-specific graph.
Branch composition excludes nodes shared by multiple displayed routes.
"""

from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import networkx as nx
import numpy as np
import pandas as pd

import preview_angular_all_roots_dualspace_breast as base


EXPR = (base.PROJECT / "outputs" / "reports_FROZEN"
        / "trajectory_evidence_frozen_A0" / "node_expression_cache"
        / "xenium_breast_node_expression.npz")
OUT = base.OUTPUT / "biology_first_legend"
PROGRAMS = {
    "C13": [
        ("Luminal epithelial", "FOXA1 · GATA3 · MLPH", ["FOXA1", "GATA3", "MLPH"], "#5c4a97"),
        ("Cycling epithelial", "MKI67 · TOP2A · CENPF", ["MKI67", "TOP2A", "CENPF"], "#cf7435"),
        ("CLDN4/CEACAM6-rich", "CLDN4 · CEACAM6 · KLF5", ["CLDN4", "CEACAM6", "KLF5"], "#c53f60"),
    ],
    "C00": [
        ("IL7R/CCR7-rich T-like", "IL7R · CCR7 · LTB", ["IL7R", "CCR7", "LTB"], "#574da2"),
        ("CCL5/GZMA-rich", "CCL5 · GZMA · CD69", ["CCL5", "GZMA", "CD69"], "#d07e32"),
        ("CD8/NKG7-rich", "CD8A · NKG7 · PRF1", ["CD8A", "NKG7", "PRF1"], "#087c8d"),
    ],
    "C21": [
        ("IL7R/CCR7-rich T-like", "IL7R · CCR7 · LTB", ["IL7R", "CCR7", "LTB"], "#574da2"),
        ("CCL5/GZMA-rich", "CCL5 · GZMA · CD69", ["CCL5", "GZMA", "CD69"], "#d07e32"),
        ("CD8/NKG7-rich", "CD8A · NKG7 · PRF1", ["CD8A", "NKG7", "PRF1"], "#087c8d"),
    ],
}
MIXED = "#9aa4ae"
PROCESS = {
    "C13": "Luminal epithelial-state heterogeneity: cycling and CLDN4/CEACAM6-rich branches",
    "C00": "T-cell marker gradient: IL7R/CCR7-rich to cytotoxic-associated states",
    "C21": "IL7R-rich T-cell branches within the shared immune-state component",
}
SHORT = {
    "C13": ["luminal", "cycling", "CLDN4"],
    "C00": ["IL7R", "CCL5", "CD8/NKG7"],
    "C21": ["IL7R", "CCL5", "CD8/NKG7"],
}


def call_programs(root, component_nodes, expression, gene_index):
    nodes = np.array(sorted(component_nodes), dtype=int)
    matrix = expression[nodes]
    scores = []
    for _, _, genes, _ in PROGRAMS[root]:
        data = matrix[:, [gene_index[g] for g in genes]]
        median = np.median(data, axis=0)
        iqr = np.quantile(data, .75, axis=0) - np.quantile(data, .25, axis=0)
        scores.append(np.mean(np.clip((data - median) / (iqr + .1), -3, 3), axis=1))
    scores = np.column_stack(scores)
    ordered = np.sort(scores, axis=1)
    top = np.argmax(scores, axis=1)
    # Unresolved cells stay neutral rather than receiving a forced subtype.
    accepted = (ordered[:, -1] > .25) & ((ordered[:, -1] - ordered[:, -2]) > .15)
    classes = np.where(accepted, top, 3)
    return pd.DataFrame({"node_id": nodes, "program_class": classes,
                         "program_margin": ordered[:, -1] - ordered[:, -2],
                         "program_top_score": ordered[:, -1]})


def draw_biology(ax, root, cells, bounds):
    ax.scatter(cells.umap_1, cells.umap_2, s=1.45, c="#d9dee3", alpha=.18,
               linewidths=0, rasterized=True, zorder=1)
    for k in [3, 0, 1, 2]:
        sub = cells[cells.program_class.eq(k)]
        if not len(sub):
            continue
        color = MIXED if k == 3 else PROGRAMS[root][k][3]
        ax.scatter(sub.umap_1, sub.umap_2, s=2.0, c=color,
                   alpha=.37 if k == 3 else .74, linewidths=0,
                   rasterized=True, zorder=2 if k == 3 else 3)
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
def draw_topology(ax, root, path, cells, positions, bounds):
    rgba = base.cell_rgba(cells)
    ax.scatter(cells.umap_1, cells.umap_2, s=1.7, c=rgba,
               linewidths=0, rasterized=True, zorder=1)
    base.draw_edges(ax, {root: path}, positions, "umap_1", "umap_2",
                    roots_to_draw=[root], arrow_limit=7, umap_spacing=.035)
    for i, terminal in enumerate(path["terminals"]):
        x, y = positions.loc[terminal, ["umap_1", "umap_2"]]
        ax.scatter([x], [y], s=74, c="white", edgecolors=base.LINE[root],
                   linewidths=1.5, zorder=15)
        ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=6.9,
                color=base.LINE[root], fontweight="bold", zorder=16)
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def route_composition(root, path, node_calls, counts):
    routes = [nx.shortest_path(path["graph"], path["hub"], terminal)
              for terminal in path["terminals"]]
    class_by_node = node_calls.set_index("node_id").program_class
    output = []
    for i, route in enumerate(routes):
        other = set().union(*(set(r) for j, r in enumerate(routes) if j != i))
        unique = sorted(set(route) - other)
        if not unique:
            raise ValueError(f"{root}: terminal route {i + 1} has no unique nodes")
        cell_counts = np.bincount(class_by_node.loc[unique].to_numpy(int),
                                  weights=counts[unique], minlength=4)
        total = float(cell_counts.sum())
        for klass, number in enumerate(cell_counts):
            output.append({"root": root, "branch": i + 1,
                           "terminal_node": path["terminals"][i],
                           "program_class": klass,
                           "program_label": "Mixed / unresolved" if klass == 3
                           else PROGRAMS[root][klass][0],
                           "n_unique_nodes": len(unique), "n_cells": int(number),
                           "fraction": float(number / total)})
    return pd.DataFrame(output)


def draw_routes(ax, root, composition):
    colors = [p[3] for p in PROGRAMS[root]] + [MIXED]
    route_labels = []
    for branch in range(1, 5):
        subset = composition[composition.branch.eq(branch)].sort_values("program_class")
        fractions = subset.fraction.to_numpy(float)
        best = int(np.argmax(fractions[:3]))
        route_labels.append(
            f"{branch}  {SHORT[root][best]} {fractions[best]:.0%}"
            if fractions[best] >= .15 else f"{branch}  unresolved"
        )
        left = 0
        for row in subset.itertuples():
            ax.barh(branch, row.fraction, left=left, color=colors[row.program_class],
                    height=.54, edgecolor="white", linewidth=.5)
            left += row.fraction
    ax.set_xlim(0, 1)
    ax.set_ylim(4.55, .45)
    ax.set_yticks([1, 2, 3, 4], route_labels, fontsize=8.2)
    ax.set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"], fontsize=8)
    ax.tick_params(length=0)
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#b9c0ca")
    ax.set_title("c  Marker-program composition of unique route segments",
                 loc="left", fontsize=11, fontweight="bold", pad=10)


def draw_one(root, path, raw, root_pt, node_calls, positions, composition,
             evidence_line):
    cells = raw.merge(root_pt[["node_id", "pt_display"]], on="node_id", how="left",
                      validate="many_to_one")
    cells["root"] = np.where(cells.pt_display.notna(), root, None)
    cells = cells.merge(node_calls[["node_id", "program_class"]], on="node_id",
                        how="left", validate="many_to_one")
    xy = positions.loc[list(path["graph"].nodes), ["umap_1", "umap_2"]].to_numpy(float)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    pad = np.maximum(.55, .15 * (hi - lo))
    bounds = ((lo[0] - pad[0], hi[0] + pad[0]),
              (lo[1] - pad[1], hi[1] + pad[1]))
    fig = plt.figure(figsize=(13.6, 9.0), facecolor="white")
    ax1 = fig.add_axes([.035, .37, .35, .50])
    legend = fig.add_axes([.397, .42, .19, .38])
    ax2 = fig.add_axes([.60, .37, .36, .50])
    bar = fig.add_axes([.21, .105, .67, .19])
    legend.axis("off")
    draw_biology(ax1, root, cells, bounds)
    draw_topology(ax2, root, path, cells, positions, bounds)
    draw_routes(bar, root, composition)
    legend.text(0, .99, "CELL-COLOR KEY", fontweight="bold", fontsize=9.5,
                color="#364152", transform=legend.transAxes)
    for k, (name, markers, _, color) in enumerate(PROGRAMS[root]):
        y = .78 - k * .225
        legend.scatter([.05], [y], s=95, c=color, transform=legend.transAxes)
        legend.text(.16, y + .035, name, fontsize=8.0, fontweight="bold",
                    va="center", transform=legend.transAxes, wrap=True)
        legend.text(.16, y - .065, markers, fontsize=7.1, color="#667085",
                    va="center", transform=legend.transAxes, wrap=True)
    legend.scatter([.05], [.08], s=95, c=MIXED, transform=legend.transAxes)
    legend.text(.16, .08, "Mixed / unresolved", fontsize=8.0,
                color="#667085", va="center", transform=legend.transAxes)
    fig.text(.035, .965, f"xenium_breast  |  {base.ROOT_LABELS[root]}  |  linked biological views",
             fontsize=16, fontweight="bold", va="top")
    fig.text(.035, .925, PROCESS[root], fontsize=12.4, fontweight="bold",
             color=base.LINE[root], va="top")
    fig.text(.035, .885,
             evidence_line, fontsize=8.8, color="#526070")
    fig.text(.035, .845, "a  Marker-program background", fontsize=11,
             fontweight="bold")
    fig.text(.60, .845, "b  Formal graph + root pseudotime", fontsize=11,
             fontweight="bold")
    grad_ax = fig.add_axes([.68, .325, .25, .014])
    t = np.linspace(0, 1, 256)
    early = np.array(to_rgba(base.STOPS[root][0])[:3])
    late = np.array(to_rgba(base.STOPS[root][1])[:3])
    grad_ax.imshow((early[None, :] * (1 - t[:, None]) + late[None, :] * t[:, None])[None, :, :],
                   aspect="auto", extent=(0, 1, 0, 1))
    grad_ax.set_xticks([0, 1], ["early", "late"], fontsize=7.5)
    grad_ax.set_yticks([])
    grad_ax.tick_params(length=0)
    for spine in grad_ax.spines.values():
        spine.set_visible(False)
    fig.text(.60, .328, "Pseudotime", fontsize=8.5, color=base.LINE[root],
             fontweight="bold")
    fig.text(.035, .045,
             "Background = full component; graph = four selected routes. Tips 1–4 correspond to c; shared trunk excluded."
             "  Grey = outside/unresolved; program calls exploratory; arrows = inferred edges only.",
             fontsize=8.3, color="#536070")
    stem = OUT / f"xenium_breast_{root}_biology_first_linked"
    fig.savefig(stem.with_suffix(".png"), dpi=230)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def main():
    mpl.use("Agg")
    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    OUT.mkdir(parents=True, exist_ok=True)
    paths, _, all_pt = base.load_formal_paths()
    raw = base.load_cells()
    positions = raw.groupby("node_id").agg(
        spatial_1=("spatial_1", "mean"), spatial_2=("spatial_2", "mean"))
    umap = pd.read_csv(base.EMBED / f"{base.DATASET}_metacells_selected_umap.csv")
    positions = positions.join(umap.set_index("node_id")[["umap_1", "umap_2"]],
                               validate="one_to_one")
    cache = np.load(EXPR, allow_pickle=True)
    expression, counts = cache["expression"], cache["counts"]
    gene_index = {gene: i for i, gene in enumerate(cache["genes"].astype(str))}
    assert expression.shape[0] == len(positions)
    call_rows, composition_rows = [], []
    trend_table = pd.read_csv(
        base.PROJECT / "outputs" / "reports_FROZEN"
        / "trajectory_evidence_frozen_A0" / "TRAJECTORY_PROGRAM_TRENDS.csv"
    )
    gene_table = pd.read_csv(
        base.PROJECT / "outputs" / "reports_FROZEN"
        / "trajectory_evidence_frozen_A0" / "TRAJECTORY_GENE_TRENDS.csv"
    )
    component_sets = {root: set(all_pt.loc[all_pt.root.eq(root), "node_id"])
                      for root in base.ROOTS}
    overlap = len(component_sets["C00"] & component_sets["C21"])
    jaccard = overlap / len(component_sets["C00"] | component_sets["C21"])
    evidence_rows = []
    for root, path in paths.items():
        root_pt = all_pt[all_pt.root.eq(root)]
        calls = call_programs(root, root_pt.node_id, expression, gene_index)
        calls["root"] = root
        call_rows.append(calls)
        composition = route_composition(root, path, calls, counts)
        composition_rows.append(composition)
        trends = trend_table[
            trend_table.dataset.eq(base.DATASET)
            & trend_table.root_cluster.eq(root)
            & trend_table.component_id.eq(path["component"])
        ].set_index("program_code")
        if root == "C13":
            genes = gene_table[
                gene_table.dataset.eq(base.DATASET)
                & gene_table.root_cluster.eq(root)
                & gene_table.component_id.eq(path["component"])
            ].set_index("gene")
            evidence_line = (
                "Frozen evidence: epithelial distal–anchor "
                f"{trends.loc['epithelial', 'distal_minus_anchor']:+.2f}; "
                f"cycling peaks mid ({trends.loc['cell_cycle', 'intermediate_score']:.2f}); "
                f"CLDN4 {genes.loc['CLDN4', 'distal_minus_anchor_z']:+.2f}, "
                f"CEACAM6 {genes.loc['CEACAM6', 'distal_minus_anchor_z']:+.2f} z."
            )
        else:
            evidence_line = (
                "Frozen evidence: memory-like distal–anchor "
                f"{trends.loc['t_memory', 'distal_minus_anchor']:+.2f}; "
                f"cytotoxic {trends.loc['t_cytotoxic', 'distal_minus_anchor']:+.2f}; "
                f"R1/R2 share {overlap} nodes (Jaccard {jaccard:.2f})."
            )
        evidence_rows.append({"root": root, "process_label": PROCESS[root],
                              "evidence_line": evidence_line,
                              "overlap_C00_C21": overlap if root != "C13" else 0,
                              "jaccard_C00_C21": jaccard if root != "C13" else 0})
        draw_one(root, path, raw, root_pt, calls, positions, composition,
                 evidence_line)
        print(root, calls.program_class.value_counts().sort_index().to_dict(),
              composition.groupby("branch").n_cells.sum().to_dict())
    pd.concat(call_rows).to_csv(OUT / "node_marker_program_calls.csv", index=False)
    pd.concat(composition_rows).to_csv(OUT / "route_program_composition.csv", index=False)
    pd.DataFrame(evidence_rows).to_csv(OUT / "process_label_evidence.csv", index=False)
    (OUT / "figure_methods.json").write_text(json.dumps({
        "plot_archetype": "linked UMAP identity/topology plus route composition",
        "conclusion": "Each frozen representative route has a measurable, marker-associated cellular context.",
        "program_call": "Within-root component median/IQR-scaled 3-gene score; top >0.25 and margin >0.15, else mixed/unresolved.",
        "cell_weights": "formal metacell counts from the node-expression cache",
        "route_composition": "unique nodes of each hub-to-terminal route; shared trunk excluded",
        "topology": "unchanged frozen representative path edges; arrows only on exact_anchor_outward edges",
        "label_layout": "no text or leader lines on biological UMAP; adjacent categorical legend",
        "review_risks": ["program calls are not independently validated cell types",
                         "UMAP distances are display coordinates, not graph distances",
                         "route composition is not STREAM branch density or fate probability",
                         "pseudotime does not prove lineage conversion"],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
