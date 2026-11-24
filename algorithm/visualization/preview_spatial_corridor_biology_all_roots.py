#!/usr/bin/env python3
"""Local cell corridors and angular spatial trajectories for all breast roots.

Coordinates are the original tissue coordinates. Branches come from a subset
of the frozen formal root-to-terminal tree. Short turns are removed only for
display; no trajectory or direction is inferred by this script.
"""

from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import preview_angular_all_roots_dualspace_breast as base
import preview_stream_paga_biology_breast as bio


OUT = base.OUTPUT / "spatial_corridor_biology_all_roots"
INK = "#18212B"
UNKNOWN = "#B7BEC7"
TERMINALS = {"C13": (0, 1, 3), "C00": (0, 2, 3), "C21": (0, 1, 3)}
RADIUS = {"C13": 260.0, "C00": 310.0, "C21": 310.0}
PROGRAMS = {
    "C13": [
        ("Luminal identity", ("FOXA1", "GATA3", "MLPH", "AGR3"), "#6655A4"),
        ("Cycling program", ("MKI67", "TOP2A", "CENPF"), "#D87B39"),
        ("CLDN4/CEACAM6-rich", ("CLDN4", "CEACAM6", "KLF5"), "#C5466B"),
        ("Basal-like identity", ("KRT5", "KRT14", "KRT15"), "#238F8D"),
    ],
    "C00": [
        ("IL7R/CCR7-rich T-like", ("IL7R", "CCR7", "LTB"), "#4869A8"),
        ("CCL5/GZMA-rich", ("CCL5", "GZMA", "CD69"), "#D5843C"),
        ("CD8/NKG7-rich", ("CD8A", "NKG7", "PRF1"), "#188B9A"),
        ("Checkpoint-associated", ("TIGIT", "LAG3", "HAVCR2", "PDCD1"), "#8861B8"),
        ("Cycling program", ("MKI67", "TOP2A", "CENPF"), "#C95777"),
    ],
    "C21": [
        ("IL7R/CCR7-rich T-like", ("IL7R", "CCR7", "LTB"), "#32866D"),
        ("CCL5/GZMA-rich", ("CCL5", "GZMA", "CD69"), "#C19437"),
        ("CD8/NKG7-rich", ("CD8A", "NKG7", "PRF1"), "#4774B3"),
        ("Checkpoint-associated", ("TIGIT", "LAG3", "HAVCR2", "PDCD1"), "#A65C9E"),
        ("Cycling program", ("MKI67", "TOP2A", "CENPF"), "#D06A5B"),
    ],
}


def component_graph(root, path, edges):
    e=edges.loc[edges.root_cluster.eq(root)&edges.component_id.eq(path["component"])]
    graph=nx.Graph()
    graph.add_edges_from(zip(e.node_u.astype(int),e.node_v.astype(int)))
    if not nx.is_tree(graph) or not set(path["graph"].nodes).issubset(graph.nodes):
        raise ValueError(f"{root}: incompatible frozen component tree")
    return graph


def selected_tree(path, root):
    graph=nx.Graph()
    for i in TERMINALS[root]:
        graph.add_edges_from(nx.utils.pairwise(nx.shortest_path(
            path["graph"],path["hub"],path["terminals"][i])))
    if not nx.is_tree(graph):raise ValueError("Selected route union is not a tree")
    return graph


def structural_chains(graph, hub):
    depth=nx.single_source_shortest_path_length(graph,hub)
    important={n for n in graph if graph.degree[n]!=2}|{hub}
    used=set()
    chains=[]
    for start in sorted(important,key=lambda n:(depth[n],n)):
        for neighbor in sorted(graph.neighbors(start)):
            edge=base.key(start,neighbor)
            if edge in used:continue
            chain=[start,neighbor];used.add(edge)
            previous,node=start,neighbor
            while node not in important:
                nxt=next(n for n in graph.neighbors(node) if n!=previous)
                used.add(base.key(node,nxt))
                chain.append(nxt);previous,node=node,nxt
            if depth[chain[0]]>depth[chain[-1]]:chain.reverse()
            chains.append(chain)
    if len(used)!=graph.number_of_edges():raise AssertionError("Some route edges lost")
    return chains


def rdp(points, tolerance):
    if len(points)<3:return [0,len(points)-1]
    a,b=points[0],points[-1]
    direction=b-a
    if np.linalg.norm(direction)<1e-8:
        distance=np.linalg.norm(points[1:-1]-a,axis=1)
    else:
        fraction=np.clip(((points[1:-1]-a)@direction)/(direction@direction),0,1)
        distance=np.linalg.norm(points[1:-1]-(a+fraction[:,None]*direction),axis=1)
    middle=int(np.argmax(distance))+1
    if distance[middle-1]<=tolerance:return [0,len(points)-1]
    left=rdp(points[:middle+1],tolerance)
    right=rdp(points[middle:],tolerance)
    return left[:-1]+[middle+j for j in right]


def display_line(chain, positions, tolerance, coord_cols=("spatial_1","spatial_2")):
    original=positions.loc[chain,list(coord_cols)].to_numpy(float)
    work=original.copy()
    for i in range(1,len(work)-1):
        work[i]=np.median(original[max(0,i-1):min(len(original),i+2)],axis=0)
    keep=rdp(work,tolerance)
    return work[keep], [chain[i] for i in keep]


def sampled_corridor(lines, spacing=45):
    samples=[]
    for line in lines:
        for a,b in nx.utils.pairwise(line):
            pieces=max(2,int(np.ceil(np.linalg.norm(b-a)/spacing)))
            t=np.linspace(0,1,pieces)
            samples.append(a[None,:]+t[:,None]*(b-a)[None,:])
    return np.vstack(samples)


def classify_nodes(root, node_ids, cache):
    all_genes={g:i for i,g in enumerate(cache["genes"].astype(str))}
    nodes=np.asarray(sorted(node_ids),dtype=int)
    matrix=cache["expression"][nodes]
    scores=[];coverage=[]
    for name,markers,_ in PROGRAMS[root]:
        measured=[g for g in markers if g in all_genes]
        if len(measured)<2:raise ValueError(f"{root}: insufficient marker coverage for {name}")
        x=matrix[:,[all_genes[g] for g in measured]]
        median=np.median(x,axis=0)
        iqr=np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0)
        scores.append(np.mean(np.clip((x-median)/(iqr+.1),-3,3),axis=1))
        coverage.append({"program":name,"measured_markers":measured})
    scores=np.column_stack(scores)
    rank=np.sort(scores,axis=1)
    winner=np.argmax(scores,axis=1)
    confident=(rank[:,-1]>.25)&((rank[:,-1]-rank[:,-2])>.15)
    winner=np.where(confident,winner,len(PROGRAMS[root]))
    df=pd.DataFrame({"node_id":nodes,"color_class":winner,
                     "top_score":rank[:,-1],"margin":rank[:,-1]-rank[:,-2]})
    for j in range(scores.shape[1]):
        df[f"score_{j}"]=scores[:,j]
        df[f"positive_{j}"]=scores[:,j]>.25
    return df,coverage


def add_segment_arrows(ax, lines, line_nodes, pseudotime):
    # Each displayed segment receives a black arrow ON the polyline. A tied
    # pseudotime uses the rooted chain order strictly as a display convention.
    counts={"pseudotime":0,"tie_root_to_tip":0,"pseudotime_toward_root":0}
    for line,nodes in zip(lines,line_nodes):
        for (a,b),(u,v) in zip(nx.utils.pairwise(line),nx.utils.pairwise(nodes)):
            delta=float(pseudotime[v])-float(pseudotime[u])
            if abs(delta)>1e-9:
                direction=np.sign(delta)
                counts["pseudotime"]+=1
                counts["pseudotime_toward_root"]+=int(direction<0)
            else:
                direction=1
                counts["tie_root_to_tip"]+=1
            start,end=(a,b) if direction>0 else (b,a)
            aa=start+.37*(end-start);bb=start+.68*(end-start)
            ax.add_patch(FancyArrowPatch(aa,bb,arrowstyle="-|>",
                                         mutation_scale=10.2,lw=1.65,
                                         facecolor=INK,edgecolor=INK,zorder=8))
    return counts


def plot_root(root,path,full_graph,raw,positions,cache,pseudotime):
    graph=selected_tree(path,root)
    chains=structural_chains(graph,path["hub"])
    # A small tolerance keeps several spatial turns per major branch.
    tolerance=115 if root=="C13" else 135
    display=[display_line(chain,positions,tolerance) for chain in chains]
    lines=[item[0] for item in display]
    line_nodes=[item[1] for item in display]
    corridor=sampled_corridor(lines)
    component=raw.loc[raw.node_id.isin(full_graph.nodes)].copy()
    distance,_=cKDTree(corridor).query(component[["spatial_1","spatial_2"]].to_numpy(float))
    cells=component.loc[distance<=RADIUS[root]].copy()
    if len(cells)<500:raise ValueError(f"{root}: too few nearby cells")
    calls,coverage=classify_nodes(root,full_graph.nodes,cache)
    cells=cells.merge(calls[["node_id","color_class"]],on="node_id",
                      how="left",validate="many_to_one")
    if cells.color_class.isna().any():raise ValueError("Nearby cells lack marker calls")
    cells.color_class=cells.color_class.astype(int)
    extent=np.vstack([cells[["spatial_1","spatial_2"]].to_numpy(float),*lines])
    low=extent.min(axis=0);high=extent.max(axis=0)
    pad=np.maximum((high-low)*.045,110)
    width,height=(high-low+2*pad)
    fig_width=9.4
    fig_height=float(np.clip(fig_width*height/width+1.15,4.6,7.8))
    fig,ax=plt.subplots(figsize=(fig_width,fig_height),facecolor="white")
    fig.subplots_adjust(left=.035,right=.98,top=.91,bottom=.10)
    unknown=len(PROGRAMS[root])
    for klass in [unknown,*range(unknown)]:
        subset=cells.loc[cells.color_class.eq(klass)]
        if subset.empty:continue
        color=UNKNOWN if klass==unknown else PROGRAMS[root][klass][2]
        ax.scatter(subset.spatial_1,subset.spatial_2,
                   s=2.5 if klass==unknown else 3.5,
                   c=color,alpha=.30 if klass==unknown else .76,
                   linewidths=0,rasterized=True,zorder=1 if klass==unknown else 2)
    for line in lines:
        ax.plot(line[:,0],line[:,1],c="white",lw=4.6,
                solid_joinstyle="round",solid_capstyle="round",zorder=5)
        ax.plot(line[:,0],line[:,1],c=INK,lw=2.3,
                solid_joinstyle="round",solid_capstyle="round",zorder=6)
    arrow_counts=add_segment_arrows(ax,lines,line_nodes,pseudotime)
    if arrow_counts["pseudotime"]+arrow_counts["tie_root_to_tip"] != sum(len(line)-1 for line in lines):
        raise AssertionError(f"{root}: a displayed segment is missing an arrow")
    hub=positions.loc[path["hub"],["spatial_1","spatial_2"]].to_numpy(float)
    ax.scatter(*hub,marker="*",s=190,c=INK,edgecolors="white",lw=.7,zorder=10)
    for idx in TERMINALS[root]:
        t=path["terminals"][idx]
        xy=positions.loc[t,["spatial_1","spatial_2"]].to_numpy(float)
        ax.scatter(*xy,s=54,facecolors="white",edgecolors=INK,lw=1.5,zorder=10)
    handles=[]
    counts=cells.color_class.value_counts().to_dict()
    for i,(name,_,color) in enumerate(PROGRAMS[root]):
        if counts.get(i,0)==0:continue
        handles.append(Line2D([],[],ls="",marker="o",markersize=6,
                              markerfacecolor=color,markeredgecolor="none",label=name))
    if counts.get(unknown,0):
        handles.append(Line2D([],[],ls="",marker="o",markersize=6,
                              markerfacecolor=UNKNOWN,markeredgecolor="none",label="Unresolved"))
    legend=ax.legend(handles=handles,title="Nearby cell marker program",
                     loc="upper right",bbox_to_anchor=(.995,.995),
                     frameon=True,facecolor="white",edgecolor="#D9DFE4",
                     framealpha=.97,fontsize=7.5,title_fontsize=8,
                     labelspacing=.28,handletextpad=.35,borderpad=.55)
    legend._legend_box.align="left"
    ax.set_xlim(low[0]-pad[0],high[0]+pad[0])
    ax.set_ylim(low[1]-pad[1],high[1]+pad[1])
    ax.set_aspect("equal",adjustable="box")
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.04,.97,f"{base.ROOT_LABELS[root]}  |  local spatial trajectory",
             va="top",fontsize=14.2,fontweight="bold",color=INK)
    fig.text(.04,.027,"Only nearby cells. Black arrows: increasing pseudotime; ties: root-to-tip display order. "
             "Star = root; circles = selected tips.",
             fontsize=6.7,color="#5A6876")
    stem=OUT/f"xenium_breast_{root}_local_spatial_biology"
    fig.savefig(stem.with_suffix(".png"),dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    labels={i:p[0] for i,p in enumerate(PROGRAMS[root])}|{unknown:"Unresolved"}
    table=pd.DataFrame([{"root":root,"class":i,"label":labels[i],"n_cells":n,
                         "fraction":n/len(cells)} for i,n in sorted(counts.items())])
    table.to_csv(OUT/f"{root}_local_label_counts.csv",index=False)
    calls.to_csv(OUT/f"{root}_node_program_scores.csv",index=False)
    print(root,"cells",len(cells),"nodes",graph.number_of_nodes(),
          "chains",len(chains),"display vertices",sum(map(len,lines)),
          "arrows",arrow_counts)
    print(table[["label","n_cells","fraction"]].to_string(index=False,
                                                         float_format=lambda v:f"{v:.3f}"))
    return {"root":root,"cells":len(cells),"display_nodes":graph.number_of_nodes(),
            "selected_terminal_indices":TERMINALS[root],"chains":len(chains),
            "display_vertices":sum(map(len,lines)),"arrow_counts":arrow_counts,
            "corridor_radius_tissue_units":RADIUS[root],"marker_coverage":coverage}


def main():
    mpl.use("Agg")
    mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
                         "pdf.fonttype":42,"svg.fonttype":"none"})
    OUT.mkdir(parents=True,exist_ok=True)
    paths,_,_=base.load_formal_paths()
    raw=base.load_cells()
    positions=raw.groupby("node_id").agg(spatial_1=("spatial_1","mean"),
                                          spatial_2=("spatial_2","mean"))
    edges=pd.read_csv(base.DIRECTION/"DIRECTED_COMPONENT_MST_EDGES.csv")
    cache=np.load(bio.EXPR,allow_pickle=True)
    all_pt=pd.read_csv(base.DIRECTION/"NODE_PSEUDOTIME_BY_ROOT_COMPONENT.csv")
    records=[]
    for root,path in paths.items():
        full=component_graph(root,path,edges)
        pt=(all_pt.loc[all_pt.root_cluster.eq(root)&
                       all_pt.component_id.eq(path["component"]),
                       ["node_id","constrained_pseudotime"]]
            .set_index("node_id").constrained_pseudotime)
        if not set(path["graph"].nodes).issubset(pt.index):
            raise ValueError(f"{root}: selected path lacks pseudotime")
        records.append(plot_root(root,path,full,raw,positions,cache,pt))
    (OUT/"figure_methods.json").write_text(json.dumps({
        "claim":"marker heterogeneity beside selected root-specific spatial branches",
        "source":"frozen formal component tree; raw tissue cell positions; node-aggregated expression",
        "layout":"original tissue coordinate system, no coordinate warping",
        "branches":"three of four frozen representative terminals per root; degree-2 chains mildly median-filtered and RDP-simplified; root and branch nodes preserved",
        "cells":"only component cells inside a spatial-distance corridor around displayed branch polylines",
        "color":"root-specific marker programs, top score >0.25 and margin >0.15; independent program scores retained for overlap audit",
        "arrow_direction":"each displayed straight segment is oriented from lower to higher formal constrained pseudotime; exact ties use root-to-tip chain order as a visualization convention, not an inferred direction",
        "caveats":["display lines approximate formal branch geometry","colors are exploratory programs rather than validated cell types","nearby cells are not necessarily members of the drawn graph path"],
        "roots":records,
    },indent=2),encoding="utf-8")


if __name__=="__main__":main()
