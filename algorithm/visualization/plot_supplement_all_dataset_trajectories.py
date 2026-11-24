"""Two data-driven supplementary plates using the frozen Fig. 1 graph exports."""
from __future__ import annotations

import json
from pathlib import Path
import string

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import preview_spatial_corridor_biology_all_roots as corridor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/Supplement_all_dataset_trajectories'
DATASETS = ['GSE','GSE193460','GSE245582','xenium_breast','xenium_lung',
            'xenium_melanoma','xenium_pancreatic']
COLORS = ['#c62838','#1656c0','#087a48','#8655a7']
LIGHT = ['#efa5ae','#a8c7fa','#a5d8b6','#cfb7e5']
DARK = ['#7b001a','#07357c','#064a2d','#48306b']
INK = '#18212b'
IDENTITIES = {
    'tumor_epithelial':('Epithelial / tumor','#ba536c'),
    'lymphoid':('Lymphoid','#5279b6'),
    'myeloid':('Myeloid','#ab8741'),
    'endothelial_vascular':('Endothelial / vascular','#27979b'),
    'stromal_fibroblast':('Stromal / fibroblast','#8a6eaa'),
    'perivascular_smooth_muscle':('Perivascular / muscle','#658d54'),
}
STATES = [
    ('Cycling', ['MKI67','TOP2A','CENPF','PCNA','TYMS'], '#df843c'),
    ('Cytotoxic markers', ['NKG7','PRF1','GZMB','GZMA','CCL5'], '#258ba0'),
    ('IFN-response markers', ['ISG15','IFIT1','IFIT3','MX1','STAT1','IFI6'], '#c05294'),
    ('Stress-response markers', ['HSPA1A','HSPA1B','HSP90AA1','DDIT3','ATF3'], '#b39538'),
    ('ECM-remodeling markers', ['COL1A1','COL1A2','COL3A1','FN1','POSTN'], '#6d8c40'),
]


def load_dataset(ds):
    embed=ROOT/'outputs/v7_8_1_frozen_A0_trajectory_aware_embedding'/ds
    formal=ROOT/'NicheForest_v7_4_6_all7_reproduction_bundle/data'/ds
    raw=pd.read_csv(embed/'raw_cell_dense_coordinates.csv.gz',
                    usecols=['cell_id','spatial_1','spatial_2','umap_1','umap_2'])
    mapping=pd.read_csv(formal/'cell_to_metacell.csv',usecols=['cell_name','metacell_id'])
    domains=pd.read_csv(formal/'biological_node_domains.csv')
    raw['cell_id']=raw.cell_id.astype(str)
    mapping['cell_name']=mapping.cell_name.astype(str)
    raw=raw.merge(mapping,left_on='cell_id',right_on='cell_name',how='left',validate='one_to_one')
    raw=raw.merge(domains[['node_id','node_name']],left_on='metacell_id',right_on='node_name',
                  how='left',validate='many_to_one')
    assert raw.node_id.notna().all(), ds
    raw.node_id=raw.node_id.astype(int)
    pos=raw.groupby('node_id')[['spatial_1','spatial_2']].mean()
    umap=pd.read_csv(embed/f'{ds}_metacells_selected_umap.csv').set_index('node_id')
    pos=pos.join(umap[['umap_1','umap_2']],validate='one_to_one')
    assert pos.notna().all().all()
    audit=pd.read_csv(ROOT/'outputs/CB_figure_v7_8_1_FROZEN_trajectory_embedding_connected_rebuild'
                      /'05_tree_refinement_before_after_v6_2'/ds/'connected_representative_display_audit.csv')
    edges=pd.read_csv(ROOT/'outputs/v7_8_1_frozen_direction'/ds/'DIRECTED_COMPONENT_MST_EDGES.csv')
    pt=pd.read_csv(ROOT/'outputs/v7_8_1_frozen_direction'/ds/'NODE_PSEUDOTIME_BY_ROOT_COMPONENT.csv')
    cache=np.load(ROOT/'outputs/reports_v7_8_1_FROZEN/trajectory_evidence_v781_frozen_A0'
                  /'node_expression_cache'/f'{ds}_node_expression.npz',allow_pickle=True)
    assert cache['expression'].shape[0]==len(domains), ds
    paths=[]; parts=[]
    for j,row in enumerate(audit.itertuples()):
        e=edges[(edges.root_cluster==row.root_id)&(edges.component_id==row.component_id)
                &(edges.component_hash==row.component_hash)]
        full=nx.Graph();full.add_edges_from(zip(e.node_u.astype(int),e.node_v.astype(int)))
        assert nx.is_tree(full) and int(row.display_hub_node) in full
        tips=[int(v) for v in str(row.requested_terminals).split('|')]
        summary_graph=nx.Graph();summary_graph.add_node(int(row.display_hub_node))
        for t in tips:
            summary_graph.add_edges_from(nx.utils.pairwise(nx.shortest_path(full,int(row.display_hub_node),t)))
        # Preserve precisely the three published breast routes; all supplied
        # representative terminal routes are retained for the other datasets.
        if ds=='xenium_breast':
            tips=[tips[k] for k in corridor.TERMINALS[row.root_id]]
        graph=nx.Graph();graph.add_node(int(row.display_hub_node))
        for t in tips:
            route=nx.shortest_path(full,int(row.display_hub_node),t)
            graph.add_edges_from(nx.utils.pairwise(route))
        sub=pt[(pt.root_cluster==row.root_id)&(pt.component_id==row.component_id)
               &(pt.component_hash==row.component_hash)].copy()
        assert set(full).issubset(set(sub.node_id))
        sub['root_index']=j;sub['root']=row.root_id
        vals=sub.constrained_pseudotime
        sub['pt_display']=0.
        positive=vals>0
        sub.loc[positive,'pt_display']=.07+.93*vals[positive].rank(pct=True)
        parts.append(sub[['node_id','root','root_index','pt_display','root_membership_U']])
        paths.append(dict(root=row.root_id,j=j,hub=int(row.display_hub_node),tips=tips,
                          graph=graph,summary_graph=summary_graph,full=full,pt=sub.set_index('node_id').constrained_pseudotime,
                          component=int(row.component_id),hash=row.component_hash))
    membership=pd.concat(parts).sort_values(['node_id','root_membership_U','root_index'],
                                            ascending=[True,False,True]).drop_duplicates('node_id')
    raw=raw.merge(membership[['node_id','root_index','pt_display']],on='node_id',how='left',validate='many_to_one')
    print(ds,len(raw),'cells;',len(pos),'nodes;',len(paths),'roots',flush=True)
    return dict(ds=ds,raw=raw,pos=pos,domains=domains.set_index('node_id'),cache=cache,paths=paths)


def axis_clean(ax):
    ax.set_aspect('equal',adjustable='box');ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():sp.set_visible(False)


def arrow(ax,a,b,color=INK,size=7,lw=.9):
    if np.linalg.norm(b-a)<1e-12:return
    ax.add_patch(FancyArrowPatch(a+.38*(b-a),a+.67*(b-a),arrowstyle='-|>',
                 mutation_scale=size,lw=lw,color=color,zorder=9))


def chains_for(p,pos,tolerance):
    chains=corridor.structural_chains(p['graph'],p['hub'])
    return [corridor.display_line(c,pos,tolerance) for c in chains]


def draw_route(ax,p,pos,cols,color=INK,simplify=False,tolerance=0,linewidth=1.2):
    audit=[]
    segments=chains_for(p,pos,tolerance) if simplify else [
        (pos.loc[[u,v],list(cols)].to_numpy(float),[u,v]) for u,v in p['graph'].edges]
    for line,nodes in segments:
        ax.plot(line[:,0],line[:,1],c='white',lw=linewidth+1.0,zorder=5)
        ax.plot(line[:,0],line[:,1],c=color,lw=linewidth,zorder=6,
                solid_joinstyle='round',solid_capstyle='round')
        for (a,b),(u,v) in zip(nx.utils.pairwise(line),nx.utils.pairwise(nodes)):
            delta=float(p['pt'][v]-p['pt'][u]);start,end=(a,b) if delta>=0 else (b,a)
            arrow(ax,start,end,color,size=6 if simplify else 5,lw=.85)
            audit.append(dict(source=int(u),target=int(v),delta_pseudotime=delta,
                              direction_rule='pseudotime' if abs(delta)>1e-9 else 'root_to_tip_display_tie'))
    hub=pos.loc[p['hub'],list(cols)].to_numpy(float)
    ax.scatter(*hub,s=65,marker='*',c=color,edgecolor='white',lw=.5,zorder=11)
    if simplify:
        for tip in p['tips']:
            xy=pos.loc[tip,list(cols)].to_numpy(float)
            ax.scatter(*xy,s=19,facecolor='white',edgecolor=color,lw=.9,zorder=11)
    return audit,segments


def draw_summary(fig,axes,d,row):
    cells=d['raw'];rgba=np.tile(to_rgba('#c9ced4',.25),(len(cells),1))
    for p in d['paths']:
        j=p['j'];mask=cells.root_index.eq(j).to_numpy()
        t=cells.loc[mask,'pt_display'].to_numpy()**.76
        rgba[mask,:3]=np.array(to_rgba(DARK[j])[:3])[None,:]*(1-t[:,None])+np.array(to_rgba(LIGHT[j])[:3])[None,:]*t[:,None]
        rgba[mask,3]=.72
    for ax,cols,label in zip(axes,[('spatial_1','spatial_2'),('umap_1','umap_2')],['Tissue','UMAP']):
        original_box=ax.get_position()
        ax.scatter(cells[cols[0]],cells[cols[1]],c=rgba,s=.42,lw=0,rasterized=True)
        for p in d['paths']:
            summary_path=dict(p,graph=p['summary_graph'])
            draw_route(ax,summary_path,d['pos'],cols,COLORS[p['j']],linewidth=.95)
        ax.margins(.035);axis_clean(ax)
        if cols[0]=='spatial_1':ax.invert_yaxis()
        fig.text(original_box.x0,original_box.y1+.001,label,fontsize=9)
    # One small root-specific gradient per dataset, outside the data area.
    y=1-(row+1)/7+.010
    for p in d['paths']:
        j=p['j'];x=.09+j*.205
        ga=fig.add_axes([x,y,.15,.0045])
        t=np.linspace(0,1,128)
        rgb=np.array(to_rgba(DARK[j])[:3])[None,:]*(1-t[:,None])+np.array(to_rgba(LIGHT[j])[:3])[None,:]*t[:,None]
        ga.imshow(rgb[None,:,:],aspect='auto');ga.set_axis_off()
        fig.text(x,y+.006,f"R{j} · {p['root']}   early → late",fontsize=7.5,color=COLORS[j])


def biological_calls(d,p):
    nodes=np.array(sorted(p['full']),dtype=int)
    if d['ds']=='xenium_breast':
        calls,coverage=corridor.classify_nodes(p['root'],p['full'],d['cache'])
        programs=corridor.PROGRAMS[p['root']]
        labels={k:(v[0],v[2]) for k,v in enumerate(programs)}
        labels[len(programs)]=('Unresolved','#b7bec7')
        calls['label']=calls.color_class.map({k:v[0] for k,v in labels.items()})
        colors={v[0]:v[1] for v in labels.values()}
        return calls.set_index('node_id'),colors,coverage
    # Identity comes from the frozen compartment annotations. Functional-state
    # overlays use only measured marker genes, with conservative score gates.
    dom=d['domains'].reindex(nodes)
    labels=dom.major_compartment.map(lambda x:IDENTITIES.get(x,('Unresolved','#b7bec7'))[0])
    colors={v[0]:v[1] for v in IDENTITIES.values()};colors['Unresolved']='#b7bec7'
    calls=pd.DataFrame({'label':labels.to_numpy()},index=nodes)
    gene_index={str(g).upper():k for k,g in enumerate(d['cache']['genes'])}
    matrix=d['cache']['expression'][nodes]
    measured=[];scores=[]
    for label,genes,color in STATES:
        present=[g for g in genes if g in gene_index]
        if len(present)<2:continue
        data=matrix[:,[gene_index[g] for g in present]]
        median=np.median(data,axis=0);iqr=np.quantile(data,.75,axis=0)-np.quantile(data,.25,axis=0)
        score=np.mean(np.clip((data-median)/(iqr+.1),-3,3),axis=1)
        score[(data>0).sum(axis=1)<2]=-3
        scores.append(score);measured.append({'program':label,'measured_markers':present,'color':color})
        calls['score_'+label]=score
    if scores:
        scores=np.column_stack(scores);win=scores.argmax(axis=1)
        for k,item in enumerate(measured):
            keep=(win==k)&(scores[:,k]>.65)
            if item['program']=='Cytotoxic markers':
                keep &= dom.major_compartment.isin(['lymphoid','myeloid']).to_numpy()
            if item['program']=='ECM-remodeling markers':
                keep &= dom.major_compartment.isin(['stromal_fibroblast','perivascular_smooth_muscle','tumor_epithelial']).to_numpy()
            # Keep identity and functional state visible together in one label.
            for identity in labels[keep].unique():
                mask=keep&(labels.to_numpy()==identity)
                combined=identity+' · '+item['program']
                calls.loc[nodes[mask],'label']=combined
                colors[combined]=item['color']
    calls.index.name='node_id'
    return calls,colors,measured


def draw_local(ax,d,p):
    original_box=ax.get_position()
    pos=d['pos'];extent=np.ptp(pos.loc[list(p['full']),['spatial_1','spatial_2']].to_numpy(float),axis=0)
    diag=float(np.linalg.norm(extent))
    if d['ds']=='xenium_breast':
        tolerance=115 if p['root']=='C13' else 135;radius=corridor.RADIUS[p['root']]
    else:
        tolerance=diag*.012;radius=diag*.032
    display=chains_for(p,pos,tolerance)
    lines=[x[0] for x in display]
    cells=d['raw'][d['raw'].node_id.isin(p['full'])].copy()
    if d['ds']!='xenium_breast':
        center=pos.reindex(cells.node_id)[['spatial_1','spatial_2']].to_numpy(float)
        dispersion=np.linalg.norm(cells[['spatial_1','spatial_2']].to_numpy(float)-center,axis=1)
        radius=max(radius,float(np.median(dispersion))*.65)
    samples=corridor.sampled_corridor(lines,spacing=max(diag/180,1e-5))
    distance,_=cKDTree(samples).query(cells[['spatial_1','spatial_2']].to_numpy(float))
    cells=cells[distance<=radius].copy()
    # Very sparse spot datasets retain a neighborhood wide enough to show the
    # observed support, without adding any off-component cells.
    if len(cells)<12:
        cells=d['raw'][d['raw'].node_id.isin(p['graph'])].copy()
    calls,colors,coverage=biological_calls(d,p)
    cells['label']=cells.node_id.map(calls.label)
    assert cells.label.notna().all()
    count=cells.label.value_counts()
    # Keep crowded legends readable; the complete calls remain in source data.
    kept=set(count.head(6).index)
    if len(count)>6:
        cells.loc[~cells.label.isin(kept),'label']='Other nearby states'
        colors['Other nearby states']='#aeb6c0'
    counts=cells.label.value_counts()
    for label in counts.index:
        sub=cells[cells.label==label]
        ax.scatter(sub.spatial_1,sub.spatial_2,s=1.8,c=colors[label],alpha=.70,
                   lw=0,rasterized=True,zorder=1)
    arrows,_=draw_route(ax,p,pos,('spatial_1','spatial_2'),simplify=True,
                        tolerance=tolerance,linewidth=1.4)
    axis_clean(ax);ax.margins(.08);ax.invert_yaxis()
    handles=[Line2D([],[],marker='o',ls='',ms=3.6,color=colors[label],label=label)
             for label in counts.index]
    leg=ax.legend(handles=handles,loc='upper right',
                  bbox_to_anchor=(original_box.x1,original_box.y1),bbox_transform=ax.figure.transFigure,fontsize=6.5,
                  frameon=True,facecolor='white',edgecolor='#e2e5e9',framealpha=.94,
                  handletextpad=.3,labelspacing=.25,borderpad=.45)
    leg.set_zorder(20)
    ax.figure.text(original_box.x0,original_box.y1+.003,
                   f"({p['j']+1})  R{p['j']} · {p['root']}",fontsize=9,weight='bold')
    stem=OUT/'source_data'/f"{d['ds']}_{p['root']}"
    calls.to_csv(str(stem)+'_node_annotations.csv')
    cells[['cell_id','node_id','spatial_1','spatial_2','label']].to_csv(str(stem)+'_corridor_cells.csv.gz',index=False)
    pd.DataFrame(arrows).to_csv(str(stem)+'_display_arrows.csv',index=False)
    return dict(dataset=d['ds'],root=p['root'],component=p['component'],hash=p['hash'],
                formal_nodes=len(p['full']),display_nodes=len(p['graph']),display_edges=p['graph'].number_of_edges(),
                tips=p['tips'],corridor_radius=radius,display_tolerance=tolerance,
                nearby_cells=len(cells),legend_counts=counts.to_dict(),marker_coverage=coverage,
                arrow_segments=len(arrows),tied_arrows=sum(x['direction_rule']=='root_to_tip_display_tie' for x in arrows))


def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',dpi=300)
    fig.savefig(OUT/f'{name}.svg',dpi=220)
    fig.savefig(OUT/f'{name}.png',dpi=220)
    fig.savefig(OUT/f'{name}_preview.png',dpi=90)
    print('Saved',name,flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'source_data').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Arial','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    summary=plt.figure(figsize=(14,28),facecolor='white')
    sg=summary.add_gridspec(7,2,left=.065,right=.98,top=.97,bottom=.025,hspace=.38,wspace=.12)
    biology=plt.figure(figsize=(20,28),facecolor='white')
    bg=biology.add_gridspec(7,12,left=.028,right=.985,top=.97,bottom=.025,hspace=.39,wspace=.65)
    manifest=[]
    for row,ds in enumerate(DATASETS):
        d=load_dataset(ds)
        sa=[summary.add_subplot(sg[row,k]) for k in range(2)]
        draw_summary(summary,sa,d,row)
        sy=sg[row,0].get_position(summary).y1
        summary.text(.018,sy+.006,string.ascii_uppercase[row],fontsize=14,weight='bold')
        summary.text(.065,sy+.017,ds,fontsize=11,weight='bold')
        by=bg[row,0].get_position(biology).y1
        biology.text(.007,by+.006,string.ascii_uppercase[row],fontsize=14,weight='bold')
        biology.text(.028,by+.017,ds,fontsize=11,weight='bold')
        n=len(d['paths']);step=12//n
        for k,p in enumerate(d['paths']):
            ax=biology.add_subplot(bg[row,k*step:(k+1)*step])
            manifest.append(draw_local(ax,d,p))
        pd.DataFrame([dict(root=p['root'],source=u,target=v) for p in d['paths']
                      for u,v in p['graph'].edges]).to_csv(OUT/'source_data'/f'{ds}_representative_graph_edges.csv',index=False)
    summary.text(.065,.004,'Root colors: root-specific pseudotime (early → late). Grey: outside selected root components. Stars: root anchors.',fontsize=8,color='#5c6671')
    biology.text(.028,.005,'Only trajectory-nearby cells are shown. Black arrows: increasing pseudotime; ties: root-to-tip display order. Stars: root anchors; circles: selected tips.',fontsize=8,color='#5c6671')
    save(summary,'Supplement_B_all_datasets_merged_trajectories')
    save(biology,'Supplement_C_all_datasets_spatial_biology')
    (OUT/'source_data/manifest.json').write_text(json.dumps({'datasets':DATASETS,'roots':manifest,
        'scope':'All seven datasets and all twenty selected roots; same frozen graph and embedding exports as Figure 1.',
        'identity':'Frozen broad compartment annotation; breast reuses Figure 1 marker-program calls.',
        'functional_states':'Measured-marker robust z scores; at least two measured genes, score > 0.65. Post hoc labels.',
        'layout':'Two supplementary image plates, seven dataset rows. No figure-level title.'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
