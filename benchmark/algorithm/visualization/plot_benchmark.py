"""Redraw the frozen Figure 3 evidence, without cropping reference figures.

Contract: reconstruction example -> 1,000 supplied-root construction pairs ->
historical seven-dataset context. Preserve every observation, direction, metric,
and unavailable topology score. No inference is rerun or repaired for appearance.
Python/matplotlib. An asymmetric scientific composite at 180 mm width.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import anndata as ad
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[2]
INK='#202a33'; MUTED='#5f6d78'; LIGHT='#dbe2e7'
MSF='#427ca5'; NF='#d97b36'
ROOT_C={'R0':'#c94d60','R1':'#377dab'}
BRANCH_C={'R0_b0':'#bb3854','R0_b1':'#ef9d98','R1_b0':'#377dab','R1_b1':'#7bc3cb'}
FAMILY_C={'ambig':'#427ca5','confound':'#4c9586','conflict':'#d38c3f',
          'lowsnr':'#bd6275','sep':'#897a68','mix':'#8d74aa'}
METHODS=['current branch-scale T0','root-conditioned MSF','CellRank-on-DPT','PAGA+DPT','Palantir','stLearn']
LABELS={'current branch-scale T0':'Current T0','root-conditioned MSF':'Root-MSF',
        'CellRank-on-DPT':'CellRank','PAGA+DPT':'PAGA+DPT','Palantir':'Palantir','stLearn':'stLearn'}
METHOD_C={'current branch-scale T0':NF,'root-conditioned MSF':MSF,'CellRank-on-DPT':'#6baaa5',
          'PAGA+DPT':'#c89d59','Palantir':'#9584b1','stLearn':'#c887a1'}
METRICS=[('Coverage','coverage'),('Branch ARI','branch_ARI_rootwise'),
         ('State-edge F1','topology_F1'),('Signed pseudotime rho','pseudotime_spearman')]


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def setup():
    mpl.rcParams.update({'font.family':'Arial','font.size':6.2,'axes.labelsize':6.2,
        'axes.linewidth':.55,'xtick.labelsize':5.8,'ytick.labelsize':5.8,
        'axes.spines.top':False,'axes.spines.right':False,'text.color':INK,
        'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'legend.frameon':False,'legend.fontsize':5.8,'savefig.facecolor':'white'})


def bootstrap_mean(v,seed,n=4000):
    rng=np.random.default_rng(seed);v=np.asarray(v,float)
    draws=v[rng.integers(0,len(v),(n,len(v)))].mean(1)
    return np.quantile(draws,[.025,.975])


def load(args):
    pkg=Path(args.data_root);bench=Path(args.benchmark_root)
    frozen=pkg/'results/NicheForest_paper_frozen_candidate'
    sim=frozen/'P4_root_conditioned_benchmark/runs/sim_K2_sep'
    adata_path=sim/'forest_inputs/metacell_adata.h5ad'
    a=ad.read_h5ad(adata_path)
    phase=sim/'forest_run/phase_a'
    node_path=phase/'T0_STATE_NODE_ASSIGNMENTS.csv'
    edge_path=phase/'T0_STATE_BACKBONE_EDGES.csv'
    nodes=pd.read_csv(node_path);edges=pd.read_csv(edge_path)
    coords=np.asarray(a.obsm['spatial'],float)
    dom=(nodes.sort_values(['node_id','membership','root_id'],ascending=[True,False,True],kind='mergesort')
         .drop_duplicates('node_id').set_index('node_id').reindex(np.arange(len(a))))
    assert dom.membership.notna().all() and len(a)==630
    roots=nodes.loc[nodes.is_root.astype(bool)].drop_duplicates(['root_id','node_id'])
    assert set(roots.root_id)=={'R0','R1'} and len(roots)==2
    assert all(str(a.obs.iloc[int(r.node_id)].origin)==r.root_id[-1] for r in roots.itertuples())
    assert len(edges)==12
    truth=a.uns['milestone_network'].copy()
    assert len(truth)==9
    unitmap=nodes[['unit_id','node_id']].drop_duplicates()
    assert unitmap.unit_id.is_unique
    assert set(edges.support_source)|set(edges.support_target)<=set(unitmap.unit_id)

    summary_root=Path(getattr(args,'summary_root',None) or ROOT/'summary')
    overall_path=summary_root/'overall_metrics.csv';condition_path=summary_root/'condition_metrics.csv'
    overall=pd.read_csv(overall_path).set_index('metric').loc[[m for m,c in METRICS]].reset_index()
    cond=pd.read_csv(condition_path)
    protocol=json.loads((bench/'SELECTION_PROTOCOL.json').read_text())
    files=sorted((bench/'runs').glob('*/METRICS.json'))
    assert len(files)==protocol['n_datasets']==1000
    records=[];file_hashes={}
    for p in files:
        part=json.loads(p.read_text())
        assert len(part)==2
        records.extend(part)
        file_hashes[str(p)]=sha(p)
    raw=pd.DataFrame(records)
    assert len(raw)==2000 and raw[['dataset','method']].duplicated().sum()==0
    assert raw.dataset.nunique()==1000 and raw.method.nunique()==2
    assert raw.groupby('dataset').seed.nunique().eq(1).all()
    assert raw.root_F1.eq('N/A_ROOTS_SUPPLIED').all()
    assert raw.is_forest.all() and raw.candidate_legal.all()
    assert all(raw.groupby('condition').dataset.nunique().to_dict()[k]==v for k,v in protocol['replicates_by_condition'].items())
    pairs=raw.pivot(index=['dataset','condition','seed'],columns='method',values=[c for m,c in METRICS])
    pairs.columns=[f'{m}__{method}' for m,method in pairs.columns]
    pairs=pairs.reset_index()
    for label,col in METRICS:
        pairs[col+'_delta']=pairs[col+'__current branch-scale T0']-pairs[col+'__root-conditioned MSF']
        r=overall.loc[overall.metric.eq(label)].iloc[0]
        assert np.isclose(pairs[col+'__current branch-scale T0'].mean(),r.nicheforest_mean,atol=1e-12)
        assert np.isclose(pairs[col+'__root-conditioned MSF'].mean(),r.rootmsf_mean,atol=1e-12)
        assert np.isclose(pairs[col+'_delta'].mean(),r.paired_mean_difference,atol=1e-12)
    for r in cond.itertuples():
        p=pairs[pairs.condition.eq(r.condition)]
        assert len(p)==r.N
        assert np.isclose(p.branch_ARI_rootwise_delta.mean(),r.branch_ari_diff,atol=1e-12)
        assert np.isclose(p.pseudotime_spearman_delta.mean(),r.pseudotime_diff,atol=1e-12)
        assert int((p.branch_ARI_rootwise_delta>1e-12).sum())==r.branch_ari_wins
    ci=[]
    for i,r in enumerate(cond.itertuples()):
        p=pairs[pairs.condition.eq(r.condition)]
        lo,hi=bootstrap_mean(p.branch_ARI_rootwise_delta,seed=45100+i)
        ci.append(dict(condition=r.condition,ci95_lower=lo,ci95_upper=hi))
    cond=cond.merge(pd.DataFrame(ci),on='condition',validate='one_to_one')

    p4_path=frozen/'P4_root_conditioned_benchmark/P4_CORE_COMPARISON.csv'
    p3_path=frozen/'P3_native_single_start_benchmark/P3_DATASET_RESULTS.csv'
    p4=pd.read_csv(p4_path);p3=pd.read_csv(p3_path)
    historical=[]
    for r in p4.itertuples():
        historical.append(dict(dataset=r.dataset,method=r.method,coverage=r.coverage,
            pt_abs_sr=abs(r.pseudotime_spearman),branch_ari=r.branch_ARI_rootwise,
            topology_f1=r.topology_F1,track='primary root-conditioned'))
    for r in p3[p3.status.eq('PASS')].itertuples():
        historical.append(dict(dataset=r.dataset,method=r.method,coverage=r.coverage,
            pt_abs_sr=abs(r.forest_pt_SR_covered),branch_ari=r.branch_ARI_covered,
            topology_f1=np.nan,track='contextual native single-start'))
    historical=pd.DataFrame(historical)
    assert len(historical)==42 and historical.groupby('method').dataset.nunique().eq(7).all()
    means=historical.groupby('method')[['coverage','pt_abs_sr','branch_ari','topology_f1']].mean().reindex(METHODS)
    expected=pd.read_csv(pkg/'figures/paper_main_composites_current_frozen_gradient/SOURCE_DATA_Figure1_current_benchmark_means.csv').set_index('method').reindex(METHODS)
    assert np.allclose(means,expected[means.columns],equal_nan=True,atol=1e-12)
    assert historical.loc[historical.track.eq('contextual native single-start'),'topology_f1'].isna().all()
    paths=[adata_path,node_path,edge_path,p4_path,p3_path,overall_path,condition_path,bench/'SELECTION_PROTOCOL.json']
    inputs={str(p):sha(p) for p in paths}
    return dict(a=a,coords=coords,nodes=nodes,edges=edges,dom=dom,roots=roots,truth=truth,
        overall=overall,conditions=cond,pairs=pairs,historical=historical,means=means,
        hashes=inputs,run_hashes=file_hashes,protocol=protocol)


class Plate:
    def __init__(self):
        self.w=180/25.4;self.h=10.75
        self.fig=plt.figure(figsize=(self.w,self.h),facecolor='white')
        self.groups={}
    def ax(self,x,y,w,h,group=None):
        a=self.fig.add_axes([x/self.w,1-(y+h)/self.h,w/self.w,h/self.h])
        if group:self.groups.setdefault(group,[]).append(a)
        return a
    def text(self,x,y,value,size=6.3,weight='normal',color=INK,**kw):
        return self.fig.text(x/self.w,1-y/self.h,value,ha='left',va='top',fontsize=size,fontweight=weight,color=color,**kw)
    def section(self,letter,title,y,subtitle=None):
        self.text(.13,y,letter,10,'bold')
        self.text(.34,y+.005,title,8.7,'bold')
        if subtitle:self.text(.34,y+.20,subtitle,6.0,color=MUTED)
    def panel(self,x,y,n,title):self.text(x,y,f'({n})  {title}',7.1,'bold')


def map_limits(ax,coords):
    low=coords.min(0);high=coords.max(0);pad=(high-low)*.05
    ax.set_xlim(low[0]-pad[0],high[0]+pad[0]);ax.set_ylim(low[1]-pad[1],high[1]+pad[1])
    ax.set_aspect('equal',adjustable='box');ax.axis('off')


def root_marks(ax,d,white=False,label=False):
    for r in d['roots'].itertuples():
        x,y=d['coords'][int(r.node_id)]
        ax.scatter(x,y,marker='*',s=43,facecolor='white' if white else ROOT_C[r.root_id],
            edgecolor=INK if white else 'white',lw=.55,zorder=8)
        if label:ax.annotate(r.root_id,(x,y),xytext=(5,-3),textcoords='offset points',fontsize=5.7,fontweight='bold')


def draw_maps(p,d):
    p.section('A','Representative two-root synthetic reconstruction',.10)
    coords=d['coords'];obs=d['a'].obs;dom=d['dom']
    positions=[(.18,.49,3.12,.73),(3.82,.49,3.12,.73),(.18,1.60,3.12,.73),(3.82,1.60,3.12,.73)]
    titles=['Ground-truth origins','Predicted branches and roots','Cell-level structural pseudotime','Directed trajectory trees']
    for j,((x,y,w,h),title) in enumerate(zip(positions,titles),1):
        p.panel(x,y-.19,j,title);ax=p.ax(x,y,w,h,'A')
        if j==1:
            for origin,color in [('0',ROOT_C['R0']),('1',ROOT_C['R1'])]:
                m=obs.origin.astype(str).eq(origin).to_numpy()
                ax.scatter(coords[m,0],coords[m,1],s=2.0,color=color,alpha=.74,lw=0)
            handles=[Line2D([],[],marker='o',ls='',color=ROOT_C[f'R{i}'],label=f'Origin {i}',markersize=3) for i in range(2)]
            ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.02),ncol=2,fontsize=5.5,handletextpad=.4,columnspacing=1.3,borderaxespad=0)
        if j==2:
            for b,color in BRANCH_C.items():
                m=dom.branch_id.eq(b).to_numpy()
                ax.scatter(coords[m,0],coords[m,1],s=2,color=color,alpha=.8,lw=0)
            root_marks(ax,d,label=True)
            handles=[Line2D([],[],marker='o',ls='',color=c,label=b.replace('_',': '),markersize=2.8) for b,c in BRANCH_C.items()]
            ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.02),ncol=4,fontsize=5.1,handletextpad=.2,columnspacing=.6,borderaxespad=0)
        if j==3:
            sc=ax.scatter(coords[:,0],coords[:,1],s=2.1,c=dom.pseudotime.to_numpy(float),cmap='viridis',vmin=0,vmax=1,lw=0)
            root_marks(ax,d,white=True)
            cax=p.ax(x+w+.06,y+.08,.045,h-.13,'A')
            cb=p.fig.colorbar(sc,cax=cax);cb.set_ticks([0,.5,1]);cb.ax.tick_params(labelsize=5,length=1.5,pad=1);cb.outline.set_linewidth(.4)
            cb.solids.set_rasterized(False)
        if j==4:
            ax.scatter(coords[:,0],coords[:,1],s=1.7,color='#dce2e6',alpha=.65,lw=0)
            unit=dict(zip(d['nodes'].unit_id,d['nodes'].node_id))
            for r in d['edges'].itertuples():
                u,v=int(unit[r.support_source]),int(unit[r.support_target])
                ax.add_patch(FancyArrowPatch(coords[u],coords[v],arrowstyle='-|>',mutation_scale=5.3,lw=.8,color=ROOT_C[r.root_id],shrinkA=1,shrinkB=1,zorder=4))
                ax.scatter(coords[[u,v],0],coords[[u,v],1],s=6,color=ROOT_C[r.root_id],edgecolor='white',lw=.25,zorder=5)
            root_marks(ax,d)
        map_limits(ax,coords)


def positions_for_graph(g,root):
    """A topology-preserving left-to-right layout, including non-root sources."""
    if root.startswith('O'):
        # These positions use milestone identities, not inferred extra edges.
        o=root.split('_')[0]
        if o=='O0':raw={'M0':(0,.5),'M1':(1,.5),'M2':(2,.8),'M3':(3,.8),'M4':(2,-.4)}
        else:raw={'M0':(0,.5),'M1':(1,.5),'M2':(2,1.05),'M3':(2,.05),'M4':(3,.05),'M5':(2,-.95)}
        return {n:raw[n.split('_')[1]] for n in g}
    if root=='R0':
        return {'root':(0,.55),'trunk':(1,.55),'b0_s0':(2,1),'b0_s1':(3,1),
                'b0_s2':(3,-.05),'b1_s0':(1,-.60),'b1_s1':(1,-1.55),'b1_s2':(2,-1.55)}
    return {'root':(0,.55),'trunk':(1,.55),'b0_s0':(2,.85),'b0_s1':(3,1.3),
            'b0_s2':(3,.35),'b1_s0':(2,-.85)}


def draw_graph(ax,g,pos,root,color):
    # The state graph can contain convergent or root-disconnected sources.
    # Preserve each edge exactly; layout never fills missing connections.
    assert set(g)==set(pos)
    for u,v in g.edges:
        ax.add_patch(FancyArrowPatch(pos[u],pos[v],arrowstyle='-|>',mutation_scale=6.1,
            color='#6a7782',lw=.75,shrinkA=14.2,shrinkB=15.0,zorder=1))
    for n,(x,y) in pos.items():
        isroot=n==root or n=='root'
        fill=color if isroot else '#f1f5f7'
        edge=color if isroot else '#b5c1ca'
        box=FancyBboxPatch((x-.24,y-.31),.48,.62,boxstyle='round,pad=0.025,rounding_size=.12',
            facecolor=fill,edgecolor=edge,lw=.6,zorder=3)
        ax.add_patch(box)
        label=n.split('_',1)[1] if n.startswith('O') else n
        ax.text(x,y,label,ha='center',va='center',fontsize=5.2,
            color='white' if isroot else INK,fontweight='bold' if isroot else 'normal',zorder=4)
    ax.set_xlim(-.36,3.36);ys=[q[1] for q in pos.values()];ax.set_ylim(min(ys)-.43,max(ys)+.43)
    ax.axis('off')


def draw_topology(p,d):
    p.panel(.18,2.54,5,'Ground-truth and predicted topology')
    p.text(.34,2.78,'Ground truth',6.7,'bold');p.text(4.05,2.78,'Predicted T0 state topology',6.7,'bold')
    truth=d['truth'];pg=[]
    for idx in range(2):
        rid=f'R{idx}';origin=f'O{idx}';y=3.02+idx*.58
        p.text(.18,y+.06,f'O{idx}',6.3,'bold');p.text(3.82,y+.06,rid,6.3,'bold')
        te=truth[truth['from'].str.startswith(origin)]
        g=nx.from_pandas_edgelist(te,'from','to',create_using=nx.DiGraph)
        ax=p.ax(.39,y,2.88,.47,'A5')
        pos=positions_for_graph(g,origin+'_M0');draw_graph(ax,g,pos,origin+'_M0',ROOT_C[rid])
        part=d['edges'][d['edges'].root_id.eq(rid)]
        g2=nx.from_pandas_edgelist(part,'source_state','target_state',create_using=nx.DiGraph)
        assert nx.is_directed_acyclic_graph(g2)
        ax=p.ax(4.03,y,2.88,.47,'A5');pos2=positions_for_graph(g2,rid)
        assert set(g2)==set(pos2)
        draw_graph(ax,g2,pos2,rid,ROOT_C[rid])
        for u,v in g2.edges:pg.append(dict(root_id=rid,source=u,target=v))
    p.text(.39,4.10,'Root and state labels retain their original identities; all 9 truth edges and 12 predicted edges are shown.',5.8,color=MUTED)
    return pd.DataFrame(pg)


def half_violin(ax,values,x,color,rng):
    values=np.asarray(values,float)
    if np.ptp(values)>1e-9:
        grid=np.linspace(values.min(),values.max(),180)
        density=gaussian_kde(values)(grid);density=density/density.max()*.23
        ax.fill_betweenx(grid,x,x+density,color=color,alpha=.25,lw=.4,edgecolor=color)
    jit=rng.uniform(-.19,-.04,len(values))
    ax.scatter(x+jit,values,s=.23,color=color,alpha=.19,lw=0)
    ax.scatter([x],[np.mean(values)],s=12,marker='D',color=color,edgecolor='white',lw=.4,zorder=6)


def draw_overall(p,d):
    p.section('B','Expanded supplied-root construction benchmark',4.39,
        '1,000 paired datasets · 12 conditions · ground-truth roots supplied to both methods')
    p.panel(.18,4.91,1,'Overall score profile and paired effects')
    handles=[Line2D([],[],marker='o',ls='',color=c,label=s,markersize=3.5) for c,s in [(MSF,'Root-MSF'),(NF,'NicheForest')]]
    p.fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.975,1-4.88/p.h),ncol=2,fontsize=6,handletextpad=.4,columnspacing=1.3)
    xs=[.42,2.12,3.82,5.52]
    rng=np.random.default_rng(714)
    display=['Coverage','Branch ARI','State-edge F1','Pseudotime correlation']
    for i,((label,col),x,title) in enumerate(zip(METRICS,xs,display)):
        p.text(x,5.16,title,6.8,'bold')
        ax=p.ax(x,5.40,1.13,.69,'B1')
        for k,(method,color) in enumerate([('root-conditioned MSF',MSF),('current branch-scale T0',NF)]):
            vals=d['pairs'][col+'__'+method].to_numpy()
            half_violin(ax,vals,k,color,rng)
        ax.set_xlim(-.32,1.32);ax.set_ylim(-.27,1.06)
        ax.set_yticks([-.25,0,.5,1]);ax.set_xticks([0,1],['MSF','NF']);ax.tick_params(length=2,pad=1,labelsize=5.4)
        if i==0:ax.set_ylabel('Score',labelpad=1,fontsize=5.9)
        else:ax.set_yticklabels([]);ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#aebac3')
        r=d['overall'].iloc[i]
        p.text(x,6.23,f'Means  {r.rootmsf_mean:.4f} / {r.nicheforest_mean:.4f}',5.7,color=MUTED)
        eff=p.ax(x,6.45,1.13,.19,'B1')
        eff.axvline(0,lw=.55,color='#9ba7b0',ls=(0,(3,2)))
        eff.plot([r.ci95_lower,r.ci95_upper],[0,0],lw=1.7,color=NF,zorder=3)
        eff.scatter(r.paired_mean_difference,0,s=15,color=NF,edgecolor='white',lw=.4,zorder=4)
        eff.set_xlim(-.06,.105);eff.set_ylim(-1,1);eff.set_yticks([])
        eff.set_xticks([-.05,0,.10],['−.05','0','+.10']);eff.tick_params(length=2,pad=1,labelsize=5.1)
        eff.spines['left'].set_visible(False);eff.spines['bottom'].set_color('#aebac3')
        digits=5 if i==0 else 3
        p.text(x,6.82,f'Δ {r.paired_mean_difference:+.{digits}f}',6.0,'bold')
    p.text(.42,7.05,'Dots: paired-dataset scores; diamonds: means. Lower intervals: mean Δ and paired bootstrap 95% CI (NF − MSF).',5.7,color=MUTED)


def draw_conditions(p,d):
    p.panel(.18,7.24,2,'Condition-wise Branch ARI')
    ax=p.ax(.93,7.57,2.04,1.64,'B2')
    cond=d['conditions'].sort_values('branch_ari_diff',ascending=False).reset_index(drop=True)
    rng=np.random.default_rng(1217)
    for i,r in enumerate(cond.itertuples()):
        color=FAMILY_C[r.condition.split('_')[0]]
        vals=d['pairs'].loc[d['pairs'].condition.eq(r.condition),'branch_ARI_rootwise_delta'].to_numpy()
        if np.ptp(vals)>0:
            grid=np.linspace(vals.min(),vals.max(),140);density=gaussian_kde(vals)(grid);density=density/density.max()*.26
            ax.fill_between(grid,i-density,i+density,color=color,alpha=.14,lw=0)
        ax.scatter(vals,i+rng.uniform(-.16,.16,len(vals)),s=.7,c=color,alpha=.25,lw=0)
        ax.plot([r.ci95_lower,r.ci95_upper],[i,i],color=color,lw=1.6,zorder=4)
        ax.scatter(r.branch_ari_diff,i,s=12,c=color,edgecolor='white',lw=.45,zorder=5)
        ax.text(1.025,i,f'{100*r.branch_ari_wins/r.N:.0f}%',transform=ax.get_yaxis_transform(),fontsize=5.6,ha='left',va='center')
    ax.set_yticks(range(len(cond)),cond.condition);ax.set_ylim(len(cond)-.5,-.5)
    ax.axvline(0,color='#9aa7b1',ls=(0,(3,2)),lw=.65,zorder=0)
    ax.set_xlim(-.35,.38);ax.set_xticks([-.3,-.15,0,.15,.3]);ax.tick_params(axis='y',length=0,pad=3,labelsize=5.8)
    ax.tick_params(axis='x',length=2,pad=2,labelsize=5.5)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Paired Branch ARI difference (NF − MSF)',labelpad=2,fontsize=5.9)
    ax.text(1.02,1.075,'NF wins',transform=ax.transAxes,fontsize=5.8,ha='left')
    p.text(.93,9.48,'Each condition: n = 83–84 pairs; point / interval = mean / 95% CI.',5.3,color=MUTED)


def draw_tradeoff(p,d):
    p.panel(3.84,7.24,3,'Branch–ordering trade-off')
    ax=p.ax(4.22,7.57,2.60,1.64,'B3')
    ax.axvline(0,color='#a5afb7',ls=(0,(3,2)),lw=.65)
    ax.axhline(0,color='#a5afb7',ls=(0,(3,2)),lw=.65)
    offsets={'ambig_K2':(4,-5),'ambig_K3':(4,5),'ambig_K4':(4,5),
        'conflict_K3':(4,3),'confound_K3':(4,4),'lowsnr_K3':(4,8),
        'sep_K2':(4,5),'sep_K3':(-5,-7),'sep_K4':(5,-7),
        'mix_K2':(-5,5),'mix_K3':(-4,5),'mix_K4':(-4,5)}
    for r in d['conditions'].itertuples():
        color=FAMILY_C[r.condition.split('_')[0]]
        ax.scatter(r.pseudotime_diff,r.branch_ari_diff,s=20,c=color,edgecolor='white',lw=.6,zorder=5)
        dx,dy=offsets[r.condition]
        ax.annotate(r.condition,(r.pseudotime_diff,r.branch_ari_diff),xytext=(dx,dy),textcoords='offset points',
            fontsize=5.3,ha='left' if dx>0 else 'right',va='center')
    ax.set_xlim(-.145,.092);ax.set_ylim(-.065,.234)
    ax.set_xticks([-.1,-.05,0,.05]);ax.set_yticks([-.05,0,.1,.2]);ax.tick_params(length=2,pad=2,labelsize=5.5)
    ax.set_xlabel('Δ pseudotime correlation (NF − MSF)',labelpad=2,fontsize=5.9)
    ax.set_ylabel('Δ Branch ARI',labelpad=2,fontsize=5.9)
    p.text(4.22,9.48,'Each dot is one condition mean; positive values favor NF.',5.3,color=MUTED)


def draw_history(p,d):
    p.section('C','Historical comparison across trajectory-inference methods',9.82,
        'Seven synthetic datasets; native single-start methods are contextual comparisons')
    p.panel(.18,10.28,1,'Mean scores')
    # Historical evidence is deliberately given a second, wider panel export
    # as well as a compact position in the main page.
    metrics=['coverage','pt_abs_sr','branch_ari','topology_f1']
    names=['Coverage','PT |SR|','Branch ARI','Topology F1']
    ax=p.ax(.76,10.58,1.30,1.06,'C1')
    mat=d['means'][metrics].to_numpy()
    cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#edf0f3')
    im=ax.pcolormesh(np.arange(5)-.5,np.arange(7)-.5,np.ma.masked_invalid(mat),vmin=0,vmax=1,cmap=cmap,shading='flat')
    ax.set_xlim(-.5,3.5);ax.set_ylim(5.5,-.5)
    for i in range(6):
        for j in range(4):
            v=mat[i,j];ax.text(j,i,'N/A' if not np.isfinite(v) else f'{v:.2f}',ha='center',va='center',fontsize=5.1,color='white' if np.isfinite(v) and v>.64 else INK)
    ax.set_yticks(range(6),[LABELS[m] for m in METHODS]);ax.set_xticks(range(4),names,rotation=55,ha='right')
    ax.tick_params(length=0,labelsize=5.0,pad=2)
    for s in ax.spines.values():s.set_visible(False)
    cax=p.ax(2.10,10.58,.035,1.06,'C1');cb=p.fig.colorbar(im,cax=cax);cb.set_ticks([0,.5,1]);cb.ax.tick_params(length=1,pad=1,labelsize=5.0);cb.outline.set_linewidth(.4)
    cb.solids.set_rasterized(False)

    p.panel(2.40,10.28,2,'Dataset-wise score distributions')
    # Small-n distributions: all seven observations, IQR and median, no KDE.
    for j,(metric,title) in enumerate(zip(metrics,names)):
        ax=p.ax(2.44+j*.65,10.58,.50,1.06,'C2')
        methods=[m for m in METHODS if d['historical'].loc[d['historical'].method.eq(m),metric].notna().any()]
        for k,method in enumerate(methods):
            v=d['historical'].loc[d['historical'].method.eq(method),metric].dropna().to_numpy()
            rng=np.random.default_rng(8000+j*20+k)
            ax.scatter(k+rng.uniform(-.15,.15,len(v)),v,s=3.3,color=METHOD_C[method],alpha=.85,lw=.15,edgecolor='white',zorder=4)
            q=np.quantile(v,[.25,.5,.75]);ax.plot([k,k],[q[0],q[2]],lw=2,color=METHOD_C[method],alpha=.5)
            ax.plot([k-.20,k+.20],[q[1],q[1]],lw=.7,color=INK,zorder=5)
        ax.set_ylim(-.04,1.04);ax.set_yticks([0,.5,1]);ax.tick_params(axis='y',length=1.5,pad=1,labelsize=5.0)
        if j>0:ax.set_yticklabels([]);ax.spines['left'].set_visible(False)
        short={'current branch-scale T0':'T0','root-conditioned MSF':'MSF','CellRank-on-DPT':'CR','PAGA+DPT':'PAGA','Palantir':'Pal','stLearn':'stL'}
        ax.set_xticks(range(len(methods)),[short[m] for m in methods],rotation=90)
        ax.tick_params(axis='x',labelsize=5.0,length=0,pad=2)
        ax.set_title(title,fontsize=5.3,pad=4,fontweight='bold')
    p.panel(5.35,10.28,3,'Branch–topology balance')
    ax=p.ax(5.62,10.58,1.20,1.06,'C3')
    primary=d['historical'][d['historical'].track.eq('primary root-conditioned')]
    wide=primary.pivot(index='dataset',columns='method',values=['branch_ari','topology_f1'])
    for dataset,row in wide.iterrows():
        ax.plot([row[('branch_ari',m)] for m in ['root-conditioned MSF','current branch-scale T0']],
            [row[('topology_f1',m)] for m in ['root-conditioned MSF','current branch-scale T0']],lw=.5,color='#cbd2d8',zorder=1)
    for method in ['root-conditioned MSF','current branch-scale T0']:
        part=primary[primary.method.eq(method)]
        ax.scatter(part.branch_ari,part.topology_f1,s=11,color=METHOD_C[method],edgecolor='white',lw=.4,label=LABELS[method],zorder=3)
    ax.set_xlim(-.012,.30);ax.set_ylim(.44,.67);ax.set_xticks([0,.15,.30]);ax.set_yticks([.45,.55,.65]);ax.tick_params(length=2,pad=1,labelsize=5.0)
    ax.set_xlabel('Branch ARI',labelpad=1,fontsize=5.4);ax.set_ylabel('Topology F1',labelpad=1,fontsize=5.4)
    ax.legend(loc='lower right',fontsize=5.0,handletextpad=.3,labelspacing=.3)
    p.text(.18,12.00,'N/A: topology scores are unavailable under the common state-edge definition. B and C are distinct benchmark cohorts.',5.6,color=MUTED)


def export_sources(d,out,pred):
    sources=out/'source_data';sources.mkdir(parents=True,exist_ok=True)
    a=d['dom'].reset_index().copy();a['x']=d['coords'][:,0];a['y']=d['coords'][:,1]
    a['truth_origin']=d['a'].obs.origin.astype(str).to_numpy();a['truth_branch']=d['a'].obs.branch.astype(str).to_numpy()
    a.to_csv(sources/'A_synthetic_nodes.csv',index=False)
    d['truth'].to_csv(sources/'A_truth_edges.csv',index=False)
    d['edges'].to_csv(sources/'A_supported_backbone_edges.csv',index=False)
    pred.to_csv(sources/'A5_predicted_state_edges.csv',index=False)
    d['pairs'].to_csv(sources/'B_1000_paired_dataset_scores.csv',index=False)
    d['overall'].to_csv(sources/'B_overall_frozen_summary.csv',index=False)
    d['conditions'].to_csv(sources/'B_condition_summary_with_CI.csv',index=False)
    d['historical'].to_csv(sources/'C_historical_42_method_dataset_scores.csv',index=False)
    d['means'].to_csv(sources/'C_historical_means.csv')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/historical_comparisons')
    parser.add_argument('--benchmark-root',type=Path,default=ROOT/'data/supplied_root')
    parser.add_argument('--summary-root',type=Path,default=ROOT/'summary')
    parser.add_argument('--outdir',type=Path,default=ROOT/'outputs/Figure3_redesign')
    args=parser.parse_args();out=args.outdir;out.mkdir(parents=True,exist_ok=True)
    setup();d=load(args)
    print('Verified 630 synthetic nodes, 9 truth / 12 predicted edges, 1000 paired runs, 42 historical records',flush=True)
    # A full native redraw, not a raster assembly. The tall final figure keeps
    # all original content and improves B without squeezing historical C.
    p=Plate();p.h=12.16
    actual_height=10.75
    p.fig.set_size_inches(p.w,actual_height)
    draw_maps(p,d);pred=draw_topology(p,d)
    draw_overall(p,d);draw_conditions(p,d);draw_tradeoff(p,d);draw_history(p,d)
    assert d['pairs'].select_dtypes('number').notna().all().all()
    assert d['pairs'].filter(regex='pseudotime_spearman__').min().min()>-.27
    assert d['pairs'].branch_ARI_rootwise_delta.between(-.35,.38).all()
    stem=out/'Figure3_redesigned_ABC'
    p.fig.savefig(stem.with_suffix('.png'),dpi=400)
    p.fig.savefig(stem.with_suffix('.pdf'))
    p.fig.savefig(stem.with_suffix('.svg'))
    # Standalone region exports retain every mark and editable vector label.
    regions={'A5_topology_detail':(.10,2.50,6.91,1.77),
             'B_benchmark_detail':(.10,4.33,6.91,5.34),
             'C_historical_detail':(.10,9.76,6.91,2.37)}
    for name,(x,y,w,h) in regions.items():
        scale_y=actual_height/p.h
        box=mpl.transforms.Bbox.from_bounds(x,(p.h-y-h)*scale_y,w,h*scale_y)
        p.fig.savefig(out/(name+'.png'),dpi=500,bbox_inches=box)
        p.fig.savefig(out/(name+'.pdf'),bbox_inches=box)
    # Retain separate native-rendered panels for an eventual editable assembly.
    panel_dir=out/'panels';panel_dir.mkdir(exist_ok=True)
    panel_regions={
        'A1':(.10,.27,3.28,1.08),'A2':(3.75,.27,3.29,1.08),
        'A3':(.10,1.38,3.50,1.04),'A4':(3.75,1.38,3.29,1.04),
        'A5':regions['A5_topology_detail'],
        'B1':(.10,4.86,6.91,2.34),'B2':(.10,7.20,3.35,2.45),
        'B3':(3.75,7.20,3.26,2.45),
        'C1':(.10,10.22,2.17,1.80),'C2':(2.37,10.22,2.74,1.76),
        'C3':(5.30,10.22,1.71,1.76)}
    for name,(x,y,w,h) in panel_regions.items():
        box=mpl.transforms.Bbox.from_bounds(x,(p.h-y-h)*actual_height/p.h,w,h*actual_height/p.h)
        p.fig.savefig(panel_dir/(name+'.png'),dpi=500,bbox_inches=box)
        p.fig.savefig(panel_dir/(name+'.pdf'),bbox_inches=box)
    plt.close(p.fig)
    export_sources(d,out,pred)
    manifest=dict(contract='A reconstruction -> B supplied-root 1000-pair construction benchmark -> C historical 7-dataset contextual comparison',
        physical_size_mm=[180,actual_height*25.4],source_inputs_sha256=d['hashes'],raw_1000_run_sha256=d['run_hashes'],
        counts=dict(synthetic_nodes=630,truth_edges=9,predicted_state_edges=12,paired_datasets=1000,historical_datasets=7,historical_records=42),
        primary_scope='Both methods received the true roots in B; this is downstream construction, not root discovery.',
        C_scope='Historical native single-start comparisons remain distinct from the primary root-conditioned cohort; topology F1 stays unavailable for four methods.',
        uncertainty=dict(overall='Original frozen paired-bootstrap 95% CI, unchanged',conditions='Percentile bootstrap 95% CI of paired differences, 4000 resamples per condition; seed 45100 + original CSV row index'),
        topology='All supplied directed edges preserved, including convergent and root-disconnected state sources; no invented connectors.',
        visualization_references=['https://www.nature.com/articles/s41592-019-0470-3','https://www.nature.com/articles/s41587-019-0071-9'],
        software=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,matplotlib=mpl.__version__,anndata=version('anndata')))
    (out/'reproduction_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(stem.with_suffix('.png'),flush=True)


if __name__=='__main__':main()
