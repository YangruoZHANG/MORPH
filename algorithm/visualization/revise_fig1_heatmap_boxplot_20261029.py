"""Observed gene evidence above ordered program distributions; no new inference.

Contract: distinguish the measured genes underlying each biological interpretation
from the distribution of its descriptive program score. Reuse accepted A/C/D.
Python mixed-modality composition, 7.2 x 10 inches; editable B PDF/SVG and source CSV.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba
from matplotlib.transforms import Bbox

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/Figure1_heatmap_boxplot_B_20261029'
OUT.mkdir(parents=True,exist_ok=True)
OLD=ROOT/'outputs/Figures_124_unified_trajectories_20261025/figure1'
EVID=ROOT/'tmp/breast_review_20261029/assets'
W,H=1600.,2222.222222
plt.rcParams.update({'font.family':'Arial','font.size':6.5,'axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':.5,'pdf.fonttype':42,'svg.fonttype':'none',
    'text.color':'#202B36','axes.labelcolor':'#202B36','xtick.color':'#647587','ytick.color':'#647587'})
CASES=[('R0','Stromal ECM remodelling',['PCOLCE','FBLN1','MMP1','MMP2'],'ECM program','#C53D56'),
       ('R1','B-cell secretory states',['PRDM1','MZB1','TNFRSF17','DERL3','SLAMF7','TENT5C'],'Secretory program','#387FB5'),
       ('R2','Stage-conditioned epithelium',['FOXA1','KRT5','MKI67','CLDN4','FASN','SCD','SQLE','SEC24A'],'Lipid-related four-gene score','#27916D')]
heat_rows=[];box_rows=[]
def rect(x,y,w,h):return [x/W,1-(y+h)/H,w/W,h/H]
def label(fig,x,y,s,size=7,bold=False,color='#202B36'):
    fig.text(x/W,1-y/H,s,ha='left',va='top',fontsize=size,weight='bold' if bold else 'normal',color=color)
def build(background):
    fig=plt.figure(figsize=(7.2,10),facecolor='white')
    if background:
        ax=fig.add_axes([0,0,1,1]);ax.imshow(plt.imread(OLD/'Figure1_checked_state_forests.png'),aspect='auto');ax.axis('off')
    fig.patches.append(Rectangle((0,1-1162/H),1,(1162-570)/H,transform=fig.transFigure,facecolor='white',edgecolor='none',zorder=2))
    label(fig,40,594,'B  Measured gene expression and biological program states',11.2,True)
    label(fig,44,638,'Gene-level expression along inferred order, with program-score distributions across four ordered groups',7,color='#647587')
    for k,(rid,title,genes,program,col) in enumerate(CASES):
        x=40+k*516.8
        data=pd.read_csv(EVID/f'{rid}_observed_evidence.csv')
        label(fig,x+3,680,f'({k+1}) {rid}: {title}',8,True)
        label(fig,x+112,713,f'{len(data):,} states',6.3,color='#647587')
        ax=fig.add_axes(rect(x+112,739,338,156),zorder=4)
        matrix=np.array([[data.loc[data.decile.eq(d),g].median() for d in range(1,11)] for g in genes])
        assert np.isfinite(matrix).all()
        im=ax.imshow(matrix,cmap='RdBu_r',vmin=-1.5,vmax=1.5,aspect='auto',interpolation='nearest')
        ax.set_yticks(range(len(genes)),genes,fontsize=6)
        ax.set_xticks([0,4,9],['1','5','10'])
        ax.tick_params(length=0,labelsize=6,pad=2)
        ax.set_xlabel('Within-forest state-order decile',fontsize=6,labelpad=2)
        for sp in ax.spines.values():sp.set_visible(False)
        if rid=='R2':ax.axhline(3.5,c='white',lw=1)
        for g,row in zip(genes,matrix):
            for d,value in enumerate(row,1):
                heat_rows.append({'root':rid,'gene':g,'decile':d,'median_gene_z_score':value,'n_states':int(data.decile.eq(d).sum())})
        if k==2:
            cax=fig.add_axes(rect(1543,739,9,156),zorder=4)
            cb=fig.colorbar(im,cax=cax,ticks=[-1.5,0,1.5]);cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=5.5,length=1,pad=1)
            label(fig,1534,716,'z score',5.7,color='#647587')
        label(fig,x+76,946,program,7,True)
        ax=fig.add_axes(rect(x+76,982,374,92),zorder=4)
        vals=[data.loc[data.quartile.eq(q),'display_program'].to_numpy() for q in range(4)]
        b=ax.boxplot(vals,tick_labels=['S0','S1','S2','S3'],widths=.52,patch_artist=True,showfliers=False,
            boxprops={'linewidth':.65,'edgecolor':col},medianprops={'linewidth':1,'color':'#202B36'},
            whiskerprops={'linewidth':.6,'color':col},capprops={'linewidth':.6,'color':col})
        for q,patch in enumerate(b['boxes']):patch.set_facecolor(to_rgba(col,.16+.16*q))
        ax.axhline(0,lw=.5,ls='--',color='#ABB4BE',zorder=0)
        ax.set_xlabel('Within-forest order quartile',fontsize=6,labelpad=2)
        ax.set_ylabel('Score',fontsize=6,labelpad=2);ax.tick_params(length=2,labelsize=6,pad=2)
        ax.grid(axis='y',alpha=.12,lw=.4)
        for q,v in enumerate(vals):
            lo,med,hi=np.quantile(v,[.25,.5,.75]);iqr=hi-lo
            box_rows.append({'root':rid,'program':program,'quartile':q,'n_states':len(v),'q25':lo,'median':med,'q75':hi,
                'whisker_low':v[v>=lo-1.5*iqr].min(),'whisker_high':v[v<=hi+1.5*iqr].max()})
        data.to_csv(OUT/f'SOURCE_DATA_{rid}_observed_states.csv',index=False)
    label(fig,116,1137,'Heatmaps: decile medians of within-identity gene z scores (shared scale). Boxes: median, IQR and 1.5-IQR whiskers; outliers hidden.',5.9,color='#647587')
    label(fig,116,1157,'S0-S3: order quartiles. R2 lipid score: FASN/SCD/SQLE/SEC24A. Descriptive evidence from one section; no lineage or causal claim.',5.9,color='#647587')
    return fig

stem='Figure1_heatmap_boxplot_B'
fig=build(True)
fig.savefig(OUT/f'{stem}.png',dpi=400)
fig.savefig(OUT/f'{stem}_preview.png',dpi=200)
plt.close(fig)
fig=build(False)
fig.savefig(OUT/'B_overlay.pdf',transparent=True)
fig.savefig(OUT/'B_overlay.svg',transparent=True)
fig.savefig(OUT/'Figure1_B_detail.png',dpi=400,bbox_inches=Bbox.from_extents(0,10*(1-1175/H),7.2,10*(1-580/H)))
plt.close(fig)
pd.DataFrame(heat_rows).drop_duplicates().to_csv(OUT/'SOURCE_DATA_gene_heatmaps.csv',index=False)
pd.DataFrame(box_rows).drop_duplicates().to_csv(OUT/'SOURCE_DATA_boxplot_summary.csv',index=False)
(OUT/'Figure_legend.txt').write_text('''Figure 1 | Spatially resolved trajectories and measured biological program states.
A/C/D reuse the accepted unified artwork unchanged. B upper panels: selected measured genes across ten ranked within-forest pseudotime groups, using decile medians of within-identity gene z scores. Original whole-cell counts were divided by the total measured-panel count for each cell, scaled by median panel depth, log1p transformed and standardised gene-wise within published cell identity; exact-member mean aggregation produced state values. All heatmaps share the same colour limits, -1.5 to +1.5. R0 and R1 genes are the ECM and secretory panels discussed in the biological review. R2 shows four illustrative epithelial markers and all four members of the displayed lipid-related score; the white separator distinguishes these two gene sets. Flat genes (including MMP1) are retained; marker members do not all rise in parallel. B lower panels: existing ECM and Secretory program-score distributions for R0/R1 and the descriptive mean four-gene lipid-related score for R2, in within-forest order quartiles S0-S3. Box centre lines are medians, boxes IQR, and whiskers extreme observations within 1.5 IQR; outliers hidden. Exact state counts per quartile are supplied. These are descriptive distributions of state units from a single section, not independent patient replicates or an independent validation. R2 uses a clinical-stage prior. The four-gene lipid score is not a complete pathway assay. No new fitting, tests or significance annotations were introduced. Target-gene-excluded refit tests remain available in the original frozen source table and are not replaced by these descriptive plots. A: partitions and identity composition; C: root-specific tissue routes; D: supplied-start external-method outputs projected to tissue. Neither route geometry nor the expression plots prove ancestry or causal relationships.
''',encoding='utf-8')
(OUT/'provenance.json').write_text(json.dumps({'freeze':'v70_20261025','A_C_D':'unchanged accepted unified artwork','B':'observed state expression and score exports used in senior review','new_inference_or_tests':False,'genes':{r:g for r,_,g,_,_ in CASES}},indent=2),encoding='utf-8')
print(OUT)
