"""Recompute measured-panel enrichment on MORPH, restoring the five-part figure narrative."""
import os,sys,json,hashlib,shutil,re
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'algorithm/visualization'))
import plot_forest_contacts as ref
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata,ttest_1samp,hypergeom
from scipy.sparse.csgraph import dijkstra
from scipy.cluster.hierarchy import linkage,leaves_list,dendrogram
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge,Circle,PathPatch
from matplotlib.path import Path as MPath
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from matplotlib.transforms import Bbox

FROZEN=ref.FROZEN;OUT=ROOT/'outputs/Figure4_enrichment_reference_style'
GROUPS=[('R0','Stromal'),('R1','B_cell'),('R1','CD8_T'),('R1','Macrophage'),('R2','Malignant_epithelial'),('R2','Stromal')]
NAMES=['R0-b0','R1-b0','R1-b1','R1-b2','R2-b0','R2-b1']
COL=['#C53D56','#387FB5','#71A5C9','#A4C8E0','#27916D','#88BFA0']
IDENT=['Stromal','B cells','CD8 T','Myeloid','Tumour','Stromal']
MODCOL={'Immune identity':'#4B78A8','Cell cycle':'#C7535A','Immune signalling':'#8C5E9F','Stress / damage':'#A36C3D','Metabolic':'#4F8C78','Growth / secretion':'#8A7A38','Tissue remodelling':'#A35C83','Other':'#B8BEC5'}
DISPLAY_MODULES={'Stromal':{'Stress / damage','Metabolic','Growth / secretion','Tissue remodelling'},'B_cell':{'Immune identity','Immune signalling','Cell cycle','Stress / damage','Metabolic','Growth / secretion'},'CD8_T':{'Immune identity','Immune signalling','Cell cycle','Stress / damage','Metabolic','Growth / secretion'},'Macrophage':{'Immune identity','Immune signalling','Stress / damage','Metabolic','Growth / secretion'},'Malignant_epithelial':{'Cell cycle','Stress / damage','Metabolic','Growth / secretion','Tissue remodelling'}}
INK='#202B36'
def bh(p):
 p=np.asarray(p,float);order=np.argsort(p);q=np.empty(len(p));q[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];return np.minimum(q,1)
def module(s):
 s=s.lower()
 for terms,m in [(['cell cycle','g2-m','e2f','mitotic','proliferation','spindle','dna replication'],'Cell cycle'),(['interferon','inflamm','il-6','il-2','tnf','complement','allograft','cytokine'],'Immune signalling'),(['t_cell','b_cell','nk_like','myeloid','dendritic','treg','cytotoxic','naive_memory','lymph','antigen','t cell','b cell'],'Immune identity'),(['hypoxia','uv response','p53','apoptosis','stress','dna damage'],'Stress / damage'),(['glycolysis','oxidative','cholesterol','adipogenesis','xenobiotic','heme','metabol','lipid'],'Metabolic'),(['mtorc1','kras','androgen','estrogen','secretion','secretory','transport','vesicle'],'Growth / secretion'),(['junction','mesenchymal','myogenesis','coagulation','angiogenesis','matrix','collagen','adhesion','remodel'],'Tissue remodelling')]:
  if any(t in s for t in terms):return m
 return 'Other'
def pretty(s,maxlen=34):
 s=s.split('::')[-1];s=re.sub(r'\s*(GO:\d+|R-HSA-\d+)','',s);s=s.replace('_',' ').replace('TNF-alpha Signaling via NF-kB','TNF / NF-kB').replace('Interferon Gamma Response','IFN-gamma response').replace('Epithelial Mesenchymal Transition','EMT').replace('IL-6/JAK/STAT3 Signaling','IL-6 / JAK / STAT3')
 return s if len(s)<=maxlen else s[:maxlen-1]+'…'
def blend(c,f):return tuple(1-f+f*np.array(to_rgb(c)))
def source_audit():
 records=json.loads((FROZEN/'SHA256_MANIFEST.json').read_text())
 for r in records:
  assert hashlib.sha256((FROZEN/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
 return len(records)

def analyse(d):
 catalog=pd.read_csv(ref.shared.BASE/'complete_catalog/EVERY_ORIGINAL_PROGRAM_AND_EXCLUSION.csv')
 assert len(catalog)==43
 genes=pd.Index(d['raw'].var_names.astype(str));x=sparse.csr_matrix(d['nuc']);depth=np.asarray(x.sum(1)).ravel();median=np.median(depth[depth>0]);norm=np.log1p(x.toarray()/np.maximum(depth[:,None],1)*median).astype(np.float32)
 # Exact core-cell aggregation; no inferred location or invented expression.
 unitgene=np.asarray(d['mean']@norm[d['core'].raw_row.to_numpy(int)])
 assert unitgene.shape==(len(d['units']),len(genes))
 z=StandardScaler().fit_transform(unitgene)
 score=np.column_stack([z[:,genes.get_indexer(s.split(';'))].mean(1) for s in catalog.measured_genes])
 pd.DataFrame(score,columns=catalog.program).to_csv(OUT/'source_data/ALL_UNIT_43_PROGRAM_SCORES.csv',index_label='node_id')
 catalog.to_csv(OUT/'source_data/ORIGINAL_43_PROGRAM_CATALOG.csv',index=False)
 frames=[];gene_tests=[];control=[];block_differences={}
 for j,key in enumerate(GROUPS):
  n,e=d['fits'][key];ids=n.node_id.to_numpy(int);tau=(rankdata(n.pseudotime,method='average')-1)/max(len(n)-1,1);step=np.minimum((tau*4).astype(int),3)
  a=n.copy();a['cohort']=NAMES[j];a['root_id']=key[0];a['step']=step;a['tau_rank']=tau;a['colour']=COL[j];frames.append(a)
  # For each ordered quartile, compare to the other quartiles in the SAME
  # root/identity forest using paired 500-um spatial-block means.
  blockxy=np.floor(d['xy'][ids]/500).astype(int);_,block=np.unique(blockxy,axis=0,return_inverse=True);nb=block.max()+1
  for st in range(4):
   m=step==st;n1=np.bincount(block[m],minlength=nb);n0=np.bincount(block[~m],minlength=nb);valid=(n1>0)&(n0>0)
   mean1=np.zeros((nb,len(genes)));mean0=np.zeros_like(mean1);np.add.at(mean1,block[m],unitgene[ids[m]]);np.add.at(mean0,block[~m],unitgene[ids[~m]])
   diff=mean1[valid]/n1[valid,None]-mean0[valid]/n0[valid,None]
   block_differences[NAMES[j],st]=diff/np.maximum(unitgene[ids].std(0),.05)
   if len(diff)>=8:
    p=ttest_1samp(diff,0,axis=0,alternative='greater').pvalue;p=np.nan_to_num(p,nan=1)
    effect=diff.mean(0)/np.maximum(unitgene[ids].std(0),.05)
   else:p=np.ones(len(genes));effect=np.zeros(len(genes))
   gene_tests.extend({'cohort':NAMES[j],'step':st,'gene':g,'paired_blocks':int(valid.sum()),'standardized_effect':float(effect[k]),'p':float(p[k])} for k,g in enumerate(genes))
  # Same frozen candidate graph, same observed units and source states.
  sub=d['g'][ids][:,ids];seeds=np.flatnonzero(n.is_source_state.to_numpy(bool));dist,pred,_=dijkstra(sub,directed=False,indices=seeds,min_only=True,return_predecessors=True)
  for k,v in enumerate(pred):
   if v>=0:control.append({'cohort':NAMES[j],'source':int(ids[v]),'target':int(ids[k])})
 assignments=pd.concat(frames,ignore_index=True);assignments.to_csv(OUT/'source_data/FROZEN_COHORT_STEP_ASSIGNMENTS.csv',index=False)
 gt=pd.DataFrame(gene_tests);gt['q_global']=bh(gt.p);gt['foreground']=gt.q_global.le(.05)&gt.standardized_effect.ge(.30);gt.to_csv(OUT/'source_data/ALL_STEP_GENE_TESTS.csv',index=False)
 control=pd.DataFrame(control);control.to_csv(OUT/'source_data/MATCHED_SOURCE_SHORTEST_PATH_EDGES.csv',index=False)
 # Background is the measured assay panel, never the entire genome.
 universe=set(genes);sets=[]
 folder=ROOT/'outputs/NicheForest_paper_frozen_candidate/P8_breast_biological_validation/gene_sets'
 for family in ['GO_Biological_Process_2023','Reactome_2022']:
  src=folder/(family+'.gmt');shutil.copy2(src,OUT/'source_data'/src.name)
  for line in src.read_text().splitlines():
   tok=line.split('\t');gs=set(tok[2:])&universe
   if 3<=len(gs)<=len(universe)*.8:sets.append((family,tok[0],gs))
 for r in catalog.itertuples():sets.append(('Primary_43',r.program,set(r.measured_genes.split(';'))))
 # Duplicate measured gene-sets retain term labels but are accounted for
 # in the complete, joint BH family, rather than selecting favourable terms.
 enrichment=[]
 for (cohort,st),g in gt.groupby(['cohort','step'],sort=False):
  fg=set(g.loc[g.foreground,'gene']);N=len(fg)
  for family,name,gs in sets:
   hit=sorted(fg&gs);p=float(hypergeom.sf(len(hit)-1,len(universe),len(gs),N)) if hit else 1.
   enrichment.append({'cohort':cohort,'step':int(st),'family':family,'term':name,'module':module(name),'n_foreground':N,'n_background':len(universe),'n_set_measured':len(gs),'overlap':len(hit),'overlap_genes':';'.join(hit),'fold_enrichment':len(hit)/max(N*len(gs)/len(universe),1e-12),'p':p})
 enrich=pd.DataFrame(enrichment);enrich['q_global']=bh(enrich.p);enrich['significant']=enrich.q_global.le(.05)&enrich.overlap.ge(3)
 enrich.to_csv(OUT/'source_data/ALL_MEASURED_PANEL_GO_REACTOME_ORA.csv',index=False)
 enrich.loc[enrich.significant].to_csv(OUT/'source_data/SIGNIFICANT_ENRICHMENTS.csv',index=False)
 print('ORA significant',int(enrich.significant.sum()),'/',len(enrich),'tests; foreground genes',gt.groupby(['cohort','step']).foreground.sum().to_dict(),flush=True)
 # Continuous competitive gene-set testing complements the thresholded ORA.
 # Each spatial block supplies one observed programme-vs-complement contrast.
 # Correlated genes are aggregated BEFORE testing; genes are not replicates.
 weights=np.zeros((len(genes),len(sets)))
 for k,(_,_,gs) in enumerate(sets):
  include=np.array([g in gs for g in genes]);weights[include,k]=1/include.sum();weights[~include,k]=-1/(~include).sum()
 rows=[]
 for (cohort,st),diff in block_differences.items():
  values=diff@weights
  if len(values)>=8:
   pp=np.nan_to_num(ttest_1samp(values,0,axis=0,alternative='greater').pvalue,nan=1);effects=values.mean(0)
  else:pp=np.ones(len(sets));effects=np.zeros(len(sets))
  for k,(family,term,gs) in enumerate(sets):rows.append({'cohort':cohort,'step':st,'family':family,'term':term,'module':module(term),'n_background':len(genes),'n_set_measured':len(gs),'paired_blocks':len(diff),'competitive_effect':float(effects[k]),'p':float(pp[k]),'measured_genes':';'.join(sorted(gs))})
 enrich=pd.DataFrame(rows);enrich['q_global']=bh(enrich.p);enrich['significant']=enrich.q_global.le(.05)&enrich.competitive_effect.gt(0)
 enrich.to_csv(OUT/'source_data/ALL_COMPETITIVE_SPATIAL_BLOCK_GENE_SET_TESTS.csv',index=False)
 enrich.loc[enrich.significant].to_csv(OUT/'source_data/SIGNIFICANT_COMPETITIVE_ENRICHMENTS.csv',index=False)
 print('Competitive spatial-block enrichments',int(enrich.significant.sum()),'/',len(enrich),flush=True)
 # Entire 43-program matrix is descriptive. Frozen gene-disjoint tests,
 # not the post-hoc heatmap, provide the separate independent evidence.
 effect=np.column_stack([score[f.node_id].mean(0) for f in frames])
 effect=effect/np.maximum(score[np.unique(assignments.node_id)].std(0)[:,None],.05)
 pd.DataFrame(effect,index=catalog.program,columns=NAMES).to_csv(OUT/'source_data/COHORT_PROGRAM_EFFECT_HEATMAP.csv')
 story=[]
 for j,key in enumerate(GROUPS):
  f=frames[j]
  for st in range(4):
   sel=f.loc[f.step.eq(st)];hit=enrich.loc[enrich.cohort.eq(NAMES[j])&enrich.step.eq(st)&enrich.significant&enrich.module.isin(DISPLAY_MODULES[key[1]])&~enrich.term.str.contains('viral|HIV|kidney|renal',case=False,regex=True)].sort_values(['q_global','p','term'])
   mscore=score[sel.node_id].mean(0);best=int(mscore.argmax())
   if len(hit):r=hit.iloc[0];term=r.term;mod=r.module;status='Competitive enrichment q<=0.05';q=float(r.q_global)
   else:term=catalog.iloc[best].program;mod=module(term);status='descriptive score only';q=None
   story.append({'cohort':NAMES[j],'root_id':key[0],'identity':key[1],'step':st,'n_units':len(sel),'n_observed_cells':int(sel.n_raw_cells.sum()),'term':term,'module':mod,'evidence':status,'competitive_q':q,'top_program_score':float(mscore[best])})
 story=pd.DataFrame(story);story.to_csv(OUT/'source_data/ORDERED_STEP_FUNCTIONAL_STORY.csv',index=False)
 # Common, ordinary program PCA with a single invertible radial compression.
 unique=np.unique(assignments.node_id);pc=PCA(2,random_state=42).fit_transform(StandardScaler().fit_transform(score[unique]));pc-=np.median(pc,axis=0)
 r=np.linalg.norm(pc,axis=1);s=max(float(np.median(r)),1e-8);coord=pc/(s+r[:,None])*.98
 xy=np.full((len(d['units']),2),np.nan);xy[unique]=coord
 pd.DataFrame({'node_id':unique,'disk_x':coord[:,0],'disk_y':coord[:,1],'PC1':pc[:,0],'PC2':pc[:,1]}).to_csv(OUT/'source_data/COMMON_PROGRAM_DISK_COORDINATES.csv',index=False)
 modules=list(MODCOL);modscore=np.column_stack([score[:,[i for i,v in enumerate(catalog.program) if module(v)==m]].mean(1) if any(module(v)==m for v in catalog.program) else np.zeros(len(score)) for m in modules]);dominant=np.argmax(modscore,axis=1)
 np.savez_compressed(OUT/'_build/plot_payload.npz',score=score,unitgene=unitgene,xy=xy,unique=unique,dominant=dominant,effect=effect)
 return catalog,assignments,enrich,story,xy,unique,dominant,effect,control

def draw_A(ax,d,a,story):
 ax.set_aspect('equal');ax.axis('off');ax.set_xlim(-1.45,1.6);ax.set_ylim(-1.3,1.3)
 angle=90.;gap=3;span=(360-gap*len(GROUPS))/len(GROUPS)
 for j,key in enumerate(GROUPS):
  end=angle+span;mid=np.deg2rad((angle+end)/2)
  ax.add_patch(Wedge((0,0),.59,angle,end,width=.24,fc=blend(COL[j],.55),ec='white',lw=.7))
  ax.text(.46*np.cos(mid),.46*np.sin(mid),NAMES[j],ha='center',va='center',fontsize=4.3,rotation=(angle+end)/2-90,color=INK)
  for st in range(4):
   r=story.loc[story.cohort.eq(NAMES[j])&story.step.eq(st)].iloc[0];lo=angle+span*st/4;hi=angle+span*(st+1)/4
   c=MODCOL[r.module] if r.evidence.startswith('Competitive') else blend(MODCOL[r.module],.35)
   ax.add_patch(Wedge((0,0),.96,lo,hi,width=.33,fc=c,ec='white',lw=.6))
   th=np.deg2rad((lo+hi)/2);ax.text(.80*np.cos(th),.80*np.sin(th),f'S{st}',ha='center',va='center',fontsize=3.5,color=INK)
  ax.annotate(f'{NAMES[j]}\n{IDENT[j]}',xy=(np.cos(mid)*.98,np.sin(mid)*.98),xytext=(np.cos(mid)*1.20,np.sin(mid)*1.13),fontsize=4.2,color=COL[j],ha='center',va='center',arrowprops=dict(arrowstyle='-',lw=.45,color=COL[j]))
  angle=end+gap
 # Root ring follows cohort order; R0 one, R1 three, R2 two.
 angle=90
 for rid,count in [('R0',1),('R1',3),('R2',2)]:
  end=angle+count*(span+gap)-gap;mid=np.deg2rad((angle+end)/2)
  ax.add_patch(Wedge((0,0),.31,angle,end,width=.14,fc=ref.RC[rid],ec='white',lw=.7));ax.text(.24*np.cos(mid),.24*np.sin(mid),rid,ha='center',va='center',fontsize=4.5,color='white',weight='bold');angle=end+gap
 ax.text(0,.02,f'{a.node_id.nunique():,}',ha='center',fontsize=5.8,weight='bold',color=INK);ax.text(0,-.09,'unique units',ha='center',fontsize=3.0,color=INK)
 handles=[Line2D([],[],marker='o',ls='',c=c,markersize=2.7,label=m) for m,c in MODCOL.items() if m!='Other']
 ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.48,1.09),ncol=3,fontsize=3.1,columnspacing=.6,handletextpad=.15,labelspacing=.2)
 ax.text(-1.40,-1.26,'Equal cohort / step sectors. Pale: descriptive programme;\nsolid: measured-panel enrichment. S0–S3: order quartiles.',fontsize=3.5,color='#657581')

def draw_B(ax,d,cat,eff):
 order=leaves_list(linkage(eff,method='average',metric='euclidean'));mat=eff[order]
 im=ax.imshow(mat,aspect='auto',cmap='RdBu_r',vmin=-1.5,vmax=1.5,interpolation='nearest')
 ax.set_yticks(range(43),[pretty(cat.iloc[i].program,29) for i in order],fontsize=3.0);ax.set_xticks(range(6),[f'{n}\n{v}' for n,v in zip(NAMES,IDENT)],fontsize=4.1);ax.tick_params(length=0,pad=1)
 tests=d['tests'];tests=tests.loc[tests.method.eq('Multipath')&tests.count_view.eq('WholeCell')]
 for y,i in enumerate(order):
  name=cat.iloc[i].program
  for j,key in enumerate(GROUPS):
   rows=tests.loc[tests.panel.eq(name)&tests.root_id.eq(key[0])&tests.identity.eq(key[1])]
   if len(rows) and bool(rows.iloc[0].qualified_complete):ax.plot(j,y,'o',ms=1.9,mfc='none',mec=INK,mew=.5)
 for s in ax.spines.values():s.set_visible(False)
 cb=ax.figure.colorbar(im,ax=ax,fraction=.020,pad=.02,shrink=.50);cb.ax.tick_params(labelsize=3.5,length=1);cb.set_label('Standardized programme mean',fontsize=4)
 ax.text(0,-.12,'43 measured programmes. Open dots: frozen gene-disjoint test passed.\nIdentity-overlapping signatures remain descriptive, not independent evidence.',transform=ax.transAxes,fontsize=3.3,va='top',color='#657581')

def disk_setup(ax):
 ax.add_patch(Circle((0,0),1,fc='none',ec=INK,lw=.65));ax.set_aspect('equal');ax.set_xlim(-1.08,1.08);ax.set_ylim(-1.08,1.08)
 ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.tick_params(labelsize=4,length=1,pad=1)
 ax.set_xlabel('Programme disk coordinate 1',fontsize=4,labelpad=1);ax.set_ylabel('Programme disk coordinate 2',fontsize=4,labelpad=1)
 for s in ax.spines.values():s.set_visible(False)
def skeleton(ax,d,xy,control=None,coloured=False):
 displayed=[]
 for j,key in enumerate(GROUPS):
  ed=ref.shared.segment_edges(d['paths'][key])
  if control is not None:
   parent=dict(control.loc[control.cohort.eq(NAMES[j]),['target','source']].to_numpy(int));paths=[]
   for original in d['paths'][key]:
    path=[int(original[-1])]
    while path[-1] in parent:path.append(int(parent[path[-1]]))
    paths.append(path[::-1])
   ed=ref.shared.segment_edges(paths)
   displayed.extend({'cohort':NAMES[j],'source':int(u),'target':int(v)} for u,v in ed)
   ax.add_collection(LineCollection(xy[ed],colors=INK,lw=.46,alpha=.70,zorder=4));continue
  ax.add_collection(LineCollection(xy[ed],colors=COL[j] if coloured else INK,lw=.46,alpha=.70,zorder=4))
 for j,key in enumerate(GROUPS):
  if key not in ref.CASES:continue
  n,_=d['fits'][key];u=int(n.loc[n.is_source_state,'node_id'].iloc[0]);q=xy[u]
  ax.scatter(*q,marker='*',s=18,c=ref.RC[key[0]],ec='white',lw=.25,zorder=8);ax.annotate(key[0],q,xytext=(3,2),textcoords='offset points',fontsize=4.3,weight='bold')
 if control is not None:pd.DataFrame(displayed).to_csv(OUT/'source_data/DISPLAYED_MATCHED_SHORTEST_PATH_EDGES.csv',index=False)

def draw_C(ax,d,a,xy,unique,dominant,which,control):
 disk_setup(ax)
 if which==0:
  # Shared observed nodes remain at their single common coordinate.
  shared=a.groupby('node_id').cohort.nunique();shared=set(shared[shared>1].index)
  for j,name in enumerate(NAMES):
   ids=a.loc[a.cohort.eq(name)&~a.node_id.isin(shared),'node_id'].to_numpy(int);ax.scatter(*xy[ids].T,s=1.0,c=COL[j],alpha=.45,lw=0,rasterized=True)
  ids=np.array(sorted(shared),int);ax.scatter(*xy[ids].T,s=1,c='#9A83AF',alpha=.40,lw=0,rasterized=True);skeleton(ax,d,xy,coloured=True)
  handles=[Line2D([],[],marker='o',ls='',c=COL[k],markersize=2,label=f'{NAMES[k]} {IDENT[k]}') for k in range(6)]+[Line2D([],[],marker='o',ls='',c='#9A83AF',markersize=2,label='Shared units')]
  ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(-.07,-.01),fontsize=3.2,ncol=2,handletextpad=.15,columnspacing=.4,labelspacing=.15)
 elif which==1:
  # Shared nodes have several cohort-specific orders; display their median,
  # explicitly not a new globally calibrated clock.
  v=a.groupby('node_id').tau_rank.median().reindex(unique).to_numpy();sc=ax.scatter(*xy[unique].T,s=1,c=v,cmap='viridis',vmin=0,vmax=1,lw=0,alpha=.6,rasterized=True);skeleton(ax,d,xy)
  cb=ax.figure.colorbar(sc,ax=ax,fraction=.030,pad=.035,shrink=.65);cb.ax.tick_params(labelsize=3.5,length=1);cb.set_label('Within-cohort order',fontsize=4)
 elif which==2:
  for k,(m,c) in enumerate(MODCOL.items()):
   ids=unique[dominant[unique]==k];ax.scatter(*xy[ids].T,s=1,c=c,alpha=.52,lw=0,rasterized=True)
  skeleton(ax,d,xy)
 else:
  for k,(m,c) in enumerate(MODCOL.items()):
   ids=unique[dominant[unique]==k];ax.scatter(*xy[ids].T,s=1,c=c,alpha=.52,lw=0,rasterized=True)
  skeleton(ax,d,xy,control=control)

def draw_D(ax,d,a,story,index):
 name=NAMES[index];f=a.loc[a.cohort.eq(name)]
 ax.scatter(*d['rawxy'][::3].T,s=.10,c='#D7DDE1',alpha=.32,lw=0,rasterized=True)
 labels=[]
 for st in range(4):
  r=story.loc[story.cohort.eq(name)&story.step.eq(st)].iloc[0];ids=f.loc[f.step.eq(st),'node_id'].to_numpy(int);rr=d['members'].loc[d['members'].unit_node_id.isin(ids),'raw_row'].to_numpy(int)
  c=MODCOL[r.module];ax.scatter(*d['rawxy'][rr].T,s=.24,c=c,alpha=.55,lw=0,rasterized=True)
  pts=d['xy'][ids];H,xe,ye=np.histogram2d(pts[:,0],pts[:,1],bins=36,range=[[d['rawxy'][:,0].min(),d['rawxy'][:,0].max()],[d['rawxy'][:,1].min(),d['rawxy'][:,1].max()]]);H=gaussian_filter(H,1)
  if H.max()>0:ax.contour((xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2,H.T,levels=[H.max()*.35],colors=[c],linewidths=.35,linestyles=['-','--','-.',':'][st],alpha=.8)
  q=np.median(pts,axis=0);ax.annotate(f'S{st}',q,xytext=((st%2)*12-6,(st//2)*12-6),textcoords='offset points',fontsize=3.5,color=INK,bbox=dict(fc='white',ec=c,lw=.3,pad=.6),arrowprops=dict(arrowstyle='-',color=c,lw=.25))
  labels.append(f'S{st}: {pretty(r.term,18)}'+(' *' if r.evidence.startswith('Competitive') else ''))
 ref.shared.clean(ax,d['rawxy'],True,pad=.025);ref.scale(ax,1000)
 ax.text(.00,1.08,'\n'.join(['; '.join(labels[:2]),'; '.join(labels[2:])]),transform=ax.transAxes,fontsize=3.3,va='bottom',color=INK,linespacing=1.12)

def ribbon(ax,a,b,c,d,colour):
 r=.80;point=lambda t:(r*np.cos(t),r*np.sin(t));aa=np.linspace(b,a,12);bb=np.linspace(d,c,12)
 verts=[point(a),(0,0),(0,0),point(d)]+[point(t) for t in bb[1:]]+[(0,0),(0,0),point(b)]+[point(t) for t in aa[1:]]+[point(a)]
 codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4]+[MPath.LINETO]*11+[MPath.CURVE4]*3+[MPath.LINETO]*11+[MPath.CLOSEPOLY]
 ax.add_patch(PathPatch(MPath(verts,codes),fc=colour,ec='none',alpha=.22))
def draw_E(ax,enrich):
 hits=enrich.loc[enrich.significant&enrich.family.ne('Primary_43')&enrich.module.ne('Other')&~enrich.term.str.contains('viral|HIV|kidney|renal',case=False,regex=True)].copy()
 identity_of=dict(zip(NAMES,[k[1] for k in GROUPS]))
 hits=hits.loc[[m in DISPLAY_MODULES[identity_of[c]] for m,c in zip(hits.module,hits.cohort)]].copy()
 hits['evidence']=np.minimum(-np.log10(hits.q_global.clip(lower=1e-12)),12)*hits.competitive_effect
 links=hits.sort_values(['evidence','term'],ascending=[False,True]).drop_duplicates(['cohort','module']).copy()
 # Report all significant modules. No links are made for unsupported groups.
 links.to_csv(OUT/'source_data/CIRCOS_SIGNIFICANT_LINKS.csv',index=False)
 ax.set_aspect('equal');ax.axis('off');ax.set_xlim(-1.40,1.45);ax.set_ylim(-1.40,1.40)
 if len(links)==0:
  ax.add_patch(Circle((0,0),1,fc='none',ec='#C7CFD5'));ax.text(0,0,'No GO / Reactome term\npassed the joint FDR gate',ha='center',fontsize=6);return
 mods=[m for m in MODCOL if m in set(links.module)];bs={};ms={};start=np.deg2rad(100);span=np.deg2rad(145);gap=.055
 for i,name in enumerate(NAMES):bs[name]=(start+i*(span/6+gap),start+i*(span/6+gap)+span/6)
 start=np.deg2rad(-75);tot=links.groupby('module').evidence.sum();cursor=start
 for m in mods:
  width=np.deg2rad(145)*tot[m]/tot.sum();ms[m]=(cursor,cursor+width);cursor+=width+gap
 bc={n:bs[n][0] for n in NAMES};mc={m:ms[m][0] for m in mods}
 for r in links.itertuples():
  width=(bs[r.cohort][1]-bs[r.cohort][0])*r.evidence/links.loc[links.cohort.eq(r.cohort),'evidence'].sum();ww=(ms[r.module][1]-ms[r.module][0])*r.evidence/tot[r.module]
  ribbon(ax,bc[r.cohort],bc[r.cohort]+width,mc[r.module],mc[r.module]+ww,COL[NAMES.index(r.cohort)]);bc[r.cohort]+=width;mc[r.module]+=ww
 for name,(a,b) in bs.items():
  c=COL[NAMES.index(name)];ax.add_patch(Wedge((0,0),1,np.degrees(a),np.degrees(b),width=.14,fc=c,ec='white',lw=.5));t=(a+b)/2;ax.text(1.12*np.cos(t),1.12*np.sin(t),name,ha='center',va='center',rotation=np.degrees(t)-90,fontsize=4.4,color=c,weight='bold')
 for name,(a,b) in ms.items():
  c=MODCOL[name];ax.add_patch(Wedge((0,0),1,np.degrees(a),np.degrees(b),width=.14,fc=c,ec='white',lw=.5));t=(a+b)/2
  short={'Tissue remodelling':'Remodelling','Growth / secretion':'Secretion / growth','Stress / damage':'Stress','Immune signalling':'Immune signals','Immune identity':'Immune programmes'}.get(name,name)
  if b-a>=.22:ax.text(1.13*np.cos(t),1.13*np.sin(t),short,ha='center',va='center',rotation=np.degrees(t)-90,fontsize=3.5,color=c,weight='bold')
 ax.text(0,.04,'Shared enriched\nfunctions',ha='center',fontsize=6,weight='bold');ax.text(0,-.16,'GO / Reactome\njoint q ≤ 0.05',ha='center',fontsize=4,color='#657581')
 ax.text(-1.35,-1.35,'Widths: maximum step enrichment evidence\nper cohort and functional module.',fontsize=3.7,color='#657581')

def render(d,payload):
 cat,a,enrich,story,xy,unique,dominant,eff,control=payload
 plt.rcParams.update({'font.family':'Arial','font.size':6,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False,'axes.linewidth':.55})
 fig=plt.figure(figsize=(7.2,10));assets={};elements=[]
 def heading(name,text,x,y,w,size=8.0):
  fig.text(x,y,text,weight='bold',color=INK,fontsize=size,va='top');elements.append({'kind':'text','name':name,'text':text,'x':1600*x,'y':(1-y)*2210,'w':1600*w,'h':55,'size':size*1600/(7.2*72),'bold':True})
 def panel(name,fn,rect):
  before=set(fig.axes);ax=fig.add_axes(rect);fn(ax)
  file=OUT/'panels'/f'Figure4_{name}.png'
  # Exact subplot extent including legends, scales and labels; copied image
  # remains a faithful crop of the assembled real-data panel.
  fig.canvas.draw();boxes=[v.get_tightbbox(fig.canvas.get_renderer()) for v in fig.axes if v not in before];box=Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).expanded(1.015,1.02)
  for ext in ['png','pdf','svg']:fig.savefig(file.with_suffix('.'+ext),dpi=450,facecolor='white',bbox_inches=box,pad_inches=0)
  assets[name]={'path':str(file)};elements.append({'kind':'image','name':name,'key':name,'x':box.x0/7.2*1600,'y':(10-box.y1)/10*2210,'w':box.width/7.2*1600,'h':box.height/10*2210})
 heading('A','A  Roots, cell-type forests and ordered steps',.025,.983,.40,7.3)
 heading('B','B  Functional profiles across 43 measured programmes',.465,.983,.52,7.3)
 panel('A',lambda ax:draw_A(ax,d,a,story),(.035,.802,.385,.143))
 panel('B',lambda ax:draw_B(ax,d,cat,eff),(.625,.784,.325,.170))
 heading('C','C  Biological-program disk map',.025,.754,.95,8.5)
 ctitle=['(1) Root-conditioned cohorts and actual paths','(2) Ordered states in the same programme space','(3) Dominant biological-programme modules','(4) Matched shortest-path forests in the same map']
 for k in range(4):
  x=.075+(k%2)*.48;y=.540 if k<2 else .315
  heading('C'+str(k+1)+' title',ctitle[k],x-.035,y+.190,.46,5.7)
  panel('C'+str(k+1),lambda ax,k=k:draw_C(ax,d,a,xy,unique,dominant,k,control),(x,y,.345,.175))
 # Compact shared legends, rather than using a different coordinate system.
 for k,(m,c) in enumerate(MODCOL.items()):
  x=.04+(k%4)*.245;y=.292-(k//4)*.009
  fig.text(x,y,'●',color=c,fontsize=4.5);fig.text(x+.010,y,m,fontsize=4,color=INK)
  elements.append({'kind':'text','name':'Module swatch '+m,'text':'●','x':1600*x,'y':(1-y)*2210-10,'w':16,'h':17,'size':13.9,'bold':False,'color':c})
  elements.append({'kind':'text','name':'Module legend '+m,'text':m,'x':1600*(x+.010),'y':(1-y)*2210-10,'w':350,'h':17,'size':12.3,'bold':False,'color':INK})
 heading('D','D  Spatial step-niche atlas',.025,.265,.60,8)
 heading('E','E  Circos functional map',.665,.265,.32,8)
 # Original full-section coordinates and scale; no warped tissue map.
 for k,j in enumerate([0,1,2,4]):
  x=.025+(k%2)*.30;y=.034 if k>=2 else .151
  heading('D'+str(k+1)+' title',f'({k+1}) {NAMES[j]}: {IDENT[j]}',x,y+.103,.29,5.0)
  panel('D'+str(k+1),lambda ax,j=j:draw_D(ax,d,a,story,j),(x,y,.27,.075))
 panel('E',lambda ax:draw_E(ax,enrich),(.665,.042,.315,.200))
 foot='Frozen MORPH throughout. Common PCA disk, not hyperbolic inference; shared-node orders are medians across cohorts. Gene sets are restricted to the313 measured genes.\nCompetitive enrichment uses paired spatial-block programme-versus-complement contrasts and joint BH correction. * in D: enrichment-supported; other labels are descriptive.'
 fig.text(.025,.004,foot,fontsize=3.6,color='#657581',va='bottom');elements.append({'kind':'text','name':'Interpretation scope','text':foot,'x':40,'y':2170,'w':1520,'h':35,'size':8.8,'bold':False,'color':'#657581'})
 stem=OUT/'Figure4_root_conditioned_biological_enrichment'
 for ext in ['png','pdf','svg']:fig.savefig(stem.with_suffix('.'+ext),dpi=450,facecolor='white')
 plt.close(fig)
 (OUT/'_build/figure4_manifest.json').write_text(json.dumps({'figure':4,'exact_text_positions':True,'assets':assets,'elements':elements},indent=2))

def main():
 for folder in ['panels','source_data','_build']:(OUT/folder).mkdir(parents=True,exist_ok=True)
 contract={'claim':'Frozen root-conditioned cell-state forests show distinct and shared measured functional programmes and spatially distributed ordered states.','archetype':'asymmetric mixed-modality figure','evidence':'A nested architecture; B complete 43-program descriptive profiles with independently validated dots; C shared programme disk; D original-section step niches; E current measured-panel ORA.','backend':'Python matplotlib','export':'7.2 by 10 inches, PNG450dpi, editable-text PDF/SVG, separate panels and source CSV.','roots_and_graphs':'Only frozen MORPH, never old883-node results. No inference edge or source anchor changed.','cohort_definition':'Six named root/identity forests already used in frozen functional tests; labels b0 etc are cohort display IDs, not newly inferred clonal branches.','step_definition':'Within each frozen cohort, rank pseudotime into four quartiles. No stage-specific discovery claim.','gene_tests':'Nuclear log-depth-normalized expression, exact observed-member unit means. Each step versus all other steps of the SAME root/identity forest, paired mean contrasts within500um blocks; one-sided paired t-test if >=8 blocks, global BH over all6x4x313 tests, positive standardized effect>=.30 and q<=.05. Single-section exploratory comparisons, not patient replicates.','ORA':'GO2023 and Reactome2022 original local libraries; measured-panel intersection>=3 and <=80% assay universe, hypergeometric upper tail, BH jointly over every cohort/step/eligibleterm; report q<=.05 and overlap>=3. Primary43 included in same correction. No whole-genome background, no reused old effects.','disk':'PCA of43 measured programmes; one common radial compression x=.98 PC/(median radius+radius); no pseudotime radius, no root offsets, no hyperbolic/lineage claim.','review_risks':'Exploratory within-section ORA depends on frozen estimated order and targeted panel. It is not independent validation. Frozen held-panel tests are separately indicated;43 is not43 independent signatures. All gene tests and terms exported including nonsignificant results.'}
 (OUT/'_build/FIGURE_CONTRACT.json').write_text(json.dumps(contract,indent=2))
 contract['competitive_enrichment']='Complementary continuous gene-set analysis, retaining the thresholded ORA including its negative result. Within each paired500um block, average gene-standardized step contrasts over the measured genes in the set, minus the mean of genes outside the set. Test these block-level contrasts with a one-sided t-test; retain the joint BH family across all24 cohort/step comparisons and all837 eligible term labels. Genes are not treated as independent replicates. This is exploratory same-section enrichment, not independent validation.'
 contract['display_annotation_filter']={'allowed_modules_by_original_identity':{k:sorted(v) for k,v in DISPLAY_MODULES.items()},'scope':'Display labels and chord summarization only. All tested terms and q-values remain exported, including incompatible-name signatures. Broad modules avoid interpreting e.g. a small T-cell gene-set overlap as stromal-to-T-cell differentiation. No p or q value is changed.'}
 contract['control_display']='Shortest-path comparator uses exactly the same selected endpoint nodes as the current forest display, the same common coordinates, point opacity and line style. Full comparator edges and displayed endpoint-matched paths are both exported. No full-vs-pruned density comparison is made.'
 contract['evidence']='A nested architecture; B all43 descriptive programme profiles plus frozen independent evidence; C common disk; D spatial steps; E continuous competitive gene-set enrichment. Both full ORA and competitive test families are retained.'
 (OUT/'_build/FIGURE_CONTRACT.json').write_text(json.dumps(contract,indent=2))
 count=source_audit();ref.shared.OUT=OUT;d=ref.shared.load();payload=analyse(d);render(d,payload)
 assert source_audit()==count
 ora=pd.read_csv(OUT/'source_data/ALL_MEASURED_PANEL_GO_REACTOME_ORA.csv')
 (OUT/'source_data/SOURCE_AND_QA.json').write_text(json.dumps({'frozen_files_checked':count,'frozen_files_modified':0,'all_skeleton_edges_frozen':True,'old_883_node_or_7_branch_results_used':False,'programmes':43,'independent_eligible_programmes':24,'measured_background_genes':313,'ora_tests':len(ora),'ora_significant':int(ora.significant.sum()),'competitive_gene_set_tests':len(payload[2]),'competitive_significant':int(payload[2].significant.sum()),'edit_scope':'Figure4 interpretation display and new exploratory enrichment only; Fig1/2/3 and frozen methods unchanged.'},indent=2))
 shutil.copy2(Path(__file__),OUT/Path(__file__).name)
 (OUT/'Figure4_legend.tex').write_text(r'''\textbf{Root-conditioned trajectories resolve distinct and shared biological programmes.}
\textbf{A}, Nested display of six root/identity forests and four within-cohort state-order quartiles. Sector widths are equal for readability; they do not encode cell abundance. Solid step fills denote measured-panel enrichment and pale fills denote descriptive programme scores. Display cohort labels do not imply clonal branches.
\textbf{B}, Complete profiles of the 43 measured programmes. Values are mean globally standardized programme activities in each cohort. Open dots mark independently eligible, gene-disjoint frozen conditional-state tests passing the complete-family gate. Identity-overlapping signatures remain descriptive.
\textbf{C}, A common PCA-based programme disk, coloured by cohort, within-cohort order or dominant functional module. A single monotonic radial compression bounds all coordinates. No root-dependent displacement or pseudotime-derived radius is used. Shared nodes retain one coordinate; their order colour is the median cohort-specific rank. Lines in panels1--3 are actual frozen parent edges; panel4 shows source-matched shortest-path forests on the same observed units and frozen candidate graph, displaying paths to the same endpoint nodes with identical line styling and point opacity.
\textbf{D}, Original-tissue distributions of four ordered quartiles in representative cohorts. Colours denote the selected functional programme class and contours indicate smoothed support density. Asterisks identify labels supported by the competitive enrichment analysis; remaining labels describe measured programme activities.
\textbf{E}, Links show significant competitive GO/Reactome enrichment. Gene sets are restricted to the 313 measured genes, with at least three measured members. For each $500\,\mu\mathrm{m}$ block containing both groups, gene-standardized expression contrasts compare a step to other steps of the same root/identity forest. Set-member mean contrasts minus the mean over genes outside the set define one competitive contrast per block. One-sided block-level tests receive joint Benjamini--Hochberg correction over every eligible term and all 24 cohort/step comparisons. Links require $q\leq0.05$ and a positive contrast; widths summarize maximum step-specific evidence for each cohort/module. Thresholded foreground-gene over-representation tests are also supplied in full; none passed the joint FDR threshold. These analyses are exploratory annotations of a single targeted-gene section, not independent validation, experimentally established lineage or causal effects.
''',encoding='utf8')
 print('COMPLETE',OUT,flush=True)
if __name__=='__main__':main()
