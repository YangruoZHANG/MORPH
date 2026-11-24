"""One source registry for Figures1/2/4, actual edges and measured cells only.

Main figures are descriptive state-forest projections, not measured migration
or experimental lineage tracing. Native comparisons are freshly run on the
same current source/observation inputs. Figure3 remains teammate-owned.
"""
import os,sys,json,hashlib,argparse
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'algorithm/visualization'))
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from inspect_current_state_paths_20261025 import representative_paths
BASE=ROOT/'outputs/Count_spatial_agreement_states_v70_20261025/ensemble'
ORIGINAL=ROOT/'outputs/NicheForest_current_research_freeze_20261024'
OUT=ROOT/'outputs/Checked_visualization_suite_20261025'
INK='#202B36';ROOTCOL={'R0':'#C53D56','R1':'#387FB5','R2':'#27916D'}
CASES=[('R0','Stromal'),('R1','B_cell'),('R2','Malignant_epithelial')]
COLORS={'Stromal-low':'#8C9E89','Stromal-high':'#D39A55','B-low':'#477DB2','B-secretory':'#8D72B3','Epi-luminal':'#6A51A3','Epi-basal':'#178F8D','Epi-cycling':'#E17832','Epi-junction':'#C9416B','Epi-unresolved':'#B8C0CA'}
LABELS={'Stromal-low':'ECM-low stromal','Stromal-high':'ECM-high stromal','B-low':'Secretory-low B','B-secretory':'Secretory-high B','Epi-luminal':'Luminal-like tumour','Epi-basal':'Basal-like tumour','Epi-cycling':'Cycling tumour','Epi-junction':'CLDN4/CEACAM6-high','Epi-unresolved':'Unresolved tumour'}
MARKERS={'ECM':['PCOLCE','FBLN1','MMP2','MMP1'],'Secretory':['MZB1','PRDM1','TNFRSF17','DERL3','SLAMF7','TENT5C'],'Luminal':['FOXA1','GATA3','MLPH','AGR3'],'Basal':['KRT5','KRT14','KRT15','KRT6B'],'Cycling':['MKI67','TOP2A','CENPF','PCLAF'],'Junction':['CLDN4','CEACAM6','KLF5']}

def clean(ax,xy,spatial=True,pad=.055):
 ax.set_xticks([]);ax.set_yticks([])
 for s in ax.spines.values():s.set_visible(False)
 ax.set_aspect('equal');lo=np.min(xy,axis=0);hi=np.max(xy,axis=0);span=np.maximum(hi-lo,1);ax.set_xlim(lo[0]-pad*span[0],hi[0]+pad*span[0]);ax.set_ylim(lo[1]-pad*span[1],hi[1]+pad*span[1])
 if spatial:ax.invert_yaxis()

def bar(ax,length=1000):
 x0,x1=ax.get_xlim();y0,y1=ax.get_ylim();x=x0+.055*(x1-x0);y=y0+.06*(y1-y0);ax.plot([x,x+length],[y,y],color=INK,lw=1.1);ax.text(x+length/2,y-.022*abs(y1-y0),f'{length/1000:g} mm',fontsize=5.3,ha='center',va='bottom')

def segment_edges(paths):return np.array(sorted(set((int(u),int(v)) for p in paths for u,v in zip(p[:-1],p[1:]))),int).reshape(-1,2)

def epithelial_stage_paths(nodes,edges):
 # Cover the original three clinical annotation strata, rather than choosing
 # three almost-identical invasive endpoints. This is display stratification,
 # not inferred fate annotation, shape optimization or creation of an edge.
 n=nodes.set_index('node_id');parent=dict(zip(edges.target.astype(int),edges.source.astype(int)));children=set(edges.source.astype(int));leaves=[int(i) for i in n.index if i not in children and i in parent];routes={}
 for v in leaves:
  path=[v]
  while path[-1] in parent:path.append(parent[path[-1]])
  routes[v]=path[::-1]
 used=set();selected=[]
 for category in [('DCIS_1','DCIS_2'),('Invasive_Tumor',),('Prolif_Invasive_Tumor',)]:
  options=[i for i in routes if n.loc[i,'published_cell_type'] in category]
  if not options:continue
  v=max(options,key=lambda i:(sum(n.loc[u,'n_raw_cells'] for u in routes[i] if u not in used),n.loc[i,'pseudotime'],-i));selected.append(routes[v]);used.update(routes[v])
 return selected

def lines(ax,xy,ed,color=INK,width=.72,spatial_scale=None):
 if not len(ed):return
 seg=xy[ed];long=np.zeros(len(ed),bool)
 ax.add_collection(LineCollection(seg,colors='white',linewidths=width+1.1,zorder=5))
 for mask,style in [(~long,'solid'),(long,(0,(3,2)))]:
  if mask.any():ax.add_collection(LineCollection(seg[mask],colors=color,linewidths=width,linestyles=style,zorder=6))
 for a,b in seg:
  if np.linalg.norm(b-a)>1e-8:ax.add_patch(FancyArrowPatch(a+.44*(b-a),a+.62*(b-a),arrowstyle='-|>',mutation_scale=2.8,color=color,lw=.32,zorder=7))

def panel(name,fn,size=(5.4,4.15)):
 # Author each panel at its final physical size, rather than shrinking a
 # 5-inch plot (and its text) into a 2-inch paper column afterwards.
 family,key=name.split('_',1);row=key[0];dims={'Figure1':{'A':(2.196,1.68),'B':(3.33,1.88),'C':(2.196,1.63),'D':(2.196,1.55)},'Figure2':{'A':(2.196,1.27),'B':(2.196,1.25),'C':(3.25,1.33),'D':(2.196,1.30),'E':(1.656,.77)},'Figure4':{'A':(2.88,1.78) if key=='A1' else (3.492,2.30),'B':(3.312,1.70),'C':(2.196,1.20),'D':(2.196,1.11),'E':(3.312,1.30)}};size=dims[family][row]
 fig,ax=plt.subplots(figsize=size);fn(ax)
 if ax.get_xlabel() or ax.get_ylabel() or any(t.get_text() for t in ax.get_xticklabels()):fig.tight_layout(pad=.55)
 else:fig.subplots_adjust(left=.025,right=.82 if name=='Figure2_B3' else .975,bottom=.025,top=.75 if name.startswith('Figure1_C') or name.startswith('Figure4_D') else .975)
 file=OUT/'panels'/f'{name}.png';fig.savefig(file,dpi=400,facecolor='white');fig.savefig(file.with_suffix('.pdf'));fig.savefig(file.with_suffix('.svg'));plt.close(fig);return str(file)

def plate(number,assets,rows,foot):
 fig=plt.figure(figsize=(7.2,10.0));elements=[]
 for letter,title,y,items in rows:
  fig.text(.025,y,letter+'  '+title,fontsize=10.4,weight='bold',va='top',color=INK);elements.append({'kind':'text','name':letter,'text':letter+'  '+title,'x':40,'y':(1-y)*2210,'w':1520,'h':52,'size':25,'bold':True})
  for key,title,rect in items:
   ax=fig.add_axes(rect);ax.imshow(plt.imread(assets[key]));ax.axis('off');fig.text(rect[0]+.002,rect[1]+rect[3]+.006,title,fontsize=6.7,weight='bold',color=INK,va='bottom');elements.extend([{'kind':'image','name':key,'key':key,'x':rect[0]*1600,'y':(1-rect[1]-rect[3])*2210,'w':rect[2]*1600,'h':rect[3]*2210},{'kind':'text','name':key+' title','text':title,'x':rect[0]*1600,'y':(1-rect[1]-rect[3]-.029)*2210,'w':rect[2]*1600,'h':40,'size':16,'bold':True}])
 fig.text(.028,.009,foot,fontsize=4.7,color='#667784',va='bottom');stem=OUT/f'Figure{number}_checked_state_forests'
 for ext in ['png','pdf','svg']:fig.savefig(stem.with_suffix('.'+ext),dpi=400,facecolor='white')
 plt.close(fig);manifest={'figure':number,'assets':{k:{'path':v} for k,v in assets.items()},'elements':elements};(OUT/'_build'/f'figure{number}_manifest.json').write_text(json.dumps(manifest,indent=2));return manifest

def load():
 d={};d['cells']=pd.read_csv(ORIGINAL/'data/PUBLISHED_CELL_METADATA.csv',dtype={'cell_name':str});d['raw']=ad.read_h5ad(ORIGINAL/'data/BREAST_ORIGINAL_COUNTS.h5ad');d['units']=pd.read_csv(BASE/'EXACT_STATE_UNITS.csv');d['members']=pd.read_csv(BASE/'EVERY_OBSERVED_CELL_STATE_MEMBERSHIP.csv');d['xy']=d['units'][['spatial_x','spatial_y']].to_numpy();d['rawxy']=d['cells'][['spatial_x','spatial_y']].to_numpy();d['roots']=pd.read_csv(ORIGINAL/'data/CANONICAL_SPATIAL_ANCHORS.csv').set_index('root_id');embedding=pd.read_csv(ORIGINAL/'data/RAW_CELL_UMAP_COORDINATES.csv',dtype={'cell_name':str});assert np.array_equal(embedding.cell_name,d['cells'].cell_name);d['rawumap']=embedding[['umap_1','umap_2']].to_numpy();reps=d['units'].representative_cell_name.astype(str);rawindex=pd.Index(d['cells'].cell_name).get_indexer(reps);assert (rawindex>=0).all();d['umap']=d['rawumap'][rawindex];d['g']=sparse.load_npz(BASE/'WholeCell/FULL_CANDIDATE_COSTS.npz');d['fits']={};d['paths']={};d['scale']=np.ones(len(d['units']))
 for identity in d['units'].identity.unique():
  ids=np.flatnonzero(d['units'].identity.eq(identity));distance,_=cKDTree(d['xy'][ids]).query(d['xy'][ids],k=min(31,len(ids)));d['scale'][ids]=np.maximum(distance[:,-1],1e-8)
 for f in (BASE/'WholeCell').glob('*/FULL_STATE_NODES.csv'):
  key=tuple(f.parent.name.split('_',1));n=pd.read_csv(f);e=pd.read_csv(f.parent/'FULL_STATE_EDGES.csv');d['fits'][key]=(n,e);d['paths'][key]=epithelial_stage_paths(n,e) if key==('R2','Malignant_epithelial') else representative_paths(n,e)
 d['tests']=pd.read_csv(BASE/'complete_catalog/COMPLETE_FAMILY_STATE_ASSOCIATIONS.csv');d['unit_of_raw']=np.full(len(d['cells']),-1,int);d['unit_of_raw'][d['members'].raw_row]=d['members'].unit_node_id
 # Measured marker scores, raw-cell depth normalization then exact unit mean.
 x=sparse.csr_matrix(d['raw'].X);nuc=sparse.load_npz(ROOT/'outputs/Original_nuclear_molecule_audit_v46_20261024/NUCLEAR_Q20_ORIGINAL_CELL_GENE_ORDER.npz');mean=sparse.load_npz(BASE/'MEAN_STATE_OPERATOR.npz');core=pd.read_csv(ROOT/'outputs/Marker_disjoint_lineage_state_forests_v59_20261025/EXACT_CORE_CELL_INPUT.csv');d['core']=core;d['mean']=mean;d['nuc']=nuc;depth=np.maximum(np.asarray(x.sum(1)).ravel(),1);scores={}
 for name,genes in MARKERS.items():
  cols=d['raw'].var_names.get_indexer(genes);cols=cols[cols>=0];v=np.log1p(x[:,cols].toarray()/depth[:,None]*np.median(depth));z=np.zeros_like(v)
  for identity in d['cells'].identity.dropna().unique():
   ids=np.flatnonzero(d['cells'].identity.eq(identity));z[ids]=StandardScaler().fit_transform(v[ids])
  scores[name]=np.asarray(mean@z[core.raw_row.to_numpy(int)]).mean(1)
 d['scores']=pd.DataFrame(scores);cls=np.repeat('Other',len(d['units'])).astype(object);ids=d['units'].identity.eq('Stromal');cls[ids]=np.where(d['scores'].loc[ids,'ECM']>0,'Stromal-high','Stromal-low');ids=d['units'].identity.eq('B_cell');cls[ids]=np.where(d['scores'].loc[ids,'Secretory']>0,'B-secretory','B-low');ids=d['units'].identity.eq('Malignant_epithelial');v=d['scores'].loc[ids,['Luminal','Basal','Cycling','Junction']].to_numpy();cls[ids]=np.where(v.max(1)>0,np.array(['Epi-luminal','Epi-basal','Epi-cycling','Epi-junction'])[v.argmax(1)],'Epi-unresolved');d['classes']=cls;d['units'].assign(display_class=cls).to_csv(OUT/'source_data/EXACT_UNITS_AND_DISPLAY_MARKER_PROGRAMS.csv',index=False);d['scores'].to_csv(OUT/'source_data/MEASURED_DISPLAY_MARKER_SCORES.csv',index=False)
 return d

def nearby_units(d,pathnodes,identity,support=None,graph=None):
 g=d['g'] if graph is None else graph;take=set(map(int,pathnodes));allowed=None if support is None else set(map(int,support))
 for u in pathnodes:
  nn=g.indices[g.indptr[u]:g.indptr[u+1]];distance=np.linalg.norm(d['xy'][nn]-d['xy'][u],axis=1);ok=(distance<=np.sqrt(d['scale'][u]*d['scale'][nn]))&d['units'].identity.iloc[nn].eq(identity).to_numpy()
  take.update(int(i) for i in nn[ok] if allowed is None or int(i) in allowed)
 return np.array(sorted(take),int)

def background(ax,d,unitids,legend=False):
 rows=d['members'].loc[d['members'].unit_node_id.isin(unitids),'raw_row'].to_numpy(int);context=cKDTree(d['rawxy'][rows]).query(d['rawxy'])[0]<=30 if len(rows) else np.zeros(len(d['cells']),bool);ax.scatter(*d['rawxy'][context].T,s=.18,c='#DDE3E8',alpha=.35,lw=0,rasterized=True,zorder=1);present=[]
 for c in COLORS:
  own=rows[d['classes'][d['unit_of_raw'][rows]]==c]
  if len(own):ax.scatter(*d['rawxy'][own].T,s=.85,c=COLORS[c],alpha=.78,lw=0,rasterized=True,zorder=2);present.append(c)
 if legend:ax.legend(handles=[Line2D([],[],marker='o',ms=2.6,ls='',c=COLORS[c],label=LABELS[c]) for c in present],loc='upper right',bbox_to_anchor=(1,1.29),ncol=2,fontsize=4.9,frameon=False,labelspacing=.3,borderpad=0,handletextpad=.25,columnspacing=.6)
 return rows

def trajectory(ax,d,key,legend=True,full=False):
 n,_=d['fits'][key];paths=d['paths'][key];ed=segment_edges(paths);rows=context_along_edges(ax,d,ed,key[1],legend=legend);ids=np.unique(d['unit_of_raw'][rows]);lines(ax,d['xy'],ed,spatial_scale=d['scale']);root=d['roots'].loc[key[0],['x','y']].to_numpy(float);ax.scatter(*root,marker='*',s=35,c=INK,edgecolor='white',lw=.5,zorder=10)
 for p in paths:ax.scatter(*d['xy'][p[-1]],s=10,fc='white',ec=INK,lw=.65,zorder=9)
 extent=d['rawxy'] if full else np.vstack([d['rawxy'][rows],d['xy'][np.unique(ed)],root]);clean(ax,extent);bar(ax);return ed,ids

def context_along_edges(ax,d,ed,identity,legend=False):
 # Real local tissue context is not an asserted list of ancestors. The same
 # corridor rule applies to our and native-method paths; it does not feed the
 # inference or tests, and it never creates or moves a cell/trajectory point.
 samples=np.concatenate([np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/35))) for a,b in d['xy'][ed]]);mask=cKDTree(samples).query(d['rawxy'])[0]<=150;ax.scatter(*d['rawxy'][mask].T,s=.13,c='#DDE3E8',alpha=.30,lw=0,rasterized=True);rows=np.flatnonzero(mask&d['cells'].identity.eq(identity).to_numpy()&(d['unit_of_raw']>=0));present=[]
 for c in COLORS:
  own=rows[d['classes'][d['unit_of_raw'][rows]]==c]
  if len(own):ax.scatter(*d['rawxy'][own].T,s=.24,c=COLORS[c],alpha=.78,lw=0,rasterized=True,zorder=2);present.append(c)
 if legend:ax.legend(handles=[Line2D([],[],marker='o',ms=2.6,ls='',c=COLORS[c],label=LABELS[c]) for c in present],loc='upper right',bbox_to_anchor=(1,1.29),ncol=2,fontsize=4.9,frameon=False,labelspacing=.3,borderpad=0,handletextpad=.25,columnspacing=.6)
 return rows

def overview(ax,d,space='spatial',candidate=False):
 xy=d['xy'] if space=='spatial' else d['umap'];rawxy=d['rawxy'] if space=='spatial' else d['rawumap'];ax.scatter(*rawxy[::3].T,s=.16,c='#DFE4E8',alpha=.32,lw=0,rasterized=True)
 for key in CASES:
  n,_=d['fits'][key];ids=n.node_id.to_numpy(int)
  if space=='spatial':
   rr=d['members'].loc[d['members'].unit_node_id.isin(ids),'raw_row'].to_numpy(int);ax.scatter(*rawxy[rr].T,s=.12,c=ROOTCOL[key[0]],alpha=.27,lw=0,rasterized=True)
  else:ax.scatter(*xy[ids].T,s=.28,c=ROOTCOL[key[0]],alpha=.30,lw=0,rasterized=True)
  if candidate:
   coo=d['g'][ids][:,ids].tocoo();keep=coo.row<coo.col;u=ids[coo.row[keep]];v=ids[coo.col[keep]];ix=np.linspace(0,len(u)-1,min(3000,len(u)),dtype=int);ax.add_collection(LineCollection(xy[np.column_stack([u[ix],v[ix]])],colors=ROOTCOL[key[0]],lw=.2,alpha=.09,rasterized=True));continue
  ed=segment_edges(d['paths'][key]);lines(ax,xy,ed,ROOTCOL[key[0]],.75,d['scale'] if space=='spatial' else None)
  point=d['roots'].loc[key[0],['x','y']].to_numpy(float) if space=='spatial' else xy[int(n.loc[n.is_source_state,'node_id'].iloc[0])];ax.scatter(*point,marker='*' if space=='spatial' else 'o',s=45 if space=='spatial' else 17,c=ROOTCOL[key[0]],ec='white',lw=.5,zorder=10)
 clean(ax,rawxy,space=='spatial')
 if space=='spatial':bar(ax)
 # Identity colours are defined once in the figure caption and neighbouring
 # labelled root panels. Do not cover measured tissue with repeated legends.

def native_panel(ax,d,method):
 allids=[];empty=[]
 for key in CASES:
  folder=BASE/'native_methods'/key[0];status=json.loads((folder/'STATUS.json').read_text());entry=next(x for x in status if x['method']==method)
  if entry['status']!='completed':raise RuntimeError(f'Native method did not complete: {entry}')
  e=pd.read_csv(folder/f'{method}_PATH_EDGES.csv');ed=e[['source','target']].to_numpy(int)
  if not len(ed):empty.append(key[0]);continue
  context_along_edges(ax,d,ed,key[1]);lines(ax,d['xy'],ed,width=.55,spatial_scale=d['scale']);allids.extend(np.unique(ed))
  point=d['roots'].loc[key[0],['x','y']].to_numpy(float);ax.scatter(*point,marker='*',s=44,c=ROOTCOL[key[0]],ec='white',lw=.4,zorder=10)
 clean(ax,d['rawxy']);bar(ax)
 if empty:ax.text(.02,.99,', '.join(empty)+': no terminal called',transform=ax.transAxes,va='top',fontsize=5.5,color='#697884')

def figure1(d):
 assets={};palette=json.loads((ORIGINAL/'data/FIXED_PARTITION_PALETTE.json').read_text());cluster=d['cells'].fixed_graph_partition;top=list(dict.fromkeys(cluster.value_counts().head(10).index.tolist()+['C13','C00','C21']))
 for j,space in enumerate(['spatial','umap'],1):
  def draw(ax,space=space):
   xy=d['rawxy'] if space=='spatial' else d['rawumap']
   for name,color in palette.items():
    ids=np.flatnonzero(cluster.eq(name))[::2];ax.scatter(*xy[ids].T,s=.25,c=[color],alpha=.72,lw=0,rasterized=True)
   for name in top:
    q=np.median(xy[cluster.eq(name)],axis=0);ax.text(*q,name,ha='center',fontsize=4.6,bbox=dict(fc='white',ec='#D4DCE1',lw=.3,pad=.5,alpha=.85))
   clean(ax,xy,space=='spatial')
  assets[f'A{j}']=panel(f'Figure1_A{j}',draw)
 # Same fixed64-cluster composition, top identity strip and lower purity.
 # Redraw at final size so the legend remains readable in the paper.
 comp=pd.read_csv(ORIGINAL/'evidence/FIG1_CLUSTER_COMPOSITION.csv',index_col=0);comp.to_csv(OUT/'source_data/UNCHANGED_FIXED_CLUSTER_COMPOSITION.csv');dominant=comp.idxmax(0);family_palette=dict(zip(comp.index,['#4B9DA8','#4D68A8','#C55068','#8C70AE','#36958A','#BDC4CC','#779550','#CA8A48']));short={'Endothelial':'Endo.','Lymphoid':'Lymph.','Malignant_epithelial':'Tumour','Myeloid':'Myeloid','Myoepithelial':'Myoepi.','Other/unresolved':'Unres.','Perivascular':'Perivasc.','Stromal':'Stromal'};fig=plt.figure(figsize=(2.196,1.68));ax=fig.add_axes([.24,.29,.64,.48]);im=ax.imshow(comp,aspect='auto',cmap='Blues',vmin=0,vmax=1);ax.set_yticks(range(len(comp)),[short.get(s,s) for s in comp.index],fontsize=4.9);ax.set_xticks([]);ax.tick_params(length=0)
 for label,name in zip(ax.get_yticklabels(),comp.index):label.set_color(family_palette[name])
 cb=fig.add_axes([.91,.29,.025,.48]);fig.colorbar(im,cax=cb,ticks=[0,1]);cb.tick_params(labelsize=4.6,length=1)
 strip=fig.add_axes([.24,.81,.64,.05]);strip.imshow(np.array([[matplotlib.colors.to_rgba(family_palette[dominant[k]]) for k in comp.columns]]),aspect='auto');strip.axis('off');purity=fig.add_axes([.24,.10,.64,.10]);purity.bar(range(len(comp.columns)),comp.max(0),color=[family_palette[dominant[k]] for k in comp.columns],width=.95,lw=0);purity.set_ylim(0,1);purity.set_xlim(-.5,len(comp.columns)-.5);purity.set_xticks([]);purity.set_yticks([0,1],['0','1'],fontsize=4.6);purity.set_ylabel('Purity',fontsize=4.8,labelpad=1);fig.text(.56,.21,'64 fixed graph partitions',ha='center',fontsize=4.8)
 assets['A3']=str(OUT/'panels/Figure1_A3.png')
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'panels/Figure1_A3.{ext}',dpi=400)
 plt.close(fig)
 for j,space in enumerate(['spatial','umap'],1):assets[f'B{j}']=panel(f'Figure1_B{j}',lambda ax,s=space:overview(ax,d,s),size=(7.0,4.9))
 paths=[];stats=[]
 for j,key in enumerate(CASES,1):
  assets[f'C{j}']=panel(f'Figure1_C{j}',lambda ax,k=key:trajectory(ax,d,k),size=(5.5,4.9));ed=segment_edges(d['paths'][key]);extent=np.ptp(d['xy'][np.unique(ed)],axis=0)/np.ptp(d['rawxy'],axis=0);stats.append({'root_id':key[0],'identity':key[1],'n_actual_edges_shown':len(ed),'x_fraction_of_section':extent[0],'y_fraction_of_section':extent[1],'median_edge_um':float(np.median(np.linalg.norm(d['xy'][ed[:,0]]-d['xy'][ed[:,1]],axis=1)))})
  paths.extend({'root_id':key[0],'identity':key[1],'path_id':p,'order':k,'unit_node_id':int(v)} for p,route in enumerate(d['paths'][key]) for k,v in enumerate(route))
 pd.DataFrame(paths).to_csv(OUT/'source_data/EXACT_DISPLAYED_FOREST_PATHS.csv',index=False);pd.DataFrame(stats).to_csv(OUT/'source_data/FIGURE1_ACTUAL_PATH_EXTENTS.csv',index=False)
 for j,method in enumerate(['CellRank_DPT','PAGA_DPT','Palantir'],1):assets[f'D{j}']=panel(f'Figure1_D{j}',lambda ax,m=method:native_panel(ax,d,m),size=(5.5,4.9))
 rows=[('A','Fixed graph partitions and cell identities',.981,[(f'A{j+1}',s,(.025+j*.323,.766,.305,.168)) for j,s in enumerate(['(1) Original tissue coordinates','(2) Fixed expression UMAP','(3) Identity composition by partition'])]),('B','Root-conditioned state trajectories',.731,[('B1','(1) Original tissue section',(.025,.503,.465,.188)),('B2','(2) Same states and edges in UMAP',(.515,.503,.46,.188))]),('C','Tissue trajectories with identity-specific marker states',.468,[(f'C{j+1}',s,(.025+j*.323,.264,.305,.163)) for j,s in enumerate(['(1) R0: stromal states','(2) R1: B-cell states','(3) R2: stage-conditioned epithelium'])]),('D','Expression-based methods projected into tissue',.229,[(f'D{j+1}',s,(.025+j*.323,.034,.305,.155)) for j,s in enumerate(['(1) CellRank (DPT kernel)','(2) PAGA+DPT','(3) Palantir'])])]
 return plate(1,assets,rows,'Measured cells and inferred state edges only; lines do not establish physical migration or experimental ancestry.\nC: cropped tissue context, not an ancestor assignment. D: current common supplied-start samples; colours follow C. R2 uses the clinical-stage prior.')

def affinities(d):
 units=d['units'];prob=np.zeros((len(units),3))
 for identity in units.identity.unique():
  ids=np.flatnonzero(units.identity.eq(identity));rids=[rid for rid in ROOTCOL if (rid,identity) in d['fits']]
  if not rids:continue
  dist=np.array([dijkstra(d['g'],directed=False,indices=d['fits'][rid,identity][0].loc[lambda a:a.is_source_state,'node_id'].to_numpy(int),min_only=True)[ids] for rid in rids]);near=dist.min(0);valid=np.isfinite(near);positive=near[valid&(near>0)];scale=np.median(positive) if len(positive) else 1;w=np.zeros_like(dist);w[:,valid]=np.exp(-(dist[:,valid]-near[valid])/max(scale,1e-8))
  for k,rid in enumerate(rids):
   keep=np.isin(ids,d['fits'][rid,identity][0].node_id);w[k,~keep]=0
  w/=np.maximum(w.sum(0),1e-12)
  for k,rid in enumerate(rids):prob[ids,int(rid[1])]=w[k]
 pd.DataFrame(prob,columns=['R0','R1','R2']).to_csv(OUT/'source_data/ROOT_SUPPORT_AFFINITIES.csv',index_label='unit_node_id');return prob

def contacts(d):
 x=sparse.csr_matrix(d['raw'].X);depth=np.maximum(np.asarray(x.sum(1)).ravel(),1);records=[]
 for event,skey,tkey,ligand,receptor,description in [('E1',('R0','Stromal'),('R1','B_cell'),'CXCL12','CXCR4','Recruitment-like hypothesis'),('E2',('R2','Malignant_epithelial'),('R1','CD8_T'),'CD274','PDCD1','Inhibition-like hypothesis')]:
  sourceunits=d['fits'][skey][0].node_id;targetunits=d['fits'][tkey][0].node_id;sr=d['members'].loc[d['members'].unit_node_id.isin(sourceunits),'raw_row'].to_numpy(int);tr=d['members'].loc[d['members'].unit_node_id.isin(targetunits),'raw_row'].to_numpy(int);lv=np.asarray(x[:,d['raw'].var_names.get_loc(ligand)].toarray()).ravel();rv=np.asarray(x[:,d['raw'].var_names.get_loc(receptor)].toarray()).ravel();sr=sr[lv[sr]>0];tr=tr[rv[tr]>0]
  if not len(sr) or not len(tr):continue
  tree=cKDTree(d['rawxy'][sr]);near=tree.query_ball_point(d['rawxy'][tr],r=60)
  for target,nn in zip(tr,near):
   for j in nn:
    source=int(sr[j]);distance=float(np.linalg.norm(d['rawxy'][source]-d['rawxy'][target]));score=float(np.sqrt(np.log1p(lv[source]/depth[source]*np.median(depth))*np.log1p(rv[target]/depth[target]*np.median(depth)))*np.exp(-.5*(distance/30)**2));records.append({'event':event,'source_raw_row':source,'target_raw_row':int(target),'source_root':skey[0],'source_identity':skey[1],'target_root':tkey[0],'target_identity':tkey[1],'ligand':ligand,'receptor':receptor,'distance_um':distance,'descriptive_contact_score':score,'hypothesis':description,'formal_call':False})
 result=pd.DataFrame(records);result.to_csv(OUT/'source_data/ALL_OBSERVED_LIGAND_RECEPTOR_CONTACT_CANDIDATES.csv',index=False);return result

def figure2(d):
 assets={};prob=affinities(d);valid=prob.sum(1)>0;xy=d['umap']
 for j,key in enumerate(CASES,1):
  def draw(ax,key=key):
   ax.scatter(*d['rawumap'][::4].T,s=.16,c='#E0E5E9',alpha=.3,lw=0,rasterized=True);n=d['fits'][key][0];ids=n.node_id.to_numpy(int);ax.scatter(*xy[ids].T,s=1.2,c=ROOTCOL[key[0]],alpha=.45,lw=0,rasterized=True);lines(ax,xy,segment_edges(d['paths'][key]),ROOTCOL[key[0]],.65);clean(ax,d['rawumap'],False)
  assets[f'A{j}']=panel(f'Figure2_A{j}',draw)
 for j in [1,2,3]:
  def draw(ax,j=j):
   ax.scatter(*xy[~valid].T,s=.5,c='#E3E7EB',lw=0,rasterized=True)
   if j==1:
    for k,rid in enumerate(ROOTCOL):
     ix=valid&(prob.argmax(1)==k);ax.scatter(*xy[ix].T,s=.6,c=ROOTCOL[rid],alpha=.6,lw=0,rasterized=True)
   elif j==2:
    shared=(prob>=.2).sum(1)>1
    for mask,c in [(valid&~shared,'#8296AB'),(valid&shared,'#9275B3')]:ax.scatter(*xy[mask].T,s=.65,c=c,alpha=.65,lw=0,rasterized=True)
    ax.legend(handles=[Line2D([],[],ls='',marker='o',ms=3,c=c,label=t) for c,t in [('#8296AB','One regional support'),('#9275B3','Shared regional support')]],fontsize=6,loc='upper right')
   else:
    entropy=-(prob*np.log(np.maximum(prob,1e-15))).sum(1)/np.log(3);im=ax.scatter(*xy[valid].T,s=.65,c=entropy[valid],cmap='YlOrBr',vmin=0,vmax=1,lw=0,rasterized=True);cax=ax.figure.add_axes([.85,.17,.025,.66]);cb=ax.figure.colorbar(im,cax=cax,ticks=[0,.5,1]);cb.ax.tick_params(labelsize=4.8,length=1)
   clean(ax,d['rawumap'],False)
  assets[f'B{j}']=panel(f'Figure2_B{j}',draw)
 old=ROOT/'outputs/Regional_source_ensemble_states_v63_20261025/ensemble/WholeCell';changes=[]
 for key,(n,e) in d['fits'].items():
  f=old/f'{key[0]}_{key[1]}/FULL_STATE_EDGES.csv';earlier=pd.read_csv(f).set_index('target').source;later=e.set_index('target').source;common=earlier.index.intersection(later.index);eligible=common[earlier.loc[common].isin(n.node_id)];changes.append({'root_id':key[0],'identity':key[1],'n_common_targets':len(common),'n_targets_with_old_parent_still_present':len(eligible),'n_reselected_parents_on_retained_candidates':int(np.sum(earlier.loc[eligible].to_numpy()!=later.loc[eligible].to_numpy()))})
 changed=pd.DataFrame(changes);changed.to_csv(OUT/'source_data/ACTUAL_PARENT_RESELECTION_NOT_JUST_PRUNING.csv',index=False)
 def rewiring(ax):
  a=changed.set_index(['root_id','identity']).loc[CASES];value=(a.n_reselected_parents_on_retained_candidates/a.n_targets_with_old_parent_still_present).tolist();value.append(changed.n_reselected_parents_on_retained_candidates.sum()/changed.n_targets_with_old_parent_still_present.sum());ax.barh(range(4),value,color=[*ROOTCOL.values(),'#8497AA'],height=.6);ax.set_yticks(range(4),['R0 stromal','R1 B cells','R2 epithelial','All18 forests'],fontsize=6);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('Reselected parent fraction');ax.grid(axis='x',alpha=.12)
 assets['C1']=panel('Figure2_C1',rewiring,size=(6,5))
 oldbench=pd.read_csv(ROOT/'outputs/Regional_state_matched_unit_benchmark_v66_20261025/ALL_MATCHED_UNIT_SCORES.csv');newbench=pd.read_csv(ROOT/'outputs/Count_spatial_agreement_matched_unit_benchmark_v70_20261025/ALL_MATCHED_UNIT_SCORES.csv');a=oldbench.loc[oldbench.method.eq('RegionalMultipath')].set_index('case');b=newbench.loc[newbench.method.eq('RegionalMultipath')].set_index('case');delta=b[['fine_edge_F1','pseudotime_spearman']]-a[['fine_edge_F1','pseudotime_spearman']];delta.to_csv(OUT/'source_data/PAIRED_60_DEVELOPMENT_EFFECTS.csv');rng=np.random.default_rng(202610250);ci=np.quantile(np.array([delta.iloc[rng.integers(0,len(delta),len(delta))].mean().to_numpy() for _ in range(10000)]),[.025,.975],axis=0)
 def effects(ax):
  for j,col in enumerate(delta):
   values=delta[col].to_numpy();ax.scatter(values,np.repeat(j,len(values))+np.linspace(-.12,.12,len(values)),s=7,c='#B5C8D7',alpha=.6);mean=values.mean();ax.errorbar(mean,j,xerr=[[mean-ci[0,j]],[ci[1,j]-mean]],fmt='D',ms=5,c=INK,capsize=3,zorder=5)
  ax.axvline(0,c='#9EABB7',ls='--',lw=.7);ax.set_yticks([0,1],['State-adjacency F1','Pseudotime correlation']);ax.set_xlabel('Paired change from bounded-spatial model');ax.grid(axis='x',alpha=.12);ax.set_ylim(1.5,-.5)
 assets['C2']=panel('Figure2_C2',effects,size=(6,5));contact=contacts(d);events=contact.sort_values('descriptive_contact_score',ascending=False).drop_duplicates('event')
 def eventmap(ax):
  overview(ax,d)
  for r in events.itertuples():
   q=d['rawxy'][r.target_raw_row];ax.scatter(*q,s=35,fc='white',ec=INK,lw=.8,zorder=10);ax.annotate(r.event,q,xytext=(5,4),textcoords='offset points',fontsize=7,weight='bold')
 assets['D1']=panel('Figure2_D1',eventmap,size=(7,5.2))
 for j,event in enumerate(['E1','E2'],2):
  def eventzoom(ax,event=event):
   a=events.loc[events.event.eq(event)]
   if not len(a):ax.text(.5,.5,'No eligible measured contact',ha='center',transform=ax.transAxes);ax.axis('off');return
   r=a.iloc[0];q=d['rawxy'][int(r.target_raw_row)];rows=np.flatnonzero(np.linalg.norm(d['rawxy']-q,axis=1)<=350);ax.scatter(*d['rawxy'][rows].T,s=.7,c='#DDE3E8',alpha=.5,lw=0);pair=contact.loc[contact.event.eq(event)];local=pair.loc[np.linalg.norm(d['rawxy'][pair.target_raw_row.to_numpy(int)]-q,axis=1)<=350]
   for column,c in [('source_raw_row',ROOTCOL[r.source_root]),('target_raw_row',ROOTCOL[r.target_root])]:
    ix=np.unique(local[column]);ax.scatter(*d['rawxy'][ix].T,s=4,c=c,alpha=.8,lw=0)
   u=d['rawxy'][int(r.source_raw_row)];v=d['rawxy'][int(r.target_raw_row)];ax.add_patch(FancyArrowPatch(u,v,arrowstyle='-|>',mutation_scale=8,ls='--',lw=1,color=INK,zorder=5));clean(ax,np.vstack([q-350,q+350]));bar(ax,200);ax.text(.02,.99,str(r.ligand)+' / '+str(r.receptor)+'\nCandidate only',va='top',transform=ax.transAxes,fontsize=6)
  assets[f'D{j}']=panel(f'Figure2_D{j}',eventzoom,size=(4.3,3.3))
 for j,(space,candidate) in enumerate([('umap',True),('umap',False),('spatial',True),('spatial',False)],1):assets[f'E{j}']=panel(f'Figure2_E{j}',lambda ax,s=space,c=candidate:overview(ax,d,s,c),size=(4.4,4.0))
 rows=[('A','Root-specific trajectories in one fixed expression space',.982,[(f'A{j+1}',s,(.025+j*.323,.806,.305,.127)) for j,s in enumerate(['(1) R0: stromal states','(2) R1: B-cell states','(3) R2: epithelial states'])]),('B','Soft regional assignment and shared support',.777,[(f'B{j+1}',s,(.025+j*.323,.610,.305,.125)) for j,s in enumerate(['(1) Highest regional support','(2) Shared versus single support','(3) Membership entropy'])]),('C','Evidence-driven changes in the reconstructed forests',.578,[('C1','(1) Matched parent reselection',(.025,.396,.45,.133)),('C2','(2) Same-unit simulated development comparison',(.52,.396,.455,.133))]),('D','Cross-tree contacts and signalling-compatible hypotheses',.367,[(f'D{j+1}',s,(.025+j*.323,.183,.305,.130)) for j,s in enumerate(['(1) Observed contact locations','(2) Recruitment-like contact','(3) Inhibition-like contact'])]),('E','Count-admissible candidates and representative forests',.137,[(f'E{j+1}',s,(.025+j*.244,.022,.23,.077)) for j,s in enumerate(['(1) UMAP candidates','(2) UMAP forest','(3) Tissue candidates','(4) Tissue forest'])])]
 return plate(2,assets,rows,'C: 60 reused development simulations; points are datasets, diamonds means, intervals paired bootstrap 95% CI.\nD: measured ligand/receptor-positive contacts are hypotheses, not causal calls or fitted promotion/inhibition effects.')

def figure4(d):
 assets={};groups=[('R0','Stromal'),('R1','B_cell'),('R1','CD8_T'),('R1','Macrophage'),('R2','Malignant_epithelial'),('R2','Stromal')];tests=d['tests'].loc[d['tests'].method.eq('Multipath')];all_panels=tests.panel.drop_duplicates().tolist();panels=[x for x in all_panels if '::' not in x or x.split('::')[-1] in ['Cholesterol Homeostasis','Complement','mTORC1 Signaling','Xenobiotic Metabolism','STRESS_RESPONSE','heme Metabolism']];mat=np.full((len(panels),len(groups)*2),np.nan);mark=np.zeros_like(mat,bool)
 for i,name in enumerate(panels):
  for j,key in enumerate(groups):
   for k,view in enumerate(['WholeCell','Nuclear']):
    a=tests.loc[tests.panel.eq(name)&tests.root_id.eq(key[0])&tests.identity.eq(key[1])&tests.count_view.eq(view)]
    if len(a):mat[i,2*j+k]=a.iloc[0].partial_rho;mark[i,2*j+k]=bool(a.iloc[0].usable_for_conditional_state_naming)
 pd.DataFrame(mat,index=panels,columns=[r+' '+t+' '+v for r,t in groups for v in ['WholeCell','Nuclear']]).to_csv(OUT/'source_data/FIGURE4_DISPLAYED_FUNCTION_MATRIX.csv')
 tests.pivot_table(index='panel',columns=['root_id','identity','count_view'],values='partial_rho').to_csv(OUT/'source_data/FIGURE4_COMPLETE_FUNCTION_MATRIX.csv')
 def composition(ax):
  a=pd.read_csv(BASE/'FULL_RECONSTRUCTION_AUDIT.csv');a=a.loc[a.count_view.eq('WholeCell')];table=a.pivot(index='identity',columns='root_id',values='n_observed_raw_cells').fillna(0);bottom=np.zeros(len(table))
  for rid in ROOTCOL:ax.barh(range(len(table)),table[rid],left=bottom,color=ROOTCOL[rid],height=.72,label=rid);bottom+=table[rid]
  ax.set_yticks(range(len(table)),table.index.str.replace('_',' '),fontsize=6);ax.set_xlabel('Root-cell assignments (shared cells can recur)');ax.legend(fontsize=6,ncol=3);ax.grid(axis='x',alpha=.12)
 assets['A1']=panel('Figure4_A1',composition,size=(6,4.3))
 def heat(ax):
  alias=['ECM','T naive / trafficking','T cytotoxic','T checkpoint','Luminal','Junctional','B secretory','Contractile','Inflammatory myeloid','C1Q / MRC1','Stress','Cholesterol','Complement','mTORC1','Xenobiotic','Heme'];im=ax.imshow(mat,cmap='RdBu_r',vmin=-.65,vmax=.65,aspect='auto');ax.set_yticks(range(len(panels)),alias,fontsize=5.3);ax.set_xticks(np.arange(len(groups))*2+.5,['R0\nStromal','R1\nB','R1\nCD8','R1\nMyeloid','R2\nTumour','R2\nStromal'],fontsize=5.0);ax.tick_params(length=0)
  for i,j in zip(*np.where(mark)):ax.text(j,i,'•',fontsize=6.3,ha='center',va='center')
  for j in np.arange(.5,len(groups)*2,2):ax.axvline(j,c='white',lw=.5,alpha=.5)
  ax.figure.colorbar(im,ax=ax,fraction=.025,pad=.025,label='Partial rank correlation');ax.set_title('Columns: whole-cell / nuclear features',fontsize=6.5,pad=7)
 assets['A2']=panel('Figure4_A2',heat,size=(7.4,6.6))
 # Ordinary shared PCA of measured marker-program scores. No root offsets,
 # artificial radial pseudotime or claim of hyperbolic/ancestral geometry.
 pc=PCA(2,random_state=42).fit_transform(StandardScaler().fit_transform(d['scores']));pd.DataFrame(pc,columns=['PC1','PC2']).to_csv(OUT/'source_data/SHARED_MEASURED_PROGRAM_PCA.csv',index_label='unit_node_id')
 for j in [1,2]:
  def draw(ax,j=j):
   ax.scatter(*pc.T,s=.4,c='#D7DFE5',alpha=.2,lw=0,rasterized=True)
   for key in CASES:
    n=d['fits'][key][0];ids=n.node_id.to_numpy(int);color=ROOTCOL[key[0]] if j==1 else rankdata(n.pseudotime)/len(n);ax.scatter(*pc[ids].T,s=1.5,c=color,cmap='viridis' if j==2 else None,vmin=0 if j==2 else None,vmax=1 if j==2 else None,alpha=.55,lw=0,rasterized=True)
    for p in d['paths'][key]:ax.plot(*pc[p].T,c=ROOTCOL[key[0]] if j==1 else INK,lw=.55,alpha=.6)
   ax.set_xlabel('Program-profile PC1');ax.set_ylabel('Program-profile PC2');ax.grid(alpha=.12)
   if j==2:ax.figure.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='viridis'),ax=ax,fraction=.03,label='Within-cohort state-order rank')
  assets[f'B{j}']=panel(f'Figure4_B{j}',draw,size=(6.1,4.3))
 for j,(key,programs) in enumerate([(CASES[0],['ECM']),(CASES[1],['Secretory']),(CASES[2],['Luminal','Basal','Cycling','Junction'])],1):
  def draw(ax,key=key,programs=programs):
   n=d['fits'][key][0];ids=n.node_id.to_numpy(int);bins=pd.qcut(pd.Series(n.pseudotime.to_numpy()).rank(method='first'),10,labels=False).to_numpy();rows=[]
   for k,name in enumerate(programs):
    values=d['scores'][name].to_numpy()[ids];med=np.array([np.median(values[bins==i]) for i in range(10)]);lo=np.array([np.quantile(values[bins==i],.25) for i in range(10)]);hi=np.array([np.quantile(values[bins==i],.75) for i in range(10)]);color=ROOTCOL[key[0]] if len(programs)==1 else ['#6A51A3','#178F8D','#E17832','#C9416B'][k];ax.plot(range(1,11),med,c=color,lw=1.15,marker='o',ms=2.8,label=name);ax.fill_between(range(1,11),lo,hi,color=color,alpha=.11);rows.extend({'root_id':key[0],'identity':key[1],'program':name,'order_decile':i+1,'median':med[i],'q25':lo[i],'q75':hi[i]} for i in range(10))
   pd.DataFrame(rows).to_csv(OUT/f'source_data/FIGURE4_DESCRIPTIVE_TRENDS_{key[0]}.csv',index=False);ax.axhline(0,lw=.5,c='#C5CED6');ax.set_xlabel('Within-cohort order decile');ax.set_ylabel('Measured program z score');ax.legend(fontsize=5.7,ncol=2 if len(programs)>1 else 1);ax.grid(alpha=.12)
  assets[f'C{j}']=panel(f'Figure4_C{j}',draw,size=(5.3,3.8))
 for j,key in enumerate(CASES,1):assets[f'D{j}']=panel(f'Figure4_D{j}',lambda ax,k=key:trajectory(ax,d,k),size=(5.3,4.0))
 def robustness(ax):
  a=pd.read_csv(BASE/'technical_thinning/EVERY_FOREST_THREE_TECHNICAL_REPEATS.csv');a=a.loc[a.count_view.eq('WholeCell')]
  for j,key in enumerate(CASES):
   v=a.loc[a.root_id.eq(key[0])&a.identity.eq(key[1]),'order_spearman_on_common_units'];ax.scatter(v,np.repeat(j,len(v)),s=20,c=ROOTCOL[key[0]]);ax.plot([v.min(),v.max()],[j,j],c=ROOTCOL[key[0]],lw=1)
  ax.set_yticks(range(3),['R0 stromal','R1 B cells','R2 epithelial']);ax.set_xlabel('State-order correlation after 80% molecule thinning');ax.set_xlim(.75,1);ax.grid(axis='x',alpha=.12)
 assets['E1']=panel('Figure4_E1',robustness,size=(6,2.9))
 def scope(ax):
  labels=['ECM programme\nR0 stromal','Secretory programme\nR1 B cells','Luminal programme\nR2 epithelial'];names=['STROMAL_ECM_NONIDENTITY','SECRETORY_B','LUMINAL_IDENTITY'];a=[]
  for key,name in zip(CASES,names):
   q=tests.loc[tests.root_id.eq(key[0])&tests.identity.eq(key[1])&tests.panel.eq(name)].set_index('count_view');a.append([q.loc[v,'partial_rho'] for v in ['WholeCell','Nuclear']])
  a=np.array(a);xx=np.arange(3);ax.bar(xx-.16,a[:,0],width=.3,color='#477DB2',label='Whole-cell features');ax.bar(xx+.16,a[:,1],width=.3,color='#9DBDD0',label='Nuclear features');ax.set_xticks(xx,labels,fontsize=5.8);ax.set_ylabel('Held-panel partial correlation');ax.axhline(0,c='#B0BCC6',lw=.6);ax.legend(fontsize=5.5)
 assets['E2']=panel('Figure4_E2',scope,size=(6,2.9))
 rows=[('A','Cell identities and gene-disjoint functional evidence',.982,[('A1','(1) Identity composition across regional forests',(.025,.744,.40,.178)),('A2','(2) Prespecified programme families',(.49,.700,.485,.230))]),('B','Shared biological-program space without root-specific offsets',.670,[('B1','(1) Cohort identity and actual paths',(.025,.455,.46,.170)),('B2','(2) Within-cohort ordered states',(.52,.455,.455,.170))]),('C','Measured programmes along inferred state order',.425,[(f'C{j+1}',s,(.025+j*.323,.255,.305,.120)) for j,s in enumerate(['(1) Stromal ECM state','(2) B-cell secretory state','(3) Epithelial marker states'])]),('D','Technical reproducibility and evidence strength',.220,[('E1','(1) Three technical molecule perturbations',(.025,.045,.46,.130)),('E2','(2) Whole-cell and nuclear feature views',(.52,.045,.455,.130))])]
 return plate(4,assets,rows,'Dots in A2: complete-family evidence gate passed, not independent pathways. C: descriptive medians and IQR, not biological replicates.\nR2 epithelial direction uses the declared clinical-stage prior; its functional interpretation is weaker than the R0/R1 examples. No cross-lineage conversion is inferred.')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--only',choices=['1','2','4','all'],default='all');args=ap.parse_args()
 for sub in ['panels','source_data','_build']: (OUT/sub).mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
 contract={'conclusion':'Source-region-conditioned within-identity state forests and their measured programme associations.','archetype':'asymmetric mixed-modality figure','data':'One verified original breast count/molecule source and one current count-spatial-agreement reconstruction; no old P6 relation/enrichment outputs.','root_contract':'Original region anchor CSV and exact raw source-cell roster unchanged. Spatial anchors and source-state representatives are different objects.','geometry':'Actual observed member representatives and parent edges; no interpolation, jitter, root offset or length-selected curve. Dashed state edges exceed empirical local physical scale.','display_paths':'At most3 actual source-to-leaf paths selected by uncovered raw-cell mass, then state order and ID. No physical length, shape, marker or embedding separation in selection.','background':'Same-identity observed member cells in displayed states and physically supported one-hop neighbours; no coloured corridor across distant nonlocal edges.','marker_colours':'Unit-mean measured marker z scores only, display labels never used as new identity or functional test. ECM/secretory above cohort mean; epithelial dominant positive measured programme, otherwise unresolved.','native_comparison':'Fresh same-input supplied-start state-unit runs; qualitative projections, not automatic root recovery or biological accuracy proof.','statistics':'Complete2448held-panel tests, all qualified and unqualified rows retained.60reused development simulations,3technical molecule perturbations, not independent patients.','export':'183mm-width figures, common Arial, upper-case panel letters and numbered subpanels, PDF/SVG/400dpiPNG, separate real panels and editable title manifests.','review_risks':['State association is not physical migration or measured ancestry.','Clinical stage and published identity are inputs.','R2 epithelial functional evidence is not as robust as R0/R1.','Old Fig3 benchmark numbers are not evidence for this new constructor.']};(OUT/'_build/FIGURE_CONTRACT.json').write_text(json.dumps(contract,indent=2));d=load();manifest=[]
 for number,fn in [(1,figure1),(2,figure2),(4,figure4)]:
  if args.only in [str(number),'all']:manifest.append(fn(d));print('Figure',number,'rendered from current source',flush=True)
 contract.update({'geometry':'Actual observed-member representatives and solid actual parent edges, arrows on every nonzero segment. No interpolation, jitter, root offset or length-selected curve. Dashed links occur only for explicitly hypothetical ligand-receptor contacts.','display_paths':'At most3 actual source-to-leaf paths selected by uncovered raw-cell mass, then state order and ID. R2 epithelial paths additionally cover available original DCIS/invasive/proliferating-invasive endpoint strata using the same mass rule within strata. No physical length, shape or embedding-separation target.','background':'Observed cells within150um of actual displayed segments. Coloured context is the corresponding supplied identity with nuclear marker support; grey is other measured local context. This is not an ancestor assignment. The same rule applies to native comparators.','marker_colours':'Measured raw-cell marker z scores standardized within supplied identity, then exact unit means. ECM/secretory positive relative to identity mean; epithelial dominant positive programme, otherwise unresolved. Display scores are descriptive and distinct from held-panel tests.','function_matrix':'Main heatmap shows16 declared programme families. Full34-programme by all current cohort/view matrix and complete2448-test results remain available.'})
 (OUT/'_build/FIGURE_CONTRACT.json').write_text(json.dumps(contract,indent=2))
 (OUT/'_build/NEW_FIGURES.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
