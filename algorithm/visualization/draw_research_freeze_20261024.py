"""Data-driven current figures: real-cell geometry and explicit source semantics.

Display paths are selected from actual parent edges; they do not define the
model. Marker smoothing is a display-only field within known cell identity.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('NICHEFOREST_RELEASE',str(ROOT/'outputs/NicheForest_current_research_freeze_20261024')))
INK='#202B36';ROOTCOL={'R0':'#C53D56','R1':'#387FB5','R2':'#27916D'}
CASES=[('R0','Stromal'),('R1','B_cell'),('R2','Malignant_epithelial')]
LABELS={
 'Stromal-low':'Stromal cells: lower remodelling',
 'Stromal-high':'Stromal cells: remodelling-high',
 'B-naive':'B cells: naive-associated',
 'B-secretory':'B cells: secretory-associated',
 'B-mixed':'B cells: mixed marker support',
 'Epi-luminal':'Tumor epithelial: luminal-associated',
 'Epi-basal':'Tumor epithelial: basal-associated',
 'Epi-cycling':'Tumor epithelial: cycling',
 'Epi-CLDN4':'Tumor epithelial: CLDN4/CEACAM6-rich',
 'Epi-mixed':'Tumor epithelial: mixed marker support'}
COLORS={'Stromal-low':'#8C9E89','Stromal-high':'#D39A55','B-naive':'#477DB2','B-secretory':'#8D72B3','B-mixed':'#B8C0CA',
 'Epi-luminal':'#6A51A3','Epi-basal':'#178F8D','Epi-cycling':'#E17832','Epi-CLDN4':'#C9416B','Epi-mixed':'#B8C0CA'}
MARKERS={
 'remodelling':['LUM','POSTN','PCOLCE','FBLN1','MMP2','MMP1'],
 'B-naive':['TCL1A','BANK1','MS4A1','CD19','CD79B'],
 'B-secretory':['MZB1','PRDM1','TNFRSF17','DERL3','SLAMF7','TENT5C'],
 'Epi-luminal':['FOXA1','GATA3','MLPH','AGR3','KRT8'],
 'Epi-basal':['KRT5','KRT14','KRT15','KRT6B'],
 'Epi-cycling':['MKI67','TOP2A','CENPF','PCLAF'],
 'Epi-CLDN4':['CLDN4','CEACAM6','KLF5']}

def setup():
 plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.titlesize':8,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})

def save(fig,stem):
 for ext in ['png','pdf','svg']:fig.savefig(OUT/(stem+'.'+ext),dpi=350,facecolor='white')

def clean(ax,xy,spatial=True,pad=.06):
 ax.set_xticks([]);ax.set_yticks([])
 for spine in ax.spines.values():spine.set_visible(False)
 ax.set_aspect('equal',adjustable='box')
 if len(xy):
  lo=np.nanmin(xy,axis=0);hi=np.nanmax(xy,axis=0);span=np.maximum(hi-lo,1)
  ax.set_xlim(lo[0]-pad*span[0],hi[0]+pad*span[0]);ax.set_ylim(lo[1]-pad*span[1],hi[1]+pad*span[1])
 if spatial:ax.invert_yaxis()

def load():
 d={};d['raw']=ad.read_h5ad(OUT/'data/BREAST_ORIGINAL_COUNTS.h5ad')
 d['cells']=pd.read_csv(OUT/'data/PUBLISHED_CELL_METADATA.csv',dtype={'cell_name':str})
 d['obs']=pd.read_csv(OUT/'data/MODEL_CELL_INPUT.csv',dtype={'cell_name':str}).set_index('node_id')
 d['nodes']=pd.read_csv(OUT/'forest/FINAL_NODE_ASSIGNMENTS.csv',dtype={'cell_name':str})
 d['edges']=pd.read_csv(OUT/'forest/FINAL_DIRECTED_EDGES.csv')
 d['base']=pd.read_csv(OUT/'forest/MATCHED_STAGE_ADMISSIBLE_T0_SPT_EDGES.csv')
 local=OUT/'forest/spatial_realization_soft'
 if (local/'LOCAL_ROUTE_EDGES.csv').exists():
  d['local_nodes']=pd.read_csv(local/'LOCAL_ROUTE_NODES.csv',dtype={'cell_name':str})
  d['local_edges']=pd.read_csv(local/'LOCAL_ROUTE_EDGES.csv')
 d['roots']=pd.read_csv(OUT/'data/CANONICAL_SPATIAL_ANCHORS.csv').set_index('root_id')
 umapfile=OUT/'data/RAW_CELL_UMAP_COORDINATES.csv'
 embedding=pd.read_csv(umapfile,dtype={'cell_name':str}).set_index('cell_name')
 assert np.array_equal(embedding.index,d['cells'].cell_name)
 d['spatial']=d['cells'][['spatial_x','spatial_y']].to_numpy()
 d['umap']=embedding[['umap_1','umap_2']].to_numpy();assert np.isfinite(d['umap']).all()
 d['model_xy']=d['obs'][['spatial_x','spatial_y']].to_numpy()
 d['model_umap']=d['umap'][d['obs'].raw_row.to_numpy(int)]
 return d

def marker_classes(d):
 path=OUT/'data/DISPLAY_MARKER_CLASSES.csv'
 if path.exists():
  frame=pd.read_csv(path,dtype={'cell_name':str});assert np.array_equal(frame.cell_name,d['obs'].cell_name);return frame
 classes=np.repeat('Other',len(d['obs'])).astype(object);scores={}
 counts=sparse.csr_matrix(d['raw'].X)[d['obs'].raw_row.to_numpy(int)]
 for identity in ['Stromal','B_cell','Malignant_epithelial']:
  ids=np.flatnonzero(d['obs'].identity.to_numpy()==identity);xy=d['model_xy'][ids]
  names=['remodelling'] if identity=='Stromal' else ['B-naive','B-secretory'] if identity=='B_cell' else ['Epi-luminal','Epi-basal','Epi-cycling','Epi-CLDN4']
  dist,nn=cKDTree(xy).query(xy,k=min(11,len(ids)));weight=np.exp(-.5*(dist/30)**2)*(dist<=75)
  weight/=np.maximum(weight.sum(1,keepdims=True),1e-12)
  means=[]
  for name in names:
   genes=[d['raw'].var_names.get_loc(g) for g in MARKERS[name] if g in d['raw'].var_names]
   train=np.setdiff1d(np.arange(d['raw'].n_vars),genes);depth=np.maximum(np.asarray(counts[ids][:,train].sum(1)).ravel(),1)
   zz=np.log1p(counts[ids][:,genes].toarray()/depth[:,None]*np.median(depth));z=StandardScaler().fit_transform(zz).mean(1)
   smooth=(z[nn]*weight).sum(1);means.append(smooth);full=np.full(len(classes),np.nan);full[ids]=smooth;scores[name]=full
  values=np.column_stack(means)
  if identity=='Stromal':classes[ids]=np.where(values[:,0]>=np.quantile(values[:,0],.75),'Stromal-high','Stromal-low')
  elif identity=='B_cell':
   delta=values[:,1]-values[:,0];classes[ids]=np.where(delta>.3,'B-secretory',np.where(delta<-.3,'B-naive','B-mixed'))
  else:
   best=np.argmax(values,axis=1);top=np.sort(values,axis=1);clear=(top[:,-1]>.15)&((top[:,-1]-top[:,-2])>.15)
   classes[ids]=np.where(clear,np.asarray(names)[best],'Epi-mixed')
 frame=pd.DataFrame({'node_id':np.arange(len(classes)),'cell_name':d['obs'].cell_name.to_numpy(),'identity':d['obs'].identity.to_numpy(),'display_class':classes,**scores})
 frame.to_csv(path,index=False)
 (OUT/'data/DISPLAY_CLASSIFICATION_PROTOCOL.json').write_text(json.dumps({
  'identity':'Original-study annotation, fixed; a functional panel never changes a cell into another identity.',
  'markers':MARKERS,'display_only_smoothing':'At most11 nearest same-identity actual cells within75um, Gaussian30um, mean program z-score; used for display categories only, not inference, testing or cell coordinates.',
  'Stromal':'Top quartile local remodelling score versus lower/intermediate; relative descriptive state, not a newly proven fibroblast subtype.',
  'B':'Secretory-minus-naive local score >0.3 or <-0.3, else mixed.',
  'Epithelial':'Only four relevant epithelial programs compete, require best local z>.15 and margin>.15, else mixed. DCIS/invasive clinical annotation remains a separate input, not inferred from these colors.',
  'no_jitter':True,'all_statistics':'Computed from unsmoothed raw counts with whole-program exclusion, never from this display smoothing.'},indent=2))
 return frame

def choose_paths(d,rid,identity,max_paths=3,baseline=False,spatial_local=False):
 nsource=d['local_nodes'] if spatial_local else d['nodes']
 nodes=nsource.loc[nsource.root_id.eq(rid)&nsource.identity.eq(identity)].set_index('node_id')
 edges=d['base'] if baseline else d['local_edges'] if spatial_local else d['edges']
 edges=edges.loc[edges.root_id.eq(rid)&edges.identity.eq(identity)]
 parent=dict(zip(edges.target.astype(int),edges.source.astype(int)));children=set(edges.source.astype(int))
 leaves=[i for i in nodes.index if i not in children and i in parent]
 tau=nodes.pseudotime.to_dict();leaf=sorted(leaves,key=lambda i:(-tau[i],i))[:max(200,int(.1*len(leaves)))]
 chosen=[];routes=[];used=set()
 for _ in range(max_paths):
  best=None;bestkey=None;bestroute=None
  for tip in leaf:
   if tip in chosen:continue
   route=[tip]
   while route[-1] in parent:route.append(parent[route[-1]])
   route=route[::-1];length=len(set(route)-used)
   separation=min(np.linalg.norm(d['model_xy'][tip]-d['model_xy'][p]) for p in chosen) if chosen else 1
   key=(length+min(separation/150,8),tau[tip],-tip)
   if bestkey is None or key>bestkey:best,bestkey,bestroute=tip,key,route
  if best is None:break
  chosen.append(best);routes.append(bestroute);used.update(bestroute)
 return routes

def segments(routes):return np.array([(u,v) for path in routes for u,v in zip(path[:-1],path[1:])],int).reshape(-1,2)

def corridor_mask(xy,edge_xy,radius=200):
 if not len(edge_xy):return np.zeros(len(xy),bool)
 samples=np.concatenate([np.linspace(a,b,max(2,int(np.linalg.norm(a-b)/35))) for a,b in edge_xy])
 return cKDTree(samples).query(xy)[0]<=radius

def draw_segments(ax,xy,ed,color=INK,lw=.9,arrows=True):
 if not len(ed):return
 seg=xy[ed];ax.add_collection(LineCollection(seg,colors='white',linewidths=lw+1.8,zorder=4))
 ax.add_collection(LineCollection(seg,colors=color,linewidths=lw,zorder=5))
 if arrows:
  for a,b in seg:
   if np.linalg.norm(a-b)<1e-10:continue
   ax.add_patch(FancyArrowPatch(a+.40*(b-a),a+.68*(b-a),arrowstyle='-|>',mutation_scale=5.3,lw=.6,color=color,zorder=6))

def draw_local(ax,d,classes,rid,identity,routes=None,legend=True):
 if routes is None:routes=choose_paths(d,rid,identity,spatial_local=True)
 ed=segments(routes);coords=d['model_xy'];pathcoords=coords[ed]
 eligible=np.flatnonzero(d['obs'].identity.to_numpy()==identity)
 keep=eligible[corridor_mask(coords[eligible],pathcoords,200)]
 context=np.flatnonzero(corridor_mask(d['spatial'],pathcoords,200))
 ax.scatter(d['spatial'][context,0],d['spatial'][context,1],s=.35,color='#E1E5E8',alpha=.32,lw=0,rasterized=True,zorder=1)
 for label in COLORS:
  selected=keep[classes.display_class.to_numpy()[keep]==label]
  if len(selected):ax.scatter(coords[selected,0],coords[selected,1],s=1.9,color=COLORS[label],alpha=.82,lw=0,rasterized=True,zorder=2)
 draw_segments(ax,coords,ed)
 root=d['roots'].loc[rid,['x','y']].to_numpy(float)
 ax.scatter(*root,marker='*',s=70,color=INK,edgecolor='white',linewidth=.6,zorder=10)
 for path in routes:
  ax.scatter(*coords[path[-1]],s=13,facecolor='white',edgecolor=INK,lw=.7,zorder=9)
  # This is a coordinate-reference link to a source cell, not a biological edge.
  ax.plot([root[0],coords[path[0],0]],[root[1],coords[path[0],1]],color='#9AA5AE',lw=.5,ls=':',zorder=3)
 clean(ax,np.vstack([coords[keep],root[None,:]]))
 if legend:
  present=[c for c in COLORS if c in set(classes.display_class.iloc[keep])]
  handles=[Line2D([],[],marker='o',ls='',ms=3,color=COLORS[c],label=LABELS[c]) for c in present]
  ax.legend(handles=handles,loc='upper right',fontsize=5.1,frameon=True,facecolor='white',edgecolor='#E2E6EB',framealpha=.96,borderpad=.4,handletextpad=.3)
 return ed,keep

def native_edges(rid,method,seed=202610440):
 prefix=OUT/'native_inputs'
 if not (prefix/str(seed)).exists():prefix=ROOT/'outputs/Native_regional_comparison_20261024'
 folder=prefix/'R0_stromal_secondary'/str(seed)/rid if rid=='R0' else prefix/str(seed)/rid
 return pd.read_csv(folder/f'{method}_PATH_EDGES.csv'),folder

def draw_native(ax,d,classes,method):
 allids=[];all_edges=[];observed=set()
 for rid,identity in CASES:
  edges,folder=native_edges(rid,method);ed=edges[['source','target']].to_numpy(int);all_edges.extend(ed)
  eligible=np.flatnonzero(d['obs'].identity.to_numpy()==identity)
  if len(ed):keep=eligible[corridor_mask(d['model_xy'][eligible],d['model_xy'][ed],200)]
  else:
   root=d['roots'].loc[rid,['x','y']].to_numpy(float);keep=eligible[np.linalg.norm(d['model_xy'][eligible]-root,axis=1)<200]
  allids.extend(keep)
  for label in COLORS:
   sel=keep[classes.display_class.to_numpy()[keep]==label]
   if len(sel):
    observed.add(label);ax.scatter(d['model_xy'][sel,0],d['model_xy'][sel,1],s=.55,c=COLORS[label],alpha=.64,lw=0,rasterized=True)
  draw_segments(ax,d['model_xy'],ed,lw=.7)
  root=d['roots'].loc[rid,['x','y']].to_numpy(float);ax.scatter(*root,marker='*',s=43,c=INK,edgecolor='white',lw=.4,zorder=10)
  ax.annotate(rid,root,xytext=(3,5),textcoords='offset points',fontsize=6,weight='bold',color=INK)
 clean(ax,d['model_xy'][allids])
 handles=[Line2D([],[],marker='o',ls='',ms=2.6,color=COLORS[c],label=LABELS[c]) for c in COLORS if c in observed]
 ax.legend(handles=handles,loc='upper right',fontsize=4.6,frameon=True,facecolor='white',edgecolor='#E2E6EB',framealpha=.96,borderpad=.25,labelspacing=.2,handletextpad=.25)

def overview(ax,d,space='spatial'):
 xy=d[space];model=d['model_xy'] if space=='spatial' else d['model_umap']
 ax.scatter(xy[::3,0],xy[::3,1],s=.22,color='#DDE2E6',alpha=.5,lw=0,rasterized=True)
 for rid,identity in CASES:
  nodes=d['nodes'].loc[d['nodes'].root_id.eq(rid)&d['nodes'].identity.eq(identity)];ids=nodes.node_id.to_numpy(int)
  ax.scatter(model[ids[::2],0],model[ids[::2],1],s=.30,c=ROOTCOL[rid],alpha=.50,lw=0,rasterized=True)
  if space=='spatial':
   routes=choose_paths(d,rid,identity,spatial_local=True);draw_segments(ax,model,segments(routes),ROOTCOL[rid],.9)
  else:
   # Expression-space curves summarize actual-cell order bins; they are not
   # literal edges or displaced cells. All underlying cells remain unchanged.
   order=nodes.pseudotime.to_numpy();binid=pd.qcut(pd.Series(order).rank(method='first'),18,labels=False).to_numpy()
   centers=np.array([np.median(model[ids[binid==b]],axis=0) for b in range(18)])
   from scipy.ndimage import gaussian_filter1d
   centers=gaussian_filter1d(centers,1,axis=0,mode='nearest')
   ax.plot(centers[:,0],centers[:,1],c=ROOTCOL[rid],lw=1.5,zorder=5)
   for b in [3,8,13]:ax.add_patch(FancyArrowPatch(centers[b],centers[b+1],arrowstyle='-|>',mutation_scale=6,lw=1,color=ROOTCOL[rid],zorder=6))
   pd.DataFrame({'root_id':rid,'identity':identity,'bin':np.arange(18),'umap_1':centers[:,0],'umap_2':centers[:,1]}).to_csv(OUT/'data'/f'EXPRESSION_ORDER_SUMMARY_{rid}.csv',index=False)
  if space=='spatial':
   root=d['roots'].loc[rid,['x','y']].to_numpy(float);ax.scatter(*root,marker='*',s=58,c=ROOTCOL[rid],edgecolor='white',lw=.5,zorder=10)
  else:
   # A spatial virtual point cannot be placed in expression space.
   seeds=nodes.loc[nodes.is_source_cell.astype(str).str.lower().eq('true'),'node_id'].to_numpy(int)
   ax.scatter(model[seeds,0],model[seeds,1],s=6,facecolors='none',edgecolor=ROOTCOL[rid],lw=.5,zorder=7)
 clean(ax,xy,spatial=space=='spatial')

def panel_image(stem,func,size=(4.6,3.8)):
 fig,ax=plt.subplots(figsize=size);func(ax);fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98)
 path=OUT/'panels'/(stem+'.png');fig.savefig(path,dpi=400,facecolor='white',bbox_inches='tight',pad_inches=.05);fig.savefig(path.with_suffix('.svg'),bbox_inches='tight',pad_inches=.05);fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.05);plt.close(fig)
 return str(path)

def figure1(d,classes):
 assets={};cells=d['cells'];cluster=cells.fixed_graph_partition;names=sorted(cluster.unique(),key=lambda k:int(k[1:]))
 palette_file=OUT/'data/FIXED_PARTITION_PALETTE.json'
 if palette_file.exists():palette=json.loads(palette_file.read_text())
 else:
  import importlib.util
  spec=importlib.util.spec_from_file_location('approved_palette',ROOT/'algorithm/可视化/make_v7_4_6_current_clusters_legacy_three_panel.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  palette=dict(zip(names,module.get_cluster_colors(len(names))));palette_file.write_text(json.dumps(palette,indent=2))
 ranks=cluster.value_counts().head(10).index.tolist();top=list(dict.fromkeys(ranks+['C13','C00','C21']))
 for index,space in enumerate(['spatial','umap'],1):
  def plot(ax,space=space):
   xy=d[space]
   for name in names:
    ids=np.flatnonzero(cluster.to_numpy()==name)[::2];ax.scatter(xy[ids,0],xy[ids,1],s=.28,c=[palette[name]],alpha=.72,lw=0,rasterized=True)
   for name in top:
    ids=np.flatnonzero(cluster.to_numpy()==name);p=np.median(xy[ids],axis=0)
    ax.text(*p,name,fontsize=5.2,weight='bold',ha='center',bbox=dict(facecolor='white',alpha=.85,edgecolor='#D4DCE1',lw=.3,pad=.5))
   clean(ax,xy,spatial=space=='spatial')
  assets[f'A{index}']=panel_image(f'Figure1_A{index}',plot)
 major=cells.identity.fillna('Other/unresolved').replace({'CD4_T':'Lymphoid','CD8_T':'Lymphoid','B_cell':'Lymphoid','Macrophage':'Myeloid','Dendritic':'Myeloid','Mast':'Myeloid'})
 comp=pd.crosstab(cluster,major,normalize='index').T
 dominant=comp.idxmax(0);order=sorted(names,key=lambda k:(dominant[k],-float(comp[k].max()),k));comp=comp[order]
 comp.to_csv(OUT/'evidence/FIG1_CLUSTER_COMPOSITION.csv')
 fig=plt.figure(figsize=(4.8,3.8));ax=fig.add_axes([.27,.26,.63,.57]);image=ax.imshow(comp,aspect='auto',cmap='Blues',vmin=0,vmax=1)
 ax.set_yticks(range(len(comp)),comp.index,fontsize=6);ax.set_xticks(range(len(order)),order,rotation=90,fontsize=3.4);ax.tick_params(length=0)
 bar=fig.add_axes([.92,.26,.02,.57]);fig.colorbar(image,cax=bar,label='Cell fraction')
 family_palette=dict(zip(comp.index,['#4B9DA8','#4D68A8','#C55068','#8C70AE','#36958A','#BDC4CC','#779550','#CA8A48']))
 from matplotlib.colors import to_rgba
 strip=fig.add_axes([.27,.855,.63,.055]);strip.imshow(np.array([[to_rgba(family_palette[dominant[k]]) for k in order]]),aspect='auto');strip.axis('off')
 purity=fig.add_axes([.27,.085,.63,.10]);purity.bar(np.arange(len(order)),comp.max(0),color=[family_palette[dominant[k]] for k in order],width=.9,lw=0)
 purity.set_xlim(-.5,len(order)-.5);purity.set_ylim(0,1);purity.set_xticks([]);purity.set_yticks([0,1],['0','1'],fontsize=4.5);purity.set_ylabel('Purity',fontsize=5)
 assets['A3']=str(OUT/'panels/Figure1_A3.png');fig.savefig(assets['A3'],dpi=400);fig.savefig(OUT/'panels/Figure1_A3.pdf');plt.close(fig)
 for i,space in enumerate(['spatial','umap'],1):assets[f'B{i}']=panel_image(f'Figure1_B{i}',lambda ax,s=space:overview(ax,d,s),size=(7,4.2))
 source_paths=[]
 for i,(rid,identity) in enumerate(CASES,1):
  routes=choose_paths(d,rid,identity,spatial_local=True)
  assets[f'C{i}']=panel_image(f'Figure1_C{i}',lambda ax,r=rid,t=identity,paths=routes:draw_local(ax,d,classes,r,t,paths))
  for pathid,path in enumerate(routes):source_paths.extend([{'root_id':rid,'identity':identity,'path_id':pathid,'order':j,'node_id':int(n),'cell_name':str(d['obs'].cell_name.iloc[n])} for j,n in enumerate(path)])
 pd.DataFrame(source_paths).to_csv(OUT/'data/FIG1_REPRESENTATIVE_ACTUAL_PATHS.csv',index=False)
 for i,method in enumerate(['CellRank_DPT','PAGA_DPT','Palantir'],1):assets[f'D{i}']=panel_image(f'Figure1_D{i}',lambda ax,m=method:draw_native(ax,d,classes,m))
 # Flat aligned plate, with no global figure title.
 fig=plt.figure(figsize=(8.2,11.3));elements=[]
 def heading(y,text):fig.text(.025,y,text,fontsize=12,weight='bold',va='top',color=INK);elements.append({'kind':'text','text':text,'x':40,'y':(1-y)*2210,'w':1520,'h':50,'size':26,'bold':True,'name':text})
 def image(key,rect,title):
  ax=fig.add_axes(rect);ax.imshow(plt.imread(assets[key]));ax.axis('off');fig.text(rect[0]+.003,rect[1]+rect[3]+.008,title,fontsize=7.8,weight='bold',color=INK,va='bottom')
  elements.append({'kind':'image','key':key,'name':key,'x':rect[0]*1600,'y':(1-rect[1]-rect[3])*2210,'w':rect[2]*1600,'h':rect[3]*2210})
  elements.append({'kind':'text','name':key+' title','text':title,'x':(rect[0]+.003)*1600,'y':(1-rect[1]-rect[3]-.011)*2210,'w':rect[2]*1600,'h':30,'size':17,'bold':True})
 heading(.982,'A  Fixed graph partitions and original-study cell identities')
 for j,title in enumerate(['(1) Tissue graph partitions','(2) Actual-cell expression UMAP','(3) Published identity composition']):image('A'+str(j+1),(.025+j*.323,.760,.305,.177),title)
 heading(.724,'B  Cell-state order and its tissue-local realization')
 image('B1',(.025,.480,.462,.206),'(1) Source-reachable local tissue routes');image('B2',(.515,.480,.46,.206),'(2) Expression-space order summaries')
 heading(.452,'C  Tissue-local routes and identity-specific marker states')
 titles=['(1) C13: resident stromal cells','(2) C00: B cells','(3) C21: malignant epithelial cells']
 for j,title in enumerate(titles):image('C'+str(j+1),(.025+j*.323,.255,.305,.162),title)
 heading(.226,'D  Native supplied-cell trajectories in original tissue coordinates')
 for j,title in enumerate(['(1) CellRank (DPT kernel)','(2) PAGA+DPT','(3) Palantir']):image('D'+str(j+1),(.025+j*.323,.035,.305,.157),title)
 foot='Stars: fixed regional anchors. Circles: selected route endpoints. B2 summarizes expression-state order; C shows its partial local tissue realization.'
 fig.text(.027,.012,foot,fontsize=5.5,color='#657581')
 save(fig,'Figure1_current');plt.close(fig)
 manifest={'figure':1,'assets':{k:{'path':v} for k,v in assets.items()},'elements':elements}
 (OUT/'_build/figure1_manifest.json').write_text(json.dumps(manifest,indent=2))

def main():
 setup();d=load();classes=marker_classes(d);figure1(d,classes)
 print('Current Figure1 generated from canonical real-cell result',flush=True)

if __name__=='__main__':main()
