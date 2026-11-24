"""Internal actual-edge visual inspection, no fabricated trajectory geometry."""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.path.insert(0,str(ROOT/'algorithm/visualization'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
from draw_research_freeze_20261024 import corridor_mask,draw_segments,clean,ROOTCOL,INK
BASE=ROOT/'outputs/Regional_source_ensemble_states_v63_20261025/ensemble'
OUT=ROOT/'outputs/Current_state_visual_QA_20261025'
if os.environ.get('NICHEFOREST_COUNT_SPATIAL_AGREEMENT')=='1':
 BASE=ROOT/'outputs/Count_spatial_agreement_states_v70_20261025/ensemble'
 OUT=ROOT/'outputs/Count_spatial_agreement_visual_QA_20261025'

def representative_paths(nodes,edges,maximum=3):
 n=nodes.set_index('node_id');parent=dict(zip(edges.target.astype(int),edges.source.astype(int)));children={int(i):[] for i in n.index}
 for v,u in parent.items():children[u].append(v)
 source=n.index[n.is_source_state].to_numpy(int);mass=n.n_raw_cells.to_dict();submass=dict(mass)
 for v in n.sort_values('pseudotime',ascending=False).index:
  if v in parent:submass[parent[v]]+=submass[v]
 leaves=[v for v in n.index if not children[v] and v in parent];routes={}
 for v in leaves:
  route=[v]
  while route[-1] in parent:route.append(parent[route[-1]])
  routes[v]=route[::-1]
 # A fixed display summary, independent of coordinates, desired length,
 # marker color or source-label separation. No alteration of the model.
 used=set();selected=[]
 for j in range(maximum):
  if not routes:break
  tip=max(routes,key=lambda v:(sum(mass[u] for u in routes[v] if u not in used),n.loc[v,'pseudotime'],-v));path=routes.pop(tip);selected.append(path);used.update(path)
 return selected

def main():
 OUT.mkdir(parents=True,exist_ok=True);units=pd.read_csv(BASE/'EXACT_STATE_UNITS.csv');xy=units[['spatial_x','spatial_y']].to_numpy();root=pd.read_csv(ROOT/'outputs/NicheForest_current_research_freeze_20261024/data/CANONICAL_SPATIAL_ANCHORS.csv').set_index('root_id');cells=pd.read_csv(ROOT/'outputs/NicheForest_current_research_freeze_20261024/data/PUBLISHED_CELL_METADATA.csv');rawxy=cells[['spatial_x','spatial_y']].to_numpy();folders=sorted((BASE/'WholeCell').glob('*/FULL_STATE_NODES.csv'));fig,axes=plt.subplots(6,3,figsize=(15,23));records=[];paths=[]
 for ax,file in zip(axes.flat,folders):
  rid,identity=file.parent.name.split('_',1);n=pd.read_csv(file);e=pd.read_csv(file.parent/'FULL_STATE_EDGES.csv');routes=representative_paths(n,e);ed=np.array([(u,v) for p in routes for u,v in zip(p[:-1],p[1:])],int);keep=corridor_mask(rawxy,xy[ed],150);ax.scatter(*rawxy[keep].T,s=.2,c='#dce2e7',alpha=.3,lw=0,rasterized=True);own=keep&cells.identity.eq(identity).to_numpy();ax.scatter(*rawxy[own].T,s=.5,c=ROOTCOL[rid],alpha=.5,lw=0,rasterized=True);draw_segments(ax,xy,ed,lw=.7);point=root.loc[rid,['x','y']].to_numpy(float);ax.scatter(*point,marker='*',s=50,c=ROOTCOL[rid],edgecolor='white',zorder=10);clean(ax,rawxy);ax.set_title(rid+' '+identity,loc='left',fontsize=10)
  coords=xy[np.unique(ed)];span=np.ptp(coords,axis=0)/np.ptp(rawxy,axis=0);length=np.linalg.norm(xy[ed[:,1]]-xy[ed[:,0]],axis=1);records.append({'root_id':rid,'identity':identity,'n_actual_edges_displayed':len(set(map(tuple,ed))),'tissue_x_span_fraction':float(span[0]),'tissue_y_span_fraction':float(span[1]),'median_actual_edge_um':float(np.median(length)),'max_actual_edge_um':float(length.max()),'fraction_all_raw_cells_in_context':float(keep.mean())})
  paths.extend({'root_id':rid,'identity':identity,'path_id':j,'order':k,'unit_node_id':int(v)} for j,p in enumerate(routes) for k,v in enumerate(p))
 plt.tight_layout();fig.savefig(OUT/'INTERNAL_ALL_18_ACTUAL_PATHS.png',dpi=160);plt.close(fig);pd.DataFrame(records).to_csv(OUT/'EVERY_FOREST_DISPLAY_GEOMETRY.csv',index=False);pd.DataFrame(paths).to_csv(OUT/'EXACT_REPRESENTATIVE_PATHS.csv',index=False);print(pd.DataFrame(records).to_string(index=False))

if __name__=='__main__':main()
