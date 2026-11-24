"""Compare complete frozen paths with an explicitly new spatial constructor.

No movement of roots/cells, no reuse of old program p-values, no overwrite of
the frozen result. Local candidates precede inference, rather than filtering
an already computed molecular order. Clinical-stage prior remains an input.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components,dijkstra
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'outputs/NicheForest_current_research_freeze_20261024'
OUT=ROOT/'outputs/Trajectory_continuity_revision_20261024'
sys.path.insert(0,str(ROOT/'algorithm/experimental'))
from cell_resolved_multipath_v14_20261023 import constructors
from stage_admissible_epithelial_v16_20261023 import stage_fit
CASES=[('R0','Stromal'),('R1','B_cell'),('R2','Malignant_epithelial')]
PROTOCOL={
 'conclusion':'Evaluate whether tissue continuity improves when physical adjacency is supplied before, rather than after, estimating state order.',
 'scope':'Development comparison, not adopted or a replacement of the frozen molecular result. Current three illustration cohorts were selected previously, not a new independent accuracy benchmark.',
 'protected':'Exact frozen nodes/cohort, exact frozen source-cell IDs and spatial anchors, original counts and identities. Only source-reachable support in the new graph is retained. All abstentions reported.',
 'geometry':'Within full published identity NB Pearson residual theta100 and PCA15. Local symmetric union30 physical-nearest-neighbor candidates within150um, standardized expression distance<=2.5. Cost=expression_distance/sqrt(expression30NN_scales)+0.5*physical_length/150.',
 'ordering':'Same alpha.97 discounted hitting solver and minimum-cost higher-potential parent selection. Clinical admissibility prior for epithelial graph only. Not constrained to frozen global tau.',
 'sensitivity':'Fixed radius100/200um and physical k15/50, same source sets. No radius selection on longest/most attractive paths.',
 'validation':'Three molecule80% thinnings for ordering/support stability and matchedSPT controls. Whole-program-exclusion descriptive/partial tests for declared illustration panels; no reuse of frozen q values or claims.',
 'display':'Full frozen paths retain actual global edges, long ones dashed as state associations. Spatial candidate paths retain actual local edges. All nodes at measured coordinates. No fake branch, curve shortcut or cross-type conversion.',
 'publication_ready':False}

def load():
 raw=ad.read_h5ad(BASE/'data/BREAST_ORIGINAL_COUNTS.h5ad');obs=pd.read_csv(BASE/'data/MODEL_CELL_INPUT.csv',dtype={'cell_name':str}).set_index('node_id')
 nodes=pd.read_csv(BASE/'forest/FINAL_NODE_ASSIGNMENTS.csv',dtype={'cell_name':str});edges=pd.read_csv(BASE/'forest/FINAL_DIRECTED_EDGES.csv')
 return raw,obs,nodes,edges

def geometry(counts,xy,genes=None,radius=150,knn=30):
 if genes is None:genes=np.arange(counts.shape[1])
 x=counts[:,genes].toarray().astype(float) if sparse.issparse(counts) else np.asarray(counts[:,genes],float)
 depth=np.maximum(x.sum(1),1);freq=x.sum(0)/max(x.sum(),1);mu=depth[:,None]*freq[None,:]
 rr=np.clip((x-mu)/np.sqrt(np.maximum(mu+mu**2/100,1e-8)),-np.sqrt(len(x)),np.sqrt(len(x)))
 pc=PCA(min(15,len(x)-1,len(genes)),random_state=42).fit_transform(rr)
 distances,neighbors=NearestNeighbors(n_neighbors=min(31,len(pc)),n_jobs=2).fit(pc).kneighbors(pc)
 scales=np.maximum(distances[:,-1],1e-6)
 d,nn=cKDTree(xy).query(xy,k=min(knn+1,len(xy)),distance_upper_bound=radius)
 u=np.repeat(np.arange(len(xy)),nn.shape[1]);v=nn.ravel();physical=d.ravel()
 good=(v<len(xy))&(u!=v);u=u[good];v=v[good];physical=physical[good]
 ex=np.linalg.norm(pc[u]-pc[v],axis=1)/np.sqrt(scales[u]*scales[v]);good=ex<=2.5
 u=u[good];v=v[good];cost=ex[good]+.5*physical[good]/150
 # Maximum combines reciprocal entries with identical symmetric cost, never
 # sums duplicates into a different geometry.
 directed=sparse.csr_matrix((cost,(u,v)),shape=(len(xy),len(xy)));matrix=directed.maximum(directed.T)
 return matrix,pc

def fit_cohort(matrix,obs,frame):
 ids=frame.node_id.to_numpy(int);seeds=frame.loc[frame.is_source_cell.astype(str).str.lower().eq('true'),'node_id'].to_numpy(int)
 sub=matrix[ids][:,ids];local_seeds=pd.Index(ids).get_indexer(seeds)
 reach=np.isfinite(dijkstra(sub,directed=False,indices=local_seeds,min_only=True));ids=ids[reach]
 identity=frame.identity.iloc[0];backward=matrix[ids][:,ids].tocoo()
 if identity=='Malignant_epithelial':
  stages=obs.published_cell_type.map({'DCIS_1':0,'DCIS_2':0,'Invasive_Tumor':1,'Prolif_Invasive_Tumor':1}).to_numpy()
  allowed=stages[ids[backward.col]]<=stages[ids[backward.row]]
  backward=sparse.csr_matrix((backward.data[allowed],(backward.row[allowed],backward.col[allowed])),shape=backward.shape)
 else:backward=backward.tocsr()
 si=pd.Index(ids).get_indexer(seeds);assert (si>=0).all()
 keep=np.isfinite(dijkstra(backward.T.tocsr(),directed=True,indices=si,min_only=True));ids=ids[keep];backward=backward[keep][:,keep].tocsr();si=pd.Index(ids).get_indexer(seeds)
 w=backward.copy();w.data=np.exp(-w.data**2);degree=np.asarray(w.sum(1)).ravel();p=sparse.diags(1/np.maximum(degree,1e-300))@w
 free=np.setdiff1d(np.arange(len(ids)),si);h=np.ones(len(ids));a=sparse.eye(len(free),format='csc')-.97*p[free][:,free];b=.97*np.asarray(p[free][:,si].sum(1)).ravel()
 # The old CG stopping criterion only bounds a global residual. For long
 # local paths it can leave tiny remote potentials inaccurate or negative.
 # A direct M-matrix solve plus a relative per-cell equation check is used;
 # negative/zero solutions are never clipped into artificial trajectories.
 if len(free):
  if os.environ.get('NICHEFOREST_POSITIVE_ITERATION')=='1':
   h[free]=0
   for iteration in range(6000):
    previous=h.copy();h=.97*(p@h);h[si]=1
    if iteration%25==0 and (h[free]>0).all():
     error=np.max(np.abs(h[free]-previous[free])/np.maximum(h[free],1e-300))
     if error<1e-11:break
   else:raise RuntimeError('Positive fixed-point solver did not meet per-cell tolerance')
  else:h[free]=spsolve(a,b)
 assert np.isfinite(h).all() and (h>0).all() and (h<=1+1e-9).all()
 relative=np.max(np.abs(h[free]-.97*(p@h)[free])/np.maximum(h[free],1e-300)) if len(free) else 0
 assert relative<1e-8,relative
 tau=-np.log(h);parent=np.full(len(ids),-1,int)
 for v in free:
  start,end=backward.indptr[v:v+2];nn=backward.indices[start:end];cost=backward.data[start:end];ok=h[nn]>h[v];assert ok.any()
  options=nn[ok];values=cost[ok];parent[v]=options[np.lexsort((ids[options],values))[0]]
 sp_tau,sp_parent,sp_source=dijkstra(backward.T.tocsr(),directed=True,indices=si,min_only=True,return_predecessors=True)
 assert (tau[free]>tau[parent[free]]).all()
 return {'nodes':ids,'seeds':seeds,'tau':tau,'h':h,'parent_local':parent,'SPT_tau':sp_tau,'SPT_parent_local':np.where(sp_parent<0,-1,sp_parent),'maximum_relative_cell_equation_error':float(relative)}

def frame_from_fit(obs,fit,rid,identity):
 n=obs.iloc[fit['nodes']].copy();n['node_id']=fit['nodes'];n['root_id']=rid;n['pseudotime']=fit['tau']/max(max(fit['tau']),1e-12);n['is_source_cell']=fit['parent_local']<0
 ids=fit['nodes'];p=fit['parent_local'];xy=obs[['spatial_x','spatial_y']].to_numpy()
 edges=pd.DataFrame([{'root_id':rid,'identity':identity,'source':int(ids[u]),'target':int(ids[v]),'spatial_length_um':float(np.linalg.norm(xy[ids[u]]-xy[ids[v]]))} for v,u in enumerate(p) if u>=0])
 return n,edges

def main():
 OUT.mkdir(exist_ok=True,parents=True);(OUT/'PROTOCOL.json').write_text(json.dumps(PROTOCOL,indent=2))
 raw,obs,nodes,edges=load();x=sparse.csr_matrix(raw.X)[obs.raw_row.to_numpy(int)].astype(float)
 fullxy=obs[['spatial_x','spatial_y']].to_numpy();audit=[];stability=[];sensitivity=[]
 matrices={};frames=[];edgeframes=[];reference={}
 for rid,identity in CASES:
  own=np.flatnonzero(obs.identity.to_numpy()==identity);subobs=obs.iloc[own].reset_index(drop=True);xy=fullxy[own];counts=x[own]
  local,pc=geometry(counts,xy);coo=local.tocoo();matrix=sparse.csr_matrix((coo.data,(own[coo.row],own[coo.col])),shape=(len(obs),len(obs)))
  matrixpath=OUT/(rid+'_CANDIDATES.npz');sparse.save_npz(matrixpath,matrix);np.save(OUT/(rid+'_PCA.npy'),pc)
  frame=nodes.loc[nodes.root_id.eq(rid)&nodes.identity.eq(identity)];fit=fit_cohort(matrix,obs,frame);assert fit is not None
  n,e=frame_from_fit(obs,fit,rid,identity);frames.append(n);edgeframes.append(e);reference[rid]=fit
  n.to_csv(OUT/(rid+'_NODES.csv'),index=False);e.to_csv(OUT/(rid+'_EDGES.csv'),index=False)
  old=edges.loc[edges.root_id.eq(rid)&edges.identity.eq(identity)];oldset=set(map(tuple,old[['source','target']].to_numpy(int)))
  record={'root_id':rid,'identity':identity,'frozen_global_n':len(frame),'new_reachable_n':len(n),'fraction_of_frozen_support':len(n)/len(frame),'exact_source_n':len(fit['seeds']),'new_edges':len(e),
   'new_edges_not_frozen_global':sum((a,b) not in oldset for a,b in e[['source','target']].itertuples(index=False,name=None)),
   'edge_median_um':e.spatial_length_um.median(),'edge_max_um':e.spatial_length_um.max()}
  pd.DataFrame([record]).to_csv(OUT/(rid+'_AUDIT.csv'),index=False);audit.append(record);print(record,flush=True)
  # Sensitivity uses identical source/cohort, with no curve-driven selection.
  for rad,k in [(100,30),(200,30),(150,15),(150,50)]:
   graph,_=geometry(counts,xy,radius=rad,knn=k);coo=graph.tocoo();m=sparse.csr_matrix((coo.data,(own[coo.row],own[coo.col])),shape=(len(obs),len(obs)))
   f=fit_cohort(m,obs,frame);common=np.intersect1d(fit['nodes'],f['nodes']);union=np.union1d(fit['nodes'],f['nodes']);a=pd.Index(fit['nodes']).get_indexer(common);b=pd.Index(f['nodes']).get_indexer(common)
   from scipy.stats import spearmanr
   sensitivity.append({'root_id':rid,'identity':identity,'radius_um':rad,'knn':k,'n_nodes':len(f['nodes']),'support_jaccard':len(common)/len(union),'order_rho':spearmanr(fit['tau'][a],f['tau'][b]).statistic})
  for seed in [202610520,202610521,202610522]:
   thin=counts.copy();thin.data=np.random.default_rng(seed).binomial(thin.data.astype(int),.8).astype(float)
   graph,_=geometry(thin,xy);coo=graph.tocoo();m=sparse.csr_matrix((coo.data,(own[coo.row],own[coo.col])),shape=(len(obs),len(obs)));f=fit_cohort(m,obs,frame)
   common=np.intersect1d(fit['nodes'],f['nodes']);a=pd.Index(fit['nodes']).get_indexer(common);b=pd.Index(f['nodes']).get_indexer(common)
   for method,field in [('Multipath','tau'),('SPT','SPT_tau')]:stability.append({'root_id':rid,'identity':identity,'seed':seed,'method':method,'order_rho':spearmanr(fit[field][a],f[field][b]).statistic})
  pd.DataFrame(sensitivity).to_csv(OUT/'SENSITIVITY.csv',index=False);pd.DataFrame(stability).to_csv(OUT/'MOLECULE_THINNING.csv',index=False)
 pd.concat(frames,ignore_index=True).to_csv(OUT/'SPATIAL_CANDIDATE_NODES.csv',index=False);pd.concat(edgeframes,ignore_index=True).to_csv(OUT/'SPATIAL_CANDIDATE_EDGES.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'MODEL_AUDIT.csv',index=False)
 print('Development comparison complete. Frozen result not overwritten.',flush=True)

if __name__=='__main__':main()
