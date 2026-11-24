"""Cell-resolved implementation of the tested count-aware multipath model.

No spatial tiling, invented cell, displaced root, or dense N-by-N matrix.
Supplied spatial seeds are kept separate from inferred expression order.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.sparse.csgraph import dijkstra,connected_components
from scipy.sparse.linalg import cg
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/Cell_resolved_multipath'
PROTOCOL={
 'population':'Every published unambiguous CD4/CD8 T cell in the provided breast section. Annotation-assisted identity gate is an input, not evidence of unsupervised purity.',
 'units':'Original single cells; no grid, centroid substitution, old metacell support selection or cell duplication.',
 'features':'Measured counts only, theta100 NB Pearson residuals, clipped sqrt(N), PCA15. Whole-panel validation excludes all target genes before depth, residual, PCA and graph fitting.',
 'candidate':'Same-identity reciprocal30NN, self-scaled expression distance +0.1*d/(d+350); sparse implementation of the already tested v6/v10 geometry. No forced bridges.',
 'roots':'Exact original three spatial input coordinates, unchanged. Every observed same-identity cell within150um is a seed; >=6 seed cells required, otherwise abstain. Root-region membership via graph distances as in v6.',
 'support':'Softmax negative graph distance with common median nearest-root cost scale, >=0.2, union seeds; discard components without a seed. Nodes can be shared across roots.',
 'potential':'W=exp(-cost^2); D=diag(W1). On non-seeds solve (D-alpha W)h=alpha W_to_seeds 1 with h_seed=1, alpha=.97. tau=-log(h). Preconditioned conjugate gradients; verify relative linear residual <1e-8.',
 'tree':'Each nonseed selects minimum-cost candidate neighbor with strictly larger h, stable cell-ID tie-break. This maximizes total log conductance among forests respecting the new potential; not constrained to T0 ordering.',
 'validation':'10 binomial80%-molecule thinnings, seeds202610350..359, matched SPT and MSF controls. These are technical perturbations, not biological replicates. Roots and raw-cell IDs remain fixed.',
 'interpretation':'Root-conditioned within-T-cell state organization. No CD4-to-CD8, T-to-tumor/stroma, migration, or tumor-induced causal conversion inferred.',
 'protected':'AMCP, supplied root coordinates, original scoring, Fig3 and old release remain unchanged. No automatic publication-ready flag.'}

def candidate(counts,xy,identity,genes=None,knn=30,theta=100,pcs=15,spatial=.1):
 if genes is None:genes=np.arange(counts.shape[1])
 x=counts[:,genes];depth=np.maximum(x.sum(1),1);freq=x.sum(0)/max(x.sum(),1)
 mu=depth[:,None]*freq[None,:]
 r=np.clip((x-mu)/np.sqrt(np.maximum(mu+mu*mu/theta,1e-8)),-np.sqrt(len(x)),np.sqrt(len(x)))
 pc=PCA(min(pcs,len(x)-1,len(genes)),random_state=42).fit_transform(r)
 rr=[];cc=[];dd=[];scales=np.zeros(len(x))
 for label in np.unique(identity):
  ix=np.flatnonzero(identity==label);k=min(knn,len(ix)-1)
  d,nn=NearestNeighbors(n_neighbors=k+1,n_jobs=2).fit(pc[ix]).kneighbors(pc[ix])
  for j,node in enumerate(ix):
   # Explicit self removal, including duplicate-expression profiles.
   good=nn[j]!=j;neigh=nn[j][good][:k];dist=d[j][good][:k]
   scales[node]=max(dist[-1],1e-8)
   rr.extend(np.repeat(node,len(neigh)));cc.extend(ix[neigh]);dd.extend(dist)
 directed=sparse.csr_matrix((np.ones(len(rr),bool),(rr,cc)),shape=(len(x),len(x)))
 mask=directed.multiply(directed.T).tocoo();keep=mask.row<mask.col;u=mask.row[keep];v=mask.col[keep]
 ex=np.linalg.norm(pc[u]-pc[v],axis=1);sp=np.linalg.norm(xy[u]-xy[v],axis=1)
 cost=ex/np.sqrt(scales[u]*scales[v])+spatial*sp/(sp+350)
 matrix=sparse.csr_matrix((np.r_[cost,cost],(np.r_[u,v],np.r_[v,u])),shape=(len(x),len(x)))
 return matrix,pc

def support_sets(matrix,obs,roots,radius=150):
 xy=obs[['spatial_x','spatial_y']].to_numpy();identities=obs.identity.to_numpy();fits={};audit=[]
 distances={};seedsets={}
 for row in roots.itertuples():
  for identity in np.unique(identities):
   seeds=np.flatnonzero((identities==identity)&(np.linalg.norm(xy-[row.x,row.y],axis=1)<=radius))
   eligible=len(seeds)>=6
   audit.append({'root_id':row.root_id,'identity':identity,'fixed_x':row.x,'fixed_y':row.y,'n_seeds':len(seeds),'eligible':eligible,'radius_um':radius,'seed_ids':';'.join(obs.cell_name.iloc[seeds])})
   if not eligible:continue
   key=(row.root_id,identity);seedsets[key]=seeds
   distances[key]=dijkstra(matrix,directed=False,indices=seeds,min_only=True)
 for identity in np.unique(identities):
  keys=[key for key in distances if key[1]==identity];ids=np.flatnonzero(identities==identity)
  if not keys:continue
  dd=np.array([distances[key][ids] for key in keys]);nearest=dd.min(0);valid=np.isfinite(nearest)
  positive=nearest[valid&(nearest>0)];scale=np.median(positive) if len(positive) else 1
  weights=np.zeros(dd.shape);weights[:,valid]=np.exp(-(dd[:,valid]-nearest[valid])/max(scale,1e-8))
  probs=weights/np.maximum(weights.sum(0),1e-12)
  for j,key in enumerate(keys):
   seeds=seedsets[key];selected=np.union1d(ids[(probs[j]>=.2)&np.isfinite(dd[j])],seeds)
   sub=matrix[selected][:,selected].tocsr();position=pd.Index(selected).get_indexer(seeds);assert (position>=0).all()
   nc,components=connected_components(sub,directed=False)
   keep=np.isin(components,np.unique(components[position]));selected=selected[keep]
   fits[key]={'nodes':selected,'seeds':seeds,'membership':dict(zip(ids,probs[j]))}
 return fits,pd.DataFrame(audit)

def constructors(matrix,item,alpha=.97):
 nodes=item['nodes'];n=len(nodes);ix=pd.Index(nodes);si=ix.get_indexer(item['seeds']);assert (si>=0).all()
 sub=matrix[nodes][:,nodes].tocsr();w=sub.copy();w.data=np.exp(-np.minimum(w.data*w.data,700))
 degree=np.asarray(w.sum(1)).ravel();free=np.setdiff1d(np.arange(n),si)
 h=np.ones(n);residual=0.
 if len(free):
  a=sparse.diags(degree[free])-alpha*w[free][:,free]
  b=alpha*np.asarray(w[free][:,si].sum(1)).ravel()
  pre=sparse.diags(1/np.maximum(degree[free],1e-12))
  value,info=cg(a.tocsr(),b,M=pre,rtol=1e-12,atol=0,maxiter=5000)
  assert info==0,info
  residual=float(np.linalg.norm(a@value-b)/max(np.linalg.norm(b),1e-12));assert residual<1e-8
  h[free]=value
 assert np.isfinite(h).all() and (h>0).all() and (h<=1+1e-9).all()
 tau=-np.log(h);parent=np.full(n,-1,int)
 for v in free:
  start,end=sub.indptr[v:v+2];neighbors=sub.indices[start:end];cost=sub.data[start:end]
  ok=h[neighbors]>h[v];assert ok.any(),('potential maximum principle',v)
  options=neighbors[ok];values=cost[ok];parent[v]=options[np.lexsort((nodes[options],values))[0]]
 assert set(np.flatnonzero(parent<0))==set(si)
 assert (tau[parent[free]]<tau[free]).all()
 d,prev,source=dijkstra(sub,directed=False,indices=si,min_only=True,return_predecessors=True)
 assert np.isfinite(d).all()
 return {'nodes':nodes,'seeds':item['seeds'],'tau':tau,'parent_local':parent,'h':h,'SPT_tau':d,
         'SPT_parent_local':np.where(prev<0,-1,prev),'relative_linear_residual':residual}

def load_input():
 raw=ad.read_h5ad(ROOT/'outputs/NicheForest_paper_frozen_candidate/P6_frozen_breast/relation_input/event/synthetic.h5ad')
 cells=pd.read_csv(ROOT/'outputs/Manuscript_method_biology_alignment/RAW_CELLS_WITH_PUBLISHED_ANNOTATIONS.csv',dtype={'cell_name':str})
 assert np.array_equal(raw.obs_names.astype(str).str.replace('CELL_','',regex=False),cells.cell_name)
 sel=cells.loc[cells.published_cell_type.isin(['CD4+_T_Cells','CD8+_T_Cells'])].copy()
 sel['raw_row']=sel.index;sel['identity']=sel.published_cell_type;sel=sel.reset_index(drop=True)
 counts=sparse.csr_matrix(raw.X)[sel.raw_row].toarray().astype(float)
 roots=pd.read_csv(ROOT/'outputs/Root_region_state_forest/CANONICAL_ROOTS.csv')
 return raw,counts,sel,roots

def run():
 OUT.mkdir(exist_ok=True,parents=True);pp=OUT/'PROTOCOL.json'
 if pp.exists():assert json.loads(pp.read_text())==PROTOCOL
 else:pp.write_text(json.dumps(PROTOCOL,indent=2))
 print('protocol',hashlib.sha256(pp.read_bytes()).hexdigest(),flush=True)
 raw,counts,obs,roots=load_input();xy=obs[['spatial_x','spatial_y']].to_numpy();identity=obs.identity.to_numpy()
 obs.to_csv(OUT/'EXACT_CELL_INPUT.csv',index_label='node_id');roots.to_csv(OUT/'IMMUTABLE_SPATIAL_ROOTS.csv',index=False)
 a=ad.AnnData(X=sparse.csr_matrix(counts),obs=obs.copy());a.var_names=raw.var_names;a.obs_names=obs.cell_name.to_numpy();a.obsm['spatial']=xy
 a.write_h5ad(OUT/'EXACT_T_CELL_COUNTS.h5ad',compression='gzip')
 matrix,pc=candidate(counts,xy,identity);sparse.save_npz(OUT/'CANDIDATE_COSTS.npz',matrix);np.save(OUT/'COUNT_AWARE_PCA.npy',pc)
 fits,audit=support_sets(matrix,obs,roots);audit.to_csv(OUT/'ROOT_REGION_SEEDS.csv',index=False)
 reference={};records=[]
 for key,item in fits.items():
  fitted=constructors(matrix,item);reference[key]=fitted;ids=fitted['nodes'];tau=fitted['tau'];parent=fitted['parent_local']
  nodes=obs.iloc[ids].copy();nodes['node_id']=ids;nodes['pseudotime']=tau/(max(tau) or 1);nodes['unscaled_potential_distance']=tau;nodes['is_region_seed']=parent<0
  nodes['membership']=[item['membership'][int(i)] for i in ids]
  label=key[0]+'_'+key[1]
  nodes.to_csv(OUT/f'{label}_NODES.csv',index=False)
  edge=[{'source':int(ids[parent[j]]),'target':int(ids[j]),'cost':float(matrix[ids[parent[j]],ids[j]])} for j in np.flatnonzero(parent>=0)]
  pd.DataFrame(edge).to_csv(OUT/f'{label}_EDGES.csv',index=False)
  old=np.column_stack([fitted['SPT_parent_local'],np.arange(len(ids))]);new=np.column_stack([parent,np.arange(len(ids))])
  changed=int(np.sum(parent!=fitted['SPT_parent_local']))
  records.append({'root_id':key[0],'identity':key[1],'nodes':len(ids),'seeds':len(item['seeds']),'edges':len(edge),'replaced_parent_edges_vs_SPT':changed,'relative_linear_residual':fitted['relative_linear_residual']})
 pd.DataFrame(records).to_csv(OUT/'FULL_FIT_AUDIT.csv',index=False)
 stability=[]
 for seed in range(202610350,202610360):
  thinned=np.random.default_rng(seed).binomial(counts.astype(int),.8).astype(float)
  trial,_=candidate(thinned,xy,identity);other,_=support_sets(trial,obs,roots)
  for key,original in reference.items():
   if key not in other:continue
   fitted=constructors(trial,other[key]);common=np.intersect1d(original['nodes'],fitted['nodes']);union=np.union1d(original['nodes'],fitted['nodes'])
   oi=pd.Index(original['nodes']).get_indexer(common);ti=pd.Index(fitted['nodes']).get_indexer(common)
   from scipy.stats import spearmanr
   for name,field in [('Hitting_0.97','tau'),('State_SPT','SPT_tau')]:
    stability.append({'seed':seed,'root_id':key[0],'identity':key[1],'method':name,'support_jaccard':len(common)/len(union),
      'order_spearman':spearmanr(original[field][oi],fitted[field][ti]).statistic,'n_common':len(common)})
  print('cell-resolved molecule thinning',seed,flush=True)
 pd.DataFrame(stability).to_csv(OUT/'MOLECULE_THINNING_STABILITY.csv',index=False)
 print(pd.DataFrame(records).to_string(index=False),flush=True)
 print(pd.DataFrame(stability).groupby(['root_id','identity','method']).order_spearman.median().to_string(),flush=True)

if __name__=='__main__':run()
