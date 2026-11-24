"""Explicit stage-prior sensitivity, separate from unsupervised claims.

Clinical stage admissibility is an INPUT. Zero backward stage edges is thus
not an accuracy metric. Test a genuine reorientation with independent marker
exclusion before considering this annotation-assisted alternative.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import bicgstab,gmres
from identity_conditioned_forest import ROOT,load,per_identity_candidates
from multipath_ordering import support_sets,constructors
OUT=ROOT/'outputs/Stage_admissible_epithelial'
PROTOCOL={
 'status':'Histopathology-annotation-assisted sensitivity, not an unsupervised or ancestry-confirmed result',
 'stage_prior':'Published DCIS1/2 stage0; published Invasive/ProliferatingInvasive stage1. Forbid stage1-parent -> stage0-child, allow within-stage and stage0 -> stage1. Labels are inputs, not heldout truth.',
 'source_frontier':'Within original150um anchor region use the lowest stage with >=6 observed cells; no coordinate or upstream root changes. If no eligible stage, abstain. Rare single DCIS cell cannot replace a supported invasive source.',
 'support':'Same original regional root support, then directed seed-reachable nodes only; unreachable nodes explicit abstentions, no bridge.',
 'potential':'Backward parent-candidate Markov chain with absorption at selected seeds, alpha=.97. Solve (I-alpha P_FF)h=alpha P_FS1, unique contraction. A directed transition prior has no symmetric Dirichlet-energy claim.',
 'tree':'Minimum-cost higher-h eligible parent, independent of T0 order and parent set, seed nodes immutable. Clinical-stage reversal reduction is enforced, not evidence of improved accuracy.',
 'comparison':'Unconstrained v15 on exactly matched retained nodes and original seed subset when reporting edge roughness. No improvement can be attributed solely to node removal.',
 'protected':'No original root, AMCP, Fig3, official scoring or existing deck/manuscript overwrite.'}

def stage_fit(matrix,obs,item,alpha=.97):
 stages=obs.published_cell_type.map({'DCIS_1':0,'DCIS_2':0,'Invasive_Tumor':1,'Prolif_Invasive_Tumor':1}).to_numpy()
 seeds=item['seeds'];eligible=[s for s in [0,1] if np.sum(stages[seeds]==s)>=6]
 if not eligible:return None
 source_stage=min(eligible);seeds=seeds[stages[seeds]==source_stage]
 ids=item['nodes'];ix=pd.Index(ids);sub=matrix[ids][:,ids].tocoo()
 # row=child, col=parent: upstream clinical stage cannot be greater.
 allowed=stages[ids[sub.col]]<=stages[ids[sub.row]]
 backward=sparse.csr_matrix((sub.data[allowed],(sub.row[allowed],sub.col[allowed])),shape=sub.shape)
 si=ix.get_indexer(seeds);assert (si>=0).all()
 reachable=np.isfinite(dijkstra(backward.T.tocsr(),directed=True,indices=si,min_only=True))
 ids=ids[reachable];backward=backward[reachable][:,reachable].tocsr();si=pd.Index(ids).get_indexer(seeds)
 w=backward.copy();w.data=np.exp(-np.minimum(w.data*w.data,700))
 degree=np.asarray(w.sum(1)).ravel();p=sparse.diags(1/np.maximum(degree,1e-300))@w
 free=np.setdiff1d(np.arange(len(ids)),si);h=np.ones(len(ids));residual=0.
 if len(free):
  a=sparse.eye(len(free),format='csr')-alpha*p[free][:,free]
  b=alpha*np.asarray(p[free][:,si].sum(1)).ravel()
  value,info=bicgstab(a,b,rtol=1e-12,atol=0,maxiter=5000)
  if info!=0:value,info=gmres(a,b,rtol=1e-12,atol=0,restart=100,maxiter=1000)
  assert info==0,info
  residual=float(np.linalg.norm(a@value-b)/max(np.linalg.norm(b),1e-12));assert residual<1e-8;h[free]=value
 assert (h>0).all() and (h<=1+1e-8).all()
 tau=-np.log(h);parent=np.full(len(ids),-1,int)
 for v in free:
  start,end=backward.indptr[v:v+2];nn=backward.indices[start:end];cost=backward.data[start:end]
  good=h[nn]>h[v];assert good.any()
  options=nn[good];values=cost[good];parent[v]=options[np.lexsort((ids[options],values))[0]]
 assert (stages[ids[parent[free]]]<=stages[ids[free]]).all()
 sp_tau,sp_parent,sp_source=dijkstra(backward.T.tocsr(),directed=True,indices=si,min_only=True,return_predecessors=True)
 assert np.isfinite(sp_tau).all()
 return {'nodes':ids,'seeds':seeds,'source_stage':source_stage,'tau':tau,'parent_local':parent,'residual':residual,
         'SPT_tau':sp_tau,'SPT_parent_local':np.where(sp_parent<0,-1,sp_parent),
         'unreachable_n':int((~reachable).sum()),'source_excluded_higher_stage_cells':int(len(item['seeds'])-len(seeds))}

def run():
 OUT.mkdir(exist_ok=True,parents=True);pp=OUT/'PROTOCOL.json'
 if pp.exists():assert json.loads(pp.read_text())==PROTOCOL
 else:pp.write_text(json.dumps(PROTOCOL,indent=2))
 raw,counts,obs,roots=load();matrix=sparse.load_npz(ROOT/'outputs/Niche_conditioned_identity_forests/CANDIDATE_COSTS.npz')
 fits,seeds=support_sets(matrix,obs,roots);audit=[]
 for key,item in fits.items():
  if key[1]!='Malignant_epithelial':continue
  fitted=stage_fit(matrix,obs,item)
  if fitted is None:continue
  ids=fitted['nodes'];parent=fitted['parent_local'];nodes=obs.iloc[ids].copy();nodes['node_id']=ids;nodes['pseudotime']=fitted['tau']/(max(fitted['tau']) or 1);nodes['is_region_seed']=parent<0
  nodes.to_csv(OUT/f'{key[0]}_NODES.csv',index=False)
  edges=[{'source':int(ids[parent[j]]),'target':int(ids[j]),'cost':float(matrix[ids[parent[j]],ids[j]])} for j in np.flatnonzero(parent>=0)]
  pd.DataFrame(edges).to_csv(OUT/f'{key[0]}_EDGES.csv',index=False)
  unconstrained=constructors(matrix,{'nodes':ids,'seeds':fitted['seeds']})
  audit.append({'root_id':key[0],'n_nodes':len(ids),'n_seeds':len(fitted['seeds']),'source_stage':fitted['source_stage'],'unreachable_abstentions':fitted['unreachable_n'],
     'initial_seed_cells_excluded_higher_stage':fitted['source_excluded_higher_stage_cells'],'residual':fitted['residual'],
     'replaced_parent_edges_on_matched_nodes':int(np.sum(unconstrained['parent_local']!=parent)),
     'forward_clinical_stage_edges':int(sum(obs.published_cell_type.iloc[e['source']].startswith('DCIS') and not obs.published_cell_type.iloc[e['target']].startswith('DCIS') for e in edges))})
  print(audit[-1],flush=True)
 pd.DataFrame(audit).to_csv(OUT/'STAGE_PRIOR_STRUCTURAL_AUDIT.csv',index=False)

if __name__=='__main__':run()
