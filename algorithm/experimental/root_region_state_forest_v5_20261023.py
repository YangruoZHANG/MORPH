"""Root-region, identity-stratified state forest with explicit uncertainty.

This development version preserves supplied root *locations*, does not label
one arbitrary nearest cell subtype as a biological progenitor, and validates
all measured non-construction panels. It never overwrites the paper release.
"""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
import json
import hashlib
import numpy as np
import pandas as pd
import networkx as nx
import anndata as ad
from scipy.spatial.distance import pdist,squareform
from scipy.stats import rankdata,spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rebuild_local_multiroot_forest_20261023 import ROOT

OUT=ROOT/'outputs/Root_region_state_forest_v5_20261023'
SRC=ROOT/'outputs/Local_multiroot_revision_v3_20261023'
P8=ROOT/'outputs/NicheForest_paper_frozen_candidate/P8_breast_biological_validation'
SEEDS=[202610260,202610261,202610262,202610263,202610264]
PROTOCOL={
 'status':'Development candidate; no automatic freezing or publication claim',
 'root_locations':'Exactly selected_roots_breast.csv, never replaced by a centroid or display hub',
 'root_semantics':'Spatial root regions supplied by upstream root finder, not verified progenitor cells',
 'initial_states':'All observed local units within 150um of supplied root, stratified by published CD4/CD8 identity; >=2 units and >=6 cells per root/identity else abstain. No closest-subtype selection.',
 'units':'Same 1009 local pseudobulks from v3, >=3 cells and one published CD4/CD8 identity per 150um tile',
 'identity_input':'Explicit annotation-assisted mode, not unsupervised identity inference',
 'graph':'Within identity, reciprocal expression 15NN on standardized logCP10K 50 original construction genes, PCA<=15. Five 80% gene resamples, consensus frequency>=3/5. No forced bridges.',
 'edge_cost':'Mean self-scaled expression distance + 0.1*d/(d+350) + 1-consensus frequency',
 'forest':'Multisource shortest-path forest. All region seeds are sources with zero initial distance; ties deterministic; no edges between cell identities or between disconnected components.',
 'root_membership':'Softmax of negative multisource graph distances with common median nearest-root distance scale; >=0.2; then seed-connected subgraph; root regions can have overlapping state supports',
 'ordering':'Distance on retained candidate graph before fine tree selection, not distance accumulated on an MST',
 'branch_policy':'Do not call every fine-tree leaf a biological endpoint. Fine connectivity is interpolation; no unvalidated fixed branch number or forced terminal label.',
 'program_validation':'All 43 original primary external/Hallmark measured panels excluding original 50 construction genes; within-identity mean gene-z score; partial rank vs pseudotime controlling panel depth, unit cell count and x/y; 999 spatial-block permutations, BH over all valid root/identity/panel tests, abs(rho)>=0.3 and leave-one-gene-out sign agreement>=0.8',
 'start_state_check':'Compare every region seed distribution with all units of same identity for naive/memory, cytotoxic and checkpoint panels. Diagnostic, not a root-reselection criterion.',
 'cell_subsample_check':'Five fixed 80% within-unit cell subsamples; root regions unchanged; compare root/identity support Jaccard and order rank correlation. Roots absent from a candidate component cause explicit abstention.',
 'seed_radius_sensitivity_um':[100,150,200],
 'evidence_boundary':'Panel-derived annotations and original construction-gene selection are not independent of expression. One section does not establish ancestry or tumor-induced causation.',
 'protected':'No AMCP, formal scoring, Fig3, old P6/P8/P9, PPTX or paper edits before evidence gates pass',
 'selection':'No favorable panel selection, no post-outcome threshold changes',
 'seeds':SEEDS}


def bh(p):
 p=np.asarray(p);idx=np.argsort(p);q=np.empty(len(p));q[idx]=np.minimum.accumulate((p[idx]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
 return np.minimum(q,1)


def candidate(log,xy,genes,identities):
 pc=PCA(min(15,len(genes),len(log)-1),random_state=42).fit_transform(StandardScaler().fit_transform(log[:,genes]))
 ex=squareform(pdist(pc));same=identities[:,None]==identities[None,:]
 distance=ex.copy();distance[~same]=np.inf
 nn=np.argsort(distance,axis=1)[:,1:16]
 mask=np.zeros(ex.shape,bool);mask[np.arange(len(log))[:,None],nn]=True
 a,b=np.where(np.triu(mask&mask.T&same,1))
 scale=np.maximum(np.take_along_axis(distance,nn[:,-1:],axis=1).ravel(),1e-8)
 sp=np.linalg.norm(xy[a]-xy[b],axis=1)
 return pd.DataFrame({'source':a,'target':b,'weight':ex[a,b]/np.sqrt(scale[a]*scale[b])+.1*sp/(sp+350)})


def consensus_graph(counts,xy,genes,identity):
 log=np.log1p(counts/np.maximum(counts.sum(1,keepdims=True),1)*1e4)
 frames=[]
 for seed in SEEDS:
  ix=np.random.default_rng(seed).choice(genes,int(.8*len(genes)),replace=False)
  frames.append(candidate(log,xy,ix,identity))
 cc=pd.concat(frames).groupby(['source','target']).agg(mean_cost=('weight','mean'),n=('weight','size')).reset_index()
 cc['frequency']=cc.n/len(SEEDS);cc=cc.loc[cc.frequency>=.6].copy()
 cc['weight']=cc.mean_cost+1-cc.frequency
 g=nx.from_pandas_edgelist(cc,'source','target',edge_attr=['weight','frequency'])
 g.add_nodes_from(range(len(counts)))
 return g,cc,log


def root_graphs(g,obs,roots,radius):
 results={};seed_audit=[]
 distances={};seed_sets={}
 xy=obs[['spatial_x','spatial_y']].to_numpy()
 for row in roots.itertuples():
  rid=row.root_id
  for identity in sorted(obs.identity.unique()):
   near=(obs.identity.to_numpy()==identity)&(np.linalg.norm(xy-np.array([row.x,row.y]),axis=1)<=radius)
   seeds=set(obs.index[near]);nc=int(obs.loc[sorted(seeds),'n_cells'].sum())
   valid=len(seeds)>=2 and nc>=6
   seed_audit.append({'root_id':rid,'identity':identity,'radius_um':radius,'n_seed_units':len(seeds),'n_seed_cells':nc,'eligible':valid,
                      'fixed_root_x':row.x,'fixed_root_y':row.y,'seed_nodes':';'.join(map(str,sorted(seeds)))})
   if valid:
    d,p=nx.multi_source_dijkstra(g,sorted(seeds),weight='weight')
    distances[rid,identity]=d;seed_sets[rid,identity]=seeds
 for identity in sorted(obs.identity.unique()):
  keys=[key for key in distances if key[1]==identity]
  if not keys:continue
  ids=obs.index[obs.identity.eq(identity)].to_numpy(int)
  dd=np.array([[distances[key].get(int(i),np.inf) for i in ids] for key in keys])
  mm=dd.min(0);valid=np.isfinite(mm);scale=np.median(mm[valid&(mm>0)]) if (valid&(mm>0)).any() else 1
  weights=np.zeros(dd.shape);weights[:,valid]=np.exp(-(dd[:,valid]-mm[valid])/max(scale,1e-8))
  probs=weights/np.maximum(weights.sum(0),1e-12)
  for j,key in enumerate(keys):
   seeds=seed_sets[key]
   support=set(ids[(probs[j]>=.2)&np.isfinite(dd[j])])|seeds
   local=g.subgraph(support).copy()
   d,paths=nx.multi_source_dijkstra(local,sorted(seeds),weight='weight')
   nodes=sorted(d);tree=nx.DiGraph();tree.add_nodes_from(nodes)
   origins={}
   for v,path in paths.items():
    origins[v]=path[0]
    if len(path)>1:
     u=path[-2];tree.add_edge(u,v,**g[u][v])
   assert nx.is_directed_acyclic_graph(tree)
   assert all(tree.in_degree(v)<=1 for v in tree)
   assert all(tree.in_degree(v)==0 for v in seeds)
   assert nx.is_forest(tree.to_undirected())
   results[key]={'nodes':nodes,'tree':tree,'distance':d,'seeds':seeds,'origin':origins,
                 'membership':dict(zip(ids,probs[j]))}
 return results,pd.DataFrame(seed_audit)


def partial_ranks(tau,score,design):
 a=rankdata(tau);b=rankdata(score,axis=0)
 a=a-design@np.linalg.lstsq(design,a,rcond=None)[0]
 b=b-design@np.linalg.lstsq(design,b,rcond=None)[0]
 return a,b,a@b/np.maximum(np.linalg.norm(a)*np.linalg.norm(b,axis=0),1e-12)


def main():
 OUT.mkdir(exist_ok=True,parents=True);pp=OUT/'PROTOCOL.json'
 if pp.exists():assert json.loads(pp.read_text())==PROTOCOL
 else:pp.write_text(json.dumps(PROTOCOL,indent=2))
 print('Protocol',hashlib.sha256(pp.read_bytes()).hexdigest(),flush=True)
 data=ad.read_h5ad(SRC/'LOCAL_COUNTS.h5ad');counts=data.X.toarray()
 obs=pd.read_csv(SRC/'LOCAL_UNITS.csv').set_index('node_id')
 roots=pd.read_csv(ROOT/'data/选择的root/selected_roots_breast.csv')
 roots['root_cluster']=roots.root_id;roots['root_id']=['R0','R1','R2']
 roots.to_csv(OUT/'CANONICAL_ROOTS.csv',index=False)
 role=pd.read_csv(SRC/'GENE_ROLES.csv');genes=np.flatnonzero(role.role.eq('construction'))
 identity=obs.identity.to_numpy();xy=obs[['spatial_x','spatial_y']].to_numpy()
 g,cc,log=consensus_graph(counts,xy,genes,identity)
 cc.to_csv(OUT/'STATE_CANDIDATES.csv',index=False)
 primary,seed_audit=root_graphs(g,obs,roots,150);seed_audit.to_csv(OUT/'ROOT_REGION_SEEDS.csv',index=False)
 cat=pd.read_csv(P8/'P8_INDEPENDENT_VS_POSTHOC_EVIDENCE.csv')
 cat=cat.loc[cat.family.isin(['external_curated','Hallmark_2020'])&cat.n_independent_genes.ge(3)].copy()
 programs={r.family+'::'+r.gene_set:[data.var_names.get_loc(gene) for gene in r.independent_genes.split(';') if gene in data.var_names] for r in cat.itertuples()}
 assert all(not set(ii)&set(genes) for ii in programs.values())
 cat.to_csv(OUT/'PROGRAM_CATALOG.csv',index=False)
 z=np.zeros(log.shape)
 for ident in obs.identity.unique():
  ii=np.flatnonzero(identity==ident);z[ii]=StandardScaler().fit_transform(log[ii])
 panel=np.column_stack([z[:,ii].mean(1) for ii in programs.values()])
 pd.DataFrame(panel,columns=list(programs)).to_csv(OUT/'IDENTITY_RESTRICTED_PANEL_SCORES.csv',index_label='node_id')
 tests=[];starts=[];summary=[]
 for (rid,ident),result in primary.items():
  label=rid+'_'+ident.replace('+','').replace('_Cells','')
  ids=np.array(result['nodes']);d=result['distance'];tau=np.array([d[i] for i in ids]);tau=tau/(max(tau) or 1)
  nodes=obs.loc[ids].copy();nodes['pseudotime']=tau;nodes['membership']=[result['membership'][i] for i in ids]
  nodes['is_region_seed']=[i in result['seeds'] for i in ids];nodes['seed_origin']=[result['origin'][i] for i in ids]
  nodes.to_csv(OUT/f'{label}_NODES.csv',index_label='node_id')
  er=[]
  for u,v,e in result['tree'].edges(data=True):
   er.append({'source':u,'target':v,'weight':e['weight'],'bootstrap_support':e['frequency'],
              'spatial_um':float(np.linalg.norm(xy[u]-xy[v])),'semantics':'same-identity state adjacency, not migration'})
  pd.DataFrame(er,columns=['source','target','weight','bootstrap_support','spatial_um','semantics']).to_csv(OUT/f'{label}_EDGES.csv',index=False)
  summary.append({'root_id':rid,'identity':ident,'n_nodes':len(ids),'n_seeds':len(result['seeds']),
                  'n_components':nx.number_connected_components(result['tree'].to_undirected()),'n_cells':int(nodes.n_cells.sum())})
  if len(ids)<20:continue
  xx=np.column_stack([np.log1p(obs.loc[ids,'n_cells']),np.log1p(counts[ids].sum(1)),xy[ids]])
  xx=rankdata(xx,axis=0);xx=(xx-xx.mean(0))/np.maximum(xx.std(0),1)
  design=np.column_stack([np.ones(len(ids)),xx])
  a,b,rho=partial_ranks(tau,panel[ids],design)
  blocks=(xy[ids,0]>np.median(xy[ids,0])).astype(int)*2+(xy[ids,1]>np.median(xy[ids,1])).astype(int)
  rng=np.random.default_rng(SEEDS[0]+int(rid[-1]));null=[]
  for _ in range(999):
   ix=np.arange(len(ids))
   for block in np.unique(blocks):
    jj=np.flatnonzero(blocks==block);ix[jj]=rng.permutation(jj)
   trial=a[ix];trial-=design@np.linalg.lstsq(design,trial,rcond=None)[0]
   null.append(trial@b/np.maximum(np.linalg.norm(trial)*np.linalg.norm(b,axis=0),1e-12))
  pv=(1+(np.abs(null)>=np.abs(rho)).sum(0))/1000
  for k,(program,ii) in enumerate(programs.items()):
   signs=[]
   for gene in ii:
    other=[t for t in ii if t!=gene]
    loo=z[ids][:,other].mean(1)
    signs.append(np.sign(partial_ranks(tau,loo,design)[2])==np.sign(rho[k]))
   tests.append({'root_id':rid,'identity':ident,'program':program,'n_nodes':len(ids),
                 'partial_rho':rho[k],'spatial_permutation_p':pv[k],'leave_one_gene_sign_agreement':float(np.mean(signs))})
  for program in ['external_curated::NAIVE_MEMORY_T','external_curated::CYTOTOXIC_EFFECTOR','external_curated::CHECKPOINT_EXHAUSTION']:
   if program not in programs:continue
   k=list(programs).index(program);allids=obs.index[obs.identity.eq(ident)].to_numpy()
   seedvalues=panel[sorted(result['seeds']),k]
   starts.append({'root_id':rid,'identity':ident,'program':program,'n_seed_units':len(seedvalues),
                   'seed_mean_z':float(seedvalues.mean()),'seed_median_population_percentile':float(np.median([np.mean(panel[allids,k]<=v) for v in seedvalues]))})
  print('primary',summary[-1],flush=True)
 results=pd.DataFrame(tests)
 if len(results):
  results['q_all_tests']=bh(results.spatial_permutation_p)
  results['qualified']=results.q_all_tests.le(.1)&results.partial_rho.abs().ge(.3)&results.leave_one_gene_sign_agreement.ge(.8)
 results.to_csv(OUT/'ALL_PROGRAM_TRENDS.csv',index=False)
 pd.DataFrame(summary).to_csv(OUT/'ROOT_IDENTITY_FORESTS.csv',index=False)
 pd.DataFrame(starts).to_csv(OUT/'START_STATE_DIAGNOSTICS.csv',index=False)
 sensitivity=[]
 for radius in [100,200]:
  trial,audit=root_graphs(g,obs,roots,radius);audit.to_csv(OUT/f'ROOT_REGION_SEEDS_RADIUS_{radius}.csv',index=False)
  for key,orig in primary.items():
   other=trial.get(key)
   a=set(orig['nodes']);b=set(other['nodes']) if other else set();common=sorted(a&b)
   correlation=spearmanr([orig['distance'][i] for i in common],[other['distance'][i] for i in common]).statistic if len(common)>=20 else np.nan
   sensitivity.append({'root_id':key[0],'identity':key[1],'radius_um':radius,'support_jaccard':len(common)/len(a|b),
                       'pseudotime_spearman':correlation,'abstained':other is None})
 pd.DataFrame(sensitivity).to_csv(OUT/'SEED_RADIUS_SENSITIVITY.csv',index=False)
 raw=ad.read_h5ad(ROOT/'outputs/NicheForest_paper_frozen_candidate/P6_frozen_breast/relation_input/event/synthetic.h5ad')
 raw_names=raw.obs_names.astype(str).str.replace('CELL_','',regex=False)
 mapping=pd.read_csv(SRC/'CELL_TO_LOCAL_UNIT.csv',dtype={'cell_name':str})
 rawids=pd.Index(raw_names).get_indexer(mapping.cell_name);assert (rawids>=0).all()
 rows=[]
 for seed in SEEDS:
  rng=np.random.default_rng(seed);keep=[]
  for _,group in mapping.groupby('node_id'):
   keep.extend(rng.choice(group.index,max(1,int(.8*len(group))),replace=False))
  keep=np.array(keep);unit=mapping.loc[keep,'node_id'].to_numpy()
  aggregate=sparse_aggregate(raw,rawids[keep],unit,len(obs))
  trial,_,_=consensus_graph(aggregate,xy,genes,identity)
  forest,_=root_graphs(trial,obs,roots,150)
  for key,orig in primary.items():
   other=forest.get(key);a=set(orig['nodes']);b=set(other['nodes']) if other else set();common=sorted(a&b)
   corr=spearmanr([orig['distance'][i] for i in common],[other['distance'][i] for i in common]).statistic if len(common)>=20 else np.nan
   rows.append({'seed':seed,'root_id':key[0],'identity':key[1],'support_jaccard':len(common)/len(a|b),'pseudotime_spearman':corr})
  print('cell bootstrap',seed,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'CELL_SUBSAMPLE_STABILITY.csv',index=False)
 (OUT/'STATUS.json').write_text(json.dumps({'promoted':False,'n_program_tests':len(results),
  'n_qualified_programs':int(results.qualified.sum()) if len(results) else 0,'n_root_identity_forests':len(primary),
  'protected_artifacts_modified':False,'interpretation':'Spatial-root-conditioned state forests, not validated progenitor-to-descendant lineages.'},indent=2))
 print('qualified',results.loc[results.qualified].to_string(index=False),flush=True)


def sparse_aggregate(raw,indices,units,n):
 from scipy import sparse
 a=sparse.csr_matrix((np.ones(len(indices)),(units,np.arange(len(indices)))),shape=(n,len(indices)))
 return (a@sparse.csr_matrix(raw.X)[indices]).toarray()


if __name__=='__main__':main()
