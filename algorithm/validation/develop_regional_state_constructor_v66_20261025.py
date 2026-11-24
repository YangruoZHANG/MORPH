"""Matched-unit supplied-root development benchmark; not old Fig3 results.

Both constructors receive EXACTLY the same coordinate-only units, feature
graph, supplied source cells, support and state grouping. Truth is aggregated
only after fitting. This tests construction at the new observation resolution;
it is not single-cell parent accuracy or a root-finding benchmark.
"""
import os,sys,json,hashlib
from pathlib import Path
from types import SimpleNamespace
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
ROOT=Path(__file__).resolve().parents[2]
for p in ['algorithm/validation','algorithm/experimental','algorithm/experimental/final_tree_method_development']:sys.path.insert(0,str(ROOT/p))
import numpy as np
import pandas as pd
import anndata as ad
import networkx as nx
from scipy import sparse
from scipy.sparse.csgraph import dijkstra,minimum_spanning_tree
from regional_source_ensemble_units_v63_20261025 import make_units,geometry
from score_current_constructor_with_original_metrics_20261024 import supports,truth_context,Q,BRANCH
from run_paper_baseline_adapter_smoke import roots_from_truth
from cell_resolved_multipath_v14_20261023 import constructors
from rooted_conductance_gradient_forest_v67_20261025 import gradient_parents
from undiscounted_root_passage_v68_20261025 import passage_fit
from count_spatial_agreement_kernel_v70_20261025 import agreement_graph
AGREEMENT=os.environ.get('NICHEFOREST_COUNT_SPATIAL_AGREEMENT')=='1'
FLOW=os.environ.get('NICHEFOREST_CONDUCTANCE_GRADIENT')=='1'
PASSAGE=os.environ.get('NICHEFOREST_UNDISCOUNTED_PASSAGE')=='1'
OUT=ROOT/'outputs/Regional_state_matched_unit_benchmark_v66_20261025'
if FLOW:OUT=ROOT/'outputs/Rooted_gradient_matched_unit_benchmark_v67_20261025'
if PASSAGE:OUT=ROOT/'outputs/Root_passage_matched_unit_benchmark_v68_20261025'
if AGREEMENT:OUT=ROOT/'outputs/Count_spatial_agreement_matched_unit_benchmark_v70_20261025'
SOURCE=ROOT/'outputs/Local_state_geometry_v31_20261024/independent_simulation'

def aggregated_truth(a,unitnames,member,rootnames,seeds,unitof):
 raw=truth_context(a,rootnames,seeds);prob={};support={};branch={};ptime={};names=a.obs_names.astype(str)
 for unit,ids in member.groupby('unit_node_id').core_cell_id:
  rawids=ids.to_numpy(int);uid=str(unitnames[unit])
  for rid in raw.roots:
   for state in raw.graphs[rid]:
    value=float(np.mean([raw.state_probability.get((str(names[i]),rid,state),0.) for i in rawids]))
    if value>0:prob[uid,rid,state]=value
    if value>=.2:support.setdefault((uid,rid),set()).add(state)
   own=[int(i) for i in rawids if (str(names[i]),rid) in raw.branch]
   if own:
    labels=pd.Series([raw.branch[str(names[i]),rid] for i in own]);branch[uid,rid]=str(labels.value_counts().sort_index().idxmax());ptime[uid,rid]=float(np.mean([raw.pseudotime[str(names[i]),rid] for i in own]))
 return SimpleNamespace(roots=raw.roots,graphs=raw.graphs,anchors={f'R{j}':str(unitnames[unitof[s]]) for j,s in enumerate(seeds)},branch=branch,pseudotime=ptime,state_probability=prob,state_support=support,state_support_threshold=.2)

def run_case(file,dest):
 a=ad.read_h5ad(file);names=a.obs_names.astype(str);seeds=[a.obs_names.get_loc(s) for s in roots_from_truth(a)];n=a.n_obs
 obs=pd.DataFrame({'identity':['Unknown']*n,'published_cell_type':['Unknown']*n,'spatial_x':a.obsm['spatial'][:,0],'spatial_y':a.obsm['spatial'][:,1],'cell_name':names,'raw_row':np.arange(n),'original_model_node_id':np.arange(n)})
 source=pd.DataFrame({'count_view':['WholeCell']*len(seeds),'root_id':[f'R{j}' for j in range(len(seeds))],'core_node_id':seeds});units,member,aggregate,mean,_=make_units(obs,source,'ensemble',maximum=9);x=sparse.csr_matrix(a.X);g,_=geometry(x,obs,units,mean,np.arange(a.n_vars));unitof=member.set_index('core_cell_id').loc[np.arange(n),'unit_node_id'].to_numpy(int);unitnames=units.unit_id.to_numpy();items=supports(g,[unitof[s] for s in seeds]);c=(aggregate@x).toarray();method_names=['RegionalMultipath','Matched_RootMSF']+(['ConductanceGradient'] if FLOW else [])+(['UndiscountedPassage'] if PASSAGE else []);result={name:{'nodes':[],'edges':[],'backbone':[]} for name in method_names}
 if AGREEMENT:
  g=agreement_graph(g,units);items=supports(g,[unitof[s] for s in seeds])
 for j,item in enumerate(items):
  rid=f'R{j}';ids=item['nodes'];f=constructors(g,item);local=g[ids][:,ids];tree=minimum_spanning_tree(local);tree=tree.maximum(tree.T);tau,parent=dijkstra(tree,directed=False,indices=pd.Index(ids).get_loc(unitof[seeds[j]]),return_predecessors=True);parent=np.where(parent<0,-1,parent);labels,_=BRANCH.stable_hvg_kmeans(c,list(map(int,ids)),complexity_penalty=.1)
  constructors_to_test=[('RegionalMultipath',f['tau'],f['parent_local']),('Matched_RootMSF',tau,parent)]
  if FLOW:constructors_to_test.append(('ConductanceGradient',f['tau'],gradient_parents(g,f)[0]))
  if PASSAGE:
   alternative=passage_fit(g,item);constructors_to_test.append(('UndiscountedPassage',alternative['tau'],alternative['parent_local']))
  for method,order,parents in constructors_to_test:
   r=result[method]
   for k,node in enumerate(ids):r['nodes'].append({'root_id':rid,'node_id':int(node),'unit_id':str(unitnames[node]),'branch_id':rid+'_'+labels[int(node)],'pseudotime':float(order[k]),'is_root':int(node)==unitof[seeds[j]]})
   for k,p in enumerate(parents):
    if p<0:continue
    u,v=int(ids[p]),int(ids[k]);r['edges'].append({'root_id':rid,'source':u,'target':v,'source_name':str(unitnames[u]),'target_name':str(unitnames[v])})
    if labels[u]!=labels[v]:r['backbone'].append({'root_id':rid,'source_state':labels[u],'target_state':labels[v]})
 truth=aggregated_truth(a,unitnames,member,list(map(str,a.uns['roots'])),seeds,unitof);records=[]
 member.to_csv(dest/'EXACT_COORDINATE_ONLY_UNIT_MEMBERSHIP.csv',index=False);units.to_csv(dest/'EXACT_OBSERVED_UNIT_REPRESENTATIVES.csv',index=False)
 for method,r in result.items():
  r={k:pd.DataFrame(v,columns={'nodes':['root_id','node_id','unit_id','branch_id','pseudotime','is_root'],'edges':['root_id','source','target','source_name','target_name'],'backbone':['root_id','source_state','target_state']}[k]) for k,v in r.items()};folder=dest/method;folder.mkdir(exist_ok=True)
  for key,frame in r.items():frame.to_csv(folder/(key.upper()+'.csv'),index=False)
  scores=Q.t0_metrics(file.parent.name,method,r,truth);scores.to_csv(folder/'UNCHANGED_ORIGINAL_SCORER.csv',index=False);row=scores.loc[scores.root_id.eq('ALL')].iloc[0];records.append({'method':method,'n_raw_cells':n,'n_units':len(units),'fine_edge_F1':row.fine_edge_F1,'branch_ARI_rootwise':row.branch_ARI_rootwise,'pseudotime_spearman':row.pseudotime_spearman,'coverage':r['nodes'].unit_id.nunique()/len(units)})
 return records

def main():
 OUT.mkdir(parents=True,exist_ok=True);protected=ROOT/'algorithm/experimental/final_tree_method_development/run_phase_a_case_v1_1.py';before=hashlib.sha256(protected.read_bytes()).hexdigest();protocol={'scope':'ALL available existing development datasets reused, enumerated explicitly in the manifest; not new confirmatory samples. Matched-unit RootMSF comparison, not the historical1000cohort.','inputs':'Only counts, measured coordinates, one exact supplied source cell/root. No identity, stage, time, origin or branch truth used in fitting. Coordinate/source-pure pools<=9cells and300um; per-raw-cell Pearson PCA then unit mean. Source-cell minimum6 rule is inapplicable because exact synthetic roots supplied equally, not estimated regional anchors.','truth_adaptation':'AFTER fitting, average soft milestone percentages over exact member cells; same original.2support gate. Within-root true time mean and modal branch label (lexicographic tie). Mixed units explicitly retained, not truth-filtered. This is an aggregate-unit task, not cell-level accuracy.','fair_comparison':'Identical units/candidates/root/support/program labels, differing only constructor. Original scoring code unchanged. No original Fig3 benchmark numbers transferred.','release':'All outcomes reported; no automatic paper-final promotion.'};(OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2));records=[]
 protocol['constructors_tested']=['RegionalMultipath','Matched_RootMSF']+(['ConductanceGradient'] if FLOW else [])+(['UndiscountedPassage'] if PASSAGE else [])
 if AGREEMENT:protocol['geometry_change']='Same count-admissible edges. Replace bounded additive spatial term by product of molecular Gaussian and self-scaled spatial Cauchy evidence; physical bandwidth is existing30NN scale, no added fitted coefficient/radius.'
 if FLOW:protocol['gradient_rule']='Same discounted potential; choose positive conductance times rootward potential difference, without any new parameter. Independent parent-only ablation.'
 if PASSAGE:protocol['passage_rule']='Replace finite discounted hitting potential by undiscounted expected source-passage time, solving grounded Markov Bellman equations. No alpha search or added parameter; same minimum-cost strictly upstream extraction.'
 (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
 files=sorted(SOURCE.glob('*/EXACT_COUNTS_AND_TRUTH.h5ad'));pd.DataFrame({'case':[f.parent.name for f in files],'input':[str(f.relative_to(ROOT)) for f in files]}).to_csv(OUT/'ALL_REUSED_DEVELOPMENT_INPUTS.csv',index=False)
 for file in sorted(SOURCE.glob('*/EXACT_COUNTS_AND_TRUTH.h5ad')):
  dest=OUT/file.parent.name;dest.mkdir(exist_ok=True);checkpoint=dest/'SCORES.csv'
  if checkpoint.exists():local=pd.read_csv(checkpoint)
  else:local=pd.DataFrame(run_case(file,dest));local.to_csv(checkpoint,index=False)
  local['case']=file.parent.name;records.extend(local.to_dict('records'));print(file.parent.name,'matched-unit constructors scored',flush=True)
 a=pd.DataFrame(records);a.to_csv(OUT/'ALL_MATCHED_UNIT_SCORES.csv',index=False);paired=[]
 for metric in ['fine_edge_F1','branch_ARI_rootwise','pseudotime_spearman','coverage']:
  t=a.pivot(index='case',columns='method',values=metric)
  comparisons=[('RegionalMultipath','Matched_RootMSF')]+([('ConductanceGradient','Matched_RootMSF'),('ConductanceGradient','RegionalMultipath')] if FLOW else [])+([('UndiscountedPassage','Matched_RootMSF'),('UndiscountedPassage','RegionalMultipath')] if PASSAGE else [])
  for method,baseline in comparisons:
   delta=t[method]-t[baseline];paired.append({'method':method,'baseline':baseline,'metric':metric,'mean_gain':float(delta.mean()),'worst_case_gain':float(delta.min()),'n_datasets':len(delta)})
 p=pd.DataFrame(paired);p.to_csv(OUT/'ALL_PAIRED_DEVELOPMENT_OUTCOMES.csv',index=False);assert hashlib.sha256(protected.read_bytes()).hexdigest()==before;(OUT/'UNCHANGED_SCORER_SHA256.json').write_text(json.dumps({'sha256':before,'modified':False},indent=2));print(p.to_string(index=False))

if __name__=='__main__':main()
