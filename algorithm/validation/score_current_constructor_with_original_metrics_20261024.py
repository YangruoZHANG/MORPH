"""Fresh supplied-root simulations scored by unchanged original scoring code.

The crossed program-label ablation exposes which module contributes branch
ARI. It must not be hidden or attributed to edge rewiring alone.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,importlib.util,hashlib
from types import SimpleNamespace
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.sparse.csgraph import dijkstra,connected_components,minimum_spanning_tree
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'algorithm/experimental'));sys.path.insert(0,str(ROOT/'algorithm/experimental/final_tree_method_development'))
from cell_resolved_multipath_v14_20261023 import candidate,constructors
from run_paper_baseline_adapter_smoke import roots_from_truth
import run_phase_a_case_v1_1 as Q
import branch_program_estimator_v9 as BRANCH
OUT=ROOT/'outputs/Current_constructor_original_metrics_20261024'
PROTOCOL={
 'scope':'New24 dataset supplied-root development confirmation, NOT a relabeling of original1000/P4/P6 results and not a root-finding benchmark.',
 'data':'12 frozen simulator conditions x2 new seeds202610470..471, 300genes35cells/edge; no origin/branch/time/stage labels used in fitting. True initial cell only supplied equally to methods.',
 'geometry':'Current count-aware per-identity model with identity unknown for every simulated cell; same reciprocal30 candidates for all methods. Clinical-stage adapter disabled when no such prior exists.',
 'membership':'Same graph-distance soft memberships>=.2, source-connected support. One supplied true seed per root, no uncertain-region expansion on these exact-root synthetic inputs.',
 'constructors':'Alpha.97 multipath forest vs matched rootMSF, identical candidates/nodes/seeds.',
 'branch_labels':'Original unmodified v9 stable-HVG KMeans complexity0.10 program partitions versus original unmodified topology_branch_labels. Cross these labels over both constructors to isolate the effect of grouping from edge reconstruction.',
 'scoring':'Call original run_phase_a_case_v1_1.t0_metrics unchanged with truth state supports>=.20 and probabilities. Equal-root ALL aggregation as original. Original scoring files hashed before/after.',
 'metrics':['coverage','branch_ARI_rootwise','fine_edge_F1','pseudotime_spearman'],
 'gate':'Report all paired outcomes and module ablations. Do not promote paper-ready merely because ordering or a baseline-dependent branch metric improves. This smaller new cohort cannot inherit the sample size or statistics of the old1000 experiment.'}

def supports(matrix,seeds):
 dist=np.array([dijkstra(matrix,directed=False,indices=int(s)) for s in seeds]);nearest=dist.min(0);valid=np.isfinite(nearest)
 pos=nearest[valid&(nearest>0)];scale=np.median(pos) if len(pos) else 1
 w=np.zeros(dist.shape);w[:,valid]=np.exp(-(dist[:,valid]-nearest[valid])/max(scale,1e-8));probs=w/np.maximum(w.sum(0),1e-12)
 fits=[]
 for j,s in enumerate(seeds):
  selected=np.union1d(np.flatnonzero((probs[j]>=.2)&np.isfinite(dist[j])),[s]);local=matrix[selected][:,selected]
  nc,components=connected_components(local,directed=False);rootpos=pd.Index(selected).get_loc(s);selected=selected[components==components[rootpos]]
  fits.append({'nodes':selected,'seeds':np.array([s])})
 return fits

def truth_context(a,rootnames,seeds):
 allgraph=nx.DiGraph()
 mn=a.uns['milestone_network']
 for r in mn.itertuples(index=False):allgraph.add_edge(str(r[0]),str(r[1]),length=float(r[2]))
 graphs={};branch={};ptime={};state_probability={};state_support={};anchors={}
 mp=a.uns['milestone_percentages'];names=a.obs_names.astype(str)
 probabilities={(str(r.cell_id),str(r.milestone_id)):float(r.percentage) for r in mp.itertuples()}
 for j,root in enumerate(rootnames):
  rid=f'R{j}';ids={root}|nx.descendants(allgraph,root);graphs[rid]=allgraph.subgraph(ids).copy();anchors[rid]=str(names[seeds[j]])
  for i,name in enumerate(names):
   active={s for s in ids if probabilities.get((name,s),0)>=.2}
   if active:state_support[name,rid]=active
   for s in ids:
    p=probabilities.get((name,s),0)
    if p>0:state_probability[name,rid,s]=p
   if str(a.obs.origin.iloc[i])==str(j):branch[name,rid]=str(a.obs.branch.iloc[i]);ptime[name,rid]=float(a.uns['timecourse'][i])
 return SimpleNamespace(roots=list(graphs),graphs=graphs,anchors=anchors,branch=branch,pseudotime=ptime,
                       state_support=state_support,state_probability=state_probability,state_support_threshold=.2)

def fit_results(a):
 c=a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
 matrix,pc=candidate(c,a.obsm['spatial'],np.repeat('unknown',a.n_obs))
 seeds=[int(a.obs_names.get_loc(s)) for s in roots_from_truth(a)];ss=supports(matrix,seeds)
 results={name:{'nodes':[],'edges':[],'backbone':[]} for name in ['Multipath_program','RootMSF_topology','Multipath_topology','RootMSF_program']}
 names=a.obs_names.astype(str)
 for j,item in enumerate(ss):
  rid=f'R{j}';ids=item['nodes'];hit=constructors(matrix,item);sub=matrix[ids][:,ids];mst=minimum_spanning_tree(sub).maximum(minimum_spanning_tree(sub).T)
  localroot=pd.Index(ids).get_loc(seeds[j]);distance,parent=dijkstra(mst,directed=False,indices=localroot,return_predecessors=True);parent=np.where(parent<0,-1,parent)
  program,audit=BRANCH.stable_hvg_kmeans(c,list(map(int,ids)),complexity_penalty=.1)
  for constructor,tau,parents in [('Multipath',hit['tau'],hit['parent_local']),('RootMSF',distance,parent)]:
   graph=nx.Graph();graph.add_nodes_from(map(int,ids));graph.add_edges_from((int(ids[p]),int(ids[k])) for k,p in enumerate(parents) if p>=0)
   topo=Q.P.topology_branch_labels(graph,int(seeds[j]));branches={'program':program,'topology':topo}
   for grouping,labels in branches.items():
    name=constructor+'_'+grouping;result=results[name];t=dict(zip(map(int,ids),map(float,tau)))
    for node in ids:result['nodes'].append({'root_id':rid,'node_id':int(node),'unit_id':str(names[node]),'branch_id':rid+'_'+labels[int(node)],'pseudotime':t[int(node)],'is_root':int(node)==seeds[j]})
    for u,v in graph.edges():result['edges'].append({'root_id':rid,'source':u,'target':v,'source_name':str(names[u]),'target_name':str(names[v])})
    for u,v in graph.edges():
     if labels[u]==labels[v]:continue
     if (t[u],u)>(t[v],v):u,v=v,u
     result['backbone'].append({'root_id':rid,'source_state':labels[u],'target_state':labels[v]})
 for result in results.values():
  for key in result:result[key]=pd.DataFrame(result[key],columns={'nodes':['root_id','node_id','unit_id','branch_id','pseudotime','is_root'],'edges':['root_id','source','target','source_name','target_name'],'backbone':['root_id','source_state','target_state']}[key])
 return results,seeds

def run():
 OUT.mkdir(exist_ok=True,parents=True);pp=OUT/'PROTOCOL.json'
 if pp.exists():assert json.loads(pp.read_text())==PROTOCOL
 else:pp.write_text(json.dumps(PROTOCOL,indent=2))
 original=ROOT/'algorithm/experimental/final_tree_method_development/run_phase_a_case_v1_1.py';before=hashlib.sha256(original.read_bytes()).hexdigest()
 path=ROOT/'outputs/Fig3_teammate_reproduction_20261015/NicheForest_Fig3_Reproduction/simulate/make_multiorigin_spatial.py'
 spec=importlib.util.spec_from_file_location('frozen_sim',path);sim=importlib.util.module_from_spec(spec);spec.loader.exec_module(sim)
 rows=[]
 for condition,cfg in sim.grid_conditions():
  for seed in [202610470,202610471]:
   dest=OUT/f'{condition}_{seed}';dest.mkdir(exist_ok=True)
   if (dest/'SCORES.csv').exists():rows.extend(pd.read_csv(dest/'SCORES.csv').to_dict('records'));print('resume',condition,seed,flush=True);continue
   a=sim.simulate(name=condition,condition=condition,seed=seed,cells_per_edge=35,**cfg);a.write_h5ad(dest/'EXACT_SIMULATED_COUNTS_AND_TRUTH.h5ad')
   results,seeds=fit_results(a)
   # Truth is handed only to the unmodified scorer after fitting has finished.
   truth=truth_context(a,list(map(str,a.uns['roots'])),seeds)
   local=[]
   for method,result in results.items():
    folder=dest/method;folder.mkdir(exist_ok=True)
    for name,frame in result.items():frame.to_csv(folder/(name.upper()+'.csv'),index=False)
    metrics=Q.t0_metrics(condition,method,result,truth);metrics.to_csv(folder/'ORIGINAL_SCORER_ALL_OUTPUTS.csv',index=False)
    row=metrics.loc[metrics.root_id.eq('ALL')].iloc[0]
    local.append({'condition':condition,'seed':seed,'method':method,'coverage':result['nodes'].unit_id.nunique()/a.n_obs,
        'branch_ARI_rootwise':row.branch_ARI_rootwise,'fine_edge_F1':row.fine_edge_F1,'pseudotime_spearman':row.pseudotime_spearman,'is_forest':row.is_forest})
   pd.DataFrame(local).to_csv(dest/'SCORES.csv',index=False);rows.extend(local);print('scored fresh cohort',condition,seed,flush=True)
 result=pd.DataFrame(rows);result.to_csv(OUT/'ALL_METRICS.csv',index=False)
 assert hashlib.sha256(original.read_bytes()).hexdigest()==before
 (OUT/'ORIGINAL_SCORER_INTEGRITY.json').write_text(json.dumps({'path':str(original),'sha256_before':before,'sha256_after':before,'modified':False},indent=2))
 result.groupby('method')[PROTOCOL['metrics']].mean().to_csv(OUT/'METHOD_MEANS.csv')
 print(result.groupby('method')[PROTOCOL['metrics']].mean().to_string(),flush=True)

if __name__=='__main__':run()
