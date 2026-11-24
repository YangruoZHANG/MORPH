"""Three technical molecule-thinning repeats, not biological replication.

Units, source-cell IDs, author identity priors and identity marker exclusions are
frozen before repeats. Count thinning changes inference features only. Record
all forests, support losses, order and exact parent agreement without selecting
the most stable root/identity or using trajectory appearance.
"""
import os,sys,json,argparse
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
ROOT=Path(__file__).resolve().parents[2]
for p in ['algorithm/experimental','algorithm/validation']:sys.path.insert(0,str(ROOT/p))
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.stats import spearmanr
from rebuild_regional_source_ensembles_v63_20261025 import BASE,CORE,OUT,geometry,supports,fit,MARKERS
from rooted_conductance_gradient_forest_v67_20261025 import gradient_parents
FLOW=os.environ.get('NICHEFOREST_CONDUCTANCE_GRADIENT')=='1'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['ensemble','pool_only','source_only'],default='ensemble');args=ap.parse_args();dest=OUT/args.mode/('gradient_technical_thinning' if FLOW else 'technical_thinning');dest.mkdir(parents=True,exist_ok=True)
 raw=ad.read_h5ad(BASE/'data/BREAST_ORIGINAL_COUNTS.h5ad');genes=pd.Index(raw.var_names);obs=pd.read_csv(CORE/'EXACT_CORE_CELL_INPUT.csv',dtype={'cell_name':str});units=pd.read_csv(OUT/args.mode/'EXACT_STATE_UNITS.csv');mean=sparse.load_npz(OUT/args.mode/'MEAN_STATE_OPERATOR.npz');prior=genes.get_indexer(sorted(set(sum(MARKERS.values(),[]))));train=np.setdiff1d(np.arange(raw.n_vars),prior);whole=sparse.csr_matrix(raw.X,dtype=np.int64)[obs.raw_row.to_numpy(int)];nuc=sparse.load_npz(ROOT/'outputs/Original_nuclear_molecule_audit_v46_20261024/NUCLEAR_Q20_ORIGINAL_CELL_GENE_ORDER.npz')[obs.raw_row.to_numpy(int)];records=[]
 for view,x in [('WholeCell',whole),('Nuclear',nuc)]:
  base={}
  for file in (OUT/args.mode/view).glob('*/FULL_STATE_NODES.csv'):
   key=tuple(file.parent.name.split('_',1));a=pd.read_csv(file).set_index('node_id');e=pd.read_csv(file.parent/'FULL_STATE_EDGES.csv');base[key]=(a,e.set_index('target').source)
  if FLOW:
   fullgraph=sparse.load_npz(OUT/args.mode/view/'FULL_CANDIDATE_COSTS.npz')
   for key,(a,_) in list(base.items()):
    ref={'nodes':a.index.to_numpy(int),'tau':a.pseudotime.to_numpy(float),'seeds':a.index[a.is_source_state].to_numpy(int)};p,_=gradient_parents(fullgraph,ref,stage=units.stage.to_numpy());free=np.flatnonzero(p>=0);base[key]=(a,pd.Series(ref['nodes'][p[free]],index=ref['nodes'][free]))
  for seed in [202615801,202615802,202615803]:
   checkpoint=dest/f'{view}_{seed}.json'
   if checkpoint.exists():records.extend(json.loads(checkpoint.read_text())['records']);continue
   thin=x.copy();thin.data=np.random.default_rng(seed).binomial(thin.data,.8);thin.eliminate_zeros();g,_=geometry(thin,obs,units,mean,train);items,_=supports(g,units);local=[]
   for key,(original,parent) in base.items():
    if key not in items:raise AssertionError(f'Frozen source became ineligible: {key}')
    f=fit(g,units,key,items[key]);np.testing.assert_array_equal(np.sort(f['seeds']),np.sort(original.index[original.is_source_state]));tau=pd.Series(f['tau'],index=f['nodes']);common=np.intersect1d(tau.index,original.index);p=gradient_parents(g,f,stage=units.stage.to_numpy())[0] if FLOW else f['parent_local'];child=np.flatnonzero(p>=0);pred=pd.Series(f['nodes'][p[child]],index=f['nodes'][child]);targets=np.intersect1d(pred.index,parent.index)
    local.append({'mode':args.mode,'count_view':view,'seed':seed,'root_id':key[0],'identity':key[1],'n_full_units':len(original),'n_thinned_units':len(tau),'support_jaccard':len(common)/len(np.union1d(tau.index,original.index)),'order_spearman_on_common_units':float(spearmanr(tau.loc[common],original.loc[common,'pseudotime']).statistic),'exact_parent_agreement_on_common_targets':float(np.mean(pred.loc[targets].to_numpy()==parent.loc[targets].to_numpy())),'all_exact_source_unit_IDs_preserved':True})
   checkpoint.write_text(json.dumps({'technical_not_biological_repeat':True,'retained_molecule_probability':.8,'records':local},indent=2));records.extend(local);print(args.mode,view,seed,'complete',flush=True)
 a=pd.DataFrame(records);a.to_csv(dest/'EVERY_FOREST_THREE_TECHNICAL_REPEATS.csv',index=False);print(a.groupby('count_view')[['support_jaccard','order_spearman_on_common_units','exact_parent_agreement_on_common_targets']].agg(['mean','min']).to_string())

if __name__=='__main__':main()
