"""Fresh-input reproduction and declared clinical-prior diagnostic, no refit search.

Never changes the candidate result. A successful check permits a versioned
state-organization freeze, not a publication/ancestry validation certificate.
"""
import os,sys,json,hashlib
from pathlib import Path
os.environ['NICHEFOREST_POSITIVE_ITERATION']='1'
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='2'
ROOT=Path(__file__).resolve().parents[2]
for p in ['algorithm/validation','algorithm/forest']:sys.path.insert(0,str(ROOT/p))
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.stats import spearmanr
from run_morph import BASE,CORE,OUT,geometry,supports,fit,MARKERS
from regional_source_ensembles import make_units
from stage_aware_ordering import fit_cohort
DEST=ROOT/'outputs/morph_reproduction_audit'

def main():
 DEST.mkdir(parents=True,exist_ok=True);src=OUT/'ensemble'
 raw=ad.read_h5ad(BASE/'data/BREAST_ORIGINAL_COUNTS.h5ad');obs=pd.read_csv(CORE/'EXACT_CORE_CELL_INPUT.csv',dtype={'cell_name':str});source=pd.read_csv(CORE/'EXACT_OBSERVED_REGIONAL_SOURCE_CELL_CONTRACT.csv',dtype={'original_cell_name':str})
 units,members,aggregate,mean,_=make_units(obs,source,'ensemble')
 savedunits=pd.read_csv(src/'EXACT_STATE_UNITS.csv',dtype={'representative_cell_name':str});savedmembers=pd.read_csv(src/'EVERY_OBSERVED_CELL_STATE_MEMBERSHIP.csv',dtype={'cell_name':str})
 pd.testing.assert_frame_equal(units,savedunits,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-10)
 pd.testing.assert_frame_equal(members,savedmembers,check_dtype=False)
 assert (aggregate!=sparse.load_npz(src/'EXACT_COUNT_AGGREGATION.npz')).nnz==0
 rows=obs.raw_row.to_numpy(int);nuc=sparse.load_npz(ROOT/'data/breast_nuclear/NUCLEAR_Q20_ORIGINAL_CELL_GENE_ORDER.npz')[rows];whole=sparse.csr_matrix(raw.X)[rows]
 prior=raw.var_names.get_indexer(sorted(set(sum(MARKERS.values(),[]))));genes=np.setdiff1d(np.arange(raw.n_vars),prior);records=[];diagnostics=[]
 for view,x in [('WholeCell',whole),('Nuclear',nuc)]:
  graph,features=geometry(x,obs,units,mean,genes);saved=sparse.load_npz(src/view/'FULL_CANDIDATE_COSTS.npz')
  assert np.array_equal(graph.indptr,saved.indptr) and np.array_equal(graph.indices,saved.indices)
  np.testing.assert_allclose(graph.data,saved.data,rtol=1e-11,atol=1e-12)
  np.testing.assert_allclose(features,np.load(src/view/'MEAN_NORMALIZED_RAW_CELL_PC_STATE_FEATURES.npy'),rtol=1e-10,atol=1e-10)
  assert (aggregate@x!=sparse.load_npz(src/f'EXACT_SUMMED_{view}_COUNTS.npz')).nnz==0
  items,_=supports(graph,units)
  for key,item in items.items():
   fitted=fit(graph,units,key,item);folder=src/view/f'{key[0]}_{key[1]}';n=pd.read_csv(folder/'FULL_STATE_NODES.csv');e=pd.read_csv(folder/'FULL_STATE_EDGES.csv');ids=fitted['nodes'];parents=fitted['parent_local'];free=np.flatnonzero(parents>=0);actual=np.column_stack([ids[parents[free]],ids[free]])
   np.testing.assert_array_equal(ids,n.node_id.to_numpy(int));np.testing.assert_array_equal(actual,e[['source','target']].to_numpy(int));np.testing.assert_allclose(fitted['tau'],n.pseudotime,rtol=1e-9,atol=1e-10);np.testing.assert_array_equal(np.sort(fitted['seeds']),np.sort(n.loc[n.is_source_state,'node_id']))
   records.append({'view':view,'root_id':key[0],'identity':key[1],'n_states':len(ids),'graph_rebuilt_from_raw_counts':True,'node_IDs_identical':True,'source_IDs_identical':True,'every_parent_identical':True,'maximum_order_difference':float(np.max(np.abs(fitted['tau']-n.pseudotime)))})
   # A fixed mechanism ablation: remove ONLY stage-direction admissibility.
   # Same current support/source and molecular/spatial candidates. No new
   # source, radius, discount or visual selection. Diagnostic, not adopted.
   if key==('R2','Malignant_epithelial'):
    neutral=units.copy();neutral.loc[neutral.identity.eq('Malignant_epithelial'),'identity']='Epithelial_stage_diagnostic';frame=neutral.iloc[ids].copy();frame['is_source_cell']=np.isin(ids,fitted['seeds']);unconstrained=fit_cohort(graph,neutral,frame);np.testing.assert_array_equal(unconstrained['nodes'],ids)
    p=unconstrained['parent_local'];ff=np.flatnonzero(p>=0);edges=np.column_stack([ids[p[ff]],ids[ff]]);stage=units.stage.to_numpy();length=np.linalg.norm(units[['spatial_x','spatial_y']].to_numpy()[edges[:,0]]-units[['spatial_x','spatial_y']].to_numpy()[edges[:,1]],axis=1)
    diagnostics.append({'view':view,'root_id':key[0],'identity':key[1],'n_same_states':len(ids),'n_same_sources':len(fitted['seeds']),'order_SR_stage_vs_no_stage':float(spearmanr(fitted['tau'],unconstrained['tau']).statistic),'same_parent_fraction':float(np.mean(parents[free]==p[free])),'no_stage_reverse_clinical_edges':int(np.sum(stage[edges[:,0]]>stage[edges[:,1]])),'no_stage_total_edges':len(edges),'no_stage_median_edge_um':float(np.median(length)),'scope':'Clinical-prior dependence diagnostic only, not independent developmental ground truth.'})
  print(view,'raw-count graph and',len(items),'forests exactly reproduced',flush=True)
 pd.DataFrame(records).to_csv(DEST/'FRESH_RAW_INPUT_REPRODUCTION.csv',index=False);pd.DataFrame(diagnostics).to_csv(DEST/'R2_CLINICAL_PRIOR_ABLATION.csv',index=False)
 verdict={'fresh_raw_count_reproduction_passed':True,'n_forests_reproduced':len(records),'same_units_sources_counts_candidates_and_every_parent':True,'clinical_prior_diagnostic':diagnostics,'freeze_scope':'Reproducible within-identity cellular-state organization; not experimentally validated single-cell ancestry.','publication_accepted':False,'no_parameters_selected_on_attractive_shape':True}
 (DEST/'READINESS.json').write_text(json.dumps(verdict,indent=2));print(json.dumps(verdict,indent=2))

if __name__=='__main__':main()
