"""Gene-disjoint local-state/source-ensemble correction and held-panel tests."""
import os,sys,json,argparse,hashlib
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[2]
for p in ['algorithm/experimental','algorithm/validation']:sys.path.insert(0,str(ROOT/p))
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from regional_source_ensemble_units_v63_20261025 import make_units,geometry as original_geometry,supports
from count_spatial_agreement_kernel_v70_20261025 import agreement_graph
from cell_resolved_multipath_v14_20261023 import constructors
from improve_trajectory_continuity_v20_20261024 import fit_cohort
from rebuild_nuclear_supported_cores_v51_20261024 import test,PANELS
from audit_multifamily_nuclear_identity_v54_20261024 import MARKERS
from validate_niche_programs_v16_20261024 import EXTRA
from root_region_state_forest_v5_20261023 import bh
BASE=ROOT/'outputs/NicheForest_current_research_freeze_20261024';CORE=ROOT/'outputs/Marker_disjoint_lineage_state_forests_v59_20261025';OUT=ROOT/'outputs/Regional_source_ensemble_states_v63_20261025'
AGREEMENT=os.environ.get('NICHEFOREST_COUNT_SPATIAL_AGREEMENT')=='1'
if AGREEMENT:OUT=ROOT/'outputs/Count_spatial_agreement_states_v70_20261025'

def geometry(raw_counts,raw_obs,units,mean,genes):
 g,features=original_geometry(raw_counts,raw_obs,units,mean,genes)
 return (agreement_graph(g,units) if AGREEMENT else g),features

def fit(g,units,key,item):
 if key[1]=='Malignant_epithelial':
  frame=units.iloc[item['nodes']].copy();frame['node_id']=item['nodes'];frame['is_source_cell']=np.isin(item['nodes'],item['seeds']);return fit_cohort(g,units,frame)
 return constructors(g,item)

def panels(genes):
 prior=set(sum(MARKERS.values(),[]));records=[];used=set()
 for name,original in list(PANELS.items())+list(EXTRA.items()):
  original=sorted(set(original)&set(genes));score=sorted(set(original)-prior);signature=frozenset(score);eligible=len(score)>=3 and signature not in used
  if eligible:used.add(signature)
  records.append({'name':name,'entire_withheld_panel':';'.join(original),'nonidentity_score_genes':';'.join(score),'n_score_genes':len(score),'eligible':eligible,'reason':'fewer than3nonidentity genes' if len(score)<3 else 'duplicate gene set' if not eligible else ''})
 return pd.DataFrame(records)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['ensemble','pool_only','source_only'],default='ensemble');args=parser.parse_args();dest=OUT/args.mode;dest.mkdir(parents=True,exist_ok=True)
 protocol={'hypothesis':'The regional anchor defines an initial state estimate, not many diverse cells all experimentally known to be time0. Preserve every actual source-cell ID but aggregate its normalized source state into ONE observed regional ensemble per root/identity. Independently, reduce count noise through small identity/stage-pure spatial states.','mode':args.mode,'raw_population':'Same99224nuclear-supported, non-conflicted original cells as the marker-disjoint core; original author labels are priors. No new cells, counts, positions or source cells.','local_pools':'Coordinate-only widest-axis median partition, <=original median metacell membership (9), maximum diameter300um matching the pre-existing radius150um source-niche diameter. No minimum pool mass or forced bridge. Source ensembles contain all original same-root/same-identity/same-stage source cells, even if >9. No genes select memberships.','observation_model':'Fit unchanged per-RAW-cell theta100Pearson residual/PCA15 within identity; exclude ALL27identity genes and every tested entire gene panel before any feature fitting. Mean normalized raw-cell PCs inside units; NEVER apply a single-cell NB model to variable-mass summed counts. Exact count sums are separately preserved.','graph':'Same reciprocal30NN/self-tuned molecular cost + original bounded0.1*d/(d+350), same.97multipath solver. No spatial edge cap or path-length objective. Source eligibility remains>=6 RAW cells, not6pooled units.','coordinate':'Each display coordinate is an observed member nearest its coordinate mean; no artificial cell or displaced raw cell. Original regional anchor positions unchanged.','response':'Mean per-raw-cell nuclear programme scores within a unit, not repeated cell-level observations masquerading as independent replicates. Partial rank includes mean nonpanel nuclear depth, mean nucleus area, raw-cell mass and measured representative x/y;9999block flips at250/500/750um, joint family BH.','clinical_prior':'Original DCIS/invasive input admissibility retained. Earliest eligible stage source roster was already frozen on raw cells before pooling; do not re-require6pooled units or silently reselect source stage.','scope':'A region-conditioned state forest, not proof of literal single-cell ancestry or migration. All prior raw source IDs preserved as explicit members. No automatic paper-final flag or main artifact overwrite.'};(dest/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
 if AGREEMENT:
  protocol['graph']='Identical count-admissible reciprocal30NN edge set. Gaussian molecular evidence times Cauchy physical evidence, with physical30NN self-scales. Cost=sqrt(molecular_cost**2+log1p(d**2/(scale_i*scale_j))). Same.97multipath solver, source IDs and minimum raw source mass6; no new candidate, spatial radius or length target.'
  (dest/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
 raw=ad.read_h5ad(BASE/'data/BREAST_ORIGINAL_COUNTS.h5ad');genes=pd.Index(raw.var_names);raw_obs=pd.read_csv(CORE/'EXACT_CORE_CELL_INPUT.csv',dtype={'cell_name':str});source=pd.read_csv(CORE/'EXACT_OBSERVED_REGIONAL_SOURCE_CELL_CONTRACT.csv',dtype={'original_cell_name':str});units,member,aggregate,mean,maximum=make_units(raw_obs,source,args.mode);units.to_csv(dest/'EXACT_STATE_UNITS.csv',index=False);member.to_csv(dest/'EVERY_OBSERVED_CELL_STATE_MEMBERSHIP.csv',index=False);sparse.save_npz(dest/'EXACT_COUNT_AGGREGATION.npz',aggregate);sparse.save_npz(dest/'MEAN_STATE_OPERATOR.npz',mean);rawrows=raw_obs.raw_row.to_numpy(int);whole=sparse.csr_matrix(raw.X,dtype=np.int64)[rawrows];nuc=sparse.load_npz(ROOT/'outputs/Original_nuclear_molecule_audit_v46_20261024/NUCLEAR_Q20_ORIGINAL_CELL_GENE_ORDER.npz')[rawrows];pooled={'WholeCell':aggregate@whole,'Nuclear':aggregate@nuc};views={'WholeCell':whole,'Nuclear':nuc}
 for view,matrix in pooled.items():
  np.testing.assert_array_equal(np.asarray(matrix.sum(0)),np.asarray(views[view].sum(0)));sparse.save_npz(dest/f'EXACT_SUMMED_{view}_COUNTS.npz',matrix)
 prior=genes.get_indexer(sorted(set(sum(MARKERS.values(),[]))));retained=np.setdiff1d(np.arange(raw.n_vars),prior);full={};fit_audits=[]
 for view,x in views.items():
  folder=dest/view;folder.mkdir(exist_ok=True);graphfile=folder/'FULL_CANDIDATE_COSTS.npz'
  if graphfile.exists():g=sparse.load_npz(graphfile)
  else:g,features=geometry(x,raw_obs,units,mean,retained);sparse.save_npz(graphfile,g);np.save(folder/'MEAN_NORMALIZED_RAW_CELL_PC_STATE_FEATURES.npy',features)
  items,sourceaudit=supports(g,units);sourceaudit.to_csv(folder/'RAW_MASS_SOURCE_AUDIT.csv',index=False)
  for key,item in items.items():
   f=fit(g,units,key,item);full[view,key]=f;rid,identity=key;where=folder/f'{rid}_{identity}';where.mkdir(exist_ok=True);ids=f['nodes'];frame=units.iloc[ids].copy();frame['pseudotime']=f['tau'];frame['is_source_state']=np.isin(ids,f['seeds']);frame.to_csv(where/'FULL_STATE_NODES.csv',index=False);p=f['parent_local'];free=np.flatnonzero(p>=0);pd.DataFrame({'source':ids[p[free]],'target':ids[free]}).to_csv(where/'FULL_STATE_EDGES.csv',index=False);fit_audits.append({'count_view':view,'root_id':rid,'identity':identity,'n_units':len(ids),'n_observed_raw_cells':int(units.n_raw_cells.iloc[ids].sum()),'n_root_state_units':len(f['seeds']),'n_source_raw_cells':int(units.n_raw_cells.iloc[f['seeds']].sum())})
  print(args.mode,view,'full state reconstruction',len(units),'units',flush=True)
 pd.DataFrame(fit_audits).to_csv(dest/'FULL_RECONSTRUCTION_AUDIT.csv',index=False);cat=panels(genes);cat.to_csv(dest/'LOCKED_MECHANISM_AND_REMAINDER_PANEL_CATALOG.csv',index=False);cells=pd.read_csv(BASE/'data/PUBLISHED_CELL_METADATA.csv',dtype={'cell_name':str});metadata=pd.read_csv(ROOT/'data/external_prior_knowledge/breast_original_study_reference/xe1_cells.csv.gz').set_index('cell_id');area_raw=metadata.loc[cells.iloc[rawrows].cell_name.astype(int)].nucleus_area.to_numpy();area=mean@np.log1p(area_raw);xy=units[['spatial_x','spatial_y']].to_numpy();records=[]
 for index,panel in cat.loc[cat.eligible].iterrows():
  target=genes.get_indexer(panel.nonidentity_score_genes.split(';'));held=genes.get_indexer(panel.entire_withheld_panel.split(';'));train=np.setdiff1d(retained,held);raw_depth=np.maximum(np.asarray(nuc[:,np.setdiff1d(np.arange(raw.n_vars),held)].sum(1)).ravel(),1);value=np.log1p(nuc[:,target].toarray()/raw_depth[:,None]*np.median(raw_depth));z=np.zeros_like(value)
  for identity in raw_obs.identity.unique():
   ids=np.flatnonzero(raw_obs.identity.eq(identity));z[ids]=StandardScaler().fit_transform(value[ids])
  unit_z=np.asarray(mean@z);score=unit_z.mean(1);depth=np.asarray(mean@np.log1p(raw_depth)).ravel()
  for view,x in views.items():
   checkpoint=dest/f'HOLDOUT_{index:02d}_{view}.json'
   if checkpoint.exists():records.extend(json.loads(checkpoint.read_text())['tests']);continue
   g,_=geometry(x,raw_obs,units,mean,train);items,_=supports(g,units);rows=[]
   for key,item in items.items():
    f=fit(g,units,key,item)
    if len(f['nodes'])<20:continue
    rid,identity=key;original=full[view,key];np.testing.assert_array_equal(np.sort(f['seeds']),np.sort(original['seeds']));description=float(spearmanr(original['tau'],score[original['nodes']]).statistic)
    for method,field in [('Multipath','tau'),('Matched_SPT','SPT_tau')]:
     r=test(f,field,score,unit_z,depth,area,xy,300+int(index),extra_nuisance=np.log1p(units.n_raw_cells.to_numpy()));r.update({'panel':panel['name'],'count_view':view,'root_id':rid,'identity':identity,'method':method,'full_data_descriptive_SR':description});rows.append(r)
   checkpoint.write_text(json.dumps({'entire_panel_withheld':panel.entire_withheld_panel.split(';'),'nonidentity_response_genes':panel.nonidentity_score_genes.split(';'),'tests':rows},indent=2));records.extend(rows);print(args.mode,view,panel['name'],'held-panel checks complete',flush=True)
 a=pd.DataFrame(records)
 for scale in [250,500,750]:a[f'q_{scale}']=bh(a[f'p_{scale}'].to_numpy())
 a['qualified']=a.partial_rho.abs().ge(.3)&a.LOO_sign_agreement.ge(.8)&a[[f'q_{s}' for s in [250,500,750]]].le(.1).all(axis=1);a['full_sign_agrees']=np.sign(a.partial_rho)==np.sign(a.full_data_descriptive_SR);a['usable_for_conditional_state_naming']=a.qualified&a.full_sign_agrees;a.to_csv(dest/'ALL_UNIT_LEVEL_FUNCTION_TESTS.csv',index=False)
 invariant={'n_raw_observed_cells':len(raw_obs),'n_state_units':len(units),'default_non_source_maximum_raw_cells':maximum,'all_whole_and_nuclear_gene_count_totals_exactly_preserved':True,'every_observed_cell_assigned_exactly_once':True,'every_original_raw_source_cell_ID_preserved':True,'representatives_are_measured_members_not_virtual_cells':True,'identity_and_stage_pure_units':True,'original_root_anchor_input_sha256':hashlib.sha256((ROOT/'data/选择的root/selected_roots_breast.csv').read_bytes()).hexdigest(),'source_ensemble_not_individual_cell_founder':True,'whole_constructor_publication_accepted':False,'main_figures_pptx_latex_and_fig3_modified':False};(dest/'AGGREGATION_SOURCE_AND_COORDINATE_INVARIANTS.json').write_text(json.dumps(invariant,indent=2));print(a.loc[a.usable_for_conditional_state_naming,['count_view','root_id','identity','panel','method','partial_rho']].to_string(index=False),flush=True)

if __name__=='__main__':main()
