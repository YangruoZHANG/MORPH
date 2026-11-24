"""Biological identity safety revision on exact same-section raw molecules.

No automatic identity reassignment: only published T/stromal identities with
compatible nuclear marker evidence enter corresponding within-identity trees.
Conflicted/insufficient cells remain in the roster/context. Regional source
eligibility changes are explicitly exported, never hidden as root invariance.
"""
import os,sys,json,hashlib
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[2]
for p in ['algorithm/forest','algorithm/validation']:sys.path.insert(0,str(ROOT/p))
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.stats import rankdata,spearmanr
from sklearn.preprocessing import StandardScaler
from multipath_ordering import candidate,support_sets,constructors
from epithelial_stage_constraints import stage_fit
from regional_state_forest import partial_ranks,bh
BASE=ROOT/'data/breast_reference'
NUCLEAR=ROOT/'data/breast_nuclear'
OUT=ROOT/'outputs/Nuclear_supported_identity_cores'
IDENTITY={'T':['CD3D','CD3E','CD3G','CD247','TRAC'],'Stromal':['LUM','PDGFRA','PDGFRB','POSTN']}
MULTIFAMILY=os.environ.get('NICHEFOREST_MULTIFAMILY_IDENTITY')=='1'
ALL_IDENTITIES=os.environ.get('NICHEFOREST_ALL_SUPPORTED_IDENTITIES')=='1'
MARKER_DISJOINT=os.environ.get('NICHEFOREST_DISJOINT_IDENTITY_FEATURES')=='1'
_IDENTITY_FEATURE_COLUMNS=None
if MULTIFAMILY:
 from nuclear_identity_markers import MARKERS
 IDENTITY=MARKERS
 OUT=ROOT/'outputs/Multifamily_supported_identity_cores'
IDENTITY_FAMILY={'CD4_T':'T','CD8_T':'T','Stromal':'Stromal'}
if ALL_IDENTITIES:
 if not MULTIFAMILY:raise ValueError('All-identity reconstruction requires all-family nuclear evidence.')
 IDENTITY_FAMILY.update({'B_cell':'B','Malignant_epithelial':'Epithelial','Myoepithelial':'Epithelial','Macrophage':'Myeloid','Dendritic':'Myeloid','Endothelial':'Endothelial','Perivascular':'Stromal'})
 OUT=ROOT/'outputs/All_lineage_nuclear_supported_cores'
if MARKER_DISJOINT:
 if not ALL_IDENTITIES:raise ValueError('The marker-disjoint hypothesis is registered for the full identity population.')
 OUT=ROOT/'data/breast_state_inputs'
PANELS={'STROMAL_ECM_NONIDENTITY':['PCOLCE','FBLN1','MMP2','MMP1'],'T_NAIVE_TRAFFICKING':['CCR7','IL7R','TCF7','SELL','CD27'],'T_CYTOTOXIC':['GZMA','GZMB','PRF1','NKG7','CCL5'],'T_CHECKPOINT_ASSOCIATED':['PDCD1','LAG3','TIGIT','HAVCR2','CTLA4']}

def configure_identity_feature_columns(gene_names):
 global _IDENTITY_FEATURE_COLUMNS
 names=pd.Index(gene_names);prior=set(sum(IDENTITY.values(),[]));_IDENTITY_FEATURE_COLUMNS=names.get_indexer(sorted(prior));assert (_IDENTITY_FEATURE_COLUMNS>=0).all()

def make_graph(x,obs,genes,identities=None):
 genes=np.asarray(genes,int)
 if MARKER_DISJOINT:
  if _IDENTITY_FEATURE_COLUMNS is None:raise RuntimeError('Identity-prior feature exclusion must be configured before fitting.')
  genes=np.setdiff1d(genes,_IDENTITY_FEATURE_COLUMNS)
 matrices=[]
 for identity in sorted(obs.identity.unique() if identities is None else identities):
  ix=np.flatnonzero(obs.identity.eq(identity));g,_=candidate(x[ix][:,genes].toarray(),obs.iloc[ix][['spatial_x','spatial_y']].to_numpy(),np.repeat(identity,len(ix)));coo=g.tocoo();matrices.append(sparse.csr_matrix((coo.data,(ix[coo.row],ix[coo.col])),shape=(len(obs),len(obs))))
 return sum(matrices,sparse.csr_matrix((len(obs),len(obs))))

def fit_identity(g,obs,key,item):
 return stage_fit(g,obs,item) if key[1]=='Malignant_epithelial' else constructors(g,item)

def test(fit,field,score,z,depth,area,xy,index,extra_nuisance=None):
 ids=fit['nodes'];columns=[depth[ids],area[ids],xy[ids]]
 if extra_nuisance is not None:columns.append(np.asarray(extra_nuisance)[ids])
 design=rankdata(np.column_stack(columns),axis=0);design=(design-design.mean(0))/np.maximum(design.std(0),1);design=np.column_stack([np.ones(len(ids)),design]);a,b,rho=partial_ranks(fit[field],score[ids],design)
 loo=[np.sign(partial_ranks(fit[field],np.delete(z[ids],k,axis=1).mean(1),design)[2])==np.sign(rho) for k in range(z.shape[1])];out={'n_cells':len(ids),'partial_rho':float(rho),'LOO_sign_agreement':float(np.mean(loo)),'effect_screened':bool(abs(rho)<.3)}
 for scale in [250,500,750]:
  _,block=np.unique(np.floor(xy[ids]/scale).astype(int),axis=0,return_inverse=True);sums=np.bincount(block,weights=a*b);p=1.
  if abs(rho)>=.3 and len(sums)>=12:
   rng=np.random.default_rng(202615200+index+scale);exceed=0
   for start in range(0,9999,1000):exceed+=int(np.sum(np.abs(rng.choice([-1,1],size=(min(1000,9999-start),len(sums)))@sums)>=abs(sums.sum())))
   p=(1+exceed)/10000
  out[f'p_{scale}']=p;out[f'n_blocks_{scale}']=len(sums)
 return out

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 parity=json.loads((NUCLEAR/'RECONSTRUCTION_PARITY.json').read_text());assert parity['n_count_entries_different']==0
 rootfile=ROOT/'data/选择的root/selected_roots_breast.csv';rootsha=hashlib.sha256(rootfile.read_bytes()).hexdigest();assert rootsha=='baba3838c27d42a392707c95f1dc7a4e4491a4af914ef1abc5db27efba9f7fc2'
 protocol={'role':'Identity-safety reconstruction and fresh function checks, not a paper-final release.','data':'Exact original whole-cell matrix plus official-source nuclear-overlapping molecule subset, every source count entry verified. No new patient and no counts imputed.','identity_evidence':IDENTITY,'eligibility':'Retain a published CD4/CD8 cell only if >=2 nuclear T markers and <2 nuclear stromal markers. Retain a published Stromal cell only for the converse. Both-positive and both-low cells are context/uncertain, not renamed, deleted from raw data or interpreted as cross-lineage intermediates. These are annotation-assisted evidence gates, not independently proven cell-type accuracy.','roots':'Original anchor coordinates unchanged. Existing radius150um/minimum6 regional source rule reapplied AFTER identity QC. This can change actual source cells; all such changes exported. It is not preservation of the old P6 single-root metacell or proof of ancestry.','observations':'Whole-cell versus nuclear count feature views are an explicit mechanism contrast, not selected by visual shape. Same candidate/count model, sources and graph parameters, no new spatial edge.','functional_panels':PANELS,'primary_evidence':'Nuclear marker activity of withheld NON-IDENTITY genes. Entire tested panel removed before depth normalization, PCA, graph, support selection and order. Eligibility marker genes disjoint from every tested functional panel. Published labels remain priors.','statistics':'Partial rank controlling nonpanel nuclear depth, original nucleus area and x/y;9999 spatial-block sign flips at250/500/750um,>=12blocks;absrho>=.3,LOOsign>=.8,BH jointly across ALL views/methods/roots/identities at each scale. Effects below.3 conservatively get p=1. No old q values transferred.','boundaries':'No T-to-stroma conversion, migration, causal tumor influence or experimental lineage proof. Root anchors organize state neighborhoods, not experimentally identified founders. Original scoring, root finding, Fig3 and manuscript artifacts unchanged.'}
 if MULTIFAMILY:
  protocol['identity_evidence']=IDENTITY
  protocol['eligibility']='Published CD4/CD8 cells require nuclear T marker support and NO other supported family. Published Stromal cells require nuclear stromal support and NO other supported family. Family support is>=2 distinct measured nuclear markers, uniformly for all6 declared families. No automatic reassignment of conflicting cells; all remain observed context. The additional families address observed residual identity conflicts, not trajectory score/length optimization.'
  protocol['primary_evidence']='Nuclear marker activity of withheld NON-IDENTITY genes; entire tested panel removed from inference. Tests overlapping ANY of the6 identity-family marker panels are not independent claims. All gene/source/cell changes explicit.'
 if ALL_IDENTITIES:
  protocol['eligible_published_identity_to_nuclear_family']=IDENTITY_FAMILY
  protocol['eligibility']='Each retained published identity requires>=2 distinct measured nuclear markers from its declared broad family and NO other supported family. No author label is reassigned. Rare/unsupported identities, multi-family conflicts and insufficient nuclei remain context. The same rule applies to every identity; no favorable trajectory chooses its mask.'
  protocol['malignant_stage_prior']='As in the previous model, DCIS/invasive author labels are explicit input constraints; select the lowest stage with>=6 observed eligible regional source cells. Directed seed-unreachable cells are exported as abstentions. Stage reversal avoidance is enforced, not a recovery-accuracy result.'
 if MARKER_DISJOINT:protocol['identity_function_separation']='Exclude ALL27 nuclear identity-prior genes from trajectory geometry in every full and held-panel fit. Identity eligibility and within-identity state order are explicitly separate; this is a fixed information-separation correction, not selection of a favorable subset or a visual parameter.'
 (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
 raw=ad.read_h5ad(BASE/'data/BREAST_ORIGINAL_COUNTS.h5ad');genes=pd.Index(raw.var_names);whole=sparse.csr_matrix(raw.X,dtype=np.int32);nuc=sparse.load_npz(NUCLEAR/'NUCLEAR_Q20_ORIGINAL_CELL_GENE_ORDER.npz');original=pd.read_csv(BASE/'data/MODEL_CELL_INPUT.csv',dtype={'cell_name':str});cells=pd.read_csv(BASE/'data/PUBLISHED_CELL_METADATA.csv',dtype={'cell_name':str});metadata=pd.read_csv(ROOT/'data/external_prior_knowledge/breast_original_study_reference/xe1_cells.csv.gz').set_index('cell_id');area_all=metadata.loc[cells.cell_name.astype(int)].nucleus_area.to_numpy();roots=pd.read_csv(BASE/'data/CANONICAL_SPATIAL_ANCHORS.csv')
 configure_identity_feature_columns(genes)
 detected={key:np.asarray((nuc[:,[genes.get_loc(g) for g in panel]]>0).sum(1)).ravel() for key,panel in IDENTITY.items()};t=detected['T']>=2;s=detected['Stromal']>=2;status=np.where(t&~s,'T_supported_nucleus',np.where(s&~t,'stromal_supported_nucleus',np.where(t&s,'mixed_identity_evidence','insufficient_nuclear_evidence')))
 if MULTIFAMILY:
  number=np.column_stack([value>=2 for value in detected.values()]).sum(1);t=t&(number==1);s=s&(number==1);status=np.where(t,'T_supported_nucleus',np.where(s,'stromal_supported_nucleus',np.where(number>1,'mixed_identity_evidence',np.where(number==1,'other_identity_family_supported','insufficient_nuclear_evidence'))))
 roster=cells.copy();roster['nuclear_T_marker_genes_detected']=detected['T'];roster['nuclear_stromal_marker_genes_detected']=detected['Stromal'];roster['nuclear_identity_evidence_status']=status;roster.to_csv(OUT/'ALL_ORIGINAL_CELLS_IDENTITY_EVIDENCE.csv',index=False)
 if ALL_IDENTITIES:
  supported=np.column_stack([value>=2 for value in detected.values()]);families=list(detected);only=np.argmax(supported,axis=1);number=supported.sum(1);status=np.array([families[int(k)]+'_supported_nucleus' if n==1 else 'mixed_identity_evidence' if n>1 else 'insufficient_nuclear_evidence' for k,n in zip(only,number)])
  roster['nuclear_identity_evidence_status']=status
  for family,value in detected.items():roster['n_'+family+'_nuclear_marker_genes']=value
  roster.to_csv(OUT/'ALL_ORIGINAL_CELLS_IDENTITY_EVIDENCE.csv',index=False)
 core=original.loc[original.identity.isin(IDENTITY_FAMILY)].copy();rr=core.raw_row.to_numpy(int)
 if ALL_IDENTITIES:eligible=np.array([(detected[IDENTITY_FAMILY[i]][int(r)]>=2) and number[int(r)]==1 for i,r in zip(core.identity,rr)])
 else:eligible=np.where(core.identity.eq('Stromal').to_numpy(),s[rr]&~t[rr],t[rr]&~s[rr])
 core['nuclear_core_eligible']=eligible;core['nuclear_identity_evidence_status']=status[rr];core.to_csv(OUT/'ORIGINAL_MODEL_CELLS_CORE_DISPOSITION.csv',index=False);obs=core.loc[eligible].copy().rename(columns={'node_id':'original_model_node_id'}).reset_index(drop=True);obs.index.name='nuclear_core_node_id';obs.to_csv(OUT/'EXACT_CORE_CELL_INPUT.csv');ix=obs.raw_row.to_numpy(int);counts={'WholeCell':whole[ix],'Nuclear':nuc[ix]};nx=nuc[ix];xy=obs[['spatial_x','spatial_y']].to_numpy();area=area_all[ix];oldnodes=pd.read_csv(BASE/'forest/FINAL_NODE_ASSIGNMENTS.csv');oldsource=oldnodes.loc[oldnodes.is_source_cell&oldnodes.identity.isin(IDENTITY_FAMILY)].copy();oldsource['retained_in_new_identity_core']=oldsource.node_id.isin(obs.original_model_node_id);oldsource['nuclear_identity_evidence_status']=status[original.iloc[oldsource.node_id.to_numpy(int)].raw_row.to_numpy(int)];oldsource.to_csv(OUT/'EVERY_OLD_REGIONAL_SOURCE_CELL_DISPOSITION.csv',index=False)
 all_source=[];fullfits={}
 for view,x in counts.items():
  dest=OUT/view;dest.mkdir(exist_ok=True);graphfile=dest/'FULL_CANDIDATE_COSTS.npz'
  if graphfile.exists():g=sparse.load_npz(graphfile);assert g.shape==(len(obs),len(obs))
  else:g=make_graph(x,obs,np.arange(raw.n_vars));sparse.save_npz(graphfile,g)
  supports,sourceaudit=support_sets(g,obs,roots);sourceaudit['count_view']=view;all_source.append(sourceaudit)
  for key,item in supports.items():
   fit=fit_identity(g,obs,key,item)
   if fit is None:continue
   fullfits[view,key]=fit;rid,identity=key;folder=dest/f'{rid}_{identity}';folder.mkdir(exist_ok=True);ids=fit['nodes'];frame=obs.iloc[ids].copy();frame['node_id']=ids;frame['pseudotime']=fit['tau'];frame['is_regional_source']=np.isin(ids,fit['seeds']);frame.to_csv(folder/'FULL_NODE_ASSIGNMENTS.csv',index=False);p=fit['parent_local'];free=np.flatnonzero(p>=0);pd.DataFrame({'source_core_node_id':ids[p[free]],'target_core_node_id':ids[free],'source_original_model_node_id':obs.original_model_node_id.to_numpy()[ids[p[free]]],'target_original_model_node_id':obs.original_model_node_id.to_numpy()[ids[free]]}).to_csv(folder/'FULL_DIRECTED_EDGES.csv',index=False)
   (folder/'OBSERVATION_AND_SOURCE_AUDIT.json').write_text(json.dumps({'n_cells':len(ids),'n_actual_sources':len(fit['seeds']),'actual_original_model_source_ids':obs.original_model_node_id.iloc[fit['seeds']].astype(int).tolist(),'n_directed_unreachable_abstentions':int(fit.get('unreachable_n',0)),'clinical_stage_prior':identity=='Malignant_epithelial','source_stage':int(fit['source_stage']) if 'source_stage' in fit else None,'source_excluded_higher_stage_cells':int(fit.get('source_excluded_higher_stage_cells',0)),'experimental_not_paper_final':True},indent=2))
  print(view,'identity-supported full reconstruction',len(obs),'core cells',len(supports),'regional forests',flush=True)
 pd.concat(all_source).to_csv(OUT/'ALL_REAPPLIED_REGIONAL_SOURCE_AUDITS.csv',index=False)
 tests=[];descriptive=[]
 for panel_index,(panel,names) in enumerate(PANELS.items()):
  target=np.array([genes.get_loc(g) for g in names]);train=np.setdiff1d(np.arange(raw.n_vars),target);depth=np.maximum(np.asarray(nx[:,train].sum(1)).ravel(),1);expression=np.log1p(nx[:,target].toarray()/depth[:,None]*np.median(depth));z=np.zeros_like(expression)
  for identity in obs.identity.unique():
   ii=np.flatnonzero(obs.identity.eq(identity));z[ii]=StandardScaler().fit_transform(expression[ii])
  score=z.mean(1)
  for view,x in counts.items():
   cache=OUT/f'HOLDOUT_{panel}_{view}.json'
   if cache.exists():
    saved=json.loads(cache.read_text());tests.extend(saved['tests']);descriptive.extend(saved['descriptive']);continue
   relevant=['Stromal'] if panel.startswith('STROMAL') else ['CD4_T','CD8_T'];g=make_graph(x,obs,train,identities=relevant);supports,sourceaudit=support_sets(g,obs,roots);local=[];de=[]
   for key,item in supports.items():
    rid,identity=key
    if (panel.startswith('STROMAL') and identity!='Stromal') or (panel.startswith('T_') and identity not in ['CD4_T','CD8_T']):continue
    fit=fit_identity(g,obs,key,item)
    if fit is None:continue
    if len(fit['nodes'])<20:continue
    for method,field in [('Multipath','tau'),('Matched_SPT','SPT_tau')]:
     result=test(fit,field,score,z,depth,area,xy,panel_index);result.update({'panel':panel,'count_view':view,'root_id':rid,'identity':identity,'method':method});local.append(result)
    full=fullfits.get((view,key))
    if full is not None:de.append({'panel':panel,'count_view':view,'root_id':rid,'identity':identity,'full_data_descriptive_SR':float(spearmanr(full['tau'],score[full['nodes']]).statistic)})
   cache.write_text(json.dumps({'tests':local,'descriptive':de,'withheld_genes':names},indent=2));tests.extend(local);descriptive.extend(de);pd.DataFrame(tests).to_csv(OUT/'ALL_FUNCTION_TESTS.partial.csv',index=False);print(view,panel,'all eligible roots held-out nuclear function checks complete',flush=True)
 result=pd.DataFrame(tests)
 for scale in [250,500,750]:result[f'q_{scale}']=bh(result[f'p_{scale}'].to_numpy())
 result['qualified']=result.partial_rho.abs().ge(.3)&result.LOO_sign_agreement.ge(.8)&result[[f'q_{scale}' for scale in [250,500,750]]].le(.1).all(axis=1);result=result.merge(pd.DataFrame(descriptive),on=['panel','count_view','root_id','identity'],how='left');result['full_sign_agrees']=np.sign(result.partial_rho)==np.sign(result.full_data_descriptive_SR);result['usable_for_state_naming']=result.qualified&result.full_sign_agrees;result.to_csv(OUT/'ALL_NUCLEAR_FUNCTION_TESTS.csv',index=False)
 assert hashlib.sha256(rootfile.read_bytes()).hexdigest()==rootsha
 (OUT/'DECISION.json').write_text(json.dumps({'whole_constructor_publication_accepted':False,'original_root_input_sha256':rootsha,'original_raw_cell_count':raw.n_obs,'n_eligible_core_cells':len(obs),'n_nuclear_function_tests':len(result),'n_qualified_with_full_sign_agreement':int(result.usable_for_state_naming.sum()),'annotation_safety_only_not_lineage_validation':True,'regional_source_membership_changes_explicitly_exported':True,'figures_pptx_latex_and_fig3_modified':False},indent=2));print(result.loc[result.usable_for_state_naming,['count_view','root_id','identity','panel','method','partial_rho']].to_string(index=False),flush=True)

if __name__=='__main__':main()
