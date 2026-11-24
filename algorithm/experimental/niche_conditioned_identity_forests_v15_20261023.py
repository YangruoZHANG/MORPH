"""All-identity sensitivity to the interpretation of fixed spatial anchors.

This is an EXPLICIT alternative regional-anchor model. It does not relabel
the original T-like root node as an epithelial progenitor. Every lineage gate
has its own seed cells and graph; no T-cell-to-stroma edge can be created.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from cell_resolved_multipath_v14_20261023 import candidate,support_sets,constructors,ROOT
OUT=ROOT/'outputs/Niche_conditioned_identity_forests_v15_20261023'
IDENTITIES={
 'Invasive_Tumor':'Malignant_epithelial','Prolif_Invasive_Tumor':'Malignant_epithelial','DCIS_1':'Malignant_epithelial','DCIS_2':'Malignant_epithelial',
 'CD4+_T_Cells':'CD4_T','CD8+_T_Cells':'CD8_T','B_Cells':'B_cell',
 'Macrophages_1':'Macrophage','Macrophages_2':'Macrophage','IRF7+_DCs':'Dendritic','LAMP3+_DCs':'Dendritic',
 'Endothelial':'Endothelial','Stromal':'Stromal','Myoepi_ACTA2+':'Myoepithelial','Myoepi_KRT15+':'Myoepithelial',
 'Perivascular-Like':'Perivascular','Mast_Cells':'Mast'}
PROTOCOL={
 'status':'Alternative spatial-niche-anchor formulation under audit, not a validated cell-ancestry model or an automatic replacement of original root semantics',
 'anchor':'Exact supplied spatial coordinates; upstream root finding untouched. Root points identify regions. Separate observed >=6 cells of each identity inside150um supply its conditional state prior. Other identities do NOT descend from the original T-like root cell.',
 'identity_map':IDENTITIES,
 'identity_gate':'Explicit original-study annotations. Unknown and hybrid calls excluded from trajectory fitting, retained in contextual tissue plots. Malignant epithelial states grouped together, while immune/stromal lineages stay separate.',
 'geometry':'Within-EACH-identity theta100 Pearson residuals and PCA15, mutual30NN, bounded spatial cost0.1*d/(d+350). Every node is an original single cell; no tile or common-identity PCA distortion.',
 'potential':'Same alpha=.97 multipath solver and freely reoriented parent selection as v14. Candidate support/probability .2 and radius150 fixed. No forced connectors.',
 'validation':'Five80%-molecule thinnings202610390..394. Report all identity/root combinations and every abstention. No selection of a root-specific identity because its curve looks attractive.',
 'biology':'Within-identity state gradients and neighborhood composition. Separate epithelial cancer states, T-cell states and stromal states, with NO assertion of T-to-cancer/stroma conversion or tumor-caused fate.',
 'protected':'AMCP, original root nodes, scoring and Fig3 are unchanged. New regional-anchor claims must be labeled separately from cell-root claims.'}

def load():
 raw=ad.read_h5ad(ROOT/'outputs/NicheForest_paper_frozen_candidate/P6_frozen_breast/relation_input/event/synthetic.h5ad')
 cells=pd.read_csv(ROOT/'outputs/Manuscript_method_biology_alignment_20261022/RAW_CELLS_WITH_PUBLISHED_ANNOTATIONS.csv',dtype={'cell_name':str})
 assert np.array_equal(raw.obs_names.astype(str).str.replace('CELL_','',regex=False),cells.cell_name)
 cells['identity']=cells.published_cell_type.map(IDENTITIES)
 sel=cells.loc[cells.identity.notna()].copy();sel['raw_row']=sel.index;sel=sel.reset_index(drop=True)
 counts=sparse.csr_matrix(raw.X)[sel.raw_row].astype(float)
 roots=pd.read_csv(ROOT/'outputs/Root_region_state_forest_v5_20261023/CANONICAL_ROOTS.csv')
 return raw,counts,sel,roots

def per_identity_candidates(counts,obs,genes=None,**options):
 rr=[];cc=[];dd=[];pcs=[]
 for identity in sorted(obs.identity.unique()):
  ids=np.flatnonzero(obs.identity.to_numpy()==identity)
  x=counts[ids].toarray() if sparse.issparse(counts) else counts[ids]
  a,pc=candidate(x,obs.loc[ids,['spatial_x','spatial_y']].to_numpy(),np.repeat(identity,len(ids)),genes=genes,**options)
  coo=a.tocoo();rr.extend(ids[coo.row]);cc.extend(ids[coo.col]);dd.extend(coo.data)
  pcs.append((ids,pc))
 return sparse.csr_matrix((dd,(rr,cc)),shape=(len(obs),len(obs))),pcs

def run():
 OUT.mkdir(exist_ok=True,parents=True);pp=OUT/'PROTOCOL.json'
 if pp.exists():assert json.loads(pp.read_text())==PROTOCOL
 else:pp.write_text(json.dumps(PROTOCOL,indent=2))
 print('protocol',hashlib.sha256(pp.read_bytes()).hexdigest(),flush=True)
 raw,counts,obs,roots=load();obs.to_csv(OUT/'EXACT_CELL_INPUT.csv',index_label='node_id')
 roots.to_csv(OUT/'IMMUTABLE_SPATIAL_ROOTS.csv',index=False)
 a=ad.AnnData(X=counts.copy(),obs=obs.copy());a.var_names=raw.var_names;a.obs_names=obs.cell_name.to_numpy();a.obsm['spatial']=obs[['spatial_x','spatial_y']].to_numpy()
 a.write_h5ad(OUT/'EXACT_ALL_IDENTITY_COUNTS.h5ad',compression='gzip')
 matrix,pcs=per_identity_candidates(counts,obs);sparse.save_npz(OUT/'CANDIDATE_COSTS.npz',matrix)
 fits,seedaudit=support_sets(matrix,obs,roots);seedaudit.to_csv(OUT/'ROOT_REGION_SEEDS.csv',index=False)
 reference={};audit=[]
 for key,item in fits.items():
  fitted=constructors(matrix,item);ids=fitted['nodes'];parent=fitted['parent_local'];tau=fitted['tau'];label=key[0]+'_'+key[1]
  frame=obs.iloc[ids].copy();frame['node_id']=ids;frame['pseudotime']=tau/(max(tau) or 1);frame['unscaled_potential_distance']=tau;frame['is_region_seed']=parent<0
  frame['membership']=[item['membership'][int(i)] for i in ids];frame.to_csv(OUT/f'{label}_NODES.csv',index=False)
  edges=[{'source':int(ids[parent[j]]),'target':int(ids[j]),'cost':float(matrix[ids[parent[j]],ids[j]])} for j in np.flatnonzero(parent>=0)]
  pd.DataFrame(edges).to_csv(OUT/f'{label}_EDGES.csv',index=False)
  reference[key]=fitted
  audit.append({'root_id':key[0],'identity':key[1],'n_nodes':len(ids),'n_seeds':len(item['seeds']),'n_edges':len(edges),'replaced_parent_edges':int(np.sum(parent!=fitted['SPT_parent_local'])),'linear_residual':fitted['relative_linear_residual']})
  print('full niche forest',key,len(ids),flush=True)
 pd.DataFrame(audit).to_csv(OUT/'ALL_IDENTITY_FOREST_AUDITS.csv',index=False)
 # Distributions around unchanged root points, before any expression fitting.
 spatial=obs[['spatial_x','spatial_y']].to_numpy();context=[]
 for row in roots.itertuples():
  nearby=obs.loc[np.linalg.norm(spatial-[row.x,row.y],axis=1)<=150]
  for celltype,n in nearby.published_cell_type.value_counts().items():context.append({'root_id':row.root_id,'published_cell_type':celltype,'n_cells':int(n),'fraction_among_modelled':n/len(nearby)})
 pd.DataFrame(context).to_csv(OUT/'ROOT_REGION_CELL_COMPOSITION.csv',index=False)
 stability=[]
 from scipy.stats import spearmanr
 for seed in range(202610390,202610395):
  thin=counts.copy();thin.data=np.random.default_rng(seed).binomial(thin.data.astype(int),.8).astype(float)
  trial,_=per_identity_candidates(thin,obs);other,_=support_sets(trial,obs,roots)
  for key,original in reference.items():
   if key not in other:continue
   fitted=constructors(trial,other[key]);common=np.intersect1d(original['nodes'],fitted['nodes']);union=np.union1d(original['nodes'],fitted['nodes'])
   oi=pd.Index(original['nodes']).get_indexer(common);ti=pd.Index(fitted['nodes']).get_indexer(common)
   for name,field in [('Hitting_0.97','tau'),('State_SPT','SPT_tau')]:
    stability.append({'seed':seed,'root_id':key[0],'identity':key[1],'method':name,'support_jaccard':len(common)/len(union),'order_spearman':spearmanr(original[field][oi],fitted[field][ti]).statistic})
  pd.DataFrame(stability).to_csv(OUT/'MOLECULE_THINNING_STABILITY.partial.csv',index=False)
  print('all-identity thinning',seed,flush=True)
 pd.DataFrame(stability).to_csv(OUT/'MOLECULE_THINNING_STABILITY.csv',index=False)
 print(pd.DataFrame(stability).groupby(['identity','method']).order_spearman.median().unstack().to_string(),flush=True)

if __name__=='__main__':run()
