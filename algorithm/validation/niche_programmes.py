"""Full regional-anchor alternative: all identities and all measured programs.

Clinical annotations are explicit priors, not independent validation truth.
New cell-state panels are declared here before any corresponding result.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import sys,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,rankdata
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'algorithm/forest'))
from identity_conditioned_forest import load,per_identity_candidates
from multipath_ordering import support_sets,constructors
from epithelial_stage_constraints import stage_fit
from regional_state_forest import bh,partial_ranks
OUT=ROOT/'outputs/Niche_program_validation'
EXTRA={
 'LUMINAL_IDENTITY':['FOXA1','GATA3','MLPH','AGR3','KRT8'],
 'BASAL_IDENTITY':['KRT5','KRT14','KRT15','KRT6B'],
 'EPITHELIAL_JUNCTION':['EPCAM','CDH1','JUP','DSP','DSC2'],
 'SECRETORY_B':['MZB1','PRDM1','TNFRSF17','DERL3','SLAMF7','TENT5C'],
 'NAIVE_B_ASSOCIATED':['TCL1A','BANK1','MS4A1','CD19','CD79B'],
 'B_CELL_IDENTITY':['CD79A','CD79B','MS4A1','CD19','BANK1'],
 'STROMAL_REMODELLING':['LUM','POSTN','PCOLCE','FBLN1','MMP2','MMP1'],
 'CONTRACTILE_STATE':['ACTA2','ACTG2','MYH11','MYLK'],
 'INFLAMMATORY_MACROPHAGE':['CD14','LYZ','C15ORF48','CCL8','APOC1'],
 'C1Q_MRC1_MACROPHAGE':['C1QA','C1QC','MRC1','CD163']}
PROTOCOL={
 'model':'Explicit region-anchor alternative, v15 per-identity single-cell geometry plus v16 clinical-stage constraint only in malignant epithelial cells. Original root finding unchanged; no claim all cell types descend from its T-like root node.',
 'programs':'All original43 measured primary panels plus10 prespecified measured identity/state panels. Report all root/identity outcomes. Additional panels are secondary biological interpretation and not retrospective replacements for failed original hypotheses.',
 'extra_panels':EXTRA,
 'priors':'Published identity and epithelial DCIS/invasive staging remain model inputs even when a gene panel is excluded. Whole-panel exclusion controls construction-gene reuse, not independence from the original annotation.',
 'validation':'Rebuild each identity residuals/PCA/graph, membership, seed reachability and potential after excluding the ENTIRE tested panel. Same-root-stage-admissible SPT control. Scores normalized by non-panel total, within-identity gene z-scores.',
 'statistics':'Partial rank controlling non-panel depth and x/y; 9999 whole-spatial-block sign flips at250/500/750um, >=12 blocks, BH jointly over all original+secondary tests separately per method and scale; absrho>=.3, LOOsign>=.8 and q<=.1 allscales. For absrho<.3 retain p=1 conservatively and explicitly label effect-screened; no claim of a computed small p value.',
 'full_data':'Descriptive full-fit correlations and deciles kept separate; biological naming requires heldout effect qualified AND full-data sign agreement.',
 'borders':'No ancestry, migration, causal promotion/inhibition, clinical outcome prediction or genome-wide GSEA claim. A clinical-stage prior cannot validate itself.',
 'protected':'No AMCP/scoring/Fig3/old artifact edits, no automatic final release.'}

def fit_all(matrix,obs,roots):
 support,seeds=support_sets(matrix,obs,roots);fits={}
 for key,item in support.items():
  fitted=stage_fit(matrix,obs,item) if key[1]=='Malignant_epithelial' else constructors(matrix,item)
  if fitted is not None:fits[key]=fitted
 return fits,seeds

def test(item,field,score,z,xy,depth,index):
 ids=item['nodes'];tau=item[field];design=rankdata(np.column_stack([depth[ids],xy[ids]]),axis=0)
 design=(design-design.mean(0))/np.maximum(design.std(0),1);design=np.column_stack([np.ones(len(ids)),design])
 a,b,rho=partial_ranks(tau,score[ids],design)
 loo=[np.sign(partial_ranks(tau,z[ids][:,[k for k in range(z.shape[1]) if k!=j]].mean(1),design)[2])==np.sign(rho) for j in range(z.shape[1])]
 record={'partial_rho':float(rho),'LOO_sign_agreement':float(np.mean(loo)),'n_cells':len(ids),'effect_screened':bool(abs(rho)<.3)}
 for scale in [250,500,750]:
  _,block=np.unique(np.floor(xy[ids]/scale).astype(int),axis=0,return_inverse=True);sums=np.bincount(block,weights=a*b)
  p=1.
  if abs(rho)>=.3 and len(sums)>=12:
   rng=np.random.default_rng(202610410+index+scale);exceed=0
   for start in range(0,9999,1000):
    null=np.abs(rng.choice([-1,1],size=(min(1000,9999-start),len(sums)))@sums)
    exceed+=np.sum(null>=abs(sums.sum()))
   p=(1+exceed)/10000
  record[f'p_{scale}']=p;record[f'n_blocks_{scale}']=len(sums)
 return record

def run():
 OUT.mkdir(exist_ok=True,parents=True);p=OUT/'PROTOCOL.json'
 if p.exists():assert json.loads(p.read_text())==PROTOCOL
 else:p.write_text(json.dumps(PROTOCOL,indent=2))
 raw,counts,obs,roots=load();xy=obs[['spatial_x','spatial_y']].to_numpy();identity=obs.identity.to_numpy()
 # The full graph is exactly v15; only the declared malignant-stage adapter changes.
 from scipy import sparse
 full=sparse.load_npz(ROOT/'outputs/Niche_conditioned_identity_forests/CANDIDATE_COSTS.npz')
 fullfits,seeds=fit_all(full,obs,roots);seeds.to_csv(OUT/'FULL_SOURCE_AUDIT.csv',index=False)
 original=pd.read_csv(ROOT/'outputs/Count_aware_state_forest/PROGRAM_CATALOG.csv')
 panels=[{'program':r.family+'::'+r.gene_set,'panel_genes':r.panel_genes,'tier':'original_primary'} for r in original.itertuples()]
 panels.extend([{'program':'secondary_curated::'+name,'panel_genes':';'.join(genes),'tier':'new_secondary'} for name,genes in EXTRA.items()])
 pd.DataFrame(panels).to_csv(OUT/'LOCKED_PROGRAM_CATALOG.csv',index=False)
 rows=[];descriptive=[];deciles=[];gene_manifest=[]
 requested=os.environ.get('NICHEFOREST_PANEL_RANGE')
 chosen=set(range(len(panels))) if not requested else set(range(*map(int,requested.split(':'))))
 for index,panel in enumerate(panels):
  if index not in chosen:continue
  # Resume only complete panel checkpoints, preserving the locked catalog.
  checkpoint=OUT/f'panel_{index:02d}.json'
  if checkpoint.exists():
   saved=json.loads(checkpoint.read_text());rows.extend(saved['tests']);descriptive.extend(saved['descriptive']);deciles.extend(saved['deciles']);gene_manifest.extend(saved['genes']);print('resume panel',index,flush=True);continue
  start=time.perf_counter();target=np.array([raw.var_names.get_loc(g) for g in panel['panel_genes'].split(';') if g in raw.var_names]);assert len(target)>=3
  train=np.setdiff1d(np.arange(raw.n_vars),target);depth=np.maximum(np.asarray(counts[:,train].sum(1)).ravel(),1)
  scoregenes=np.log1p(counts[:,target].toarray()/depth[:,None]*np.median(depth));z=np.zeros(scoregenes.shape)
  for ident in np.unique(identity):
   ids=np.flatnonzero(identity==ident);z[ids]=StandardScaler().fit_transform(scoregenes[ids])
  score=z.mean(1);matrix,_=per_identity_candidates(counts,obs,genes=train);fits,seeds=fit_all(matrix,obs,roots)
  tr=[];de=[];bi=[]
  for key,item in fits.items():
   if len(item['nodes'])<20:continue
   for method,field in [('Multipath','tau'),('Matched_SPT','SPT_tau')]:
    record=test(item,field,score,z,xy,depth,index)
    record.update({'program':panel['program'],'tier':panel['tier'],'root_id':key[0],'identity':key[1],'method':method});tr.append(record)
   fullitem=fullfits[key];ids=fullitem['nodes'];rho=float(spearmanr(fullitem['tau'],score[ids]).statistic)
   de.append({'program':panel['program'],'root_id':key[0],'identity':key[1],'full_data_spearman':rho})
   bins=pd.qcut(pd.Series(fullitem['tau']).rank(method='first'),10,labels=False).to_numpy()
   for b in range(10):
    selected=ids[bins==b];bi.append({'program':panel['program'],'root_id':key[0],'identity':key[1],'decile':b,'n_cells':len(selected),'mean_score':float(score[selected].mean()),'mean_tau':float(fullitem['tau'][bins==b].mean())})
  ge=[{'program':panel['program'],'withheld_gene':raw.var_names[i]} for i in target]
  temporary=checkpoint.with_suffix(f'.{os.getpid()}.tmp')
  temporary.write_text(json.dumps({'tests':tr,'descriptive':de,'deciles':bi,'genes':ge},indent=2))
  temporary.replace(checkpoint)
  rows.extend(tr);descriptive.extend(de);deciles.extend(bi);gene_manifest.extend(ge)
  if not requested:pd.DataFrame(rows).to_csv(OUT/'ALL_TESTS.partial.csv',index=False)
  print('full niche program',index,panel['program'],'seconds',round(time.perf_counter()-start,1),flush=True)
 if requested:
  print('Worker complete. No partial-family q-values or final table emitted.',requested,flush=True)
  return
 result=pd.DataFrame(rows)
 for scale in [250,500,750]:
  result[f'q_{scale}']=np.nan
  for method in result.method.unique():
   mask=result.method.eq(method);result.loc[mask,f'q_{scale}']=bh(result.loc[mask,f'p_{scale}'])
 result['qualified']=result.partial_rho.abs().ge(.3)&result.LOO_sign_agreement.ge(.8)&result[[f'q_{s}' for s in [250,500,750]]].le(.1).all(axis=1)
 result=result.merge(pd.DataFrame(descriptive),on=['program','root_id','identity'],how='left')
 result['full_data_sign_agrees']=np.sign(result.partial_rho)==np.sign(result.full_data_spearman)
 result['usable_for_naming']=result.qualified&result.full_data_sign_agrees
 result.to_csv(OUT/'ALL_PROGRAM_TESTS.csv',index=False);pd.DataFrame(descriptive).to_csv(OUT/'FULL_DATA_DESCRIPTIVE_TRENDS.csv',index=False)
 pd.DataFrame(deciles).to_csv(OUT/'FULL_DATA_PROGRAM_DECILES.csv',index=False);pd.DataFrame(gene_manifest).to_csv(OUT/'WITHHELD_GENE_MANIFEST.csv',index=False)
 print(result.groupby('method')[['qualified','usable_for_naming']].sum().to_string(),flush=True)
 print(result.loc[result.method.eq('Multipath')&result.usable_for_naming,['root_id','identity','program','partial_rho','full_data_spearman']].to_string(index=False),flush=True)

if __name__=='__main__':run()
