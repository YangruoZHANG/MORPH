"""Audit other lineage-marker conflicts before further cell-state naming.

Evidence flags are not new cell-type truth or a claim of biological hybrids.
The predefined threshold is the same two distinct measured markers used by
the T/stromal audit. No trajectory, length, direction or enriched hit chooses it.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/breast_reference'
CORE=ROOT/'outputs/Nuclear_supported_identity_cores'
OUT=ROOT/'outputs/Multifamily_nuclear_identity_audit'
MARKERS={'T':['CD3D','CD3E','CD3G','CD247','TRAC'],'Stromal':['LUM','PDGFRA','PDGFRB','POSTN'],'B':['MS4A1','CD19','CD79A','CD79B','BANK1'],'Epithelial':['EPCAM','KRT8','KRT7','KRT5','KRT14'],'Myeloid':['AIF1','LYZ','CD14','CD163'],'Endothelial':['PECAM1','VWF','CLDN5','ESM1']}

def main():
 OUT.mkdir(parents=True,exist_ok=True);raw=ad.read_h5ad(BASE/'data/BREAST_ORIGINAL_COUNTS.h5ad',backed='r');nuc=sparse.load_npz(ROOT/'data/breast_nuclear/NUCLEAR_Q20_ORIGINAL_CELL_GENE_ORDER.npz');cells=pd.read_csv(BASE/'data/PUBLISHED_CELL_METADATA.csv',dtype={'cell_name':str});genes=pd.Index(raw.var_names);roster=cells.copy();flags={}
 for family,markers in MARKERS.items():
  assert set(markers)<=set(genes);values=np.asarray((nuc[:,genes.get_indexer(markers)]>0).sum(1)).ravel();roster['n_'+family+'_nuclear_marker_genes']=values;flags[family]=values>=2;roster[family+'_nuclear_evidence']=flags[family]
 roster['n_nuclear_identity_families_supported']=np.column_stack(list(flags.values())).sum(1);roster.to_csv(OUT/'ALL_ORIGINAL_CELL_MULTIFAMILY_EVIDENCE.csv',index=False)
 core=pd.read_csv(CORE/'EXACT_CORE_CELL_INPUT.csv',dtype={'cell_name':str});q=roster.iloc[core.raw_row.to_numpy(int)].reset_index(drop=True);rows=[]
 for identity,ix in core.groupby('identity').groups.items():
  keep=np.array(list(ix));expected='Stromal' if identity=='Stromal' else 'T';other=[k for k in MARKERS if k!=expected];conflict=np.column_stack([flags[k][core.iloc[keep].raw_row.to_numpy(int)] for k in other]).any(1)
  rows.append({'published_identity':identity,'n_core_cells':len(keep),'n_any_other_family_marker_conflict':int(conflict.sum()),'fraction_any_other_family_marker_conflict':float(conflict.mean())})
  for family in other:rows[-1]['fraction_also_'+family]=float(q.iloc[keep][family+'_nuclear_evidence'].mean())
 pd.DataFrame(rows).to_csv(OUT/'CORE_CONFLICT_AUDIT.csv',index=False)
 sources=pd.read_csv(CORE/'EVERY_OLD_REGIONAL_SOURCE_CELL_DISPOSITION.csv',dtype={'cell_name':str});mapping=roster.set_index('cell_name');sourceflags=mapping.loc[sources.cell_name,list(k+'_nuclear_evidence' for k in MARKERS)].reset_index(drop=True);pd.concat([sources.reset_index(drop=True),sourceflags],axis=1).to_csv(OUT/'ALL_ORIGINAL_SOURCE_MULTIFAMILY_AUDIT.csv',index=False)
 protocol={'marker_panels':MARKERS,'evidence':'At least2 measured distinct nuclear marker genes per family; not a validated celltype classifier, hybrid proof, lineage tracing or automatic author-label correction.','preselection':'No fine trajectory, enriched-program sign, spatial shape or desired length chooses a marker threshold.','status':'Audit only. Raw counts, published labels, root input and old trees not overwritten.'};(OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2));print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
