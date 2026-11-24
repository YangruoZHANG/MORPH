"""Exact contact-scoring function used by the approved main visualizations.
Extracted without modifying the function body. This is descriptive contact
scoring, not conditional-mutual-information estimation or a significance test.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
OUT = None

def contacts(d):
 x=sparse.csr_matrix(d['raw'].X);depth=np.maximum(np.asarray(x.sum(1)).ravel(),1);records=[]
 for event,skey,tkey,ligand,receptor,description in [('E1',('R0','Stromal'),('R1','B_cell'),'CXCL12','CXCR4','Recruitment-like hypothesis'),('E2',('R2','Malignant_epithelial'),('R1','CD8_T'),'CD274','PDCD1','Inhibition-like hypothesis')]:
  sourceunits=d['fits'][skey][0].node_id;targetunits=d['fits'][tkey][0].node_id;sr=d['members'].loc[d['members'].unit_node_id.isin(sourceunits),'raw_row'].to_numpy(int);tr=d['members'].loc[d['members'].unit_node_id.isin(targetunits),'raw_row'].to_numpy(int);lv=np.asarray(x[:,d['raw'].var_names.get_loc(ligand)].toarray()).ravel();rv=np.asarray(x[:,d['raw'].var_names.get_loc(receptor)].toarray()).ravel();sr=sr[lv[sr]>0];tr=tr[rv[tr]>0]
  if not len(sr) or not len(tr):continue
  tree=cKDTree(d['rawxy'][sr]);near=tree.query_ball_point(d['rawxy'][tr],r=60)
  for target,nn in zip(tr,near):
   for j in nn:
    source=int(sr[j]);distance=float(np.linalg.norm(d['rawxy'][source]-d['rawxy'][target]));score=float(np.sqrt(np.log1p(lv[source]/depth[source]*np.median(depth))*np.log1p(rv[target]/depth[target]*np.median(depth)))*np.exp(-.5*(distance/30)**2));records.append({'event':event,'source_raw_row':source,'target_raw_row':int(target),'source_root':skey[0],'source_identity':skey[1],'target_root':tkey[0],'target_identity':tkey[1],'ligand':ligand,'receptor':receptor,'distance_um':distance,'descriptive_contact_score':score,'hypothesis':description,'formal_call':False})
 result=pd.DataFrame(records);result.to_csv(OUT/'source_data/ALL_OBSERVED_LIGAND_RECEPTOR_CONTACT_CANDIDATES.csv',index=False);return result
