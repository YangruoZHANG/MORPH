"""Local identity-pure states and one observed source ensemble per root/type.

The original region supplied many diverse observed source cells, not many
experimentally established founders. Preserve EXACTLY those source-cell IDs,
but estimate one initial regional state from their normalized mean features.
Non-source states are small coordinate-only local pools. No invented cell,
count, source, coordinate, trajectory length target or gene-driven pooling.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from source_preserving_state_units import candidate_from_pcs,supports
ROOT=Path(__file__).resolve().parents[2]

def spatial_partitions(xy,ids,maximum,diameter=300.):
 pending=[np.sort(np.asarray(ids,int))];result=[]
 while pending:
  members=pending.pop()
  if len(members)<=1 or (len(members)<=maximum and float(np.max(pdist(xy[members])))<=diameter):result.append(members);continue
  axis=int(np.argmax(np.ptp(xy[members],axis=0)));ordered=members[np.lexsort((members,xy[members,axis]))];middle=len(ordered)//2;pending.extend([ordered[:middle],ordered[middle:]])
 return result

def make_units(obs,source_contract,mode='ensemble',maximum=None):
 if maximum is None:
  old=pd.read_csv(ROOT/'data/breast_state_inputs/reference_metacell_membership.csv',usecols=['metacell_index']);maximum=int(round(old.metacell_index.value_counts().median()))
 if mode not in ['ensemble','pool_only','source_only']:raise ValueError('Unknown observation ablation.')
 source_contract=source_contract.loc[source_contract.count_view.eq('WholeCell')];rids=sorted(source_contract.root_id.unique());source=np.zeros((len(obs),len(rids)),bool)
 for j,rid in enumerate(rids):
  ids=source_contract.loc[source_contract.root_id.eq(rid),'core_node_id'].to_numpy(int);source[ids,j]=True
 stage=obs.published_cell_type.map({'DCIS_1':0,'DCIS_2':0,'Invasive_Tumor':1,'Prolif_Invasive_Tumor':1}).fillna(-1).to_numpy(int);signature=(source*(2**np.arange(len(rids)))).sum(1);keys=pd.DataFrame({'identity':obs.identity,'stage':stage,'source_signature':signature});xy=obs[['spatial_x','spatial_y']].to_numpy();groups=[]
 for (_,_,sig),ids in keys.groupby(['identity','stage','source_signature'],sort=True).indices.items():
  ids=np.asarray(ids,int)
  if sig and mode in ['ensemble','source_only']:
   assert len(ids)==1 or np.max(pdist(xy[ids]))<=300+1e-8,'Source ensemble exceeds the original radius150um niche diameter.';groups.append(np.sort(ids))
  elif mode=='source_only':groups.extend(np.array([i],int) for i in ids)
  else:groups.extend(spatial_partitions(xy,ids,maximum,300.))
 groups.sort(key=lambda group:int(group.min()));metadata=[];rows=[];cols=[];members=[]
 for unit,ids in enumerate(groups):
  center=xy[ids].mean(0);representative=int(ids[np.lexsort((ids,np.linalg.norm(xy[ids]-center,axis=1)))[0]]);r=obs.iloc[representative]
  assert len(set(obs.iloc[ids].identity))==1 and len(set(stage[ids]))==1 and len(set(signature[ids]))==1
  metadata.append({'node_id':unit,'unit_id':f'S{unit:05d}','identity':r.identity,'published_cell_type':r.published_cell_type,'stage':int(stage[representative]),'n_raw_cells':len(ids),'is_source_ensemble':bool(signature[representative]) and mode in ['ensemble','source_only'],'spatial_x':float(xy[representative,0]),'spatial_y':float(xy[representative,1]),'representative_core_cell_id':representative,'representative_cell_name':str(r.cell_name),**{f'source_{rid}':bool(source[representative,j]) for j,rid in enumerate(rids)}})
  rows.extend(np.repeat(unit,len(ids)));cols.extend(ids);members.extend({'unit_node_id':unit,'unit_id':f'S{unit:05d}','core_cell_id':int(i),'original_model_node_id':int(obs.iloc[i].original_model_node_id),'raw_row':int(obs.iloc[i].raw_row),'cell_name':str(obs.iloc[i].cell_name)} for i in ids)
 aggregate=sparse.csr_matrix((np.ones(len(cols),np.int64),(rows,cols)),shape=(len(groups),len(obs)));mass=np.asarray(aggregate.sum(1)).ravel();mean=sparse.diags(1/mass)@aggregate;units=pd.DataFrame(metadata);member=pd.DataFrame(members)
 assert len(member)==len(obs) and member.core_cell_id.nunique()==len(obs)
 for j,rid in enumerate(rids):
  actual=set(member.loc[member.unit_node_id.isin(units.loc[units[f'source_{rid}'],'node_id']),'core_cell_id']);assert actual==set(np.flatnonzero(source[:,j])),'Pooling changed actual source-cell IDs.'
 return units,member,aggregate,mean,maximum

def geometry(raw_counts,raw_obs,units,mean,genes):
 genes=np.asarray(genes,int);features=np.zeros((len(units),15))
 for identity in sorted(raw_obs.identity.unique()):
  ids=np.flatnonzero(raw_obs.identity.eq(identity));x=raw_counts[ids][:,genes].toarray().astype(float);depth=np.maximum(x.sum(1),1);frequency=x.sum(0)/max(x.sum(),1);mu=depth[:,None]*frequency[None,:];residual=np.clip((x-mu)/np.sqrt(np.maximum(mu+mu*mu/100,1e-8)),-np.sqrt(len(ids)),np.sqrt(len(ids)));k=min(15,len(ids)-1,len(genes))
  if k<1:continue
  pc=PCA(k,random_state=42).fit_transform(residual);features[:,:k]+=mean[:,ids]@pc
 return candidate_from_pcs(features,units[['spatial_x','spatial_y']].to_numpy(),units.identity.to_numpy()),features
