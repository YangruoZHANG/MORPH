"""Spatially pooled observations; molecular state inference, NOT migration.

Only new construction choice: recover the original observation granularity
with identity/stage/source-pure, coordinate-only balanced partitions. Every
count is an exact sum, every display coordinate is a measured member cell.
Existing geometry, alpha, affiliation cutoff and parent objective unchanged.
No trial radius, desired path length, branch count or marker-based partition.
"""
import os,sys,json,hashlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'algorithm/experimental'))
from cell_resolved_multipath_v14_20261023 import candidate,constructors
from stage_admissible_epithelial_v16_20261023 import stage_fit
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from scipy.sparse.csgraph import dijkstra,connected_components
BASE=ROOT/'outputs/NicheForest_current_research_freeze_20261024'
MEAN_PC=os.environ.get('NICHEFOREST_STATE_UNITS_MEAN_PC')=='1'
OUT=ROOT/('outputs/Exposure_corrected_state_units_v23_20261024' if MEAN_PC else 'outputs/Source_preserving_state_units_v22_20261024')
MAPFILE=ROOT/'outputs/NicheForest_paper_frozen_candidate/P6_frozen_breast/relation_input/formal_metacell/cell_to_metacell_mapping.csv'
PROTOCOL={
 'change':'Observation model only: coordinate-only source/identity/stage-pure count pooling, then the existing molecular graph/multipath model. Remove the rejected physical150um edge ceiling. Spatial adjacency is NOT developmental ancestry.',
 'granularity':'Upper leaf mass is the median observed membership size of the original5405metacells, rounded to integer; extracted once, no tuning on markers/plots. Balanced spatial binary partitions split the widest coordinate at its median until that upper mass. Actual original-member sums, no synthetic cell or substituted counts.',
 'purity':'Pool only within published identity, malignant DCIS/invasive stage, and the full vector of existing20 frozen source-cell flags. This preserves every supplied original source-cell ID exactly; source flags are input, not response labels.',
 'sources':'Fixed three spatial anchors and every original declared source cell, unchanged. Root eligibility evaluated in original raw-cell counts. A pooled unit is a source iff all members were sources for that root/identity.',
 'coordinates':'Representative is an observed member closest to the member-coordinate mean, stable raw-ID tie-break. Members and coordinates exported. No fictional centroid treated as a cell.',
 'counts':'Exact integer sums over the313 measured genes. Pooling independent of construction/validation genes. Published identity and clinical stage remain explicit priors, not independent ground truth.',
 'geometry':'Same per-identity theta100 Pearson residual PCA15 and reciprocal30NN with bounded0.1*d/(d+350) spatial cost; no physical edge cap or forced connection.',
 'inference':'Same affinity>=.2, same alpha.97 hitting potential and original minimum-cost increasing-potential parent objective. Stage-admissibility prior retained. Source raw mass replaces the old minimum6unit check; all stage-source decisions frozen before pooling.',
 'validation':'Whole-panel exclusion before depth/residual/PCA/graph/affiliation/order; pooled target scores evaluated once per observed unit with spatial-block sign flips and joint BH. Five molecule80%thinnings, exact-root/source/mass invariants. No old q values transferred.',
 'interpretation':'Within-identity molecular state organization localized in tissue. Nonlocal edges are state associations, not migration, T-to-stroma conversion or causal tumor influence.',
 'status':'Development experiment. No new figure/final release until validation passes.'}
if MEAN_PC:
 PROTOCOL=dict(PROTOCOL)
 PROTOCOL['exposure_correction']='Fit the unchanged per-RAW-cell theta100 Pearson-residual PCA15 within identity, then average exact member PC scores. Do not apply a single-cell NB model to variable-mass pooled sums. The graph uses expected per-cell molecular state, not total count mass. Whole-panel exclusion occurs before every raw-cell normalization/PCA and pooling step.'
_RAW_GEOMETRY=None

def configure_raw_geometry(raw_counts,raw_obs,members):
 global _RAW_GEOMETRY
 unitof=members.set_index('raw_node_id').loc[raw_obs.index,'unit_node_id'].to_numpy(int)
 mass=np.bincount(unitof,minlength=int(members.unit_node_id.max())+1)
 mean=sparse.diags(1/mass)@sparse.csr_matrix((np.ones(len(raw_obs)),(unitof,np.arange(len(raw_obs)))),shape=(len(mass),len(raw_obs)))
 _RAW_GEOMETRY=(sparse.csr_matrix(raw_counts),raw_obs,mean)

def mean_cell_pcs(genes=None):
 if _RAW_GEOMETRY is None:raise RuntimeError('Configure exact raw-cell counts/membership before exposure-corrected geometry')
 counts,obs,mean=_RAW_GEOMETRY
 if genes is None:genes=np.arange(counts.shape[1])
 pcs=np.zeros((mean.shape[0],min(15,len(genes))))
 for identity in sorted(obs.identity.unique()):
  ix=np.flatnonzero(obs.identity.to_numpy()==identity);x=counts[ix][:,genes].toarray().astype(float)
  depth=np.maximum(x.sum(1),1);freq=x.sum(0)/max(x.sum(),1);mu=depth[:,None]*freq[None,:]
  residual=np.clip((x-mu)/np.sqrt(np.maximum(mu+mu*mu/100,1e-8)),-np.sqrt(len(x)),np.sqrt(len(x)))
  k=min(15,len(ix)-1,len(genes))
  if k<1:continue
  pc=PCA(k,random_state=42).fit_transform(residual)
  pcs[:,:k]+=mean[:,ix]@pc
 return pcs

def candidate_from_pcs(pc,xy,identity):
 rr=[];cc=[];dd=[];scales=np.zeros(len(pc))
 for label in sorted(np.unique(identity)):
  ix=np.flatnonzero(identity==label);k=min(30,len(ix)-1)
  if k<1:continue
  distances,neighbors=NearestNeighbors(n_neighbors=k+1,n_jobs=2).fit(pc[ix]).kneighbors(pc[ix])
  for j,node in enumerate(ix):
   good=neighbors[j]!=j;nn=neighbors[j][good][:k];dist=distances[j][good][:k];scales[node]=max(dist[-1],1e-8)
   rr.extend(np.repeat(node,len(nn)));cc.extend(ix[nn]);dd.extend(dist)
 directed=sparse.csr_matrix((np.ones(len(rr),bool),(rr,cc)),shape=(len(pc),len(pc)))
 reciprocal=directed.multiply(directed.T).tocoo();keep=reciprocal.row<reciprocal.col;u=reciprocal.row[keep];v=reciprocal.col[keep]
 expr=np.linalg.norm(pc[u]-pc[v],axis=1)/np.sqrt(scales[u]*scales[v]);distance=np.linalg.norm(xy[u]-xy[v],axis=1)
 cost=expr+.1*distance/(distance+350)
 return sparse.csr_matrix((np.r_[cost,cost],(np.r_[u,v],np.r_[v,u])),shape=directed.shape)

def load():
 raw=ad.read_h5ad(BASE/'data/BREAST_ORIGINAL_COUNTS.h5ad')
 obs=pd.read_csv(BASE/'data/MODEL_CELL_INPUT.csv',dtype={'cell_name':str}).set_index('node_id')
 nodes=pd.read_csv(BASE/'forest/FINAL_NODE_ASSIGNMENTS.csv',dtype={'cell_name':str})
 roots=pd.read_csv(BASE/'data/CANONICAL_SPATIAL_ANCHORS.csv')
 return raw,obs,nodes,roots

def partition(xy,ids,maximum):
 stack=[np.sort(np.asarray(ids,int))];out=[]
 while stack:
  ix=stack.pop()
  if len(ix)<=maximum:out.append(ix);continue
  axis=int(np.argmax(np.ptp(xy[ix],axis=0)))
  ordered=ix[np.lexsort((ix,xy[ix,axis]))];middle=len(ix)//2
  stack.extend([ordered[middle:],ordered[:middle]])
 return out

def make_units(raw,obs,nodes):
 old=pd.read_csv(MAPFILE,usecols=['metacell_index'])
 maximum=int(round(old.metacell_index.value_counts().median()))
 assert maximum>=2
 rids=sorted(nodes.root_id.unique(),key=lambda r:int(r[1:]))
 source=np.zeros((len(obs),len(rids)),bool)
 for j,rid in enumerate(rids):
  subset=nodes.loc[nodes.root_id.eq(rid)&nodes.is_source_cell.astype(str).str.lower().eq('true')]
  source[subset.node_id.to_numpy(int),j]=True
 stage=obs.published_cell_type.map({'DCIS_1':0,'DCIS_2':0,'Invasive_Tumor':1,'Prolif_Invasive_Tumor':1}).fillna(-1).astype(int).to_numpy()
 signature=(source*(2**np.arange(len(rids)))).sum(1)
 keys=pd.DataFrame({'identity':obs.identity.to_numpy(),'stage':stage,'source_signature':signature})
 xy=obs[['spatial_x','spatial_y']].to_numpy();members=[]
 for _,ix in keys.groupby(['identity','stage','source_signature'],sort=True).indices.items():members+=partition(xy,ix,maximum)
 members.sort(key=lambda ix:int(ix.min()));rows=[];membership=[];r=[];c=[]
 for unit,ix in enumerate(members):
  center=xy[ix].mean(0);representative=int(ix[np.lexsort((ix,np.linalg.norm(xy[ix]-center,axis=1)))[0]])
  first=obs.iloc[representative]
  assert (stage[ix]==stage[ix[0]]).all() and (signature[ix]==signature[ix[0]]).all()
  rows.append({'node_id':unit,'unit_id':f'U{unit:05d}','identity':first.identity,'published_cell_type':str(first.published_cell_type),'stage':int(stage[ix[0]]),'n_raw_cells':len(ix),'spatial_x':xy[representative,0],'spatial_y':xy[representative,1],'representative_raw_node_id':representative,'representative_cell_name':str(first.cell_name),**{f'source_{rid}':bool(source[ix[0],j]) for j,rid in enumerate(rids)}})
  membership.extend({'unit_id':f'U{unit:05d}','unit_node_id':unit,'raw_node_id':int(v),'raw_row':int(obs.iloc[v].raw_row),'cell_name':str(obs.iloc[v].cell_name)} for v in ix)
  r.extend(np.repeat(unit,len(ix)));c.extend(ix)
 aggregate=sparse.csr_matrix((np.ones(len(c)),(r,c)),shape=(len(members),len(obs)))
 counts=aggregate@sparse.csr_matrix(raw.X)[obs.raw_row.to_numpy(int)]
 np.testing.assert_array_equal(np.asarray(counts.sum(0)),np.asarray(sparse.csr_matrix(raw.X)[obs.raw_row.to_numpy(int)].sum(0)))
 frame=pd.DataFrame(rows).set_index('node_id')
 member=pd.DataFrame(membership)
 assert len(member)==len(obs) and member.raw_node_id.nunique()==len(obs)
 for j,rid in enumerate(rids):
  actual=set(member.loc[member.unit_node_id.isin(frame.index[frame[f'source_{rid}']]),'raw_node_id'])
  assert actual==set(np.flatnonzero(source[:,j]))
 return counts.tocsr(),frame,member,maximum

def geometry(counts,units,genes=None):
 if MEAN_PC:
  pc=mean_cell_pcs(genes)
  return candidate_from_pcs(pc,units[['spatial_x','spatial_y']].to_numpy(),units.identity.to_numpy())
 rr=[];cc=[];dd=[]
 for identity in sorted(units.identity.unique()):
  ix=np.flatnonzero(units.identity.to_numpy()==identity)
  if len(ix)<2:continue
  g,pc=candidate(counts[ix].toarray(),units.iloc[ix][['spatial_x','spatial_y']].to_numpy(),np.repeat(identity,len(ix)),genes=genes)
  coo=g.tocoo();rr.extend(ix[coo.row]);cc.extend(ix[coo.col]);dd.extend(coo.data)
 return sparse.csr_matrix((dd,(rr,cc)),shape=(len(units),len(units)))

def supports(graph,units,minimum_source_raw_cells=6):
 fits={};audit=[];xy=units[['spatial_x','spatial_y']].to_numpy()
 for identity in sorted(units.identity.unique()):
  ix=np.flatnonzero(units.identity.to_numpy()==identity);distances=[];keys=[];source_sets=[]
  for rid in sorted([c.removeprefix('source_') for c in units.columns if c.startswith('source_R')],key=lambda r:int(r[1:])):
   seeds=ix[units.iloc[ix][f'source_{rid}'].to_numpy(bool)]
   mass=int(units.n_raw_cells.iloc[seeds].sum())
   audit.append({'root_id':rid,'identity':identity,'source_unit_count':len(seeds),'source_raw_cell_count':mass,'eligible':mass>=minimum_source_raw_cells})
   if mass<minimum_source_raw_cells:continue
   keys.append((rid,identity));source_sets.append(seeds)
   distances.append(dijkstra(graph,directed=False,indices=seeds,min_only=True)[ix])
  if not keys:continue
  dd=np.array(distances);nearest=dd.min(0);valid=np.isfinite(nearest);positive=nearest[valid&(nearest>0)]
  scale=np.median(positive) if len(positive) else 1
  weights=np.zeros(dd.shape);weights[:,valid]=np.exp(-(dd[:,valid]-nearest[valid])/max(scale,1e-8));probs=weights/np.maximum(weights.sum(0),1e-12)
  for k,key in enumerate(keys):
   seeds=source_sets[k];ids=np.union1d(ix[(probs[k]>=.2)&np.isfinite(dd[k])],seeds)
   sub=graph[ids][:,ids];_,component=connected_components(sub,directed=False);si=pd.Index(ids).get_indexer(seeds)
   ids=ids[np.isin(component,np.unique(component[si]))]
   fits[key]={'nodes':ids,'seeds':seeds}
 return fits,pd.DataFrame(audit)

def fit_all(graph,units):
 support,audit=supports(graph,units);fits={}
 for key,item in support.items():
  if key[1]=='Malignant_epithelial':
   # Source-stage selection was frozen on RAW cells before pooling. Do not
   # silently require six pooled units or select a new source stage here.
   from improve_trajectory_continuity_v20_20261024 import fit_cohort
   frame=units.iloc[item['nodes']].copy();frame['node_id']=item['nodes'];frame['is_source_cell']=np.isin(item['nodes'],item['seeds'])
   result=fit_cohort(graph,units,frame)
  else:result=constructors(graph,item)
  fits[key]=result
 return fits,audit

def run():
 OUT.mkdir(exist_ok=True,parents=True);pp=OUT/'PROTOCOL.json'
 if pp.exists():assert json.loads(pp.read_text())==PROTOCOL
 else:pp.write_text(json.dumps(PROTOCOL,indent=2))
 raw,obs,nodes,roots=load();counts,units,members,mass=make_units(raw,obs,nodes)
 if MEAN_PC:configure_raw_geometry(sparse.csr_matrix(raw.X)[obs.raw_row.to_numpy(int)],obs,members)
 units.to_csv(OUT/'EXACT_OBSERVATION_UNITS.csv',index_label='node_id');members.to_csv(OUT/'EXACT_UNIT_MEMBERS.csv',index=False)
 roots.to_csv(OUT/'UNCHANGED_SPATIAL_ANCHORS.csv',index=False)
 a=ad.AnnData(X=counts,obs=units.copy());a.var_names=raw.var_names;a.obs_names=units.unit_id.to_numpy();a.write_h5ad(OUT/'EXACT_POOLED_COUNTS.h5ad',compression='gzip')
 graph=geometry(counts,units);sparse.save_npz(OUT/'MOLECULAR_CANDIDATES.npz',graph);fits,audit=fit_all(graph,units)
 audit.to_csv(OUT/'EXACT_SOURCE_AUDIT.csv',index=False);nodeframes=[];edges=[];records=[]
 for (rid,identity),fit in fits.items():
  ids=fit['nodes'];p=fit['parent_local'];tau=fit['tau'];f=units.iloc[ids].copy();f['node_id']=ids;f['root_id']=rid;f['pseudotime']=tau/max(max(tau),1e-12);f['potential_distance']=tau;f['is_source_cell']=p<0;nodeframes.append(f)
  for v,u in enumerate(p):
   if u>=0:edges.append({'root_id':rid,'identity':identity,'source':int(ids[u]),'target':int(ids[v]),'state_edge_not_migration':True})
  records.append({'root_id':rid,'identity':identity,'n_units':len(ids),'n_raw_cells':int(units.n_raw_cells.iloc[ids].sum()),'n_source_units':int((p<0).sum()),'n_source_raw_cells':int(units.n_raw_cells.iloc[ids[p<0]].sum()),'edges':int((p>=0).sum())})
 pd.concat(nodeframes,ignore_index=True).to_csv(OUT/'STATE_NODE_ASSIGNMENTS.csv',index=False);pd.DataFrame(edges).to_csv(OUT/'STATE_DIRECTED_EDGES.csv',index=False);pd.DataFrame(records).to_csv(OUT/'FULL_MODEL_AUDIT.csv',index=False)
 (OUT/'AGGREGATION_INVARIANTS.json').write_text(json.dumps({'raw_cell_count':len(obs),'observed_units':len(units),'maximum_unit_raw_cells':mass,'all_313_gene_totals_preserved':True,'every_raw_cell_assigned_exactly_once':True,'all_original_source_ids_preserved':True,'representatives_are_measured_members':True,'root_coordinates_unchanged':True,'old_mapping_used_only_to_measure_original_unit_size_not_group_cells':True,'published_identity_and_stage_are_priors_not_validation':True,'final_release_accepted':False},indent=2))
 print(pd.DataFrame(records).to_string(index=False),flush=True)

if __name__=='__main__':run()
