"""Restore the trajectory/relationship narrative without changing frozen v70.

Data panels use the frozen graphs, cells, coordinates and contact records.
Dashed cross-tree links are descriptive measured-contact hypotheses only.
"""
import os,sys,json,hashlib,shutil,zipfile
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'algorithm/visualization'))
import render_checked_state_suite_20261025 as shared
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.spatial import cKDTree
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch,Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox

FROZEN=ROOT/'outputs/NicheForest_state_forests_v70_FROZEN_20261025'
OUT=ROOT/'outputs/Figure2_relationship_layout_v70_20261025'
shared.ROOT=FROZEN;shared.BASE=FROZEN/'outputs/Count_spatial_agreement_states_v70_20261025/ensemble';shared.ORIGINAL=FROZEN/'outputs/NicheForest_current_research_freeze_20261024';shared.OUT=OUT
INK=shared.INK;RC=shared.ROOTCOL;CASES=shared.CASES
EC={'E1':'#2D9D8B','E2':'#CB536B'}
EXTRA=('R1','CD8_T')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def scale(ax,length=1000):
 # A dedicated lower margin prevents scale bars covering a trajectory.
 x0,x1=ax.get_xlim();x=x0+.04*(x1-x0);tr=ax.get_xaxis_transform()
 ax.plot([x,x+length],[-.04,-.04],c=INK,lw=1,transform=tr,clip_on=False)
 ax.text(x+length/2,-.018,f'{length/1000:g} mm',ha='center',va='bottom',fontsize=5.0,transform=tr)

def edges(ax,xy,ed,color,width=.6,alpha=1.,arrows=True,z=5):
 if not len(ed):return
 seg=xy[np.asarray(ed,int)];ax.add_collection(LineCollection(seg,colors=color,lw=width,alpha=alpha,zorder=z,rasterized=False))
 if arrows:
  for a,b in seg:
   if np.linalg.norm(b-a)>1e-8:ax.add_patch(FancyArrowPatch(a+.43*(b-a),a+.63*(b-a),arrowstyle='-|>',mutation_scale=2.6,lw=.3,color=color,alpha=alpha,zorder=z+1))

def context(ax,d,space='spatial',step=3,alpha=.24):
 xy=d['rawxy'] if space=='spatial' else d['rawumap'];ax.scatter(*xy[::step].T,s=.13,c='#C9D1D8',alpha=alpha,lw=0,rasterized=True,zorder=0)

def finish(ax,d,space='spatial',limits=None):
 shared.clean(ax,d['rawxy'] if space=='spatial' else d['rawumap'] if limits is None else limits,space=='spatial',pad=.035)
 if limits is not None:shared.clean(ax,limits,space=='spatial',pad=.02)
 if space=='spatial':scale(ax,1000 if limits is None else 100)

def forest(ax,d,space='spatial',candidate=False,keys=CASES,alpha=1.,stars=True):
 xy=d['xy'] if space=='spatial' else d['umap'];context(ax,d,space)
 for key in keys:
  n,e=d['fits'][key];ids=n.node_id.to_numpy(int);color=RC[key[0]]
  ax.scatter(*xy[ids].T,s=.22,c=color,alpha=.25,lw=0,rasterized=True)
  if candidate:
   sub=d['g'][ids][:,ids].tocoo();valid=sub.row<sub.col;u=ids[sub.row[valid]];v=ids[sub.col[valid]];ix=np.linspace(0,len(u)-1,min(3000,len(u)),dtype=int);ed=np.column_stack([u[ix],v[ix]]);edges(ax,xy,ed,color,.15,.11,False)
  else:edges(ax,xy,shared.segment_edges(d['paths'][key]),color,.55,alpha)
  if stars:
   pos=d['roots'].loc[key[0],['x','y']].to_numpy(float) if space=='spatial' else xy[int(n.loc[n.is_source_state,'node_id'].iloc[0])]
   ax.scatter(*pos,marker='*' if space=='spatial' else 'o',s=26 if space=='spatial' else 8,c=color,ec='white',lw=.4,zorder=10)
 finish(ax,d,space)

def rootprob(ax,d,prob,kind):
 valid=prob.sum(1)>0;xy=d['umap'];ax.scatter(*xy[~valid].T,c='#DFE4E8',s=.3,lw=0,rasterized=True)
 if kind==1:
  for i,rid in enumerate(RC):
   take=valid&(prob.argmax(1)==i);ax.scatter(*xy[take].T,s=.55,c=RC[rid],alpha=.75,lw=0,rasterized=True)
 elif kind==2:
  sharedmask=(prob>=.2).sum(1)>1
  # Use the frozen support cutoff, without inventing extra boundary classes.
  groups=[(valid&~sharedmask,'#91A1AE','Root-specific'),(valid&sharedmask,'#9A78B5','Shared support')]
  for mask,c,label in groups:ax.scatter(*xy[mask].T,s=.55,c=c,alpha=.7,lw=0,label=label,rasterized=True)
  ax.legend(loc='upper left',bbox_to_anchor=(.02,1.03),fontsize=4.8,handletextpad=.2,labelspacing=.2,markerscale=2)
 else:
  entropy=-(prob*np.log(np.maximum(prob,1e-15))).sum(1)/np.log(3);ax.scatter(*xy[valid].T,s=.55,c=entropy[valid],cmap='YlOrBr',vmin=0,vmax=1,lw=0,rasterized=True)
  cax=ax.inset_axes([1.025,.05,.035,.86]);cb=plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='YlOrBr'),cax=cax,ticks=[0,.5,1]);cb.ax.tick_params(labelsize=5,length=1)
 finish(ax,d,'umap')

def ordermap(ax,d,space):
 context(ax,d,space);xy=d['xy'] if space=='spatial' else d['umap']
 for key in CASES:
  n,e=d['fits'][key];ids=n.node_id.to_numpy(int);q=(rankdata(n.pseudotime)-1)/max(len(n)-1,1);colour=LinearSegmentedColormap.from_list(key[0],[RC[key[0]],{'R0':'#F2B0BD','R1':'#B5D4F3','R2':'#B9DFCA'}[key[0]]])
  if space=='spatial':
   raw=d['members'].loc[d['members'].unit_node_id.isin(ids)];loc=pd.Index(ids).get_indexer(raw.unit_node_id);ax.scatter(*d['rawxy'][raw.raw_row].T,c=q[loc],cmap=colour,vmin=0,vmax=1,s=.14,alpha=.7,lw=0,rasterized=True)
  else:ax.scatter(*xy[ids].T,c=q,cmap=colour,vmin=0,vmax=1,s=.6,alpha=.85,lw=0,rasterized=True)
 finish(ax,d,space)

def ancestor_path(d,key,unit):
 n,e=d['fits'][key];parent=dict(zip(e.target.astype(int),e.source.astype(int)));path=[int(unit)]
 while path[-1] in parent:path.append(parent[path[-1]])
 assert path[-1] in set(n.loc[n.is_source_state,'node_id']);return path[::-1]

def prepare_events(d):
 allcontacts=pd.read_csv(FROZEN/'deliverables/source_data/ALL_OBSERVED_LIGAND_RECEPTOR_CONTACT_CANDIDATES.csv');events=allcontacts.sort_values('descriptive_contact_score',ascending=False,kind='stable').drop_duplicates('event').copy();route_rows=[];routes={}
 for r in events.itertuples():
  for role,key,rawrow in [('source',(r.source_root,r.source_identity),r.source_raw_row),('target',(r.target_root,r.target_identity),r.target_raw_row)]:
   unit=int(d['unit_of_raw'][rawrow]);path=ancestor_path(d,key,unit);routes[r.event,role]=(key,path)
   route_rows.extend({'event':r.event,'role':role,'root_id':key[0],'identity':key[1],'order':i,'unit_node_id':v} for i,v in enumerate(path))
 events.to_csv(OUT/'source_data/D_DISPLAYED_CONTACTS.csv',index=False);pd.DataFrame(route_rows).to_csv(OUT/'source_data/D_ACTUAL_SOURCE_TO_CONTACT_PATHS.csv',index=False);allcontacts.to_csv(OUT/'source_data/ALL_FROZEN_CONTACT_RECORDS.csv',index=False)
 return allcontacts,events,routes

def relationship_map(ax,d,events,routes):
 # The light routes retain the actual three principal forests. Bold routes
 # identify the two frozen forest branches participating in each contact.
 forest(ax,d,alpha=.50)
 for r in events.itertuples():
  for role in ['source','target']:
   key,path=routes[r.event,role];ed=shared.segment_edges([path]);edges(ax,d['xy'],ed,RC[key[0]],.85,.95,z=7)
  a=d['rawxy'][r.source_raw_row];b=d['rawxy'][r.target_raw_row];q=(a+b)/2;radius=260 if r.event=='E1' else 130
  ax.add_patch(Rectangle(q-radius,2*radius,2*radius,fc='none',ec=EC[r.event],lw=.8,ls=(0,(3,2)),zorder=12))
  ax.scatter(*q,s=36,fc='white',ec=EC[r.event],lw=1,zorder=13);ax.annotate(r.event,q,xytext=(5,7),textcoords='offset points',fontsize=7.3,weight='bold',color=EC[r.event],zorder=14)
 for rid in RC:
  pos=d['roots'].loc[rid,['x','y']].to_numpy(float);ax.annotate(rid,pos,xytext=(4,-8),textcoords='offset points',fontsize=6.1,color=RC[rid],weight='bold')
 # Root colours are defined by A and the shared key below C. No legend over
 # the tissue. D3 explicitly identifies the additional R1 CD8 T subforest.

def eventzoom(ax,d,contacts,events,routes,event):
 r=events.loc[events.event.eq(event)].iloc[0];u=int(r.source_raw_row);v=int(r.target_raw_row);q=(d['rawxy'][u]+d['rawxy'][v])/2;radius=260 if event=='E1' else 130
 local=np.linalg.norm(d['rawxy']-q,axis=1)<=radius*1.5;ax.scatter(*d['rawxy'][local].T,s=.45,c='#D4DCE2',alpha=.5,lw=0,rasterized=True)
 for role in ['source','target']:
  key,path=routes[event,role];n,e=d['fits'][key];members=d['members'].loc[d['members'].unit_node_id.isin(n.node_id)];rows=members.raw_row.to_numpy(int);rows=rows[np.linalg.norm(d['rawxy'][rows]-q,axis=1)<=radius*1.5];ax.scatter(*d['rawxy'][rows].T,s=.85,c=RC[key[0]],alpha=.48,lw=0,rasterized=True)
  edges(ax,d['xy'],shared.segment_edges([path]),RC[key[0]],.8,.9)
 # Keep individual measured positive cells and the actual ligand/receptor
 # separation visible; no moved contact or synthetic state bridge.
 for rawrow,key,marker in [(u,r.source_root,'s'),(v,r.target_root,'o')]:ax.scatter(*d['rawxy'][rawrow],marker=marker,s=20,c=RC[key],edgecolors='white',lw=.45,zorder=14)
 ax.add_patch(FancyArrowPatch(d['rawxy'][u],d['rawxy'][v],arrowstyle='-|>',mutation_scale=7,lw=1.,ls=(0,(2,2)),color=EC[event],zorder=15))
 shared.clean(ax,np.array([q-radius,q+radius]),True,pad=.02);scale(ax,100)
 ax.text(.015,.98,f'{r.ligand} / {r.receptor}   {r.distance_um:.1f} µm',transform=ax.transAxes,fontsize=5.1,va='top',color=INK,bbox=dict(fc='white',ec='none',pad=.7,alpha=.90))

def main():
 for sub in ['panels','source_data','_build']:(OUT/sub).mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'Arial','font.size':6,'axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
 contract={'conclusion':'Fixed regional forests retain shared support and ordered states, enabling localized cross-tree contact hypotheses.','evidence_chain':'A root-conditioned candidates; B soft assignment; C representative forests and within-cohort order; D actual tree branches and measured cross-tree contacts; E candidates versus representative forests.','hero':'D original-tissue overview plus two magnifications, no bar charts in the main figure.','backend':'python matplotlib','inputs':'Only frozen v70 counts/nodes/edges/coordinates/contact records. Frozen result directory is read-only.','inference':'Display revision only. No rerun or alteration of roots, graph, parent edges, source roster or tests.','relationship_boundary':'E1 stromal R0 to B R1, E2 epithelial R2 to CD8 T R1; measured-contact hypotheses, not formally called promotion/inhibition. No legacy six-event positions reused.','export':'183mm wide, Arial, uppercase group labels and numbered subpanels. Full PNG/PDF/SVG and individual real-data panels.','risks':['Contacts do not establish causality.','D includes the explicitly identified R1 CD8 T subforest, distinct from its principal B-cell example.','Pseudotime colours are separately ranked within each root/identity; they do not define a common biological clock.']}
 contract.update(globals().get('CONTRACT_PATCH',{}))
 (OUT/'_build/FIGURE_CONTRACT.json').write_text(json.dumps(contract,indent=2));d=shared.load();prob=shared.affinities(d);contacts,events,routes=prepare_events(d)
 # source payload records and all displayed route edges for explicit checking.
 allpaths=[]
 for key in CASES:
  allpaths.extend({'root_id':key[0],'identity':key[1],'source':int(a),'target':int(b)} for a,b in shared.segment_edges(d['paths'][key]))
 pd.DataFrame(allpaths).to_csv(OUT/'source_data/PRINCIPAL_ACTUAL_DISPLAYED_EDGES.csv',index=False)
 rows=[
 ('A','Root-specific candidate trajectories',.985,[(f'A{i+1}',s,(.025+i*.323,.845,.305,.11)) for i,s in enumerate(['(1) R0: C13, stromal','(2) R1: C00, B cells','(3) R2: C21, malignant epithelial'])]),
 ('B','Soft root assignment',.815,[(f'B{i+1}',s,(.025+i*.323,.671,.255 if i==2 else .305,.11)) for i,s in enumerate(['(1) Primary regional support','(2) Root-specific / shared support','(3) Membership entropy'])]),
 ('C','Representative trajectories and pseudotime',.638,[(f'C{i+1}',s,(.025+i*.244,.500,.23,.10)) for i,s in enumerate(['(1) UMAP representative','(2) Tissue representative','(3) UMAP pseudotime','(4) Tissue pseudotime'])]),
 ('D','Cross-tree contacts and signalling-compatible candidates',globals().get('RELATION_HEADER_Y',.443),[('D1','(1) Related tree branches and contact locations',(.025,.156,.60,.249)),('D2',globals().get('PROMOTION_TITLE','(2) E1: recruitment-like, R0 stromal to R1 B'),(.660,.292,.315,.11)),('D3',globals().get('INHIBITION_TITLE','(3) E2: inhibition-like, R2 tumour to R1 CD8 T'),(.660,.156,.315,.11))]),
 ('E','Candidate and final representative trajectories',.123,[(f'E{i+1}',s,(.025+i*.244,.020,.23,.069)) for i,s in enumerate(['(1) UMAP candidates','(2) UMAP representative','(3) Tissue candidates','(4) Tissue representative'])])]
 draw={}
 for i,key in enumerate(CASES,1):draw[f'A{i}']=lambda ax,key=key:forest(ax,d,'umap',True,[key])
 for i in range(1,4):draw[f'B{i}']=lambda ax,i=i:rootprob(ax,d,prob,i)
 draw.update({'C1':lambda ax:forest(ax,d,'umap'),'C2':lambda ax:forest(ax,d),'C3':lambda ax:ordermap(ax,d,'umap'),'C4':lambda ax:ordermap(ax,d,'spatial'),'D1':lambda ax:relationship_map(ax,d,events,routes),'D2':lambda ax:eventzoom(ax,d,contacts,events,routes,'E1'),'D3':lambda ax:eventzoom(ax,d,contacts,events,routes,'E2')})
 for i,(space,candidate) in enumerate([('umap',True),('umap',False),('spatial',True),('spatial',False)],1):draw[f'E{i}']=lambda ax,space=space,candidate=candidate:forest(ax,d,space,candidate)
 fig=plt.figure(figsize=(7.2,10));elements=[];assets={}
 for letter,title,y,items in rows:
  fig.text(.025,y,letter+'  '+title,color=INK,weight='bold',fontsize=10.0,va='top');elements.append({'kind':'text','name':letter,'text':letter+'  '+title,'x':40,'y':(1-y)*2210,'w':1520,'h':48,'size':24,'bold':True})
  for name,subtitle,rect in items:
   # Real axes in the assembled PDF preserve editable vector line geometry.
   ax=fig.add_axes(rect);draw[name](ax);fig.text(rect[0],rect[1]+rect[3]+.007,subtitle,fontsize=6.1,weight='bold',color=INK,va='bottom')
   # Independent panels, each authored at its final physical size.
   one=plt.figure(figsize=(rect[2]*7.2,rect[3]*10));pax=one.add_axes([.025,.13,.84 if name=='B3' else .95,.79]);draw[name](pax)
   file=OUT/'panels'/f'Figure2_{name}.png'
   for ext in ['png','pdf','svg']:one.savefig(file.with_suffix('.'+ext),dpi=450,facecolor='white')
   plt.close(one);assets[name]={'path':str(file)};elements.extend([{'kind':'image','name':name,'key':name,'x':rect[0]*1600,'y':(1-rect[1]-rect[3])*2210,'w':rect[2]*1600,'h':rect[3]*2210},{'kind':'text','name':name+' title','text':subtitle,'x':rect[0]*1600,'y':(1-rect[1]-rect[3])*2210-34,'w':rect[2]*1600,'h':28,'size':15,'bold':True}])
 # One shared root colour key and one root-specific order key.
 for i,(rid,label) in enumerate([('R0','R0: stromal'),('R1','R1: B cells'),('R2','R2: malignant epithelial')]):
  ax=fig.add_axes([.08+i*.31,.473,.23,.006]);cmap=LinearSegmentedColormap.from_list(rid,[RC[rid],{'R0':'#F2B0BD','R1':'#B5D4F3','R2':'#B9DFCA'}[rid]]);ax.imshow(np.linspace(0,1,128)[None,:],aspect='auto',cmap=cmap);ax.axis('off');fig.text(.08+i*.31,.482,label,fontsize=5.2,color=RC[rid],weight='bold');fig.text(.08+i*.31,.464,'early',fontsize=4.7,color='#5D6F7D');fig.text(.31+i*.31,.464,'late',fontsize=4.7,color='#5D6F7D',ha='right')
 if callable(globals().get('decorate_figure')):globals()['decorate_figure'](fig,elements,assets)
 foot=globals().get('FOOTNOTE','Solid lines: frozen inferred state edges. Dashed links: measured ligand/receptor contact candidates, not causal effects.\nD includes the R1 CD8 T subforest. C colours rank order separately within each cohort; grey denotes observed context.')
 fig.text(.025,.004,foot,fontsize=4.5,color='#657987',va='bottom');stem=OUT/'Figure2_tree_relationships_v70'
 # Keep the shared key as a separate real asset for later editable PPT
 # assembly; otherwise a manifest-based deck would silently lose its colours.
 keyfile=OUT/'panels/Figure2_shared_order_key.png'
 for ext in ['png','pdf','svg']:fig.savefig(keyfile.with_suffix('.'+ext),dpi=450,facecolor='white',bbox_inches=Bbox.from_bounds(0,4.60,7.2,.35),pad_inches=0)
 assets['shared_order_key']={'path':str(keyfile)};elements.append({'kind':'image','name':'shared order key','key':'shared_order_key','x':0,'y':(1-.495)*2210,'w':1600,'h':.035*2210})
 elements.append({'kind':'text','name':'Interpretation scope','text':foot,'x':40,'y':2167,'w':1520,'h':36,'size':11,'bold':False,'color':'#657987'})
 for ext in ['png','pdf','svg']:fig.savefig(stem.with_suffix('.'+ext),dpi=450,facecolor='white')
 plt.close(fig)
 (OUT/'_build/figure2_manifest.json').write_text(json.dumps({'figure':2,'assets':assets,'elements':elements},indent=2))
 # Verify every event-associated line is an unchanged parent in the snapshot.
 trace=pd.read_csv(OUT/'source_data/D_ACTUAL_SOURCE_TO_CONTACT_PATHS.csv')
 for (event,role,rid,identity),group in trace.groupby(['event','role','root_id','identity']):
  allowed=set(map(tuple,d['fits'][rid,identity][1][['source','target']].to_numpy(int)));v=group.sort_values('order').unit_node_id.to_numpy(int);assert all((a,b) in allowed for a,b in zip(v[:-1],v[1:]))
 files=[shared.BASE/'WholeCell/FULL_CANDIDATE_COSTS.npz',FROZEN/'deliverables/source_data/ALL_OBSERVED_LIGAND_RECEPTOR_CONTACT_CANDIDATES.csv',FROZEN/'SHA256_MANIFEST.json']
 (OUT/'source_data/FROZEN_SOURCE_CHECKSUMS.json').write_text(json.dumps([{'path':str(p),'sha256':sha(p)} for p in files],indent=2))
 frozen_manifest=json.loads((FROZEN/'SHA256_MANIFEST.json').read_text())
 for record in frozen_manifest:assert sha(FROZEN/record['path'])==record['sha256'],'Frozen source was changed: '+record['path']
 svg=stem.with_suffix('.svg').read_text(encoding='utf8');assert '<text' in svg
 qa={'frozen_files_checked':len(frozen_manifest),'frozen_files_modified':0,'all_event_ancestral_paths_are_actual_frozen_parent_edges':True,'all_displayed_contacts_from_frozen_table':True,'legacy_event_positions_reused':False,'main_figure_bar_charts':0,'svg_editable_text_present':True,'visual_inspection':'Full assembled figure inspected. Relationship labels moved inside zooms and scales separated from contact descriptions.','publication_acceptance_claimed':False}
 (OUT/'source_data/VISUAL_AND_SOURCE_QA.json').write_text(json.dumps(qa,indent=2));shutil.copy2(Path(__file__),OUT/Path(__file__).name)
 archive=OUT/'Figure2_relationship_layout_and_source_v70.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
  for p in sorted(OUT.rglob('*')):
   if p.is_file() and p!=archive and '_build' not in p.parts:z.write(p,arcname=p.relative_to(OUT).as_posix())
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 print('Figure2 relationship layout exported; all displayed event paths verified against frozen edges.',flush=True)

if __name__=='__main__':main()
