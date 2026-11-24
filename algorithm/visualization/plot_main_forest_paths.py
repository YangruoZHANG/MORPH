"""One principal edge table and renderer for Fig1 and Fig2; separate contact helpers."""
import sys,json,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'algorithm/visualization'))
import plot_forest_contacts as base
import plot_forest_cluster_context as colour
import plot_contact_symbols as symbols
import plot_breast_figures as normal
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch,Rectangle
from matplotlib.lines import Line2D

OUT=ROOT/'outputs/Figures_124_unified_trajectories'
OLD=ROOT/'outputs/Figures_124_normal_orientation'
FIG2=OUT/'figure2';AUX='#697781'
original_forest=base.forest
original_zoom=colour.eventzoom

def principal(d):
 if '_unified_principal' not in d:
  source=pd.read_csv(OLD/'figure1/source_data/EXACT_DISPLAYED_FOREST_PATHS.csv');rows=[]
  for (rid,identity,pid),g in source.groupby(['root_id','identity','path_id']):
   v=g.sort_values('order').unit_node_id.to_numpy(int);rows.extend((rid,identity,int(u),int(w)) for u,w in zip(v[:-1],v[1:]))
  table=pd.DataFrame(sorted(set(rows)),columns=['root_id','identity','source','target'])
  for key in base.CASES:
   expected=set(map(tuple,table.loc[table.root_id.eq(key[0])&table.identity.eq(key[1]),['source','target']].to_numpy(int)))
   assert expected==set(map(tuple,base.shared.segment_edges(d['paths'][key])))
  d['_unified_principal']=table;d['_main_edge_pairs']=set(map(tuple,table[['source','target']].to_numpy(int)))
  table.to_csv(FIG2/'source_data/SHARED_FIG1_FIG2_PRINCIPAL_EDGES.csv',index=False)
 return d['_unified_principal']

def unified_forest(ax,d,space='spatial',candidate=False,keys=base.CASES,alpha=1.,stars=True):
 principal(d)
 if not candidate and list(keys)==list(base.CASES):
  # Same function, halo, arrows, anchors, crop and widths as Fig1B.
  base.shared.overview(ax,d,space)
 else:original_forest(ax,d,space,candidate,keys,alpha,stars)

def auxiliary(ax,d,ed):
 ed=np.asarray(ed,int).reshape(-1,2)
 if not len(ed):return
 seg=d['xy'][ed]
 ax.add_collection(LineCollection(seg,colors='white',lw=1.4,zorder=3))
 ax.add_collection(LineCollection(seg,colors=AUX,lw=.55,linestyles=(0,(3,2)),alpha=.90,zorder=4))
 for a,b in seg:
  if np.linalg.norm(b-a)>1e-8:ax.add_patch(FancyArrowPatch(a+.44*(b-a),a+.62*(b-a),arrowstyle='-|>',mutation_scale=2.8,lw=.32,color=AUX,zorder=4))

def map_relations(ax,d,events,routes):
 table=principal(d);colour.cluster_cells(ax,d,np.arange(len(d['cells'])),.16,.40)
 extra=[]
 for row in events.itertuples():
  for role in ['source','target']:
   key,path=routes[row.event,role];ed=base.shared.segment_edges([path])
   subset=table.loc[table.root_id.eq(key[0])&table.identity.eq(key[1]),['source','target']]
   main=set(map(tuple,subset.to_numpy(int)))
   extra.extend({'root_id':key[0],'identity':key[1],'source':int(u),'target':int(v)} for u,v in ed if (u,v) not in main)
 aux=pd.DataFrame(extra).drop_duplicates().sort_values(['root_id','identity','source','target'])
 aux.to_csv(FIG2/'source_data/D_SEPARATE_AUXILIARY_EDGES.csv',index=False)
 auxiliary(ax,d,aux[['source','target']].to_numpy(int))
 for key in base.CASES:
  ed=table.loc[table.root_id.eq(key[0])&table.identity.eq(key[1]),['source','target']].to_numpy(int)
  base.shared.lines(ax,d['xy'],ed,base.RC[key[0]],.75,d['scale'])
  q=d['roots'].loc[key[0],['x','y']].to_numpy(float)
  ax.scatter(*q,marker='*',s=45,c=base.RC[key[0]],ec='white',lw=.5,zorder=10)
  ax.annotate(key[0],q,xytext=(4,-8),textcoords='offset points',fontsize=6.1,color=base.RC[key[0]],weight='bold')
 base.shared.clean(ax,d['rawxy'],True);base.scale(ax,1000)
 for row in events.itertuples():
  q=(d['rawxy'][int(row.source_raw_row)]+d['rawxy'][int(row.target_raw_row)])/2;radius=260 if row.event=='E1' else 130
  ax.add_patch(Rectangle(q-radius,2*radius,2*radius,fc='none',ec=colour.STYLE[row.event]['colour'],lw=.8,ls=(0,(3,2)),zorder=12))
  symbols.symbol(ax,d,row,17)
 ax.legend(handles=[Line2D([],[],c=AUX,lw=.55,ls=(0,(3,2)),label='Extra contact-support paths (including R1 CD8 T)')],loc='lower right',bbox_to_anchor=(1,-.10),fontsize=4.1,handlelength=2.6,handletextpad=.5)

def zoom(ax,d,contacts,events,routes,event):
 principal(d);save=base.edges
 def split_edges(ax2,xy,ed,color,width=.6,alpha=1.,arrows=True,z=5):
  ed=np.asarray(ed,int).reshape(-1,2);mask=np.array([tuple(v) in d['_main_edge_pairs'] for v in ed],bool)
  auxiliary(ax2,d,ed[~mask]);base.shared.lines(ax2,xy,ed[mask],color,.75,d['scale'])
 base.edges=split_edges
 try:original_zoom(ax,d,contacts,events,routes,event)
 finally:base.edges=save
 if event=='E2':ax.text(.015,.90,'Grey dashed: R1 CD8 T support path',transform=ax.transAxes,fontsize=4.1,color=AUX,va='top',bbox=dict(fc='white',ec='none',pad=.6,alpha=.9))

def main():
 for sub in ['panels','source_data','_build']:(FIG2/sub).mkdir(parents=True,exist_ok=True)
 # Preserve the approved Fig1 and Fig4 byte-for-byte, including their panels.
 for n in [1,4]:shutil.copytree(OLD/f'figure{n}',OUT/f'figure{n}',dirs_exist_ok=True)
 base.shared.clean=normal.normal_clean;base.shared.bar=normal.normal_bar
 colour.OUT=FIG2;symbols.OUT=FIG2;symbols.STYLE=colour.STYLE;symbols.symbol=normal.normal_symbol
 base.OUT=FIG2;base.shared.OUT=FIG2;base.forest=unified_forest;base.relationship_map=map_relations;base.eventzoom=zoom;base.decorate_figure=symbols.decorate
 base.RELATION_HEADER_Y=.451;base.PROMOTION_TITLE='(2) Promotion-like: R0 stromal to R1 B';base.INHIBITION_TITLE='(3) Inhibition-like: R2 tumour to R1 CD8 T'
 base.FOOTNOTE='Coloured principal trajectories use exactly the same 283 edges and renderer as Fig1. Grey dashed lines are separate contact-support paths, including R1 CD8 T.\nD overview: fixed cluster colours. Zooms: orange / violet for candidate-associated cohorts, other cells grey. Candidates are not causal effects. Tissue y is not inverted.'
 base.CONTRACT_PATCH={'principal_contract':'Direct comparison to Fig1 exact path table:283 edges, same renderer overview/lines, identical roots/UMAP/original coordinates.','D_layers':'283 coloured main edges;24 additional measured contact-support parent edges separately grey/dashed, including11 R1 CD8 T edges. They are not blue B-cell main-trajectory edges.','orientation':'Ascending original-tissue y retained; R2 bottom-right.','relation_colours':colour.STYLE,'science':'No inference, parent selection, cell selection, functional test or root source is rerun.'}
 shutil.copy2(Path(__file__),FIG2/Path(__file__).name)
 base.main()
 a=pd.read_csv(FIG2/'source_data/SHARED_FIG1_FIG2_PRINCIPAL_EDGES.csv');b=pd.read_csv(FIG2/'source_data/PRINCIPAL_ACTUAL_DISPLAYED_EDGES.csv')
 assert set(map(tuple,a.to_numpy()))==set(map(tuple,b.to_numpy()))
 aux=pd.read_csv(FIG2/'source_data/D_SEPARATE_AUXILIARY_EDGES.csv')
 assert len(a)==283 and len(aux)==24
 assert len(aux.loc[aux.root_id.eq('R1')&aux.identity.eq('CD8_T')])==11
 qa={'main_edges_fig1_and_fig2':len(a),'main_edge_sets_identical':True,'same_main_renderer':'render_checked_state_suite.overview and lines','D_auxiliary_edges_separate':len(aux),'D_CD8_T_helper_edges_not_blue_B_main':11,'raw_coordinates_and_roots_unchanged':True,'new_inference_or_enrichment':False,'figure1_and_figure4_files_unchanged':True}
 for n in [1,4]:
  for ext in ['png','pdf','svg']:
   name=f'Figure{n}_checked_state_forests.{ext}' if n==1 else f'Figure4_root_conditioned_biological_enrichment.{ext}'
   assert hashlib.sha256((OLD/f'figure{n}'/name).read_bytes()).digest()==hashlib.sha256((OUT/f'figure{n}'/name).read_bytes()).digest()
 (OUT/'UNIFIED_TRAJECTORY_QA.json').write_text(json.dumps(qa,indent=2))
 (OUT/'说明_轨迹统一.md').write_text('Fig1与Fig2的主轨迹统一为同一份283条边，使用同一绘图函数、线宽、描边、箭头与坐标。Fig2D额外24条实测接触定位路径独立为灰色虚线，其中11条属于R1来源CD8 T森林，不再混入蓝色B细胞主轨迹。Fig2A/E候选图仍明确为候选，不等同于最终代表图。Fig1、Fig4和全部冻结结果未修改。原始切片维持正常方向，R2右下。\n',encoding='utf8')
 print('COMPLETE',OUT,flush=True)
if __name__=='__main__':main()
