"""Cluster-coloured overview; relation-coloured participating cells in square zooms."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'algorithm/visualization'))
import redraw_figure2_relationships_v70_20261025 as base
import redraw_figure2_relation_symbols_v70_20261025 as previous
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

OUT=ROOT/'outputs/Figure2_relation_coloured_context_v70_20261025'
STYLE={k:dict(v) for k,v in previous.STYLE.items()}
STYLE['collision']['colour']='#414B57'
STYLE['E1']['colour']='#D68A00'
STYLE['E2']['colour']='#5A3D9A'
GREY='#D4DCE2'

def setup(d):
 if '_fixed_cluster_palette' not in d:
  path=base.shared.ORIGINAL/'data/FIXED_PARTITION_PALETTE.json'
  palette=json.loads(path.read_text(encoding='utf8'))
  cluster=d['cells'].fixed_graph_partition.astype(str).to_numpy()
  assert set(cluster)<=set(palette), 'Every observed cell must retain its fixed cluster colour.'
  d['_fixed_cluster_palette']=palette;d['_fixed_cluster']=cluster
  shutil.copy2(path,OUT/'source_data/FIXED_PARTITION_PALETTE.json')
 return d['_fixed_cluster_palette'],d['_fixed_cluster']

def cluster_cells(ax,d,rows,size,alpha):
 palette,cluster=setup(d)
 for name,colour in palette.items():
  rr=rows[cluster[rows]==name]
  if len(rr):ax.scatter(*d['rawxy'][rr].T,s=size,c=colour,alpha=alpha,lw=0,rasterized=True,zorder=1)

def relationship_map(ax,d,events,routes):
 cluster_cells(ax,d,np.arange(len(d['cells'])),.16,.40)
 for key in base.CASES:
  base.edges(ax,d['xy'],base.shared.segment_edges(d['paths'][key]),base.RC[key[0]],.55,.50)
  q=d['roots'].loc[key[0],['x','y']].to_numpy(float)
  ax.scatter(*q,marker='*',s=26,c=base.RC[key[0]],ec='white',lw=.4,zorder=10)
 base.finish(ax,d)
 for row in events.itertuples():
  for role in ['source','target']:
   key,path=routes[row.event,role]
   base.edges(ax,d['xy'],base.shared.segment_edges([path]),base.RC[key[0]],.85,.95,z=7)
  q=(d['rawxy'][int(row.source_raw_row)]+d['rawxy'][int(row.target_raw_row)])/2
  radius=260 if row.event=='E1' else 130
  ax.add_patch(Rectangle(q-radius,2*radius,2*radius,fc='none',ec=STYLE[row.event]['colour'],lw=.8,ls=(0,(3,2)),zorder=12))
  previous.symbol(ax,d,row,17)
 for rid in base.RC:
  q=d['roots'].loc[rid,['x','y']].to_numpy(float)
  ax.annotate(rid,q,xytext=(4,-8),textcoords='offset points',fontsize=6.1,color=base.RC[rid],weight='bold')
 pd.DataFrame({'raw_row':np.arange(len(d['cells'])),'fixed_graph_partition':d['_fixed_cluster']}).to_csv(OUT/'source_data/D1_CLUSTER_BACKGROUND.csv',index=False)

def eventzoom(ax,d,contacts,events,routes,event):
 row=events.loc[events.event.eq(event)].iloc[0]
 u,v=int(row.source_raw_row),int(row.target_raw_row)
 q=(d['rawxy'][u]+d['rawxy'][v])/2;radius=260 if event=='E1' else 130
 local=np.flatnonzero(np.linalg.norm(d['rawxy']-q,axis=1)<=radius*1.5)
 roles=np.full(len(d['cells']),'unrelated',object)
 for role in ['source','target']:
  key,path=routes[event,role];nodes,_=d['fits'][key]
  rr=d['members'].loc[d['members'].unit_node_id.isin(nodes.node_id),'raw_row'].to_numpy(int)
  assert d['cells'].iloc[rr].identity.eq(key[1]).all()
  roles[rr]=role+': '+key[0]+' '+key[1]
 involved=local[roles[local]!='unrelated'];other=local[roles[local]=='unrelated']
 ax.scatter(*d['rawxy'][other].T,s=.45,c=GREY,alpha=.50,lw=0,rasterized=True,zorder=0)
 ax.scatter(*d['rawxy'][involved].T,s=.85,c=STYLE[event]['colour'],alpha=.65,lw=0,rasterized=True,zorder=1)
 for role in ['source','target']:
  key,path=routes[event,role]
  base.edges(ax,d['xy'],base.shared.segment_edges([path]),base.RC[key[0]],.8,.9)
 for rr,rid,marker in [(u,row.source_root,'s'),(v,row.target_root,'D')]:
  ax.scatter(*d['rawxy'][rr],marker=marker,s=14,c=STYLE[event]['colour'],ec='white',lw=.45,zorder=16)
 previous.symbol(ax,d,row,19)
 base.shared.clean(ax,np.array([q-radius,q+radius]),True,pad=.02);base.scale(ax,100)
 ax.text(.015,.98,f'{row.ligand} / {row.receptor}   {row.distance_um:.1f} µm',transform=ax.transAxes,fontsize=5.1,va='top',color=base.INK,bbox=dict(fc='white',ec='none',pad=.7,alpha=.90))
 palette,cluster=setup(d)
 table=pd.DataFrame({'raw_row':local,'cell_name':d['cells'].iloc[local].cell_name.to_numpy(),'identity':d['cells'].iloc[local].identity.to_numpy(),'fixed_graph_partition':cluster[local],'context_role':roles[local],'display_colour':[STYLE[event]['colour'] if roles[i]!='unrelated' else GREY for i in local]})
 assert (table.loc[table.context_role.eq('unrelated'),'display_colour']==GREY).all()
 assert (table.loc[~table.context_role.eq('unrelated'),'display_colour']==STYLE[event]['colour']).all()
 assert roles[u]!='unrelated' and roles[v]!='unrelated'
 table.to_csv(OUT/f'source_data/{event}_LOCAL_CELL_COLOURS.csv',index=False)

def main():
 for folder in ['source_data','panels','_build']:(OUT/folder).mkdir(parents=True,exist_ok=True)
 previous.OUT=OUT;previous.STYLE=STYLE
 base.OUT=OUT;base.shared.OUT=OUT
 base.relationship_map=relationship_map;base.eventzoom=eventzoom;base.decorate_figure=previous.decorate
 base.RELATION_HEADER_Y=.451
 base.PROMOTION_TITLE='(2) Promotion-like: R0 stromal to R1 B'
 base.INHIBITION_TITLE='(3) Inhibition-like: R2 tumour to R1 CD8 T'
 base.FOOTNOTE='D overview: fixed cluster colours. Zooms: participating source/target cohorts use orange (promotion-like) or violet (inhibition-like); other cells are grey.\nRelation colours identify candidate-associated cohorts, not proven signalling in every cell. Frozen trees, roots and contact positions are unchanged.'
 base.CONTRACT_PATCH={'relation_symbols':STYLE,'D_background':'Unchanged Fig1A fixed graph-partition palette, all observed cells.','D_local_selection':'Exact-member cells in each frozen source and target identity forest use the relation colour; all other local observed cells grey.','layout':'Restore previous square zooms, original contact positions and rotated symbols.'}
 (OUT/'source_data/RELATION_SYMBOL_DEFINITIONS.json').write_text(json.dumps(STYLE,indent=2,ensure_ascii=False),encoding='utf8')
 (OUT/'配色与筛选说明.md').write_text('''# 本次仅修改D区显示

恢复上一版本的布局和方形局部图。促进样候选使用橙色 #D68A00 的箭头；抑制样候选使用深紫色 #5A3D9A 的T形符号；碰撞/交汇使用深灰色 #414B57 的叉号。红、蓝、绿仍仅表示三棵树，不新增碰撞事件。

左侧全部实测细胞按Fig1A固定聚类及原始64簇色表着色，不重新聚类。右侧彩色细胞只来自对应关系的冻结源/靶细胞群：促进图的R0基质细胞和R1 B细胞统一橙色；抑制图的R2恶性上皮和R1 CD8 T细胞统一深紫色。其他局部细胞一律灰色，不再沿用聚类色。树线保留各自root颜色。彩色群中的每个细胞并不都被认定为发生了信号交流。方形/菱形分别标出实际配体/受体阳性接触细胞，同样使用关系色。

根、树边、细胞位置、接触位置、距离和冻结统计均未改变。局部细胞筛选表和固定色表附在source_data，可逐细胞核对。促进和抑制仍为实测近邻支持的候选，而不是已验证因果关系。
''',encoding='utf8')
 (OUT/'Figure2_caption_addendum.tex').write_text(r'\textbf{D}, Overview cells retain fixed graph-partition colours, as in Fig.~1A. In each magnification, exact-member cells of the participating source and target forests share the relation colour: orange for the promotion-like candidate and dark violet for the inhibition-like candidate. Other observed cells are grey; tree edges retain their root colours. Squares and diamonds mark measured ligand-positive and receptor-positive cells. Coloured cohorts indicate candidate-associated populations, not established signalling in every cell. Grey $\times$, orange $\rightarrow$ and violet $\dashv$ distinguish collision/crossing, promotion-like and inhibition-like classes; no independently called collision event is added. Frozen roots, edges and contact positions are unchanged.'+'\n',encoding='utf8')
 shutil.copy2(Path(__file__),OUT/Path(__file__).name)
 shutil.copy2(Path(previous.__file__),OUT/Path(previous.__file__).name)
 base.main()

if __name__=='__main__':main()
