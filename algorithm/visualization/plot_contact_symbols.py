"""Three relation symbols, without inventing a third frozen event class."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'algorithm/visualization'))
import plot_forest_contacts as base
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

OUT=ROOT/'outputs/Figure2_relation_symbols'
STYLE={
 'collision':{'symbol':r'$\times$','unicode':'×','colour':'#8D68AD','label':'Collision / crossing'},
 'E1':{'symbol':r'$\rightarrow$','unicode':'→','colour':'#2D9D8B','label':'Promotion-like candidate'},
 'E2':{'symbol':r'$\dashv$','unicode':'⊣','colour':'#CB536B','label':'Inhibition-like candidate'}}
ORIGINAL_ZOOM=base.eventzoom

def symbol(ax,d,row,size):
 a=d['rawxy'][int(row.source_raw_row)];b=d['rawxy'][int(row.target_raw_row)];q=(a+b)/2
 # The axis inverts tissue y. Rotate only the symbol to follow the actual
 # ligand-to-receptor direction; its measured position stays fixed.
 delta=b-a;angle=float(np.degrees(np.arctan2(-delta[1],delta[0])))
 style=STYLE[row.event]
 ax.text(*q,style['symbol'],color=style['colour'],fontsize=size,ha='center',va='center',rotation=angle,rotation_mode='anchor',weight='bold',zorder=20,path_effects=[pe.withStroke(linewidth=2.1,foreground='white')])

def relationship_map(ax,d,events,routes):
 base.forest(ax,d,alpha=.5)
 for row in events.itertuples():
  for role in ['source','target']:
   key,path=routes[row.event,role];base.edges(ax,d['xy'],base.shared.segment_edges([path]),base.RC[key[0]],.85,.95,z=7)
  a=d['rawxy'][row.source_raw_row];b=d['rawxy'][row.target_raw_row];q=(a+b)/2;radius=260 if row.event=='E1' else 130
  ax.add_patch(Rectangle(q-radius,2*radius,2*radius,fc='none',ec=STYLE[row.event]['colour'],lw=.8,ls=(0,(3,2)),zorder=12))
  symbol(ax,d,row,17)
 for rid in base.RC:
  q=d['roots'].loc[rid,['x','y']].to_numpy(float);ax.annotate(rid,q,xytext=(4,-8),textcoords='offset points',fontsize=6.1,color=base.RC[rid],weight='bold')

def eventzoom(ax,d,contacts,events,routes,event):
 ORIGINAL_ZOOM(ax,d,contacts,events,routes,event)
 # Only remove the old contact arrow and its two highlighted raw-cell
 # markers. Retain the actual tree arrows, observations and label positions.
 ax.patches[-1].remove();ax.collections[-1].remove();ax.collections[-1].remove()
 row=events.loc[events.event.eq(event)].iloc[0]
 for rawrow,rid,marker in [(int(row.source_raw_row),row.source_root,'s'),(int(row.target_raw_row),row.target_root,'D')]:ax.scatter(*d['rawxy'][rawrow],marker=marker,s=14,c=base.RC[rid],ec='white',lw=.45,zorder=16)
 symbol(ax,d,row,19)

def decorate(fig,elements,assets):
 for i,key in enumerate(['collision','E1','E2']):
  x=.033+i*.322;st=STYLE[key]
  fig.text(x,.428,st['symbol'],fontsize=10.5,color=st['colour'],ha='left',va='center')
  fig.text(x+.027,.428,st['label'],fontsize=5.5,color=base.INK,ha='left',va='center')
 # Keep the key as an independent editable/vector asset for future assembly.
 keyfile=OUT/'panels/Figure2_relation_symbol_key.png'
 for ext in ['png','pdf','svg']:fig.savefig(keyfile.with_suffix('.'+ext),dpi=450,facecolor='white',bbox_inches=Bbox.from_bounds(0,4.20,7.2,.16),pad_inches=0)
 assets['relation_symbol_key']={'path':str(keyfile)};elements.append({'kind':'image','name':'three relation symbol key','key':'relation_symbol_key','x':0,'y':(1-.436)*2210,'w':1600,'h':.016*2210})
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'Figure2D_three_relation_symbols.{ext}',dpi=450,facecolor='white',bbox_inches=Bbox.from_bounds(.02*7.2,1.40,.96*7.2,3.20),pad_inches=0)

def main():
 for folder in ['source_data','panels','_build']:(OUT/folder).mkdir(parents=True,exist_ok=True)
 base.OUT=OUT;base.shared.OUT=OUT;base.relationship_map=relationship_map;base.eventzoom=eventzoom;base.decorate_figure=decorate
 base.RELATION_HEADER_Y=.451
 base.PROMOTION_TITLE='(2) Promotion-like: R0 stromal to R1 B'
 base.INHIBITION_TITLE='(3) Inhibition-like: R2 tumour to R1 CD8 T'
 base.FOOTNOTE='Symbols denote relation classes; promotion/inhibition remain measured-contact hypotheses, not causal effects.\nThe collision key is defined for consistency; no independently called collision event is added. Frozen trees are unchanged.'
 base.CONTRACT_PATCH={'relation_symbols':STYLE,'collision_evidence':'Legend definition only. No new collision event is inferred or plotted.','old_event_markers':'E1/E2 labels and event-circle rings removed from D. Raw positive cells use squares/diamonds.','legend':'Three symbol shapes and colours provide redundant encoding under nature-figure integrity checks.'}
 (OUT/'source_data/RELATION_SYMBOL_DEFINITIONS.json').write_text(json.dumps(STYLE,indent=2,ensure_ascii=False),encoding='utf8')
 (OUT/'符号说明.md').write_text('''# D区三类关系符号

- 紫色 ×：碰撞/交汇类关系。
- 青绿色 →：促进样候选；当前对应基质CXCL12与B细胞CXCR4的实测近邻。
- 玫红色 ⊣：抑制样候选；当前对应恶性上皮CD274与CD8 T细胞PDCD1的实测近邻。

D总览及两个局部图使用相同符号和颜色，不再用E1/E2文字和事件圆圈。箭头/T形符号按实测配体到受体方向旋转，位置仍为原接触细胞对中点。局部方形和菱形标记分别表示实际配体阳性与受体阳性细胞，不是关系种类。

冻结数据当前只提供两类信号接触候选，没有单独通过判定的碰撞事件。×在本版中定义于图例，但不凭空加到树线交叉点上。二维线条相交不等于细胞碰撞。促进、抑制保留“候选”字样，不修改冻结结果。
''',encoding='utf8')
 (OUT/'Figure2_caption_addendum.tex').write_text(r'\textbf{D}, Relation classes use purple $\times$ (collision/crossing), teal $\rightarrow$ (promotion-like candidate), and rose $\dashv$ (inhibition-like candidate). Symbols replace event-number labels and circular event markers. The latter two mark measured ligand--receptor contact candidates; their direction follows ligand to receptor. Collision is defined in the key but no independently called collision event is added. Square and diamond raw-cell markers identify ligand-positive and receptor-positive cells, respectively. The frozen trees and candidate locations are unchanged.'+'\n',encoding='utf8')
 shutil.copy2(Path(__file__),OUT/Path(__file__).name)
 base.main()

if __name__=='__main__':main()
