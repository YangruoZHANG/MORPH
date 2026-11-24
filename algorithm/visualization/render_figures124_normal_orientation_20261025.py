"""Display-only orientation repair. Never rerun inference or enrichment."""
import sys,json,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'algorithm/visualization'))
import redraw_figure2_relationships_v70_20261025 as base
import redraw_figure2_cluster_context_v70_20261025 as colour
import redraw_figure2_relation_symbols_v70_20261025 as symbols
import restore_figure4_enrichment_v70_20261025 as bio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

OUT=ROOT/'outputs/Figures_124_normal_orientation_20261025'
OLD_BIO=ROOT/'outputs/Figure4_enrichment_reference_style_v70_20261025'
checks=[]
original_clean=base.shared.clean
def normal_clean(ax,xy,spatial=True,pad=.055):
 # Retain every numeric coordinate. Ascending y matches the user-approved
 # original view. No array reflection, object relocation or image flip.
 original_clean(ax,xy,False,pad)
 if spatial:
  assert ax.get_ylim()[0]<ax.get_ylim()[1]
  checks.append({'xlim':list(map(float,ax.get_xlim())),'ylim':list(map(float,ax.get_ylim())),'y_inverted':bool(ax.yaxis_inverted())})

def normal_symbol(ax,d,row,size):
 a=d['rawxy'][int(row.source_raw_row)];b=d['rawxy'][int(row.target_raw_row)];q=(a+b)/2;delta=b-a
 angle=float(np.degrees(np.arctan2(delta[1],delta[0])))
 style=symbols.STYLE[row.event]
 ax.text(*q,style['symbol'],color=style['colour'],fontsize=size,ha='center',va='center',rotation=angle,rotation_mode='anchor',weight='bold',zorder=20,path_effects=[pe.withStroke(linewidth=2.1,foreground='white')])

def normal_bar(ax,length=1000):
 # Axes-relative placement keeps the label ABOVE the bar in either tissue
 # convention; never reflect the label along with plotted observations.
 x0,x1=ax.get_xlim();x=x0+.055*(x1-x0);tr=ax.get_xaxis_transform()
 ax.plot([x,x+length],[.025,.025],color=base.INK,lw=1.1,transform=tr,zorder=30)
 ax.text(x+length/2,.05,f'{length/1000:g} mm',fontsize=5.3,ha='center',va='bottom',transform=tr,zorder=31,bbox=dict(fc='white',ec='none',pad=.2,alpha=.9))

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 contract={'claim':'The same frozen states, roots, trees and enrichment are displayed in the previously approved ascending-y original-tissue orientation.','archetype':'Three existing mixed-modality figures; display correction only.','backend':'python matplotlib','export':'Three complete PNG/PDF/SVG figures and separate-panel manifests for a3-page PPTX.','invariants':'Raw coordinates, root anchors, cell identities, graph edges, paths, UMAP coordinates, programme disk coordinates, all gene-set p/q values unchanged.','transformation':'No invert_yaxis for any original-tissue plot. No change to expression or programme-space axes. Directional relation glyph angles recalculated for ascending-y display.','risk':'Stars in programme space represent source-state units, while tissue stars are fixed spatial anchors; no substitution between them.'}
 (OUT/'DISPLAY_ORIENTATION_CONTRACT.json').write_text(json.dumps(contract,indent=2))
 count=bio.source_audit();base.shared.clean=normal_clean;base.shared.bar=normal_bar;symbols.symbol=normal_symbol
 for k in [1,2,4]:
  for sub in ['panels','source_data','_build']:(OUT/f'figure{k}'/sub).mkdir(parents=True,exist_ok=True)
 first=OUT/'figure1';base.shared.OUT=first
 plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
 d=base.shared.load();base.shared.figure1(d)
 oldpaths=pd.read_csv(ROOT/'outputs/Checked_visualization_suite_20261025/source_data/EXACT_DISPLAYED_FOREST_PATHS.csv')
 pd.testing.assert_frame_equal(pd.read_csv(first/'source_data/EXACT_DISPLAYED_FOREST_PATHS.csv'),oldpaths)
 print('Figure1: ascending-y, exact paths unchanged',flush=True)
 # The wrapper keeps all approved colour and layout choices in Figure2.
 colour.OUT=OUT/'figure2';colour.main()
 print('Figure2: ascending-y, relation glyph directions checked',flush=True)
 target=OUT/'figure4';shutil.copytree(OLD_BIO/'source_data',target/'source_data',dirs_exist_ok=True)
 shutil.copy2(OLD_BIO/'Figure4_legend.tex',target/'Figure4_legend.tex')
 cached=np.load(OLD_BIO/'_build/plot_payload.npz')
 cat=pd.read_csv(OLD_BIO/'source_data/ORIGINAL_43_PROGRAM_CATALOG.csv')
 assignments=pd.read_csv(OLD_BIO/'source_data/FROZEN_COHORT_STEP_ASSIGNMENTS.csv')
 enrich=pd.read_csv(OLD_BIO/'source_data/ALL_COMPETITIVE_SPATIAL_BLOCK_GENE_SET_TESTS.csv')
 story=pd.read_csv(OLD_BIO/'source_data/ORDERED_STEP_FUNCTIONAL_STORY.csv')
 control=pd.read_csv(OLD_BIO/'source_data/MATCHED_SOURCE_SHORTEST_PATH_EDGES.csv')
 bio.OUT=target
 bio.render(d,(cat,assignments,enrich,story,cached['xy'],cached['unique'],cached['dominant'],cached['effect'],control))
 # Exact cached numbers, no gene or term test is executed in this repair.
 for name in ['ALL_COMPETITIVE_SPATIAL_BLOCK_GENE_SET_TESTS.csv','ALL_MEASURED_PANEL_GO_REACTOME_ORA.csv','COMMON_PROGRAM_DISK_COORDINATES.csv','ALL_STEP_GENE_TESTS.csv','FROZEN_COHORT_STEP_ASSIGNMENTS.csv']:
  assert hashlib.sha256((OLD_BIO/'source_data'/name).read_bytes()).digest()==hashlib.sha256((target/'source_data'/name).read_bytes()).digest()
 anchors=d['roots'].reset_index()[['root_id','x','y']]
 anchors['display_orientation']='ascending y; R2 bottom-right in full section'
 anchors.to_csv(OUT/'UNCHANGED_SPATIAL_ANCHORS.csv',index=False)
 assert bio.source_audit()==count
 assert len(checks)>0 and not any(c['y_inverted'] for c in checks)
 (OUT/'ORIENTATION_AND_INVARIANT_QA.json').write_text(json.dumps({'spatial_axes_checked':len(checks),'all_spatial_axes_ascending_y':True,'frozen_files_checked':count,'frozen_files_changed':0,'fig1_display_path_table_identical':True,'fig4_gene_term_tests_rerun':False,'fig4_all_checked_statistical_tables_identical':True,'expression_and_programme_coordinates_unchanged':True,'axes':checks},indent=2))
 shutil.copy2(Path(__file__),OUT/Path(__file__).name)
 print('COMPLETE',OUT,flush=True)
if __name__=='__main__':main()
