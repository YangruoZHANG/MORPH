"""Portable commands for the Figure 3 handoff. Run from any working directory."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
BENCH=ROOT/'data/supplied_root'
FROZEN=ROOT/'data/historical_comparisons'
DEV=ROOT/'algorithm/evaluation/branch_state'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(key,'1')
os.environ.setdefault('NUMBA_DISABLE_JIT','1')


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def run_plot():
    subprocess.run([sys.executable,str(ROOT/'algorithm/visualization/plot_benchmark.py'),
        '--data-root',str(FROZEN),'--benchmark-root',str(BENCH),
        '--summary-root',str(ROOT/'summary'),
        '--outdir',str(ROOT/'reproduced/Figure3')],check=True,cwd=ROOT)


def verify(full=False):
    inventory=json.loads((ROOT/'PACKAGE_INVENTORY.json').read_text(encoding='utf-8'))
    for i,item in enumerate(inventory):
        path=ROOT/item['path']
        if not path.is_file() or path.stat().st_size!=item['bytes']:raise ValueError(path)
        if full:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1048576),b''):h.update(block)
            if h.hexdigest()!=item['sha256']:raise ValueError('Hash mismatch: '+str(path))
        if i%5000==0:print('Verified',i,'files',flush=True)
    assert len(list((BENCH/'datasets').glob('*.h5ad')))==1000
    assert len(list((BENCH/'runs').glob('*/METRICS.json')))==1000
    print('Integrity PASS:',len(inventory),'files; 1000 simulated datasets / paired metric records.',flush=True)


def scores(limit=None,historical=False):
    """Recompute the four published metrics using the original truth mapping.

    Additional exploratory topology diagnostics are deliberately not rerun;
    their original complete scorer remains in the algorithm directory.
    """
    import networkx as nx
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import adjusted_rand_score
    d=load_module('handoff_original_hierarchical',DEV/'hierarchical_branch_diagnostics.py')
    Q=d.V2.Q
    if historical:
        base=FROZEN/'results/NicheForest_paper_frozen_candidate/P4_root_conditioned_benchmark/runs'
        paths=sorted(base.glob('*/forest_run/phase_a/T0_STATE_NODE_ASSIGNMENTS.csv'))
        expected=pd.read_csv(base.parent/'P4_CORE_COMPARISON.csv')
    else:
        paths=sorted((BENCH/'runs').glob('*/forest_run/phase_a/T0_STATE_NODE_ASSIGNMENTS.csv'))
    if limit:paths=paths[:limit]
    records=[];errors=[]
    for index,path in enumerate(paths):
        phase=path.parent;run=phase.parent;folder=run.parent;dataset=folder.name
        inputs=folder/'forest_inputs'
        fixed=run/'supplied_root_inputs/FIXED_ROOT_INPUT.csv'
        truth=d.load_truth_for_metacells(run,inputs,fixed)
        n=pd.read_csv(inputs/'cell_to_metacell_mapping.csv').shape[0]
        for case,method in [('current_root_conditioned_msf','root-conditioned MSF'),
                             ('hierarchical_branch_program_state_fine','current branch-scale T0')]:
            case_dir=phase/case
            nodes=pd.read_csv(case_dir/'T0_NODE_ASSIGNMENTS.csv',float_precision='round_trip')
            edges=pd.read_csv(case_dir/'T0_FINE_TRAJECTORY_EDGES.csv',float_precision='round_trip')
            per=defaultdict(list)
            for root in truth.roots:
                no=nodes[nodes.root_id.astype(str)==root]
                ed=edges[edges.root_id.astype(str)==root]
                pairset=set()
                for e in ed.itertuples(index=False):
                    options=Q.plausible_pairs(truth,root,str(e.source_name),str(e.target_name))
                    pairset.update(Q.P.pair(a,b) for _,a,b in options)
                truepairs={Q.P.pair(a,b) for a,b in truth.graphs[root].edges()}
                per['topology_F1'].append(Q.P.prf(truepairs,pairset)[2])
                pairs=[(truth.branch.get((str(r.unit_id),root)),str(r.branch_id))
                       for r in no.itertuples(index=False)
                       if truth.branch.get((str(r.unit_id),root)) is not None]
                per['branch_ARI_rootwise'].append(adjusted_rand_score([a for a,b in pairs],[b for a,b in pairs]) if len(pairs)>=2 else np.nan)
                pts=[(truth.pseudotime.get((str(r.unit_id),root)),float(r.pseudotime)) for r in no.itertuples(index=False)]
                pts=[(a,b) for a,b in pts if a is not None and np.isfinite(a)]
                per['pseudotime_spearman'].append(spearmanr([a for a,b in pts],[b for a,b in pts]).statistic if len(pts)>=3 else np.nan)
                graph=nx.Graph();graph.add_nodes_from(no.unit_id.astype(str))
                graph.add_edges_from(zip(ed.source_name.astype(str),ed.target_name.astype(str)))
                assert nx.is_forest(graph)
            row=dict(dataset=dataset,method=method,coverage=nodes.unit_id.nunique()/n)
            row.update({key:float(np.nanmean(vals)) for key,vals in per.items()})
            frozen=(expected[(expected.dataset==dataset)&(expected.method==method)].iloc[0].to_dict()
                    if historical else next(x for x in json.loads((folder/'METRICS.json').read_text()) if x['method']==method))
            for key in ['coverage','topology_F1','branch_ARI_rootwise','pseudotime_spearman']:
                if not np.isclose(row[key],float(frozen[key]),atol=1e-10,equal_nan=True):
                    errors.append(dict(dataset=dataset,method=method,metric=key,recomputed=row[key],frozen=frozen[key]))
            records.append(row)
        if (index+1)%25==0:print('Rescored',index+1,'datasets; mismatches:',len(errors),flush=True)
    out=ROOT/'reproduced';out.mkdir(exist_ok=True)
    name='historical' if historical else '1000'
    pd.DataFrame(records).to_csv(out/f'rescored_{name}.csv',index=False)
    (out/f'rescore_{name}_audit.json').write_text(json.dumps(dict(datasets=len(paths),records=len(records),mismatches=errors),indent=2),encoding='utf-8')
    if errors:raise ValueError(f'{len(errors)} metric mismatches; see reproduced/rescore_{name}_audit.json')
    print('Metric reconstruction PASS:',len(paths),'datasets /',len(records)*4,'metric values.',flush=True)


def summary():
    import numpy as np
    import pandas as pd
    from scipy.stats import binomtest
    rows=[]
    for p in sorted((BENCH/'runs').glob('*/METRICS.json')):rows.extend(json.loads(p.read_text()))
    frame=pd.DataFrame(rows);out=ROOT/'reproduced';out.mkdir(exist_ok=True)
    result=[]
    frozen=pd.read_csv(ROOT/'summary/overall_metrics.csv')
    for label,key in [('Coverage','coverage'),('Branch ARI','branch_ARI_rootwise'),('State-edge F1','topology_F1'),('Signed pseudotime rho','pseudotime_spearman')]:
        tab=frame.pivot(index='dataset',columns='method',values=key)
        delta=tab['current branch-scale T0']-tab['root-conditioned MSF']
        wins=int((delta>1e-12).sum());losses=int((delta < -1e-12).sum())
        row=dict(metric=label,nicheforest_mean=tab['current branch-scale T0'].mean(),rootmsf_mean=tab['root-conditioned MSF'].mean(),paired_mean_difference=delta.mean(),wins=wins,ties=len(delta)-wins-losses,losses=losses,p_value=binomtest(wins,wins+losses).pvalue)
        orig=frozen[frozen.metric==label].iloc[0]
        for k in ['nicheforest_mean','rootmsf_mean','paired_mean_difference','wins','ties','losses','p_value']:
            assert np.isclose(row[k],orig[k],atol=1e-12), (label,k)
        # Preserve the published frozen CI exactly. Its original random draws
        # were not archived; condition CIs are reproducible in the plot script.
        row.update(ci95_lower=orig.ci95_lower,ci95_upper=orig.ci95_upper)
        result.append(row)
    pvals=np.array([r['p_value'] for r in result]);order=np.argsort(pvals)
    adjusted=np.minimum.accumulate((pvals[order]*len(pvals)/np.arange(1,len(pvals)+1))[::-1])[::-1]
    for j,q in zip(order,np.minimum(adjusted,1.)):
        result[j]['q_value']=float(q)
        original=frozen[frozen.metric==result[j]['metric']].iloc[0]
        assert np.isclose(q,original.q_value,rtol=1e-9,atol=1e-300)
    pd.DataFrame(result).to_csv(out/'overall_metrics_reaggregated.csv',index=False)
    print('Summary means, paired deltas, W/T/L, exact sign p-values and BH q-values PASS. Frozen CI carried with provenance.',flush=True)


def generate(limit=None):
    import pandas as pd
    import anndata as ad
    import numpy as np
    g=load_module('handoff_simulator',ROOT/'simulate/make_multiorigin_spatial.py')
    configs=dict(g.grid_conditions());m=pd.read_csv(BENCH/'datasets/manifest.csv')
    if limit:m=m.head(limit)
    out=ROOT/'reproduced/simulated_datasets';out.mkdir(parents=True,exist_ok=True)
    for i,r in enumerate(m.itertuples()):
        a=g.simulate(name=r.dataset,condition=r.condition,seed=int(r.seed),cells_per_edge=70,**configs[r.condition])
        original=ad.read_h5ad(BENCH/'datasets'/f'{r.dataset}.h5ad')
        assert a.shape==original.shape and np.array_equal(a.X,original.X),r.dataset
        assert np.array_equal(a.obsm['spatial'],original.obsm['spatial']),r.dataset
        a.write_h5ad(out/f'{r.dataset}.h5ad',compression='gzip')
        if (i+1)%25==0:print('Regenerated',i+1,flush=True)
    print('Simulation reproduction PASS:',len(m),'datasets.',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['plot','verify','scores','summary','generate','rerun'])
    ap.add_argument('--full',action='store_true');ap.add_argument('--limit',type=int)
    ap.add_argument('--historical',action='store_true');ap.add_argument('--workers',type=int,default=1)
    a=ap.parse_args()
    if a.command=='plot':run_plot()
    elif a.command=='verify':verify(a.full)
    elif a.command=='scores':scores(a.limit,a.historical)
    elif a.command=='summary':summary()
    elif a.command=='generate':generate(a.limit)
    else:
        sys.path.insert(0,str(ROOT/'algorithm/validation'))
        import anndata as ad
        import pandas as pd
        from run_supplied_root_benchmark import _run_core
        data=pd.read_csv(BENCH/'datasets/manifest.csv').dataset.tolist()
        if a.limit:data=data[:a.limit]
        for ds in data:
            path=BENCH/'datasets'/f'{ds}.h5ad';out=ROOT/'reproduced/inference_runs'/ds
            out.mkdir(parents=True,exist_ok=True);rows,status=_run_core(ad.read_h5ad(path),path,out)
            (out/'METRICS.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print('Fresh inference complete:',ds,flush=True)


if __name__=='__main__':main()
