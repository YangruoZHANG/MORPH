"""Recompute the cross-tree candidate contacts displayed in the approved main."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import main_contact_kernel as kernel

HERE=Path(__file__).resolve().parent
COHORTS=[('R0','Stromal'),('R1','B_cell'),('R2','Malignant_epithelial'),('R1','CD8_T')]

def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results')
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):parser.error('Use a fresh empty output folder; existing results are never overwritten.')
    for directory in ['inputs','reference','evidence','provenance']:
        if output.is_relative_to(HERE/directory):parser.error('Output must not be inside preserved input/reference/evidence folders.')
    (output/'source_data').mkdir(parents=True,exist_ok=True)
    inputs=HERE/'inputs'
    raw=ad.read_h5ad(inputs/'raw_counts.h5ad')
    cells=pd.read_csv(inputs/'cells.csv',dtype={'cell_name':str})
    members=pd.read_csv(inputs/'cell_to_state.csv')
    assert np.array_equal(raw.obs_names.astype(str),cells.cell_name.to_numpy()),'Raw-cell order does not match the fixed metadata'
    assert len(cells)==raw.n_obs
    assert members.raw_row.between(0,raw.n_obs-1).all()
    assert cells[['spatial_x','spatial_y']].notna().all().all()
    fits={key:(pd.read_csv(inputs/'cohorts'/('_'.join(key)+'.csv')),None) for key in COHORTS}
    data={'raw':raw,'rawxy':cells[['spatial_x','spatial_y']].to_numpy(), 'members':members,'fits':fits}
    kernel.OUT=output
    evidence_before={p.relative_to(HERE).as_posix():sha(p) for p in (HERE/'reference').glob('fixed_*.csv')}
    candidates=kernel.contacts(data)
    displayed=candidates.sort_values('descriptive_contact_score',ascending=False,kind='stable').drop_duplicates('event').copy()
    candidates.to_csv(output/'all_contact_candidates.csv',index=False)
    displayed.to_csv(output/'displayed_contacts.csv',index=False)
    keys=['event','source_raw_row','target_raw_row']
    for actual,name in [(candidates,'all_contact_candidates.csv'),(displayed,'displayed_contacts.csv')]:
        expected=pd.read_csv(HERE/'reference'/name)
        pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),expected.sort_values(keys).reset_index(drop=True),check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-10)
    assert not candidates.formal_call.any()
    assert evidence_before=={name:sha(HERE/name) for name in evidence_before}
    manifest={'complete':True,'reference_comparison_passed':True,'raw_cells':raw.n_obs,'genes':raw.n_vars,
              'tested_contact_candidates':len(candidates),'displayed_candidates':len(displayed),
              'formal_CMI_E_significance_test_executed':False,'trajectory_refit_executed':False,
              'fixed_trajectory_reference_files_unchanged':True,
              'scope':'Exact descriptive ligand-receptor candidate-contact computation used in main; frozen trajectories are inputs.',
              'displayed':displayed[['event','source_root','source_identity','target_root','target_identity','ligand','receptor','distance_um','descriptive_contact_score']].to_dict('records'),
              'packages':{p:metadata.version(p) for p in ['numpy','scipy','pandas','anndata','h5py']}}
    (output/'VERIFICATION.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
