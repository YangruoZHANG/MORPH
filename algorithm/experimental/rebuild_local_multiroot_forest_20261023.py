"""Prospective annotation-assisted diagnostic revision, never overwrites P6.

Fixed supplied root coordinates; local raw-cell aggregates over the entire
observed T-cell population, not over old-tree support sets. This is a test of
the construction assumptions, not a claim of a validated new lineage method.
"""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import json
import hashlib
import numpy as np
import pandas as pd
import anndata as ad
import networkx as nx
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import dijkstra
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/Local_multiroot_revision_v3_20261023'
BIO = ROOT / 'outputs/Manuscript_method_biology_alignment_20261022'
P6 = ROOT / 'outputs/NicheForest_paper_frozen_candidate/P6_frozen_breast'
P8 = ROOT / 'outputs/NicheForest_paper_frozen_candidate/P8_breast_biological_validation'
SEEDS = [202610240, 202610241, 202610242, 202610243, 202610244]
PROTOCOL = {
    'status': 'Experimental annotation-assisted construction diagnostic; not frozen',
    'root_contract': 'Read exact supplied C13/C00/C21 x,y; immutable virtual-source coordinates. Map to nearest eligible local T-cell unit within 200um; record displacement. This changes the analysis unit, not root discovery.',
    'population': 'All published CD4+ or CD8+ T cells in the existing breast section; no old P6 support selection. Hybrid/unlabelled cells excluded from the state graph but retained in the source dataset.',
    'units': '150um fixed spatial grid, separately aggregated CD4 and CD8 identities, >=3 cells. Coordinates are actual aggregate centroids, never replaced by the root coordinate.',
    'annotation_boundary': 'Annotation-assisted diagnostic; purity is imposed and is not a validation endpoint. A successor would require an explicitly annotation-assisted method or independently learned identity gates.',
    'feature_roles': 'Original 50 construction genes only for graph fitting; remaining measured genes for conditional evaluation. Old construction-gene selection and published annotation were not gene-independent.',
    'graph': 'Union of expression mutual-15NN edges and spatial-8NN edges that are within the 90th percentile of expression mutual-neighbor distances; spatial cap 200/350/500um; no forced bridges.',
    'weight': 'Expression distance divided by geometric mean of endpoint 15NN scales plus 0.35 * spatial distance / radius.',
    'ensemble': 'Five fixed 80% construction-gene subsamples. Consensus edges require >=3/5 appearances; edge cost = mean observed cost + (1-frequency).',
    'soft_membership': 'Softmax of negative root-specific shortest-path costs divided by median finite nearest-root cost. No anatomical offset or root-specific coordinate warp.',
    'support': 'For each root, probability>=0.2 and finite distance; include its mapped seed, then retain only its connected component. No grafting between components.',
    'constructors': ['Consensus rooted shortest-path tree', 'Matched consensus Root-MSF'],
    'primary_radius_um': 350,
    'sensitivity_radii_um': [200, 350, 500],
    'primary_endpoint': 'Paired held-out-gene spatial-block CV R2, comparing identical-support rooted SPT vs MST; report expression+space-only control as well.',
    'evaluation': '4x4 spatial quantile blocks, GroupKFold(4), Ridge(alpha=10). Predict normalized log heldout counts using train PC15+space, or these features plus cubic tree pseudotime. Report heldout edge roughness, support stability and reachable coverage.',
    'abstention': 'Unmapped root (>200um), disconnected support and components <20 units are reported, not rescued by forced edges.',
    'interpretation': 'Within-identity state organization, not proof of lineage, migration, CD4/CD8 interconversion or tumor-induced causation. Tissue context is a separate layer.',
    'selection': '350um declared primary before evaluation; sensitivity variants are not selected on favorable pathways. No automatic promotion.',
    'protected': 'Do not modify AMCP, scoring, original P6/P8/P9, paper, figures or PPTX.'
}


def graph_from_features(log_train, xy, genes, radius):
    pca = PCA(n_components=min(15, len(genes), len(xy)-1), random_state=42)
    pc = pca.fit_transform(StandardScaler().fit_transform(log_train[:, genes]))
    ex = squareform(pdist(pc))
    sp = squareform(pdist(xy))
    k = min(15, len(xy)-1)
    nearest = np.argsort(ex, axis=1)[:, 1:k+1]
    knn = np.zeros(ex.shape, dtype=bool)
    knn[np.arange(len(xy))[:, None], nearest] = True
    mutual = knn & knn.T
    scale = np.maximum(np.take_along_axis(ex, nearest[:, -1:], axis=1).ravel(), 1e-8)
    threshold = np.quantile(ex[mutual], .90) if mutual.any() else 0
    spatial = np.zeros(ex.shape, dtype=bool)
    sn = np.argsort(sp, axis=1)[:, 1:9]
    spatial[np.arange(len(xy))[:, None], sn] = True
    eligible = (mutual | ((spatial | spatial.T) & (ex <= threshold))) & (sp <= radius)
    u, v = np.where(np.triu(eligible, 1))
    w = ex[u, v] / np.sqrt(scale[u]*scale[v]) + .35*sp[u, v]/radius
    return pd.DataFrame({'source':u, 'target':v, 'weight':w}), pc


def sparse_graph(frame, n):
    u, v = frame.source.to_numpy(int), frame.target.to_numpy(int)
    w = frame.weight.to_numpy(float)
    return sparse.csr_matrix((np.r_[w,w], (np.r_[u,v],np.r_[v,u])), shape=(n,n))


def soft_support(frame, n, anchors):
    distances = dijkstra(sparse_graph(frame, n), directed=False, indices=anchors)
    nearest = np.min(distances, axis=0)
    finite = nearest[np.isfinite(nearest) & (nearest > 0)]
    temperature = np.median(finite) if len(finite) else 1.
    logits = -distances / max(temperature, 1e-8)
    reachable = np.isfinite(nearest)
    probabilities = np.zeros((n, len(anchors)))
    probabilities[reachable] = softmax(logits[:, reachable].T, axis=1)
    g = nx.from_pandas_edgelist(frame, 'source','target', edge_attr='weight')
    g.add_nodes_from(range(n))
    supports = []
    for j, anchor in enumerate(anchors):
        nodes = set(np.flatnonzero((probabilities[:, j] >= .2) & np.isfinite(distances[j])))
        nodes.add(anchor)
        supports.append(set(nx.node_connected_component(g.subgraph(nodes), anchor)))
    return g, probabilities, supports


def cv_score(features, target, xy):
    bx = pd.qcut(pd.Series(xy[:,0]).rank(method='first'), 4, labels=False).to_numpy()
    by = pd.qcut(pd.Series(xy[:,1]).rank(method='first'), 4, labels=False).to_numpy()
    blocks = bx*4+by
    pred = np.empty_like(target)
    for train, test in GroupKFold(4).split(features, groups=blocks):
        sx = StandardScaler().fit(features[train])
        sy = StandardScaler().fit(target[train])
        model = Ridge(alpha=10).fit(sx.transform(features[train]),sy.transform(target[train]))
        pred[test] = sy.inverse_transform(model.predict(sx.transform(features[test])))
    sst = ((target-target.mean(0))**2).sum(0)
    good = sst > 1e-10
    return float(np.mean(1-((pred-target)**2).sum(0)[good]/sst[good]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT/'PROTOCOL.json'
    if protocol.exists():
        assert json.loads(protocol.read_text()) == PROTOCOL
    else:
        protocol.write_text(json.dumps(PROTOCOL, indent=2))
    print('Protocol', hashlib.sha256(protocol.read_bytes()).hexdigest(), flush=True)
    raw = ad.read_h5ad(P6/'relation_input/event/synthetic.h5ad')
    cells = pd.read_csv(BIO/'RAW_CELLS_WITH_PUBLISHED_ANNOTATIONS.csv', dtype={'cell_name':str})
    assert np.array_equal(raw.obs_names.astype(str).str.replace('CELL_','',regex=False), cells.cell_name)
    sel = cells.loc[cells.published_cell_type.isin(['CD4+_T_Cells','CD8+_T_Cells'])].copy()
    sel['grid_x'] = np.floor(sel.spatial_x/150).astype(int)
    sel['grid_y'] = np.floor(sel.spatial_y/150).astype(int)
    sel['unit'] = sel.groupby(['published_cell_type','grid_x','grid_y'],sort=True).ngroup()
    sizes = sel.unit.value_counts()
    sel = sel.loc[sel.unit.isin(sizes.index[sizes >= 3])].copy()
    sel['node_id'] = pd.Categorical(sel.unit, categories=sorted(sel.unit.unique())).codes
    n = int(sel.node_id.max()+1)
    obs = sel.groupby('node_id').agg(n_cells=('cell_name','size'), identity=('published_cell_type','first'),
                                   spatial_x=('spatial_x','mean'),spatial_y=('spatial_y','mean'))
    mat = sparse.csr_matrix((np.ones(len(sel)),(sel.node_id,np.arange(len(sel)))),shape=(n,len(sel)))
    counts = (mat @ sparse.csr_matrix(raw.X)[sel.index.to_numpy()]).toarray()
    train_names = set(pd.read_csv(P8/'P8_CONSTRUCTION_GENE_MANIFEST.csv').gene)
    train = np.flatnonzero(raw.var_names.isin(train_names))
    hold = np.flatnonzero(~raw.var_names.isin(train_names))
    # All feature splits use the same training-only depth normalizer.
    depth = np.maximum(counts[:,train].sum(1,keepdims=True),1)
    log = np.log1p(counts/depth*np.median(depth))
    xy = obs[['spatial_x','spatial_y']].to_numpy()
    supplied = pd.read_csv(ROOT/'data/选择的root/selected_roots_breast.csv')
    anchors = []
    root_rows = []
    for j, row in enumerate(supplied.itertuples()):
        distance = np.linalg.norm(xy-np.array([row.x,row.y]),axis=1)
        seed = int(np.argmin(distance))
        assert distance[seed] <= 200, ('abstain: supplied root has no local unit',row.root_id,distance[seed])
        anchors.append(seed)
        root_rows.append({'root_id':f'R{j}', 'root_cluster':row.root_id,'fixed_x':row.x,'fixed_y':row.y,
                          'mapped_node':seed,'mapped_x':xy[seed,0],'mapped_y':xy[seed,1],
                          'mapping_distance_um':distance[seed],'mapped_identity':obs.loc[seed,'identity'],
                          'mapped_n_cells':int(obs.loc[seed,'n_cells'])})
    assert len(set(anchors)) == 3
    pd.DataFrame(root_rows).to_csv(OUT/'IMMUTABLE_ROOT_CONTRACT.csv',index=False)
    obs.to_csv(OUT/'LOCAL_UNITS.csv',index_label='node_id')
    sel[['cell_name','metacell_index','node_id','published_cell_type','spatial_x','spatial_y']].to_csv(OUT/'CELL_TO_LOCAL_UNIT.csv',index=False)
    pd.DataFrame({'gene':raw.var_names,'role':np.where(raw.var_names.isin(train_names),'construction','heldout')}).to_csv(OUT/'GENE_ROLES.csv',index=False)
    ad.AnnData(sparse.csr_matrix(counts),obs=obs.astype({'identity':str}),var=pd.DataFrame(index=raw.var_names)).write_h5ad(OUT/'LOCAL_COUNTS.h5ad')
    print('Local units',n,'cells',len(sel),'roots',root_rows,flush=True)
    evaluations, stability, coverage = [], [], []
    for radius in [200,350,500]:
        dest = OUT/f'radius_{radius}'
        dest.mkdir(exist_ok=True)
        frames, replicate_supports = [], []
        for seed in SEEDS:
            genes = np.random.default_rng(seed).choice(len(train),int(.8*len(train)),replace=False)
            edges, _ = graph_from_features(log[:,train],xy,genes,radius)
            frames.append(edges)
            _, _, sup = soft_support(edges,n,anchors)
            replicate_supports.append(sup)
        stacked = pd.concat(frames)
        consensus = stacked.groupby(['source','target']).agg(mean_weight=('weight','mean'),count=('weight','size')).reset_index()
        consensus['frequency'] = consensus['count']/len(SEEDS)
        consensus = consensus.loc[consensus.frequency >= .6].copy()
        consensus['weight'] = consensus.mean_weight+1-consensus.frequency
        consensus.to_csv(dest/'CONSENSUS_CANDIDATES.csv',index=False)
        g, probs, supports = soft_support(consensus,n,anchors)
        pd.DataFrame(probs,columns=['R0','R1','R2']).to_csv(dest/'REFITTED_SOFT_MEMBERSHIP.csv',index_label='node_id')
        _, pc = graph_from_features(log[:,train],xy,np.arange(len(train)),radius)
        for j, anchor in enumerate(anchors):
            rid = f'R{j}'
            support = supports[j]
            for a in range(len(SEEDS)):
                for b in range(a+1,len(SEEDS)):
                    aa,bb = replicate_supports[a][j],replicate_supports[b][j]
                    stability.append({'radius_um':radius,'root_id':rid,'seed_a':SEEDS[a],'seed_b':SEEDS[b],
                                      'support_jaccard':len(aa&bb)/len(aa|bb)})
            sub = g.subgraph(support).copy()
            lengths, paths = nx.single_source_dijkstra(sub,anchor,weight='weight')
            spt = nx.Graph()
            spt.add_nodes_from(support)
            for v, path in paths.items():
                if len(path)>1:
                    u = path[-2]
                    spt.add_edge(u,v,weight=sub[u][v]['weight'])
            mst = nx.minimum_spanning_tree(sub,weight='weight')
            ids = np.array(sorted(support))
            coverage.append({'radius_um':radius,'root_id':rid,'n_nodes':len(ids),'n_cells':int(obs.loc[ids,'n_cells'].sum()),
                             'fraction_all_T_cells':float(obs.loc[ids,'n_cells'].sum()/len(sel)),
                             'component_testable':len(ids)>=20,'candidate_component_size':len(nx.node_connected_component(g,anchor))})
            base_features = np.column_stack([pc[ids],xy[ids]])
            base = cv_score(base_features,log[ids][:,hold],xy[ids]) if len(ids)>=20 else np.nan
            for name, tree in [('Consensus_SPT',spt),('Matched_Root_MSF',mst)]:
                assert nx.is_tree(tree)
                depth_tree = nx.single_source_dijkstra_path_length(tree,anchor,weight='weight')
                scale = max(depth_tree.values()) or 1
                tau = np.array([depth_tree[i]/scale for i in ids])
                order = obs.loc[ids].copy()
                order['pseudotime'] = tau
                order['root_id'] = rid
                order['is_mapped_seed'] = ids==anchor
                order.to_csv(dest/f'{rid}_{name}_NODES.csv',index_label='node_id')
                edge_rows = []
                for u,v in tree.edges:
                    if depth_tree[u]>depth_tree[v]:u,v=v,u
                    edge_rows.append({'source':u,'target':v,'weight':sub[u][v]['weight'],
                                      'spatial_um':float(np.linalg.norm(xy[u]-xy[v]))})
                pd.DataFrame(edge_rows,columns=['source','target','weight','spatial_um']).to_csv(dest/f'{rid}_{name}_EDGES.csv',index=False)
                if len(ids)<20:continue
                features = np.column_stack([base_features,tau,tau**2,tau**3])
                score = cv_score(features,log[ids][:,hold],xy[ids])
                z = StandardScaler().fit_transform(log[ids][:,hold])
                positions = {v:i for i,v in enumerate(ids)}
                a = np.array([positions[e['source']] for e in edge_rows])
                b = np.array([positions[e['target']] for e in edge_rows])
                rr = {'radius_um':radius,'root_id':rid,'method':name,'n_nodes':len(ids),
                      'heldout_R2':score,'expression_space_only_R2':base,'increment_vs_expression_space':score-base,
                      'heldout_edge_roughness':float(np.mean((z[a]-z[b])**2)),
                      'spatial_edge_p95_um':float(np.quantile([e['spatial_um'] for e in edge_rows],.95)),
                      'CD8_fraction_tau_rho':float(spearmanr(tau,(obs.loc[ids,'identity']=='CD8+_T_Cells').astype(float)).statistic)}
                evaluations.append(rr)
                print(rr,flush=True)
    ev = pd.DataFrame(evaluations)
    ev.to_csv(OUT/'HELDOUT_COMPARISON.csv',index=False)
    pd.DataFrame(stability).to_csv(OUT/'GENE_SUBSAMPLE_SUPPORT_STABILITY.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'COVERAGE_AND_ABSTENTION.csv',index=False)
    pairs = ev.pivot(index=['radius_um','root_id'],columns='method',values='heldout_R2')
    pairs['SPT_minus_MSF_R2'] = pairs.Consensus_SPT-pairs.Matched_Root_MSF
    pairs.to_csv(OUT/'PAIRED_CONSTRUCTOR_EFFECT.csv')
    (OUT/'STATUS.json').write_text(json.dumps({'promoted':False,'n_raw_T_cells':int(cells.published_cell_type.isin(['CD4+_T_Cells','CD8+_T_Cells']).sum()),
        'n_retained_T_cells':len(sel),'n_units':n,'n_construction_genes':len(train),'n_heldout_genes':len(hold),
        'primary_radius_um':350,'old_results_modified':False,'conclusion':'Inspect heldout effects and stability; no automatic promotion.'},indent=2))


if __name__ == '__main__':
    main()
