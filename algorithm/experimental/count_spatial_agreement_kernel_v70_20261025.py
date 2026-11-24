"""Agreement kernel on unchanged count-admissible edges, with no added radius.

The old bounded d/(d+350) penalty becomes almost constant for millimetre-scale
separations. A short expression edge can therefore join distant tissue states.
Replace that bounded additive term by a positive product of evidence kernels:
 W_ij = exp(-standardized_molecular_distance_ij**2)
        / (1 + physical_distance_ij**2 / (local_scale_i*local_scale_j)).
Physical scale is the existing 30-neighbour design applied within identity,
not a fitted length parameter. The Cauchy factor never deletes a candidate or
creates a bridge. The corresponding cost sqrt(-log(W)) enters BOTH regional
support inference and the unchanged discounted hitting solver.

This is a falsifiable model change, not a drawing rule. It must pass held-gene,
technical-repeat and original-scorer tests before use in paper figures.
"""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

def agreement_graph(original_cost,units,neighbors=30):
 xy=units[['spatial_x','spatial_y']].to_numpy(float);identity=units.identity.to_numpy();scale=np.ones(len(units))
 for key in np.unique(identity):
  ids=np.flatnonzero(identity==key);k=min(neighbors,len(ids)-1)
  if k<1:continue
  d,_=cKDTree(xy[ids]).query(xy[ids],k=k+1);scale[ids]=np.maximum(d[:,-1],1e-8)
 coo=original_cost.tocoo();distance=np.linalg.norm(xy[coo.row]-xy[coo.col],axis=1)
 molecular=coo.data-.1*distance/(distance+350.)
 if molecular.min(initial=0)<-1e-8:raise ValueError('Input is not the declared molecular-plus-bounded-spatial candidate cost.')
 molecular=np.maximum(molecular,0);spatial_log=np.log1p(distance*distance/(scale[coo.row]*scale[coo.col]));cost=np.sqrt(molecular*molecular+spatial_log);result=sparse.csr_matrix((cost,(coo.row,coo.col)),shape=coo.shape)
 if result.nnz!=original_cost.nnz:raise AssertionError('Agreement kernel altered candidate edge membership.')
 return result
