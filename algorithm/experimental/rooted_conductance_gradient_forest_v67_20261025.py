"""Extract a forest from rooted conductance gradients, not near-flat cost ties.

Given the same graph, source set and fitted discounted hitting potential h,
choose each child's higher-h candidate parent maximizing
    exp(-cost**2) * (h_parent - h_child).
No new parameter, node, candidate edge, potential, source or display coordinate
is introduced. Stable log arithmetic avoids underflow for distant states.

For symmetric conductances the underlying equation is a resistor network with
positive diagonal leakage: alpha*L*h+(1-alpha)*D*h=0 off the source boundary.
For an asymmetric clinical-stage-admissible graph this is ONLY a directed
conductance-gradient extraction rule, not a conserved electrical-current claim.
Every edge strictly increases h towards a source. Thus independent local
maxima yield the globally maximum sum log-gradient weight rooted forest on
this potential DAG; no cycles or disconnected non-source maxima are allowed.
This mathematical property is NOT biological or benchmark validation.
"""
import numpy as np
import pandas as pd

def gradient_parents(matrix,fit,stage=None):
 ids=np.asarray(fit['nodes'],int);tau=np.asarray(fit['tau'],float);seed_ids=np.asarray(fit['seeds'],int);source=pd.Index(ids).get_indexer(seed_ids)
 if (source<0).any() or not np.isfinite(tau).all():raise ValueError('Invalid source/order contract.')
 sub=matrix[ids][:,ids].tocsr();parent=np.full(len(ids),-1,int);log_flux=np.full(len(ids),np.nan)
 for v in np.setdiff1d(np.arange(len(ids)),source):
  lo,hi=sub.indptr[v:v+2];nn=sub.indices[lo:hi];cost=sub.data[lo:hi];ok=tau[nn]<tau[v]
  if stage is not None:ok&=np.asarray(stage)[ids[nn]]<=np.asarray(stage)[ids[v]]
  nn=nn[ok];cost=cost[ok]
  if not len(nn):raise ValueError('A non-source has no strictly upstream admissible candidate.')
  delta=tau[v]-tau[nn];value=-np.minimum(cost*cost,700)-tau[nn]+np.log(-np.expm1(-delta));best=np.lexsort((ids[nn],-value))[0];parent[v]=nn[best];log_flux[v]=value[best]
 free=np.flatnonzero(parent>=0)
 assert set(np.flatnonzero(parent<0))==set(source) and (tau[parent[free]]<tau[free]).all()
 return parent,log_flux
