"""Discount-free first-passage-time limit on the unchanged candidate graph.

Solve E_i[T_S]=1+sum_j P_ij E_j[T_S] off the supplied source set.
This removes the finite-discount horizon instead of searching alpha for the
best-looking tree. Source-connectedness, positivity, the local Bellman equation
and strictly upstream parent choices are checked. It is a hypothesis, not a
validated replacement of pseudotime or ancestry.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra

def passage_fit(matrix,item):
 ids=np.asarray(item['nodes'],int);sub=matrix[ids][:,ids].tocsr();seed=np.asarray(item['seeds'],int);si=pd.Index(ids).get_indexer(seed)
 if (si<0).any():raise ValueError('Source missing from support.')
 if not np.isfinite(dijkstra(sub,directed=False,indices=si,min_only=True)).all():raise ValueError('Unreachable nodes cannot receive an invented passage time.')
 w=sub.copy();w.data=np.exp(-np.minimum(w.data*w.data,700));degree=np.asarray(w.sum(1)).ravel();free=np.setdiff1d(np.arange(len(ids)),si);tau=np.zeros(len(ids));parent=np.full(len(ids),-1,int)
 if len(free):
  a=sparse.diags(degree[free])-w[free][:,free];tau[free]=spsolve(a.tocsc(),degree[free]);error=np.max(np.abs(a@tau[free]-degree[free])/np.maximum(degree[free],1e-12))
  if not np.isfinite(tau).all() or (tau[free]<=0).any() or error>1e-7:raise ArithmeticError(f'Passage solve invalid: relative equation error {error}')
 else:error=0.
 for v in free:
  start,end=sub.indptr[v:v+2];nn=sub.indices[start:end];cost=sub.data[start:end];ok=tau[nn]<tau[v]
  if not ok.any():raise ArithmeticError('First-passage Bellman maximum principle violated.')
  nn=nn[ok];cost=cost[ok];parent[v]=nn[np.lexsort((ids[nn],cost))[0]]
 assert set(np.flatnonzero(parent<0))==set(si)
 return {'nodes':ids,'seeds':seed,'tau':tau,'parent_local':parent,'maximum_relative_Bellman_equation_error':float(error)}
