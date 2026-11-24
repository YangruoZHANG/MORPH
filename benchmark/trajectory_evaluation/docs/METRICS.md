# Metric contract

All reported scores except signed PT are in `[0, 1]` where larger is better; PT is
in `[-1, 1]`. Prediction files and simulated observations are treated as frozen
inputs: no score is used to refit, select or repair a method.

## Coverage

Fraction of cells in the evaluation origin with a finite native pseudotime. It is
reported separately and never hidden inside a quality-only average.

## PT

Signed Spearman correlation between native predicted pseudotime and true continuous
time, evaluated on covered cells. The sign is not flipped post hoc. Every method
received the same supplied starting cell in the frozen benchmark.

## Cordist

Spearman correlation between reference and predicted pairwise trajectory-geodesic
distances on a deterministic sample of at most 10,000 cell pairs. Reference distances
are exact distances between continuous positions on the truth tree. Predicted distances
use the method's native weighted graph when at least half of covered cells map to it.
For Palantir and CellRank, which exported no tree, absolute native-pseudotime distance
is the declared fallback. `cordist_mode` and `cordist_pair_coverage` remain in the
audit table.

## Branch F1

Each branch is a root-to-leaf lineage. A cell on the shared trunk belongs to every
downstream lineage, so the cell sets may overlap. Branch names are ignored. For every
true lineage, the best predicted lineage is selected by cell-set Jaccard overlap and
averaged to obtain recovery. The reverse matching gives relevance. `Branch F1` is
their harmonic mean.

This is deliberately different from the previous hard-label ARI and from exclusive
edge-segment clustering. Both force each shared progenitor into exactly one segment and
therefore over-penalize otherwise correct terminal lineages. A wrong split, missed fate,
or wrong cell placement still lowers Jaccard overlap. For fate-only output, missing fate
labels form the shared trunk; this uses prediction output only and does not impute truth
labels. `branch_mode`, recovery, relevance and coverage are retained for audit. The
exclusive maximal-segment mapping is also retained in `segment_f1_audit` so the more
stringent dynbenchmark-like view remains fully visible.

## Topo 1-HIM

The predicted graph is projected post hoc to truth milestones by majority cell votes;
this mapping is evaluation-only. Hamming adjacency disagreement and the normalized
Ipsen–Mikhailov Laplacian-spectrum distance are combined as

`HIM = sqrt((Hamming^2 + IM^2) / 2)`.

The table reports `1 - HIM`. Palantir and CellRank do not export a unique native tree.
To keep the manuscript heatmap complete, their native fate/macrostate labels are
represented by a declared `derived_fate_star`: the earliest 20% of covered cells and
unassigned cells form one shared trunk, connected to one terminal node per native
label. This deterministic surrogate uses no truth information and is used only for
Topo 1-HIM. `topology_mode` distinguishes it from native graphs in the audit table.

## Feat-imp

Following dynbenchmark, random-forest impurity importance is computed from log-normalized
expression. For the reference trajectory, targets are distances to all true milestones.
For a graph prediction, targets are distances to its terminal/junction milestones; a
branchless prediction uses its native pseudotime. The score is the reference-importance-
weighted correlation of feature-importance ranks, with negative correlations clipped
to zero. Reference fits are cached and reused across methods.

## Aggregation

Roots are averaged within each simulated dataset first; simulated datasets then receive
equal weight. The headline `manuscript_table.csv` and `summary_long.csv` use the 25
positive trajectory-recovery datasets and exclude the five `spatial_null` datasets.
The latter are a negative-control rejection task and are reported separately in
`spatial_null_negative_control.csv` as output coverage and `1 - coverage` rejection
rate. `summary_all_conditions_long.csv` preserves the original 30-dataset aggregation
for audit only. Missing structural outputs are excluded only for the affected structural
metric and remain visible through `n_valid` and `n_planned`.

