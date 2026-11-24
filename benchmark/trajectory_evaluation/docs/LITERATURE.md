# Literature basis

The six-column panel follows the evaluation dimensions established by Saelens et al.
rather than inventing a score tailored to MORPH.

1. Saelens W, Cannoodt R, Todorov H, Saeys Y. **A comparison of single-cell
   trajectory inference methods.** *Nature Biotechnology* 37, 547–554 (2019).
   DOI: https://doi.org/10.1038/s41587-019-0071-9
2. Dynbenchmark metric characterization and robustness experiments:
   https://github.com/dynverse/dynbenchmark_results/tree/master/02-metrics/01-metric_characterisation
3. Dynbenchmark accuracy analysis (HIM, F1 branches, geodesic-distance correlation,
   weighted feature-importance correlation):
   https://github.com/dynverse/dynbenchmark_results/tree/master/06-benchmark
4. Todorov H et al. **Cell-connectivity-guided trajectory inference from single-cell
   data.** *Nature Biotechnology* / PMC10474950 (2023). The study summarizes overall
   accuracy with the same four complementary trajectory dimensions.

The literature reports synthetic data because its exact reference trajectory is known.
It also treats ordering, topology, branch assignment and feature relevance as separate
questions; a high pseudotime score cannot compensate invisibly for a missing tree.

