# 论文正文修改建议：轨迹准确性与空间空对照分离报告

本文档中的文字可直接用于修改论文 Methods、Results、Discussion、图注和表注。
方括号内的图表编号请替换成论文中的实际编号。

## 一、修改原则

原汇总把 25 个正向轨迹恢复数据集与 5 个 `spatial_null` 数据集共同平均。
这会把两个不同问题混在一起：

1. 正向数据集回答“方法能否恢复存在的轨迹”；
2. `spatial_null` 回答“当空间坐标与表达轨迹脱钩时，方法是否仍然输出轨迹”。

Coverage 定义为获得有限伪时间值的细胞比例，是输出覆盖率而不是准确率。非空间
方法通常为所有细胞强制返回伪时间，因此 Coverage 接近 1 并不表示其轨迹正确。
MORPH 在 `spatial_null` 中仅保留根细胞，反映其空间支持筛选产生的主动拒绝行为。
因此，主准确性表应只汇总 25 个正向数据集；5 个空间空对照单独报告 Output
coverage 和 Rejection rate (`1 - coverage`)。

## 二、Methods 建议文字

### 中文版

虚拟数据评测被预先划分为轨迹恢复任务和空间空对照任务。轨迹恢复任务包含25个
具有有效表达—空间轨迹对应关系的数据集，覆盖分叉、三分叉、深层树、低计数和
多起点条件。我们在这些数据集上计算输出覆盖率（获得有限伪时间值的细胞比例）、
带方向的伪时间 Spearman 相关（PT）、真实与预测轨迹测地距离相关（Cordist）、
root-to-leaf 谱系集合的 Branch F1、Hamming–Ipsen–Mikhailov 拓扑相似度
（Topo 1-HIM）以及加权特征重要性相关（Feat-imp）。多起点数据首先在各根内
评分，再在数据集内等权平均；所有数据集在最终汇总中等权。置信区间通过数据集
层面的 bootstrap 获得。

另设置5个空间空对照数据集，其中表达轨迹保持不变，而空间坐标被随机置换。该
对照用于评价空间一致性筛选产生的拒绝行为，而不纳入正向轨迹准确性均值。我们
单独报告 Output coverage 及 Rejection rate（1 − Output coverage）。Coverage
仅衡量方法是否为细胞返回有效伪时间，不应解释为轨迹准确率。

Palantir 和 CellRank 不返回唯一的原生树。为避免汇总热图出现结构性缺失，我们
仅在 Topo 1-HIM 中使用预先规定的 `derived_fate_star` 表示：预测伪时间最早的
20%细胞及未分配细胞构成公共主干，每个原生 fate 或 macrostate 标签构成一个
末端节点。该派生结构不使用真实标签，并在审计表中与原生树明确区分。

### English version

The simulation benchmark was prespecified as two separate tasks: positive trajectory
recovery and a spatial-null diagnostic. The positive task comprised 25 datasets with
valid expression–spatial trajectory correspondence, spanning bifurcating,
trifurcating, deep-tree, low-count, and multi-origin conditions. We evaluated output
coverage (the fraction of cells with finite pseudotime), signed pseudotime Spearman
correlation (PT), correlation of reference and inferred trajectory-geodesic distances
(Cordist), Branch F1 over overlapping root-to-leaf lineage cell sets,
Hamming–Ipsen–Mikhailov topology similarity (Topo 1-HIM), and reference-weighted
feature-importance correlation (Feat-imp). Multi-origin datasets were first scored
within each supplied root and then averaged within dataset; datasets received equal
weight in the final summary. Confidence intervals were obtained by dataset-level
bootstrap resampling.

Five additional spatial-null datasets retained the expression trajectory while the
spatial coordinates were randomly permuted. These datasets were used to quantify
rejection induced by spatial-consistency filtering and were not pooled with positive
trajectory-recovery accuracy. We separately report Output coverage and Rejection rate
(1 − Output coverage). Coverage measures whether a method returns finite pseudotime
for a cell and must not be interpreted as trajectory accuracy.

Palantir and CellRank do not return a unique native tree. To avoid structural missing
cells in the summary heatmap, a prespecified `derived_fate_star` was used only for
Topo 1-HIM: the earliest 20% of cells by predicted pseudotime and unassigned cells
formed a shared trunk, and each native fate or macrostate label formed one terminal
node. This surrogate used no reference labels and was explicitly distinguished from
native graphs in the audit table.

## 三、Results 建议文字

### 中文版

在25个正向轨迹数据集上，MORPH 对所有目标细胞均产生有效输出（Coverage =
1.000），并取得 PT = 0.978、Cordist = 0.959、Branch F1 = 0.744、Topo 1-HIM
= 1.000 和 Feat-imp = 0.627。不同指标反映互补的轨迹属性，因此未合并为单一
总分。MORPH 在细胞排序、全局轨迹距离和特征相关性方面表现稳定；Branch F1
采用允许共享主干细胞同时属于多个下游谱系的 root-to-leaf 集合定义。作为严格
敏感性分析，排他性的 edge-segment F1 仍保留在完整审计结果中。

在5个空间空对照中，MORPH 的平均 Output coverage 为0.0037，对应 Rejection
rate = 0.9963。其余基线方法的 Output coverage 为1.000，主要因为这些方法无论
空间是否与表达轨迹一致都会为全部细胞返回伪时间。该结果应解释为方法输出策略
和空间一致性门控的差异，而不是非空间方法的准确率为100%，也不应单独作为
MORPH 优于非空间方法的证据。

### English version

Across the 25 positive trajectory datasets, MORPH returned a finite prediction for all
target cells (Coverage = 1.000) and achieved PT = 0.978, Cordist = 0.959, Branch F1 =
0.744, Topo 1-HIM = 1.000, and Feat-imp = 0.627. Because these metrics capture
complementary properties of a trajectory, they were not collapsed into a single
composite score. Branch F1 was defined over root-to-leaf lineage sets, allowing shared
trunk cells to belong to multiple downstream lineages; the more stringent exclusive
edge-segment F1 was retained as a sensitivity analysis in the full audit table.

In the five spatial-null controls, MORPH had a mean Output coverage of 0.0037,
corresponding to a Rejection rate of 0.9963. The baseline methods had Output coverage
of 1.000 because they returned pseudotime for all cells irrespective of whether spatial
coordinates remained concordant with the expression trajectory. This result reflects
different output and spatial-gating behavior; it should not be interpreted as 100%
accuracy for non-spatial methods or, by itself, as evidence that MORPH is superior.

## 四、图注建议

### 主热图中文图注

**图[X]｜正向虚拟轨迹数据上的多维度评测。** 数值为25个正向轨迹数据集的
等权均值，越高越好。Coverage 表示获得有限伪时间值的细胞比例，而非准确率。
PT、Cordist、Branch F1、Topo 1-HIM 和 Feat-imp 分别评价细胞排序、轨迹测地
距离、root-to-leaf 谱系归属、拓扑结构和轨迹相关特征排序。`spatial_null`
数据未纳入本图，而在独立负对照分析中报告。

### Main heatmap English caption

**Figure [X] | Multidimensional evaluation on positive synthetic trajectories.**
Values are equal-weight means across 25 positive trajectory datasets; higher is better.
Coverage denotes the fraction of cells with finite pseudotime and is not an accuracy
measure. PT, Cordist, Branch F1, Topo 1-HIM, and Feat-imp evaluate cell ordering,
trajectory-geodesic distances, root-to-leaf lineage assignment, topology, and
trajectory-associated feature ranking, respectively. Spatial-null datasets were
excluded from this panel and analyzed separately as negative controls.

### 空对照表注

**表[X]｜空间空对照中的输出覆盖与拒绝率。** 表达轨迹保持不变，空间坐标被
随机置换。Rejection rate 定义为 `1 − Output coverage`。该分析描述方法是否在
缺少空间一致性时仍输出轨迹，不作为正向轨迹准确率排序。

## 五、Discussion/Limitations 建议文字

### 中文版

需要指出，空间空对照中的高拒绝率既反映 MORPH 对空间一致性的敏感性，也反映
其允许放弃不受支持预测的设计。多数基线方法被设计为对所有细胞返回伪时间，因而
在 Coverage 指标上天然接近1。由此，空对照中的 Coverage/Rejection rate 不应
作为跨方法准确率排名，而应与正向数据上的 PT、Cordist、Branch F1、拓扑和
特征重要性结果结合解释。此外，Palantir 和 CellRank 的拓扑值来自预先规定的
fate-star 派生表示，并非原生树输出。

### English version

The high spatial-null rejection rate reflects both MORPH's sensitivity to spatial
consistency and its design choice to abstain when support is insufficient. Most
baseline methods are designed to return pseudotime for every cell and therefore have
coverage near one by construction. Consequently, spatial-null Coverage/Rejection rate
should be interpreted as an output-behavior diagnostic rather than a cross-method
accuracy ranking, and should be considered jointly with PT, Cordist, Branch F1,
topology, and feature-importance performance on positive datasets. In addition, the
topology values for Palantir and CellRank are based on a prespecified fate-star
surrogate rather than a native tree output.

## 六、建议避免的表述

- 不要写“其他方法 Coverage = 1，因此准确率为100%”。
- 不要把 MORPH 在空对照中的拒绝率直接表述为优于所有非空间方法。
- 不要把 Palantir/CellRank 的 Topo 1-HIM 称为原生树准确率。
- 不要将25个正向数据与5个空间空对照重新混合成一个主准确性均值。
- 不要只报告质量均值而省略 Coverage、有效数据集数和置信区间。

## 七、对应结果文件

- 主热图数值：`results/manuscript_table.csv`
- 主分析置信区间：`results/summary_long.csv`
- 空间空对照：`results/spatial_null_negative_control.csv`
- 30数据集混合汇总（仅审计）：`results/summary_all_conditions_long.csv`
- 完整逐数据集结果：`results/by_dataset.csv`
- 完整逐根审计：`results/by_root.csv`

