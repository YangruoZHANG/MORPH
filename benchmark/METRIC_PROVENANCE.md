# Fig. 3 指标、真值和绘图来源

## 面板来源

| 面板 | 数据 | 原始实现/入口 |
|---|---|---|
| A(1) | sim_K2_sep / metacell_adata.h5ad：obsm spatial、obs origin | plot_figure3_redesign.py / draw_maps |
| A(2) | T0_STATE_NODE_ASSIGNMENTS.csv：dominant membership、branch_id、is_root | draw_maps；同 node 取最大 membership，tie 按 root_id |
| A(3) | 同一节点表 pseudotime | draw_maps；不额外拟合时间 |
| A(4) | T0_STATE_BACKBONE_EDGES.csv：support_source/support_target → unit_id → node_id | draw_maps |
| A(5) | h5ad uns milestone_network、全部 T0 state edges | draw_topology；保留原有汇聚与未连接到 root 的粗状态来源，不补边 |
| B(1) | 1000 个 METRICS.json + frozen overall_metrics.csv | draw_overall |
| B(2) | 1000 个逐对差值 + condition_metrics.csv | draw_conditions / bootstrap_mean |
| B(3) | 每个 condition 的平均 ΔARI 和 Δsigned Spearman | draw_tradeoff |
| C(1–3) | P4_CORE_COMPARISON.csv + P3_DATASET_RESULTS.csv | load / draw_history |

## 四个主要指标

1. Coverage：全部 root 的预测中 `unit_id` 的并集大小 / 输入细胞数。一个细胞出现在多个 root 中仍只计一次。原始入口：`algorithm/validation/run_paper_p4_root_conditioned_benchmark.py::_run_core`。
2. Branch ARI：每个 root 内，预测 `branch_id` 与真值 `TRUE_CELL_BRANCH.csv` 的 edge-identity branch 标签计算 adjusted_rand_score，再对 root 等权平均。不能将所有 root 的标签直接拼接后算一次 ARI。实现：`run_phase_a_case_v1_1.py::t0_metrics`，随后 `run_phase_a_case_v2.py::dual_scale_metrics` 汇总。
3. State-edge F1：使用 `METACELL_STATE_COMPOSITION.csv` 中 probability >= 0.20 的状态支持。对每条预测细边枚举两端状态支持的组合，去除同状态组合，将得到的无向状态对去重，与真值无向 milestone 边集求 precision、recall、F1；最后对 root 等权平均。不是直接比较图中 12 条粗状态边与 9 条真值边，也不是只按 dominant state 投影。实现：`run_phase_a_case_v1_1.py::plausible_pairs` / `t0_metrics`。
4. Signed pseudotime Spearman：在每个 root 中，将预测节点时间与真值 root-conditioned 时间匹配，计算 Spearman，再对 root 等权平均。共享 metacell 的真值使用对应 root 的有效记录。B 保留符号。C 的历史显示用其绝对值，见绘图的 historical 数据处理。

`dual_scale_metrics` 还包含更详细的 coarse branch/topology 诊断；最终 Fig. 3B 的 `topology_F1` 字段映射到该 scorer 的 `fine_edge_F1`，并非将其它诊断指标改名。

## 统计摘要

- 每个模拟集是一对比较，Δ = NicheForest − Root-MSF；总体均值对数据集等权。
- win/loss 按配对差值的正负计数，ties 排除在 exact two-sided sign test 的试验次数之外。
- 四个主指标的 p 值使用 Benjamini–Hochberg 校正为 q。
- Overall CI 精确沿用已冻结表。条件 CI 的抽样方案、次数与随机种子见 README。
- 可接受结构检查：root support connectivity、is_forest、candidate_legal；这些检查的原始状态和逐次诊断表已保存。两种方法都可能满足某一检查，不能把共有属性当作独占优势。

## 图中数字的快速对照

| B 总体指标 | Root-MSF | NicheForest | 平均配对差 |
|---|---:|---:|---:|
| Coverage | 1.0000000000 | 0.9996669572 | −0.0003330428 |
| Branch ARI | 0.0387762700 | 0.1177600214 | +0.0789837514 |
| State-edge F1 | 0.5753942379 | 0.5718778121 | −0.0035164258 |
| Signed Spearman | 0.3293419100 | 0.2988080245 | −0.0305338855 |

这些数值应通过包内逐次 METRICS.json 重新汇总；本表用于人工对照。
