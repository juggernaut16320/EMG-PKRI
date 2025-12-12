# Day3-Day12 最小闭环执行方案

> **目标**：在 6 天内完成核心实验闭环，验证 EMG 方法的有效性

## 📊 方案概述

### 核心路径（必须完成）
完成从 baseline 评估 → 不确定性分析 → 知识后验 → EMG 融合 → 效果验证的完整闭环，证明 EMG 方法优于 baseline。

### 时间预算
- **总时间**：6 天
- **核心任务**：4-5 天
- **缓冲时间**：1-2 天

---

## ✅ 任务清单

### 阶段 1：基础准备（0.5 天）

#### 任务 1.1：完成 Day1 剩余工作
**任务**：实现 `hardset_maker.py`  
**时间**：0.5 天（开发 2-3h，运行 1-2h）

**工作内容**：
- 加载训练好的 baseline 模型
- 对 dev/test 集进行预测，得到 baseline 的预测结果
- 使用 `gemma-3-27b` 对相同样本进行打标（teacher 预测）
- 找出分歧样本：
  - baseline 预测错误但 teacher 预测正确
  - baseline 高置信但预测错误
- 从分歧样本中抽取 500-2000 条作为 `hard_eval_set.jsonl`

**输出**：
- `data/hard_eval_set.jsonl`

**依赖**：
- ✅ Day2 训练好的 baseline 模型
- ✅ `data/dev.jsonl` 和 `data/test.jsonl`

---

### 阶段 2：Baseline 评估（0.5 天）

#### 任务 2.1：Day3 - 基线评估 + 高置信错误分析
**时间**：0.5 天（开发 2-3h，运行 0.5-1h）

**工作内容**：
- 实现 `scripts/eval_baseline.py`
- 在 `test.jsonl` 上评估 baseline：
  - Accuracy / F1 / confusion matrix
- 在 `hard_eval_set.jsonl` 上单独评估
- 选出一批高置信错误样本（预测概率>0.8但错了）

**输出**：
- `output/metrics_baseline.json`
- `output/high_conf_error_samples.jsonl`
- 评估报告

**依赖**：
- ✅ Day2 baseline 模型
- ✅ `data/test.jsonl`
- ✅ `data/hard_eval_set.jsonl`（任务 1.1 输出）

---

### 阶段 3：不确定性分析（0.5 天）

#### 任务 3.1：Day4 - 不确定性指标 u 构建 + 分桶分析
**时间**：0.5 天（开发 3-4h，运行 0.5h）

**工作内容**：
- 实现 `scripts/uncertainty_analysis.py`
- 对 dev/test 的每条样本，计算：
  - `u = 1 - max_c pθ(c|x)`
- 按 u 分桶（10 桶），统计：
  - 样本数、错误率、平均置信度等
- 绘制 `u_vs_error.png` 图表

**输出**：
- `output/uncertainty_buckets.csv`
- `output/u_vs_error.png`
- 不确定性分析报告

**依赖**：
- ✅ Day3 baseline 评估结果（包含预测概率）

---

### 阶段 4：知识后验构建（1 天）

#### 任务 4.1：Day6 - 构造规则版知识后验 q₀(c|z)
**时间**：1 天（开发 4-5h，运行 1-2h）

**工作内容**：
- 实现 `scripts/q0_builder.py`
- 基于色情/涉政/辱骂等词表、alias、正则规则构造知识信号
- 输出 coarse 二分类：[p_non_sensitive, p_sensitive]
- 内部可按 porn/politics/abuse 设计不同强度权重
- 对 train/dev/test/hardset 全部生成 q₀ 后验

**输出**：
- `data/q0_train.jsonl`
- `data/q0_dev.jsonl`
- `data/q0_test.jsonl`
- `data/q0_hard.jsonl`

**依赖**：
- ✅ 词表文件（需要准备或创建 `configs/lexicons/`）
- ✅ `data/train.jsonl`, `data/dev.jsonl`, `data/test.jsonl`, `data/hard_eval_set.jsonl`

**注意**：
- 如果时间紧张，可以先用简化版（仅词表匹配）
- 后续可以扩展词频/PMI、embedding cosine 等特征

---

### 阶段 5：EMG 融合（1 天）

#### 任务 5.1：Day7 - EMG 分桶 α 搜索
**时间**：0.5 天（开发 3-4h，运行 1-2h）

**工作内容**：
- 实现 `scripts/emg_bucket_search.py`
- 对每个 u bucket，枚举 α ∈ {0, 0.25, 0.5, 0.75, 1}
- 在 dev 集上算 NLL / F1，选出每个 bucket 的 α*

**输出**：
- `output/bucket_alpha_star.csv`

**依赖**：
- ✅ Day4 不确定性分桶结果
- ✅ Day6 q₀ 后验结果
- ✅ Day3 baseline 预测概率

#### 任务 5.2：Day8 - PAV 保序回归生成单调 α(u)
**时间**：0.5 天（开发 1-2h，运行 0.1h）

**工作内容**：
- 实现 `scripts/emg_fit_alpha_u.py`
- 用 PAV 对 (u_bucket, α*) 做保序回归
- 输出 α(u) 查表 / 函数

**输出**：
- `output/alpha_u_lut.json`（查表）
- `output/alpha_u_curve.png`

**依赖**：
- ✅ Day7 `bucket_alpha_star.csv`

---

### 阶段 6：效果验证（0.5 天）

#### 任务 6.1：Day9 - EMG 效果验证
**时间**：0.5 天（开发 3-4h，运行 0.5h）

**工作内容**：
- 实现 `scripts/eval_emg.py`
- 在 test & hardset 上对比：
  1. baseline
  2. 固定 α 融合
  3. EMG（α(u) 自适应）

**输出**：
- `output/metrics_emg.json`
- `output/comparison_table.csv`
- `output/comparison_charts.png`

**依赖**：
- ✅ Day3 baseline 预测
- ✅ Day6 q₀ 后验
- ✅ Day8 α(u) 查表

---

### 阶段 7：总结报告（0.5 天）

#### 任务 7.1：Day12 - 可视化 + 初步结论总结
**时间**：0.5 天（开发 3-4h，运行 0.5h）

**工作内容**：
- 实现 `scripts/generate_reports.py`
- 汇总各天输出的 metrics / 图表 / CSV
- 自动化生成 3-5 张核心图
- 输出对比表
- 写一份 `initial_experiment_summary.md`

**输出**：
- `output/initial_experiment_summary.md`
- `output/figures/`（核心图表）
- `output/comparison_tables.csv`

**依赖**：
- ✅ 前面所有阶段的输出结果

---

## 📅 时间安排建议

### 方案 A：顺序执行（推荐）
```
Day 1: 任务 1.1 (hardset_maker) + 任务 2.1 (baseline 评估)
Day 2: 任务 3.1 (不确定性分析) + 开始任务 4.1 (知识后验)
Day 3: 完成任务 4.1 (知识后验)
Day 4: 任务 5.1 (EMG 搜索) + 任务 5.2 (PAV 拟合)
Day 5: 任务 6.1 (EMG 验证)
Day 6: 任务 7.1 (总结报告) + 缓冲/优化
```

### 方案 B：并行优化
- 任务 1.1 和任务 2.1 可以并行（如果 hardset_maker 先完成）
- 任务 4.1 运行时间较长，可以在后台运行
- 任务 5.1 的网格搜索可以并行化

---

## 🔄 可选任务（时间允许时）

### Day5 - 校准分析（可选）
**时间**：0.5 天  
**工作内容**：
- 绘制 Reliability Diagram
- 计算 ECE/MCE
- 可以与 Day4 合并实现

**优先级**：中（有助于理解 baseline 的过度自信问题）

### Day10 - PKRI 构建（可选）
**时间**：1.5 天  
**工作内容**：
- 构建轻量特征（词表命中、embedding cosine、子标签匹配等）
- 用 LR / LightGBM 训练 PKRI 模型
- 输出 q_PKRI(c|z)

**优先级**：低（可以延后，不影响核心闭环）

### Day11 - EMG + PKRI 联合融合（可选）
**时间**：0.5 天  
**工作内容**：
- 在 `eval_emg.py` 中增加 `--knowledge_source qpkri` 模式
- 对比 EMG+q₀ vs EMG+q_PKRI

**优先级**：低（依赖 Day10，可以延后）

---

## ⚠️ 风险与应对

### 风险 1：知识后验构建耗时过长
**应对**：
- 先用简化版（仅词表匹配）
- 后续可以扩展特征

### 风险 2：EMG 搜索计算量大
**应对**：
- 减少 α 枚举粒度（如 {0, 0.5, 1}）
- 减少 u 分桶数量（如 5 桶）
- 并行化计算

### 风险 3：调试时间超出预期
**应对**：
- 预留 1-2 天缓冲时间
- 优先保证核心功能，细节可以后续优化

### 风险 4：词表资源不足
**应对**：
- 先使用简单的敏感词列表
- 可以从子标签数据中提取关键词

---

## 📋 检查清单

### 阶段 1 检查点
- [ ] `hardset_maker.py` 实现完成
- [ ] `data/hard_eval_set.jsonl` 生成（500-2000 条）

### 阶段 2 检查点
- [ ] `eval_baseline.py` 实现完成
- [ ] baseline 在 test 和 hard 集上的评估结果
- [ ] 高置信错误样本已识别

### 阶段 3 检查点
- [ ] `uncertainty_analysis.py` 实现完成
- [ ] u 分桶结果和图表生成
- [ ] 不确定性 u 与错误率的相关性验证

### 阶段 4 检查点
- [ ] `q0_builder.py` 实现完成
- [ ] 所有数据集的 q₀ 后验已生成
- [ ] 词表资源准备完成

### 阶段 5 检查点
- [ ] `emg_bucket_search.py` 实现完成
- [ ] 每个 bucket 的最优 α* 已找到
- [ ] `emg_fit_alpha_u.py` 实现完成
- [ ] α(u) 查表已生成

### 阶段 6 检查点
- [ ] `eval_emg.py` 实现完成
- [ ] baseline vs 固定融合 vs EMG 的对比结果
- [ ] EMG 效果验证完成

### 阶段 7 检查点
- [ ] `generate_reports.py` 实现完成
- [ ] 核心图表已生成
- [ ] 实验总结报告已完成

---

## 🎯 成功标准

### 最小闭环成功标准
1. ✅ 完成 baseline 评估，了解错误分布
2. ✅ 证明不确定性 u 与错误率有相关性
3. ✅ 成功构建知识后验 q₀
4. ✅ 找到最优 α(u) 门控函数
5. ✅ 验证 EMG 优于 baseline 和固定融合
6. ✅ 生成完整的实验报告

### 核心指标
- **F1 提升**：EMG 相比 baseline 的 F1 提升
- **高置信错误减少**：在 hard 集上的错误率降低
- **不确定性相关性**：u 与错误率的相关系数 > 0.3

---

## 📝 执行建议

1. **每日检查点**：每天结束时检查当天任务完成情况
2. **代码复用**：Day9 和 Day11 可以复用部分代码
3. **并行运行**：计算密集型任务（如 q₀ 生成、EMG 搜索）可以在后台运行
4. **版本控制**：每个阶段完成后提交代码，便于回滚
5. **文档同步**：及时更新各 Day 的文档，记录实现细节

---

## 🔗 相关文档

- [ReadMe.md](./ReadMe.md) - 完整项目说明
- [Day1.md](./Day1.md) - Day1 任务详情
- [Day2.md](./Day2.md) - Day2 任务详情（已完成）
- [Day3.md](./Day3.md) - Day3 任务详情（待创建）
- [Day4.md](./Day4.md) - Day4 任务详情（待创建）
- ...

---

**最后更新**：2025-12-12  
**状态**：待执行

