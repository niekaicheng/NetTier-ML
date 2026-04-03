# 实验 E2: Stage 1 (RF) 特征重要性分析 — SHAP

> **归属**: Stage 1 (Traffic Filtering)  
> **目标**: 通过 SHAP (SHapley Additive exPlanations) 博弈论归因方法解释随机森林的决策依据，验证模型聚焦于有意义的网络流量特征。

---

## 1. 实验目标 (Objectives)

1. **主目标**: 揭示 Stage 1 RF 模型在二分类（Benign vs Attack）任务中的特征决策逻辑。
2. **次要目标**:
    * 对比 SHAP 归因 与 RF 内置 Gini Importance 的排序一致性。
    * 验证模型关注的特征是否符合网络安全领域知识 ([Sharafaldin'18])。
    * 为 E10（Stage 2 可解释性）建立跨模型对比基线。

## 2. 理论依据 (Theoretical Basis)

**为什么使用 SHAP？**
* **Shapley 一致性公理**: SHAP 基于合作博弈论的 Shapley 值，满足局部精确性、缺失性、一致性三大公理，提供理论上唯一公平的特征归因方案。
* **交互效应捕获**: 与 Gini Importance 仅测量单特征分裂贡献不同，SHAP 能捕获特征间的协同交互效应。
* **TreeExplainer 高效算法**: 对树模型有多项式时间精确解，无需近似。

## 3. 实验配置 (Configuration)

| 参数 | 值 |
|------|------|
| 模型 | `stage1_rf_stratified.joblib` (50 trees, 76 features) |
| 任务 | 二分类 (Benign=0, Attack=1) |
| SHAP 方法 | TreeExplainer (精确计算) |
| 采样策略 | 分层抽样，每类最多 500 个样本 |
| 总样本数 | **6,068** |
| 特征数 | **76** |
| 数据来源 | `archive/*.parquet` (CIC-IDS2017) |

## 4. 实验结果 (Results)

### Top-10 特征排名

| 排名 | 特征 | SHAP (mean\|abs\|) | RF Gini Importance | RF 排名 |
|:---:|------|:---:|:---:|:---:|
| 1 | Bwd Packet Length Std | 0.031994 | 0.060602 | 4 |
| 2 | Init Bwd Win Bytes | 0.028535 | 0.011535 | **23** |
| 3 | Bwd Packet Length Mean | 0.027812 | 0.072389 | 3 |
| 4 | Packet Length Variance | 0.024618 | 0.105389 | 1 |
| 5 | Fwd IAT Min | 0.023699 | 0.011646 | **22** |
| 6 | Avg Packet Size | 0.020887 | 0.075228 | 2 |
| 7 | Fwd IAT Total | 0.020051 | 0.034290 | 8 |
| 8 | Avg Bwd Segment Size | 0.019977 | 0.056386 | 7 |
| 9 | Bwd Packet Length Max | 0.018957 | 0.056640 | 6 |
| 10 | Bwd Packets Length Total | 0.018582 | 0.022153 | 17 |

### 排名一致性验证

| 指标 | 值 | 解读 |
|------|------|------|
| Spearman ρ | **0.9444** | SHAP 与 RF Gini 高度一致 |
| p-value | 1.70e-37 | 极显著 |
| SHAP 计算耗时 | 235.1s | — |

### 关键发现

1. **`Init Bwd Win Bytes` 的排名跃升**: SHAP 将其从 Gini 的 #23 提升至 #2，跃升 21 位。这是 SHAP 捕获特征交互效应的直接体现——该特征与周围时序/载荷特征协同作用时，对决策边界的影响远超其独立分裂贡献。
2. **特征聚类覆盖两大域知识维度**: Payload 长度统计 + IAT 时间间隔，与 [Sharafaldin'18] 确立的 DoS/BruteForce 核心特征完全吻合。
3. **双向判别特征**: `Fwd IAT Min` 呈双向分布——低值（高频攻击轰炸）映射负区，高值（正常流量）映射正区，单一特征同时编码攻击和正常签名。

## 5. 输出文件 (Output)

| 文件 | 说明 |
|------|------|
| `results/E2_shap_summary.png` | SHAP Beeswarm 汇总图 |
| `results/E2_shap_bar.png` | SHAP 柱状图 |
| `results/E2_shap_vs_rf.png` | SHAP vs RF 重要性对比散点图 |
| `results/E2_feature_importance.csv` | 76 特征完整排名 CSV |
| `results/E2_shap_results.json` | 结构化结果 JSON |
| `results/E2_shap_report.txt` | 详细文字报告 |

## 6. 后续衔接

* **向前**: 建立 Stage 1 特征归因基线。
* **向后**: E10 将对 Stage 2 (TransECA-Net) 执行 IG + Attention 归因，形成跨模型三角验证。
* **代码**: `src/experiments/exp_e2_shap.py`
