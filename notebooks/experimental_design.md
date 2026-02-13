# 实验设计方法论 — Hierarchical IDS Framework

> **目标**：为分层网络入侵检测框架设计严谨的实验方案，涵盖正则化、拟合优化、统计推断替代方案、Bias-Variance Trade-off、Bootstrap/CV 方法论、以及全面的可视化与评估策略。
> 
> **核心参考文献**：本文档深度融合以下8篇关键论文内容作为实验设计的学术支撑。
> 
> | 类别 | 简称 | 论文 | 在本项目中的角色 |
> |---|---|---|---|
> | **数据** | **[Sharafaldin'18]** | Toward Generating a New Intrusion Detection Dataset... | 数据集基础 (CIC-IDS2017) |
> | **数据** | **[Ring'19]** | A Survey of Network-based Intrusion Detection Data Sets | **数据集选型依据 (Why CIC-IDS2017?)** |
> | **数据** | **[Moustafa'15]** | UNSW-NB15: A comprehensive data set... | **跨数据集泛化验证基准** |
> | **模型** | **[Liu'25]** | TransECA-Net: A Transformer-Based Model... | Stage 2 核心模型架构 |
> | **模型** | **[Kwon'17]** | Deep learning-based network anomaly detection | **Stage 2 深度学习必要性论证** |
> | **ML** | **[Doula'25]** | Analysis of ML-Based Methods for Network Traffic... | Stage 1 随机森林方法论支撑 |
> | **ML** | **[Abu Al-Haija'22]** | ML-Based Darknet Traffic Detection System... | 集成学习与SHAP解释性支撑 |
> | **ML** | **[Kaur'21]** | ML Techniques for Anomaly Detection in Network Traffic | **异常检测通用评估标准** |

---

## 目录
1. [数据基础与选型依据](#1-数据基础与选型依据)
2. [Stage 1 实验设计：RF 正则化与文献支撑](#2-stage-1-实验设计rf-正则化与文献支撑)
3. [Stage 2 实验设计：深度学习必要性与TransECA-Net](#3-stage-2-实验设计深度学习必要性与transeca-net)
4. [P值替代方案（Random Forest 统计推断）](#4-p值替代方案random-forest-的统计推断)
5. [Bias-Variance Trade-off 分析](#5-bias-variance-trade-off-分析)
6. [Bootstrap 重采样法](#6-bootstrap-重采样法)
7. [交叉验证策略](#7-交叉验证策略-cross-validation)
8. [数据与模型结果可视化](#8-数据与模型结果可视化)
9. [综合评估框架](#9-综合评估框架)
10. [实验设计总结表](#10-实验设计总结表)

---

## 1. 数据基础与选型依据

> **核心文献**：
> - **[Sharafaldin'18]** (CIC-IDS2017 原始论文)
> - **[Ring'19]** (数据集综述与评估标准)

### 1.1 为什么选择 CIC-IDS2017? — 基于 [Ring'19] 的评估

[Ring'19] 提出了评估网络入侵检测数据集的15个关键属性。基于此标准，CIC-IDS2017 是目前最符合现代网络环境的数据集之一，优于老旧的 KDD99/NSL-KDD。

| [Ring'19] 评估标准 | CIC-IDS2017 表现 | 竞品 (NSL-KDD) | 本项目获益 |
|-------------------|------------------|---------------|-----------|
| **Year (时效性)** | 2017 (现代流量) | 2009 (过时) | 模型能识别现代攻击(如Heartbleed) |
| **Duration (时长)** | 5天 (完整工作周) | - | 包含完整的周期性流量模式 |
| **Traffic Type (流量)**| 真实+模拟 (B-Profile) | 合成/采样 | Stage 1 能学习真实背景流量分布 |
| **Labelled (标注)** | 流级+包级标注 | 流级 | 支持精细化分类 (Stage 2) |
| **Attack Diversity**| 7大类 14子类 | 4大类 | **验证分层架构处理复杂多分类的能力** |

**结论**：依据 [Ring'19]，CIC-IDS2017 满足 "Evaluation" 和 "Labeling" 的高标准，是验证 Hierarchical Framework 的最佳选择。

### 1.2 CIC-IDS2017 特征体系 ([Sharafaldin'18])

数据集包含约 **80个统计特征**，构成了 Stage 1 的输入空间：

| 特征类别 | 代表性特征 | 对 Stage 1 的作用 |
|---------|----------|-----------------|
| **时间统计** | Flow Duration, IAT Mean/Std | **RF 最依赖的核心特征** (识别DoS/BruteForce) |
| **包/字节统计**| Total Fwd/Bwd Packets, Bytes | 流量规模与非对称性判断 |
| **标志位** | SYN/FIN/RST Count | 协议违规检测 (如 PortScan) |
| **行为模式** | Subflow Stats, Window Size | 连接行为分析 |

### 1.3 实验设计 D1: 基于特征类别的贡献度分析

```python
# 实验 D1: 特征组消融实验
# 验证 [Sharafaldin'18] 中不同特征组对检测贡献的差异
feature_groups = {
    'Time': ['Flow Duration', 'Flow IAT...'],
    'Size': ['Total Packets', 'Total Bytes...'],
    'Flag': ['SYN', 'RST', 'FIN...'],
    'Header': ['Fwd Header Len', 'Bwd Header Len...']
}
# 预期：Time > Flag > Size (针对DoS/BruteForce)
```

---

## 2. Stage 1 实验设计：RF 正则化与文献支撑

> **核心文献**：
> - **[Doula'25]** (ML方法对比)
> - **[Abu Al-Haija'22]** (集成学习优势)
> - **[Kaur'21]** (ML异常检测技术)

### 2.1 技术选型论证

- **RF vs SVM/DT**: [Doula'25] 实验显示 RF 在 CIC-IDS2017 上达到 **99.86%** 准确率，显著优于单棵决策树和SVM。
- **Bagging 的必要性**: [Abu Al-Haija'22] 证明 Bagging 类集成方法（如 Bagging-DT）在 IoT 暗网流量检测中具有 **最高准确率 (99.50%)** 和极低延迟 (**9.09μs**)，完美契合 Stage 1 "快速高精" 的需求。
- **评估指标**: [Kaur'21] 强调在网络流量检测中，**Precision** (查准率) 和 **Recall** (查全率) 比 Accuracy 更重要，尤其是面对类别不平衡时。

### 2.2 Stage 1 实验方案

#### E1: 超参数调优 (Nested CV)

```python
# 基于 [Doula'25] 的建议，重点调整 n_estimators 和 max_feature
param_grid = {
    'n_estimators': [50, 100, 200],      # 集成规模
    'max_depth': [10, 20, None],         # 控制过拟合
    'max_features': ['sqrt', 'log2'],    # 特征扰动
    'class_weight': ['balanced', None]   # [Kaur'21] 强调的不平衡处理
}
```

#### E11: 推理速度基准测试 (Benchmarking)

```python
# 对标 [Abu Al-Haija'22] 的 9.09μs/sample
# 目标：Stage 1 吞吐量 > 100,000 samples/s
import time
start = time.perf_counter()
rf.predict(X_test[:10000])
latency = (time.perf_counter() - start) / 10000 * 1e6
print(f"Latency: {latency:.2f} μs (Target: < 10 μs)")
```

---

## 3. Stage 2 实验设计：深度学习必要性与 TransECA-Net

> **核心文献**：
> - **[Kwon'17]** (Deep learning-based network anomaly detection)
> - **[Liu'25]** (TransECA-Net 模型)

### 3.1 为什么 Stage 2 需要深度学习？— [Kwon'17] 论证

[Kwon'17] 在综述中指出，随着网络攻击日益复杂（如加密流量、变种攻击），传统浅层 ML（如 RF/SVM）面临**特征工程瓶颈**。
- **Deep Learning 优势**：能从原始数据中自动学习高阶特征表示（Representation Learning）。
- **本项目应用**：Stage 1 负责"量"的过滤（传统攻击），Stage 2 负责"质"的分析（复杂/加密攻击）。
- TransECA-Net 正是基于此理念，利用 **1D-CNN (局部)** + **Transformer (全局)** 提取深层流量模式。

### 3.2 TransECA-Net 架构与实验 ([Liu'25])

#### 核心组件功效

1.  **1D-CNN**: 提取局部特征（如连续的数据包大小变化）。
2.  **ECA-Net (Efficient Channel Attention)**: 自适应关注关键通道。
    - 公式：$k = \psi(C) = | \frac{\log_2(C)}{\gamma} + \frac{b}{\gamma} |_{odd}$
    - 优势：比 SE-Net 更轻量，无降维信息损失。
3.  **Transformer Encoder**: 捕获长距离依赖（如 TCP 的握手-传输-断开全过程）。

#### E7 & E8: 正则化与消融实验

为了验证 [Kwon'17] 提出的 "DL 优于 Shallow ML" 以及 [Liu'25] 的架构优势：

| 实验 ID | 模型变体 | 移除组件 | 验证假设 |
|---|---|---|---|
| **E8-A** | **TransECA-Net (Full)** | — | **DL SOTA 性能 ([Liu'25] 98.25%)** |
| **E8-B** | CNN-Only | -Transformer, -ECA | 验证长序列建模的重要性 |
| **E8-C** | RF (Stage 1 复用) | -All DL | **验证 [Kwon'17]: DL 在复杂分类上优于 ML** |

---

## 4. P值替代方案（Random Forest 的统计推断）

> **学术依据**：[Abu Al-Haija'22] 使用博弈论方法 (Shapley) 解释特征重要性

### 4.1 方案选择
RF 不提供 p-value，采用以下替代方案：

1.  **SHAP Values (推荐)**: [Abu Al-Haija'22] 验证了其在流量检测中的有效性。能给出特征对预测结果的正负贡献方向。
2.  **Permutation Importance**: 通过打乱特征值观察精度下降，可计算 statistical significance (p-value)。

```python
# E2: Permutation Importance 显著性检验
from sklearn.inspection import permutation_importance
result = permutation_importance(rf, X_test, y_test, n_repeats=30)
# 对每个特征计算 p-value (H0: 特征不重要)
```

---

## 5. Bias-Variance Trade-off 分析

### 5.1 理论映射
- **Stage 1 (RF)**: 低 Bias（强分类器），低 Variance（Bagging 机制 [Abu Al-Haija'22]）。
- **Stage 2 (DL)**: 低 Bias（深层网络 [Kwon'17]），高 Variance（参数多，易过拟合）。

### 5.2 实验 E4: Bias-Variance 分解
通过 50 次 Bootstrap 重采样训练，量化模型的 Bias 和 Variance，指导正则化策略（如 Stage 2 增加 Dropout）。

---

## 6. Bootstrap 重采样法

### 6.1 应用场景
- **性能置信区间 (CI)**: 计算 F1-score 的 95% CI，由 [Kaur'21] 推荐作为严谨的评估方式（而非单次测试结果）。
- **OOB Score**: 利用 RF 自带的 OOB (Out-of-Bag) 估计作为验证集性能的无偏估计。

---

## 7. 交叉验证策略 (Cross-Validation)

### 7.1 分层设计
- **Stage 1**: **Stratified K-Fold (5-fold)**。处理 [Sharafaldin'18] 提到的类别不平衡。
- **Stage 2**: **Repeated Random Holdout**。深度学习训练成本高，采用多次随机划分（如 5次 80/20 切分）取平均。

---

## 8. 数据与模型结果可视化

1.  **E10: ECA Attention Heatmap**: 可视化 [Liu'25] 中 ECA 模块关注的特征通道。
2.  **E12: t-SNE / UMAP**: 数据流形可视化，展示 Benign 与 Attack 的可分性。
3.  **SHAP Summary Plot**: 结合 [Abu Al-Haija'22]，展示 Top-20 特征的全局影响力。

---

## 9. 综合评估框架

> **新增关键实验**：基于 [Moustafa'15] (UNSW-NB15) 的泛化能力验证

### 9.1 泛化能力实验 (E15)
为避免模型仅拟合 CIC-IDS2017 的特定伪影（Artifacts），引入 **UNSW-NB15** 作为测试集。
- **挑战**: 两数据集特征空间不同。
- **方案**: 选取两者共有的通用特征（如 Duration, Packet Count, Byte Count）进行 **Transfer Learning** 或 **Cross-Evaluation**。
- **目的**: 证明框架不仅仅是"记住了" CIC-IDS2017，而是学到了通用的入侵模式。

### 9.2 统计检验
- **McNemar Test**: 对比 Stage 1 (RF) 与其他 ML 模型（如 XGBoost）的显著性差异。

---

## 10. 实验设计总结表

| # | 实验名称 | 方法 | 输出 | 核心文献支撑 |
|---|---|---|---|---|
| **E1** | RF 超参调优 | Nested CV + GridSearch | 最优参数 | [Doula'25] |
| **E2** | 特征重要性 | Permutation + SHAP | p-value替代 | **[Abu Al-Haija'22]** |
| **E3** | 特征选择 | Information Gain | Top-K 特征 | [Doula'25] |
| **E4** | Bias-Variance | Bootstrap 分解 | B/V 数值 | [Abu Al-Haija'22] |
| **E5** | 复杂度曲线 | Validation Curve | U型图 | — |
| **E6** | 性能 CI | 1000次 Bootstrap | 95% 置信区间 | **[Kaur'21]** |
| **E7** | DL 正则化 | Dropout/L2/Smoothing | 性能增益 | [Liu'25] |
| **E8** | **DL 消融实验** | -ECA, -Transformer | 模块贡献 | **[Liu'25], [Kwon'17]** |
| **E11**| **推理速度** | Latency Benchmarking | μs/sample | **[Abu Al-Haija'22]** |
| **E12**| 特征类别分析 | 按 [Sharafaldin'18] 分组 | 类别重要性 | **[Sharafaldin'18]** |
| **E13**| **不平衡处理**| SMOTE vs ClassWeight | F1-Macro | [Doula'25], [Ring'19] |
| **E15**| **跨库验证** | Test on UNSW-NB15 | 泛化分数 | **[Moustafa'15]** |

### 项目展讲解逻辑
1.  **数据选型 ([Ring'19])**: 为什么不用 KDD99？因为 Ring 的评估标准指向 CIC-IDS2017。
2.  **分层架构 ([Kwon'17])**: 为什么分两层？因为 Kwon 指出复杂攻击需要 DL，而 Abu Al-Haija 证明 RF 处理简单流量极快。
3.  **技术细节 ([Liu'25] + [Doula'25])**: RF 怎么调优？TransECA 怎么设计？
4.  **结果验证 ([Moustafa'15] + [Kaur'21])**: 不仅在测试集准，CI 窄，且能泛化到 UNSW-NB15。
