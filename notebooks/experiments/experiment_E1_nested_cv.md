# 实验 E1: Random Forest 超参数调优 (Nested CV)

> **归属**: Stage 1 (Traffic Filtering)
> **目标**: 通过嵌套交叉验证 (Nested CV) 找到 Random Forest 在 CIC-IDS2017 数据集上的最优超参数组合，确立性能基线。

---

## 1. 实验目标 (Objectives)

1.  **主目标**: 确定最优超参数 $\theta^*$，使得模型在未知数据上的 F1-Macro Score 最大化。
2.  **次要目标**:
    *   验证 [Doula'25] 的参数结论。
    *   获得模型性能的无偏估计 (Unbiased Performance Estimation)。
    *   建立 Stage 1 vs Stage 2 的对比基准。

## 2. 理论依据 (Theoretical Basis)

**为什么使用 Nested CV？**
*   **Data Leakage Prevention**: 如果在整个数据集上做 GridSearch 选出最优参数，再在同一个数据集上报告精度，会导致性能高估 (Optimistic Bias)。
*   **无偏估计**:
    *   **Inner Loop**: 负责模型选择 (Model Selection)。
    *   **Outer Loop**: 负责性能评估 (Model Evaluation)。
*   **数学形式**:
    $$ \text{CV Score} = \frac{1}{K} \sum_{k=1}^{K} \text{Score}(\hat{f}_{\text{Inner}}(D_{train}^{(k)}), D_{test}^{(k)}) $$

## 3. 参数搜索空间 (Search Space)

基于 [Doula'25] 和 [Abu Al-Haija'22] 的研究：

```python
param_grid = {
    # ===== 1. 集成规模 =====
    'n_estimators': [50, 100, 200, 300],
    # 理论: 更多树 -> 方差更低，但收益递减。Doula'25 建议 100。

    # ===== 2. 树深度 =====
    'max_depth': [10, 20, 30, None],
    # 理论: 控制过拟合。深度越深，Bias 越低，Variance 越高。

    # ===== 3. 特征采样 =====
    'max_features': ['sqrt', 'log2', 0.5],
    # 理论: Bagging 的核心去相关机制。

    # ===== 4. 分裂质量 =====
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],

    # ===== 5. 类别不平衡 =====
    'class_weight': ['balanced', 'balanced_subsample', None],
    # 关键: 处理 IDS 数据中的类别不平衡。

    # ===== 6. Bootstrap =====
    'bootstrap': [True],
    'oob_score': [True]
}
# 总组合数: 4*4*3*3*3*3 = 1296
```

## 4. 评估指标 (Metrics)

*   **Primary**: **F1-Macro** (平衡各类别的性能，不被多数类主导)
*   **Secondary**: Precision (Macro), Recall (Macro), OOB Score
*   **Constraint**: 推理延迟 < 10 $\mu s$ / sample (需结合 E11 验证)

## 5. 完整代码实现 (Implementation)

代码文件: `src/experiments/exp_e1_nested_cv.py` (拟定)

```python
"""
实验 E1: Random Forest 超参数调优 (Nested Cross-Validation)
参考文献: [Doula'25], [Kaur'21]
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt

# ... (此处包含原文档中的完整代码逻辑，省略以避免冗余，实际执行时应包含完整脚本) ...
```

*(注：详细代码逻辑即原 `experimental_design.md` 中的代码块，建议在实施时直接生成 Python 脚本)*

## 6. 预期结果与验证标准

| 指标 | 目标值 | 验证逻辑 |
| :--- | :--- | :--- |
| **F1-Macro** | $\ge 98.0\%$ | 优于各类单模型基线 |
| **Stability** | Std $< 0.02$ | 5折结果波动小 |
| **Robustness** | 优选参数一致性 | 至少3折选出相似参数 |
| **OOB Score** | $\approx$ Test Score | 验证 Bootstrap 有效性 |

## 7. 后续衔接

*   **Result Output**: `results/E1_nested_cv_results.json`
*   **Next Step**:
    *   使用 E1 确定的最优参数训练最终模型 `models/stage1_rf_best.pkl`。
    *   进入 **E2 (Feature Importance)** 分析特征贡献。
