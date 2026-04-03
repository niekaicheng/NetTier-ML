# 实验 E4: 偏差-方差分析 (Bias-Variance Analysis)

> **归属**: Stage 1 + Stage 2 (系统级理论分析)  
> **目标**: 通过偏差-方差分解验证两阶段模型的统计学习理论特性，为超参数选择提供理论依据。

---

## 1. 实验目标 (Objectives)

1. **主目标**: 验证 Stage 1 RF 的 $B=50$ 参数选择是否处于方差收敛点。
2. **次要目标**:
    * 分析 Stage 2 TransECA-Net 的泛化间隙 (Generalization Gap)，验证是否过拟合。
    * 绘制模型容量 vs 性能曲线，量化从 CNN-Only 到 TransECA 的必要性。
    * 通过学习曲线确认 RF 的数据饱和特性。

## 2. 理论依据 (Theoretical Basis)

**Bagging 方差分解定理 (Breiman)**:
$$\text{Var}_{\text{ensemble}} = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$
其中 $\rho$ 为树间相关系数，$B$ 为树数量。当 $B$ 足够大时，$\frac{1-\rho}{B}$ 项趋近于零，残余误差完全由 $\rho\sigma^2$ 主导。

## 3. 实验配置 (Configuration)

### 3.1 RF OOB 分析
| 参数 | 值 |
|------|------|
| 树数量范围 | 10, 20, 30, ..., 200 (共 20 个点) |
| max_depth | 20 |
| 指标 | OOB Error (1 - OOB Score) |

### 3.2 DL 学习曲线
| 参数 | 值 |
|------|------|
| 数据来源 | `results/training_log.json` (Stage 2 实际训练日志) |
| 指标 | Train Loss / Val Loss / Generalization Gap |

### 3.3 容量对比
| 参数 | 值 |
|------|------|
| 模型 | CNN-Only (2,703), Full TransECA (301,460), No-ECA (301,455), TransECA Prod (301,460) |
| 指标 | Test Accuracy, W-F1, M-F1 |

### 3.4 RF 学习曲线
| 参数 | 值 |
|------|------|
| 训练数据比例 | 1%, 2%, 5%, 10%, 20%, 50%, 100% |
| 指标 | Train Accuracy, Test Accuracy, Gap |

## 4. 实验结果 (Results)

### 4.1 RF OOB 误差收敛

| 树数量 $B$ | OOB Error | 备注 |
|:---:|:---:|------|
| 10 | 0.00361 | 初始高方差 |
| 30 | 0.00184 | 快速下降 |
| **50** | **0.00173** | **拐点** |
| 100 | 0.00170 | 收敛 |
| 200 | 0.00171 | 几乎无变化 |

**结论**: $B=50$ 后 OOB Error 从 0.00173 到 0.00171，变化仅 0.00002，已达方差收敛。

### 4.2 DL 泛化间隙

| 指标 | 值 |
|------|------|
| 初始 Gen Gap (Epoch 1) | +0.529 |
| 最终 Gen Gap (Epoch 30) | **+0.006** |
| 最终 Train Acc | 92.01% |
| 最终 Val Acc | 92.48% |

**结论**: Gen Gap = +0.006 极小，推翻了 [Kwon'19] "DL 必然高方差"的预测。AdamW + CosineAnnealing 有效抑制了过拟合。

### 4.3 容量跃迁

| 模型 | 参数量 | Test Acc | W-F1 | M-F1 |
|------|:---:|:---:|:---:|:---:|
| CNN-Only | 2,703 | 61.89% | 0.668 | 0.235 |
| TransECA Prod | 301,460 | **93.00%** | **0.950** | **0.800** |
| **跃迁幅度** | — | **+31.1pp** | **+0.282** | **+0.565** |

### 4.4 RF 学习曲线

| 训练数据比例 | Train Acc | Test Acc | Gap |
|:---:|:---:|:---:|:---:|
| 1% (2,000) | 1.0000 | 0.9894 | 0.0106 |
| 10% (20,000) | 0.9999 | 0.9962 | 0.0037 |
| 100% (200,000) | 0.9997 | **0.9981** | **0.0016** |

**结论**: Gap@100% = 0.0016，低方差已确认。同时 Test Acc 在大数据量下趋于饱和，说明 RF 的假设空间已被充分利用。

## 5. 理论预测 vs 实验验证

| 预测 | 理论来源 | 实验结果 | 是否验证 |
|------|------|------|:---:|
| RF 低方差 | Breiman: Bagging ↑B 降低 Var | OOB 50→200 几乎不变 | ✅ |
| RF 低偏差 | 决策树是强学习器 | L-Curve Gap@100% = 0.0016 | ✅ |
| DL 高方差 | [Kwon'19] | Gen Gap = +0.006（低） | ❌ 被推翻 |

## 6. 输出文件 (Output)

| 文件 | 说明 |
|------|------|
| `results/E4_oob_vs_trees.png` | OOB 误差 vs 树数量曲线 |
| `results/E4_dl_learning_curves.png` | DL Train/Val Loss 学习曲线 |
| `results/E4_complexity_vs_perf.png` | 模型容量 vs 性能图 |
| `results/E4_rf_learning_curve.png` | RF 学习曲线 |
| `results/E4_bias_variance_results.json` | 结构化结果 JSON |
| `results/E4_bias_variance_report.txt` | 详细文字报告 |

## 7. 后续衔接

* **向前**: E1 确定超参数 → E4 验证 $B=50$ 的理论最优性。
* **向后**: E11 验证 $B=50$ 带来的 6.12μs 低延迟是否满足实时需求。
* **关联**: E8 消融实验提供 CNN-Only / No-ECA / Full 的容量对比数据。
* **代码**: `src/experiments/exp_e4_bias_variance.py`
