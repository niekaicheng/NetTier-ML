# 实验 E17: 决策阈值调优 (Threshold Optimization)

> **归属**: Stage 1 (Traffic Filtering)  
> **目标**: 确定 Stage 1 RF 的最优决策阈值 $\tau$，平衡检测率 (Recall) 和通过率 ($\alpha$)。

---

## 1. 实验目标 (Objectives)

1. **主目标**: 在 RF 后验概率 $P(\text{Attack}|x)$ 上扫描阈值，找到最优操作点。
2. **次要目标**:
    * 量化 Recall-Efficiency 曲线的精确形状。
    * 确定泄漏到 Stage 2 的计算负载上界 ($\alpha$)。
    * 为系统级推理成本计算提供 $\alpha$ 参数。

## 2. 理论依据 (Theoretical Basis)

**阈值调整原理**: RF 输出后验概率 $P(\text{Attack}|x) \in [0,1]$。当 $P > \tau$ 时，样本被标记为"可疑"并转发至 Stage 2。

* $\tau$ 越低 → Recall 越高（少漏检），但 $\alpha$ 越大（更多流量转发给 Stage 2）。
* $\tau$ 越高 → Recall 降低，但 $\alpha$ 减小。
* **目标**: 找到 $\tau^*$ 使得 Recall ≥ 99.9% 且 $\alpha$ 最小化。

## 3. 实验配置 (Configuration)

| 参数 | 值 |
|------|------|
| 模型 | `models_chk/stage1_rf_best.pkl` (E1 输出) |
| 标签编码器 | `models_chk/label_encoder_e1.joblib` |
| 阈值范围 | 0.00 ~ 1.00（步长 0.01） |
| 扫描点数 | 101 |
| 数据 | CIC-IDS2017 测试集 |

## 4. 实验结果 (Results)

### 4.1 关键阈值对比

| 阈值 $\tau$ | Recall | Pass-through $\alpha$ | Precision |
|:---:|:---:|:---:|:---:|
| 0.00 | 1.000 | 100.00% | 0.145 |
| 0.01 | 0.9997 | 16.72% | 0.870 |
| **0.06** | **0.9992** | **15.25%** | **0.953** |
| 0.10 | 0.9982 | 14.89% | 0.975 |
| 0.50 | 0.9928 | 14.50% | 0.996 |
| 0.99 | 0.9465 | 13.77% | 1.000 |

### 4.2 最优操作点

| 指标 | 值 |
|------|------|
| 最优阈值 $\tau^*$ | **0.06** |
| 对应 Recall | **99.92%** (≈99.9% 目标) |
| 通过率 $\alpha$ | **15.25%** |
| Precision | 0.953 |
| FPR | ≈ 0.001 |

### 4.3 关键分析

1. **$\tau=0.06$ 是最优平衡点**: 在保持 Recall ≥ 99.9% 的约束下，$\alpha$ 从 $\tau=0.01$ 的 16.72% 下降至 15.25%，减少约 1.47pp 的计算负载。
2. **$\alpha$ 对 $\tau$ 不敏感区间**: $\tau \in [0.06, 0.99]$ 范围内，$\alpha$ 仅从 15.25% 降至 13.77%，变化仅 1.48pp —— 说明绝大多数"旁观者流量"已被首次筛选排除，剩余流量是真正的边界样本。
3. **系统推理成本输入**:
   $$E[C] = C_1 + \alpha(\tau) \cdot C_2 = 6.12 + 0.1525 \times 500 = 82.37 \mu s$$

## 5. 输出文件 (Output)

| 文件 | 说明 |
|------|------|
| `results/E17_threshold_tuning_20260218_202157.json` | 完整阈值扫描结果 JSON |
| `results/E17_threshold_tuning_20260218_202157.png` | Recall-Efficiency 曲线 |

## 6. 后续衔接

* **向前**: E1 确定最优 RF 参数 → E17 在此基础上微调操作阈值。
* **向后**: $\alpha=15.25\%$ 直接用于 E11 (延迟基准) 和 E14 (逃逸率) 的系统级计算。
* **系统级串联**: $\tau=0.06$ → Stage 1 Recall=99.9% → $\alpha=15.25\%$ → $E[C]=82.37\mu s$ → 6.07× 加速。
* **代码**: `src/experiments/exp_e17_threshold.py`
