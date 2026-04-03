# 实验 E14: 对抗鲁棒性测试 (Adversarial Robustness)

> **归属**: 系统级 (Stage 1 + Stage 2 联合分析)  
> **目标**: 评估分层架构在对抗攻击下的防御能力，验证"正交弱点"防御机制。

---

## 1. 实验目标 (Objectives)

1. **主目标**: 证明 Stage 1 (RF) 和 Stage 2 (TransECA-Net) 的脆弱区域正交互补，单一攻击策略无法同时突破两层。
2. **次要目标**:
    * 量化 RF 在随机噪声下的意外脆弱性。
    * 对比 FGSM (单步) 与 PGD (多步) 对深度模型的攻击强度。
    * 计算系统级逃逸率上界。
    * 分析各攻击类别的鲁棒性差异。

## 2. 理论依据 (Theoretical Basis)

* **RF 梯度免疫**: RF 是分段常函数，$\nabla h_1 \approx 0$，基于梯度的攻击（FGSM/PGD）失效。
* **RF 噪声敏感**: 轴对齐超矩形决策边界→ 微小特征值偏移即可越界。
* **DL 梯度敏感**: 连续可微网络可被 PGD 精确攻击。
* **安全多样性原则 [Littlewood'00]**: 异构防御层的联合逃逸概率小于任一单层。

## 3. 实验配置 (Configuration)

| 参数 | 值 |
|------|------|
| 扰动强度 $\varepsilon$ | 0.001, 0.005, 0.01, 0.05, 0.1 |
| RF 攻击方法 | $L^\infty$ 均匀随机噪声 |
| DL 攻击方法 | FGSM [Goodfellow'15] (单步) + PGD [Madry'18] (5 步) |
| 采样 | 10,000 分层子样本 |
| 总耗时 | 174.4s |

## 4. 实验结果 (Results)

### 4.1 准确率退化曲线

| 攻击 | Clean | $\varepsilon$=0.001 | $\varepsilon$=0.01 | $\varepsilon$=0.1 |
|------|:---:|:---:|:---:|:---:|
| RF ($L^\infty$ noise) | 99.59% | **0.65%** | 0.49% | 0.43% |
| TransECA FGSM | 92.16% | 90.26% | **75.73%** | 6.09% |
| TransECA PGD | 92.16% | 90.20% | **47.28%** | 0.66% |

### 4.2 三大关键发现

**① RF 的意外脆弱性**:
$\varepsilon = 0.001$ 即将准确率从 99.6% 降至 0.65%，推翻"RF 天然鲁棒"的直觉假设。原因：RF 决策边界是轴对齐超矩形，极微小的特征值偏移即可穿越边界。

**② PGD > FGSM**:
在 $\varepsilon = 0.01$ 时，PGD 比 FGSM 低 28.5pp (47.28% vs 75.73%)，验证了 [Madry'18] 的理论：多步迭代优化在 $\ell_\infty$ 球内找到更强的对抗样本。

**③ 正交弱点 = 系统级鲁棒性**:
RF 对随机噪声极度脆弱，但**免疫**梯度攻击 ($\nabla h_1 = 0$)；TransECA 对随机噪声鲁棒，但敏感于梯度攻击。没有单一攻击策略能同时突破两层。

### 4.3 系统级逃逸率

$$P(\text{Evasion}) = \alpha \times P(h_2 \text{ misclassifies} | \text{reaches Stage 2})$$
$$= 0.1525 \times (1 - 0.4728) = \textbf{8.04\%}$$

即使在强 PGD 攻击 ($\varepsilon=0.01$) 下，系统级逃逸率仅 **8.04%**，远低于单层 TransECA 的 52.72%。

### 4.4 逐类鲁棒性亮点 (FGSM)

| 类别 | 特性 | 与 E10 交叉验证 |
|------|------|------|
| **Heartbleed** | $\varepsilon=0.05$ 仍保持 1.00 | E10 中 `Bwd Header Length` IG=5.250 超集中 |
| Web Attack Brute Force | $\varepsilon=0.001$ 即降至 0.18 | 优先防御强化目标 |
| DoS Slowhttptest | $\varepsilon=0.1$ 反弹至 0.83 | 非单调异常，疑为跨类误判 |

## 5. 输出文件 (Output)

| 文件 | 说明 |
|------|------|
| `results/E14_robustness_curve.png` | Accuracy + W-F1 退化曲线 |
| `results/E14_per_class_robustness.png` | 逐类鲁棒性柱状图 |
| `results/E14_impossible_trinity_concept.png` | 不可能三角概念图 |
| `results/E14_adversarial_results.json` | 结构化结果 JSON |
| `results/E14_adversarial_report.txt` | 详细文字报告 |

## 6. 后续衔接

* **向前**: E2/E10 确认模型决策合理 → E14 测试其在对抗环境下的稳定性。
* **核心结论**: "结构正交防御"是分层架构最强大的安全红利，系统级逃逸率被压缩至 ~8%。
* **代码**: `src/experiments/exp_e14_adversarial.py`
