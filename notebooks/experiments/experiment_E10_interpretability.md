# 实验 E10: Stage 2 可解释性分析 (Integrated Gradients + Attention)

> **归属**: Stage 2 (Deep Learning Classification)  
> **目标**: 可视化 TransECA-Net 的决策依据，与 E2 (SHAP) 形成跨模型三角验证。

---

## 1. 实验目标 (Objectives)

1. **主目标**: 解构 Stage 2 深度学习模型的决策逻辑，验证其关注的特征是否与领域知识一致。
2. **次要目标**:
    * 验证 IG (Integrated Gradients) 与 SHAP 在核心特征优先级上的部分收敛。
    * 分析 Attention Rollout 的注意力权重分布。
    * 验证 ECA 通道权重的均匀性 (与 E8 消融结论交叉验证)。
    * 分解 15 个攻击类别的逐类归因模式差异。

## 2. 理论依据 (Theoretical Basis)

**三种归因方法的理论保证**:
* **Integrated Gradients (IG)**: 满足完备性公理 $\sum IG_i = F(x) - F(x')$，梯度沿输入空间路径积分。
* **Attention Rollout**: 通过层间注意力权重递归乘积，近似全局特征关注分布。
* **ECA Channel Attention**: 直接提取 ECA 模块的通道权重，反映通道选择性。

## 3. 实验配置 (Configuration)

| 参数 | 值 |
|------|------|
| 模型 | `stage2_transeca.pth` (d=128, h=8, L=3, 301,460 params) |
| IG 样本数 | **405** (分层抽样) |
| IG 积分步数 | **50** |
| IG 计算耗时 | 19.73s |
| Attention 计算耗时 | 0.23s |
| ECA 计算耗时 | 0.17s |
| 总耗时 | 27.77s |

## 4. 实验结果 (Results)

### 4.1 Integrated Gradients Top-10

| 排名 | 特征 | IG (mean\|abs\|) |
|:---:|------|:---:|
| 1 | **Init Fwd Win Bytes** | **2.107** |
| 2 | Flow Packets/s | 1.708 |
| 3 | Bwd Header Length | 1.591 |
| 4 | Fwd Seg Size Min | 1.554 |
| 5 | Fwd Header Length | 1.288 |
| 6 | Flow Bytes/s | 1.288 |
| 7 | PSH Flag Count | 1.236 |
| 8 | Flow IAT Max | 1.099 |
| 9 | Fwd IAT Total | 1.035 |
| 10 | Bwd IAT Total | 0.862 |

### 4.2 Attention Rollout Top-5

| 排名 | 特征 |
|:---:|------|
| 1 | **Fwd IAT Mean** |
| 2 | Fwd Pkt Len Max |
| 3 | Flow Bytes/s |
| 4 | Flow IAT Min |
| 5 | Init Fwd Win Bytes |

### 4.3 ECA 通道权重统计

| 指标 | 值 |
|------|------|
| 均值 | 0.449 |
| 标准差 | 0.009 |
| **CV** | **0.020** |
| 最大通道 | Ch127 (0.488) |

### 4.4 跨模型部分收敛验证

两个理论独立的归因方法 (SHAP: Shapley 公理; IG: 完备性公理)，应用于架构迥异的模型 (RF vs TransECA-Net)，在核心物理优先级上展现**部分收敛**:

| 收敛维度 | SHAP (E2) | IG (E10) | Attention (E10) |
|------|------|------|------|
| TCP 初始窗口 | #2 (Bwd) | **#1 (Fwd)** | #5 |
| IAT 时序系列 | Top-5/7/12/13 | Top-8/9/10 | #1/#4 |
| 领域知识对齐 | ✅ [Sharafaldin'18] | ✅ | ✅ |

**关键分歧**: `Bwd Packet Length Std` 在 SHAP 中绝对 #1，但在 IG 中降至中游 (0.804)。假设：Transformer 的序列注意力将单点密度分散到更广的时序维度。

**IG 独占发现**: `PSH Flag Count` (IG=1.236) 是 IG 独有的高排名特征，RF 完全未能捕获此经典注入攻击序列信号。

### 4.5 逐类 IG 归因极值

| 攻击类别 | 极值特征 | IG 值 | 物理含义 |
|------|------|:---:|------|
| FTP-Patator | Init Fwd Win Bytes | **+6.024** | 暴力破解的 TCP 窗口异常 |
| Bot | Fwd IAT Total | **-4.702** | 僵尸网络的时序逆向行为 |
| Heartbleed | Bwd Header Length | **+5.250** | 内存越界读取的协议层特征 |
| Benign | (全局) | 平坦 | 无极端激活峰值 |

## 5. 输出文件 (Output)

| 文件 | 说明 |
|------|------|
| `results/E10_ig_global_importance.png` | 全局 IG 特征重要性柱状图 |
| `results/E10_ig_per_class.png` | 15类 × Top-15特征 热力图 |
| `results/E10_attention_heatmap.png` | Attention Rollout 热力图 |
| `results/E10_eca_channel_weights.png` | ECA 128 通道权重分布图 |
| `results/E10_interpretability_results.json` | 结构化结果 JSON |
| `results/E10_interpretability_report.txt` | 详细文字报告 |

## 6. 后续衔接

* **向前**: E2 建立 Stage 1 特征归因基线 → E10 完成跨模型三角验证。
* **向后**: ECA CV=0.020 直接验证 E8 消融结论（ECA 边际贡献有限）。
* **关联**: E14 的 Heartbleed 在对抗攻击下保持 1.0 准确率，与 E10 的 `Bwd Header Length` IG=5.250 超集中激活相互印证。
* **代码**: `src/experiments/exp_e10_interpretability.py`
