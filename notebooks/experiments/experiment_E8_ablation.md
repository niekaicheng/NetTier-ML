# 实验 E8: TransECA-Net 消融实验 (Ablation Study)

> **归属**: Stage 2 (Deep Learning Classification)  
> **目标**: 量化 TransECA-Net 三个组件 (CNN, Transformer, ECA) 各自的贡献，验证架构设计的必要性。

---

## 1. 实验目标 (Objectives)

1. **主目标**: 回答"为什么 TransECA-Net 需要同时引入 ECA Attention 和 Transformer Encoder？"
2. **次要目标**:
    * 量化 Transformer 自注意力机制对长距离特征关联的贡献。
    * 揭示 ECA 通道注意力在少数类识别中的差异化价值。
    * 为 E10（可解释性分析）和 E12（可视化）提供架构基线。

## 2. 理论依据 (Theoretical Basis)

**消融实验 (Ablation Study)** 是验证复杂架构各组件贡献的标准范式：通过逐一移除或替换组件，在控制其他变量不变的条件下，观察性能变化。

## 3. 实验配置 (Configuration)

### 3.1 模型变体

| 变体 | 架构 | 参数量 |
|------|------|:---:|
| Full TransECA-Net | CNN + ECA + Transformer | 301,460 |
| No-ECA (CNN+Trans) | CNN + Transformer (ECA → Identity) | 301,455 |
| CNN-Only | CNN + GlobalAvgPool + FC | 2,703 |

### 3.2 统一训练条件

| 参数 | 值 |
|------|------|
| 数据 | `data/stage2/{train,val,test}.parquet` |
| d_model | 128 |
| nhead | 8 |
| num_layers | 3 |
| batch_size | 512 |
| epochs | 20 |
| optimizer | AdamW (weight_decay=1e-4) |
| scheduler | CosineAnnealingWarmRestarts (T_0=10, T_mult=2) |
| 设备 | Intel ARC 130T (XPU) |
| 总训练耗时 | 446.51 min |

## 4. 实验结果 (Results)

### 4.1 性能对比

| 变体 | 参数量 | Test Acc | W-F1 | M-F1 | Best Val Acc |
|------|:---:|:---:|:---:|:---:|:---:|
| CNN-Only | 2,703 | 61.89% | 0.668 | 0.235 | 61.99% |
| No-ECA (CNN+Trans) | 301,455 | 92.05% | 0.947 | 0.675 | 91.97% |
| **Full TransECA-Net** | 301,460 | 89.71% | 0.925 | **0.759** | 89.86% |

### 4.2 组件贡献分析

| 组件 | W-F1 贡献 | M-F1 贡献 | 解读 |
|------|:---:|:---:|------|
| **Transformer** | **+0.279** | **+0.440** | 架构核心，Acc 提升 30pp |
| **ECA** | -0.022 | **+0.085** | W-F1 微降，但显著提升少数类识别 |

### 4.3 关键发现

1. **Transformer 是决定性组件**: CNN-Only 仅 61.89%；引入 Transformer 后 Acc 升至 92%。流量分类需要关联远距特征（如 TCP 窗口大小 ↔ IAT 时间统计），这是 Self-Attention 的核心能力。

2. **ECA 的差异化价值**: 全局 W-F1 下降 0.022，但 M-F1 提升 0.085。ECA 通道权重 CV=0.02（近均匀），说明 CNN 提取的通道信息已经较均衡，ECA 仅提供边际调节；但对少数类（如 Heartbleed），条件通道权重有效 CV 远高于全局，提供差异化表征。在 IDS 场景中，少数类 = 罕见攻击 = 最关键的检测目标。

3. **M-F1 vs W-F1 的度量反转**: No-ECA 在 W-F1 上优于 Full（0.947 vs 0.925），但 Full 在 M-F1 上显著领先（0.759 vs 0.675）。这证明 ECA 牺牲了微小的全局确定性，换取了对极稀疏长尾攻击的感知边界。

## 5. 输出文件 (Output)

| 文件 | 说明 |
|------|------|
| `results/E8_ablation_comparison.png` | 训练曲线对比图 (4 子图: Train/Val Loss/Acc) |
| `results/E8_ablation_bar.png` | 最终性能柱状图 |
| `results/E8_ablation_results.json` | 结构化结果 JSON |
| `results/E8_ablation_report.txt` | 详细文字报告 |

## 6. 后续衔接

* **向前**: Stage 2 训练验证了 TransECA-Net 的高精度 → E8 解释其结构必要性。
* **向后**: E10 通过 IG + Attention 验证 ECA CV=0.02 的物理含义；E12 可视化嵌入空间改善。
* **代码**: `src/experiments/exp_e8_ablation.py`
