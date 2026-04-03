# 实验复现指南

> 项目：6800GNetTier-ML — 网络入侵检测系统（NIDS）两阶段机器学习研究  
> 最后更新：2026-04-02

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [执行顺序总览](#3-执行顺序总览)
4. [Stage 1 训练](#4-stage-1-训练)
5. [Stage 2 训练](#5-stage-2-训练)
6. [各实验脚本](#6-各实验脚本)
7. [预期结果](#7-预期结果)
8. [已知不可复现项](#8-已知不可复现项)

---

## 1. 环境准备

### 1.1 Python 版本

推荐 Python 3.10+。

### 1.2 安装依赖

**精确复现**（使用锁定版本，推荐）：

```bash
pip install -r requirements-lock.txt
```

**宽松安装**（可能有微小精度差异）：

```bash
pip install -r requirements.txt
```

### 1.3 锁定版本清单

| 包 | 版本 |
|----|------|
| torch | 2.11.0 |
| numpy | 2.3.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.7 |
| seaborn | 0.13.2 |
| joblib | 1.5.3 |
| pillow | 12.0.0 |
| sympy | 1.14.0 |

### 1.4 工作目录

所有脚本均需在项目根目录下执行：

```bash
cd /path/to/6800GNetTier-ML
```

---

## 2. 数据准备

### 2.1 原始数据（必须已存在）

确认 `archive/` 目录下有以下 10 个文件：

```
archive/
├── Benign-Monday-no-metadata.parquet
├── Bruteforce-Tuesday-no-metadata.parquet
├── DoS-Wednesday-no-metadata.parquet
├── Infiltration-Thursday-no-metadata.parquet
├── WebAttacks-Thursday-no-metadata.parquet
├── Botnet-Friday-no-metadata.parquet
├── DDoS-Friday-no-metadata.parquet
├── Portscan-Friday-no-metadata.parquet
├── UNSW_NB15_training-set.parquet   ← E15 实验专用
└── UNSW_NB15_testing-set.parquet    ← E15 实验专用
```

### 2.2 中间数据（由 Stage 1 自动生成）

`data/stage2/` 目录下的 train/val/test 分割文件由 `train_stage1.py` 自动生成，**无需手动准备**。

---

## 3. 执行顺序总览

各脚本之间存在严格依赖，必须按以下顺序执行：

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: train_stage1.py                                      │
│   输出 → models_chk/stage1_rf_stratified.joblib             │
│        → models_chk/preprocessor_stratified.joblib          │
│        → data/stage2/train.parquet                          │
│        → data/stage2/val.parquet                            │
│        → data/stage2/test.parquet                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ STEP 2: train_stage2_optimized.py                            │
│   输出 → models_chk/stage2_transeca.pth                     │
│        → models_chk/stage2_transeca_best.pth                │
│        → results/stage2_optimized_report.txt                │
└──────┬─────────────────────────────────────────────┬────────┘
       │                                             │
       │  STEP 3a: exp_e1_nested_cv.py (独立)        │  STEP 3b: 依赖 Stage 1+2 的实验（可并行）
       │    输出 → models_chk/stage1_rf_best.pkl     │    exp_e2_shap.py
       │         → models_chk/label_encoder_e1.joblib│    exp_e4_bias_variance.py
       │                   │                         │    exp_e6_bootstrap_ci.py
       │                   ▼                         │    exp_e8_ablation.py
       │           exp_e17_threshold.py              │    exp_e10_interpretability.py
       │           exp_e11_latency.py                │    exp_e12_visualization.py
       │                                             │    exp_e14_adversarial.py
       │                                             │    exp_e15_generalization.py
       └─────────────────────────────────────────────┘
```

---

## 4. Stage 1 训练

### 命令

```bash
python train_stage1.py
```

### 说明

- 从 `archive/` 加载 CIC-IDS2017 全部 8 个 parquet 文件（排除 UNSW）
- 使用分层混合采样（80% 训练 / 10% 验证 / 10% 测试）
- 训练二分类随机森林（Benign vs Attack）
- 对验证集和测试集做 CV Mining，将 Stage 1 误分类的难例输出给 Stage 2

### 关键超参数

| 参数 | 值 |
|------|----|
| n_estimators | 50 |
| max_depth | 20 |
| class_weight | balanced |
| random_state | 42 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `models_chk/stage1_rf_stratified.joblib` | 训练好的 RF 模型 |
| `models_chk/preprocessor_stratified.joblib` | 特征预处理器 |
| `data/stage2/train.parquet` | Stage 2 训练集（Stage 1 难例） |
| `data/stage2/val.parquet` | Stage 2 验证集 |
| `data/stage2/test.parquet` | Stage 2 测试集 |
| `results/stage1_report.txt` | Stage 1 评估报告 |

---

## 5. Stage 2 训练

### 命令

```bash
python train_stage2_optimized.py
```

### 说明

- 加载 `data/stage2/{train,val,test}.parquet`（Stage 1 输出的难例数据）
- 训练 TransECA-Net（Transformer + ECA 注意力）多分类模型
- 使用 Early Stopping（patience=5）和 Cosine Annealing 学习率调度
- **随机种子已固定（SEED=42）**，结果可精确复现

### 关键超参数

| 参数 | 值 |
|------|----|
| d_model | 128 |
| nhead | 8 |
| num_layers | 3 |
| batch_size | 512 |
| learning_rate | 0.001 |
| epochs (最大) | 30 |
| optimizer | AdamW (weight_decay=1e-4) |
| scheduler | CosineAnnealingWarmRestarts (T_0=10) |
| random_seed | 42 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `models_chk/stage2_transeca.pth` | 最终模型权重 |
| `models_chk/stage2_transeca_best.pth` | Early Stopping 保存的最优权重 |
| `results/stage2_optimized_report.txt` | Stage 2 评估报告 |
| `results/training_log.json` | 逐 epoch 训练日志 |

---

## 6. 各实验脚本

### 6.1 E1 — 嵌套交叉验证

```bash
python src/experiments/exp_e1_nested_cv.py
```

**依赖**：`archive/*.parquet`（原始数据，独立于 Stage 1 产出）  
**说明**：重新训练 RF 并做 5-fold 外层 × 3-fold 内层嵌套 CV，使用 10% 数据采样  
**输出**：`models_chk/stage1_rf_best.pkl`、`models_chk/label_encoder_e1.joblib`、`results/E1_nested_cv_results_*.json`

---

### 6.2 E2 — SHAP 特征重要性

```bash
python src/experiments/exp_e2_shap.py
```

**依赖**：`models_chk/stage1_rf_stratified.joblib`、`models_chk/preprocessor_stratified.joblib`（需先完成 Step 1）  
**输出**：`results/E2_shap_*.png`、`results/E2_shap_results.json`、`results/E2_feature_importance.csv`

---

### 6.3 E4 — 偏差方差分析

```bash
python src/experiments/exp_e4_bias_variance.py
```

**依赖**：`models_chk/preprocessor_stratified.joblib`、`results/training_log.json`（需先完成 Step 1 & 2）  
**输出**：`results/E4_*.png`、`results/E4_bias_variance_results.json`

---

### 6.4 E6 — Bootstrap 置信区间

```bash
python src/experiments/exp_e6_bootstrap_ci.py
```

**依赖**：`models_chk/stage1_rf_stratified.joblib`、`models_chk/preprocessor_stratified.joblib`、`data/stage2/test.parquet`（需先完成 Step 1 & 2）  
**输出**：`results/E6_bootstrap_*.png`、`results/E6_bootstrap_ci_results.json`

---

### 6.5 E8 — 消融实验

```bash
python src/experiments/exp_e8_ablation.py
```

**依赖**：`data/stage2/{train,val,test}.parquet`（需先完成 Step 1）  
**说明**：从头训练多个消融版本，对比各模块贡献  
**输出**：`results/E8_ablation_*.png`、`results/E8_ablation_results.json`

---

### 6.6 E10 — 可解释性分析

```bash
python src/experiments/exp_e10_interpretability.py
```

**依赖**：`models_chk/stage2_transeca.pth`、`models_chk/preprocessor_stratified.joblib`、`data/stage2/test.parquet`（需先完成 Step 1 & 2）  
**输出**：`results/E10_attention_heatmap.png`、`results/E10_ig_*.png`、`results/E10_interpretability_results.json`

---

### 6.7 E11 — 延迟基准测试

```bash
python src/experiments/exp_e11_latency.py
```

**依赖**：`models_chk/stage1_rf_best.pkl`（需先完成 E1）  
**⚠️ 注意**：结果受硬件（CPU/GPU 型号、频率、系统负载）影响，**无法精确复现**，仅作量级参考  
**输出**：`results/E11_latency_benchmark_<timestamp>.json`

---

### 6.8 E12 — 特征空间可视化

```bash
python src/experiments/exp_e12_visualization.py
```

**依赖**：`models_chk/stage2_transeca.pth`、`models_chk/preprocessor_stratified.joblib`、`data/stage2/test.parquet`（需先完成 Step 1 & 2）  
**输出**：`results/E12_tsne_*.png`、`results/E12_umap_*.png`

---

### 6.9 E14 — 对抗鲁棒性测试

```bash
python src/experiments/exp_e14_adversarial.py
```

**依赖**：`models_chk/stage2_transeca.pth`、`models_chk/preprocessor_stratified.joblib`、`data/stage2/test.parquet`（需先完成 Step 1 & 2）  
**输出**：`results/E14_robustness_curve.png`、`results/E14_adversarial_results.json`

---

### 6.10 E15 — 跨数据集泛化测试

```bash
python src/experiments/exp_e15_generalization.py
```

**依赖**：`archive/UNSW_NB15_{training,testing}-set.parquet`（独立数据，无需 Stage 1/2）  
**说明**：在 UNSW-NB15 数据集上从头微调，测试跨数据集泛化能力  
**输出**：`models_chk/e15_transeca_unsw.pth`、`results/E15_*.png`、`results/E15_generalization_results.json`

---

### 6.11 E17 — 决策阈值调优

```bash
python src/experiments/exp_e17_threshold.py
```

**依赖**：`models_chk/stage1_rf_best.pkl`、`models_chk/label_encoder_e1.joblib`（需先完成 E1）  
**输出**：`results/E17_threshold_tuning_<timestamp>.json`、`results/E17_threshold_tuning_<timestamp>.png`

---

## 7. 预期结果

与 `results/` 目录下当前存档对比：

| 实验 | 预期误差 | 说明 |
|------|---------|------|
| Stage 1 | ±0% | 随机种子完全固定 |
| E1 | ±0% | 随机种子完全固定 |
| E2 | ±0% | 基于 Stage 1 RF，确定性计算 |
| E17 | ±0% | 基于 Stage 1 RF，确定性计算 |
| Stage 2 | ±0%* | 随机种子已修复，*需相同硬件环境 |
| E4、E6、E8、E10、E12、E14、E15 | ±0%* | 依赖 Stage 2，同上 |
| E11 | 不可比 | 硬件依赖，每次运行结果不同 |

> *注：不同操作系统、CUDA 版本下浮点运算顺序可能导致 ±0.01% 以内的差异，属正常范围。

---

## 8. 已知不可复现项

### E11 延迟测试

延迟数值受以下因素影响，**无法跨机器复现**：

- CPU / GPU 型号与主频
- 系统内存带宽
- 运行时系统负载
- PyTorch JIT 编译缓存状态
- 操作系统调度策略

建议：将 E11 结果仅用于同一机器上不同模型之间的**相对比较**，不用于跨环境的绝对数值对比。

---

## 快速复现（完整流程）

```bash
# 1. 进入项目目录
cd /path/to/6800GNetTier-ML

# 2. 安装精确依赖
pip install -r requirements-lock.txt

# 3. 训练 Stage 1（约 5~10 分钟）
python train_stage1.py

# 4. 训练 Stage 2（约 15~30 分钟）
python train_stage2_optimized.py

# 5. 运行独立实验（可并行）
python src/experiments/exp_e1_nested_cv.py

# 6. 运行依赖 Stage 1+2 的实验（可并行）
python src/experiments/exp_e2_shap.py
python src/experiments/exp_e4_bias_variance.py
python src/experiments/exp_e6_bootstrap_ci.py
python src/experiments/exp_e8_ablation.py
python src/experiments/exp_e10_interpretability.py
python src/experiments/exp_e12_visualization.py
python src/experiments/exp_e14_adversarial.py
python src/experiments/exp_e15_generalization.py

# 7. 运行依赖 E1 的实验
python src/experiments/exp_e17_threshold.py

# 8. 延迟测试（结果仅供参考）
python src/experiments/exp_e11_latency.py
```
