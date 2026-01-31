# Training Data Manifest (学习数据记录报告)

**生成时间:** 2026-01-31
**来源依据:** `src/train_stage1.py`, `src/train_stage2.py`

本文档记录了 Hierarchical IDS 模型在训练过程中实际“学习”过的数据来源、采样策略及预处理配置。

---

## 1. 数据来源 (Data Source)

*   **原始数据集**: CIC-IDS2017
*   **存储路径**: `d:\6800G\archive\`
*   **文件格式**: Parquet

## 2. 数据集统计 (Data Statistics)

以下是 `archive` 目录中每个文件的实际数据量统计：

| 文件名 (File Name) | 数据行数 (Row Count) | 文件大小 (Size) | 主要内容 |
| :--- | :--- | :--- | :--- |
| Benign-Monday-no-metadata.parquet | 458,831 | 54.14 MB | 纯正常流量 |
| Botnet-Friday-no-metadata.parquet | 176,038 | 18.94 MB | ARES 僵尸网络 |
| Bruteforce-Tuesday-no-metadata.parquet | 389,714 | 44.00 MB | FTP/SSH 暴力破解 |
| DDoS-Friday-no-metadata.parquet | 221,264 | 24.13 MB | LOIT 等 DDoS 攻击 |
| DoS-Wednesday-no-metadata.parquet | 584,991 | 65.04 MB | Hulk, Slowloris, Heartbleed |
| Infiltration-Thursday-no-metadata.parquet | 207,630 | 22.07 MB | 内网渗透 |
| Portscan-Friday-no-metadata.parquet | 119,522 | 12.96 MB | 端口扫描 |
| WebAttacks-Thursday-no-metadata.parquet | 155,820 | 16.84 MB | XSS, SQL 注入, Brute Force |
| **总计 (Total)** | **2,313,810** | **258.12 MB** | **全量数据** |

## 3. 训练集构成 (Training Composition)

模型分为两个阶段训练，各自使用了不同的数据子集。

### Stage 1: 过滤器 (Filter Model)
*   **模型文件**: `models_chk/stage1_rf.joblib`
*   **学习目标**: 区分 "正常 (Benign)" 与 "异常 (Malicious)"。
*   **数据采样**:

| 类型 | 来源文件 (Subset) | 采样数量 | 备注 |
| :--- | :--- | :--- | :--- |
| **正常流量** | `Benign` | **50,000** | 随机采样，作为基准 |
| **攻击流量** | `PortScan` | ~15,000 | 常见扫描攻击 |
|  | `DoS` | ~15,000 | 拒绝服务攻击 (Hulk, Slowloris等) |
|  | `WebAttacks` | ~15,000 | XSS, SQL注入, BruteForce |
|  | `Botnet` | ~15,000 | 僵尸网络 |
| **总计** | | **~110,000** | 接近 1:1 的正负样本比例 (平衡训练) |

### Stage 2: 深度分类器 (Deep Classifier)
*   **模型文件**: `models_chk/stage2_transeca.pth`
*   **学习目标**: 区分具体攻击类型 (11分类)。
*   **数据采样**:

| 类别 (Class) | 对应攻击类型 | 采样策略 |
| :--- | :--- | :--- |
| **0** | Benign | 5,000 (作为背景噪音) |
| **1-10** | DoS, DDoS, PortScan, WebAttacks, Botnet | 各类约 5,000 |
| **总计** | | **~25,000** | 用于快速验证 TransECA-Net 结构 |

## 3. 预处理配置 (Learned Preprocessing)

这些参数“冻结”了模型对世界的认知范围，存储于 `models_chk/preprocessor.joblib`。

*   **特征清洗**: 移除了 `Flow ID`, `Source IP`, `Timestamp` 等 7 列非通用特征。
*   **特征缩放 (Scaling)**: 使用 `MinMaxScaler` 将 77 个统计特征压缩至 [0, 1] 区间。
    *   *模型记录了这 11 万条数据的最大值/最小值。*
*   **标签映射 (Encoding)**:
    *   `Benign` -> 0
    *   `Bot` -> 1
    *   `DDoS` -> 2
    *   ... (详见 `results/stage1_report.txt` 中的类别列表)

## 4. 验证集 (Validation)
在 `run_pipeline.py` 中，使用了独立的 **20,000** 条混合数据（包含上述所有类型）来测试最终的流水线性能，确保模型不仅仅是记住了训练数据。
