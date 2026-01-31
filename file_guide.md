# Project File Guide (项目文件详细说明)

本文档对当前项目目录下的文件进行分类和标注，帮助您快速识别核心文件与临时文件。

## 1. 核心代码 (Core System) —— **[保留 / 勿删]**
这些是项目运行的基础，缺失会导致系统无法工作。

*   **`src/`**: 源代码目录。
    *   `processing/`: 数据加载与预处理逻辑。
    *   `models/`: 模型定义（RF, TransECA-Net, Hybrid）。
    *   `presentation/`: 仪表盘代码。
*   **`main.py`**: 项目入口文件。
*   **`train_stage1.py`**: 第一阶段（过滤器）训练脚本。
*   **`train_stage2.py`**: 第二阶段（分类器）训练脚本。
*   **`run_pipeline.py`**: 端到端测试与验证脚本。
*   **`analyze_results.py`**: 结果分析与绘图脚本。
*   **`run_dashboard.bat`**: 启动可视化仪表盘的快捷脚本。

## 2. 模型与数据 (Models & Data) —— **[保留 / 核心资产]**
系统的“大脑”和“记忆”。

*   **`models_chk/`**: 存放训练好的模型权重 (`.joblib`, `.pth`) 和预处理标准 (`preprocessor.joblib`)。
*   **`archive/`**: 原始数据集 (Parquet文件)。

## 3. 结果与报告 (Results & Reports) —— **[保留 / 作业提交]**
用于展示工作成果的文件。

*   **`Project_Proposal.docx`**: **[重要]** 您的项目提案（Word版），用于提交。
*   **`data_manifest.md`**: **[重要]** 详细的训练数据统计报告。
*   **`results/`**: 存放生成的图表（混淆矩阵、特征重要性）和文本报告。
    *   `stage1_cm.png`, `stage1_report.txt` 等。

## 4. 辅助文档 (Auxiliary Docs) —— **[参考]**
*   **`notebooks/`**: 存放您之前创建的一些文本笔记（如 `firstrunreport.txt` 等），内容已整合到上述正式报告中，可作为备份保留。
*   **`requirement.txt`**: 原始需求文档。

## 5. 临时文件 (Temporary) —— **[可删除 / 已归档]**
*   **`temp/`**: 刚才整理进去的中间临时文件（如 `read_pdf.py`, `debug_etl.py` 等）。如果您不再需要调试，可以整个文件夹删除。

---
**建议**：
提交作业或备份时，只需打包 **Group 1, 2, 3** 中的文件即可（`archive/` 数据集通常太大不需提交，只提交代码报告和 `models_chk/` 即可）。
