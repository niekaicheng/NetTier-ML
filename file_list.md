# 项目文件清单

> 生成日期：2026-04-02  
> 项目：6800GNetTier-ML — 网络入侵检测系统（NIDS）两阶段机器学习研究

### 标记说明

| 标记 | 含义 |
|------|------|
| `[待删-H]` | 确定未使用，可安全删除（高优先级） |
| `[待删-M]` | 过时版本或未被论文引用，建议删除（中优先级） |
| `[待删-L]` | 一次性工具脚本，任务已完成（低优先级） |
| `[重复]` | 与其他目录下的文件内容完全相同 |
| `[草稿]` | 历史草稿，已被当前版本取代 |
| `[编译产物]` | LaTeX 编译自动生成，可随时重新生成 |
| *(无标记)* | 正常使用中 |

---

## 核心入口文件

| 文件 | 介绍 |
|------|------|
| `main.py` | 主程序入口，协调整个 ML 流水线执行 `[待删-M]` 仅有 TODO 骨架，无实际实现，未被任何脚本调用 |
| `run_pipeline.py` | 完整实验流水线运行脚本，按序执行各实验 |
| `train_stage1.py` | 第一阶段训练脚本：训练随机森林分类器 |
| `train_stage2.py` | 第二阶段训练脚本：训练 TransECA-Net Transformer 模型 |
| `train_stage2_optimized.py` | 优化版 Stage 2 训练，含超参数调优 |
| `run_dashboard.bat` | Windows 批处理文件，启动可视化 Dashboard |

---

## src/ 源代码

### 模型 (`src/models/`)

| 文件 | 介绍 |
|------|------|
| `src/models/stage1_rf.py` | 随机森林模型定义与训练逻辑（Stage 1） |
| `src/models/stage2_transeca.py` | TransECA-Net：基于 Transformer + ECA 注意力机制的深度学习模型（Stage 2） |
| `src/models/hybrid.py` | 两阶段混合模型集成接口 |
| `src/models/test_transeca.py` | TransECA-Net 单元测试 `[待删-L]` 临时测试脚本，不属于主工作流 |

### 数据处理 (`src/processing/`)

| 文件 | 介绍 |
|------|------|
| `src/processing/loader.py` | 标准数据加载器，读取 parquet 格式数据集 |
| `src/processing/loader_stratified.py` | 分层采样数据加载器，处理类别不平衡 |
| `src/processing/loader_time_aware.py` | 时间感知数据加载器，保持时序特性 `[待删-L]` 为不存在的 E16 实验设计，无任何调用方 |
| `src/processing/preprocess.py` | 特征工程与数据预处理（归一化、编码等） |
| `src/processing/pipeline_utils.py` | 流水线通用工具函数 |
| `src/processing/convert_csv_to_parquet.py` | 将原始 CSV 数据集转换为 Parquet 格式 `[待删-L]` 数据已完成转换，不再需要 |

### 实验 (`src/experiments/`)

| 文件 | 介绍 |
|------|------|
| `src/experiments/exp_e1_nested_cv.py` | E1：嵌套交叉验证，评估模型泛化性 |
| `src/experiments/exp_e2_shap.py` | E2：SHAP 特征重要性分析 |
| `src/experiments/exp_e4_bias_variance.py` | E4：偏差-方差权衡分析与学习曲线 |
| `src/experiments/exp_e6_bootstrap_ci.py` | E6：Bootstrap 置信区间估计 |
| `src/experiments/exp_e8_ablation.py` | E8：消融实验，验证各模块贡献 |
| `src/experiments/exp_e10_interpretability.py` | E10：模型可解释性分析（Integrated Gradients + Attention 热图） |
| `src/experiments/exp_e11_latency.py` | E11：推理延迟基准测试 |
| `src/experiments/exp_e12_visualization.py` | E12：t-SNE / UMAP 特征空间可视化 |
| `src/experiments/exp_e14_adversarial.py` | E14：对抗鲁棒性测试 |
| `src/experiments/exp_e15_generalization.py` | E15：跨数据集泛化测试（在 UNSW-NB15 上验证） |
| `src/experiments/exp_e17_threshold.py` | E17：决策阈值调优，优化精确率/召回率权衡 |

### 其他 (`src/`)

| 文件 | 介绍 |
|------|------|
| `src/presentation/dashboard.py` | Streamlit 可视化 Dashboard，展示实验结果 |
| `src/utils/training_logger.py` | 训练过程日志记录工具 |
| `src/utils/count_rows.py` | 统计数据集行数工具 `[待删-L]` 临时工具，未被任何模块引用 |
| `src/__init__.py` | 包初始化文件 |

---

## 工具脚本（根目录）

| 文件 | 介绍 |
|------|------|
| `analyze_results.py` | 汇总分析所有实验结果，生成报告 |
| `check_data_suitability.py` | 检查数据集是否适合研究目标 |
| `fix_figures.py` | 修复/重新生成论文所需图表 `[待删-M]` 未被任何脚本调用 |
| `convert_to_single_col.py` | 将双栏 LaTeX 转换为单栏格式（v1） `[待删-M]` v2 已替代，且两者均未被调用 |
| `convert_to_single_col_v2.py` | 单栏转换脚本 v2，功能更完善 |

---

## 数据文件 (`archive/` & `data/`)

| 文件 | 介绍 |
|------|------|
| `archive/Benign-Monday-no-metadata.parquet` | CIC-IDS2017：周一正常流量 |
| `archive/Bruteforce-Tuesday-no-metadata.parquet` | CIC-IDS2017：周二暴力破解攻击 |
| `archive/DoS-Wednesday-no-metadata.parquet` | CIC-IDS2017：周三 DoS 攻击 |
| `archive/Infiltration-Thursday-no-metadata.parquet` | CIC-IDS2017：周四渗透攻击 |
| `archive/WebAttacks-Thursday-no-metadata.parquet` | CIC-IDS2017：周四 Web 攻击 |
| `archive/Botnet-Friday-no-metadata.parquet` | CIC-IDS2017：周五僵尸网络 |
| `archive/DDoS-Friday-no-metadata.parquet` | CIC-IDS2017：周五 DDoS 攻击 |
| `archive/Portscan-Friday-no-metadata.parquet` | CIC-IDS2017：周五端口扫描 |
| `archive/UNSW_NB15_training-set.parquet` | UNSW-NB15 训练集（跨数据集泛化测试用） |
| `archive/UNSW_NB15_testing-set.parquet` | UNSW-NB15 测试集 |
| `data/stage2_train_data.parquet` | Stage 1 输出后用于训练 Stage 2 的中间数据 `[待删-M]` 已被 data/stage2/ 目录下的 train/val/test 分割版本取代 |

---

## 模型检查点 (`models_chk/`)

| 文件 | 介绍 |
|------|------|
| `models_chk/stage1_rf.joblib` | Stage 1 随机森林模型（标准版） |
| `models_chk/stage1_rf_stratified.joblib` | Stage 1 随机森林模型（分层采样版） |
| `models_chk/stage1_rf_best.pkl` | Stage 1 最优随机森林模型 |
| `models_chk/stage2_transeca.pth` | Stage 2 TransECA-Net 模型权重 |
| `models_chk/stage2_config.json` | Stage 2 模型配置（num_features/num_classes/超参数），训练后自动生成 |
| `models_chk/stage2_transeca_best.pth` | Stage 2 最优模型权重（early stopping 保存点） |
| `models_chk/e15_transeca_unsw.pth` | E15 跨数据集实验的 UNSW 微调模型 |
| `models_chk/preprocessor.joblib` | 标准数据预处理器（scaler/encoder） |
| `models_chk/preprocessor_stratified.joblib` | 分层采样版预处理器 |
| `models_chk/label_encoder_e1.joblib` | E1 实验的标签编码器 |

---

## 论文 / LaTeX (`acmart-primary/`)

| 文件 | 介绍 |
|------|------|
| `acmart-primary/experiment_report.tex` | 主 LaTeX 论文源文件（当前版本） |
| `acmart-primary/experiment_report copy.tex` | 论文草稿备份 `[草稿]` 已被 experiment_report.tex 取代 |
| `acmart-primary/experiment_report copy 2.tex` | 论文草稿备份 v2 `[草稿]` 已被 experiment_report.tex 取代 |
| `acmart-primary/experiment_report.pdf` | 编译后的论文 PDF（当前版本） |
| `acmart-primary/experiment_report_v24.pdf` | 论文第 24 版 PDF |
| `acmart-primary/experiment_report_final.pdf` | 论文 final 版 PDF |
| `acmart-primary/experiment_report_v24.aux` | LaTeX 辅助文件（v24） `[编译产物]` |
| `acmart-primary/experiment_report_v24.log` | LaTeX 编译日志（v24） `[编译产物]` |
| `acmart-primary/experiment_report_v24.out` | hyperref 信息文件（v24） `[编译产物]` |
| `acmart-primary/acmart.bib` | BibTeX 参考文献数据库 |
| `acmart-primary/acmart.cls` | ACM 论文模板样式文件 |
| `acmart-primary/acmart-tagged.cls` | ACM 论文模板（tagged 版） |
| `acmart-primary/Makefile` | LaTeX 编译自动化 |
| `acmart-primary/E_LogicFlow.png` | 实验逻辑流程图（中文版） |
| `acmart-primary/performance_radar_chart.png` | 性能雷达图 |
| `acmart-primary/E10_attention_heatmap.png` | E10 Attention 热图 |
| `acmart-primary/E10_eca_channel_weights.png` | E10 ECA 通道权重图 |
| `acmart-primary/E10_ig_global_importance.png` | E10 全局特征重要性图 |
| `acmart-primary/E10_ig_per_class.png` | E10 各类别特征重要性图 |
| `acmart-primary/E12_tsne_embedding.png` | E12 t-SNE 嵌入可视化 |
| `acmart-primary/E12_tsne_raw.png` | E12 t-SNE 原始空间可视化 |
| `acmart-primary/E12_umap_embedding.png` | E12 UMAP 嵌入可视化 |
| `acmart-primary/E12_umap_raw.png` | E12 UMAP 原始空间可视化 |
| `acmart-primary/E14_impossible_trinity_concept.png` | E14 不可能三角示意图 |
| `acmart-primary/E14_per_class_robustness.png` | E14 各类别鲁棒性图 |
| `acmart-primary/E14_robustness_curve.png` | E14 鲁棒性曲线图 |
| `acmart-primary/E15_cross_dataset_comparison.png` | E15 跨数据集对比图 |
| `acmart-primary/E15_unsw_confusion_matrix.png` | E15 UNSW 混淆矩阵 |
| `acmart-primary/E15_unsw_training_curves.png` | E15 UNSW 训练曲线 |
| `acmart-primary/E17_threshold_tuning_20260218_202157.png` | E17 阈值调优图 |
| `acmart-primary/E2_shap_summary.png` | E2 SHAP 汇总图 |
| `acmart-primary/E2_shap_vs_rf.png` | E2 SHAP 与 RF 重要性对比图 |
| `acmart-primary/E4_complexity_vs_perf.png` | E4 复杂度与性能关系图 |
| `acmart-primary/E4_dl_learning_curves.png` | E4 深度学习学习曲线 |
| `acmart-primary/E4_oob_vs_trees.png` | E4 OOB 误差与树数量关系图 |
| `acmart-primary/E4_rf_learning_curve.png` | E4 随机森林学习曲线 |
| `acmart-primary/E6_bootstrap_distributions.png` | E6 Bootstrap 分布图 |
| `acmart-primary/E8_ablation_bar.png` | E8 消融实验柱状图 |
| `acmart-primary/E8_ablation_comparison.png` | E8 消融实验对比图 |

---

## 实验结果 (`results/`)

### 数值结果 & 报告

| 文件 | 介绍 |
|------|------|
| `results/E1_nested_cv_results_20260218_104008.json` | E1 嵌套CV结果（第1次运行） |
| `results/E1_nested_cv_results_20260218_201215.json` | E1 嵌套CV结果（第2次运行） |
| `results/E2_shap_results.json` | E2 SHAP 分析数值结果 |
| `results/E2_shap_report.txt` | E2 SHAP 文字报告 |
| `results/E2_feature_importance.csv` | E2 特征重要性 CSV 数据 |
| `results/E4_bias_variance_results.json` | E4 偏差-方差分析结果 |
| `results/E4_bias_variance_report.txt` | E4 偏差-方差文字报告 |
| `results/E6_bootstrap_ci_results.json` | E6 Bootstrap 置信区间结果 |
| `results/E6_bootstrap_ci_report.txt` | E6 Bootstrap 文字报告 |
| `results/E8_ablation_results.json` | E8 消融实验结果 |
| `results/E8_ablation_report.txt` | E8 消融实验文字报告 |
| `results/E10_interpretability_results.json` | E10 可解释性分析结果 |
| `results/E10_interpretability_report.txt` | E10 可解释性文字报告 |
| `results/E11_latency_benchmark_20260218_154755.json` | E11 延迟基准（运行1） |
| `results/E11_latency_benchmark_20260218_165043.json` | E11 延迟基准（运行2） |
| `results/E11_latency_benchmark_20260218_174235.json` | E11 延迟基准（运行3） |
| `results/E11_latency_benchmark_20260218_175053.json` | E11 延迟基准（运行4） |
| `results/E11_latency_benchmark_20260218_175204.json` | E11 延迟基准（运行5） |
| `results/E11_latency_benchmark_20260218_175301.json` | E11 延迟基准（运行6） |
| `results/E11_latency_benchmark_20260218_175335.json` | E11 延迟基准（运行7） |
| `results/E11_latency_benchmark_20260218_181015.json` | E11 延迟基准（运行8） |
| `results/E11_latency_benchmark_20260218_182455.json` | E11 延迟基准（运行9） |
| `results/E11_latency_benchmark_20260218_201957.json` | E11 延迟基准（运行10） |
| `results/E12_visualization_results.json` | E12 可视化分析结果 |
| `results/E12_visualization_report.txt` | E12 可视化文字报告 |
| `results/E14_adversarial_results.json` | E14 对抗测试结果 |
| `results/E14_adversarial_report.txt` | E14 对抗测试文字报告 |
| `results/E15_generalization_results.json` | E15 泛化测试结果 |
| `results/E15_generalization_report.txt` | E15 泛化测试文字报告 |
| `results/E17_threshold_tuning_20260218_144007.json` | E17 阈值调优结果（第1次） |
| `results/E17_threshold_tuning_20260218_202157.json` | E17 阈值调优结果（第2次） |
| `results/stage1_report.txt` | Stage 1 总体评估报告 |
| `results/stage2_optimized_report.txt` | Stage 2 优化版评估报告 |
| `results/training_log.json` | 训练过程完整日志 |
| `results/E_LogicFlow.mmd` | 实验逻辑流程图源码（Mermaid，中文） |
| `results/E_LogicFlow_en.mmd` | 实验逻辑流程图源码（Mermaid，英文） |

### 可视化图表

| 文件 | 介绍 |
|------|------|
| `results/E2_shap_summary.png` | E2 SHAP 汇总图 |
| `results/E2_shap_bar.png` | E2 SHAP 柱状图 `[待删-M]` 已被 E2_shap_summary.png 取代，未在论文中使用 |
| `results/E2_shap_vs_rf.png` | E2 SHAP 与 RF 重要性对比图 |
| `results/E4_complexity_vs_perf.png` | E4 复杂度与性能关系图 |
| `results/E4_dl_learning_curves.png` | E4 深度学习学习曲线 |
| `results/E4_oob_vs_trees.png` | E4 OOB 误差与树数量关系图 |
| `results/E4_rf_learning_curve.png` | E4 随机森林学习曲线 |
| `results/E6_bootstrap_distributions.png` | E6 Bootstrap 分布图 |
| `results/E8_ablation_bar.png` | E8 消融实验柱状图 |
| `results/E8_ablation_comparison.png` | E8 消融实验对比图 |
| `results/E10_attention_heatmap.png` | E10 Attention 热图 |
| `results/E10_eca_channel_weights.png` | E10 ECA 通道权重图 |
| `results/E10_ig_global_importance.png` | E10 全局特征重要性图 |
| `results/E10_ig_per_class.png` | E10 各类别特征重要性图 |
| `results/E12_binary_view.png` | E12 二分类视角可视化 `[待删-M]` 未被论文采纳的中间产物 |
| `results/E12_tsne_embedding.png` | E12 t-SNE 嵌入空间图 |
| `results/E12_tsne_raw.png` | E12 t-SNE 原始空间图 |
| `results/E12_umap_embedding.png` | E12 UMAP 嵌入空间图 |
| `results/E12_umap_raw.png` | E12 UMAP 原始空间图 `[待删-M]` 论文只引用 embedding 版本 |
| `results/E14_impossible_trinity_concept.png` | E14 不可能三角示意图 |
| `results/E14_per_class_robustness.png` | E14 各类别鲁棒性图 |
| `results/E14_robustness_curve.png` | E14 鲁棒性曲线图 |
| `results/E15_cross_dataset_comparison.png` | E15 跨数据集对比图 |
| `results/E15_unsw_confusion_matrix.png` | E15 UNSW 混淆矩阵 |
| `results/E15_unsw_training_curves.png` | E15 UNSW 训练曲线 |
| `results/E17_threshold_tuning_20260218_144007.png` | E17 阈值调优图（第1次） `[待删-M]` 已被 202157 更新版本取代 |
| `results/E17_threshold_tuning_20260218_202157.png` | E17 阈值调优图（第2次，最新） |
| `results/E_LogicFlow.png` | 实验逻辑流程图（中文） |
| `results/E_LogicFlow_en.png` | 实验逻辑流程图（英文） |
| `results/performance_radar_chart.png` | 整体性能雷达图 |
| `results/stage1_cm.png` | Stage 1 混淆矩阵 `[待删-M]` 早期过时输出，未在论文中使用 |
| `results/stage1_feature_importance.png` | Stage 1 特征重要性图 `[待删-M]` 已被 E2 SHAP 系列图取代 |
| `results/stage1_feature_importance.csv` | Stage 1 特征重要性 CSV 数据 |
| `results/stage2_confusion_matrix.png` | Stage 2 混淆矩阵 `[待删-M]` 未被论文引用 |
| `results/stage2_training_history.png` | Stage 2 训练历史曲线 `[待删-M]` 未被论文引用 |

---

## 文档 (`notebooks/`)

| 文件 | 介绍 |
|------|------|
| `notebooks/project_proposal.md` | 项目提案，研究目标与背景 |
| `notebooks/experimental_design.md` | 实验设计文档（中文） |
| `notebooks/experimental_design_en.md` | 实验设计文档（英文） |
| `notebooks/experiment_report.md` | Markdown 格式实验报告 |
| `notebooks/experiment_report_en.md` | Markdown 格式实验报告（英文） |
| `notebooks/experiment_report_zh.md` | Markdown 格式实验报告（中文） `[草稿]` 与 experiment_report.md 内容高度重叠 |
| `notebooks/experiment_report.docx` | Word 格式实验报告 `[草稿]` md 的衍生版本，主版本为 .tex |
| `notebooks/experiment_report_en.docx` | Word 格式实验报告（英文） `[草稿]` md 的衍生版本 |
| `notebooks/experiment_report_zh.docx` | Word 格式实验报告（中文） `[草稿]` md 的衍生版本 |
| `notebooks/experiment_E1_nested_cv.md` | E1 嵌套CV实验详细记录 |
| `notebooks/verification_logic_chain.md` | 实验验证逻辑链文档（中文） |
| `notebooks/verification_logic_chain_en.md` | 实验验证逻辑链文档（英文） |
| `notebooks/verification_logic_chain.docx` | 验证逻辑链 Word 文档（中文） `[草稿]` md 的衍生版本 |
| `notebooks/verification_logic_chain_en.docx` | 验证逻辑链 Word 文档（英文） `[草稿]` md 的衍生版本 |
| `notebooks/literature_analysis.md` | 文献综述分析 |
| `notebooks/report.md` | 综合报告 |
| `notebooks/experimentalRank` | 实验优先级排序文档（中文） |
| `notebooks/experimentalRank_en` | 实验优先级排序文档（英文） |

---

## 参考文献 (`referencepapers/`)

| 文件 | 介绍 |
|------|------|
| `referencepapers/1Toward Generating...pdf` | CIC-IDS2017 数据集原始论文 |
| `referencepapers/2applsci-15-02977TransECA-Net...pdf` | TransECA-Net 模型原始论文（Stage 2 架构来源） |
| `referencepapers/10UNSW-NB15...pdf` | UNSW-NB15 数据集论文 |
| `referencepapers/8Machine_Learning_Techniques...pdf` | 网络异常检测机器学习综述 |
| `referencepapers/9Network Traffic Analysis Based on GNN.pdf` | 基于图神经网络的流量分析综述 |
| `referencepapers/A_detailed_analysis_of_the_KDD_CUP_99...pdf` | KDD Cup 99 数据集分析 |
| `referencepapers/Analysis of Machine Learning-Based Methods...pdf` | 基于 ML 的网络流量分析方法综述 |
| `referencepapers/Intelligent_network_traffic_analysis...pdf` | 面向网络安全的智能流量分析 |
| `referencepapers/Machine_Learning_in_Network_Anomaly_Detection_A_Survey.pdf` | 网络异常检测 ML 综述 |
| `referencepapers/paper/1Toward Generating...pdf` | CIC-IDS2017 数据集论文 `[重复]` 与 referencepapers/ 下同名文件完全相同 |
| `referencepapers/paper/2applsci-15-02977TransECA-Net...pdf` | TransECA-Net 论文 `[重复]` 与 referencepapers/ 下同名文件完全相同 |
| `referencepapers/paper/3A survey of deep learning-based network anomaly detection.txt` | 基于深度学习的网络异常检测综述（文本） |
| `referencepapers/paper/4S016740481930118X-main.pdf` | 参考文献 4 |
| `referencepapers/paper/6.pdf` | 参考文献 6 |
| `referencepapers/paper/7electronics-11-00556.pdf` | 参考文献 7（Electronics 期刊） |
| `referencepapers/paper/8Machine_Learning_Techniques...pdf` | 网络异常检测机器学习综述 `[重复]` 与 referencepapers/ 下同名文件完全相同 |
| `referencepapers/paper/10UNSW-NB15...pdf` | UNSW-NB15 数据集论文 `[重复]` 与 referencepapers/ 下同名文件完全相同 |
| `referencepapers/paper/no2.pdf` ~ `no25.pdf` | 编号参考文献（no2/3/4/6/12/13/22/23/24/25） |

---

## 配置与其他

| 文件 | 介绍 |
|------|------|
| `requirements.txt` | Python 依赖包列表 |
| `requirement_en.txt` | 英文版项目需求说明文档 |
| `.gitignore` | Git 忽略规则 |
| `.vscode/settings.json` | VS Code 编辑器配置 |
| `data_manifest.md` | 数据集清单，记录各数据文件来源与描述 |
| `file_guide_en.md` | 项目文件结构导航指南（英文） |
| `deep_learning_feasibility_analysis.md` | 深度学习可行性分析报告 |
| `STAGE2_DL_SUMMARY.md` | Stage 2 深度学习阶段总结 |
| `cite6800G.txt` | 引用格式参考文本 |
| `paperlist.txt` | 参考论文列表 |
| `paperref.txt` | 参考文献引用字符串 |
| `CIC_IDS2017_Presentation.pptx` | 项目演示文稿（PowerPoint） |
| `acmart-primary.zip` | LaTeX 论文目录的压缩备份 |

---

## 待删除汇总

| 优先级 | 文件数 | 说明 |
|--------|--------|------|
| `[待删-H]` | 0 | （已升级为 M/L，按实际情况分类） |
| `[编译产物]` | 11 | acmart-primary/ 下 .aux/.log/.out/.bbl/.blg |
| `[草稿]` | 9 | 论文草稿 copy.tex、notebooks/ 下 docx 及重复 md |
| `[重复]` | 4 | referencepapers/paper/ 下与上级目录完全相同的 PDF |
| `[待删-M]` | 11 | 过时图表、data/stage2_train_data.parquet、main.py 等 |
| `[待删-L]` | 4 | 一次性工具脚本（loader_time_aware、count_rows 等） |
| **合计** | **39** | |
