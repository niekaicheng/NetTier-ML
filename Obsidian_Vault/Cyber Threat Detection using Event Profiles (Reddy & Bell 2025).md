# Cyber Threat Detection using Event Profiles
**Authors**: Varimadugu Lokeshwar Reddy, Briso Becky Bell
**Date/Venue**: 9th International Conference on ICECA, 2025

## 核心提要 (Core Summary)
这篇文章聚焦应对在机器学习 NIDS 中的“算力负担”以及由来已久的“黑盒效应（不被安全操作员信任、难以追溯，即 [[语义鸿沟 (Semantic Gap)]]）”。
本研究证明了：如果我们不要盲目地把海量的散装“原始日志 (Raw Logs)”当作底层输入投喂给深沉笨重的架构（如 [[CNN 网络入侵检测]]），而是把它们结构化成 **[[安全事件概化 (Event Profiling)]]** 再喂给基础的 **人工神经网络 (ANN)**，同时外包一层 **[[贝叶斯增强可解释性 (Bayesian Inference)]]**，能够取得又快又好又透明的效果。

## 解决的问题
1. **多余的维度灾难对抗**
   原始数据（如 SIEM / IDS Log）充斥了无数高频冗长却没有意义的噪点（Noise）。
2. **缓解“黑盒 (Black-box)”特质**
   单纯的网络（不论多深）在检测出异常时，只能输出一个 Confidence Score（置信度）。这显然不能说明 "为什么认为这是攻击"，在真刀真枪的安防部门（Security Operations Center, SOC）是难以落地的。借助有向无环图 (DAG)，系统在给出判定时，能够向人类操作员阐明概率依存关系。

## 核心结论
- **算法轻量但高效**：在这个特定的架构下（Event Profiling + Bayesian Reasoning），经典浅显的神经网络 (ANN) 在准确率居然达到了惊人的 **97.4%**，反杀了更笨重且占用巨大算力和时间成本的深度 CNN 架构 (95.6%)，以及随机森林RF (94.2%)。
- **完美的物联网/边缘适用性**：这再次呼应了 [[AB-TRAP 框架]] 中探讨的 RealizAtion（落地问题）—— 高准确率并伴有“高计算效率 (Efficiency in Computation)”的框架，才能完美下放部署至 IoT 与前端。
