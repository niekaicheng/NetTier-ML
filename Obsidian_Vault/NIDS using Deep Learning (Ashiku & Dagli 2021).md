# NIDS using Deep Learning
**Authors**: Lirim Ashiku, Cihan Dagli
**Date/Venue**: Procedia Computer Science, Complex Adaptive Systems Conference, 2021

## 核心提要 (Core Summary)
这篇文章重点探讨了在 NIDS 中应用真正的**深度神经网络 (DNN)** 来取代传统的浅层分类器（Shallow Classifiers, 如随机森林、支持向量机）。
核心目标是赋予 NIDS 强大的自主学习能力，以从复杂的现代系统流量中提取高层次特征，乃至识别未知的**零日攻击 (Zero-day attacks)**。

## 核心技术点
1. **采用新一代数据集**
   作者尖锐地指出了旧版 DARPA 和 [[NSL-KDD 数据集]] 存在的大段冗余与脱离现代真实网络环境的问题（类似此前提到的语义鸿沟）。本研究全面转向了由澳大利亚大学构建的 **[[UNSW-NB15 数据集]]**，它包含了高达 40 多种现代流量特征和 9 个大类的现代攻击类型（包含 CVE 公开攻击字典）。
2. **深度学习架构重构 ([[CNN 网络入侵检测]])**
   放弃了传统前馈神经网络 (FNN) 中的点积乘法计算，转而使用**双重堆叠卷积系统 (Double Stacked CNN)**。
   配合最大池化 (Max Pooling) 实现特征降采样，结合 Dropout 机制来遏制过拟合 (Overfitting)。这种高维感知野（Receptive Field）对零日攻击的变种极为敏感。
3. **半动态超参数优化 (Semi-dynamic Hyperparameter Optimization)**
   针对此类超深层网络，为了防止在巨大的样本空间中迷失，探索出了一种能动态调节 Batch Size 与 Learning Rate 并在最佳收敛点（EarlyStopping）提早截断的调参手段。

## 结论
依托于 UNSW-NB15 数据集和极具针对性的 CNN 搭建范式，模型在不依赖严重数据重采样（Bootstrapping）的情况下，多分类最高获得了 95.6% 的惊人准确率。这宣告了深度学习在极其严苛的新一代多模态 NIDS 挑战中的适用性。
