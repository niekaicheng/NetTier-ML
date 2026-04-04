# 新文献与实验报告匹配度脱轨核查分析

**核查目标**：评估 `referencepapers/newpaper` 目录下的 5 篇新论文是否能够整合到 `experiment_report.tex` 作为支持性参考文献。
**实验报告基准架构**：双层检测系统 Hierarchical IDS。Stage 1 使用随机森林 (RF) 挡在前端做高速物理防御与初判，Stage 2 使用深度学习 (TransECA-Net: CNN+ECA+Transformer) 啃“硬骨头”。使用了 SHAP 与 IG (积分梯度) 提供严密的可解释性。

**核心结论警告：**
这 5 篇新文献中有 3 篇与当前的 LaTeX 实验报告数据和架构理念存在**直接致命的冲突**。如果试图将它们作为“正面支撑”引入，会导致整篇报告底层逻辑崩溃。唯一的安全引入方式是将其作为**“Related Work (相关工作) 中的被超越/批判对象”**。

详细对比结果如下：

---

### 1. 存在致命的【架构逆向冲突】
**对应论文**：`A Deep Learning Approach to Network Intrusion Detection (Shone et al. 2017)`
*   **新论文主张**：使用 NDAE（非对称深度自编码器 DL）在前端做重度特征降维，然后把提取好的维度交给 RF（随机森林 ML）分类。即：**DL 初筛 $\to$ ML 分类**。
*   **报告实际内容**：微秒级（6.12 $\mu s$）的高速 ML (RF) 在最前端（Stage 1）负责纳秒初筛防御，过滤掉大量报文后，交给后方的重型 DL (TransECA-Net) 精准攻击识别。即：**ML 初筛 $\to$ DL 分类**。
*   **评判**：**绝对冲突**。两者防御管线的承压逻辑是完全倒置的。

###  2. 存在致命的【自我数据打脸冲突】
**对应论文**：`NIDS using Deep Learning (Ashiku & Dagli 2021)`
*   **新论文主张**：推崇纯 CNN (卷积神经网络) 架构就可以完美通过现代网络流量（UNSW-NB15）分类测试。
*   **报告实际内容**：在 **Phase 2 (E8 消融实验)** 中，报告明确用实验图表数据抨击了了 CNN-Only 架构。文本批判记录：“CNN-Only 组显示出高达 50% 的剧烈验证集震荡...证明纯卷积机制根本无法胜任长时序特征”。
*   **评判**：**极其矛盾（自我逻辑崩塌）**。不能在报告正文中用消融实验超度纯 CNN，而在引言里又拿赞美 CNN 的文章作支撑背书。

###  3. 存在较重的【技术流派偏移】
**对应论文 1**：`Cyber Threat Detection based on Artificial Neural Networks using Event Profiles (Reddy & Bell 2025)`
*   **偏差分析**：Reddy 缓解黑盒和语义鸿沟的方法，是构建“Event Profiles”（事件概貌）和“贝叶斯生成图”。但您的系统用的是暴力的原包统计特征，并通过博弈论机制（SHAP）与积分算子（IG）硬撕出来的数学级特征级解释。是两个截然不同的派系路线。

**对应论文 2**：`An End-to-End Framework for Machine Learning-Based NIDS (Bertoli 2021, AB-TRAP)`
*   **偏差分析**：该文全篇主打底层的 **eBPF/XDP Kernel 层下发挂载能力**。而本份实验报告关注的是 PyTorch 环境下超参极值、对抗扰动 (PGD) 、假阳性截断等模型内在属性。

### 4. 罕见的【最高级哲学精神契合】
**对应论文**：`Outside the Closed World (Sommer & Paxson 2010)` （以及原库中包含的 `Ahmad 2020 综述`）
*   **评判**：**完美动机基石**。
*   **原因**：2010年，安全大牛深刻指出“在 NIDS 中误差极其昂贵（False Positives 杀人）”。这为您报告的整个 Phase 1 —— 拼命卡死 FPR 到 **0.007%**，极限压榨 $\tau=0.06$ 阈值筛选提供了坚不可摧的安全理论护城河！而冗长的解释性 SHAP 图表，正是您对安全人员所谓“理解鸿沟（Semantic Gap）”在物理层面上实施的最暴击的实弹工业回应。此文献应当占据 Background 开拓性的引用身位。

---

###  补救与改写建议
**结论**：不要倒向文献，要让文献为您服务。
目前使用基于 RF 不可微梯度特性 $\nabla h_1 \simeq 0$ 防住了 PGD 的设计，惊艳度极高！
对于 Shone (2017) 和 Ashiku (2021)，**请放入报告的【Related Work (相关工作)】章节作为靶子**：
1. *"有别于 Shone(2017) 将庞大的特征降维抛给深度自编码器的低效流水线，本文首创微秒级 ML 优先挡板机制..."*
2. *"针对 Ashiku(2021) 孤立利用 CNN 极易因为丢失长序列依赖导致模型大幅波动的现象，本系统 E8 实验用确凿对比论证了 Transformer 的不可或缺性..."*
