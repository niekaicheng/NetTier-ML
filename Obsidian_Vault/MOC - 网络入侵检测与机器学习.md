# MOC - 网络入侵检测与机器学习 (Machine Learning in NIDS)
**标签 (Tags)**: #moc #nids #machine-learning

这是关于机器学习与网络入侵检测技术结合的知识地图。

## 核心文献 (Core Papers)
- [[Outside the Closed World (Sommer & Paxson 2010)]] - 本领域的开创性批判文献。指出了 ML 在网络异常检测中的固有硬伤。
- [[A Deep Learning Approach to NIDS (Shone et al. 2017)]] - 提出了使用堆叠非对称深层自编码器(NDAE)结合随机森林的方案，大幅降低NIDS训练时间。
- [[A Survey of DL-based Network Anomaly Detection (Kwon et al. 2017)]] - 早期系统性综述。全面梳理了将受限玻尔兹曼机 (RBM)、深层隐马尔可夫及 RNN 等深度架构首次规模化引入网络异常检测的里程碑。
- [[A Survey of NIDS Data Sets (Ring et al. 2019)]] - 权威数据集族谱综述。为超过 34 种著名的 NIDS 数据集搭建了包含 15 项核心属性的评估雷达，并对“流 vs 包”给出了深度界定。
- [[An End-to-End Framework for ML-Based NIDS (Bertoli et al. 2021)]] - 提出了 AB-TRAP 框架，解决 NIDS 数据集老化和工程落地（Realization/Deployment）等全生命周期挑战。
- [[NIDS using Deep Learning (Ashiku & Dagli 2021)]] - 探讨如何使用 CNN 架构和半动态超参数优化处理 UNSW-NB15 数据集中的零日攻击。
- [[Cyber Threat Detection using Event Profiles (Reddy & Bell 2025)]] - 通过融合构建“事件概化体(Event Profiling)”和“贝叶斯推断”，打破深度神经网络的内部黑盒，缓解语义鸿沟。
- [[NIDS - A Systematic Study of ML and DL (Ahmad et al. 2020)]] - 体系化综述。宏观回顾了基于机器学习和深度学习算法在 NIDS 领域的演进、优劣比对及数据集发展史。
- [[Toward Generating CIC-IDS2017 Dataset (Sharafaldin et al. 2018)]] - 标志性数据集论文。痛陈十年间安全数据集的落伍，并提出了基于 11 项核心准则构建的全新 CIC-IDS2017 评估基准。
- [[TransECA-Net for Encrypted Traffic Classification (Liu et al. 2025)]] - 针对加密隧道的流分类应用。利用 ECANet 和 Transformer 多头注意力彻底取代旧有 LSTM，实现超长的并行序列捕获技术。

## 核心概念 (Key Concepts)
网络流量极其复杂，并且与通用机器学习所适用的环境不同：
- [[网络流量的自相似性]] - 推翻了传统泊松分布对流量到达率的假设，揭示了不同时间尺度上的突发性。
- [[异常检测的误差代价]] - 在安全运维中，False Positives 会迅速耗尽安全分析师的人力成本。
- [[封闭世界假设 (Closed World Assumption)]] - 安全领域的无效假设：即非白即黑（不正常的流量就一定是攻击）。
- [[语义鸿沟 (Semantic Gap)]] - 数学上识别出的“离群点”无法被直接翻译成操作员可以理解的“攻击行为”。
- [[非对称深度自编码器 (NDAE)]] - 仅包含编码阶段的无监督特征提取深度学习架构，极大减少了计算开销。
- [[NSL-KDD 数据集]] - 改进自 KDD Cup '99 的经典网络入侵检测评估基准数据集。
- [[AB-TRAP 框架]] - 从生成数据集、训练、直至落地的全链路 NIDS 模型生命周期框架。
- [[eBPF 与模型落地部署]] - 解决资源受限设备或内核级防御的高维方法，是 NIDS 模型走出实验室（Realization）的关键。
- [[UNSW-NB15 数据集]] - 包含现代网络流量和多类合成零日攻击的新一代基准数据集。
- [[CNN 网络入侵检测]] - 用卷积操作提取一维流量或图像化流量特征的深度防御技术。
- [[安全事件概化 (Event Profiling)]] - 摒弃底层零碎日志，将孤立的杂乱流量信息抽缩并格式化为富有上下文的独立“事件矢量”进行输入降噪。
- [[贝叶斯增强可解释性 (Bayesian Inference)]] - 引入基于有向无环图 (DAG) 的概率推断，让 AI 模型不仅会报警，还能交代原因链。
- [[深度学习与机器学习比较 (ML vs DL in NIDS)]] - 分类器选型指南：传统浅层学习依赖人工特征工程，而深度网络自动提取高维特征但有落地的算力包袱。
- [[CIC-IDS2017 数据集]] - 由加拿大网络安全研究所发布的现象级基准数据集，严格遵循 11 项网络真实环境评估准则。
- [[IDS 数据集11项评估标准 (11 Criteria)]] - 衡量一个安全数据集是否具有高学术价值和工业对抗意义的黄金底盘标准。
- [[加密流量分类 (Encrypted Traffic Classification)]] - NIDS 的深水区挑战，解决报文内容完全乱码化和动态端口导致的检测瘫痪问题。
- [[ECANet 注意力机制在 NIDS 的应用]] - 一种一维轻量级通道注意力机制，能自适应加权流量特征而几乎不增加算力开销。
- [[Transformer 对比 LSTM在流量检测中的优势]] - 多头注意力直接克服了 LSTM 长序列遗忘和单线程计算的桎梏，是大模型降维打击 NIDS 序列的方法。
- [[ISCX VPN-nonVPN 数据集]] - 用于衡量加密流量探测的专用 12 分类基准数据集。
- [[深层信念网络在流量检测的应用 (DBN - RBM)]] - 深度学习介入 NIDS 早期提取高维特征的排头兵，基于受限玻尔兹曼机堆叠的概率生成模型。
- [[基于包与基于流的数据对比 (Packet vs Flow-based Data)]] - 判断使用 DPI 深层探查还是元数据轻量监控的根本前设。
- [[NIDS 数据集 15 项评估属性 (15 Properties)]] - 用来将任意流量捕获库转化为体系化论文评估数据集的核心雷达评分项。

## 相关领域
- [[深度学习网络入侵检测系统 (DL-NIDS)]]
- [[基于规则的检测 (Rule-based Misuse Detection)]]
