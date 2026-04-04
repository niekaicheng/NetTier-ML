# Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization
**Authors**: Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani
**Date/Venue**: 4th International Conference on Information Systems Security and Privacy (ICISSP), 2018

## 核心提要 (Core Summary)
这是一篇在 IDS/NIDS 数据集构建领域极具**“破局意义”的奠基之作**。
此论文对从 1998 年（DARPA98）至 2016 年间市面上存在的 11 大公开数据集（KDD'99, DEFCON, CAIDA, Kyoto, ISCX2012 等）进行了系统性清算，直指它们由于“攻击载荷被完全匿名化去重（Anonymized）”、“缺乏现代协议（如缺失 HTTPS）”、“攻击类型单一”等问题，导致当今许多基于异常（Anomaly-based）的论文得出的结论如同空中楼阁。

为彻底解决此痛点，加拿大网络安全研究所（CIC）根据著名的 [[IDS 数据集11项评估标准 (11 Criteria)]] 全新部署了一套极具对抗深度的攻防拓扑，并以此诞生了现象级的数据演武场——**[[CIC-IDS2017 数据集]]**。

## 解决的问题
1. **打破脱离现实的实验室虚假流量**
   部署了 B-Profile Agent，用于模拟真实人类交互时产生的良性背景流量（涵盖 HTTP, HTTPS, FTP, SSH 及邮件流）。
2. **囊括最前沿全维度攻击池**
   从 Brute Force 到 DoS 变种 (如心脏出血 Heartbleed、Slowloris)，直至深度渗透 (Infiltration)、PortScan 以及 Botnet 控制，全量囊括。
3. **提供统一的机器可读特征基准**
   通过开源自身研发的提取提取器 CICFlowMeter，将所有抓包（pcap）统一清洗出了 80 余项关键指标集（流量间隔、均值、流持续时间等等），直接喂食给当红算法验证性能。

## 核心结论
- 在使用了 RandomForestRegressor 挑选最敏锐的攻击分类特征，分别比对了 7 种流行浅层分类器后，**随机森林 (Random Forest, RF)** 凭借以仅仅 74.39 秒的最短预测时长达成了 98% 的 Precision，证明了**“只要前置提取特征足够纯净，浅层树依然称王”**的数据界新法则。
