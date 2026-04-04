# UNSW-NB15 数据集

## 定义与来源
**UNSW-NB15** 是由澳大利亚新南威尔士大学 (University of New South Wales) 教授 Nour Moustafa 等人建立的全面网络入侵检测系统演练数据集。

## 为什么抛弃 KDD？
过去的十几年间，业界普遍使用 KDD Cup '99 及其翻新版 [[NSL-KDD 数据集]]。但网络世界变天了：
- **旧版脱节**：1999 年缺乏现今庞大的 IoT 流量以及复杂的高级持续性威胁 (APT)。
- **冗余率高**：旧数据集训练出的模型经常“背题”，难以防御新变种。

## UNSW-NB15 的现代特征
- 包含极大量的**正常真实当代流量**，混合了人工生成的**现代化攻击面流量**。
- 分解为 **9 大攻击血统 (Families)**：例如 Fuzzers、Analysis、Backdoors、DoS、Exploits、Generic、Reconnaissance、Shellcode 和 Worms。
- **关联 CVE**：漏洞具有直接映射到工业界公共漏洞库 (CVE) 的能力，这大大缩小了 [[语义鸿沟 (Semantic Gap)]]。

> 相关文献：[[NIDS using Deep Learning (Ashiku & Dagli 2021)]] 用其验证了深度卷积模型的泛化能力。
