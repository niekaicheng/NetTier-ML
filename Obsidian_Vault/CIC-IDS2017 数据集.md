# CIC-IDS2017 数据集

## 概念定义
这是由加拿大新布伦瑞克大学 (University of New Brunswick) 加拿大网络安全研究院（CIC）于 2017 年打造的一款标志性高拟真新一代网络入侵检测防御评估数据集。
**在学术地位上，它与 [[UNSW-NB15 数据集]] 一起，正式埋葬了老迈失真的 KDD99 和 NSL-KDD，强行拔高了近今年 NIDS 论文成果验收的标准线。**

## 数据集特征亮点
1. 完全通过高度模拟化的 Attack-Network 和具有物理实体和独立操作系统的 Victim-Network 进行内网/公网的双向对抗捕获得出。
2. 捕获历时 5 天的长效攻击（涵盖：良性白流量日、暴力破解日、DoS与心脏出血日、网页渗透日以及僵尸网络/DDoS日）。
3. 提供了原生 Pcap 的同时，也极其友善地配套提供了预先由 CICFlowMeter 提取的 **80 个标准网络统计物理特征**（包含 IAT 最小均值、Subflow 包长等等）为快速训练建立底层特征对齐。

>相关溯源：发源于重要文献 [[Toward Generating CIC-IDS2017 Dataset (Sharafaldin et al. 2018)]]，并遵循了 [[IDS 数据集11项评估标准 (11 Criteria)]] 之规范底本。
