# A Survey of Network-based Intrusion Detection Data Sets
**Authors**: Markus Ring, Sarah Wunderlich, Deniz Scheuring, Dieter Landes, Andreas Hotho
**Date/Venue**: Computers & Security, 2019

## 核心提要 (Core Summary)
目前阻碍 NIDS 落地与异常检测算法飞跃的头号元凶，就是“缺乏代表性的、高质量的网络评估数据集”。市面上的数据要么太老（如 DARPA, KDD CUP 99 彻底过时），要么缺乏脱敏，要么缺失对应的标签。
这篇综述（2019 年）提供了一个非常硬核的数据集索引，研究者彻底拆解并分类了市面上存在的多达 34 种主要安全评估数据集。

## 解决的问题
1. **建立挑选依据指南**：
   引入了 [[NIDS 数据集 15 项评估属性 (15 Properties)]]，从创建时间、数据格式、标签到来源背景网络类型，将寻找适配模型训练数据集的过程结构化。这比之前的盲目试用提供了宏观上帝视角。
2. **底层格式明晰**：
   专门将所有评估基准的基础格式划分为三大类进行批判：[[基于包与基于流的数据对比 (Packet vs Flow-based Data)]]（也就是 packet-based， flow-based 还是 hybrid other）。例如：有的模型专门吃特征值统计流，那么就该用 NetFlow/IPFIX 类型的数据去灌；若要用深度学习去算字节间概率，则要寻觅全包留存集（.pcap）。
3. **数据集间血脉梳理**：
   给出了各大数据集的衍生关联（譬如 NSL-KDD 是 KDD99 的删减，Botnet dataset 杂交自 ISOT 和 CTU-13 等）。

## 核心结论
- **Flow 数据的抬头**：随着加密和流量爆发，带载荷的全包（Packet）捕捉越来越难以合规化和高负荷长期运行，含有聚合状态的 Flow 数据将在工业界更有话语权。
- **良性用户行为模拟之痛**：多数数据集里含有极为丰富的攻击，却往往无法模拟足够随机、多变的“正常人流量（Normal user behavior）”，这也是现在检测模型上生产就高假阳性疯狂报错的病根。
