# 安全事件概化 (Event Profiling)

## 定义
在检测网络入侵和处理安全防线 (如防火墙、SIEM、IPS) 时，系统产生的原始日志数据（Raw Logs）是极其零碎、非结构化且充满高维噪音的。
**事件概化 (Event Profiling)** 指的是在将数据喂入机器学习（ML/DL）进行特征学习前的一套系统化降噪处理标准：将独立的 IP 游走事件、时间戳集、端口探测频次、报文长度进行标准化、类别化（Categorical Encoding）甚至是图聚合融合。最终吐出的不再是单行的碎日志，而是一个描述局部连贯行为的**行为特征文件（Profiles Vectors）**。

## 核心战术价值点
- **暴降唯度难题 (Dimensionality Reduction)**：极度清晰干瘪的 Profile 能让下游接收的神经网络不用去耗费巨大算力寻找长距离联系。
- **让简单网络发光**：正如研究证明，在高质量的 Profiling 投喂下，基础的人工神经网络 (ANN) 就可完胜耗时耗力的深度卷积系统（CNN）在 NIDS 领域的准确率。
> 相关参考：[[Cyber Threat Detection using Event Profiles (Reddy & Bell 2025)]]
