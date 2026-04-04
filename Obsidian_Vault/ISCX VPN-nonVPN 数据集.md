# ISCX VPN-nonVPN 数据集

## 概念定义
这是基于多类加密应用协议搭建的场景化评估基准。该数据集并非用于基础入侵（如 DDoS），而是专长于衡量 AI 识别[[加密流量分类 (Encrypted Traffic Classification)]] 这类“黑暗森林”行为的能力。

## 数据集特征
1. **多重服务模拟分类 (12 Classes)**：
   包含了 6 种完全裸奔的流量：浏览 (Browsing), 邮件 (Email), 聊天 (Chat), 流媒体 (Streaming), 文件传输 (File Transfer), 网络电话 (VoIP)。
   还并行捕获了 6 种这套动作套上 **VPN 前缀协议** 后的暗河封包：VPN-Chat, VPN-Streaming 等。
2. **截断像素化 (Image representation)**：
   在很多如 [[TransECA-Net for Encrypted Traffic Classification (Liu et al. 2025)]] 论文中，人们只拦截各个会话连接（Flow/Session）的前 784 bytes，然后通过转化成为 0-255 的灰阶数值，塞入深度（例如包含 28x28 像素池的 [[CNN 网络入侵检测]]）模型中分析。由于是加密报文，这些截断图肉眼看就是白底噪音点，但 AI 却能读懂里面潜藏的空间波浪。
