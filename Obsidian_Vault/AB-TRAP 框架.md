# AB-TRAP 框架

## 定义
在检测系统研究中，不能仅仅关注 AI 模型的训练。**AB-TRAP** (Attack, Bona Fide, Train, RealizAtion, and Performance) 是由 Bertoli 等人在 2021 年提出的五篇式端到端开发周期框架，用以指导 ML-Based NIDS 项目从概念到最终落地的闭环：

1. **A - Attack Dataset (攻击数据集生成)**
   - 不要依赖陈旧的公共数据，要用当下最新的攻击载荷、扫描器（如 Zmap、masscan 等）在无害环境中打出攻击特征流量。
2. **B - Bona fide Dataset (安全白数据集)**
   - “Bona fide”指真正的合法业务流量。采集该流量后与步骤 1 的数据实施“掺沙子 (Salting)”混合。
3. **TR - Train (训练阶段)**
   - 使用混合数据集，选取适合的评估指标（如 F1-Score / ROC / AUC）针对应用场景挑选最合理的算法（未必需要 DL，如决策树在特定情况也很优秀）。
4. **A - RealizAtion (系统落地实现)**
   - 模型不是代码文件，需要真刀真枪跑在路由器/IoT设备上。必须考虑部署是在云端、雾计算中心，还是内存吃紧的网卡级别（如配合 [[eBPF 与模型落地部署]]）。
5. **P - Performance evaluation (服役性能考评)**
   - AI 防御不能拖垮主系统。衡量 NIDS 组件占用的 CPU/RAM 开销以及对网络吞吐率 (Throughput) 造成的下降。

> 参考文献溯源：[[An End-to-End Framework for ML-Based NIDS (Bertoli et al. 2021)]]
