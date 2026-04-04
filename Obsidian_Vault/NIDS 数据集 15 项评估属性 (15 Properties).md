# NIDS 数据集 15 项评估属性 (15 Properties)

针对学术界常常随意使用不严谨或老套数据集进行 AI 模型训练的情况，[[A Survey of NIDS Data Sets (Ring et al. 2019)]] 中整理提炼出衡量任何一个安全靶场数据集合规与优秀的 15 条通用属性标尺。它分为以下 5 大类别：

## 1. 通用信息 (General Information)
1. **产生年份 (Year of creation)**：越新越好，过老的数据可能充斥着现在绝迹的旧版本协议从而让模型沾染严重的领域漂移。
2. **公开获取 (Public availability)**：不开源的数据集无法提供学术可复现性验证。
3. **正常流包含情况 (Normal user behavior)**：光打攻击没有用，模型极需要海量“平民流量”拉低假阳报警率。
4. **异常流包含情况 (Attack traffic)**：必须明确混杂了何种类型的攻击（DoS，探针，零日 等）。

## 2. 数据性质 (Nature of data)
5. **元数据附赠 (Metadata)**：提供网络拓扑或配置的白皮书支持，防止研究员瞎猜参数意义。
6. **格式 (Format)**：是原始的 Packet 格式、轻量的 Flow 还是整合日志（如详见 [[基于包与基于流的数据对比 (Packet vs Flow-based Data)]]）。
7. **脱敏度 (Anonymity)**：处理过 IP 或者是清洗了恶意 payload，关系到是否能够进行原始重现。

## 3. 数据体量 (Data volume)
8. **数据条数 (Count)**：包含的包或者日志流条目总数度量级。
9. **捕获持续时间 (Duration)**：长周期的监控能够保留企业周间日或白昼黑夜等“周期节律（Periodic effects）”。

## 4. 采集环境 (Recording environment)
10. **流量种类 (Kind of traffic)**：是在真实服务器采集、虚拟机沙盒测试床仿真、还是脚本计算合成出来的？
11. **背景网络 (Type of network)**：来源是家庭局域网，还是 ISP 骨干网大动脉，决定了背景噪声级别不同。
12. **路由完整性 (Complete network)**：是否具备多路由分发和跳板复杂拓扑情况。

## 5. 模型验证向支持 (Evaluation)
13. **切分辨识 (Predefined splits)**：是否提供官方标定的 Train/Test 集，来确保不同机构训练出来的横向对比公平性。
14. **类别均衡 (Balanced)**：标签样本数量是否高度左倾，以防随机森林或前馈网络产生极化骗分拟合。
15. **标注情况 (Labeled)**：包含攻击帧的打标（二分类还是多分类溯源标示）。这也是监督学习算法活下去的最基础燃料。

只有跨越了这 15 大雷达属性中的多数合格线（如：[[CIC-IDS2017 数据集]] 和 [[UNSW-NB15 数据集]]），才能被引荐成为真正的 NIDS SOTA 模型评测基准。
