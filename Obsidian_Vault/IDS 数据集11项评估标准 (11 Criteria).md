# IDS 数据集11项评估标准 (11 Criteria)

## 缘起
任何科研最怕“在错误的垃圾数据上寻找最优解”。2016 年 Gharib 等人提出了验证一个系统级网安数据集是否及格的 11 项黄金基准。而在后续的 [[Toward Generating CIC-IDS2017 Dataset (Sharafaldin et al. 2018)]] 论文中，作者正是以此为指导方针批判了市面上存留的所有数据集，并创立了满足这大圆满 11 宗的最强集合。

## 具体指标
1. **网络配置完备性 (Complete Network Configuration)**：绝不能是只有两台互开的蜜罐游戏，要有路由器、交换机以及横跨 Win/Linux/Mac 的多种端体系。
2. **流量覆盖完整 (Complete Trafﬁc)**：拥有真实化人类日常背景流量 (B-Profile) 作掩护。
3. **标签详尽 (Labelling)**：拥有极度精细的时间戳挂钩标签分类。
4. **全向互动捕获 (Complete Interaction)**：包含 LAN 内网通信互刷以及外网（Internet）通信。
5. **完整保留度 (Complete Capture)**：使用镜像端口捕获原汁原味的报文，而非阉割掉特征信息。
6. **全协议覆盖 (Available Protocols)**：提供当前社会流行的 HTTP, HTTPS, FTP, SSH 等。
7. **攻击战术多样性 (Attack Diversity)**：需涵盖从浅层的拒绝服务直至深层次漏洞爆破与横向扫描等全方位维度的威胁集合。
8. **异构性捕获 (Heterogeneity)**：不仅抓核心路由的包，还要有终端设备的反馈与交互底本提取。
9. **特征集预置 (Feature Set)**：除原始报文外，提取出通用的计算元特征（如子流吞吐、包间间隔）。
10. **元数据溯源 (MetaData)**：详细发布攻击发起拓扑图及对应精确时间点，方便后来者复现切片。
11. **非匿名化要求 (Anonymity)**：不能打码脱敏（如早年 CAIDA 掩盖 Payload 和目标），这样会破坏包载荷学习算法（指名道姓批评语义鸿沟的源头）。
