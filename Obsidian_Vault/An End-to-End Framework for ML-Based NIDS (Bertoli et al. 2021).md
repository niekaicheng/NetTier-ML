# An End-to-End Framework for Machine Learning-Based NIDS
**Authors**: Gustavo de Carvalho Bertoli et al.
**Date/Venue**: IEEE Access, 2021

## 核心提要 (Core Summary)
随着 IoT / 5G 的爆发，针对 NIDS 的研究和实现要求越来越高。然而，学术界经常使用的基准数据集已经变得极为“陈旧（Obsolete）”且缺乏现实世界（Real-world）背景（正如此前 [[Semantic Gap]] 涉及的窘境）。
本作不仅探讨了如何解决数据集老化，而且创造性地提出了从“**发现问题到上线防御部署的整个端到端框架**”，即 [[AB-TRAP 框架]]。

## 解决的核心痛点：
相比于仅在实验室内比对 F1-score 的研究，这篇文章真正走出了实验室：
1. **数据集腐烂、失去时效**
   - 旧数据集（如 [[NSL-KDD 数据集]]、DARPA 98）已失去现实检验能力，因为攻击面已经变化（各种 IoT 新协议等）。
2. **缺乏系统性的落地评估 (Realization & Performance)**
   - 模型好并不代表能部署。例如 IoT 设备只有极少 CPU 和 RAM，若选用笨重深层网络，或者需要在用户态（Userspace）慢速拦截流量，会导致整个网络瘫痪。

## 技术干货
- 提出了基于内核级高效拦截或用户级的模型部署方式，并在论文中探讨了利用 [[eBPF 与模型落地部署]] 和 LKM (Linux Kernel Module) 执行智能包过滤的可行性。
- 用真实的扫描软件环境生成**纯净的攻击数据集 (Attack)**，在真实网关提取**干净的业务数据集 (Bona fide)**，然后进行混合学习 (Train)。

## 结论
未来最好的 NIDS 是**可演进（Evolving）、可复现（Reproducible），且能在部署环境证明其系统开销（CPU/RAM 评估合法）的**。仅仅追求实验室模型精度是不够的。
