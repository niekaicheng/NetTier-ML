# Project Proposal: A Hierarchical Network Intrusion Detection Framework

**Team Members:** 
**Project Type:** Application/Empirical

## Project Description
With the rapid growth of encrypted traffic and high-speed networks, traditional Intrusion Detection Systems (IDS) face a critical trade-off between detection accuracy and processing throughput. Deep learning models offer high precision but suffer from high latency, while statistical methods are fast but often struggle with complex attack patterns. This project proposes a **Hierarchical Network Intrusion Detection Framework** that integrates statistical machine learning with deep sequence modeling to balance efficiency and accuracy. 

The system utilizes a two-stage "funnel" architecture: **Stage 1** employs a lightweight Random Forest model to rapidly filter out benign traffic (which constitutes the majority of network data) using statistical flow features. **Stage 2** deploys a deep learning model, **TransECA-Net** (combining 1D-CNN, Efficient Channel Attention, and Transformers), to perform fine-grained classification on the suspicious traffic identified by Stage 1. We will implement this framework using Python and PyTorch, training on the real-world **CIC-IDS2017** dataset. The goal is to empirically demonstrate that this hierarchical approach can achieve high throughput (handling >90% of traffic at low cost) while maintaining state-of-the-art detection rates for complex attacks.

## Preliminary Literature List (for Review)
1. **Sharafaldin, I., et al.** (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*. (CIC-IDS2017 Source Paper)
2. **Liu, Z., et al.** (2025). *TransECA-Net: A Transformer-Based Model for Encrypted Traffic Classification*. Appl. Sci.
3. **Umer, M. F., et al.** (2017). *Machine Learning in Network Anomaly Detection: A Survey*.
4. **Ring, M., et al.** (2019). *A Survey of Network-Based Intrusion Detection Data Sets*. (Context on datasets like KDD/UNSW)
5. **Doula, et al.** *Analysis of Machine Learning-Based Methods for Network Traffic*.
6. **Alpates.** *Intelligent Network Traffic Analysis Leveraging Machine Learning for Enhanced Cybersecurity*.
7. *Machine Learning Techniques for Anomaly Detection in Network Traffic*.
8. *Network Traffic Analysis Based on Graph Neural Networks: A Scoping Review*.
9. *A Detailed Analysis of the KDD CUP 99 Data Set*.
10. *UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems*.
