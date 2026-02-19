# Project Proposal: A Hierarchical Network Intrusion Detection Framework

**Team Members:** 
**Project Type:** Application/Empirical

## Project Description
With the rapid growth of encrypted traffic and high-speed networks, traditional Intrusion Detection Systems (IDS) face a critical trade-off between detection accuracy and processing throughput. Deep learning models offer high precision but suffer from high latency, while statistical methods are fast but often struggle with complex attack patterns. This project proposes a **Hierarchical Network Intrusion Detection Framework** that integrates statistical machine learning with deep sequence modeling to balance efficiency and accuracy. 

The system utilizes a **Two-Stage Funnel** architecture with a **Stratified Mixed-Supervision** strategy:

1.  **Stage 1 (Fast Filter)**: A lightweight **Random Forest** trained on a **Mixed Benign Baseline** (80% of Monday + Benign samples from Tue-Fri) and **70-80% of Known Attacks** (all types). Its goal is to maximize **Recall** (>99.9%) to filter out >90% of benign traffic, passing any "suspicious" (Not Clearly Benign) traffic to Stage 2.
2.  **Stage 2 (Deep Analyst)**: A **TransECA-Net** deep learning model trained on the "suspicious" subset of the **Training Set** (Stage 1 Positives). Its goal is to maximize **Precision** in identifying specific attack types (e.g., DoS, BruteForce, Botnet) from the complex traffic that bypassed Stage 1.
3.  **Evaluation (Stict Isolation)**: The remaining **20-30% of Data** (covering all days and attack types) is reserved as a strict Test Set to verify the framework's generalization on **Known Attack Types** and unseen Benign variations.

We will implement this framework using Python and PyTorch, benchmarking against the **CIC-IDS2017** dataset.

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
