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

## Literature (ACM Format)
[1] Q. Abu Al-Haija, A. Odeh, and H. Qattous. 2022. ML-Based darknet traffic detection system. IEEE Access 10 (2022), 87608–87621.

[2] S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, and J. W. Vaughan. 2010. A theory of learning from different domains. Machine Learning 79 (2010), 151–175.

[3] L. Breiman. 2001. Random forests. Machine Learning 45, 1 (2001), 5–32.

[4] G. C. Cawley and N. L. C. Talbot. 2010. On over-fitting in model selection and subsequent selection bias in performance evaluation. Journal of Machine Learning Research 11 (2010), 2079–2107.

[5] J. Wang. 2025. Analysis of machine learning-based methods for network traffic anomaly detection and prediction. In Proceedings of the 2nd International Conference on Data Science and Engineering (ICDSE). 550–554.

[6] B. Efron. 1979. Bootstrap methods: another look at the jackknife. The Annals of Statistics 7, 1 (1979), 1–26.

[7] R. Singh, N. Srivastava, and A. Kumar. 2021. Machine learning techniques for anomaly detection in network traffic. In Proceedings of the 2021 Sixth International Conference on Image Information Processing (ICIIP). 261–266.

[8] D. Kwon, H. Kim, J. Kim, S. C. Suh, I. Kim, and K. J. Kim. 2017. Deep learning-based network anomaly detection. Cluster Computing 22, 1 (2017), 209–224.

[9] Z. Liu, et al. 2025. TransECA-Net: A transformer-based model for encrypted traffic classification. Applied Sciences 15 (2025).

[10] I. Loshchilov and F. Hutter. 2016. SGDR: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983 (2016).

[11] I. Loshchilov and F. Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017).

[12] S. M. Lundberg and S. I. Lee. 2017. A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 30.

[13] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. 2018. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations (ICLR).

[14] L. McInnes, J. Healy, and J. Melville. 2018. UMAP: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426 (2018).

[15] N. Moustafa and J. Slay. 2015. UNSW-NB15: a comprehensive data set for network intrusion detection systems. In Proceedings of the 2015 Military Communications and Information Systems Conference (MilCIS). 1–6.

[16] M. Ring, S. Wunderlich, D. Scheuring, D. Landes, and A. Hotho. 2019. A survey of network-based intrusion detection data sets. Computers & Security 86 (2019), 147–167.

[17] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani. 2018. Toward generating a new intrusion detection dataset and intrusion traffic characterization. In Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP). 108–116.

[18] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. 2014. Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research 15, 1 (2014), 1929–1958.

[19] M. Sundararajan, A. Taly, and Q. Yan. 2017. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning (ICML). 3319–3328.

[20] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 30.

[21] L. van der Maaten and G. Hinton. 2008. Visualizing data using t-SNE. Journal of Machine Learning Research 9, 11 (2008), 2579-2605.

[22] B. Littlewood and L. Strigini. 2004. Redundancy and diversity in security. IEEE Security & Privacy 2, 3 (2004), 56–61.

[23] Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, and Q. Hu. 2020. ECA-Net: Efficient channel attention for deep convolutional neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 11534–11542.

[24] I. J. Goodfellow, J. Shlens, and C. Szegedy. 2015. Explaining and harnessing adversarial examples. In International Conference on Learning Representations (ICLR).

[25] S. Abnar and W. Zuidema. 2020. Quantifying attention flow in transformers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL). 4190–4197.
