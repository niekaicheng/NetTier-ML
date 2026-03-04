# Experimental Design Methodology — Hierarchical IDS Framework

> **Objective**: Design rigorous experimental protocols for the hierarchical network intrusion detection framework, covering regularization, fitting optimization, statistical inference alternatives, Bias-Variance Trade-off, Bootstrap/CV methodology, and comprehensive visualization and evaluation strategies.
> 
> **Core References**: This document deeply integrates the following 8 key papers as academic support for the experimental design.
> 
> | Category | Abbreviation | Paper | Role in This Project |
> |---|---|---|---|
> | **Data** | **[Sharafaldin'18]** | Toward Generating a New Intrusion Detection Dataset... | Dataset foundation (CIC-IDS2017) |
> | **Data** | **[Ring'19]** | A Survey of Network-based Intrusion Detection Data Sets | **Dataset selection rationale (Why CIC-IDS2017?)** |
> | **Data** | **[Moustafa'15]** | UNSW-NB15: A comprehensive data set... | **Cross-dataset generalization benchmark** |
> | **Model** | **[Liu'25]** | TransECA-Net: A Transformer-Based Model... | Stage 2 core model architecture |
> | **Model** | **[Kwon'17]** | Deep learning-based network anomaly detection | **Stage 2 deep learning necessity justification** |
> | **ML** | **[Doula'25]** | Analysis of ML-Based Methods for Network Traffic... | Stage 1 Random Forest methodology support |
> | **ML** | **[Abu Al-Haija'22]** | ML-Based Darknet Traffic Detection System... | Ensemble learning and SHAP interpretability support |
> | **ML** | **[Kaur'21]** | ML Techniques for Anomaly Detection in Network Traffic | **Anomaly detection general evaluation criteria** |

---

## Table of Contents
1. [Data Foundation and Selection Rationale](#1-data-foundation-and-selection-rationale)
2. [Stage 1 Experimental Design: RF Regularization and Literature Support](#2-stage-1-experimental-design-rf-regularization-and-literature-support)
3. [Stage 2 Experimental Design: Deep Learning Necessity and TransECA-Net](#3-stage-2-experimental-design-deep-learning-necessity-and-transeca-net)
4. [P-value Alternatives (Random Forest Statistical Inference)](#4-p-value-alternatives-random-forest-statistical-inference)
5. [Bias-Variance Trade-off Analysis](#5-bias-variance-trade-off-analysis)
6. [Bootstrap Resampling Method](#6-bootstrap-resampling-method)
7. [Cross-Validation Strategy](#7-cross-validation-strategy)
8. [Data and Model Result Visualization](#8-data-and-model-result-visualization)
9. [Comprehensive Evaluation Framework](#9-comprehensive-evaluation-framework)
10. [Experimental Design Summary Table](#10-experimental-design-summary-table)
11. [Hierarchical Framework Mathematical Proof](#11-hierarchical-framework-mathematical-proof)

---

## 1. Data Foundation and Selection Rationale

> **Core References**:
> - **[Sharafaldin'18]** (CIC-IDS2017 original paper)
> - **[Ring'19]** (Dataset survey and evaluation criteria)

### 1.1 Why CIC-IDS2017? — Evaluation Based on [Ring'19]

[Ring'19] proposed 15 key attributes for evaluating network intrusion detection datasets. Based on these criteria, CIC-IDS2017 is one of the most suitable datasets for modern network environments, superior to the outdated KDD99/NSL-KDD.

| [Ring'19] Evaluation Criteria | CIC-IDS2017 Performance | Competitor (NSL-KDD) | Benefit to This Project |
|-------------------|------------------|---------------|-----------|
| **Year (Timeliness)** | 2017 (Modern traffic) | 2009 (Outdated) | Model can identify modern attacks (e.g., Heartbleed) |
| **Duration** | 5 days (Full work week) | - | Contains complete periodic traffic patterns |
| **Traffic Type** | Real + Simulated (B-Profile) | Synthetic/Sampled | Stage 1 can learn real background traffic distribution |
| **Labelled** | Flow-level + Packet-level | Flow-level | Supports fine-grained classification (Stage 2) |
| **Attack Diversity** | 7 major categories, 14 subcategories | 4 major categories | **Validates hierarchical architecture's ability to handle complex multi-classification** |

**Conclusion**: According to [Ring'19], CIC-IDS2017 meets the high standards for "Evaluation" and "Labeling", making it the best choice for validating the Hierarchical Framework.

### 1.2 CIC-IDS2017 Feature System ([Sharafaldin'18])

The dataset contains approximately **80 statistical features**, forming the input space for Stage 1:

| Feature Category | Representative Features | Role in Stage 1 |
|---------|----------|-----------------|
| **Temporal Statistics** | Flow Duration, IAT Mean/Std | **Core features most relied upon by RF** (identifying DoS/BruteForce) |
| **Packet/Byte Statistics** | Total Fwd/Bwd Packets, Bytes | Traffic volume and asymmetry assessment |
| **Flag Bits** | SYN/FIN/RST Count | Protocol violation detection (e.g., PortScan) |
| **Behavioral Patterns** | Subflow Stats, Window Size | Connection behavior analysis |

### 1.3 Experiment Design D1: Feature Category Contribution Analysis

```python
# Experiment D1: Feature Group Ablation
# Validate the differential contribution of feature groups from [Sharafaldin'18]
feature_groups = {
    'Time': ['Flow Duration', 'Flow IAT...'],
    'Size': ['Total Packets', 'Total Bytes...'],
    'Flag': ['SYN', 'RST', 'FIN...'],
    'Header': ['Fwd Header Len', 'Bwd Header Len...']
}
# Expected: Time > Flag > Size (for DoS/BruteForce)
```

---

## 2. Stage 1 Experimental Design: RF Regularization and Literature Support

> **Core References**:
> - **[Doula'25]** (ML method comparison)
> - **[Abu Al-Haija'22]** (Ensemble learning advantages)
> - **[Kaur'21]** (ML anomaly detection techniques)

### 2.1 Technology Selection Justification

- **RF vs SVM/DT**: [Doula'25] experiments show RF achieves **99.86%** accuracy on CIC-IDS2017, significantly outperforming single decision trees and SVMs.
- **Necessity of Bagging**: [Abu Al-Haija'22] demonstrates that Bagging-type ensemble methods (e.g., Bagging-DT) achieve the **highest accuracy (99.50%)** and extremely low latency (**9.09μs**) in IoT darknet traffic detection, perfectly fitting Stage 1's "fast and accurate" requirements.
- **Evaluation Metrics**: [Kaur'21] emphasizes that in network traffic detection, **Precision** and **Recall** are more important than Accuracy, especially when facing class imbalance.

### 2.2 Stage 1 Experiment Plan

#### E1: Hyperparameter Tuning (Nested CV)

> **See standalone document for details**: [experiment_E1_nested_cv.md](./experiment_E1_nested_cv.md)

**Experiment Summary**:
To avoid Grid Search overfitting risk and obtain unbiased model performance estimates, we employ a **Nested Cross-Validation (5×3)** strategy.

*   **Outer Loop**: 5-Fold Stratified K-Fold for evaluating model generalization performance.
*   **Inner Loop**: 3-Fold Stratified K-Fold for searching optimal hyperparameters within the training envelope.
*   **Search Space**: Covers `n_estimators`, `max_depth`, `class_weight` and other key parameters (1,296 combinations total).
*   **Objective**: Determine the optimal Stage 1 Random Forest configuration and establish the F1-Macro baseline.

---

---

#### E11: Inference Speed and Resource Benchmarking (Heterogeneous Resource Profiling)

```python
# Benchmarking against [Abu Al-Haija'22]'s 9.09μs/sample
# Target: Stage 1 throughput > 100,000 samples/s
import time
start = time.perf_counter()
rf.predict(X_test[:10000])
latency = (time.perf_counter() - start) / 10000 * 1e6
print(f"Latency: {latency:.2f} μs (Target: < 10 μs)")

# Added: Energy consumption and throughput metrics
# Metric 1: Throughput (pps - packets per second)
# Metric 2: Energy Consumption (Joules/sample)
```

#### E17: Stage 1 Threshold Trade-off (Threshold Tuning)

> **New key experiment**: Addressing the "Recall vs Efficiency" Trade-off

*   **Objective**: Find the optimal confidence threshold $\tau$ for Stage 1 such that under the constraint $Recall \ge 99.9\%$, the pass-through rate $\alpha$ to Stage 2 is minimized.
*   **Method**: Plot **Recall-Efficiency Curve**.
*   **Utility Function**: $U = F1_{system} - \lambda \times Latency_{total}$

---

## 3. Stage 2 Experimental Design: Deep Learning Necessity and TransECA-Net

> **Core References**:
> - **[Kwon'17]** (Deep learning-based network anomaly detection)
> - **[Liu'25]** (TransECA-Net model)

### 3.1 Why Does Stage 2 Need Deep Learning? — [Kwon'17] Justification

[Kwon'17] points out in the survey that as network attacks become increasingly sophisticated (e.g., encrypted traffic, attack variants), traditional shallow ML (e.g., RF/SVM) faces a **feature engineering bottleneck**.
- **Deep Learning Advantage**: Ability to automatically learn high-order feature representations from raw data (Representation Learning).
- **Application in This Project**: Stage 1 handles "quantity" filtering (conventional attacks), Stage 2 handles "quality" analysis (complex/encrypted attacks).
- TransECA-Net is based on this philosophy, utilizing **1D-CNN (local)** + **Transformer (global)** to extract deep traffic patterns.

### 3.2 TransECA-Net Architecture and Experiments ([Liu'25])

#### Core Component Functions

1.  **1D-CNN**: Extracts local features (e.g., consecutive packet size changes).
2.  **ECA-Net (Efficient Channel Attention)**: Adaptively attends to key channels.
    - Formula: $k = \psi(C) = | \frac{\log_2(C)}{\gamma} + \frac{b}{\gamma} |_{odd}$
    - Advantage: Lighter than SE-Net, no dimensionality reduction information loss.
3.  **Transformer Encoder**: Captures long-range dependencies (e.g., TCP handshake-transfer-teardown process).

#### E8: Ablation Study

> **Note**: Original E7 (DL Regularization) highly overlaps with E8 and has been merged. Regularization parameters (Dropout/L2) are training details; E8 directly validates architectural contributions through component removal.

To validate [Kwon'17]'s claim that "DL outperforms Shallow ML" and [Liu'25]'s architectural advantages:

| Experiment ID | Model Variant | Removed Component | Hypothesis Being Tested |
|---|---|---|---|
| **E8-A** | **TransECA-Net (Full)** | — | **DL SOTA performance ([Liu'25] 98.25%)** |
| **E8-B** | No-ECA (CNN+Trans) | -ECA | Validate ECA channel attention contribution |
| **E8-C** | CNN-Only | -Transformer, -ECA | Validate importance of long-sequence modeling |

---

## 4. P-value Alternatives (Random Forest Statistical Inference)

> **Academic Basis**: [Abu Al-Haija'22] uses game-theoretic methods (Shapley) for feature importance explanation

### 4.1 Approach Selection
RF does not provide p-values; the following alternatives are adopted:

1.  **SHAP TreeExplainer (Actually Used)**: [Abu Al-Haija'22] validated its effectiveness in traffic detection. Provides directional (positive/negative) contribution of features to predictions. Computes exact Shapley values using tree structure.
2.  **SHAP vs RF Importance Comparison**: Spearman rank correlation comparing SHAP and RF's built-in Gini importance rankings for consistency.

```python
# E2: SHAP TreeExplainer Feature Importance Analysis
import shap
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_sample)  # Stratified sampling 6068 samples
# Output: Summary Plot, Bar Plot, SHAP vs RF Comparison, Spearman ρ
```

> **E3 (Feature Selection) not executed separately**: RF has built-in feature importance ranking, and E2 SHAP already provides deeper feature contribution analysis, making independent Top-K selection of **marginal benefit**.

---

## 5. Bias-Variance Trade-off Analysis

### 5.1 Theoretical Mapping
- **Stage 1 (RF)**: Low Bias (strong classifier), Low Variance (Bagging mechanism [Abu Al-Haija'22]).
- **Stage 2 (DL)**: Low Bias (deep network [Kwon'17]), High Variance (many parameters, prone to overfitting).

### 5.2 Experiment E4: Bias-Variance Decomposition (Optional)

> **Priority**: Optional. Theoretical depth analysis, but less directly impactful than E1/E8 in engineering applications.
> **E5 (Validation Curve) not executed separately**: E1 Nested CV's grid search already contains hyperparameter vs. performance curve information; conclusions are fully covered.

Going beyond simple Bootstrap with a more rigorous analysis:

1.  **RF**: Plot **OOB Error vs Tree Count** curve to analyze Variance.
2.  **DL**: Plot **Train/Val Loss** curves and **Generalization Gap**.
3.  **Complexity Analysis**: Plot Model Complexity (parameter count/depth) vs Variance curve for intuitive Trade-off visualization.

---

## 6. Bootstrap Resampling Method

### 6.1 Application Scenarios
- **Performance Confidence Intervals (CI)**: Compute 95% CI for F1-scores, recommended by [Kaur'21] as a rigorous evaluation approach (rather than single test results).
- **OOB Score**: Utilize RF's built-in OOB (Out-of-Bag) estimate as an unbiased validation set performance estimate.

---

## 7. Cross-Validation Strategy

### 7.1 Stratified Design
- **Stage 1**: **Stratified K-Fold (5-fold)**. Handles the class imbalance mentioned in [Sharafaldin'18].
- **Stage 2**: **Repeated Random Holdout**. Due to high deep learning training costs, multiple random splits (e.g., 5× 80/20 splits) are averaged.

### 7.2 E16: Strict Test Isolation and Full Attack Coverage Validation

> **Objective**: Validate the model's generalization ability on "known attack types" and robustness to mixed background traffic.

*   **Data Splitting Logic (Stratified Mixed Split)**:
    *   **Train Set**:
        *   **Benign Baseline**: 80% Monday data + 80% Tue-Fri Benign data. Establishing the most robust normal traffic baseline.
        *   **Attack Coverage**: 80% of **ALL** Attack types (from Tue-Fri). Ensuring Stage 2 has seen all attack patterns.
        *   **Stage 1 Training**: Binary (Benign vs Attack).
        *   **Stage 2 Training**: Multi-class (On Stage 1 Suspicious).
    *   **Test Set**:
        *   **Reserved 20%**: All days and all attack types at this ratio.
        *   **Validation Objective**: Ensure no training data leaks into the test set; evaluate model detection rate on **known threat variants** and **unknown background traffic**.
*   **Expected**: High Recall (>99.5%) and High Precision (>95%), since the training set covers all attack distributions.

---

## 8. Data and Model Result Visualization

1.  **E10: Interpretability Upgrade**: 
    *   **RF**: SHAP Summary Plot (Global Importance).
    *   **TransECA-Net**: Introduce **Integrated Gradients** or **Attention Rollout**.
    *   **Objective**: Demonstrate that TransECA-Net attends to key Payload/Header bytes rather than noise. Validate Transformer's long-range dependency capture capability.
2.  **E12: t-SNE / UMAP**: Data manifold visualization showing separability between Benign and Attack.

---

## 9. Comprehensive Evaluation Framework

> **New key experiment**: Generalization validation based on [Moustafa'15] (UNSW-NB15)

### 9.1 Generalization Experiment (E15)
To prevent the model from merely fitting CIC-IDS2017-specific artifacts, **UNSW-NB15** is introduced as a test set.
- **Challenge**: Different feature spaces between the two datasets.
- **Approach**: Select shared generic features (e.g., Duration, Packet Count, Byte Count) for **Transfer Learning** or **Cross-Evaluation**.
- **Purpose**: Demonstrate that the framework has learned generalizable intrusion patterns, not just "memorized" CIC-IDS2017.

### 9.2 Statistical Tests
- **McNemar Test**: Compare Stage 1 (RF) with other ML models (e.g., XGBoost) for statistically significant differences.

### 9.3 E14: Adversarial Robustness Testing

> **Argument for enhanced defense capability**

*   **Method**: Generate adversarial examples using **FGSM** or **DeepFool**.
*   **Comparison**: 
    - Stage 1 (RF) perturbation resistance (typically robust to discrete features).
    - Stage 2 (TransECA-Net) perturbation resistance (DL may be sensitive).
*   **Expected**: Demonstrate that the hierarchical architecture combines RF's robustness with DL's high sensitivity.

---

## 10. Experimental Design Summary Table (Updated 2026-03-03)

### 10.1 Actual Execution Order

| Execution Order | # | Experiment Name | Method | Status | Core Literature Support |
|:---:|---|---|---|:---:|---|
| 1 | **E1+E16** | RF Hyperparameter Tuning + Time Split | Nested CV + Time-aware Split | Done | [Doula'25] |
| 2 | **E17** | Threshold Optimization | Threshold Tuning ($\alpha$ vs Recall) | Done | — |
| 3 | **E11** | Resource and Speed | Throughput/Latency Benchmark | Done | **[Abu Al-Haija'22]** |
| 4 | — | Stage 1 Full Training | 5-Fold CV Mining → S2 Data | Done | — |
| 5 | — | Stage 2 Training | TransECA-Net (30 epochs, XPU) | Done | **[Liu'25]** |
| 6 | **E8** | DL Ablation Study | -ECA, -Transformer | Done | **[Liu'25], [Kwon'17]** |
| 7 | **E15** | Cross-Dataset Validation | Architecture Generalization on UNSW-NB15 | Done | **[Moustafa'15]** |
| 8 | **E2** | Feature Importance | SHAP TreeExplainer | Done | **[Abu Al-Haija'22]** |
| 9 | **E6** | Performance CI | 1000× Bootstrap | Done | **[Kaur'21]** |
| 10 | **E10** | Interpretability | Integrated Gradients / Attention | Done | — |
| 11 | **E14** | Adversarial Robustness | FGSM/PGD Attack | Done | FGSM ε=0.01 Acc=0.7573, PGD ε=0.01 Acc=0.4728, RF ε=0.001→Acc=0.47 |
| 12 | **E4** | Bias-Variance | OOB+Gap+Complexity+LC | Done | RF OOB@50=0.0017, DL gap=+0.006, LC gap@full=0.0016 |
| 13 | **E12** | Feature Space Visualization | t-SNE / UMAP | Done | **[Sharafaldin'18]** |

### 10.2 Experiments Not Executed and Rationale

| # | Original Experiment Name | Original Method | Reason for Non-Execution |
|---|---|---|---|
| **E3** | Feature Selection (Top-K) | Information Gain | RF has built-in feature importance ranking; E2 (SHAP) already provides deeper feature contribution analysis, making independent Top-K selection of **extremely low marginal benefit** |
| **E5** | Complexity Curve | Validation Curve | E1 (Nested CV) grid search already contains complete hyperparameter vs. performance curve information; **conclusions fully covered by E1** |
| **E7** | DL Regularization | Dropout/L2/Smoothing | E8 (Ablation) already validates architectural contribution through component removal; regularization tuning is a training detail, **highly overlapping with E8** |
| **E13** | Imbalance Handling | SMOTE vs ClassWeight | `class_weight` already included in E1's hyperparameter search space; **merged into E1 execution** |

> **Note**: The original design has no E9 identifier (E8 is directly followed by E10).

### Project Presentation Logic
1.  **Dataset Selection ([Ring'19])**: Why not KDD99? Because Ring's evaluation criteria point to CIC-IDS2017.
2.  **Hierarchical Architecture ([Kwon'17])**: Why two stages? Because Kwon shows complex attacks need DL, while Abu Al-Haija proves RF handles simple traffic extremely fast.
3.  **Technical Details ([Liu'25] + [Doula'25])**: How is RF tuned? How is TransECA designed?
4.  **Result Validation ([Moustafa'15] + [Kaur'21])**: Not only accurate on the test set with narrow CIs, but also generalizes to UNSW-NB15.

---

## 11. Hierarchical Framework Mathematical Proof

Beyond citing literature, the hierarchical architecture design of this project can be rigorously proven mathematically through **Expected Computational Cost** and **Conditional Probability**.

### 11.1 Expected Computational Cost Minimization (Cost Example)

Assuming the total system computational cost is $C_{total}$, define the following variables:
- $C_1$: Stage 1 (Random Forest) average inference time (approximately $9\mu s$ per E11 experiment).
- $C_2$: Stage 2 (TransECA-Net) average inference time (deep learning is typically slower, approximately $500\mu s$).
- $\alpha$: Pass-through Rate, i.e., the proportion that Stage 1 classifies as "Suspicious" and forwards to Stage 2.
  - For normal background traffic, $\alpha \approx 0$ (RF filters out the vast majority).
  - For attack traffic, $\alpha \approx 1$ (complex attacks that RF cannot determine).

**Expected cost per inference $E[C]$:**
$$ E[C] = C_1 + \alpha \cdot C_2 $$

Since normal traffic constitutes the vast majority of network traffic (e.g., $>99\%$), the overall $\alpha$ is very small (e.g., $\alpha_{total} \approx 0.05$).
$$ E[C] \approx 9\mu s + 0.05 \times 500\mu s = 9 + 25 = 34\mu s $$

**Compared to a single-layer deep learning model (TransECA-Net only):**
$$ C_{DL\_only} = C_2 = 500\mu s $$

**Conclusion**: $E[C] \ll C_{DL\_only}$. The hierarchical architecture improves theoretical speed by **14×** ($500/34$) while maintaining deep learning accuracy.

### 11.2 Cost-Sensitive Risk Minimization

Define total risk $R_{total}$ as the weighted sum of false positives and false negatives:
$$ R_{total} = C_{FN} \cdot P(FN) + C_{FP} \cdot P(FP) + C_{comp} \cdot E[Computational\_Cost] $$

Where:
*   $C_{FN}$: Cost of missed detection (extremely high, leads to successful intrusion).
*   $C_{FP}$: Cost of false alarm (moderate, causes administrator fatigue).
*   $C_{comp}$: Unit cost of computational resources.

**Derivation Refinement**:
The advantage of the hierarchical architecture is that it allows us to use the inexpensive $C_1$ (Stage 1) to dramatically reduce $P(FN)$ (through a high Recall threshold), paying the expensive $C_2$ (Stage 2) only for a small fraction of samples ($\alpha$) to reduce $P(FP)$. This proves that the framework is the global optimum for **Risk-Cost Utility**.

Define events:
- $D$: System successfully detects an attack.
- $S_1$: Stage 1 output (0=Normal, 1=Suspicious).
- $S_2$: Stage 2 output (0=Normal, 1=Attack).

The system's detection logic is a variant of a **Serial System** (Filter-and-Refine):
1. If $S_1=0$, output Normal directly (fast path).
2. If $S_1=1$, execute $S_2$, output $S_2$'s result.

The system's **Recall (True Positive Rate, TPR)** can be expressed as:
$$ P(D|Attack) = P(S_1=1|Attack) \times P(S_2=1|Attack, S_1=1) $$

- $P(S_1=1|Attack)$: Stage 1's Recall. RF needs a low threshold (e.g., confidence > 0.3 treated as Suspicious) to ensure this term approaches 100%.
- $P(S_2=1|Attack, S_1=1)$: Stage 2's Recall on hard examples. This is where Deep Learning plays its role.

**False Positive Rate (FPR)**:
$$ P(D|Normal) = P(S_1=1|Normal) \times P(S_2=1|Normal, S_1=1) $$

- $P(S_1=1|Normal)$: Stage 1's false positive rate. Even if RF produces false positives (classifying normal traffic as suspicious), as long as the second layer TransECA-Net correctly identifies them, $P(S_2=1|Normal...) \approx 0$, and the total false positive rate remains extremely low.

**Conclusion**: The hierarchical architecture ensures high recall through $S_1$ (better safe than sorry) and high precision through $S_2$ (reducing false positives), thus mathematically achieving dual optimization of Precision and Recall.
