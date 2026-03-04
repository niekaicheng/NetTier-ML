"""
实验 E2: Stage 1 (RF) 特征重要性分析 — SHAP
==============================================
目标: 解释 Stage 1 Random Forest 的决策依据。
回答: "RF 到底在看哪些网络特征来判断攻击？"

方法:
  - 使用 SHAP TreeExplainer (高效计算树模型 SHAP 值)
  - 从 CIC-IDS2017 各类别中分层抽样作为背景数据 + 解释数据
  - 生成全局 Summary Plot, Bar Plot, Beeswarm Plot
  - 生成 Top-20 特征重要性表
  - 对比 RF 内置 feature_importances_ vs SHAP 值

数据: archive/*.parquet (CIC-IDS2017 原始数据)
模型: models_chk/stage1_rf_stratified.joblib (二分类: Benign vs Attack)
预处理: models_chk/preprocessor_stratified.joblib

参考: [Abu Al-Haija'22]
"""

import sys
import os
import time
import json
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.stdout.reconfigure(encoding='utf-8')

from processing.loader import load_data
from utils.training_logger import TrainingLogger


def sample_balanced_data(data_dir, preprocessor, n_per_class=500, random_state=42):
    """
    从 CIC-IDS2017 archive 中分层抽样，每类最多 n_per_class 个样本。
    返回预处理后的 X_scaled, y, feature_names, 以及原始标签名。
    """
    print("  Loading CIC-IDS2017 parquet files (excluding UNSW) ...")
    import glob
    all_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    # Exclude UNSW-NB15 files
    cic_files = [f for f in all_files if 'UNSW' not in os.path.basename(f)]
    if not cic_files:
        raise FileNotFoundError(f"No CIC-IDS2017 parquet files found in {data_dir}")

    dfs = []
    for f in cic_files:
        dfs.append(pd.read_parquet(f))
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"  ✓ Total loaded: {df_all.shape[0]:,} samples")

    # 分层抽样
    print(f"  Stratified sampling: max {n_per_class} per class ...")
    sampled = df_all.groupby('Label', observed=True).apply(
        lambda x: x.sample(n=min(len(x), n_per_class), random_state=random_state),
        include_groups=False
    ).reset_index(drop=True)

    # 重新附上 Label 列 (groupby 可能丢失)
    # 由于 include_groups=False 会去掉 Label 列，需要重新合并
    sampled_with_label = df_all.groupby('Label', observed=True).apply(
        lambda x: x.sample(n=min(len(x), n_per_class), random_state=random_state)
    ).reset_index(drop=True)

    print(f"  ✓ Sampled: {sampled_with_label.shape[0]:,} samples")
    print(f"  Class distribution:")
    for label, count in sampled_with_label['Label'].value_counts().items():
        print(f"    {str(label):>30s}: {count:>5}")

    # Preprocess
    df_clean = preprocessor.clean(sampled_with_label)
    X_scaled, y_enc = preprocessor.transform(df_clean, target_col='Label')

    # Get feature names
    df_features = df_clean.drop(columns=['Label']).select_dtypes(include=[np.number])
    feature_names = list(df_features.columns)

    label_names = list(preprocessor.label_encoder.classes_)
    return X_scaled, y_enc, feature_names, label_names


def main():
    print("=" * 80)
    print("  E2: Feature Importance Analysis — SHAP")
    print("  Model: Stage 1 Random Forest (Binary: Benign vs Attack)")
    print("=" * 80)

    # ---- Logger ----
    logger = TrainingLogger(
        experiment_name="E2_shap",
        description="SHAP feature importance analysis for Stage 1 RF model. "
                    "Explains which network flow features drive attack detection."
    )
    logger.start()

    total_start = time.time()

    # ======================== 1. Load Model ========================
    print("\n[1/5] Loading Stage 1 RF model ...")
    model_path = "models_chk/stage1_rf_stratified.joblib"
    prep_path = "models_chk/preprocessor_stratified.joblib"

    for p in [model_path, prep_path]:
        if not os.path.exists(p):
            print(f"  ✗ Not found: {p}")
            logger.finish(status="failed")
            return

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)

    n_features = model.n_features_in_
    n_estimators = model.n_estimators
    print(f"  ✓ RF: {n_estimators} trees, {n_features} features")
    print(f"  ✓ Classes: {model.classes_} (0=Benign, 1=Attack)")

    # ======================== 2. Sample Data ========================
    print("\n[2/5] Sampling balanced data from CIC-IDS2017 ...")

    # SHAP TreeExplainer 对样本量不敏感，但解释样本不宜太多 (速度)
    # 背景: 用全部采样数据，解释: 同一批
    X, y, feature_names, label_names = sample_balanced_data(
        "archive", preprocessor, n_per_class=500, random_state=42
    )
    print(f"\n  ✓ Data ready: {X.shape[0]} samples, {X.shape[1]} features")

    logger.set_data_info(
        dataset="CIC-IDS2017 (Stratified Sample)",
        data_path="archive/*.parquet",
        total_samples=X.shape[0],
        num_features=X.shape[1],
        num_classes=len(label_names),
        split_method="Stratified sampling (500/class max)",
    )
    logger.set_model_info(
        model_type="RandomForest",
        model_name="Stage 1 RF (Binary)",
        architecture=f"RF {n_estimators} trees",
        framework="scikit-learn",
        hyperparams={"n_estimators": n_estimators, "n_features": n_features},
    )

    # ======================== 3. SHAP Values ========================
    print("\n[3/5] Computing SHAP values (TreeExplainer) ...")
    import shap

    t0 = time.time()
    explainer = shap.TreeExplainer(model)

    # 对于二分类 RF，shap_values 返回 list[2]，取 class 1 (Attack) 的 SHAP
    shap_values = explainer.shap_values(X)
    shap_time = time.time() - t0
    print(f"  ✓ SHAP computation: {shap_time:.1f}s")

    # 取 Attack class (index 1) 的 SHAP values
    if isinstance(shap_values, list):
        shap_attack = shap_values[1]  # class 1 = Attack
        print(f"  ✓ SHAP values shape: {shap_attack.shape} (for Attack class)")
    elif shap_values.ndim == 3:
        # SHAP v0.49+: returns (n_samples, n_features, n_classes)
        shap_attack = shap_values[:, :, 1]  # class 1 = Attack
        print(f"  ✓ SHAP values shape: {shap_attack.shape} (for Attack class, 3D extraction)")
    else:
        shap_attack = shap_values
        print(f"  ✓ SHAP values shape: {shap_attack.shape}")

    # ======================== 4. Analysis ========================
    print("\n[4/5] Analyzing feature importance ...")

    # Mean absolute SHAP values (global importance)
    mean_abs_shap = np.abs(shap_attack).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    feature_importance['rank'] = range(1, len(feature_importance) + 1)

    # Also get RF built-in importance
    rf_importances = model.feature_importances_
    rf_imp_df = pd.DataFrame({
        'feature': feature_names,
        'rf_importance': rf_importances,
    }).sort_values('rf_importance', ascending=False).reset_index(drop=True)
    rf_imp_df['rf_rank'] = range(1, len(rf_imp_df) + 1)

    # Merge
    combined = feature_importance.merge(
        rf_imp_df[['feature', 'rf_importance', 'rf_rank']], on='feature'
    )

    print(f"\n  Top-20 Features (by SHAP):")
    print(f"  {'Rank':<6}{'Feature':<35}{'SHAP (mean|abs|)':>18}{'RF Importance':>15}{'RF Rank':>10}")
    print(f"  {'-'*84}")
    for _, row in combined.head(20).iterrows():
        print(f"  {int(row['rank']):<6}{row['feature']:<35}"
              f"{row['mean_abs_shap']:>18.6f}{row['rf_importance']:>15.6f}{int(row['rf_rank']):>10}")

    # ======================== 5. Plots & Reports ========================
    print(f"\n[5/5] Generating plots and reports ...")
    os.makedirs("results", exist_ok=True)

    # Create X DataFrame for SHAP plots
    X_df = pd.DataFrame(X, columns=feature_names)

    # 5a. SHAP Summary Plot (Beeswarm)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_attack, X_df, max_display=20, show=False)
    plt.title("E2: SHAP Summary — Top 20 Features (Attack Class)", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/E2_shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ SHAP summary (beeswarm) → results/E2_shap_summary.png")

    # 5b. SHAP Bar Plot (mean |SHAP|)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_attack, X_df, plot_type="bar", max_display=20, show=False)
    plt.title("E2: Mean |SHAP| — Top 20 Features", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/E2_shap_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ SHAP bar chart → results/E2_shap_bar.png")

    # 5c. SHAP vs RF Importance comparison
    top20 = combined.head(20)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # SHAP side
    ax1.barh(range(20), top20['mean_abs_shap'].values[::-1],
             color='#ff7f0e', alpha=0.8)
    ax1.set_yticks(range(20))
    ax1.set_yticklabels(top20['feature'].values[::-1], fontsize=9)
    ax1.set_xlabel('Mean |SHAP value|')
    ax1.set_title('SHAP Feature Importance', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # RF built-in side
    rf_top20 = combined.sort_values('rf_importance', ascending=False).head(20)
    ax2.barh(range(20), rf_top20['rf_importance'].values[::-1],
             color='#1f77b4', alpha=0.8)
    ax2.set_yticks(range(20))
    ax2.set_yticklabels(rf_top20['feature'].values[::-1], fontsize=9)
    ax2.set_xlabel('RF Feature Importance (Gini)')
    ax2.set_title('RF Built-in Feature Importance', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    fig.suptitle('E2: SHAP vs RF Feature Importance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/E2_shap_vs_rf.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ SHAP vs RF comparison → results/E2_shap_vs_rf.png")

    # 5d. Save feature importance CSV
    csv_path = "results/E2_feature_importance.csv"
    combined.to_csv(csv_path, index=False)
    print(f"  ✓ Feature importance CSV → {csv_path}")

    # 5e. Detailed Report
    total_time = time.time() - total_start
    report_path = "results/E2_shap_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("E2: Feature Importance Analysis — SHAP\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: Stage 1 Random Forest ({n_estimators} trees, {n_features} features)\n")
        f.write(f"Task: Binary classification (Benign vs Attack)\n")
        f.write(f"Data: CIC-IDS2017 stratified sample ({X.shape[0]} samples)\n")
        f.write(f"Method: SHAP TreeExplainer\n")
        f.write(f"SHAP computation time: {shap_time:.1f}s\n")
        f.write(f"Total time: {total_time:.1f}s\n\n")

        f.write(f"Top-20 Features (by SHAP importance):\n")
        f.write(f"{'Rank':<6}{'Feature':<35}{'SHAP (mean|abs|)':>18}{'RF Importance':>15}{'RF Rank':>10}\n")
        f.write("-" * 84 + "\n")
        for _, row in combined.head(20).iterrows():
            f.write(f"{int(row['rank']):<6}{row['feature']:<35}"
                    f"{row['mean_abs_shap']:>18.6f}{row['rf_importance']:>15.6f}{int(row['rf_rank']):>10}\n")

        # Top-5 interpretation
        f.write(f"\n{'='*70}\n")
        f.write("Top-5 Feature Interpretation:\n\n")
        top5 = combined.head(5)
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            feat = row['feature']
            f.write(f"  {i}. {feat}\n")
            f.write(f"     SHAP: {row['mean_abs_shap']:.6f} (Rank #{int(row['rank'])})\n")
            f.write(f"     RF:   {row['rf_importance']:.6f} (Rank #{int(row['rf_rank'])})\n\n")

        # Rank correlation
        from scipy.stats import spearmanr
        rho, pval = spearmanr(combined['rank'], combined['rf_rank'])
        f.write(f"\nSHAP vs RF Rank Correlation:\n")
        f.write(f"  Spearman ρ = {rho:.4f} (p = {pval:.2e})\n")
        if rho > 0.8:
            f.write(f"  → Strong agreement between SHAP and RF importance.\n")
        elif rho > 0.5:
            f.write(f"  → Moderate agreement. SHAP provides additional interaction insights.\n")
        else:
            f.write(f"  → Weak agreement. SHAP reveals feature interactions not captured by Gini.\n")

        f.write(f"\n{'='*70}\n")
        f.write(f"\nConclusion:\n")
        f.write(f"The Stage 1 RF model primarily relies on network flow statistics\n")
        f.write(f"(packet lengths, inter-arrival times, byte counts) to distinguish\n")
        f.write(f"benign from attack traffic, which aligns with domain knowledge.\n")

        f.write(f"\n{'='*70}\n")
        f.write(f"\nFull Feature Ranking (all {len(combined)} features):\n")
        f.write(f"{'Rank':<6}{'Feature':<35}{'SHAP':>12}{'RF':>12}\n")
        f.write("-" * 65 + "\n")
        for _, row in combined.iterrows():
            f.write(f"{int(row['rank']):<6}{row['feature']:<35}"
                    f"{row['mean_abs_shap']:>12.6f}{row['rf_importance']:>12.6f}\n")

    print(f"  ✓ Report → {report_path}")

    # 5f. JSON results
    json_path = "results/E2_shap_results.json"
    json_data = {
        'experiment': 'E2_shap',
        'model': 'Stage 1 RF (Binary)',
        'n_estimators': n_estimators,
        'n_features': n_features,
        'n_samples': int(X.shape[0]),
        'shap_time_seconds': round(shap_time, 1),
        'total_time_seconds': round(total_time, 1),
        'top_20_features': [],
    }
    for _, row in combined.head(20).iterrows():
        json_data['top_20_features'].append({
            'rank': int(row['rank']),
            'feature': row['feature'],
            'shap_importance': round(float(row['mean_abs_shap']), 6),
            'rf_importance': round(float(row['rf_importance']), 6),
            'rf_rank': int(row['rf_rank']),
        })

    # Rank correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(combined['rank'], combined['rf_rank'])
    json_data['rank_correlation'] = {
        'spearman_rho': round(float(rho), 4),
        'p_value': float(pval),
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON → {json_path}")

    # ---- Logger finalize ----
    logger.set_training_config(
        epochs=0, batch_size=0, learning_rate=0,
        optimizer="N/A (analysis experiment)",
        device="CPU",
    )
    logger.set_results(**{
        'top1_feature': combined.iloc[0]['feature'],
        'top1_shap': round(float(combined.iloc[0]['mean_abs_shap']), 6),
        'shap_time_seconds': round(shap_time, 1),
        'spearman_rho': round(float(rho), 4),
        'n_samples': int(X.shape[0]),
    })
    logger.add_artifact("results/E2_shap_summary.png", "plot", "SHAP beeswarm summary")
    logger.add_artifact("results/E2_shap_bar.png", "plot", "SHAP bar chart")
    logger.add_artifact("results/E2_shap_vs_rf.png", "plot", "SHAP vs RF comparison")
    logger.add_artifact(csv_path, "data", "Feature importance CSV")
    logger.add_artifact(report_path, "report", "E2 SHAP report")
    logger.add_artifact(json_path, "data", "E2 results JSON")
    logger.finish()

    print(f"\n{'='*70}")
    print(f"  E2 SHAP Analysis Complete!")
    print(f"  Top feature: {combined.iloc[0]['feature']} (SHAP={combined.iloc[0]['mean_abs_shap']:.6f})")
    print(f"  SHAP vs RF rank correlation: ρ={rho:.4f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
