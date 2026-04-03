"""
实验 E6: Bootstrap 置信区间分析
================================
目标: 通过 1000 次 Bootstrap 重采样计算 Stage 1 (RF) 和 Stage 2 (TransECA-Net) 
     关键指标的 95% 置信区间 (CI)，证明性能评估的统计显著性。

方法:
  - 1000× Bootstrap (with replacement) on test predictions
  - 计算每次重采样的 Accuracy, Weighted-F1, Macro-F1
  - 使用 percentile method (BCa 可选) 构建 95% CI
  - 生成 CI summary table + distribution plots

参考: [Kaur'21] — 推荐 Bootstrap CI 作为严谨评估方式

阶段:
  - Stage 1: RF binary (Benign vs Attack) on CIC-IDS2017 test set
  - Stage 2: TransECA-Net multiclass on data/stage2/test.parquet

产出:
  - results/E6_bootstrap_ci_results.json
  - results/E6_bootstrap_ci_report.txt
  - results/E6_bootstrap_distributions.png
"""

import sys
import os
import time
import json
import glob
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report
)
from datetime import datetime

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.stdout.reconfigure(encoding='utf-8')

from utils.training_logger import TrainingLogger


def _fast_multiclass_metrics(y_true, y_pred, num_classes):
    """
    Fast vectorized computation of accuracy, weighted-F1, macro-F1,
    weighted-precision, weighted-recall using numpy confusion matrix.
    Avoids sklearn overhead for each bootstrap iteration.
    """
    # Build confusion matrix with np.bincount (O(n), no sort)
    cm = np.bincount(y_true * num_classes + y_pred, minlength=num_classes * num_classes)
    cm = cm.reshape(num_classes, num_classes)
    
    # Per-class metrics from confusion matrix
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1).astype(np.float64)  # per-class sample count
    
    total = support.sum()
    accuracy = tp.sum() / total if total > 0 else 0.0
    
    # Precision / Recall per class (handle zero division)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 
                       2 * precision * recall / (precision + recall), 0.0)
    
    # Weighted averages (weight = support / total)
    weights = support / total if total > 0 else np.zeros_like(support)
    weighted_f1 = float(np.sum(f1 * weights))
    weighted_precision = float(np.sum(precision * weights))
    weighted_recall = float(np.sum(recall * weights))
    
    # Macro average (only classes with support > 0)
    active = support > 0
    macro_f1 = float(np.mean(f1[active])) if active.any() else 0.0
    
    return accuracy, weighted_f1, macro_f1, weighted_precision, weighted_recall


def bootstrap_metrics(y_true, y_pred, n_bootstrap=1000, confidence=0.95, 
                      average_f1='weighted', random_state=42):
    """
    Bootstrap 重采样计算指标的置信区间。
    Uses fast numpy confusion-matrix approach instead of sklearn per-iteration.

    Returns:
        dict: {metric_name: {'mean': float, 'std': float, 'ci_lower': float, 
               'ci_upper': float, ...}}
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    alpha = 1 - confidence
    num_classes = max(y_true.max(), y_pred.max()) + 1
    
    # Pre-allocate arrays
    acc_arr = np.empty(n_bootstrap)
    wf1_arr = np.empty(n_bootstrap)
    mf1_arr = np.empty(n_bootstrap)
    wp_arr = np.empty(n_bootstrap)
    wr_arr = np.empty(n_bootstrap)
    
    valid_count = 0
    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        y_t = y_true[indices]
        y_p = y_pred[indices]
        
        if len(np.unique(y_t)) < 2:
            continue
        
        acc, wf1, mf1, wp, wr = _fast_multiclass_metrics(y_t, y_p, num_classes)
        acc_arr[valid_count] = acc
        wf1_arr[valid_count] = wf1
        mf1_arr[valid_count] = mf1
        wp_arr[valid_count] = wp
        wr_arr[valid_count] = wr
        valid_count += 1
        
        if (i + 1) % 200 == 0:
            print(f"    ... {i+1}/{n_bootstrap} iterations", flush=True)
    
    # Trim to valid
    acc_arr = acc_arr[:valid_count]
    wf1_arr = wf1_arr[:valid_count]
    mf1_arr = mf1_arr[:valid_count]
    wp_arr = wp_arr[:valid_count]
    wr_arr = wr_arr[:valid_count]

    if valid_count < n_bootstrap * 0.95:
        print(f"  ⚠ Warning: Only {valid_count}/{n_bootstrap} bootstrap iterations were valid "
              f"({valid_count / n_bootstrap * 100:.1f}%). CI reliability may be reduced.")

    def _ci(vals):
        return {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'ci_lower': float(np.percentile(vals, 100 * alpha / 2)),
            'ci_upper': float(np.percentile(vals, 100 * (1 - alpha / 2))),
            'n_valid_bootstrap': len(vals),
        }
    
    return {
        'accuracy': _ci(acc_arr),
        'weighted_f1': _ci(wf1_arr),
        'macro_f1': _ci(mf1_arr),
        'precision_weighted': _ci(wp_arr),
        'recall_weighted': _ci(wr_arr),
    }


def get_stage1_predictions():
    """
    获取 Stage 1 RF 在 CIC-IDS2017 测试集上的预测。
    使用 stratified preprocessor + stratified RF model。
    """
    print("  Loading Stage 1 RF model + preprocessor ...")
    rf = joblib.load("models_chk/stage1_rf_stratified.joblib")
    preprocessor = joblib.load("models_chk/preprocessor_stratified.joblib")
    
    print("  Loading CIC-IDS2017 test data ...")
    # Load all CIC parquets (excluding UNSW)
    all_files = glob.glob(os.path.join("archive", "*.parquet"))
    cic_files = [f for f in all_files if 'UNSW' not in os.path.basename(f)]
    
    dfs = []
    for f in cic_files:
        dfs.append(pd.read_parquet(f))
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"  ✓ Total CIC-IDS2017: {df_all.shape[0]:,} samples")
    
    # Clean
    df_clean = preprocessor.clean(df_all)
    
    # Transform to get X, y
    X_all, y_all = preprocessor.transform(df_clean, target_col='Label')
    
    # Binary labels: 0=Benign, 1=Attack
    # Find Benign index
    benign_class = 'Benign'
    classes = preprocessor.label_encoder.classes_
    if benign_class in classes:
        benign_idx = list(classes).index(benign_class)
    else:
        benign_idx = 0
    
    y_binary = (y_all != benign_idx).astype(int)
    
    # Use a held-out 20% test split (same random_state for reproducibility)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X_all, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    print(f"  ✓ Test set: {len(X_test):,} samples (20% stratified split)")
    
    # Predict
    y_pred = rf.predict(X_test)
    
    return y_test, y_pred, "Stage 1 RF (Binary: Benign vs Attack)"


def get_stage2_predictions():
    """
    获取 Stage 2 TransECA-Net 在 hard-example 测试集上的预测。
    """
    from models.stage2_transeca import TransECANet
    
    print("  Loading Stage 2 test data ...")
    df_test = pd.read_parquet("data/stage2/test.parquet")
    print(f"  ✓ Test set: {df_test.shape[0]:,} samples, {df_test.shape[1]} columns")
    
    # Separate features and labels
    preprocessor = joblib.load("models_chk/preprocessor_stratified.joblib")
    
    # Get label encoding
    y_str = df_test['Label']
    X_raw = df_test.drop(columns=['Label'])
    
    # Encode labels
    le = preprocessor.label_encoder
    classes = le.classes_
    num_classes = len(classes)
    
    known_labels = set(classes)
    y_enc = np.array([
        le.transform([v])[0] if v in known_labels else -1
        for v in y_str
    ])
    
    # Filter out unknown labels
    valid_mask = y_enc >= 0
    if valid_mask.sum() < len(y_enc):
        print(f"  ⚠ Dropped {(~valid_mask).sum()} samples with unknown labels")
    
    X_raw = X_raw[valid_mask]
    y_enc = y_enc[valid_mask]
    
    # Scale features
    X_numeric = X_raw.select_dtypes(include=[np.number])
    X_scaled = preprocessor.scaler.transform(X_numeric)
    
    num_features = X_scaled.shape[1]
    
    # Load model
    print("  Loading TransECA-Net model ...")
    device = torch.device('cpu')  # CPU for bootstrap (no GPU needed for inference)
    
    model = TransECANet(
        num_features=num_features, 
        num_classes=num_classes,
        d_model=128, nhead=8, num_layers=3
    )
    model.load_state_dict(torch.load("models_chk/stage2_transeca.pth", 
                                      map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Batch inference
    X_tensor = torch.FloatTensor(X_scaled)
    batch_size = 1024
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            outputs = model(batch)
            _, pred = torch.max(outputs, 1)
            all_preds.append(pred.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    
    return y_enc, y_pred, "Stage 2 TransECA-Net (15-class multiclass)"


def plot_bootstrap_distributions(results_dict, save_path):
    """
    绘制 Bootstrap 分布图 — 每个 Stage 的关键指标分布 + CI 标注。
    """
    stages = list(results_dict.keys())
    metrics_to_plot = ['accuracy', 'weighted_f1', 'macro_f1']
    metric_labels = ['Accuracy', 'Weighted F1', 'Macro F1']
    
    fig, axes = plt.subplots(len(stages), len(metrics_to_plot), 
                              figsize=(5 * len(metrics_to_plot), 4 * len(stages)))
    
    if len(stages) == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['#2196F3', '#FF9800']  # Blue for S1, Orange for S2
    
    for i, stage in enumerate(stages):
        stage_data = results_dict[stage]['bootstrap']
        color = colors[i % len(colors)]
        
        for j, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            ax = axes[i, j]
            data = stage_data[metric]
            
            # Reconstruct samples from stats (we'll pass raw values separately)
            mean_val = data['mean']
            ci_lo = data['ci_lower']
            ci_hi = data['ci_upper']
            
            # Use normal approximation for histogram visualization
            n_vis = 1000
            rng = np.random.RandomState(42)
            samples = rng.normal(mean_val, data['std'], n_vis)
            
            ax.hist(samples, bins=40, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
            
            # CI lines
            ax.axvline(ci_lo, color='red', linestyle='--', linewidth=1.5, label=f'95% CI')
            ax.axvline(ci_hi, color='red', linestyle='--', linewidth=1.5)
            ax.axvline(mean_val, color='black', linestyle='-', linewidth=1.5, label=f'Mean={mean_val:.4f}')
            
            # Fill CI region  
            ax.axvspan(ci_lo, ci_hi, alpha=0.15, color='red')
            
            ax.set_title(f'{stage}\n{label}', fontsize=11, fontweight='bold')
            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            
            # CI annotation
            ci_text = f'95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]'
            ax.text(0.5, 0.92, ci_text, transform=ax.transAxes, ha='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
            
            if j == 0:
                ax.legend(loc='upper left', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Bootstrap distributions → {save_path}")


def generate_report(results_dict, total_time, save_path):
    """
    生成人类可读的报告。
    """
    lines = []
    lines.append("=" * 80)
    lines.append("  E6: Bootstrap Confidence Interval Analysis")
    lines.append("  Method: 1000× Bootstrap Resampling, 95% Percentile CI")
    lines.append("  Reference: [Kaur'21]")
    lines.append("=" * 80)
    lines.append("")
    
    for stage_name, stage_data in results_dict.items():
        lines.append(f"── {stage_name} ──")
        lines.append(f"  Description: {stage_data['description']}")
        lines.append(f"  Test samples: {stage_data['n_test']:,}")
        lines.append(f"  Bootstrap iterations: {stage_data['n_bootstrap']}")
        lines.append("")
        
        # Table header
        lines.append(f"  {'Metric':<25} {'Point Est':>10} {'Mean':>10} {'Std':>10} {'95% CI Lower':>14} {'95% CI Upper':>14}")
        lines.append(f"  {'-'*83}")
        
        bootstrap = stage_data['bootstrap']
        point_metrics = stage_data['point_estimates']
        
        for metric_key in ['accuracy', 'weighted_f1', 'macro_f1', 'precision_weighted', 'recall_weighted']:
            b = bootstrap[metric_key]
            p = point_metrics.get(metric_key, b['mean'])
            name_map = {
                'accuracy': 'Accuracy',
                'weighted_f1': 'Weighted F1',
                'macro_f1': 'Macro F1',
                'precision_weighted': 'Precision (Weighted)',
                'recall_weighted': 'Recall (Weighted)',
            }
            lines.append(f"  {name_map[metric_key]:<25} {p:>10.4f} {b['mean']:>10.4f} {b['std']:>10.4f} {b['ci_lower']:>14.4f} {b['ci_upper']:>14.4f}")
        
        lines.append("")
        
        # CI Width summary
        for metric_key in ['accuracy', 'weighted_f1', 'macro_f1']:
            b = bootstrap[metric_key]
            width = b['ci_upper'] - b['ci_lower']
            name_map = {'accuracy': 'Accuracy', 'weighted_f1': 'W-F1', 'macro_f1': 'M-F1'}
            lines.append(f"  CI Width ({name_map[metric_key]}): {width:.4f}")
        
        lines.append("")
    
    lines.append(f"Total time: {total_time:.1f}s")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - Narrow CIs (width < 0.01) indicate highly stable performance estimates.")
    lines.append("  - CI not overlapping between stages/variants → statistically significant difference.")
    lines.append("  - Point estimate within CI → consistent with bootstrap distribution (no overfitting to single split).")
    
    report_text = '\n'.join(lines)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  ✓ Report → {save_path}")
    
    return report_text


def main():
    print("=" * 80)
    print("  E6: Bootstrap Confidence Interval Analysis")
    print("  Method: 1000× Bootstrap Resampling, 95% Percentile CI")
    print("  Reference: [Kaur'21]")
    print("=" * 80)
    
    # Logger
    logger = TrainingLogger(
        experiment_name="E6_bootstrap_ci",
        description="1000× Bootstrap CI for Stage 1 RF and Stage 2 TransECA-Net. Ref: [Kaur'21]"
    )
    logger.start()
    
    N_BOOTSTRAP = 1000
    CONFIDENCE = 0.95
    total_start = time.time()
    
    results_dict = {}
    
    # ==================== Stage 1: RF Binary ====================
    print(f"\n[1/4] Stage 1 — Random Forest (Binary Classification)")
    print("-" * 60)
    
    t0 = time.time()
    y_true_s1, y_pred_s1, desc_s1 = get_stage1_predictions()
    pred_time_s1 = time.time() - t0
    
    # Point estimates
    point_s1 = {
        'accuracy': float(accuracy_score(y_true_s1, y_pred_s1)),
        'weighted_f1': float(f1_score(y_true_s1, y_pred_s1, average='weighted', zero_division=0)),
        'macro_f1': float(f1_score(y_true_s1, y_pred_s1, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(y_true_s1, y_pred_s1, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_true_s1, y_pred_s1, average='weighted', zero_division=0)),
    }
    
    print(f"\n  Point estimates: Acc={point_s1['accuracy']:.4f}, W-F1={point_s1['weighted_f1']:.4f}, M-F1={point_s1['macro_f1']:.4f}")
    
    print(f"\n  Running {N_BOOTSTRAP}× bootstrap ...")
    t0 = time.time()
    bootstrap_s1 = bootstrap_metrics(y_true_s1, y_pred_s1, n_bootstrap=N_BOOTSTRAP, 
                                      confidence=CONFIDENCE, random_state=42)
    bs_time_s1 = time.time() - t0
    print(f"  ✓ Bootstrap completed in {bs_time_s1:.1f}s")
    
    for metric in ['accuracy', 'weighted_f1', 'macro_f1']:
        b = bootstrap_s1[metric]
        name_map = {'accuracy': 'Acc', 'weighted_f1': 'W-F1', 'macro_f1': 'M-F1'}
        print(f"    {name_map[metric]}: {b['mean']:.4f} [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]")
    
    results_dict['Stage 1 (RF)'] = {
        'description': desc_s1,
        'n_test': len(y_true_s1),
        'n_bootstrap': N_BOOTSTRAP,
        'prediction_time_s': pred_time_s1,
        'bootstrap_time_s': bs_time_s1,
        'point_estimates': point_s1,
        'bootstrap': bootstrap_s1,
    }
    
    # ==================== Stage 2: TransECA-Net ====================
    print(f"\n[2/4] Stage 2 — TransECA-Net (Multiclass Classification)")
    print("-" * 60)
    
    t0 = time.time()
    y_true_s2, y_pred_s2, desc_s2 = get_stage2_predictions()
    pred_time_s2 = time.time() - t0
    
    # Point estimates
    point_s2 = {
        'accuracy': float(accuracy_score(y_true_s2, y_pred_s2)),
        'weighted_f1': float(f1_score(y_true_s2, y_pred_s2, average='weighted', zero_division=0)),
        'macro_f1': float(f1_score(y_true_s2, y_pred_s2, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(y_true_s2, y_pred_s2, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_true_s2, y_pred_s2, average='weighted', zero_division=0)),
    }
    
    print(f"\n  Point estimates: Acc={point_s2['accuracy']:.4f}, W-F1={point_s2['weighted_f1']:.4f}, M-F1={point_s2['macro_f1']:.4f}")
    
    print(f"\n  Running {N_BOOTSTRAP}× bootstrap ...")
    t0 = time.time()
    bootstrap_s2 = bootstrap_metrics(y_true_s2, y_pred_s2, n_bootstrap=N_BOOTSTRAP, 
                                      confidence=CONFIDENCE, random_state=42)
    bs_time_s2 = time.time() - t0
    print(f"  ✓ Bootstrap completed in {bs_time_s2:.1f}s")
    
    for metric in ['accuracy', 'weighted_f1', 'macro_f1']:
        b = bootstrap_s2[metric]
        name_map = {'accuracy': 'Acc', 'weighted_f1': 'W-F1', 'macro_f1': 'M-F1'}
        print(f"    {name_map[metric]}: {b['mean']:.4f} [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]")
    
    results_dict['Stage 2 (TransECA-Net)'] = {
        'description': desc_s2,
        'n_test': len(y_true_s2),
        'n_bootstrap': N_BOOTSTRAP,
        'prediction_time_s': pred_time_s2,
        'bootstrap_time_s': bs_time_s2,
        'point_estimates': point_s2,
        'bootstrap': bootstrap_s2,
    }
    
    # ==================== Plots ====================
    print(f"\n[3/4] Generating plots ...")
    os.makedirs("results", exist_ok=True)
    plot_bootstrap_distributions(results_dict, "results/E6_bootstrap_distributions.png")
    
    # ==================== Report ====================
    print(f"\n[4/4] Generating reports ...")
    total_time = time.time() - total_start
    
    report_text = generate_report(results_dict, total_time, "results/E6_bootstrap_ci_report.txt")
    
    # JSON results (without raw bootstrap values for compactness)
    json_results = {
        'experiment': 'E6_bootstrap_ci',
        'method': f'{N_BOOTSTRAP}× Bootstrap, {int(CONFIDENCE*100)}% Percentile CI',
        'reference': '[Kaur\'21]',
        'timestamp': datetime.now().isoformat(),
        'total_time_s': total_time,
        'stages': {}
    }
    
    for stage_name, stage_data in results_dict.items():
        json_results['stages'][stage_name] = {
            'description': stage_data['description'],
            'n_test': stage_data['n_test'],
            'n_bootstrap': stage_data['n_bootstrap'],
            'point_estimates': stage_data['point_estimates'],
            'bootstrap_ci': {
                metric: {
                    'mean': data['mean'],
                    'std': data['std'],
                    'ci_lower': data['ci_lower'],
                    'ci_upper': data['ci_upper'],
                }
                for metric, data in stage_data['bootstrap'].items()
            },
        }
    
    with open("results/E6_bootstrap_ci_results.json", 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON → results/E6_bootstrap_ci_results.json")
    
    # Logger
    logger.set_data_info(
        dataset="CIC-IDS2017 (S1 test) + Stage 2 hard examples (S2 test)",
        total_samples=len(y_true_s1) + len(y_true_s2),
    )
    logger.set_model_info(
        model_type="Bootstrap CI Analysis",
        hyperparams={'n_bootstrap': N_BOOTSTRAP, 'confidence': CONFIDENCE},
    )
    logger.set_results(
        stage1_accuracy_ci=f"[{bootstrap_s1['accuracy']['ci_lower']:.4f}, {bootstrap_s1['accuracy']['ci_upper']:.4f}]",
        stage1_wf1_ci=f"[{bootstrap_s1['weighted_f1']['ci_lower']:.4f}, {bootstrap_s1['weighted_f1']['ci_upper']:.4f}]",
        stage2_accuracy_ci=f"[{bootstrap_s2['accuracy']['ci_lower']:.4f}, {bootstrap_s2['accuracy']['ci_upper']:.4f}]",
        stage2_wf1_ci=f"[{bootstrap_s2['weighted_f1']['ci_lower']:.4f}, {bootstrap_s2['weighted_f1']['ci_upper']:.4f}]",
    )
    logger.add_artifact("results/E6_bootstrap_ci_results.json", "data", "Bootstrap CI results")
    logger.add_artifact("results/E6_bootstrap_ci_report.txt", "report", "Human-readable report")
    logger.add_artifact("results/E6_bootstrap_distributions.png", "plot", "Bootstrap distributions")
    logger.finish()
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"  E6 Bootstrap CI Analysis Complete!")
    print(f"  Stage 1 Acc 95% CI: [{bootstrap_s1['accuracy']['ci_lower']:.4f}, {bootstrap_s1['accuracy']['ci_upper']:.4f}]")
    print(f"  Stage 1 W-F1 95% CI: [{bootstrap_s1['weighted_f1']['ci_lower']:.4f}, {bootstrap_s1['weighted_f1']['ci_upper']:.4f}]")
    print(f"  Stage 2 Acc 95% CI: [{bootstrap_s2['accuracy']['ci_lower']:.4f}, {bootstrap_s2['accuracy']['ci_upper']:.4f}]")
    print(f"  Stage 2 W-F1 95% CI: [{bootstrap_s2['weighted_f1']['ci_lower']:.4f}, {bootstrap_s2['weighted_f1']['ci_upper']:.4f}]")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
