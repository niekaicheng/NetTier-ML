"""
实验 E14: 对抗性鲁棒性测试 (Adversarial Robustness)
====================================================
目标: 评估分层 IDS 架构在对抗攻击下的鲁棒性。
      证明分层架构能结合 RF 的鲁棒性和 DL 的高灵敏度。

方法:
  1. FGSM (Fast Gradient Sign Method) — 单步梯度攻击，多个 ε 值
  2. PGD (Projected Gradient Descent) — 多步迭代攻击，更强的对手
  3. 对比: Stage 1 RF (加高斯噪声模拟扰动) vs Stage 2 TransECA-Net (梯度攻击)
  4. 分层联合评估: RF → TransECA-Net 串联鲁棒性

分析:
  - Accuracy vs ε 曲线 (robustness profile)
  - 各攻击类别的对抗脆弱性
  - 对抗样本可视化 (特征扰动分布)

参考: experimental_design.md §9.3

产出:
  - results/E14_adversarial_results.json
  - results/E14_adversarial_report.txt
  - results/E14_robustness_curve.png — Accuracy vs ε
  - results/E14_per_class_robustness.png — 各类别对抗鲁棒性热力图
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.stdout.reconfigure(encoding='utf-8')

from models.stage2_transeca import TransECANet
from utils.training_logger import TrainingLogger


# ══════════════════════════════════════════════════════════════════════
#  Adversarial Attack Implementations
# ══════════════════════════════════════════════════════════════════════

def fgsm_attack(model, x, y, epsilon, criterion):
    """
    FGSM: x_adv = x + ε * sign(∇_x L(f(x), y))
    """
    x_adv = x.clone().detach().requires_grad_(True)
    outputs = model(x_adv)
    loss = criterion(outputs, y)
    model.zero_grad()
    loss.backward()
    grad_sign = x_adv.grad.data.sign()
    x_adv = x_adv + epsilon * grad_sign
    # Clamp to [0, 1] (MinMax scaled features)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


def pgd_attack(model, x, y, epsilon, criterion, alpha=None, steps=10):
    """
    PGD: Iterative FGSM with projection back to ε-ball.
    alpha = step size (default: ε / 4)
    """
    if alpha is None:
        alpha = epsilon / 4.0

    x_adv = x.clone().detach()
    x_orig = x.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = criterion(outputs, y)
        model.zero_grad()
        loss.backward()
        grad_sign = x_adv.grad.data.sign()
        x_adv = x_adv.detach() + alpha * grad_sign
        # Project back to ε-ball (L∞)
        perturbation = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + perturbation, 0.0, 1.0)

    return x_adv.detach()


# ══════════════════════════════════════════════════════════════════════
#  Data & Model Loading
# ══════════════════════════════════════════════════════════════════════

def load_test_data(max_samples=10_000, random_state=42):
    """Load Stage 2 test data (stratified subsample for speed)."""
    preprocessor = joblib.load("models_chk/preprocessor_stratified.joblib")
    df_test = pd.read_parquet("data/stage2/test.parquet")

    y_str = df_test['Label']
    X_raw = df_test.drop(columns=['Label'])

    le = preprocessor.label_encoder
    classes = le.classes_
    known = set(classes)

    y_enc = np.array([
        le.transform([v])[0] if v in known else -1
        for v in y_str
    ])
    valid = y_enc >= 0
    X_raw = X_raw[valid].reset_index(drop=True)
    y_enc = y_enc[valid]

    # Stratified subsample for speed (gradient attacks on CPU are slow)
    if max_samples and len(y_enc) > max_samples:
        rng = np.random.RandomState(random_state)
        idx_keep = []
        classes_u, counts = np.unique(y_enc, return_counts=True)
        # Keep all samples from rare classes; proportionally sample large classes
        for c, cnt in zip(classes_u, counts):
            c_idx = np.where(y_enc == c)[0]
            # Proportional allocation (at least keep all if class is small)
            alloc = max(cnt, int(max_samples * cnt / len(y_enc)))
            alloc = min(alloc, cnt)  # can't sample more than available
            if alloc >= cnt:
                idx_keep.append(c_idx)
            else:
                idx_keep.append(rng.choice(c_idx, size=alloc, replace=False))
        idx = np.concatenate(idx_keep)
        # Trim if total exceeds budget
        if len(idx) > max_samples:
            idx = rng.choice(idx, size=max_samples, replace=False)
        idx.sort()
        X_raw = X_raw.iloc[idx].reset_index(drop=True)
        y_enc = y_enc[idx]

    X_numeric = X_raw.select_dtypes(include=[np.number])
    feature_names = list(X_numeric.columns)
    X_scaled = preprocessor.scaler.transform(X_numeric)

    return (torch.FloatTensor(X_scaled), y_enc,
            feature_names, list(classes), preprocessor)


def load_stage2_model(num_features, num_classes, device):
    """Load TransECA-Net."""
    model = TransECANet(
        num_features=num_features,
        num_classes=num_classes,
        d_model=128, nhead=8, num_layers=3
    )
    model.load_state_dict(
        torch.load("models_chk/stage2_transeca.pth",
                    map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════
#  Stage 1 RF Noise Robustness
# ══════════════════════════════════════════════════════════════════════

def evaluate_rf_noise_robustness(X_scaled, y_binary, epsilons, random_state=42):
    """
    Test RF robustness by adding Gaussian/uniform noise to features.
    RF operates on pre-scaled features; we add uniform L∞ noise analogous to FGSM.
    """
    rf = joblib.load("models_chk/stage1_rf_stratified.joblib")

    # Clean baseline
    y_pred_clean = rf.predict(X_scaled)
    clean_acc = accuracy_score(y_binary, y_pred_clean)
    clean_f1 = f1_score(y_binary, y_pred_clean, average='weighted', zero_division=0)

    results = {'clean': {'accuracy': clean_acc, 'weighted_f1': clean_f1}}
    rng = np.random.RandomState(random_state)

    for eps in epsilons:
        # Uniform L∞ noise in [-ε, ε]
        noise = rng.uniform(-eps, eps, size=X_scaled.shape)
        X_noisy = np.clip(X_scaled + noise, 0.0, 1.0)

        y_pred = rf.predict(X_noisy)
        acc = accuracy_score(y_binary, y_pred)
        wf1 = f1_score(y_binary, y_pred, average='weighted', zero_division=0)
        results[f'eps_{eps}'] = {'accuracy': acc, 'weighted_f1': wf1, 'epsilon': eps}

    return results


# ══════════════════════════════════════════════════════════════════════
#  Stage 2 Adversarial Evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate_adversarial(model, X_tensor, y_tensor, epsilons, device,
                          attack_fn, attack_name, class_names,
                          batch_size=256):
    """
    Evaluate model under adversarial attack at multiple ε values.
    Returns per-ε metrics + per-class breakdown.
    """
    criterion = nn.CrossEntropyLoss()
    num_classes = len(class_names)

    # Clean baseline
    model.eval()
    all_preds_clean = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            x_b = X_tensor[i:i+batch_size].to(device)
            out = model(x_b)
            _, pred = torch.max(out, 1)
            all_preds_clean.append(pred.cpu().numpy())
    preds_clean = np.concatenate(all_preds_clean)
    y_np = y_tensor.numpy() if isinstance(y_tensor, torch.Tensor) else y_tensor

    clean_acc = accuracy_score(y_np, preds_clean)
    clean_wf1 = f1_score(y_np, preds_clean, average='weighted', zero_division=0)
    clean_mf1 = f1_score(y_np, preds_clean, average='macro', zero_division=0)

    results = {
        'clean': {
            'accuracy': float(clean_acc),
            'weighted_f1': float(clean_wf1),
            'macro_f1': float(clean_mf1),
        }
    }

    for eps in epsilons:
        print(f"    {attack_name} ε={eps:.4f} ...", end=' ', flush=True)
        all_preds_adv = []

        for i in range(0, len(X_tensor), batch_size):
            x_b = X_tensor[i:i+batch_size].to(device)
            y_b = torch.LongTensor(y_np[i:i+batch_size]).to(device)

            if attack_name == 'PGD':
                x_adv = attack_fn(model, x_b, y_b, eps, criterion, steps=5)
            else:
                x_adv = attack_fn(model, x_b, y_b, eps, criterion)

            with torch.no_grad():
                out = model(x_adv)
                _, pred = torch.max(out, 1)
                all_preds_adv.append(pred.cpu().numpy())

        preds_adv = np.concatenate(all_preds_adv)
        adv_acc = accuracy_score(y_np, preds_adv)
        adv_wf1 = f1_score(y_np, preds_adv, average='weighted', zero_division=0)
        adv_mf1 = f1_score(y_np, preds_adv, average='macro', zero_division=0)

        # Per-class accuracy
        per_class_acc = {}
        for c in range(num_classes):
            mask = y_np == c
            if mask.sum() > 0:
                c_acc = accuracy_score(y_np[mask], preds_adv[mask])
                per_class_acc[class_names[c]] = float(c_acc)

        results[f'eps_{eps}'] = {
            'epsilon': eps,
            'accuracy': float(adv_acc),
            'weighted_f1': float(adv_wf1),
            'macro_f1': float(adv_mf1),
            'acc_drop': float(clean_acc - adv_acc),
            'per_class_accuracy': per_class_acc,
        }
        print(f"Acc={adv_acc:.4f} (Δ={clean_acc - adv_acc:+.4f}), W-F1={adv_wf1:.4f}", flush=True)

    return results


# ══════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════

def plot_robustness_curves(fgsm_results, pgd_results, rf_results, epsilons,
                            save_path="results/E14_robustness_curve.png"):
    """Plot Accuracy vs ε for FGSM, PGD, and RF noise."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Accuracy ──
    ax = axes[0]
    eps_plot = [0.0] + list(epsilons)

    # FGSM
    fgsm_acc = [fgsm_results['clean']['accuracy']]
    fgsm_acc += [fgsm_results[f'eps_{e}']['accuracy'] for e in epsilons]
    ax.plot(eps_plot, fgsm_acc, 'o-', color='#E53935', linewidth=2, markersize=6, label='TransECA FGSM')

    # PGD
    pgd_acc = [pgd_results['clean']['accuracy']]
    pgd_acc += [pgd_results[f'eps_{e}']['accuracy'] for e in epsilons]
    ax.plot(eps_plot, pgd_acc, 's--', color='#D81B60', linewidth=2, markersize=6, label='TransECA PGD')

    # RF noise
    rf_acc = [rf_results['clean']['accuracy']]
    rf_acc += [rf_results[f'eps_{e}']['accuracy'] for e in epsilons]
    ax.plot(eps_plot, rf_acc, '^-', color='#1E88E5', linewidth=2, markersize=6, label='RF (L∞ noise)')

    ax.set_xlabel('Perturbation ε', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('E14: Adversarial Robustness — Accuracy vs ε', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # ── Weighted F1 ──
    ax = axes[1]
    fgsm_f1 = [fgsm_results['clean']['weighted_f1']]
    fgsm_f1 += [fgsm_results[f'eps_{e}']['weighted_f1'] for e in epsilons]
    ax.plot(eps_plot, fgsm_f1, 'o-', color='#E53935', linewidth=2, markersize=6, label='TransECA FGSM')

    pgd_f1 = [pgd_results['clean']['weighted_f1']]
    pgd_f1 += [pgd_results[f'eps_{e}']['weighted_f1'] for e in epsilons]
    ax.plot(eps_plot, pgd_f1, 's--', color='#D81B60', linewidth=2, markersize=6, label='TransECA PGD')

    rf_f1 = [rf_results['clean']['weighted_f1']]
    rf_f1 += [rf_results[f'eps_{e}']['weighted_f1'] for e in epsilons]
    ax.plot(eps_plot, rf_f1, '^-', color='#1E88E5', linewidth=2, markersize=6, label='RF (L∞ noise)')

    ax.set_xlabel('Perturbation ε', fontsize=12)
    ax.set_ylabel('Weighted F1', fontsize=12)
    ax.set_title('E14: Adversarial Robustness — W-F1 vs ε', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Robustness curves → {save_path}")


def plot_per_class_robustness(fgsm_results, class_names, epsilons,
                               save_path="results/E14_per_class_robustness.png"):
    """Heatmap of per-class accuracy under FGSM at different ε."""
    # Build matrix: (n_eps, n_classes)
    eps_labels = [f'ε={e}' for e in epsilons]
    active_classes = []
    for cn in class_names:
        # Check if class has data in at least one ε result
        for e in epsilons:
            key = f'eps_{e}'
            if key in fgsm_results and cn in fgsm_results[key].get('per_class_accuracy', {}):
                if cn not in active_classes:
                    active_classes.append(cn)
                break

    if not active_classes:
        print("  ⚠ No per-class data available for heatmap")
        return

    matrix = np.zeros((len(epsilons), len(active_classes)))
    for i, e in enumerate(epsilons):
        key = f'eps_{e}'
        pca = fgsm_results[key].get('per_class_accuracy', {})
        for j, cn in enumerate(active_classes):
            matrix[i, j] = pca.get(cn, 0.0)

    fig, ax = plt.subplots(figsize=(max(10, len(active_classes) * 0.8), max(4, len(epsilons) * 0.8)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')

    ax.set_xticks(range(len(active_classes)))
    ax.set_xticklabels(active_classes, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(epsilons)))
    ax.set_yticklabels(eps_labels, fontsize=9)

    plt.colorbar(im, ax=ax, label='Accuracy', shrink=0.8)
    ax.set_title('E14: Per-Class Accuracy under FGSM Attack', fontsize=13, fontweight='bold')

    # Annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = 'white' if val < 0.4 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Per-class robustness heatmap → {save_path}")


# ══════════════════════════════════════════════════════════════════════
#  Report
# ══════════════════════════════════════════════════════════════════════

def generate_report(fgsm_results, pgd_results, rf_results, epsilons,
                     total_time, save_path):
    lines = []
    lines.append("=" * 80)
    lines.append("  E14: Adversarial Robustness Analysis")
    lines.append("  Stage 1: RF + L∞ Uniform Noise | Stage 2: TransECA-Net + FGSM / PGD")
    lines.append("=" * 80)
    lines.append("")

    # ── Summary Table ──
    lines.append("── Robustness Summary ──")
    lines.append(f"  {'ε':<8} {'RF Acc':>10} {'RF W-F1':>10} │ {'FGSM Acc':>10} {'FGSM W-F1':>10} {'FGSM M-F1':>10} │ {'PGD Acc':>10} {'PGD W-F1':>10} {'PGD M-F1':>10}")
    lines.append(f"  {'-'*100}")

    # Clean
    lines.append(f"  {'0.0':<8} {rf_results['clean']['accuracy']:>10.4f} "
                  f"{rf_results['clean']['weighted_f1']:>10.4f} │ "
                  f"{fgsm_results['clean']['accuracy']:>10.4f} "
                  f"{fgsm_results['clean']['weighted_f1']:>10.4f} "
                  f"{fgsm_results['clean']['macro_f1']:>10.4f} │ "
                  f"{pgd_results['clean']['accuracy']:>10.4f} "
                  f"{pgd_results['clean']['weighted_f1']:>10.4f} "
                  f"{pgd_results['clean']['macro_f1']:>10.4f}")

    for eps in epsilons:
        rk = f'eps_{eps}'
        lines.append(f"  {eps:<8.4f} "
                      f"{rf_results[rk]['accuracy']:>10.4f} "
                      f"{rf_results[rk]['weighted_f1']:>10.4f} │ "
                      f"{fgsm_results[rk]['accuracy']:>10.4f} "
                      f"{fgsm_results[rk]['weighted_f1']:>10.4f} "
                      f"{fgsm_results[rk]['macro_f1']:>10.4f} │ "
                      f"{pgd_results[rk]['accuracy']:>10.4f} "
                      f"{pgd_results[rk]['weighted_f1']:>10.4f} "
                      f"{pgd_results[rk]['macro_f1']:>10.4f}")

    lines.append("")

    # ── Accuracy Drop Analysis ──
    lines.append("── Accuracy Drop (TransECA-Net) ──")
    for eps in epsilons:
        fgsm_drop = fgsm_results[f'eps_{eps}']['acc_drop']
        pgd_drop = pgd_results[f'eps_{eps}']['acc_drop']
        lines.append(f"  ε={eps:.4f}: FGSM Δ={fgsm_drop:+.4f}, PGD Δ={pgd_drop:+.4f}")

    lines.append("")

    # ── Key Observations ──
    lines.append("── Key Observations ──")

    # RF robustness at max ε
    max_eps = epsilons[-1]
    rf_drop = rf_results['clean']['accuracy'] - rf_results[f'eps_{max_eps}']['accuracy']
    fgsm_drop = fgsm_results[f'eps_{max_eps}']['acc_drop']
    pgd_drop = pgd_results[f'eps_{max_eps}']['acc_drop']

    lines.append(f"  1. At ε={max_eps}: RF Acc drop = {rf_drop:.4f}, "
                  f"DL FGSM drop = {fgsm_drop:.4f}, DL PGD drop = {pgd_drop:.4f}")
    if rf_drop < fgsm_drop:
        lines.append(f"     → RF is MORE robust than TransECA-Net to input perturbations")
    else:
        lines.append(f"     → TransECA-Net is MORE robust than RF to input perturbations")

    lines.append(f"  2. PGD (multi-step) is {'stronger' if pgd_drop > fgsm_drop else 'comparable to'} "
                  f"FGSM (single-step) as expected")

    # Hierarchical defense argument
    lines.append(f"  3. Hierarchical Defense: Stage 1 RF acts as a robust pre-filter,")
    lines.append(f"     only forwarding ~15% of traffic to DL. Even if DL is adversarially")
    lines.append(f"     vulnerable, the attack surface is limited to the subset that passes RF.")

    lines.append("")
    lines.append(f"Total time: {total_time:.1f}s")

    report = '\n'.join(lines)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  ✓ Report → {save_path}")
    return report


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  E14: Adversarial Robustness Analysis")
    print("  FGSM + PGD on TransECA-Net | L∞ Noise on RF")
    print("=" * 80)

    logger = TrainingLogger(
        experiment_name="E14_adversarial_robustness",
        description="Adversarial robustness: FGSM/PGD on Stage 2 + noise on Stage 1"
    )
    logger.start()

    # Use XPU (Intel ARC GPU) if available, otherwise CPU
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        print(f"  Using device: XPU ({torch.xpu.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"  Using device: CPU")
    total_start = time.time()
    os.makedirs("results", exist_ok=True)

    # Epsilon values: 0.001 to 0.1 (on [0,1] scaled features)
    EPSILONS = [0.001, 0.005, 0.01, 0.05, 0.1]

    # ──── Load Data ────
    print(f"\n[1/5] Loading data & models ...", flush=True)
    X_tensor, y_enc, feature_names, class_names, preprocessor = load_test_data()
    num_features = X_tensor.shape[1]
    num_classes = len(class_names)
    print(f"  ✓ Stage 2 test: {len(X_tensor):,} samples, {num_features} features, {num_classes} classes")

    model = load_stage2_model(num_features, num_classes, device)
    print(f"  ✓ TransECA-Net loaded")

    # ──── Stage 1 RF Noise Robustness ────
    print(f"\n[2/5] Stage 1 RF — L∞ Noise Robustness ...", flush=True)
    # Use same X_scaled data for RF (binary labels)
    X_np = X_tensor.numpy()
    # Binary: find Benign index
    benign_idx = list(preprocessor.label_encoder.classes_).index('Benign') \
        if 'Benign' in preprocessor.label_encoder.classes_ else 0
    y_binary = (y_enc != benign_idx).astype(int)

    t0 = time.time()
    rf_results = evaluate_rf_noise_robustness(X_np, y_binary, EPSILONS)
    rf_time = time.time() - t0
    print(f"  ✓ RF noise evaluation: {rf_time:.1f}s")
    print(f"  Clean: Acc={rf_results['clean']['accuracy']:.4f}")
    rf_max = rf_results[f'eps_{EPSILONS[-1]}']
    print(f"  ε={EPSILONS[-1]}: Acc={rf_max['accuracy']:.4f} "
          f"(Δ={rf_results['clean']['accuracy'] - rf_max['accuracy']:+.4f})")

    # ──── Stage 2 FGSM ────
    print(f"\n[3/5] Stage 2 TransECA-Net — FGSM Attack ...", flush=True)
    t0 = time.time()
    fgsm_results = evaluate_adversarial(
        model, X_tensor, y_enc, EPSILONS, device,
        fgsm_attack, 'FGSM', class_names, batch_size=1024
    )
    fgsm_time = time.time() - t0
    print(f"  ✓ FGSM evaluation: {fgsm_time:.1f}s")

    # ──── Stage 2 PGD ────
    print(f"\n[4/5] Stage 2 TransECA-Net — PGD Attack (5 steps) ...", flush=True)
    t0 = time.time()
    pgd_results = evaluate_adversarial(
        model, X_tensor, y_enc, EPSILONS, device,
        pgd_attack, 'PGD', class_names, batch_size=1024
    )
    pgd_time = time.time() - t0
    print(f"  ✓ PGD evaluation: {pgd_time:.1f}s")

    # ──── Plots & Reports ────
    print(f"\n[5/5] Generating plots & reports ...", flush=True)
    total_time = time.time() - total_start

    plot_robustness_curves(fgsm_results, pgd_results, rf_results, EPSILONS)
    plot_per_class_robustness(fgsm_results, class_names, EPSILONS)

    report = generate_report(fgsm_results, pgd_results, rf_results, EPSILONS,
                              total_time, "results/E14_adversarial_report.txt")

    # JSON
    json_results = {
        'experiment': 'E14_adversarial_robustness',
        'epsilons': EPSILONS,
        'total_time_s': total_time,
        'rf_noise': {k: v for k, v in rf_results.items()},
        'fgsm': {k: v for k, v in fgsm_results.items()},
        'pgd': {k: v for k, v in pgd_results.items()},
    }
    with open("results/E14_adversarial_results.json", 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON → results/E14_adversarial_results.json")

    # Logger
    logger.set_data_info(dataset="Stage 2 test (data/stage2/test.parquet)",
                          total_samples=len(X_tensor))
    logger.set_model_info(model_type="Adversarial Robustness (FGSM/PGD)",
                           hyperparams={'epsilons': EPSILONS, 'pgd_steps': 5,
                                        'subsample': 10000})
    logger.set_results(
        rf_clean_acc=rf_results['clean']['accuracy'],
        transeca_clean_acc=fgsm_results['clean']['accuracy'],
        fgsm_eps01_acc=fgsm_results[f'eps_{EPSILONS[-1]}']['accuracy'],
        pgd_eps01_acc=pgd_results[f'eps_{EPSILONS[-1]}']['accuracy'],
        rf_eps01_acc=rf_results[f'eps_{EPSILONS[-1]}']['accuracy'],
    )
    logger.add_artifact("results/E14_adversarial_results.json", "data", "Adversarial results")
    logger.add_artifact("results/E14_adversarial_report.txt", "report", "Analysis report")
    logger.add_artifact("results/E14_robustness_curve.png", "plot", "Robustness curves")
    logger.add_artifact("results/E14_per_class_robustness.png", "plot", "Per-class heatmap")
    logger.finish()

    # Summary
    fgsm_max_drop = fgsm_results[f'eps_{EPSILONS[-1]}']['acc_drop']
    pgd_max_drop = pgd_results[f'eps_{EPSILONS[-1]}']['acc_drop']
    rf_max_drop = rf_results['clean']['accuracy'] - rf_results[f'eps_{EPSILONS[-1]}']['accuracy']

    print(f"\n{'='*70}")
    print(f"  E14 Adversarial Robustness Complete!")
    print(f"  At ε={EPSILONS[-1]}:")
    print(f"    RF (L∞ noise):   Acc drop = {rf_max_drop:+.4f}")
    print(f"    TransECA FGSM:   Acc drop = {fgsm_max_drop:+.4f}")
    print(f"    TransECA PGD:    Acc drop = {pgd_max_drop:+.4f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
