"""
实验 E10: Stage 2 TransECA-Net 可解释性分析
=============================================
目标: 可视化 TransECA-Net 的决策依据，证明模型关注的是网络安全领域有意义的特征。

方法:
  1. Integrated Gradients (IG) — 精确的输入级特征归因 (captum)
  2. Attention Rollout — 提取 Transformer 自注意力权重，分析特征间关系
  3. ECA Channel Attention — 可视化通道注意力权重

分析:
  - 全局 IG 归因: 对各攻击类别计算平均 IG 值，得到 Top 特征
  - Per-class IG: 不同攻击关注不同特征
  - Attention Heatmap: Transformer 特征交互关系
  - IG vs SHAP (E2) 对比: TransECA-Net vs RF 关注的特征是否一致

产出:
  - results/E10_ig_global_importance.png — 全局 IG 特征重要性
  - results/E10_ig_per_class.png — 各类别 IG 归因热力图
  - results/E10_attention_heatmap.png — Transformer Attention Rollout
  - results/E10_eca_channel_weights.png — ECA 通道注意力
  - results/E10_interpretability_report.txt — 分析报告
  - results/E10_interpretability_results.json — 结构化结果

参考: experimental_design.md §8
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
import matplotlib.colors as mcolors

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.stdout.reconfigure(encoding='utf-8')

from models.stage2_transeca import TransECANet, ECAModule
from utils.training_logger import TrainingLogger


# ──────────────────────────────────────────────────────────────────────
#  Attention Hook — 提取 Transformer Self-Attention 权重
# ──────────────────────────────────────────────────────────────────────

class AttentionHookManager:
    """
    Register forward hooks on MultiheadAttention layers to capture attention weights.
    """
    def __init__(self, model):
        self.attention_weights = []
        self.hooks = []
        self._register(model)

    def _register(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)

    def _hook_fn(self, module, input, output):
        # MultiheadAttention returns (attn_output, attn_output_weights)
        # attn_output_weights: (B, L, L) when average_attn_weights=True (default)
        if len(output) >= 2 and output[1] is not None:
            self.attention_weights.append(output[1].detach().cpu())

    def clear(self):
        self.attention_weights = []

    def remove(self):
        for h in self.hooks:
            h.remove()


class ECAHookManager:
    """Capture ECA sigmoid output to see channel attention weights."""
    def __init__(self, model):
        self.eca_weights = []
        self.hooks = []
        self._register(model)

    def _register(self, model):
        for name, module in model.named_modules():
            if isinstance(module, ECAModule):
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)

    def _hook_fn(self, module, input, output):
        # ECA: output = x * attention_weights
        # We need the attention weights (sigmoid output), not the final output.
        # Re-compute: attention = output / input  (element-wise), but safer to hook sigmoid.
        # Instead, let's compute it: attn = output / (input[0] + 1e-10)
        x_in = input[0]  # (B, C, L)
        with torch.no_grad():
            # ECA attention is channel-wise (same across L): compute mean ratio
            # Or we can access internal: module.avg_pool -> module.conv -> module.sigmoid
            y = module.avg_pool(x_in)  # (B, C, 1)
            y = module.conv(y.transpose(-1, -2)).transpose(-1, -2)
            y = module.sigmoid(y)  # (B, C, 1) — channel attention weights
            self.eca_weights.append(y.squeeze(-1).detach().cpu())  # (B, C)

    def clear(self):
        self.eca_weights = []

    def remove(self):
        for h in self.hooks:
            h.remove()


# ──────────────────────────────────────────────────────────────────────
#  Model wrapper to enable attention weight output for Transformer
# ──────────────────────────────────────────────────────────────────────

def enable_attention_output(model):
    """
    Patch TransformerEncoderLayer to output attention weights.
    PyTorch's TransformerEncoderLayer doesn't output attn weights by default.
    We need to set need_weights=True on the self_attn call.
    """
    for module in model.modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            # Monkey-patch the forward to pass need_weights=True
            original_forward = module.forward

            def make_patched(orig_fwd, mod):
                def patched_forward(src, src_mask=None, src_key_padding_mask=None, **kwargs):
                    # Use the internal _sa_block with need_weights
                    # For simplicity, we'll override the self_attn call
                    x = src
                    # Self Attention block
                    attn_out, attn_weights = mod.self_attn(
                        x, x, x,
                        attn_mask=src_mask,
                        key_padding_mask=src_key_padding_mask,
                        need_weights=True,
                        average_attn_weights=True
                    )
                    x = mod.norm1(x + mod.dropout1(attn_out))
                    # Feedforward block
                    ff_out = mod.linear2(mod.dropout(mod.activation(mod.linear1(x))))
                    x = mod.norm2(x + mod.dropout2(ff_out))
                    return x
                return patched_forward

            module.forward = make_patched(original_forward, module)


# ──────────────────────────────────────────────────────────────────────
#  Data Loading
# ──────────────────────────────────────────────────────────────────────

def load_test_data():
    """Load Stage 2 test data, return X_tensor, y_enc, feature_names, class_names."""
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

    X_numeric = X_raw.select_dtypes(include=[np.number])
    feature_names = list(X_numeric.columns)
    X_scaled = preprocessor.scaler.transform(X_numeric)

    return (torch.FloatTensor(X_scaled), y_enc,
            feature_names, list(classes))


def load_model(num_features, num_classes, device):
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


# ──────────────────────────────────────────────────────────────────────
#  1. Integrated Gradients
# ──────────────────────────────────────────────────────────────────────

def compute_integrated_gradients(model, X_tensor, y_enc, class_names,
                                  device, n_samples=500, n_steps=50):
    """
    Compute IG attributions per class using captum.
    Returns ig_global (76,), ig_per_class dict {class_idx: (76,)}.
    """
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(model)
    num_classes = len(class_names)

    # Sample n_samples from test set (stratified)
    rng = np.random.RandomState(42)
    sample_idx = []
    unique_classes = np.unique(y_enc)
    per_class = max(1, n_samples // len(unique_classes))
    for c in unique_classes:
        c_idx = np.where(y_enc == c)[0]
        chosen = rng.choice(c_idx, size=min(per_class, len(c_idx)), replace=False)
        sample_idx.extend(chosen)
    sample_idx = np.array(sample_idx)[:n_samples]

    X_sample = X_tensor[sample_idx].to(device)
    y_sample = y_enc[sample_idx]

    # Baseline = zero vector (represents "absence of information")
    baseline = torch.zeros_like(X_sample[0:1]).to(device)

    # Compute IG for each sample targeting its true class
    all_attr = torch.zeros(len(sample_idx), X_tensor.shape[1])

    batch_size = 64
    for start in range(0, len(sample_idx), batch_size):
        end = min(start + batch_size, len(sample_idx))
        x_batch = X_sample[start:end]
        y_batch = y_sample[start:end]

        # IG per sample
        for i in range(len(x_batch)):
            xi = x_batch[i:i+1]
            target_class = int(y_batch[i])
            bl = baseline.expand_as(xi)

            attr = ig.attribute(xi, baselines=bl, target=target_class,
                                n_steps=n_steps, internal_batch_size=64)
            all_attr[start + i] = attr.squeeze(0).detach().cpu()

        if (end) % 100 == 0 or end == len(sample_idx):
            print(f"    ... IG computed for {end}/{len(sample_idx)} samples", flush=True)

    # Global importance: mean |attribution|
    ig_global = all_attr.abs().mean(dim=0).numpy()  # (76,)

    # Per-class importance
    ig_per_class = {}
    for c in unique_classes:
        mask = y_sample == c
        if mask.sum() > 0:
            ig_per_class[int(c)] = all_attr[mask].abs().mean(dim=0).numpy()

    return ig_global, ig_per_class, sample_idx


# ──────────────────────────────────────────────────────────────────────
#  2. Attention Rollout
# ──────────────────────────────────────────────────────────────────────

def compute_attention_rollout(model, X_tensor, y_enc, device, n_samples=200):
    """
    Extract and aggregate attention weights across all Transformer layers.
    Attention Rollout: multiply attention matrices layer by layer.
    Returns: rollout (76, 76) averaged over samples.
    """
    # Enable attention weight output
    enable_attention_output(model)

    attn_hook = AttentionHookManager(model)

    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_tensor), size=min(n_samples, len(X_tensor)), replace=False)

    rollout_accum = None
    count = 0

    batch_size = 64
    for start in range(0, len(idx), batch_size):
        end = min(start + batch_size, len(idx))
        x_batch = X_tensor[idx[start:end]].to(device)

        attn_hook.clear()
        with torch.no_grad():
            _ = model(x_batch)

        # attn_hook.attention_weights: list of (B, L, L), one per layer
        if not attn_hook.attention_weights:
            print("  ⚠ No attention weights captured. Skipping rollout.")
            attn_hook.remove()
            return None

        B = x_batch.shape[0]
        num_layers = len(attn_hook.attention_weights)
        seq_len = attn_hook.attention_weights[0].shape[-1]

        for b in range(B):
            # Rollout: start with identity, multiply through layers
            rollout = torch.eye(seq_len)
            for layer_idx in range(num_layers):
                attn = attn_hook.attention_weights[layer_idx][b]  # (L, L)
                # Add identity (residual connections)
                attn = 0.5 * attn + 0.5 * torch.eye(seq_len)
                # Normalize rows
                attn = attn / attn.sum(dim=-1, keepdim=True)
                rollout = torch.matmul(attn, rollout)

            if rollout_accum is None:
                rollout_accum = rollout.numpy()
            else:
                rollout_accum += rollout.numpy()
            count += 1

    attn_hook.remove()

    if count > 0:
        rollout_accum /= count

    return rollout_accum  # (76, 76)


# ──────────────────────────────────────────────────────────────────────
#  3. ECA Channel Attention
# ──────────────────────────────────────────────────────────────────────

def compute_eca_attention(model, X_tensor, device, n_samples=500):
    """
    Extract ECA channel attention weights averaged over samples.
    Returns: eca_weights (d_model,)
    """
    eca_hook = ECAHookManager(model)

    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_tensor), size=min(n_samples, len(X_tensor)), replace=False)

    eca_accum = None
    count = 0

    batch_size = 128
    for start in range(0, len(idx), batch_size):
        end = min(start + batch_size, len(idx))
        x_batch = X_tensor[idx[start:end]].to(device)

        eca_hook.clear()
        with torch.no_grad():
            _ = model(x_batch)

        if eca_hook.eca_weights:
            weights = eca_hook.eca_weights[0].numpy()  # (B, C)
            if eca_accum is None:
                eca_accum = weights.sum(axis=0)
            else:
                eca_accum += weights.sum(axis=0)
            count += weights.shape[0]

    eca_hook.remove()

    if count > 0:
        return eca_accum / count  # (d_model,)
    return None


# ──────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_ig_global(ig_global, feature_names, top_k=20, save_path="results/E10_ig_global_importance.png"):
    """Bar chart of top-K features by Integrated Gradients attribution."""
    sorted_idx = np.argsort(ig_global)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(10, 7))
    names = [feature_names[i] for i in sorted_idx]
    values = ig_global[sorted_idx]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_k))
    bars = ax.barh(range(top_k), values[::-1], color=colors[::-1], edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel('Mean |Integrated Gradients Attribution|', fontsize=11)
    ax.set_title('E10: Top-20 Features by Integrated Gradients\n(Stage 2 TransECA-Net)', fontsize=13, fontweight='bold')
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ IG global importance → {save_path}")


def plot_ig_per_class(ig_per_class, feature_names, class_names, top_k=15,
                       save_path="results/E10_ig_per_class.png"):
    """Heatmap of per-class IG attributions for top features."""
    # Determine top features globally
    all_class_data = np.array([ig_per_class[c] for c in sorted(ig_per_class.keys())])
    global_mean = all_class_data.mean(axis=0)
    top_feat_idx = np.argsort(global_mean)[::-1][:top_k]

    # Filter to classes with enough support
    active_classes = sorted(ig_per_class.keys())
    class_labels = [class_names[c] if c < len(class_names) else f"Class {c}" for c in active_classes]

    matrix = np.array([ig_per_class[c][top_feat_idx] for c in active_classes])  # (n_classes, top_k)
    feat_labels = [feature_names[i] for i in top_feat_idx]

    fig, ax = plt.subplots(figsize=(12, max(6, len(active_classes) * 0.5)))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    ax.set_xticks(range(top_k))
    ax.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(active_classes)))
    ax.set_yticklabels(class_labels, fontsize=9)

    plt.colorbar(im, ax=ax, label='Mean |IG Attribution|', shrink=0.8)
    ax.set_title('E10: Per-Class Integrated Gradients Attribution\n(Top-15 Features)', fontsize=13, fontweight='bold')

    # Add value annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = 'white' if val > matrix.max() * 0.7 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=6, color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ IG per-class heatmap → {save_path}")


def plot_attention_heatmap(rollout, feature_names, top_k=20,
                            save_path="results/E10_attention_heatmap.png"):
    """Heatmap of Attention Rollout (top-K features × top-K features)."""
    if rollout is None:
        print("  ⚠ Skipping attention heatmap (no rollout data)")
        return

    # Select top-K features by attention received (column sum)
    col_sum = rollout.sum(axis=0)
    top_idx = np.argsort(col_sum)[::-1][:top_k]

    sub_rollout = rollout[np.ix_(top_idx, top_idx)]
    feat_labels = [feature_names[i] if i < len(feature_names) else f"F{i}" for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sub_rollout, cmap='Blues', interpolation='nearest')

    ax.set_xticks(range(top_k))
    ax.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(feat_labels, fontsize=8)

    plt.colorbar(im, ax=ax, label='Attention Weight', shrink=0.8)
    ax.set_title('E10: Transformer Attention Rollout\n(Top-20 Features by Attention Received)',
                  fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Attention heatmap → {save_path}")

    return col_sum  # full attention-received scores


def plot_eca_weights(eca_weights, save_path="results/E10_eca_channel_weights.png"):
    """Bar chart of ECA channel attention weights."""
    if eca_weights is None:
        print("  ⚠ Skipping ECA plot (no data)")
        return

    d_model = len(eca_weights)
    fig, ax = plt.subplots(figsize=(12, 4))

    colors = plt.cm.coolwarm(eca_weights / eca_weights.max())
    ax.bar(range(d_model), eca_weights, color=colors, edgecolor='none', width=1.0)
    ax.set_xlabel('Channel Index', fontsize=11)
    ax.set_ylabel('ECA Attention Weight', fontsize=11)
    ax.set_title(f'E10: ECA Channel Attention Weights (d_model={d_model})', fontsize=13, fontweight='bold')
    ax.axhline(y=eca_weights.mean(), color='red', linestyle='--', linewidth=1, label=f'Mean={eca_weights.mean():.3f}')
    ax.legend()

    # Annotate top-5 channels
    top5 = np.argsort(eca_weights)[::-1][:5]
    for ch in top5:
        ax.annotate(f'Ch{ch}\n{eca_weights[ch]:.3f}',
                     xy=(ch, eca_weights[ch]),
                     xytext=(ch, eca_weights[ch] + 0.01),
                     ha='center', fontsize=7, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ ECA channel weights → {save_path}")


# ──────────────────────────────────────────────────────────────────────
#  Report
# ──────────────────────────────────────────────────────────────────────

def generate_report(ig_global, ig_per_class, rollout, eca_weights,
                     feature_names, class_names, total_time, save_path):
    """Generate human-readable report."""
    lines = []
    lines.append("=" * 80)
    lines.append("  E10: TransECA-Net Interpretability Analysis")
    lines.append("  Model: Stage 2 TransECA-Net (d=128, h=8, L=3)")
    lines.append("  Methods: Integrated Gradients + Attention Rollout + ECA Weights")
    lines.append("=" * 80)
    lines.append("")

    # IG Global
    lines.append("── 1. Integrated Gradients (Global Feature Attribution) ──")
    sorted_idx = np.argsort(ig_global)[::-1]
    lines.append(f"  {'Rank':<6} {'Feature':<35} {'Mean |IG|':>12}")
    lines.append(f"  {'-'*53}")
    for rank, idx in enumerate(sorted_idx[:20], 1):
        lines.append(f"  {rank:<6} {feature_names[idx]:<35} {ig_global[idx]:>12.6f}")
    lines.append("")

    # IG Per-class highlights
    lines.append("── 2. Per-Class Top Feature (by IG) ──")
    for c in sorted(ig_per_class.keys()):
        c_ig = ig_per_class[c]
        top_idx = np.argmax(c_ig)
        cname = class_names[c] if c < len(class_names) else f"Class {c}"
        lines.append(f"  {cname:<30} → {feature_names[top_idx]:<30} (IG={c_ig[top_idx]:.6f})")
    lines.append("")

    # Attention
    if rollout is not None:
        lines.append("── 3. Attention Rollout (Top Features by Attention Received) ──")
        col_sum = rollout.sum(axis=0)
        attn_sorted = np.argsort(col_sum)[::-1]
        for rank, idx in enumerate(attn_sorted[:15], 1):
            fname = feature_names[idx] if idx < len(feature_names) else f"F{idx}"
            lines.append(f"  {rank:<6} {fname:<35} {col_sum[idx]:>12.6f}")
        lines.append("")

    # ECA
    if eca_weights is not None:
        lines.append("── 4. ECA Channel Attention ──")
        lines.append(f"  Channels: {len(eca_weights)}")
        lines.append(f"  Mean weight: {eca_weights.mean():.4f}")
        lines.append(f"  Std: {eca_weights.std():.4f}")
        lines.append(f"  Max channel: {np.argmax(eca_weights)} (weight={eca_weights.max():.4f})")
        lines.append(f"  Min channel: {np.argmin(eca_weights)} (weight={eca_weights.min():.4f})")
        selectivity = eca_weights.std() / eca_weights.mean() if eca_weights.mean() > 0 else 0
        lines.append(f"  Selectivity (CV): {selectivity:.4f}")
        lines.append("")

    # Cross-method comparison
    lines.append("── 5. Cross-Method Feature Ranking Comparison ──")
    ig_rank = np.argsort(np.argsort(ig_global)[::-1]) + 1  # rank array
    if rollout is not None:
        col_sum = rollout.sum(axis=0)
        attn_rank = np.argsort(np.argsort(col_sum)[::-1]) + 1
        # Spearman
        from scipy.stats import spearmanr
        rho, pval = spearmanr(ig_rank, attn_rank)
        lines.append(f"  IG vs Attention Rollout: Spearman ρ = {rho:.4f} (p={pval:.2e})")
    lines.append("")

    lines.append(f"Total time: {total_time:.1f}s")

    report = '\n'.join(lines)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  ✓ Report → {save_path}")
    return report


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  E10: TransECA-Net Interpretability Analysis")
    print("  Methods: Integrated Gradients + Attention Rollout + ECA Channel Weights")
    print("=" * 80)

    logger = TrainingLogger(
        experiment_name="E10_interpretability",
        description="TransECA-Net interpretability: IG, Attention Rollout, ECA weights."
    )
    logger.start()

    device = torch.device('cpu')  # CPU for interpretability (gradient-based)
    total_start = time.time()

    os.makedirs("results", exist_ok=True)

    # ──── Load data & model ────
    print(f"\n[1/5] Loading data & model ...", flush=True)
    X_tensor, y_enc, feature_names, class_names = load_test_data()
    num_features = X_tensor.shape[1]
    num_classes = len(class_names)
    print(f"  ✓ Data: {len(X_tensor):,} samples, {num_features} features, {num_classes} classes")

    model = load_model(num_features, num_classes, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model: TransECA-Net ({total_params:,} params)")

    # ──── 1. Integrated Gradients ────
    print(f"\n[2/5] Computing Integrated Gradients (500 samples, 50 steps) ...", flush=True)
    t0 = time.time()
    ig_global, ig_per_class, ig_sample_idx = compute_integrated_gradients(
        model, X_tensor, y_enc, class_names, device, n_samples=500, n_steps=50
    )
    ig_time = time.time() - t0
    print(f"  ✓ IG completed in {ig_time:.1f}s")

    top5_idx = np.argsort(ig_global)[::-1][:5]
    print(f"  Top-5 IG features: {', '.join([feature_names[i] for i in top5_idx])}")

    # ──── 2. Attention Rollout ────
    print(f"\n[3/5] Computing Attention Rollout (200 samples) ...", flush=True)
    t0 = time.time()
    rollout = compute_attention_rollout(model, X_tensor, y_enc, device, n_samples=200)
    attn_time = time.time() - t0
    print(f"  ✓ Attention rollout completed in {attn_time:.1f}s")

    if rollout is not None:
        col_sum = rollout.sum(axis=0)
        top5_attn = np.argsort(col_sum)[::-1][:5]
        print(f"  Top-5 Attention features: {', '.join([feature_names[i] for i in top5_attn])}")

    # ──── 3. ECA Channel Attention ────
    print(f"\n[4/5] Computing ECA Channel Attention (500 samples) ...", flush=True)

    # Need a fresh model without patched attention for ECA hooks
    model_eca = load_model(num_features, num_classes, device)
    t0 = time.time()
    eca_weights = compute_eca_attention(model_eca, X_tensor, device, n_samples=500)
    eca_time = time.time() - t0
    print(f"  ✓ ECA weights completed in {eca_time:.1f}s")
    if eca_weights is not None:
        print(f"  ECA: mean={eca_weights.mean():.4f}, std={eca_weights.std():.4f}, "
              f"top_ch={np.argmax(eca_weights)}")

    # ──── 4. Plots & Reports ────
    print(f"\n[5/5] Generating plots & reports ...", flush=True)

    plot_ig_global(ig_global, feature_names)
    plot_ig_per_class(ig_per_class, feature_names, class_names)
    attn_col_sum = plot_attention_heatmap(rollout, feature_names)
    plot_eca_weights(eca_weights)

    total_time = time.time() - total_start
    report = generate_report(ig_global, ig_per_class, rollout, eca_weights,
                              feature_names, class_names, total_time,
                              "results/E10_interpretability_report.txt")

    # JSON results
    json_results = {
        'experiment': 'E10_interpretability',
        'methods': ['Integrated Gradients', 'Attention Rollout', 'ECA Channel Attention'],
        'model': 'TransECA-Net (d=128, h=8, L=3)',
        'total_params': total_params,
        'ig_samples': len(ig_sample_idx),
        'ig_steps': 50,
        'ig_time_s': ig_time,
        'attn_time_s': attn_time,
        'eca_time_s': eca_time,
        'total_time_s': total_time,
        'ig_top20': [
            {'rank': r+1, 'feature': feature_names[i], 'ig_mean_abs': float(ig_global[i])}
            for r, i in enumerate(np.argsort(ig_global)[::-1][:20])
        ],
        'eca_stats': {
            'mean': float(eca_weights.mean()) if eca_weights is not None else None,
            'std': float(eca_weights.std()) if eca_weights is not None else None,
            'top_channel': int(np.argmax(eca_weights)) if eca_weights is not None else None,
        },
    }

    with open("results/E10_interpretability_results.json", 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON → results/E10_interpretability_results.json")

    # Logger
    logger.set_data_info(dataset="Stage 2 test (data/stage2/test.parquet)",
                          total_samples=len(X_tensor))
    logger.set_model_info(model_type="TransECA-Net Interpretability",
                           hyperparams={'ig_samples': 500, 'ig_steps': 50,
                                        'attn_samples': 200, 'eca_samples': 500})
    logger.set_results(
        ig_top1=feature_names[np.argmax(ig_global)],
        ig_top1_value=float(ig_global.max()),
        ig_time=ig_time, attn_time=attn_time, eca_time=eca_time,
    )
    logger.add_artifact("results/E10_ig_global_importance.png", "plot", "IG global feature importance")
    logger.add_artifact("results/E10_ig_per_class.png", "plot", "IG per-class heatmap")
    logger.add_artifact("results/E10_attention_heatmap.png", "plot", "Attention rollout heatmap")
    logger.add_artifact("results/E10_eca_channel_weights.png", "plot", "ECA channel weights")
    logger.add_artifact("results/E10_interpretability_report.txt", "report", "Analysis report")
    logger.add_artifact("results/E10_interpretability_results.json", "data", "JSON results")
    logger.finish()

    # Summary
    print(f"\n{'='*70}")
    print(f"  E10 Interpretability Analysis Complete!")
    print(f"  IG Top Feature: {feature_names[np.argmax(ig_global)]} "
          f"(IG={ig_global.max():.6f})")
    if eca_weights is not None:
        print(f"  ECA Selectivity (CV): {eca_weights.std()/eca_weights.mean():.4f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
