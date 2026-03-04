"""
实验 E12: 特征空间可视化 (t-SNE / UMAP)
=========================================
目标: 可视化 Stage 2 数据的高维特征空间，展示各攻击类别的可分性和聚类结构。

方法:
  1. t-SNE on raw scaled features (76-dim → 2D) — 原始特征空间
  2. UMAP on raw scaled features                — 保持全局拓扑结构
  3. t-SNE on TransECA-Net embeddings (128-dim → 2D) — 学习后的表征空间
  4. UMAP on TransECA-Net embeddings             — 学习后表征的全局结构
  5. Binary view: Benign vs Attack 可分性

分析:
  - 原始特征 vs 学习表征的聚类质量对比
  - 各攻击类别的重叠程度和分离度
  - 分层架构的合理性证据

参考: experimental_design.md §8

产出:
  - results/E12_tsne_raw.png           — t-SNE on raw features
  - results/E12_umap_raw.png           — UMAP on raw features
  - results/E12_tsne_embedding.png     — t-SNE on TransECA embeddings
  - results/E12_umap_embedding.png     — UMAP on TransECA embeddings
  - results/E12_binary_view.png        — Binary (Benign/Attack) visualization
  - results/E12_visualization_report.txt
  - results/E12_visualization_results.json
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
from matplotlib.lines import Line2D

import torch
from sklearn.manifold import TSNE
import umap

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.stdout.reconfigure(encoding='utf-8')

from models.stage2_transeca import TransECANet
from utils.training_logger import TrainingLogger


# ══════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_data(max_samples=8000, random_state=42):
    """
    Load Stage 2 test data with stratified subsample for visualization.
    8K samples is a good balance: enough for structure, fast for t-SNE/UMAP.
    """
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

    # Stratified subsample
    if max_samples and len(y_enc) > max_samples:
        rng = np.random.RandomState(random_state)
        idx_keep = []
        classes_u, counts = np.unique(y_enc, return_counts=True)
        for c, cnt in zip(classes_u, counts):
            c_idx = np.where(y_enc == c)[0]
            alloc = max(cnt, int(max_samples * cnt / len(y_enc)))
            alloc = min(alloc, cnt)
            if alloc >= cnt:
                idx_keep.append(c_idx)
            else:
                idx_keep.append(rng.choice(c_idx, size=alloc, replace=False))
        idx = np.concatenate(idx_keep)
        if len(idx) > max_samples:
            idx = rng.choice(idx, size=max_samples, replace=False)
        idx.sort()
        X_raw = X_raw.iloc[idx].reset_index(drop=True)
        y_enc = y_enc[idx]

    X_numeric = X_raw.select_dtypes(include=[np.number])
    feature_names = list(X_numeric.columns)
    X_scaled = preprocessor.scaler.transform(X_numeric)

    return X_scaled, y_enc, feature_names, list(classes), preprocessor


def extract_embeddings(model, X_scaled, device, batch_size=1024):
    """
    Extract penultimate-layer embeddings (128-dim) from TransECA-Net.
    This is the output after global_pool, before the FC classification head.
    """
    model.eval()
    X_tensor = torch.FloatTensor(X_scaled)
    embeddings = []

    # Hook to capture the penultimate layer
    hook_output = {}

    def hook_fn(module, input, output):
        hook_output['embedding'] = output

    # Register hook on the FC layer's input (= global_pool output squeezed)
    # We hook on global_pool and squeeze manually
    handle = model.global_pool.register_forward_hook(hook_fn)

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            x_b = X_tensor[i:i+batch_size].to(device)
            _ = model(x_b)
            # hook_output['embedding'] shape: (B, d_model, 1)
            emb = hook_output['embedding'].squeeze(-1).cpu().numpy()  # (B, d_model)
            embeddings.append(emb)

    handle.remove()
    return np.concatenate(embeddings, axis=0)


# ══════════════════════════════════════════════════════════════════════
#  Visualization
# ══════════════════════════════════════════════════════════════════════

# Color palette for 15 classes — maximally distinct
CLASS_COLORS = [
    '#2196F3',  # Benign - blue
    '#FF9800',  # Bot - orange
    '#E53935',  # DDoS - red
    '#9C27B0',  # DoS GoldenEye - purple
    '#F44336',  # DoS Hulk - dark red
    '#FF5722',  # DoS Slowhttptest - deep orange
    '#E91E63',  # DoS slowloris - pink
    '#4CAF50',  # FTP-Patator - green
    '#795548',  # Heartbleed - brown
    '#607D8B',  # Infiltration - blue grey
    '#00BCD4',  # PortScan - cyan
    '#8BC34A',  # SSH-Patator - light green
    '#FFC107',  # Web Attack Brute Force - amber
    '#CDDC39',  # Web Attack SQL Injection - lime
    '#FF6F00',  # Web Attack XSS - dark amber
]


def make_scatter(ax, coords_2d, y_enc, class_names, title, point_size=5, alpha=0.5):
    """Create a single t-SNE/UMAP scatter plot."""
    unique_classes = np.unique(y_enc)

    for c in unique_classes:
        mask = y_enc == c
        color = CLASS_COLORS[c % len(CLASS_COLORS)]
        count = mask.sum()
        label = f"{class_names[c]} ({count})"
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   c=color, s=point_size, alpha=alpha,
                   edgecolors='none', label=label, rasterized=True)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=10)
    ax.set_ylabel('Dim 2', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.15)


def plot_single(coords_2d, y_enc, class_names, title, save_path,
                point_size=5, alpha=0.5):
    """Full-page scatter plot with legend."""
    fig, ax = plt.subplots(figsize=(12, 9))
    make_scatter(ax, coords_2d, y_enc, class_names, title, point_size, alpha)

    # Create legend with larger markers
    handles = []
    unique_classes = np.unique(y_enc)
    for c in unique_classes:
        color = CLASS_COLORS[c % len(CLASS_COLORS)]
        count = (y_enc == c).sum()
        handles.append(Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, markersize=7,
                              label=f"{class_names[c]} ({count})"))

    ax.legend(handles=handles, fontsize=7, loc='center left',
              bbox_to_anchor=(1.01, 0.5), framealpha=0.9, ncol=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {title} → {save_path}")


def plot_binary_view(tsne_raw, umap_raw, tsne_emb, umap_emb,
                     y_enc, class_names, save_path):
    """Binary (Benign vs Attack) 2x2 comparison."""
    benign_idx = class_names.index('Benign') if 'Benign' in class_names else 0
    y_binary = (y_enc != benign_idx).astype(int)
    binary_names = ['Benign', 'Attack']
    binary_colors = ['#2196F3', '#E53935']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    data_pairs = [
        (tsne_raw, 't-SNE (Raw Features)'),
        (umap_raw, 'UMAP (Raw Features)'),
        (tsne_emb, 't-SNE (TransECA Embeddings)'),
        (umap_emb, 'UMAP (TransECA Embeddings)'),
    ]

    for ax, (coords, title) in zip(axes.ravel(), data_pairs):
        for b in [0, 1]:
            mask = y_binary == b
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=binary_colors[b], s=4, alpha=0.4,
                       edgecolors='none', label=f"{binary_names[b]} ({mask.sum()})",
                       rasterized=True)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.15)
        ax.set_xlabel('Dim 1', fontsize=9)
        ax.set_ylabel('Dim 2', fontsize=9)

    plt.suptitle('E12: Binary View — Benign vs Attack', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Binary view → {save_path}")


# ══════════════════════════════════════════════════════════════════════
#  Clustering Quality Metrics
# ══════════════════════════════════════════════════════════════════════

def compute_silhouette(coords_2d, y_enc, sample_size=2000):
    """Compute silhouette score on a subsample (for speed)."""
    from sklearn.metrics import silhouette_score
    n = len(y_enc)
    if n > sample_size:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=sample_size, replace=False)
        return silhouette_score(coords_2d[idx], y_enc[idx])
    return silhouette_score(coords_2d, y_enc)


# ══════════════════════════════════════════════════════════════════════
#  Report
# ══════════════════════════════════════════════════════════════════════

def generate_report(sil_scores, n_samples, n_classes, total_time, save_path):
    lines = []
    lines.append("=" * 80)
    lines.append("  E12: Feature Space Visualization (t-SNE / UMAP)")
    lines.append("  Raw Features vs TransECA-Net Embeddings")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  Samples: {n_samples:,}")
    lines.append(f"  Classes: {n_classes}")
    lines.append("")

    lines.append("── Silhouette Scores (higher = better separation) ──")
    lines.append(f"  {'Method':<30s} {'Silhouette':>10}")
    lines.append(f"  {'-'*42}")
    for name, score in sil_scores.items():
        lines.append(f"  {name:<30s} {score:>10.4f}")

    lines.append("")

    # Interpretation
    raw_avg = (sil_scores.get('t-SNE (Raw)', 0) + sil_scores.get('UMAP (Raw)', 0)) / 2
    emb_avg = (sil_scores.get('t-SNE (Embedding)', 0) + sil_scores.get('UMAP (Embedding)', 0)) / 2

    if emb_avg > raw_avg:
        improvement = (emb_avg - raw_avg) / max(abs(raw_avg), 1e-6) * 100
        lines.append(f"  → TransECA embeddings improve cluster separation by {improvement:.1f}%")
        lines.append(f"     (avg silhouette: raw={raw_avg:.4f} → embedding={emb_avg:.4f})")
    else:
        lines.append(f"  → Raw features already have good separation (silhouette {raw_avg:.4f})")
        lines.append(f"     Embeddings: {emb_avg:.4f}")

    lines.append("")
    lines.append("── Key Observations ──")
    lines.append("  1. t-SNE preserves local neighborhood structure (good for cluster shape)")
    lines.append("  2. UMAP preserves global topology (good for cluster relationships)")
    lines.append("  3. TransECA embeddings compress 76 features → 128-dim learned representation")
    lines.append("  4. Binary view shows how well the hierarchical design separates Benign/Attack")
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
    print("  E12: Feature Space Visualization (t-SNE / UMAP)")
    print("  Raw Features + TransECA Embeddings")
    print("=" * 80)

    logger = TrainingLogger(
        experiment_name="E12_visualization",
        description="t-SNE/UMAP on raw features and TransECA-Net embeddings"
    )
    logger.start()

    total_start = time.time()
    os.makedirs("results", exist_ok=True)

    # ──── Load Data ────
    print(f"\n[1/7] Loading data ...", flush=True)
    X_scaled, y_enc, feature_names, class_names, preprocessor = load_data(max_samples=8000)
    n_samples = len(y_enc)
    n_classes = len(np.unique(y_enc))
    print(f"  ✓ {n_samples:,} samples, {X_scaled.shape[1]} features, {n_classes} classes")

    # ──── Extract Embeddings ────
    print(f"\n[2/7] Extracting TransECA-Net embeddings ...", flush=True)
    device = torch.device('cpu')  # CPU is fine for inference on 8K samples
    model = TransECANet(
        num_features=X_scaled.shape[1],
        num_classes=len(class_names),
        d_model=128, nhead=8, num_layers=3
    )
    model.load_state_dict(
        torch.load("models_chk/stage2_transeca.pth",
                    map_location=device, weights_only=True)
    )
    model.eval()

    embeddings = extract_embeddings(model, X_scaled, device)
    print(f"  ✓ Embeddings: {embeddings.shape} (128-dim)")

    # ──── t-SNE ────
    print(f"\n[3/7] t-SNE on raw features ({X_scaled.shape[1]}-dim → 2D) ...", flush=True)
    t0 = time.time()
    tsne_raw = TSNE(n_components=2, perplexity=30, random_state=42,
                     max_iter=1000, learning_rate='auto', init='pca').fit_transform(X_scaled)
    print(f"  ✓ t-SNE (raw): {time.time()-t0:.1f}s")

    print(f"\n[4/7] t-SNE on TransECA embeddings (128-dim → 2D) ...", flush=True)
    t0 = time.time()
    tsne_emb = TSNE(n_components=2, perplexity=30, random_state=42,
                     max_iter=1000, learning_rate='auto', init='pca').fit_transform(embeddings)
    print(f"  ✓ t-SNE (embedding): {time.time()-t0:.1f}s")

    # ──── UMAP ────
    print(f"\n[5/7] UMAP on raw features ...", flush=True)
    t0 = time.time()
    umap_raw = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                          random_state=42).fit_transform(X_scaled)
    print(f"  ✓ UMAP (raw): {time.time()-t0:.1f}s")

    print(f"\n[6/7] UMAP on TransECA embeddings ...", flush=True)
    t0 = time.time()
    umap_emb = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                          random_state=42).fit_transform(embeddings)
    print(f"  ✓ UMAP (embedding): {time.time()-t0:.1f}s")

    # ──── Plots ────
    print(f"\n[7/7] Generating plots & report ...", flush=True)

    plot_single(tsne_raw, y_enc, class_names,
                'E12: t-SNE — Raw Scaled Features (76-dim)',
                'results/E12_tsne_raw.png')

    plot_single(umap_raw, y_enc, class_names,
                'E12: UMAP — Raw Scaled Features (76-dim)',
                'results/E12_umap_raw.png')

    plot_single(tsne_emb, y_enc, class_names,
                'E12: t-SNE — TransECA-Net Embeddings (128-dim)',
                'results/E12_tsne_embedding.png')

    plot_single(umap_emb, y_enc, class_names,
                'E12: UMAP — TransECA-Net Embeddings (128-dim)',
                'results/E12_umap_embedding.png')

    plot_binary_view(tsne_raw, umap_raw, tsne_emb, umap_emb,
                     y_enc, class_names, 'results/E12_binary_view.png')

    # ── Silhouette Scores ──
    print("  Computing silhouette scores ...", flush=True)
    sil_scores = {
        't-SNE (Raw)': compute_silhouette(tsne_raw, y_enc),
        'UMAP (Raw)': compute_silhouette(umap_raw, y_enc),
        't-SNE (Embedding)': compute_silhouette(tsne_emb, y_enc),
        'UMAP (Embedding)': compute_silhouette(umap_emb, y_enc),
    }
    for name, score in sil_scores.items():
        print(f"    {name}: {score:.4f}")

    # ── Report ──
    total_time = time.time() - total_start

    generate_report(sil_scores, n_samples, n_classes, total_time,
                     'results/E12_visualization_report.txt')

    # ── JSON ──
    json_results = {
        'experiment': 'E12_visualization',
        'total_time_s': total_time,
        'n_samples': n_samples,
        'n_classes': n_classes,
        'raw_features_dim': X_scaled.shape[1],
        'embedding_dim': embeddings.shape[1],
        'silhouette_scores': {k: float(v) for k, v in sil_scores.items()},
        'tsne_config': {'perplexity': 30, 'max_iter': 1000, 'init': 'pca'},
        'umap_config': {'n_neighbors': 15, 'min_dist': 0.1},
    }
    with open("results/E12_visualization_results.json", 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON → results/E12_visualization_results.json")

    # ── Logger ──
    logger.set_data_info(dataset="Stage 2 test (data/stage2/test.parquet)",
                          total_samples=n_samples)
    logger.set_model_info(model_type="t-SNE/UMAP Visualization",
                           hyperparams={'tsne_perplexity': 30, 'umap_n_neighbors': 15,
                                        'max_samples': 8000})
    logger.set_results(**{k.replace(' ', '_').replace('(', '').replace(')', ''): float(v)
                          for k, v in sil_scores.items()})
    logger.add_artifact("results/E12_tsne_raw.png", "plot", "t-SNE raw features")
    logger.add_artifact("results/E12_umap_raw.png", "plot", "UMAP raw features")
    logger.add_artifact("results/E12_tsne_embedding.png", "plot", "t-SNE embeddings")
    logger.add_artifact("results/E12_umap_embedding.png", "plot", "UMAP embeddings")
    logger.add_artifact("results/E12_binary_view.png", "plot", "Binary view comparison")
    logger.add_artifact("results/E12_visualization_report.txt", "report", "Analysis report")
    logger.add_artifact("results/E12_visualization_results.json", "data", "Full results")
    logger.finish()

    # Summary
    best_method = max(sil_scores, key=sil_scores.get)
    print(f"\n{'='*70}")
    print(f"  E12 Visualization Complete!")
    print(f"  Best silhouette: {best_method} = {sil_scores[best_method]:.4f}")
    print(f"  Silhouette (raw avg): {(sil_scores['t-SNE (Raw)'] + sil_scores['UMAP (Raw)'])/2:.4f}")
    print(f"  Silhouette (emb avg): {(sil_scores['t-SNE (Embedding)'] + sil_scores['UMAP (Embedding)'])/2:.4f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
