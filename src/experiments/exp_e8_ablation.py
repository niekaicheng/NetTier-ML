"""
实验 E8: TransECA-Net 消融实验 (Ablation Study)
================================================
目标: 验证各个组件 (CNN, ECA, Transformer) 的贡献。
回答: "为什么要引入 ECA Attention 和 Transformer Encoder？"

Variations:
  1. Full TransECA-Net  (CNN + ECA + Transformer)  — 完整模型
  2. No-ECA            (CNN + Transformer)          — 移除 ECA 注意力
  3. CNN-Only          (CNN only)                    — 仅 CNN 基线

训练条件: 与 S2 训练完全一致
  - Data: data/stage2/{train,val,test}.parquet
  - d_model=128, nhead=8, num_layers=3
  - BS=512, AdamW, CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
  - 类别权重, AMP on XPU/CUDA
  - 20 epochs (enough to see convergence differences)
"""

import sys
import os
import time
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.stdout.reconfigure(encoding='utf-8')

from processing.preprocess import DataPreprocessor
from models.stage2_transeca import TransECANet, ECAModule
from utils.training_logger import TrainingLogger


# ============================================================
# Model Variants
# ============================================================

class TransNet_NoECA(TransECANet):
    """Variant 2: CNN + Transformer (ECA replaced with Identity)"""
    def __init__(self, num_features, num_classes, d_model=128, nhead=8, num_layers=3):
        super().__init__(num_features, num_classes, d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.eca = nn.Identity()


class CNN_Only(nn.Module):
    """Variant 3: CNN baseline (no ECA, no Transformer)"""
    def __init__(self, num_features, num_classes, d_model=128, nhead=8, num_layers=3):
        # nhead/num_layers unused, kept for consistent API
        super().__init__()
        self.conv1 = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)                    # (B, 1, F)
        x = self.relu(self.bn1(self.conv1(x)))  # (B, d_model, F)
        x = self.global_pool(x).squeeze(-1)     # (B, d_model)
        return self.fc(x)


# ============================================================
# Training & Evaluation
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, amp_device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast(amp_device):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return running_loss / len(loader), 100 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, amp_device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return running_loss / len(loader), 100 * correct / total


@torch.no_grad()
def predict_all(model, loader, device, use_amp, amp_device):
    model.eval()
    all_preds = []
    for inputs, _ in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_preds.append(outputs.argmax(dim=1).cpu().numpy())
    return np.concatenate(all_preds)


def train_variant(name, model, train_loader, val_loader, criterion,
                  device, use_amp, amp_device, scaler_factory,
                  epochs=20, lr=0.001):
    """Train a model variant and return training history + best state_dict."""
    print(f"\n{'='*70}")
    print(f"  Training: {name}")
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    print(f"{'='*70}")

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = scaler_factory() if use_amp else None

    best_val_loss = float('inf')
    best_state = None
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                 optimizer, scaler, device, use_amp, amp_device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp, amp_device)
        scheduler.step(epoch + 1)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            saved = " ✓saved"

        dt = time.time() - t0
        print(f"  E{epoch+1:02d}/{epochs} | "
              f"TL={train_loss:.4f} TA={train_acc:.2f}% | "
              f"VL={val_loss:.4f} VA={val_acc:.2f}% | "
              f"{dt:.1f}s{saved}", flush=True)

    print(f"  → Best Val Acc: {best_val_acc:.2f}%")
    return history, best_state, best_val_acc, params


def plot_ablation_comparison(all_histories, save_path='results/E8_ablation_comparison.png'):
    """绘制消融实验对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'Full TransECA-Net': '#1f77b4', 'No-ECA (CNN+Trans)': '#ff7f0e', 'CNN-Only': '#2ca02c'}

    for name, hist in all_histories.items():
        c = colors.get(name, 'gray')
        axes[0, 0].plot(hist['train_loss'], label=name, color=c)
        axes[0, 1].plot(hist['val_loss'], label=name, color=c)
        axes[1, 0].plot(hist['train_acc'], label=name, color=c)
        axes[1, 1].plot(hist['val_acc'], label=name, color=c)

    titles = ['Train Loss', 'Val Loss', 'Train Accuracy (%)', 'Val Accuracy (%)']
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('E8: Ablation Study — Component Contribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Ablation comparison plot → {save_path}")


def plot_ablation_bar(results_summary, save_path='results/E8_ablation_bar.png'):
    """绘制消融实验结果柱状图"""
    names = list(results_summary.keys())
    test_acc = [results_summary[n]['test_accuracy'] for n in names]
    f1_w = [results_summary[n]['weighted_f1'] for n in names]
    f1_m = [results_summary[n]['macro_f1'] for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, test_acc, width, label='Test Accuracy (%)', color='#1f77b4')
    bars2 = ax.bar(x, [f * 100 for f in f1_w], width, label='Weighted F1 (×100)', color='#ff7f0e')
    bars3 = ax.bar(x + width, [f * 100 for f in f1_m], width, label='Macro F1 (×100)', color='#2ca02c')

    ax.set_ylabel('Score')
    ax.set_title('E8: Ablation Study — Test Performance Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 在柱上标数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Ablation bar chart → {save_path}")


# ============================================================
# Main
# ============================================================

def name_delta(a, b):
    d = a - b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def main():
    print("=" * 80)
    print("  E8: TransECA-Net Ablation Study")
    print("  Variants: Full | No-ECA | CNN-Only")
    print("=" * 80)

    # ---- Logger ----
    logger = TrainingLogger(
        experiment_name="E8_ablation",
        description="Ablation study: Full TransECA-Net vs No-ECA vs CNN-Only. "
                    "Validates ECA and Transformer contributions."
    )
    logger.start()

    # ======================== 1. Data Loading ========================
    print("\n[1/5] Loading Stage 2 data ...")
    try:
        df_train = pd.read_parquet("data/stage2/train.parquet")
        df_val   = pd.read_parquet("data/stage2/val.parquet")
        df_test  = pd.read_parquet("data/stage2/test.parquet")
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        logger.finish(status="failed")
        return

    print(f"  ✓ Train: {df_train.shape[0]:,} | Val: {df_val.shape[0]:,} | Test: {df_test.shape[0]:,}")

    # ======================== 2. Preprocessing ========================
    print("\n[2/5] Preprocessing ...")
    preprocessor = None
    for pp_path in ["models_chk/preprocessor_stratified.joblib", "models_chk/preprocessor.joblib"]:
        try:
            preprocessor = joblib.load(pp_path)
            print(f"  ✓ Preprocessor: {pp_path}")
            break
        except Exception:
            continue

    if preprocessor is None:
        print("  ✗ No preprocessor found"); logger.finish(status="failed"); return

    def prep(df):
        df_c = preprocessor.clean(df)
        X, y = preprocessor.transform(df_c, target_col='Label')
        valid = y >= 0
        return X[valid], y[valid]

    X_train, y_train = prep(df_train)
    X_val, y_val     = prep(df_val)
    X_test, y_test   = prep(df_test)

    # Class remapping
    all_labels = np.concatenate([y_train, y_val, y_test])
    unique_classes = np.unique(all_labels)
    class_names_all = [str(c) for c in preprocessor.label_encoder.classes_]
    class_names = [class_names_all[i] for i in unique_classes]
    num_classes = len(unique_classes)
    num_features = X_train.shape[1]

    if num_classes < len(preprocessor.label_encoder.classes_):
        class_map = {old: new for new, old in enumerate(unique_classes)}
        y_train = np.array([class_map[y] for y in y_train])
        y_val   = np.array([class_map[y] for y in y_val])
        y_test  = np.array([class_map[y] for y in y_test])

    print(f"  ✓ Features: {num_features} | Classes: {num_classes}")
    print(f"  ✓ Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Class weights
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = np.clip(class_weights, None, np.median(class_weights) * 50)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # ======================== 3. DataLoaders ========================
    print("\n[3/5] Preparing DataLoaders ...")
    batch_size = 512
    # num_workers=0 to avoid Windows DataLoader deadlock when training multiple models sequentially
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )
    print(f"  ✓ BS={batch_size} | Train batches: {len(train_loader)} | Val: {len(val_loader)}")

    # ======================== 4. Device ========================
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  ✓ Device: CUDA — {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu")
        props = torch.xpu.get_device_properties(0)
        print(f"  ✓ Device: XPU — {torch.xpu.get_device_name(0)} ({props.total_memory // 2**30}GB)")
    else:
        print("  ⚠ Device: CPU")

    use_amp = device.type in ('cuda', 'xpu')
    amp_device = device.type if use_amp else 'cpu'
    scaler_factory = lambda: torch.amp.GradScaler(amp_device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))

    # Logger metadata
    logger.set_data_info(
        dataset="CIC-IDS2017 (Stage 2 Hard Examples)",
        data_path="data/stage2/",
        total_samples=len(X_train) + len(X_val) + len(X_test),
        train_samples=len(X_train), val_samples=len(X_val), test_samples=len(X_test),
        num_features=num_features, num_classes=num_classes,
        split_method="Stage 1 CV Mining + Prediction",
    )
    logger.set_model_info(
        model_type="TransECA-Net (Ablation)",
        model_name="E8 Ablation — Full vs No-ECA vs CNN-Only",
        architecture="1D-CNN + ECA + Transformer (ablated)",
        framework="PyTorch",
        hyperparams={"d_model": 128, "nhead": 8, "num_layers": 3, "variants": 3},
    )
    logger.set_training_config(
        epochs=20, batch_size=batch_size, learning_rate=0.001,
        optimizer="AdamW (weight_decay=1e-4)",
        scheduler="CosineAnnealingWarmRestarts (T_0=10, T_mult=2)",
        loss_function="CrossEntropyLoss (class-weighted)",
        device=str(device),
    )

    # ======================== 5. Train & Evaluate Each Variant ========================
    print(f"\n[4/5] Training 3 variants (20 epochs each) ...")

    d_model, nhead, n_layers, epochs = 128, 8, 3, 20

    variants = {
        "Full TransECA-Net": TransECANet(num_features, num_classes,
                                          d_model=d_model, nhead=nhead, num_layers=n_layers),
        "No-ECA (CNN+Trans)": TransNet_NoECA(num_features, num_classes,
                                              d_model=d_model, nhead=nhead, num_layers=n_layers),
        "CNN-Only": CNN_Only(num_features, num_classes, d_model=d_model),
    }

    all_histories = {}
    results_summary = {}
    total_start = time.time()

    for name, model in variants.items():
        history, best_state, best_val_acc, n_params = train_variant(
            name, model, train_loader, val_loader, criterion,
            device, use_amp, amp_device, scaler_factory, epochs=epochs,
        )
        all_histories[name] = history

        # Load best weights and run test
        model.load_state_dict(best_state)
        model.to(device)
        preds = predict_all(model, test_loader, device, use_amp, amp_device)

        test_acc = accuracy_score(y_test, preds) * 100
        w_f1 = f1_score(y_test, preds, average='weighted')
        m_f1 = f1_score(y_test, preds, average='macro')

        report = classification_report(y_test, preds, target_names=class_names, digits=4)
        print(f"\n  {name} Test Results:")
        print(f"    Accuracy: {test_acc:.2f}% | Weighted F1: {w_f1:.4f} | Macro F1: {m_f1:.4f}")

        results_summary[name] = {
            'test_accuracy': round(test_acc, 2),
            'weighted_f1': round(w_f1, 4),
            'macro_f1': round(m_f1, 4),
            'best_val_accuracy': round(best_val_acc, 2),
            'params': n_params,
            'report': report,
        }

    total_time = time.time() - total_start

    # ======================== 6. Summary & Plots ========================
    print(f"\n[5/5] Results & Visualization ...")
    print(f"\n{'='*70}")
    print(f"  E8 Ablation Summary (20 epochs each)")
    print(f"{'='*70}")
    print(f"  {'Variant':<22} {'Params':>8} {'Test Acc':>10} {'W-F1':>8} {'M-F1':>8} {'Best Val':>10}")
    print(f"  {'-'*68}")
    for name, r in results_summary.items():
        print(f"  {name:<22} {r['params']:>8,} {r['test_accuracy']:>9.2f}% "
              f"{r['weighted_f1']:>8.4f} {r['macro_f1']:>8.4f} {r['best_val_accuracy']:>9.2f}%")

    # Contribution analysis
    full_f1 = results_summary['Full TransECA-Net']['weighted_f1']
    noeca_f1 = results_summary['No-ECA (CNN+Trans)']['weighted_f1']
    cnn_f1 = results_summary['CNN-Only']['weighted_f1']
    print(f"\n  Component Contributions (Weighted F1):")
    print(f"    ECA contribution:         {name_delta(full_f1, noeca_f1)}  (Full - No-ECA)")
    print(f"    Transformer contribution: {name_delta(noeca_f1, cnn_f1)}  (No-ECA - CNN-Only)")
    print(f"    Combined (ECA+Trans):     {name_delta(full_f1, cnn_f1)}  (Full - CNN-Only)")
    print(f"\n  Total time: {total_time/60:.1f} min")

    # Plots
    os.makedirs("results", exist_ok=True)
    plot_ablation_comparison(all_histories)
    plot_ablation_bar(results_summary)

    # Save detailed report
    report_path = "results/E8_ablation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("E8: TransECA-Net Ablation Study\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Variants: Full TransECA-Net | No-ECA (CNN+Transformer) | CNN-Only\n")
        f.write(f"Epochs: 20 | BS: {batch_size} | Device: {device}\n")
        f.write(f"d_model={d_model}, nhead={nhead}, num_layers={n_layers}\n")
        f.write(f"Total training time: {total_time/60:.1f} min\n\n")
        f.write(f"{'Variant':<22} {'Params':>8} {'Test Acc':>10} {'W-F1':>8} {'M-F1':>8} {'Best Val':>10}\n")
        f.write("-" * 70 + "\n")
        for name, r in results_summary.items():
            f.write(f"{name:<22} {r['params']:>8,} {r['test_accuracy']:>9.2f}% "
                    f"{r['weighted_f1']:>8.4f} {r['macro_f1']:>8.4f} {r['best_val_accuracy']:>9.2f}%\n")
        f.write(f"\nComponent Contributions (Weighted F1):\n")
        f.write(f"  ECA:         {name_delta(full_f1, noeca_f1)}\n")
        f.write(f"  Transformer: {name_delta(noeca_f1, cnn_f1)}\n")
        f.write(f"  Combined:    {name_delta(full_f1, cnn_f1)}\n")
        f.write(f"\n{'='*70}\n")
        for name, r in results_summary.items():
            f.write(f"\n{name} — Classification Report:\n")
            f.write("-" * 70 + "\n")
            f.write(r['report'] + "\n")
    print(f"  ✓ Report → {report_path}")

    # Save JSON results
    json_path = "results/E8_ablation_results.json"
    json_data = {
        'experiment': 'E8_ablation',
        'epochs': epochs,
        'device': str(device),
        'd_model': d_model, 'nhead': nhead, 'num_layers': n_layers,
        'total_time_minutes': round(total_time / 60, 2),
        'variants': {},
    }
    for name, r in results_summary.items():
        json_data['variants'][name] = {
            'params': r['params'],
            'test_accuracy': r['test_accuracy'],
            'weighted_f1': r['weighted_f1'],
            'macro_f1': r['macro_f1'],
            'best_val_accuracy': r['best_val_accuracy'],
        }
    json_data['contributions'] = {
        'eca_wf1_delta': round(full_f1 - noeca_f1, 4),
        'transformer_wf1_delta': round(noeca_f1 - cnn_f1, 4),
        'combined_wf1_delta': round(full_f1 - cnn_f1, 4),
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON → {json_path}")

    # Logger
    logger.set_results(**{
        'full_test_accuracy': results_summary['Full TransECA-Net']['test_accuracy'],
        'full_weighted_f1': results_summary['Full TransECA-Net']['weighted_f1'],
        'noeca_test_accuracy': results_summary['No-ECA (CNN+Trans)']['test_accuracy'],
        'noeca_weighted_f1': results_summary['No-ECA (CNN+Trans)']['weighted_f1'],
        'cnn_test_accuracy': results_summary['CNN-Only']['test_accuracy'],
        'cnn_weighted_f1': results_summary['CNN-Only']['weighted_f1'],
        'eca_contribution_wf1': round(full_f1 - noeca_f1, 4),
        'transformer_contribution_wf1': round(noeca_f1 - cnn_f1, 4),
        'total_time_minutes': round(total_time / 60, 2),
    })
    logger.add_artifact(report_path, "report", "Ablation study report")
    logger.add_artifact(json_path, "data", "Ablation results JSON")
    logger.add_artifact("results/E8_ablation_comparison.png", "plot", "Training curves comparison")
    logger.add_artifact("results/E8_ablation_bar.png", "plot", "Bar chart comparison")
    logger.finish()

    print(f"\n{'='*70}")
    print(f"  E8 Ablation Study Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
