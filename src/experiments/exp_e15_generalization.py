"""
实验 E15: 架构泛化验证 (Architecture Generalization on UNSW-NB15)
================================================================
目标: 证明 TransECA-Net 架构不仅在 CIC-IDS2017 上有效，
      在完全不同的数据集 (UNSW-NB15) 上从头训练也能取得良好效果。
回答: "模型架构是否过拟合了 CIC-IDS2017？"

方案 B — Architecture Generalization:
  - 在 UNSW-NB15 上从头训练一个 TransECA-Net
  - 使用 UNSW-NB15 自带的 training/testing split
  - 从 training set 中切出 15% 作为 validation
  - 对比 CIC-IDS2017 上的 S2 训练结果

数据: archive/UNSW_NB15_{training,testing}-set.parquet
  - Training: 175,341 samples, 36 columns
  - Testing:  82,332 samples
  - 10 类: Normal + 9 attack categories
  - 特征: 34 数值/类别特征 (与 CIC-IDS2017 完全不同)

训练条件 (与 S2/E8 一致):
  - d_model=128, nhead=8, num_layers=3
  - BS=512, AdamW, CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
  - 类别权重, AMP on XPU/CUDA
  - 30 epochs
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.stdout.reconfigure(encoding='utf-8')

from models.stage2_transeca import TransECANet
from utils.training_logger import TrainingLogger


# ============================================================
# UNSW-NB15 Preprocessor
# ============================================================

class UNSWPreprocessor:
    """
    UNSW-NB15 专用预处理器。
    - 独立于 CIC-IDS2017 的预处理管线
    - One-hot 编码类别特征 (proto, service, state)
    - 标准化数值特征
    - LabelEncoder 编码 attack_cat
    """

    # 需要排除的列
    DROP_COLS = ['label']  # 二分类 label，我们用 attack_cat 做多分类
    TARGET_COL = 'attack_cat'
    CAT_COLS = ['proto', 'service', 'state']

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.num_cols_ = None
        self.fitted = False

    def fit_transform(self, df):
        """Fit on training data and transform."""
        df = df.copy()

        # 1. Encode target
        y = self.label_encoder.fit_transform(df[self.TARGET_COL].astype(str))

        # 2. Separate features
        df = df.drop(columns=[self.TARGET_COL] + self.DROP_COLS, errors='ignore')

        # 3. Categorical → One-hot
        cat_data = df[self.CAT_COLS].astype(str)
        cat_encoded = self.ohe.fit_transform(cat_data)

        # 4. Numerical features
        self.num_cols_ = [c for c in df.columns if c not in self.CAT_COLS]
        num_data = df[self.num_cols_].values.astype(np.float32)

        # Handle NaN/Inf
        num_data = np.nan_to_num(num_data, nan=0.0, posinf=0.0, neginf=0.0)

        # 5. Scale numerical
        num_scaled = self.scaler.fit_transform(num_data)

        # 6. Combine
        X = np.hstack([num_scaled, cat_encoded]).astype(np.float32)
        self.fitted = True
        return X, y

    def transform(self, df):
        """Transform using fitted encoders/scaler."""
        assert self.fitted, "Must call fit_transform first"
        df = df.copy()

        # 1. Encode target
        y = self.label_encoder.transform(df[self.TARGET_COL].astype(str))

        # 2. Separate features
        df = df.drop(columns=[self.TARGET_COL] + self.DROP_COLS, errors='ignore')

        # 3. Categorical
        cat_data = df[self.CAT_COLS].astype(str)
        cat_encoded = self.ohe.transform(cat_data)

        # 4. Numerical
        num_data = df[self.num_cols_].values.astype(np.float32)
        num_data = np.nan_to_num(num_data, nan=0.0, posinf=0.0, neginf=0.0)
        num_scaled = self.scaler.transform(num_data)

        # 5. Combine
        X = np.hstack([num_scaled, cat_encoded]).astype(np.float32)
        return X, y


# ============================================================
# Training & Evaluation (same as E8/S2)
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
    """Evaluate without AMP (XPU Transformer fused kernel issue)."""
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
def predict_all(model, loader, device):
    """Predict without AMP."""
    model.eval()
    all_preds = []
    for inputs, _ in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_preds.append(outputs.argmax(dim=1).cpu().numpy())
    return np.concatenate(all_preds)


# ============================================================
# Plotting
# ============================================================

def plot_training_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax1.plot(history['val_loss'], label='Val Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('E15: UNSW-NB15 — Training/Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['train_acc'], label='Train Acc', marker='o', markersize=3)
    ax2.plot(history['val_acc'], label='Val Acc', marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('E15: UNSW-NB15 — Training/Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Training curves → {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('E15: UNSW-NB15 — TransECA-Net Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix → {save_path}")


def plot_cross_dataset_comparison(cic_results, unsw_results, save_path):
    """
    对比 CIC-IDS2017 (S2) vs UNSW-NB15 (E15) 的性能。
    """
    datasets = ['CIC-IDS2017\n(S2 Training)', 'UNSW-NB15\n(E15 Generalization)']
    test_acc = [cic_results['test_accuracy'], unsw_results['test_accuracy']]
    w_f1 = [cic_results['weighted_f1'] * 100, unsw_results['weighted_f1'] * 100]
    m_f1 = [cic_results['macro_f1'] * 100, unsw_results['macro_f1'] * 100]

    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, test_acc, width, label='Test Accuracy (%)', color='#1f77b4')
    bars2 = ax.bar(x, w_f1, width, label='Weighted F1 (×100)', color='#ff7f0e')
    bars3 = ax.bar(x + width, m_f1, width, label='Macro F1 (×100)', color='#2ca02c')

    ax.set_ylabel('Score')
    ax.set_title('E15: Architecture Generalization — CIC-IDS2017 vs UNSW-NB15', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Cross-dataset comparison → {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("  E15: Architecture Generalization on UNSW-NB15")
    print("  Method: Train fresh TransECA-Net from scratch on UNSW-NB15")
    print("=" * 80)

    # ---- Logger ----
    logger = TrainingLogger(
        experiment_name="E15_generalization",
        description="Architecture generalization: Train TransECA-Net from scratch on UNSW-NB15 "
                    "to verify architecture is not overfit to CIC-IDS2017."
    )
    logger.start()

    # ======================== 1. Data Loading ========================
    print("\n[1/6] Loading UNSW-NB15 data ...")

    train_path = "archive/UNSW_NB15_training-set.parquet"
    test_path  = "archive/UNSW_NB15_testing-set.parquet"

    for p in [train_path, test_path]:
        if not os.path.exists(p):
            print(f"  ✗ File not found: {p}")
            logger.finish(status="failed")
            return

    df_train_full = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    print(f"  ✓ UNSW Training set: {df_train_full.shape[0]:,} samples, {df_train_full.shape[1]} columns")
    print(f"  ✓ UNSW Testing set:  {df_test.shape[0]:,} samples")

    # Split training → train + validation (85/15 stratified)
    df_train, df_val = train_test_split(
        df_train_full, test_size=0.15, random_state=42,
        stratify=df_train_full['attack_cat']
    )
    print(f"  ✓ Split: Train {df_train.shape[0]:,} | Val {df_val.shape[0]:,} | Test {df_test.shape[0]:,}")

    print(f"\n  Attack Category Distribution (Train):")
    for cat, count in df_train['attack_cat'].value_counts().items():
        print(f"    {str(cat):>20s}: {count:>7,}")

    # ======================== 2. Preprocessing ========================
    print("\n[2/6] Preprocessing (UNSW-specific) ...")
    preprocessor = UNSWPreprocessor()

    X_train, y_train = preprocessor.fit_transform(df_train)
    X_val, y_val = preprocessor.transform(df_val)
    X_test, y_test = preprocessor.transform(df_test)

    class_names = list(preprocessor.label_encoder.classes_)
    num_classes = len(class_names)
    num_features = X_train.shape[1]

    print(f"  ✓ Features (after encoding): {num_features}")
    print(f"  ✓ Classes: {num_classes} — {class_names}")
    print(f"  ✓ Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # ======================== 3. DataLoaders ========================
    print("\n[3/6] Preparing DataLoaders ...")

    # Class weights
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = np.clip(class_weights, None, np.median(class_weights) * 50)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"  ✓ Class weights computed (max ratio: {class_weights.max()/class_weights.min():.1f}x)")

    batch_size = 512
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
    print(f"  ✓ BS={batch_size} | Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    # ======================== 4. Device & Model ========================
    print("\n[4/6] Setting up model & device ...")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  ✓ Device: CUDA — {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu")
        props = torch.xpu.get_device_properties(0)
        print(f"  ✓ Device: XPU — {torch.xpu.get_device_name(0)} ({props.total_memory // 2**30}GB)")
    else:
        print("  ⚠ Device: CPU (training will be slow)")

    use_amp = device.type in ('cuda', 'xpu')
    amp_device = device.type if use_amp else 'cpu'

    d_model, nhead, n_layers = 128, 8, 3
    epochs = 30

    model = TransECANet(num_features, num_classes,
                        d_model=d_model, nhead=nhead, num_layers=n_layers)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ TransECA-Net: {n_params:,} parameters")
    print(f"  ✓ Config: d_model={d_model}, nhead={nhead}, num_layers={n_layers}")
    print(f"  ✓ Epochs: {epochs}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler(amp_device) if use_amp else None

    # Logger metadata
    logger.set_data_info(
        dataset="UNSW-NB15 (Architecture Generalization)",
        data_path="archive/UNSW_NB15_*.parquet",
        total_samples=len(X_train) + len(X_val) + len(X_test),
        train_samples=len(X_train), val_samples=len(X_val), test_samples=len(X_test),
        num_features=num_features, num_classes=num_classes,
        class_distribution={class_names[i]: int(c) for i, c in enumerate(class_counts)},
        split_method="Original train/test split + 15% val from train (stratified)",
        split_ratios="Train 85% / Val 15% of training-set, Test = testing-set",
    )
    logger.set_model_info(
        model_type="TransECA-Net",
        model_name="TransECA-Net on UNSW-NB15",
        architecture="1D-CNN + ECA + Transformer Encoder",
        framework="PyTorch",
        hyperparams={"d_model": d_model, "nhead": nhead, "num_layers": n_layers,
                     "n_params": n_params},
    )
    logger.set_training_config(
        epochs=epochs, batch_size=batch_size, learning_rate=0.001,
        optimizer="AdamW (weight_decay=1e-4)",
        scheduler="CosineAnnealingWarmRestarts (T_0=10, T_mult=2)",
        loss_function="CrossEntropyLoss (class-weighted)",
        device=str(device),
    )

    # ======================== 5. Training ========================
    print(f"\n[5/6] Training TransECA-Net on UNSW-NB15 ({epochs} epochs) ...")
    print(f"{'='*70}")

    best_val_loss = float('inf')
    best_state = None
    best_val_acc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    total_start = time.time()

    for epoch in range(epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, use_amp, amp_device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, use_amp, amp_device
        )
        scheduler.step(epoch + 1)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            saved = " ✓saved"

        dt = time.time() - t0
        print(f"  E{epoch+1:02d}/{epochs} | "
              f"TL={train_loss:.4f} TA={train_acc:.2f}% | "
              f"VL={val_loss:.4f} VA={val_acc:.2f}% | "
              f"{dt:.1f}s{saved}", flush=True)

    train_time = time.time() - total_start
    print(f"\n  Training complete: {train_time/60:.1f} min")
    print(f"  Best model: Epoch {best_epoch}, Val Acc {best_val_acc:.2f}%")

    # ======================== 6. Evaluation ========================
    print(f"\n[6/6] Testing on UNSW-NB15 test set ...")
    print(f"{'='*70}")

    model.load_state_dict(best_state)
    model.to(device)

    preds = predict_all(model, test_loader, device)

    test_acc = accuracy_score(y_test, preds) * 100
    w_f1 = f1_score(y_test, preds, average='weighted')
    m_f1 = f1_score(y_test, preds, average='macro')
    report = classification_report(y_test, preds, target_names=class_names, digits=4)

    print(f"\n  UNSW-NB15 Test Results:")
    print(f"    Test Accuracy:  {test_acc:.2f}%")
    print(f"    Weighted F1:    {w_f1:.4f}")
    print(f"    Macro F1:       {m_f1:.4f}")
    print(f"    Best Val Acc:   {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"\n  Classification Report:")
    print(report)

    # ---- CIC-IDS2017 reference results (S2 training) ----
    cic_results = {
        'test_accuracy': 93.00,
        'weighted_f1': 0.95,
        'macro_f1': 0.80,  # approximate from S2 report
    }
    # Try to load actual S2 results if available
    for s2_json in ['results/stage2_results.json']:
        if os.path.exists(s2_json):
            try:
                with open(s2_json, 'r') as f:
                    s2_data = json.load(f)
                cic_results.update(s2_data)
            except Exception:
                pass

    unsw_results = {
        'test_accuracy': round(test_acc, 2),
        'weighted_f1': round(w_f1, 4),
        'macro_f1': round(m_f1, 4),
    }

    # ======================== 7. Save Results ========================
    os.makedirs("results", exist_ok=True)

    # 7a. Training curves
    plot_training_history(history, "results/E15_unsw_training_curves.png")

    # 7b. Confusion matrix
    plot_confusion_matrix(y_test, preds, class_names, "results/E15_unsw_confusion_matrix.png")

    # 7c. Cross-dataset comparison bar chart
    plot_cross_dataset_comparison(cic_results, unsw_results,
                                  "results/E15_cross_dataset_comparison.png")

    # 7d. Detailed report
    report_path = "results/E15_generalization_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("E15: Architecture Generalization — UNSW-NB15\n")
        f.write("=" * 70 + "\n\n")
        f.write("Method: Train TransECA-Net from scratch on UNSW-NB15\n")
        f.write("Purpose: Verify architecture is not overfit to CIC-IDS2017\n\n")

        f.write(f"Dataset: UNSW-NB15\n")
        f.write(f"  Training set: {len(X_train):,} samples\n")
        f.write(f"  Validation:   {len(X_val):,} samples (15% stratified from train)\n")
        f.write(f"  Test set:     {len(X_test):,} samples (original testing-set)\n")
        f.write(f"  Features:     {num_features} (after One-hot encoding)\n")
        f.write(f"  Classes:      {num_classes} — {class_names}\n\n")

        f.write(f"Model: TransECA-Net\n")
        f.write(f"  d_model={d_model}, nhead={nhead}, num_layers={n_layers}\n")
        f.write(f"  Parameters: {n_params:,}\n")
        f.write(f"  Epochs: {epochs} | BS: {batch_size} | Device: {device}\n")
        f.write(f"  Training time: {train_time/60:.1f} min\n")
        f.write(f"  Best model: Epoch {best_epoch}\n\n")

        f.write("=" * 70 + "\n")
        f.write("Cross-Dataset Comparison\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Dataset':<25} {'Test Acc':>10} {'W-F1':>8} {'M-F1':>8}\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'CIC-IDS2017 (S2)':<25} {cic_results['test_accuracy']:>9.2f}% "
                f"{cic_results['weighted_f1']:>8.4f} {cic_results['macro_f1']:>8.4f}\n")
        f.write(f"{'UNSW-NB15 (E15)':<25} {unsw_results['test_accuracy']:>9.2f}% "
                f"{unsw_results['weighted_f1']:>8.4f} {unsw_results['macro_f1']:>8.4f}\n\n")

        gap_acc = cic_results['test_accuracy'] - unsw_results['test_accuracy']
        gap_wf1 = cic_results['weighted_f1'] - unsw_results['weighted_f1']
        f.write(f"Performance Gap:\n")
        f.write(f"  Accuracy:    {gap_acc:+.2f}%\n")
        f.write(f"  Weighted F1: {gap_wf1:+.4f}\n\n")

        if unsw_results['weighted_f1'] >= 0.80:
            f.write("Conclusion: TransECA-Net achieves strong performance on UNSW-NB15,\n")
            f.write("confirming the architecture generalizes beyond CIC-IDS2017.\n")
        elif unsw_results['weighted_f1'] >= 0.65:
            f.write("Conclusion: TransECA-Net achieves reasonable performance on UNSW-NB15.\n")
            f.write("The architecture demonstrates transferability, though dataset-specific\n")
            f.write("tuning may further improve results.\n")
        else:
            f.write("Conclusion: Performance on UNSW-NB15 is below expectations.\n")
            f.write("Further investigation into dataset-specific preprocessing is recommended.\n")

        f.write(f"\n{'='*70}\n")
        f.write(f"\nUNSW-NB15 Classification Report:\n")
        f.write("-" * 70 + "\n")
        f.write(report + "\n")

        f.write(f"\nClass Distribution (Training):\n")
        f.write("-" * 40 + "\n")
        for i, name in enumerate(class_names):
            f.write(f"  {name:>20s}: {class_counts[i]:>7,}\n")

    print(f"  ✓ Report → {report_path}")

    # 7e. JSON results
    json_path = "results/E15_generalization_results.json"
    json_data = {
        'experiment': 'E15_generalization',
        'method': 'Architecture Generalization (train from scratch on UNSW-NB15)',
        'dataset': 'UNSW-NB15',
        'model': 'TransECA-Net',
        'd_model': d_model, 'nhead': nhead, 'num_layers': n_layers,
        'params': n_params,
        'epochs': epochs, 'best_epoch': best_epoch,
        'batch_size': batch_size,
        'device': str(device),
        'training_time_minutes': round(train_time / 60, 2),
        'num_features': num_features,
        'num_classes': num_classes,
        'class_names': class_names,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'unsw_results': {
            'test_accuracy': round(test_acc, 2),
            'weighted_f1': round(w_f1, 4),
            'macro_f1': round(m_f1, 4),
            'best_val_accuracy': round(best_val_acc, 2),
        },
        'cic_reference': cic_results,
        'performance_gap': {
            'accuracy': round(cic_results['test_accuracy'] - test_acc, 2),
            'weighted_f1': round(cic_results['weighted_f1'] - w_f1, 4),
        },
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON → {json_path}")

    # 7f. Save model checkpoint
    model_path = "models_chk/e15_transeca_unsw.pth"
    torch.save(best_state, model_path)
    print(f"  ✓ Model → {model_path}")

    # ---- Logger finalize ----
    logger.set_results(**{
        'unsw_test_accuracy': round(test_acc, 2),
        'unsw_weighted_f1': round(w_f1, 4),
        'unsw_macro_f1': round(m_f1, 4),
        'unsw_best_val_accuracy': round(best_val_acc, 2),
        'best_epoch': best_epoch,
        'training_time_minutes': round(train_time / 60, 2),
        'params': n_params,
        'cic_test_accuracy_ref': cic_results['test_accuracy'],
        'cic_weighted_f1_ref': cic_results['weighted_f1'],
        'gap_accuracy': round(cic_results['test_accuracy'] - test_acc, 2),
        'gap_weighted_f1': round(cic_results['weighted_f1'] - w_f1, 4),
    })
    logger.add_artifact(report_path, "report", "E15 generalization report")
    logger.add_artifact(json_path, "data", "E15 results JSON")
    logger.add_artifact("results/E15_unsw_training_curves.png", "plot", "Training curves")
    logger.add_artifact("results/E15_unsw_confusion_matrix.png", "plot", "Confusion matrix")
    logger.add_artifact("results/E15_cross_dataset_comparison.png", "plot", "Cross-dataset comparison")
    logger.add_artifact(model_path, "model", "TransECA-Net trained on UNSW-NB15")
    logger.finish()

    print(f"\n{'='*70}")
    print(f"  E15 Architecture Generalization Complete!")
    print(f"  UNSW-NB15: Acc={test_acc:.2f}% | W-F1={w_f1:.4f} | M-F1={m_f1:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
