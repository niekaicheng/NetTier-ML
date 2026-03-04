"""
优化的 Stage 2 训练脚本 (v2)
================================
使用 Stage 1 生成的难例数据 (data/stage2/*.parquet) 训练 TransECA-Net

改进点 (vs v1):
1. 使用 Stage 1 CV Mining 产生的 data/stage2/*.parquet (非原始数据)
2. 增加训练 Epochs（50轮）
3. 添加学习率调度器 + Cosine Annealing
4. 添加 Early Stopping
5. 启用混合精度训练（AMP）
6. 类别权重处理不平衡
7. 详细的训练日志 (TrainingLogger)
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.stdout.reconfigure(encoding='utf-8')

from processing.preprocess import DataPreprocessor
from models.stage2_transeca import TransECANet
from utils.training_logger import TrainingLogger


class EarlyStopping:
    """Early Stopping 机制"""
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_acc = 0
        
    def __call__(self, val_loss, val_acc, model, path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_acc = val_acc
            self.save_checkpoint(model, path)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_acc = val_acc
            self.save_checkpoint(model, path)
            self.counter = 0
            
    def save_checkpoint(self, model, path):
        if self.verbose:
            print(f'✓ Validation loss decreased ({self.best_loss:.4f}). Saving model...')
        torch.save(model.state_dict(), path)


def plot_training_history(history, save_path='results/stage2_training_history.png'):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='results/stage2_confusion_matrix.png'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Stage 2: TransECA-Net Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")


def main():
    print("="*80)
    print("Stage 2 优化训练流程")
    print("="*80)
    
    # 初始化训练记录器
    logger = TrainingLogger(
        experiment_name="stage2_transeca_optimized",
        description="Stage 2 TransECA-Net multiclass classifier, optimized training with LR scheduler, early stopping, AMP"
    )
    logger.start()
    
    # ======================== 1. 数据加载 ========================
    print("\n[1/7] Loading Stage 2 data from data/stage2/*.parquet ...")
    
    import joblib
    
    try:
        df_train = pd.read_parquet("data/stage2/train.parquet")
        df_val   = pd.read_parquet("data/stage2/val.parquet")
        df_test  = pd.read_parquet("data/stage2/test.parquet")
    except FileNotFoundError as e:
        print(f"  ✗ Stage 2 data not found: {e}")
        print("    Please run train_stage1.py first to generate Stage 2 data.")
        logger.finish(status="failed")
        return
    
    print(f"  ✓ Train: {df_train.shape[0]:,} samples, {df_train.shape[1]} columns")
    print(f"  ✓ Val:   {df_val.shape[0]:,} samples")
    print(f"  ✓ Test:  {df_test.shape[0]:,} samples")
    print(f"  ✓ Total: {df_train.shape[0] + df_val.shape[0] + df_test.shape[0]:,} samples")
    print(f"\n  Train Label Distribution:")
    for label, count in df_train['Label'].value_counts().items():
        print(f"    {label:>30s}: {count:>7,}")
    
    # ======================== 2. 数据预处理 ========================
    print("\n[2/7] Preprocessing data ...")
    
    # 使用 Stage 1 的预处理器保持一致性
    preprocessor = None
    for pp_path in ["models_chk/preprocessor_stratified.joblib", "models_chk/preprocessor.joblib"]:
        try:
            preprocessor = joblib.load(pp_path)
            print(f"  ✓ Loaded preprocessor from {pp_path}")
            break
        except Exception:
            continue
    
    if preprocessor is None:
        print("  ⚠ No saved preprocessor found. Creating new one.")
        preprocessor = DataPreprocessor()
        df_train_clean = preprocessor.clean(df_train)
        X_train, y_train_enc = preprocessor.fit_transform(df_train_clean, target_col='Label')
        joblib.dump(preprocessor, "models_chk/preprocessor_stage2.joblib")
    else:
        df_train_clean = preprocessor.clean(df_train)
        X_train, y_train_enc = preprocessor.transform(df_train_clean, target_col='Label')
    
    df_val_clean = preprocessor.clean(df_val)
    X_val, y_val_enc = preprocessor.transform(df_val_clean, target_col='Label')
    
    df_test_clean = preprocessor.clean(df_test)
    X_test, y_test_enc = preprocessor.transform(df_test_clean, target_col='Label')
    
    # 过滤无效标签 (-1 = unseen by label encoder)
    for name, X_arr, y_arr in [("Train", X_train, y_train_enc), ("Val", X_val, y_val_enc), ("Test", X_test, y_test_enc)]:
        invalid = (y_arr < 0).sum() if y_arr is not None else 0
        if invalid > 0:
            print(f"  ⚠ {name}: removing {invalid} samples with unseen labels")
    
    valid_train = y_train_enc >= 0
    valid_val = y_val_enc >= 0
    valid_test = y_test_enc >= 0
    X_train, y_train_enc = X_train[valid_train], y_train_enc[valid_train]
    X_val, y_val_enc = X_val[valid_val], y_val_enc[valid_val]
    X_test, y_test_enc = X_test[valid_test], y_test_enc[valid_test]
    
    # 重新映射类别 (Stage 2 数据可能不包含所有原始类别)
    all_labels = np.concatenate([y_train_enc, y_val_enc, y_test_enc])
    unique_classes = np.unique(all_labels)
    class_names_all = [str(c) for c in preprocessor.label_encoder.classes_]
    class_names = [class_names_all[i] for i in unique_classes]
    num_classes = len(unique_classes)
    num_features = X_train.shape[1]
    
    if num_classes < len(preprocessor.label_encoder.classes_):
        print(f"  ℹ Remapping: {num_classes}/{len(preprocessor.label_encoder.classes_)} classes present in Stage 2 data")
        class_map = {old: new for new, old in enumerate(unique_classes)}
        y_train_enc = np.array([class_map[y] for y in y_train_enc])
        y_val_enc = np.array([class_map[y] for y in y_val_enc])
        y_test_enc = np.array([class_map[y] for y in y_test_enc])
    
    print(f"\n  ✓ Features: {num_features}")
    print(f"  ✓ Classes:  {num_classes}")
    print(f"  ✓ Mapping:  {dict(enumerate(class_names))}")
    print(f"  ✓ Final: Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")
    
    # 计算类别权重 (处理不平衡)
    class_counts = np.bincount(y_train_enc, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = np.clip(class_weights, None, np.median(class_weights) * 50)  # cap extreme weights
    class_weights = class_weights / class_weights.sum() * num_classes  # normalize
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"  ✓ Class weights computed (max ratio: {class_weights.max()/class_weights.min():.1f}x)")
    
    # 记录数据信息
    logger.set_data_info(
        dataset="CIC-IDS2017 (Stage 2 Hard Examples)",
        data_path="data/stage2/",
        total_samples=len(X_train) + len(X_val) + len(X_test),
        train_samples=len(X_train),
        val_samples=len(X_val),
        test_samples=len(X_test),
        num_features=num_features,
        num_classes=num_classes,
        class_distribution={class_names[i]: int(c) for i, c in enumerate(class_counts)},
        split_method="Stage 1 CV Mining + Prediction",
        split_ratios="Pre-split from Stage 1 (70/10/20)"
    )
    
    # ======================== 3. DataLoader ========================
    print("\n[3/7] Preparing DataLoaders ...")
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_enc, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_enc, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_enc, dtype=torch.long)
    
    batch_size = 512
    num_workers = 4  # GPU内存充足时可用多线程加载
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    print(f"  ✓ Batch size: {batch_size}")
    print(f"  ✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # ======================== 4. 模型初始化 ========================
    print("\n[4/7] Initializing model ...")
    device = torch.device("cpu")
    
    # Priority 1: CUDA (NVIDIA)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  ✓ Device: CUDA — {torch.cuda.get_device_name(0)}")
        print(f"    Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    # Priority 2: XPU (Intel ARC — PyTorch 2.10+ 原生支持)
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu")
        props = torch.xpu.get_device_properties(0)
        print(f"  ✓ Device: XPU — {torch.xpu.get_device_name(0)}")
        print(f"    Memory: {props.total_memory / 1024**3:.1f} GB | EUs: {props.gpu_eu_count}")
    else:
        print("  ⚠ Device: CPU (no GPU acceleration)")

    
    model = TransECANet(
        num_features=num_features, 
        num_classes=num_classes,
        d_model=128,  # 从64增加到128（更强表达能力）
        nhead=8,      # 从4增加到8
        num_layers=3  # 从2增加到3
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Model Parameters: {total_params:,} (Trainable: {trainable_params:,})")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度器
    epochs = 30
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # 混合精度训练（GPU only: CUDA or XPU）
    use_amp = device.type in ('cuda', 'xpu')
    amp_device = device.type if use_amp else 'cpu'
    scaler = torch.amp.GradScaler(amp_device) if use_amp else None
    if use_amp:
        print(f"  ✓ Mixed Precision Training (AMP) on {amp_device}")
    
    # 记录模型和训练配置
    logger.set_model_info(
        model_type="TransECA-Net",
        model_name="Stage 2 Multiclass Classifier",
        architecture="1D-CNN + ECA Attention + Transformer Encoder",
        framework="PyTorch",
        hyperparams={"d_model": 128, "nhead": 8, "num_layers": 3},
        total_params=total_params,
        trainable_params=trainable_params,
    )
    logger.set_training_config(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.001,
        optimizer="AdamW (weight_decay=1e-4)",
        scheduler="CosineAnnealingWarmRestarts (T_0=10, T_mult=2)",
        loss_function="CrossEntropyLoss (class-weighted)",
        device=str(device),
        early_stopping=True,
        early_stopping_patience=10,
    )
    
    # ======================== 5. 训练循环 ========================
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"\n[5/7] Training — {epochs} epochs, batch_size={batch_size}")
    print(f"{'='*80}")
    
    start_time = time.time()
    num_train_batches = len(train_loader)
    
    for epoch in range(epochs):
        # -------- Training --------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / (batch_idx + 1) * (num_train_batches - batch_idx - 1)
                print(f"    Epoch {epoch+1} batch {batch_idx+1}/{num_train_batches} "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
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
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印进度
        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {lr_now:.6f}", flush=True)
        
        # 记录到 logger
        logger.log_epoch(
            epoch=epoch + 1,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
            train_acc=round(train_acc, 2),
            val_acc=round(val_acc, 2),
            learning_rate=lr_now,
        )
        
        # 学习率调度
        scheduler.step(epoch + 1)
        
        # Early Stopping
        early_stopping(val_loss, val_acc, model, "models_chk/stage2_transeca_best.pth")
        if early_stopping.early_stop:
            print(f"\n  ✓ Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"  Training completed in {training_time/60:.2f} min | "
          f"Best Val Acc: {early_stopping.best_acc:.2f}% | "
          f"Epochs: {epoch+1}/{epochs}")
    print(f"{'='*80}")
    
    # ======================== 6. 加载最佳模型并测试 ========================
    print(f"\n[6/7] Evaluating on test set ({len(X_test):,} samples) ...")
    
    model.load_state_dict(torch.load("models_chk/stage2_transeca_best.pth", weights_only=True))
    model.eval()
    
    # Batch inference for large test set
    all_preds = []
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred_batch = torch.max(outputs, 1)
            all_preds.append(pred_batch.cpu().numpy())
    
    preds = np.concatenate(all_preds)
    y_test = y_test_enc
    
    # ======================== 7. 评估与可视化 ========================
    print("\n" + "="*80)
    print("  Test Set Classification Report")
    print("="*80)
    report = classification_report(y_test, preds, target_names=class_names)
    print(report)
    
    # 保存报告
    print(f"[7/7] Saving results ...")
    os.makedirs("results", exist_ok=True)
    with open("results/stage2_optimized_report.txt", "w", encoding='utf-8') as f:
        f.write(f"Stage 2 (TransECA-Net) — Optimized Training Report\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Data: Stage 1 Hard Examples (data/stage2/*.parquet)\n")
        f.write(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}\n")
        f.write(f"  Features: {num_features} | Classes: {num_classes}\n\n")
        f.write(f"Model: TransECA-Net (d=128, h=8, L=3)\n")
        f.write(f"  Parameters: {total_params:,}\n\n")
        f.write(f"Training: {epoch+1}/{epochs} epochs, BS={batch_size}\n")
        f.write(f"  Time: {training_time/60:.2f} min\n")
        f.write(f"  Best Val Acc: {early_stopping.best_acc:.2f}%\n\n")
        f.write(f"{'='*80}\n")
        f.write("Test Classification Report\n")
        f.write(f"{'='*80}\n\n")
        f.write(report)
    
    print(f"  ✓ Report → results/stage2_optimized_report.txt")
    
    # 绘制训练曲线
    plot_training_history(history)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(y_test, preds, class_names)
    
    # 保存最终模型
    torch.save(model.state_dict(), "models_chk/stage2_transeca.pth")
    print(f"  ✓ Model → models_chk/stage2_transeca.pth")
    
    # 记录结果和产物
    logger.set_results(
        best_val_accuracy=early_stopping.best_acc,
        training_time_minutes=round(training_time / 60, 2),
        total_epochs_run=epoch + 1,
        classification_report=report,
    )
    logger.add_artifact("models_chk/stage2_transeca_best.pth", "model", "Best model (early stopping)")
    logger.add_artifact("models_chk/stage2_transeca.pth", "model", "Final model")
    logger.add_artifact("results/stage2_optimized_report.txt", "report", "Classification report")
    logger.add_artifact("results/stage2_training_history.png", "plot", "Training curves")
    logger.add_artifact("results/stage2_confusion_matrix.png", "plot", "Confusion matrix")
    logger.finish()
    
    print("\n" + "="*80)
    print("  Stage 2 Training Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
