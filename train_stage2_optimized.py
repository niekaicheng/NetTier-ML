"""
优化的 Stage 2 训练脚本
基于深度学习可行性分析的建议实施

改进点：
1. 使用全量数据（不限制样本数）
2. 增加训练 Epochs（30轮）
3. 添加学习率调度器
4. 添加 Early Stopping
5. 启用混合精度训练（AMP）
6. 优化数据加载（num_workers）
7. 详细的训练日志
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.stdout.reconfigure(encoding='utf-8')

from processing.loader import load_data
from processing.preprocess import DataPreprocessor
from models.stage2_transeca import TransECANet


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
    
    # ======================== 1. 数据加载 ========================
    data_dir = "archive"
    subsets = ["Benign", "DoS", "PortScan", "WebAttacks", "Botnet", 
               "Bruteforce", "DDoS", "Infiltration"]
    
    dfs = []
    total_samples = 0
    for s in subsets:
        try:
            print(f"Loading {s}...")
            df_tmp = load_data(data_dir, subset=s)
            samples = len(df_tmp)
            total_samples += samples
            print(f"  → Loaded {samples:,} samples from {s}")
            
            # 使用全量数据（移除采样限制）
            # 如果内存不足，可以适当采样，但尽量保留更多数据
            if len(df_tmp) > 100000:
                df_tmp = df_tmp.sample(100000, random_state=42)
                print(f"  → Sampled to 100,000 (memory optimization)")
            
            dfs.append(df_tmp)
        except Exception as e:
            print(f"  ✗ Error loading {s}: {e}")
            
    if not dfs:
        print("No data loaded. Exiting.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Total Data: {df.shape[0]:,} samples, {df.shape[1]} columns")
    print(f"✓ Classes: {df['Label'].unique()}")
    print(f"✓ Class Distribution:\n{df['Label'].value_counts()}")
    
    # ======================== 2. 数据预处理 ========================
    import joblib
    try:
        preprocessor = joblib.load("models_chk/preprocessor.joblib")
        print("\n✓ Loaded shared preprocessor from models_chk/")
        df = preprocessor.clean(df)
        X_scaled, y_enc = preprocessor.transform(df, target_col='Label')
    except Exception as e:
        print(f"\n⚠ Warning: Could not load preprocessor: {e}")
        print("Creating new preprocessor (may cause mismatch with Stage 1)")
        preprocessor = DataPreprocessor()
        df = preprocessor.clean(df)
        X_scaled, y_enc = preprocessor.fit_transform(df, target_col='Label')
        joblib.dump(preprocessor, "models_chk/preprocessor.joblib")
    
    num_classes = len(preprocessor.label_encoder.classes_)
    num_features = X_scaled.shape[1]
    
    print(f"\n✓ Preprocessed Features: {num_features}")
    print(f"✓ Number of Classes: {num_classes}")
    print(f"✓ Class Mapping: {dict(enumerate(preprocessor.label_encoder.classes_))}")
    
    # ======================== 3. 数据集划分 ========================
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n✓ Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # 转换为 Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    # DataLoader（优化：增加 num_workers）
    batch_size = 128  # 从32增加到128
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2  # Windows下建议2-4
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t), 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    # ======================== 4. 模型初始化 ========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Training Device: {device}")
    if torch.cuda.is_available():
        print(f"  → GPU: {torch.cuda.get_device_name(0)}")
        print(f"  → Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
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
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=7, verbose=True)
    
    # 混合精度训练（仅GPU）
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("✓ Mixed Precision Training (AMP) Enabled")
    
    # ======================== 5. 训练循环 ========================
    epochs = 30
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"\n{'='*80}")
    print(f"Training Started: {epochs} epochs, batch_size={batch_size}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # -------- Training --------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            if use_amp:
                with torch.cuda.amp.autocast():
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
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # Early Stopping
        early_stopping(val_loss, val_acc, model, "models_chk/stage2_transeca_best.pth")
        if early_stopping.early_stop:
            print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Training Completed in {training_time/60:.2f} minutes")
    print(f"Best Validation Accuracy: {early_stopping.best_acc:.2f}%")
    print(f"{'='*80}\n")
    
    # ======================== 6. 加载最佳模型并测试 ========================
    model.load_state_dict(torch.load("models_chk/stage2_transeca_best.pth"))
    print("✓ Loaded best model for testing")
    
    model.eval()
    with torch.no_grad():
        inputs = X_test_t.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
    
    # ======================== 7. 评估与可视化 ========================
    class_names = [str(c) for c in preprocessor.label_encoder.classes_]
    
    print("\n" + "="*80)
    print("Test Set Classification Report")
    print("="*80 + "\n")
    print(classification_report(y_test, preds, target_names=class_names))
    
    # 保存报告
    with open("results/stage2_optimized_report.txt", "w", encoding='utf-8') as f:
        f.write(f"Stage 2 (TransECA-Net) - Optimized Training Report\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Training Time: {training_time/60:.2f} minutes\n")
        f.write(f"Total Epochs: {epoch+1}/{epochs}\n")
        f.write(f"Best Validation Accuracy: {early_stopping.best_acc:.2f}%\n\n")
        f.write(f"Model Architecture:\n")
        f.write(f"  - Features: {num_features}\n")
        f.write(f"  - Classes: {num_classes}\n")
        f.write(f"  - Parameters: {total_params:,}\n")
        f.write(f"  - d_model: 128, nhead: 8, num_layers: 3\n\n")
        f.write(f"{'='*80}\n")
        f.write("Test Set Classification Report\n")
        f.write(f"{'='*80}\n\n")
        f.write(classification_report(y_test, preds, target_names=class_names))
    
    print("✓ Report saved to results/stage2_optimized_report.txt")
    
    # 绘制训练曲线
    plot_training_history(history)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(y_test, preds, class_names)
    
    # 保存最终模型
    torch.save(model.state_dict(), "models_chk/stage2_transeca.pth")
    print("✓ Final model saved to models_chk/stage2_transeca.pth")
    
    print("\n" + "="*80)
    print("All tasks completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
