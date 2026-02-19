
"""
实验 E8: TransECA-Net 消融实验 (Ablation Study)
目标: 验证各个组件 (CNN, ECA, Transformer) 的贡献。
Variations:
1. Full Model (TransECA-Net)
2. No-ECA (CNN + Transformer)
3. CNN-Only (No Transformer, No ECA)
"""
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Add src
sys.path.append(os.path.join(os.getcwd(), 'src'))
from models.stage2_transeca import TransECANet
from processing.loader_stratified import load_stratified_mixed_split

# Define Variants
class TransNet_NoECA(TransECANet):
    def __init__(self, num_features, num_classes, d_model=64):
        super().__init__(num_features, num_classes, d_model=d_model)
        # Override ECA with Identity
        self.eca = nn.Identity()

class CNN_Only(nn.Module):
    def __init__(self, num_features, num_classes, d_model=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

def train_and_eval(model_name, model, train_loader, test_loader, device, epochs=5):
    print(f"\n--- Training {model_name} ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")
        
    # Eval
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            pred = outputs.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            truths.extend(y_batch.numpy())
            
    print(f"Results for {model_name}:")
    print(classification_report(truths, preds, digits=4))
    
    # Return Weighted F1
    from sklearn.metrics import f1_score
    return f1_score(truths, preds, average='weighted')

def main():
    print("Running E8 Ablation Study...")
    
    # Load Data (Sampled for speed in this demo script)
    # in Real E8, use full dataset
    try:
        # Load Stratified Split (we only need df_train to split into train/val)
        # Using larger ratio for meaningful results
        df, _ = load_stratified_mixed_split("archive", 
                                           benign_monday_ratio=0.1, 
                                           benign_other_ratio=0.1, 
                                           attack_ratio=0.1)
    except:
        return

    # Preprocess
    preprocessor = joblib.load("models_chk/preprocessor.joblib")
    df = preprocessor.clean(df)
    X, y = preprocessor.transform(df, target_col='Label')
    
    num_features = X.shape[1]
    num_classes = len(preprocessor.label_encoder.classes_)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    # Tensor
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Models
    models = {
        "Full TransECA": TransECANet(num_features, num_classes),
        "No-ECA": TransNet_NoECA(num_features, num_classes),
        "CNN-Only": CNN_Only(num_features, num_classes)
    }
    
    results = {}
    for name, model in models.items():
        score = train_and_eval(name, model, train_loader, test_loader, device)
        results[name] = score
        
    print("\n--- Ablation Summary (Weighted F1) ---")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
