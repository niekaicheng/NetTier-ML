
import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.stdout.reconfigure(encoding='utf-8')

from processing.loader import load_data
from processing.preprocess import DataPreprocessor
from models.stage2_transeca import TransECANet

def main():
    # Load Multiple Attack Types for Multiclass
    data_dir = "archive"
    # Match Stage 1 subsets + Benign
    subsets = ["Benign", "DoS", "PortScan", "WebAttacks", "Botnet"]
    
    dfs = []
    for s in subsets:
        try:
            print(f"Loading {s}...")
            df_tmp = load_data(data_dir, subset=s)
            if len(df_tmp) > 5000: df_tmp = df_tmp.sample(5000, random_state=42)
            dfs.append(df_tmp)
        except:
            pass
            
    if not dfs:
        print("No data loaded.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"Total Data: {df.shape}")
    print(f"Classes: {df['Label'].unique()}")
    
    # Preprocess
    # Load shared preprocessor
    import joblib
    try:
        preprocessor = joblib.load("models_chk/preprocessor.joblib")
        print("Loaded shared preprocessor.")
        df = preprocessor.clean(df)
        X_scaled, y_enc = preprocessor.transform(df, target_col='Label')
    except Exception as e:
        print(f"Error loading preprocessor: {e}. Create a new one (Mismatch Risk!)")
        preprocessor = DataPreprocessor()
        df = preprocessor.clean(df)
        X_scaled, y_enc = preprocessor.fit_transform(df, target_col='Label')
    
    num_classes = len(preprocessor.label_encoder.classes_)
    num_features = X_scaled.shape[1]
    
    print(f"Features: {num_features}, Classes: {num_classes}")
    print(f"Class Map: {dict(enumerate(preprocessor.label_encoder.classes_))}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42)
    
    # To Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    model = TransECANet(num_features=num_features, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train Loop
    epochs = 1 # Quick test
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
        
    # Save FIRST
    torch.save(model.state_dict(), "models_chk/stage2_transeca.pth")
    print("Model saved.")

    # Evaluate
    model.eval()
    with torch.no_grad():
        inputs = X_test_t.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
        
    try:
        print(classification_report(y_test, preds, target_names=[str(c) for c in preprocessor.label_encoder.classes_]))
    except Exception as e:
        print(f"Eval warning: {e}")

if __name__ == "__main__":
    main()
