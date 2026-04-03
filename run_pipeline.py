
import sys
import os
import pandas as pd
import numpy as np
import torch
import time
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.stdout.reconfigure(encoding='utf-8')

from processing.loader import load_data
from processing.preprocess import DataPreprocessor
from models.hybrid import HierarchicalIDS

def main():
    # 1. Load Mixed Data (Benign + various Attacks)
    data_dir = "archive"
    print("Loading Test Data for Pipeline...")
    
    # We load a sample of everything for evaluation
    subsets = ["Benign", "DoS", "PortScan", "WebAttacks"]
    dfs = []
    for s in subsets:
        try:
            df_tmp = load_data(data_dir, subset=s)
            if len(df_tmp) > 5000: df_tmp = df_tmp.sample(5000, random_state=42)
            dfs.append(df_tmp)
        except: pass
        
    if not dfs: return
    df = pd.concat(dfs, ignore_index=True)
    
    # 2. Preprocess
    # Load saved preprocessor to ensure consistent scaling
    try:
        import joblib
        preprocessor = joblib.load("models_chk/preprocessor.joblib")
        print("Loaded saved preprocessor.")
        
        # We need to handle if 'Label' is unexpected in transform?
        # preprocessor.transform logic checks target_col.
        # But wait, preprocessor.label_encoder was fitted on TRAIN labels.
        # Test labels might contain new classes or subset.
        # The 'transform' method in 'preprocess.py' handles target_col.
        
        # We assume labels in test set are subset of train set for encoder to work.
        # If not, it might fail. 'train_stage1' trained on mixed attacks so it should be fine.
        
        df = preprocessor.clean(df)
        X_scaled, y_enc = preprocessor.transform(df, target_col='Label')
        
    except Exception as e:
        print(f"Error using saved preprocessor: {e}. Fallback to fitting (Bad for accuracy).")
        preprocessor = DataPreprocessor()
        df = preprocessor.clean(df)
        X_scaled, y_enc = preprocessor.fit_transform(df, target_col='Label')
    
    # Check classes
    classes = preprocessor.label_encoder.classes_
    print(f"Evaluation Classes: {classes}")
    
    # 3. Initialize Hybrid Model
    stage1_path = "models_chk/stage1_rf.joblib"
    stage2_path = "models_chk/stage2_transeca.pth"
    
    if not os.path.exists(stage1_path) or not os.path.exists(stage2_path):
        print("Models not ready yet.")
        return

    # 从训练时保存的配置文件读取 Stage 2 参数，避免从测试数据推断 num_classes 导致维度不匹配
    import json
    config_path = "models_chk/stage2_config.json"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
        stage2_params = {
            "num_features": model_config["num_features"],
            "num_classes": model_config["num_classes"],
            "d_model": model_config.get("d_model", 128),
            "nhead": model_config.get("nhead", 8),
            "num_layers": model_config.get("num_layers", 3),
        }
        print(f"Loaded Stage 2 config: {model_config['num_classes']} classes, {model_config['num_features']} features")
    else:
        # 兼容旧版本（无配置文件时从当前数据推断，存在类别不匹配风险）
        print("Warning: stage2_config.json not found, inferring num_classes from test data (may mismatch training).")
        stage2_params = {
            "num_features": X_scaled.shape[1],
            "num_classes": len(classes),
        }
    
    ids = HierarchicalIDS(stage1_path, stage2_path, stage2_params)
    
    # 4. Run Prediction
    X_torch = torch.tensor(X_scaled, dtype=torch.float32)
    
    start_time = time.time()
    preds = ids.predict(X_scaled, X_torch)
    end_time = time.time()
    
    # 5. Metrics
    print("\n=== Hierarchical IDS Performance ===")
    print(f"Latency: {end_time - start_time:.4f} seconds for {len(df)} samples")
    print(f"Throughput: {len(df) / (end_time - start_time):.2f} samples/sec")
    
    print("\nClassification Report:")
    print(classification_report(y_enc, preds, target_names=[str(c) for c in classes]))

if __name__ == "__main__":
    main()
