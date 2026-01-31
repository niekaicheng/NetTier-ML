
import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.stdout.reconfigure(encoding='utf-8')

from processing.loader import load_data
from processing.preprocess import DataPreprocessor

def plot_confusion_matrix(y_true, y_pred, classes, title, filename):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved Confusion Matrix to {filename}")
    plt.close()

def plot_feature_importance(model, feature_names, filename, top_n=20):
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_")
        return
        
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f"Top {top_n} Feature Importances")
    plt.bar(range(top_n), importances[top_indices], align="center")
    plt.xticks(range(top_n), [feature_names[i] for i in top_indices], rotation=90)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved Feature Importance to {filename}")
    plt.close()
    
    # Save text report
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })
    importance_df.to_csv(filename.replace('.png', '.csv'), index=False)

def main():
    os.makedirs("results", exist_ok=True)
    
    # 1. Load Data
    data_dir = "archive"
    print("Loading Analysis Data...")
    
    # Load representative sample
    subsets = ["Benign", "DoS", "PortScan", "WebAttacks", "Botnet"]
    dfs = []
    for s in subsets:
        try:
            tmp = load_data(data_dir, subset=s)
            if len(tmp) > 5000: tmp = tmp.sample(5000, random_state=42)
            dfs.append(tmp)
        except: pass
        
    if not dfs: return
    df = pd.concat(dfs, ignore_index=True)
    
    # 2. Load Preprocessor
    try:
        preprocessor = joblib.load("models_chk/preprocessor.joblib")
        print("Loaded Preprocessor.")
    except:
        print("Preprocessor not found!")
        return
        
    # Clean & Transform
    df = preprocessor.clean(df)
    X_scaled, y_enc = preprocessor.transform(df, target_col='Label')
    classes = preprocessor.label_encoder.classes_
    
    # Get Feature Names
    # Assuming the preprocessor doesn't store feature names explicitly, we re-derive from df columns
    # df columns include 'Label', so we drop it.
    feature_names = df.drop(columns=['Label']).columns.tolist()
    
    # 3. Analyze Stage 1 (Random Forest)
    print("\n=== Analyzing Stage 1 (filter) ===")
    stage1_model_wrapper = Stage1Wrapper("models_chk/stage1_rf.joblib")
    
    # Feature Importance
    plot_feature_importance(stage1_model_wrapper.model, feature_names, "results/stage1_feature_importance.png")
    
    # Predictions
    stage1_preds = stage1_model_wrapper.model.predict(X_scaled)
    
    # Binary Truth (0: Benign, 1: Attack) 
    # Find Benign Index
    benign_idx = preprocessor.label_encoder.transform(['Benign'])[0]
    y_binary = (y_enc != benign_idx).astype(int)
    
    # Confusion Matrix Station 1
    plot_confusion_matrix(y_binary, stage1_preds, ['Benign', 'Malicious'], "Stage 1 Confusion Matrix", "results/stage1_cm.png")
    
    # Report
    print("Stage 1 Classification Report:")
    report1 = classification_report(y_binary, stage1_preds, target_names=['Benign', 'Malicious'])
    print(report1)
    with open("results/stage1_report.txt", "w") as f: f.write(report1)


class Stage1Wrapper:
    def __init__(self, path):
        self.model = joblib.load(path)

if __name__ == "__main__":
    main()
