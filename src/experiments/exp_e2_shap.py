
"""
实验 E2: 特征重要性分析 (SHAP)
目标: 解释 Random Forest 模型的决策依据。
参考文献: [Abu Al-Haija'22]
"""
import sys
import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Add src
sys.path.append(os.path.join(os.getcwd(), 'src'))
from processing.loader import load_data

def main():
    print("Running E2: SHAP Analysis...")
    
    # Load Model
    model_path = "models_chk/stage1_rf_stratified.joblib"
    prep_path = "models_chk/preprocessor_stratified.joblib"
    
    if not os.path.exists(model_path):
        print("Model not found. Run Stage 1 training first.")
        return
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    
    # Load Sample Data (Background for SHAP)
    # SHAP needs a background dataset to simulate "missing" features.
    print("Loading Background Data...")
    try:
        # Load a small balanced sample
        df_benign = load_data("archive", subset="Benign").sample(100, random_state=42)
        df_dos = load_data("archive", subset="DoS").sample(100, random_state=42)
        df = pd.concat([df_benign, df_dos], ignore_index=True)
    except:
        print("Data loading failed.")
        return
        
    # Preprocess
    df = preprocessor.clean(df)
    X, _ = preprocessor.transform(df, target_col='Label')
    
    # SHAP Explainer
    # TreeExplainer is optimized for Trees (RF, XGBoost)
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Plot
    print("Generating Plots...")
    os.makedirs("results", exist_ok=True)
    
    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=preprocessor.feature_cols, show=False)
    plt.savefig("results/E2_shap_summary.png", bbox_inches='tight')
    plt.close()
    
    print("SHAP Summary saved to results/E2_shap_summary.png")

if __name__ == "__main__":
    main()
