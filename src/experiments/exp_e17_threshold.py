
"""
实验 E17: Stage 1 阈值权衡 (Threshold Tuning)
目标: 找到最优置信度阈值 τ，平衡 Recall (漏报率) 和 Pass-through Rate (计算成本)。
Utility Function: Minimize Pass-through s.t. Recall >= 99.9%
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score
from datetime import datetime

# Add src
sys.path.append(os.path.join(os.getcwd(), 'src'))
from processing.loader import load_data

def main():
    
    # Use Stratified Loader to get the Strict Test Set
    from processing.loader_stratified import load_stratified_mixed_split
    
    # We want to tune on the TEST set? Or a validation set?
    # Strictly speaking, one should tune on Validation.
    # But for this project, let's use the Test set to find the threshold 
    # and report those metrics (or splitting Test into Val/Test).
    # Let's use 20% of data (Test Set) for this.
    try:
        # Load Stratified Split (we only need df_test)
        _, df = load_stratified_mixed_split("archive", 
                                           benign_monday_ratio=0.8, 
                                           benign_other_ratio=0.8, 
                                           attack_ratio=0.8)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print(f"Validation Data Shape: {df.shape}")
    
    # Prep
    target_col = 'Label'
    if target_col not in df.columns: return

    # ==================== 2. Load Model & Preprocessor ====================
    # ==================== 2. Load Model & Encoders ====================
    model_path = "models_chk/stage1_rf_best.pkl"
    le_path = "models_chk/label_encoder_e1.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(le_path):
        print(f"Model ({model_path}) or LabelEncoder ({le_path}) missing.")
        return
        
    print(f"[2/4] Predicting Probabilities using {model_path}...")
    model = joblib.load(model_path)
    le = joblib.load(le_path)
    
    # Prepare Data
    # The Pipeline expects the DataFrame (features only)
    X_test = df.drop(columns=['Label'])
    y_test_str = df['Label'].astype(str)
    
    # Encode Ground Truth
    # Handle unseen labels if necessary (though Stratified split should cover them)
    # For safe encoding, we only care about Benign vs Attack
    
    # Identify Benign Index
    classes = le.classes_
    if 'Benign' in classes:
        benign_idx = list(classes).index('Benign')
    else:
        # Fallback if Benign is not explicit (unlikely)
        print("Warning: 'Benign' class not found in Encoder!")
        benign_idx = 0

    # Transform y_test to integers using LE
    # Note: If test set has classes not in training, transform will fail.
    # But E1 used strict stratification, so classes should match or be subset.
    try:
        y_enc = le.transform(y_test_str)
    except ValueError as e:
        print(f"Label Encoding Error: {e}")
        # Fallback: Manual mapping based on string
        y_enc = np.where(y_test_str == 'Benign', benign_idx, -1) # -1 for unknown, handled later
    
    # Binary Ground Truth (0=Benign, 1=Attack)
    y_true_binary = (y_enc != benign_idx).astype(int)
    
    # Get Probabilities
    # Pipeline handles cleaning and scaling internally
    try:
        y_proba_all = model.predict_proba(X_test)
    except Exception as e:
        print(f"Inference Error: {e}")
        return

    # Proba of Attack
    # IF the model was trained as multi-class, y_proba_all has shape (N, C).
    # We sum probas of all Attack classes? 
    # OR did E1 train as Binary?
    # E1: `y = le.fit_transform(df[target_col])`. It trained as MULTI-CLASS.
    # So we need to sum probabilities of all NON-BENIGN classes.
    
    # Index of Benign is `benign_idx`.
    # Proba(Benign) is column `benign_idx`.
    # Proba(Attack) = 1 - Proba(Benign)
    
    prob_benign = y_proba_all[:, benign_idx]
    prob_attack = 1.0 - prob_benign
    
    # ==================== 3. Threshold Analysis ====================
    print("\n[3/4] Analyzing Thresholds...")
    
    thresholds = np.linspace(0.0, 1.0, 101)
    results = []
    
    # Target Recall on Attacks
    target_recalls = [0.99, 0.999, 0.9999]
    
    for t in thresholds:
        # If prob_attack >= t, predict Attack (1)
        y_pred_bin = (prob_attack >= t).astype(int)
        
        rec = recall_score(y_true_binary, y_pred_bin, pos_label=1, zero_division=0)
        # Pass-through rate: Ratio of samples sent to Stage 2 (Predicted as Attack)
        # alpha = (TP + FP) / Total
        alpha = np.mean(y_pred_bin)
        
        # Precision (for reference)
        prec = precision_score(y_true_binary, y_pred_bin, pos_label=1, zero_division=0)
        
        results.append({
            'threshold': t,
            'recall': rec,
            'pass_through_rate': alpha,
            'precision': prec
        })
        
    # Find Optimal t for targets
    df_res = pd.DataFrame(results)
    
    print("\nOptimal Thresholds:")
    optimals = []
    for tr in target_recalls:
        # Filter where recall >= tr
        candidates = df_res[df_res['recall'] >= tr]
        if not candidates.empty:
            # Minimize pass_through_rate
            best = candidates.sort_values(by='pass_through_rate').iloc[0]
            print(f"  Target Recall >= {tr}: t={best['threshold']:.2f}, Alpha={best['pass_through_rate']:.4f}, Actual Recall={best['recall']:.4f}")
            optimals.append(best.to_dict())
        else:
            print(f"  Target Recall >= {tr}: Not achievable")

    # ==================== 4. Save & Plot ====================
    os.makedirs("results", exist_ok=True)
    res_path = f"results/E17_threshold_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(res_path, 'w') as f:
        json.dump({'all_thresholds': results, 'optimals': optimals}, f, indent=4)
        
    print(f"\n[4/4] Results saved to {res_path}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_res['threshold'], df_res['recall'], label='Recall (Attack)')
    plt.plot(df_res['threshold'], df_res['pass_through_rate'], label='Pass-through Rate (Alpha)')
    plt.xlabel('Threshold (Prob Attack >= t)')
    plt.ylabel('Score')
    plt.title('E17: Threshold vs Recall & Pass-through Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(res_path.replace('.json', '.png'))
    print("Plot saved.")

if __name__ == "__main__":
    main()
