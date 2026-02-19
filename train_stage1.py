
import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# Force utf-8 for Windows consoles to avoid charmap errors
sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.join(os.getcwd(), 'src'))

from processing.loader_stratified import load_stratified_mixed_split
from processing.preprocess import DataPreprocessor
from models.stage1_rf import Stage1Filter

def main():
    data_dir = "archive"
    
    # 1. Load Data using Stratified Mixed Split (User Request)
    # - 80% of Monday Benign + 80% other Benign -> Train Baseline
    # - 70-80% of ALL Attacks -> Train Coverage
    # - Rest (20-30%) -> Strict Test Isolation
    try:
        print("Loading data with Stratified Mixed Split...")
        # Using 80% for all as a good baseline
        df_train, df_test = load_stratified_mixed_split(
            data_dir, 
            benign_monday_ratio=0.8,
            benign_other_ratio=0.8,
            attack_ratio=0.8
        )
        
        print(f"Initial Train Shape: {df_train.shape} (80%)")
        print(f"Initial Test Shape: {df_test.shape} (20%)")

        # User Request: Strict separation. Split 80% Train into -> 70% Train + 10% Val.
        print(f"Splitting Train (80%) into Train (70%) and Val (10%)...")
        df_train_final, df_val = train_test_split(
            df_train, 
            test_size=0.125, # 0.125 * 0.8 = 0.1
            stratify=df_train['Label'], 
            random_state=42
        )
        # Update ref
        df_train = df_train_final
        
        print(f"Final Shapes:")
        print(f"- Train: {df_train.shape} (70%)")
        print(f"- Val:   {df_val.shape}   (10%)")
        print(f"- Test:  {df_test.shape}  (20%)")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Preprocess
    # We need to fit preprocessor on TRAIN only
    preprocessor = DataPreprocessor()
    
    # Clean and Fit-Transform Train
    print("Preprocessing Train...")
    df_train = preprocessor.clean(df_train)
    X_train_scaled, y_train_enc = preprocessor.fit_transform(df_train, target_col='Label')
    
    print("Preprocessing Val...")
    df_val = preprocessor.clean(df_val)
    X_val_scaled, y_val_enc = preprocessor.transform(df_val, target_col='Label')
    
    # Clean and Transform Test
    print("Preprocessing Test...")
    df_test = preprocessor.clean(df_test)
    X_test_scaled, y_test_enc = preprocessor.transform(df_test, target_col='Label') # Use transform, not fit
    
    # 3. Prepare Binary Labels for Stage 1 (0=Benign, 1=Attack)
    # Get Label Encoder classes
    classes = preprocessor.label_encoder.classes_
    print(f"Classes: {classes}")
    
    # Find index of 'Benign'
    if 'Benign' in classes:
        benign_idx = list(classes).index('Benign')
    else:
        # Fallback/Check
        print("Warning: 'Benign' class not found in LabelEncoder? Mapping anyway.")
        benign_idx = 0 
        
    # Convert to Binary
    # Benign -> 0
    # Attack -> 1
    y_train_binary = (y_train_enc != benign_idx).astype(int)
    y_val_binary = (y_val_enc != benign_idx).astype(int)
    y_test_binary = (y_test_enc != benign_idx).astype(int)
    
    print(f"Train Class Dist (Binary): 0={sum(y_train_binary==0)}, 1={sum(y_train_binary==1)}")
    
    # 4. Train Stage 1 Model (Random Forest)
    # Use Best Params from E1 if available, otherwise default
    print("Training Stage 1 RF (Binary) on 70% Train Set...")
    model = Stage1Filter(n_estimators=50, max_depth=20, class_weight='balanced') 
    model.train(X_train_scaled, y_train_binary)
    
    # 5. Evaluate
    print("\n--- Evaluation on Validation Set (10%) ---")
    model.evaluate(X_val_scaled, y_val_binary)

    print("\n--- Evaluation on Test Set (20% - Strict Holdout) ---")
    model.evaluate(X_test_scaled, y_test_binary)
    
    # 6. Save Model AND Preprocessor
    os.makedirs("models_chk", exist_ok=True)
    model.save("models_chk/stage1_rf_stratified.joblib")
    joblib.dump(preprocessor, "models_chk/preprocessor_stratified.joblib")
    joblib.dump(preprocessor.label_encoder, "models_chk/label_encoder_e1.joblib") # Useful for E17
    
    print("Model and Preprocessor saved to models_chk/ (suffixed _stratified)")
    
    # 7. GENERATE STAGE 2 DATA (For All Splits)
    print("\n=== Generating Stage 2 Data for All Splits ===")
    os.makedirs("data/stage2", exist_ok=True)

    # A. Stage 2 TRAIN (from Stage 1 Train via CV)
    from sklearn.model_selection import cross_val_predict
    from sklearn.ensemble import RandomForestClassifier

    print("[Stage 2 Train] Generating via 5-Fold CV on Stage 1 Train (70%)...")
    # Re-instantiate RF for CV to prevent leakage/overfitting
    rf_for_cv = RandomForestClassifier(n_estimators=50, max_depth=20, class_weight='balanced', n_jobs=-1, random_state=42)
    y_train_pred_cv = cross_val_predict(rf_for_cv, X_train_scaled, y_train_binary, cv=5, n_jobs=-1)
    
    mask_train = (y_train_pred_cv == 1) # Suspicious
    if mask_train.any():
        df_train[mask_train].to_parquet("data/stage2/train.parquet")
        print(f"- Saved {sum(mask_train)} samples to data/stage2/train.parquet")
    else:
        print("- Warning: No Suspicious samples found in Train CV!")

    # B. Stage 2 VAL (from Stage 1 Val via Prediction)
    print("[Stage 2 Val] Generating via Model Prediction on Stage 1 Val (10%)...")
    y_val_pred = model.predict(X_val_scaled)
    mask_val = (y_val_pred == 1)
    if mask_val.any():
        df_val[mask_val].to_parquet("data/stage2/val.parquet")
        print(f"- Saved {sum(mask_val)} samples to data/stage2/val.parquet")
    else:
        print("- Warning: No Suspicious samples found in Val!")
        
    # C. Stage 2 TEST (from Stage 1 Test via Prediction)
    print("[Stage 2 Test] Generating via Model Prediction on Stage 1 Test (20%)...")
    y_test_pred = model.predict(X_test_scaled)
    mask_test = (y_test_pred == 1)
    if mask_test.any():
        df_test[mask_test].to_parquet("data/stage2/test.parquet")
        print(f"- Saved {sum(mask_test)} samples to data/stage2/test.parquet")
    else:
        print("- Warning: No Suspicious samples found in Test!")
        
    print("Stage 2 Data Generation Complete.")

if __name__ == "__main__":
    main()
