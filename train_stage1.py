
import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.getcwd(), 'src'))

from processing.loader import load_data
from processing.preprocess import DataPreprocessor
from models.stage1_rf import Stage1Filter

def main():
    data_dir = "archive"
    
    # 1. Load Data (Benign + Multiple Attacks)
    try:
        print("Loading Benign data...")
        df_benign = load_data(data_dir, subset="Benign")
        # Ensure balanced-ish: 50k Benign
        if len(df_benign) > 50000:
            df_benign = df_benign.sample(n=50000, random_state=42)
            
        print("Loading Attack data (Mixed)...")
        attack_subsets = ["PortScan", "DoS", "WebAttacks", "Botnet"]
        attack_dfs = []
        for s in attack_subsets:
            try:
                tmp = load_data(data_dir, subset=s)
                if len(tmp) > 15000: tmp = tmp.sample(15000, random_state=42)
                attack_dfs.append(tmp)
            except: pass
            
        if attack_dfs:
            df_attack = pd.concat(attack_dfs, ignore_index=True)
            print(f"Attack samples: {len(df_attack)}")
        else:
            print("No attack data found!")
            return
            
        df = pd.concat([df_benign, df_attack], ignore_index=True)
        print(f"Total Data: {df.shape}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Preprocess
    preprocessor = DataPreprocessor()
    df = preprocessor.clean(df)
    
    # Create Binary Label: 0=Benign, 1=Malicious
    # Assuming 'Label' column exists and "Benign" is the string
    # We use preprocessor to get X_scaled, but we need binary y.
    
    # Custom fit_transform for Stage 1 to get Binary Y
    # But preprocessor gives multiclass Y.
    # So we do:
    X_scaled, y_enc = preprocessor.fit_transform(df, target_col='Label')
    
    # We need to map y_enc back to binary.
    # We know 'Benign' is likely class 0 if alphabetical.
    # Let's verify.
    benign_idx = preprocessor.label_encoder.transform(['Benign'])[0]
    y_binary = (y_enc != benign_idx).astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)
    
    # 3. Train
    model = Stage1Filter()
    model.train(X_train, y_train)
    
    # 4. Evaluate
    model.evaluate(X_test, y_test)
    
    # 5. Save Model AND Preprocessor
    os.makedirs("models_chk", exist_ok=True)
    model.save("models_chk/stage1_rf.joblib")
    
    # Save Preprocessor
    joblib.dump(preprocessor, "models_chk/preprocessor.joblib")
    print("Preprocessor saved.")

    # Note: Stage 2 training should ideally use this SAME preprocessor or refit on same data logic.
    # For now, we assume Stage 1's preprocessor covers enough range.

if __name__ == "__main__":
    main()
