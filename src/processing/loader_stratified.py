
import pandas as pd
import os
import glob
import sys

# Add src to path if needed (though usually done by caller)
# sys.path.append(os.path.join(os.getcwd(), 'src'))
from processing.loader import load_data

def load_stratified_mixed_split(data_dir, 
                                benign_monday_ratio=0.8, 
                                benign_other_ratio=0.8, 
                                attack_ratio=0.8,
                                random_state=42):
    """
    Loads data with specific stratification strategies:
    1. Monday Benign: Take `benign_monday_ratio` for Train.
    2. Other Days Benign: Take `benign_other_ratio` for Train.
    3. Attacks (All Files): Take `attack_ratio` for Train.
    
    Returns:
        df_train (pd.DataFrame): Mixed Train Set
        df_test (pd.DataFrame): Strict Isolation Test Set (Remaining Data)
    """
    print(f"Loading Stratified Mixed Split from {data_dir}...")
    
    train_dfs = []
    test_dfs = []
    
    # 1. Monday Benign (The Baseline)
    # File: Benign-Monday-no-metadata.parquet
    mon_file = os.path.join(data_dir, "Benign-Monday-no-metadata.parquet")
    if os.path.exists(mon_file):
        print("Processing Monday Benign...", flush=True)
        df_mon = pd.read_parquet(mon_file)
        # Sample
        df_mon_train = df_mon.sample(frac=benign_monday_ratio, random_state=random_state)
        df_mon_test = df_mon.drop(df_mon_train.index)
        
        train_dfs.append(df_mon_train)
        test_dfs.append(df_mon_test)
    else:
        print("Warning: Monday file not found!", flush=True)
        
    # 2. Process All Other Files
    # We need to distinguish Benign from Attack within these files
    # Pattern: *-Day-no-metadata.parquet
    all_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    
    for f in all_files:
        if "Benign-Monday" in f: continue # Already handled
        
        fname = os.path.basename(f)
        print(f"Processing {fname}...", flush=True)
        
        try:
            df = pd.read_parquet(f)
            
            # Split by Label if possible
            if 'Label' in df.columns:
                # Benign
                df_benign = df[df['Label'] == 'Benign']
                if not df_benign.empty:
                    df_b_train = df_benign.sample(frac=benign_other_ratio, random_state=random_state)
                    df_b_test = df_benign.drop(df_b_train.index)
                    train_dfs.append(df_b_train)
                    test_dfs.append(df_b_test)
                    
                # Attacks (All non-Benign)
                df_attack = df[df['Label'] != 'Benign']
                if not df_attack.empty:
                    df_a_train = df_attack.sample(frac=attack_ratio, random_state=random_state)
                    df_a_test = df_attack.drop(df_a_train.index)
                    train_dfs.append(df_a_train)
                    test_dfs.append(df_a_test)
            else:
                # Assume based on filename if Label missing?
                # But our verify script showed Label exists.
                print(f"  Warning: No Label column in {fname}", flush=True)
                
        except Exception as e:
            print(f"  Error reading {fname}: {e}", flush=True)
            
    # Concatenate
    print("Concatenating Train Sets...", flush=True)
    df_train = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    
    print("Concatenating Test Sets...", flush=True)
    df_test = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    
    return df_train, df_test

if __name__ == "__main__":
    # Test
    try:
        tr, te = load_stratified_mixed_split("archive", benign_monday_ratio=0.1, benign_other_ratio=0.1, attack_ratio=0.1)
        print(f"Train: {tr.shape}")
        print(f"Test: {te.shape}")
        if not tr.empty:
            print("Train Labels:", tr['Label'].value_counts())
    except Exception as e:
        print(e)
