
"""
Data Loader for E16: Time-aware Split
Logic:
    Train: Monday, Tuesday, Wednesday
    Test: Thursday, Friday
"""

import pandas as pd
import os
import glob
import sys

# Add src
sys.path.append(os.path.join(os.getcwd(), 'src'))
from processing.loader import load_data

def load_time_aware_split(data_dir):
    """
    Returns (df_train, df_test) based on day of week.
    """
    # Days
    train_days = ["Monday", "Tuesday", "Wednesday"]
    test_days = ["Thursday", "Friday"]
    
    print(f"Loading Time-aware Split from {data_dir}...")
    
    # Get all files
    pattern = os.path.join(data_dir, "*.parquet")
    files = glob.glob(pattern)
    
    train_files = []
    test_files = []
    
    for f in files:
        fname = os.path.basename(f).lower()
        if any(d.lower() in fname for d in train_days):
            train_files.append(f)
        elif any(d.lower() in fname for d in test_days):
            test_files.append(f)
        else:
            print(f"Warning: File {fname} does not match Mon-Fri pattern. Ignoring.")
            
    print(f"Train Files ({len(train_files)}): {[os.path.basename(f) for f in train_files]}")
    print(f"Test Files ({len(test_files)}): {[os.path.basename(f) for f in test_files]}")
    
    # Load
    def load_files(file_list):
        dfs = []
        for f in file_list:
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()

    df_train = load_files(train_files)
    df_test = load_files(test_files)
    
    return df_train, df_test

if __name__ == "__main__":
    # Test
    try:
        tr, te = load_time_aware_split("archive")
        print(f"Train Shape: {tr.shape}")
        print(f"Test Shape: {te.shape}")
        
        # Check Label overlap?
        if not tr.empty:
            print("Train Labels:", tr['Label'].unique() if 'Label' in tr.columns else "No Label")
        if not te.empty:
            print("Test Labels:", te['Label'].unique() if 'Label' in te.columns else "No Label")
            
    except Exception as e:
        print(e)
