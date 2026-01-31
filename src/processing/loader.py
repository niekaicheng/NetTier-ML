
import pandas as pd
import os
import glob

def load_data(data_dir: str, subset: str = None) -> pd.DataFrame:
    """
    Load parquet files from the data directory.
    
    Args:
        data_dir (str): Path to directory containing parquet files.
        subset (str, optional): distinct keyword to filter files (e.g., 'Benign', 'DDoS').
        
    Returns:
        pd.DataFrame: Combined dataframe.
    """
    pattern = os.path.join(data_dir, "*.parquet")
    files = glob.glob(pattern)
    
    if subset:
        files = [f for f in files if subset.lower() in f.lower()]
        
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir} matching {subset}")
        
    dfs = []
    print(f"Loading {len(files)} files...")
    for f in files:
        try:
            df = pd.read_parquet(f)
            # Add a label column based on filename if not present (optional logic)
            # But the requirement says "Labelled" files exist, we assume they have labels?
            # Let's inspect columns later. For now, just load.
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    # Test
    try:
        df = load_data("archive", subset="Benign")
        print(f"Loaded Benign shape: {df.shape}")
    except Exception as e:
        print(e)
