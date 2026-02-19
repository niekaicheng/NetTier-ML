
import pandas as pd
import glob
import os

def convert_csv_to_parquet(csv_dir="archive", output_dir="archive"):
    print(f"Converting CSVs in {csv_dir} to Parquet...")
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    if not csv_files:
        print("No CSV files found.")
        return

    for f in csv_files:
        fname = os.path.basename(f)
        parquet_name = fname.replace(".csv", ".parquet")
        output_path = os.path.join(output_dir, parquet_name)
        
        if os.path.exists(output_path):
            print(f"Skipping {fname} (Parquet exists)")
            continue
            
        print(f"Converting {fname}...", flush=True)
        try:
            # Read CSV (handle encoding if needed, CIC-IDS2017 often has quirks)
            try:
                df = pd.read_csv(f, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(f, encoding='cp1252')
            
            # Basic cleaning of column names
            df.columns = df.columns.str.strip()
            
            # Save
            df.to_parquet(output_path, index=False)
            print(f"Saved {output_path}")
            
            # Remove CSV to save space in Colab?
            # os.remove(f) 
        except Exception as e:
            print(f"Error converting {fname}: {e}")

if __name__ == "__main__":
    convert_csv_to_parquet()
