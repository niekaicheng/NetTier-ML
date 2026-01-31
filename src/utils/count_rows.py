
import os
import glob
import pandas as pd
from fastparquet import ParquetFile

def count_rows():
    data_dir = "archive"
    pattern = os.path.join(data_dir, "*.parquet")
    files = glob.glob(pattern)
    
    print(f"| File Name | Row Count | Size (MB) |")
    print(f"| :--- | :--- | :--- |")
    
    total_rows = 0
    total_size = 0
    
    results = []
    
    for f in files:
        try:
            # Use ParquetFile to get count without loading all data
            pf = ParquetFile(f)
            rows = pf.count()
            
            size_mb = os.path.getsize(f) / (1024 * 1024)
            name = os.path.basename(f)
            
            print(f"| {name} | {rows:,} | {size_mb:.2f} MB |")
            
            total_rows += rows
            total_size += size_mb
            results.append((name, rows, size_mb))
        except Exception as e:
            print(f"| {os.path.basename(f)} | Error: {e} | - |")

    print(f"| **Total** | **{total_rows:,}** | **{total_size:.2f} MB** |")
    
    return results

if __name__ == "__main__":
    count_rows()
