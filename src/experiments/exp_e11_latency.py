
"""
实验 E11: 推理速度与资源基准测试
参考文献: [Abu Al-Haija'22]
目标: 验证 Stage 1 本身是否满足 < 10μs/sample 的即时性要求，并评估吞吐量。
"""
import sys
import os
import time
import numpy as np
import pandas as pd
import joblib
# import psutil # Unavailable
from datetime import datetime
import json

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

# Add src
sys.path.append(os.path.join(os.getcwd(), 'src'))
from processing.loader import load_data

def main():
    print("[1/4] Loading Test Data...", flush=True)
    
    # Load a substantial amount of data to get stable metrics (e.g. 100k samples)
    data_dir = "archive"
    # Load Benign + DoS for testing
    dfs = []
    
    print("Loading Benign...", flush=True)
    try:
        df1 = load_data(data_dir, subset="Benign")
        # Use 50k for robust testing
        if len(df1) > 50000: 
            df1 = df1.sample(50000, random_state=42)
        dfs.append(df1)
        
        print("Loading DoS...", flush=True)
        df2 = load_data(data_dir, subset="DoS")
        if len(df2) > 50000: 
            df2 = df2.sample(50000, random_state=42)
        dfs.append(df2)
    except:
        print("Data loading failed (files missing?)", flush=True)
        return
    
    print("Done Loading. Concatenating...", flush=True)

    if not dfs: return
    df = pd.concat(dfs, ignore_index=True)
    
    # We need X (features)
    # Assume 'Label' is present
    if 'Label' in df.columns:
        X = df.drop(columns=['Label'])
    else:
        X = df

    print(f"Test Data Shape: {X.shape}", flush=True)

    # ==================== 2. Load Model ====================
    # Use the best model from E1
    model_path = "models_chk/stage1_rf_best.pkl"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run E1 first or train a dummy model.")
        # Train a dummy for structure verification?
        # Better to exit than mislead with dummy performance.
        return
        
    print(f"[2/4] Loading Model from {model_path}...", flush=True)
    pipeline = joblib.load(model_path)
    
    # OPTIMIZATION: Set n_jobs=1 for low-latency inference
    # n_jobs=-1 causes high overhead for small batches/single samples
    print("optimizing RF for inference (n_jobs=1)...", flush=True)
    if 'rf' in pipeline.named_steps:
        pipeline.named_steps['rf'].n_jobs = 1
    
    # Warmup
    print("Warming up model...", flush=True)
    pipeline.predict(X.iloc[:100])
    
    # ==================== 3. Benchmarking ====================
    print("\n[3/4] Starting Benchmark...", flush=True)
    
    batch_sizes = [1, 32, 64, 128, 1000, 10000]
    results = []
    
    # process = psutil.Process(os.getpid())
    
    for bs in batch_sizes:
        # Prepare batches
        num_samples = len(X)
        num_batches = num_samples // bs
        if num_batches == 0: 
            print(f"Skipping Batch Size {bs} (Not enough data)", flush=True)
            continue
        
        # Dynamic limit based on batch size to avoid long wait times for small batches
        # BS=1: 1000 samples is enough for stable mean. 100k takes forever.
        if bs < 100:
            limit_samples = min(num_samples, 2000)
        else:
            limit_samples = min(num_samples, 100000)
            
        limit_batches = limit_samples // bs
        
        print(f"  Testing Batch Size: {bs} (Limit: {limit_samples} samples)", flush=True)
        
        latencies = []
        start_time = time.perf_counter()
        # cpu_start = process.cpu_percent()
        
        for i in range(limit_batches):
            batch = X.iloc[i*bs : (i+1)*bs]
            
            t0 = time.perf_counter()
            _ = pipeline.predict(batch)
            t1 = time.perf_counter()
            
            latencies.append(t1 - t0)
            
        total_time = time.perf_counter() - start_time
        # cpu_end = process.cpu_percent()
        cpu_end = 0 # Placeholder
        
        # Metrics
        avg_latency_per_batch = np.mean(latencies)
        avg_latency_per_sample = avg_latency_per_batch / bs
        throughput = (limit_batches * bs) / total_time
        
        print(f"    Latency (per sample): {avg_latency_per_sample*1e6:.2f} μs", flush=True)
        print(f"    Throughput: {throughput:.2f} samples/s", flush=True)
        # print(f"    CPU Usage: {cpu_end}%")
        
        results.append({
            'batch_size': bs,
            'latency_us_per_sample': avg_latency_per_sample * 1e6,
            'throughput_samples_per_sec': throughput,
            'cpu_usage_percent': cpu_end
        })

    # ==================== 4. Save Results ====================
    os.makedirs("results", exist_ok=True)
    res_path = f"results/E11_latency_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[4/4] Results saved to {res_path}", flush=True)

if __name__ == "__main__":
    main()
