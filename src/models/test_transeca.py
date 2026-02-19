
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from models.stage2_transeca import TransECANet

def test_transeca():
    print("Testing TransECANet Architecture...")
    
    # Mock Data
    batch_size = 4
    num_features = 78 # Approx CIC-IDS2017 feature count after cleaning
    num_classes = 8   # Benign + 7 Attack Classes
    
    x = torch.randn(batch_size, num_features)
    print(f"Input Shape: {x.shape}")
    
    # Initialize Model
    try:
        model = TransECANet(num_features=num_features, num_classes=num_classes)
        print("Model Initialized Successfully.")
    except Exception as e:
        print(f"Model Initialization Failed: {e}")
        return

    # Forward Pass
    try:
        y = model(x)
        print(f"Output Shape: {y.shape}")
        
        expected_shape = (batch_size, num_classes)
        if y.shape == expected_shape:
            print("[OK] Forward Pass Successful. Dimensions Match.")
        else:
            print(f"[FAIL] Dimension Mismatch. Expected {expected_shape}, got {y.shape}")
            
    except Exception as e:
        print(f"Forward Pass Failed: {e}")

if __name__ == "__main__":
    test_transeca()
