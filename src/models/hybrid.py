
import numpy as np
import torch
from .stage1_rf import Stage1Filter
from .stage2_transeca import TransECANet

class HierarchicalIDS:
    def __init__(self, stage1_path, stage2_path, stage2_params):
        self.stage1 = Stage1Filter()
        self.stage1.load(stage1_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stage2 = TransECANet(**stage2_params).to(self.device)
        self.stage2.load_state_dict(torch.load(stage2_path, map_location=self.device))
        self.stage2.eval()
        
    def predict(self, X_numpy, X_torch):
        """
        X_numpy: For Stage 1 (RF)
        X_torch: For Stage 2 (TransECA-Net)
        """
        # 1. Stage 1 Filter
        print("Stage 1 Filtering...")
        binary_preds = self.stage1.predict(X_numpy) # 0: Benign, 1: Malicious
        
        # Initialize final preds with 0 (Benign)
        # Note: mapping depends on Stage 2 classes. 
        # Usually Stage 2 has Benign=0 too.
        final_preds = np.zeros_like(binary_preds)
        
        # 2. Identify Suspicious
        mask = (binary_preds == 1)
        suspicious_count = np.sum(mask)
        print(f"Stage 1 flagged {suspicious_count}/{len(X_numpy)} as suspicious.")
        
        if suspicious_count > 0:
            # Prepare input for Stage 2
            # We need X_torch corresponding to mask
            X_mal = X_torch[mask].to(self.device)
            
            with torch.no_grad():
                outputs = self.stage2(X_mal)
                _, preds_mal = torch.max(outputs, 1)
                
            # Update final preds
            # Note: Stage 2 Prediction is the Class Index.
            # We assume Class Index aligns with global classes or need remapping.
            # Use 'final_preds' to store Class Indices.
            final_preds[mask] = preds_mal.cpu().numpy()
            
        return final_preds
