
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        # Columns to drop based on common IDS practices (IPs, Ports, Timestamps)
        self.drop_cols = [
            "Flow ID", "Source IP", "Source Port", "Destination IP", 
            "Destination Port", "Protocol", "Timestamp"
        ]
        
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove Infinity, NaN, and irrelevant columns.
        """
        # Replace Inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Drop NaN
        df = df.dropna()
        
        # Drop columns if they exist
        existing_drop_cols = [c for c in self.drop_cols if c in df.columns]
        if existing_drop_cols:
            df = df.drop(columns=existing_drop_cols)
            
        print(f"Cleaned data shape: {df.shape}")
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = "Label"):
        """
        Fit scaler and label encoder, then transform.
        """
        # Separate features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
            
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Encode Labels
        y_enc = self.label_encoder.fit_transform(y)
        
        # Scale Features
        # Ensure all are numeric
        X_numeric = X.select_dtypes(include=[np.number])
        X_scaled = self.scaler.fit_transform(X_numeric)
        
        return X_scaled, y_enc
        
    def transform(self, df: pd.DataFrame, target_col: str = "Label"):
        """
        Transform new data using fitted scalers.
        """
        if target_col in df.columns:
            y = df[target_col]
            # Handle unseen labels map to 'Unknown' or similar? 
            # For simplicity, we assume consistent labels or crash.
            # Or better, just transform.
            try:
                y_enc = self.label_encoder.transform(y)
            except ValueError:
                 # Fallback for unseen labels if necessary
                y_enc = np.zeros(len(y)) - 1 # Invalid
                
            X = df.drop(columns=[target_col])
        else:
            y_enc = None
            X = df
            
        X_numeric = X.select_dtypes(include=[np.number])
        X_scaled = self.scaler.transform(X_numeric)
        
        return X_scaled, y_enc

if __name__ == "__main__":
    # Test logic
    pass
