
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

class FeatureCleaner(BaseEstimator, TransformerMixin):
    """
    Cleaner that removes Infinity, NaN, and irrelevant columns.
    Compatible with sklearn Pipeline.
    """
    def __init__(self, drop_cols=None):
        self.drop_cols = drop_cols or [
            "Flow ID", "Source IP", "Source Port", "Destination IP", 
            "Destination Port", "Protocol", "Timestamp"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Replace Inf
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Drop Cols
        existing_drop = [c for c in self.drop_cols if c in X.columns]
        if existing_drop:
            X = X.drop(columns=existing_drop)
            
        # Drop NaN (Row-wise)
        # Note: In a pipeline, dropping rows in transform is tricky because y must be aligned.
        # But sklearn pipeline doesn't support dropping rows easily in transform if y is separate.
        # Imputation is better. Let's fill NaN with 0 or mean.
        # For IDS, 0 is often safe for missing stats, or mean.
        # Let's use simple fillna(0) for now to keep shapes consistent.
        X = X.fillna(0)
        
        return X

class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Wrapper around MinMaxScaler that preserves DataFrame columns (optional).
    """
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.columns = None

    def fit(self, X, y=None):
        # Select numeric
        self.columns = X.select_dtypes(include=[np.number]).columns
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X = X.copy()
        if self.columns is None:
             self.columns = X.select_dtypes(include=[np.number]).columns
        
        # Transform numeric
        X[self.columns] = self.scaler.transform(X[self.columns])
        return X
