
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

class Stage1Filter:
    def __init__(self, n_estimators=100, max_depth=None, class_weight='balanced'):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight, # 'balanced' helps with recall
            n_jobs=-1,
            random_state=42
        )
        
    def train(self, X_train, y_train):
        print("Training Random Forest...")
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test):
        print("Evaluating...")
        y_pred = self.model.predict(X_test)
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        return y_pred
        
    def save(self, path):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
        
    def load(self, path):
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
        
    def predict(self, X):
        return self.model.predict(X)
