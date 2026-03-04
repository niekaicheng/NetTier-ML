
"""
实验 E1: Random Forest 超参数调优 (Nested Cross-Validation)
参考文献: [Doula'25], [Kaur'21]
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from processing.loader import load_data
from processing.pipeline_utils import FeatureCleaner, FeatureScaler
from utils.training_logger import TrainingLogger

def main():
    start_time = datetime.now()
    print(f"Experiment E1 Started at: {start_time}")
    
    # 初始化训练记录器
    logger = TrainingLogger(
        experiment_name="E1_nested_cv",
        description="RF hyperparameter tuning via Nested 5-Fold CV with GridSearch. Ref: [Doula'25], [Kaur'21]"
    )
    logger.start()
    
    # ==================== 1. 数据加载 ====================
    print("[1/5] Loading CIC-IDS2017 data (Stratified for E1)...") 
    data_dir = "archive"
    
    from processing.loader_stratified import load_stratified_mixed_split
    
    # Load Stratified Split
    # We only need TRAIN set for Nested CV (which does its own internal splitting)
    # But E1 methodology (Nested) typically uses the "Development Set".
    # Here df_train IS the development set. df_test is the strict holdout.
    try:
        df, _ = load_stratified_mixed_split(data_dir, 
                                           benign_monday_ratio=0.1, # Reduced for E1 tuning speed
                                           benign_other_ratio=0.1, 
                                           attack_ratio=0.1)
    except Exception as e:
        print(f"Error loading data: {e}")
        logger.finish(status="failed")
        return
        
    print(f"Data Shape: {df.shape}")



    # ==================== 2. 预处理 (Pipeline Prep) ====================
    # Define Target and Features
    target_col = 'Label'
    if target_col not in df.columns:
        print("Label column missing!")
        return

    # Encode Label (Outcome)
    le = LabelEncoder()
    df[target_col] = df[target_col].astype(str) # Ensure string
    y = le.fit_transform(df[target_col])
    X = df.drop(columns=[target_col])
    
    # Save encoders
    os.makedirs("models_chk", exist_ok=True)
    joblib.dump(le, "models_chk/label_encoder_e1.joblib")
    
    print(f"Classes: {le.classes_}")
    
    # 记录数据信息
    logger.set_data_info(
        dataset="CIC-IDS2017",
        data_path="archive",
        total_samples=len(df),
        num_features=X.shape[1],
        num_classes=len(le.classes_),
        split_method="Stratified Mixed Split (10% sample for tuning speed)",
    )

    # Define Pipeline (Leak-free)
    # 1. Clean
    # 2. Scale
    # 3. Model
    # Note: RF doesn't strictly need scaling, but it helps convergence speed sometimes and essential for other models.
    # We include it to be consistent with 'pipeline_utils' design.
    pipeline = Pipeline([
        ('cleaner', FeatureCleaner()),
        ('scaler', FeatureScaler()),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1)) # n_jobs=-1 here for parallel trees
    ])

    # ==================== 3. Nested CV Configuration ====================
    print("\n[2/5] Configuring Nested CV...")
    
    # Outer CV (Performance Eval)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Inner CV (Hyperparam Tuning)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Param Grid (Prefix with 'rf__')
    param_grid = {
        'rf__n_estimators': [50, 100],        # Reduced for demo/testing speed (Real E1: [50, 100, 200])
        'rf__max_depth': [10, 20, None],
        'rf__max_features': ['sqrt', 'log2'],
        'rf__min_samples_split': [2, 10],
        'rf__class_weight': ['balanced', None]
    }
    
    # 记录模型和训练配置
    logger.set_model_info(
        model_type="RandomForest",
        model_name="Stage 1 RF (Nested CV)",
        framework="scikit-learn",
        hyperparams={"param_grid": param_grid},
    )
    logger.set_training_config(
        cv_folds=5,
        device="CPU",
        inner_cv_folds=3,
    )
    
    # Scorers
    scorers = {
        'f1_macro': make_scorer(f1_score, average='macro'),
        'precision_macro': make_scorer(precision_score, average='macro'),
        'recall_macro': make_scorer(recall_score, average='macro')
    }

    # ==================== 4. Execution Loop ====================
    print("\n[3/5] Starting Nested CV Loop...")
    
    nested_results = []
    best_params_list = []
    
    fold = 1
    for train_idx, test_idx in outer_cv.split(X, y):
        print(f"\n--- Outer Fold {fold}/5 ---")
        
        # Split Data (Pandas indexing)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner GridSearch
        clf = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            scoring='f1_macro',
            n_jobs=2, # Parallelize grid search (total cores = n_jobs_grid * n_jobs_rf)
            verbose=1
        )
        
        clf.fit(X_train, y_train)
        
        # Best Model
        best_model = clf.best_estimator_
        best_params = clf.best_params_
        best_params_list.append(best_params)
        
        # Evaluate on Outer Test
        y_pred = best_model.predict(X_test)
        
        scores = {
            'fold': fold,
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'best_params': best_params
        }
        nested_results.append(scores)
        
        print(f"  Fold Score: F1={scores['f1_macro']:.4f}")
        print(f"  Best Params: {best_params}")
        
        # 记录到 logger
        logger.log_cv_fold(
            fold=fold,
            metrics={"f1_macro": scores['f1_macro'], "precision_macro": scores['precision_macro'], "recall_macro": scores['recall_macro']},
            best_params=best_params,
        )
        
        fold += 1

    # ==================== 5. Reporting ====================
    print("\n[4/5] Aggregating Results...")
    
    df_res = pd.DataFrame(nested_results)
    mean_f1 = df_res['f1_macro'].mean()
    std_f1 = df_res['f1_macro'].std()
    
    print(f"\nNested CV F1-Macro: {mean_f1:.4f} ± {std_f1:.4f}")
    
    # Save Results
    os.makedirs("results", exist_ok=True)
    res_path = f"results/E1_nested_cv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert for JSON serialization
    results_json = {
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'folds': nested_results,
        'final_best_params_mode': str(best_params_list[-1]) # Simplified mode selection
    }
    
    with open(res_path, 'w') as f:
        json.dump(results_json, f, indent=4, default=str)
        
    print(f"Results saved to {res_path}")

    # ==================== 6. Final Model Training ====================
    print("\n[5/5] Training Final Model (Best Config) on ALL Data...")
    
    # Use params from last fold as 'best' (or mode)
    final_pipeline = Pipeline([
        ('cleaner', FeatureCleaner()),
        ('scaler', FeatureScaler()),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    final_pipeline.set_params(**best_params_list[-1])
    final_pipeline.fit(X, y)
    
    joblib.dump(final_pipeline, "models_chk/stage1_rf_best.pkl")
    print("Final model saved to models_chk/stage1_rf_best.pkl")
    
    # End Time
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nExperiment Completed.")
    print(f"Start Time: {start_time}")
    print(f"End Time:   {end_time}")
    print(f"Duration:   {duration}")
    
    # Update JSON with timestamps
    results_json['start_time'] = str(start_time)
    results_json['end_time'] = str(end_time)
    results_json['duration_seconds'] = duration.total_seconds()
    
    # Re-save JSON
    with open(res_path, 'w') as f:
        json.dump(results_json, f, indent=4, default=str)
    print(f"Updated results JSON with timestamps.")
    
    # 记录结果并完成
    logger.set_results(
        mean_f1_macro=mean_f1,
        std_f1_macro=std_f1,
        final_best_params=str(best_params_list[-1]),
    )
    logger.add_artifact(res_path, "results", "Nested CV results JSON")
    logger.add_artifact("models_chk/stage1_rf_best.pkl", "model", "Final RF model (best config)")
    logger.add_artifact("models_chk/label_encoder_e1.joblib", "encoder", "Label encoder")
    logger.finish()

if __name__ == "__main__":
    main()
