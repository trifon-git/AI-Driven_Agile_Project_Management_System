import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import os

INPUT_DATA = r'c:\Users\trifo\OneDrive\Υπολογιστής\v0.1 - Antigrivity\OUTPUT\training_data_v2_no_emb.csv'
MODEL_DIR = r'c:\Users\trifo\OneDrive\Υπολογιστής\v0.1 - Antigrivity\MODELS'

def train_model():
    print(f"Loading data from {INPUT_DATA}...")
    if not os.path.exists(INPUT_DATA):
        print(f"Error: Could not find {INPUT_DATA}")
        return
        
    df = pd.read_csv(INPUT_DATA)
    
    features = [
        'JR_story_points', 
        'JR_estimated_hours_from_sp',
        'dev_avg_hours_historic',
        'dev_tasks_count',
        'dev_project_familiarity',
        'JR_project_key_encoded',
        'JR_issue_type_encoded',
        'JR_status_encoded'
    ]
    
    # Add any embedding columns if they exist
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    features.extend(emb_cols)
    
    print(f"Using {len(features)} features.")
    
    X = df[features]
    y = df['target_hours_capped']
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    estimators = [
        ('et', ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ]
    
    stacked_model = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=-1
    )
    
    print("Evaluating model with 5-Fold CV...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    
    for train_idx, test_idx in kf.split(X_imputed):
        X_train, X_test = X_imputed[train_idx], X_imputed[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        stacked_model.fit(X_train, y_train)
        preds = stacked_model.predict(X_test)
        
        r2_scores.append(r2_score(y_test, preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, preds)))
        mae_scores.append(mean_absolute_error(y_test, preds))
        
    print("\n--- Model Performance (CV) ---")
    print(f"R2 Score: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")
    print(f"RMSE:     {np.mean(rmse_scores):.4f}")
    print(f"MAE:      {np.mean(mae_scores):.4f}")
    
    # Train final model on all data
    print("\nTraining final model on full dataset...")
    stacked_model.fit(X_imputed, y)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'stacked_estimator_v2.pkl')
    imputer_path = os.path.join(MODEL_DIR, 'imputer_v2.pkl')
    
    joblib.dump(stacked_model, model_path)
    joblib.dump(imputer, imputer_path)
    
    print(f"Final model saved to {model_path}")

if __name__ == '__main__':
    train_model()
