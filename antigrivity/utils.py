"""
Antigrivity — Shared Utilities
===============================
Common helpers used across the pipeline.
"""

import re
import numpy as np
import pandas as pd
import joblib
import os
from antigrivity.config import (
    EFFORT_MODEL_PATH, EFFORT_MODEL_TUNED_PATH,
    EFFORT_MODEL_V2_PATH, TRAINING_DATASET_CSV,
)



def clean_text(text):
    """Clean text for NLP features: lowercase, remove special chars, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()



_cached_model = None
_cached_imputer = None
_cached_transformer = None
_cached_training_data = None


def load_model(use_tuned=True):
    """Load the best available effort estimation model (v2 > tuned > base)."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    # Priority: v2 (stacked) → tuned ExtraTrees → base model
    candidates = [
        EFFORT_MODEL_V2_PATH,
        EFFORT_MODEL_TUNED_PATH if use_tuned else EFFORT_MODEL_PATH,
        EFFORT_MODEL_PATH if use_tuned else EFFORT_MODEL_TUNED_PATH,
    ]

    for path in candidates:
        if os.path.exists(path):
            loaded = joblib.load(path)

            # Support pipelines saved as dicts: {'imputer':..., 'transformer':..., 'model':...}
            if isinstance(loaded, dict):
                if 'model' in loaded:
                    _cached_model = loaded['model']
                    # optionally cache imputer/transformer when present
                    global _cached_imputer, _cached_transformer
                    _cached_imputer = loaded.get('imputer')
                    _cached_transformer = loaded.get('transformer')
                    print(f"[ModelLoader] Loaded pipeline dict: {os.path.basename(path)}")
                    return _cached_model
                # fallback: find first value that looks like a model (has predict)
                for v in loaded.values():
                    if hasattr(v, 'predict'):
                        _cached_model = v
                        print(f"[ModelLoader] Loaded model from dict: {os.path.basename(path)}")
                        return _cached_model
                # not a supported dict
                raise TypeError(f"Loaded object from {path} is a dict but contains no model-like entry")

            _cached_model = loaded
            print(f"[ModelLoader] Loaded: {os.path.basename(path)}")
            return _cached_model

    # Helpful debug info when no model is found
    existence = [(p, os.path.exists(p)) for p in candidates]
    debug_lines = "\n".join([f"{p} -> exists={exists}" for p, exists in existence])
    raise FileNotFoundError(
        f"No trained model found. Run the training pipeline first.\n"
        f"Checked candidate paths:\n{debug_lines}\n"
        f"Expected at: {EFFORT_MODEL_V2_PATH} or {EFFORT_MODEL_TUNED_PATH}"
    )


def load_training_data():
    """Load the training dataset (cached after first call)."""
    global _cached_training_data
    if _cached_training_data is not None:
        return _cached_training_data

    if not os.path.exists(TRAINING_DATASET_CSV):
        raise FileNotFoundError(
            f"Training dataset not found at: {TRAINING_DATASET_CSV}\n"
            f"Run the ETL + training set pipeline first."
        )

    df = pd.read_csv(TRAINING_DATASET_CSV)
    
    # Map V2 CSV column names to V1 internal app schema naming
    rename_map = {
        'JR_assignee': 'feature_assignee',
        'TS_timesheet_hours': 'target_actual_hours',
        'JR_project_key': 'feature_project_key',
        'JR_issue_type': 'feature_issue_type',
        'JR_summary': 'feature_summary_clean',
        'JR_story_points': 'feature_story_points'
    }
    df = df.rename(columns=rename_map)
    
    _cached_training_data = df
    return _cached_training_data



# Default values for optional features the model may expect
_OPTIONAL_FEATURE_DEFAULTS = {
    'feature_original_estimate_hours': 0.0,
    'feature_summary_length': 0,
    'feature_created_day': 'Monday',
    'feature_created_month': 1,
    'feature_has_parent': 0,
    'feature_has_sprint': 0,
    'feature_sprint_count': 0,
    'feature_calendar_days_open': 7,
    'feature_priority': 'Medium',
    'feature_description_length': 0,
    'feature_comments_count': 0,
    'feature_inward_links': 0,
    'feature_outward_links': 0,
    'feature_total_links': 0,
}


def _get_model_feature_names(model):
    """Discover all column names the model's ColumnTransformer expects."""
    preprocessor = model.named_steps['preprocessor']
    columns = set()
    for name, transformer, cols in preprocessor.transformers:
        if isinstance(cols, str):
            columns.add(cols)
        elif isinstance(cols, list):
            columns.update(cols)
    return columns


def _build_input_dataframe(task_dict, assignee, model):
    """
    Build a DataFrame with ALL columns the model expects.
    Fills missing optional features with sensible defaults.
    """
    # Base features (always provided)
    row = {
        'feature_summary_clean': clean_text(task_dict.get('summary', '')),
        'feature_issue_type': task_dict.get('type', 'Task'),
        'feature_project_key': task_dict.get('project', ''),
        'feature_assignee': assignee,
        'feature_story_points': float(task_dict.get('sp', 0)),
    }

    # Enrich with computed defaults where possible
    summary_clean = row['feature_summary_clean']
    row['feature_summary_length'] = len(summary_clean)
    row['feature_original_estimate_hours'] = float(task_dict.get('sp', 0)) * 0.5

    # Discover what other columns the model needs and fill defaults
    from sklearn.ensemble import StackingRegressor
    if isinstance(model, StackingRegressor):
        output_row = {}
        sp = float(task_dict.get('sp', 0))
        output_row['JR_story_points'] = sp
        output_row['JR_estimated_hours_from_sp'] = sp * 0.5
        
        train_df = load_training_data()
        if 'target_actual_hours' in train_df.columns:
            history = train_df[train_df['feature_assignee'] == assignee]
            if len(history) > 0:
                output_row['dev_avg_hours_historic'] = history['target_actual_hours'].mean()
                output_row['dev_tasks_count'] = len(history)
                output_row['dev_project_familiarity'] = len(history[history['feature_project_key'] == task_dict.get('project', '')])
            else:
                output_row['dev_avg_hours_historic'] = 0.0
                output_row['dev_tasks_count'] = 0
                output_row['dev_project_familiarity'] = 0
        else:
             output_row['dev_avg_hours_historic'] = 0.0
             output_row['dev_tasks_count'] = 0
             output_row['dev_project_familiarity'] = 0

        # Categoricals - lazy load and fit the raw CSV to mimic OrdinalEncoder
        from sklearn.preprocessing import OrdinalEncoder
        raw_data_path = os.path.join(os.path.dirname(TRAINING_DATASET_CSV), "jira_with_timesheet_hours.csv")
        cats = ['JR_project_key', 'JR_issue_type', 'JR_status']
        if os.path.exists(raw_data_path):
            raw_df = pd.read_csv(raw_data_path, low_memory=False)
            for c in cats:
                if c not in raw_df.columns: raw_df[c] = 'Unknown'
                raw_df[c] = raw_df[c].fillna('Unknown')
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            encoder.fit(raw_df[cats])
            query = pd.DataFrame([{
                'JR_project_key': task_dict.get('project', 'Unknown'),
                'JR_issue_type': task_dict.get('type', 'Task'),
                'JR_status': task_dict.get('status', 'To Do')
            }])
            encoded = encoder.transform(query)[0]
            output_row['JR_project_key_encoded'] = encoded[0]
            output_row['JR_issue_type_encoded'] = encoded[1]
            output_row['JR_status_encoded'] = encoded[2]
        else:
            output_row['JR_project_key_encoded'] = 0
            output_row['JR_issue_type_encoded'] = 0
            output_row['JR_status_encoded'] = 0
            
        return pd.DataFrame([output_row])

    expected_cols = _get_model_feature_names(model)
    for col in expected_cols:
        if col not in row:
            row[col] = _OPTIONAL_FEATURE_DEFAULTS.get(col, 0)

    return pd.DataFrame([row])


def predict_effort(model, task_dict, assignee):
    """
    Predict effort in hours for a task assigned to a specific developer.

    Parameters
    ----------
    model : sklearn Pipeline
        The trained effort estimation pipeline.
    task_dict : dict
        Must contain keys: 'summary', 'type', 'project', 'sp'
    assignee : str
        Full name of the developer.

    Returns
    -------
    float : predicted hours (minimum 0.5)
    """
    input_data = _build_input_dataframe(task_dict, assignee, model)

    from sklearn.ensemble import StackingRegressor
    if isinstance(model, StackingRegressor):
        # V2 StackedRegressor predicts directly in hours
        # Prefer cached imputer (loaded with pipeline dict), fallback to known imputer file
        global _cached_imputer
        if _cached_imputer is not None:
            import numpy as np
            if not hasattr(_cached_imputer, "_fill_dtype"): _cached_imputer._fill_dtype = np.float64
            input_data = _cached_imputer.transform(input_data)
        else:
            imputer_path = os.path.join(os.path.dirname(EFFORT_MODEL_V2_PATH), 'imputer_v2.pkl')
            if os.path.exists(imputer_path):
                imputer = joblib.load(imputer_path)
                import numpy as np
                if not hasattr(imputer, "_fill_dtype"): imputer._fill_dtype = np.float64
                input_data = imputer.transform(input_data)

        pred_hours = float(model.predict(input_data)[0])
    else:
        # V1 model predicted on log scale
        pred_log = model.predict(input_data)[0]
        pred_hours = float(np.expm1(pred_log))
        
    return max(0.5, round(pred_hours, 2))


def predict_effort_range(model, task_dict, assignee):
    """
    Predict effort range (P25, P50, P75) using individual trees in the ensemble.

    Returns
    -------
    tuple : (low, mid, high) in hours
    """
    input_data = _build_input_dataframe(task_dict, assignee, model)

    from sklearn.ensemble import StackingRegressor
    if isinstance(model, StackingRegressor):
        # Use cached imputer if available, else fallback to file-based imputer
        global _cached_imputer
        if _cached_imputer is not None:
            import numpy as np
            if not hasattr(_cached_imputer, "_fill_dtype"): _cached_imputer._fill_dtype = np.float64
            input_data = _cached_imputer.transform(input_data)
        else:
            imputer_path = os.path.join(os.path.dirname(EFFORT_MODEL_V2_PATH), 'imputer_v2.pkl')
            if os.path.exists(imputer_path):
                imputer = joblib.load(imputer_path)
                import numpy as np
                if not hasattr(imputer, "_fill_dtype"): imputer._fill_dtype = np.float64
                input_data = imputer.transform(input_data)

        # Stacking Regressors don't have standard trees to extract percentiles easily.
        # Fallback to simple +- 25% boundary variance for prediction points.
        pred_hours = float(model.predict(input_data)[0])
        pred_hours = max(pred_hours, 0.5)
        return (round(pred_hours * 0.75, 2), round(pred_hours, 2), round(pred_hours * 1.5, 2))
        
    # Get the preprocessor and regressor from the pipeline
    preprocessor = model.named_steps['preprocessor']
    regressor = model.named_steps['regressor']

    X_transformed = preprocessor.transform(input_data)

    # Get predictions from individual trees
    tree_preds = np.array([
        tree.predict(X_transformed)[0]
        for tree in regressor.estimators_
    ])

    # Convert from log space and compute quantiles
    tree_preds_hours = np.expm1(tree_preds)
    tree_preds_hours = np.maximum(tree_preds_hours, 0.5)

    low = round(float(np.percentile(tree_preds_hours, 25)), 2)
    mid = round(float(np.percentile(tree_preds_hours, 50)), 2)
    high = round(float(np.percentile(tree_preds_hours, 75)), 2)

    return (low, mid, high)



def get_developer_history(training_df):
    """
    Build a developer skill profile from training data.

    Returns
    -------
    dict : {developer_name: {project_key: count, ...}, ...}
    """
    profiles = {}
    for dev, group in training_df.groupby('feature_assignee'):
        project_counts = group['feature_project_key'].value_counts().to_dict()
        type_counts = group['feature_issue_type'].value_counts().to_dict()
        avg_hours = float(group['target_actual_hours'].mean())
        total_tasks = len(group)

        profiles[dev] = {
            'projects': project_counts,
            'issue_types': type_counts,
            'avg_hours_per_task': avg_hours,
            'total_tasks': total_tasks,
        }

    return profiles


def find_best_developer(task_dict, training_df, team_names):
    """
    Find the best-fit developer for a task based on historical experience.

    Returns
    -------
    str : developer name
    """
    project = task_dict.get('project', '')
    issue_type = task_dict.get('type', 'Task')

    # Filter history for this project + type
    matches = training_df[
        (training_df['feature_project_key'] == project) &
        (training_df['feature_issue_type'] == issue_type)
    ]

    if not matches.empty:
        # Return most frequent assignee (who is also in the team)
        freq = matches['feature_assignee'].value_counts()
        for dev in freq.index:
            if dev in team_names:
                return dev

    # Fallback: just project match
    proj_matches = training_df[training_df['feature_project_key'] == project]
    if not proj_matches.empty:
        freq = proj_matches['feature_assignee'].value_counts()
        for dev in freq.index:
            if dev in team_names:
                return dev

    # Fallback: first available team member
    return list(team_names)[0] if team_names else "Unassigned"
