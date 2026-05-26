"""
Effort Estimator Wrapper
========================
High-level interface to the trained ML model for effort prediction.
Provides point estimates and probabilistic ranges.
"""

import numpy as np
import pandas as pd
from antigrivity.utils import (
    load_model, load_training_data, clean_text,
    predict_effort, predict_effort_range, get_developer_history
)
from antigrivity.config import TEAM


class EffortEstimator:
    """
    Wrapper around the trained sklearn pipeline for effort estimation.

    Provides:
    - Point estimates: predict()
    - Probabilistic ranges: predict_range() using individual tree predictions
    - Developer velocity profiles from historical data
    """

    def __init__(self):
        self.model = load_model(use_tuned=True)
        self.training_data = load_training_data()
        self.developer_profiles = get_developer_history(self.training_data)

    def predict(self, task_dict, assignee):
        """
        Predict effort in hours for a task assigned to a developer.

        Parameters
        ----------
        task_dict : dict with 'summary', 'type', 'project', 'sp'
        assignee : str

        Returns
        -------
        float : predicted hours
        """
        return predict_effort(self.model, task_dict, assignee)

    def predict_range(self, task_dict, assignee):
        """
        Predict effort range (P25, P50, P75) using ensemble tree variance.

        Returns
        -------
        tuple : (low, mid, high)
        """
        return predict_effort_range(self.model, task_dict, assignee)

    def predict_for_all_developers(self, task_dict, dev_list=None):
        """
        Predict effort for the specified team members (or all if dev_list is None).
        
        Parameters
        ----------
        task_dict : dict
            Task details (summary, sp, etc)
        dev_list : list of str, optional
            List of developer names to predict for.

        Returns
        -------
        list of dict: sorted by predicted hours (ascending)
            Each dict: {developer, predicted_hours, range_low, range_mid, range_high}
        """
        results = []
        target_devs = dev_list if dev_list is not None else list(TEAM.keys())
        
        for dev_name in target_devs:
            if dev_name not in TEAM:
                continue
            hours = self.predict(task_dict, dev_name)
            low, mid, high = self.predict_range(task_dict, dev_name)
            results.append({
                "developer": dev_name,
                "predicted_hours": hours,
                "range_low": low,
                "range_mid": mid,
                "range_high": high,
            })
        return sorted(results, key=lambda x: x["predicted_hours"])

    def get_developer_velocity(self, assignee, project_key=None):
        """
        Get historical velocity for a developer (hours per task, optionally filtered by project).

        Returns
        -------
        dict : {avg_hours, total_tasks, projects}
        """
        profile = self.developer_profiles.get(assignee)
        if not profile:
            return {"avg_hours": None, "total_tasks": 0, "projects": {}}

        if project_key:
            proj_data = self.training_data[
                (self.training_data['feature_assignee'] == assignee) &
                (self.training_data['feature_project_key'] == project_key)
            ]
            if not proj_data.empty:
                return {
                    "avg_hours": float(proj_data['target_actual_hours'].mean()),
                    "total_tasks": len(proj_data),
                    "projects": {project_key: len(proj_data)},
                }

        return profile

    def find_similar_tasks(self, task_dict, top_n=5):
        """
        Find the most similar historical tasks using TF-IDF similarity.

        Parameters
        ----------
        task_dict : dict with 'summary'
        top_n : int

        Returns
        -------
        DataFrame : top N similar tasks with their actual hours
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        df = self.training_data.copy()
        if df.empty:
            return pd.DataFrame()

        query = clean_text(task_dict.get('summary', ''))
        corpus = df['feature_summary_clean'].fillna('').tolist()

        vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        tfidf_matrix = vectorizer.fit_transform(corpus + [query])

        query_vec = tfidf_matrix[-1]
        corpus_matrix = tfidf_matrix[:-1]

        similarities = cosine_similarity(query_vec, corpus_matrix).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]

        result = df.iloc[top_indices][
            ['feature_summary_clean', 'feature_project_key', 'feature_assignee',
             'feature_story_points', 'target_actual_hours']
        ].copy()
        result['similarity'] = similarities[top_indices]

        return result
