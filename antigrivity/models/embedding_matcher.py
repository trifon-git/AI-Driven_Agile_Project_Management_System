"""
Embedding-Based Developer Similarity Search
=============================================
Uses sentence-transformers to match new tasks to the developer
who has historically done the most semantically similar work.

This is more powerful than keyword/TF-IDF matching because it
understands meaning, not just word overlap.

Example: "Add payment gateway" ≈ "Integrate Stripe checkout"
         even when they share no common words.
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
from antigrivity.config import MODEL_DIR

# Cache paths
_CACHE_DIR = MODEL_DIR
_EMBEDDINGS_CACHE = os.path.join(_CACHE_DIR, "developer_embeddings_cache.pkl")

# Module-level singletons (lazy loaded)
_model = None
_cache = None  # {developer: {"summaries": [...], "embeddings": np.ndarray}}


def _get_model():
    """Lazy-load the sentence transformer model (cached after first call)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # 'intfloat/multilingual-e5-large' provides advanced multilingual
            # semantic matching using 1024 dimensional embeddings 
            _model = SentenceTransformer("intfloat/multilingual-e5-large")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
    return _model


def build_developer_profiles(training_df: pd.DataFrame, force_rebuild: bool = False) -> dict:
    """
    Build embedding profiles for each developer from training data.

    For each developer, encodes all their historical task summaries
    into vectors and computes a profile matrix.

    Parameters
    ----------
    training_df : pd.DataFrame
        The training dataset with 'feature_assignee' and 'feature_summary_clean'.
    force_rebuild : bool
        If True, ignore cached profiles and rebuild from scratch.

    Returns
    -------
    dict : {developer_name: {"summaries": [...], "embeddings": np.ndarray, "hours": [...]}}
    """
    global _cache

    # Try loading from cache
    if not force_rebuild and os.path.exists(_EMBEDDINGS_CACHE):
        _cache = joblib.load(_EMBEDDINGS_CACHE)
        print(f"[EmbeddingMatcher] Loaded developer profiles from cache "
              f"({len(_cache)} developers)")
        return _cache

    print("[EmbeddingMatcher] Building developer embedding profiles...")
    model = _get_model()

    profiles = {}
    for dev_name, group in training_df.groupby("feature_assignee"):
        summaries = group["feature_summary_clean"].fillna("").tolist()
        hours = group["target_actual_hours"].tolist() if "target_actual_hours" in group else []

        if not summaries:
            continue

        # Encode all summaries for this developer in one batch
        embeddings = model.encode(summaries, batch_size=64, show_progress_bar=False,
                                   normalize_embeddings=True)

        profiles[dev_name] = {
            "summaries": summaries,
            "embeddings": embeddings,   # shape: (num_tasks, embedding_dim)
            "hours": hours,
            "count": len(summaries),
        }

    # Save cache
    os.makedirs(_CACHE_DIR, exist_ok=True)
    joblib.dump(profiles, _EMBEDDINGS_CACHE)
    print(f"[EmbeddingMatcher] Built profiles for {len(profiles)} developers. "
          f"Cache saved to {_EMBEDDINGS_CACHE}")

    _cache = profiles
    return profiles


def find_best_developer_by_embedding(task_summary: str,
                                      candidate_developers: list,
                                      profiles: dict,
                                      top_k: int = 3) -> list:
    """
    Find the best developer(s) for a task using semantic similarity.

    Encodes the task summary and computes cosine similarity against
    each developer's historical task embeddings. The developer with
    the highest average similarity to their top-K past tasks wins.

    Parameters
    ----------
    task_summary : str
        The new task summary to match.
    candidate_developers : list of str
        Subset of developers to consider (for constraint enforcement).
    profiles : dict
        Developer embedding profiles from build_developer_profiles().
    top_k : int
        How many of each developer's nearest tasks to average over.

    Returns
    -------
    list of dict : ranked candidates, each with:
        {developer, score, top_matches: [{summary, similarity, hours}]}
    """
    if not profiles:
        # Fallback: return candidates with equal score
        return [{"developer": d, "score": 0.0, "top_matches": []} for d in candidate_developers]

    model = _get_model()
    # E5 models require 'query: ' prefix
    query_embedding = model.encode([f"query: {task_summary}"], normalize_embeddings=True)[0]  # shape: (dim,)

    results = []
    for dev_name in candidate_developers:
        profile = profiles.get(dev_name)
        if not profile:
            # Developer has no history — assign neutral score
            results.append({
                "developer": dev_name,
                "score": 0.0,
                "top_matches": [],
                "num_historical_tasks": 0,
            })
            continue

        dev_embeddings = profile["embeddings"]  # shape: (num_tasks, dim)
        summaries = profile["summaries"]
        hours = profile["hours"]

        # Cosine similarity: query · each task embedding (both normalized)
        similarities = cosine_similarity([query_embedding], dev_embeddings).flatten()  # shape: (num_tasks,)

        # Get top-K most similar tasks
        k = min(top_k, len(similarities))
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        top_sim = similarities[top_indices]
        avg_score = float(np.mean(top_sim))

        top_matches = []
        for idx in top_indices:
            top_matches.append({
                "summary": summaries[idx],
                "similarity": float(similarities[idx]),
                "hours": hours[idx] if hours else None,
            })

        results.append({
            "developer": dev_name,
            "score": avg_score,
            "top_matches": top_matches,
            "num_historical_tasks": profile["count"],
        })

    # Sort by similarity score (descending — higher is better)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def find_global_closest_tasks(task_summary: str, candidate_developers: list, profiles: dict, top_k: int = 5) -> list:
    """
    Finds the single closest historical tasks globally across all developers,
    ignoring developer averages entirely.
    """
    if not profiles:
        return []

    model = _get_model()
    # e5 models require 'query: ' prefix for downstream asymmetric semantic tasks
    query_embedding = model.encode([f"query: {task_summary}"], normalize_embeddings=True)[0]

    all_tasks = []
    
    for dev_name in candidate_developers:
        profile = profiles.get(dev_name)
        if not profile:
            continue
            
        dev_embeddings = profile["embeddings"]
        # Use explicit cosine similarity from scikit-learn
        similarities = cosine_similarity([query_embedding], dev_embeddings).flatten()
        
        # We only need the top relatively few from each dev to save memory sorting
        k = min(top_k, len(similarities))
        top_indices = np.argpartition(similarities, -k)[-k:]
        
        for idx in top_indices:
            all_tasks.append({
                "developer": dev_name,
                "similarity": float(similarities[idx]),
                "summary": profile["summaries"][idx],
                "hours": profile["hours"][idx] if profile["hours"] else None
            })
            
    # Sort globally by similarity (highest first)
    all_tasks.sort(key=lambda x: x["similarity"], reverse=True)
    return all_tasks[:top_k]


def get_developer_similarity_score(task_summary: str, dev_name: str, profiles: dict) -> float:
    """
    Get a single similarity score for one developer/task pair.

    Returns 0.0 if the developer has no history.
    """
    results = find_best_developer_by_embedding(task_summary, [dev_name], profiles, top_k=5)
    return results[0]["score"] if results else 0.0


def invalidate_cache():
    """Delete the embeddings cache (force rebuild on next use)."""
    global _cache
    _cache = None
    if os.path.exists(_EMBEDDINGS_CACHE):
        os.remove(_EMBEDDINGS_CACHE)
        print("[EmbeddingMatcher] Cache invalidated.")
