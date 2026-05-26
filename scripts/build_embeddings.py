"""
Pre-Build Developer Embedding Profiles
=======================================
Run this ONCE after you have training data to build and cache
the semantic embedding profiles used by the Allocation Agent.

After running, profiles are saved to:
    MODELS/developer_embeddings_cache.pkl

Subsequent planning runs will load from cache instantly (~50ms)
instead of recomputing (~30-60s for first build).

Usage:
    cd "c:\\Users\\trifo\\OneDrive\\Υπολογιστής\\v0.1 - Antigrivity"
    python scripts/build_embeddings.py             # Build profiles
    python scripts/build_embeddings.py --report    # Build + print stats
    python scripts/build_embeddings.py --rebuild   # Force rebuild (ignore cache)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antigrivity.config import MODEL_DIR
from antigrivity.models.embedding_matcher import build_developer_profiles, invalidate_cache


def main():
    parser = argparse.ArgumentParser(description="Build developer embedding profiles")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild even if cache exists")
    parser.add_argument("--report", action="store_true",
                        help="Print detailed stats after build")
    args = parser.parse_args()

    # 1. Load training data
    TRAINING_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OUTPUT", "jira_with_timesheet_hours.csv")
    print("\n📂 Loading training data...")
    if not os.path.exists(TRAINING_DATA):
        print(f"❌ Training dataset not found at: {TRAINING_DATA}")
        print("   Run the merge script first:")
        print("   python scripts/01_merge_data.py")
        sys.exit(1)

    df = pd.read_csv(TRAINING_DATA)
    # Map new columns to expected internal names
    df = df.rename(columns={
        'JR_summary': 'feature_summary_clean',
        'JR_assignee': 'feature_assignee',
        'TS_timesheet_hours': 'target_actual_hours'
    })
    
    df = df.dropna(subset=["feature_summary_clean", "feature_assignee"])
    df = df[df["feature_summary_clean"].astype(str).str.strip() != ""]

    print(f"   Loaded {len(df)} tasks from {df['feature_assignee'].nunique()} developers")

    # 2. Build profiles
    if args.rebuild:
        print("\n🔄 Forcing cache rebuild...")
        invalidate_cache()

    print("\n🧠 Building embedding profiles...")
    print("   (First run downloads 'all-MiniLM-L6-v2' ~80MB if not cached)")

    import time
    t0 = time.time()
    profiles = build_developer_profiles(df, force_rebuild=args.rebuild)
    elapsed = time.time() - t0

    print(f"\n✅ Profiles built in {elapsed:.1f}s")
    print(f"   Cache saved to: {os.path.join(MODEL_DIR, 'developer_embeddings_cache.pkl')}")

    # 3. Optional detailed report
    if args.report or True:   # Always print summary
        print("\n" + "=" * 60)
        print("  DEVELOPER PROFILE SUMMARY")
        print("=" * 60)

        from antigrivity.config import TEAM
        for dev_name in TEAM:
            profile = profiles.get(dev_name)
            if profile:
                emb = profile["embeddings"]
                hours = profile["hours"]
                avg_h = np.mean(hours) if hours else 0
                print(f"\n  {dev_name}")
                print(f"    Tasks encoded : {profile['count']}")
                print(f"    Embedding dim : {emb.shape[1]}")
                print(f"    Avg hours/task: {avg_h:.1f}h")

                # Show top 5 most "representative" tasks
                # (task whose embedding is closest to the mean profile)
                if profile["count"] > 0:
                    mean_vec = np.mean(emb, axis=0)
                    sims = emb @ mean_vec
                    top_idx = np.argsort(sims)[-5:][::-1]
                    print(f"    Top tasks (most representative):")
                    for idx in top_idx:
                        s = profile["summaries"][idx]
                        print(f"      • {s[:70]}")
            else:
                print(f"\n  {dev_name}")
                print(f"    ⚠️  No historical tasks in training data")

        print("\n" + "=" * 60)
        print("\n💡 To test similarity lookup, run:")
        print('   python scripts/build_embeddings.py')
        print('   Then query in Python:')
        print('   from antigrivity.models.embedding_matcher import *')
        print('   profiles = build_developer_profiles(df)')
        print('   results = find_best_developer_by_embedding("Create landing page", list(profiles.keys()), profiles)')
        print('   print(results[:3])\n')


if __name__ == "__main__":
    main()
