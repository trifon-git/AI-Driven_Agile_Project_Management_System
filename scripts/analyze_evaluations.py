import os
import csv
from pathlib import Path

# Paths to the CSV files that PMs fill out
OUTPUT_DIR = Path("OUTPUT/eval_candidates")

def analyze_decomposition():
    """Analyze the PM scores for the Decomposition benchmark."""
    csv_path = OUTPUT_DIR / "decomposition_eval.csv"
    
    if not csv_path.exists():
        print(f"File not found: {csv_path}. Please run generate_benchmarks.py first.")
        return
        
    print("\n" + "=" * 50)
    print(" Decomposition Analysis Report ")
    print("=" * 50)
    
    model_stats = {}
    
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["Model"]
            pm_score = row.get("PM Score (1-5)", "").strip()
            completeness = row.get("Completeness (1-5)", "").strip()
            
            if not pm_score or not completeness:
                continue # Skip unrated
                
            try:
                score = int(pm_score)
                comp = int(completeness)
                
                if model not in model_stats:
                    model_stats[model] = {"total_score": 0, "total_comp": 0, "count": 0}
                    
                model_stats[model]["total_score"] += score
                model_stats[model]["total_comp"] += comp
                model_stats[model]["count"] += 1
                
            except ValueError:
                continue
                
    if not model_stats:
        print("No scored data found in the decomposition CSV yet.")
        return
        
    for model, stats in model_stats.items():
        avg_score = stats["total_score"] / stats["count"]
        avg_comp = stats["total_comp"] / stats["count"]
        print(f"Model: {model}")
        print(f"  Average PM Score:   {avg_score:.2f} / 5.0")
        print(f"  Avg Completeness:   {avg_comp:.2f} / 5.0")
        print(f"  Tasks Evaluated:    {stats['count']}")
        print("-" * 50)

def analyze_allocation():
    """Analyze the PM assignments and scores against AI assignments."""
    csv_path = OUTPUT_DIR / "allocation_eval.csv"
    
    if not csv_path.exists():
        print(f"File not found: {csv_path}. Please run generate_benchmarks.py first.")
        return
        
    print("\n" + "=" * 50)
    print(" Allocation Analysis Report ")
    print("=" * 50)
    
    algo_stats = {}
    
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row.get("Algorithm", "Unknown").strip()
            ai_assignee = row.get("AI Assignee", "").strip()
            pm_gt = row.get("PM Ground Truth Assignee", "").strip()
            score_str = row.get("Assignment Score (1-5)", "").strip()
            
            if not pm_gt:
                continue # Ground truth not filled out
                
            if algo not in algo_stats:
                algo_stats[algo] = {
                    "matches_gt": 0,
                    "total_tasks": 0,
                    "total_score": 0,
                    "scored_tasks": 0
                }
                
            algo_stats[algo]["total_tasks"] += 1
            
            if ai_assignee.lower() == pm_gt.lower():
                algo_stats[algo]["matches_gt"] += 1
                
            if score_str:
                try:
                    algo_stats[algo]["total_score"] += int(score_str)
                    algo_stats[algo]["scored_tasks"] += 1
                except ValueError:
                    pass
                    
    if not algo_stats:
        print("No ground truth data entered in the allocation CSV yet.")
        return
        
    for algo, stats in algo_stats.items():
        if stats["total_tasks"] == 0:
            continue
            
        accuracy = (stats["matches_gt"] / stats["total_tasks"]) * 100
        
        print(f"Algorithm: {algo}")
        print(f"  Total Tasks Evaluated:       {stats['total_tasks']}")
        print(f"  Accuracy vs PM Ground Truth: {accuracy:.1f}%")
        
        if stats["scored_tasks"] > 0:
            avg_score = stats["total_score"] / stats["scored_tasks"]
            print(f"  Average Subjective Score:    {avg_score:.2f} / 5.0")
        print("-" * 50)

if __name__ == "__main__":
    analyze_decomposition()
    analyze_allocation()
    print("\nNote: Fill out the CSV files in OUTPUT/eval_candidates/ before running this script for meaningful results.")
