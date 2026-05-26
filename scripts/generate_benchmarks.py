from asyncio import base_tasks
import os
import sys
import csv
import json
import copy
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import antigrivity.config
from antigrivity.agents.decomposition import DecompositionAgent
from antigrivity.agents.estimation import EstimationAgent
from antigrivity.agents.allocation import AllocationAgent
from antigrivity.agents.allocation_gnn import GNNAllocationAgent

# Predefined projects for evaluation
PROJECTS = [
    {
        "key": "EVAL1",
        "description": "Design and implement a single page website to showcase Imiskoubrias upcoming show and boost their merch sales via stampaland's eshop.  Project: Static, no admin. Technologies used: Next js"
    },
    {
        "key": "EVAL2",
        "description": "Design and implement a 5page website for ambassador travel, a local transfers and tours  company based on previous project White Hawk. The pages are, Homepage (with custom design feedback from client), tours, transfers, about and contact. They also wanted the extra feature of the whats app chat bubble already implemented on white hawk. Technologies: payload Cms"
    },
    {
        "key": "EVAL3",
        "description": "Implement a store theme ecommerce solution for equin brandhouse. Basic Web Bunch ecommerce implementation, no extra features."
    }
]

# Models to test for task splitting
LLM_MODELS = [
    # "gemma4:e2b",
    # "gemma4:e4b",
    # "gemma4:26b",
    # "gemma4:31b",
    # "qwen3.5:9b",
    # "qwen3.5:4b",
    # "qwen3.5:27b",
    # "qwen3.5:35b",
    "qwen3.6:27b",
    # "qwen3.6:35b",
    # "Llama3.2:latest",
    # "llama3.3:latest",
    # "gemma3:4b",
    # "gemma3:12b",
    # "gemma3:27b",
    # "gpt-oss:20b",
    # "nemotron3:33b",
]

OUTPUT_DIR = Path("OUTPUT/eval_candidates")

def generate_decomposition_benchmarks():
    """Run different LLMs to generate task splits and save to CSV."""
    print("--- Running Decomposition Benchmarks ---")
    
    csv_path_trad = OUTPUT_DIR / "decomposition_eval_traditional.csv"
    csv_path_gnn = OUTPUT_DIR / "decomposition_eval_gnn.csv"
    
    # Headers for manual evaluation (removed Algorithm)
    headers = [
        "Project Key", "Project Description", "Model", "Task ID", "Type", "Summary", "Description", "Story Points",
        "Predicted Hours", "AI Assignee", "PM Score (1-5)", "Completeness (1-5)", "Assignment Score (1-5)", "PM Ground Truth Assignee", "Comments"
    ]
    
    completed_runs = set()
    file_exists = csv_path_trad.exists()
    if file_exists:
        with open(csv_path_trad, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "Project Key" in row and "Model" in row:
                    completed_runs.add((row["Project Key"], row["Model"]))
    
    mode_trad = "a" if file_exists else "w"
    mode_gnn = "a" if csv_path_gnn.exists() else "w"

    with open(csv_path_trad, mode=mode_trad, newline="", encoding="utf-8") as f_trad, \
         open(csv_path_gnn, mode=mode_gnn, newline="", encoding="utf-8") as f_gnn:
        
        writer_trad = csv.DictWriter(f_trad, fieldnames=headers)
        if f_trad.tell() == 0:
            writer_trad.writeheader()
            
        writer_gnn = csv.DictWriter(f_gnn, fieldnames=headers)
        if f_gnn.tell() == 0:
            writer_gnn.writeheader()
        
        for model_name in LLM_MODELS:
            pending_projects = [p for p in PROJECTS if (p["key"], model_name) not in completed_runs]
            
            if not pending_projects:
                print(f"Skipping {model_name} (all projects already evaluated).")
                continue
                
            print(f"\n--- Pulling model {model_name} ---")
            import subprocess
            subprocess.run(["ollama", "pull", model_name])
            
            for project in pending_projects:
                print(f"Generating tasks for {project['key']} using {model_name}...")
                
                # Override config
                antigrivity.config.OLLAMA_MODEL = model_name
                
                try:
                    agent = DecompositionAgent()
                    est_agent = EstimationAgent()
                    context = {
                        "project_description": project["description"],
                        "project_key": project["key"],
                        "tasks": []
                    }

                    context = agent.run(context)
                    base_tasks = context.get("tasks", [])

                    if not base_tasks:
                        print(f"Warning: {model_name} returned no tasks.")
                        continue
                    
                    # Run Traditional Allocation -> Estimation
                    print("Running Traditional Allocation and Estimation...")
                    trad_agent = AllocationAgent()
                    trad_context = {"tasks": copy.deepcopy(base_tasks), "constraints": []}
                    trad_context = trad_agent.run(trad_context)
                    
                    trad_est_context = {"tasks": trad_context["plan"].tasks}
                    trad_est_context = est_agent.run(trad_est_context)
                    trad_final_tasks = trad_est_context.get("tasks", [])
                    
                    trad_assignments = {t.id: t.assigned_to for t in trad_final_tasks}
                    trad_hours = {t.id: getattr(t, 'predicted_hours', "") for t in trad_final_tasks}
                    
                    # Run GNN Allocation -> Estimation
                    print("Running GNN Allocation and Estimation...")
                    gnn_agent = GNNAllocationAgent()
                    gnn_context = {"tasks": copy.deepcopy(base_tasks), "constraints": []}
                    
                    try:
                        gnn_context = gnn_agent.run(gnn_context)
                        
                        gnn_est_context = {"tasks": gnn_context["plan"].tasks}
                        gnn_est_context = est_agent.run(gnn_est_context)
                        gnn_final_tasks = gnn_est_context.get("tasks", [])
                        
                        gnn_assignments = {t.id: t.assigned_to for t in gnn_final_tasks}
                        gnn_hours = {t.id: getattr(t, 'predicted_hours', "") for t in gnn_final_tasks}
                    except Exception as e:
                        print(f"Warning: GNN Allocation failed: {e}")
                        gnn_final_tasks = None
                        gnn_assignments = {}
                        gnn_hours = {}

                    for task in base_tasks:
                        desc = task.description
                        
                        # Traditional Row
                        writer_trad.writerow({
                            "Project Key": project["key"],
                            "Project Description": project["description"],
                            "Model": model_name,
                            "Task ID": task.id,
                            "Type": task.type,
                            "Summary": task.summary,
                            "Description": desc,
                            "Story Points": task.sp,
                            "Predicted Hours": trad_hours.get(task.id, ""),
                            "AI Assignee": trad_assignments.get(task.id, "Unassigned"),
                            "PM Score (1-5)": "",
                            "Completeness (1-5)": "",
                            "Assignment Score (1-5)": "",
                            "PM Ground Truth Assignee": "",
                            "Comments": ""
                        })
                        
                        # GNN Row
                        writer_gnn.writerow({
                            "Project Key": project["key"],
                            "Project Description": project["description"],
                            "Model": model_name,
                            "Task ID": task.id,
                            "Type": task.type,
                            "Summary": task.summary,
                            "Description": desc,
                            "Story Points": task.sp,
                            "Predicted Hours": gnn_hours.get(task.id, ""),
                            "AI Assignee": gnn_assignments.get(task.id, "Unassigned") if gnn_final_tasks else "N/A",
                            "PM Score (1-5)": "",
                            "Completeness (1-5)": "",
                            "Assignment Score (1-5)": "",
                            "PM Ground Truth Assignee": "",
                            "Comments": ""
                        })

                    # Force flush to disk immediately after each project/model combo finishes
                    f_trad.flush()
                    os.fsync(f_trad.fileno())
                    f_gnn.flush()
                    os.fsync(f_gnn.fileno())
                except Exception as e:
                    print(f"Error running {model_name}: {e}")
            
            print(f"Removing model {model_name} to free up space...")
            subprocess.run(["ollama", "rm", model_name])
            
            print(f"Waiting 10 seconds before the next model...")
            import time
            time.sleep(10)
                    
    print(f"Decomposition benchmarks saved to:\n  - {csv_path_trad}\n  - {csv_path_gnn}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_decomposition_benchmarks()
    print("\nBenchmark generation complete! Check OUTPUT/eval_candidates/ directory.")
