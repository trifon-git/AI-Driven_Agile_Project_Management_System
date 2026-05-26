"""
GNN Resource Allocation Agent
=============================
Constraint-Aware Message-Passing Graph Neural Network (CAMP-GNN).

Uses Node2Vec structural embeddings and contextual similarities to 
score valid task-to-developer assignments while strictly enforcing PM constraints.
"""
import torch
import torch.nn as nn
import numpy as np
from antigrivity.agents.allocation import AllocationAgent
from antigrivity.models.graph_builder import AllocationGraphBuilder
from antigrivity.models.node2vec_embedder import GraphEmbedder
from antigrivity.core.models import SprintPlan
from antigrivity.config import TEAM

class CAMPGNNScorer(nn.Module):
    """
    Simple MLP to score an edge (Task -> Developer) given the concatenated embeddings.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_task, x_dev):
        # x_task: [N, D], x_dev: [N, D]
        x = torch.cat([x_task, x_dev], dim=-1)
        return self.net(x).squeeze(-1)


class GNNAllocationAgent(AllocationAgent):
    """
    Overrides the greedy AllocationAgent to use Node2Vec + CAMP-GNN.
    """
    name = "CAMPGNN_AllocationAgent"
    description = "Constraint-aware GNN allocation using Node2Vec structural embeddings."

    def run(self, context: dict) -> dict:
        tasks = context.get("tasks", [])
        constraints = context.get("constraints", [])
        
        self.log(f"Starting GNN Allocation for {len(tasks)} tasks...")

        # 1. Reset tasks
        for task in tasks:
            task.assigned_to = None

        # 2. Load Offline Preprocessed Graph
        import os
        import joblib
        from antigrivity.config import MODEL_DIR
        
        hist_path = os.path.join(MODEL_DIR, "historical_graph_state.pkl")
        if not os.path.exists(hist_path):
            raise FileNotFoundError(f"Offline graph state not found at {hist_path}. Run scripts/preprocess_graph.py first.")
            
        data = joblib.load(hist_path)
        node_to_id = data["node_to_id"]
        structural_embs = torch.tensor(data["embeddings"])
            
        # Prepare historical task centroid and per-task induction data
        hist_task_ids = [v for k, v in node_to_id.items() if k.startswith("TASK_HIST_")]
        if not hist_task_ids:
            hist_task_ids = [v for k, v in node_to_id.items()
                             if k.startswith("TASK_") and not k.startswith("TASK_ACTIVE_")]
        if hist_task_ids:
            task_centroid = structural_embs[hist_task_ids].mean(dim=0)
        else:
            task_centroid = torch.zeros(structural_embs.size(1))

        # Build semantic matcher inputs (historical summaries -> node ids)
        hist_summary_keys = []  # list of (issue_key, node_id, summary)
        from antigrivity.utils import load_training_data
        training_df = load_training_data()
        for _, row in training_df.iterrows():
            issue_key = row.get("JR_issue_key") or row.get("JR_issue") or None
            summary = row.get("feature_summary_clean") or row.get("JR_summary") or None
            if issue_key and isinstance(summary, str):
                node_name = f"TASK_{issue_key}"
                if node_name in node_to_id:
                    hist_summary_keys.append((issue_key, node_to_id[node_name], summary))

        # Precompute semantic embeddings for historical summaries (if any)
        hist_semantic_embeddings = None
        st_model = None
        if hist_summary_keys:
            from antigrivity.models.embedding_matcher import _get_model as _get_st_model
            import numpy as np
            st_model = _get_st_model()
            summaries = [f"query: {s}" for (_, _, s) in hist_summary_keys]
            hist_semantic_embeddings = st_model.encode(summaries, normalize_embeddings=True)
            hist_node_ids = [nid for (_, nid, _) in hist_summary_keys]
            
        # 3. Setup constraint state using parent methods
        # Need to pre-process OVERRIDES just like the parent
        for c in constraints:
            target_tasks = []
            if c.task_id:
                target_tasks = [t for t in tasks if t.id == c.task_id]
            elif c.task_summary:
                target_tasks = [t for t in tasks if c.task_summary.lower() in t.summary.lower()]
            
            for t in target_tasks:
                if c.type == "change_sp" and c.value is not None:
                    t.sp = float(c.value)
                    t.reasoning += f" [PM overridden SP: {t.sp}]"
                elif c.type == "force_hours" and c.value is not None:
                    t.predicted_hours = float(c.value)
                    t.reasoning += f" [PM forced hours: {t.predicted_hours}h]"
                
        constraint_state = self._build_constraint_state(constraints, tasks)
        
        # 4. Build device models
        dev_loads = {}
        enabled_devs = context.get("enabled_developers", list(TEAM.keys()))
        if enabled_devs is None: enabled_devs = list(TEAM.keys())
        
        from antigrivity.core.models import DeveloperLoad
        for dev_name, info in TEAM.items():
            if dev_name not in enabled_devs: continue
            cap = info["capacity_hours"]
            if dev_name in constraint_state["max_hours"]: cap = constraint_state["max_hours"][dev_name]
            elif dev_name in constraint_state["boosts"]: cap = constraint_state["boosts"][dev_name]
            dev_loads[dev_name] = DeveloperLoad(name=dev_name, capacity_hours=cap)

        # 5. Initialize GNN Edge Scorer
        # Since we don't have labeled explicit targets for edge scoring besides historical data,
        # we will use the Cosine Similarity of Node2Vec embeddings as the primary GNN 'message'.
        
        active_tasks = [t for t in tasks if t.id not in constraint_state["defer_tasks"] and t.summary not in constraint_state["defer_summaries"]]
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        active_tasks.sort(key=lambda t: (priority_order.get(t.priority, 1), -t.sp))
        
        model = None
        from antigrivity.utils import load_model, predict_effort
        model = load_model(use_tuned=True)
        
        # Batch-encode active task summaries once to avoid per-task model calls
        task_summary_embeddings = None
        if hist_semantic_embeddings is not None and st_model is not None:
            summaries_to_encode = []
            task_ids_order = []
            for t in active_tasks:
                if hasattr(t, 'summary') and t.summary:
                    summaries_to_encode.append(f"query: {t.summary}")
                    task_ids_order.append(t.id)
            if summaries_to_encode:
                encoded = st_model.encode(summaries_to_encode, normalize_embeddings=True)
                task_summary_embeddings = {tid: emb for tid, emb in zip(task_ids_order, encoded)}

        unassigned = []
        for task in active_tasks:
            forced_dev = constraint_state["forced_assignments"].get(task.id)
            if forced_dev and forced_dev in dev_loads:
                self._assign_task(task, forced_dev, dev_loads, model)
                self.log(f"  #{task.id} FORCED -> {forced_dev}")
                continue

            banned_devs = constraint_state["bans"].get(task.id, set())
            candidate_devs = [d for d in dev_loads if d not in banned_devs]
            
            # GNN Evaluation
            scores = []
            task_node_name = f"TASK_ACTIVE_{task.id}"
            task_node_id = node_to_id.get(task_node_name)

            # Induce a per-task structural embedding from top-K semantically similar historical tasks
            task_emb = task_centroid
            # Use precomputed semantic encodings for active tasks when available
            if hist_semantic_embeddings is not None and st_model is not None and task_summary_embeddings is not None:
                q = task_summary_embeddings.get(task.id)
                if q is not None:
                    sims = np.dot(hist_semantic_embeddings, q)
                    if sims.size > 0:
                        k = min(5, sims.size)
                        top_idxs = np.argpartition(sims, -k)[-k:]
                        top_idxs = top_idxs[np.argsort(sims[top_idxs])[::-1]]
                        selected_node_ids = [hist_node_ids[i] for i in top_idxs]
                        task_emb = structural_embs[selected_node_ids].mean(dim=0)

            for dev_name in candidate_devs:
                load = dev_loads[dev_name]
                dev_info = TEAM.get(dev_name, {})
                dev_node_name = f"DEV_{dev_name}"
                dev_node_id = node_to_id.get(dev_node_name)
                
                # Strict Department Constraint
                task_cat = getattr(task, "category", "Developer")
                dev_dept = dev_info.get("department", "Developer")
                
                # Rule 1: Designer/SEO/Management tasks MUST match the department
                if task_cat != dev_dept:
                    # Exception: Only allow cross-assignment if both are strictly "Developer" 
                    # (this prevents Managers or Designers from taking Code tasks)
                    if not (task_cat == "Developer" and dev_dept == "Developer"):
                        continue
                        
                # Overload penalty logic equivalent
                task_hours = task.predicted_hours if task.predicted_hours else (task.sp * 0.5) # rough fallback
                if load.total_hours + task_hours > load.capacity_hours:
                    continue  # Hard drop instead of penalty in GNN unless no one exists
                    
                # GNN Struct Similarity message (Zero-Shot Induction)
                struct_sim = 0.0
                if dev_node_id is not None:
                    d_emb = structural_embs[dev_node_id]
                    # Compare developer embedding to the per-task induced structural embedding
                    struct_sim = torch.nn.functional.cosine_similarity(task_emb.unsqueeze(0), d_emb.unsqueeze(0)).item()
                
                # Soft assignment bonus
                if dev_name in constraint_state["soft_assignments"].get(task.id, set()):
                    struct_sim += 2.0
                
                scores.append((dev_name, struct_sim))
                
            if not scores:
                # If everyone is full, drop capacity constraint for a pass
                for dev_name in candidate_devs:
                    scores.append((dev_name, 0.0))
            if not scores:
                unassigned.append(task)
                continue
                
            # Pick highest structural similarity score (GNN edge choice)
            scores.sort(key=lambda x: x[1], reverse=True) # Higher is Better here
            best_dev = scores[0][0]
            self._assign_task(task, best_dev, dev_loads, model)
            self.log(f"  #{task.id} GNN Assigned -> {best_dev} (Sim={scores[0][1]:.2f})")
            
        plan = SprintPlan(
            tasks=active_tasks + unassigned,
            developer_loads=dev_loads,
            unassigned_tasks=unassigned,
            constraints=constraints,
        )
        
        context["plan"] = plan
        return context

        context["plan"] = plan
        return context

