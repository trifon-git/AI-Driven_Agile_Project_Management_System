"""
Graph Builder
=============
Constructs the assignment and dependency graph from historical training data
and current active Sprint tasks.

Produces `edge_index` and Node ID mappings for PyTorch Geometric processing.
"""
import networkx as nx
import torch
from antigrivity.utils import load_training_data
from antigrivity.config import TEAM

class AllocationGraphBuilder:
    def __init__(self):
        self.node_to_id = {}
        self.id_to_node = {}
        self.node_data = {}
        self.current_node_id = 0
        self.edge_list = []
        self.weights = []
        
    def _get_node_id(self, name: str, data: dict = None) -> int:
        if name not in self.node_to_id:
            node_id = self.current_node_id
            self.node_to_id[name] = node_id
            self.id_to_node[node_id] = name
            if data:
                self.node_data[node_id] = data
            self.current_node_id += 1
            return node_id
        return self.node_to_id[name]

    def build_from_history_and_active(self, active_tasks):
        """
        active_tasks: List of Task objects
        Returns: edge_index (2xN tensor), node mappings
        """
        # Load historical assignment data to learn Dev <-> Task type affinity
        df = load_training_data()
        
        # Add Dev Nodes
        for dev_name in TEAM.keys():
            self._get_node_id(f"DEV_{dev_name}")
            
        # Add Historical Task Assignments
        for _, row in df.iterrows():
            task_key = row.get("JR_issue_key", "UNKNOWN")
            assignee = row.get("feature_assignee", "Unassigned")
            summary = row.get("feature_summary_clean", "")
            project = row.get("feature_project_key", "UNK")
            
            if assignee in TEAM:
                dev_id = self._get_node_id(f"DEV_{assignee}")
                task_id = self._get_node_id(f"TASK_{task_key}", data={"summary": summary, "project": project})
                
                # Bi-directional assignment edge
                self.edge_list.append((dev_id, task_id))
                self.edge_list.append((task_id, dev_id))
                self.weights.extend([1.0, 1.0])
                
                # Add Parent Dependency if exists
                parent = row.get("JR_parent_key")
                if parent and str(parent) != "nan":
                    parent_id = self._get_node_id(f"TASK_{parent}")
                    # Directed edge: task depends on parent
                    self.edge_list.append((task_id, parent_id))
                    self.weights.append(1.0)
                    
        # Add Active Tasks from current context
        for task in active_tasks:
            task_id = self._get_node_id(f"TASK_ACTIVE_{task.id}", data={"summary": getattr(task, 'summary', ''), "project": getattr(task, 'project', '')})
            
            # Dependencies
            for dep_id in getattr(task, "dependencies", []):
                dep_node = self._get_node_id(f"TASK_ACTIVE_{dep_id}")
                self.edge_list.append((task_id, dep_node))
                self.weights.append(2.0) # Active dependencies have higher weight

        if not self.edge_list:
            return torch.empty((2, 0), dtype=torch.long), self.node_to_id, self.id_to_node, self.node_data

        edge_index = torch.tensor(self.edge_list, dtype=torch.long).t().contiguous()
        return edge_index, self.node_to_id, self.id_to_node, self.node_data

