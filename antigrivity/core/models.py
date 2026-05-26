"""
Core Data Models
=================
Defines the shared data classes used across the entire planning system:
Task, Constraint, DeveloperLoad, and SprintPlan.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any



@dataclass
class Task:
    """A single decomposed task in the sprint backlog."""
    id: int
    summary: str
    type: str                         # Bug, Story, Task, Sub-task
    project: str                      # Jira project key
    category: str = "Developer"       # Developer, Designer, Manager, Operations
    description: str = ""             # Detailed instructions/context
    sp: float = 0.0                   # Story points
    priority: str = "Medium"          # High, Medium, Low
    dependencies: List[int] = field(default_factory=list)
    predicted_hours: float = 0.0
    range_low: float = 0.0
    range_mid: float = 0.0
    range_high: float = 0.0
    assigned_to: Optional[str] = None
    reasoning: str = ""               # Why this estimate / assignment

    def to_dict(self):
        return {
            "id": self.id,
            "summary": self.summary,
            "description": self.description,
            "type": self.type,
            "category": self.category,
            "project": self.project,
            "sp": self.sp,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "predicted_hours": self.predicted_hours,
            "range_low": self.range_low,
            "range_mid": self.range_mid,
            "range_high": self.range_high,
            "assigned_to": self.assigned_to,
            "reasoning": self.reasoning,
        }


@dataclass
class DeveloperLoad:
    """Tracks a developer's assigned tasks and workload."""
    name: str
    capacity_hours: float
    assigned_tasks: List[Task] = field(default_factory=list)
    total_hours: float = 0.0

    @property
    def utilization(self):
        return (self.total_hours / self.capacity_hours * 100) if self.capacity_hours > 0 else 0

    @property
    def remaining_hours(self):
        return max(0, self.capacity_hours - self.total_hours)


@dataclass
class Constraint:
    """
    A planning constraint from the PM's feedback.
    
    Types:
    - max_hours: Hard cap on hours (e.g., for leave)
    - capacity_boost: Increase cap for overtime
    - assign_task: Force assignment (Hard bound)
    - soft_assign: Prefer assignment (Weighted preference)
    - ban_developer: Prevent assignment
    - defer_task: Move out of sprint
    - change_priority: Set level (High/Med/Low)
    - change_sp: Override story points
    - force_hours: Override ML estimate with fixed hours
    - change_category: Move between Developer/Designer/Operations
    """
    type: str
    developer: Optional[str] = None
    task_id: Optional[int] = None
    task_summary: Optional[str] = None
    value: Any = None
    reason: str = ""

    def to_dict(self):
        return {k: v for k, v in {
            "type": self.type,
            "developer": self.developer,
            "task_id": self.task_id,
            "task_summary": self.task_summary,
            "value": self.value,
            "reason": self.reason,
        }.items() if v is not None}


@dataclass
class SprintPlan:
    """The complete sprint plan output."""
    tasks: List[Task] = field(default_factory=list)
    developer_loads: Dict[str, DeveloperLoad] = field(default_factory=dict)
    unassigned_tasks: List[Task] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    version: int = 1
    timestamp: str = ""

    def summary_stats(self):
        total_hours = sum(dl.total_hours for dl in self.developer_loads.values())
        total_capacity = sum(dl.capacity_hours for dl in self.developer_loads.values())
        return {
            "total_tasks": len(self.tasks),
            "assigned_tasks": len(self.tasks) - len(self.unassigned_tasks),
            "unassigned_tasks": len(self.unassigned_tasks),
            "total_hours": round(total_hours, 1),
            "total_capacity": total_capacity,
            "team_utilization": round(total_hours / total_capacity * 100, 1) if total_capacity > 0 else 0,
            "num_constraints": len(self.constraints),
        }
