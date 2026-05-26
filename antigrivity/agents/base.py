"""
Agent Base Class
=================
Defines the Agent interface that all planning agents must implement.

For data classes (Task, Constraint, SprintPlan, etc.), see antigrivity.core.models.
For the Orchestrator, see antigrivity.core.orchestrator.
"""


class Agent:
    """Base class for all agents in the planning system."""

    name: str = "BaseAgent"
    description: str = ""

    def run(self, context: dict) -> dict:
        """
        Execute the agent's logic.

        Parameters
        ----------
        context : dict
            Shared context passed between agents.

        Returns
        -------
        dict : updated context with agent's outputs
        """
        raise NotImplementedError

    def log(self, message: str):
        """Log an agent message (for the UI to display)."""
        print(f"[{self.name}] {message}")


# Backward-compatible re-exports so existing imports keep working
from antigrivity.core.models import Task, Constraint, DeveloperLoad, SprintPlan  # noqa: E402, F401
from antigrivity.core.orchestrator import Orchestrator, PipelineState, StateTransition  # noqa: E402, F401
