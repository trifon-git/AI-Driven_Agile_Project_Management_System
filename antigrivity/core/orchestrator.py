"""
Orchestrator (State Machine)
=============================
Coordinates the multi-agent planning pipeline using a state machine.
Each state either calls an agent or performs a validation check.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List
import time
import copy

from antigrivity.core.models import Task, Constraint, DeveloperLoad, SprintPlan



class PipelineState(Enum):
    """States in the orchestrator's planning pipeline."""
    DECOMPOSE = "decompose"
    VALIDATE_TASKS = "validate_tasks"
    RETRY_DECOMPOSE = "retry_decompose"
    ESTIMATE = "estimate"
    VALIDATE_ESTIMATES = "validate_estimates"
    ALLOCATE = "allocate"
    REVIEW = "review"
    AUTO_REBALANCE = "auto_rebalance"
    DONE = "done"


@dataclass
class StateTransition:
    """Records one state transition in the pipeline."""
    from_state: str
    to_state: str
    reason: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%H:%M:%S")


# Thresholds for validation gates
_MIN_TASKS = 2               # Retry decomposition if fewer tasks
_MAX_TASK_HOURS = 100.0      # Clamp any estimate above this
_MIN_TASK_HOURS = 0.1        # Clamp any estimate below this
_MAX_UTIL_PCT = 120.0        # Trigger auto-rebalance above this
_MAX_UNASSIGNED_PCT = 30.0   # Trigger auto-rebalance above this
_MAX_ITERATIONS = 8          # Safety cap to prevent infinite loops


class Orchestrator:
    """
    Coordinates the multi-agent planning pipeline using a state machine.

    Each state either calls an agent or performs a validation check.
    Transitions are driven by inspecting the previous output, enabling
    automatic retries, outlier correction, and workload rebalancing.

    States: decompose → validate_tasks → allocate → estimate → validate_estimates
            → review → done

    Conditional paths:
        validate_tasks     → retry_decompose  (if too few tasks)
        validate_estimates → clamp outliers   (if extreme values)
        review             → auto_rebalance   (loop back to allocate if overloaded)
    """

    def __init__(self):
        from antigrivity.agents.decomposition import DecompositionAgent
        from antigrivity.agents.estimation import EstimationAgent
        from antigrivity.agents.allocation import AllocationAgent
        from antigrivity.agents.negotiation import NegotiationAgent
        from antigrivity.config import USE_GNN_ALLOCATION

        self.decomposition = DecompositionAgent()
        self.estimation = EstimationAgent()
        
        if USE_GNN_ALLOCATION:
            from antigrivity.agents.allocation_gnn import GNNAllocationAgent
            self.allocation = GNNAllocationAgent()
        else:
            self.allocation = AllocationAgent()
            
        self.negotiation = NegotiationAgent()
        self.current_plan = None
        self.plan_history = []
        self.state_log: List[StateTransition] = []
        self._last_context = {}

    # Public API (unchanged signatures)

    def run_planning_cycle(self, project_description: str, project_key: str = "PROJ",
                           constraints: List[Constraint] = None,
                           enabled_developers: List[str] = None) -> SprintPlan:
        """
        Run the full agent pipeline as a state machine.

        Parameters
        ----------
        project_description : str
            Free-text description of the project/sprint requirements.
        project_key : str
            Jira project key for the tasks.
        constraints : list of Constraint, optional
            Pre-existing constraints.
        enabled_developers : list of str, optional
            List of developer names to consider for this cycle.

        Returns
        -------
        SprintPlan
        """
        context = {
            "project_description": project_description,
            "project_key": project_key,
            "constraints": constraints or [],
            "tasks": [],
            "plan": None,
            "enabled_developers": enabled_developers,
        }

        retries = {"decompose": 1, "estimate": 1, "rebalance": 2}
        state = PipelineState.DECOMPOSE
        self.state_log = []
        iterations = 0

        while state != PipelineState.DONE and iterations < _MAX_ITERATIONS:
            iterations += 1
            prev_state = state

            if state == PipelineState.DECOMPOSE:
                context = self.decomposition.run(context)
                self._log_agent_output("DecompositionAgent", context)
                state = PipelineState.VALIDATE_TASKS
                self._log_transition(prev_state, state, "Decomposition complete")

            elif state == PipelineState.VALIDATE_TASKS:
                tasks = context.get("tasks", [])
                if len(tasks) >= _MIN_TASKS:
                    state = PipelineState.ALLOCATE
                    self._log_transition(prev_state, state,
                                         f"{len(tasks)} tasks produced — valid")
                elif retries["decompose"] > 0:
                    retries["decompose"] -= 1
                    state = PipelineState.RETRY_DECOMPOSE
                    self._log_transition(prev_state, state,
                                         f"Only {len(tasks)} tasks — retrying with enriched prompt")
                else:
                    state = PipelineState.ALLOCATE
                    self._log_transition(prev_state, state,
                                         f"Only {len(tasks)} tasks but retries exhausted — proceeding")

            elif state == PipelineState.RETRY_DECOMPOSE:
                context["project_description"] = (
                    context["project_description"]
                    + "\n\nIMPORTANT: Please break this down into MORE granular, "
                    "smaller tasks. At least 5-8 tasks are expected."
                )
                context["tasks"] = []  # Reset
                context = self.decomposition.run(context)
                self._log_agent_output("DecompositionAgent", context)
                state = PipelineState.VALIDATE_TASKS
                self._log_transition(prev_state, state, "Re-decomposition complete")

            elif state == PipelineState.ALLOCATE:
                context = self.allocation.run(context)
                self._log_agent_output("AllocationAgent", context)
                state = PipelineState.ESTIMATE
                self._log_transition(prev_state, state, "Allocation complete")

            elif state == PipelineState.ESTIMATE:
                context = self.estimation.run(context)
                self._log_agent_output("EstimationAgent", context)
                state = PipelineState.VALIDATE_ESTIMATES
                self._log_transition(prev_state, state, "Estimation complete")

            elif state == PipelineState.VALIDATE_ESTIMATES:
                tasks = context.get("tasks", [])
                clamped = 0
                for task in tasks:
                    if task.predicted_hours > _MAX_TASK_HOURS:
                        task.predicted_hours = _MAX_TASK_HOURS
                        task.reasoning += f" [clamped from >{_MAX_TASK_HOURS}h]"
                        clamped += 1
                    elif task.predicted_hours < _MIN_TASK_HOURS:
                        task.predicted_hours = _MIN_TASK_HOURS
                        task.reasoning += f" [clamped from <{_MIN_TASK_HOURS}h]"
                        clamped += 1

                # Update total hours in plan loads since estimates may have changed
                if "plan" in context:
                    for dl in context["plan"].developer_loads.values():
                        dl.total_hours = sum(t.predicted_hours for t in dl.assigned_tasks)

                state = PipelineState.REVIEW
                reason = "Estimates valid" if clamped == 0 else f"Clamped {clamped} outlier estimate(s)"
                self._log_transition(prev_state, state, reason)

            elif state == PipelineState.REVIEW:
                plan = context.get("plan")
                if not plan:
                    state = PipelineState.DONE
                    self._log_transition(prev_state, state, "No plan produced — finishing")
                    continue

                # Check utilization
                max_util = max(
                    (dl.utilization for dl in plan.developer_loads.values()),
                    default=0
                )
                total_tasks = len(plan.tasks)
                unassigned_pct = (
                    len(plan.unassigned_tasks) / total_tasks * 100
                    if total_tasks > 0 else 0
                )

                issues = []
                if max_util > _MAX_UTIL_PCT:
                    issues.append(f"max utilization {max_util:.0f}%")
                if unassigned_pct > _MAX_UNASSIGNED_PCT:
                    issues.append(f"unassigned {unassigned_pct:.0f}%")

                if not issues or retries["rebalance"] <= 0:
                    state = PipelineState.DONE
                    reason = "Plan acceptable" if not issues else (
                        f"Issues remain ({', '.join(issues)}) but rebalance retries exhausted"
                    )
                    self._log_transition(prev_state, state, reason)
                else:
                    retries["rebalance"] -= 1
                    state = PipelineState.AUTO_REBALANCE
                    self._log_transition(prev_state, state,
                                         f"Auto-rebalancing: {', '.join(issues)}")

            elif state == PipelineState.AUTO_REBALANCE:
                plan = context["plan"]
                # Add a ban_developer constraint to explicitly remove them from their largest task
                for dev_name, dl in plan.developer_loads.items():
                    if dl.utilization > _MAX_UTIL_PCT and dl.assigned_tasks:
                        # Find the task that takes the most hours for this overloaded dev
                        largest_task = max(dl.assigned_tasks, key=lambda t: t.predicted_hours)
                        context["constraints"].append(Constraint(
                            type="ban_developer",
                            developer=dev_name,
                            task_id=largest_task.id,
                            reason=f"Auto-rebalance: overloaded at {dl.utilization:.0f}% capacity",
                        ))
                state = PipelineState.ALLOCATE
                self._log_transition(prev_state, state,
                                     "Auto-constraints added — re-allocating")

        # Finalize
        if context.get("plan"):
            self.current_plan = context["plan"]
        else:
            # Edge case: pipeline ended without producing a plan
            self.current_plan = SprintPlan()

        self.current_plan.version = len(self.plan_history) + 1
        self.current_plan.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.plan_history.append(self.current_plan)
        self._last_context = context

        return self.current_plan

    def apply_feedback(self, feedback_text: str) -> SprintPlan:
        """
        Process PM feedback and re-optimize the plan.

        Enters the state machine at ALLOCATE after negotiation parses
        the feedback into constraints, then runs review/rebalance.

        Parameters
        ----------
        feedback_text : str
            Natural language feedback from the PM.

        Returns
        -------
        SprintPlan : re-optimized plan
        """
        if not self.current_plan:
            raise ValueError("No current plan to modify. Run run_planning_cycle() first.")

        context = {
            "feedback_text": feedback_text,
            "current_plan": self.current_plan,
            "tasks": copy.deepcopy(self.current_plan.tasks),
            "constraints": copy.deepcopy(list(self.current_plan.constraints)),
            "enabled_developers": list(self.current_plan.developer_loads.keys()),
        }

        # Parse feedback into constraints
        context = self.negotiation.run(context)
        self._log_agent_output("NegotiationAgent", context)
        self._log_transition(
            PipelineState.REVIEW, PipelineState.ALLOCATE,
            f"PM feedback: '{feedback_text[:60]}'"
        )

        # Re-enter state machine at ALLOCATE → REVIEW → (maybe REBALANCE) → DONE
        retries = {"rebalance": 2}
        state = PipelineState.ALLOCATE
        iterations = 0

        while state != PipelineState.DONE and iterations < _MAX_ITERATIONS:
            iterations += 1
            prev_state = state

            if state == PipelineState.ALLOCATE:
                context = self.allocation.run(context)
                self._log_agent_output("AllocationAgent", context)
                state = PipelineState.REVIEW
                self._log_transition(prev_state, state, "Re-allocation complete")

            elif state == PipelineState.REVIEW:
                plan = context.get("plan")
                if not plan:
                    state = PipelineState.DONE
                    self._log_transition(prev_state, state, "No plan")
                    continue

                max_util = max(
                    (dl.utilization for dl in plan.developer_loads.values()),
                    default=0
                )
                total_tasks = len(plan.tasks)
                unassigned_pct = (
                    len(plan.unassigned_tasks) / total_tasks * 100
                    if total_tasks > 0 else 0
                )

                issues = []
                if max_util > _MAX_UTIL_PCT:
                    issues.append(f"max utilization {max_util:.0f}%")
                if unassigned_pct > _MAX_UNASSIGNED_PCT:
                    issues.append(f"unassigned {unassigned_pct:.0f}%")

                if not issues or retries["rebalance"] <= 0:
                    state = PipelineState.DONE
                    reason = "Plan acceptable" if not issues else (
                        f"Issues remain ({', '.join(issues)}) but retries exhausted"
                    )
                    self._log_transition(prev_state, state, reason)
                else:
                    retries["rebalance"] -= 1
                    state = PipelineState.AUTO_REBALANCE
                    self._log_transition(prev_state, state,
                                         f"Auto-rebalancing: {', '.join(issues)}")

            elif state == PipelineState.AUTO_REBALANCE:
                plan = context["plan"]
                for dev_name, dl in plan.developer_loads.items():
                    if dl.utilization > _MAX_UTIL_PCT:
                        cap_hours = round(dl.capacity_hours * 0.95, 1)
                        context["constraints"].append(Constraint(
                            type="max_hours",
                            developer=dev_name,
                            value=cap_hours,
                            reason=f"Auto-rebalance: was at {dl.utilization:.0f}%",
                        ))
                state = PipelineState.ALLOCATE
                self._log_transition(prev_state, state,
                                     "Auto-constraints added — re-allocating")

        self.current_plan = context.get("plan", SprintPlan())
        self.current_plan.version = len(self.plan_history) + 1
        self.current_plan.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.plan_history.append(self.current_plan)

        return self.current_plan


    def _log_transition(self, from_state, to_state, reason: str):
        """Record a state transition."""
        from_name = from_state.value if isinstance(from_state, PipelineState) else str(from_state)
        to_name = to_state.value if isinstance(to_state, PipelineState) else str(to_state)
        transition = StateTransition(
            from_state=from_name,
            to_state=to_name,
            reason=reason,
        )
        self.state_log.append(transition)
        print(f"[Orchestrator] {from_name} → {to_name}: {reason}")

    def _log_agent_output(self, agent_name: str, context: dict):
        """Print detailed agent output to the terminal."""
        separator = "-" * 60
        print(f"\n{separator}")
        print(f"  OUTPUT: {agent_name}")
        print(separator)

        tasks = context.get("tasks", [])

        if agent_name == "DecompositionAgent":
            for t in tasks:
                deps = f" (depends on: {t.dependencies})" if t.dependencies else ""
                print(f"  #{t.id:>2}  [{t.type:<8}]  SP:{t.sp:<3}  {t.priority:<6}  {t.summary}{deps}")
                if t.description:
                    desc_snipped = t.description[:80] + ("..." if len(t.description) > 80 else "")
                    print(f"        └─ {desc_snipped}")
            print(f"  Total: {len(tasks)} tasks, {sum(t.sp for t in tasks)} SP")

        elif agent_name == "EstimationAgent":
            for t in tasks:
                print(f"  #{t.id:>2}  {t.predicted_hours:>5.1f}h  "
                      f"(range: {t.range_low:.1f}-{t.range_high:.1f}h)  {t.summary[:50]}")
                if t.reasoning:
                    print(f"        → {t.reasoning[:80]}")
            total_h = sum(t.predicted_hours for t in tasks)
            print(f"  Total: {total_h:.1f}h estimated across {len(tasks)} tasks")

        elif agent_name == "AllocationAgent":
            plan = context.get("plan")
            if plan:
                for dev_name, dl in plan.developer_loads.items():
                    bar_len = int(dl.utilization / 5)
                    bar = "█" * bar_len + "░" * (20 - bar_len)
                    print(f"  {dev_name:<25}  {dl.total_hours:>5.1f}h / {dl.capacity_hours}h  "
                          f"{bar}  {dl.utilization:.0f}%")
                    for t in dl.assigned_tasks:
                        print(f"    • [{t.predicted_hours:>5.1f}h] {t.summary[:45]}")
                if plan.unassigned_tasks:
                    print(f"  ⚠ Unassigned: {len(plan.unassigned_tasks)} tasks")

        elif agent_name == "NegotiationAgent":
            constraints = context.get("constraints", [])
            print(f"  Active constraints: {len(constraints)}")
            for c in constraints:
                parts = [c.type]
                if c.developer:
                    parts.append(c.developer)
                if c.value is not None:
                    parts.append(str(c.value))
                if c.reason:
                    parts.append(f"({c.reason})")
                print(f"    • {' | '.join(parts)}")

        print(separator + "\n")
