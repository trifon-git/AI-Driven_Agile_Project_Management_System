"""
Effort Estimation Agent
========================
Uses the trained ML model to predict effort for each task,
enriching the backlog with hours estimates and confidence ranges.
"""

from antigrivity.agents.base import Agent, Task
from antigrivity.config import TEAM, SP_TO_HOURS


class EstimationAgent(Agent):
    """
    Effort Estimation Agent — predicts hours for each task.

    For each task:
    1. Runs the ML model to get point estimate + probabilistic range
    2. Identifies best-fit developers based on historical skill match
    3. Finds similar past tasks as supporting evidence
    """

    name = "EstimationAgent"
    description = "Predicts effort using ML model with probabilistic ranges"

    def __init__(self):
        self._estimator = None
        self._estimator_loaded = False   # True once we've tried loading (even if failed)

    @property
    def estimator(self):
        """Lazy-load the estimator. Raises if model files are missing."""
        if not self._estimator_loaded:
            self._estimator_loaded = True
            from antigrivity.models.estimator import EffortEstimator
            self._estimator = EffortEstimator()
            self.log("ML model loaded successfully")
        return self._estimator

    def run(self, context: dict) -> dict:
        tasks = context.get("tasks", [])
        self.log(f"Estimating effort for {len(tasks)} tasks...")

        est = self.estimator   # Raises if model missing
        model_available = est is not None

        for task in tasks:
            task_dict = {
                "summary": task.summary,
                "type": task.type,
                "project": task.project,
                "sp": task.sp,
            }

            enabled_devs = context.get("enabled_developers")
            
            # If the task is already assigned (Allocate ran first), predict specifically for that assignee
            if task.assigned_to:
                target_dev = task.assigned_to
                best = {"predicted_hours": est.predict(task_dict, target_dev)}
                low, mid, high = est.predict_range(task_dict, target_dev)
                best["range_low"] = low
                best["range_mid"] = mid
                best["range_high"] = high
            else:
                # Fallback to predicting for all devs and picking the absolute best
                all_predictions = est.predict_for_all_developers(task_dict, dev_list=enabled_devs)
                if all_predictions:
                    best = all_predictions[0]
                else:
                    best = None

            if best:
                task.predicted_hours = best["predicted_hours"]
                task.range_low = best["range_low"]
                task.range_mid = best["range_mid"]
                task.range_high = best["range_high"]

                similar = est.find_similar_tasks(task_dict, top_n=2)
                similar_text = ""
                if not similar.empty:
                    for _, row in similar.iterrows():
                        similar_text += (
                            f" | Similar: '{row['feature_summary_clean'][:40]}' "
                            f"took {row['target_actual_hours']:.1f}h"
                        )
                task.reasoning = (
                    f"ML estimate: {task.predicted_hours}h "
                    f"(range: {task.range_low}-{task.range_high}h)"
                    f"{similar_text}"
                )
                self.log(f"  #{task.id} ML: {task.predicted_hours:.1f}h ({task.range_low:.1f}-{task.range_high:.1f}h) — {task.summary[:50]}")

        self.log("Estimation complete (ML model)")
        context["tasks"] = tasks
        return context
