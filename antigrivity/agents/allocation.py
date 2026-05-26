"""
Resource Allocation Agent
==========================
Constraint-aware task assignment using penalty-based optimization.

Minimizes a weighted penalty function across:
- Skill mismatch (developer hasn't done similar work)
- Overload (exceeding capacity)
- Underload (uneven distribution)
- Constraint violations (hard/soft from PM feedback)
"""

import re
from collections import defaultdict
from antigrivity.agents.base import Agent, Task, DeveloperLoad, Constraint, SprintPlan
from antigrivity.config import TEAM
from antigrivity.utils import load_training_data, load_model, predict_effort, predict_effort_range
from antigrivity.models.embedding_matcher import build_developer_profiles, find_best_developer_by_embedding


# Penalty weights  (lower = preferred)
W_SKILL_MISMATCH = 20.0      # Max penalty for out-of-role assignment
W_EMBEDDING_MISMATCH = 5.0   # Max penalty for no semantic similarity
W_OVERLOAD = 10.0
W_UNDERLOAD = 0.5
W_PRIORITY = 3.0


class AllocationAgent(Agent):
    """
    Resource Allocation Agent — assigns tasks to developers.

    Uses a penalty-based greedy assignment with iterative swap improvement:
    1. Sort tasks by priority (High first) to assign critical work first
    2. For each task, score all eligible developers
    3. Assign to the developer with the lowest penalty score
    4. Run swap-based improvement passes to balance workload
    """

    name = "AllocationAgent"
    description = "Constraint-aware task-to-developer assignment with penalty optimization"

    def run(self, context: dict) -> dict:
        tasks = context.get("tasks", [])
        constraints = context.get("constraints", [])

        self.log(f"Allocating {len(tasks)} tasks with {len(constraints)} constraints...")

        # Reset all task assignments (critical for re-allocation after feedback)
        for task in tasks:
            task.assigned_to = None

        # 1. PRE-PROCESS TASK OVERRIDES (change_sp, force_hours, change_category)
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
                elif c.type == "change_category" and c.value is not None:
                    t.category = str(c.value)
                    t.reasoning += f" [PM switched category to {t.category}]"

        # 2. Build constraint lookup
        constraint_state = self._build_constraint_state(constraints, tasks)
        self.log(f"Constraint state: max_hours={constraint_state['max_hours']}, boosts={constraint_state['boosts']}, forced={constraint_state['forced_assignments']}, bans={dict(constraint_state['bans'])}, deferred_ids={constraint_state['defer_tasks']}")

        # 3. Build developer loads
        enabled_devs = context.get("enabled_developers")
        dev_loads = {}
        for dev_name, info in TEAM.items():
            if enabled_devs is not None and dev_name not in enabled_devs:
                continue
            
            # Use max_hours if set, otherwise check for capacity_boost, otherwise default
            cap = info["capacity_hours"]
            if dev_name in constraint_state["max_hours"]:
                cap = constraint_state["max_hours"][dev_name]
                self.log(f"  {dev_name}: capped to {cap}h (max_hours)")
            elif dev_name in constraint_state["boosts"]:
                cap = constraint_state["boosts"][dev_name]
                self.log(f"  {dev_name}: boosted to {cap}h (overtime)")
                
            dev_loads[dev_name] = DeveloperLoad(
                name=dev_name,
                capacity_hours=cap,
            )

        # Load training data for skill matching
        training_df = load_training_data()
        model = load_model(use_tuned=True)

        # Build semantic embedding profiles
        embedding_profiles = build_developer_profiles(training_df)

        # Remove deferred tasks
        active_tasks = []
        deferred = []
        for task in tasks:
            if task.id in constraint_state["defer_tasks"] or \
               task.summary in constraint_state["defer_summaries"]:
                deferred.append(task)
                self.log(f"  Deferred: #{task.id} {task.summary[:40]}")
            else:
                active_tasks.append(task)

        # Sort by priority
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        active_tasks.sort(key=lambda t: (priority_order.get(t.priority, 1), -t.sp))

        # Greedy assignment
        unassigned = []
        for task in active_tasks:
            # Check for forced assignment
            forced_dev = constraint_state["forced_assignments"].get(task.id)
            if forced_dev and forced_dev in dev_loads:
                self._assign_task(task, forced_dev, dev_loads, model)
                self.log(f"  #{task.id} FORCED -> {forced_dev} — {task.summary[:40]}")
                continue

            # Score all eligible developers
            scores = []
            banned_devs = constraint_state["bans"].get(task.id, set())
            candidate_devs = [d for d in dev_loads if d not in banned_devs]

            # Get semantic similarity ranking
            embedding_ranks = {}
            if embedding_profiles and candidate_devs:
                ranked = find_best_developer_by_embedding(
                    task.summary, candidate_devs, embedding_profiles, top_k=5
                )
                embedding_ranks = {r["developer"]: r for r in ranked}

            for dev_name, load in dev_loads.items():
                if dev_name in banned_devs:
                    continue

                score = self._score_assignment(
                    task, dev_name, load, embedding_ranks, constraint_state
                )
                scores.append((dev_name, score))

            if not scores:
                unassigned.append(task)
                self.log(f"  #{task.id} UNASSIGNED (no eligible devs) — {task.summary[:40]}")
                continue

            scores.sort(key=lambda x: x[1])
            best_dev = scores[0][0]
            best_score = scores[0][1]
            self._assign_task(task, best_dev, dev_loads, model)
            self.log(f"  #{task.id} -> {best_dev} (score={best_score:.1f}) — {task.summary[:40]}")

        # Swap improvement
        self._improve_by_swaps(dev_loads, model, constraint_state, embedding_profiles)

        # Build plan
        plan = SprintPlan(
            tasks=active_tasks + unassigned + deferred,
            developer_loads=dev_loads,
            unassigned_tasks=unassigned + deferred,
            constraints=constraints,
        )

        # Log final utilization summary
        self.log(f"Assignment complete: {len(active_tasks) - len(unassigned)} assigned, {len(unassigned)} unassigned, {len(deferred)} deferred")
        for dn, dl in dev_loads.items():
            self.log(f"  {dn}: {dl.total_hours:.1f}/{dl.capacity_hours}h ({dl.utilization:.0f}%) — {len(dl.assigned_tasks)} tasks")

        context["plan"] = plan
        return context

    def _build_constraint_state(self, constraints, tasks=None):
        """Parse constraints into lookup structures.
        
        Resolves task_summary to actual task IDs via substring matching
        so that constraints like 'assign Frontend to X' work correctly.
        """
        state = {
            "max_hours": {},
            "boosts": {},
            "forced_assignments": {},
            "soft_assignments": defaultdict(set),  # task_id -> set of preferred devs
            "bans": defaultdict(set),
            "defer_tasks": set(),
            "defer_summaries": set(),
        }

        tasks = tasks or []

        for c in constraints:
            # Resolve task IDs: use explicit task_id, or match task_summary against tasks
            resolved_task_ids = []
            if c.task_id:
                resolved_task_ids = [c.task_id]
            elif c.task_summary and tasks:
                needle = c.task_summary.lower()
                resolved_task_ids = [t.id for t in tasks if needle in t.summary.lower()]
                if resolved_task_ids:
                    self.log(f"  Resolved '{c.task_summary}' -> task IDs {resolved_task_ids}")
                else:
                    self.log(f"  ⚠️ Could not match '{c.task_summary}' to any task")

            if c.type == "max_hours" and c.developer:
                state["max_hours"][c.developer] = c.value
            elif c.type == "capacity_boost" and c.developer:
                state["boosts"][c.developer] = c.value
            elif c.type == "assign_task" and c.developer and resolved_task_ids:
                for tid in resolved_task_ids:
                    state["forced_assignments"][tid] = c.developer
            elif c.type == "soft_assign" and c.developer and resolved_task_ids:
                for tid in resolved_task_ids:
                    state["soft_assignments"][tid].add(c.developer)
            elif c.type == "ban_developer" and c.developer and resolved_task_ids:
                for tid in resolved_task_ids:
                    state["bans"][tid].add(c.developer)
            elif c.type == "defer_task":
                for tid in resolved_task_ids:
                    state["defer_tasks"].add(tid)
                if c.task_summary:
                    state["defer_summaries"].add(c.task_summary)

        return state

    def _assign_task(self, task, dev_name, dev_loads, model):
        """Assign a task to a developer, updating predicted hours if model available."""
        # Only re-predict if NOT forced by PM
        is_forced = "PM forced hours" in task.reasoning
        
        if model and not is_forced:
            task_dict = {"summary": task.summary, "type": task.type,
                         "project": task.project, "sp": task.sp}
            task.predicted_hours = predict_effort(model, task_dict, dev_name)
            task.range_low, task.range_mid, task.range_high = predict_effort_range(model, task_dict, dev_name)
            
            # Update reasoning
            task.reasoning = re.sub(r"ML estimate: [\d.]+h", f"ML estimate: {task.predicted_hours}h", task.reasoning)
            task.reasoning = re.sub(r"range: [\d.]+-[\d.]+h", f"range: {task.range_low}-{task.range_high}h", task.reasoning)

        task.assigned_to = dev_name
        dev_loads[dev_name].assigned_tasks.append(task)
        dev_loads[dev_name].total_hours += task.predicted_hours

    def _score_assignment(self, task, dev_name, load, embedding_ranks, constraint_state=None):
        """
        Score penalty for assigning task to developer. Lower = better.
        """
        dev_info = TEAM.get(dev_name, {})

        # 1. STRICT DEPARTMENT FILTER
        task_category = getattr(task, "category", "Developer")
        dev_dept = dev_info.get("department", "Developer")

        if task_category == "Designer" and dev_dept != "Designer":
            return 1000000.0
        if task_category == "Developer" and dev_dept != "Developer":
            return 1000000.0
        if task_category == "Manager" and dev_dept != "Management":
            return 1000000.0
        if task_category == "Operations" and dev_dept != "Operations":
            return 1000000.0

        score = 0.0

        # 2. Soft Assignment Bonus
        if constraint_state and dev_name in constraint_state["soft_assignments"].get(task.id, set()):
            score -= 50.0  # Large bonus to prefer this person

        # 1. Overload penalty (exponential)
        task_hours = task.predicted_hours
        projected_load = load.total_hours + task_hours
        if projected_load > load.capacity_hours:
            overflow = projected_load - load.capacity_hours
            score += W_OVERLOAD * (overflow ** 1.5)

        # 2. EXPLICIT ROLE/SKILL PENALTY (independent of embeddings)
        role = dev_info.get("role", "").lower()
        task_text = (task.summary + " " + task.description + " " + task.type).lower()
        
        # Strip punctuation for cleaner word matching
        import re
        task_words = set(re.findall(r'\b\w+\b', task_text))

        design_kw = {"design", "ui", "ux", "mockup", "layout", "branding", "figma", "responsive", "visual"}
        seo_kw = {"seo", "analytics", "content", "meta", "keyword", "copywriting", "social", "marketing", "ads"}
        dev_kw = {"api", "backend", "frontend", "database", "code", "integration", "deploy", "schema", "bug", "test", "testing", "fix", "verify", "qa"}
        mgmt_kw = {"strategy", "planning", "account", "client", "stakeholder", "business"}

        design_hits = len(task_words & design_kw)
        seo_hits = len(task_words & seo_kw)
        dev_hits = len(task_words & dev_kw)
        mgmt_hits = len(task_words & mgmt_kw)
        
        hits = {
            "design": design_hits,
            "seo": seo_hits,
            "dev": dev_hits,
            "mgmt": mgmt_hits
        }
        
        max_hits = max(hits.values())
        if max_hits == 0:
            primary_domains = {"dev"}  # Default un-categorized tasks to dev
        else:
            primary_domains = {k for k, v in hits.items() if v == max_hits}

        dev_domains = set()
        if "designer" in role: dev_domains.add("design")
        if "seo" in role or "operations" in role: dev_domains.add("seo")
        if "developer" in role: dev_domains.add("dev")
        if "ceo" in role or "account" in role: dev_domains.add("mgmt")

        domain_penalty = 0.0
        # If the developer doesn't have ANY roles matching the task's primary domain(s), penalize
        if not (primary_domains & dev_domains):
            domain_penalty += W_SKILL_MISMATCH

        score += domain_penalty

        # 3. EMBEDDING SIMILARITY (granular assignment signal)
        embed_info = embedding_ranks.get(dev_name)
        if embed_info and embed_info["score"] > 0:
            similarity = embed_info["score"]
            score += W_EMBEDDING_MISMATCH * (1.0 - similarity)
        else:
            # No embedding history — fall back to a minor penalty if domain matched
            score += W_EMBEDDING_MISMATCH * (0.3 if domain_penalty == 0 else 1.0)

        # 3. Seniority-complexity match
        seniority = dev_info.get("seniority", "mid")
        if task.sp >= 8 and seniority == "junior":
            score += W_PRIORITY * 1.5  # Complex tasks → penalize juniors
        elif task.sp <= 2 and seniority == "senior":
            score += W_UNDERLOAD * 0.3  # Trivial tasks → slight penalty for seniors

        # 4. Underload bonus (balance workload)
        utilization = load.total_hours / load.capacity_hours if load.capacity_hours > 0 else 0
        score += W_UNDERLOAD * utilization

        # 5. Extra penalty for High priority tasks with poor fit
        if task.priority == "High" and (not embed_info or embed_info["score"] < 0.3):
            score += W_PRIORITY

        return score

    def _improve_by_swaps(self, dev_loads, model, constraint_state, embedding_profiles=None, passes=2):
        """
        Iterative improvement: swap tasks between overloaded and underloaded developers.
        """
        for _ in range(passes):
            # Find overloaded and underloaded devs
            overloaded = [(name, dl) for name, dl in dev_loads.items()
                          if dl.utilization > 95]
            underloaded = [(name, dl) for name, dl in dev_loads.items()
                           if dl.utilization < 60]

            if not overloaded or not underloaded:
                break

            for over_name, over_dl in overloaded:
                for under_name, under_dl in underloaded:
                    # Try swapping the smallest task from overloaded to underloaded
                    if not over_dl.assigned_tasks:
                        continue

                    # Find a swappable task (not forced, not banned)
                    for task in sorted(over_dl.assigned_tasks, key=lambda t: t.predicted_hours):
                        if task.id in constraint_state["forced_assignments"]:
                            continue
                        if under_name in constraint_state["bans"].get(task.id, set()):
                            continue

                        # NEW: Respect department during swaps
                        under_info = TEAM.get(under_name, {})
                        if getattr(task, "category", "Developer") == "Designer" and under_info.get("department") != "Designer":
                            continue
                        if getattr(task, "category", "Developer") == "Developer" and under_info.get("department") != "Developer":
                            continue
                        if getattr(task, "category", "Developer") == "Operations" and under_info.get("department") != "Operations":
                            continue

                        # Check if underloaded dev can absorb this task
                        if under_dl.total_hours + task.predicted_hours <= under_dl.capacity_hours:
                            # Execute swap
                            over_dl.assigned_tasks.remove(task)
                            over_dl.total_hours -= task.predicted_hours

                            # Re-estimate for new developer
                            if model:
                                task_dict = {"summary": task.summary, "type": task.type,
                                             "project": task.project, "sp": task.sp}
                                task.predicted_hours = predict_effort(model, task_dict, under_name)

                            task.assigned_to = under_name
                            under_dl.assigned_tasks.append(task)
                            under_dl.total_hours += task.predicted_hours
                            break
