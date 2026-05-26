"""
Negotiation Agent (Text-to-Constraint)
=======================================
Uses Ollama (local LLM) to parse natural language feedback
from the Project Manager into structured planning constraints.

Replaces the brittle regex parser from v0.1 with LLM understanding.
"""

import json
import re
import ollama
from antigrivity.agents.base import Agent, Constraint
import antigrivity.config
from antigrivity.config import TEAM


NEGOTIATION_PROMPT = """You are a senior Project Management assistant for the Antigrivity sprint planner.
Your goal is to translate Project Manager feedback into a JSON array of structured constraints.

SYSTEM CAPABILITIES (Constraint Types):
1. "max_hours": Hard limit on a developer's capacity (use for leave/holidays).
2. "capacity_boost": Increase capacity for a specific person (overtime).
3. "assign_task": FORCE a specific task to a person.
4. "soft_assign": PREFER a person for a task (weighted preference).
5. "ban_developer": Prevent someone from doing a specific task.
6. "defer_task": Move a task out of the current sprint.
7. "change_priority": Update task priority (High, Medium, Low).
8. "change_sp": Manual override of task Story Points.
9. "force_hours": Manual override of task duration in hours.
10. "change_category": Change task department (Developer, Designer, Operations).

INTERPRETATION EXAMPLES:
- "Marios is on leave this Friday" -> [{{"type": "max_hours", "developer": "Marios Anagnostopoulos", "value": 28, "reason": "Leave"}}]
- "Marios can work 10 hours extra this week" -> [{{"type": "capacity_boost", "developer": "Marios Anagnostopoulos", "value": 45, "reason": "Overtime"}}]
- "Task 5 is actually 13 points, not 5" -> [{{"type": "change_sp", "task_id": 5, "value": 13, "reason": "PM estimate override"}}]
- "That landing page task should be 20 hours flat" -> [{{"type": "force_hours", "task_summary": "landing page", "value": 20, "reason": "Fixed duration"}}]
- "Move the SEO work to Maria, Nikos is too slow" -> [{{"type": "assign_task", "developer": "Maria Messari", "task_summary": "SEO"}}, {{"type": "ban_developer", "developer": "Nikos Haliotis", "task_summary": "SEO"}}]
- "Try to give the frontend tasks to Marios Kontis" -> [{{"type": "soft_assign", "developer": "Marios Kontis", "task_summary": "frontend"}}]

TEAM MEMBERS:
{team_members}

CURRENT PLAN STATE:
{plan_summary}

TASK BACKLOG:
{task_list}

PM FEEDBACK: "{feedback}"

INSTRUCTIONS:
- Match names to the EXACT full names in the team list.
- Use 'task_id' if numeric, otherwise match with 'task_summary'.
- Return ONLY the JSON array. No preamble.

JSON SCHEMA:
[
  {{
    "type": "type_from_list",
    "developer": "Full Name or null",
    "task_id": number or null,
    "task_summary": "text or null",
    "value": number/string or null,
    "reason": "explanation"
  }}
]
"""


class NegotiationAgent(Agent):
    """
    Negotiation Agent — translates PM feedback into planning constraints.
    """

    name = "NegotiationAgent"
    description = "Parses natural language PM feedback into structured planning constraints"

    def run(self, context: dict) -> dict:
        feedback_text = context.get("feedback_text", "")
        current_plan = context.get("current_plan")
        tasks = context.get("tasks", [])
        existing_constraints = context.get("constraints", [])

        self.log(f"Processing feedback with LLM: '{feedback_text[:80]}...'")

        team_members = "\n".join(f"- {name}" for name in TEAM.keys())

        plan_summary = ""
        if current_plan:
            for dev_name, dl in current_plan.developer_loads.items():
                plan_summary += f"- {dev_name}: {dl.total_hours:.1f}/{dl.capacity_hours}h ({dl.utilization:.0f}%)\n"

        task_list = ""
        for task in tasks:
            task_list += f"- ID:{task.id} [{task.type}] {task.summary} ({task.predicted_hours:.1f}h)\n"

        prompt = NEGOTIATION_PROMPT.format(
            team_members=team_members,
            plan_summary=plan_summary or "No current plan",
            task_list=task_list or "No tasks",
            feedback=feedback_text,
        )

        prompt_with_directive = prompt.strip()
        
        raw_text = ""
        for attempt in range(2):
            response = ollama.chat(
                model=antigrivity.config.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are a senior Project Management assistant. Output ONLY valid JSON arrays. Do NOT output markdown, formatting, or any extra whitespace."},
                    {"role": "user", "content": prompt_with_directive},
                    {"role": "assistant", "content": "["}
                ],
                options={
                    "seed": antigrivity.config.OLLAMA_SEED,
                    "temperature": 0.1,
                    "num_predict": 1024,
                    "num_ctx": antigrivity.config.OLLAMA_NUM_CTX,
                    "repeat_penalty": 1.2,
                    "top_p": 0.9,
                    "stop": ["```", "</s>", "<|im_end|>"]
                }
            )
            raw_text = "[" + response["message"]["content"].strip()
            done_reason = response.get("done_reason", "unknown")
            eval_count = response.get("eval_count", 0)
            
            self.log(f"LLM raw response ({len(raw_text)} chars, reason: {done_reason}, tokens: {eval_count})")
            if len(raw_text) > 0:
                self.log(f"Content preview: {raw_text[:500]}")
            
            if raw_text and ("[" in raw_text or "{" in raw_text):
                break
            self.log(f"Empty/non-JSON on attempt {attempt + 1}, retrying...")

        new_constraints = self._parse_constraints(raw_text)
        self.log(f"Parsed {len(new_constraints)} constraints")


        all_constraints = list(existing_constraints) + new_constraints
        context["constraints"] = all_constraints
        return context


    def _parse_constraints(self, raw_text):
        """Robustly parse LLM JSON into Constraint objects."""
        text = raw_text.strip()

        # For reasoning models: strip `<think>...</think>` blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        if "```" in text:
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        start = text.find("{") if "[" not in text or (text.find("{") < text.find("[") and text.find("{") != -1) else text.find("[")
        end = max(text.rfind("}"), text.rfind("]")) + 1
        
        if start == -1 or end == 0:
            return []

        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError as e:
            # Auto-truncate unfinished JSON arrays
            cleaned = text[start:end]
            if not cleaned.strip().endswith("]"):
                last_brace = cleaned.rfind("}")
                if last_brace != -1:
                    cleaned = cleaned[:last_brace+1] + "\n]"
            try:
                data = json.loads(cleaned)
            except Exception as e2:
                return []

        raw_list = data if isinstance(data, list) else [data]

        constraints = []
        for raw in raw_list:
            if not isinstance(raw, dict) or "type" not in raw:
                continue
                
            constraints.append(Constraint(
                type=raw["type"],
                developer=self._match_developer(raw.get("developer")),
                task_id=int(raw["task_id"]) if raw.get("task_id") and str(raw["task_id"]).isdigit() else None,
                task_summary=raw.get("task_summary"),
                value=raw.get("value"),
                reason=raw.get("reason", "Parsed feedback"),
            ))
        return constraints

    def _match_developer(self, name):
        """
        Fuzzy match a nickname or partial name to the full team name.
        Uses EMPLOYEE_MAP from config.py as a primary source of truth.
        """
        if not name:
            return None

        from antigrivity.config import EMPLOYEE_MAP
        name_clean = str(name).strip().capitalize()
        name_lower = name_clean.lower()

        # 1. Check the Explicit Alias Map (EMPLOYEE_MAP)
        if name_clean in EMPLOYEE_MAP:
            return EMPLOYEE_MAP[name_clean]
        
        # 2. Check for values in the Alias Map
        for alias, full_name in EMPLOYEE_MAP.items():
            if name_lower in alias.lower() or alias.lower() in name_lower:
                return full_name

        # 3. Exact Team Match
        for dev in TEAM:
            if dev.lower() == name_lower:
                return dev

        # 4. Partial Team Match (e.g., 'Marios' -> 'Marios Anagnostopoulos')
        for dev in TEAM:
            if name_lower in dev.lower() or dev.lower() in name_lower:
                return dev

        # 5. First-name or Surname matching
        for dev in TEAM:
            parts = dev.lower().split()
            if any(p == name_lower for p in parts):
                return dev

        return name  # Return original if no match found

    def _fallback_parse(self, feedback_text, tasks):
        """
        Simple rule-based fallback parser (similar to v0.1 but improved).
        Used when Ollama is not available.
        """
        import re
        constraints = []
        text = feedback_text.lower()

        # Pattern: "limit X to Y hours"
        limit_match = re.search(r'limit\s+(\w+)\s+to\s+(\d+)\s*h', text)
        if limit_match:
            dev = self._match_developer(limit_match.group(1).capitalize())
            hours = int(limit_match.group(2))
            if dev:
                constraints.append(Constraint(
                    type="max_hours", developer=dev, value=hours,
                    reason="User requested hour limit"
                ))

        # Pattern: "assign task X to Y"
        assign_match = re.search(r'assign\s+task\s+(\d+)\s+to\s+(\w+)', text)
        if assign_match:
            task_id = int(assign_match.group(1))
            dev = self._match_developer(assign_match.group(2).capitalize())
            if dev:
                constraints.append(Constraint(
                    type="assign_task", developer=dev, task_id=task_id,
                    reason="User forced assignment"
                ))

        # Pattern: "X cannot do task Y" or "remove X from task Y"
        ban_match = re.search(r'(\w+)\s+cannot\s+do\s+task\s+(\d+)', text)
        if ban_match:
            dev = self._match_developer(ban_match.group(1).capitalize())
            task_id = int(ban_match.group(2))
            if dev:
                constraints.append(Constraint(
                    type="ban_developer", developer=dev, task_id=task_id,
                    reason="User banned developer from task"
                ))

        # Pattern: "defer X" or "move X to next sprint"
        defer_match = re.search(r'(?:defer|move|postpone)\s+(?:task\s+)?(\d+|.+?)(?:\s+to\s+next)', text)
        if defer_match:
            matched = defer_match.group(1).strip()
            try:
                task_id = int(matched)
                constraints.append(Constraint(
                    type="defer_task", task_id=task_id,
                    reason="User deferred task"
                ))
            except ValueError:
                constraints.append(Constraint(
                    type="defer_task", task_summary=matched,
                    reason="User deferred task"
                ))

        return constraints
