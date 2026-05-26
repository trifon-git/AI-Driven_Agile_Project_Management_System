"""
Task Decomposition Agent
=========================
Uses Ollama (local LLM) to break a project description into atomic tasks.
Retrieves similar historical tasks via RAG for grounding.
"""

import json
import re
import ollama
from antigrivity.agents.base import Agent, Task
import antigrivity.config
from antigrivity.config import TEAM
from antigrivity.utils import load_training_data, clean_text


DECOMPOSITION_PROMPT = """You are an experienced Agile Scrum Master for a web development agency.

Your job is to exhaustively break down a project description into a highly detailed, comprehensive list of atomic, actionable tasks suitable for a sprint backlog. 
You must generate AS MANY DETAILED TASKS AS POSSIBLE to cover every single development, design, testing, and operational step.

AVAILABLE TEAM:
{team_info}

RULES:
1. Break down the work into highly granular tasks. Each task must be small enough for ONE team member to complete in 1-8 hours.
2. Do NOT group large features into single tasks. Split them into backend, frontend, database, design, API, testing, and deployment components.
3. Create tasks that match the team's skills (design tasks for designers, SEO tasks for SEO specialists, dev tasks for developers, etc.).
4. Issue types:
   - "Story": A user-facing feature or capability described in the project.
   - "Task": Non-feature work (setup, deployment, documentation, research).
   - "Bug": ONLY use if the project description explicitly mentions a fix or an existing broken feature. Do NOT invent bugs or fixes if they are not strictly mentioned.
   - "Sub-task": Use extensively to break down Stories and Tasks into atomic steps.
5. Include story point estimates (1, 2, 3, 5) using the Fibonacci scale, favoring smaller points due to high granularity.
6. Identify dependencies between tasks (which tasks must come first).
7. Assign priority: "High", "Medium", or "Low".
8. Be extremely specific and technical — define exact database tables, UI components, API routes, or tools required.
9. Consider seniority: complex architectural tasks should be sized for senior developers; simpler tasks for juniors.
10. Leave no stone unturned. For example, if a login flow is requested, create separate tasks for UI design, frontend form, state management, API route, database schema, email sending (for forgot password), and unit testing.
11. Do NOT invent features or requirements that are not mentioned in the project description. Only decompose exactly what is described.

SIMILAR PAST TASKS (for reference):
{similar_tasks}

PROJECT DESCRIPTION:
{project_description}

PROJECT KEY: {project_key}

Respond with ONLY a JSON array. Each element must have exactly these fields:
{{
  "summary": "Brief task title",
  "description": "2-3 sentences explaining exactly what needs to be done, technical requirements, and acceptance criteria.",
  "type": "Story|Task|Sub-task",
  "category": "Developer|Designer|Manager|Operations",
  "sp": <story points as number>,
  "priority": "High|Medium|Low",
  "dependencies": [<list of task indices this depends on, 0-indexed>]
}}

Respond with ONLY the JSON array, no other text.

"""


class DecompositionAgent(Agent):
    """
    Task Decomposition Agent — breaks project descriptions into atomic tasks.

    Uses RAG: retrieves similar past tasks from training data to ground
    the LLM's output in real project history.
    """

    name = "DecompositionAgent"
    description = "Breaks project descriptions into atomic sprint tasks using LLM + RAG"

    def run(self, context: dict) -> dict:
        project_description = context.get("project_description", "")
        project_key = context.get("project_key", "PROJ")

        self.log(f"Decomposing project: {project_description[:80]}...")

        # RAG: Find similar historical tasks
        similar_tasks_text = self._retrieve_similar_tasks(project_description)

        # Build team info string
        team_lines = []
        for name, info in TEAM.items():
            skills = ", ".join(info.get("skills", []))
            dept = info.get("department", "Developer")
            team_lines.append(f"- {name} (Role: {info['role']}, Dept: {dept}, {info.get('seniority', 'mid')}): {skills}")
        team_info = "\n".join(team_lines)

        # LLM: Generate task breakdown
        prompt = DECOMPOSITION_PROMPT.format(
            team_info=team_info,
            similar_tasks=similar_tasks_text,
            project_description=project_description,
            project_key=project_key,
        )

        prompt_with_directive = prompt.strip()

        # Try disabling thinking mode if the model supports it

        # Retry up to 3 times if the LLM returns an empty response
        raw_text = ""
        for attempt in range(3):
            temp = 0.3 + (attempt * 0.2)  # Increase temperature on retries
            self.log(f"LLM attempt {attempt + 1}/3 (temp={temp})...")
            
            response = ollama.chat(
                model=antigrivity.config.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are a strictly technical Agile Scrum Master. Output ONLY valid JSON arrays. Do NOT output markdown, formatting, or any extra whitespace. Do NOT output reasoning or <think> tags."},
                    {"role": "user", "content": prompt_with_directive},
                    {"role": "assistant", "content": "["}
                ],
                think=False,
                options={
                    "seed": antigrivity.config.OLLAMA_SEED,
                    "temperature": temp,
                    "num_predict": antigrivity.config.OLLAMA_NUM_PREDICT,
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
                self.log(f"Content preview: {raw_text[:800]}")

            # If model returned only an opening bracket or extremely short output, ask it to continue
            short_output = raw_text.strip() in ("[",) or len(raw_text.strip()) <= 2
            if short_output:
                self.log("Received very short output from model; attempting a continuation request...")
                cont_resp = ollama.chat(
                    model=antigrivity.config.OLLAMA_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a strictly technical Agile Scrum Master. Continue the previously-started JSON array and output ONLY the remaining JSON array items and the closing bracket. No extra text."},
                        {"role": "assistant", "content": raw_text},
                        {"role": "user", "content": "Continue and finish the JSON array. Output only valid JSON array."}
                    ],
                    think=False,
                    options={"seed": antigrivity.config.OLLAMA_SEED, "temperature": 0.0, "num_predict": antigrivity.config.OLLAMA_NUM_PREDICT}
                )
                cont_text = cont_resp.get("message", {}).get("content", "").strip()
                if cont_text:
                    # append continuation to original
                    raw_text = (raw_text + cont_text).strip()
                    done_reason = cont_resp.get("done_reason", done_reason)
                    eval_count += cont_resp.get("eval_count", 0)
                    self.log(f"Continuation returned ({len(cont_text)} chars), new length {len(raw_text)}")

            if raw_text and ("[" in raw_text or "{" in raw_text):
                break  # Got a valid-looking response
            else:
                self.log(f"Empty or non-JSON response on attempt {attempt + 1}, retrying...")

        if not raw_text:
            raise ValueError(
                f"LLM returned empty response after 3 attempts. "
                f"Check that model '{antigrivity.config.OLLAMA_MODEL}' is working correctly. "
                f"Try running: ollama run {antigrivity.config.OLLAMA_MODEL} \"Say hello\""
            )

        tasks = self._parse_tasks(raw_text, project_key)

        self.log(f"Generated {len(tasks)} tasks")
        context["tasks"] = tasks
        return context


    def _retrieve_similar_tasks(self, description, top_n=8):
        """Retrieve similar historical tasks using TF-IDF similarity (RAG)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        df = load_training_data()
        if df.empty:
            return "No historical data available."

        query = clean_text(description)
        corpus = df['feature_summary_clean'].fillna('').tolist()

        vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        tfidf_matrix = vectorizer.fit_transform(corpus + [query])

        query_vec = tfidf_matrix[-1]
        corpus_matrix = tfidf_matrix[:-1]

        similarities = cosine_similarity(query_vec, corpus_matrix).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]

        lines = []
        for idx in top_indices:
            row = df.iloc[idx]
            sim = similarities[idx]
            if sim > 0.05:  # Only include reasonably similar tasks
                lines.append(
                    f"- [{row.get('feature_issue_type', 'Task')}] "
                    f"{row.get('feature_summary_clean', 'N/A')} "
                    f"(SP: {row.get('feature_story_points', '?')}, "
                    f"Actual: {row.get('target_actual_hours', '?'):.1f}h, "
                    f"Project: {row.get('feature_project_key', '?')})"
                )

        return "\n".join(lines) if lines else "No closely matching historical tasks found."

    def _parse_tasks(self, raw_text, project_key):
        """Parse LLM JSON response into Task objects with robust recovery."""
        text = raw_text.strip()

        # For reasoning models: strip `<think>...</think>` blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Strip markdown code fences
        if "```" in text:
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        self.log(f"Cleaned text for parsing ({len(text)} chars): {text[:200]}")

        # Try to find a JSON array first, then fall back to a single object
        start = text.find("[")
        end = text.rfind("]") + 1

        if start == -1 or end == 0:
            # Maybe LLM returned a single object instead of an array
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError(f"No JSON found in LLM response: {text[:300]}")

        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError as e:
            self.log(f"JSON decode error: {e}")
            # Try to fix common issues: trailing commas, etc.
            cleaned = text[start:end]
            cleaned = cleaned.replace(",]", "]").replace(",}", "}")
            
            # --- Auto-fix truncated outputs (e.g. from reason: length) ---
            # If the model hit the max token length, the JSON will be cut off mid-word.
            if not cleaned.strip().endswith("]"):
                # Find the last fully completed task object
                last_brace = cleaned.rfind("}")
                if last_brace != -1:
                    cleaned = cleaned[:last_brace+1] + "\n]"
                    self.log("Attempted to auto-close a truncated JSON array.")

            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError as e2:
                # Last resort fallback: manually parse individual objects if the array is broken
                self.log(f"Second decode attempt failed: {e2}. Trying object-by-object fallback.")
                data = []
                # Fallback: manually parse individual objects if the array is broken
                
                # Match all { ... } blocks that contain the word "summary"
                blocks = re.findall(r'{[^{}]*"summary"[^{}]*}', cleaned)
                for block in blocks:
                    try:
                        data.append(json.loads(block))
                    except:
                        pass
                
                if not data:
                    raise ValueError(f"Could not parse valid JSON out of truncated response: {e}")
                
        raw_tasks = data if isinstance(data, list) else [data]

        tasks = []
        for i, raw in enumerate(raw_tasks):
            if not isinstance(raw, dict):
                self.log(f"Skipping non-dict task entry: {raw}")
                continue

            task = Task(
                id=i + 1,
                summary=raw.get("summary", f"Task {i+1}"),
                description=raw.get("description", ""),
                type=raw.get("type", "Task"),
                category=raw.get("category", "Developer"),
                project=project_key,
                sp=float(raw.get("sp", 3)),
                priority=raw.get("priority", "Medium"),
                dependencies=[d + 1 for d in raw.get("dependencies", []) if isinstance(d, int)],
            )
            tasks.append(task)

        if not tasks:
            raise ValueError(f"LLM returned JSON but no valid tasks were parsed. Raw: {text[:300]}")

        return tasks

