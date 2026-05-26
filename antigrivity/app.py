"""
AI-Driven Agile Project Management System
=============================================
Interactive web interface for the multi-agent sprint planning system.

Run with: streamlit run antigrivity/app.py
"""

import streamlit as st
import pandas as pd
import time
import sys
import os
import warnings
import logging
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antigrivity.agents.base import Orchestrator, Constraint, SprintPlan
from antigrivity.config import TEAM


st.set_page_config(
    page_title="AI-Driven Agile Project Management System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        opacity: 0.8;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Developer cards */
    .dev-card {
        background: linear-gradient(145deg, #1e1e2e, #252540);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        color: #e0e0e0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .dev-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .dev-name {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }

    /* Load bar */
    .load-bar-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
        margin: 0.4rem 0;
    }
    .load-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease;
    }
    .load-ok { background: linear-gradient(90deg, #00b894, #55efc4); }
    .load-warn { background: linear-gradient(90deg, #fdcb6e, #e17055); }
    .load-critical { background: linear-gradient(90deg, #e17055, #d63031); }

    /* Task pill */
    .task-pill {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .task-hours {
        font-weight: 600;
        color: #74b9ff;
        white-space: nowrap;
        margin-left: 1rem;
    }

    /* Stats cards */
    .stat-card {
        background: linear-gradient(145deg, #1e1e2e, #2d2d44);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #74b9ff;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Constraint badge */ 
    .constraint-badge {
        background: rgba(116, 185, 255, 0.15);
        border: 1px solid rgba(116, 185, 255, 0.3);
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
        color: #74b9ff;
    }

    /* Feedback area */
    .feedback-area {
        background: linear-gradient(145deg, #1a1a2e, #1e293b);
        border: 1px solid rgba(116, 185, 255, 0.2);
        border-radius: 14px;
        padding: 1.5rem;
    }

    /* Priority indicators */
    .priority-high { color: #e17055; font-weight: 600; }
    .priority-medium { color: #fdcb6e; font-weight: 500; }
    .priority-low { color: #00b894; font-weight: 400; }

    /* Hide Streamlit's default styling elements */
    .stDeployButton { display: none; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def init_session():
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
    if "plan" not in st.session_state:
        st.session_state.plan = None
    if "plan_history" not in st.session_state:
        st.session_state.plan_history = []
    if "feedback_log" not in st.session_state:
        st.session_state.feedback_log = []
    if "tasks_generated" not in st.session_state:
        st.session_state.tasks_generated = False

init_session()


st.markdown("""
<div class="main-header">
    <h1>🚀 AI-Driven Agile Project Management System</h1>
    <p>Multi-Agent System for Human-Centric Agile Planning</p>
</div>
""", unsafe_allow_html=True)


# SIDEBAR — PROJECT INPUT
with st.sidebar:
    st.markdown("### 📋 Project Input")

    project_desc = st.text_area(
        "Project Description",
        placeholder="Describe the project or sprint requirements...\n\nExample: Build an e-commerce checkout with Stripe integration, order confirmation emails, and a mobile-responsive design.",
        height=180,
        key="project_desc",
    )

    project_key = st.text_input(
        "Project Key",
        value="PROJ",
        help="Jira project key (e.g., ES, COM, BI)",
    )

    st.markdown("---")
    st.markdown("### 🏢 Team Management")

    # Show team with toggle and editable capacity
    enabled_developers = []
    team_capacities = {}
    for dev_name, info in TEAM.items():
        # Default 5 employees enabled, historical ones disabled
        is_historical = dev_name in ["Andreas Pelekoudas", "Stavros Messaris"]
        default_val = not is_historical
        
        cols = st.columns([0.2, 0.8])
        is_enabled = cols[0].checkbox("Enable", value=default_val, key=f"enable_{dev_name}", label_visibility="collapsed")
        
        if is_enabled:
            enabled_developers.append(dev_name)
            first_name = dev_name.split()[0]
            role = info.get("role", "Developer")
            cap = cols[1].slider(
                f"{first_name} ({role})",
                min_value=0, max_value=60,
                value=info["capacity_hours"],
                key=f"cap_{dev_name}",
            )
            team_capacities[dev_name] = cap
        else:
            cols[1].markdown(f"~~{dev_name}~~ *(Inactive)*")

    st.markdown("---")

    generate_btn = st.button(
        "🤖 Generate Sprint Plan",
        type="primary",
        disabled=not project_desc.strip() or not enabled_developers,
        use_container_width=True
    )


# MAIN CONTENT — PLAN GENERATION

def generate_plan(enabled_devs):
    """Run the full agent pipeline."""
    # Update team capacities from sidebar
    for dev_name, cap in team_capacities.items():
        if dev_name in TEAM:
            TEAM[dev_name]["capacity_hours"] = cap

    with st.spinner("🧠 Initializing agents..."):
        orchestrator = Orchestrator()
        st.session_state.orchestrator = orchestrator

    with st.spinner("📝 Decomposing project into tasks..."):
        plan = orchestrator.run_planning_cycle(
            project_description=project_desc,
            project_key=project_key,
            enabled_developers=enabled_devs,
        )

    st.session_state.plan = plan
    st.session_state.plan_history = [plan]
    st.session_state.tasks_generated = True
    st.session_state.feedback_log = []


if generate_btn:
    generate_plan(enabled_developers)



def get_load_class(utilization):
    if utilization > 90:
        return "load-critical"
    elif utilization > 70:
        return "load-warn"
    return "load-ok"


def get_priority_class(priority):
    return f"priority-{priority.lower()}"


def render_plan(plan: SprintPlan):
    """Render the sprint plan visualization."""

    stats = plan.summary_stats()

    cols = st.columns(5)
    stat_data = [
        (str(stats["total_tasks"]), "Total Tasks"),
        (str(stats["assigned_tasks"]), "Assigned"),
        (str(stats["unassigned_tasks"]), "Unassigned"),
        (f"{stats['total_hours']}h", "Total Hours"),
        (f"{stats['team_utilization']}%", "Utilization"),
    ]

    for col, (value, label) in zip(cols, stat_data):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    dev_cols = st.columns(len(plan.developer_loads))

    for col, (dev_name, load) in zip(dev_cols, plan.developer_loads.items()):
        with col:
            util = load.utilization
            load_class = get_load_class(util)
            bar_width = min(util, 100)

            status_emoji = "✅" if util <= 90 else "🔥" if util > 95 else "⚠️"

            tasks_html = ""
            sorted_assigned = sorted(load.assigned_tasks, key=lambda t: int(re.search(r'\d+', str(t.id)).group()) if re.search(r'\d+', str(t.id)) else 0)
            for task in sorted_assigned:
                priority_cls = get_priority_class(task.priority)
                tasks_html += f"""
                <div class="task-pill">
                    <span><span class="{priority_cls}">●</span> {task.summary[:35]}{'...' if len(task.summary) > 35 else ''}</span>
                    <span class="task-hours">{task.predicted_hours:.1f}h</span>
                </div>
                """

            first_name = dev_name.split()[0]
            dev_role = TEAM.get(dev_name, {}).get("role", "")

            st.markdown(f"""
            <div class="dev-card">
                <div class="dev-name">{status_emoji} {first_name}</div>
                <div style="font-size: 0.75rem; color: #74b9ff; margin-bottom: 0.4rem; opacity: 0.8;">{dev_role}</div>
                <div style="font-size: 0.85rem; color: #aaa;">
                    {load.total_hours:.1f}h / {load.capacity_hours}h ({util:.0f}%)
                </div>
                <div class="load-bar-bg">
                    <div class="load-bar-fill {load_class}" style="width: {bar_width}%"></div>
                </div>
                <details>
                    <summary style="font-size: 0.85rem; color: #aaa; cursor: pointer; margin-top: 0.8rem; outline: none; user-select: none;">
                        Assigned Tasks ({len(sorted_assigned)})
                    </summary>
                    <div style="margin-top: 0.5rem;">
                        {tasks_html}
                    </div>
                </details>
            </div>
            """, unsafe_allow_html=True)

    if plan.unassigned_tasks:
        st.markdown("---")
        st.markdown("### ⚠️ Unassigned / Deferred Tasks")
        for task in plan.unassigned_tasks:
            st.markdown(f"- **[{task.type}]** {task.summary} ({task.sp} SP)")

    if plan.constraints:
        st.markdown("---")
        st.markdown("### 🔒 Active Constraints")
        badges = ""
        for c in plan.constraints:
            label = f"{c.type}"
            if c.developer:
                label += f": {c.developer.split()[0]}"
            if c.value:
                label += f" → {c.value}"
            if c.task_summary:
                label += f" ({c.task_summary[:25]})"
            badges += f'<span class="constraint-badge">{label}</span> '
        st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)


if st.session_state.plan:
    plan = st.session_state.plan

    st.markdown(f"### 📊 Sprint Plan v{plan.version}")

    # Show estimation mode warning if ML model was missing
    if hasattr(st.session_state, "orchestrator") and st.session_state.orchestrator:
        ctx = getattr(st.session_state.orchestrator, "_last_context", {})
        if ctx.get("estimation_warning"):
            st.warning(ctx["estimation_warning"])

    render_plan(plan)

    with st.expander("📋 Full Task Breakdown", expanded=False):
        task_data = []
        sorted_all_tasks = sorted(plan.tasks, key=lambda t: int(re.search(r'\d+', str(t.id)).group()) if re.search(r'\d+', str(t.id)) else 0)
        for task in sorted_all_tasks:
            task_data.append({
                "ID": task.id,
                "Summary": task.summary,
                "Description": task.description,
                "Type": task.type,
                "Category": getattr(task, "category", "Developer"),
                "Priority": task.priority,
                "Predicted Hours": f"{task.predicted_hours:.1f}h",
                "Range": f"{task.range_low:.1f}-{task.range_high:.1f}h",
                "Assigned To": task.assigned_to or "—",
                "Reasoning": task.reasoning,
            })
        if task_data:
            st.dataframe(pd.DataFrame(task_data), width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("""
    <div class="feedback-area">
        <h3>💬 Negotiate the Plan</h3>
        <p style="color: #888; font-size: 0.9rem;">
            Type natural language feedback to modify the plan. Examples:<br>
            • "Limit Nikos to 20 hours"<br>
            • "Assign the Stripe task to Maria"<br>
            • "Marios Kontis cannot do task 3"<br>
            • "Move the SEO task to next sprint"
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("feedback_form", clear_on_submit=True):
        feedback_text = st.text_input(
            "Your feedback:",
            placeholder="Type your constraint or modification...",
            label_visibility="collapsed",
        )
        apply_btn = st.form_submit_button("🔄 Apply Feedback & Re-optimize",
                                          type="primary", use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        approve_btn = st.button("✅ Approve Plan", use_container_width=True)
    with col2:
        export_btn = st.button("📥 Export CSV", use_container_width=True)

    if apply_btn and feedback_text:
        with st.spinner("🔄 Re-optimizing plan..."):
            orchestrator = st.session_state.orchestrator
            new_plan = orchestrator.apply_feedback(feedback_text)
            st.session_state.plan = new_plan
            st.session_state.plan_history.append(new_plan)
            st.session_state.feedback_log.append({
                "version": new_plan.version,
                "feedback": feedback_text,
                "timestamp": time.strftime("%H:%M:%S"),
            })
        st.rerun()

    if approve_btn:
        st.success("✅ Sprint Plan approved! Final version: v" + str(plan.version))
        st.balloons()

    if export_btn:
        rows = []
        for task in plan.tasks:
            rows.append({
                "ID": task.id,
                "Summary": task.summary,
                "Type": task.type,
                "Project": task.project,
                "Story Points": task.sp,
                "Priority": task.priority,
                "Predicted Hours": task.predicted_hours,
                "Range Low": task.range_low,
                "Range High": task.range_high,
                "Assigned To": task.assigned_to or "",
                "Reasoning": task.reasoning,
            })
        csv_df = pd.DataFrame(rows)
        csv_data = csv_df.to_csv(index=False)
        st.download_button(
            "📥 Download Sprint Plan CSV",
            data=csv_data,
            file_name=f"sprint_plan_v{plan.version}.csv",
            mime="text/csv",
        )

    if st.session_state.feedback_log:
        with st.expander("📜 Feedback History"):
            for entry in reversed(st.session_state.feedback_log):
                st.markdown(
                    f"**v{entry['version']}** ({entry['timestamp']}) — "
                    f"_{entry['feedback']}_"
                )

    if len(st.session_state.plan_history) > 1:
        with st.expander("📊 Plan Version Comparison"):
            comparison_data = []
            for p in st.session_state.plan_history:
                stats = p.summary_stats()
                comparison_data.append({
                    "Version": f"v{p.version}",
                    "Tasks": stats["total_tasks"],
                    "Assigned": stats["assigned_tasks"],
                    "Total Hours": f"{stats['total_hours']}h",
                    "Utilization": f"{stats['team_utilization']}%",
                    "Constraints": stats["num_constraints"],
                })
            st.dataframe(pd.DataFrame(comparison_data), width="stretch", hide_index=True)

    # State machine transitions log
    orchestrator = st.session_state.orchestrator
    if orchestrator and orchestrator.state_log:
        with st.expander("🔀 Pipeline State Transitions"):
            for t in orchestrator.state_log:
                st.markdown(
                    f"**`{t.from_state}`** → **`{t.to_state}`** "
                    f"<span style='color:#888;font-size:0.85rem;'>({t.timestamp})</span><br>"
                    f"<span style='color:#74b9ff;'>{t.reason}</span>",
                    unsafe_allow_html=True,
                )

else:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; color: #666;">
        <h2 style="color: #74b9ff;">👈 Enter a Project Description</h2>
        <p style="font-size: 1.1rem;">
            Describe your project requirements in the sidebar, then click<br>
            <strong>"🤖 Generate Sprint Plan"</strong> to start the AI planning pipeline.
        </p>
        <p style="font-size: 0.9rem; color: #555; margin-top: 1rem;">
            The system will: decompose tasks → estimate effort → assign to developers → let you negotiate
        </p>
    </div>
    """, unsafe_allow_html=True)
