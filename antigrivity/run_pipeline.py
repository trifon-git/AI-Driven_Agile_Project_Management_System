"""
Antigrivity — CLI Pipeline Runner
===================================
Entry point for running the planning pipeline from the command line.

Usage:
    python -m antigrivity.run_pipeline --demo           # Run with sample project
    python -m antigrivity.run_pipeline --desc "..."     # Run with custom description
    python -m antigrivity.run_pipeline --interactive     # Interactive mode with feedback
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antigrivity.agents.base import Orchestrator


DEMO_PROJECT = (
    "Build a restaurant website with: online menu display with categories and images, "
    "table reservation form with date/time picker and email confirmation, "
    "Google Maps integration showing the restaurant location, "
    "responsive mobile design, SEO optimization, "
    "and an admin panel to update menu items and view reservations."
)
import torch
import numpy as np
import random

def set_deterministic_seed(seed=42):
    # Lock Python & Numpy
    random.seed(seed)
    np.random.seed(seed)
    
    # Lock PyTorch Operations
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Call BEFORE the Orchestrator starts
set_deterministic_seed(42)


def print_plan(plan):
    """Pretty-print a sprint plan to the console."""
    stats = plan.summary_stats()

    print("\n" + "=" * 70)
    print(f"  SPRINT PLAN v{plan.version}  |  {plan.timestamp}")
    print("=" * 70)

    print(f"\n  Tasks: {stats['total_tasks']}  |  "
          f"Assigned: {stats['assigned_tasks']}  |  "
          f"Hours: {stats['total_hours']}h / {stats['total_capacity']}h  |  "
          f"Utilization: {stats['team_utilization']}%\n")

    for dev_name, load in plan.developer_loads.items():
        util = load.utilization
        bar_len = int(util / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        status = "✅"
        if util > 95: status = "🔥"
        elif util > 80: status = "⚠️"
        elif util < 40: status = "❄️"

        print(f"  {dev_name:<25} {load.total_hours:>5.1f}h / {load.capacity_hours}h "
              f"{bar} {util:.0f}% {status}")

        for task in load.assigned_tasks:
            priority_marker = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(task.priority, "⚪")
            print(f"    {priority_marker} [{task.predicted_hours:5.1f}h] {task.summary}")
        print()

    if plan.unassigned_tasks:
        print(f"  ⚠️  UNASSIGNED / DEFERRED ({len(plan.unassigned_tasks)}):")
        for task in plan.unassigned_tasks:
            print(f"    • {task.summary} ({task.sp} SP)")

    if plan.constraints:
        print(f"\n  🔒 Active Constraints ({len(plan.constraints)}):")
        for c in plan.constraints:
            print(f"    • {c.type}: {c.developer or ''} {c.task_summary or ''} {c.value or ''}")

    print("\n" + "=" * 70)


def print_state_log(orchestrator):
    """Print the state machine transition log."""
    if not orchestrator.state_log:
        return
    print("\n  🔀 Pipeline State Transitions:")
    for t in orchestrator.state_log:
        print(f"    [{t.timestamp}] {t.from_state} → {t.to_state}: {t.reason}")
    print()


def run_interactive(orchestrator, plan):
    """Interactive feedback loop."""
    while True:
        print("\n" + "-" * 50)
        feedback = input("💬 Enter feedback (or 'exit' to finish): ").strip()

        if feedback.lower() in ('exit', 'quit', 'done', ''):
            print("✅ Plan finalized!")
            break

        print("🔄 Re-optimizing...")
        plan = orchestrator.apply_feedback(feedback)
        print_plan(plan)


def main():
    parser = argparse.ArgumentParser(description="Antigrivity Sprint Planner")
    parser.add_argument("--demo", action="store_true", help="Run with demo project")
    parser.add_argument("--desc", type=str, help="Project description")
    parser.add_argument("--project-key", type=str, default="PROJ", help="Jira project key")
    parser.add_argument("--interactive", action="store_true", help="Enable feedback loop")
    parser.add_argument("--test", action="store_true", help="Quick smoke test")

    args = parser.parse_args()

    description = args.desc or (DEMO_PROJECT if args.demo or args.test else None)

    if not description:
        print("Error: Provide --desc 'description' or use --demo")
        parser.print_help()
        sys.exit(1)

    print("Antigrivity Sprint Planner")
    print("=" * 40)

    print("Initializing agents...")
    orchestrator = Orchestrator()

    print(f"Planning for: {description[:80]}...")
    plan = orchestrator.run_planning_cycle(
        project_description=description,
        project_key=args.project_key,
    )

    print_plan(plan)
    print_state_log(orchestrator)

    if args.interactive:
        run_interactive(orchestrator, plan)
    elif args.test:
        print("\n✅ Smoke test passed — pipeline executed successfully.")


if __name__ == "__main__":
    main()
