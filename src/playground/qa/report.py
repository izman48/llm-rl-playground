"""Run the exploit generators across every task and report the result.

`run_report()` returns one row per strategy: its **worst case** (highest reward)
across all tasks, so a single passing row means the category is defeated on the
whole task set, not just one example. The honest reference is reported as its
**worst** (lowest) reward across tasks — it must still be 1.0 everywhere.

`main()` prints a human-readable table and returns True iff the environment held
up: the reference scored 1.0 on every task and every exploit category scored
at/below threshold on every task.
"""
from __future__ import annotations

from .. import tasks
from ..grader import GradeResult, grade
from .exploits import STRATEGIES

THRESHOLD = 0.20
SWEEP_TIMEOUT = 1.0  # short: the timeout defense doesn't need 5s to demonstrate, and
                     # infinite_loop runs once per task, so keep the sweep fast.


def run_report() -> list[tuple[str, GradeResult]]:
    """One row per strategy = worst case across all tasks (plus the reference)."""
    all_tasks = tasks.all_tasks()
    rows: list[tuple[str, GradeResult]] = []

    # Honest control: the reference must score 1.0 on EVERY task -> report the min.
    ref = [grade(t, t.reference, timeout=SWEEP_TIMEOUT) for t in all_tasks]
    rows.append(("honest_reference", min(ref, key=lambda r: r.reward)))

    # Each exploit must stay low on EVERY task -> report the max (its best shot).
    for name, strat in STRATEGIES:
        results = [grade(t, strat(t), timeout=SWEEP_TIMEOUT) for t in all_tasks]
        rows.append((name, max(results, key=lambda r: r.reward)))
    return rows


def _verdict(name: str, res: GradeResult) -> bool:
    if name == "honest_reference":
        return res.reward == 1.0
    return res.reward <= THRESHOLD


def main() -> bool:
    rows = run_report()
    all_tasks = tasks.all_tasks()
    n_tasks = len(all_tasks)
    n_held = sum(len(t.held_out_cases) for t in all_tasks)

    print("=" * 78)
    print("  llm-rl-playground - Reward-Hacking QA Report")
    print(f"  tasks: {n_tasks}   |   held-out tests across tasks: {n_held}")
    print("  each exploit category is run against EVERY task; the row shows its worst case.")
    print("=" * 78)
    print(f"  {'exploit':<22} {'max reward':>10}  {'flagged by detector':<22} verdict")
    print("  " + "-" * 72)

    all_ok = True
    for name, res in rows:
        ok = _verdict(name, res)
        all_ok = all_ok and ok
        flagged = ", ".join(res.hack_signals) if res.hack_signals else "-"
        if name == "honest_reference":
            label = "min reward"
            verdict = "OK legit" if ok else "FAIL"
        else:
            label = "max reward"
            verdict = "caught" if ok else "GAMEABLE!"
        print(f"  {name:<22} {res.reward:>10.2f}  {flagged[:21]:<22} {verdict}")

    print("  " + "-" * 72)
    n_ex = len(rows) - 1
    if all_ok:
        print(f"  RESULT: reference=1.0 on every task and all {n_ex} exploit categories "
              f"<= {THRESHOLD:.2f} across all {n_tasks} tasks. Environment is NOT gameable.")
    else:
        print("  RESULT: an exploit scored too high OR the reference failed. "
              "Environment is GAMEABLE.")
    print("=" * 78)
    return all_ok


def cli_main() -> None:
    """Console entry point (``uv run rl-qa``)."""
    import sys

    sys.exit(0 if main() else 1)
