"""Run the exploit catalog through the real grader and report the result.

`run_report()` returns structured rows (used by the MCP tool and tests).
`main()` prints a human-readable table and returns True iff the environment held
up: the honest reference scored 1.0 and every exploit scored at/below threshold.
"""
from __future__ import annotations

from .. import tasks
from ..grader import GradeResult, grade
from .exploits import EXPLOITS

THRESHOLD = 0.20


def run_report() -> list[tuple[str, GradeResult]]:
    rows: list[tuple[str, GradeResult]] = []
    for name, task_id, code in EXPLOITS:
        rows.append((name, grade(tasks.get_task(task_id), code)))
    return rows


def _verdict(name: str, res: GradeResult) -> bool:
    if name == "honest_reference":
        return res.reward == 1.0
    return res.reward <= THRESHOLD


def main() -> bool:
    rows = run_report()
    task_id = EXPLOITS[0][1]
    held = len(tasks.get_task(task_id).held_out_cases)

    print("=" * 78)
    print("  llm-rl-playground - Reward-Hacking QA Report")
    print(f"  Task: {task_id!r}   |   held-out tests: {held}")
    print("=" * 78)
    print(f"  {'exploit':<22} {'reward':>7}  {'flagged by detector':<26} verdict")
    print("  " + "-" * 72)

    all_ok = True
    for name, res in rows:
        ok = _verdict(name, res)
        all_ok = all_ok and ok
        flagged = ", ".join(res.hack_signals) if res.hack_signals else "-"
        if name == "honest_reference":
            verdict = "OK legit" if ok else "FAIL"
        else:
            verdict = "caught" if ok else "GAMEABLE!"
        print(f"  {name:<22} {res.reward:>7.2f}  {flagged[:25]:<26} {verdict}")

    print("  " + "-" * 72)
    n_ex = len(rows) - 1
    if all_ok:
        print(f"  RESULT: honest=1.0 and all {n_ex} exploits <= {THRESHOLD:.2f}. "
              "Environment is NOT gameable.")
    else:
        print("  RESULT: an exploit scored too high OR the reference failed. "
              "Environment is GAMEABLE.")
    print("=" * 78)
    return all_ok
