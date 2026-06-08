"""Verifier / reward function for the code track.

Reward = fraction of HELD-OUT tests that pass. The grader also runs the static
hack-signal detector and reports the sandbox status, so callers (rollouts, MCP)
can surface a reward-hack rate alongside the pass rate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .qa.checks import detect_hack_signals
from .sandbox import run_candidate
from .tasks import Task


@dataclass
class GradeResult:
    reward: float
    n_passed: int
    n_total: int
    per_test: list[dict[str, Any]]
    hack_signals: list[str]
    sandbox_status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "n_passed": self.n_passed,
            "n_total": self.n_total,
            "per_test": self.per_test,
            "hack_signals": self.hack_signals,
            "sandbox_status": self.sandbox_status,
        }


def _normalize(value: Any) -> Any:
    # Compare like-for-like with the JSON-roundtripped candidate output.
    return json.loads(json.dumps(value))


def grade(task: Task, solution_code: str, timeout: float = 5.0) -> GradeResult:
    """Score ``solution_code`` against ``task``'s held-out cases."""
    cases = task.held_out_cases
    inputs = [list(c.args) for c in cases]
    expected = [_normalize(c.expected) for c in cases]

    hack_signals = detect_hack_signals(solution_code)
    run = run_candidate(solution_code, task.entry_point, inputs, timeout=timeout)

    per_test: list[dict[str, Any]] = []
    n_passed = 0
    if run.status != "ok" or run.outputs is None:
        for c in cases:
            per_test.append({"args": list(c.args), "passed": False, "reason": run.status})
    else:
        for c, exp, out in zip(cases, expected, run.outputs):
            ok = bool(out.get("ok")) and out.get("value") == exp
            per_test.append(
                {
                    "args": list(c.args),
                    "passed": ok,
                    "expected": exp,
                    "got": out.get("value") if out.get("ok") else None,
                    "error": out.get("error"),
                }
            )
            if ok:
                n_passed += 1

    n_total = len(cases)
    reward = n_passed / n_total if n_total else 0.0
    return GradeResult(reward, n_passed, n_total, per_test, hack_signals, run.status)
