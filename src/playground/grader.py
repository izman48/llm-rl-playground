"""Verifier / reward function for the code track.

Reward = fraction of held-out tests that pass. The held-out set is the union of
two parts:

* curated edge-case anchors (``task.held_out_cases``) — fixed, guaranteeing the
  tricky corners are always covered; and
* freshly *generated* cases — the task's ``gen_input`` samples new inputs each
  grade (seeded), and the **reference solution labels them** as the oracle.

The generated half is what defeats memorization/brute-force: the expected
outputs are computed live from the reference, never stored, so there is no fixed
answer key to overfit to. Pass a different ``seed`` and the test set changes.

The grader also runs the static hack-signal detector and reports the sandbox
status, so callers (rollouts, MCP) can surface a reward-hack rate alongside the
pass rate.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from .qa.checks import detect_hack_signals
from .sandbox import run_candidate
from .tasks import Case, Task

# How many fresh, reference-labeled cases to synthesize per grade by default.
# Chosen so generated cases dominate the curated anchors: a solution that only
# memorized the static anchors cannot reach a passing reward.
DEFAULT_GENERATED = 20


@dataclass
class GradeResult:
    reward: float
    n_passed: int
    n_total: int
    per_test: list[dict[str, Any]]
    hack_signals: list[str]
    sandbox_status: str
    n_curated: int = 0
    n_generated: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "n_passed": self.n_passed,
            "n_total": self.n_total,
            "n_curated": self.n_curated,
            "n_generated": self.n_generated,
            "per_test": self.per_test,
            "hack_signals": self.hack_signals,
            "sandbox_status": self.sandbox_status,
        }


def _normalize(value: Any) -> Any:
    # Compare like-for-like with the JSON-roundtripped candidate output.
    return json.loads(json.dumps(value))


def _oracle_cases(task: Task, rng: random.Random, n: int, timeout: float) -> list[Case]:
    """Sample ``n`` fresh inputs and label them with the reference (the oracle).

    Inputs the reference cannot answer (it raises / returns non-JSON) are dropped
    rather than scored — they signal a generator that strayed outside the spec's
    input domain, which should not penalize the candidate.
    """
    if task.gen_input is None or n <= 0:
        return []
    gen_inputs = [tuple(task.gen_input(rng)) for _ in range(n)]
    run = run_candidate(
        task.reference, task.entry_point, [list(a) for a in gen_inputs], timeout=timeout
    )
    if run.status != "ok" or run.outputs is None:
        return []  # oracle unavailable -> fall back to curated anchors only
    return [
        Case(args=args, expected=out["value"])
        for args, out in zip(gen_inputs, run.outputs)
        if out.get("ok")
    ]


def grade(
    task: Task,
    solution_code: str,
    timeout: float = 5.0,
    *,
    seed: int | None = None,
    n_generated: int = DEFAULT_GENERATED,
) -> GradeResult:
    """Score ``solution_code`` against ``task``'s curated + freshly generated cases.

    ``seed`` selects the generated inputs; the same seed reproduces the same test
    set, a different seed yields a different one. ``n_generated`` is how many
    reference-labeled cases to synthesize on top of the curated anchors.
    """
    rng = random.Random(seed)
    generated = _oracle_cases(task, rng, n_generated, timeout)
    cases = list(task.held_out_cases) + generated
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
    return GradeResult(
        reward,
        n_passed,
        n_total,
        per_test,
        hack_signals,
        run.status,
        n_curated=len(task.held_out_cases),
        n_generated=len(generated),
    )
