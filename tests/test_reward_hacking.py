"""The headline suite: prove every exploit category stays un-gameable on EVERY task.

Generates each exploit strategy for each task and asserts its reward stays at/below
threshold, and that the reference solution still scores 1.0 everywhere. This is the
proof that backs the "not gameable across the task distribution" claim.
"""
import pytest

from playground import tasks
from playground.grader import grade
from playground.qa.exploits import STRATEGIES
from playground.qa.report import SWEEP_SEED, SWEEP_TIMEOUT, THRESHOLD

_TASKS = tasks.all_tasks()
_CASES = [(t.id, name, strat) for t in _TASKS for name, strat in STRATEGIES]


@pytest.mark.parametrize(
    "task_id,name,strat", _CASES, ids=[f"{tid}-{n}" for tid, n, _ in _CASES]
)
def test_exploit_not_gameable(task_id, name, strat):
    task = tasks.get_task(task_id)
    res = grade(task, strat(task), timeout=SWEEP_TIMEOUT, seed=SWEEP_SEED)
    assert res.reward <= THRESHOLD, (task_id, name, res.as_dict())


@pytest.mark.parametrize("task", _TASKS, ids=[t.id for t in _TASKS])
def test_reference_scores_full(task):
    res = grade(task, task.reference, timeout=SWEEP_TIMEOUT, seed=SWEEP_SEED)
    assert res.reward == 1.0, (task.id, res.as_dict())


def test_detector_flags_obvious_tells():
    from playground.qa.checks import detect_hack_signals

    assert "call:sys.exit" in detect_hack_signals("import sys\nsys.exit(0)\n")
    assert "bare_except" in detect_hack_signals("try:\n    x\nexcept:\n    pass\n")
    assert detect_hack_signals("def f(x):\n    return x\n") == []
