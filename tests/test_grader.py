import pytest

from playground import tasks
from playground.grader import grade


@pytest.mark.parametrize("task", tasks.all_tasks(), ids=lambda t: t.id)
def test_reference_solution_scores_full(task):
    res = grade(task, task.reference)
    assert res.reward == 1.0, res.per_test
    assert res.sandbox_status == "ok"


def test_wrong_solution_scores_below_full():
    t = tasks.get_task("reverse_string")
    res = grade(t, "def reverse_string(s):\n    return s\n")  # identity, not reverse
    assert res.reward < 1.0


def test_crash_scores_zero():
    t = tasks.get_task("fibonacci")
    res = grade(t, "def fibonacci(n):\n    return 1 / 0\n")
    assert res.reward == 0.0
    assert all(not pt["passed"] for pt in res.per_test)
