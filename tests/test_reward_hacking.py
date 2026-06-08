"""The headline suite: prove every exploit category stays un-gameable."""
import pytest

from playground import tasks
from playground.grader import grade
from playground.qa.exploits import EXPLOITS
from playground.qa.report import THRESHOLD


@pytest.mark.parametrize("name,task_id,code", EXPLOITS, ids=[e[0] for e in EXPLOITS])
def test_exploit_not_gameable(name, task_id, code):
    res = grade(tasks.get_task(task_id), code)
    if name == "honest_reference":
        assert res.reward == 1.0
    else:
        assert res.reward <= THRESHOLD, (name, res.as_dict())


def test_detector_flags_obvious_tells():
    from playground.qa.checks import detect_hack_signals

    assert "call:sys.exit" in detect_hack_signals("import sys\nsys.exit(0)\n")
    assert "bare_except" in detect_hack_signals("try:\n    x\nexcept:\n    pass\n")
    assert detect_hack_signals("def f(x):\n    return x\n") == []
