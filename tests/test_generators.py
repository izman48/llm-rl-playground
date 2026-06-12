"""Prove the dynamic generator defeats memorizing a fixed test set.

The headline property: with a *static* test set a lookup table scores 1.0, but
once the grader synthesizes fresh, reference-labeled cases the same lookup table
can no longer reach a passing reward — it has to generalize, not memorize.
"""
import random

import pytest

from playground import tasks
from playground.grader import _oracle_cases, grade


def _memorize_static(task: tasks.Task) -> str:
    """A candidate that hardcodes ONLY the curated static cases, else returns None.

    Keyed on ``repr(args)`` so list-argument tasks (unhashable) still build.
    """
    answers = {repr(tuple(c.args)): c.expected for c in task.held_out_cases}
    return (
        f"def {task.entry_point}(*args):\n"
        f"    answers = {answers!r}\n"
        f"    return answers.get(repr(args))\n"
    )


def test_every_task_has_a_generator():
    missing = [t.id for t in tasks.all_tasks() if t.gen_input is None]
    assert not missing, missing


@pytest.mark.parametrize("task", tasks.all_tasks(), ids=lambda t: t.id)
def test_reference_scores_full_on_generated_cases(task):
    res = grade(task, task.reference, seed=123, n_generated=25)
    assert res.reward == 1.0, res.as_dict()
    assert res.n_generated > 0  # the oracle actually labeled fresh cases


def test_generated_inputs_vary_by_seed():
    task = tasks.get_task("max_subarray")
    a = _oracle_cases(task, random.Random(1), 10, timeout=5.0)
    b = _oracle_cases(task, random.Random(2), 10, timeout=5.0)
    assert [c.args for c in a] != [c.args for c in b]


@pytest.mark.parametrize("task", tasks.all_tasks(), ids=lambda t: t.id)
def test_static_memorization_passes_only_without_generation(task):
    code = _memorize_static(task)
    # With no generated cases (the old static-only world), memorizing wins.
    assert grade(task, code, n_generated=0).reward == 1.0
    # With fresh generated cases, the same lookup table can't reach a pass.
    res = grade(task, code, seed=7, n_generated=20)
    assert res.reward < 1.0, res.as_dict()


def test_overfit_to_one_seed_fails_on_another():
    task = tasks.get_task("count_vowels")
    # Memorize exactly the cases produced at seed=1, then face a different seed.
    seen = _oracle_cases(task, random.Random(1), 20, timeout=5.0)
    answers = {repr(tuple(c.args)): c.expected for c in seen}
    code = (
        f"def {task.entry_point}(*args):\n"
        f"    answers = {answers!r}\n"
        f"    return answers.get(repr(args))\n"
    )
    # It can't beat even its own seed (curated anchors aren't in the table),
    # and it does strictly worse on an unseen seed.
    assert grade(task, code, seed=2, n_generated=20).reward < 1.0
