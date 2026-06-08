"""A Gymnasium-compatible RL environment over the coding tasks.

Single-step (contextual-bandit) by default — the honest shape of RLVR: observe a
task spec, emit code, receive a reward. Set ``max_attempts > 1`` for an agentic
mode where a failed attempt is followed by non-leaking feedback (counts only —
never the held-out expected values) so the agent can revise.

Uses ``gymnasium`` when installed; otherwise falls back to a tiny base class with
the same ``reset``/``step`` signatures so the core stays dependency-free.
"""
from __future__ import annotations

import random
from typing import Any

try:  # pragma: no cover - exercised by environment, not unit-tested both ways
    import gymnasium as gym
    from gymnasium import spaces

    _Base = gym.Env
    _HAVE_GYM = True
except Exception:  # gymnasium not installed
    spaces = None  # type: ignore[assignment]
    _Base = object  # type: ignore[assignment, misc]
    _HAVE_GYM = False

from .grader import GradeResult, grade
from .tasks import Task, all_tasks, get_task


def _format_feedback(res: GradeResult) -> str:
    # Deliberately leaks NO expected values — only counts/status/signals.
    parts = [f"passed {res.n_passed}/{res.n_total} held-out tests "
             f"(sandbox: {res.sandbox_status})."]
    if res.hack_signals:
        parts.append("warning - suspicious patterns detected: "
                     + ", ".join(res.hack_signals))
    return " ".join(parts)


class CodeEnv(_Base):
    """Observation = task prompt (str). Action = candidate code (str)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        task_ids: list[str] | None = None,
        max_attempts: int = 1,
        timeout: float = 5.0,
        seed: int | None = None,
    ) -> None:
        if _HAVE_GYM:
            super().__init__()
        self._tasks: list[Task] = (
            [get_task(t) for t in task_ids] if task_ids else all_tasks()
        )
        self.max_attempts = max_attempts
        self.timeout = timeout
        self._rng = random.Random(seed)
        self._task: Task | None = None
        self._attempts = 0
        if _HAVE_GYM:
            self.observation_space = spaces.Text(max_length=10_000)
            self.action_space = spaces.Text(max_length=50_000)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any]]:
        if _HAVE_GYM:
            super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        if options and options.get("task_id"):
            self._task = get_task(options["task_id"])
        else:
            self._task = self._rng.choice(self._tasks)
        self._attempts = 0
        info = {"task_id": self._task.id, "entry_point": self._task.entry_point}
        return self._task.prompt(), info

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        assert self._task is not None, "call reset() before step()"
        self._attempts += 1
        res = grade(self._task, action, timeout=self.timeout)
        terminated = res.reward >= 1.0 or self._attempts >= self.max_attempts
        info = {"task_id": self._task.id, **res.as_dict()}
        if terminated:
            obs = self._task.prompt()
        else:
            obs = self._task.prompt() + "\n\nPrevious attempt: " + _format_feedback(res)
        return obs, res.reward, terminated, False, info
