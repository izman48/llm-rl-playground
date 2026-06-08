"""Run episodes of an agent against the code environment and collect metrics."""
from __future__ import annotations

from typing import Any

from .agents import Agent
from .env import CodeEnv
from .tasks import all_tasks


def run_rollouts(
    agent: Agent,
    n: int = 10,
    task_ids: list[str] | None = None,
    max_attempts: int = 1,
    timeout: float = 5.0,
) -> dict[str, Any]:
    env = CodeEnv(task_ids=task_ids, max_attempts=max_attempts, timeout=timeout)
    ids = task_ids or [t.id for t in all_tasks()]

    episodes: list[dict[str, Any]] = []
    for i in range(n):
        task_id = ids[i % len(ids)]
        obs, info = env.reset(options={"task_id": task_id})
        reward, last_info, done = 0.0, {}, False
        while not done:
            code = agent.act(obs, info)
            obs, reward, terminated, truncated, last_info = env.step(code)
            done = terminated or truncated
        episodes.append(
            {
                "task_id": task_id,
                "reward": reward,
                "hack_signals": last_info.get("hack_signals", []),
                "sandbox_status": last_info.get("sandbox_status"),
            }
        )

    n_total = len(episodes) or 1
    solved = sum(1 for e in episodes if e["reward"] >= 1.0)
    hacked = sum(1 for e in episodes if e["hack_signals"])
    summary = {
        "episodes": len(episodes),
        "mean_reward": sum(e["reward"] for e in episodes) / n_total,
        "pass_rate": solved / n_total,
        "hack_rate": hacked / n_total,
    }
    return {"summary": summary, "episodes": episodes}
