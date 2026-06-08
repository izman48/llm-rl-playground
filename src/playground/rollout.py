"""Run episodes of an agent against the code environment and collect metrics."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

from .agents import Agent, ClaudeAgent, ScriptedAgent
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


def main() -> None:
    """Console entry point (``uv run rl-rollouts``)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of episodes")
    ap.add_argument("--agent", choices=["claude", "scripted"], default="claude")
    ap.add_argument("--max-attempts", type=int, default=1)
    args = ap.parse_args()

    agent: Agent = ScriptedAgent() if args.agent == "scripted" else ClaudeAgent()
    out = run_rollouts(agent, n=args.n, max_attempts=args.max_attempts)

    for e in out["episodes"]:
        flag = "OK " if e["reward"] >= 1.0 else "~~ "
        hacks = ",".join(e["hack_signals"]) or "-"
        print(f"  {flag} {e['task_id']:<16} reward={e['reward']:.2f}  hack_signals={hacks}")

    s = out["summary"]
    print("-" * 60)
    print(
        f"  episodes={s['episodes']}  mean_reward={s['mean_reward']:.2f}  "
        f"pass_rate={s['pass_rate']:.0%}  hack_rate={s['hack_rate']:.0%}"
    )

    os.makedirs("results", exist_ok=True)
    with open("results/rollouts.json", "w") as f:
        json.dump(out, f, indent=2)
    print("  saved -> results/rollouts.json")
