#!/usr/bin/env python
"""Run an agent against the gym and report pass rate + reward-hack rate.

Default agent `claude` needs ANTHROPIC_API_KEY (+ `pip install -e ".[agent]"`).
Use `--agent scripted` for an offline baseline that needs no key.
"""
import argparse
import json
import os

import _bootstrap  # noqa: F401
from playground.agents import ClaudeAgent, ScriptedAgent
from playground.rollout import run_rollouts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of episodes")
    ap.add_argument("--agent", choices=["claude", "scripted"], default="claude")
    ap.add_argument("--max-attempts", type=int, default=1)
    args = ap.parse_args()

    agent = ScriptedAgent() if args.agent == "scripted" else ClaudeAgent()
    out = run_rollouts(agent, n=args.n, max_attempts=args.max_attempts)

    for e in out["episodes"]:
        flag = "OK " if e["reward"] >= 1.0 else "~~ "
        hacks = ",".join(e["hack_signals"]) or "-"
        print(f"  {flag} {e['task_id']:<16} reward={e['reward']:.2f}  hack_signals={hacks}")

    s = out["summary"]
    print("-" * 60)
    print(f"  episodes={s['episodes']}  mean_reward={s['mean_reward']:.2f}  "
          f"pass_rate={s['pass_rate']:.0%}  hack_rate={s['hack_rate']:.0%}")

    os.makedirs("results", exist_ok=True)
    with open("results/rollouts.json", "w") as f:
        json.dump(out, f, indent=2)
    print("  saved -> results/rollouts.json")


if __name__ == "__main__":
    main()
