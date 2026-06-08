#!/usr/bin/env python
"""Run an agent against the gym and report pass rate + reward-hack rate.

Default agent `claude` needs ANTHROPIC_API_KEY (+ `uv run --extra agent rl-rollouts`).
Use `--agent scripted` for an offline baseline that needs no key.
"""
import _bootstrap  # noqa: F401
from playground.rollout import main

if __name__ == "__main__":
    main()
