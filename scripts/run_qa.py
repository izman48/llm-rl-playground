#!/usr/bin/env python
"""Prove the environment can't be reward-hacked. No API key needed."""
import _bootstrap  # noqa: F401  (puts src/ on sys.path)
from playground.qa.report import cli_main

if __name__ == "__main__":
    cli_main()
