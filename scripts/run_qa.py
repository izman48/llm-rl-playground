#!/usr/bin/env python
"""Prove the environment can't be reward-hacked. No API key needed."""
import sys

import _bootstrap  # noqa: F401  (puts src/ on sys.path)
from playground.qa.report import main

if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
