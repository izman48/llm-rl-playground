#!/usr/bin/env python
"""Track 2: validate the truthfulness grader against gold labels (no API key)."""
import _bootstrap  # noqa: F401
from playground.truthfulness.meta_eval import main

if __name__ == "__main__":
    main()
