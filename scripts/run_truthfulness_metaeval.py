#!/usr/bin/env python
"""Track 2: validate the truthfulness grader against gold labels (no API key)."""
import json

import _bootstrap  # noqa: F401
from playground.truthfulness import meta_eval

if __name__ == "__main__":
    out = meta_eval.evaluate()
    for r in out["rows"]:
        mark = "OK " if r["correct"] else "XX "
        print(f"  {mark} {r['name']:<22} reward={r['reward']:.2f}  "
              f"pred_truthful={r['predicted_truthful']}  gold={r['gold_truthful']}")
    print("-" * 60)
    print(f"  n={out['n']}  accuracy={out['accuracy']:.0%}  "
          f"cohen_kappa={out['cohen_kappa']}")
    print(json.dumps(out["rows"], indent=2)[:0])  # (rows already printed)
