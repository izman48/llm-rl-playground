"""Track 2 - grading open-ended answers and catching lies (weak-verifier track).

Code has a strong verifier (tests). Most user queries do not. This track turns an
unverifiable "is it true?" into many smaller verifiable checks: decompose an
answer into atomic claims, verify each against provided sources, validate
citations, and reward calibrated uncertainty. The judge is fallible, so it has
its own reward-hacking QA and a meta-eval against gold labels.
"""
