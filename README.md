# llm-rl-playground

A small **RL playground for LLMs**: build reward environments, graders, and the
**reward-hacking QA** that proves they can't be gamed — across two tracks.
**Track 1 — code:** a strong, verifiable reward (RLVR). **Track 2 — truthfulness:**
grading open-ended answers and catching lies, where there is *no* ground-truth checker.

> No model training, no GPU. This builds the *environments, graders, and anti-cheat QA* —
> the slice an RL-data/environments team owns. Training the model is a separate concern.

---

## For reviewers (start here)

A working miniature of the RL-data workflow: build the execution environment, write the
grader, and **prove the grader can't be reward-hacked** — for both verifiable (code) and
non-verifiable (truthfulness) tasks.

**60-second tour:**
1. `python scripts/run_qa.py` — every cheat attempt scores ~0 (the anti-reward-hacking proof).
2. Skim `src/playground/sandbox.py` (isolated execution), `grader.py` (held-out-test reward),
   and `qa/exploits.py` + `qa/checks.py` (exploit catalog + detector).
3. `python scripts/run_rollouts.py --agent scripted` — the full rollout loop end-to-end (no key).

**Where each responsibility shows up:**

| Responsibility | In this repo |
|---|---|
| the execution environments RL tasks run in | `src/playground/env.py` (Gymnasium) + `sandbox.py` |
| prompts, evals, and **graders** | `grader.py`, `truthfulness/grader.py`, `rollout.py` |
| QA frameworks to catch **reward hacking** | `qa/` + `tests/test_reward_hacking.py` |
| **sandboxing** execution environments | `sandbox.py` (timeouts, rlimits, no-answers-on-disk) |
| third-party tool/API connector (**MCP servers**) | `src/playground/mcp/server.py` |
| RL on LLMs / reward design | code track (RLVR) + truthfulness track |

The unifying idea: **a reward is only as good as its verifier; the job is making verifiers
harder to fool than the policy is at fooling them.**

---

## Requirements

- **Python 3.11+**
- **An Anthropic API key** — only for the live Claude agent (rollouts) and the online
  truthfulness decomposition. **Everything else, including the headline QA, needs no key.**
- **Node.js** — optional, only for the MCP Inspector.

## Setup

```bash
git clone <your-repo-url>
cd llm-rl-playground
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

pip install -e .            # core only — runs the QA + tests below
# pip install -e ".[all]"   # + Claude agent, gymnasium, and the MCP server
```

The core (environment, grader, reward-hacking QA) is **pure standard library** — it installs
and runs with zero third-party packages. The agent, MCP server, and gymnasium are optional
extras, so a reviewer is never blocked on a dependency or a key.

## Run it — least-setup first

**1 · Prove the environment can't be cheated** — *no key, ~5s. The headline.*
```bash
python scripts/run_qa.py
```
→ a table where every planted cheat (hardcoding, reading the test file, fake-exit, infinite
loop, …) scores ~0, ending in `Environment is NOT gameable.`

**2 · Run the test suite** — *no key.*
```bash
pip install pytest && pytest -v        # or: pip install -e ".[dev]"
```
→ sandbox, grader, environment, reward-hacking, MCP-logic, and truthfulness suites green.

**3 · Run the rollout loop** — *`--agent scripted` needs no key; `claude` needs a key + `[agent]`.*
```bash
python scripts/run_rollouts.py --agent scripted        # offline baseline
python scripts/run_rollouts.py --agent claude --n 10   # real Claude (claude-opus-4-8)
```
→ per-episode rewards + a summary: **pass rate** and **reward-hack rate**; saved to
`results/rollouts.json`.

**4 · Validate the truthfulness grader (Track 2)** — *no key.*
```bash
python scripts/run_truthfulness_metaeval.py
```
→ the grader's lie-detection vs. gold labels (accuracy + Cohen's κ) on the honeypots.

**5 · Drive it live over MCP** — *needs `[mcp]`.*
```bash
python scripts/run_mcp.py        # serves over stdio
```
See "Drive it from Claude Desktop" below.

---

## How Track 1 defends against reward hacking

| Cheat attempt | Why it fails |
|---|---|
| Hardcode the visible-test answers | Scored on **held-out** tests it never saw |
| Read the hidden test file off disk | **Expected outputs never enter the sandbox** |
| Overwrite the assert / fake a pass with `sys.exit(0)` | Results come from a **nonce-marked, per-test protocol**, not exit code or stdout |
| Infinite loop to stall the grader | Wall-clock **timeout** + resource limits |
| Bare `except:` to swallow failures | Flagged by the **static (AST) detector**; still fails the real tests |

Defenses are **structural and per-category**: each is fixed once and protects every task —
that's what makes the anti-cheat scale (you don't re-patch per problem).

## Track 2 — verifying open-ended answers (the don't-lie track)

Code is the easy case. Most user queries have no automatic checker, so Track 2 turns one
unverifiable judgment into many smaller verifiable ones: **decompose** an answer into atomic
claims, **verify** each against provided sources, **check citations**, and reward calibrated
**abstention**. The grader has its **own** reward-hacking QA (confident liar, fabricated
citation, negation flip, vague dodge) and a **meta-eval** (`meta_eval.py`) that validates it
against gold labels — because a weak verifier you haven't validated is not trustworthy.
Honest framing: truthfulness verification is **not solved**, it's a *managed gap*.

## Drive it from Claude Desktop (MCP)

The gym is also an MCP server. Point Claude Desktop at it:

```jsonc
// claude_desktop_config.json
{
  "mcpServers": {
    "llm-rl-playground": {
      "command": "python",
      "args": ["/absolute/path/to/llm-rl-playground/scripts/run_mcp.py"]
    }
  }
}
```

Then ask Claude *"solve the tasks in the gym."* Watch it call `get_task` → write code →
`submit_solution` → read its reward live; if it tries a trick, the `hack_signals` come back.
Inspect the tools directly with `npx @modelcontextprotocol/inspector python scripts/run_mcp.py`.

## Threat model (honest limits)

The sandbox is a *best-effort educational* boundary, not a security product: process
isolation, resource limits, timeouts, and a design where the answer key never reaches the
sandbox. It does **not** provide kernel-level isolation; for untrusted code at scale you'd
reach for containers / gVisor / network namespaces. Naming the gap is deliberate.

## Layout

```
src/playground/
  tasks.py        sandbox.py     grader.py     env.py
  agents.py       rollout.py
  qa/             checks.py  exploits.py  report.py
  mcp/            server.py
  truthfulness/   grader.py  claims.py  honeypots.py  meta_eval.py  exploits.py
scripts/   run_qa.py  run_rollouts.py  run_mcp.py  run_truthfulness_metaeval.py
tests/     test_sandbox/grader/env/reward_hacking/mcp_server/truthfulness
```
