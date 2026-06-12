# llm-rl-playground

A small **RL playground for LLMs**: build reward environments, graders, and the
**reward-hacking QA** that proves they can't be gamed — across two tracks.
**Track 1 — code:** a strong, verifiable reward (RLVR). **Track 2 — truthfulness:**
grading open-ended answers and catching lies, where there is *no* ground-truth checker.

> No model training, no GPU. This builds the *environments, graders, and anti-cheat QA* —
> the slice an RL-data/environments team owns. Training the model is a separate concern.

---

## Overview

A working miniature of the RL-data workflow: build the execution environment, write the
grader, and **prove the grader can't be reward-hacked** — for both verifiable (code) and
non-verifiable (truthfulness) tasks.

**60-second tour:**
1. `uv run rl-qa` — every cheat attempt scores ~0 on every task (the anti-reward-hacking proof).
2. Skim `src/playground/sandbox.py` (isolated execution), `grader.py` (held-out +
   generated-test reward), `tasks.py` (specs + input generators), and `qa/exploits.py` +
   `qa/checks.py` (exploit catalog + detector).
3. `uv run rl-rollouts --agent scripted` — the full rollout loop end-to-end (no key).

**Where each responsibility shows up:**

| Responsibility | In this repo |
|---|---|
| the execution environments RL tasks run in | `src/playground/env.py` (Gymnasium) + `sandbox.py` |
| prompts, evals, and **graders** | `grader.py`, `truthfulness/grader.py`, `rollout.py` |
| QA frameworks to catch **reward hacking** | `qa/` + `tests/test_reward_hacking.py` |
| **sandboxing** execution environments | `sandbox.py` (timeout + best-effort rlimits — CPU; memory where supported, no-answers-on-disk) |
| third-party tool/API connector (**MCP servers**) | `src/playground/mcp/server.py` |
| RL on LLMs / reward design | code track (RLVR) + truthfulness track |

The unifying idea: **a reward is only as good as its verifier; the job is making verifiers
harder to fool than the policy is at fooling them.**

---

## Requirements

- **Python 3.11+** and **[uv](https://docs.astral.sh/uv/)**
- **An Anthropic API key** — only for the live Claude agent (rollouts) and the online
  truthfulness decomposition. **Everything else, including the headline QA, needs no key.**
- **Node.js** — optional, only for the MCP Inspector.

## Setup

```bash
git clone <your-repo-url>
cd llm-rl-playground
```

**`uv run` syncs the environment automatically** — no manual install step. The core
(environment, grader, reward-hacking QA) is **pure standard library**; optional extras
(`agent`, `mcp`, `rl`, `dev`) are pulled in on demand with `--extra`.

| Command | Extra needed |
|---|---|
| `uv run rl-qa` | none |
| `uv run --extra dev pytest -q` | `dev` |
| `uv run rl-rollouts --agent scripted` | none |
| `uv run --extra agent rl-rollouts --agent claude` | `agent` |
| `uv run rl-metaeval` | none |
| `uv run --extra mcp rl-mcp` | `mcp` |

Use `--all-extras` instead of `--extra <name>` if you want every optional dependency at once.

## Run it — least-setup first

**1 · Prove the environment can't be cheated** — *no key, ~10s. The headline.*
```bash
uv run rl-qa
```
→ a table where every planted cheat (hardcoding, reading the test file, fake-exit, infinite
loop, …) — generated for and run against **all 8 tasks** — scores ~0, ending in
`Environment is NOT gameable.`

**2 · Run the test suite** — *no key.*
```bash
uv run --extra dev pytest -q
```
→ sandbox, grader, environment, reward-hacking, MCP-logic, and truthfulness suites green.

**3 · Run the rollout loop** — *`--agent scripted` needs no key; `claude` needs a key + `agent` extra.*
```bash
uv run rl-rollouts --agent scripted                  # offline baseline
ANTHROPIC_API_KEY=sk-ant-... uv run --extra agent rl-rollouts --agent claude --n 10
```
→ per-episode rewards + a summary: **pass rate** and **reward-hack rate**; saved to
`results/rollouts.json`.

**4 · Validate the truthfulness grader (Track 2)** — *no key.*
```bash
uv run rl-metaeval
```
→ the grader's lie-detection vs. gold labels (accuracy + Cohen's κ) on the honeypots.

**5 · Drive it live over MCP** — *needs `mcp` extra.*
```bash
uv run --extra mcp rl-mcp        # serves over stdio
```
See "Drive it over MCP" below.

The `scripts/run_*.py` wrappers still work (`uv run python scripts/run_qa.py`, etc.) if you
prefer explicit paths.

---

## How Track 1 defends against reward hacking

| Cheat attempt | Why it fails |
|---|---|
| Hardcode the visible-test answers | Scored on **held-out** tests it never saw |
| Memorize / overfit the held-out set itself | The held-out set isn't fixed — most cases are **generated fresh each grade** (see below), so there's no stable answer set to memorize |
| Read the hidden test file off disk | **Expected outputs never enter the sandbox** |
| Overwrite the assert / fake a pass with `sys.exit(0)` | Results come from a **nonce-marked, per-test protocol**, not exit code or stdout |
| Infinite loop to stall the grader | Wall-clock **timeout** + resource limits |
| Bare `except:` to swallow failures | Flagged by the **static (AST) detector**; still fails the real tests |

Defenses are **structural and per-category**: each is fixed once and protects every task —
that's what makes the anti-cheat scale (you don't re-patch per problem).

**Dynamic test generation (differential testing).** A fixed held-out set is still
something an RL agent can overfit to with enough episodes — if the tests never change, the
reward signal stops distinguishing "solved the task" from "memorized these inputs." So the
grader doesn't rely on a fixed set. Each task ships an **input generator**; at grade time it
samples fresh inputs (seeded — the env uses a new seed every step) and the hidden **reference
solution labels them** as the oracle. The expected outputs are computed live and never stored,
so there is no answer key to overfit to: a different seed is a different test set, and a
solution has to actually generalize. The curated edge cases remain as fixed anchors on top, so
the tricky corners (empty input, negatives, …) are always covered. `tests/test_generators.py`
proves the property: a lookup table of the static cases scores 1.0 with generation off and
**below a pass with it on**.

## Track 2 — verifying open-ended answers (the don't-lie track)

Code is the easy case. Most user queries have no automatic checker. Track 2 is an
**illustrative sketch** of how you'd attack that harder problem — a toy, not a finished
verifier. It turns one unverifiable judgment into smaller checkable ones: **decompose** an
answer into atomic claims (an optional online step via Claude in `claims.py`, *not exercised
by the keyless QA*), **verify** each against provided sources (a **toy token-overlap proxy**
standing in for an NLI/LLM verifier), **check citations** exist, and give honest **abstention**
a small fixed credit. The grader has its **own** reward-hacking fixtures (confident liar,
fabricated citation, negation flip, vague dodge) and a **meta-eval** (`meta_eval.py`) that
measures grader-vs-gold agreement (accuracy + Cohen's κ) on a **small n=5 labeled set — a smoke
check, not a statistical guarantee**. Honest framing: truthfulness verification is **not
solved**; this track is a signpost to the next step (see *What's missing / what's next*).

## Drive it over MCP

The gym is also an MCP server, exposing four tools: `list_tasks`, `get_task`,
`submit_solution`, and `run_qa`.

**The one prerequisite:** the `mcp` extra must be available when the server starts —
`uv run --extra mcp` pulls it in automatically.

### Claude Desktop

**Option A — one-click bundle (`.mcpb`).** A pre-built
[`llm-rl-playground.mcpb`](llm-rl-playground.mcpb) ships in this repo. In Claude Desktop go to
**Settings → Extensions** and drag the `.mcpb` onto the window (or *Install from file…*). During
install, set the extension's **command** to `uv` and **args** to
`run --directory /absolute/path/to/llm-rl-playground --extra mcp rl-mcp` (or point
**Python executable** at any interpreter that already has `mcp`). Then toggle the extension on.

Rebuild the bundle after changing the server:

```bash
npm install -g @anthropic-ai/mcpb
mcpb pack . llm-rl-playground.mcpb      # uses manifest.json + .mcpbignore
```

**Option B — manual config.** Edit `claude_desktop_config.json` (**Settings → Developer → Edit
Config**), then restart Claude Desktop:

```jsonc
{
  "mcpServers": {
    "llm-rl-playground": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/absolute/path/to/llm-rl-playground",
        "--extra", "mcp",
        "rl-mcp"
      ]
    }
  }
}
```

### Claude Code

Register the server with one command (run it from anywhere; use absolute paths):

```bash
claude mcp add llm-rl-playground -- \
  uv run --directory /absolute/path/to/llm-rl-playground --extra mcp rl-mcp
```

Verify and inspect:

```bash
claude mcp list                      # shows llm-rl-playground ✓ connected
claude mcp get llm-rl-playground     # shows the command it will run
```

Inside a session, `/mcp` lists the server and its four tools. Use `claude mcp remove
llm-rl-playground` to undo. (The `.mcpb` bundle is a Claude Desktop installer format — for
Claude Code, use `claude mcp add` as above.)

### MCP Inspector (quick visual test, no host)

```bash
npx @modelcontextprotocol/inspector uv run --extra mcp rl-mcp
```

Opens a browser UI to click each tool and read the JSON responses directly.

---

Once connected, ask Claude *"solve the tasks in the gym."* Watch it call `get_task` → write code
→ `submit_solution` → read its reward live; if it tries a trick, the `hack_signals` come back.

## What's missing / what's next

Stated plainly, so the gaps aren't hidden:

- **The sandbox is best-effort, not a security product.** Process isolation, wall-clock
  timeout, CPU-time limit, and a design where the answer key never reaches the sandbox — but
  **no kernel-level isolation**, and the memory limit (`RLIMIT_AS`) is a **no-op on macOS**.
  For untrusted code at scale: containers / gVisor / network namespaces.
- **Verifiable rewards only cover code.** The strong, un-gameable reward is RLVR on code
  (Track 1), proven across all tasks by `uv run rl-qa`. **Non-verifiable (truthfulness)
  rewards are the sketched next step**, not a finished feature — today's Track 2 is a toy: a
  token-overlap verifier, an n=5 meta-eval, and a decomposition step not wired into the keyless QA.
- **Scaling the anti-cheat.** Add more tasks and exploit categories to the generator
  (`qa/exploits.py` — each new strategy is checked against the whole task set automatically),
  feed the per-category detector signals into a monitor/classifier, and for Track 2 swap in a
  real NLI/LLM verifier with a larger labeled honeypot set so the meta-eval becomes meaningful.

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
manifest.json  .mcpbignore  llm-rl-playground.mcpb   # MCP bundle (one-click install)
```
