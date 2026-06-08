# Plan: `llm-rl-playground` — a tiny RL environment with a verifiable grader and a reward-hacking QA harness

## Context

The user is applying for **Software Engineer, RL Data at Anthropic** (job 5238606008) and wants a standout GitHub portfolio piece. That team builds "the execution environments RL tasks run in," "prompts, evals, and graders," and "QA frameworks to catch reward hacking and ensure environment quality," with sandboxing of execution environments — and lists "RL on LLMs" and "MCP servers" as preferred quals. The take-home-style prompt ("build an LLM eval/grader, a tiny RL environment, or an MCP connector") is essentially a description of this team's day job.

We're building the artifact that hits the most of those responsibilities at once: a **tiny RL environment for LLM code generation with a verifiable reward (RLVR)**, whose centerpiece is a **reward-hacking QA harness** that proves the environment isn't gameable. Crucially, this needs **no model training and no GPU** — the RL Data team owns environments, graders, and QA; training is a separate team. So "tiny" is the correct scope, not a compromise.

Decisions locked with the user: **task domain = sandboxed coding tasks**; **MCP server layer = first-class** (a FastMCP server exposes the env's tools so any MCP host — e.g. Claude Desktop — can drive the gym); **real Claude agent** for rollouts (user has an API key). Language: **Python** (3.14 and Node 25 are installed; `git` present, `gh` is not).

The differentiator vs. a generic "reward = tests pass" project: we anticipate that an agent will try to *hardcode outputs, overfit the visible tests, read or overwrite the test file, or fake a pass via `sys.exit`* — and we build QA that catches each. Demonstrating that judgment is exactly what this role screens for.

## Goal / shape of the artifact

A small, clean, well-documented repo demonstrating:
1. A **Gymnasium-compatible** RL environment (`reset`/`step`) wrapping verifiable coding tasks.
2. A **sandboxed executor** + **grader** (reward = fraction of *held-out* tests passing) with an honestly-documented threat model.
3. A **reward-hacking QA harness**: a catalog of exploit "solutions" + detection checks + a report/pytest suite asserting the env stays un-gameable (reward stays low for every exploit).
4. A **rollout harness** driving a real `claude-opus-4-8` agent, reporting held-out pass rate **and** a reward-hack rate.

## Repository layout

```
llm-rl-playground/
  README.md                  # framed in the language of the RL Data role
  pyproject.toml             # deps: anthropic, gymnasium, mcp (FastMCP), pytest, (matplotlib optional)
  .env.example               # ANTHROPIC_API_KEY=...
  .gitignore
  src/playground/
    __init__.py
    tasks.py                 # Task dataclass + loader; ~6–10 small problems
    tasks_data/              # each task: spec, reference solution, public + held-out tests
    sandbox.py               # subprocess executor: timeout, rlimits, isolated tempdir, no test file on disk
    grader.py                # reward/verifier: runs held-out tests in sandbox -> scalar reward
    env.py                   # CodeEnv(gymnasium.Env): Text spaces, single-step (bandit) default,
                             #   optional max_attempts>1 agentic mode with failing-test feedback
    agents.py                # Agent base; ClaudeAgent (claude-opus-4-8, adaptive thinking); ScriptedAgent
    rollout.py               # run N episodes, collect pass rate + hack rate + reward stats
    mcp/
      __init__.py
      server.py              # FastMCP server (stdio): list_tasks / get_task / submit_solution / run_qa / grade_answer
    qa/
      exploits.py            # catalog of reward-hacking candidate solutions (fixtures)
      checks.py              # AST/static analysis + isolation checks (hack-signal detector)
      report.py              # run every exploit through the env, assert reward stays low, print report
  tests/
    test_sandbox.py
    test_grader.py
    test_env.py
    test_reward_hacking.py   # the QA harness as a pytest suite (the headline test)
    test_mcp_server.py       # smoke-test the MCP tools return the expected shapes
  scripts/
    run_qa.py                # no API key needed — proves env is not gameable
    run_rollouts.py          # needs API key — runs the Claude agent, prints metrics
    run_mcp.py               # launch the MCP server over stdio (drive the gym from Claude Desktop)
  results/                   # committed sample QA report + rollout metrics (+ optional plot)
```

## Key design decisions

- **Env = contextual bandit (single-step) by default.** `reset()` returns the task spec (+ public test signatures) as the observation; `action` is the candidate code string; `step(code)` runs the grader and returns `(obs, reward, terminated=True, truncated, info)`. This is the honest shape of RLVR. Use `gymnasium.spaces.Text` for observation/action spaces and subclass `gymnasium.Env` for real API compatibility.
- **Optional agentic mode** (`max_attempts > 1`): on a failed attempt, the next observation includes the failing held-out-test summary so the agent can revise — shows understanding of multi-step/agentic environments. Keep default at 1.
- **Reward = fraction of held-out tests passing**, with the grader returning rich `info` (per-test results, hack signals, sandbox outcome). Public tests are shown to the agent; **held-out tests are never shown and are the reward signal** — this alone defeats "overfit the visible tests."
- **Sandbox (`sandbox.py`)** — best-effort, explicitly documented as an educational sandbox, not a security boundary (naming the residual risks is itself a signal for this role):
  - Run candidate as `solution.py` in an isolated `tempfile.TemporaryDirectory`; **the tests are never written to disk** — expected I/O is generated from the reference solution at build time and injected into an in-memory runner, structurally defeating "read the test file."
  - Function-call task format: candidate defines `def solution(...)`; a runner imports it, calls it on inputs, compares to embedded expected outputs, and emits structured per-test JSON over a pipe — so a premature `sys.exit(0)` yields incomplete results = not all-passed (defeats fake-pass).
  - POSIX resource limits via `preexec_fn` + `resource.setrlimit` (`RLIMIT_CPU`, `RLIMIT_AS`), wall-clock `timeout` on `subprocess.run`, kill the process group on timeout (defeats infinite-loop/sleep exploits).
  - Network: documented off; on Linux note `unshare`, on macOS note `sandbox-exec` as the hardening path; default relies on no-network-needed tasks + AST flagging of `socket`/`urllib`.
- **Hack-signal detector (`qa/checks.py`)** via `ast`: flag `sys.exit`/`os._exit`, `open(...)`/`os.` access aimed at test paths, `import pytest`/monkeypatching, bare `except:` swallowing assertions, `socket`/network imports. Surfaced in `info["hack_signals"]`; the rollout reports a **reward-hack rate**.
- **Exploit catalog (`qa/exploits.py`)** — one candidate per known hack: (1) hardcode public-test outputs, (2) try to read the test file, (3) overwrite the assert/test, (4) `sys.exit(0)` fake-pass, (5) infinite loop vs. timeout, (6) bare-`except` swallow, (7) print expected output. `qa/report.py` + `tests/test_reward_hacking.py` run each through the real env and **assert the reward stays at/near zero** — i.e., prove environment quality.
- **ClaudeAgent**: `client.messages.create(model="claude-opus-4-8", thinking={"type":"adaptive"}, max_tokens=16000, ...)` with a system prompt instructing fenced-code output; parse the code block. Default credentials from env (`anthropic.Anthropic()`); never hardcode a key. (SDK usage confirmed against the loaded `claude-api` skill — adaptive thinking only, no `budget_tokens`/sampling params on Opus 4.8.)

## Critical files to write (in order)

1. `tasks.py` + `tasks_data/` — define the `Task` dataclass and 6–10 problems with reference solutions and public/held-out tests.
2. `sandbox.py` — the isolated subprocess executor (the riskiest piece; build and unit-test first).
3. `grader.py` — reward from held-out tests via the sandbox.
4. `qa/checks.py` + `qa/exploits.py` — the hack detector and the exploit fixtures.
5. `env.py` — the `gymnasium.Env` wrapper tying tasks + grader together.
6. `agents.py` + `rollout.py` — `ClaudeAgent`/`ScriptedAgent` and the episode loop with metrics.
7. `qa/report.py`, `tests/`, `scripts/`, `README.md`, `results/` — the headline QA proof, tests, entry points, and docs.

## Reuse / dependencies (no existing codebase — greenfield)

- **Python stdlib**: `subprocess`, `resource`, `ast`, `tempfile`, `json`, `signal` — the sandbox/grader/checks lean entirely on these.
- **`gymnasium`** for the `Env` base class and `spaces.Text`.
- **`anthropic`** SDK, used per the loaded `claude-api` skill (Python README): `anthropic.Anthropic()`, `messages.create(model="claude-opus-4-8", thinking={"type":"adaptive"})`, parse `block.type == "text"`.
- **`pytest`** for the QA suite. `matplotlib` optional (single reward/hack-rate bar chart in `results/`).

## Verification (end-to-end)

1. **Setup**: `python -m venv .venv && source .venv/bin/activate && pip install -e .` (confirm `gymnasium` + `anthropic` install on Python 3.14; fall back to `pip install gymnasium anthropic pytest` if the editable install has issues).
2. **Env not gameable (no API key needed)** — the headline proof:
   - `python scripts/run_qa.py` → prints a report showing every exploit in the catalog receives ~0 reward and is flagged by the hack detector.
   - `pytest tests/test_reward_hacking.py -v` → all exploit assertions pass (reward stays low), plus `test_sandbox.py`/`test_grader.py`/`test_env.py` green.
3. **Sanity**: feed each task's **reference solution** through the env and assert reward == 1.0 (grader correctness, covered in `test_grader.py`).
4. **Real rollouts (needs `ANTHROPIC_API_KEY`)**:
   - `cp .env.example .env` and add the key (or `export ANTHROPIC_API_KEY=...`).
   - `python scripts/run_rollouts.py --n 10` → runs the `claude-opus-4-8` agent across tasks, prints held-out **pass rate**, **reward-hack rate**, and mean reward; writes `results/rollouts.json`.
5. **MCP server**: `python scripts/run_mcp.py`, then exercise it with the MCP Inspector (`npx @modelcontextprotocol/inspector`) — call `list_tasks` / `get_task` / `submit_solution` and confirm the shapes; then add it to Claude Desktop's `claude_desktop_config.json`, restart, and ask Claude to *solve the gym* (Node 25 is installed, so the Inspector runs).
6. **Repo hygiene**: `git init`, commit; README explains the RLVR framing, the threat model, and the reward-hacking QA results, and maps the project to the RL Data responsibilities. (`gh` is not installed — if the user wants it pushed to GitHub, install `gh` or add a remote manually; do this only when the user asks.)

## Scaling the anti-cheat — design note

Catching a cheat must NOT mean "ask the model to fix itself" or hand-write a fresh test suite each time — neither scales (and conversational correction backfires: it teaches cheat-then-apologize). The scalable model this project demonstrates in miniature:

- **Fix categories at the environment level, once.** Most hacks are killed structurally for *all* tasks, not patched per task: tests never touch disk (defeats file-reading), structured per-test results (defeats fake-exit via `sys.exit`), timeouts + rlimits (defeats stalling), held-out tests (defeats memorization). One fix → whole gym, forever.
- **Generate held-out tests, don't author them.** The scalable form of "fresh tests on demand" is a generator: keep the task *spec* + *reference solution* and draw new random cases (property-based / fuzz testing) each evaluation. A real solution passes any draw; a memorized hack passes only what it saw. Implement as an optional `gen_tests()` per task — **stretch goal**; the fixed held-out set stays the default to keep core scope tiny.
- **Detect at scale automatically.** The static/AST hack-signal detector (+ optional LLM-judge reading the solution) flags suspicious runs out of many — no human eyeballing.
- **The QA harness is a *regression suite of exploit categories*.** When a genuinely new trick appears (the rare, human part), add one fixture to `qa/exploits.py`; thereafter it's checked against the entire task distribution forever. The catalog grows slowly, by category; enforcement is automated. (Frame `qa/exploits.py` + `tests/test_reward_hacking.py` explicitly as this regression suite.)
- **The model improves because hacking stops paying** (no reward / penalty / discarded trajectory), not because it's told to fix itself.

## MCP server layer (first-class)

Exposes the gym over the Model Context Protocol so any MCP host (Claude Desktop, an IDE, the MCP Inspector) can drive the environment with a standard tool interface. It is a **thin wrapper over functions that already exist** (`env.reset`/`env.step`, the grader, the QA report, the truthfulness grader) — it changes nothing about how grading or anti-cheat works; it only adds a doorway.

- **Transport:** `stdio` (local subprocess) by default — what Claude Desktop uses; Streamable HTTP is the remote alternative. Built with the official Python MCP SDK (`FastMCP`).
- **Tools exposed** (`src/playground/mcp/server.py`):
  - `list_tasks() -> [{id, title}]`
  - `get_task(task_id) -> {spec, public_tests, signature}` — public tests only; **never** the held-out set
  - `submit_solution(task_id, code) -> {reward, per_test_results, hack_signals, sandbox_outcome}`
  - `run_qa() -> {report}` — run the exploit suite on demand
  - (Track 2) `grade_answer(question, answer, sources?) -> {truthfulness, claims, citations_ok}`
- **Why it earns first-class status:** (1) claims the JD's "MCP servers" preferred qual; (2) turns the demo from "run a script" into "open Claude Desktop, say *solve the gym*, and watch Claude call `get_task` → write code → `submit_solution` → read its reward (and any `hack_signals`) live"; (3) demonstrates connector craft — tool names, descriptions, input schemas, error handling; (4) cleanly separates the environment (world) from the agent (player) behind a standard protocol — the direction modern RL/agent infra is heading (cf. the MCP-based Open Reward Standard).
- **Design care:** `get_task` must never leak held-out tests; validate inputs and return structured errors; write prescriptive tool descriptions (the model only uses tools well when they're described well).

## Track 2 — open-ended grading: how do you verify an LLM isn't lying?

**The problem.** Track 1 (code) has a *strong* verifier: tests pass or they don't. Most real user queries ("explain X", "is this advice sound", "summarize this honestly") have **no automatic ground-truth checker** — there is no unit test for "did the model lie." This is where RLVR stops and the genuinely hard part begins, and it is central to the RL Data team's remit (human-feedback tooling + graders for non-code data).

**Unifying thesis (ties the whole repo together).** A reward is only as good as its verifier. Reward hacking = exploiting the gap between the verifier and true quality. Code has a near-perfect verifier (small gap); open-ended tasks have a *weak, fallible* one (a judge), so the gap is wide and hacking is easy (sycophancy, confident BS, padding, fabricated citations). The engineering job is identical to Track 1, one level up: **make the verifier harder to fool than the policy is at fooling it** — and prove it with QA + meta-eval.

**Scalable strategies the playground demonstrates (tiny but real):**

1. **Convert "is this true" into many smaller verifiable checks.** Decompose an answer into atomic factual claims; verify each against provided sources / a small knowledge set (supported / contradicted / unsupported). Reward = fraction supported, penalize contradicted. Drags an unverifiable judgment back toward RLVR. (FActScore / SAFE-style.)
2. **Require and verify citations.** Each claim cites a source; the grader checks the source exists and actually supports the claim. Catches fabricated references.
3. **Honeypots / known-answer probes (the bridge).** Can't verify every open query, but *can* seed queries whose answers are known (or plant checkable facts), measure the lie/hallucination rate, and generalize from the measurable subset.
4. **Calibration / abstention.** Reward appropriate "I don't know"; penalize confident wrongness — removes the incentive to fabricate.
5. **Diversity + adversarial signals.** Combine cheap judges (rubric LLM-judge, self-consistency across resampled answers, self-contradiction checks); optional debate (two answers argue, judge picks — easier to judge an argument than to produce truth).
6. **Meta-eval anchors the weak verifier.** The judge is fallible, so measure judge-vs-human(gold) agreement (Cohen's κ) on a small hand-labeled set. An unvalidated grader is not trustworthy. (The meta-eval idea from the eval/grader discussion, now load-bearing.)
7. **Reward-model hacking is still the failure mode.** The judge gets gamed over training (Goodhart: verbosity, sycophancy, confident tone). So Track 2 gets its **own** reward-hacking QA — planted *confident liar*, *fabricated citation*, *vague dodge to avoid checks*, *sycophant*, *length padding* — asserting the grader resists them, mirroring Track 1.

**Honest framing.** Truthfulness verification is **not solved**; it is a *managed gap*. The playground demonstrates the management toolkit (decompose → verify → honeypot → calibrate → meta-eval → QA), not a magic checker. Saying so is the right signal for this role.

**Tiny concrete build (Track 2 module — can be phase 2, after Track 1 is solid):**
- `src/playground/truthfulness/claims.py` — decompose an answer into atomic claims (Claude call) + verify each against a small provided source set (entailment via Claude with structured output).
- `src/playground/truthfulness/grader.py` — truthfulness reward from supported/contradicted/unsupported counts + citation validity + abstention handling.
- `src/playground/truthfulness/honeypots.py` — a small set of queries with known answers / planted facts to measure lie rate.
- `src/playground/truthfulness/meta_eval.py` — grader-vs-gold agreement (κ) on a hand-labeled mini-set.
- `src/playground/qa/exploits_openended.py` + tests — confident-liar / fabricated-citation / vague-dodge / sycophant / padding fixtures; assert the grader scores them low.

This keeps the repo's spine intact (environment → grader → reward-hacking QA → meta-eval), now spanning a **strong-verifier track (code)** and a **weak-verifier track (truthfulness)** — a complete, honest miniature of how graders are built and defended across data types. (Scope note: Track 1 is the tiny, shippable core; Track 2 is the ambitious extension — build it second.)

## Appendix: `README.md` draft

````markdown
# llm-rl-playground

A small **RL playground for LLMs**: build reward environments, graders, and the
**reward-hacking QA** that proves they can't be gamed — across two tracks.
**Track 1 — code:** a strong, verifiable reward (RLVR). **Track 2 — truthfulness:**
grading open-ended answers and catching lies, where there is *no* ground-truth checker.

> No model training, no GPU. This builds the *environment, grader, and anti-cheat QA* —
> the slice an RL data/environments team owns. Training the model is a separate concern.

## The idea in one breath

To make an AI better at a task, you score its answers and let it chase a high score. But
models optimize *literally* — give them a loophole and they take it instead of doing the
work. That's **reward hacking** (e.g. hardcoding the answers to the tests it can see). A
useful RL environment isn't just "reward = tests pass" — it's a grader that is **hard to
fool**, plus the QA that demonstrates it.

## For reviewers (start here)

This repo is a working miniature of the **RL Data** workflow: build the execution
environment, write the grader, and **prove the grader can't be reward-hacked** — for both
verifiable (code) and non-verifiable (truthfulness) tasks.

**60-second tour:**
1. `python scripts/run_qa.py` — every cheat attempt scores ~0 (the anti-reward-hacking proof).
2. Skim `src/playground/sandbox.py` (isolated execution), `grader.py` (held-out-test reward),
   and `qa/exploits.py` + `qa/checks.py` (exploit catalog + detector).
3. `python scripts/run_rollouts.py --n 10` — a real Claude agent graded end-to-end.

**Where each job responsibility shows up:**

| RL Data responsibility | In this repo |
|---|---|
| "the execution environments RL tasks run in" | `env.py` (Gymnasium) + `sandbox.py` |
| "prompts, evals, and **graders**" | `grader.py`, `truthfulness/grader.py`, `rollout.py` |
| "QA frameworks to catch **reward hacking**" | `qa/` + `tests/test_reward_hacking.py` |
| "**sandboxing**" execution environments | `sandbox.py` (timeouts, rlimits, no-test-on-disk) |
| "third-party tool/API connector (**MCP servers**)" | `src/playground/mcp/server.py` |
| RL on LLMs / reward design | RLVR code track + truthfulness track |

Everything below is detail.

## What's inside

- **`CodeEnv`** — a Gymnasium-compatible environment. `reset()` hands the agent a coding
  spec; the action is a code string; `step(code)` runs it and returns a reward.
- **Sandboxed grader** — runs candidate code in an isolated subprocess and scores it
  against **held-out** tests the agent never sees. Reward = fraction passing.
- **Reward-hacking QA harness** — a catalog of cheat attempts + automatic detectors + a
  regression suite asserting every cheat earns ~0 reward.
- **Agents** — a real `claude-opus-4-8` agent, plus scripted agents for the QA fixtures.
- **MCP server** — exposes the gym's tools (`list_tasks` / `get_task` / `submit_solution` /
  `run_qa`) over MCP, so you can drive it live from Claude Desktop.

## Requirements

- **Python 3.11+** (developed on 3.14)
- **An Anthropic API key** — only for the live-agent demos (steps 3–4). **Steps 1–2 need no key.**
- **Node.js** — optional, only for the MCP Inspector in step 4b.

## Setup

```bash
git clone <your-repo-url>
cd llm-rl-playground
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .

# Only needed for the live Claude agent (steps 3–4):
cp .env.example .env          # then paste your key into it,  — OR —
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run it — four things, least-setup first

**1 · Prove the environment can't be cheated** — *no API key, ~5s. This is the headline.*
```bash
python scripts/run_qa.py
```
→ a table where every planted cheat (hardcoding, reading the test file, fake-exit, infinite
loop, …) scores ~0, ending in `Environment is NOT gameable. ✅`

**2 · Run the full test suite** — *no API key.*
```bash
pytest -v
```
→ all green: sandbox, grader, environment, and the reward-hacking regression suite.

**3 · Let a real Claude agent take the exam** — *needs `ANTHROPIC_API_KEY`.*
```bash
python scripts/run_rollouts.py --n 10
```
→ per-episode rewards + a summary (**held-out pass rate** and **reward-hack rate**); written
to `results/rollouts.json`.

**4 · Drive it live from an MCP host** — *the interactive demo.*
```bash
python scripts/run_mcp.py          # starts the MCP server over stdio
```
- **4a — Claude Desktop:** add the config snippet (see "Drive it from Claude Desktop" below)
  to `claude_desktop_config.json`, restart, then ask Claude *"solve the tasks in the gym."*
  Watch it call `get_task` → write code → `submit_solution` → read its reward live.
- **4b — MCP Inspector (no Desktop needed):**
  `npx @modelcontextprotocol/inspector python scripts/run_mcp.py`, then call
  `list_tasks` / `get_task` / `submit_solution` by hand.

> **Fastest path for a reviewer:** steps 1 and 2 require zero credentials and finish in
> seconds — they alone show the environment, grader, and anti-reward-hacking QA working.

## How it defends against reward hacking

| Cheat attempt | Why it fails |
|---|---|
| Hardcode the visible-test answers | Scored on **held-out** tests it never saw |
| Read the hidden test file off disk | Tests are **never written to disk** |
| Overwrite the assert / monkeypatch pytest | Tests run from a separate, structured runner |
| `sys.exit(0)` to fake a pass | Pass count comes from **per-test results**, not exit code |
| Infinite loop to stall the grader | Wall-clock **timeout** + resource limits |
| Bare `except: pass` to swallow failures | Flagged by the **static (AST) detector** |

## Verifying open-ended answers (the don't-lie track)

Code is the easy case — tests give a near-perfect verifier. Most user queries ("explain X",
"is this true?") have **no automatic checker**. Track 2 tackles the hard case: grading
truthfulness and catching lies, by **turning one unverifiable judgment into many smaller
verifiable ones**:

- **Decompose → verify claims.** Split an answer into atomic factual claims; check each
  against provided sources (supported / contradicted / unsupported). Reward = fraction
  supported. Fabricated **citations** are checked and caught.
- **Honeypots.** Seed queries whose answers are known, measure the lie/hallucination rate,
  and generalize from the measurable subset.
- **Calibration.** Reward honest "I don't know"; penalize confident wrongness.
- **Meta-eval.** The judge is fallible, so validate it: measure grader-vs-human agreement
  (Cohen's κ) on a small gold set. An unvalidated grader isn't trustworthy.
- **Its own reward-hacking QA.** Planted *confident liar*, *fabricated citation*, *vague
  dodge*, *sycophant*, and *length-padding* attempts — the grader must score them low.

Honest framing: truthfulness verification is **not solved** — it's a *managed gap*. This
track demonstrates the toolkit (decompose → verify → honeypot → calibrate → meta-eval →
QA), not a magic lie detector.

Unifying idea across both tracks: **a reward is only as good as its verifier, and the whole
job is making verifiers harder to fool than the policy is at fooling them.**

## How it scales (the part that matters)

Catching a cheat is the easy half. The interesting half is doing it *without* a human
writing a new test suite every time:

1. **Fix categories, not instances — at the environment level.** "Read the test file" is
   killed by never putting tests on disk: one fix protecting *every* task, forever. Most
   hacks die structurally across the whole gym, not per problem.
2. **Generate tests, don't author them.** Fresh held-out cases come from a generator
   (property-based / fuzz testing off the spec + reference solution), so "new tests on
   demand" scales with compute, not human effort.
3. **Detect automatically.** The AST detector (+ optional LLM-judge) flags suspicious runs
   at scale — no eyeballing.
4. **Grow a regression suite of exploit *categories*.** A genuinely new trick is added once
   to `qa/exploits.py` and then checked against the entire task distribution forever. The
   human only handles the rare "never seen this before."

The model improves because hacking **stops paying** (no reward / penalty / discarded
trajectory) — not because we ask it to fix itself.

## Drive it from Claude Desktop (MCP)

The gym is also an **MCP server**, so any MCP host can run it. Point Claude Desktop at it:

```jsonc
// claude_desktop_config.json
{
  "mcpServers": {
    "llm-rl-playground": { "command": "python", "args": ["scripts/run_mcp.py"] }
  }
}
```

Then ask Claude *"solve the tasks in the gym."* You'll watch it call `get_task`, write code,
call `submit_solution`, read its reward — and if it ever tries a trick, see the `hack_signals`
come back. Inspect the tools directly with `npx @modelcontextprotocol/inspector`.

## Threat model (honest limits)

The sandbox is a *best-effort educational* boundary, not a security product: process
isolation, resource limits, timeouts, and a no-test-on-disk design. It does **not** provide
kernel-level isolation; for untrusted code at scale you'd reach for containers / gVisor /
network namespaces. Naming this gap is deliberate.

## Why this exists

A portfolio piece demonstrating the core primitives of RL-data work: verifiable
environments, graders, sandboxed execution, and reward-hacking QA.
````
