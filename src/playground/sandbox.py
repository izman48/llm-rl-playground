"""Best-effort sandboxed execution of candidate solutions.

This is an *educational* sandbox, not a security boundary. It uses process
isolation, a wall-clock timeout, and best-effort POSIX resource limits (CPU time
always; address-space/memory only where the OS enforces RLIMIT_AS — not macOS) —
and, most importantly, a design where **the expected outputs never enter the
sandbox**. The subprocess receives only the function arguments; it must actually
compute the answers, because they are not present anywhere on disk or in the
environment. That single property structurally defeats the most common reward
hacks (reading the answer key, hardcoding hidden answers).

Result integrity: the runner emits results on a stdout line prefixed with a
per-run nonce that the candidate cannot guess. The runner reads the whole job
from stdin *before* importing the candidate (so candidate code cannot intercept
the nonce or inputs) and runs under ``-E -s`` (no PYTHON* env vars, no user
site). Anything the candidate prints itself lacks the nonce and is ignored, so
it cannot forge a "pass".

Residual risk (documented on purpose): no kernel-level isolation. For untrusted
code at scale you would use containers / gVisor / network namespaces.
"""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

# Runner harness written into the sandbox dir. It reads the job from stdin
# (consuming it fully *before* importing the candidate so candidate code cannot
# read the nonce/inputs), runs each case, and writes a nonce-marked results line.
_RUNNER = r'''
import json, sys

_job = json.loads(sys.stdin.read())          # consume stdin before candidate import
_nonce = _job["nonce"]
_entry = _job["entry_point"]
_inputs = _job["inputs"]

_results = []
try:
    import solution
    _fn = getattr(solution, _entry, None)
    if not callable(_fn):
        _results = [{"error": "missing_entry_point"} for _ in _inputs]
    else:
        for _args in _inputs:
            try:
                _val = _fn(*_args)
                json.dumps(_val)              # must be JSON-serializable
                _results.append({"ok": True, "value": _val})
            except SystemExit:
                _results.append({"error": "system_exit"})
            except BaseException as e:         # report any per-case failure
                _results.append({"error": type(e).__name__})
except BaseException as e:                      # import-time failure
    _results = [{"error": "import_error:" + type(e).__name__} for _ in _inputs]

sys.stdout.write("\n" + _nonce + json.dumps(_results) + "\n")
sys.stdout.flush()
'''


@dataclass
class RunResult:
    status: str  # "ok" | "timeout" | "no_output" | "error"
    outputs: list[dict[str, Any]] | None
    stderr: str


def _limit_resources(mem_mb: int, cpu_s: int):
    def _apply() -> None:  # runs in the child before exec (POSIX only)
        try:
            import resource

            mem = mem_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            except (ValueError, OSError):
                pass  # RLIMIT_AS unreliable on some platforms (e.g. macOS)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        except Exception:
            pass

    return _apply


def run_candidate(
    solution_code: str,
    entry_point: str,
    inputs: list[list[Any]],
    timeout: float = 5.0,
    mem_mb: int = 512,
    cpu_s: int = 10,
) -> RunResult:
    """Run ``entry_point(*case)`` for each case in an isolated subprocess.

    ``inputs`` are arguments only; expected outputs are never provided.
    """
    nonce = secrets.token_hex(16)
    with tempfile.TemporaryDirectory(prefix="codegym_") as tmp:
        with open(os.path.join(tmp, "solution.py"), "w") as f:
            f.write(solution_code)
        with open(os.path.join(tmp, "runner.py"), "w") as f:
            f.write(_RUNNER)

        job = json.dumps({"nonce": nonce, "entry_point": entry_point, "inputs": inputs})
        try:
            proc = subprocess.run(
                # -E ignore PYTHON* env vars, -s no user site. (NOT -I/-P: those
                # would drop the script dir from sys.path and break `import solution`.)
                [sys.executable, "-E", "-s", "runner.py"],
                cwd=tmp,
                input=job,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limit_resources(mem_mb, cpu_s) if os.name == "posix" else None,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            return RunResult("timeout", None, "wall-clock timeout")

        for line in proc.stdout.splitlines():
            if line.startswith(nonce):
                try:
                    return RunResult("ok", json.loads(line[len(nonce):]), proc.stderr)
                except json.JSONDecodeError:
                    return RunResult("error", None, proc.stderr)
        return RunResult("no_output", None, (proc.stderr or proc.stdout)[-2000:])
