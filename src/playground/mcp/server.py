"""FastMCP server exposing the gym over the Model Context Protocol (stdio).

Thin wrapper over functions that already exist (tasks, grader, QA report). It
changes nothing about how grading or anti-cheat works — it only adds a standard
doorway so any MCP host (Claude Desktop, the MCP Inspector) can drive the gym.

The ``*_impl`` functions hold the logic and are import-safe for unit tests; the
decorated tools call them, so the module can be tested without an MCP runtime.
"""
from __future__ import annotations

from typing import Any

from .. import tasks
from ..grader import grade
from ..qa.report import run_report


def list_tasks_impl() -> list[dict[str, Any]]:
    return [{"id": t.id, "title": t.title} for t in tasks.all_tasks()]


def get_task_impl(task_id: str) -> dict[str, Any]:
    t = tasks.get_task(task_id)
    return {
        "id": t.id,
        "title": t.title,
        "entry_point": t.entry_point,
        "prompt": t.prompt(),
        # PUBLIC cases only — held-out tests are never exposed.
        "public_cases": [
            {"args": list(c.args), "expected": c.expected} for c in t.public_cases
        ],
    }


def submit_solution_impl(task_id: str, code: str) -> dict[str, Any]:
    return grade(tasks.get_task(task_id), code).as_dict()


def run_qa_impl() -> dict[str, Any]:
    rows = run_report()
    return {
        "results": [
            {"exploit": name, "reward": res.reward, "hack_signals": res.hack_signals}
            for name, res in rows
        ]
    }


def build_server():  # pragma: no cover - requires the `mcp` package at runtime
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("llm-rl-playground")

    @server.tool()
    def list_tasks() -> list[dict[str, Any]]:
        """List the coding tasks available in the gym."""
        return list_tasks_impl()

    @server.tool()
    def get_task(task_id: str) -> dict[str, Any]:
        """Get a task's spec and PUBLIC example cases (held-out tests are never returned)."""
        return get_task_impl(task_id)

    @server.tool()
    def submit_solution(task_id: str, code: str) -> dict[str, Any]:
        """Run a Python solution sandboxed against held-out tests; returns reward + signals."""
        return submit_solution_impl(task_id, code)

    @server.tool()
    def run_qa() -> dict[str, Any]:
        """Run the reward-hacking QA suite; returns each exploit's reward (all should be ~0)."""
        return run_qa_impl()

    return server


def main() -> None:  # pragma: no cover
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
