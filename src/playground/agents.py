"""Agents that produce candidate code from a task observation.

- ``ScriptedAgent``: returns fixed code per task (defaults to reference solutions)
  — used for offline tests and as a sanity baseline; needs no API key.
- ``ClaudeAgent``: calls the Anthropic API (``claude-opus-4-8``, adaptive
  thinking) and extracts the code block from the reply.
"""
from __future__ import annotations

import re

from .tasks import all_tasks

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the first fenced code block out of a model reply (fallback: raw text)."""
    m = _CODE_BLOCK.search(text)
    return m.group(1).strip() if m else text.strip()


class Agent:
    def act(self, observation: str, info: dict) -> str:
        raise NotImplementedError


class ScriptedAgent(Agent):
    def __init__(self, solutions: dict[str, str] | None = None) -> None:
        self._sol = solutions or {t.id: t.reference for t in all_tasks()}

    def act(self, observation: str, info: dict) -> str:
        return self._sol.get(info.get("task_id", ""), "def solution(*a, **k):\n    return None\n")


class ClaudeAgent(Agent):
    """Real Claude agent. Requires ``anthropic`` and ANTHROPIC_API_KEY."""

    SYSTEM = (
        "You are a careful Python programmer. Read the problem and reply with a "
        "single self-contained Python function that solves it. Respond with ONLY "
        "one ```python code block containing the required function (plus any "
        "helpers); no prose, no explanation."
    )

    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 16_000) -> None:
        import anthropic  # imported lazily so the core stays dependency-free

        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def act(self, observation: str, info: dict) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            thinking={"type": "adaptive"},
            system=self.SYSTEM,
            messages=[{"role": "user", "content": observation}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return extract_code(text)
