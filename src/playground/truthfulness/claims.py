"""Decompose a free-text answer into structured, verifiable claims using Claude.

This is the *online* part of Track 2 (needs anthropic + an API key). It converts
a natural-language answer into the `Answer`/`Claim` structure that the offline,
rule-based grader then verifies against sources. Separating decomposition from
verification is deliberate: the verifier never trusts the model's self-assessment.
"""
from __future__ import annotations

import json

from .grader import Answer, Claim, Source

_SYSTEM = (
    "You extract atomic factual claims from an answer and map each to the source "
    "that is most relevant. Return ONLY JSON: a list of objects with keys "
    "'text' (the atomic claim), 'citation' (a source id from the provided list, or "
    "'NONE' if no source is relevant), and 'asserts' (one of 'supports', "
    "'contradicts', 'unknown' describing what the answer implies about that source). "
    "Do not judge truth yourself; just structure the answer faithfully."
)


def decompose_with_claude(
    answer_text: str,
    sources: list[Source],
    model: str = "claude-opus-4-8",
    max_tokens: int = 4_000,
) -> Answer:
    import anthropic  # lazy import; core stays dependency-free

    client = anthropic.Anthropic()
    src_block = "\n".join(f"[{s.id}] {s.text}" for s in sources)
    user = f"SOURCES:\n{src_block}\n\nANSWER:\n{answer_text}\n\nReturn the JSON list."

    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    raw = json.loads(text)

    claims = tuple(
        Claim(
            text=item["text"],
            citation=item.get("citation", "NONE"),
            asserts=item.get("asserts", "unknown"),
        )
        for item in raw
    )
    return Answer(claims=claims, abstained=not claims)
