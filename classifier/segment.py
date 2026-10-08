"""Clause segmentation: sentence split + conjunction heuristics (Phase 2).

Deliberately not a dependency parser. Handles:
  - multi-sentence prompts (implicit split on sentence boundaries)
  - coordinating connectors: "and then", "also", "additionally", ";"
Protects against false splits where 'and' appears inside quotes/strings.
"""
from __future__ import annotations
import re

_CONNECTORS = re.compile(
    r"(?:,\s*|\s+)(?:and then|and also|also|additionally|then|besides|"
    r"next|after that|furthermore|moreover)(?:\s+)", re.IGNORECASE)

# Plain "and" is only a task boundary when it introduces another explicit
# imperative. This handles "write a story and extract its lessons" without
# breaking noun phrases such as "research and development".
_PLAIN_AND_TASK = re.compile(
    r"\s+and\s+(?=(?:analyze|build|compose|create|describe|draft|explain|extract|"
    r"find|fix|generate|give|identify|implement|list|provide|return|show|suggest|"
    r"summarize|tell|write)\b)",
    re.IGNORECASE,
)

_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")

# An 'and' inside quotes or code fences must not trigger a split.
_PROTECTED = re.compile(r'["\'].*?["\']|`[^`]*`|```.*?```', re.DOTALL)

def segment(prompt: str) -> list[str]:
    if not prompt.strip():
        return []
    # mask protected regions so connectors inside them are invisible
    # rebuild the string by inserting tokens at match spans (avoids
    # masked.replace() corrupting duplicate quoted strings).
    masked_parts: list[str] = []
    keep: dict[str, str] = {}
    prev_end = 0
    for i, m in enumerate(_PROTECTED.finditer(prompt)):
        tok = f"\x00{i}\x00"
        keep[tok] = m.group(0)
        masked_parts.append(prompt[prev_end:m.start()])
        masked_parts.append(tok)
        prev_end = m.end()
    masked_parts.append(prompt[prev_end:])
    masked = "".join(masked_parts)

    pieces: list[str] = []
    for sent in _SENT_SPLIT.split(masked):
        parts = []
        for explicit_part in _CONNECTORS.split(sent):
            parts.extend(_PLAIN_AND_TASK.split(explicit_part))
        parts = [p.strip() for p in parts if p.strip()]
        pieces.extend(parts if parts else [sent.strip()])

    out = []
    for p in pieces:
        for tok, orig in keep.items():
            p = p.replace(tok, orig, 1)  # replace at most once per token
        out.append(p)
    return out
