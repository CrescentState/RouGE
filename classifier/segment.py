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

_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")

# An 'and' inside quotes or code fences must not trigger a split.
_PROTECTED = re.compile(r'["\'].*?["\']|`[^`]*`|```.*?```', re.DOTALL)

def segment(prompt: str) -> list[str]:
    if not prompt.strip():
        return []
    # mask protected regions so connectors inside them are invisible
    masked = prompt
    keep: dict[str, str] = {}
    for i, m in enumerate(_PROTECTED.finditer(prompt)):
        tok = f"\x00{i}\x00"
        keep[tok] = m.group(0)
        masked = masked.replace(m.group(0), tok)

    pieces: list[str] = []
    for sent in _SENT_SPLIT.split(masked):
        parts = [p.strip() for p in _CONNECTORS.split(sent) if p.strip()]
        pieces.extend(parts if parts else [sent.strip()])

    out = []
    for p in pieces:
        for tok, orig in keep.items():
            p = p.replace(tok, orig)
        out.append(p)
    return out
