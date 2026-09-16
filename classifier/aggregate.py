"""Aggregate per-clause labels into the prompt-level label (plan 4.4.3).

Rules:
  - single clause -> its own entropy
  - all clauses same task_type -> that entropy
  - same entropy, different types -> that entropy
  - different types with different entropy -> mixed_intent
"""
from __future__ import annotations
import json
import pathlib

from .contracts import ClassificationResult, Clause, Label, TaskType
from .segment import segment
from .entropy import base_entropy_of, escalate, estimate_tokens
from .embedder import embed, cosine, get_backend
from .contracts import CONF_FLOOR, MARGIN, FALLBACK_CONF_FLOOR, FALLBACK_MARGIN

_BANK: dict[str, list[tuple[str, list[float]]]] | None = None

def _load_bank() -> dict[str, list[tuple[str, list[float]]]]:
    global _BANK
    if _BANK is None:
        path = pathlib.Path(__file__).parent / "bank" / "prototype_bank.json"
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Prototype bank not found at {path}. "
                "Place classifier/bank/prototype_bank.json or configure a custom path."
            )
        if not data:
            return {}
        _BANK = {t: [(ex, embed(ex)) for ex in examples] for t, examples in data.items()}
    return _BANK

def classify_clause(text: str) -> tuple[TaskType, float, float]:
    vec = embed(text)
    sims: dict[str, float] = {}
    for ttype, pairs in _load_bank().items():
        if not pairs:
            continue
        sims[ttype] = max(cosine(vec, v) for _, v in pairs)
    if not sims:
        return "unknown", 0.0, 0.0
    ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
    best_type, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    # select thresholds per backend (classify-time, no import side-effects)
    if get_backend() == "hash-fallback":
        conf_floor, margin_thresh = FALLBACK_CONF_FLOOR, FALLBACK_MARGIN
    else:
        conf_floor, margin_thresh = CONF_FLOOR, MARGIN
    gap = best - second
    # clamp outputs to Clause contract ranges before rounding
    conf_out = round(max(0.0, min(1.0, best)), 4)
    gap_out = round(max(0.0, min(2.0, gap)), 4)
    if best < conf_floor or gap < margin_thresh:
        return "unknown", conf_out, gap_out
    return best_type, conf_out, gap_out

def classify(prompt: str, prompt_id: str = "adhoc") -> ClassificationResult:
    clauses: list[Clause] = []
    for i, text in enumerate(segment(prompt), start=1):
        ttype, conf, margin = classify_clause(text)
        toks = estimate_tokens(text)
        clauses.append(Clause(
            id=i, text=text, task_type=ttype, confidence=conf,
            margin=margin, token_count=toks,
            base_entropy=escalate(base_entropy_of(ttype), toks),
        ))
    return ClassificationResult(
        prompt_id=prompt_id, clauses=clauses,
        aggregate_label=_aggregate(clauses),
        segmentation_method="sentence_split_conjunction",
    )

def _aggregate(clauses: list[Clause]) -> Label:
    """Aggregate per-clause labels into the prompt-level label.

    Rules:
      - no clauses -> simple
      - single clause -> its own entropy
      - all clauses same task_type -> that entropy
      - different task types -> mixed_intent (conservative: any intent mix
        triggers the mixed-intent path, prompting DAG decomposition).
      - same entropy across all clauses -> _simple_or_complex(entropies)
    """
    if not clauses:
        return "simple"
    entropies = {c.base_entropy for c in clauses}
    types = {c.task_type for c in clauses}
    if len(types) > 1:
        return "mixed_intent"
    if len(entropies) > 1:
        return "mixed_intent"
    return _simple_or_complex(entropies)

def _simple_or_complex(e: set[str]) -> Label:
    return "complex" if e == {"high"} else "simple"