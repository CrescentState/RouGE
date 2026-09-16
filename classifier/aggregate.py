"""Aggregate per-clause labels into the prompt-level label (plan 4.4.3).

Rules:
  - single clause -> its own entropy
  - all clauses same task_type -> that entropy
  - same entropy, different types -> that entropy
  - different types with different entropy -> mixed_intent
"""
from __future__ import annotations
from .contracts import ClassificationResult, Clause, Label, TaskType
from .segment import segment
from .entropy import base_entropy_of, escalate, estimate_tokens
from .embedder import embed, cosine
from .contracts import CONF_FLOOR, MARGIN
import json
import pathlib

_BANK: dict[str, list[tuple[str, list[float]]]] | None = None

def _load_bank() -> dict[str, list[tuple[str, list[float]]]]:
    global _BANK
    if _BANK is None:
        path = pathlib.Path(__file__).parent / "bank" / "prototype_bank.json"
        data = json.loads(path.read_text())
        _BANK = {t: [(ex, embed(ex)) for ex in examples] for t, examples in data.items()}
    return _BANK

def classify_clause(text: str) -> tuple[TaskType, float, float]:
    vec = embed(text)
    sims: dict[str, float] = {}
    for ttype, pairs in _load_bank().items():
        sims[ttype] = max(cosine(vec, v) for _, v in pairs)
    ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
    best_type, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if best < CONF_FLOOR or (best - second) < MARGIN:
        return "unknown", best, best - second
    return best_type, best, best - second

def classify(prompt: str, prompt_id: str = "adhoc") -> ClassificationResult:
    clauses: list[Clause] = []
    for i, text in enumerate(segment(prompt), start=1):
        ttype, conf, margin = classify_clause(text)
        toks = estimate_tokens(text)
        clauses.append(Clause(
            id=i, text=text, task_type=ttype, confidence=round(conf, 4),
            margin=round(margin, 4), token_count=toks,
            base_entropy=escalate(base_entropy_of(ttype), toks),
        ))
    return ClassificationResult(
        prompt_id=prompt_id, clauses=clauses,
        aggregate_label=_aggregate(clauses),
        segmentation_method="sentence_split_conjunction",
    )

def _aggregate(clauses: list[Clause]) -> Label:
    if not clauses:
        return "simple"
    entropies = {c.base_entropy for c in clauses}
    types = {c.task_type for c in clauses}
    if len(entropies) > 1:
        return "mixed_intent"
    # single entropy across all clauses
    if len(types) > 1:
        # different types, same entropy -> still one execution tier
        return _simple_or_complex(entropies)
    return _simple_or_complex(entropies)

def _simple_or_complex(e: set[str]) -> Label:
    return "complex" if e == {"high"} else "simple"
