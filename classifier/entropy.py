"""Per-clause entropy labeling (Phase 2, plan section 4.4.2).

base_entropy: task-type prior (low = summarization/extraction,
high = code_generation, unknown treated as high/conservative).
Length-aware escalation: long inputs escalate low -> high because
processing cost and context risk grow with token count.
Thresholds are initial values; calibrate against Part B's KV/latency
measurements once engines are live (plan note, week 2).
"""
from __future__ import annotations
from .contracts import Clause, Entropy

TYPE_PRIOR = {
    "summarization": "low",
    "extraction": "low",
    "code_generation": "high",
    # A standalone creative request can be executed directly. When it is
    # combined with another task type, aggregation still produces mixed_intent.
    "creative_writing": "low",
    "unknown": "high",          # conservative fallback
}

# ~4,000 tokens: from report section 6.2 (initial, tunable)
LENGTH_ESCALATION_TOKENS = 4000

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def base_entropy_of(task_type: str) -> Entropy:
    return TYPE_PRIOR[task_type]

def escalate(base: Entropy, token_count: int) -> Entropy:
    if base == "low" and token_count >= LENGTH_ESCALATION_TOKENS:
        return "high"
    return base
