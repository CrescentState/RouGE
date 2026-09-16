"""Contract 1: classifier output schema.

This JSON is the interface consumed by the decomposer (Part C) and
scheduler (Part D). Every prompt produces exactly one of these objects.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

TaskType = Literal["summarization", "extraction", "code_generation", "unknown"]
Entropy = Literal["low", "high"]
Label = Literal["simple", "complex", "mixed_intent"]

class Clause(BaseModel):
    id: int
    text: str
    task_type: TaskType
    confidence: float = Field(ge=0.0, le=1.0)   # best cosine similarity
    margin: float = Field(ge=0.0, le=2.0)       # best - second best
    token_count: int
    base_entropy: Entropy

class ClassificationResult(BaseModel):
    prompt_id: str
    clauses: list[Clause]
    aggregate_label: Label
    segmentation_method: Literal["sentence_split_conjunction"]
    notes: Optional[str] = None

# Tuning knobs (documented defaults; calibrate against eval fixtures)
CONF_FLOOR = 0.45    # below this, clause task_type becomes "unknown"
MARGIN     = 0.08    # below best-second gap, same -> "unknown"

# Fallback thresholds for hash-fallback embedder (lower raw similarities
# from pure token-overlap embedder so conservative path isn't triggered
# on CPU-only boxes; overridden at classify-time via get_backend()).
FALLBACK_CONF_FLOOR = 0.20
FALLBACK_MARGIN = 0.01

# Empty prompt contract: zero clauses → simple label (valid per
# contract spec "exactly one object per prompt"; documented behaviour).