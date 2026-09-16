from .aggregate import classify  # noqa: F401
from .contracts import Clause, ClassificationResult, CONF_FLOOR, MARGIN, TaskType, Entropy, Label  # noqa: F401
from .embedder import embed, cosine, get_backend, _hash_embed  # noqa: F401
from .segment import segment  # noqa: F401
from .entropy import base_entropy_of, escalate, estimate_tokens  # noqa: F401

__all__ = [
    "classify",
    "Clause",
    "ClassificationResult",
    "CONF_FLOOR",
    "MARGIN",
    "TaskType",
    "Entropy",
    "Label",
    "embed",
    "cosine",
    "get_backend",
    "_hash_embed",
    "segment",
    "base_entropy_of",
    "escalate",
    "estimate_tokens",
]