"""Embedding backend.

Uses sentence-transformers (all-MiniLM-L6-v2) when available (GPU box,
per Phase-2 plan). Falls back to a deterministic lexical hashing embedder
so the module runs and tests anywhere (e.g. CPU-only environments).
"""
from __future__ import annotations
import math, re
from collections import Counter

try:
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    _DIM = 384
    BACKEND = "minilm"
except Exception:  # pragma: no cover - fallback path
    _MODEL = None
    _DIM = 256
    BACKEND = "hash-fallback"

_DIMS = 256

def embed(text: str) -> list[float]:
    if _MODEL is not None:
        v = _MODEL.encode(text, normalize_embeddings=True)
        return [float(x) for x in v]
    return _hash_embed(text)

def _hash_embed(text: str) -> list[float]:
    v = [0.0] * _DIMS
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = hash(tok) % _DIMS
        v[h] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))
