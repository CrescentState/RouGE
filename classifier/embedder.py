"""Embedding backend.

Uses sentence-transformers (all-MiniLM-L6-v2) when available (GPU box,
per Phase-2 plan). Falls back to a deterministic lexical hashing embedder
so the module runs and tests anywhere (e.g. CPU-only environments).
"""
from __future__ import annotations
import hashlib
import math
import re

DIM = 256
_MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MINILM_DIM = 384

_MODEL = None
_MODEL_ATTEMPTED = False


def _get_model():
    """Lazily load MiniLM once; return None on CPU-only boxes."""
    global _MODEL, _MODEL_ATTEMPTED
    if not _MODEL_ATTEMPTED:
        _MODEL_ATTEMPTED = True
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer(_MINILM_NAME)
        except (ImportError, OSError):
            _MODEL = None
    return _MODEL


def get_backend() -> str:
    """Return the active embedding backend: 'minilm' or 'hash-fallback'."""
    return "minilm" if _get_model() is not None else "hash-fallback"


def get_dim() -> int:
    return _MINILM_DIM if get_backend() == "minilm" else DIM


def embed(text: str) -> list[float]:
    model = _get_model()
    if model is not None:
        v = model.encode(text, normalize_embeddings=True)
        return [float(x) for x in v]
    return _hash_embed(text)

def _hash_embed(text: str) -> list[float]:
    v = [0.0] * DIM
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM
        v[h] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))