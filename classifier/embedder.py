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
        except ImportError:
            _MODEL = None
            return _MODEL
        try:
            # Prefer the local cache without touching the network: instant
            # when cached, and no minutes-long retry storm when offline.
            _MODEL = SentenceTransformer(_MINILM_NAME, local_files_only=True)
        except Exception:
            try:
                _MODEL = SentenceTransformer(_MINILM_NAME)
            except Exception:
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


def embed_many(texts: list[str]) -> list[list[float]]:
    """Embed a batch at once. A single model.encode(list) call replaces N
    per-text calls, which is what makes first-use classification slow
    (the 140-anchor prototype bank would otherwise cost 140 round trips)."""
    model = _get_model()
    if model is not None:
        vecs = model.encode(list(texts), normalize_embeddings=True)
        return [[float(x) for x in v] for v in vecs]
    return [_hash_embed(text) for text in texts]

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