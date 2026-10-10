"""
decompose.py

Phase 3: Recursive VRAM-Gated Decomposition (core contribution of DAG-C).

Wired up to Person B's Contract 3 FastAPI server:
  - `call_int4_engine`   -> real outgoing HTTP call to /generate (INT4)
  - `call_int8_engine`   -> real fallback HTTP call to /generate (INT8)
  - `get_free_vram_mb`   -> live query to /telemetry (RTX 5050 PyNVML)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional
import requests

from classifier import classify
from Setup.generation import CONTEXT_SIZE
from .dag_models import DagNode, ValidatedDag, validate_dag_response

# --------------------------------------------------------------------------- #
# Server Configuration
# --------------------------------------------------------------------------- #

API_URL = os.getenv("ROUGE_API_URL", "http://127.0.0.1:8000").rstrip("/")
_MOCK_FREE_VRAM_MB = 2048.0  # Safe fallback if server is unreachable


# --------------------------------------------------------------------------- #
# Live Telemetry - Network Query to Person B's API
# --------------------------------------------------------------------------- #

def get_free_vram_mb() -> float:
    """
    Live query to Person B's /telemetry endpoint for free VRAM on GPU 0.
    Falls back to mock value if the API endpoint cannot be reached.
    """
    try:
        response = requests.get(f"{API_URL}/telemetry", timeout=3)
        response.raise_for_status()
        data = response.json()
        return float(data.get("free_vram_mb", _MOCK_FREE_VRAM_MB))
    except Exception as e:
        print(f"[WARN] Live telemetry query to {API_URL} failed ({e}), using fallback {_MOCK_FREE_VRAM_MB}MB")
        return float(_MOCK_FREE_VRAM_MB)


def has_vram_headroom(required_mb: float, safety_margin_mb: float = 512) -> bool:
    free_mb = get_free_vram_mb()
    return free_mb - safety_margin_mb >= required_mb


# --------------------------------------------------------------------------- #
# Engine HTTP Calls (Contract 3 / Fallback)
# --------------------------------------------------------------------------- #

INT4_DAG_SYSTEM_PROMPT = (
    "You are a strict task decomposition planner. Preserve every explicit deliverable "
    "in the user's request and express only the essential work as a small directed "
    "acyclic graph (DAG). Create no more than 4 nodes. Do not create generic background "
    "tasks unless the user requested them. When one requested result depends on another, "
    "for example lessons extracted from a story that must first be written, create an "
    "edge from the producing task to the consuming task. Mark a task atomic when it can "
    "be answered directly and must not be decomposed again. Every node text must be a "
    "specific imperative instruction grounded in the user's topic. Never use placeholder "
    "phrases such as 'first task', 'requested deliverable', or 'dependent task'. For a "
    "request that creates content and then analyzes it, make the content itself the first "
    "node and make the analysis explicitly refer to the generated content.\n"
    "Respond ONLY with a JSON object matching this schema:\n"
    "{\n"
    '  "nodes": [\n'
    '    {"node_id": "n0", "text": "SPECIFIC_FIRST_TASK_FROM_USER_REQUEST", "intent_type": "simple", "token_length": 6, "atomic": true},\n'
    '    {"node_id": "n1", "text": "SPECIFIC_DEPENDENT_TASK_FROM_USER_REQUEST", "intent_type": "simple", "token_length": 7, "atomic": true}\n'
    "  ],\n"
    '  "edges": [\n'
    '    {"from": "n0", "to": "n1"}\n'
    "  ]\n"
    "}\n"
    "allowed intent_type values: 'simple', 'complex', 'mixed_intent'. Each node must "
    "contain atomic as true or false. Never repeat the parent task as a child. "
    "Do not include any conversational filler."
)

_GENERIC_TASK_MARKERS = (
    "specific_first_task_from_user_request",
    "specific_dependent_task_from_user_request",
    "requested deliverable",
    "first task",
    "dependent task",
    "next task",
)
_ATOMIC_ARTIFACT_TASK = re.compile(
    r"\b(?:write|create|compose|draft|tell|give)\b.*"
    r"\b(?:story|poem|letter|email|essay|article|dialogue|script)\b",
    re.IGNORECASE,
)

FALLBACK_SYSTEM_PROMPT = (
    "Answer every explicit part of the original user request in one coherent, "
    "self-contained response. Follow any requested format, ordering, headings, item "
    "counts, or schema exactly. Produce requested artifacts such as stories, code, "
    "letters, or plans in full, then include any requested analysis or extracted points. "
    "Begin directly and do not discuss routing, DAGs, fallback, or internal processing."
)
FALLBACK_MAX_TOKENS = 1024


def _extract_json_payload(raw_text: str) -> dict:
    """Helper to parse JSON even if the model encloses it in markdown code blocks."""
    raw_clean = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_clean, re.DOTALL)
    if match:
        raw_clean = match.group(1)
    return json.loads(raw_clean)


def call_int4_engine(node_text: str, node_id: str, request_id: Optional[str] = None) -> dict:
    """
    Calls the INT4 model over HTTP to decompose a complex/mixed node into a sub-DAG.
    """
    payload = {
        "request_id": request_id or f"decompose:{node_id}",
        "baseline_mode": "VRAM-gated",
        "prompt": node_text,
        "engine": "INT4",
        "system_prompt": INT4_DAG_SYSTEM_PROMPT,
        # Valid DAGs are short JSON; a tight cap halves the cost of the
        # planning calls that fail validation anyway.
        "max_tokens": 256,
    }

    resp = requests.post(f"{API_URL}/generate", json=payload, timeout=60)
    resp.raise_for_status()
    raw_response = resp.json()["response"]

    # Parse and wrap into the Contract 3 Int4Response schema expected by dag_models.py
    dag_dict = _extract_json_payload(raw_response)

    return {
        "status": 200,
        "model": "qwen2.5-1.5b-int4",
        "clause_id": node_id,
        "dag": dag_dict,
    }


@dataclass
class GenerationResult:
    response: str
    input_tokens: int = 0
    generated_tokens: int = 0
    max_tokens_used: int = 0
    context_size: int = CONTEXT_SIZE


@dataclass
class TokenCountResult:
    input_tokens: int
    context_size: int
    safety_margin_tokens: int
    available_output_tokens: int
    reserved_output_tokens: int
    fits: bool


def generate(
    prompt: str,
    *,
    engine: str = "INT8",
    request_id: str = "fallback_int8",
    system_prompt: str = "You are a helpful AI assistant. Answer concisely and solve the prompt fully.",
    max_tokens: int = 512,
    purpose: str = "execution",
    baseline_mode: Optional[str] = None,
) -> GenerationResult:
    """Execute one prompt on the requested engine (Contract 3 /generate).

    Routing policy lives in pipeline.py: simple work goes to INT4 (cheaper,
    faster), complex work and all synthesis stay on INT8. Both engines serve
    the same Qwen 2.5 1.5B tokenizer, so token budgets are interchangeable.
    """
    if engine not in ("INT4", "INT8"):
        raise ValueError("engine must be INT4 or INT8")
    payload = {
        "request_id": request_id,
        "baseline_mode": baseline_mode or f"VRAM-gated:{purpose}",
        "prompt": prompt,
        "engine": engine,
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
    }
    resp = requests.post(f"{API_URL}/generate", json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    return GenerationResult(
        response=data["response"],
        input_tokens=int(data.get("input_tokens", 0)),
        generated_tokens=int(data.get("generated_tokens", 0)),
        max_tokens_used=int(data.get("max_tokens_used", max_tokens)),
        context_size=int(data.get("context_size", CONTEXT_SIZE)),
    )


def generate_int8(
    prompt: str,
    *,
    request_id: str = "fallback_int8",
    system_prompt: str = "You are a helpful AI assistant. Answer concisely and solve the prompt fully.",
    max_tokens: int = 512,
    purpose: str = "execution",
    baseline_mode: Optional[str] = None,
) -> GenerationResult:
    """
    Phase 2.5 Fallback: sends the full original prompt directly to the INT8 engine.
    """
    return generate(
        prompt,
        engine="INT8",
        request_id=request_id,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        purpose=purpose,
        baseline_mode=baseline_mode,
    )


def generate_int4(
    prompt: str,
    *,
    request_id: str = "simple_int4",
    system_prompt: str = "You are a helpful AI assistant. Answer concisely and solve the prompt fully.",
    max_tokens: int = 512,
    purpose: str = "execution",
    baseline_mode: Optional[str] = None,
) -> GenerationResult:
    """Execute simple (low-entropy) work on the cheaper INT4 engine."""
    return generate(
        prompt,
        engine="INT4",
        request_id=request_id,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        purpose=purpose,
        baseline_mode=baseline_mode,
    )


def call_int8_engine(
    prompt: str,
    *,
    request_id: str = "fallback_int8",
    system_prompt: str = "You are a helpful AI assistant. Answer concisely and solve the prompt fully.",
    max_tokens: int = 512,
    purpose: str = "execution",
) -> str:
    """Backward-compatible text-only INT8 client."""
    return generate_int8(
        prompt,
        request_id=request_id,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        purpose=purpose,
    ).response


def count_int8_tokens(
    prompt: str,
    *,
    system_prompt: str,
    reserved_output_tokens: int,
    safety_margin_tokens: int = 128,
) -> TokenCountResult:
    payload = {
        "prompt": prompt,
        "engine": "INT8",
        "system_prompt": system_prompt,
        "reserved_output_tokens": reserved_output_tokens,
        "safety_margin_tokens": safety_margin_tokens,
    }
    resp = requests.post(f"{API_URL}/token-count", json=payload, timeout=15)
    resp.raise_for_status()
    return TokenCountResult(**resp.json())


# --------------------------------------------------------------------------- #
# Decomposition Tree Data Structure
# --------------------------------------------------------------------------- #

@dataclass
class DecompNode:
    node_id: str
    text: str
    intent_type: str
    token_length: int
    depth: int
    atomic: bool = False
    dependencies: List[str] = field(default_factory=list)
    children: List["DecompNode"] = field(default_factory=list)
    resolved_via: Optional[str] = None  # "decomposed" | "leaf" | "int8_fallback"
    response: Optional[str] = None
    generation: Optional[GenerationResult] = None
    failure_reason: Optional[str] = None
    # Clause-level task type from re-classification ("summarization" |
    # "extraction" | "code_generation" | "creative_writing" | "unknown" |
    # "mixed"). Drives engine routing: creative leaves need INT8 quality.
    task_type: str = "unknown"


MAX_RECURSION_DEPTH = 4
EST_VRAM_COST_PER_DECOMPOSE_MB = 350
# Total INT4 planning calls allowed per pipeline run. Advisory prompts make
# the small planner repeat the parent task at every level; without a global
# budget, recursion burns dozens of planning calls before the leaf limit
# rescues the run (measured: 12 fallback leaves, ~50s of planning).
MAX_PLANNING_ATTEMPTS = 4
# Consecutive planning failures (malformed graph, repeated parent, generic
# placeholders) before the circuit breaker routes nodes straight to leaves
# instead of burning an INT8 fallback call per node.
MAX_CONSECUTIVE_PLANNING_FAILURES = 2


def new_attempt_tracker(limit: int = MAX_PLANNING_ATTEMPTS) -> dict:
    """Mutable per-run planning budget shared across decompose() recursion."""
    return {
        "used": 0,
        "limit": limit,
        "fail_streak": 0,
        "capped": [],
        "breaker": [],
    }


def needs_further_decomposition(node: DagNode) -> bool:
    return not node.atomic and node.intent_type in {"complex", "mixed_intent"}


def decompose(
    node: DagNode,
    depth: int,
    original_full_prompt: str,
    pipeline_id: str = "adhoc",
    inherited_dependencies: Optional[List[str]] = None,
    task_type: str = "unknown",
    attempt_tracker: dict | None = None,
) -> DecompNode:
    result = DecompNode(
        node_id=node.node_id,
        text=node.text,
        intent_type=node.intent_type,
        token_length=node.token_length,
        depth=depth,
        atomic=node.atomic,
        dependencies=list(inherited_dependencies or []),
        task_type=task_type,
    )
    tracker = attempt_tracker if attempt_tracker is not None else new_attempt_tracker()

    if depth >= MAX_RECURSION_DEPTH:
        result.resolved_via = "leaf"
        return result

    if not needs_further_decomposition(node):
        result.resolved_via = "leaf"
        return result

    if not has_vram_headroom(EST_VRAM_COST_PER_DECOMPOSE_MB):
        result.resolved_via = "leaf"
        return result

    if tracker["used"] >= tracker["limit"]:
        tracker["capped"].append(node.node_id)
        result.resolved_via = "leaf"
        return result
    tracker["used"] += 1

    try:
        raw_response = call_int4_engine(
            node.text,
            node.node_id,
            request_id=f"{pipeline_id}:decompose:{node.node_id}",
        )
        validated: ValidatedDag = validate_dag_response(raw_response)
        parent_text = " ".join(node.text.lower().split())
        if any(" ".join(child.text.lower().split()) == parent_text for child in validated.nodes):
            raise ValueError("decomposition repeated the parent task")
        if any(
            marker in child.text.lower()
            for child in validated.nodes
            for marker in _GENERIC_TASK_MARKERS
        ):
            raise ValueError("decomposition produced generic placeholder task text")
        tracker["fail_streak"] = 0
    except Exception as e:
        tracker["fail_streak"] += 1
        if tracker["fail_streak"] >= MAX_CONSECUTIVE_PLANNING_FAILURES:
            # Circuit breaker: the planner is stuck (e.g. repeating the
            # parent task every level). Answer this node as a plain leaf
            # instead of burning another full INT8 fallback call.
            tracker["breaker"].append(node.node_id)
            result.resolved_via = "leaf"
            result.failure_reason = (
                f"planning circuit breaker after {tracker['fail_streak']} "
                f"consecutive failures: {e}"
            )
            return result
        print(f"[WARN] INT4 decomposition or validation failed ({e}). Triggering Phase 2.5 INT8 fallback.")
        result.resolved_via = "int8_fallback"
        result.failure_reason = str(e)
        try:
            result.generation = generate_int8(
                original_full_prompt,
                request_id=f"{pipeline_id}:fallback:{node.node_id}",
                system_prompt=FALLBACK_SYSTEM_PROMPT,
                max_tokens=FALLBACK_MAX_TOKENS,
                purpose="fallback",
            )
            result.response = result.generation.response
        except Exception as fallback_error:
            result.failure_reason = f"{e}; INT8 fallback failed: {fallback_error}"
        return result

    node_by_id = {n.node_id: n for n in validated.nodes}
    prefix = f"{node.node_id}_"
    incoming: dict[str, list[str]] = {child_id: [] for child_id in node_by_id}
    for edge in validated.edges:
        incoming[edge.to].append(f"{prefix}{edge.from_}")
    for child_id in validated.topo_order:
        child_node = node_by_id[child_id]
        child_classification = classify(
            child_node.text,
            prompt_id=f"{pipeline_id}:{node.node_id}:{child_node.node_id}",
        )
        # Dominant clause task type for engine routing downstream. Mixed
        # multi-clause children route to INT8 (safe default).
        clause_types = [
            clause.task_type for clause in child_classification.clauses
        ] or ["unknown"]
        # Content artifacts are useful leaf outputs, not planning containers.
        # Small planners otherwise tend to split a short story into redundant
        # introduction/body/conclusion nodes and lose the requested artifact.
        atomic = child_node.atomic or bool(_ATOMIC_ARTIFACT_TASK.search(child_node.text))
        child_node = child_node.model_copy(
            update={
                "node_id": f"{node.node_id}_{child_node.node_id}",
                "intent_type": child_classification.aggregate_label,
                "token_length": sum(
                    clause.token_count for clause in child_classification.clauses
                ),
                "atomic": atomic,
            }
        )
        child_result = decompose(
            child_node,
            depth + 1,
            original_full_prompt,
            pipeline_id=pipeline_id,
            inherited_dependencies=list(
                dict.fromkeys([*result.dependencies, *incoming[child_id]])
            ),
            task_type=(
                clause_types[0]
                if len(set(clause_types)) == 1
                else "mixed"
            ),
            attempt_tracker=tracker,
        )
        result.children.append(child_result)

    result.resolved_via = "decomposed"
    return result


def print_tree(node: DecompNode, indent: int = 0):
    pad = "  " * indent
    print(f"{pad}- [{node.resolved_via}] ({node.intent_type}, {node.token_length}tok) {node.text[:60]}")
    for child in node.children:
        print_tree(child, indent + 1)
