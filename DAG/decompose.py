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
import re
from dataclasses import dataclass, field
from typing import List, Optional
import requests

from dag_models import DagNode, ValidatedDag, DagValidationError, validate_dag_response

# --------------------------------------------------------------------------- #
# Server Configuration
# --------------------------------------------------------------------------- #

API_URL = "http://localhost:8000"
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
    "You are a strict task decomposition planner. Given an input task, break it into "
    "atomic sub-tasks and express them as a valid directed acyclic graph (DAG).\n"
    "Respond ONLY with a JSON object matching this schema:\n"
    "{\n"
    '  "nodes": [\n'
    '    {"node_id": "n0", "text": "task text", "intent_type": "single_intent", "token_length": 5}\n'
    "  ],\n"
    '  "edges": [\n'
    '    {"from": "n0", "to": "n1"}\n'
    "  ]\n"
    "}\n"
    "allowed intent_type values: 'single_intent', 'mixed_intent', 'length_escalated'. "
    "Do not include any conversational filler."
)


def _extract_json_payload(raw_text: str) -> dict:
    """Helper to parse JSON even if the model encloses it in markdown code blocks."""
    raw_clean = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_clean, re.DOTALL)
    if match:
        raw_clean = match.group(1)
    return json.loads(raw_clean)


def call_int4_engine(node_text: str, node_id: str) -> dict:
    """
    Calls the INT4 model over HTTP to decompose a complex/mixed node into a sub-DAG.
    """
    payload = {
        "request_id": node_id,
        "baseline_mode": "VRAM-gated",
        "prompt": node_text,
        "engine": "INT4",
        "system_prompt": INT4_DAG_SYSTEM_PROMPT,
        "max_tokens": 512,
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


def call_int8_engine(full_prompt: str) -> str:
    """
    Phase 2.5 Fallback: sends the full original prompt directly to the INT8 engine.
    """
    payload = {
        "request_id": "fallback_int8",
        "baseline_mode": "VRAM-gated",
        "prompt": full_prompt,
        "engine": "INT8",
        "system_prompt": "You are a helpful AI assistant. Answer concisely and solve the prompt fully.",
        "max_tokens": 1024,
    }
    resp = requests.post(f"{API_URL}/generate", json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()["response"]


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
    children: List["DecompNode"] = field(default_factory=list)
    resolved_via: Optional[str] = None  # "decomposed" | "leaf" | "int8_fallback"


MAX_RECURSION_DEPTH = 4
LENGTH_ESCALATION_TOKEN_THRESHOLD = 30
EST_VRAM_COST_PER_DECOMPOSE_MB = 350


def needs_further_decomposition(node: DagNode) -> bool:
    if node.intent_type == "mixed_intent":
        return True
    if node.intent_type == "length_escalated":
        return True
    if node.token_length > LENGTH_ESCALATION_TOKEN_THRESHOLD:
        return True
    return False


def decompose(node: DagNode, depth: int, original_full_prompt: str) -> DecompNode:
    result = DecompNode(
        node_id=node.node_id,
        text=node.text,
        intent_type=node.intent_type,
        token_length=node.token_length,
        depth=depth,
    )

    if depth >= MAX_RECURSION_DEPTH:
        result.resolved_via = "leaf"
        return result

    if not needs_further_decomposition(node):
        result.resolved_via = "leaf"
        return result

    if not has_vram_headroom(EST_VRAM_COST_PER_DECOMPOSE_MB):
        result.resolved_via = "leaf"
        return result

    try:
        raw_response = call_int4_engine(node.text, node.node_id)
        validated: ValidatedDag = validate_dag_response(raw_response)
    except (DagValidationError, Exception) as e:
        print(f"[WARN] INT4 decomposition or validation failed ({e}). Triggering Phase 2.5 INT8 fallback.")
        call_int8_engine(original_full_prompt)
        result.resolved_via = "int8_fallback"
        return result

    node_by_id = {n.node_id: n for n in validated.nodes}
    for child_id in validated.topo_order:
        child_node = node_by_id[child_id]
        child_result = decompose(child_node, depth + 1, original_full_prompt)
        result.children.append(child_result)

    result.resolved_via = "decomposed"
    return result


def print_tree(node: DecompNode, indent: int = 0):
    pad = "  " * indent
    print(f"{pad}- [{node.resolved_via}] ({node.intent_type}, {node.token_length}tok) {node.text[:60]}")
    for child in node.children:
        print_tree(child, indent + 1)

