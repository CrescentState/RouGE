from __future__ import annotations
import json
from typing import List, Optional
import networkx as nx
from pydantic import BaseModel, Field, ValidationError, field_validator

class DagValidationError(Exception):
    def __init__(self, reason: str, raw: Optional[str] = None):
        self.reason = reason
        self.raw = raw
        super().__init__(reason)

# --------------------------------------------------------------------------- #
# Pydantic schema (structural validation)
# --------------------------------------------------------------------------- #

class DagNode(BaseModel):
    node_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    intent_type: str  # "simple" | "complex" | "mixed_intent"
    token_length: int = Field(..., ge=0)
    atomic: bool = False

    @field_validator("intent_type")
    @classmethod
    def known_intent_type(cls, v: str) -> str:
        allowed = {"simple", "complex", "mixed_intent"}
        if v not in allowed:
            raise ValueError(f"intent_type '{v}' not in {allowed}")
        return v


class DagEdge(BaseModel):
    from_: str = Field(..., alias="from", min_length=1)
    to: str = Field(..., min_length=1)

    model_config = {"populate_by_name": True}


class DagPayload(BaseModel):
    # Small local models are much more reliable when the planner is forced to
    # produce only the essential deliverables instead of an unbounded task list.
    nodes: List[DagNode] = Field(..., min_length=1, max_length=4)
    edges: List[DagEdge] = Field(default_factory=list, max_length=12)


class Int4Response(BaseModel):
    status: int
    model: str
    clause_id: str
    latency_ms: Optional[int] = None
    dag: DagPayload


# --------------------------------------------------------------------------- #
# Result object handed back to the recursive decomposition logic
# --------------------------------------------------------------------------- #

class ValidatedDag(BaseModel):
    clause_id: str
    nodes: List[DagNode]
    edges: List[DagEdge] = Field(default_factory=list)
    topo_order: List[str]  # node_ids in a valid execution order

    model_config = {"arbitrary_types_allowed": True}


# --------------------------------------------------------------------------- #
# Core validation entry point
# --------------------------------------------------------------------------- #

def validate_dag_response(raw_response) -> ValidatedDag:
    # Step 1: get a dict, regardless of whether we were handed a string or dict.
    if isinstance(raw_response, (str, bytes)):
        try:
            payload = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError) as exc:
            raise DagValidationError(f"malformed JSON: {exc}", raw=str(raw_response)) from exc
    else:
        payload = raw_response

    # Step 2: structural / type validation via Pydantic.
    try:
        parsed = Int4Response.model_validate(payload)
    except ValidationError as exc:
        raise DagValidationError(f"schema validation failed: {exc}", raw=str(payload)) from exc

    # Step 3: build a networkx DiGraph and check it's actually a DAG.
    graph = nx.DiGraph()
    node_ids = [n.node_id for n in parsed.dag.nodes]

    if len(node_ids) != len(set(node_ids)):
        raise DagValidationError("duplicate node_id values in DAG", raw=str(payload))

    normalized_texts = [" ".join(node.text.lower().split()) for node in parsed.dag.nodes]
    if len(normalized_texts) != len(set(normalized_texts)):
        raise DagValidationError("duplicate task text values in DAG", raw=str(payload))

    graph.add_nodes_from(node_ids)

    for edge in parsed.dag.edges:
        if edge.from_ not in graph or edge.to not in graph:
            raise DagValidationError(
                f"edge references unknown node: {edge.from_} -> {edge.to}",
                raw=str(payload),
            )
        graph.add_edge(edge.from_, edge.to)

    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph, orientation="original")
        cycle_str = " -> ".join(u for u, v, *_ in cycle) + f" -> {cycle[0][0]}"
        raise DagValidationError(f"cyclic dependency detected: {cycle_str}", raw=str(payload))

    # Step 4: topological sort - this is the execution order the recursive
    # decomposition step (Phase 3) will walk over.
    topo_order = list(nx.topological_sort(graph))

    return ValidatedDag(
        clause_id=parsed.clause_id,
        nodes=parsed.dag.nodes,
        edges=parsed.dag.edges,
        topo_order=topo_order,
    )
