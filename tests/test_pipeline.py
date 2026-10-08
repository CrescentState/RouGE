from __future__ import annotations

import importlib

import pytest
from pydantic import ValidationError

import pipeline
from classifier.contracts import ClassificationResult, Clause
from DAG.dag_models import DagNode, DagValidationError, validate_dag_response
from DAG.decompose import (
    DecompNode,
    GenerationResult,
    TokenCountResult,
    decompose,
    needs_further_decomposition,
)

decompose_module = importlib.import_module("DAG.decompose")


def classification(label: str) -> ClassificationResult:
    entropy = "low" if label == "simple" else "high"
    return ClassificationResult(
        prompt_id="test",
        clauses=[
            Clause(
                id=1,
                text="test prompt",
                task_type="summarization" if label == "simple" else "code_generation",
                confidence=0.9,
                margin=0.5,
                token_count=3,
                base_entropy=entropy,
            )
        ],
        aggregate_label=label,
        segmentation_method="sentence_split_conjunction",
    )


def test_dag_uses_classifier_labels():
    simple = DagNode(node_id="n1", text="summarize", intent_type="simple", token_length=3)
    complex_node = DagNode(node_id="n2", text="write code", intent_type="complex", token_length=3)

    assert not needs_further_decomposition(simple)
    assert needs_further_decomposition(complex_node)
    with pytest.raises(ValidationError):
        DagNode(node_id="old", text="legacy", intent_type="single_intent", token_length=1)


def test_atomic_complex_node_is_not_decomposed(monkeypatch):
    node = DagNode(
        node_id="story",
        text="Write a complete story about discipline",
        intent_type="complex",
        token_length=8,
        atomic=True,
    )
    monkeypatch.setattr(
        decompose_module,
        "call_int4_engine",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("atomic tasks must not be decomposed")
        ),
    )

    result = decompose(node, 1, node.text, pipeline_id="atomic")

    assert result.resolved_via == "leaf"
    assert result.atomic is True


def test_dag_rejects_more_than_four_planned_nodes():
    payload = {
        "status": 200,
        "model": "test",
        "clause_id": "root",
        "dag": {
            "nodes": [
                {
                    "node_id": f"n{index}",
                    "text": f"task {index}",
                    "intent_type": "simple",
                    "token_length": 2,
                    "atomic": True,
                }
                for index in range(5)
            ],
            "edges": [],
        },
    }

    with pytest.raises(DagValidationError, match="schema validation failed"):
        validate_dag_response(payload)


def test_single_leaf_is_returned_without_synthesis(monkeypatch):
    monkeypatch.setattr(pipeline, "classify", lambda *args, **kwargs: classification("simple"))
    monkeypatch.setattr(
        pipeline,
        "decompose",
        lambda *args, **kwargs: DecompNode(
            node_id="root",
            text="test prompt",
            intent_type="simple",
            token_length=3,
            depth=0,
            resolved_via="leaf",
        ),
    )
    calls = []

    def fake_int8(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return GenerationResult(response="leaf answer", input_tokens=10, generated_tokens=3)

    monkeypatch.setattr(pipeline, "generate_int8", fake_int8)

    result = pipeline.run_pipeline("test prompt", pipeline_id="run1")

    assert result.final_answer == "leaf answer"
    assert len(calls) == 1
    assert calls[0][1]["purpose"] == "leaf"


def test_multiple_leaves_are_synthesized_in_order(monkeypatch):
    monkeypatch.setattr(
        pipeline, "classify", lambda *args, **kwargs: classification("mixed_intent")
    )
    tree = DecompNode(
        node_id="root",
        text="test prompt",
        intent_type="mixed_intent",
        token_length=3,
        depth=0,
        resolved_via="decomposed",
        children=[
            DecompNode("root_n1", "first task", "simple", 2, 1, resolved_via="leaf"),
            DecompNode("root_n2", "second task", "simple", 2, 1, resolved_via="leaf"),
        ],
    )
    monkeypatch.setattr(pipeline, "decompose", lambda *args, **kwargs: tree)
    calls = []

    def fake_int8(prompt, **kwargs):
        calls.append((prompt, kwargs))
        if kwargs["purpose"] == "synthesis":
            return GenerationResult(response="combined answer")
        return GenerationResult(response=f"answer for {prompt}")

    monkeypatch.setattr(pipeline, "generate_int8", fake_int8)
    monkeypatch.setattr(
        pipeline,
        "count_int8_tokens",
        lambda *args, **kwargs: TokenCountResult(100, 2048, 128, 1820, 512, True),
    )

    result = pipeline.run_pipeline("test prompt", pipeline_id="run2")

    assert result.final_answer == "combined answer"
    assert [call[1]["purpose"] for call in calls] == ["leaf", "leaf", "synthesis"]
    assert calls[2][0].index("first task") < calls[2][0].index("second task")


def test_dependency_output_is_given_to_downstream_task(monkeypatch):
    prompt = (
        "Give me a story on discipline and extract key points on it and how to "
        "implement it in our daily lives"
    )
    monkeypatch.setattr(
        pipeline, "classify", lambda *args, **kwargs: classification("mixed_intent")
    )
    tree = DecompNode(
        node_id="root",
        text=prompt,
        intent_type="mixed_intent",
        token_length=24,
        depth=0,
        resolved_via="decomposed",
        children=[
            DecompNode(
                "root_n1",
                "Write a complete short story illustrating discipline",
                "complex",
                10,
                1,
                atomic=True,
                resolved_via="leaf",
            ),
            DecompNode(
                "root_n2",
                "Extract the key lessons from the generated story",
                "simple",
                9,
                1,
                atomic=True,
                dependencies=["root_n1"],
                resolved_via="leaf",
            ),
            DecompNode(
                "root_n3",
                "Turn those lessons into practical daily actions",
                "simple",
                8,
                1,
                atomic=True,
                dependencies=["root_n1", "root_n2"],
                resolved_via="leaf",
            ),
        ],
    )
    monkeypatch.setattr(pipeline, "decompose", lambda *args, **kwargs: tree)
    leaf_prompts = []

    def fake_generate(generation_prompt, **kwargs):
        if kwargs["purpose"] == "leaf":
            leaf_prompts.append(generation_prompt)
            if len(leaf_prompts) == 1:
                return GenerationResult(response="Maya practiced every dawn until she succeeded.")
            if len(leaf_prompts) == 2:
                assert "Maya practiced every dawn" in generation_prompt
                return GenerationResult(response="Consistency matters more than motivation.")
            assert "Consistency matters more than motivation" in generation_prompt
            return GenerationResult(response="Choose one habit and practice it daily.")
        assert kwargs["purpose"] == "synthesis"
        assert "Maya practiced every dawn" in generation_prompt
        assert kwargs["max_tokens"] == 1024
        assert "stories" in kwargs["system_prompt"]
        return GenerationResult(response="Story\n...\nKey points\n...\nDaily actions\n...")

    monkeypatch.setattr(pipeline, "generate_int8", fake_generate)
    monkeypatch.setattr(
        pipeline,
        "count_int8_tokens",
        lambda *args, **kwargs: TokenCountResult(300, 4096, 128, 3668, 1024, True),
    )

    result = pipeline.run_pipeline(prompt, pipeline_id="discipline")

    assert len(result.task_results) == 3
    assert len(leaf_prompts) == 3
    assert result.final_answer.startswith("Story")


def test_fallback_response_is_reused(monkeypatch):
    monkeypatch.setattr(pipeline, "classify", lambda *args, **kwargs: classification("complex"))
    tree = DecompNode(
        node_id="root",
        text="test prompt",
        intent_type="complex",
        token_length=3,
        depth=0,
        resolved_via="int8_fallback",
        response="already answered",
        failure_reason="invalid DAG",
    )
    monkeypatch.setattr(pipeline, "decompose", lambda *args, **kwargs: tree)

    def unexpected_call(*args, **kwargs):
        raise AssertionError("the stored fallback response must not be executed again")

    monkeypatch.setattr(pipeline, "generate_int8", unexpected_call)

    result = pipeline.run_pipeline("test prompt", pipeline_id="run3")

    assert result.final_answer == "already answered"
    assert result.task_results[0].response == "already answered"
    assert "invalid DAG" in result.warnings[0]


def test_missing_fallback_response_is_a_pipeline_error(monkeypatch):
    monkeypatch.setattr(pipeline, "classify", lambda *args, **kwargs: classification("complex"))
    monkeypatch.setattr(
        pipeline,
        "decompose",
        lambda *args, **kwargs: DecompNode(
            node_id="root",
            text="test prompt",
            intent_type="complex",
            token_length=3,
            depth=0,
            resolved_via="int8_fallback",
            failure_reason="INT4 and INT8 unavailable",
        ),
    )

    with pytest.raises(pipeline.PipelineExecutionError, match="INT4 and INT8 unavailable"):
        pipeline.run_pipeline("test prompt", pipeline_id="run4")


def test_decompose_stores_fallback_response(monkeypatch):
    monkeypatch.setattr(decompose_module, "has_vram_headroom", lambda *args: True)
    monkeypatch.setattr(
        decompose_module,
        "call_int4_engine",
        lambda *args, **kwargs: {"not": "a valid DAG response"},
    )
    fallback_calls = []

    def fake_fallback(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return GenerationResult(response="fallback answer")

    monkeypatch.setattr(decompose_module, "generate_int8", fake_fallback)
    root = DagNode(
        node_id="root",
        text="write a program",
        intent_type="complex",
        token_length=4,
    )

    result = decompose(root, 0, "write a program", pipeline_id="run5")

    assert result.resolved_via == "int8_fallback"
    assert result.response == "fallback answer"
    assert result.failure_reason
    assert len(fallback_calls) == 1
    assert fallback_calls[0][1]["system_prompt"] == decompose_module.FALLBACK_SYSTEM_PROMPT
    assert fallback_calls[0][1]["max_tokens"] == 1024


def test_planner_schema_example_has_known_edge_endpoints():
    prompt = decompose_module.INT4_DAG_SYSTEM_PROMPT

    assert '"node_id": "n0"' in prompt
    assert '"node_id": "n1"' in prompt
    assert '"from": "n0", "to": "n1"' in prompt


def test_decompose_preserves_topological_dependencies(monkeypatch):
    monkeypatch.setattr(decompose_module, "has_vram_headroom", lambda *args: True)
    monkeypatch.setattr(
        decompose_module,
        "call_int4_engine",
        lambda *args, **kwargs: {
            "status": 200,
            "model": "test",
            "clause_id": "root",
            "dag": {
                "nodes": [
                    {
                        "node_id": "story",
                        "text": "Write a short story about discipline",
                        "intent_type": "complex",
                        "token_length": 8,
                        "atomic": False,
                    },
                    {
                        "node_id": "lessons",
                        "text": "Extract lessons from the generated story",
                        "intent_type": "simple",
                        "token_length": 7,
                        "atomic": True,
                    },
                ],
                "edges": [{"from": "story", "to": "lessons"}],
            },
        },
    )
    monkeypatch.setattr(
        decompose_module,
        "classify",
        lambda *args, **kwargs: classification("complex"),
    )
    root = DagNode(
        node_id="root",
        text="Write a story and extract its lessons",
        intent_type="mixed_intent",
        token_length=9,
    )

    result = decompose(root, 0, root.text, pipeline_id="dependencies")

    assert [child.node_id for child in result.children] == [
        "root_story",
        "root_lessons",
    ]
    assert result.children[0].dependencies == []
    assert result.children[1].dependencies == ["root_story"]
    assert all(child.resolved_via == "leaf" for child in result.children)
    assert result.children[0].atomic is True


def test_oversized_synthesis_uses_hierarchical_reduction(monkeypatch):
    monkeypatch.setattr(
        pipeline, "classify", lambda *args, **kwargs: classification("mixed_intent")
    )
    tree = DecompNode(
        node_id="root",
        text="test prompt",
        intent_type="mixed_intent",
        token_length=3,
        depth=0,
        resolved_via="decomposed",
        children=[
            DecompNode("root_n1", "first task", "simple", 2, 1, resolved_via="leaf"),
            DecompNode("root_n2", "second task", "simple", 2, 1, resolved_via="leaf"),
        ],
    )
    monkeypatch.setattr(pipeline, "decompose", lambda *args, **kwargs: tree)
    calls = []

    def fake_generate(prompt, **kwargs):
        calls.append(kwargs["purpose"])
        if kwargs["purpose"] == "reduction":
            return GenerationResult(response="short summary", generated_tokens=20)
        if kwargs["purpose"] == "synthesis":
            return GenerationResult(response="bounded final answer", generated_tokens=30)
        return GenerationResult(response=f"long answer for {prompt}", generated_tokens=500)

    def fake_count(prompt, **kwargs):
        is_original_only = '"ordered_task_results": []' in prompt
        is_reduced = "short summary" in prompt
        is_intermediate = kwargs["system_prompt"] == pipeline.REDUCTION_SYSTEM_PROMPT
        fits = is_original_only or is_reduced or is_intermediate
        return TokenCountResult(
            input_tokens=500 if fits else 1800,
            context_size=2048,
            safety_margin_tokens=128,
            available_output_tokens=1420 if fits else 120,
            reserved_output_tokens=kwargs["reserved_output_tokens"],
            fits=fits,
        )

    monkeypatch.setattr(pipeline, "generate_int8", fake_generate)
    monkeypatch.setattr(pipeline, "count_int8_tokens", fake_count)

    result = pipeline.run_pipeline("test prompt", pipeline_id="run6")

    assert result.final_answer == "bounded final answer"
    assert result.synthesis_rounds == 1
    assert calls == ["leaf", "leaf", "reduction", "synthesis"]
    assert "hierarchical reduction" in result.warnings[0]


def test_leaf_limit_bypasses_individual_execution(monkeypatch):
    monkeypatch.setattr(
        pipeline, "classify", lambda *args, **kwargs: classification("mixed_intent")
    )
    tree = DecompNode(
        node_id="root",
        text="test prompt",
        intent_type="mixed_intent",
        token_length=3,
        depth=0,
        resolved_via="decomposed",
        children=[
            DecompNode(
                f"root_n{i}", f"task {i}", "simple", 2, 1, resolved_via="leaf"
            )
            for i in range(pipeline.MAX_EXECUTABLE_LEAVES + 1)
        ],
    )
    monkeypatch.setattr(pipeline, "decompose", lambda *args, **kwargs: tree)
    calls = []

    def fake_generate(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return GenerationResult(response="direct bounded answer")

    monkeypatch.setattr(pipeline, "generate_int8", fake_generate)

    result = pipeline.run_pipeline("test prompt", pipeline_id="run7")

    assert result.final_answer == "direct bounded answer"
    assert len(calls) == 1
    assert calls[0][1]["purpose"] == "leaf_limit_fallback"
    assert f"{pipeline.MAX_EXECUTABLE_LEAVES + 1} leaves" in result.warnings[0]
    assert f"{pipeline.MAX_EXECUTABLE_LEAVES}-leaf limit" in result.warnings[0]


def test_synthesis_rejects_original_prompt_that_cannot_fit(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "count_int8_tokens",
        lambda *args, **kwargs: TokenCountResult(1800, 2048, 128, 120, 512, False),
    )

    with pytest.raises(
        pipeline.PipelineExecutionError,
        match="original prompt cannot fit",
    ):
        pipeline._synthesize(
            "oversized original",
            [pipeline.TaskResult("n1", "task", "answer", "leaf")],
            "run8",
        )
