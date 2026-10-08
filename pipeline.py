"""End-to-end orchestration for the integrated RouGE pipeline."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field

from classifier import ClassificationResult, classify
from DAG.dag_models import DagNode
from Setup.generation import CONTEXT_SIZE
from DAG.decompose import (
    DecompNode,
    FALLBACK_MAX_TOKENS,
    FALLBACK_SYSTEM_PROMPT,
    GenerationResult,
    TokenCountResult,
    count_int8_tokens,
    decompose,
    generate_int8,
)


SYNTHESIS_SYSTEM_PROMPT = (
    "You are the final response synthesizer. Answer every explicit deliverable in the "
    "original user prompt using the ordered task results. Produce one coherent, "
    "self-contained answer. Preserve requested primary artifacts such as stories, code, "
    "letters, or plans in full; do not replace them with a description or summary. Then "
    "include requested analysis, key points, and practical actions in clearly separated "
    "sections. Before finishing, verify that no requested deliverable is missing. Do not "
    "mention task routing, model precision, DAGs, or internal processing."
)
REDUCTION_SYSTEM_PROMPT = (
    "Reduce the ordered task results for a later synthesis step. Preserve every requested "
    "deliverable, conclusion, constraint, and dependency. Never replace a requested "
    "primary artifact such as a story, code block, letter, or plan with a description of "
    "it; retain that artifact and compress explanatory material first. Do not add new "
    "information or discuss internal processing."
)
LEAF_SYSTEM_PROMPT = (
    "Execute only the current task while respecting the original user request. Use the "
    "provided dependency results when the task refers to an earlier result. Produce the "
    "actual requested artifact, not an explanation of what it would contain. Begin "
    "directly; never announce that you will do the task later. Do not mention routing, "
    "DAGs, nodes, dependency data, or internal processing."
)

LEAF_MAX_TOKENS = 512
FINAL_MAX_TOKENS = 1024
INTERMEDIATE_MAX_TOKENS = 256
SAFETY_MARGIN_TOKENS = 128
MAX_EXECUTABLE_LEAVES = 4
MAX_REDUCTION_ROUNDS = 8


class PipelineExecutionError(RuntimeError):
    """A user-facing error raised when an execution stage cannot finish."""

    def __init__(self, stage: str, reason: str):
        self.stage = stage
        self.reason = reason
        super().__init__(f"{stage} failed: {reason}")


@dataclass
class TaskResult:
    node_id: str
    instruction: str
    response: str
    resolved_via: str
    input_tokens: int = 0
    generated_tokens: int = 0
    max_tokens_used: int = 0
    context_size: int = CONTEXT_SIZE


@dataclass
class PipelineResult:
    pipeline_id: str
    classification: ClassificationResult
    tree: DecompNode
    task_results: list[TaskResult]
    final_answer: str
    warnings: list[str] = field(default_factory=list)
    synthesis_rounds: int = 0
    final_generation: GenerationResult | None = None


def _executable_nodes(node: DecompNode) -> list[DecompNode]:
    nodes: list[DecompNode] = []
    if node.resolved_via in {"leaf", "int8_fallback"}:
        nodes.append(node)
    for child in node.children:
        nodes.extend(_executable_nodes(child))
    return nodes


def _synthesis_prompt(original_prompt: str, tasks: list[TaskResult]) -> str:
    payload = {
        "original_prompt": original_prompt,
        "ordered_task_results": [
            {
                "task": task.instruction,
                "result": task.response,
            }
            for task in tasks
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _leaf_prompt(
    original_prompt: str,
    node: DecompNode,
    dependency_results: list[TaskResult],
) -> str:
    return json.dumps(
        {
            "original_prompt": original_prompt,
            "current_task": node.text,
            "dependency_results": [
                {"task": task.instruction, "result": task.response}
                for task in dependency_results
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def _results_for_dependencies(
    node: DecompNode, completed: list[TaskResult]
) -> list[TaskResult]:
    if not node.dependencies:
        return []
    return [
        result
        for result in completed
        if any(
            result.node_id == dependency
            or result.node_id.startswith(f"{dependency}_")
            for dependency in node.dependencies
        )
    ]


def _reduction_prompt(tasks: list[TaskResult]) -> str:
    return json.dumps(
        {
            "ordered_task_results": [
                {"task": task.instruction, "result": task.response}
                for task in tasks
            ]
        },
        ensure_ascii=False,
        indent=2,
    )


def _task_from_generation(
    *,
    node_id: str,
    instruction: str,
    resolved_via: str,
    generation: GenerationResult,
) -> TaskResult:
    return TaskResult(
        node_id=node_id,
        instruction=instruction,
        response=generation.response,
        resolved_via=resolved_via,
        input_tokens=generation.input_tokens,
        generated_tokens=generation.generated_tokens,
        max_tokens_used=generation.max_tokens_used,
        context_size=generation.context_size,
    )


def _count_for_synthesis(prompt: str, system_prompt: str, reserve: int) -> TokenCountResult:
    return count_int8_tokens(
        prompt,
        system_prompt=system_prompt,
        reserved_output_tokens=reserve,
        safety_margin_tokens=SAFETY_MARGIN_TOKENS,
    )


def _build_reduction_batches(tasks: list[TaskResult]) -> list[list[TaskResult]]:
    batches: list[list[TaskResult]] = []
    current: list[TaskResult] = []

    for task in tasks:
        candidate = [*current, task]
        count = _count_for_synthesis(
            _reduction_prompt(candidate),
            REDUCTION_SYSTEM_PROMPT,
            INTERMEDIATE_MAX_TOKENS,
        )
        if count.fits:
            current = candidate
            continue

        if not current:
            raise PipelineExecutionError(
                "synthesis",
                f"task {task.node_id} cannot fit in an intermediate reduction request",
            )

        batches.append(current)
        current = [task]
        single_count = _count_for_synthesis(
            _reduction_prompt(current),
            REDUCTION_SYSTEM_PROMPT,
            INTERMEDIATE_MAX_TOKENS,
        )
        if not single_count.fits:
            raise PipelineExecutionError(
                "synthesis",
                f"task {task.node_id} cannot fit in an intermediate reduction request",
            )

    if current:
        batches.append(current)
    return batches


def _synthesize(
    original_prompt: str,
    tasks: list[TaskResult],
    run_id: str,
) -> tuple[GenerationResult, int]:
    original_only = _count_for_synthesis(
        _synthesis_prompt(original_prompt, []),
        SYNTHESIS_SYSTEM_PROMPT,
        FINAL_MAX_TOKENS,
    )
    if not original_only.fits:
        raise PipelineExecutionError(
            "synthesis",
            "the original prompt cannot fit with the reserved final-answer budget",
        )

    current = tasks
    rounds = 0
    while True:
        final_prompt = _synthesis_prompt(original_prompt, current)
        final_count = _count_for_synthesis(
            final_prompt,
            SYNTHESIS_SYSTEM_PROMPT,
            FINAL_MAX_TOKENS,
        )
        if final_count.fits:
            return (
                generate_int8(
                    final_prompt,
                    request_id=f"{run_id}:synthesis",
                    system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                    max_tokens=FINAL_MAX_TOKENS,
                    purpose="synthesis",
                ),
                rounds,
            )

        if rounds >= MAX_REDUCTION_ROUNDS:
            raise PipelineExecutionError(
                "synthesis", "hierarchical reduction exceeded its maximum number of rounds"
            )

        previous_tokens = final_count.input_tokens
        batches = _build_reduction_batches(current)
        reduced: list[TaskResult] = []
        for index, batch in enumerate(batches, start=1):
            generation = generate_int8(
                _reduction_prompt(batch),
                request_id=f"{run_id}:reduce:{rounds + 1}:{index}",
                system_prompt=REDUCTION_SYSTEM_PROMPT,
                max_tokens=INTERMEDIATE_MAX_TOKENS,
                purpose="reduction",
            )
            reduced.append(
                _task_from_generation(
                    node_id=f"reduction_{rounds + 1}_{index}",
                    instruction=f"Ordered reduction batch {index}",
                    resolved_via="hierarchical_reduction",
                    generation=generation,
                )
            )

        reduced_count = _count_for_synthesis(
            _synthesis_prompt(original_prompt, reduced),
            SYNTHESIS_SYSTEM_PROMPT,
            FINAL_MAX_TOKENS,
        )
        if reduced_count.input_tokens >= previous_tokens and len(reduced) >= len(current):
            raise PipelineExecutionError(
                "synthesis", "hierarchical reduction did not reduce the synthesis payload"
            )

        current = reduced
        rounds += 1


def run_pipeline(prompt: str, pipeline_id: str | None = None) -> PipelineResult:
    """Classify, decompose, execute, and synthesize one user prompt."""
    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise PipelineExecutionError("classification", "the prompt is empty")

    run_id = pipeline_id or uuid.uuid4().hex[:12]

    try:
        classification = classify(clean_prompt, prompt_id=run_id)
    except Exception as exc:
        raise PipelineExecutionError("classification", str(exc)) from exc

    root = DagNode(
        node_id="root",
        text=clean_prompt,
        intent_type=classification.aggregate_label,
        token_length=sum(clause.token_count for clause in classification.clauses),
    )

    try:
        tree = decompose(
            root,
            depth=0,
            original_full_prompt=clean_prompt,
            pipeline_id=run_id,
        )
    except Exception as exc:
        raise PipelineExecutionError("decomposition", str(exc)) from exc

    warnings: list[str] = []
    task_results: list[TaskResult] = []
    executable_nodes = _executable_nodes(tree)

    if len(executable_nodes) > MAX_EXECUTABLE_LEAVES:
        try:
            generation = generate_int8(
                clean_prompt,
                request_id=f"{run_id}:leaf-limit-fallback",
                system_prompt=FALLBACK_SYSTEM_PROMPT,
                max_tokens=FALLBACK_MAX_TOKENS,
                purpose="leaf_limit_fallback",
            )
        except Exception as exc:
            raise PipelineExecutionError("leaf-limit fallback", str(exc)) from exc
        task_results.append(
            _task_from_generation(
                node_id="root",
                instruction=clean_prompt,
                resolved_via="leaf_limit_fallback",
                generation=generation,
            )
        )
        warnings.append(
            f"Decomposition produced {len(executable_nodes)} leaves; "
            f"the {MAX_EXECUTABLE_LEAVES}-leaf limit triggered direct INT8 execution."
        )

    for node in [] if task_results else executable_nodes:
        if node.resolved_via == "int8_fallback":
            if not node.response:
                raise PipelineExecutionError(
                    "fallback",
                    node.failure_reason or f"no response was returned for {node.node_id}",
                )
            generation = node.generation or GenerationResult(response=node.response)
            if node.failure_reason:
                warnings.append(
                    f"{node.node_id} used INT8 fallback after: {node.failure_reason}"
                )
        else:
            try:
                generation = generate_int8(
                    _leaf_prompt(
                        clean_prompt,
                        node,
                        _results_for_dependencies(node, task_results),
                    ),
                    request_id=f"{run_id}:leaf:{node.node_id}",
                    system_prompt=LEAF_SYSTEM_PROMPT,
                    max_tokens=LEAF_MAX_TOKENS,
                    purpose="leaf",
                )
            except Exception as exc:
                raise PipelineExecutionError(
                    "leaf execution", f"{node.node_id}: {exc}"
                ) from exc

        task_results.append(
            _task_from_generation(
                node_id=node.node_id,
                instruction=node.text,
                resolved_via=node.resolved_via or "leaf",
                generation=generation,
            )
        )

    if not task_results:
        raise PipelineExecutionError("execution", "decomposition produced no executable tasks")

    synthesis_rounds = 0
    final_generation: GenerationResult | None = None
    if len(task_results) == 1:
        final_answer = task_results[0].response
    else:
        try:
            final_generation, synthesis_rounds = _synthesize(
                clean_prompt, task_results, run_id
            )
            final_answer = final_generation.response
            if synthesis_rounds:
                warnings.append(
                    f"Synthesis used {synthesis_rounds} hierarchical reduction round(s) "
                    "to stay within the model context window."
                )
        except Exception as exc:
            if isinstance(exc, PipelineExecutionError):
                raise
            raise PipelineExecutionError("synthesis", str(exc)) from exc

    return PipelineResult(
        pipeline_id=run_id,
        classification=classification,
        tree=tree,
        task_results=task_results,
        final_answer=final_answer,
        warnings=warnings,
        synthesis_rounds=synthesis_rounds,
        final_generation=final_generation,
    )
