"""End-to-end orchestration for the integrated RouGE pipeline."""

from __future__ import annotations

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from classifier import ClassificationResult, Clause, classify, estimate_tokens
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
    generate_int4,
    generate_int8,
    new_attempt_tracker,
)


SYNTHESIS_SYSTEM_PROMPT = (
    "You are the final response synthesizer. Answer every explicit deliverable in the "
    "original user prompt using the ordered task results. Produce one coherent, "
    "self-contained answer. Preserve requested primary artifacts such as stories, code, "
    "letters, or plans in full; do not replace them with a description or summary. "
    "Reproduce concrete lists, names, and numbers from the task results verbatim; "
    "never drop a requested list while keeping only its commentary. Then "
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
    "actual requested artifact, not an explanation of what it would contain. Prefer "
    "concrete specifics from the provided context over generic commentary; if a detail "
    "is genuinely unavailable, say so once and still deliver the requested structure. "
    "Begin directly; never announce that you will do the task later. Do not mention "
    "routing, DAGs, nodes, dependency data, or internal processing."
)

LEAF_MAX_TOKENS = 512
FINAL_MAX_TOKENS = 1024
INTERMEDIATE_MAX_TOKENS = 256
# Single unknown-intent prompts at or below this size skip decomposition and
# are answered directly by INT8 (e.g. greetings like "Hi there", which the
# prototype bank cannot match and would otherwise burn an INT4 planning call
# plus a fallback call). Calibrate against classifier/data/evaluation.jsonl.
TRIVIAL_TOKEN_LIMIT = 32
SAFETY_MARGIN_TOKENS = 128
MAX_EXECUTABLE_LEAVES = 4
MAX_REDUCTION_ROUNDS = 8
# Independent leaves in one dependency batch run concurrently. Effective
# parallelism also needs the API to offload blocking llama.cpp calls (it
# does, with a per-engine lock: cross-engine leaves overlap, same-engine
# leaves serialize on the shared llama.cpp context).
MAX_LEAF_WORKERS = 4


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
    engine: str = "INT8"  # engine that produced response ("INT4" or "INT8")


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


def _root_task_type(classification: ClassificationResult) -> str:
    """Dominant clause task type for the root node (drives engine routing).

    Single-clause prompts inherit their clause type, so a direct creative or
    code prompt routes to INT8 even without decomposition.
    """
    types = {clause.task_type for clause in classification.clauses}
    if len(types) == 1:
        return next(iter(types))
    return "mixed"


_LIST_ITEM = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.*\S)")
# Minimum fraction of leaf list items that must survive (verbatim substring)
# in the synthesized answer before it is trusted over concatenation.
SYNTHESIS_ITEM_RECALL_FLOOR = 0.5


def _leaf_list_items(tasks: list[TaskResult]) -> list[str]:
    items: list[str] = []
    for task in tasks:
        for line in task.response.splitlines():
            match = _LIST_ITEM.match(line)
            if match:
                items.append(match.group(1).strip())
    return items


def _synthesis_kept_items(final_answer: str, items: list[str]) -> bool:
    if len(items) < 2:
        return True
    lowered = final_answer.lower()
    kept = sum(1 for item in items if item[:60].lower() in lowered)
    return kept / len(items) >= SYNTHESIS_ITEM_RECALL_FLOOR


# Prompts at or below this length skip embedding entirely: there is no
# task verb to find in "Thank you", so classification can only add latency.
# Kept clear of multi-word test prompts; courtesies are shorter.
ULTRA_SHORT_CHARS = 10


def _stub_classification(clean_prompt: str, prompt_id: str) -> ClassificationResult:
    """Transparent stand-in for ultra-short prompts: one unknown clause."""
    return ClassificationResult(
        prompt_id=prompt_id,
        clauses=[
            Clause(
                id=1,
                text=clean_prompt,
                task_type="unknown",
                confidence=0.0,
                margin=0.0,
                token_count=estimate_tokens(clean_prompt),
                base_entropy="high",
            )
        ],
        aggregate_label="complex",
        segmentation_method="sentence_split_conjunction",
        notes="classifier skipped: ultra-short prompt",
    )


def _is_trivial_prompt(classification: ClassificationResult) -> bool:
    """True when decomposition cannot add value: every clause is unknown and
    the whole prompt fits within TRIVIAL_TOKEN_LIMIT."""
    if not classification.clauses:
        return False
    if any(clause.task_type != "unknown" for clause in classification.clauses):
        return False
    return (
        sum(clause.token_count for clause in classification.clauses)
        <= TRIVIAL_TOKEN_LIMIT
    )


def _is_untyped_prompt(classification: ClassificationResult) -> bool:
    """True when no clause matched any known task type (any size).

    The INT4 planner grounds its DAG in typed intents; with zero typed
    clauses it repeats the parent or emits placeholders until the budget or
    breaker stops it (measured: 12 wasted leaves, ~50s). Untyped prompts go
    straight to one INT8 call however large they are.
    """
    return bool(classification.clauses) and all(
        clause.task_type == "unknown" for clause in classification.clauses
    )


def _direct_answer(
    *,
    run_id: str,
    clean_prompt: str,
    classification: ClassificationResult,
    token_length: int,
    engine: str,
    purpose: str,
    resolved_via: str,
    note: str,
    classify_note: str,
) -> PipelineResult:
    """Answer without decomposition using a single engine call."""
    run_generate = generate_int4 if engine == "INT4" else generate_int8
    try:
        generation = run_generate(
            clean_prompt,
            request_id=f"{run_id}:{purpose}",
            system_prompt=FALLBACK_SYSTEM_PROMPT,
            max_tokens=FALLBACK_MAX_TOKENS,
            purpose=purpose,
        )
    except Exception as exc:
        raise PipelineExecutionError(f"{purpose} execution", str(exc)) from exc
    task = _task_from_generation(
        node_id="root",
        instruction=clean_prompt,
        resolved_via=resolved_via,
        generation=generation,
        engine=engine,
    )
    return PipelineResult(
        pipeline_id=run_id,
        classification=classification,
        tree=DecompNode(
            node_id="root",
            text=clean_prompt,
            intent_type=classification.aggregate_label,
            token_length=token_length,
            depth=0,
            resolved_via="leaf",
        ),
        task_results=[task],
        final_answer=generation.response,
        warnings=[note, classify_note],
    )


def _deps_satisfied(node: DecompNode, done_ids: set[str]) -> bool:
    """True when every dependency of node already has a completed result."""
    for dependency in node.dependencies:
        if not any(
            done_id == dependency or done_id.startswith(f"{dependency}_")
            for done_id in done_ids
        ):
            return False
    return True


def _engine_for_node(node: DecompNode) -> str:
    """Route simple extraction/summarization leaves to INT4, rest to INT8.

    INT4 is cheaper and faster and adequate for low-entropy extraction and
    summarization. Complex/mixed leaves, fallbacks, and synthesis stay on
    INT8 for quality — as do creative_writing leaves (the Q4 model announces
    creative tasks instead of performing them) and code_generation leaves
    (precision-sensitive: Q4 overcomplicates simple functions).
    """
    if node.resolved_via != "leaf" or node.intent_type != "simple":
        return "INT8"
    if node.task_type in ("creative_writing", "code_generation"):
        return "INT8"
    return "INT4"


def _task_from_generation(
    *,
    node_id: str,
    instruction: str,
    resolved_via: str,
    generation: GenerationResult,
    engine: str = "INT8",
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
        engine=engine,
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

    classify_started = time.perf_counter()
    if len(clean_prompt) <= ULTRA_SHORT_CHARS:
        classification = _stub_classification(clean_prompt, run_id)
        classify_note = "Classification skipped (ultra-short prompt)."
    else:
        try:
            classification = classify(clean_prompt, prompt_id=run_id)
        except Exception as exc:
            raise PipelineExecutionError("classification", str(exc)) from exc
        classify_note = (
            f"Classification took {time.perf_counter() - classify_started:.2f}s."
        )

    root = DagNode(
        node_id="root",
        text=clean_prompt,
        intent_type=classification.aggregate_label,
        token_length=sum(clause.token_count for clause in classification.clauses),
    )
    root_task_type = _root_task_type(classification)

    if _is_trivial_prompt(classification):
        # Greetings and other tiny out-of-bank prompts: answer directly with
        # one cheap INT4 call instead of burning an INT4 planning call (which
        # the planner cannot ground) plus a fallback call.
        return _direct_answer(
            run_id=run_id,
            clean_prompt=clean_prompt,
            classification=classification,
            token_length=root.token_length,
            engine="INT4",
            purpose="trivial_direct",
            resolved_via="trivial_direct",
            note=(
                "Prompt served directly by INT4 without decomposition "
                "(trivial out-of-bank prompt)."
            ),
            classify_note=classify_note,
        )

    if _is_untyped_prompt(classification):
        # No recognized task type at all: planning would fail, so answer
        # directly with one INT8 call (quality-safe for untyped content).
        return _direct_answer(
            run_id=run_id,
            clean_prompt=clean_prompt,
            classification=classification,
            token_length=root.token_length,
            engine="INT8",
            purpose="untyped_direct",
            resolved_via="untyped_direct",
            note=(
                "Prompt served directly by INT8 without decomposition "
                "(no recognized task type)."
            ),
            classify_note=classify_note,
        )

    warnings: list[str] = []
    attempt_tracker = new_attempt_tracker()
    try:
        tree = decompose(
            root,
            depth=0,
            original_full_prompt=clean_prompt,
            pipeline_id=run_id,
            attempt_tracker=attempt_tracker,
            task_type=root_task_type,
        )
    except Exception as exc:
        raise PipelineExecutionError("decomposition", str(exc)) from exc
    if attempt_tracker["capped"]:
        warnings.append(
            f"Planning budget exhausted; {len(attempt_tracker['capped'])} "
            "node(s) answered directly without further decomposition."
        )
    if attempt_tracker["breaker"]:
        warnings.append(
            f"Planning circuit breaker routed {len(attempt_tracker['breaker'])} "
            "node(s) straight to leaves after repeated planner failures."
        )
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
        if node.resolved_via != "int8_fallback":
            continue
        if not node.response:
            raise PipelineExecutionError(
                "fallback",
                node.failure_reason or f"no response was returned for {node.node_id}",
            )
        task_results.append(
            _task_from_generation(
                node_id=node.node_id,
                instruction=node.text,
                resolved_via=node.resolved_via,
                generation=node.generation or GenerationResult(response=node.response),
                engine="INT8",
            )
        )
        if node.failure_reason:
            warnings.append(
                f"{node.node_id} used INT8 fallback after: {node.failure_reason}"
            )

    pending = [
        node
        for node in executable_nodes
        if node.resolved_via != "int8_fallback" and not task_results
    ]
    completed: dict[str, TaskResult] = {
        task.node_id: task for task in task_results
    }

    def _run_leaf_request(
        node: DecompNode, dependency_results: list[TaskResult]
    ) -> tuple[GenerationResult, str]:
        engine = _engine_for_node(node)
        run_generate = generate_int4 if engine == "INT4" else generate_int8
        generation = run_generate(
            _leaf_prompt(clean_prompt, node, dependency_results),
            request_id=f"{run_id}:leaf:{node.node_id}",
            system_prompt=LEAF_SYSTEM_PROMPT,
            max_tokens=LEAF_MAX_TOKENS,
            purpose="leaf",
        )
        return generation, engine

    while pending:
        ready = [
            node for node in pending if _deps_satisfied(node, set(completed))
        ]
        if not ready:
            raise PipelineExecutionError(
                "execution",
                "leaf dependencies cannot be satisfied (circular or missing)",
            )
        snapshots = {
            node.node_id: _results_for_dependencies(
                node, list(completed.values())
            )
            for node in ready
        }
        with ThreadPoolExecutor(
            max_workers=min(MAX_LEAF_WORKERS, len(ready))
        ) as executor:
            futures = {
                executor.submit(_run_leaf_request, node, snapshots[node.node_id]): node
                for node in ready
            }
            for future, node in futures.items():
                try:
                    generation, engine = future.result()
                except Exception as exc:
                    raise PipelineExecutionError(
                        "leaf execution", f"{node.node_id}: {exc}"
                    ) from exc
                completed[node.node_id] = _task_from_generation(
                    node_id=node.node_id,
                    instruction=node.text,
                    resolved_via=node.resolved_via or "leaf",
                    generation=generation,
                    engine=engine,
                )
        pending = [node for node in pending if node.node_id not in completed]

    if completed and not task_results:
        # Deterministic tree order (was implicit in the old sequential loop).
        task_results = [completed[node.node_id] for node in executable_nodes]

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
            items = _leaf_list_items(task_results)
            if not _synthesis_kept_items(final_answer, items):
                # Synthesis dropped concrete leaf content (e.g. a requested
                # list): concatenation preserves every deliverable with no
                # extra LLM call.
                final_answer = "\n\n".join(
                    f"### {task.instruction}\n{task.response}"
                    for task in task_results
                )
                warnings.append(
                    "Synthesis dropped leaf list content; concatenated leaf "
                    "answers used instead."
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
        warnings=[*warnings, classify_note],
        synthesis_rounds=synthesis_rounds,
        final_generation=final_generation,
    )
