"""Streamlit interface for the integrated RouGE pipeline."""

from __future__ import annotations

import re

import streamlit as st

from benchmark import BenchmarkOutcome, latest_opposite_run, run_benchmark
from DAG.decompose import DecompNode, get_free_vram_mb
from Setup.generation import CONTEXT_SIZE


st.set_page_config(page_title="RouGE Unified Pipeline", layout="wide")


@st.cache_resource(show_spinner=False)
def _warm_classification_backend() -> str:
    """Load the MiniLM embedding model once per Streamlit process.

    The first classify() call otherwise pays ~8s of model load inside a
    measured middleware run. Warming at startup keeps that cost out of
    benchmark timings.
    """
    from classifier import get_backend

    return get_backend()


_warm_backend = _warm_classification_backend()


def _mermaid_id(node_id: str) -> str:
    """Return an identifier accepted by Mermaid."""
    return "node_" + re.sub(r"[^a-zA-Z0-9_]", "_", node_id)


def build_mermaid_chart(
    node: DecompNode,
    graph_lines: list[str] | None = None,
    parent_id: str | None = None,
) -> str:
    if graph_lines is None:
        graph_lines = ["graph TD"]

    safe_text = node.text.replace('"', "'").replace("\n", " ")[:60]
    if node.atomic:
        safe_text = f"[atomic] {safe_text}"
    current_id = _mermaid_id(node.node_id)

    if node.resolved_via == "int8_fallback":
        shape = f'{current_id}>"INT8 Fallback:<br>{safe_text}"]'
    elif node.resolved_via == "decomposed":
        shape = f'{current_id}{{"INT4 Decomposed:<br>{safe_text}"}}'
    else:
        shape = f'{current_id}["Leaf Node:<br>{safe_text}"]'

    graph_lines.append(shape)
    if parent_id:
        graph_lines.append(f"{parent_id} --> {current_id}")

    for child in node.children:
        build_mermaid_chart(child, graph_lines, current_id)

    for dependency in node.dependencies:
        graph_lines.append(f"{_mermaid_id(dependency)} -. context .-> {current_id}")

    return "\n".join(graph_lines)


def _format_metric(value: float | None, suffix: str, decimals: int = 2) -> str:
    if value is None:
        return "Unavailable"
    return f"{value:.{decimals}f} {suffix}".strip()


def render_run_metrics(outcome: BenchmarkOutcome) -> None:
    metrics = outcome.metrics
    st.subheader("Run Measurements")
    first_row = st.columns(4)
    first_row[0].metric("Mode", "Normal INT8" if outcome.mode == "normal" else "RouGE")
    first_row[1].metric("End-to-end time", _format_metric(metrics.duration_seconds, "s"))
    first_row[2].metric("GPU energy", _format_metric(metrics.energy_joules, "J"))
    first_row[3].metric("Peak VRAM", _format_metric(metrics.peak_used_vram_mb, "MB", 0))

    second_row = st.columns(4)
    second_row[0].metric("Average power", _format_metric(metrics.average_power_w, "W"))
    second_row[1].metric("Peak power", _format_metric(metrics.peak_power_w, "W"))
    second_row[2].metric("Peak temperature", _format_metric(metrics.peak_temperature_c, "°C", 0))
    second_row[3].metric("Peak GPU utilization", _format_metric(metrics.peak_gpu_util_pct, "%", 0))
    st.caption(
        f"Run ID: {outcome.run_id} · Telemetry samples: {metrics.sample_count} · "
        f"Input/output tokens observed: {outcome.input_tokens}/{outcome.output_tokens}"
    )


def _as_float(row: dict[str, str], field: str) -> float | None:
    try:
        return float(row[field]) if row.get(field) not in {None, ""} else None
    except (TypeError, ValueError):
        return None


def render_latest_comparison(outcome: BenchmarkOutcome) -> None:
    opposite = latest_opposite_run(outcome.prompt_hash, outcome.mode)
    st.subheader("Latest Same-Prompt Comparison")
    if opposite is None:
        other = "Middleware" if outcome.mode == "normal" else "Normal"
        st.info(
            f"Run this same prompt once in {other} mode to populate the comparison."
        )
        return

    fields = [
        ("End-to-end time", "duration_seconds", "s"),
        ("GPU energy", "energy_joules", "J"),
        ("Average power", "average_power_w", "W"),
        ("Peak power", "peak_power_w", "W"),
        ("Peak VRAM", "peak_used_vram_mb", "MB"),
        ("Peak temperature", "peak_temperature_c", "°C"),
        ("Peak GPU utilization", "peak_gpu_util_pct", "%"),
    ]
    current_metrics = outcome.metrics.__dict__
    rows = []
    for label, field, unit in fields:
        current = current_metrics.get(field)
        previous = _as_float(opposite, field)
        normal = current if outcome.mode == "normal" else previous
        middleware = current if outcome.mode == "middleware" else previous
        delta = None if normal in {None, 0} or middleware is None else (
            (middleware - normal) / normal * 100.0
        )
        if delta is None:
            delta_cell = "—"
        elif field == "duration_seconds" and max(normal, middleware) < 1.0:
            delta_cell = f"{middleware - normal:+.2f} {unit} (noise)"
        elif field == "energy_joules" and max(normal, middleware) < 10.0:
            delta_cell = f"{middleware - normal:+.2f} {unit} (noise)"
        else:
            delta_cell = f"{delta:+.1f}%"
        rows.append(
            {
                "Metric": label,
                "Normal": "—" if normal is None else f"{normal:.2f} {unit}",
                "Middleware": "—" if middleware is None else f"{middleware:.2f} {unit}",
                "Middleware vs Normal": delta_cell,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "Negative percentages mean the middleware used less or completed faster. "
        "A single pair is preliminary; repeat runs under similar GPU conditions."
    )


def render_session_answer_comparison(
    outcome: BenchmarkOutcome, outcomes: dict[tuple[str, str], BenchmarkOutcome]
) -> bool:
    normal = outcomes.get((outcome.prompt_hash, "normal"))
    middleware = outcomes.get((outcome.prompt_hash, "middleware"))
    if normal is None or middleware is None or not normal.success or not middleware.success:
        return False

    st.subheader("Answer Comparison")
    normal_tab, middleware_tab = st.tabs(["Normal INT8", "RouGE Middleware"])
    with normal_tab:
        st.markdown(normal.answer)
    with middleware_tab:
        st.markdown(middleware.answer)
    return True


def render_middleware_details(outcome: BenchmarkOutcome) -> None:
    result = outcome.pipeline_result
    if result is None:
        return
    st.subheader("Phase 1: Classification")
    st.success(
        f"Classified as **{result.classification.aggregate_label}** "
        f"across {len(result.classification.clauses)} clause(s)."
    )
    if result.classification.clauses:
        st.dataframe(
            [clause.model_dump() for clause in result.classification.clauses],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Phase 2: VRAM-Gated Decomposition")
    st.markdown(f"```mermaid\n{build_mermaid_chart(result.tree)}\n```")

    st.subheader("Phase 3: Task Execution")
    tabs = st.tabs(
        [f"Task {index}" for index in range(1, len(result.task_results) + 1)]
    )
    for tab, task in zip(tabs, result.task_results):
        with tab:
            st.caption(
                f"Node: {task.node_id} · Resolved via: {task.resolved_via} · "
                f"Engine: {task.engine} · "
                f"Tokens: {task.input_tokens} in / {task.generated_tokens} out"
            )
            st.markdown("**Instruction**")
            st.code(task.instruction, language=None)
            st.markdown("**Response**")
            st.write(task.response)


st.title("RouGE Architecture: End-to-End Pipeline")
st.markdown(
    f"**Hardware Status:** NVIDIA GPU | "
    f"**Free VRAM:** `{get_free_vram_mb():.0f} MB` | "
    f"**Context:** `{CONTEXT_SIZE} tokens`"
)
st.divider()

prompt = st.text_area(
    "Enter a user prompt:",
    height=120,
    placeholder="e.g., Summarize the incident report and then draft a follow-up email...",
)

use_middleware = st.toggle(
    "Use RouGE middleware",
    value=False,
    help="Off sends the prompt directly to INT8. On runs classification, DAG routing, execution, and synthesis.",
)
selected_mode = "middleware" if use_middleware else "normal"
st.caption(
    "Selected mode: **RouGE Middleware**" if use_middleware
    else "Selected mode: **Normal Direct INT8**"
)

if st.button("Execute Prompt", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        spinner = (
            "Classifying, decomposing, executing, and synthesizing..."
            if use_middleware
            else "Running direct INT8 baseline..."
        )
        with st.spinner(spinner):
            outcome = run_benchmark(prompt, selected_mode)
        st.session_state["last_benchmark_outcome"] = outcome
        stored_outcomes = st.session_state.setdefault("benchmark_outcomes", {})
        stored_outcomes[(outcome.prompt_hash, outcome.mode)] = outcome

outcome = st.session_state.get("last_benchmark_outcome")
if outcome is not None:
    if not outcome.success:
        st.error(f"{outcome.mode.title()} execution failed: {outcome.error}")
    else:
        render_run_metrics(outcome)
        render_latest_comparison(outcome)
        if outcome.mode == "middleware":
            render_middleware_details(outcome)
        compared_answers = render_session_answer_comparison(
            outcome, st.session_state.get("benchmark_outcomes", {})
        )
        if not compared_answers:
            st.subheader("Final Answer")
            st.markdown(outcome.answer)

    if outcome.warnings:
        with st.expander("Run warnings"):
            for warning in outcome.warnings:
                st.warning(warning)
