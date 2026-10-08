"""Run-level benchmarking for direct INT8 and RouGE middleware execution."""

from __future__ import annotations

import csv
import hashlib
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal

import pynvml

from DAG.decompose import GenerationResult, generate_int8
from pipeline import FINAL_MAX_TOKENS, PipelineResult, run_pipeline


ExecutionMode = Literal["normal", "middleware"]
SAMPLE_INTERVAL_SECONDS = 0.2
PROJECT_ROOT = Path(__file__).resolve().parent
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
RUNS_CSV = EVALUATION_DIR / "benchmark_runs.csv"
SAMPLES_CSV = EVALUATION_DIR / "benchmark_samples.csv"

DIRECT_SYSTEM_PROMPT = (
    "Answer every explicit part of the user's request in one coherent, self-contained "
    "response. Produce requested artifacts such as stories, code, letters, or plans in "
    "full rather than describing them. Use clear sections when the request has multiple "
    "deliverables."
)


@dataclass(frozen=True)
class TelemetrySample:
    elapsed_seconds: float
    used_vram_mb: float
    free_vram_mb: float
    power_w: float
    temperature_c: float
    gpu_util_pct: float


@dataclass(frozen=True)
class RunMetrics:
    duration_seconds: float
    sample_count: int
    energy_joules: float | None = None
    average_power_w: float | None = None
    peak_power_w: float | None = None
    average_used_vram_mb: float | None = None
    peak_used_vram_mb: float | None = None
    minimum_free_vram_mb: float | None = None
    peak_temperature_c: float | None = None
    average_gpu_util_pct: float | None = None
    peak_gpu_util_pct: float | None = None


@dataclass
class BenchmarkOutcome:
    run_id: str
    mode: ExecutionMode
    prompt_hash: str
    started_at: float
    success: bool
    metrics: RunMetrics
    answer: str = ""
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    task_count: int = 0
    synthesis_rounds: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    pipeline_result: PipelineResult | None = None
    generation: GenerationResult | None = None


class TelemetrySampler:
    """Sample whole-device NVIDIA telemetry while one benchmark run executes."""

    def __init__(self, interval_seconds: float = SAMPLE_INTERVAL_SECONDS):
        self.interval_seconds = interval_seconds
        self.samples: list[TelemetrySample] = []
        self.error: str = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._handle = None

    def start(self) -> None:
        self._started = time.perf_counter()
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._sample_once()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception as exc:
            self.error = str(exc)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._sample_once()
            except Exception as exc:
                self.error = str(exc)
                return

    def _sample_once(self) -> None:
        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        self.samples.append(
            TelemetrySample(
                elapsed_seconds=time.perf_counter() - self._started,
                used_vram_mb=info.used / (1024**2),
                free_vram_mb=info.free / (1024**2),
                power_w=pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0,
                temperature_c=float(
                    pynvml.nvmlDeviceGetTemperature(
                        self._handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                ),
                gpu_util_pct=float(utilization.gpu),
            )
        )

    def stop(self, duration_seconds: float) -> RunMetrics:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 3))
        if self._handle is not None:
            try:
                self._sample_once()
            except Exception as exc:
                self.error = str(exc)
        return aggregate_metrics(self.samples, duration_seconds)


def aggregate_metrics(
    samples: list[TelemetrySample], duration_seconds: float
) -> RunMetrics:
    if not samples:
        return RunMetrics(duration_seconds=duration_seconds, sample_count=0)

    energy_joules = 0.0
    for previous, current in zip(samples, samples[1:]):
        elapsed = max(0.0, current.elapsed_seconds - previous.elapsed_seconds)
        energy_joules += elapsed * (previous.power_w + current.power_w) / 2.0

    def average(values: list[float]) -> float:
        return sum(values) / len(values)

    powers = [sample.power_w for sample in samples]
    used_vram = [sample.used_vram_mb for sample in samples]
    free_vram = [sample.free_vram_mb for sample in samples]
    temperatures = [sample.temperature_c for sample in samples]
    utilization = [sample.gpu_util_pct for sample in samples]
    return RunMetrics(
        duration_seconds=duration_seconds,
        sample_count=len(samples),
        energy_joules=energy_joules,
        average_power_w=average(powers),
        peak_power_w=max(powers),
        average_used_vram_mb=average(used_vram),
        peak_used_vram_mb=max(used_vram),
        minimum_free_vram_mb=min(free_vram),
        peak_temperature_c=max(temperatures),
        average_gpu_util_pct=average(utilization),
        peak_gpu_util_pct=max(utilization),
    )


def prompt_fingerprint(prompt: str) -> str:
    normalized = " ".join(prompt.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _middleware_token_totals(result: PipelineResult) -> tuple[int, int]:
    generations = list(result.task_results)
    input_tokens = sum(item.input_tokens for item in generations)
    output_tokens = sum(item.generated_tokens for item in generations)
    if result.final_generation is not None:
        input_tokens += result.final_generation.input_tokens
        output_tokens += result.final_generation.generated_tokens
    return input_tokens, output_tokens


def run_benchmark(
    prompt: str,
    mode: ExecutionMode,
    *,
    run_id: str | None = None,
    persist: bool = True,
    sampler_factory: Callable[[], TelemetrySampler] = TelemetrySampler,
) -> BenchmarkOutcome:
    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise ValueError("the prompt is empty")
    if mode not in {"normal", "middleware"}:
        raise ValueError("mode must be 'normal' or 'middleware'")

    identifier = run_id or uuid.uuid4().hex[:12]
    started_at = time.time()
    sampler = sampler_factory()
    sampler.start()
    started = time.perf_counter()
    answer = ""
    error = ""
    warnings: list[str] = []
    task_count = 0
    synthesis_rounds = 0
    input_tokens = 0
    output_tokens = 0
    pipeline_result: PipelineResult | None = None
    generation: GenerationResult | None = None

    try:
        if mode == "normal":
            generation = generate_int8(
                clean_prompt,
                request_id=f"{identifier}:normal",
                system_prompt=DIRECT_SYSTEM_PROMPT,
                max_tokens=FINAL_MAX_TOKENS,
                purpose="normal",
                baseline_mode="normal:int8",
            )
            answer = generation.response
            input_tokens = generation.input_tokens
            output_tokens = generation.generated_tokens
            task_count = 1
        else:
            pipeline_result = run_pipeline(clean_prompt, pipeline_id=identifier)
            answer = pipeline_result.final_answer
            warnings = list(pipeline_result.warnings)
            task_count = len(pipeline_result.task_results)
            synthesis_rounds = pipeline_result.synthesis_rounds
            input_tokens, output_tokens = _middleware_token_totals(pipeline_result)
    except Exception as exc:
        error = str(exc)

    duration_seconds = time.perf_counter() - started
    metrics = sampler.stop(duration_seconds)
    if sampler.error:
        warnings.append(f"Telemetry sampling was incomplete: {sampler.error}")

    outcome = BenchmarkOutcome(
        run_id=identifier,
        mode=mode,
        prompt_hash=prompt_fingerprint(clean_prompt),
        started_at=started_at,
        success=not error,
        metrics=metrics,
        answer=answer,
        error=error,
        warnings=warnings,
        task_count=task_count,
        synthesis_rounds=synthesis_rounds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        pipeline_result=pipeline_result,
        generation=generation,
    )
    if persist:
        save_outcome(outcome, sampler.samples)
    return outcome


RUN_FIELDS = [
    "run_id",
    "timestamp",
    "prompt_hash",
    "mode",
    "success",
    "duration_seconds",
    "sample_count",
    "energy_joules",
    "average_power_w",
    "peak_power_w",
    "average_used_vram_mb",
    "peak_used_vram_mb",
    "minimum_free_vram_mb",
    "peak_temperature_c",
    "average_gpu_util_pct",
    "peak_gpu_util_pct",
    "input_tokens",
    "output_tokens",
    "task_count",
    "synthesis_rounds",
    "warning_count",
    "error",
]
SAMPLE_FIELDS = ["run_id", "prompt_hash", "mode", *TelemetrySample.__dataclass_fields__]
_CSV_LOCK = threading.Lock()


def _append_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def save_outcome(outcome: BenchmarkOutcome, samples: list[TelemetrySample]) -> None:
    metrics = asdict(outcome.metrics)
    run_row = {
        "run_id": outcome.run_id,
        "timestamp": round(outcome.started_at, 3),
        "prompt_hash": outcome.prompt_hash,
        "mode": outcome.mode,
        "success": outcome.success,
        **metrics,
        "input_tokens": outcome.input_tokens,
        "output_tokens": outcome.output_tokens,
        "task_count": outcome.task_count,
        "synthesis_rounds": outcome.synthesis_rounds,
        "warning_count": len(outcome.warnings),
        "error": outcome.error,
    }
    sample_rows = [
        {
            "run_id": outcome.run_id,
            "prompt_hash": outcome.prompt_hash,
            "mode": outcome.mode,
            **asdict(sample),
        }
        for sample in samples
    ]
    with _CSV_LOCK:
        _append_csv(RUNS_CSV, RUN_FIELDS, [run_row])
        _append_csv(SAMPLES_CSV, SAMPLE_FIELDS, sample_rows)


def latest_opposite_run(
    prompt_hash: str, mode: ExecutionMode, path: Path = RUNS_CSV
) -> dict[str, str] | None:
    if not path.exists():
        return None
    opposite = "middleware" if mode == "normal" else "normal"
    with path.open(newline="", encoding="utf-8") as stream:
        matches = [
            row
            for row in csv.DictReader(stream)
            if row.get("prompt_hash") == prompt_hash
            and row.get("mode") == opposite
            and row.get("success", "").lower() == "true"
        ]
    return matches[-1] if matches else None
