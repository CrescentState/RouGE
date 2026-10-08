from __future__ import annotations

from types import SimpleNamespace

import benchmark
from DAG.decompose import GenerationResult


class FakeSampler:
    instances = []

    def __init__(self):
        self.samples = [
            benchmark.TelemetrySample(0.0, 1000, 7000, 10, 45, 20),
            benchmark.TelemetrySample(1.0, 1200, 6800, 20, 47, 60),
            benchmark.TelemetrySample(2.0, 1100, 6900, 10, 46, 30),
        ]
        self.error = ""
        self.started = False
        self.stopped = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True

    def stop(self, duration_seconds):
        self.stopped = True
        return benchmark.aggregate_metrics(self.samples, duration_seconds)


def test_aggregate_metrics_integrates_energy():
    metrics = benchmark.aggregate_metrics(
        [
            benchmark.TelemetrySample(0.0, 1000, 7000, 10, 45, 20),
            benchmark.TelemetrySample(1.0, 1200, 6800, 20, 47, 60),
            benchmark.TelemetrySample(2.0, 1100, 6900, 10, 46, 30),
        ],
        duration_seconds=2.0,
    )

    assert metrics.energy_joules == 30.0
    assert metrics.average_power_w == 40 / 3
    assert metrics.peak_used_vram_mb == 1200
    assert metrics.minimum_free_vram_mb == 6800
    assert metrics.peak_temperature_c == 47
    assert metrics.peak_gpu_util_pct == 60


def test_normal_mode_is_one_direct_int8_call(monkeypatch):
    calls = []

    def fake_generate(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return GenerationResult(
            response="direct answer",
            input_tokens=12,
            generated_tokens=25,
        )

    monkeypatch.setattr(benchmark, "generate_int8", fake_generate)
    monkeypatch.setattr(
        benchmark,
        "run_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("normal mode must not invoke the middleware")
        ),
    )

    outcome = benchmark.run_benchmark(
        "test prompt",
        "normal",
        run_id="normal-run",
        persist=False,
        sampler_factory=FakeSampler,
    )

    assert outcome.success
    assert outcome.answer == "direct answer"
    assert outcome.task_count == 1
    assert outcome.input_tokens == 12
    assert outcome.output_tokens == 25
    assert len(calls) == 1
    assert calls[0][1]["baseline_mode"] == "normal:int8"
    assert calls[0][1]["max_tokens"] == 1024


def test_middleware_mode_uses_pipeline(monkeypatch):
    result = SimpleNamespace(
        final_answer="middleware answer",
        warnings=["test warning"],
        task_results=[
            SimpleNamespace(input_tokens=10, generated_tokens=20),
            SimpleNamespace(input_tokens=11, generated_tokens=21),
        ],
        synthesis_rounds=1,
        final_generation=GenerationResult(
            response="middleware answer", input_tokens=30, generated_tokens=40
        ),
    )
    monkeypatch.setattr(benchmark, "run_pipeline", lambda *args, **kwargs: result)
    monkeypatch.setattr(
        benchmark,
        "generate_int8",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("middleware mode must not use direct execution")
        ),
    )

    outcome = benchmark.run_benchmark(
        "test prompt",
        "middleware",
        run_id="middleware-run",
        persist=False,
        sampler_factory=FakeSampler,
    )

    assert outcome.success
    assert outcome.answer == "middleware answer"
    assert outcome.task_count == 2
    assert outcome.synthesis_rounds == 1
    assert outcome.input_tokens == 51
    assert outcome.output_tokens == 81
    assert outcome.warnings == ["test warning"]


def test_sampler_stops_and_failure_is_returned(monkeypatch):
    FakeSampler.instances.clear()
    monkeypatch.setattr(
        benchmark,
        "generate_int8",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("engine down")),
    )

    outcome = benchmark.run_benchmark(
        "test prompt",
        "normal",
        persist=False,
        sampler_factory=FakeSampler,
    )

    assert not outcome.success
    assert outcome.error == "engine down"
    assert FakeSampler.instances[-1].started
    assert FakeSampler.instances[-1].stopped


def test_history_matches_latest_opposite_mode(monkeypatch, tmp_path):
    runs_path = tmp_path / "benchmark_runs.csv"
    samples_path = tmp_path / "benchmark_samples.csv"
    monkeypatch.setattr(benchmark, "RUNS_CSV", runs_path)
    monkeypatch.setattr(benchmark, "SAMPLES_CSV", samples_path)
    prompt_hash = benchmark.prompt_fingerprint("same prompt")
    metrics = benchmark.RunMetrics(duration_seconds=1.0, sample_count=1)
    sample = benchmark.TelemetrySample(0.0, 1000, 7000, 10, 45, 20)

    normal = benchmark.BenchmarkOutcome(
        run_id="normal-1",
        mode="normal",
        prompt_hash=prompt_hash,
        started_at=1.0,
        success=True,
        metrics=metrics,
    )
    middleware = benchmark.BenchmarkOutcome(
        run_id="middleware-1",
        mode="middleware",
        prompt_hash=prompt_hash,
        started_at=2.0,
        success=True,
        metrics=metrics,
    )
    benchmark.save_outcome(normal, [sample])
    benchmark.save_outcome(middleware, [sample])

    match = benchmark.latest_opposite_run(prompt_hash, "normal", runs_path)

    assert match is not None
    assert match["run_id"] == "middleware-1"
    assert len(runs_path.read_text().splitlines()) == 3
    assert len(samples_path.read_text().splitlines()) == 3
