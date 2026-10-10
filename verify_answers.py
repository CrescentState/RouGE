"""One-shot Normal-vs-Middleware answer comparison for a given prompt.

Usage (API must be up):
    venv/Scripts/python.exe verify_answers.py "your prompt here"
Prints both answers, token/time stats, and a deliverable-coverage checklist.
"""
import sys

from benchmark import run_benchmark

CHECKS = [
    ("mentions ICE/fuel costs", ["ice", "internal combustion", "petrol", "diesel", "fuel"]),
    ("mentions EV/charging", ["ev", "electric", "charg", "battery"]),
    ("compares running costs", ["cost", "cheaper", "expensive", "price", "econom"]),
    ("addresses short trips", ["short", "city", "daily", "commute"]),
    ("addresses long trips", ["long", "highway", "trip", "range"]),
    ("gives a recommendation", ["recommend", "better for you", "suggest", "should", "ideal", "suitable"]),
]


def coverage(answer: str) -> dict[str, bool]:
    low = answer.lower()
    return {label: any(k in low for k in keys) for label, keys in CHECKS}


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "Between internal combustion engine and EV which is better? "
        "The person does not travel a fixed kilometer distance everyday, "
        "instead sometimes short distances sometimes long distances."
    )
    normal = run_benchmark(prompt, "normal", persist=False)
    middleware = run_benchmark(prompt, "middleware", persist=False)

    print("=" * 70)
    print(f"NORMAL ok={normal.success} time={normal.metrics.duration_seconds:.2f}s "
          f"in/out={normal.input_tokens}/{normal.output_tokens}")
    print("=" * 70)
    print(normal.answer if normal.success else f"ERROR: {normal.error}")
    print()
    print("=" * 70)
    print(f"MIDDLEWARE ok={middleware.success} time={middleware.metrics.duration_seconds:.2f}s "
          f"in/out={middleware.input_tokens}/{middleware.output_tokens} "
          f"tasks={middleware.task_count} synth={middleware.synthesis_rounds}")
    if middleware.pipeline_result:
        for t in middleware.pipeline_result.task_results:
            print(f"  task {t.node_id} engine={t.engine} via={t.resolved_via} "
                  f"out={t.generated_tokens}")
    print("=" * 70)
    print(middleware.answer if middleware.success else f"ERROR: {middleware.error}")
    print()
    if normal.success and middleware.success:
        cn, cm = coverage(normal.answer), coverage(middleware.answer)
        print(f"{'deliverable':<28}{'normal':<10}{'middleware'}")
        for (label, _) in CHECKS:
            print(f"{label:<28}{str(cn[label]):<10}{str(cm[label])}")
        print(f"\nnormal coverage: {sum(cn.values())}/{len(CHECKS)}  "
              f"middleware coverage: {sum(cm.values())}/{len(CHECKS)}")
    for w in middleware.warnings:
        print("WARN:", w)


if __name__ == "__main__":
    main()
