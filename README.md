# RouGE — Route-Gated Execution

**RouGE** is a recursive, VRAM-gated routing middleware for heterogeneous quantized LLM inference on edge hardware.

It sits between the client and two locally hosted inference engines (an INT4 and an INT8 quantization of the same base model, served via llama.cpp). For every incoming prompt, RouGE:

1. **Classifies** each clause of the prompt by task type and entropy (simple / complex / mixed-intent), using prototype-bank embeddings with a length-aware escalation model.
2. **Decomposes** mixed-intent prompts into a dependency DAG of single-intent sub-tasks, parsed by the low-cost INT4 engine with JSON-schema-constrained output.
3. **Gates** deeper recursion on live GPU memory: if free VRAM (total − engine weights − current KV cache) is sufficient, pending sub-tasks are further decomposed mid-execution; otherwise they are routed monolithically.
4. **Executes** sub-tasks across both engines — independent low-entropy nodes in parallel on INT4, high-entropy nodes on INT8 — then stitches ordered node outputs into a single response.

Every component degrades gracefully: any classification, parsing, or engine failure falls back to monolithic INT8 routing. Nothing hangs, nothing errors out to the user.

## Architecture

```
                        ┌─────────────────────────────┐
                        │      FastAPI Gateway        │
  prompt ──────────────▶│  classify → segment → gate  │
                        │  schedule → stitch          │
                        └──────┬──────────────┬───────┘
                               │              │
                        ┌──────▼─────┐  ┌─────▼──────┐
                        │ INT4 engine│  │ INT8 engine│
                        │ (llama.cpp)│  │ (llama.cpp)│
                        └──────┬─────┘  └─────┬──────┘
                               └──────┬───────┘
                        ┌─────────────▼─────────────┐
                        │  PyNVML telemetry (CSV)   │
                        │  power · VRAM · temp · util│
                        └───────────────────────────┘
```

## Project layout

```
gateway/       FastAPI middleware: API, scheduler, stitching
classifier/    clause segmentation, prototype bank, entropy labeling
decomposer/    DAG parsing + validation + recursion gating
telemetry/     PyNVML memory/power logging
engines/       llama.cpp wrappers (INT4 / INT8 HTTP clients)
eval/          baselines, datasets, metrics, plots
configs/       engine ports, quant levels, thresholds, presets
```

## Component contracts

| Contract | Producer | Consumer |
|----------|----------|----------|
| 1 — Classifier output (clause labels JSON) | `classifier/` | decomposer, scheduler |
| 2 — Validated DAG (JSON) | `decomposer/` | scheduler |
| 3 — Engine HTTP API | llama.cpp servers | gateway |

## Quick start

```bash
# 1. Start engines (INT4 and INT8 quantizations of the same base model)
llama-server -m model-Q4_K_M.gguf --port 8081 -ngl 99 &
llama-server -m model-Q8_0.gguf   --port 8082 -ngl 99 &

# 2. Install middleware dependencies
pip install -r requirements.txt

# 3. Configure engines + thresholds
cp configs/default.yaml configs/local.yaml

# 4. Run the gateway
uvicorn gateway.api:app --port 8000
```

## Baselines

All four comparison conditions (Always-INT8, monolithic PAQ-style, fixed-depth decomposition, full gated recursion) are config presets over the same harness — see `configs/`.

## Status

Active development · 6-week build plan · target: workshop paper

## License

MIT (recommended — pending final decision)