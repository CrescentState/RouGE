# RouGE Classifier Module: Clause Labeling (Contract 1)

## Overview

The classifier converts a raw user prompt into structured per-clause labels — task type, confidence, margin, entropy — plus a single prompt-level label (`simple` / `complex` / `mixed_intent`). This is Contract 1: its JSON output is the input to the DAG decomposition and routing stages.

## Pipeline

```
prompt ──▶ segment.py ──▶ embedder.py ──▶ aggregate.py ──▶ ClassificationResult
             (clauses)      (vector)       (label + Clause[])
                              │
                              └─ entropy.py (task prior + length escalation)
```

1. **`segment.py`** — splits the prompt into clauses on sentence boundaries and coordinating connectors (`and then`, `also`, `then`, `;`, …). Plain `and` is a boundary only when it introduces a recognized task verb, so `write a story and extract its lessons` splits while `research and development` does not. Connector words inside quotes / code fences are protected.
2. **`embedder.py`** — embeds each clause. Uses `sentence-transformers/all-MiniLM-L6-v2` when available; otherwise falls back to a deterministic lexical-hash embedder so the module runs and tests on any box.
3. **`entropy.py`** — labels each clause low/high entropy. Task-type prior: `summarization`/`extraction`/`creative_writing` → low, `code_generation`/`unknown` → high. Clauses below 4000 tokens stay at the prior; longer inputs escalate `low → high`.
4. **`aggregate.py`** — matches each clause against the **prototype bank** (`bank/prototype_bank.json`), computes best cosine similarity + margin, and returns a `Clause`. It then folds the clauses into one prompt-level label.

## Bank and evaluation data

The prototype bank contains 49 diverse anchors for each supported learned task type: summarization, extraction, code generation, and creative writing. `unknown` is intentionally not a prototype category; it is produced when confidence or margin falls below the configured thresholds.

`data/evaluation.jsonl` contains held-out fixtures covering all four learned task types, mixed intent, unknown prompts, and segmentation edge cases. Each record includes the expected clause task types and aggregate label. See `data/README.md` for the schema. Asset tests ensure the bank remains balanced, fixture IDs remain unique, segmentation expectations are valid, and evaluation prompts do not leak into the prototype bank.

## Label rules (`aggregate.py`)

| Clause set | Prompt label |
|---|---|
| zero clauses (empty prompt) | `simple` |
| single clause | its own entropy |
| all same task type / same entropy | `simple` (low) or `complex` (high) |
| different task types or entropy mix | `mixed_intent` |

Unknown/mismatched clauses are labeled conservatively **high** entropy.

## Contract 1 schema (`contracts.py`)

```json
{
  "prompt_id": "adhoc",
  "clauses": [
    {
      "id": 1,
      "text": "Extract all dates from this text.",
      "task_type": "extraction",
      "confidence": 0.95,
      "margin": 0.34,
      "token_count": 8,
      "base_entropy": "low"
    }
  ],
  "aggregate_label": "simple",
  "segmentation_method": "sentence_split_conjunction"
}
```

`task_type` ∈ `summarization | extraction | code_generation | creative_writing | unknown`
`base_entropy` ∈ `low | high`
`aggregate_label` ∈ `simple | complex | mixed_intent`

(The example above is illustrative schema shape only; measured `confidence` for this exact prototype prompt is ~0.95.)

## Usage

```python
from classifier import classify
result = classify("Summarize the log and fix the errors.", prompt_id="demo")
print(result.aggregate_label, [c.task_type for c in result.clauses])
```

## Tuning constants

| Constant | Value | Notes |
|---|---|---|
| `CONF_FLOOR` | 0.45 | below this, `task_type` becomes `unknown` |
| `MARGIN` | 0.08 | best − second best gap floor |
| `FALLBACK_CONF_FLOOR` / `FALLBACK_MARGIN` | 0.20 / 0.01 | used when the hash-fallback embedder is active |
| `LENGTH_ESCALATION_TOKENS` | 4000 | tokens at which `low` escalates to `high` |
| `estimate_tokens()` | `len(text) // 4` | character-based token estimate |

Tune the prototype examples in `bank/prototype_bank.json`; recalibrate `CONF_FLOOR` / `MARGIN` against evaluation fixtures.

## Tests

```bash
uv run pytest classifier/tests -q
```

Covers: simple label, mixed-intent splitting, no false split inside quotes, conservative unknown handling, and JSON serializability of the output.

## Integration and known limitations

* `pipeline.py` uses `classify()` for root prompts and for every generated DAG node. Its aggregate labels are the canonical routing labels used by the decomposer.
* Out-of-bank prompts (confidence below `CONF_FLOOR`) become `unknown` → high entropy → `complex`/`mixed_intent`, sending them down the expensive decomposition path.
* With the hash-fallback embedder, similarities are lower; the relaxed fallback thresholds above prevent the conservative path from triggering on CPU-only boxes.
