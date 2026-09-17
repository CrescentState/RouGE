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

1. **`segment.py`** — splits the prompt into clauses on sentence boundaries and coordinating connectors (`and then`, `also`, `then`, `;`, …). Connector words inside quotes / code fences are protected so they never trigger a false split.
2. **`embedder.py`** — embeds each clause. Uses `sentence-transformers/all-MiniLM-L6-v2` when available; otherwise falls back to a deterministic lexical-hash embedder so the module runs and tests on any box.
3. **`entropy.py`** — labels each clause low/high entropy. Task-type prior: `summarization`/`extraction` → low, `code_generation`/`unknown` → high. Clauses below 4000 tokens stay at the prior; longer inputs escalate `low → high`.
4. **`aggregate.py`** — matches each clause against the **prototype bank** (`bank/prototype_bank.json`), computes best cosine similarity + margin, and returns a `Clause`. It then folds the clauses into one prompt-level label.

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

`task_type` ∈ `summarization | extraction | code_generation | unknown`
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

## Known limitations

* The UI (`ui.py`) still uses a placeholder classifier — this module is not wired into the running pipeline yet. `classify()` here is the intended replacement.
* Out-of-bank prompts (confidence below `CONF_FLOOR`) become `unknown` → high entropy → `complex`/`mixed_intent`, sending them down the expensive decomposition path.
* With the hash-fallback embedder, similarities are lower; the relaxed fallback thresholds above prevent the conservative path from triggering on CPU-only boxes.