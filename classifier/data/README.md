# Classifier Evaluation Data

`evaluation.jsonl` is a hand-authored, held-out evaluation set. It is intentionally separate from `bank/prototype_bank.json`; evaluation prompts must not be copied into the prototype bank.

Each line is one JSON object with:

- `id`: stable fixture identifier.
- `prompt`: raw input passed to `classifier.classify()`.
- `expected_task_types`: expected task type for each segmented clause, in order.
- `expected_aggregate`: expected `simple`, `complex`, or `mixed_intent` prompt label.
- `group`: evaluation slice used when reporting results.

The set covers summarization, extraction, code generation, mixed-intent prompts, out-of-bank prompts, sentence splitting, and protected quoted/code content. Examples are synthetic and contain no sensitive data.

The prototype bank contains representative training anchors. The evaluation set is for threshold calibration and regression measurement only.
