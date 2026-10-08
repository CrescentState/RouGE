import json
from pathlib import Path

from classifier.segment import segment


CLASSIFIER_DIR = Path(__file__).resolve().parents[1]
ALLOWED_TASK_TYPES = {
    "summarization",
    "extraction",
    "code_generation",
    "creative_writing",
    "unknown",
}
ALLOWED_AGGREGATES = {"simple", "complex", "mixed_intent"}


def test_prototype_bank_is_balanced_and_has_no_duplicates():
    bank = json.loads((CLASSIFIER_DIR / "bank" / "prototype_bank.json").read_text())

    assert set(bank) == {
        "summarization",
        "extraction",
        "code_generation",
        "creative_writing",
    }
    assert {len(examples) for examples in bank.values()} == {35}
    for examples in bank.values():
        normalized = [example.strip().lower() for example in examples]
        assert len(normalized) == len(set(normalized))


def test_evaluation_data_has_valid_unique_records():
    path = CLASSIFIER_DIR / "data" / "evaluation.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    assert len(records) >= 50
    assert len({record["id"] for record in records}) == len(records)
    assert {record["group"] for record in records} == {
        "summarization",
        "extraction",
        "code_generation",
        "creative_writing",
        "mixed_intent",
        "unknown",
        "segmentation",
    }

    for record in records:
        assert record["prompt"].strip()
        assert record["expected_aggregate"] in ALLOWED_AGGREGATES
        assert set(record["expected_task_types"]) <= ALLOWED_TASK_TYPES
        assert len(segment(record["prompt"])) == len(record["expected_task_types"])


def test_evaluation_prompts_are_held_out_from_prototype_bank():
    bank = json.loads((CLASSIFIER_DIR / "bank" / "prototype_bank.json").read_text())
    prototype_text = {
        example.strip().lower()
        for examples in bank.values()
        for example in examples
    }
    records = [
        json.loads(line)
        for line in (CLASSIFIER_DIR / "data" / "evaluation.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert not {
        record["prompt"].strip().lower() for record in records
    } & prototype_text
