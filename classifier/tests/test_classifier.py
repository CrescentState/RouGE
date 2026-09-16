import json
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from classifier import classify

def test_simple():
    r = classify("Summarize this article in five bullet points.")
    assert r.aggregate_label == "simple"
    assert r.clauses[0].task_type == "summarization"

def test_mixed_intent():
    r = classify("Summarize this server log and also write a python script to fix the errors.")
    assert r.aggregate_label == "mixed_intent"
    assert len(r.clauses) == 2

def test_no_false_split_in_quotes():
    r = classify('Write code that prints "and then again" to the console.')
    assert len(r.clauses) == 1

def test_unknown_is_conservative():
    r = classify("Tell me a story about a dragon knight in medieval France.")
    assert r.clauses[0].task_type in ("unknown", "code_generation", "summarization", "extraction")

def test_contract_serializable():
    r = classify("Extract all dates from this text. Then summarize it.")
    json.loads(r.model_dump_json())
