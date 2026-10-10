import json
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from classifier import classify
from classifier.segment import segment

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
    r = classify("Debate whether numbers exist independently of observers.")
    assert r.clauses[0].task_type == "unknown"

def test_creative_writing_is_simple():
    r = classify("Tell me a story about a dragon knight in medieval France.")
    assert r.aggregate_label == "simple"
    assert r.clauses[0].task_type == "creative_writing"

def test_story_then_extract_is_mixed_intent():
    r = classify("Give me a story on discipline and extract the key points")
    assert [clause.task_type for clause in r.clauses] == [
        "creative_writing",
        "extraction",
    ]
    assert r.aggregate_label == "mixed_intent"

def test_contract_serializable():
    r = classify("Extract all dates from this text. Then summarize it.")
    json.loads(r.model_dump_json())

def test_plain_and_before_task_verb_splits():
    assert segment("Write a story and extract its lessons") == [
        "Write a story",
        "extract its lessons",
    ]

def test_plain_and_in_noun_phrase_does_not_split():
    assert segment("Explain the relationship between research and development.") == [
        "Explain the relationship between research and development."
    ]

def test_plain_and_inside_quotes_does_not_split():
    assert segment('Explain the phrase "write and extract the result" briefly.') == [
        'Explain the phrase "write and extract the result" briefly.'
    ]

def test_and_before_point_splits_story_and_key_points():
    assert segment(
        "Generate a short story about a family and point out the key points of the story"
    ) == [
        "Generate a short story about a family",
        "point out the key points of the story",
    ]

def test_story_plus_key_points_is_mixed_intent():
    r = classify(
        "Generate a short story about a family and point out the key points of the story"
    )
    assert [clause.task_type for clause in r.clauses] == [
        "creative_writing",
        "extraction",
    ]
    assert r.aggregate_label == "mixed_intent"

def test_instruction_head_beats_pasted_attachment():
    r = classify(
        "Extract all dates from this text: kickoff 2026-03-01, review March 15 2026."
    )
    assert r.clauses[0].task_type == "extraction"
    assert r.aggregate_label == "simple"

def test_summarize_head_with_article_body():
    r = classify(
        "Summarize this article in five bullet points: The council met Tuesday."
    )
    assert r.clauses[0].task_type == "summarization"

def test_url_colon_does_not_split_instruction():
    assert segment("visit http://example.com for details") == [
        "visit http://example.com for details"
    ]
