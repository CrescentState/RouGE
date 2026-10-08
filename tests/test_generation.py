from __future__ import annotations

import pytest

from Setup.generation import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT_SIZE,
    ContextWindowExceeded,
    QWEN_STOP_SEQUENCES,
    calculate_token_budget,
    count_formatted_tokens,
    format_qwen_prompt,
    resolve_context_size,
    resolve_batch_size,
)


def test_context_size_defaults_to_4096_and_accepts_override():
    assert DEFAULT_CONTEXT_SIZE == 4096
    assert resolve_context_size("") == 4096
    assert resolve_context_size("2048") == 2048
    assert resolve_context_size(8192) == 8192


def test_batch_size_defaults_to_512_and_accepts_override():
    assert DEFAULT_BATCH_SIZE == 512
    assert resolve_batch_size("") == 512
    assert resolve_batch_size("256") == 256
    assert resolve_batch_size(128) == 128


@pytest.mark.parametrize("value", ["invalid", "16", "4096"])
def test_batch_size_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        resolve_batch_size(value)


@pytest.mark.parametrize("value", ["invalid", "512", "65536"])
def test_context_size_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        resolve_context_size(value)


def test_token_budget_clamps_requested_output():
    budget = calculate_token_budget(
        input_tokens=1500,
        requested_max_tokens=512,
        context_size=2048,
        safety_margin_tokens=128,
    )

    assert budget.available_output_tokens == 420
    assert budget.effective_max_tokens == 420
    assert (
        budget.input_tokens
        + budget.effective_max_tokens
        + budget.safety_margin_tokens
        <= budget.context_size
    )


def test_token_budget_rejects_prompt_without_useful_output_space():
    with pytest.raises(ContextWindowExceeded):
        calculate_token_budget(
            input_tokens=1900,
            requested_max_tokens=512,
            context_size=2048,
            safety_margin_tokens=128,
        )


def test_4096_context_clamps_generation_near_boundary():
    budget = calculate_token_budget(
        input_tokens=3500,
        requested_max_tokens=512,
        context_size=4096,
        safety_margin_tokens=128,
    )

    assert budget.effective_max_tokens == 468
    assert budget.input_tokens + budget.effective_max_tokens + 128 == 4096


def test_qwen_format_includes_system_user_and_assistant_boundaries():
    formatted = format_qwen_prompt("hello", "be concise")

    assert "<|im_start|>system\nbe concise<|im_end|>" in formatted
    assert "<|im_start|>user\nhello<|im_end|>" in formatted
    assert formatted.endswith("<|im_start|>assistant\n")
    assert QWEN_STOP_SEQUENCES == ["<|im_end|>"]


def test_exact_count_includes_model_bos_and_eos_tokens():
    class FakeModelConfig:
        @staticmethod
        def add_bos_token():
            return True

        @staticmethod
        def add_eos_token():
            return True

    class FakeModel:
        _model = FakeModelConfig()

        @staticmethod
        def tokenize(text, add_bos, special):
            assert text == b"formatted"
            assert add_bos is False
            assert special is True
            return [10, 11, 12]

        @staticmethod
        def token_bos():
            return 1

        @staticmethod
        def token_eos():
            return 2

    assert count_formatted_tokens(FakeModel(), "formatted") == 5
