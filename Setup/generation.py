"""Pure helpers for prompt formatting and context-window budgeting."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_CONTEXT_SIZE = 4096
MIN_CONTEXT_SIZE = 1024
MAX_CONTEXT_SIZE = 32768
DEFAULT_BATCH_SIZE = 512
MIN_BATCH_SIZE = 32
MAX_BATCH_SIZE = 2048
DEFAULT_MAX_TOKENS = 512
MAX_REQUESTED_TOKENS = 1024
DEFAULT_SAFETY_MARGIN_TOKENS = 128
MIN_USEFUL_OUTPUT_TOKENS = 32
QWEN_STOP_SEQUENCES = ["<|im_end|>"]


def resolve_context_size(value: str | int | None = None) -> int:
    """Resolve and validate the shared llama.cpp context size."""
    raw_value = os.getenv("ROUGE_CONTEXT_SIZE") if value is None else value
    if raw_value in (None, ""):
        return DEFAULT_CONTEXT_SIZE
    try:
        context_size = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("ROUGE_CONTEXT_SIZE must be an integer") from exc
    if not MIN_CONTEXT_SIZE <= context_size <= MAX_CONTEXT_SIZE:
        raise ValueError(
            f"ROUGE_CONTEXT_SIZE must be between {MIN_CONTEXT_SIZE} and "
            f"{MAX_CONTEXT_SIZE} tokens"
        )
    return context_size


CONTEXT_SIZE = resolve_context_size()


def resolve_batch_size(value: str | int | None = None) -> int:
    """Resolve llama.cpp's prompt-processing batch size."""
    raw_value = os.getenv("ROUGE_BATCH_SIZE") if value is None else value
    if raw_value in (None, ""):
        return DEFAULT_BATCH_SIZE
    try:
        batch_size = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("ROUGE_BATCH_SIZE must be an integer") from exc
    if not MIN_BATCH_SIZE <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(
            f"ROUGE_BATCH_SIZE must be between {MIN_BATCH_SIZE} and "
            f"{MAX_BATCH_SIZE} tokens"
        )
    return batch_size


BATCH_SIZE = resolve_batch_size()


class ContextWindowExceeded(ValueError):
    """Raised when a formatted prompt leaves no useful generation budget."""

    def __init__(self, input_tokens: int, context_size: int, safety_margin_tokens: int):
        self.input_tokens = input_tokens
        self.context_size = context_size
        self.safety_margin_tokens = safety_margin_tokens
        super().__init__(
            f"formatted prompt uses {input_tokens} tokens in a {context_size}-token "
            f"context with a {safety_margin_tokens}-token safety margin"
        )


@dataclass(frozen=True)
class TokenBudget:
    input_tokens: int
    context_size: int
    safety_margin_tokens: int
    available_output_tokens: int
    requested_max_tokens: int
    effective_max_tokens: int


def format_qwen_prompt(prompt: str, system_prompt: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def count_formatted_tokens(model, formatted_prompt: str) -> int:
    """Mirror llama.cpp completion tokenization, including configured BOS/EOS."""
    tokens = model.tokenize(
        formatted_prompt.encode("utf-8"),
        add_bos=False,
        special=True,
    )
    if model._model.add_bos_token() and model.token_bos() != -1:
        tokens.insert(0, model.token_bos())
    if model._model.add_eos_token() and model.token_eos() != -1:
        tokens.append(model.token_eos())
    return len(tokens)


def calculate_token_budget(
    *,
    input_tokens: int,
    requested_max_tokens: int,
    context_size: int = CONTEXT_SIZE,
    safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS,
) -> TokenBudget:
    available = context_size - input_tokens - safety_margin_tokens
    if available < MIN_USEFUL_OUTPUT_TOKENS:
        raise ContextWindowExceeded(input_tokens, context_size, safety_margin_tokens)

    return TokenBudget(
        input_tokens=input_tokens,
        context_size=context_size,
        safety_margin_tokens=safety_margin_tokens,
        available_output_tokens=available,
        requested_max_tokens=requested_max_tokens,
        effective_max_tokens=min(requested_max_tokens, available),
    )
