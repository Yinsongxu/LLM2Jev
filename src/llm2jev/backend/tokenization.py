from __future__ import annotations

from typing import TYPE_CHECKING, Any
from os.path import commonprefix

if TYPE_CHECKING:
    from ..inference.prompt import ChatPrompt


def _generated_token_texts(tokenizer: Any, token_ids: list[int]) -> tuple[str, ...]:
    """Decode with context so whitespace and special tokens retain their alignment."""
    full_text = tokenizer.decode(
        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    pieces, previous = [], 0
    for end in range(1, len(token_ids) + 1):
        text = tokenizer.decode(
            token_ids[:end], skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        # Byte-fallback tokens can temporarily decode to a replacement character.
        end_offset = max(previous, len(commonprefix((full_text, text))))
        pieces.append(full_text[previous:end_offset])
        previous = end_offset
    return tuple(pieces)


def _single_token_id(tokenizer: Any, label: str, field: str) -> int:
    token_ids = tokenizer.encode(label, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{field} must encode to exactly one token, got {token_ids}")
    return token_ids[0]


def _apply_chat_template(
    tokenizer: Any,
    prompt: ChatPrompt,
    *,
    enable_thinking: bool,
) -> str:
    return tokenizer.apply_chat_template(
        list(prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
