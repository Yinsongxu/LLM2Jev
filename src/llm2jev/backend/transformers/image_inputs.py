"""Prepare image prompts and load image inputs for Transformers."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from ...inference.prompt import ChatPrompt
from ..tokenization import _apply_chat_template


def has_images(prompts: Sequence[ChatPrompt]) -> bool:
    return any(
        part["type"] == "image_url"
        for prompt in prompts
        for message in prompt
        if isinstance(message["content"], list)
        for part in message["content"]
    )


def prepare_image_prompts(
    processor: Any,
    prompts: Sequence[ChatPrompt],
    *,
    enable_thinking: bool,
) -> tuple[list[str], list[list[str]]]:
    rendered = []
    images = []
    for prompt in prompts:
        messages = []
        prompt_images = []
        for message in prompt:
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            template_parts = []
            for part in content:
                if part["type"] == "image_url":
                    source = part["image_url"]["url"]
                    if not source.startswith(("http://", "https://", "file://", "data:")):
                        source = str(Path(source).resolve())
                    prompt_images.append(source)
                    template_parts.append({"type": "image"})
                else:
                    template_parts.append(dict(part))
            messages.append({"role": message["role"], "content": template_parts})
        rendered.append(_apply_chat_template(
            processor, tuple(messages), enable_thinking=enable_thinking,
        ))
        images.append(prompt_images)
    return rendered, images


def load_transformers_images(sources: Sequence[Sequence[str]]) -> list[list[Any]]:
    from transformers.image_utils import load_image

    loaded: dict[str, Any] = {}
    batches = []
    for row in sources:
        images = []
        for source in row:
            if source.startswith("file://"):
                parsed = urlparse(source)
                source = url2pathname(f"//{parsed.netloc}{parsed.path}")
            if source not in loaded:
                loaded[source] = load_image(source)
            images.append(loaded[source])
        batches.append(images)
    return batches
