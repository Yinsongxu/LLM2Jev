"""Explicit image evidence inside otherwise arbitrary JSON content."""

from collections.abc import Mapping
from typing import Any, TypeGuard


def is_multimodal(value: object) -> TypeGuard[Mapping[str, Any]]:
    return isinstance(value, Mapping) and value.get("type") == "multimodal"


def contains_image(value: object) -> bool:
    return is_multimodal(value) and any(
        part["type"] == "image_url" for part in value["content"]
    )


def validate_multimodal(value: object) -> None:
    if not is_multimodal(value):
        return
    if set(value) != {"type", "content"}:
        raise ValueError("multimodal must contain exactly type and content")
    content = value["content"]
    if not isinstance(content, list) or not content:
        raise ValueError("multimodal content must be a non-empty list")
    for part in content:
        if not isinstance(part, Mapping):
            raise ValueError("multimodal content parts must be objects")
        if part.get("type") == "text":
            if set(part) != {"type", "text"} or not isinstance(part["text"], str):
                raise ValueError("text parts must contain type and a text string")
        elif part.get("type") == "image_url":
            image = part.get("image_url")
            if set(part) != {"type", "image_url"} or not isinstance(image, Mapping):
                raise ValueError("image_url parts must contain type and an image_url object")
            if set(image) != {"url"} or not isinstance(image["url"], str) or not image["url"].strip():
                raise ValueError("image_url must contain a non-empty url string")
        else:
            raise ValueError("multimodal content supports only text and image_url parts")
