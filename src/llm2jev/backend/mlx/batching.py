"""Merge native MLX caches without changing model-specific cache semantics."""

from collections.abc import Sequence
from typing import Any


def _can_merge(layer: Any) -> bool:
    if not callable(getattr(layer, "merge", None)):
        return False
    # MLX's batch rotating cache cannot preserve pinned initial tokens, even
    # though RotatingKVCache.merge itself does not reject this configuration.
    if getattr(layer, "keep", 0):
        return False
    return all(_can_merge(child) for child in getattr(layer, "caches", ()))


def merge_caches(caches: Sequence[list[Any]]) -> list[Any] | None:
    """Return an independent batch cache, or None for a sequential fallback."""
    if not caches or not caches[0]:
        return None
    if any(len(cache) != len(caches[0]) for cache in caches):
        return None
    if not all(_can_merge(layer) for cache in caches for layer in cache):
        return None
    try:
        merged = [
            caches[0][index].merge([cache[index] for cache in caches])
            for index in range(len(caches[0]))
        ]
    except (NotImplementedError, ValueError):
        # Capability errors occur before a model call. Never retry a failed
        # forward pass, which may already have mutated cache/model state.
        return None
    if not all(callable(getattr(layer, "extract", None)) for layer in merged):
        return None
    return merged


def extract_cache(cache: list[Any], index: int) -> list[Any]:
    return [layer.extract(index) for layer in cache]
