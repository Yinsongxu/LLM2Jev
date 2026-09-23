"""Bounded, isolated prompt snapshots without importing the MLX runtime."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, kw_only=True)
class _Entry:
    cache: list[Any]
    nbytes: int


class PrefixCache:
    """Store model cache snapshots indexed by their logical token prefixes.

    Callers evaluate native cache state before insertion and include any other
    evidence, such as an image digest, in ``namespace``. Stored snapshots and
    returned caches never share mutable objects with callers. A zero limit
    disables storage. This helper requires external synchronization if shared
    across threads.
    """

    def __init__(
        self,
        max_entries: int = 32,
        max_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        for name, value in (("max_entries", max_entries), ("max_bytes", max_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[tuple[Hashable, tuple[int, ...]], _Entry] = OrderedDict()
        self._nbytes = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def clear(self) -> None:
        self._entries.clear()
        self._nbytes = 0

    def put(
        self,
        tokens: Sequence[int],
        cache: Sequence[Any],
        *,
        namespace: Hashable = None,
    ) -> None:
        if not tokens or not cache or self.max_entries == 0 or self.max_bytes == 0:
            return
        nbytes = sum(layer.nbytes for layer in cache)
        if nbytes > self.max_bytes:
            return
        key = (namespace, tuple(tokens))
        entry = _Entry(cache=deepcopy(list(cache)), nbytes=nbytes)
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._nbytes -= previous.nbytes
        self._entries[key] = entry
        self._nbytes += nbytes
        while len(self._entries) > self.max_entries or self._nbytes > self.max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self._nbytes -= evicted.nbytes

    def fetch(
        self,
        tokens: Sequence[int],
        *,
        namespace: Hashable = None,
    ) -> tuple[list[Any] | None, int]:
        """Return the longest usable prefix and a private, possibly trimmed copy.

        A cache with recurrent state or an exhausted sliding window may not be
        trimmable. Such entries remain reusable whenever their entire token
        sequence is a prefix of the requested tokens.
        """
        query = tuple(tokens)
        if not query:
            return None, 0
        exact_key = (namespace, query)
        exact = self._entries.get(exact_key)
        if exact is not None:
            result = deepcopy(exact.cache)
            self._entries.move_to_end(exact_key)
            return result, len(query)

        candidates = []
        # Consider equally useful entries in most-recently-used order.
        for key, entry in reversed(self._entries.items()):
            entry_namespace, prefix = key
            if entry_namespace != namespace:
                continue
            common = _common_length(query, prefix)
            if common:
                candidates.append((common, key, entry))
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)

        for common, key, entry in candidates:
            to_trim = len(key[1]) - common
            if to_trim:
                result = _trimmed_copy(entry.cache, to_trim)
                if result is None:
                    continue
            else:
                result = deepcopy(entry.cache)
            self._entries.move_to_end(key)
            return result, common
        return None, 0


def shared_prefixes(
    token_sequences: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Return distinct adjacent common prefixes, shortest first.

    Sorting sequences exposes every branching prefix without a node per token.
    Duplicates and sequences that are strict prefixes of another are retained
    while comparing, so their complete shared token path can also be reused.
    Callers must remove any token they reserve for a final scoring pass.
    """
    sequences = sorted(tuple(tokens) for tokens in token_sequences)
    prefixes = set()
    for left, right in zip(sequences, sequences[1:]):
        common = _common_length(left, right)
        if common:
            prefixes.add(left[:common])
    return tuple(sorted(prefixes, key=lambda prefix: (len(prefix), prefix)))


def _common_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def _trimmed_copy(cache: list[Any], count: int) -> list[Any] | None:
    try:
        if not all(
            callable(getattr(layer, "is_trimmable", None))
            and callable(getattr(layer, "trim", None))
            and layer.is_trimmable()
            for layer in cache
        ):
            return None
        result = deepcopy(cache)
        # A partial trim cannot be treated as a valid prefix. The original
        # entry remains intact even when a later layer cannot trim enough.
        for layer in result:
            trimmed = layer.trim(count)
            if isinstance(trimmed, bool) or trimmed != count:
                return None
        return result
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        # Cache implementations may explicitly reject trimming at runtime.
        # Falling back to another snapshot leaves all stored entries intact.
        return None
