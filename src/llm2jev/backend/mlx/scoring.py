"""Zero-generation scoring with bounded prefix reuse and native MLX batching."""

from collections.abc import Mapping, Sequence
from typing import Any

from .batching import extract_cache, merge_caches
from .prefix_cache import PrefixCache, shared_prefixes


class TextScorer:
    def __init__(
        self, model: Any, mx: Any, make_prompt_cache: Any, *,
        batch_size: int, prefill_step_size: int, submission: str,
        max_cache_entries: int, max_cache_bytes: int,
    ) -> None:
        self.model = model
        self.mx = mx
        self.make_prompt_cache = make_prompt_cache
        # These models unpack cache.state directly as (keys, values). Native
        # BatchKVCache adds offset/padding state, so even a cache with merge()
        # cannot safely be handed to their forward implementations.
        self.batch_size = 1 if _requires_individual_cache(model) else batch_size
        self.prefill_step_size = prefill_step_size
        self.submission = submission
        self.prefix_cache = PrefixCache(max_cache_entries, max_cache_bytes)

    def clear_cache(self) -> None:
        self.prefix_cache.clear()

    def score(
        self, token_sequences: Sequence[Sequence[int]], *,
        no_token_id: int, yes_token_id: int,
    ) -> list[float]:
        if self.submission == "staged" and self.prefix_cache.max_entries and self.prefix_cache.max_bytes:
            # Store branch points explicitly: recurrent and full rotating
            # caches cannot be trimmed back from a completed candidate.
            for prefix in shared_prefixes([tokens[:-1] for tokens in token_sequences]):
                cache, matched = self._fetch(prefix)
                self._prefill([prefix[matched:]], [cache])
                self.prefix_cache.put(prefix, cache)

        probabilities = []
        for start in range(0, len(token_sequences), self.batch_size):
            batch = token_sequences[start : start + self.batch_size]
            caches, suffixes = [], []
            for tokens in batch:
                cache, matched = self._fetch(tokens[:-1])
                caches.append(cache)
                suffixes.append(tokens[matched:-1])
            self._prefill(suffixes, caches)
            for tokens, cache in zip(batch, caches):
                self.prefix_cache.put(tokens[:-1], cache)

            merged = merge_caches(caches) if len(caches) > 1 else None
            if merged is not None:
                logits = self.model(self.mx.array([[tokens[-1]] for tokens in batch]), cache=merged)
                for index in range(len(batch)):
                    probabilities.append(self._probability(logits[index, -1], no_token_id, yes_token_id))
            else:
                for tokens, cache in zip(batch, caches):
                    logits = self.model(self.mx.array([[tokens[-1]]]), cache=cache)
                    probabilities.append(self._probability(logits[0, -1], no_token_id, yes_token_id))
        return probabilities

    def _fetch(self, tokens: Sequence[int]) -> tuple[list[Any], int]:
        cache, matched = self.prefix_cache.fetch(tokens)
        return (self.make_prompt_cache(self.model) if cache is None else cache), matched

    def _prefill(self, suffixes: Sequence[Sequence[int]], caches: list[list[Any]]) -> None:
        positions = [0] * len(suffixes)
        while True:
            active = [index for index, tokens in enumerate(suffixes) if positions[index] < len(tokens)]
            if not active:
                return
            # Ragged prompts advance together only while they have real input
            # tokens. This needs neither input padding nor model-specific masks.
            width = min(self.prefill_step_size, *(len(suffixes[i]) - positions[i] for i in active))
            rows = [suffixes[i][positions[i] : positions[i] + width] for i in active]
            merged = merge_caches([caches[i] for i in active]) if len(active) > 1 else None
            if merged is not None:
                self.model(self.mx.array(rows), cache=merged)
                self.mx.eval([layer.state for layer in merged])
                for row, index in enumerate(active):
                    caches[index] = extract_cache(merged, row)
            else:
                for row, index in zip(rows, active):
                    self.model(self.mx.array([row]), cache=caches[index])
                    self.mx.eval([layer.state for layer in caches[index]])
            self.mx.clear_cache()
            for index in active:
                positions[index] += width

    def _probability(self, logits: Any, no_token_id: int, yes_token_id: int) -> float:
        binary = logits[self.mx.array([no_token_id, yes_token_id])]
        return float(self.mx.softmax(binary.astype(self.mx.float32), axis=-1)[1].item())


def _requires_individual_cache(model: Any) -> bool:
    incompatible_types = {"afm7", "gemma3n", "gemma3n_text"}
    model_types = [getattr(model, "model_type", None)]
    for attribute in ("config", "args"):
        config = getattr(model, attribute, None)
        model_types.append(
            config.get("model_type") if isinstance(config, Mapping)
            else getattr(config, "model_type", None)
        )
    return any(
        isinstance(model_type, str) and model_type in incompatible_types
        for model_type in model_types
    ) or type(model).__module__ in {"mlx_lm.models.afm7", "mlx_lm.models.gemma3n"}
