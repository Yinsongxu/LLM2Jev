"""Small deterministic MLX doubles; importing these needs no ML dependencies."""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any
from unittest.mock import Mock


class FakeArray:
    def __init__(self, values: Any, *, dtype: str | None = None) -> None:
        self.values = _array_values(values)
        self.dtype = dtype

    def __getitem__(self, indices: Any) -> FakeArray:
        if not isinstance(indices, tuple):
            indices = (indices,)
        values = self.values
        for index in indices:
            if isinstance(index, FakeArray):
                index = index.values
            values = [values[item] for item in index] if isinstance(index, list) else values[index]
        return FakeArray(values, dtype=self.dtype)

    def astype(self, dtype: str) -> FakeArray:
        return FakeArray(self.values, dtype=dtype)

    def item(self) -> Any:
        return self.values

    def tolist(self) -> Any:
        return self.values

    def __float__(self) -> float:
        return float(self.values)


def _array_values(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_array_values(item) for item in value]
    return value


class FakeCacheLayer:
    def __init__(self) -> None:
        self.tokens: list[int] = []

    @property
    def state(self) -> tuple[FakeArray, ...]:
        return (FakeArray(list(self.tokens)),)

    @property
    def nbytes(self) -> int:
        return len(self.tokens) * 4

    def is_trimmable(self) -> bool:
        return True

    def trim(self, count: int) -> int:
        count = min(count, len(self.tokens))
        if count:
            del self.tokens[-count:]
        return count

    @classmethod
    def merge(cls, layers: list[FakeCacheLayer]) -> FakeBatchCacheLayer:
        return FakeBatchCacheLayer([list(layer.tokens) for layer in layers])


class FakeBatchCacheLayer:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    @property
    def state(self) -> tuple[FakeArray, ...]:
        return (FakeArray([list(row) for row in self.rows]),)

    @property
    def nbytes(self) -> int:
        return sum(map(len, self.rows)) * 4

    def extract(self, index: int) -> FakeCacheLayer:
        layer = FakeCacheLayer()
        layer.tokens = list(self.rows[index])
        return layer


class FakeTokenizer:
    def __init__(
        self,
        prompt_tokens: list[list[int]] | None = None,
        labels: dict[str, list[int]] | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens or [[11, 12, 13]]
        self.labels = {"yes": [7], "no": [9]} if labels is None else labels
        self.template_calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        self.encode_calls: list[tuple[str, bool]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.template_calls.append((messages, kwargs))
        return f"rendered prompt {len(self.template_calls) - 1}"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.encode_calls.append((text, add_special_tokens))
        if text in self.labels:
            return list(self.labels[text])
        index = int(text.removeprefix("rendered prompt "))
        return list(self.prompt_tokens[index % len(self.prompt_tokens)])


class FakeModel:
    def __init__(
        self,
        logits_by_token: dict[int, tuple[float, float]] | None = None,
        logits_by_sequence: dict[tuple[int, ...], tuple[float, float]] | None = None,
    ) -> None:
        self.logits_by_token = logits_by_token or {}
        self.logits_by_sequence = logits_by_sequence or {}
        self.calls: list[tuple[list[list[int]], list[FakeCacheLayer]]] = []
        self.contexts: list[list[list[int]]] = []
        self.eval = Mock()

    def __call__(self, tokens: FakeArray, *, cache: list[FakeCacheLayer]) -> FakeArray:
        values = tokens.tolist()
        self.calls.append((values, cache))
        for layer in cache:
            histories = layer.rows if isinstance(layer, FakeBatchCacheLayer) else [layer.tokens]
            if len(histories) != len(values):
                raise AssertionError("every input row requires its own cache state")
            for history, row in zip(histories, values):
                history.extend(row)
        histories = cache[0].rows if isinstance(cache[0], FakeBatchCacheLayer) else [cache[0].tokens]
        self.contexts.append([list(history) for history in histories])
        batches = []
        for history, input_row in zip(histories, values):
            rows = []
            prefix_length = len(history) - len(input_row)
            for index, token in enumerate(input_row):
                context = tuple(history[:prefix_length + index + 1])
                no, yes = self.logits_by_sequence.get(
                    context, self.logits_by_token.get(token, (0.0, 0.0)),
                )
                row = [10000.0] + [0.0] * 9
                row[9], row[7] = no, yes
                rows.append(row)
            batches.append(rows)
        return FakeArray(batches, dtype="float16")


class FakeMLX:
    def __init__(
        self,
        *,
        prompt_tokens: list[list[int]] | None = None,
        labels: dict[str, list[int]] | None = None,
        logits_by_token: dict[int, tuple[float, float]] | None = None,
        logits_by_sequence: dict[tuple[int, ...], tuple[float, float]] | None = None,
    ) -> None:
        self.tokenizer = FakeTokenizer(prompt_tokens, labels)
        self.model = FakeModel(logits_by_token, logits_by_sequence)
        self.caches: list[list[FakeCacheLayer]] = []
        self.softmax_inputs: list[FakeArray] = []
        self.core = ModuleType("mlx.core")
        self.core.float32 = "float32"
        self.core.array = Mock(side_effect=lambda values: FakeArray(values))
        self.core.softmax = Mock(side_effect=self.softmax)
        self.core.eval = Mock()
        self.core.clear_cache = Mock()
        self.core.synchronize = Mock()
        self.load = Mock(return_value=(self.model, self.tokenizer))
        self.make_prompt_cache = Mock(side_effect=self.new_cache)

        mlx = ModuleType("mlx")
        mlx.__path__ = []
        mlx.core = self.core
        mlx_lm = ModuleType("mlx_lm")
        mlx_lm.__path__ = []
        mlx_lm.load = self.load
        models = ModuleType("mlx_lm.models")
        models.__path__ = []
        cache_module = ModuleType("mlx_lm.models.cache")
        cache_module.make_prompt_cache = self.make_prompt_cache
        models.cache = cache_module
        mlx_lm.models = models
        self.modules = {
            "mlx": mlx,
            "mlx.core": self.core,
            "mlx_lm": mlx_lm,
            "mlx_lm.models": models,
            "mlx_lm.models.cache": cache_module,
        }

    def new_cache(self, model: FakeModel) -> list[FakeCacheLayer]:
        if model is not self.model:
            raise AssertionError("cache must belong to the loaded model")
        cache = [FakeCacheLayer(), FakeCacheLayer()]
        self.caches.append(cache)
        return cache

    def softmax(self, values: FakeArray, axis: int = -1) -> FakeArray:
        if axis != -1:
            raise AssertionError("binary logits must be normalized on their last axis")
        self.softmax_inputs.append(values)
        peak = max(values.values)
        masses = [math.exp(value - peak) for value in values.values]
        total = sum(masses)
        return FakeArray([value / total for value in masses], dtype=values.dtype)
