"""Prefill-only MLX-VLM scoring with conservative model capability boundaries."""

from __future__ import annotations

import hashlib
import inspect
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ...core.response import Usage
from ...inference.prompt import ChatPrompt
from ..base import BinaryBackendOutput
from ..tokenization import _apply_chat_template
from .batching import extract_cache, merge_caches
from .prefix_cache import PrefixCache, shared_prefixes


# These models return sequence-aligned embeddings and complete, explicit mRoPE
# positions. Other VLMs may expand tokens or carry cross-attention state, so they
# use an independent, complete prefill instead of guessing their cache layout.
_OPTIMIZED_MODEL_TYPES = frozenset({"qwen2_vl", "qwen2_5_vl"})


@dataclass(frozen=True, kw_only=True)
class _VisionPrompt:
    tokens: tuple[int, ...]
    embeddings: Any
    kwargs: dict[str, Any]
    namespace: tuple[bytes, ...]
    optimized: bool


class MLXVisionScorer:
    """Score causal VLM prompts without sampling or decoding any output token.

    Qwen2-VL and Qwen2.5-VL support chunking, shared prefixes and equally sized
    batches. Other causal VLMs use their public embedding and language-model
    interfaces for a full prefill. Image features are reused within a request
    through the model's public encoding hooks, or Qwen's merged embeddings.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        enable_thinking: bool,
        batch_size: int,
        prefill_step_size: int,
        submission: str,
        max_cache_entries: int,
        max_cache_bytes: int,
    ) -> None:
        import mlx.core as mx
        from mlx_vlm.models.base import BaseImageProcessor
        from mlx_vlm.models.cache import make_prompt_cache
        from mlx_vlm.utils import load_image, prepare_inputs, should_add_special_tokens

        if not callable(getattr(model, "get_input_embeddings", None)) or not callable(
            getattr(model, "language_model", None),
        ):
            raise ValueError("MLX multimodal scoring requires a causal language VLM")
        self.model = model
        self.processor = processor
        self._mx = mx
        self._make_cache = make_prompt_cache
        self._load_image = load_image
        self._prepare_inputs = prepare_inputs
        self._base_image_processor = BaseImageProcessor
        self._model_type = getattr(model.config, "model_type", "")
        self._add_special_tokens = should_add_special_tokens(self._model_type, processor)
        self.enable_thinking = enable_thinking
        self.batch_size = batch_size
        self.prefill_step_size = prefill_step_size
        self.submission = submission
        self.max_cache_entries = max_cache_entries
        self.max_cache_bytes = max_cache_bytes
        self._prefix_cache = PrefixCache(
            max_entries=max_cache_entries, max_bytes=max_cache_bytes,
        )

    def clear_cache(self) -> None:
        self._prefix_cache.clear()

    def score(
        self,
        prompts: Sequence[ChatPrompt],
        *,
        yes_token_id: int,
        no_token_id: int,
    ) -> BinaryBackendOutput:
        if not prompts:
            raise ValueError("prompts must not be empty")
        prepared = self._prepare(prompts)
        if self.submission == "staged":
            self._warm_prefixes(prepared)

        probabilities: list[float | None] = [None] * len(prepared)
        groups: dict[Any, list[tuple[int, _VisionPrompt, Any, int]]] = defaultdict(list)
        for index, prompt in enumerate(prepared):
            if not prompt.optimized:
                probabilities[index] = self._score_complete(
                    prompt, yes_token_id=yes_token_id, no_token_id=no_token_id,
                )
                continue
            cache, offset = self._prefix_cache.fetch(
                prompt.tokens[:-1], namespace=prompt.namespace,
            )
            if cache is None:
                cache = self._make_cache(self.model.language_model)
            # Equal lengths avoid padding-dependent attention or mRoPE changes.
            metadata_shape = tuple(
                (name, value.shape[:-1]) for name, value in sorted(prompt.kwargs.items())
            )
            key = (len(prompt.tokens), offset, metadata_shape)
            groups[key].append((index, prompt, cache, offset))

        for group in groups.values():
            for start in range(0, len(group), self.batch_size):
                batch = group[start : start + self.batch_size]
                caches = [row[2] for row in batch]
                merged = merge_caches(caches) if len(batch) > 1 else caches[0]
                if merged is None:
                    for row in batch:
                        result = self._score_batch(
                            [row], row[2], yes_token_id, no_token_id,
                        )
                        probabilities[row[0]] = result[0]
                else:
                    result = self._score_batch(batch, merged, yes_token_id, no_token_id)
                    for row, value in zip(batch, result):
                        probabilities[row[0]] = value

        if any(value is None for value in probabilities):
            raise RuntimeError("MLX-VLM did not score every prompt")
        return BinaryBackendOutput(
            yes_probabilities=[float(value) for value in probabilities],
            usage=Usage(
                input_tokens=sum(prompt.embeddings.shape[1] for prompt in prepared),
                output_tokens=0,
            ),
        )

    def _prepare(self, prompts: Sequence[ChatPrompt]) -> list[_VisionPrompt]:
        loaded: dict[str, Any] = {}
        digests: dict[str, bytes] = {}
        features: OrderedDict[tuple[bytes, ...], Any] = OrderedDict()
        feature_bytes = 0
        prepared = []
        for prompt in prompts:
            messages, sources = _image_messages(prompt)
            if len(sources) > 1 and isinstance(
                getattr(self.processor, "image_processor", None), self._base_image_processor,
            ):
                # MLX-VLM's legacy processor path only inserts one image marker.
                raise ValueError("this MLX-VLM image processor supports only one image per prompt")
            images = []
            for source in sources:
                if source not in loaded:
                    loaded[source] = self._load_image(source)
                    digests[source] = _image_digest(loaded[source])
                images.append(loaded[source])
            namespace = tuple(digests[source] for source in sources)
            rendered = _apply_chat_template(
                self.processor, messages, enable_thinking=self.enable_thinking,
            )
            inputs = dict(self._prepare_inputs(
                self.processor, images=images or None, prompts=rendered,
                image_token_index=getattr(self.model.config, "image_token_index", None),
                add_special_tokens=self._add_special_tokens, padding=False,
            ))
            ids = inputs.pop("input_ids", None)
            if ids is None or len(ids.shape) != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
                raise ValueError("each MLX-VLM prompt must encode to one non-empty token row")
            mask = inputs.pop("attention_mask", None)
            if mask is not None and (
                mask.shape != ids.shape or any(value != 1 for value in mask.tolist()[0])
            ):
                raise ValueError("MLX-VLM processor returned a padded or invalid attention mask")
            pixels = inputs.pop("pixel_values", None)
            if images and pixels is None:
                raise ValueError("MLX-VLM processor did not produce image pixel values")
            if "decoder_input_ids" in inputs:
                raise ValueError("MLX binary scoring requires a causal VLM, not an encoder-decoder")

            embedding_kwargs = dict(inputs)
            cached = features.get(namespace) if images else None
            if cached is not None:
                features.move_to_end(namespace)
                embedding_kwargs["cached_image_features"] = cached
            elif images:
                cached = _encode_image_features(self.model, pixels, inputs)
                if cached is not None:
                    embedding_kwargs["cached_image_features"] = cached

            output = self.model.get_input_embeddings(
                ids, pixels, mask=mask, **embedding_kwargs,
            )
            embeddings = output.inputs_embeds
            if (
                embeddings is None or len(embeddings.shape) != 3
                or embeddings.shape[0] != 1 or embeddings.shape[1] == 0
            ):
                raise ValueError("MLX-VLM must return non-empty causal input embeddings")
            metadata = {key: value for key, value in output.to_dict().items() if value is not None}
            metadata.pop("inputs_embeds", None)
            if "decoder_inputs_embeds" in metadata:
                raise ValueError("MLX binary scoring requires a causal VLM, not an encoder-decoder")
            tokens = tuple(ids.tolist()[0])
            optimized = (
                self._model_type in _OPTIMIZED_MODEL_TYPES
                and embeddings.shape[1] == len(tokens)
                and set(metadata) <= {"position_ids", "rope_deltas"}
                and "position_ids" in metadata
            )
            if optimized and images and cached is None:
                # Qwen's documented cached_image_features are the projected
                # image rows already merged by get_input_embeddings. Read those
                # rows without depending on the vision tower's internal layers.
                image_id = self.model.config.image_token_id
                positions = [index for index, token in enumerate(tokens) if token == image_id]
                if not positions:
                    raise ValueError("image evidence is missing from the rendered MLX-VLM prompt")
                cached = embeddings[0, self._mx.array(positions)]
            if cached is not None and namespace not in features and self.max_cache_entries:
                self._mx.eval(cached)
                size = _array_bytes(cached)
                if 0 < size <= self.max_cache_bytes:
                    while features and (
                        len(features) >= self.max_cache_entries
                        or feature_bytes + size > self.max_cache_bytes
                    ):
                        _, discarded = features.popitem(last=False)
                        feature_bytes -= _array_bytes(discarded)
                    features[namespace] = cached
                    feature_bytes += size
            prepared.append(_VisionPrompt(
                tokens=tokens, embeddings=embeddings,
                kwargs=metadata if optimized else {**inputs, **metadata},
                namespace=namespace, optimized=optimized,
            ))
        return prepared

    def _warm_prefixes(self, prompts: Sequence[_VisionPrompt]) -> None:
        if not self.max_cache_entries or not self.max_cache_bytes:
            return
        groups: dict[tuple[bytes, ...], list[_VisionPrompt]] = defaultdict(list)
        for prompt in prompts:
            if prompt.optimized:
                groups[prompt.namespace].append(prompt)
        for namespace, rows in groups.items():
            for prefix in shared_prefixes([row.tokens[:-1] for row in rows]):
                cache, offset = self._prefix_cache.fetch(prefix, namespace=namespace)
                if offset == len(prefix):
                    continue
                if cache is None:
                    cache = self._make_cache(self.model.language_model)
                row = next(row for row in rows if row.tokens[:len(prefix)] == prefix)
                self._prefill([row], cache, start=offset, stop=len(prefix))
                self._prefix_cache.put(prefix, cache, namespace=namespace)

    def _prefill(
        self, prompts: Sequence[_VisionPrompt], cache: Any, *, start: int, stop: int,
    ) -> None:
        for offset in range(start, stop, self.prefill_step_size):
            end = min(offset + self.prefill_step_size, stop)
            self._forward(prompts, cache, start=offset, stop=end)
            self._mx.eval([layer.state for layer in cache])
            self._mx.clear_cache()

    def _forward(
        self, prompts: Sequence[_VisionPrompt], cache: Any, *, start: int, stop: int,
    ) -> Any:
        mx = self._mx
        ids = mx.array([list(prompt.tokens[start:stop]) for prompt in prompts])
        embeddings = mx.concatenate(
            [prompt.embeddings[:, start:stop] for prompt in prompts], axis=0,
        )
        kwargs = {}
        for key in prompts[0].kwargs:
            values = [prompt.kwargs[key] for prompt in prompts]
            if key == "position_ids":
                values = [value[..., start:stop] for value in values]
                axis = 1 if values[0].ndim == 3 else 0
            else:
                axis = 0
            kwargs[key] = mx.concatenate(values, axis=axis)
        return self.model.language_model(
            ids, inputs_embeds=embeddings, cache=cache, **kwargs,
        ).logits

    def _score_batch(
        self,
        batch: Sequence[tuple[int, _VisionPrompt, Any, int]],
        cache: Any,
        yes_token_id: int,
        no_token_id: int,
    ) -> list[float]:
        prompts = [row[1] for row in batch]
        stop = len(prompts[0].tokens) - 1
        self._prefill(prompts, cache, start=batch[0][3], stop=stop)
        if stop:
            self._mx.eval([layer.state for layer in cache])
            for index, prompt in enumerate(prompts):
                row_cache = cache if len(batch) == 1 else extract_cache(cache, index)
                self._prefix_cache.put(
                    prompt.tokens[:-1], row_cache, namespace=prompt.namespace,
                )
        logits = self._forward(prompts, cache, start=stop, stop=stop + 1)
        return self._probabilities(logits, yes_token_id, no_token_id)

    def _score_complete(
        self, prompt: _VisionPrompt, *, yes_token_id: int, no_token_id: int,
    ) -> float:
        cache = self._make_cache(self.model.language_model)
        kwargs = dict(prompt.kwargs)
        if "attention_mask_4d" in kwargs:
            # Prefix-attention VLMs expose this mask separately from token
            # padding. Their language model consumes it through ``mask``;
            # retain the original name for models that read it from kwargs.
            kwargs["mask"] = kwargs["attention_mask_4d"]
        output = self.model.language_model(
            self._mx.array([list(prompt.tokens)]), inputs_embeds=prompt.embeddings,
            cache=cache, **kwargs,
        )
        return self._probabilities(output.logits, yes_token_id, no_token_id)[0]

    def _probabilities(self, logits: Any, yes_token_id: int, no_token_id: int) -> list[float]:
        mx = self._mx
        pair = logits[:, -1, mx.array([no_token_id, yes_token_id])]
        probabilities = mx.softmax(pair.astype(mx.float32), axis=-1)[:, 1]
        return [float(value) for value in probabilities.tolist()]


def _image_messages(prompt: ChatPrompt) -> tuple[ChatPrompt, list[str]]:
    messages = []
    sources = []
    for message in prompt:
        content = message["content"]
        if isinstance(content, str):
            messages.append(dict(message))
            continue
        parts = []
        for part in content:
            if part["type"] == "image_url":
                source = part["image_url"]["url"]
                if source.startswith("file://"):
                    parsed = urlparse(source)
                    if parsed.netloc not in {"", "localhost"}:
                        raise ValueError("file image URLs must refer to the local machine")
                    source = unquote(parsed.path)
                if not source.startswith(("http://", "https://", "data:")):
                    source = str(Path(source).expanduser().resolve())
                sources.append(source)
                parts.append({"type": "image"})
            elif part["type"] == "text":
                parts.append(dict(part))
            else:
                raise ValueError("MLX-VLM prompts support only text and image_url parts")
        messages.append({**message, "content": parts})
    return tuple(messages), sources


def _image_digest(image: Any) -> bytes:
    digest = hashlib.sha256()
    digest.update(repr((image.mode, image.size)).encode("utf-8"))
    digest.update(image.tobytes())
    return digest.digest()


def _array_bytes(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return sum(_array_bytes(part) for part in value)
    return int(getattr(value, "nbytes", 0))


def _encode_image_features(model: Any, pixels: Any, metadata: dict[str, Any]) -> Any:
    encoder = getattr(model, "encode_images", None)
    if not callable(encoder):
        encoder = getattr(model, "encode_image", None)
    if not callable(encoder):
        return None
    try:
        signature = inspect.signature(encoder)
    except (TypeError, ValueError):
        return None
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = {
        key: value for key, value in metadata.items()
        if accepts_kwargs or (
            key in signature.parameters
            and signature.parameters[key].kind in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
            }
        )
    }
    try:
        signature.bind(pixels, **kwargs)
    except TypeError:
        # A model requiring other arguments owns that preparation inside
        # get_input_embeddings; leave it on the complete model path.
        return None
    return encoder(pixels, **kwargs)
