"""Deterministic VLM contract tests plus optional tiny native vision models."""

from __future__ import annotations

import ast
import importlib.util
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from llm2jev import MLXBackend
from llm2jev.backend.mlx.vision import MLXVisionScorer, _image_digest, _image_messages
from test_mlx_fakes import FakeMLX, FakeTokenizer


class _Array:
    """Minimal nested-list array for tests that run without NumPy or MLX."""

    def __init__(self, values: Any) -> None:
        self.values = values
        self.shape = self._shape(values)
        self.ndim = len(self.shape)
        self.nbytes = math.prod(self.shape) * 4

    @staticmethod
    def _shape(values: Any) -> tuple[int, ...]:
        if not isinstance(values, list):
            return ()
        return (len(values),) + (_Array._shape(values[0]) if values else ())

    def __getitem__(self, selectors: Any) -> _Array:
        if not isinstance(selectors, tuple):
            selectors = (selectors,)
        if Ellipsis in selectors:
            index = selectors.index(Ellipsis)
            selectors = (
                selectors[:index] + (slice(None),) * (self.ndim - len(selectors) + 1)
                + selectors[index + 1:]
            )

        def select(values: Any, indices: tuple[Any, ...]) -> Any:
            if not indices:
                return values
            first, *rest = indices
            if isinstance(first, _Array):
                return [select(values[index], tuple(rest)) for index in first.values]
            if isinstance(first, slice):
                return [select(value, tuple(rest)) for value in values[first]]
            return select(values[first], tuple(rest))

        return _Array(select(self.values, selectors))

    def astype(self, dtype: Any) -> _Array:
        return self

    def tolist(self) -> Any:
        return deepcopy(self.values)


def _concatenate(arrays: list[_Array], axis: int = 0) -> _Array:
    def concat(values: list[Any], depth: int) -> Any:
        if depth == 0:
            return [item for value in values for item in value]
        return [concat(list(items), depth - 1) for items in zip(*values)]

    return _Array(concat([array.values for array in arrays], axis))


def _softmax(array: _Array, axis: int = -1) -> _Array:
    def calculate(values: list[Any]) -> list[Any]:
        if isinstance(values[0], list):
            return [calculate(row) for row in values]
        weights = [math.exp(value - max(values)) for value in values]
        return [weight / sum(weights) for weight in weights]

    return _Array(calculate(array.values))


class _Cache:
    def __init__(self, histories: list[list[float]] | None = None) -> None:
        self.histories = histories if histories is not None else [[]]

    @property
    def nbytes(self) -> int:
        return 4 * sum(map(len, self.histories))

    @property
    def state(self) -> _Array:
        return _Array(self.histories)

    def is_trimmable(self) -> bool:
        return True

    def trim(self, count: int) -> int:
        for row in self.histories:
            del row[-count:]
        return count

    @classmethod
    def merge(cls, caches: list[_Cache]) -> _Cache:
        return cls([list(cache.histories[0]) for cache in caches])

    def extract(self, index: int) -> _Cache:
        return _Cache([list(self.histories[index])])


class _Image:
    mode = "RGB"
    size = (1, 1)

    def __init__(self, color: int) -> None:
        self.color = color

    def tobytes(self) -> bytes:
        return bytes([self.color] * 3)


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.template_calls = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        self.template_calls.append((messages, kwargs))
        return repr(messages)


def _prompt(candidate: str = "A", source: str = "image.png") -> tuple[dict[str, Any], ...]:
    return ({"role": "user", "content": [
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {"url": source}},
        {"type": "text", "text": "after " + candidate},
    ]},)


def _tokenize(messages: Any, image_id: int = 14) -> list[int]:
    result = [10]
    for message in messages:
        content = message["content"]
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        for part in parts:
            if part["type"] == "image":
                result.extend([image_id] * 2)
            else:
                result.extend(20 + ord(char) % 20 for char in part["text"])
    return result + [11]


class _VisionModel:
    def __init__(self, model_type: str = "qwen2_vl") -> None:
        self.config = SimpleNamespace(model_type=model_type, image_token_id=14)
        self.eval = Mock()
        self.vision_calls = 0
        self.embedding_calls = []
        self.language_calls = []
        self.language_model = self.forward

    def get_input_embeddings(self, ids: _Array, pixels: _Array | None, **kwargs: Any) -> Any:
        self.embedding_calls.append(kwargs)
        tokens = ids.tolist()[0]
        cached = kwargs.get("cached_image_features")
        if pixels is not None and cached is None:
            self.vision_calls += 1
            cached = _Array([[value] for value in pixels.tolist() for _ in range(2)])
        features = iter(cached.tolist() if cached is not None else [])
        rows = [next(features) if token == 14 else [token / 20] for token in tokens]
        length = len(tokens)
        metadata = {
            "inputs_embeds": _Array([rows]),
            "position_ids": _Array([[[index for index in range(length)]] for _ in range(3)]),
            "rope_deltas": _Array([[0]]),
        }
        return SimpleNamespace(inputs_embeds=metadata["inputs_embeds"], to_dict=lambda: metadata)

    def forward(self, ids: _Array, *, inputs_embeds: _Array, cache: Any, **kwargs: Any) -> Any:
        self.language_calls.append((ids.tolist(), deepcopy(kwargs)))
        result = []
        for index, row in enumerate(inputs_embeds.tolist()):
            history = cache[0].histories[index]
            logits = []
            for column, value in enumerate(row):
                position = kwargs["position_ids"].tolist()[0][index][column]
                history.append(value[0] + position / 100)
                logit = [9999.0] + [0.0] * 11
                logit[7] = sum(history) / 20
                logits.append(logit)
            result.append(logits)
        return SimpleNamespace(logits=_Array(result))


class _Runtime:
    def __init__(self, *, model_type: str = "qwen2_vl") -> None:
        text_runtime = FakeMLX()
        self.modules = text_runtime.modules
        self.core = self.modules["mlx.core"]
        self.core.array = _Array
        self.core.concatenate = _concatenate
        self.core.softmax = _softmax
        self.core.synchronize = Mock()
        self.processor = _Processor()
        self.model = _VisionModel(model_type)
        self.images: dict[str, _Image] = {}
        self.load_image = Mock(side_effect=lambda source: self.images.get(source, _Image(1)))
        self.load = Mock(return_value=(self.model, self.processor))
        self.prepare = Mock(side_effect=self.prepare_inputs)
        vlm = ModuleType("mlx_vlm")
        vlm.__path__ = []
        vlm.load = self.load
        models = ModuleType("mlx_vlm.models")
        models.__path__ = []
        base = ModuleType("mlx_vlm.models.base")
        base.BaseImageProcessor = type("BaseImageProcessor", (), {})
        cache = ModuleType("mlx_vlm.models.cache")
        cache.make_prompt_cache = lambda model: [_Cache()]
        utils = ModuleType("mlx_vlm.utils")
        utils.load_image = self.load_image
        utils.prepare_inputs = self.prepare
        utils.should_add_special_tokens = lambda model_type, processor: False
        self.modules.update({
            "mlx_vlm": vlm, "mlx_vlm.models": models, "mlx_vlm.models.base": base,
            "mlx_vlm.models.cache": cache, "mlx_vlm.utils": utils,
        })

    def prepare_inputs(self, processor: Any, **kwargs: Any) -> dict[str, Any]:
        tokens = _tokenize(ast.literal_eval(kwargs["prompts"]))
        result = {"input_ids": _Array([tokens]), "attention_mask": _Array([[1] * len(tokens)])}
        if kwargs["images"]:
            result["pixel_values"] = _Array([image.color for image in kwargs["images"]])
            result["image_grid_thw"] = _Array([[1, 2, 4] for _ in kwargs["images"]])
        return result

    def scorer(self, **options: Any) -> MLXVisionScorer:
        defaults = dict(
            enable_thinking=False, batch_size=8, prefill_step_size=3,
            submission="staged", max_cache_entries=32, max_cache_bytes=1024 * 1024,
        )
        with patch.dict("sys.modules", self.modules):
            return MLXVisionScorer(self.model, self.processor, **{**defaults, **options})


class MLXVisionTests(unittest.TestCase):
    def score(self, scorer: MLXVisionScorer, prompts: Any) -> Any:
        return scorer.score(prompts, yes_token_id=7, no_token_id=9)

    def test_backend_loads_local_vlm_and_preserves_template_and_label_options(self) -> None:
        runtime = _Runtime()
        runtime.processor.tokenizer.labels.update({"positive": [7], "negative": [9]})
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", runtime.modules):
            with MLXBackend(
                directory, multimodal=True, yes_label="positive", no_label="negative",
                enable_thinking=True, batch_size=2, prefill_step_size=4,
            ) as backend:
                output = backend.score(model="alias", prompts=[_prompt()])
                self.assertEqual(output.usage.output_tokens, 0)
                self.assertIs(backend.processor, runtime.processor)
                runtime.load.assert_called_once_with(directory, local_files_only=True)
                self.assertTrue(runtime.processor.template_calls[0][1]["enable_thinking"])
            self.assertIsNone(backend.model)

    def test_backend_rejects_multitoken_labels(self) -> None:
        runtime = _Runtime()
        runtime.processor.tokenizer.labels["yes"] = [7, 8]
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", runtime.modules):
            with self.assertRaisesRegex(ValueError, "exactly one token"):
                MLXBackend(directory, multimodal=True)

    def test_shared_images_prefixes_and_real_batches_match_independent_prefill(self) -> None:
        prompts = [_prompt("A"), _prompt("B"), _prompt("longer")]
        reference_runtime = _Runtime(model_type="other_causal_vlm")
        reference = self.score(reference_runtime.scorer(), prompts)
        runtime = _Runtime()
        scorer = runtime.scorer()
        output = self.score(scorer, prompts)
        self.assertEqual(output, reference)
        self.assertEqual(runtime.model.vision_calls, 1)
        self.assertEqual(runtime.load_image.call_count, 1)
        self.assertTrue(any(len(ids) == 2 for ids, _ in runtime.model.language_calls))
        self.assertLess(
            sum(len(row) for ids, _ in runtime.model.language_calls for row in ids),
            output.usage.input_tokens,
        )
        runtime.model.language_calls.clear()
        repeated = self.score(scorer, prompts[::-1])
        self.assertEqual(repeated.yes_probabilities, output.yes_probabilities[::-1])
        self.assertTrue(all(len(row) == 1 for ids, _ in runtime.model.language_calls for row in ids))

    def test_different_images_and_changed_file_contents_do_not_reuse_prefix(self) -> None:
        runtime = _Runtime()
        source = str(Path("image.png").resolve())
        scorer = runtime.scorer()
        first = self.score(scorer, [_prompt()])
        runtime.images[source] = _Image(90)
        runtime.model.language_calls.clear()
        changed = self.score(scorer, [_prompt()])
        self.assertNotEqual(first.yes_probabilities, changed.yes_probabilities)
        self.assertGreater(len(runtime.model.language_calls), 1)
        other = str(Path("other.png").resolve())
        runtime.images[other] = _Image(3)
        output = self.score(scorer, [_prompt(source="other.png"), _prompt()])
        self.assertNotEqual(*output.yes_probabilities)
        self.assertEqual(output.yes_probabilities[1], changed.yes_probabilities[0])

    def test_interleaved_multiple_images_and_messages_keep_order(self) -> None:
        runtime = _Runtime()
        prompt = (
            {"role": "system", "content": "observe"},
            {"role": "user", "content": [
                {"type": "text", "text": "first"},
                {"type": "image_url", "image_url": {"url": "one.png"}},
                {"type": "text", "text": "between"},
                {"type": "image_url", "image_url": {"url": "two.png"}},
                {"type": "text", "text": "last"},
            ]},
        )
        self.score(runtime.scorer(), [prompt, _prompt()])
        messages = runtime.processor.template_calls[0][0]
        self.assertEqual(messages[0]["content"], "observe")
        self.assertEqual([part["type"] for part in messages[1]["content"]], [
            "text", "image", "text", "image", "text",
        ])
        self.assertEqual(len(runtime.prepare.call_args_list[0].kwargs["images"]), 2)

    def test_local_file_http_and_data_sources_are_passed_to_loader(self) -> None:
        runtime = _Runtime()
        sources = ["file:///tmp/a%20b.png", "https://example.com/a.png", "data:image/png;base64,AAA="]
        self.score(runtime.scorer(), [_prompt(source=source) for source in sources])
        self.assertEqual([call.args[0] for call in runtime.load_image.call_args_list], [
            str(Path("/tmp/a b.png").resolve()), sources[1], sources[2],
        ])
        with self.assertRaisesRegex(ValueError, "local machine"):
            _image_messages(_prompt(source="file://elsewhere/a.png"))

    def test_unknown_model_uses_complete_independent_prefill(self) -> None:
        runtime = _Runtime(model_type="other_causal_vlm")
        output = self.score(runtime.scorer(), [_prompt("A"), _prompt("B")])
        self.assertEqual(len(runtime.model.language_calls), 2)
        self.assertTrue(all(len(ids) == 1 for ids, _ in runtime.model.language_calls))
        self.assertTrue(all("image_grid_thw" in kwargs for _, kwargs in runtime.model.language_calls))
        self.assertEqual(runtime.model.vision_calls, 2)
        self.assertEqual(output.usage.output_tokens, 0)

    def test_generic_model_preserves_full_image_attention_mask(self) -> None:
        runtime = _Runtime(model_type="other_causal_vlm")
        original_embeddings = runtime.model.get_input_embeddings
        attention_mask = _Array([[[[0.0, 0.0], [0.0, 0.0]]]])

        def get_embeddings(*args: Any, **kwargs: Any) -> Any:
            output = original_embeddings(*args, **kwargs)
            metadata = output.to_dict()
            metadata["attention_mask_4d"] = attention_mask
            return SimpleNamespace(inputs_embeds=output.inputs_embeds, to_dict=lambda: metadata)

        runtime.model.get_input_embeddings = get_embeddings
        self.score(runtime.scorer(), [_prompt()])
        _, kwargs = runtime.model.language_calls[0]
        self.assertEqual(kwargs["mask"].tolist(), attention_mask.tolist())
        self.assertEqual(kwargs["attention_mask_4d"].tolist(), attention_mask.tolist())

    def test_public_image_encoder_is_reused_for_generic_vlm(self) -> None:
        runtime = _Runtime(model_type="other_causal_vlm")
        runtime.model.encode_images = Mock(return_value=_Array([[3], [3]]))
        self.score(runtime.scorer(), [_prompt("A"), _prompt("B")])
        runtime.model.encode_images.assert_called_once()
        self.assertEqual(runtime.model.vision_calls, 0)

    def test_image_encoder_receives_supported_geometry_metadata(self) -> None:
        runtime = _Runtime(model_type="other_causal_vlm")
        calls = []

        def encode_image(pixels: Any, image_grid_thw: Any = None) -> Any:
            self.assertIsNotNone(image_grid_thw)
            calls.append(image_grid_thw.tolist())
            return _Array([[3], [3]])

        runtime.model.encode_image = encode_image
        self.score(runtime.scorer(), [_prompt("A"), _prompt("B")])
        self.assertEqual(calls, [[[1, 2, 4]]])
        self.assertEqual(runtime.model.vision_calls, 0)

    def test_encoder_with_unavailable_required_metadata_uses_complete_model_path(self) -> None:
        runtime = _Runtime(model_type="other_causal_vlm")

        def encode_image(pixels: Any, unavailable_layout: Any) -> Any:
            self.fail("incompatible standalone image encoder must not be called")

        runtime.model.encode_image = encode_image
        self.score(runtime.scorer(), [_prompt()])
        self.assertEqual(runtime.model.vision_calls, 1)

    def test_actual_image_encoder_failures_are_not_retried(self) -> None:
        runtime = _Runtime(model_type="other_causal_vlm")
        runtime.model.encode_images = Mock(side_effect=ValueError("encoder failed"))
        with self.assertRaisesRegex(ValueError, "encoder failed"):
            self.score(runtime.scorer(), [_prompt()])
        runtime.model.encode_images.assert_called_once()
        self.assertEqual(runtime.model.vision_calls, 0)

    def test_disabled_cache_and_all_submission_preserve_scores(self) -> None:
        prompts = [_prompt("A"), _prompt("B")]
        reference = self.score(_Runtime().scorer(), prompts)
        for options in ({"max_cache_entries": 0}, {"max_cache_bytes": 0}, {"submission": "all"}):
            with self.subTest(options=options):
                runtime = _Runtime()
                scorer = runtime.scorer(**options)
                self.assertEqual(self.score(scorer, prompts), reference)
                if options != {"submission": "all"}:
                    self.assertEqual(len(scorer._prefix_cache), 0)
                    self.assertEqual(runtime.model.vision_calls, 2)
                scorer.clear_cache()
                self.assertEqual(len(scorer._prefix_cache), 0)

    def test_empty_and_invalid_processor_results_are_rejected(self) -> None:
        runtime = _Runtime()
        scorer = runtime.scorer()
        with self.assertRaisesRegex(ValueError, "empty"):
            self.score(scorer, [])
        for result, message in (
            ({"input_ids": _Array([[]])}, "non-empty"),
            ({"input_ids": _Array([[1]]), "attention_mask": _Array([[0]])}, "padded"),
            ({"input_ids": _Array([[1]])}, "pixel values"),
        ):
            with self.subTest(result=result), patch.object(scorer, "_prepare_inputs", return_value=result):
                with self.assertRaisesRegex(ValueError, message):
                    self.score(scorer, [_prompt()])

    def test_legacy_processor_rejects_multiple_images_instead_of_dropping_one(self) -> None:
        runtime = _Runtime()
        runtime.processor.image_processor = runtime.modules["mlx_vlm.models.base"].BaseImageProcessor()
        prompt = _prompt()[0]
        prompt["content"].append({"type": "image_url", "image_url": {"url": "second.png"}})
        with self.assertRaisesRegex(ValueError, "only one image"):
            self.score(runtime.scorer(), [(prompt,)])

    def test_image_digest_includes_dimensions(self) -> None:
        left, right = _Image(2), _Image(2)
        self.assertEqual(_image_digest(left), _image_digest(right))
        right.size = (3, 1)
        self.assertNotEqual(_image_digest(left), _image_digest(right))


@unittest.skipUnless(
    importlib.util.find_spec("mlx") is not None and importlib.util.find_spec("mlx_vlm") is not None,
    "MLX-VLM dependencies are optional",
)
class MLXVisionNativeTests(unittest.TestCase):
    """Real image preprocessing, vision attention and mRoPE without downloads."""

    def check_model(self, model_type: str) -> None:
        import base64
        import importlib

        import mlx.core as mx
        from PIL import Image
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast
        from mlx_vlm.models.cache import make_prompt_cache
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor
        from mlx_vlm.utils import load_image, prepare_inputs

        model_module = importlib.import_module(f"mlx_vlm.models.{model_type}")
        configs = importlib.import_module(f"mlx_vlm.models.{model_type}.config")
        processing = importlib.import_module(
            f"mlx_vlm.models.{model_type}.processing_{model_type}",
        )
        processor_class = getattr(
            processing, "Qwen2VLProcessor" if model_type == "qwen2_vl" else "Qwen2_5_VLProcessor",
        )
        mx.random.seed(4)
        text = configs.TextConfig(
            model_type=model_type, hidden_size=32, num_hidden_layers=2,
            intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
            rms_norm_eps=1e-6, vocab_size=64,
            rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        )
        vision_options = dict(
            depth=1, num_heads=4, patch_size=2, spatial_patch_size=2,
            spatial_merge_size=2, temporal_patch_size=2,
        )
        if model_type == "qwen2_vl":
            vision_options.update(embed_dim=32, hidden_size=32)
        else:
            vision_options.update(
                hidden_size=32, out_hidden_size=32, intermediate_size=64,
                window_size=8, fullatt_block_indexes=[0],
            )
        model = model_module.Model(configs.ModelConfig(
            text_config=text, vision_config=configs.VisionConfig(**vision_options),
            model_type=model_type, image_token_id=14, video_token_id=15,
            vision_start_token_id=13,
        ))
        model.eval()
        vocab = {
            "<unk>": 0, "<pad>": 1, "yes": 7, "no": 9, "<|vision_start|>": 13,
            "<|image_pad|>": 14, "<|video_pad|>": 15, "<|vision_end|>": 16,
            "before": 20, "after": 21, "A": 22, "B": 23, "assistant": 24, "longer": 25,
        }
        for index in range(64):
            if index not in vocab.values():
                vocab[f"unused{index}"] = index
        tokenizer_object = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        tokenizer_object.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_object, unk_token="<unk>", pad_token="<pad>",
            eos_token="<pad>", additional_special_tokens=[
                "<|vision_start|>", "<|image_pad|>", "<|video_pad|>", "<|vision_end|>",
            ],
        )
        template = (
            "{% for message in messages %}"
            "{% if message['content'] is string %}{{ message['content'] }}"
            "{% else %}{% for part in message['content'] %}"
            "{% if part['type']=='image' %}<|vision_start|><|image_pad|><|vision_end|>"
            "{% else %}{{ part['text'] }} {% endif %}{% endfor %}{% endif %} "
            "{% endfor %}assistant"
        )
        tokenizer.chat_template = template
        processor = processor_class(
            image_processor=Qwen3VLImageProcessor(
                patch_size=2, temporal_patch_size=2, merge_size=2, min_pixels=64, max_pixels=64,
            ), tokenizer=tokenizer, chat_template=template,
        )

        def reference(prompt: Any) -> tuple[float, int]:
            messages, sources = _image_messages(prompt)
            rendered = processor.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            inputs = prepare_inputs(
                processor, images=[load_image(source) for source in sources] or None,
                prompts=rendered, padding=False,
            )
            ids = inputs.pop("input_ids")
            mask = inputs.pop("attention_mask", None)
            pixels = inputs.pop("pixel_values", None)
            features = model.get_input_embeddings(ids, pixels, mask=mask, **inputs)
            metadata = {key: value for key, value in features.to_dict().items() if value is not None}
            output = model.language_model(
                ids, cache=make_prompt_cache(model.language_model), **metadata,
            )
            probability = float(mx.softmax(output.logits[0, -1, mx.array([9, 7])].astype(mx.float32))[1].item())
            return probability, features.inputs_embeds.shape[1]

        with tempfile.TemporaryDirectory() as directory:
            red, blue = Path(directory) / "red image.png", Path(directory) / "blue.png"
            Image.new("RGB", (8, 8), "red").save(red)
            Image.new("RGB", (8, 8), "blue").save(blue)
            data_url = "data:image/png;base64," + base64.b64encode(red.read_bytes()).decode("ascii")
            multiple = deepcopy(_prompt("A", str(red)))
            multiple[0]["content"].append({"type": "image_url", "image_url": {"url": str(blue)}})
            prompts = [
                _prompt("A", str(red)), _prompt("B", red.as_uri()),
                _prompt("longer B", data_url), _prompt("A", str(blue)), multiple,
                ({"role": "user", "content": "before after A"},),
            ]
            expected = [reference(prompt) for prompt in prompts]
            for submission, batch_size in (("staged", 4), ("all", 4), ("staged", 1)):
                with self.subTest(model_type=model_type, submission=submission, batch_size=batch_size):
                    scorer = MLXVisionScorer(
                        model, processor, enable_thinking=False, batch_size=batch_size,
                        prefill_step_size=3, submission=submission,
                        max_cache_entries=32, max_cache_bytes=2 * 1024 * 1024,
                    )
                    language_calls = []
                    vision_calls = []
                    language_class, vision_class = type(model.language_model), type(model.vision_tower)
                    original_language, original_vision = language_class.__call__, vision_class.__call__

                    def tracked_language(instance: Any, ids: Any, **kwargs: Any) -> Any:
                        language_calls.append(ids.shape)
                        return original_language(instance, ids, **kwargs)

                    def tracked_vision(instance: Any, *args: Any, **kwargs: Any) -> Any:
                        vision_calls.append(args[0].shape)
                        return original_vision(instance, *args, **kwargs)

                    with patch.object(language_class, "__call__", tracked_language), patch.object(
                        vision_class, "__call__", tracked_vision,
                    ):
                        output = scorer.score(prompts, yes_token_id=7, no_token_id=9)
                    for actual, (probability, _) in zip(output.yes_probabilities, expected):
                        self.assertAlmostEqual(actual, probability, delta=1e-5)
                    self.assertEqual(output.usage.input_tokens, sum(length for _, length in expected))
                    self.assertEqual(output.usage.output_tokens, 0)
                    # Three distinct ordered image sets, despite five image prompts.
                    self.assertEqual(len(vision_calls), 3)
                    if batch_size > 1:
                        self.assertTrue(any(shape[0] > 1 for shape in language_calls))
                    reordered = scorer.score(prompts[::-1], yes_token_id=7, no_token_id=9)
                    for actual, (probability, _) in zip(reordered.yes_probabilities, expected[::-1]):
                        self.assertAlmostEqual(actual, probability, delta=1e-5)

            # A new image written at the same path must produce fresh evidence.
            Image.new("RGB", (8, 8), "green").save(red)
            changed_reference, _ = reference(prompts[0])
            changed = scorer.score([prompts[0]], yes_token_id=7, no_token_id=9)
            self.assertAlmostEqual(changed.yes_probabilities[0], changed_reference, delta=1e-5)
            self.assertNotAlmostEqual(changed_reference, expected[0][0], delta=1e-5)

    def test_qwen2_vl_image_prefill_matches_complete_forward(self) -> None:
        self.check_model("qwen2_vl")

    def test_qwen2_5_vl_image_prefill_matches_complete_forward(self) -> None:
        self.check_model("qwen2_5_vl")


if __name__ == "__main__":
    unittest.main()
