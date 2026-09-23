import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llm2jev import Choice, JevRequest, LLM2Jev, MLXBackend, Noul, Score, Usage
from test_mlx_fakes import FakeBatchCacheLayer, FakeCacheLayer, FakeMLX, FakeModel


def text_prompt(text: str = "Evidence") -> tuple[dict[str, str], ...]:
    return ({"role": "user", "content": text},)


class MLXBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.model_path = Path(directory.name)

    def backend(self, runtime: FakeMLX, **kwargs) -> MLXBackend:
        modules = patch.dict("sys.modules", runtime.modules)
        modules.start()
        self.addCleanup(modules.stop)
        options = {"batch_size": 1, "max_cache_entries": 0, **kwargs}
        return MLXBackend(self.model_path, **options)

    def test_loads_local_model_once_and_ignores_request_model_name(self) -> None:
        runtime = FakeMLX()
        backend = self.backend(runtime)

        backend.score(model="a request alias", prompts=[text_prompt()])
        backend.score(model="another alias", prompts=[text_prompt()])

        runtime.load.assert_called_once_with(
            str(self.model_path), tokenizer_config={"local_files_only": True},
        )
        runtime.model.eval.assert_called_once_with()
        self.assertIs(backend.model, runtime.model)
        self.assertIs(backend.tokenizer, runtime.tokenizer)

    def test_expands_local_home_directory_before_loading(self) -> None:
        runtime = FakeMLX()
        with patch.dict("sys.modules", runtime.modules):
            MLXBackend("~")

        runtime.load.assert_called_once_with(
            str(Path.home()), tokenizer_config={"local_files_only": True},
        )

    def test_rejects_missing_directories_remote_ids_and_files_before_loading(self) -> None:
        runtime = FakeMLX()
        file_path = self.model_path / "weights.bin"
        file_path.write_bytes(b"fake")
        with patch.dict("sys.modules", runtime.modules):
            for path in (self.model_path / "missing", file_path, "organization/remote-model"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    MLXBackend(path)

        runtime.load.assert_not_called()

    def test_rejects_invalid_prefill_step_size_before_loading(self) -> None:
        runtime = FakeMLX()
        with patch.dict("sys.modules", runtime.modules):
            for size in (0, -1, True, False, 1.5, "2", None):
                with self.subTest(size=size), self.assertRaisesRegex(ValueError, "prefill_step_size"):
                    MLXBackend(self.model_path, prefill_step_size=size)

        runtime.load.assert_not_called()

    def test_missing_optional_dependencies_have_actionable_error(self) -> None:
        for unavailable in ("mlx", "mlx_lm", "mlx_lm.models.cache"):
            runtime = FakeMLX()
            modules = {**runtime.modules, unavailable: None}
            if unavailable == "mlx":
                modules["mlx.core"] = None
            with self.subTest(module=unavailable), patch.dict("sys.modules", modules):
                with self.assertRaisesRegex(ImportError, "[Mm][Ll][Xx]"):
                    MLXBackend(self.model_path)

            runtime.load.assert_not_called()

    def test_accepts_custom_single_token_labels(self) -> None:
        runtime = FakeMLX(labels={"Yes": [7], "No": [9]})
        backend = self.backend(runtime, yes_label="Yes", no_label="No")

        output = backend.score(model="local", prompts=[text_prompt()])

        self.assertEqual(output.yes_probabilities, (0.5,))
        self.assertEqual(runtime.tokenizer.encode_calls[:2], [("Yes", False), ("No", False)])

    def test_rejects_empty_multitoken_or_identical_labels(self) -> None:
        for labels in (
            {"yes": [], "no": [9]},
            {"yes": [7, 8], "no": [9]},
            {"yes": [7], "no": []},
            {"yes": [7], "no": [8, 9]},
            {"yes": [7], "no": [7]},
        ):
            runtime = FakeMLX(labels=labels)
            with self.subTest(labels=labels), patch.dict("sys.modules", runtime.modules):
                with self.assertRaises(ValueError):
                    MLXBackend(self.model_path)

            self.assertEqual(runtime.model.calls, [])

    def test_scores_only_final_binary_logits_in_prompt_order_and_float32(self) -> None:
        runtime = FakeMLX(
            prompt_tokens=[[11, 12], [13], [14, 15]],
            logits_by_token={
                11: (0, 100),
                12: (-1000, -999),
                13: (1000, 999),
                14: (100, 0),
                15: (0, 0),
            },
        )
        backend = self.backend(runtime)

        output = backend.score(model="local", prompts=[text_prompt(str(i)) for i in range(3)])

        expected = (1 / (1 + math.exp(-1)), 1 / (1 + math.exp(1)), 0.5)
        for actual, probability in zip(output.yes_probabilities, expected):
            self.assertAlmostEqual(actual, probability)
            self.assertIsInstance(actual, float)
        self.assertEqual(output.usage, Usage(input_tokens=5, output_tokens=0))
        self.assertEqual([value.tolist() for value in runtime.softmax_inputs],
                         [[-1000, -999], [1000, 999], [0, 0]])
        self.assertTrue(all(value.dtype == runtime.core.float32 for value in runtime.softmax_inputs))

    def test_chunk_boundaries_preserve_all_tokens_and_use_fresh_prompt_caches(self) -> None:
        tokens = [[11], [21, 22], [31, 32, 33], [41, 42, 43, 44, 45, 46]]
        runtime = FakeMLX(prompt_tokens=tokens)
        backend = self.backend(runtime, prefill_step_size=2)

        output = backend.score(model="local", prompts=[text_prompt(str(i)) for i in range(4)])

        self.assertEqual(output.usage, Usage(input_tokens=12, output_tokens=0))
        self.assertEqual(len(runtime.caches), 4)
        expected = [
            [[[11]]],
            [[[21]], [[22]]],
            [[[31, 32]], [[33]]],
            [[[41, 42]], [[43, 44]], [[45]], [[46]]],
        ]
        for index, cache in enumerate(runtime.caches):
            calls = [values for values, seen_cache in runtime.model.calls if seen_cache is cache]
            self.assertEqual(calls, expected[index])
            for layer in cache:
                self.assertEqual(layer.tokens, tokens[index])
        cache_evaluations = [call.args[0] for call in runtime.core.eval.call_args_list
                             if call.args and isinstance(call.args[0], list)]
        self.assertEqual(len(cache_evaluations), 5)
        self.assertEqual(runtime.core.clear_cache.call_count, 5)
        self.assertEqual(
            [states[0][0].tolist() for states in cache_evaluations],
            [[21], [31, 32], [41, 42], [41, 42, 43, 44], [41, 42, 43, 44, 45]],
        )

    def test_prefill_step_size_one_keeps_final_token_separate(self) -> None:
        runtime = FakeMLX(prompt_tokens=[[11, 12, 13]])
        backend = self.backend(runtime, prefill_step_size=1)

        backend.score(model="local", prompts=[text_prompt()])

        self.assertEqual([values for values, _ in runtime.model.calls], [[[11]], [[12]], [[13]]])
        self.assertEqual(len(runtime.caches), 1)

    def test_rejects_empty_prompt_batch_or_empty_encoded_prompt(self) -> None:
        runtime = FakeMLX(prompt_tokens=[[]])
        backend = self.backend(runtime)

        with self.assertRaises(ValueError):
            backend.score(model="local", prompts=[])
        with self.assertRaises(ValueError):
            backend.score(model="local", prompts=[text_prompt()])

        self.assertEqual(runtime.model.calls, [])

    def test_template_disables_thinking_and_encoding_does_not_add_special_tokens(self) -> None:
        runtime = FakeMLX()
        backend = self.backend(runtime)
        prompt = text_prompt()

        backend.score(model="local", prompts=[prompt])

        self.assertEqual(runtime.tokenizer.template_calls, [(list(prompt), {
            "tokenize": False, "add_generation_prompt": True, "enable_thinking": False,
        })])
        self.assertTrue(all(add_special_tokens is False
                            for _, add_special_tokens in runtime.tokenizer.encode_calls))

    def test_explicit_thinking_option_reaches_template(self) -> None:
        runtime = FakeMLX()
        backend = self.backend(runtime, enable_thinking=True)

        backend.score(model="local", prompts=[text_prompt()])

        self.assertTrue(runtime.tokenizer.template_calls[0][1]["enable_thinking"])

    def test_joins_text_parts_without_mutating_prompt(self) -> None:
        runtime = FakeMLX()
        backend = self.backend(runtime)
        parts = [{"type": "text", "text": "Evidence: "}, {"type": "text", "text": "hello"}]
        prompt = ({"role": "user", "content": parts},)

        backend.score(model="local", prompts=[prompt])

        self.assertEqual(runtime.tokenizer.template_calls[0][0],
                         [{"role": "user", "content": "Evidence: hello"}])
        self.assertIs(prompt[0]["content"], parts)
        self.assertEqual(len(parts), 2)

        # OpenAI message types also accept content parts supplied as an iterable.
        backend.score(model="local", prompts=[({"role": "user", "content": iter(parts)},)])
        self.assertEqual(runtime.tokenizer.template_calls[-1][0],
                         [{"role": "user", "content": "Evidence: hello"}])

    def test_rejects_images_anywhere_in_batch_before_model_forward(self) -> None:
        runtime = FakeMLX()
        backend = self.backend(runtime)
        image = ({"role": "user", "content": [
            {"type": "text", "text": "Describe this"},
            {"type": "image_url", "image_url": {"url": "missing.png"}},
        ]},)

        with self.assertRaisesRegex(ValueError, "[Ii]mage|text-only"):
            backend.score(model="local", prompts=[text_prompt(), image])

        self.assertEqual(runtime.model.calls, [])

    def test_mixed_jev_request_preserves_answers_probabilities_and_usage(self) -> None:
        probabilities = [0.8, 0.2, 0.25, 0.75, 0.9, 0.7, 0.3]
        runtime = FakeMLX(
            prompt_tokens=[[11, token] for token in range(20, 27)],
            logits_by_token={
                token: (math.log(1 - probability), math.log(probability))
                for token, probability in zip(range(20, 27), probabilities)
            },
        )
        backend = self.backend(runtime)
        request = JevRequest(
            state={"message": "Duplicate charge; please refund me."},
            model="request-model",
            questions={
                "department": Choice(criteria={"billing": None, "returns": None}),
                "severity": Score(criteria=["Low", "High"]),
                "refund": Noul(),
                "explicit_refund": Noul(
                    criteria={"true": "Refund requested", "false": "No refund requested"},
                ),
            },
        )

        response = LLM2Jev(backend=backend).evaluate(request)

        self.assertEqual(response.model, "request-model")
        self.assertEqual(response.answers["department"].choice, "billing")
        self.assertAlmostEqual(response.answers["department"].probabilities["billing"], 0.8)
        self.assertAlmostEqual(response.answers["severity"].score, 0.75)
        self.assertAlmostEqual(response.answers["refund"].noul, 0.9)
        self.assertAlmostEqual(response.answers["explicit_refund"].noul, 0.7)
        self.assertEqual(response.usage, Usage(input_tokens=14, output_tokens=0))
        self.assertEqual(response.to_dict()["usage"], {"input_tokens": 14, "output_tokens": 0})
        self.assertEqual(len(runtime.caches), 7)

    def test_variable_length_inputs_use_real_batches_without_padding(self) -> None:
        sequences = [[11, 12, 91], [21, 22, 23, 24, 92], [31, 93]]
        expected = (0.8, 0.2, 0.6)
        runtime = FakeMLX(
            prompt_tokens=sequences,
            logits_by_sequence={
                tuple(tokens): (math.log(1 - probability), math.log(probability))
                for tokens, probability in zip(sequences, expected)
            },
        )
        backend = self.backend(runtime, batch_size=3, prefill_step_size=2)

        output = backend.score(model="local", prompts=[text_prompt(str(i)) for i in range(3)])

        self.assertEqual([values for values, _ in runtime.model.calls], [
            [[11], [21], [31]], [[12], [22]], [[23, 24]], [[91], [92], [93]],
        ])
        self.assertEqual(runtime.model.contexts[-1], sequences)
        self.assertTrue(all(isinstance(layer, FakeBatchCacheLayer)
                            for layer in runtime.model.calls[-1][1]))
        for actual, probability in zip(output.yes_probabilities, expected):
            self.assertAlmostEqual(actual, probability)
        self.assertEqual(output.usage, Usage(input_tokens=10, output_tokens=0))

    def test_batch_limit_preserves_order_across_multiple_submissions(self) -> None:
        sequences = [[11, 91], [21, 22, 92], [31, 93], [94], [41, 42, 95]]
        expected = (0.1, 0.8, 0.4, 0.7, 0.3)
        runtime = FakeMLX(
            prompt_tokens=sequences,
            logits_by_sequence={
                tuple(tokens): (math.log(1 - probability), math.log(probability))
                for tokens, probability in zip(sequences, expected)
            },
        )
        backend = self.backend(runtime, batch_size=2)

        output = backend.score(model="local", prompts=[text_prompt(str(i)) for i in range(5)])

        sizes = [len(values) for values, _ in runtime.model.calls]
        self.assertEqual(max(sizes), 2)
        self.assertIn(1, sizes)
        for actual, probability in zip(output.yes_probabilities, expected):
            self.assertAlmostEqual(actual, probability)
        self.assertEqual(output.usage, Usage(input_tokens=11, output_tokens=0))

    def test_staged_prefills_shared_prefix_once_and_forks_independent_histories(self) -> None:
        sequences = [[11, 12, 21, 91], [11, 12, 22, 92]]
        runtime = FakeMLX(prompt_tokens=sequences)
        backend = self.backend(runtime, batch_size=2, max_cache_entries=8, submission="staged")

        output = backend.score(model="local", prompts=[text_prompt("a"), text_prompt("b")])

        self.assertEqual([values for values, _ in runtime.model.calls], [
            [[11, 12]], [[21], [22]], [[91], [92]],
        ])
        self.assertEqual(runtime.model.contexts[-1], sequences)
        self.assertEqual(output.usage, Usage(input_tokens=8, output_tokens=0))

    def test_staged_supports_recurrent_cache_that_cannot_be_trimmed(self) -> None:
        sequences = [[11, 12, 21, 91], [11, 12, 22, 92]]
        runtime = FakeMLX(prompt_tokens=sequences)
        backend = self.backend(runtime, batch_size=2, max_cache_entries=8, submission="staged")

        with patch.object(FakeCacheLayer, "is_trimmable", return_value=False):
            backend.score(model="local", prompts=[text_prompt("a"), text_prompt("b")])

        self.assertEqual([values for values, _ in runtime.model.calls], [
            [[11, 12]], [[21], [22]], [[91], [92]],
        ])
        self.assertEqual(runtime.model.contexts[-1], sequences)

    def test_warm_requests_only_score_last_input_tokens_and_keep_logical_usage(self) -> None:
        sequences = [[11, 12, 21, 91], [11, 12, 22, 92]]
        runtime = FakeMLX(
            prompt_tokens=sequences,
            logits_by_sequence={tuple(sequences[0]): (0.0, 1.0), tuple(sequences[1]): (1.0, 0.0)},
        )
        backend = self.backend(runtime, batch_size=2, max_cache_entries=8)
        prompts = [text_prompt("a"), text_prompt("b")]
        first = backend.score(model="local", prompts=prompts)
        runtime.model.calls.clear()

        second = backend.score(model="local", prompts=prompts)

        self.assertEqual([values for values, _ in runtime.model.calls], [[[91], [92]]])
        self.assertEqual(runtime.model.contexts[-1], sequences)
        self.assertEqual(first, second)
        self.assertEqual(second.usage, Usage(input_tokens=8, output_tokens=0))

    def test_all_submission_skips_shared_prefix_seeding_but_reuses_warm_cache(self) -> None:
        sequences = [[11, 12, 91], [11, 12, 92]]
        runtime = FakeMLX(prompt_tokens=sequences)
        backend = self.backend(runtime, batch_size=2, max_cache_entries=8, submission="all")
        prompts = [text_prompt("a"), text_prompt("b")]

        first = backend.score(model="local", prompts=prompts)
        self.assertEqual([values for values, _ in runtime.model.calls], [
            [[11, 12], [11, 12]], [[91], [92]],
        ])
        runtime.model.calls.clear()
        second = backend.score(model="local", prompts=prompts)

        self.assertEqual([values for values, _ in runtime.model.calls], [[[91], [92]]])
        self.assertEqual(runtime.model.contexts[-1], sequences)
        self.assertEqual(first, second)

    def test_zero_cache_limits_disable_reuse_without_disabling_batching(self) -> None:
        for limits in ({"max_cache_entries": 0}, {"max_cache_entries": 8, "max_cache_bytes": 0}):
            with self.subTest(limits=limits):
                runtime = FakeMLX(prompt_tokens=[[11, 12, 91], [11, 12, 92]])
                backend = self.backend(runtime, batch_size=2, **limits)
                prompts = [text_prompt("a"), text_prompt("b")]
                backend.score(model="local", prompts=prompts)
                first_calls = [values for values, _ in runtime.model.calls]
                runtime.model.calls.clear()

                backend.score(model="local", prompts=prompts)

                self.assertEqual([values for values, _ in runtime.model.calls], first_calls)
                self.assertEqual(first_calls, [[[11, 12], [11, 12]], [[91], [92]]])

    def test_single_token_candidates_batch_without_prefill_or_cache_entries(self) -> None:
        runtime = FakeMLX(prompt_tokens=[[91], [92]])
        backend = self.backend(runtime, batch_size=2, max_cache_entries=8)

        output = backend.score(model="local", prompts=[text_prompt("a"), text_prompt("b")])

        self.assertEqual([values for values, _ in runtime.model.calls], [[[91], [92]]])
        self.assertEqual(len(backend._text_scorer.prefix_cache), 0)
        self.assertEqual(output.usage, Usage(input_tokens=2, output_tokens=0))

    def test_missing_merge_support_falls_back_to_complete_individual_histories(self) -> None:
        sequences = [[11, 12, 91], [21, 92]]
        runtime = FakeMLX(prompt_tokens=sequences)
        backend = self.backend(runtime, batch_size=2)

        with patch.object(FakeCacheLayer, "merge", None):
            backend.score(model="local", prompts=[text_prompt("a"), text_prompt("b")])

        self.assertTrue(all(len(values) == 1 for values, _ in runtime.model.calls))
        self.assertEqual(runtime.model.contexts[-2:], [[sequences[0]], [sequences[1]]])

    def test_rotating_cache_with_pinned_tokens_is_never_merged(self) -> None:
        sequences = [[11, 12, 91], [21, 92]]
        runtime = FakeMLX(prompt_tokens=sequences)
        backend = self.backend(runtime, batch_size=2)

        with patch.object(FakeCacheLayer, "keep", 4, create=True), patch.object(
            FakeCacheLayer, "merge", side_effect=AssertionError("pinned state must not be merged"),
        ) as merge:
            backend.score(model="local", prompts=[text_prompt("a"), text_prompt("b")])

        merge.assert_not_called()
        self.assertTrue(all(len(values) == 1 for values, _ in runtime.model.calls))
        self.assertEqual(runtime.model.contexts[-2:], [[sequences[0]], [sequences[1]]])

    def test_merge_capability_errors_fall_back_before_forward(self) -> None:
        for error in (NotImplementedError("unsupported"), ValueError("incompatible")):
            with self.subTest(error=type(error).__name__):
                runtime = FakeMLX(prompt_tokens=[[11, 91], [21, 92]])
                backend = self.backend(runtime, batch_size=2)
                with patch.object(FakeCacheLayer, "merge", side_effect=error):
                    backend.score(model="local", prompts=[text_prompt("a"), text_prompt("b")])

                self.assertTrue(all(len(values) == 1 for values, _ in runtime.model.calls))
                self.assertEqual(runtime.model.contexts[-2:], [[[11, 91]], [[21, 92]]])

    def test_forward_failure_is_propagated_without_retrying_mutated_batch(self) -> None:
        runtime = FakeMLX(prompt_tokens=[[11, 91], [21, 92]])
        backend = self.backend(runtime, batch_size=2)
        original_forward = FakeModel.__call__

        def failing_forward(model, tokens, *, cache):
            original_forward(model, tokens, cache=cache)
            raise RuntimeError("forward failed after updating state")

        with patch.object(FakeModel, "__call__", failing_forward):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                backend.score(model="local", prompts=[text_prompt("a"), text_prompt("b")])

        self.assertEqual([values for values, _ in runtime.model.calls], [[[11], [21]]])

    def test_known_models_that_unpack_cache_state_use_individual_caches(self) -> None:
        sequences = [[11, 12, 21, 91], [11, 12, 22, 92]]
        for model_type in ("afm7", "gemma3n"):
            for source in ("model_type", "config", "args", "module"):
                with self.subTest(model_type=model_type, source=source):
                    class IndividualCacheModel(FakeModel):
                        def __call__(self, tokens, *, cache):
                            if any(isinstance(layer, FakeBatchCacheLayer) for layer in cache):
                                raise AssertionError("this model requires ordinary cache state")
                            return super().__call__(tokens, cache=cache)

                    runtime = FakeMLX(prompt_tokens=sequences)
                    runtime.model = IndividualCacheModel()
                    runtime.load.return_value = (runtime.model, runtime.tokenizer)
                    if source == "module":
                        IndividualCacheModel.__module__ = f"mlx_lm.models.{model_type}"
                    elif source == "model_type":
                        runtime.model.model_type = model_type
                    else:
                        setattr(runtime.model, source, SimpleNamespace(model_type=model_type))
                    backend = self.backend(runtime, batch_size=8, max_cache_entries=8)
                    prompts = [text_prompt("a"), text_prompt("b")]

                    first = backend.score(model="local", prompts=prompts)

                    self.assertTrue(all(len(values) == 1 for values, _ in runtime.model.calls))
                    computed = [token for values, _ in runtime.model.calls
                                for row in values for token in row]
                    self.assertEqual(computed.count(11), 1)
                    self.assertEqual(computed.count(12), 1)
                    runtime.model.calls.clear()
                    second = backend.score(model="local", prompts=prompts)
                    self.assertEqual([values for values, _ in runtime.model.calls], [[[91]], [[92]]])
                    self.assertEqual(runtime.model.contexts[-2:], [[sequences[0]], [sequences[1]]])
                    self.assertEqual(first, second)
                    self.assertEqual(second.usage, Usage(input_tokens=8, output_tokens=0))

    def test_model_type_fallback_handles_mapping_config_and_does_not_disable_other_models(self) -> None:
        for model_type, expected_sizes in (("gemma3n_text", [1, 1]), ("qwen2", [2])):
            with self.subTest(model_type=model_type):
                runtime = FakeMLX(prompt_tokens=[[91], [92]])
                runtime.model.config = {"model_type": model_type}
                backend = self.backend(runtime, batch_size=2)

                backend.score(model="local", prompts=[text_prompt("a"), text_prompt("b")])

                self.assertEqual([len(values) for values, _ in runtime.model.calls], expected_sizes)

    def test_clear_cache_recomputes_prefill_without_reloading_weights(self) -> None:
        runtime = FakeMLX(prompt_tokens=[[11, 12, 91], [11, 12, 92]])
        backend = self.backend(runtime, batch_size=2, max_cache_entries=8)
        prompts = [text_prompt("a"), text_prompt("b")]
        first = backend.score(model="local", prompts=prompts)
        first_calls = [values for values, _ in runtime.model.calls]
        backend.clear_cache()
        runtime.model.calls.clear()

        second = backend.score(model="local", prompts=prompts)

        self.assertEqual([values for values, _ in runtime.model.calls], first_calls)
        self.assertEqual(first, second)
        runtime.load.assert_called_once()
        self.assertIs(backend.model, runtime.model)

    def test_rejects_invalid_batch_cache_and_submission_options_before_loading(self) -> None:
        runtime = FakeMLX()
        invalid = {
            "batch_size": (0, -1, True, False, 1.5, "2", None),
            "max_cache_entries": (-1, True, False, 1.5, "2", None),
            "max_cache_bytes": (-1, True, False, 1.5, "2", None),
            "submission": ("parallel", "", True, None),
        }
        with patch.dict("sys.modules", runtime.modules):
            for name, values in invalid.items():
                for value in values:
                    with self.subTest(name=name, value=value):
                        with self.assertRaisesRegex(ValueError, name):
                            MLXBackend(self.model_path, **{name: value})

        runtime.load.assert_not_called()

    def test_close_releases_model_and_cache_once_and_rejects_future_scoring(self) -> None:
        runtime = FakeMLX()
        backend = self.backend(runtime, max_cache_entries=8)
        backend.score(model="local", prompts=[text_prompt()])
        scorer = backend._text_scorer
        self.assertGreater(len(scorer.prefix_cache), 0)

        backend.close()
        clear_calls = runtime.core.clear_cache.call_count
        backend.close()

        runtime.core.synchronize.assert_called_once_with()
        self.assertEqual(runtime.core.clear_cache.call_count, clear_calls)
        self.assertIsNone(backend.model)
        self.assertIsNone(backend.tokenizer)
        self.assertIsNone(backend.processor)
        self.assertIsNone(backend._text_scorer)
        self.assertEqual(len(scorer.prefix_cache), 0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.score(model="local", prompts=[text_prompt()])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.__enter__()

    def test_context_manager_closes_backend_on_normal_and_exceptional_exit(self) -> None:
        for should_raise in (False, True):
            with self.subTest(should_raise=should_raise):
                runtime = FakeMLX()
                backend = self.backend(runtime)
                caught = False
                try:
                    with backend as entered:
                        self.assertIs(entered, backend)
                        entered.score(model="local", prompts=[text_prompt()])
                        if should_raise:
                            raise ValueError("caller failure")
                except ValueError as error:
                    caught = True
                    self.assertTrue(should_raise)
                    self.assertEqual(str(error), "caller failure")
                self.assertEqual(caught, should_raise)
                self.assertIsNone(backend.model)
                runtime.core.synchronize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
