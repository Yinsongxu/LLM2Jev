import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llm2jev import JevRequest, LLM2Jev, Noul, SGLangBackend, Usage
from test_sglang_fakes import native_modules, score_result


class SGLangBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = Mock()
        self.tokenizer.encode.side_effect = lambda label, **kwargs: {
            "yes": [7], "no": [9], "multi": [3, 4], "alias": [7],
        }[label]
        self.tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: (
            "template:" + messages[-1]["content"]
        )
        self.tokenizer.return_value = {"input_ids": [[11, 12, 13], [11, 12, 14, 15]]}
        self.engine = Mock()
        self.engine.tokenizer_manager = SimpleNamespace(processor=None)
        self.engine.generate.side_effect = lambda **kwargs: [
            score_result({13: .8, 15: .3}[tokens[-1]], len(tokens))
            for tokens in kwargs["input_ids"]
        ]
        self.engine_class = Mock(return_value=self.engine)
        self.auto_tokenizer = Mock()
        self.auto_tokenizer.from_pretrained.return_value = self.tokenizer
        modules = patch.dict(sys.modules, {
            **native_modules(),
            "sglang": SimpleNamespace(Engine=self.engine_class),
            "transformers": SimpleNamespace(AutoTokenizer=self.auto_tokenizer),
        })
        modules.start()
        self.addCleanup(modules.stop)
        self.prompts = (
            ({"role": "user", "content": "first"},),
            ({"role": "user", "content": "second"},),
        )

    def test_scores_full_token_sequences_with_no_yes_normalization(self) -> None:
        options = {"schedule_policy": "lpm", "mem_fraction_static": 0.4}
        with SGLangBackend("local-model", submission="all", engine_kwargs=options) as backend:
            output = backend.score(model="request-model", prompts=self.prompts)
        self.engine_class.assert_called_once_with(model_path="local-model", **options)
        self.auto_tokenizer.from_pretrained.assert_called_once_with(
            "local-model", local_files_only=True,
        )
        self.tokenizer.assert_called_once_with(
            ["template:first", "template:second"], add_special_tokens=False,
        )
        self.engine.generate.assert_called_once_with(
            prompt=None, input_ids=[[11, 12, 13], [11, 12, 14, 15]], image_data=None,
            token_ids_logprob=[9, 7], sampling_params={"max_new_tokens": 0},
            return_logprob=True, logprob_start_len=-1,
        )
        for actual, expected in zip(output.yes_probabilities, (.8, .3)):
            self.assertAlmostEqual(actual, expected)
        self.engine.score.assert_not_called()
        self.assertEqual(output.usage, Usage(input_tokens=7, output_tokens=0))
        self.assertEqual(options, {"schedule_policy": "lpm", "mem_fraction_static": 0.4})
        self.engine.shutdown.assert_called_once()

    def test_preserves_chat_template_and_thinking_setting(self) -> None:
        with SGLangBackend("local-model", enable_thinking=True) as backend:
            backend.score(model="model", prompts=self.prompts)
        self.tokenizer.apply_chat_template.assert_any_call(
            list(self.prompts[0]), tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )

    def test_applies_standalone_chat_template_from_object_file(self) -> None:
        with tempfile.TemporaryDirectory() as model_dir:
            template_path = Path(model_dir) / "chat_template.json"
            template_path.write_text(
                json.dumps({"chat_template": "standalone template"}), encoding="utf-8",
            )
            self.tokenizer.chat_template = None
            with SGLangBackend(model_dir) as backend:
                self.assertEqual(backend.tokenizer.chat_template, "standalone template")

    def test_applies_standalone_chat_template_from_raw_string_file(self) -> None:
        with tempfile.TemporaryDirectory() as model_dir:
            template_path = Path(model_dir) / "chat_template.json"
            template_path.write_text(json.dumps("raw template"), encoding="utf-8")
            self.tokenizer.chat_template = ""
            with SGLangBackend(model_dir) as backend:
                self.assertEqual(backend.tokenizer.chat_template, "raw template")

    def test_keeps_missing_chat_template_unset(self) -> None:
        with tempfile.TemporaryDirectory() as model_dir:
            self.tokenizer.chat_template = None
            with SGLangBackend(model_dir) as backend:
                self.assertIsNone(backend.tokenizer.chat_template)

    def test_skips_standalone_file_without_chat_template_field(self) -> None:
        with tempfile.TemporaryDirectory() as model_dir:
            template_path = Path(model_dir) / "chat_template.json"
            template_path.write_text("{}", encoding="utf-8")
            self.tokenizer.chat_template = None
            with SGLangBackend(model_dir) as backend:
                self.assertIsNone(backend.tokenizer.chat_template)

    def test_prefers_existing_chat_template_over_standalone_file(self) -> None:
        with tempfile.TemporaryDirectory() as model_dir:
            template_path = Path(model_dir) / "chat_template.json"
            template_path.write_text(
                json.dumps({"chat_template": "standalone template"}), encoding="utf-8",
            )
            self.tokenizer.chat_template = "configured template"
            with SGLangBackend(model_dir) as backend:
                self.assertEqual(backend.tokenizer.chat_template, "configured template")

    def test_default_staged_submission_preserves_full_inputs_output_order_and_usage(self) -> None:
        ids = [[1, 2, 3, 10], [1, 2, 3, 11], [1, 2, 4, 12], [1, 2, 4, 13]]
        original = [tokens[:] for tokens in ids]
        self.tokenizer.return_value = {"input_ids": ids}
        self.engine.generate.side_effect = lambda **kwargs: [
            score_result((tokens[-1] - 9) / 10, len(tokens)) for tokens in kwargs["input_ids"]
        ]
        with SGLangBackend("local-model") as backend:
            output = backend.score(model="model", prompts=self.prompts * 2)
        self.assertEqual([call.kwargs["input_ids"] for call in self.engine.generate.call_args_list],
                         [[ids[0]], [ids[1], ids[2]], [ids[3]]])
        for actual, expected in zip(output.yes_probabilities, (.1, .2, .3, .4)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(output.usage, Usage(input_tokens=16, output_tokens=0))
        self.assertEqual(ids, original)
        self.engine.flush_cache.assert_not_called()

    def test_rejects_invalid_submission_and_disabled_cache_before_engine_start(self) -> None:
        for kwargs in ({"submission": "auto"},
                       {"engine_kwargs": {"disable_radix_cache": True}},
                       {"submission": "staged", "engine_kwargs": {"disable_radix_cache": True}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SGLangBackend("local-model", **kwargs)
        self.engine_class.assert_not_called()

    def test_all_submission_allows_disabled_cache(self) -> None:
        with SGLangBackend("local-model", submission="all", engine_kwargs={"disable_radix_cache": True}) as backend:
            backend.score(model="model", prompts=self.prompts)
        self.engine.generate.assert_called_once()

    def test_staged_failure_does_not_return_partial_results_or_call_later_stages(self) -> None:
        self.tokenizer.return_value = {
            "input_ids": [[1, 2, 3, 10], [1, 2, 3, 11], [1, 2, 4, 12], [1, 2, 4, 13]],
        }
        self.engine.generate.side_effect = [[score_result()], []]
        with SGLangBackend("local-model", submission="staged") as backend:
            with self.assertRaisesRegex(ValueError, "wrong number"):
                backend.score(model="model", prompts=self.prompts * 2)
        self.assertEqual(self.engine.generate.call_count, 2)

    def test_rejects_invalid_labels_before_starting_engine(self) -> None:
        for kwargs in ({"yes_label": "multi"}, {"no_label": "alias"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SGLangBackend("local-model", **kwargs)
        self.engine_class.assert_not_called()

    def test_rejects_options_that_change_input_semantics(self) -> None:
        for options in ({"enable_mis": True}, {"model_path": "other-model"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SGLangBackend("local-model", engine_kwargs=options)
        self.engine_class.assert_not_called()

    def test_rejects_empty_prompts_without_calling_engine(self) -> None:
        with SGLangBackend("local-model") as backend:
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                backend.score(model="model", prompts=[])
        self.engine.generate.assert_not_called()

    def test_rejects_malformed_scores(self) -> None:
        missing_label = score_result()
        missing_label["meta_info"]["output_token_ids_logprobs"] = [[[0, 7, None]]]
        generated = score_result()
        generated["meta_info"]["completion_tokens"] = 1
        invalid = (([score_result()], ValueError), ([score_result(), missing_label], KeyError),
                   ([score_result(), score_result(float("nan"))], ValueError),
                   ([score_result(), generated], ValueError))
        self.engine.generate.side_effect = None
        with SGLangBackend("local-model", submission="all") as backend:
            for scores, error in invalid:
                with self.subTest(scores=scores), self.assertRaises(error):
                    self.engine.generate.return_value = scores
                    backend.score(model="model", prompts=self.prompts)

    def test_releases_engine_on_error_and_close_is_idempotent(self) -> None:
        backend = SGLangBackend("local-model")
        self.engine.generate.side_effect = RuntimeError("engine failed")
        with self.assertRaisesRegex(RuntimeError, "engine failed"), backend:
            backend.score(model="model", prompts=self.prompts)
        backend.close()
        self.engine.shutdown.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.score(model="model", prompts=self.prompts)

    def test_evaluates_jev_request(self) -> None:
        request = JevRequest(
            state="package delayed", model="model",
            questions={"delivery": Noul(), "refund": Noul()},
        )
        with SGLangBackend("local-model") as backend:
            response = LLM2Jev(backend=backend).evaluate(request)
        self.assertAlmostEqual(response.answers["delivery"].noul, 0.8)
        self.assertAlmostEqual(response.answers["refund"].noul, 0.3)
        self.assertEqual(response.usage.output_tokens, 0)

    def test_missing_optional_dependency_has_install_hint(self) -> None:
        with patch.dict(sys.modules, {"sglang": None}):
            with self.assertRaisesRegex(ImportError, r"llm2jev\[sglang\]"):
                SGLangBackend("local-model")


if __name__ == "__main__":
    unittest.main()
