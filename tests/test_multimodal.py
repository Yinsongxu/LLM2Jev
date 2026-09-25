import base64
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llm2jev import (
    Choice, DefaultPromptRenderer, JevRequest, LLM2Jev, Noul, Score,
    SGLangBackend, TransformersBackend, compile_binary_questions,
)
from llm2jev.backend.transformers.image_inputs import load_transformers_images, prepare_image_prompts
from llm2jev.backend.sglang.scoring import score_output
from llm2jev.server.sglang_server import _evaluate_request, _parse_request
from test_sglang_fakes import native_modules, score_result


def image_text(path="image.png", text="Evidence"):
    return {"type": "multimodal", "content": [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": path}},
    ]}


class MultimodalContentTests(unittest.TestCase):
    def test_serializes_and_parses_both_locations_for_all_types(self):
        for kind in (Choice, Score, Noul):
            kwargs = {"criteria": {"a": None, "b": None}} if kind is Choice else (
                {"criteria": ["low", "high"]} if kind is Score else {}
            )
            request = JevRequest(
                state=image_text("shared.png"), model="local",
                questions={"q": kind(instructions=image_text("question.png"), **kwargs)},
            )
            self.assertEqual(_parse_request(request.to_dict()), request)
            tasks = compile_binary_questions(request)
            for task in tasks:
                self.assertEqual(task.context, image_text("shared.png"))
                self.assertEqual(task.objective, image_text("question.png"))

    def test_rejects_malformed_tagged_content_at_both_locations(self):
        invalid = [
            {"type": "multimodal"}, image_text(""), image_text("   "),
            image_text(None), image_text(["a.png"]), image_text(text=3),
            {**image_text(), "extra": True},
            {"type": "multimodal", "content": []},
            {"type": "multimodal", "content": "text"},
        ]
        invalid.extend({"type": "multimodal", "content": [part]} for part in (
            "text", {}, {"type": "audio"}, {"type": "text"},
            {"type": "text", "text": "ok", "extra": True},
            {"type": "image_url", "image_url": "file:///a.png"},
            {"type": "image_url", "image_url": {}},
            {"type": "image_url", "image_url": {"url": "a.png", "detail": "high"}},
        ))
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    JevRequest(state=value, model="local", questions={"q": Noul()})
                for kind in (Choice, Score, Noul):
                    kwargs = {"criteria": {"a": None}} if kind is Choice else (
                        {"criteria": ["low", "high"]} if kind is Score else {}
                    )
                    with self.assertRaises(ValueError):
                        kind(instructions=value, **kwargs)

    def test_plain_image_key_and_nested_tag_remain_json(self):
        for value in ({"image": "business value", "text": "description"},
                      {"nested": image_text()}, [image_text()], image_text()["content"]):
            request = JevRequest(state=value, model="local", questions={"q": Noul(instructions=value)})
            content = DefaultPromptRenderer().render(compile_binary_questions(request)[0])[1]["content"]
            self.assertIsInstance(content, str)
            self.assertIn('"', content)

    def test_images_follow_their_text_and_precede_candidates(self):
        request = JevRequest(
            state=image_text("shared.png", "shared"), model="local",
            questions={"q": Choice(instructions=image_text("q.png", "question"),
                                   criteria={"a": "first", "b": "second"})},
        )
        prompts = [DefaultPromptRenderer().render(t) for t in compile_binary_questions(request)]
        first, second = (p[1]["content"] for p in prompts)
        self.assertEqual([p["type"] for p in first], ["text", "text", "image_url", "text", "text", "image_url", "text"])
        self.assertEqual(first[:-1], second[:-1])
        self.assertEqual(first[0]["text"] + first[1]["text"], "Context:\nshared")
        self.assertEqual(first[2]["image_url"]["url"], "shared.png")
        self.assertIn("question", first[4]["text"])
        self.assertEqual(first[5]["image_url"]["url"], "q.png")
        self.assertIn(
            'Is this candidate "a: first" the best answer?',
            first[6]["text"],
        )
        self.assertIn("inside the context or images", prompts[0][0]["content"])

    def test_each_placement_and_noul_negative_preserve_image_prefix(self):
        for in_state in (True, False):
            request = JevRequest(
                state=image_text() if in_state else "shared", model="local",
                questions={"q": Noul(instructions="Is it red?" if in_state else image_text(text="Is it red?"),
                                     criteria={"true": "Red", "false": "Not red"})},
            )
            parts = [DefaultPromptRenderer().render(t)[1]["content"] for t in compile_binary_questions(request)]
            self.assertEqual(parts[0][:-1], parts[1][:-1])
            self.assertNotIn("above yes?", parts[0][-1]["text"])
            self.assertNotIn("above no?", parts[1][-1]["text"])
            self.assertIn("All candidates:\n- Red\n- Not red", parts[0][-1]["text"])
            self.assertIn('Is this candidate "Red" the best answer?', parts[0][-1]["text"])
            self.assertIn('Is this candidate "Not red" the best answer?', parts[1][-1]["text"])
            self.assertEqual(sum(p["type"] == "image_url" for p in parts[0]), 1)

    def test_interleaved_parts_keep_order_and_do_not_alias_request(self):
        value = image_text("first.png")
        value["content"].extend(image_text("second.png", "Compare these")["content"])
        request = JevRequest(state=value, model="local", questions={"q": Noul()})
        task = compile_binary_questions(request)[0]
        parts = DefaultPromptRenderer().render(task)[1]["content"]
        self.assertEqual(parts[1:5], value["content"])
        parts[2]["image_url"]["url"] = "changed.png"
        self.assertEqual(task.context["content"][1]["image_url"]["url"], "first.png")

    def test_text_only_and_image_only_blocks_are_valid(self):
        for content in ([{"type": "text", "text": ""}],
                        [{"type": "image_url", "image_url": {"url": "file:///a.png"}}]):
            request = JevRequest(state={"type": "multimodal", "content": content},
                                 model="local", questions={"q": Noul()})
            self.assertEqual(_parse_request(request.to_dict()), request)
            rendered = DefaultPromptRenderer().render(compile_binary_questions(request)[0])
            self.assertIsInstance(rendered[1]["content"], str if content[0]["type"] == "text" else list)


class ImageBackendTests(unittest.TestCase):
    def setUp(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is optional")
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = str((Path(directory.name) / "red.png").resolve())
        Image.new("RGB", (8, 8), "red").save(self.path)
        self.processor = Mock()
        self.processor.apply_chat_template.side_effect = lambda messages, **kwargs: "".join(
            message["content"] if isinstance(message["content"], str) else "".join(
                p["text"] if p["type"] == "text" else "<image>" for p in message["content"]
            ) for message in messages
        )
        self.request = JevRequest(
            state=image_text(self.path), model="local",
            questions={"q": Choice(instructions="What color?", criteria={"red": None, "blue": None})},
        )
        self.prompts = tuple(DefaultPromptRenderer().render(t) for t in compile_binary_questions(self.request))

    def test_separates_sources_from_template_without_loading_images(self):
        with patch("PIL.Image.open", side_effect=AssertionError("must use native loader")):
            texts, sources = prepare_image_prompts(self.processor, self.prompts, enable_thinking=False)
        self.assertNotIn(self.path, texts[0])
        self.assertIn("<image>", texts[0])
        self.assertEqual(sources, [[self.path], [self.path]])
        self.processor.apply_chat_template.assert_called_with(
            unittest.mock.ANY, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def test_missing_or_invalid_image_fails_in_transformers_loader(self):
        self.require_transformers()
        for path in (str(Path(self.path).with_name("missing.png")), __file__):
            with self.assertRaises((ValueError, OSError)):
                load_transformers_images([[path]])

    def require_transformers(self):
        try:
            from transformers.image_utils import load_image
        except ImportError:
            self.skipTest("Transformers vision dependencies are optional")
        return load_image

    def test_transformers_reuses_native_loader_for_file_uri_and_data_url(self):
        native = self.require_transformers()
        spaced = Path(self.path).with_name("red image.png")
        spaced.write_bytes(Path(self.path).read_bytes())
        data = "data:image/png;base64," + base64.b64encode(spaced.read_bytes()).decode("ascii")
        with patch("transformers.image_utils.load_image", wraps=native) as loader:
            images = load_transformers_images([[spaced.as_uri(), data], [str(spaced)]])
        self.assertEqual(loader.call_count, 2)
        loader.assert_any_call(str(spaced))
        loader.assert_any_call(data)
        self.assertIs(images[0][0], images[1][0])
        self.assertEqual(images[0][1].getpixel((0, 0)), (255, 0, 0))

    def test_urls_are_forwarded_to_native_loaders_unchanged(self):
        sources = ["file:///data/red%20image.png", "https://example.test/a.png",
                   "http://example.test/b.png", "data:image/png;base64,abc"]
        prompts = (({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": source}} for source in sources
        ]},),)
        _, actual = prepare_image_prompts(self.processor, prompts, enable_thinking=False)
        self.assertEqual(actual, [sources])
        self.require_transformers()
        with patch("transformers.image_utils.load_image") as loader:
            load_transformers_images(actual)
        self.assertEqual([call.args[0] for call in loader.call_args_list],
                         ["/data/red image.png", *sources[1:]])

    def test_distinct_images_and_mixed_text_inputs_stay_separate(self):
        from PIL import Image

        blue_path = str(Path(self.path).with_name("blue.png"))
        Image.new("RGB", (8, 8), "blue").save(blue_path)
        prompts = (
            self.prompts[0],
            ({"role": "user", "content": "text only"},),
            ({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": blue_path}},
                {"type": "image_url", "image_url": {"url": self.path}},
            ]},),
        )
        _, images = prepare_image_prompts(self.processor, prompts, enable_thinking=False)
        self.assertEqual([len(row) for row in images], [1, 0, 2])
        self.assertEqual(images, [[self.path], [], [blue_path, self.path]])

    def test_transformers_constructor_loads_image_model_and_processor(self):
        tokenizer = self.processor.tokenizer
        tokenizer.pad_token_id = 0
        tokenizer.encode.side_effect = lambda label, **kwargs: {"yes": [7], "no": [9]}[label]
        auto_processor = Mock()
        auto_processor.from_pretrained.return_value = self.processor
        auto_model = Mock()
        text_model = Mock()
        modules = {"transformers": SimpleNamespace(
            AutoProcessor=auto_processor, AutoModelForImageTextToText=auto_model,
            AutoModelForCausalLM=text_model, AutoTokenizer=Mock(),
        ), "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))}
        with patch.dict("sys.modules", modules):
            backend = TransformersBackend("local", multimodal=True)
        self.assertIs(backend.processor, self.processor)
        auto_processor.from_pretrained.assert_called_once_with("local", local_files_only=True)
        auto_model.from_pretrained.assert_called_once_with("local", local_files_only=True)
        text_model.from_pretrained.assert_not_called()

    def test_transformers_forwards_visual_metadata_and_scores_last_nonpadding_token(self):
        self.require_transformers()
        try:
            import torch
        except ImportError:
            self.skipTest("torch is optional")
        backend = TransformersBackend.__new__(TransformersBackend)
        backend._torch = torch
        backend.processor = self.processor
        backend.tokenizer = Mock()
        backend.batch_size = 2
        backend.enable_thinking = False
        backend.device = "cpu"
        backend.no_token_id, backend.yes_token_id = 9, 7
        inputs = {
            "input_ids": torch.tensor([[1, 2, 0], [0, 1, 2]]),
            "attention_mask": torch.tensor([[1, 1, 0], [0, 1, 1]]),
            "pixel_values": torch.ones(2, 3, 8, 8),
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
        }
        self.processor.return_value = inputs
        logits = torch.zeros(2, 3, 10)
        logits[0, 1, 7] = math.log(4)
        logits[1, 2, 9] = math.log(4)
        logits[0, 2, 9] = 100  # Padding must not affect the score.
        backend.model = Mock(return_value=SimpleNamespace(logits=logits))
        output = backend.score(model="local", prompts=self.prompts)
        self.assertAlmostEqual(output.yes_probabilities[0], .8, places=6)
        self.assertAlmostEqual(output.yes_probabilities[1], .2, places=6)
        self.assertEqual(output.usage.input_tokens, 4)
        self.assertEqual(output.usage.output_tokens, 0)
        self.assertIs(backend.model.call_args.kwargs["pixel_values"], inputs["pixel_values"])
        self.assertIs(backend.model.call_args.kwargs["image_grid_thw"], inputs["image_grid_thw"])
        self.assertFalse(backend.model.call_args.kwargs["use_cache"])
        self.assertEqual(len(self.processor.call_args.kwargs["images"]), 2)
        backend.tokenizer.assert_not_called()

    def test_text_transformers_rejects_image_instead_of_silently_ignoring_it(self):
        backend = TransformersBackend.__new__(TransformersBackend)
        backend.processor = None
        with self.assertRaisesRegex(ValueError, "multimodal=True"):
            backend.score(model="local", prompts=self.prompts)

    def test_sglang_native_images_all_and_staged_restore_results_and_usage(self):
        modules = patch.dict("sys.modules", native_modules())
        modules.start()
        self.addCleanup(modules.stop)
        for mode in ("all", "staged"):
            backend = SGLangBackend.__new__(SGLangBackend)
            backend.tokenizer = Mock()
            backend.enable_thinking = False
            backend.submission = mode
            backend.no_token_id, backend.yes_token_id = 9, 7
            backend.engine = Mock()
            backend.engine.tokenizer_manager = SimpleNamespace(processor=self.processor)
            backend.engine.generate.side_effect = lambda **kwargs: [
                score_result(.8 if "Candidate: red" in text else .2)
                for text in kwargs["prompt"]
            ]
            response = LLM2Jev(backend=backend).evaluate(self.request)
            self.assertEqual(response.answers["q"].choice, "red")
            self.assertEqual(response.usage.input_tokens, 200)
            self.assertEqual(response.usage.output_tokens, 0)
            self.assertEqual(backend.engine.generate.call_count, 2 if mode == "staged" else 1)
            for call in backend.engine.generate.call_args_list:
                self.assertEqual(call.kwargs["sampling_params"], {"max_new_tokens": 0})
                self.assertEqual(call.kwargs["logprob_start_len"], -1)
                self.assertEqual(call.kwargs["token_ids_logprob"], [9, 7])
                self.assertEqual(len(call.kwargs["image_data"]), len(call.kwargs["prompt"]))
                self.assertEqual(call.kwargs["image_data"][0][0].url, self.path)
            backend.engine.score.assert_not_called()

    def test_sglang_staged_failure_stops_remaining_candidates(self):
        modules = patch.dict("sys.modules", native_modules())
        modules.start()
        self.addCleanup(modules.stop)
        backend = SGLangBackend.__new__(SGLangBackend)
        backend.tokenizer = Mock()
        backend.enable_thinking = False
        backend.submission = "staged"
        backend.no_token_id, backend.yes_token_id = 9, 7
        backend.engine = Mock()
        backend.engine.tokenizer_manager = SimpleNamespace(processor=self.processor)
        backend.engine.generate.return_value = []
        with self.assertRaisesRegex(ValueError, "wrong number"):
            backend.score(model="local", prompts=self.prompts)
        backend.engine.generate.assert_called_once()


class ImageScoreOutputTests(unittest.TestCase):
    def test_uses_next_token_labels_by_id_and_stable_normalization(self):
        result = score_result()
        result["meta_info"]["output_token_ids_logprobs"] = [[[-1000, 7, None], [-1001, 9, None]]]
        result["meta_info"]["input_token_ids_logprobs"] = [[[0, 9, None], [-100, 7, None]]]
        output = score_output([result], expected_count=1, no_token_id=9, yes_token_id=7)
        self.assertAlmostEqual(output.yes_probabilities[0], 1 / (1 + math.exp(-1)))

    def test_rejects_wrong_positions_count_nonfinite_and_generated_tokens(self):
        for positions in ([], [[[0, 9, None], [0, 7, None]]] * 2,
                          [[[math.nan, 9, None], [0, 7, None]]],
                          [[[0, 9, None], [math.nan, 7, None]]],
                          [[[math.inf, 9, None], [0, 7, None]]],
                          [[[0, 9, None], [math.inf, 7, None]]],
                          [[[-math.inf, 9, None], [-math.inf, 7, None]]]):
            result = score_result()
            result["meta_info"]["output_token_ids_logprobs"] = positions
            with self.assertRaises(ValueError):
                score_output([result], expected_count=1, no_token_id=9, yes_token_id=7)
        with self.assertRaises(ValueError):
            score_output([], expected_count=1, no_token_id=9, yes_token_id=7)
        result = score_result()
        result["meta_info"]["completion_tokens"] = 1
        with self.assertRaises(ValueError):
            score_output([result], expected_count=1, no_token_id=9, yes_token_id=7)

    def test_missing_native_fields_and_labels_raise_key_error(self):
        for field in ("completion_tokens", "output_token_ids_logprobs", "prompt_tokens"):
            result = score_result()
            del result["meta_info"][field]
            with self.subTest(field=field), self.assertRaises(KeyError):
                score_output([result], expected_count=1, no_token_id=9, yes_token_id=7)
        for remaining_token in (9, 7):
            result = score_result()
            result["meta_info"]["output_token_ids_logprobs"] = [[[0, remaining_token, None]]]
            with self.subTest(token=remaining_token), self.assertRaises(KeyError):
                score_output([result], expected_count=1, no_token_id=9, yes_token_id=7)

    def test_single_zero_mass_label_is_valid(self):
        for no, yes, expected in ((-math.inf, 0, 1), (0, -math.inf, 0)):
            result = score_result()
            result["meta_info"]["output_token_ids_logprobs"] = [[[no, 9, None], [yes, 7, None]]]
            output = score_output([result], expected_count=1, no_token_id=9, yes_token_id=7)
            self.assertEqual(output.yes_probabilities, (expected,))


class ImageHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_uses_native_image_request_and_zero_generation(self):
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda label, **kwargs: {"yes": [7], "no": [9]}[label]
        requests = []

        async def generate(request, _):
            requests.append(request)
            yield [score_result() for _ in request.text]

        processor = Mock()
        processor.apply_chat_template.return_value = "rendered"
        manager = SimpleNamespace(tokenizer=tokenizer, processor=processor, generate_request=generate)
        request = JevRequest(state=image_text("file:///data/no-local-read.png"), model="local", questions={"q": Noul()})
        with patch.dict("sys.modules", native_modules()):
            response = await _evaluate_request(request, manager)
        self.assertAlmostEqual(response.answers["q"].noul, .8)
        self.assertEqual(response.usage.input_tokens, 100)
        self.assertEqual(requests[0].image_data[0][0].url, "file:///data/no-local-read.png")
        self.assertEqual(requests[0].sampling_params, {"max_new_tokens": 0})


if __name__ == "__main__":
    unittest.main()
