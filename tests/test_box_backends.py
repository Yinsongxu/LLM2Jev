import unittest
import math
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llm2jev import Box, MLXBackend, Noul, SGLangBackend, TransformersBackend, Usage
from llm2jev.backend.sglang.generation import generation_output, prepare_generation_request
from llm2jev import DefaultPromptRenderer
from llm2jev.server.sglang_server import _evaluate_request
from test_box import box_request, image_content
from test_sglang_fakes import native_modules, score_result


def generated_result(text='{"bbox_2d":[100,200,800,900]}', tokens=50, reason="stop"):
    return {"text": text, "meta_info": {
        "output_token_logprobs": [[math.log(.8), ord(char), None] for char in text] +
                                 [[0., 0, None]] * (tokens - len(text)),
        "prompt_tokens": 50, "completion_tokens": tokens,
        "finish_reason": {"type": reason},
    }}


def processor():
    result = Mock()
    result.image_processor = object()
    result.apply_chat_template.side_effect = lambda messages, **kwargs: str(messages)
    return result


class SGLangBoxTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch.dict("sys.modules", native_modules())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tokenizer = Mock()
        self.tokenizer.decode.side_effect = lambda ids, **kwargs: "".join(chr(i) for i in ids if i)
        self.tokenizer.encode.side_effect = lambda label, **kwargs: {"no": [9], "yes": [7]}[label]
        self.processor = processor()
        self.prompts = DefaultPromptRenderer().render(box_request()).box_prompts

    def test_generation_preparation_uses_native_images_without_scoring_parameters(self):
        args = prepare_generation_request(self.tokenizer, self.processor, self.prompts, max_new_tokens=128)
        self.assertIsNone(args["input_ids"])
        self.assertEqual(args["sampling_params"], {"max_new_tokens": 128, "temperature": 0})
        self.assertTrue(args["return_logprob"])
        self.assertEqual(args["logprob_start_len"], -1)
        self.assertNotIn("token_ids_logprob", args)
        self.assertEqual(len(args["image_data"][0]), 1)
        self.assertTrue(args["image_data"][0][0].url.endswith("target.png"))
        self.assertFalse(self.processor.apply_chat_template.call_args.kwargs["enable_thinking"])
        with self.assertRaisesRegex(ValueError, "multimodal"):
            prepare_generation_request(self.tokenizer, None, self.prompts, max_new_tokens=128)

    def test_generation_results_require_complete_output_and_correct_count(self):
        result = generation_output([generated_result()], expected_count=1, tokenizer=self.tokenizer)
        self.assertEqual(result.texts, ('{"bbox_2d":[100,200,800,900]}',))
        self.assertEqual(result.usage, Usage(input_tokens=50, output_tokens=50))
        for probability, piece in zip(result.token_probabilities[0], result.token_texts[0]):
            self.assertAlmostEqual(probability, .8 if piece else 1.)
        for values in ([], [generated_result(reason="length")], [generated_result(reason="abort")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                generation_output(values, expected_count=1, tokenizer=self.tokenizer)

    def test_generation_converts_zero_and_negative_infinite_logprobs_to_probabilities(self):
        result = generated_result()
        entries = result["meta_info"]["output_token_logprobs"]
        entries[0][0], entries[1][0] = 0., -math.inf
        output = generation_output([result], expected_count=1, tokenizer=self.tokenizer)
        self.assertEqual(output.token_probabilities[0][:2], (1., 0.))

    def test_generation_rejects_missing_or_incomplete_sampled_logprobs(self):
        for entries in (None, [], [[-.2, 100, None]]):
            result = generated_result()
            result["meta_info"]["output_token_logprobs"] = entries
            with self.subTest(entries=entries), self.assertRaisesRegex(ValueError, "logprobs"):
                generation_output([result], expected_count=1, tokenizer=self.tokenizer)

    def test_generation_rejects_text_that_does_not_match_token_decoding(self):
        result = generated_result()
        result["text"] = '[0,0,1000,1000]'
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            generation_output([result], expected_count=1, tokenizer=self.tokenizer)

    def test_library_generation_reuses_loaded_engine(self):
        backend = SGLangBackend.__new__(SGLangBackend)
        backend.tokenizer = self.tokenizer
        backend.engine = Mock()
        backend.engine.tokenizer_manager = SimpleNamespace(processor=self.processor)
        backend.engine.generate.return_value = [generated_result()]
        result = backend.generate(model="alias", prompts=self.prompts, max_new_tokens=128)
        self.assertEqual(result.usage.output_tokens, 50)
        args = backend.engine.generate.call_args.kwargs
        self.assertTrue(args["prompt"])
        self.assertEqual(args["sampling_params"]["temperature"], 0)
        with self.assertRaisesRegex(ValueError, "empty"):
            backend.generate(model="local", prompts=(), max_new_tokens=128)
        backend.engine = None
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.generate(model="local", prompts=self.prompts, max_new_tokens=128)

    async def test_http_box_uses_generation_probabilities_without_verification(self):
        closed, submitted = [], []

        async def generate(request, raw_request):
            submitted.append(request)
            self.assertIsNone(raw_request)
            try:
                if request.sampling_params["max_new_tokens"]:
                    yield [generated_result()]
                else:
                    yield [score_result(.72, 70)]
            finally:
                closed.append(request)

        manager = SimpleNamespace(tokenizer=self.tokenizer, processor=self.processor,
                                  generate_request=Mock(side_effect=generate))
        result = await _evaluate_request(box_request(), manager)
        self.assertEqual(len(submitted), 1)
        self.assertEqual(closed, submitted)
        self.assertAlmostEqual(result.boxes["target"].confidence, .8)
        self.assertEqual(result.usage, Usage(input_tokens=50, output_tokens=50))
        self.assertEqual(submitted[0].sampling_params["max_new_tokens"], 128)
        self.tokenizer.encode.assert_not_called()

    async def test_http_mixed_questions_preserve_box_order_and_score_original_candidates(self):
        submitted = []

        async def generate(request, raw_request):
            submitted.append(request)
            if request.sampling_params["max_new_tokens"]:
                yield [generated_result(), generated_result(text='[0,0,1000,1000]')]
            else:
                yield [score_result(.72, 70)]

        manager = SimpleNamespace(tokenizer=self.tokenizer, processor=self.processor,
                                  generate_request=Mock(side_effect=generate))
        request = box_request(first=Box(instructions="red rectangle"), present=Noul(),
                              second=Box(instructions="whole image"))
        response = await _evaluate_request(request, manager)
        self.assertEqual(tuple(response.answers), tuple(request.questions))
        self.assertAlmostEqual(response.nouls["present"].noul, .72)
        self.assertEqual(response.boxes["second"].box, {"left": 0., "top": 0., "right": 1., "bottom": 1.})
        self.assertEqual(response.usage, Usage(input_tokens=170, output_tokens=100))
        self.assertEqual([item.sampling_params["max_new_tokens"] for item in submitted], [128, 0])

    async def test_http_generation_and_assembly_failures_do_not_return_partial_answers(self):
        for text, reason in (("invalid", "stop"), ('[0,0,10,10]', "length")):
            closed = []

            async def generate(request, raw_request):
                try:
                    if request.sampling_params["max_new_tokens"]:
                        yield [generated_result(text=text, reason=reason)]
                    else:
                        yield [score_result(.72, 70)]
                finally:
                    closed.append(True)

            manager = SimpleNamespace(tokenizer=self.tokenizer, processor=self.processor,
                                      generate_request=Mock(side_effect=generate))
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                await _evaluate_request(box_request(present=Noul(), target=Box(instructions="x")), manager)
            expected_calls = 2 if reason == "stop" else 1
            self.assertEqual(manager.generate_request.call_count, expected_calls)
            self.assertEqual(closed, [True] * expected_calls)


class TransformersBoxTests(unittest.TestCase):
    def backend(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is optional")
        backend = TransformersBackend.__new__(TransformersBackend)
        backend._torch = torch
        backend.device = "cpu"
        backend.processor = processor()
        backend.tokenizer = Mock(pad_token_id=0)
        backend.tokenizer.decode.side_effect = lambda ids, **kwargs: ''.join({5: '[100,200,', 6: '800,900]', 9: ''}[i] for i in ids)
        backend.processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "pixel_values": torch.ones(1, 3, 8, 8),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
        backend.model = Mock()
        backend.model.generation_config = SimpleNamespace(eos_token_id=[9, 10])
        backend.model.generate.return_value = SimpleNamespace(
            sequences=torch.tensor([[1, 2, 3, 5, 6, 9]]),
            logits=tuple(torch.zeros(1, 11) for _ in range(3)),
        )
        return backend

    def test_generation_forwards_visual_metadata_decodes_only_new_tokens_and_counts_eos(self):
        backend = self.backend()
        request = box_request(target=Box(instructions=image_content("reference.png", "match this")))
        prompts = DefaultPromptRenderer().render(request).box_prompts
        with patch("llm2jev.backend.transformers.backend.load_transformers_images", return_value=[["target", "reference"]]):
            output = backend.generate(model="alias", prompts=prompts, max_new_tokens=128)
        self.assertEqual(output.usage, Usage(input_tokens=3, output_tokens=3))
        self.assertEqual(output.texts, ('[100,200,800,900]',))
        args = backend.model.generate.call_args.kwargs
        self.assertFalse(args["do_sample"])
        self.assertEqual(args["max_new_tokens"], 128)
        self.assertIs(args["image_grid_thw"], backend.processor.return_value["image_grid_thw"])
        self.assertEqual(backend.tokenizer.decode.call_args.args[0], [5, 6, 9])
        self.assertTrue(args["output_logits"])
        for probability in output.token_probabilities[0]:
            self.assertAlmostEqual(probability, 1 / 11, places=6)
        self.assertEqual(backend.processor.call_args.kwargs["images"], ["target", "reference"])
        self.assertFalse(backend.processor.apply_chat_template.call_args.kwargs["enable_thinking"])

    def test_truncated_output_is_rejected_even_when_json_looks_complete(self):
        backend = self.backend()
        backend.model.generate.return_value = SimpleNamespace(
            sequences=backend._torch.tensor([[1, 2, 3, 5, 6]]),
            logits=tuple(backend._torch.zeros(1, 11) for _ in range(2)),
        )
        prompts = DefaultPromptRenderer().render(box_request()).box_prompts
        with patch("llm2jev.backend.transformers.backend.load_transformers_images", return_value=[["target"]]):
            with self.assertRaisesRegex(ValueError, "truncated"):
                backend.generate(model="local", prompts=prompts, max_new_tokens=2)
        backend.model.generate.return_value.sequences = backend._torch.tensor([[1, 2, 3, 5, 9]])
        with patch("llm2jev.backend.transformers.backend.load_transformers_images", return_value=[["target"]]):
            self.assertEqual(backend.generate(model="local", prompts=prompts, max_new_tokens=2).usage.output_tokens, 2)

    def test_rejects_closed_text_only_and_empty_inputs(self):
        backend = self.backend()
        prompts = DefaultPromptRenderer().render(box_request()).box_prompts
        with self.assertRaisesRegex(ValueError, "empty"):
            backend.generate(model="local", prompts=(), max_new_tokens=128)
        backend.processor = None
        with self.assertRaisesRegex(ValueError, "multimodal=True"):
            backend.generate(model="local", prompts=prompts, max_new_tokens=128)
        backend.model = None
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.generate(model="local", prompts=prompts, max_new_tokens=128)


class MLXBoxTests(unittest.TestCase):
    def chunks(self, *args, **kwargs):
        class Logprobs:
            def reshape(self, *shape):
                return self
            def __getitem__(self, token):
                return SimpleNamespace(item=lambda: math.log(.8))
        for char in '[100,200,800,900]':
            yield SimpleNamespace(text=char, token=ord(char), logprobs=Logprobs(), finish_reason=None)
        yield SimpleNamespace(text="", token=0, logprobs=Logprobs(), finish_reason="stop",
                              prompt_tokens=55, generation_tokens=19)

    def setUp(self):
        self.native = Mock(side_effect=self.chunks)
        class BaseImageProcessor:
            pass
        patcher = patch.dict("sys.modules", {
            "mlx_vlm": SimpleNamespace(stream_generate=self.native),
            "mlx_vlm.models.base": SimpleNamespace(BaseImageProcessor=BaseImageProcessor),
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        self.legacy_processor = BaseImageProcessor()
        self.backend = MLXBackend.__new__(MLXBackend)
        self.backend._lock = RLock()
        self.backend.model = object()
        self.backend.processor = processor()
        self.backend.tokenizer = Mock()
        self.backend.tokenizer.decode.side_effect = lambda ids, **kwargs: "".join(chr(i) for i in ids if i)
        self.prompts = DefaultPromptRenderer().render(box_request()).box_prompts

    def test_native_generation_preserves_image_order_and_actual_usage(self):
        request = box_request(target=Box(instructions=image_content("file:///tmp/reference%20image.png")))
        output = self.backend.generate(model="alias", prompts=DefaultPromptRenderer().render(request).box_prompts, max_new_tokens=128)
        self.assertEqual(output.usage, Usage(input_tokens=55, output_tokens=19))
        args = self.native.call_args.kwargs
        self.assertTrue(args["image"][0].endswith("target.png"))
        self.assertEqual(args["image"][1], "/tmp/reference image.png")
        self.assertEqual(args["temperature"], 0)
        self.assertEqual(args["max_tokens"], 128)
        self.assertNotIn("prompt_cache", args)
        self.assertFalse(self.backend.processor.apply_chat_template.call_args.kwargs["enable_thinking"])
        for probability in output.token_probabilities[0]:
            self.assertAlmostEqual(probability, .8)

    def test_rejects_missing_sampled_token_logprobs(self):
        self.native.side_effect = lambda *args, **kwargs: iter([
            SimpleNamespace(text="[", token=1, logprobs=None, finish_reason=None),
        ])
        with self.assertRaisesRegex(ValueError, "sampled-token"):
            self.backend.generate(model="local", prompts=self.prompts, max_new_tokens=128)

    def test_rejects_stream_text_that_does_not_match_token_decoding(self):
        self.backend.tokenizer.decode.side_effect = None
        self.backend.tokenizer.decode.return_value = '[0,0,1000,1000]'
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            self.backend.generate(model="local", prompts=self.prompts, max_new_tokens=128)

    def test_requires_complete_generation(self):
        for reason in (None, "length", "abort"):
            self.native.side_effect = lambda *args, **kwargs: iter([SimpleNamespace(
                text="", token=0, logprobs=None, finish_reason=reason,
            )])
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, "did not complete|sampled-token"):
                self.backend.generate(model="local", prompts=self.prompts, max_new_tokens=128)

    def test_rejects_legacy_multi_image_processor_before_inference(self):
        self.backend.processor.image_processor = self.legacy_processor
        request = box_request(target=Box(instructions=image_content("reference.png")))
        with self.assertRaisesRegex(ValueError, "only one image"):
            self.backend.generate(model="local", prompts=DefaultPromptRenderer().render(request).box_prompts, max_new_tokens=128)
        self.native.assert_not_called()
        self.backend.generate(model="local", prompts=self.prompts, max_new_tokens=128)
        self.native.assert_called_once()

    def test_rejects_closed_text_only_and_empty_inputs(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.backend.generate(model="local", prompts=(), max_new_tokens=128)
        self.backend.processor = None
        with self.assertRaisesRegex(ValueError, "multimodal=True"):
            self.backend.generate(model="local", prompts=self.prompts, max_new_tokens=128)
        self.backend.model = None
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.backend.generate(model="local", prompts=self.prompts, max_new_tokens=128)
