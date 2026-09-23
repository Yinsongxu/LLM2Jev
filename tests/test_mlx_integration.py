"""Optional native MLX checks using tiny local models, with no model downloads."""

import importlib.util
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from llm2jev import MLXBackend, Usage


@unittest.skipUnless(
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None,
    "MLX dependencies are optional",
)
class MLXIntegrationTests(unittest.TestCase):
    def check_local_model(self, *, quantized: bool) -> None:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from mlx_lm.models.qwen2 import Model, ModelArgs
        from transformers import Qwen2Tokenizer

        # An ordinary MLX-LM checkpoint exercises its real loader, tokenizer,
        # attention masks and KV cache without a pretrained model or network.
        mx.random.seed(7)
        args = ModelArgs(
            model_type="qwen2", hidden_size=32, num_hidden_layers=2,
            intermediate_size=64, num_attention_heads=4,
            num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=64,
        )
        model = Model(args)
        config = asdict(args)
        if quantized:
            config["quantization"] = {"group_size": 32, "bits": 4}
            nn.quantize(model, **config["quantization"])

        words = ["yes", "no", "red", "blue", "assistant"]
        pieces = ["<unk>", "<eos>", "Ġ", *sorted(set("".join(words)))]
        vocab = {piece: index for index, piece in enumerate(pieces)}
        merges = []
        for word in words:
            for end in range(1, len(word)):
                merges.append((word[:end], word[end]))
                vocab[word[:end + 1]] = len(vocab)
        tokenizer = Qwen2Tokenizer(
            vocab=vocab, merges=merges, unk_token="<unk>", eos_token="<eos>",
        )
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['content'] }} {% endfor %}"
            "{% if add_generation_prompt %}assistant{% endif %}"
        )
        prompts = tuple(
            ({"role": "user", "content": content},)
            for content in ("", "red blue", "blue red " * 8, "red " * 270)
        )

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"HF_HUB_OFFLINE": "1"},
        ):
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
            mx.save_safetensors(
                str(path / "model.safetensors"), dict(tree_flatten(model.parameters())),
            )
            tokenizer.save_pretrained(path)

            scenarios = [
                (step_size, batch_size, submission, entries)
                for step_size in (3, 512)
                for batch_size, submission, entries in (
                    (1, "all", 0), (4, "all", 32), (4, "staged", 32),
                )
            ]
            for step_size, batch_size, submission, entries in scenarios:
                with self.subTest(quantized=quantized, scenario=(step_size, batch_size, submission, entries)):
                    backend = MLXBackend(
                        path, prefill_step_size=step_size, batch_size=batch_size,
                        submission=submission, max_cache_entries=entries,
                    )
                    output = backend.score(model="request-alias", prompts=prompts)
                    expected = []
                    input_tokens = 0
                    for prompt in prompts:
                        tokens = backend.tokenizer.apply_chat_template(
                            list(prompt), tokenize=True, add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        input_tokens += len(tokens)
                        logits = backend.model(mx.array([tokens]))[0, -1]
                        pair = logits[mx.array([backend.no_token_id, backend.yes_token_id])]
                        expected.append(float(mx.softmax(pair.astype(mx.float32))[1].item()))
                    for actual, reference in zip(output.yes_probabilities, expected):
                        self.assertAlmostEqual(actual, reference, delta=1e-5)
                    self.assertEqual(output.usage, Usage(input_tokens=input_tokens, output_tokens=0))

                    # Warm requests may reuse the matching prefix, but never
                    # inherit a different candidate's state. Count real model
                    # inputs to verify reuse and batching actually take place.
                    shapes = []
                    original_call = type(backend.model).__call__

                    def recording_call(model, inputs, *args, **kwargs):
                        shapes.append(inputs.shape)
                        return original_call(model, inputs, *args, **kwargs)

                    with patch.object(type(backend.model), "__call__", recording_call):
                        reordered = backend.score(model="another-alias", prompts=prompts[::-1])
                    for actual, reference in zip(reordered.yes_probabilities, expected[::-1]):
                        self.assertAlmostEqual(actual, reference, delta=1e-5)
                    self.assertEqual(reordered.usage, output.usage)
                    if entries:
                        self.assertEqual(sum(batch * length for batch, length in shapes), len(prompts))
                    else:
                        self.assertEqual(sum(batch * length for batch, length in shapes), input_tokens)
                    if batch_size > 1:
                        self.assertTrue(any(batch > 1 for batch, _ in shapes))
                    backend.close()

    def test_chunked_scoring_matches_full_forward(self) -> None:
        self.check_local_model(quantized=False)

    def test_quantized_checkpoint_scoring_matches_full_forward(self) -> None:
        self.check_local_model(quantized=True)

    def test_recurrent_cache_batching_and_branch_reuse_match_full_forward(self) -> None:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.models.mamba import Model, ModelArgs

        mx.random.seed(11)
        model = Model(ModelArgs(
            model_type="mamba", vocab_size=32, hidden_size=16,
            intermediate_size=32, state_size=4, num_hidden_layers=2,
            conv_kernel=4, use_bias=False, use_conv_bias=True, time_step_rank=2,
        ))
        self.check_cache_model(model, mx, make_prompt_cache)

    def test_rotating_cache_batching_and_window_overflow_preserve_scores(self) -> None:
        import mlx.core as mx
        from mlx_lm.models.cache import RotatingKVCache
        from mlx_lm.models.qwen2 import Model, ModelArgs

        mx.random.seed(13)
        model = Model(ModelArgs(
            model_type="qwen2", hidden_size=32, num_hidden_layers=2,
            intermediate_size=64, num_attention_heads=4,
            num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=32,
        ))
        self.check_cache_model(
            model, mx, lambda model: [RotatingKVCache(max_size=8) for _ in model.layers],
        )

    def check_cache_model(self, model, mx, make_cache) -> None:
        from llm2jev.backend.mlx.scoring import TextScorer

        model.eval()
        tokens = [[1, 2, 3], [1, 2, 3, 4, 5], [1, 2, 3, 6] * 7, [8]]
        expected = []
        for row in tokens:
            cache = make_cache(model)
            # A full prefill with the same model-native cache is an independent
            # reference, including the attention window for rotating caches.
            logits = model(mx.array([row]), cache=cache)[0, -1]
            expected.append(float(mx.softmax(logits[mx.array([0, 1])].astype(mx.float32))[1].item()))
        for submission in ("all", "staged"):
            scorer = TextScorer(
                model, mx, make_cache, batch_size=4, prefill_step_size=3,
                submission=submission, max_cache_entries=16, max_cache_bytes=1 << 20,
            )
            for rows, references in ((tokens, expected), (tokens[::-1], expected[::-1])):
                actual = scorer.score(rows, no_token_id=0, yes_token_id=1)
                for value, reference in zip(actual, references):
                    self.assertAlmostEqual(value, reference, delta=1e-5)
            scorer.clear_cache()


if __name__ == "__main__":
    unittest.main()
