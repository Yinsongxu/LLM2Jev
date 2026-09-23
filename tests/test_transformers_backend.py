import unittest

from llm2jev import TransformersBackend
from llm2jev.backend.tokenization import _apply_chat_template, _single_token_id


class FakeTokenizer:
    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids
        self.template_call = None

    def encode(self, label: str, *, add_special_tokens: bool) -> list[int]:
        self.encoded = (label, add_special_tokens)
        return self.token_ids

    def apply_chat_template(self, prompt, **kwargs) -> str:
        self.template_call = (prompt, kwargs)
        return "rendered prompt"


class TransformersBackendTests(unittest.TestCase):
    def test_rejects_invalid_batch_size_before_loading_model(self) -> None:
        with self.assertRaises(ValueError):
            TransformersBackend("unused", batch_size=0)

    def test_requires_a_single_token_label(self) -> None:
        tokenizer = FakeTokenizer([1, 2])

        with self.assertRaisesRegex(ValueError, "exactly one token"):
            _single_token_id(tokenizer, "yes", "yes_label")

    def test_applies_generation_template_without_thinking(self) -> None:
        tokenizer = FakeTokenizer([1])
        prompt = (
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        )

        rendered = _apply_chat_template(
            tokenizer,
            prompt,
            enable_thinking=False,
        )

        self.assertEqual(rendered, "rendered prompt")
        self.assertEqual(
            tokenizer.template_call,
            (
                list(prompt),
                {
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                },
            ),
        )

    def test_context_manager_closes_backend_and_rejects_future_scoring(self) -> None:
        backend = TransformersBackend.__new__(TransformersBackend)
        backend.model = object()
        backend.processor = object()
        backend.tokenizer = object()

        with backend as entered:
            self.assertIs(entered, backend)

        self.assertIsNone(backend.model)
        self.assertIsNone(backend.processor)
        self.assertIsNone(backend.tokenizer)
        backend.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.score(
                model="local", prompts=(({"role": "user", "content": "text"},),),
            )
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.__enter__()


if __name__ == "__main__":
    unittest.main()
