import unittest
from collections.abc import Sequence

from llm2jev import (
    BinaryBackendOutput,
    ChatPrompt,
    Choice,
    ChoiceAnswer,
    JevRequest,
    LLM2Jev,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    Usage,
)


class FakeBackend:
    def __init__(
        self,
        probabilities: Sequence[float],
        usage: Usage | None = None,
    ) -> None:
        self.probabilities = tuple(probabilities)
        self.usage = usage or Usage()
        self.calls: list[tuple[str, tuple[ChatPrompt, ...]]] = []

    def score(
        self,
        *,
        model: str,
        prompts: Sequence[ChatPrompt],
    ) -> BinaryBackendOutput:
        prompt_values = tuple(prompts)
        self.calls.append((model, prompt_values))
        return BinaryBackendOutput(
            yes_probabilities=self.probabilities,
            usage=self.usage,
        )


class FailingBackend:
    def score(
        self,
        *,
        model: str,
        prompts: Sequence[ChatPrompt],
    ) -> BinaryBackendOutput:
        raise RuntimeError("backend failed")


class LLM2JevTests(unittest.TestCase):
    def test_evaluates_mixed_request_end_to_end(self) -> None:
        request = JevRequest(
            state={"message": "Duplicate charge; please refund me."},
            model="test-model",
            questions={
                "department": Choice(criteria={"billing": None, "returns": None}),
                "severity": Score(criteria=["Low", "High"]),
                "refund": Noul(),
                "explicit_refund": Noul(
                    criteria={"true": "Refund requested", "false": "No refund requested"},
                ),
            },
        )
        backend = FakeBackend(
            [0.8, 0.2, 0.25, 0.75, 0.9, 0.7, 0.3],
            Usage(input_tokens=140, output_tokens=0),
        )

        response = LLM2Jev(backend=backend).evaluate(request)

        self.assertEqual(response.model, "test-model")
        self.assertEqual(response.usage, Usage(input_tokens=140, output_tokens=0))
        self.assertIsInstance(response.answers["department"], ChoiceAnswer)
        self.assertIsInstance(response.answers["severity"], ScoreAnswer)
        self.assertEqual(response.answers["refund"], NoulAnswer(noul=0.9))
        self.assertEqual(response.answers["explicit_refund"], NoulAnswer(noul=0.7))

        self.assertEqual(len(backend.calls), 1)
        model, prompts = backend.calls[0]
        self.assertEqual(model, "test-model")
        self.assertEqual(len(prompts), 7)
        self.assertIn("All candidates:\nbilling\nreturns", prompts[0][1]["content"])
        self.assertIn(
            'Is this candidate "No refund requested" the best answer?',
            prompts[-1][1]["content"],
        )

    def test_uses_custom_renderer_and_normalizer(self) -> None:
        class FixedRenderer:
            def render(self, question: object) -> ChatPrompt:
                return (
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "fixed"},
                )

        request = JevRequest(
            state="message",
            model="test-model",
            questions={"tone": Choice(criteria={"calm": None, "angry": None})},
        )
        backend = FakeBackend([0.9, 0.1])
        engine = LLM2Jev(
            backend=backend,
            renderer=FixedRenderer(),
            normalizer=lambda _: (0.6, 0.4),
        )

        response = engine.evaluate(request)

        answer = response.answers["tone"]
        self.assertIsInstance(answer, ChoiceAnswer)
        self.assertEqual(answer.probabilities, {"calm": 0.6, "angry": 0.4})  # type: ignore[union-attr]
        self.assertTrue(all(prompt[1]["content"] == "fixed" for prompt in backend.calls[0][1]))

    def test_rejects_backend_probability_count_mismatch(self) -> None:
        request = JevRequest(
            state="message",
            model="test-model",
            questions={"check": Noul()},
        )

        with self.assertRaises(ValueError):
            LLM2Jev(backend=FakeBackend([0.2, 0.8])).evaluate(request)

    def test_propagates_backend_errors(self) -> None:
        request = JevRequest(
            state="message",
            model="test-model",
            questions={"check": Noul()},
        )

        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            LLM2Jev(backend=FailingBackend()).evaluate(request)


class BinaryBackendOutputTests(unittest.TestCase):
    def test_copies_and_validates_probabilities(self) -> None:
        probabilities = [0.2, 0.8]

        output = BinaryBackendOutput(yes_probabilities=probabilities)
        probabilities[0] = 1.0

        self.assertEqual(output.yes_probabilities, (0.2, 0.8))
        self.assertEqual(output.usage, Usage())

    def test_rejects_invalid_usage(self) -> None:
        with self.assertRaises(ValueError):
            BinaryBackendOutput(
                yes_probabilities=[0.5],
                usage=object(),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
