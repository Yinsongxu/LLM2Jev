import unittest
from collections.abc import Sequence
from dataclasses import replace

from llm2jev import (
    ScoreOutput,
    ChatPrompt,
    Choice,
    ChoiceAnswer,
    GenerationOutput,
    JevRequest,
    LLM2Jev,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    RenderedPrompts,
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
    ) -> ScoreOutput:
        prompt_values = tuple(prompts)
        self.calls.append((model, prompt_values))
        return ScoreOutput(
            yes_probabilities=self.probabilities,
            usage=self.usage,
        )

    def generate(
        self, *, model: str, prompts: Sequence[ChatPrompt], max_new_tokens: int,
    ) -> GenerationOutput:
        raise AssertionError("binary requests must not generate text")


class FailingBackend:
    def score(
        self,
        *,
        model: str,
        prompts: Sequence[ChatPrompt],
    ) -> ScoreOutput:
        raise RuntimeError("backend failed")

    def generate(
        self, *, model: str, prompts: Sequence[ChatPrompt], max_new_tokens: int,
    ) -> GenerationOutput:
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
            def render(self, request: JevRequest) -> RenderedPrompts:
                return RenderedPrompts(
                    score_prompts=(
                        ({"role": "system", "content": "system"},
                         {"role": "user", "content": "fixed"}),
                        ({"role": "system", "content": "system"},
                         {"role": "user", "content": "fixed"}),
                    ),
                    score_question_ids=("tone", "tone"),
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

    def test_choice_candidates_switch_is_per_request_and_preserves_other_types(self) -> None:
        for description in (None, "Charges and payments"):
            with self.subTest(description=description):
                request = JevRequest(
                    state="Duplicate charge", model="test-model",
                    questions={
                        "department": Choice(
                            instructions="Which team?",
                            criteria={"billing": description, "returns": "Exchanges"},
                        ),
                        "severity": Score(criteria=["Low", "High"]),
                        "refund": Noul(criteria={"true": "Refund", "false": "No refund"}),
                    },
                )
                backend = FakeBackend([0.8, 0.2, 0.25, 0.75, 0.7, 0.3])
                engine = LLM2Jev(backend=backend)
                expected = engine.evaluate(request)
                actual = engine.evaluate(replace(request, include_choice_candidates=False))
                engine.evaluate(request)
                default, hidden, repeated = [call[1] for call in backend.calls]

                candidate = "billing" if description is None else f"billing: {description}"
                self.assertEqual(
                    hidden[0][1]["content"],
                    "Context:\nDuplicate charge\n\nQuestion:\nWhich team?\n"
                    f'Is this candidate "{candidate}" the best answer?',
                )
                self.assertNotIn("billing", hidden[1][1]["content"])
                self.assertIn('Is this candidate "returns: Exchanges" the best answer?', hidden[1][1]["content"])
                self.assertEqual(hidden[2:], default[2:])
                self.assertEqual(default, repeated)
                self.assertEqual(actual, expected)

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


class ScoreOutputTests(unittest.TestCase):
    def test_copies_and_validates_probabilities(self) -> None:
        probabilities = [0.2, 0.8]

        output = ScoreOutput(yes_probabilities=probabilities)
        probabilities[0] = 1.0

        self.assertEqual(output.yes_probabilities, (0.2, 0.8))
        self.assertEqual(output.usage, Usage())

    def test_rejects_invalid_usage(self) -> None:
        with self.assertRaises(ValueError):
            ScoreOutput(
                yes_probabilities=[0.5],
                usage=object(),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
