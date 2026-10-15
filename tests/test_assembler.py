import unittest

from llm2jev import (
    Choice,
    ChoiceAnswer,
    JevRequest,
    Noul,
    NoulAnswer,
    RenderedPrompts,
    Score,
    ScoreAnswer,
    ScoreOutput,
    Usage,
    assemble_response,
)


class AssembleResponseTests(unittest.TestCase):
    def test_maps_reordered_prompts_to_questions(self) -> None:
        request = JevRequest(
            state="message", model="local",
            questions={
                "tone": Choice(criteria={"calm": None, "angry": None}),
                "check": Noul(),
            },
        )
        response = assemble_response(
            request=request,
            rendered=RenderedPrompts(
                score_prompts=((), (), ()),
                score_question_ids=("check", "tone", "tone"),
            ),
            scored=ScoreOutput(yes_probabilities=(0.7, 0.2, 0.8)),
        )
        self.assertEqual(response.nouls["check"].noul, 0.7)
        self.assertEqual(response.choices["tone"].probabilities, {"calm": 0.2, "angry": 0.8})

    def test_assembles_mixed_response(self) -> None:
        request = JevRequest(
            state="Customer requests a refund for a duplicate charge.",
            model="jev-latest",
            questions={
                "department": Choice(
                    criteria={"billing": None, "returns": None, "shipping": None},
                ),
                "severity": Score(criteria=["Low", "Medium", "High"]),
                "refund": Noul(),
                "explicit_refund": Noul(
                    criteria={"true": "Refund requested", "false": "No refund requested"},
                ),
            },
        )

        response = assemble_response(
            request=request,
            scored=ScoreOutput(
                yes_probabilities=[0.6, 0.2, 0.2, 0.0, 0.76, 0.24, 0.91, 0.8, 0.2],
                usage=Usage(input_tokens=120, output_tokens=0),
            ),
        )

        department = response.answers["department"]
        self.assertIsInstance(department, ChoiceAnswer)
        self.assertEqual(department.choice, "billing")  # type: ignore[union-attr]
        self.assertAlmostEqual(department.confidence, 0.4)  # type: ignore[union-attr]

        severity = response.answers["severity"]
        self.assertIsInstance(severity, ScoreAnswer)
        self.assertAlmostEqual(severity.score, 1.24)  # type: ignore[union-attr]
        self.assertAlmostEqual(severity.confidence, 0.64)  # type: ignore[union-attr]

        self.assertEqual(response.answers["refund"], NoulAnswer(noul=0.91))
        self.assertEqual(response.answers["explicit_refund"], NoulAnswer(noul=0.8))
        self.assertEqual(response.usage, Usage(input_tokens=120, output_tokens=0))

    def test_normalizes_independent_choice_probabilities(self) -> None:
        request = JevRequest(
            state="message",
            model="jev-latest",
            questions={"tone": Choice(criteria={"calm": None, "angry": None})},
        )

        response = assemble_response(
            request=request,
            scored=ScoreOutput(yes_probabilities=[0.9, 0.3]),
        )

        answer = response.answers["tone"]
        self.assertIsInstance(answer, ChoiceAnswer)
        self.assertEqual(answer.probabilities, {"calm": 0.75, "angry": 0.25})  # type: ignore[union-attr]
        self.assertEqual(response.usage, Usage())

    def test_all_zero_candidates_become_uniform(self) -> None:
        request = JevRequest(
            state="message",
            model="jev-latest",
            questions={"severity": Score(criteria=["Low", "Medium", "High"])},
        )

        response = assemble_response(
            request=request,
            scored=ScoreOutput(yes_probabilities=[0.0, 0.0, 0.0]),
        )

        answer = response.answers["severity"]
        self.assertIsInstance(answer, ScoreAnswer)
        self.assertAlmostEqual(answer.score, 1.0)  # type: ignore[union-attr]
        self.assertAlmostEqual(answer.confidence, 0.0)  # type: ignore[union-attr]

    def test_uses_custom_normalizer(self) -> None:
        request = JevRequest(
            state="message",
            model="jev-latest",
            questions={"tone": Choice(criteria={"calm": None, "angry": None})},
        )

        response = assemble_response(
            request=request,
            scored=ScoreOutput(yes_probabilities=[0.9, 0.1]),
            normalizer=lambda _: (0.6, 0.4),
        )

        answer = response.answers["tone"]
        self.assertIsInstance(answer, ChoiceAnswer)
        self.assertEqual(answer.probabilities, {"calm": 0.6, "angry": 0.4})  # type: ignore[union-attr]

    def test_rejects_missing_scoring_output(self) -> None:
        request = JevRequest(
            state="message", model="jev-latest", questions={"check": Noul()},
        )
        with self.assertRaisesRegex(ValueError, "each binary task"):
            assemble_response(request=request)

    def test_rejects_probability_count_mismatch(self) -> None:
        request = JevRequest(
            state="message",
            model="jev-latest",
            questions={"check": Noul()},
        )

        with self.assertRaises(ValueError):
            assemble_response(
                request=request,
                scored=ScoreOutput(yes_probabilities=[.2, .8]),
            )

    def test_rejects_invalid_normalizer_output(self) -> None:
        request = JevRequest(
            state="message",
            model="jev-latest",
            questions={"tone": Choice(criteria={"calm": None, "angry": None})},
        )

        with self.assertRaises(ValueError):
            assemble_response(
                request=request,
                scored=ScoreOutput(yes_probabilities=[0.5, 0.5]),
                normalizer=lambda _: (1.0,),
            )

    def test_rejects_single_choice_candidate(self) -> None:
        request = JevRequest(
            state="message",
            model="jev-latest",
            questions={"only": Choice(criteria={"one": None})},
        )

        with self.assertRaises(ValueError):
            assemble_response(
                request=request,
                scored=ScoreOutput(yes_probabilities=[0.8]),
            )


if __name__ == "__main__":
    unittest.main()
