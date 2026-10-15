import unittest

from llm2jev import Choice, DefaultPromptRenderer, JevRequest, Noul, Score


class RendererPlanningTests(unittest.TestCase):
    def test_expands_candidates_and_preserves_question_ids(self):
        request = JevRequest(
            state="message", model="local", questions={
                "check": Noul(),
                "tone": Choice(criteria={"calm": None, "angry": None}),
                "severity": Score(criteria=["Low", "High"]),
            })
        rendered = DefaultPromptRenderer().render(request)
        self.assertEqual(rendered.score_question_ids,
                         ("check", "tone", "tone", "severity", "severity"))
        self.assertEqual(len(rendered.score_prompts), 5)

    def test_renders_partial_noul_criteria(self):
        rendered = DefaultPromptRenderer().render(JevRequest(
            state="message", model="local",
            questions={"check": Noul(criteria={"true": "yes"})},
        ))
        self.assertEqual(len(rendered.score_prompts), 2)
        self.assertIn('Is this candidate "yes"', rendered.score_prompts[0][1]["content"])
        self.assertIn("Candidate answer: no", rendered.score_prompts[1][1]["content"])

    def test_rejects_empty_choice(self):
        with self.assertRaises(ValueError):
            DefaultPromptRenderer().render(JevRequest(
                state="message", model="local", questions={"empty": Choice(criteria={})}))


if __name__ == "__main__":
    unittest.main()
