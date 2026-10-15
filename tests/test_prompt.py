import unittest

from llm2jev import Box, Choice, DefaultPromptRenderer, JevRequest, Noul, Score, serialize_content


def request(question, *, state="message", include=True):
    return JevRequest(state=state, model="local", questions={"q": question}, include_choice_candidates=include)


class SerializeContentTests(unittest.TestCase):
    def test_keeps_strings_readable(self):
        self.assertEqual(serialize_content("客户要求退款"), "客户要求退款")

    def test_serializes_objects_deterministically(self):
        self.assertEqual(serialize_content({"z": 1, "a": "客户", "nested": {"b": 2, "a": 1}}),
                         '{"a":"客户","nested":{"a":1,"b":2},"z":1}')


class DefaultPromptRendererTests(unittest.TestCase):
    def setUp(self):
        self.renderer = DefaultPromptRenderer()

    def test_renders_choice_and_tracks_ids(self):
        rendered = self.renderer.render(request(Choice(
            instructions="Which team?", criteria={"billing": "Charges", "technical": "Bugs"})))
        self.assertEqual(rendered.score_question_ids, ("q", "q"))
        self.assertIn('Is this candidate "billing: Charges" the best answer?',
                      rendered.score_prompts[0][1]["content"])
        self.assertIn("All candidates:\nbilling: Charges\ntechnical: Bugs", rendered.score_prompts[0][1]["content"])

    def test_renders_score_and_noul(self):
        rendered = self.renderer.render(JevRequest(
            state="message", model="local", questions={
                "score": Score(instructions="Rate", criteria=["Low", "High"]),
                "noul": Noul(instructions="Refund?", criteria={"true": "Refund", "false": "No refund"}),
            }))
        self.assertEqual(rendered.score_question_ids, ("score", "score", "noul", "noul"))
        self.assertIn('Is rating "Low" the most appropriate rating?', rendered.score_prompts[0][1]["content"])
        self.assertIn('Is this candidate "Refund" the best answer?', rendered.score_prompts[2][1]["content"])

    def test_renders_box_and_tracks_ids(self):
        value = {"type": "multimodal", "content": [
            {"type": "text", "text": "image"},
            {"type": "image_url", "image_url": {"url": "target.png"}},
        ]}
        rendered = self.renderer.render(request(Box(instructions="red target"), state=value))
        self.assertEqual(rendered.box_question_ids, ("q",))
        self.assertIn("0-1000", rendered.box_prompts[0][1]["content"][-1]["text"])

    def test_hides_choice_candidates(self):
        rendered = self.renderer.render(request(
            Choice(instructions="Which?", criteria={"a": "A", "b": "B"}), include=False))
        content = rendered.score_prompts[0][1]["content"]
        self.assertNotIn("All candidates:", content)
        self.assertIn('Is this candidate "a: A"', content)

    def test_rendering_is_repeatable(self):
        req = request(Noul(instructions="Is this true?"))
        self.assertEqual(self.renderer.render(req), self.renderer.render(req))


if __name__ == "__main__":
    unittest.main()
