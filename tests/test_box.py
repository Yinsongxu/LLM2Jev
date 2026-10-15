import json
import math
import unittest
from unittest.mock import Mock

import httpx
from openai import APIResponseValidationError

from llm2jev import (
    AsyncClient, Backend, ScoreOutput, Box, BoxAnswer, Choice, Client, GenerationOutput,
    JevRequest, JevResponse, LLM2Jev, Noul, Score, Usage, assemble_response,
    DefaultPromptRenderer,
)
from llm2jev.inference.box import coordinate_confidence, parse_box
from llm2jev.inference.request_parser import parse_question, parse_request
from llm2jev.backend.tokenization import _generated_token_texts
from llm2jev.server.transformers_server import create_app


def image_content(source="target.png", text="Target image"):
    return {"type": "multimodal", "content": [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": source}},
    ]}


def box_request(**questions):
    return JevRequest(model="local", state=image_content(), questions=questions or {
        "target": Box(instructions="the red rectangle"),
    })


def fake_backend(texts=('{"bbox_2d": [101, 202, 803, 904]}',), probabilities=(0.73,)):
    backend = Mock(spec=Backend)
    backend.generate.return_value = GenerationOutput(
        token_texts=[list(text) for text in texts],
        token_probabilities=[[.73 if index == 0 else .12] * len(text)
                        for index, text in enumerate(texts)],
        usage=Usage(input_tokens=30, output_tokens=18),
    )
    backend.score.return_value = ScoreOutput(
        yes_probabilities=probabilities, usage=Usage(input_tokens=40, output_tokens=0),
    )
    return backend


class BoxProtocolTests(unittest.TestCase):
    def test_round_trip_image_in_state_or_instructions(self):
        for state, instructions in ((image_content(), "red rectangle"),
                                    ("Find the target", image_content())):
            with self.subTest(state=state):
                request = JevRequest(state=state, model="local", questions={
                    "target": Box(instructions=instructions),
                })
                self.assertEqual(parse_request(request.to_dict()), request)
                self.assertNotIn("criteria", request.questions["target"].to_dict())
                self.assertEqual(DefaultPromptRenderer().render(request).score_prompts, ())

    def test_rejects_missing_blank_and_invalid_instructions_and_criteria(self):
        for value in (None, "", "  ", True, 2, {"type": "multimodal", "content": []}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Box(instructions=value)
        for value in ({"type": "box"}, {"type": "box", "instructions": "x", "criteria": None}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_question(value)

    def test_requires_actual_image_for_each_box_question(self):
        for state in ("image.png", {"nested": image_content()}, image_content()["content"],
                      {"type": "multimodal", "content": [{"type": "text", "text": "photo"}]}):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "require an image"):
                JevRequest(model="local", state=state, questions={"target": Box(instructions="x")})
        with self.assertRaisesRegex(ValueError, "require an image"):
            JevRequest(model="local", state="text", questions={
                "one": Box(instructions=image_content()), "two": Box(instructions="x"),
            })

    def test_answer_serialization_and_filter_preserve_coordinate_precision(self):
        answer = BoxAnswer(box={"left": .101, "top": .202, "right": .102, "bottom": .203},
                           confidence=.731)
        response = JevResponse(model="local", answers={"tiny": answer}, usage=Usage())
        self.assertEqual(response.boxes, {"tiny": answer})
        self.assertEqual(response.choices, {})
        self.assertEqual(response.to_dict()["answers"]["tiny"], {
            "type": "box", "box": dict(answer.box), "confidence": .731,
        })
        self.assertEqual(json.loads(response.json)["answers"]["tiny"]["box"], dict(answer.box))
        BoxAnswer(box={"left": 0, "top": 0, "right": 1, "bottom": 1}, confidence=1)

    def test_answer_rejects_invalid_edges_and_confidence(self):
        valid = {"left": 0, "top": 0, "right": 1, "bottom": 1}
        invalid = [{}, {**valid, "extra": .2}, {k: v for k, v in valid.items() if k != "top"}]
        invalid += [{**valid, "left": value} for value in (-.1, 1, 2, True, "0", float("nan"), float("inf"))]
        invalid += [{**valid, "top": 1}, {**valid, "right": 0}, {**valid, "bottom": 0}]
        for edges in invalid:
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                BoxAnswer(box=edges, confidence=.5)
        for confidence in (-1, 1.1, True, float("nan")):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                BoxAnswer(box=valid, confidence=confidence)

    def test_parses_qwen_json_and_normalizes_without_reordering(self):
        for text in ('[101,202,803,904]', '{"bbox_2d":[101,202,803,904]}',
                     '```json\n[{"bbox_2d":[101,202,803,904],"label":"mug"}]\n```',
                     '```\n[101,202,803,904]\n```'):
            with self.subTest(text=text):
                self.assertEqual(parse_box(text), {"left": .101, "top": .202, "right": .803, "bottom": .904})
        self.assertEqual(parse_box('[0,0,1000,1000]'), {"left": 0, "top": 0, "right": 1, "bottom": 1})

    def test_rejects_malformed_multiple_or_invalid_generated_boxes(self):
        for text in ('none', 'null', '{}', '[]', '[1,2,3]', '[1,2,3,4,5]',
                     '[true,0,10,10]', '[-1,0,10,10]', '[0,0,1001,10]',
                     '[0.1,0,10,10]', '[0,0,NaN,10]', '[0,0,Infinity,10]',
                     '[10,0,10,10]', '[20,0,10,10]', '[0,20,10,10]',
                     '[[0,0,10,10],[1,1,20,20]]',
                     '[{"bbox_2d":[0,0,10,10]},{"bbox_2d":[1,1,20,20]}]',
                     'Here is the box: [0,0,10,10]'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_box(text)

    def test_generation_preserves_images_and_question_scope(self):
        request = box_request(
            first=Box(instructions=image_content("reference.png", "Match this target")),
            second=Box(instructions="the blue rectangle"),
        )
        rendered = DefaultPromptRenderer().render(request)
        ids, prompts = rendered.box_question_ids, rendered.box_prompts
        self.assertEqual(ids, ("first", "second"))
        sources = [[part["image_url"]["url"] for part in p[1]["content"]
                    if part["type"] == "image_url"] for p in prompts]
        self.assertEqual(sources, [["target.png", "reference.png"], ["target.png"]])
        self.assertIn("0-1000", prompts[0][1]["content"][-1]["text"])
        self.assertNotIn("reference.png", str(prompts[1]))


class BoxEvaluationTests(unittest.TestCase):
    def test_confidence_uses_geometric_mean_of_coordinate_tokens_only(self):
        pieces = ['```json\n[{"label":"target 42","bbox_2d":[', '100', ',', '200',
                  ',', '800', ',', '900', ']}]\n```']
        probabilities = [0., .25, 0., 1., 0., .25, 0., 1., 0.]
        self.assertAlmostEqual(coordinate_confidence("".join(pieces), pieces, probabilities), .5)
        pieces = ['[1', '', '00,200,800,900]']
        self.assertAlmostEqual(coordinate_confidence("".join(pieces), pieces,
                                                    [.5, 0., .5]), .5)
        # A coordinate and its comma can share a token; include that token once.
        pieces = ['[', '100,200', ',', '800', ',', '900', ']']
        probabilities = [0., .125, 0., 1., 0., 1., 0.]
        self.assertAlmostEqual(coordinate_confidence("".join(pieces), pieces, probabilities), .5)

    def test_confidence_handles_certain_and_zero_probability_tokens(self):
        text = '[0,0,10,10]'
        self.assertEqual(coordinate_confidence(text, [text], [1.]), 1.)
        self.assertEqual(coordinate_confidence(text, [text], [0.]), 0.)
        pieces = ['[0,', '0,10,10]']
        self.assertEqual(coordinate_confidence(text, pieces, [.8, 0.]), 0.)
        with self.assertRaisesRegex(ValueError, "missing"):
            coordinate_confidence(text, [], [])

    def test_confidence_preserves_tiny_coordinate_probabilities(self):
        pieces = ['[100,', '200,', '800,', '900]']
        confidence = coordinate_confidence("".join(pieces), pieces, [1e-250] * 4)
        self.assertTrue(math.isclose(confidence, 1e-250, rel_tol=1e-12))

    def test_token_alignment_handles_byte_fallback_and_eos(self):
        tokenizer = Mock()
        outputs = {1: '\ufffd', 2: '红', 3: '红100', 4: '红100'}
        tokenizer.decode.side_effect = lambda ids, **kwargs: outputs[len(ids)]
        self.assertEqual(_generated_token_texts(tokenizer, [1, 2, 3, 4]), ('', '红', '100', ''))

    def test_generation_output_rejects_missing_misaligned_or_invalid_probabilities(self):
        for pieces, probabilities in (([], [[0.]]), ([['x']], []),
                                  ([['[0,0,10,10]']], [[]])):
            with self.subTest(pieces=pieces), self.assertRaises(ValueError):
                GenerationOutput(token_texts=pieces, token_probabilities=probabilities)
        for value in (-.1, 1.1, -math.inf, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GenerationOutput(token_texts=[['[0,0,10,10]']],
                                 token_probabilities=[[value]])
        for value in (0., .1, 1.):
            with self.subTest(value=value):
                output = GenerationOutput(token_texts=[['[0,0,10,10]']],
                                          token_probabilities=[[value]])
                self.assertEqual(output.token_probabilities, ((value,),))

    def test_box_only_uses_sampled_probabilities_without_scoring(self):
        backend = fake_backend()
        result = LLM2Jev(backend=backend).evaluate(box_request())
        self.assertAlmostEqual(result.boxes["target"].confidence, .73)
        self.assertEqual(result.usage, Usage(input_tokens=30, output_tokens=18))
        backend.generate.assert_called_once()
        backend.score.assert_not_called()

    def test_mixed_questions_keep_order_and_probability_behavior(self):
        request = box_request(
            present=Noul(instructions="Is the target present?"),
            target=Box(instructions="the red rectangle"),
            color=Choice(criteria={"red": "Red", "blue": "Blue"}),
            quality=Score(criteria=["Low", "High"]),
            second=Box(instructions="the blue rectangle"),
        )
        backend = fake_backend(
            texts=('[100,200,800,900]', '[0,0,1000,1000]'),
            probabilities=(.9, .8, .2, .25, .75),
        )
        result = LLM2Jev(backend=backend).evaluate(request)
        self.assertEqual(tuple(result.answers), tuple(request.questions))
        self.assertEqual(result.nouls["present"].noul, .9)
        self.assertEqual(result.choices["color"].choice, "red")
        self.assertEqual(result.scores["quality"].score, .75)
        for answer, expected in zip(result.boxes.values(), [.73, .12]):
            self.assertAlmostEqual(answer.confidence, expected)
        self.assertEqual(len(backend.score.call_args.kwargs["prompts"]), 5)

    def test_generation_failure_propagates_before_scoring(self):
        backend = fake_backend()
        backend.generate.side_effect = ValueError("generation failed")
        with self.assertRaisesRegex(ValueError, "generation failed"):
            LLM2Jev(backend=backend).evaluate(box_request(present=Noul(), target=Box(instructions="x")))
        backend.score.assert_not_called()

    def test_invalid_or_missing_generation_fails_response_assembly(self):
        for texts in ((), ('',), ('  ',), ('invalid',), ('[0,0,10,10]', '[0,0,20,20]')):
            backend = fake_backend(texts=texts)
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                LLM2Jev(backend=backend).evaluate(box_request(present=Noul(), target=Box(instructions="x")))
            backend.score.assert_called_once()

    def test_mixed_request_rejects_missing_or_extra_binary_scores(self):
        for scored in (None, ScoreOutput(yes_probabilities=[.4, .5])):
            backend = fake_backend()
            backend.score.return_value = scored
            with self.subTest(scored=scored), self.assertRaisesRegex(ValueError, "each binary task"):
                LLM2Jev(backend=backend).evaluate(box_request(present=Noul(), target=Box(instructions="x")))

    def test_assembler_requires_generation_for_box_questions(self):
        with self.assertRaisesRegex(ValueError, "generated output"):
            assemble_response(request=box_request())

    def test_assembler_requires_exactly_the_requested_generated_boxes(self):
        for texts in ((), ('[0,0,10,10]', '[0,0,20,20]')):
            generated = fake_backend(texts=texts).generate.return_value
            with self.subTest(texts=texts), self.assertRaisesRegex(ValueError, "each box question"):
                assemble_response(request=box_request(), generated=generated)
        generated = fake_backend().generate.return_value
        result = assemble_response(request=box_request(), generated=generated)
        self.assertEqual(result.boxes["target"].box, parse_box(generated.texts[0]))
        self.assertAlmostEqual(result.boxes["target"].confidence, .73)
        self.assertEqual(result.usage, generated.usage)
        binary_request = JevRequest(model="local", state="text", questions={"present": Noul()})
        with self.assertRaisesRegex(ValueError, "each box question"):
            assemble_response(request=binary_request, generated=generated,
                              scored=ScoreOutput(yes_probabilities=[.8]))

    def test_assembler_merges_usage_for_mixed_questions(self):
        request = box_request(present=Noul(), target=Box(instructions="red rectangle"))
        generated = fake_backend().generate.return_value
        scored = ScoreOutput(yes_probabilities=[.8], usage=Usage(input_tokens=40, output_tokens=0))
        result = assemble_response(request=request, scored=scored, generated=generated)
        self.assertEqual(result.usage, Usage(input_tokens=70, output_tokens=18))
        self.assertEqual(tuple(result.answers), tuple(request.questions))
        self.assertEqual(result.nouls["present"].noul, .8)

    def test_assembler_preserves_unknown_usage_per_field(self):
        request = box_request(present=Noul(), target=Box(instructions="red rectangle"))
        generated = fake_backend().generate.return_value
        for usage, expected in (
            (Usage(), Usage()),
            (Usage(input_tokens=40), Usage(input_tokens=70)),
            (Usage(output_tokens=0), Usage(output_tokens=18)),
        ):
            scored = ScoreOutput(yes_probabilities=[.8], usage=usage)
            with self.subTest(usage=usage):
                result = assemble_response(request=request, scored=scored, generated=generated)
                self.assertEqual(result.usage, expected)

    def test_unknown_usage_remains_unknown(self):
        backend = fake_backend()
        backend.generate.return_value = GenerationOutput(token_texts=[['[0,0,10,10]']], token_probabilities=[[1.]])
        self.assertEqual(LLM2Jev(backend=backend).evaluate(box_request()).usage, Usage())

    def test_generation_output_derives_text_and_copies_nested_sequences(self):
        pieces, probabilities = [['[0,0,', '10,10]']], [[.2, .3]]
        output = GenerationOutput(token_texts=pieces, token_probabilities=probabilities)
        pieces[0][0] = 'changed'
        pieces.append(['extra'])
        probabilities[0][0] = .9
        probabilities.append([0.])
        self.assertEqual(output.texts, ('[0,0,10,10]',))
        self.assertEqual(output.token_texts, (('[0,0,', '10,10]'),))
        self.assertEqual(output.token_probabilities, ((.2, .3),))

    def test_existing_answer_named_box_keeps_its_display_precision(self):
        request = JevRequest(model="local", state="text", questions={"box": Noul()})
        backend = fake_backend(probabilities=(.731,))
        result = LLM2Jev(backend=backend).evaluate(request)
        backend.generate.assert_not_called()
        self.assertEqual(json.loads(result.json)["answers"]["box"]["noul"], .73)


class BoxClientTests(unittest.IsolatedAsyncioTestCase):
    def response(self, request):
        payload = json.loads(request.content)
        self.assertEqual(payload, box_request().to_dict())
        return httpx.Response(200, json=LLM2Jev(backend=fake_backend()).evaluate(box_request()).to_dict())

    async def test_sync_and_async_sdk_round_trip_box(self):
        with Client(base_url="http://test/v1", api_key="test",
                    http_client=httpx.Client(transport=httpx.MockTransport(self.response))) as client:
            result = client.systemone.create(**box_request().to_dict())
            self.assertIsInstance(result.boxes["target"], BoxAnswer)
            result = client.systemone.create(model="local", state=image_content(),
                                             questions=box_request().questions)
            self.assertAlmostEqual(result.boxes["target"].confidence, .73)
        async with AsyncClient(base_url="http://test/v1", api_key="test",
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.response))) as client:
            result = await client.systemone.create(**box_request().to_dict())
            self.assertIsInstance(result.boxes["target"], BoxAnswer)

    async def test_http_route_supports_box_and_rejects_missing_images(self):
        app = create_app("weights", served_model_name="local", backend_factory=lambda: fake_backend())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/v1/systemone", json=box_request().to_dict())
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["answers"]["target"]["type"], "box")
                response = await client.post("/v1/systemone", json={**box_request().to_dict(), "state": "text"})
                self.assertEqual(response.status_code, 422)

    async def test_sdk_rejects_invalid_box_response(self):
        payload = LLM2Jev(backend=fake_backend()).evaluate(box_request()).to_dict()
        payload["answers"]["target"]["box"]["left"] = 2
        with Client(base_url="http://test/v1", api_key="test", http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
        )) as client:
            with self.assertRaises(APIResponseValidationError):
                client.systemone.create(**box_request().to_dict())
