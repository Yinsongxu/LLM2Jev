import json
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

import httpx
from openai import APIResponseValidationError, APITimeoutError, UnprocessableEntityError

from llm2jev import (
    AsyncClient, ScoreOutput, Choice, ChoiceAnswer, Client,
    JevResponse, Noul, NoulAnswer, Score, ScoreAnswer, Usage,
)
from llm2jev.server import mlx_server, transformers_server


QUESTIONS = {
    "department": Choice(criteria={"shipping": "Delivery", "billing": "Payments"}),
    "quality": Score(criteria=["Poor", "Good"]),
    "delivery": Noul(instructions="Is this a delivery issue?"),
}
PAYLOAD = {
    "model": "local",
    "answers": {
        "department": {"type": "choice", "choice": "shipping", "confidence": 0.8,
                       "probabilities": {"shipping": 0.8, "billing": 0.2}},
        "quality": {"type": "score", "score": 0.75, "confidence": 0.75,
                    "legend": {"0": "Poor", "1": "Good"},
                    "probabilities": {"0": 0.25, "1": 0.75}},
        "delivery": {"type": "noul", "noul": 0.9},
    },
    "usage": {"input_tokens": 100, "output_tokens": 0},
}


def sdk_response(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"object": "list", "data": [
            {"id": "local", "object": "model", "created": 0, "owned_by": "local"},
        ]})
    if request.url.path == "/v1/chat/completions":
        if json.loads(request.content).get("stream"):
            chunk = {"id": "chat-1", "object": "chat.completion.chunk", "created": 0,
                     "model": "local", "choices": [
                         {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None},
                     ]}
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json={
            "id": "chat-1", "object": "chat.completion", "created": 0, "model": "local",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"},
                         "finish_reason": "stop"}],
        })
    if request.url.path == "/v1/systemone":
        return httpx.Response(200, json=PAYLOAD)
    raise AssertionError(f"Unexpected URL: {request.url}")


class ClientTests(unittest.TestCase):
    def test_typed_response_and_dataclass_or_json_questions(self) -> None:
        for trailing_slash in ("", "/"):
            requests = []

            def handle(request):
                requests.append(request)
                return sdk_response(request)

            with Client(base_url=f"http://test/v1{trailing_slash}", api_key="secret",
                           http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
                for questions in (QUESTIONS, {name: q.to_dict() for name, q in QUESTIONS.items()},
                                  {**QUESTIONS, "delivery": QUESTIONS["delivery"].to_dict()}):
                    result = client.systemone.create(model="local", state="Package delayed", questions=questions)
                    self.assertIsInstance(result, JevResponse)
                    self.assertIsInstance(result.choices["department"], ChoiceAnswer)
                    self.assertIsInstance(result.scores["quality"], ScoreAnswer)
                    self.assertIsInstance(result.nouls["delivery"], NoulAnswer)
                    self.assertEqual(result.scores["quality"].probabilities, {0: 0.25, 1: 0.75})
                    self.assertEqual(result.to_dict(), PAYLOAD)
                for request in requests:
                    self.assertEqual(request.url.path, "/v1/systemone")
                    self.assertEqual(request.headers["authorization"], "Bearer secret")
                    self.assertNotIn("include_choice_candidates", json.loads(request.content))
            self.assertTrue(client.is_closed())

    def test_rendering_switch_multimodal_and_request_options(self) -> None:
        requests = []

        def handle(request):
            requests.append(request)
            return sdk_response(request)

        state = {"type": "multimodal", "content": [
            {"type": "text", "text": "What is shown?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        ]}
        with Client(base_url="http://test/v1", api_key="unused", timeout=20,
                       default_headers={"x-default": "original"}, default_query={"default": "yes"},
                       http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
            client.systemone.create(model="local", state=state, questions=QUESTIONS,
                                    include_choice_candidates=False,
                                    extra_headers={"x-call": "value"}, extra_query={"call": "yes"}, timeout=3)
            client.systemone.create(model="local", state=state, questions=QUESTIONS)
            self.assertEqual(json.loads(requests[0].content)["state"], state)
            self.assertIs(json.loads(requests[0].content)["include_choice_candidates"], False)
            self.assertNotIn("include_choice_candidates", json.loads(requests[1].content))
            self.assertEqual(requests[0].headers["x-default"], "original")
            self.assertEqual(requests[0].headers["x-call"], "value")
            self.assertEqual(dict(requests[0].url.params), {"default": "yes", "call": "yes"})
            self.assertEqual(requests[0].extensions["timeout"]["read"], 3)
            self.assertEqual(requests[1].extensions["timeout"]["read"], 20)

    def test_original_resources_streaming_and_with_options(self) -> None:
        with Client(base_url="http://test/v1", api_key="unused",
                       http_client=httpx.Client(transport=httpx.MockTransport(sdk_response))) as client:
            self.assertEqual(client.models.list().data[0].id, "local")
            messages = [{"role": "user", "content": "Hi"}]
            self.assertEqual(client.chat.completions.create(model="local", messages=messages)
                             .choices[0].message.content, "Hello")
            with client.chat.completions.create(model="local", messages=messages, stream=True) as stream:
                self.assertEqual([chunk.choices[0].delta.content for chunk in stream], ["Hello"])
            original = client.systemone
            copied = client.with_options(api_key="other", max_retries=0)
            self.assertIsInstance(copied, Client)
            self.assertIsNot(copied.systemone, original)
            self.assertIs(copied.systemone._client, copied)
            self.assertEqual(copied.systemone.create(model="local", state="State", questions=QUESTIONS)
                             .to_dict(), PAYLOAD)

    def test_invalid_request_is_rejected_before_http(self) -> None:
        handle = Mock(side_effect=sdk_response)
        with Client(base_url="http://test/v1", api_key="unused",
                       http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
            for override in ({"model": ""}, {"state": None}, {"questions": {}},
                             {"questions": []}, {"questions": {"q": {"type": "unknown"}}},
                             {"include_choice_candidates": "false"}, {"include_choice_candidates": 0},
                             {"include_choice_candidates": None}):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    client.systemone.create(**{"model": "local", "state": "State",
                                               "questions": QUESTIONS, **override})
        handle.assert_not_called()

    def test_malformed_success_response_raises_sdk_validation_error(self) -> None:
        malformed = [None, [], {}, {**PAYLOAD, "usage": []}, {**PAYLOAD, "answers": {}},
                     {**PAYLOAD, "answers": {"q": {"type": "unknown"}}},
                     {**PAYLOAD, "answers": {"q": {"type": "noul", "noul": 2}}}]
        for field in ("legend", "probabilities"):
            value = deepcopy(PAYLOAD)
            value["answers"]["quality"][field]["00"] = value["answers"]["quality"][field]["0"]
            malformed.append(value)
        for value in malformed + ["not-json"]:
            with self.subTest(value=value):
                def handle(request):
                    return httpx.Response(200, content="not-json") if value == "not-json" else httpx.Response(
                        200, content=json.dumps(value), headers={"content-type": "application/json"})

                with Client(base_url="http://test/v1", api_key="unused",
                               http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
                    with self.assertRaises(APIResponseValidationError) as error:
                        client.systemone.create(model="local", state="State", questions=QUESTIONS)
                    self.assertEqual(error.exception.status_code, 200)

    def test_sdk_http_errors_and_retries(self) -> None:
        handle = Mock(side_effect=[httpx.Response(500, json={"detail": "busy"}),
                                   httpx.Response(200, json=PAYLOAD),
                                   httpx.Response(422, json={"detail": "bad model"})])
        with Client(base_url="http://test/v1", api_key="unused", max_retries=1,
                       http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
            with patch("time.sleep"):
                self.assertEqual(client.systemone.create(model="local", state="State", questions=QUESTIONS)
                                 .to_dict(), PAYLOAD)
            with self.assertRaises(UnprocessableEntityError) as error:
                client.systemone.create(model="local", state="State", questions=QUESTIONS)
            self.assertEqual(error.exception.body, {"detail": "bad model"})
        self.assertEqual(handle.call_count, 3)

    def test_sdk_timeout_error(self) -> None:
        handle = Mock(side_effect=httpx.ReadTimeout("timed out"))
        with Client(base_url="http://test/v1", api_key="unused", max_retries=0,
                       http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
            with self.assertRaises(APITimeoutError):
                client.systemone.create(model="local", state="State", questions=QUESTIONS)
        self.assertEqual(handle.call_count, 1)


class AsyncClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_creation_original_resources_and_options(self) -> None:
        requests = []

        def handle(request):
            requests.append(request)
            return sdk_response(request)

        async with AsyncClient(base_url="http://test/v1", api_key="unused",
                                  http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            self.assertEqual((await client.models.list()).data[0].id, "local")
            stream = await client.chat.completions.create(
                model="local", messages=[{"role": "user", "content": "Hi"}], stream=True)
            async with stream:
                self.assertEqual([chunk.choices[0].delta.content async for chunk in stream], ["Hello"])
            copied = client.with_options(api_key="other", max_retries=0)
            self.assertIsInstance(copied, AsyncClient)
            result = await copied.systemone.create(
                model="local", state=["State"], questions={name: q.to_dict() for name, q in QUESTIONS.items()},
                include_choice_candidates=False, extra_headers={"x-call": "value"},
                extra_query={"call": "yes"}, timeout=4)
            self.assertEqual(result.to_dict(), PAYLOAD)
            request = requests[-1]
            self.assertEqual(request.headers["authorization"], "Bearer other")
            self.assertEqual(request.headers["x-call"], "value")
            self.assertEqual(request.url.params["call"], "yes")
            self.assertEqual(request.extensions["timeout"]["read"], 4)
            self.assertIs(json.loads(request.content)["include_choice_candidates"], False)
        self.assertTrue(client.is_closed())

    async def test_async_validation_and_http_error(self) -> None:
        handle = Mock(side_effect=[httpx.Response(422, json={"detail": "bad model"}),
                                   httpx.Response(200, json={})])
        async with AsyncClient(base_url="http://test/v1", api_key="unused", max_retries=0,
                                  http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            with self.assertRaises(ValueError):
                await client.systemone.create(model="local", state="State", questions={})
            handle.assert_not_called()
            with self.assertRaises(UnprocessableEntityError):
                await client.systemone.create(model="local", state="State", questions=QUESTIONS)
            with self.assertRaises(APIResponseValidationError):
                await client.systemone.create(model="local", state="State", questions=QUESTIONS)

    async def test_client_against_each_http_server_with_fake_backend(self) -> None:
        for module in (mlx_server, transformers_server):
            with self.subTest(backend=module.__name__):
                backend = Mock()
                backend.score.return_value = ScoreOutput(
                    yes_probabilities=[0.8, 0.2], usage=Usage(input_tokens=100, output_tokens=0),
                )
                app = module.create_app("local", api_key="secret", backend_factory=lambda: backend)
                async with app.router.lifespan_context(app):
                    async with AsyncClient(
                        base_url="http://test/v1", api_key="secret",
                        http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app)),
                    ) as client:
                        self.assertEqual((await client.models.list()).data[0].id, "local")
                        for enabled in (False, True):
                            result = await client.systemone.create(
                                model="local", state="Package delayed",
                                questions={"department": QUESTIONS["department"]},
                                include_choice_candidates=enabled)
                            self.assertEqual(result.choices["department"].choice, "shipping")
                            self.assertEqual(result.usage.output_tokens, 0)
                            prompts = backend.score.call_args.kwargs["prompts"]
                            self.assertEqual("All candidates:" in prompts[0][1]["content"], enabled)
