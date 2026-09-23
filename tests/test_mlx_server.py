import asyncio
import importlib.util
import threading
import unittest
from unittest.mock import patch

from llm2jev import BinaryBackendOutput, JevRequest, Noul, Usage
from llm2jev.server.mlx_server import _ModelWorker, create_app


_HAS_SERVER = all(importlib.util.find_spec(name) is not None for name in ("fastapi", "httpx"))


class FakeBackend:
    def __init__(self, on_score=None):
        self.thread_ids = [threading.get_ident()]
        self.on_score = on_score
        self.calls = []
        self.closed = False

    def score(self, *, model, prompts):
        self.thread_ids.append(threading.get_ident())
        self.calls.append((model, prompts))
        if self.on_score is not None:
            self.on_score()
        return BinaryBackendOutput(
            yes_probabilities=[0.75] * len(prompts),
            usage=Usage(input_tokens=7 * len(prompts), output_tokens=0),
        )

    def close(self):
        self.thread_ids.append(threading.get_ident())
        self.closed = True


class ModelWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_model_work_stays_on_one_background_thread(self) -> None:
        worker = _ModelWorker(FakeBackend)
        await worker.start()
        backend = worker._converter.backend
        request = JevRequest(state="text", model="local", questions={"check": Noul()})
        responses = await asyncio.gather(*(worker.evaluate(request) for _ in range(4)))
        self.assertEqual(len(responses), 4)
        await worker.close()
        self.assertTrue(backend.closed)
        self.assertEqual(len(set(backend.thread_ids)), 1)
        self.assertNotEqual(backend.thread_ids[0], threading.get_ident())
        self.assertFalse(worker.ready)

    async def test_cancellation_waits_for_active_call_before_cleanup(self) -> None:
        entered = asyncio.Event()
        released = threading.Event()
        loop = asyncio.get_running_loop()

        def block():
            loop.call_soon_threadsafe(entered.set)
            if not released.wait(timeout=5):
                raise RuntimeError("test failed to release inference")

        worker = _ModelWorker(lambda: FakeBackend(on_score=block))
        await worker.start()
        backend = worker._converter.backend
        request = JevRequest(state="text", model="local", questions={"check": Noul()})
        evaluation = asyncio.create_task(worker.evaluate(request))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            evaluation.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await evaluation
            closing = asyncio.create_task(worker.close())
            await asyncio.sleep(0)
            closing.cancel()
            await asyncio.sleep(0)
            self.assertFalse(backend.closed)
            self.assertFalse(closing.done())
        finally:
            released.set()
        with self.assertRaises(asyncio.CancelledError):
            await closing
        self.assertTrue(backend.closed)
        self.assertEqual(len(set(backend.thread_ids)), 1)

    async def test_cancelled_startup_disposes_model_after_construction_finishes(self) -> None:
        entered = asyncio.Event()
        released = threading.Event()
        created = []
        loop = asyncio.get_running_loop()

        def factory():
            loop.call_soon_threadsafe(entered.set)
            if not released.wait(timeout=5):
                raise RuntimeError("test failed to release initialization")
            backend = FakeBackend()
            created.append(backend)
            return backend

        worker = _ModelWorker(factory)
        startup = asyncio.create_task(worker.start())
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            startup.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await startup
            closing = asyncio.create_task(worker.close())
            await asyncio.sleep(0)
        finally:
            released.set()
        await closing
        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].closed)


@unittest.skipUnless(_HAS_SERVER, "requires the optional server extra and httpx")
class MLXHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        import httpx

        environment = patch.dict("os.environ", {"LLM2JEV_API_KEY": ""})
        environment.start()
        self.addCleanup(environment.stop)
        self.backends = []

        def factory():
            backend = FakeBackend()
            self.backends.append(backend)
            return backend

        self.app = create_app("weights", served_model_name="local", backend_factory=factory)
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        self.addAsyncCleanup(self.lifespan.__aexit__, None, None, None)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test",
        )
        self.addAsyncCleanup(self.client.aclose)
        self.payload = {
            "state": "Package delayed", "model": "local",
            "questions": {
                "department": {"type": "choice", "criteria": {"shipping": None, "billing": None}},
                "urgency": {"type": "score", "criteria": ["low", "high"]},
                "delivery": {"type": "noul"},
            },
        }

    async def test_wire_response_health_and_model_list(self) -> None:
        response = await self.client.post("/v1/systemone", json=self.payload)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(set(body["answers"]), {"department", "urgency", "delivery"})
        self.assertEqual(body["answers"]["department"]["choice"], "shipping")
        self.assertEqual(body["answers"]["urgency"]["score"], 0.5)
        self.assertEqual(body["usage"]["input_tokens"], 35)
        self.assertEqual(body["usage"]["output_tokens"], 0)
        health = await self.client.get("/health")
        self.assertEqual(health.json(), {"status": "ok"})
        models = await self.client.get("/v1/models")
        self.assertEqual(models.json()["object"], "list")
        self.assertEqual(models.json()["data"][0]["id"], "local")
        self.assertEqual(models.json()["data"][0]["object"], "model")

    async def test_rejects_bad_json_and_invalid_wire_without_model_call(self) -> None:
        for payload in ([], {}, {**self.payload, "questions": {}}, {**self.payload, "state": 3}):
            with self.subTest(payload=payload):
                response = await self.client.post("/v1/systemone", json=payload)
                self.assertEqual(response.status_code, 422, response.text)
        response = await self.client.post(
            "/v1/systemone", content="{broken", headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 422)
        response = await self.client.post("/v1/systemone", json={**self.payload, "model": "weights"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.backends[0].calls, [])

    async def test_multimodal_wire_is_forwarded_to_backend(self) -> None:
        state = {
            "type": "multimodal", "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}},
            ],
        }
        response = await self.client.post("/v1/systemone", json={**self.payload, "state": state})
        self.assertEqual(response.status_code, 200, response.text)
        prompts = self.backends[0].calls[0][1]
        self.assertTrue(any(
            isinstance(message["content"], (list, tuple))
            and any(part["type"] == "image_url" for part in message["content"])
            for prompt in prompts for message in prompt
        ))

    async def test_errors_do_not_expose_backend_details(self) -> None:
        for error, code in ((ValueError("private/model/path"), 422), (RuntimeError("secret"), 500)):
            with self.subTest(error=error), self.assertLogs("llm2jev.mlx_server"):
                with patch.object(self.backends[0], "score", side_effect=error):
                    response = await self.client.post("/v1/systemone", json=self.payload)
            self.assertEqual(response.status_code, code)
            self.assertNotIn(str(error), response.text)
            self.assertNotIn("Traceback", response.text)

    async def test_cancelled_client_cannot_overlap_inference_or_block_health(self) -> None:
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        active = 0
        maximum_active = 0

        def block():
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            loop.call_soon_threadsafe(started.set)
            try:
                if not release.wait(timeout=5):
                    raise RuntimeError("test failed to release inference")
            finally:
                active -= 1

        self.backends[0].on_score = block
        first = asyncio.create_task(self.client.post("/v1/systemone", json=self.payload))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            health = await asyncio.wait_for(self.client.get("/health"), timeout=1)
            self.assertEqual(health.status_code, 200)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(self.client.post("/v1/systemone", json=self.payload))
            await asyncio.sleep(0.02)
            self.assertEqual(len(self.backends[0].calls), 1)
        finally:
            release.set()
        self.assertEqual((await second).status_code, 200)
        self.assertEqual(maximum_active, 1)

    async def test_api_key_from_argument_and_environment(self) -> None:
        import httpx

        for explicit in (True, False):
            with self.subTest(explicit=explicit), patch.dict("os.environ", {"LLM2JEV_API_KEY": "env-key"}):
                app = create_app(
                    "local", api_key="argument-key" if explicit else None,
                    backend_factory=FakeBackend,
                )
                expected = "argument-key" if explicit else "env-key"
                async with app.router.lifespan_context(app):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app), base_url="http://test",
                    ) as client:
                        for path in ("/v1/models", "/v1/systemone"):
                            for token in (None, "Basic " + expected, "Bearer wrong"):
                                headers = {} if token is None else {"Authorization": token}
                                response = await client.request(
                                    "GET" if path == "/v1/models" else "POST", path,
                                    headers=headers, json=self.payload,
                                )
                                self.assertEqual(response.status_code, 401)
                                self.assertEqual(response.headers["www-authenticate"], "Bearer")
                            response = await client.request(
                                "GET" if path == "/v1/models" else "POST", path,
                                headers={"Authorization": "bearer " + expected}, json=self.payload,
                            )
                            self.assertEqual(response.status_code, 200, response.text)
                        self.assertEqual((await client.get("/health")).status_code, 200)

    async def test_uninitialized_app_reports_unavailable(self) -> None:
        import httpx

        app = create_app("local", backend_factory=FakeBackend)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            self.assertEqual((await client.get("/health")).status_code, 503)
            self.assertEqual((await client.post("/v1/systemone", json=self.payload)).status_code, 503)

    async def test_configuration_rejects_empty_names_keys_and_ambiguous_factory_options(self) -> None:
        for options in (
            {"served_model_name": ""}, {"served_model_name": 12},
            {"api_key": " "}, {"api_key": 12}, {"batch_size": 4},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                create_app("local", backend_factory=FakeBackend, **options)
        app = create_app("local", backend_factory=FakeBackend, served_model_name="alias", api_key="key")
        async with app.router.lifespan_context(app):
            self.assertTrue(app.state.llm2jev_worker.ready)

    async def test_startup_failure_closes_executor(self) -> None:
        def fail():
            raise RuntimeError("model failed to load")

        app = create_app("local", backend_factory=fail)
        with self.assertRaisesRegex(RuntimeError, "model failed to load"):
            async with app.router.lifespan_context(app):
                self.fail("startup should fail before serving requests")
        self.assertFalse(app.state.llm2jev_worker.ready)
        self.assertTrue(app.state.llm2jev_worker._executor._shutdown)


if __name__ == "__main__":
    unittest.main()
