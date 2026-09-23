import io
import unittest
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from llm2jev import Choice, JevRequest, Noul, Score
from test_sglang_fakes import native_modules, score_result
from llm2jev.server.sglang_server import (
    _evaluate_request,
    _parse_request,
    _parse_submission_args,
    _validate_server_args,
    main,
    register_systemone_route,
)


class SystemOneRequestParsingTests(unittest.TestCase):
    def test_parses_all_question_types(self) -> None:
        request = _parse_request({
            "state": {"message": "Package delayed"},
            "model": "local-model",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which department?",
                    "criteria": {"shipping": "Delivery", "billing": None},
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent?",
                    "criteria": ["low", "high"],
                },
                "delivery": {
                    "type": "noul",
                    "instructions": "Is this about delivery?",
                },
            },
        })

        self.assertIsInstance(request.questions["department"], Choice)
        self.assertIsInstance(request.questions["urgency"], Score)
        self.assertIsInstance(request.questions["delivery"], Noul)
        self.assertEqual(request.model, "local-model")

    def test_rejects_missing_and_malformed_fields(self) -> None:
        invalid = (
            [],
            {"model": "model", "questions": {"check": {"type": "noul"}}},
            {"state": "text", "model": "model", "questions": []},
            {
                "state": "text",
                "model": "model",
                "questions": {"check": {"type": "unknown"}},
            },
            {
                "state": "text",
                "model": "model",
                "questions": {"check": {"type": "choice", "criteria": []}},
            },
        )
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises((TypeError, ValueError)):
                _parse_request(payload)


class SystemOneEvaluationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        modules = patch.dict("sys.modules", native_modules())
        modules.start()
        self.addCleanup(modules.stop)

    async def test_scores_and_assembles_response(self) -> None:
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda label, **kwargs: {
            "no": [9],
            "yes": [7],
        }[label]
        tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: (
            "template:" + messages[-1]["content"]
        )
        tokenizer.return_value = {"input_ids": [[11, 12, 13], [11, 12, 14]]}
        async def generate(request, raw_request):
            yield [score_result(.8, 3), score_result(.3, 3)]

        manager = SimpleNamespace(tokenizer=tokenizer, generate_request=Mock(side_effect=generate))
        request = JevRequest(
            state="Package delayed",
            model="local-model",
            questions={
                "department": Choice(criteria={"shipping": None, "billing": None}),
            },
        )

        response = await _evaluate_request(request, manager, submission="all")

        manager.generate_request.assert_called_once()
        submitted, raw_request = manager.generate_request.call_args.args
        self.assertEqual(submitted.input_ids, [[11, 12, 13], [11, 12, 14]])
        self.assertIsNone(submitted.image_data)
        self.assertEqual(submitted.token_ids_logprob, [9, 7])
        self.assertEqual(submitted.sampling_params, {"max_new_tokens": 0})
        self.assertEqual(submitted.logprob_start_len, -1)
        self.assertTrue(submitted.return_logprob)
        self.assertIsNone(raw_request)
        self.assertEqual(response.answers["department"].choice, "shipping")
        self.assertEqual(response.usage.input_tokens, 6)
        self.assertEqual(response.usage.output_tokens, 0)

    async def test_default_stages_restore_candidate_order_and_sum_usage(self) -> None:
        # Candidate 2 seeds a separate branch before candidate 1 is submitted.
        inputs = [[1, 2, 3], [1, 2, 4], [5, 6, 7], [5, 6, 8]]
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda label, **kwargs: {"no": [9], "yes": [7]}[label]
        completed = []

        async def generate(request, raw_request):
            batch = request.input_ids
            if batch == [inputs[1], inputs[3]]:
                self.assertEqual(completed, [0, 2])
            indices = [inputs.index(tokens) for tokens in batch]
            completed.extend(indices)
            yes = [0.1, 0.9, 0.8, 0.2]
            yield [score_result(yes[index], len(inputs[index])) for index in indices]

        tokenizer.return_value = {"input_ids": inputs}
        manager = SimpleNamespace(tokenizer=tokenizer, generate_request=Mock(side_effect=generate))
        request = JevRequest(
            state="text",
            model="local-model",
            questions={
                "department": Choice(criteria={"shipping": None, "billing": None}),
                "severity": Score(criteria=["low", "high"]),
            },
        )
        response = await _evaluate_request(request, manager)

        self.assertEqual([call.args[0].input_ids for call in manager.generate_request.call_args_list],
                         [[inputs[0], inputs[2]], [inputs[1], inputs[3]]])
        self.assertEqual(completed, [0, 2, 1, 3])
        self.assertEqual(response.answers["department"].choice, "billing")
        self.assertAlmostEqual(response.answers["severity"].score, 0.2)
        self.assertEqual(response.usage.input_tokens, 12)
        self.assertEqual(response.usage.output_tokens, 0)

    async def test_stage_failure_does_not_submit_remaining_candidates(self) -> None:
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda label, **kwargs: {"no": [9], "yes": [7]}[label]
        request = JevRequest(
            state="text", model="local-model",
            questions={"check": Choice(criteria={"a": None, "b": None})},
        )
        closed = []

        async def generate(request, raw_request):
            try:
                yield []
            finally:
                closed.append(True)

        tokenizer.return_value = {"input_ids": [[1, 2], [1, 3]]}
        manager = SimpleNamespace(tokenizer=tokenizer, generate_request=Mock(side_effect=generate))
        with self.assertRaisesRegex(ValueError, "wrong number"):
            await _evaluate_request(request, manager)
        manager.generate_request.assert_called_once()
        self.assertEqual(closed, [True])

    async def test_route_uses_configured_submission(self) -> None:
        app = SimpleNamespace(state=SimpleNamespace(), routes=[], add_api_route=Mock())
        manager = SimpleNamespace(served_model_name="local-model")
        http_server = SimpleNamespace(
            app=app, get_global_state=lambda: SimpleNamespace(tokenizer_manager=manager)
        )
        payload = {
            "state": "text", "model": "local-model",
            "questions": {"check": {"type": "noul"}},
        }
        modules = {
            "fastapi": SimpleNamespace(HTTPException=Exception),
            "sglang.srt.entrypoints.http_server": http_server,
        }
        with patch.dict("sys.modules", modules):
            for mode in ("all", "staged"):
                with self.subTest(mode=mode):
                    register_systemone_route(submission=mode)
                    route = app.add_api_route.call_args.args[1]
                    response = Mock()
                    with patch("llm2jev.server.sglang_server._evaluate_request", new_callable=AsyncMock, return_value=response) as evaluate:
                        self.assertEqual(await route(payload), response.to_dict())
                        evaluate.assert_awaited_once_with(
                            _parse_request(payload), manager, submission=mode
                        )

    async def test_requires_tokenizer(self) -> None:
        request = JevRequest(
            state="text",
            model="model",
            questions={"check": Noul()},
        )
        manager = SimpleNamespace(tokenizer=None)

        with self.assertRaisesRegex(RuntimeError, "tokenization"):
            await _evaluate_request(request, manager)


class ServerArgumentTests(unittest.TestCase):
    def test_accepts_standard_single_tokenizer_http_server(self) -> None:
        _validate_server_args(
            SimpleNamespace(
                tokenizer_worker_num=1,
                skip_tokenizer_init=False,
                grpc_mode=False,
                encoder_only=False,
                use_ray=False,
                disable_radix_cache=False,
            )
        )

    def test_rejects_unsupported_server_modes(self) -> None:
        defaults = {
            "tokenizer_worker_num": 1,
            "skip_tokenizer_init": False,
            "grpc_mode": False,
            "encoder_only": False,
            "use_ray": False,
            "disable_radix_cache": False,
        }
        overrides = (
            {"tokenizer_worker_num": 2},
            {"skip_tokenizer_init": True},
            {"grpc_mode": True},
            {"encoder_only": True},
            {"use_ray": True},
            {"disable_radix_cache": True},
        )
        for override in overrides:
            with self.subTest(override=override), self.assertRaises(ValueError):
                _validate_server_args(SimpleNamespace(**(defaults | override)))

    def test_all_allows_disabled_radix_cache(self) -> None:
        _validate_server_args(SimpleNamespace(
            tokenizer_worker_num=1, skip_tokenizer_init=False,
            grpc_mode=False, encoder_only=False, use_ray=False,
            disable_radix_cache=True,
        ), submission="all")

    def test_submission_parser_preserves_sglang_arguments(self) -> None:
        native = ["--model-path", "local-model", "--port", "30001", "--disable-radix-cache"]
        self.assertEqual(_parse_submission_args(native), ("staged", native))
        for mode in ("staged", "all"):
            for option in (["--submission", mode], [f"--submission={mode}"]):
                with self.subTest(option=option):
                    self.assertEqual(_parse_submission_args(native + option), (mode, native))

    def test_submission_parser_rejects_invalid_or_missing_value(self) -> None:
        for args in (["--submission", "auto"], ["--submission"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    _parse_submission_args(args)
                self.assertEqual(error.exception.code, 2)

    def test_help_describes_submission_and_is_forwarded(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(_parse_submission_args(["--help"]), ("staged", ["--help"]))
        self.assertIn("--submission {staged,all}", output.getvalue())
        self.assertIn("default: staged", " ".join(output.getvalue().split()))

    def test_main_passes_mode_to_route_and_native_args_to_sglang(self) -> None:
        server_args = SimpleNamespace(
            tokenizer_worker_num=1, skip_tokenizer_init=False,
            grpc_mode=False, encoder_only=False, use_ray=False,
            disable_radix_cache=False,
        )
        prepare = Mock(return_value=server_args)
        launch = Mock()
        cleanup = Mock()
        modules = {
            "sglang.srt.plugins": SimpleNamespace(load_plugins=Mock()),
            "sglang.srt.server_args": SimpleNamespace(prepare_server_args=prepare),
            "sglang.srt.utils": SimpleNamespace(kill_process_tree=cleanup),
            "sglang.srt.entrypoints.http_server": SimpleNamespace(launch_server=launch),
        }
        with patch.dict("sys.modules", modules):
            for option, expected in (([], "staged"), (["--submission", "all"], "all")):
                with self.subTest(mode=expected), patch("llm2jev.server.sglang_server.register_systemone_route") as register:
                    main(["--model-path", "local-model"] + option)
                    prepare.assert_called_with(["--model-path", "local-model"])
                    register.assert_called_once_with(submission=expected)
                    launch.assert_called_with(server_args)
                    cleanup.assert_called_with(unittest.mock.ANY, include_parent=False)


if __name__ == "__main__":
    unittest.main()
