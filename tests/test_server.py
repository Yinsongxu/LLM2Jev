import io
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llm2jev.server import mlx_server, server, transformers_server
from llm2jev.inference.request_parser import parse_request
from llm2jev.server.sglang_server import _parse_request


class ServerDispatchTests(unittest.TestCase):
    def test_default_preserves_all_sglang_arguments(self) -> None:
        arguments = ["--model-path", "model", "--tp", "2", "--submission", "all"]
        with patch("llm2jev.server.sglang_server.main") as serve:
            server.main(arguments)
        serve.assert_called_once_with(arguments)

    def test_selects_backend_with_both_option_syntaxes(self) -> None:
        for backend in ("sglang", "mlx", "transformers"):
            for option in (["--backend", backend], [f"--backend={backend}"]):
                with self.subTest(backend=backend, option=option):
                    with patch(f"llm2jev.server.{backend}_server.main") as serve:
                        server.main(["--model-path", "model"] + option)
                    serve.assert_called_once_with(["--model-path", "model"])

    def test_rejects_unknown_backend(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            server.main(["--backend", "unknown"])

    def test_shared_parser_preserves_sglang_import(self) -> None:
        self.assertIs(_parse_request, parse_request)


class MLXServerArgumentTests(unittest.TestCase):
    def test_defaults_and_valid_boundaries(self) -> None:
        defaults = mlx_server._parse_args(["--model-path", "model"])
        self.assertEqual(defaults.batch_size, 8)
        self.assertEqual(defaults.prefill_step_size, 512)
        self.assertEqual(defaults.submission, "staged")
        self.assertEqual(defaults.cache_size, 32)
        self.assertEqual(defaults.cache_bytes, 512 * 1024 * 1024)
        self.assertEqual(defaults.host, "127.0.0.1")
        self.assertEqual(defaults.port, 30000)
        for port in (1, 65535):
            args = mlx_server._parse_args([
                "--model-path", "model", "--batch-size", "1",
                "--prefill-step-size", "1", "--cache-size", "0", "--cache-bytes", "0",
                "--port", str(port), "--submission", "all", "--multimodal",
            ])
            self.assertEqual(args.port, port)
            self.assertTrue(args.multimodal)
            self.assertEqual(args.cache_size, 0)
            self.assertEqual(args.cache_bytes, 0)

    def test_rejects_invalid_arguments(self) -> None:
        for option in (
            ["--batch-size", "0"], ["--batch-size", "2.5"],
            ["--prefill-step-size", "-1"], ["--cache-size", "-1"],
            ["--cache-bytes", "-1"],
            ["--port", "0"], ["--port", "65536"],
            ["--submission", "other"], ["--served-model-name", " "],
            ["--model-path", ""], ["--host", ""], ["--api-key", ""],
            ["--workers", "2"],
        ):
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    mlx_server._parse_args(["--model-path", "model"] + option)

    def test_main_passes_configuration_to_single_process_server(self) -> None:
        run = Mock()
        with patch.dict("sys.modules", {"uvicorn": SimpleNamespace(run=run)}):
            with patch.object(mlx_server, "create_app") as create:
                mlx_server.main([
                    "--model-path", "weights", "--served-model-name", "local",
                    "--host", "0.0.0.0", "--port", "8000", "--api-key", "secret",
                    "--multimodal", "--batch-size", "4", "--prefill-step-size", "64",
                    "--submission", "all", "--cache-size", "16", "--cache-bytes", "1000",
                ])
        create.assert_called_once_with(
            "weights", served_model_name="local", api_key="secret", multimodal=True,
            batch_size=4, prefill_step_size=64, submission="all", max_cache_entries=16,
            max_cache_bytes=1000,
        )
        run.assert_called_once_with(create.return_value, host="0.0.0.0", port=8000)


class TransformersServerArgumentTests(unittest.TestCase):
    def test_defaults_and_options(self) -> None:
        defaults = transformers_server._parse_args(["--model-path", "model"])
        self.assertEqual(defaults.batch_size, 8)
        self.assertEqual(defaults.dtype, "auto")
        self.assertEqual(defaults.submission, "staged")
        self.assertEqual(defaults.port, 30000)
        args = transformers_server._parse_args([
            "--model-path", "model", "--device", "cuda:1", "--dtype", "bfloat16",
            "--batch-size", "4", "--multimodal", "--submission", "all",
        ])
        self.assertEqual(args.device, "cuda:1")
        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.batch_size, 4)
        self.assertTrue(args.multimodal)
        self.assertEqual(args.submission, "all")

    def test_main_starts_server_with_transformers_options(self) -> None:
        run = Mock()
        with patch.dict("sys.modules", {"uvicorn": SimpleNamespace(run=run)}):
            with patch.object(transformers_server, "create_app") as create:
                transformers_server.main([
                    "--model-path", "weights", "--served-model-name", "local",
                    "--host", "0.0.0.0", "--port", "8000", "--api-key", "secret",
                    "--device", "cuda", "--dtype", "float16", "--batch-size", "4",
                    "--multimodal",
                ])
        create.assert_called_once_with(
            "weights", served_model_name="local", api_key="secret", device="cuda",
            dtype="float16", batch_size=4, multimodal=True,
        )
        run.assert_called_once_with(create.return_value, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    unittest.main()
