"""HTTP serving with one dedicated thread for the MLX model and its caches."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from .backend.base import BinaryBackend
from .core.request import JevRequest
from .core.response import JevResponse
from .inference.converter import LLM2Jev
from .inference.request_parser import parse_request


_logger = logging.getLogger(__name__)


class _ModelWorker:
    """Keep initialization, evaluation and disposal on the same OS thread.

    Cancelling an HTTP request cannot interrupt an active model call. The
    executor keeps later requests and shutdown behind it until that call ends.
    """

    def __init__(self, factory: Callable[[], BinaryBackend]) -> None:
        self._factory = factory
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm2jev-mlx")
        self._converter: LLM2Jev | None = None
        self._closing = False

    @property
    def ready(self) -> bool:
        return self._converter is not None and not self._closing

    def _start(self) -> None:
        self._converter = LLM2Jev(backend=self._factory())

    async def start(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._executor, self._start)

    def _evaluate(self, request: JevRequest) -> JevResponse:
        if self._converter is None:
            raise RuntimeError("Model is not initialized")
        return self._converter.evaluate(request)

    async def evaluate(self, request: JevRequest) -> JevResponse:
        if not self.ready:
            raise RuntimeError("Model is not available")
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._evaluate, request,
        )

    def _close(self) -> None:
        try:
            if self._converter is not None:
                close = getattr(self._converter.backend, "close", None)
                if close is not None:
                    close()
        finally:
            # Release the last model reference on its owning thread as well.
            self._converter = None

    async def close(self) -> None:
        self._closing = True
        future = asyncio.get_running_loop().run_in_executor(self._executor, self._close)
        cancelled = False
        try:
            # Do not cancel queued cleanup or block the event loop if shutdown
            # itself is cancelled while an inference call is still running.
            while not future.done():
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    cancelled = True
            future.result()
        finally:
            self._executor.shutdown(wait=False)
        if cancelled:
            raise asyncio.CancelledError


def create_app(
    model_path: str | Path,
    *,
    served_model_name: str | None = None,
    api_key: str | None = None,
    backend_factory: Callable[[], BinaryBackend] | None = None,
    **backend_options: Any,
) -> Any:
    """Create the MLX app; the model loads only during application startup.

    ``backend_factory`` is called on the model thread and supports deterministic
    tests without MLX or model downloads. Otherwise options go to MLXBackend.
    ``api_key=None`` uses LLM2JEV_API_KEY when the environment variable is set.
    """
    try:
        from fastapi import Body, Depends, FastAPI, Header, HTTPException
    except ImportError as error:
        raise ImportError(
            "The MLX HTTP server requires the 'server' extra; "
            "install llm2jev[mlx,server]"
        ) from error

    model_name = str(model_path) if served_model_name is None else served_model_name
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("served_model_name must be a non-empty string")
    if api_key is None:
        api_key = os.environ.get("LLM2JEV_API_KEY") or None
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        raise ValueError("api_key must be a non-empty string")
    if backend_factory is not None and backend_options:
        raise ValueError("backend_options cannot be combined with backend_factory")

    if backend_factory is None:
        def backend_factory() -> BinaryBackend:
            from .backend.mlx import MLXBackend

            return MLXBackend(model_path, **backend_options)

    @asynccontextmanager
    async def lifespan(app: Any):
        worker = _ModelWorker(backend_factory)
        app.state.llm2jev_worker = worker
        try:
            await worker.start()
            yield
        finally:
            await worker.close()

    app = FastAPI(title="LLM2Jev", lifespan=lifespan)
    app.state.llm2jev_worker = None

    async def authorize(authorization: str | None = Header(default=None)) -> None:
        if api_key is None:
            return
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            token.encode("utf-8"), api_key.encode("utf-8"),
        ):
            raise HTTPException(
                status_code=401, detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/health")
    async def health() -> dict[str, str]:
        worker = app.state.llm2jev_worker
        if worker is None or not worker.ready:
            raise HTTPException(status_code=503, detail="Model is not available")
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [{
                "id": model_name, "object": "model", "created": 0,
                "owned_by": "llm2jev",
            }],
        }

    @app.post("/v1/systemone", dependencies=[Depends(authorize)])
    async def systemone(payload: Any = Body(...)) -> dict[str, object]:
        try:
            request = parse_request(payload)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if request.model != model_name:
            raise HTTPException(status_code=404, detail="The requested model does not exist")
        worker = app.state.llm2jev_worker
        if worker is None or not worker.ready:
            raise HTTPException(status_code=503, detail="Model is not available")
        try:
            response = await worker.evaluate(request)
        except (ValueError, OSError) as error:
            _logger.warning("MLX could not evaluate a request", exc_info=True)
            raise HTTPException(
                status_code=422, detail="The model could not evaluate this request",
            ) from error
        except Exception as error:
            _logger.exception("MLX inference failed")
            raise HTTPException(status_code=500, detail="Inference failed") from error
        return response.to_dict()

    return app


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def _port(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return number


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="llm2jev-serve --backend mlx", allow_abbrev=False)
    parser.add_argument("--model-path", required=True, help="Local MLX model directory.")
    parser.add_argument("--served-model-name", help="Model ID accepted by the HTTP API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=30000)
    parser.add_argument("--api-key", help="Bearer token (default: LLM2JEV_API_KEY).")
    parser.add_argument("--multimodal", action="store_true", help="Load with MLX-VLM.")
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--prefill-step-size", type=_positive_int, default=512)
    parser.add_argument("--submission", choices=("staged", "all"), default="staged")
    parser.add_argument(
        "--cache-size", type=_nonnegative_int, default=32,
        help="Maximum cached prompt prefixes; 0 disables caching (default: 32).",
    )
    parser.add_argument(
        "--cache-bytes", type=_nonnegative_int, default=512 * 1024 * 1024,
        help="Maximum prefix cache bytes; 0 disables caching (default: 536870912).",
    )
    args = parser.parse_args(argv)
    for field in ("model_path", "served_model_name", "host", "api_key"):
        value = getattr(args, field)
        if value is not None and not value.strip():
            parser.error(f"--{field.replace('_', '-')} must not be empty")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        import uvicorn
    except ImportError as error:
        raise ImportError(
            "The MLX HTTP server requires the 'server' extra; "
            "install llm2jev[mlx,server]"
        ) from error
    app = create_app(
        args.model_path, served_model_name=args.served_model_name, api_key=args.api_key,
        multimodal=args.multimodal, batch_size=args.batch_size,
        prefill_step_size=args.prefill_step_size, submission=args.submission,
        max_cache_entries=args.cache_size, max_cache_bytes=args.cache_bytes,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
