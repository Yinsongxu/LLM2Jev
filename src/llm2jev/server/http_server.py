"""Shared HTTP API and model worker for local inference backends."""

from __future__ import annotations

import asyncio
import argparse
import logging
import os
import secrets
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from ..backend.base import BinaryBackend
from ..core.request import JevRequest
from ..core.response import JevResponse
from ..inference.converter import LLM2Jev
from ..inference.request_parser import parse_request


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def port(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return number


def add_common_server_arguments(parser: argparse.ArgumentParser, *, model_help: str) -> None:
    """Add CLI options shared by the local HTTP backends."""
    parser.add_argument("--model-path", required=True, help=model_help)
    parser.add_argument("--served-model-name", help="Model ID accepted by the HTTP API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=port, default=30000)
    parser.add_argument("--api-key", help="Bearer token (default: LLM2JEV_API_KEY).")
    parser.add_argument("--multimodal", action="store_true")


def validate_server_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for field in ("model_path", "served_model_name", "host", "api_key", "device"):
        value = getattr(args, field, None)
        if value is not None and not value.strip():
            parser.error(f"--{field.replace('_', '-')} must not be empty")


def run_http_server(app: Any, args: argparse.Namespace, *, backend_name: str) -> None:
    try:
        import uvicorn
    except ImportError as error:
        raise ImportError(
            f"The {backend_name} HTTP server requires FastAPI and Uvicorn; "
            f"install llm2jev[{backend_name}]"
        ) from error
    uvicorn.run(app, host=args.host, port=args.port)


class ModelWorker:
    """Run model setup, inference and cleanup on one dedicated thread."""

    def __init__(self, factory: Callable[[], BinaryBackend]) -> None:
        self._factory = factory
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm2jev-model")
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
        return await asyncio.get_running_loop().run_in_executor(self._executor, self._evaluate, request)

    def _close(self) -> None:
        try:
            if self._converter is not None:
                close = getattr(self._converter.backend, "close", None)
                if close is not None:
                    close()
        finally:
            self._converter = None

    async def close(self) -> None:
        self._closing = True
        future = asyncio.get_running_loop().run_in_executor(self._executor, self._close)
        cancelled = False
        try:
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


def create_http_app(
    model_path: str | Path,
    *,
    backend_factory: Callable[[], BinaryBackend],
    backend_name: str,
    served_model_name: str | None = None,
    api_key: str | None = None,
) -> Any:
    """Create the shared Jev HTTP API; the backend loads during startup."""
    try:
        from fastapi import Body, Depends, FastAPI, Header, HTTPException
    except ImportError as error:
        raise ImportError(
            f"The {backend_name} HTTP server requires FastAPI and Uvicorn; "
            f"install llm2jev[{backend_name}]"
        ) from error
    logger = logging.getLogger(f"llm2jev.{backend_name}_server")

    model_name = str(model_path) if served_model_name is None else served_model_name
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("served_model_name must be a non-empty string")
    if api_key is None:
        api_key = os.environ.get("LLM2JEV_API_KEY") or None
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        raise ValueError("api_key must be a non-empty string")

    @asynccontextmanager
    async def lifespan(app: Any):
        worker = ModelWorker(backend_factory)
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
        return {"object": "list", "data": [{
            "id": model_name, "object": "model", "created": 0, "owned_by": "llm2jev",
        }]}

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
            logger.warning("%s could not evaluate a request", backend_name, exc_info=True)
            raise HTTPException(
                status_code=422, detail="The model could not evaluate this request",
            ) from error
        except Exception as error:
            logger.exception("%s inference failed", backend_name)
            raise HTTPException(status_code=500, detail="Inference failed") from error
        return response.to_dict()

    return app
