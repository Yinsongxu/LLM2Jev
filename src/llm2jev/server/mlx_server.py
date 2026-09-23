"""HTTP serving with one dedicated thread for the MLX model and its caches."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..backend.base import BinaryBackend
from .http_server import (
    ModelWorker as _ModelWorker,
    add_common_server_arguments,
    create_http_app,
    nonnegative_int,
    positive_int,
    run_http_server,
    validate_server_arguments,
)


def create_app(
    model_path: str | Path,
    *,
    served_model_name: str | None = None,
    api_key: str | None = None,
    backend_factory: Callable[[], BinaryBackend] | None = None,
    **backend_options: Any,
) -> Any:
    """Create an MLX app; the model loads during application startup."""
    if backend_factory is not None and backend_options:
        raise ValueError("backend_options cannot be combined with backend_factory")

    if backend_factory is None:
        def backend_factory() -> BinaryBackend:
            from ..backend.mlx import MLXBackend

            return MLXBackend(model_path, **backend_options)

    return create_http_app(
        model_path, backend_factory=backend_factory, backend_name="mlx",
        served_model_name=served_model_name, api_key=api_key,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="llm2jev-serve --backend mlx", allow_abbrev=False)
    add_common_server_arguments(parser, model_help="Local MLX model directory.")
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--prefill-step-size", type=positive_int, default=512)
    parser.add_argument("--submission", choices=("staged", "all"), default="staged")
    parser.add_argument(
        "--cache-size", type=nonnegative_int, default=32,
        help="Maximum cached prompt prefixes; 0 disables caching (default: 32).",
    )
    parser.add_argument(
        "--cache-bytes", type=nonnegative_int, default=512 * 1024 * 1024,
        help="Maximum prefix cache bytes; 0 disables caching (default: 536870912).",
    )
    args = parser.parse_args(argv)
    validate_server_arguments(parser, args)
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    app = create_app(
        args.model_path, served_model_name=args.served_model_name, api_key=args.api_key,
        multimodal=args.multimodal, batch_size=args.batch_size,
        prefill_step_size=args.prefill_step_size, submission=args.submission,
        max_cache_entries=args.cache_size, max_cache_bytes=args.cache_bytes,
    )
    run_http_server(app, args, backend_name="mlx")


if __name__ == "__main__":
    main()
