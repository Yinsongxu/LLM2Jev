"""HTTP serving for the Transformers binary scoring backend."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..backend.base import BinaryBackend
from .http_server import (
    add_common_server_arguments,
    create_http_app,
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
    """Create a Transformers app; the model loads during application startup."""
    if backend_factory is not None and backend_options:
        raise ValueError("backend_options cannot be combined with backend_factory")
    if backend_factory is None:
        def backend_factory() -> BinaryBackend:
            from ..backend.transformers import TransformersBackend

            return TransformersBackend(model_path, **backend_options)

    return create_http_app(
        model_path, backend_factory=backend_factory, backend_name="transformers",
        served_model_name=served_model_name, api_key=api_key,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="llm2jev-serve --backend transformers", allow_abbrev=False)
    add_common_server_arguments(parser, model_help="Local Hugging Face model directory.")
    parser.add_argument("--device", help="Inference device (default: CUDA when available, otherwise CPU).")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto",
        help="Model weight dtype (default: auto).",
    )
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument(
        "--submission", choices=("staged", "all"), default="staged",
        help="Accepted for CLI consistency; currently has no effect in Transformers.",
    )
    args = parser.parse_args(argv)
    validate_server_arguments(parser, args)
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    app = create_app(
        args.model_path, served_model_name=args.served_model_name, api_key=args.api_key,
        device=args.device, dtype=args.dtype, batch_size=args.batch_size,
        multimodal=args.multimodal,
    )
    run_http_server(app, args, backend_name="transformers")


if __name__ == "__main__":
    main()
