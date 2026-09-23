"""Dispatch the server CLI without importing either inference runtime."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llm2jev-serve", add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--backend", choices=("sglang", "mlx"), default="sglang",
        help="Inference backend (default: sglang).",
    )
    args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if args.backend == "mlx":
        from .mlx_server import main as serve
    else:
        if "--help" in remaining or "-h" in remaining:
            print(parser.format_help())
        from .sglang_server import main as serve
    serve(remaining)


if __name__ == "__main__":
    main()
