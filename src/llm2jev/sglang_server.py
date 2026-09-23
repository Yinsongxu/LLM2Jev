from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from .backend.tokenization import _single_token_id
from .backend.sglang.scoring import prepare_score_batches, score_output
from .core.request import JevRequest
from .core.response import JevResponse, Usage
from .inference.assembler import assemble_response
from .inference.binary import compile_binary_questions
from .inference.prompt import DefaultPromptRenderer
from .inference.request_parser import (
    parse_question as _parse_question,
    parse_request as _parse_request,
)


async def _evaluate_request(
    request: JevRequest,
    tokenizer_manager: Any,
    *,
    submission: str = "staged",
) -> JevResponse:
    import asyncio

    from sglang.srt.managers.io_struct import GenerateReqInput

    tokenizer = tokenizer_manager.tokenizer
    if tokenizer is None:
        raise RuntimeError("System One requires SGLang tokenization to be enabled")

    tasks = compile_binary_questions(request)
    renderer = DefaultPromptRenderer()
    prompts = tuple(renderer.render(task) for task in tasks)
    no_token_id = _single_token_id(tokenizer, "no", "no_label")
    yes_token_id = _single_token_id(tokenizer, "yes", "yes_label")
    if no_token_id == yes_token_id:
        raise ValueError("yes_label and no_label must encode to different tokens")

    batches = await asyncio.to_thread(
        prepare_score_batches, tokenizer, getattr(tokenizer_manager, "processor", None),
        prompts, enable_thinking=False, submission=submission,
        no_token_id=no_token_id, yes_token_id=yes_token_id,
    )
    probabilities = [0.0] * len(prompts)
    input_tokens = 0
    for indices, arguments in batches:
        stream = tokenizer_manager.generate_request(GenerateReqInput(**arguments), None)
        try:
            result = await stream.__anext__()
        finally:
            await stream.aclose()
        output = score_output(
            result, expected_count=len(indices),
            no_token_id=no_token_id, yes_token_id=yes_token_id,
        )
        for index, probability in zip(indices, output.yes_probabilities):
            probabilities[index] = probability
        input_tokens += output.usage.input_tokens
    return assemble_response(
        request=request, tasks=tasks, yes_probabilities=probabilities,
        usage=Usage(input_tokens=input_tokens, output_tokens=0),
    )


def register_systemone_route(*, submission: str = "staged") -> Any:
    """Register the System One endpoint on SGLang's existing FastAPI app."""
    try:
        from fastapi import HTTPException
        from sglang.srt.entrypoints.http_server import app, get_global_state
    except ImportError as error:
        raise ImportError(
            "The System One server requires the 'sglang' extra; "
            "install llm2jev[sglang]"
        ) from error

    app.state.llm2jev_submission = submission
    if any(getattr(route, "path", None) == "/v1/systemone" for route in app.routes):
        return app

    async def systemone(payload: dict[str, object]) -> dict[str, object]:
        try:
            request = _parse_request(payload)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        tokenizer_manager = get_global_state().tokenizer_manager
        if request.model != tokenizer_manager.served_model_name:
            raise HTTPException(
                status_code=404,
                detail=f"The model {request.model!r} does not exist",
            )
        try:
            response = await _evaluate_request(
                request, tokenizer_manager, submission=app.state.llm2jev_submission
            )
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return response.to_dict()

    app.add_api_route("/v1/systemone", systemone, methods=["POST"])
    return app


def _validate_server_args(server_args: Any, *, submission: str = "staged") -> None:
    if submission == "staged" and server_args.disable_radix_cache:
        raise ValueError("staged submission requires Radix Cache; use --submission all")
    if server_args.tokenizer_worker_num != 1:
        raise ValueError("System One currently requires --tokenizer-worker-num 1")
    if server_args.skip_tokenizer_init:
        raise ValueError("System One does not support --skip-tokenizer-init")
    if server_args.grpc_mode or server_args.encoder_only or server_args.use_ray:
        raise ValueError("System One requires SGLang's standard HTTP server")


def _parse_submission_args(argv: list[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(prog="llm2jev-serve", add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--submission",
        choices=("staged", "all"),
        default="staged",
        help="Candidate submission for /v1/systemone (default: staged).",
    )
    args, remaining = parser.parse_known_args(argv)
    if "--help" in remaining or "-h" in remaining:
        print(parser.format_help())
    return args.submission, remaining


def main(argv: list[str] | None = None) -> None:
    submission, sglang_argv = _parse_submission_args(
        sys.argv[1:] if argv is None else argv
    )
    try:
        from sglang.srt.plugins import load_plugins
        from sglang.srt.server_args import prepare_server_args
        from sglang.srt.utils import kill_process_tree
    except ImportError as error:
        raise ImportError(
            "The System One server requires the 'sglang' extra; "
            "install llm2jev[sglang]"
        ) from error

    load_plugins()
    from sglang.srt.entrypoints.http_server import launch_server

    server_args = prepare_server_args(sglang_argv)
    _validate_server_args(server_args, submission=submission)
    register_systemone_route(submission=submission)
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
