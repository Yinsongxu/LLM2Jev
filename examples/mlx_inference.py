"""Run a mixed LLM2Jev request with a local MLX-LM model on Apple Silicon."""

from __future__ import annotations

import argparse

from llm2jev import (
    Choice,
    JevRequest,
    LLM2Jev,
    MLXBackend,
    Noul,
    Score,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an LLM2Jev inference example with MLX."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a local MLX-LM-compatible text model",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=512,
        help="Maximum tokens per prefill chunk (default: 512)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--submission", choices=("staged", "all"), default="staged")
    parser.add_argument("--cache-size", type=int, default=32)
    parser.add_argument("--cache-bytes", type=int, default=512 * 1024 * 1024)
    args = parser.parse_args()

    request = JevRequest(
        state="客户说包裹一直没有送到，希望查询物流并尽快处理。",
        model=args.model_path,
        questions={
            "is_delivery_issue": Noul(
                instructions="这是否是一个物流配送问题？",
            ),
            "department": Choice(
                instructions="这个请求应该交给哪个部门处理？",
                criteria={
                    "billing": "付款、账单或退款问题",
                    "shipping": "物流、配送或包裹丢失问题",
                    "technical": "产品故障或技术支持问题",
                },
            ),
            "urgency": Score(
                instructions="评估这个客户请求的紧急程度。",
                criteria=["不紧急", "比较紧急", "非常紧急"],
            ),
        },
    )

    with MLXBackend(
        args.model_path,
        prefill_step_size=args.prefill_step_size,
        batch_size=args.batch_size,
        submission=args.submission,
        max_cache_entries=args.cache_size,
        max_cache_bytes=args.cache_bytes,
    ) as backend:
        response = LLM2Jev(backend=backend).evaluate(request)
        print(response.json)


if __name__ == "__main__":
    main()
