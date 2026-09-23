"""Run a mixed LLM2Jev request with a local Transformers model."""

from __future__ import annotations

import argparse

from llm2jev import (
    Choice,
    JevRequest,
    LLM2Jev,
    Noul,
    Score,
    TransformersBackend,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an LLM2Jev inference example."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a local Transformers model",
    )
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

    with TransformersBackend(args.model_path) as backend:
        response = LLM2Jev(backend=backend).evaluate(request)
        print(response.json)


if __name__ == "__main__":
    main()
