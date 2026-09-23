"""Score image evidence with Transformers, SGLang or MLX, without decoding."""

import argparse

from llm2jev import Choice, JevRequest, LLM2Jev, MLXBackend, Noul, SGLangBackend, TransformersBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["transformers", "sglang", "mlx"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", required=True, help="Image URL, file:// URI, local path, or Base64 data URL")
    parser.add_argument("--placement", choices=["state", "instructions"], default="state")
    parser.add_argument("--device", default=None, help="Transformers device, e.g. cpu or cuda")
    parser.add_argument("--submission", choices=["all", "staged"], default="staged")
    parser.add_argument("--batch-size", type=int, default=8, help="Maximum MLX batch size")
    parser.add_argument("--prefill-step-size", type=int, default=512, help="MLX prefill chunk size")
    parser.add_argument("--cache-size", type=int, default=32, help="Maximum MLX cache entries")
    parser.add_argument("--cache-bytes", type=int, default=512 * 1024 * 1024, help="MLX cache byte budget")
    args = parser.parse_args()

    def evidence(text: str) -> dict[str, object]:
        return {"type": "multimodal", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": args.image}},
        ]}

    color_question = "Which color dominates the image?"
    person_question = "Is a person visible in the image?"
    request = JevRequest(
        state=evidence("Inspect this image.") if args.placement == "state" else "Inspect the attached evidence.",
        model=args.model_path,
        questions={
            "color": Choice(
                instructions=color_question if args.placement == "state" else evidence(color_question),
                criteria={"red": "Red", "blue": "Blue", "green": "Green"},
            ),
            "person": Noul(
                instructions=person_question if args.placement == "state" else evidence(person_question),
            ),
        },
    )
    if args.backend == "transformers":
        with TransformersBackend(
            args.model_path, multimodal=True, device=args.device, dtype="bfloat16", batch_size=1,
        ) as backend:
            print(LLM2Jev(backend=backend).evaluate(request).json)
    elif args.backend == "mlx":
        with MLXBackend(
            args.model_path, multimodal=True, submission=args.submission,
            batch_size=args.batch_size, prefill_step_size=args.prefill_step_size,
            max_cache_entries=args.cache_size, max_cache_bytes=args.cache_bytes,
        ) as backend:
            print(LLM2Jev(backend=backend).evaluate(request).json)
    else:
        with SGLangBackend(args.model_path, submission=args.submission) as backend:
            print(LLM2Jev(backend=backend).evaluate(request).json)


if __name__ == "__main__":
    main()
