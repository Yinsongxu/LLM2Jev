# Usage guide

[简体中文](usage_zh.md) · [Back to README](../README.md)

This guide covers local Python usage, the HTTP service, and candidate submission modes. Complete [installation](installation.md) first. SGLang and Transformers examples use a local Hugging Face-compatible causal language model; replace `/path/to/model` with its directory. MLX uses local MLX-LM text models or MLX-VLM image models.

## SGLang Python API

A `JevRequest` contains `state`, `model`, and `questions`. This example submits Choice, Score, and Noul questions together. The local SGLang backend defaults to staged submission:

```python
from llm2jev import Choice, JevRequest, LLM2Jev, Noul, Score, SGLangBackend

if __name__ == "__main__":
    model_path = "/path/to/model"
    request = JevRequest(
        model=model_path,
        state="My parcel arrived two weeks late, and my card was charged twice.",
        questions={
            "department": Choice(
                instructions="Which department should handle this request?",
                criteria={
                    "shipping": "Delivery problems",
                    "billing": "Charges and billing problems",
                    "returns": "Returns and exchanges",
                },
            ),
            "severity": Score(
                instructions="How severe is the problem?",
                criteria=["Low: minor impact", "Medium: impaired but usable", "High: unusable"],
            ),
            "delivery": Noul(
                instructions="Is this a delivery issue?",
            ),
        },
    )
    with SGLangBackend(model_path, submission="staged") as backend:
        response = LLM2Jev(backend=backend).evaluate(request)
        print(response.to_dict())
```

You can omit `submission="staged"`; specifying it makes the selected mode explicit. Keep the main guard because SGLang starts worker processes. For multiple requests, reuse the backend inside the same `with` block to avoid loading the model repeatedly.

The context manager shuts down the engine on exit. Pass SGLang engine options through `engine_kwargs`.

To submit all candidates together, use:

```python
with SGLangBackend(model_path, submission="all") as backend:
    response = LLM2Jev(backend=backend).evaluate(request)
```

The repository example supports both modes:

```bash
python examples/sglang_inference.py --model-path /path/to/model --submission staged
python examples/sglang_inference.py --model-path /path/to/model --submission all
```

## Transformers Backend

Run the example in the Transformers-only environment:

```bash
python examples/transformers_inference.py --model-path /path/to/model
```

Using `model_path` and `request` from the example above, replace the backend call with:

```python
from llm2jev import LLM2Jev, TransformersBackend

backend = TransformersBackend(model_path)
response = LLM2Jev(backend=backend).evaluate(request)
print(response.to_dict())
```

The Transformers backend uses CUDA when available and otherwise falls back to CPU.

## MLX Backend

On macOS with Apple Silicon, run the text example with the `mlx` extra:

```bash
uv run --extra mlx python examples/mlx_inference.py \
  --model-path /path/to/mlx-model --batch-size 8 --submission staged
```

Use the same request objects with `MLXBackend`:

```python
from llm2jev import LLM2Jev, MLXBackend

with MLXBackend("/path/to/mlx-model", batch_size=8, submission="staged") as backend:
    response = LLM2Jev(backend=backend).evaluate(request)
    print(response.to_dict())
```

MLX supports Choice, Score and Noul through the existing conversion pipeline.
Text candidates use actual variable-length batches and reusable prefix snapshots
when the model's cache supports them; unsupported cache types use sequential scoring.
`staged` explicitly prefills common prefixes, while `all` skips that warm-up and
can still reuse existing cache entries. Both preserve candidate order and logical
input-token accounting. The cache defaults to 32 entries and 512 MiB of tensors;
`max_cache_entries=0` or `max_cache_bytes=0` disables reusable storage.

For images, install `mlx-vlm` and construct `MLXBackend(..., multimodal=True)`.
Qwen2-VL and Qwen2.5-VL support image feature reuse, prefix caching and equal-length
batches; other causal VLMs use complete per-prompt prefills. Reuse a backend within
one `with` block for multiple requests. `clear_cache()` drops reusable caches;
`close()` or leaving the block releases the model. See the [MLX guide](mlx.md)
for full configuration and model support, and [Multimodal inputs](multimodal.md)
for image examples.

## System One HTTP API

`llm2jev-serve` defaults to SGLang and preserves its native server arguments.
`--backend mlx` selects the Apple Silicon service. Both serve the same
`POST /v1/systemone` request and response format, plus `/v1/models` and `/health`.
With SGLang, its other native endpoints remain available. Start SGLang with:

```bash
export LLM2JEV_API_KEY="replace-with-your-api-key"
llm2jev-serve \
  --model-path /path/to/model \
  --served-model-name local-model \
  --host 0.0.0.0 \
  --port 30000 \
  --api-key "$LLM2JEV_API_KEY"
```

Start MLX with the same model alias instead:

```bash
uv run --extra mlx --extra server llm2jev-serve \
  --backend mlx --model-path /path/to/mlx-model \
  --served-model-name local-model --host 127.0.0.1 --port 30000 \
  --batch-size 8 --submission staged --cache-size 32 --cache-bytes 536870912
```

MLX reads `LLM2JEV_API_KEY` unless `--api-key` overrides it. When configured,
`/v1` endpoints require a Bearer token; `/health` is a public readiness probe.
For MLX image models, use `--extra mlx-vlm --extra server` and add `--multimodal`.
The request's `model` must exactly match the served alias; without an alias MLX
uses the supplied model path. Unknown models return 404 and invalid requests 422.
MLX runs inference on one dedicated thread, queues concurrent requests and keeps
the event loop responsive. See [MLX HTTP serving](mlx.md#http-service) for details.

List models from either server:

```bash
curl http://localhost:30000/v1/models \
  -H "Authorization: Bearer $LLM2JEV_API_KEY"
```

Submit a System One request:

```bash
curl http://localhost:30000/v1/systemone \
  -H "Authorization: Bearer $LLM2JEV_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "The customer package has not arrived.",
    "model": "local-model",
    "questions": {
      "delivery": {
        "type": "noul",
        "instructions": "Is this a delivery issue?"
      }
    }
  }'
```

Use `--submission staged|all` to select candidate submission for `/v1/systemone`
(default: `staged`). The startup setting applies to all `/v1/systemone` requests
served by that process. This does not change the Jev request body or other native
SGLang endpoints. SGLang's `staged` requires Radix Cache; use `all` with
`--disable-radix-cache`. See [Choosing a mode](#choosing-a-mode) below.

With `--backend sglang` (the default), the command accepts SGLang's normal server arguments. It currently requires
the default single-tokenizer HTTP mode and does not support
`--skip-tokenizer-init`.

## Choosing a mode

The following table describes SGLang. MLX also offers `staged` and `all`, using
explicit prefix prefill; see [MLX batching and shared prefixes](mlx.md#batching-and-shared-prefixes)
for its semantics.

| Request pattern | Starting point | Reason |
| --- | --- | --- |
| Long context, many candidates, no relevant cached prefix | `staged` (default) | Avoids repeated processing within a cold request |
| Short input, few candidates | `all` | Extra submission rounds may cost more than they save |
| Repeated requests with mostly cached prefixes | `all` | Existing cache can be reused without establishing it in stages |
| Partial cache hits or highly varied inputs | Compare both | The benefit depends on shared computation and submission overhead |

The backend does not detect cache state and switch modes automatically. A single candidate or inputs without reusable prefixes can go in one batch even in staged mode. Having a shared prefix does not guarantee that additional rounds will be faster.

See [Performance benchmarks](shared-prefix-benchmarks.md) for measurements and test conditions.

See [Shared-prefix caching](shared-prefix-cache.md) for the reuse mechanism and output considerations.
