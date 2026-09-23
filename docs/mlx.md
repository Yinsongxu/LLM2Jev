# MLX backend

[简体中文](mlx_zh.md) · [Usage guide](usage.md) · [Back to README](../README.md)

`MLXBackend` runs local text and vision-language models on Apple Silicon macOS.
It uses the same `JevRequest`, Choice, Score, Noul, and response format as the
other backends. Python and `POST /v1/systemone` both perform binary scoring with
prefill alone: no output tokens are generated.

## Installation and examples

Choose the dependencies for your model and interface:

```bash
# Text Python API
uv sync --extra mlx
# Text HTTP service
uv sync --extra mlx --extra server
# Images, Python API and HTTP service
uv sync --extra mlx-vlm --extra server
# pip equivalent for the last command:
# python -m pip install -e '.[mlx-vlm,server]'
```

Prepare an existing local model directory containing weights, configuration and
a tokenizer or processor. Text models must be compatible with MLX-LM; image
models must be compatible with MLX-VLM. Compatible quantized models are supported.
Download or convert the model before constructing the backend; a remote model
identifier is not a local directory.

Run the examples, retaining the selected extras on `uv run` so dependency
synchronization keeps the required packages installed:

```bash
uv run --extra mlx python examples/mlx_inference.py \
  --model-path /path/to/mlx-model --batch-size 8 --submission staged

uv run --extra mlx-vlm python examples/multimodal_inference.py \
  --backend mlx --model-path /path/to/mlx-vlm --image /path/to/photo.png
```

The image example accepts local paths, `file://` URIs, HTTP(S) URLs and Base64
data URLs. Use `--placement instructions` to attach the image to each question
instead of shared `state`. See [Multimodal inputs](multimodal.md) for the wire format.

## Python API and lifecycle

```python
from llm2jev import JevRequest, LLM2Jev, MLXBackend, Noul

model_path = "/path/to/mlx-model"
request = JevRequest(
    model=model_path,
    state="The customer's parcel has not arrived.",
    questions={"delivery": Noul(instructions="Is this a delivery issue?")},
)
with MLXBackend(model_path, batch_size=8, submission="staged") as backend:
    converter = LLM2Jev(backend=backend)
    print(converter.evaluate(request).json)
    # Repeated evaluations can reuse prefixes without loading the model again.
    print(converter.evaluate(request).json)
    backend.clear_cache()  # Keep the model loaded; release reusable caches.
```

The context manager calls `close()` to release model and cache references.
Explicit `close()` is also supported and is safe to call repeatedly. Calls to
`score()` after closing raise an error. The constructor selects the model;
`request.model` labels Python responses and does not switch the loaded model.

Configuration defaults:

```python
MLXBackend(
    model_path,                  # str or pathlib.Path
    yes_label="yes",
    no_label="no",
    enable_thinking=False,
    prefill_step_size=512,
    batch_size=8,
    submission="staged",         # "staged" or "all"
    max_cache_entries=32,
    max_cache_bytes=512 * 1024 * 1024,
    multimodal=False,            # True loads the model through MLX-VLM
)
```

Both labels must encode to one token and have different token IDs. Custom labels
require a matching prompt renderer. `enable_thinking=False` is passed to the chat
template; the backend never generates a thinking trace. `prefill_step_size` and
`batch_size` must be positive integers. Cache limits must be nonnegative integers;
setting either limit to zero disables reusable cache storage.

## Batching and shared prefixes

Text candidates are scored in actual model batches up to `batch_size`, preserving
input order in the result. Variable-length prompts advance together until the
shortest remaining prompt finishes; finished rows leave the batch. This avoids
padding input tokens. Both the model and its cache must support batched cache
state, including native merging and extraction. Unsupported caches, including
rotating caches with pinned initial tokens, use sequential scoring while preserving
the model's cache behavior. AFM7 and Gemma3n in MLX-LM 0.31.3 directly unpack
single-sequence cache state, so they automatically use an effective batch size of
one (`afm7`, `gemma3n` and `gemma3n_text` model types). This fallback retains staged
prefix sharing and reuse across requests.

`submission="staged"` prefills shared token prefixes before scoring candidate
suffixes. It can reuse context within the first request as well as across later
requests. `submission="all"` skips this explicit warm-up but still reuses existing
cached prefixes. It does not disable caching or override `batch_size`.

Prefix snapshots are isolated from each candidate's mutable cache and evicted by
least recent use. The cache is bounded by both entry count and tensor bytes; a
snapshot exceeding the byte budget is not stored. These limits bound retained
reusable caches, not model weights, active inference state or total process memory.
`prefill_step_size` bounds text prefill chunks, reducing intermediate results
without shortening the full logical prompt.

## Images and model support

Use `MLXBackend(model_path, multimodal=True)` with the `mlx-vlm` extra. Images in
`state` and `instructions` use the same multimodal objects as SGLang and Transformers.
Without `multimodal=True`, image evidence raises an error.

Qwen2-VL and Qwen2.5-VL support chunked prefill, equal-length candidate batches,
image feature reuse within a request, and prefix reuse across requests. Prefix
keys include image content digests, so different images cannot accidentally share
KV state solely because their placeholder token IDs match. Changed image content
at the same source is read again on a later request. Image feature storage within
a request has its own entry and byte limits using the same configured budgets.

Other causal MLX-VLM models use independent full prefills through their embedding
and language-model interfaces. Their cache layouts and position handling may differ,
so chunking, batching and prefix optimization are not assumed for every model.
Encoder-decoder models are unsupported. MLX-VLM's legacy `BaseImageProcessor` path
accepts only one image per prompt; multiple images produce a clear error on that
processor path. Other processors can accept multiple images according to the model.

## HTTP service

```bash
export LLM2JEV_API_KEY="replace-with-your-api-key"
uv run --extra mlx --extra server llm2jev-serve \
  --backend mlx \
  --model-path /path/to/mlx-model \
  --served-model-name local-model \
  --host 127.0.0.1 --port 30000 \
  --batch-size 8 --prefill-step-size 512 \
  --submission staged --cache-size 32 --cache-bytes 536870912
```

For images, use `--extra mlx-vlm --extra server` and add `--multimodal` with a VLM
model directory. `llm2jev-serve` still defaults to SGLang when `--backend` is omitted.
The MLX service provides:

| Endpoint | Behavior |
| --- | --- |
| `POST /v1/systemone` | Existing Jev request and response JSON |
| `GET /v1/models` | OpenAI-style model list with the configured alias |
| `GET /health` | Public readiness probe; 200 when loaded, 503 when unavailable |

`--api-key` overrides `LLM2JEV_API_KEY`. When a key is configured, `/v1` endpoints
require `Authorization: Bearer <key>`. With neither configured, authentication is
disabled. The request's `model` must exactly match `--served-model-name`, or the
model path when no alias is given; unknown models return 404. Invalid requests
return 422, and internal inference errors return a generic response without model
paths or stack traces.

Model loading, scoring and disposal run on one dedicated worker thread. Concurrent
HTTP requests queue for that worker, while health checks remain responsive. Client
cancellation does not let a second request enter an active model call. Batching is
within each request; the service does not combine candidates from separate HTTP
requests into a continuous batch. The existing [web demo](../demos/web/README.md)
works with this service. Other SGLang-native endpoints are not implemented by MLX.

## Probabilities and usage

For each candidate, the final input position supplies the next-token logits.
The scorer computes `softmax([no_logit, yes_logit])[1]` in float32. `LLM2Jev`
continues to normalize candidate probabilities and assemble the final answers.
Different weights, quantization and numerical precision can produce different
probabilities across backends.

`usage.input_tokens` sums each candidate's logical input length, including repeated
context, chat-template tokens and expanded image positions for VLMs. Cache hits
and batching do not reduce this count. `usage.output_tokens` is always zero.

## Verification

```bash
uv run --extra mlx-vlm --extra server python -m unittest discover -s tests -v
uv run --extra mlx-vlm --extra server python -m compileall -q src tests examples
uv build
```

Deterministic tests cover scoring, cache isolation and eviction, image routing,
HTTP validation, authentication and cancellation. Native checks create tiny local
models and compare optimized scores with complete forward passes, including
float32 and 4-bit text checkpoints. These tests need no network or pretrained
model download. They verify numerical behavior, not trained-model decision quality
or production throughput. Tests requiring optional dependencies are skipped when
those dependencies are unavailable.

The implementation follows the label-logit scoring approach in
[NanoJev's frozen native baseline](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9/scripts/train_toy_decisions.py),
while retaining LLM2Jev's binary protocol. That NanoJev snapshot did not contain an
MLX backend.
