# LLM2Jev Web Demo

The web demo builds mixed `Choice`, `Score`, and `Noul` requests and visualizes
the response probabilities. It connects to `/v1/systemone` and `/v1/models`
provided by either the SGLang or MLX `llm2jev-serve` backend.

Start SGLang on Linux with a supported NVIDIA GPU:

```bash
uv run --extra sglang llm2jev-serve \
  --model-path /path/to/model \
  --served-model-name local-model \
  --host 127.0.0.1 --port 30000
```

Or start MLX on Apple Silicon macOS:

```bash
uv run --extra mlx --extra server llm2jev-serve \
  --backend mlx --model-path /path/to/mlx-model \
  --served-model-name local-model \
  --host 127.0.0.1 --port 30000
```

For image requests, start a compatible vision-language model. With MLX, use
`--extra mlx-vlm --extra server` and add `--multimodal`:

```bash
uv run --extra mlx-vlm --extra server llm2jev-serve \
  --backend mlx --multimodal --model-path /path/to/mlx-vlm \
  --served-model-name local-vlm --port 30000
```

See the [multimodal guide](../../docs/multimodal.md) for the request format and
[MLX guide](../../docs/mlx.md) for model support and batching options.

In another terminal, start the demo server from the repository root:

```bash
python demos/web/server.py
```

Open <http://127.0.0.1:8000> and choose the model alias advertised by the service.
The demo proxies browser requests to `http://127.0.0.1:30000` by default. To use
another endpoint or port:

```bash
python demos/web/server.py \
  --api-base http://127.0.0.1:31000 \
  --port 8080
```

When API authentication is enabled, set the same `LLM2JEV_API_KEY` in both server
terminals, or pass `--api-key` to each process. MLX and the demo read that environment
variable automatically; SGLang can receive it through `--api-key "$LLM2JEV_API_KEY"`.
The proxy adds the Bearer header when forwarding requests.
