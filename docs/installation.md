# Installation

[简体中文](installation_zh.md) · [Back to README](../README.md)

Clone the repository:

```bash
git clone https://github.com/Yinsongxu/LLM2Jev.git
cd LLM2Jev
```

SGLang is the recommended backend on Linux with a supported NVIDIA GPU. It also
installs its Transformers dependency:

```bash
uv sync --extra sglang
```

For a Transformers-only environment:

```bash
uv sync --extra transformers
```

For MLX on macOS with Apple Silicon:

```bash
uv sync --extra mlx
```

The MLX extra installs MLX and MLX-LM on this platform. Prepare a local
MLX-LM-compatible text model directory before running inference; compatible
quantized models are supported. See the [MLX guide](mlx.md).

For image models and/or HTTP serving on Apple Silicon:

```bash
uv sync --extra mlx-vlm                    # MLX-VLM Python API
uv sync --extra mlx --extra server         # Text HTTP service
uv sync --extra mlx-vlm --extra server     # Image HTTP service
```

The `mlx-vlm` extra includes MLX-LM and MLX-VLM; `server` adds FastAPI and Uvicorn.
Keep these extras on subsequent `uv run` commands. Model weights remain local
inputs and are not downloaded by the server.

For an editable pip installation, use the corresponding extra:

```bash
python -m pip install -e ".[sglang]"
# Or: python -m pip install -e ".[transformers]"
# Or, on Apple Silicon: python -m pip install -e ".[mlx]"
# With images and HTTP: python -m pip install -e ".[mlx-vlm,server]"
```

After installing with uv, activate the virtual environment:

```bash
source .venv/bin/activate
```

After installation, follow the [Usage guide](usage.md) to run Python examples or start the SGLang or MLX HTTP service.
