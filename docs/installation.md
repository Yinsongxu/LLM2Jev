# Installation

[简体中文](installation_zh.md) · [Back to README](../README.md)

## Step 1: Clone the repository

```bash
git clone https://github.com/Yinsongxu/LLM2Jev.git
cd LLM2Jev
```

## Step 2: Choose an installation method

### Using uv (recommended)

Choose one backend to install.

SGLang backend (recommended on Linux with a supported NVIDIA GPU):

```bash
uv sync --extra sglang
```

Transformers backend:

```bash
uv sync --extra transformers
```

MLX backend (Apple Silicon):

```bash
uv sync --extra mlx
```

### Using pip

Create and activate a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate
```

Then install one backend as needed:

```bash
python -m pip install -e ".[sglang]"
# Transformers backend:
python -m pip install -e ".[transformers]"
# MLX backend on Apple Silicon:
python -m pip install -e ".[mlx]"
```

After installation, see the [Usage guide](usage.md) to run Python examples or
start the SGLang, MLX, or Transformers HTTP service.
