# 安装指南

[English](installation.md) · [返回 README](../README_zh.md)

克隆仓库：

```bash
git clone https://github.com/Yinsongxu/LLM2Jev.git
cd LLM2Jev
```

在配有受支持 NVIDIA GPU 的 Linux 环境中，推荐使用 SGLang 后端；它也会安装
自身依赖的 Transformers：

```bash
uv sync --extra sglang
```

如果只需要 Transformers 后端：

```bash
uv sync --extra transformers
```

在 Apple Silicon 的 macOS 环境中使用 MLX：

```bash
uv sync --extra mlx
```

MLX extra 会在该平台安装 MLX 和 MLX-LM。推理前需要准备好本地
MLX-LM 兼容的文本模型目录，也支持兼容的量化模型，详见 [MLX 指南](mlx_zh.md)。

在 Apple Silicon 上使用图片模型或 HTTP 服务：

```bash
uv sync --extra mlx-vlm                    # MLX-VLM Python API
uv sync --extra mlx --extra server         # 文本 HTTP 服务
uv sync --extra mlx-vlm --extra server     # 图片 HTTP 服务
```

`mlx-vlm` extra 包含 MLX-LM 和 MLX-VLM，`server` 添加 FastAPI 与 Uvicorn。
后续执行 `uv run` 时保留对应 extra。模型权重需要提前准备在本地，服务不会自动下载模型。

使用 pip 可编辑安装时，选择对应的 extra：

```bash
python -m pip install -e ".[sglang]"
# 或者：python -m pip install -e ".[transformers]"
# 或者，在 Apple Silicon 上：python -m pip install -e ".[mlx]"
# 图片与 HTTP 服务：python -m pip install -e ".[mlx-vlm,server]"
```

使用 uv 安装后，激活虚拟环境：

```bash
source .venv/bin/activate
```

安装完成后，参阅[使用指南](usage_zh.md)运行 Python 示例或启动 SGLang、MLX HTTP 服务。
