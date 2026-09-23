# 安装指南

[English](installation.md) · [返回 README](../README_zh.md)

## 步骤一：克隆仓库

```bash
git clone https://github.com/Yinsongxu/LLM2Jev.git
cd LLM2Jev
```

## 步骤二：选择安装方式

### 使用 uv（推荐）

根据需要选择一个后端安装。

SGLang 后端（推荐用于配备受支持 NVIDIA GPU 的 Linux 环境）：

```bash
uv sync --extra sglang
```

Transformers 后端：

```bash
uv sync --extra transformers
```

MLX（Apple Silicon） 后端：

```bash
uv sync --extra mlx
```

### 使用 pip

先创建并激活虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
```

然后根据需要选择一个后端安装：

```bash
python -m pip install -e ".[sglang]"
# Transformers 后端：
python -m pip install -e ".[transformers]"
# Apple Silicon 上的 MLX 后端：
python -m pip install -e ".[mlx]"
```

安装完成后，参阅[使用指南](usage_zh.md)运行 Python 示例或启动 SGLang、MLX、Transformers HTTP 服务。
