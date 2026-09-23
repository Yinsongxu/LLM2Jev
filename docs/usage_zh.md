# 使用指南

[English](usage.md) · [返回 README](../README_zh.md)

请先完成[安装](installation_zh.md)。使用前准备好模型文件：SGLang 和 Transformers 使用 Hugging Face 兼容模型，MLX 使用 MLX-LM 或 MLX-VLM 模型。

## Offline: Python API

三种后端都通过 `LLM2Jev` 和 `JevRequest` 调用。按安装的后端导入对应 backend：

```python
from llm2jev import Choice, JevRequest, LLM2Jev, Noul, Score

request = JevRequest(
    model="/path/to/model",
    state="包裹晚到了两周，信用卡还被扣了两次。",
    questions={
        "department": Choice(criteria={"shipping": "物流配送", "billing": "账单问题"}),
        "severity": Score(criteria=["低", "中", "高"]),
        "delivery": Noul(instructions="这是物流配送问题吗？"),
    },
)
```

**SGLang**（Linux/NVIDIA）：

```python
from llm2jev import SGLangBackend

with SGLangBackend("/path/to/model") as backend:
    response = LLM2Jev(backend=backend).evaluate(request)
```

SGLang 默认使用 `submission="staged"`，也可设为 `"all"`。使用脚本时将调用放进
`if __name__ == "__main__":`，因为 SGLang 会启动工作进程。

**Transformers**：

```python
from llm2jev import TransformersBackend

with TransformersBackend("/path/to/model") as backend:
    response = LLM2Jev(backend=backend).evaluate(request)
```

Transformers 在 CUDA 可用时使用 CUDA，否则使用 CPU。
图片模型同样通过`TransformersBackend(..., multimodal=True)` 启用。

**MLX**（Apple Silicon）：

```python
from llm2jev import MLXBackend

with MLXBackend("/path/to/mlx-model") as backend:
    response = LLM2Jev(backend=backend).evaluate(request)
```

图片模型需使用 `MLXBackend(..., multimodal=True)`。详见[多模态输入](multimodal_zh.md)和 [MLX 部分](#offline-python-api)。

## Online: HTTP 服务

激活安装时创建的虚拟环境后，可直接运行 `llm2jev-serve`。SGLang 使用其原生启动参数，MLX 和 Transformers 使用 `--backend` 选择：

**SGLang**（Linux/NVIDIA）：

```bash
llm2jev-serve --backend sglang --model-path /path/to/model --host 127.0.0.1 --port 30000
```

**Transformers**：

```bash
llm2jev-serve --backend transformers --model-path /path/to/model \
  --served-model-name local-model --host 127.0.0.1 --port 30000
```

**MLX**（Apple Silicon）：

```bash
llm2jev-serve --backend mlx --model-path /path/to/mlx-model \
  --served-model-name local-model --host 127.0.0.1 --port 30000
```

三种服务均提供 `POST /v1/systemone` 和 `GET /v1/models`。MLX 与 Transformers 还提供公开的 `GET /health`。设置环境变量 `LLM2JEV_API_KEY` 可保护 `/v1` 接口；也可通过 `--api-key` 设置。

请求中的 `model` 必须与 `--served-model-name` 一致；未指定别名时使用模型路径。示例：

```bash
curl http://localhost:30000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": "客户的包裹一直没有送到。",
    "model": "local-model",
    "questions": {
      "delivery": {
        "type": "noul",
        "instructions": "这是物流配送问题吗？"
      }
    }
  }'
```

SGLang 和 MLX 支持 `--submission staged|all`；Transformers 也接受该参数以保持命令一致，但目前不影响其推理行为。

## 如何选择 staged 或 all？

下表给出 SGLang 的选择建议。MLX 也支持这两种模式，但通过显式前缀预热实现；Transformers 当前不应用此参数。

| 请求特点 | 建议起点 | 原因 |
| --- | --- | --- |
| 上下文长、候选多，相关缓存尚不存在 | `staged`（默认） | 避免冷请求内重复处理共享前缀 |
| 输入短、候选少 | `all` | 减少多轮提交开销 |
| 重复请求，大部分前缀已缓存 | `all` | 可直接复用缓存，无需分阶段预热 |
| 部分缓存命中或输入差异较大 | 对比两种方式 | 性能取决于共享计算量和提交开销 |

后端不会自动探测缓存状态并切换模式。SGLang 的 `staged` 需要 Radix Cache；禁用 Radix Cache 时使用 `all`。
