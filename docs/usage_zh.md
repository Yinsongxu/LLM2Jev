# 使用指南

[English](usage.md) · [返回 README](../README_zh.md)

本页介绍本地 Python 调用、HTTP 服务和候选提交模式。请先完成[安装](installation_zh.md)。SGLang 和 Transformers 示例使用本地 Hugging Face 兼容的因果语言模型，请将 `/path/to/model` 替换为实际模型目录。MLX 使用本地 MLX-LM 文本模型或 MLX-VLM 图片模型。

## SGLang Python API

`JevRequest` 包含 `state`、`model` 和 `questions`。下面的示例同时提交 Choice、Score 和 Noul 三种问题；本地 SGLang 后端默认使用分阶段提交：

```python
from llm2jev import Choice, JevRequest, LLM2Jev, Noul, Score, SGLangBackend

if __name__ == "__main__":
    model_path = "/path/to/model"
    request = JevRequest(
        model=model_path,
        state="包裹晚到了两周，信用卡还被扣了两次。",
        questions={
            "department": Choice(
                instructions="哪个部门最应该处理这个请求？",
                criteria={
                    "shipping": "物流配送问题",
                    "billing": "扣款和账单问题",
                    "returns": "退货和换货问题",
                },
            ),
            "severity": Score(
                instructions="这个问题有多严重？",
                criteria=["低：影响轻微", "中：存在问题但仍可继续使用", "高：无法继续使用"],
            ),
            "delivery": Noul(
                instructions="这是物流配送问题吗？",
            ),
        },
    )
    with SGLangBackend(model_path, submission="staged") as backend:
        response = LLM2Jev(backend=backend).evaluate(request)
        print(response.to_dict())
```

`submission="staged"` 可以省略；显式指定便于说明采用哪种方式。SGLang 会创建工作进程，因此脚本入口需要 `if __name__ == "__main__":`。批量处理多个请求时，可以在同一个 `with` 块内反复调用，避免重复加载模型。

上下文管理器会在退出时关闭引擎，`engine_kwargs` 可用于传入 SGLang 引擎配置。

要将所有候选一次提交，改用：

```python
with SGLangBackend(model_path, submission="all") as backend:
    response = LLM2Jev(backend=backend).evaluate(request)
```

仓库示例也支持切换：

```bash
python examples/sglang_inference.py --model-path /path/to/model --submission staged
python examples/sglang_inference.py --model-path /path/to/model --submission all
```

## Transformers 后端

在仅安装 Transformers 后端依赖的环境中运行示例：

```bash
python examples/transformers_inference.py --model-path /path/to/model
```

使用上例的 `model_path` 和 `request`，将后端调用部分替换为：

```python
from llm2jev import LLM2Jev, TransformersBackend

backend = TransformersBackend(model_path)
response = LLM2Jev(backend=backend).evaluate(request)
print(response.to_dict())
```

Transformers 后端会优先使用 CUDA；没有可用 GPU 时自动回退到 CPU。

## MLX 后端

在 Apple Silicon 的 macOS 环境中，使用 `mlx` extra 运行文本示例：

```bash
uv run --extra mlx python examples/mlx_inference.py \
  --model-path /path/to/mlx-model --batch-size 8 --submission staged
```

`MLXBackend` 使用同样的请求对象：

```python
from llm2jev import LLM2Jev, MLXBackend

with MLXBackend("/path/to/mlx-model", batch_size=8, submission="staged") as backend:
    response = LLM2Jev(backend=backend).evaluate(request)
    print(response.to_dict())
```

MLX 通过既有转换流程支持 Choice、Score 和 Noul。模型缓存能力允许时，文本候选会实际
组成变长 batch，并复用前缀快照；不支持的缓存类型按逐条方式评分。
`staged` 先显式预热共享前缀，`all` 省略预热但仍可命中已有缓存；两者均保持候选顺序和
逻辑输入 token 统计口径。缓存默认最多 32 条、512 MiB tensor；
`max_cache_entries=0` 或 `max_cache_bytes=0` 会禁用可复用缓存存储。

图片模型安装 `mlx-vlm`，并使用 `MLXBackend(..., multimodal=True)`。
Qwen2-VL 与 Qwen2.5-VL 支持图片特征复用、前缀缓存和等长批量评分；
其他因果 VLM 按 prompt 执行完整 prefill。在同一个 `with` 块中复用后端处理多个请求。
`clear_cache()` 清空可复用缓存，`close()` 或退出上下文会释放模型。
完整配置与模型支持范围见 [MLX 指南](mlx_zh.md)，图片示例见[多模态输入](multimodal_zh.md)。

## System One HTTP API

`llm2jev-serve` 默认使用 SGLang 并保留其原生启动参数；`--backend mlx` 选择 Apple Silicon 服务。
两者提供相同请求与响应格式的 `POST /v1/systemone`，以及 `/v1/models`、`/health`。
使用 SGLang 时，其其他原生端点仍然可用。启动 SGLang：

```bash
export LLM2JEV_API_KEY="replace-with-your-api-key"
llm2jev-serve \
  --model-path /path/to/model \
  --served-model-name local-model \
  --host 0.0.0.0 \
  --port 30000 \
  --api-key "$LLM2JEV_API_KEY"
```

也可使用相同模型别名启动 MLX：

```bash
uv run --extra mlx --extra server llm2jev-serve \
  --backend mlx --model-path /path/to/mlx-model \
  --served-model-name local-model --host 127.0.0.1 --port 30000 \
  --batch-size 8 --submission staged --cache-size 32 --cache-bytes 536870912
```

MLX 默认读取 `LLM2JEV_API_KEY`，也可通过 `--api-key` 覆盖；配置后 `/v1` 接口需要
Bearer token，`/health` 保持公开，供就绪检查使用。
MLX 图片模型改用 `--extra mlx-vlm --extra server`，并加上 `--multimodal`。
请求的 `model` 必须与服务别名完全一致，MLX 未指定别名时使用传入的模型路径。
未知模型返回 404，无效请求返回 422。MLX 使用一个专属线程执行推理，并发请求排队，
事件循环保持响应。详见 [MLX HTTP 服务](mlx_zh.md#http-服务)。

查看任一服务的模型列表：

```bash
curl http://localhost:30000/v1/models \
  -H "Authorization: Bearer $LLM2JEV_API_KEY"
```

提交 System One 请求：

```bash
curl http://localhost:30000/v1/systemone \
  -H "Authorization: Bearer $LLM2JEV_API_KEY" \
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

通过 `--submission staged|all` 选择 `/v1/systemone` 的候选提交方式，默认 `staged`。
启动时选择的模式对该服务的所有 `/v1/systemone` 请求生效。
Jev 请求体保持一致。SGLang 的 `staged` 依赖 Radix Cache；使用
`--disable-radix-cache` 时需选择 `all`。模式选择建议见[下文](#哪种方式更适合我的请求)。

使用默认的 `--backend sglang` 时，服务复用 SGLang 的启动参数，目前要求使用默认的单 tokenizer HTTP 模式，
且不能启用 `--skip-tokenizer-init`。

## 哪种方式更适合我的请求？

下表描述 SGLang 的模式选择。MLX 同样提供 `staged` 与 `all`，但采用显式前缀预热，
具体语义见 [MLX 批量评分与共享前缀](mlx_zh.md#批量评分与共享前缀)。

| 请求特点 | 建议起点 | 原因 |
| --- | --- | --- |
| 上下文长、候选多，相关缓存尚不存在 | `staged`（默认） | 避免冷请求内部重复处理长前缀 |
| 输入短、候选少 | `all` | 多轮提交的开销可能超过节省的计算 |
| 重复请求，大部分前缀已经命中缓存 | `all` | 可直接复用已有缓存，通常不需要分轮建立 |
| 部分命中或输入差异很大 | 对比两种方式 | 是否更快取决于实际共享量和提交成本 |

目前不会探测缓存状态后自动切换模式。单候选或没有可复用前缀时，分阶段规划可以直接产生一批；有共享前缀也不代表分轮一定更快。

实测数据和测试条件见[性能测评](shared-prefix-benchmarks_zh.md)。

复用原理和输出注意事项见[共享前缀说明](shared-prefix-cache_zh.md)。
