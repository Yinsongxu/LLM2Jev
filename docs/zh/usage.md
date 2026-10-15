# 使用指南

[English](../en/usage.md) · [返回 README](../../README_zh.md)

请先完成[安装](installation.md)。使用前准备好模型文件：SGLang 和 Transformers 使用 Hugging Face 兼容模型，MLX 使用 MLX-LM 或 MLX-VLM 模型。

## Offline: Python API

三种后端都通过 `LLM2Jev` 和 `JevRequest` 调用。按安装的后端导入对应 backend：

```python
from llm2jev import Choice, JevRequest, LLM2Jev, Noul, Score

request = JevRequest(
    model="/path/to/model",
    state="包裹晚到了两周，信用卡还被扣了两次。",
    # include_choice_candidates=False,  # 可选：省略 Choice 的完整候选列表
    questions={
        "department": Choice(criteria={"shipping": "物流配送", "billing": "账单问题"}),
        "severity": Score(criteria=["低", "中", "高"]),
        "delivery": Noul(instructions="这是物流配送问题吗？"),
    },
)
```

`include_choice_candidates` 是每次请求的顶层布尔参数，默认为 `True`。设为 `False` 时，默认 renderer 在每个 Choice 判断中仅保留当前候选及其说明，省略 `All candidates:` 列表。文本和多模态输入均支持；所有选项仍参与评分，Noul 和 Score 的提示词不受影响。各次请求的设置互不影响。

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

图片模型需使用 `MLXBackend(..., multimodal=True)`。详见[多模态输入](multimodal.md)和 [MLX 部分](#offline-python-api)。

自定义后端实现从 `llm2jev` 导出的 `Backend` 协议：`score()` 返回 `ScoreOutput`，`generate()` 返回 `GenerationOutput`。`Backend` 替代 `BinaryBackend`，统一要求提供这两个方法，三个内置后端均已实现。Box 仍需使用支持图片的模型及后端多模态配置。

两种输出均使用 `[0, 1]` 概率：`ScoreOutput.yes_probabilities` 在 `yes`/`no` 两者间归一化；`GenerationOutput.token_probabilities` 是每个生成 token 在完整词表上的概率，与 `token_texts` 逐项对应。

自定义 renderer 实现 `render(request: JevRequest) -> RenderedPrompts`，通过 `LLM2Jev(backend=backend, renderer=renderer)` 传入。renderer 直接处理完整请求，包括 Box 提示词，不再经过独立的二分类问题编译步骤。`DefaultPromptRenderer` 为每种题型提供一个静态方法。

`RenderedPrompts` 包含 `score_prompts`、`score_question_ids`、`box_prompts` 和 `box_question_ids`，每个提示词对应一个问题 ID。同一评分问题内，Choice 和 Score 保持 criteria 顺序；带 criteria 的 Noul 按 true、false 顺序生成两个提示词，不带 criteria 时生成一个。每个 Box 生成一个坐标提示词。

`assemble_response(request=request, rendered=rendered, scored=scored, generated=generated)` 根据这些 ID 对应后端输出与问题，统一组装答案并合并用量。不需要执行的推理阶段可省略对应输出；省略 `rendered` 时，输出须遵循默认的请求及候选顺序。可选的 `normalizer` 控制候选概率归一化。

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

请求中的 `model` 必须与 `--served-model-name` 一致；未指定别名时使用模型路径。

### Python client

`Client` 和 `AsyncClient` 继承 OpenAI Python SDK 的 client，新增 `systemone.create()`。它们包含在包内，无需额外安装。

```python
from llm2jev import Client, Choice

client = Client(base_url="http://localhost:30000/v1", api_key="unused")

print(client.models.list())
response = client.systemone.create(
    model="local-model",
    state="客户的包裹一直没有送到。",
    questions={
        "department": Choice(
            instructions="哪个部门应当处理？",
            criteria={"shipping": "物流配送", "billing": "账单问题"},
        ),
    },
    include_choice_candidates=False,
)
print(response.choices["department"].probabilities)
```

`questions` 支持 `Box`、`Choice`、`Score`、`Noul` 对象，也支持包含 `type` 的 JSON 字典，两种写法可以混用。`state` 与 offline 请求一样支持文本、JSON 和多模态内容。返回值是 `JevResponse`，包含类型化答案、`usage`、`to_dict()` 和 `json`。`include_choice_candidates` 默认 `True`，可在每次调用时设置。

异步接口使用相同参数：

```python
import asyncio
from llm2jev import AsyncClient, Noul

async def main():
    client = AsyncClient(base_url="http://localhost:30000/v1", api_key="unused")

    response = await client.systemone.create(
        model="local-model",
        state="客户的包裹一直没有送到。",
        questions={"delivery": Noul(instructions="这是物流配送问题吗？")},
    )
    print(response.nouls["delivery"].noul)

asyncio.run(main())
```

同一个 client 可以重复调用。以上短脚本省略了显式关闭，进程退出时操作系统会回收连接资源。长期运行的程序应在不再使用 client 时调用 `client.close()`；异步 client 使用 `await client.close()`。

`base_url` 应指向 llm2jev 服务的 `/v1` 地址。已启用认证时使用服务配置的 API key；未启用时可传入 `"unused"` 等占位值。OpenAI 官方服务不提供 `/systemone`。

OpenAI SDK 的 `chat.completions`、`responses`、`models` 等接口，以及构造参数、上下文管理器、`with_options()` 均保留。能否调用某个接口取决于服务端：SGLang 提供其原生的 OpenAI 兼容接口；MLX 和 Transformers 服务目前只提供 `/v1/systemone`、`/v1/models` 和 `/health`。

`systemone.create()` 支持单次调用的 `timeout`、`extra_headers` 和 `extra_query`。重试参数可在 client 初始化时配置，或通过 `client.with_options(max_retries=...)` 设置。HTTP 错误和超时沿用 OpenAI SDK 异常；成功响应的内容不合法时抛出 `openai.APIResponseValidationError`。请求参数在发送 HTTP 请求前通过现有请求模型校验。

### curl

也可以直接通过 HTTP 发送请求：

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

需要省略 Choice 的完整候选列表时，在本次 HTTP 请求顶层传入 `"include_choice_candidates": false`。三种后端均支持，无需重启服务；省略字段或传入 `true` 时渲染完整列表。参数只接受布尔值，字符串 `"false"`、数字及 `null` 会返回 422。例如：

```json
{
  "model": "local-model",
  "state": "客户的包裹一直没有送到。",
  "include_choice_candidates": false,
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "哪个部门应当处理？",
      "criteria": {"shipping": "物流配送", "billing": "账单问题"}
    }
  }
}
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
