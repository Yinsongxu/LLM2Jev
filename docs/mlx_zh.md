# MLX 后端

[English](mlx.md) · [使用指南](usage_zh.md) · [返回 README](../README_zh.md)

`MLXBackend` 在 Apple Silicon 的 macOS 环境中运行本地文本模型和视觉语言模型。
它与其他后端使用相同的 `JevRequest`、Choice、Score、Noul 和响应格式。
Python 调用和 `POST /v1/systemone` 均通过 prefill 完成二元评分，不生成输出 token。

## 安装与示例

按模型类型和调用方式选择依赖：

```bash
# 文本 Python API
uv sync --extra mlx
# 文本 HTTP 服务
uv sync --extra mlx --extra server
# 图片 Python API 和 HTTP 服务
uv sync --extra mlx-vlm --extra server
# 上一条命令对应的 pip 安装方式：
# python -m pip install -e '.[mlx-vlm,server]'
```

准备包含权重、配置和 tokenizer 或 processor 的本地模型目录。
文本模型须兼容 MLX-LM，图片模型须兼容 MLX-VLM，也支持对应运行时兼容的量化模型。
请先下载或转换模型；远程模型标识不能直接作为本地目录使用。

运行示例时在 `uv run` 中保留所选 extra，避免依赖同步移除所需包：

```bash
uv run --extra mlx python examples/mlx_inference.py \
  --model-path /path/to/mlx-model --batch-size 8 --submission staged

uv run --extra mlx-vlm python examples/multimodal_inference.py \
  --backend mlx --model-path /path/to/mlx-vlm --image /path/to/photo.png
```

图片示例支持本地路径、`file://` URI、HTTP(S) URL 和 Base64 data URL。
通过 `--placement instructions` 可把图片放入各问题的 `instructions`，默认放在共享的
`state` 中。请求格式见[多模态输入](multimodal_zh.md)。

## Python API 与生命周期

```python
from llm2jev import JevRequest, LLM2Jev, MLXBackend, Noul

model_path = "/path/to/mlx-model"
request = JevRequest(
    model=model_path,
    state="客户的包裹一直没有送到。",
    questions={"delivery": Noul(instructions="这是物流配送问题吗？")},
)
with MLXBackend(model_path, batch_size=8, submission="staged") as backend:
    converter = LLM2Jev(backend=backend)
    print(converter.evaluate(request).json)
    # 后续请求可复用前缀，无需重新加载模型。
    print(converter.evaluate(request).json)
    backend.clear_cache()  # 保留模型，释放可复用缓存。
```

退出上下文管理器时调用 `close()`，释放模型与缓存引用。
也可显式调用 `close()`，重复调用安全；关闭后再调用 `score()` 会报错。
模型由构造函数确定；Python 请求的 `request.model` 用于响应中的模型标识，不会切换模型。

配置默认值如下：

```python
MLXBackend(
    model_path,                  # str 或 pathlib.Path
    yes_label="yes",
    no_label="no",
    enable_thinking=False,
    prefill_step_size=512,
    batch_size=8,
    submission="staged",         # "staged" 或 "all"
    max_cache_entries=32,
    max_cache_bytes=512 * 1024 * 1024,
    multimodal=False,            # True 时通过 MLX-VLM 加载模型
)
```

两个标签都必须编码为单个 token，且 token ID 不同；自定义标签需要匹配的 prompt renderer。
`enable_thinking=False` 会传给 chat template，后端不会生成思考过程。
`prefill_step_size` 与 `batch_size` 必须为正整数，缓存上限须为非负整数；
任一缓存上限设为零都会禁用可复用缓存存储。

## 批量评分与共享前缀

文本候选进入实际的模型 batch，每批至多 `batch_size` 个，最终结果保持输入顺序。
变长 prompt 一起推进到最短的剩余序列结束，已完成行退出 batch，因此不需要给输入补齐 token。
模型及其缓存都需要兼容批量缓存状态，并支持原生合并与提取；不支持的缓存，以及保留固定
起始 token 的旋转缓存，会逐条评分，以保留该模型原有的缓存语义。
MLX-LM 0.31.3 中的 AFM7 与 Gemma3n 直接解包单条序列的缓存状态，因此会自动按批大小 1
评分（对应 `afm7`、`gemma3n`、`gemma3n_text` 模型类型），同时保留 staged 前缀共享和跨请求复用。

`submission="staged"` 会先对共享 token 前缀执行 prefill，再评分候选后缀，
既能在首次请求内部复用上下文，也能跨请求复用已有前缀。
`submission="all"` 省略这一步显式预热，但仍可命中已有缓存；它不会禁用缓存，
也不会覆盖 `batch_size` 的批大小限制。

缓存快照与每个候选的可变缓存隔离，并按最近最少使用规则淘汰。
条目数和 tensor 字节数共同限制缓存；单条快照超过字节上限时不会存储。
这些参数约束保留的可复用缓存，不是模型权重、推理中间状态或整个进程的内存上限。
`prefill_step_size` 限制文本 prefill 分块，降低中间结果大小，但不会缩短完整逻辑输入。

## 图片与模型支持范围

安装 `mlx-vlm` extra，并使用 `MLXBackend(model_path, multimodal=True)`。
`state` 和 `instructions` 中的图片使用与 SGLang、Transformers 相同的多模态对象。
未启用 `multimodal=True` 时，图片输入会明确报错。

Qwen2-VL 与 Qwen2.5-VL 支持分块 prefill、等长候选批量评分、请求内图片特征复用，
以及跨请求前缀复用。前缀缓存键包含图片内容摘要，因此不同图片不会因为占位 token 相同
而错误复用 KV 状态。后续请求会重新读取图片，即使地址不变也能区分已变化的内容。
请求内的图片特征存储单独使用相同的条目数和字节预算。

其他因果 MLX-VLM 模型通过自身 embedding 和语言模型接口逐条执行完整 prefill。
它们的缓存布局和位置处理可能不同，因此不会假定所有模型都支持分块、批量与前缀优化。
不支持 encoder-decoder 模型。MLX-VLM 旧式 `BaseImageProcessor` 路径每个 prompt
仅支持一张图片，多图输入会明确报错；其他 processor 可按对应模型能力处理多图。

## HTTP 服务

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

图片服务改用 `--extra mlx-vlm --extra server`，添加 `--multimodal`，并使用 VLM 模型目录。
省略 `--backend` 时，`llm2jev-serve` 仍默认使用 SGLang。
MLX 服务提供以下接口：

| 接口 | 行为 |
| --- | --- |
| `POST /v1/systemone` | 既有 Jev 请求与响应 JSON |
| `GET /v1/models` | OpenAI 风格模型列表，使用配置的模型别名 |
| `GET /health` | 公开就绪探针；加载完成返回 200，不可用时返回 503 |

`--api-key` 优先于 `LLM2JEV_API_KEY`。配置密钥后，`/v1` 接口要求
`Authorization: Bearer <key>`；两者均未配置时不启用鉴权。
请求中的 `model` 必须与 `--served-model-name` 完全一致；未指定别名时使用模型路径。
未知模型返回 404，无效请求返回 422，内部推理错误返回通用错误，不暴露模型路径或堆栈。

模型加载、评分与释放在同一专属工作线程运行。并发 HTTP 请求排队执行，
健康检查仍可及时响应；客户端取消不会导致下一个请求与正在执行的模型调用重叠。
批量评分发生在单个请求内部，服务不会把不同 HTTP 请求合并为连续批处理。
既有[网页 demo](../demos/web/README.md)可直接连接该服务。
MLX 不提供 SGLang 的其他原生接口。

## 概率与用量

每个候选在最后一个输入位置读取下一 token 的 logits，以 float32 计算
`softmax([no_logit, yes_logit])[1]`。`LLM2Jev` 继续负责候选概率归一化与答案组装。
权重、量化方式和数值精度不同，可能导致不同后端给出不同概率。

`usage.input_tokens` 累加每个候选的逻辑输入长度，包含重复上下文、chat template token，
以及 VLM 展开后的图片位置。缓存命中和批量执行不会减少这一统计值。
`usage.output_tokens` 始终为零。

## 验证

```bash
uv run --extra mlx-vlm --extra server python -m unittest discover -s tests -v
uv run --extra mlx-vlm --extra server python -m compileall -q src tests examples
uv build
```

确定性测试覆盖评分、缓存隔离与淘汰、图片路由、HTTP 校验、鉴权及取消行为。
原生验证在本地创建微型模型，将优化后的结果与完整前向计算对照，包含 float32 和 4-bit
文本模型，不需要联网或下载预训练模型。这些验证用于检查数值行为，不代表大模型决策质量
或生产吞吐量测评；缺少可选依赖时，对应测试会跳过。

本实现借鉴了 [NanoJev 的 frozen native baseline](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9/scripts/train_toy_decisions.py)
读取标签 logits 的方式，并保持 LLM2Jev 的二元协议；该 NanoJev 版本本身没有 MLX 后端。
