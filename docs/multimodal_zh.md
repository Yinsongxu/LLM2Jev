# 使用多模态数据

[English](multimodal.md) · [返回 README](../README_zh.md)

LLM2Jev 支持在 `state` 或问题的 `instructions` 中提交图文内容。

## 支持的图片形式

图片地址填写在 `image_url.url` 中，支持以下形式：

| 形式 | 示例 |
| --- | --- |
| HTTP(S) URL | `https://example.com/photo.png` |
| 本地文件 URI | `file:///data/photo.png` |
| 本地路径 | `/data/photo.png` 或 `images/photo.png` |
| Base64 data URL | `data:image/png;base64,...` |


## 构造请求

在 `state` 或 `instructions` 中使用带有 `type: "multimodal"` 标记的对象：

```json
{
  "type": "multimodal",
  "content": [
    {"type": "text", "text": "请检查这张图片。"},
    {"type": "image_url", "image_url": {"url": "file:///data/photo.png"}}
  ]
}
```

`content` 是非空列表，可包含多段文字、多张图片，也可以仅包含文字或图片；内容按列表顺序提交。文字块的 `text` 必须是字符串，图片的 `url` 必须是非空字符串。目前仅支持示例中的字段，不支持 `detail`、`file_id`、音频或视频。

必须保留外层标记对象。直接把 `content` 列表传入 `state` 或 `instructions`，会作为普通 JSON 数组转换为文字，不会加载图片。普通字符串、未标记的 JSON 对象和数组仍可按原方式使用；`criteria` 不解析图片。

下面将图片放在 `state`，让多个问题使用同一张图片。后续后端示例均使用这里的 `model_path` 和 `request`：

```python
from llm2jev import Choice, JevRequest, Noul

model_path = "/path/to/vlm"
request = JevRequest(
    model=model_path,
    state={"type": "multimodal", "content": [
        {"type": "text", "text": "检查图片"},
        {"type": "image_url", "image_url": {"url": "file:///data/photo.png"}},
    ]},
    questions={
        "person": Noul(instructions="图中有人吗？"),
        "color": Choice(
            instructions="主要颜色是什么？",
            criteria={"red": "红色", "blue": "蓝色", "green": "绿色"},
        ),
    },
)
```

也可以将图文对象放在某个问题的 `instructions` 中：

```python
instructions = JevRequest(
    model=model_path,
    state="检查所附图片。",
    questions={
        "person": Noul(instructions={
            "type": "multimodal", "content": [
                {"type": "text", "text": "图中有人吗？"},
                {"type": "image_url", "image_url": {"url": "file:///data/photo.png"}},
            ],
        }),
    },
)
```


## SGLang 后端


```python
from llm2jev import LLM2Jev, SGLangBackend

if __name__ == "__main__":
    with SGLangBackend(model_path) as backend:
        response = LLM2Jev(backend=backend).evaluate(request)
        print(response.json)
```

## Transformers 后端

创建后端时指定 `multimodal=True`：

```python
from llm2jev import LLM2Jev, TransformersBackend

with TransformersBackend(model_path, multimodal=True, dtype="bfloat16") as backend:
    response = LLM2Jev(backend=backend).evaluate(request)
    print(response.json)
```

## MLX 后端

创建后端时指定 `multimodal=True`：

```python
from llm2jev import LLM2Jev, MLXBackend

with MLXBackend(model_path, multimodal=True, batch_size=8) as backend:
    response = LLM2Jev(backend=backend).evaluate(request)
    print(response.json)
```


## HTTP API 调用

按照[使用指南](usage_zh.md#online-http-服务)启动 SGLang 多模态服务。


```bash
curl http://localhost:30000/v1/systemone \
  -H "Authorization: Bearer $LLM2JEV_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-vlm",
    "state": {
      "type": "multimodal",
      "content": [
        {"type": "text", "text": "检查图片"},
        {"type": "image_url", "image_url": {"url": "file:///data/photo.png"}}
      ]
    },
    "questions": {
      "person": {
        "type": "noul",
        "instructions": "图中有人吗？"
      },
      "color": {
        "type": "choice",
        "instructions": "主要颜色是什么？",
        "criteria": {"red": "红色", "blue": "蓝色", "green": "绿色"}
      }
    }
  }'
```

图片也可按前面的示例放在某个问题的 `instructions` 中。

## 如何选择图片位置

| 场景 | 建议位置 | 使用方式 |
| --- | --- | --- |
| 同一张图回答多个问题 | `state` | 图片只需填写一次，所有问题共享 |
| 某张图只与一个问题有关 | 该问题的 `instructions` | 图片随该问题提交 |
| 不同图片回答相同问题 | 可选 `instructions` | 固定 `state` 和问题文字，每张图分别构造请求 |
| 多张图需要联合比较或判断 | 同一个 `content` 列表 | 按所需顺序排列文字和图片 |

`state` 中的内容位于问题之前；`instructions` 中的内容位于该问题的位置。每个 `content` 列表内部的文字与图片顺序由你决定。两个位置同时放图时，模型会结合两处图片判断。

如果关注前缀缓存，通常可把重复使用的内容放在前面，但是否能复用、是否更快，需要用实际模型和请求验证，不能仅凭图片位置保证。
