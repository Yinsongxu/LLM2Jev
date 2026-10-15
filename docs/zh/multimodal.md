# 使用多模态数据

[English](../en/multimodal.md) · [返回 README](../../README_zh.md)

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

按照[使用指南](usage.md#online-http-服务)启动 SGLang 多模态服务。


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

## Box：定位图片中的目标

`Box` 返回单个目标的边界框，可与 `Choice`、`Score`、`Noul` 混合提交。首版使用 Qwen3.5 风格的 0–1000 坐标约定；其他模型需要能遵循相同的坐标和 JSON 输出要求。Transformers 和 MLX 需要以 `multimodal=True` 加载视觉模型，SGLang 需要加载支持图片的模型。

```python
from llm2jev import Box, JevRequest, Noul

request = JevRequest(
    model=model_path,
    state={"type": "multimodal", "content": [
        {"type": "text", "text": "目标图片："},
        {"type": "image_url", "image_url": {"url": "file:///data/photo.png"}},
    ]},
    questions={
        "mug_visible": Noul(instructions="目标图片中有咖啡杯吗？"),
        "mug": Box(instructions="目标图片中靠近水槽的咖啡杯"),
    },
)
```

通过上面的 `LLM2Jev.evaluate(request)` 或现有 `Client` / `AsyncClient` 的 `systemone.create` 提交。`Box` 只接受 `instructions`，不接受 `criteria`；图片必须位于共享 `state` 或该问题的 `instructions` 中。可在 `instructions` 中使用图文内容附带目标参考图；多图时请明确说明搜索哪张图、哪张图仅为参考。每个问题只返回一个框。

响应示例（数值仅用于说明格式）：

```json
{
  "type": "box",
  "box": {"top": 0.202, "bottom": 0.904, "left": 0.101, "right": 0.803},
  "confidence": 0.73
}
```

Python 中可通过 `response.boxes["mug"].box` 访问四条边。`left/right` 是图宽的比例，`top/bottom` 是图高的比例，原点在左上角，且 `left < right`、`top < bottom`。转换回像素：

```python
from PIL import Image, ImageOps

with Image.open("photo.png") as original:
    image = ImageOps.exif_transpose(original)
    edges = response.boxes["mug"].box
    pixels = (
        round(edges["left"] * image.width),
        round(edges["top"] * image.height),
        round(edges["right"] * image.width),
        round(edges["bottom"] * image.height),
    )
    image.crop(pixels).save("mug.png")
```

请使用与模型实际读取方向一致的图片；带 EXIF 方向的照片建议先转正像素，再提交同一文件和裁剪。

Box 关闭 thinking，贪心生成最多 128 个 token 的坐标 JSON。`confidence` 取实际生成的坐标 token 概率的几何平均：

```text
confidence = exp(mean(log P(token_t | 图片、指令、此前生成的 token)))
```

只统计与四个坐标数字重叠的 token，不统计 JSON 字段名、纯标点、标签、Markdown 代码块和 EOS；数字与标点合在同一个 token 时，该 token 计入一次。概率取自贪心选择前模型的完整词表分布，因此不会因为贪心解码就恒为 1。一个坐标被拆成多个 token 时，每个 token 分别参与平均。此分数表示生成确定程度，尚未校准为定位正确率，也不等同于 TypeSafe 的 confidence 算法。Box 不再追加 yes/no 验证。Box 用量包含单次生成的输入及全部输出 token（包括格式和 EOS）；Choice、Score、Noul 仍只做 prefill 评分，混合请求会累加这些评分用量。

模型输出接受四整数数组 `[x_min,y_min,x_max,y_max]`、含 `bbox_2d` 的对象或单元素对象数组，可有 Markdown 代码块；四坐标除以 1000 后返回。输出格式错误、返回多个框、坐标越界、零面积、边界反向或生成被截断时，整次请求失败；不会裁剪、交换边界或补造框。Python 抛出 `ValueError`，HTTP 返回 422。

Box 没有 `null` 或“未找到”答案。目标可能缺失时，像上例一样附加 Noul，应用自行设定阈值来决定是否采用框。Qwen3.5-4B 的合成图片冒烟已跑通混合请求，但部分宽高比下存在明显定位偏差；请用自己的图片评估定位质量和阈值。

图片继续使用本文的显式 `multimodal` 内容块。本实现暂不支持 TypeSafe 的顶层 `assets`、字符串图片引用、multipart 上传和 `criteria` 图片，也不要求其 `TypeSafe-Danger-Zone-Images` 请求头；图像尺寸和格式能力由本地后端及模型的 processor 决定。
