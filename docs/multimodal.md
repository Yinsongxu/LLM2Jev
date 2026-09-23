# Using multimodal data

[简体中文](multimodal_zh.md) · [Back to README](../README.md)

LLM2Jev supports text and images in `state` or a question's `instructions`.

## Supported image sources

Specify the image source in `image_url.url`. The following formats are supported:

| Format | Example |
| --- | --- |
| HTTP(S) URL | `https://example.com/photo.png` |
| Local file URI | `file:///data/photo.png` |
| Local path | `/data/photo.png` or `images/photo.png` |
| Base64 data URL | `data:image/png;base64,...` |

## Building a request

Use an object marked with `type: "multimodal"` in `state` or `instructions`:

```json
{
  "type": "multimodal",
  "content": [
    {"type": "text", "text": "Please inspect this image."},
    {"type": "image_url", "image_url": {"url": "file:///data/photo.png"}}
  ]
}
```

`content` is a nonempty list that can contain multiple text blocks and images, or only text or only images. Items are submitted in list order. A text block's `text` must be a string, and an image's `url` must be a nonempty string. Only the fields shown above are currently supported; `detail`, `file_id`, audio and video are not supported.

Keep the outer marker object. Passing the `content` list directly as `state` or `instructions` serializes it as ordinary JSON text and does not load images. Plain strings, unmarked JSON objects and arrays retain their existing behavior. Images are not interpreted in `criteria`.

The following example places an image in `state` so multiple questions can use it. The backend examples below use the `model_path` and `request` defined here:

```python
from llm2jev import Choice, JevRequest, Noul

model_path = "/path/to/vlm"
request = JevRequest(
    model=model_path,
    state={"type": "multimodal", "content": [
        {"type": "text", "text": "Inspect the image"},
        {"type": "image_url", "image_url": {"url": "file:///data/photo.png"}},
    ]},
    questions={
        "person": Noul(instructions="Is a person visible in the image?"),
        "color": Choice(
            instructions="Which color dominates the image?",
            criteria={"red": "Red", "blue": "Blue", "green": "Green"},
        ),
    },
)
```

You can also place the text-and-image object in a question's `instructions`:

```python
instructions = JevRequest(
    model=model_path,
    state="Inspect the attached image.",
    questions={
        "person": Noul(instructions={
            "type": "multimodal", "content": [
                {"type": "text", "text": "Is a person visible in the image?"},
                {"type": "image_url", "image_url": {"url": "file:///data/photo.png"}},
            ],
        }),
    },
)
```

## SGLang backend

```python
from llm2jev import LLM2Jev, SGLangBackend

if __name__ == "__main__":
    with SGLangBackend(model_path) as backend:
        response = LLM2Jev(backend=backend).evaluate(request)
        print(response.json)
```

## Transformers backend

Set `multimodal=True` when creating the backend:

```python
from llm2jev import LLM2Jev, TransformersBackend

backend = TransformersBackend(model_path, multimodal=True, dtype="bfloat16")
response = LLM2Jev(backend=backend).evaluate(request)
print(response.json)
```

## MLX backend

On Apple Silicon macOS, install the `mlx-vlm` extra and load a local MLX-VLM-compatible
causal model with `multimodal=True`:

```python
from llm2jev import LLM2Jev, MLXBackend

with MLXBackend(model_path, multimodal=True, batch_size=8) as backend:
    response = LLM2Jev(backend=backend).evaluate(request)
    print(response.json)
```

The repository example works with the same image sources and both placements:

```bash
uv run --extra mlx-vlm python examples/multimodal_inference.py \
  --backend mlx --model-path /path/to/mlx-vlm \
  --image /path/to/photo.png --placement state
```

Qwen2-VL and Qwen2.5-VL support image feature reuse within a request, shared prefix
caching with image content digests, chunked prefill and equal-length candidate
batches. Other causal VLMs use complete per-prompt prefills. A changed image cannot
reuse cached KV state for the previous image solely because its path or placeholder
tokens match. The legacy MLX-VLM `BaseImageProcessor` path supports only one image
per prompt and rejects multiple images; other processors follow the model's image
capacity. See [MLX model support](mlx.md#images-and-model-support) for details.

## HTTP API calls

Start SGLang following the [Usage guide](usage.md#system-one-http-api), or start
MLX-VLM on Apple Silicon:

```bash
uv run --extra mlx-vlm --extra server llm2jev-serve \
  --backend mlx --multimodal --model-path /path/to/mlx-vlm \
  --served-model-name local-vlm --port 30000
```

Both services accept the same request structure at `POST /v1/systemone`.
MLX reads `LLM2JEV_API_KEY` when configured; clients then send the Bearer header
shown below. Omit the header when authentication is disabled. Local paths and
`file://` URIs resolve on the machine running the model service.

```bash
curl http://localhost:30000/v1/systemone \
  -H "Authorization: Bearer $LLM2JEV_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-vlm",
    "state": {
      "type": "multimodal",
      "content": [
        {"type": "text", "text": "Inspect the image"},
        {"type": "image_url", "image_url": {"url": "file:///data/photo.png"}}
      ]
    },
    "questions": {
      "person": {
        "type": "noul",
        "instructions": "Is a person visible in the image?"
      },
      "color": {
        "type": "choice",
        "instructions": "Which color dominates the image?",
        "criteria": {"red": "Red", "blue": "Blue", "green": "Green"}
      }
    }
  }'
```

Images can also be placed in a question's `instructions`, as shown earlier.

## Choosing where to place images

| Scenario | Suggested location | Usage |
| --- | --- | --- |
| Multiple questions about the same image | `state` | Include the image once for all questions to share |
| An image relevant to only one question | That question's `instructions` | Attach the image to the question |
| The same question about different images | `instructions` is an option | Keep `state` and the question text fixed; build a separate request for each image |
| Comparing or evaluating multiple images together | The same `content` list | Arrange text and images in the desired order |

Content in `state` appears before the question; content in `instructions` appears within that question. You control the order of text and images in each `content` list. When both locations contain images, the model considers images from both.

For prefix caching, placing reused content earlier is usually a useful starting point. Whether caching is possible or improves speed must be verified with your actual model and requests; image placement alone does not guarantee either.
