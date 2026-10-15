# Using multimodal data

[简体中文](../zh/multimodal.md) · [Back to README](../../README.md)

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

with TransformersBackend(model_path, multimodal=True, dtype="bfloat16") as backend:
    response = LLM2Jev(backend=backend).evaluate(request)
    print(response.json)
```

## MLX backend

Set `multimodal=True` when creating the backend:

```python
from llm2jev import LLM2Jev, MLXBackend

with MLXBackend(model_path, multimodal=True, batch_size=8) as backend:
    response = LLM2Jev(backend=backend).evaluate(request)
    print(response.json)
```


## HTTP API calls

Start SGLang following the [Usage guide](usage.md#online-http-service).


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

## Box: locate a target in an image

`Box` returns one bounding box and can share a request with `Choice`, `Score`, and `Noul`. This initial implementation uses Qwen3.5's normalized 0–1000 coordinate convention; other models must follow the same coordinate and JSON output instructions. Transformers and MLX require a vision model loaded with `multimodal=True`; SGLang requires an image-capable model.

```python
from llm2jev import Box, JevRequest, Noul

request = JevRequest(
    model=model_path,
    state={"type": "multimodal", "content": [
        {"type": "text", "text": "Target image:"},
        {"type": "image_url", "image_url": {"url": "file:///data/photo.png"}},
    ]},
    questions={
        "mug_visible": Noul(instructions="Is there a coffee mug in the target image?"),
        "mug": Box(instructions="the coffee mug nearest the sink in the target image"),
    },
)
```

Submit through `LLM2Jev.evaluate(request)` as above, or the existing `Client` / `AsyncClient` `systemone.create` method. `Box` takes `instructions` and no `criteria`. An image must appear in the shared `state` or that question's `instructions`. Instructions can contain multimodal content with reference images of the target. With multiple images, identify which one to search and which are references. Each question returns one box.

Example answer (numbers illustrate the format):

```json
{
  "type": "box",
  "box": {"top": 0.202, "bottom": 0.904, "left": 0.101, "right": 0.803},
  "confidence": 0.73
}
```

In Python, access edges through `response.boxes["mug"].box`. `left/right` are fractions of image width and `top/bottom` are fractions of image height, measured from the top-left corner, with `left < right` and `top < bottom`. Convert to pixels as follows:

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

Use the same image orientation as the model input. For photos with EXIF orientation, rotate the pixels upright before submitting and cropping the same file.

Box greedily generates coordinate JSON with thinking disabled and a 128-token limit. `confidence` is the geometric mean of the probabilities assigned to the actual generated coordinate tokens:

```text
confidence = exp(mean(log P(token_t | image, instructions, preceding tokens)))
```

Only tokens overlapping the four coordinate numbers participate. JSON keys, punctuation-only tokens, labels, Markdown fences, and EOS are excluded; a token combining digits and punctuation participates once. Probabilities come from the model's full-vocabulary distribution before greedy selection, so greedy decoding does not force confidence to 1. A coordinate split across several tokens contributes each token separately. This measures generation certainty and is uncalibrated as localization accuracy; it does not reproduce TypeSafe's confidence algorithm. Box performs no additional yes/no verification. Box usage includes the single generation input and all generated output tokens, including formatting and EOS. Choice, Score, and Noul continue to use prefill scoring only; mixed requests add their scoring usage.

Accepted outputs are a four-integer array `[x_min,y_min,x_max,y_max]`, an object containing `bbox_2d`, or a singleton list of such an object, optionally in a Markdown code fence. Coordinates are divided by 1000. Invalid JSON, multiple boxes, out-of-range coordinates, zero-area or reversed bounds, and truncated generation fail the entire request; no bounds are clamped, swapped, or fabricated. Python raises `ValueError`; HTTP returns 422.

Box has no `null` or "not found" answer. If the target may be absent, add a Noul as above and choose your own threshold for using the box. Qwen3.5-4B synthetic-image smoke tests exercised mixed requests, but some aspect ratios showed substantial localization errors. Evaluate localization quality and thresholds on your own images.

Images continue to use this guide's explicit `multimodal` blocks. TypeSafe's top-level `assets`, string image references, multipart uploads, and images in `criteria` are not supported; its `TypeSafe-Danger-Zone-Images` header is not required. Image dimensions and formats depend on the local backend and model processor.
