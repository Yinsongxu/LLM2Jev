  <p align="center">
    <img src="assets/llm2jev-banner.jpeg" alt="LLM2Jev" width="100%">
  </p>

<div align="center">

# 🧠 LLM2Jev: Turn LLMs into Jev-Style Decision Models
<br/>

[![Python](https://img.shields.io/badge/python-3.12%2B-blue?style=flat-square)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square)](LICENSE)
[![Jev API](https://img.shields.io/badge/API-%2Fv1%2Fsystemone%20compatible-orange?style=flat-square)](docs/usage.md)
[![Backend](https://img.shields.io/badge/Backend-SGLang/MLX/Transformers-yellow?style=flat-square)](docs/usage.md)

[简体中文](README_zh.md)

**Turn local language models into Jev-style structured decision models. Get results from text and images with prefill alone—no token-by-token decoding required.**

</div>

> LLM2Jev is an independent open-source project. It is not affiliated with or endorsed by Jev or TypeSafe.


## 📰 News

- **September 23** - **[MLX backend](docs/usage.md#offline-python-api):** added text and image scoring, candidate batching, bounded prefix reuse, and a compatible System One HTTP service.

- **September 22** - **[Multimodal inputs](docs/multimodal.md):** added text-and-image requests for SGLang, Transformers, and the System One HTTP API.
- **September 21** - **[Web and Snake demos](#demos):** added interactive examples for composing mixed questions and model-driven decisions.
- **September 21** - **Prefix reuse on cold requests:** added staged candidate submission for reusing SGLang's Radix Cache, with [architecture](docs/request-to-model.md), [usage](docs/shared-prefix-cache.md), and [benchmark](docs/shared-prefix-benchmarks.md) documentation.
- **September 20** - **SGLang and System One API:** added the SGLang scoring backend and a compatible [`POST /v1/systemone`](docs/usage.md#online-http-service) endpoint.

## ✨ Key Features

- **Broad backend support:** run local text and vision language models with SGLang, Transformers, or MLX on Apple Silicon through the Python API or HTTP service.
- **Prefill only:** compute probabilities from logits during prefill and assemble results directly, without token-by-token decoding.
- **Apple Silicon:** run local text and image models through the [MLX backend](docs/usage.md#offline-python-api), with quantized models, batching, prefix reuse and HTTP serving.
- **Multimodal inputs:** combine text and images in `state` or `instructions`, with support for SGLang, Transformers and MLX-VLM.
- **Order-independent options:** evaluate each Choice candidate independently, so reordering options does not introduce a positional preference or change their scores.
- **Prefix reuse on cold requests:** stage candidate submissions to reuse SGLang's Radix Cache within a single request, including a first request with no relevant cached prefix.

Candidates share `state`, and candidates for the same question also share its `instructions`. The SGLang backend first scores a real `criteria` candidate to establish the prefix cache, then submits candidates that can reuse it. Each candidate is scored once, reducing repeated computation for long inputs with many candidates. The MLX backend explicitly prefills shared prefixes before scoring candidate suffixes.

![Staged candidate scoring reuses state and question instructions through SGLang Radix Cache.](assets/shared-prefix-stages.svg)

Both backends preserve the same binary scoring interface. See the [Usage guide](docs/usage.md) for details.

Learn how it works: [From Jev Request to LLM Request](docs/request-to-model.md) → [Shared-prefix design](docs/shared-prefix-cache.md).

## 🚀 Quick Start

On Linux with a supported NVIDIA GPU, run a local model through SGLang:

```bash
git clone https://github.com/Yinsongxu/LLM2Jev.git
cd LLM2Jev
uv sync --extra sglang
source .venv/bin/activate
python examples/sglang_inference.py --model-path /path/to/model
```

The example submits Choice, Score, and Noul questions and prints the response as JSON.
Replace `/path/to/model` with a local Hugging Face-compatible causal language model directory.

For image serving, see [Multimodal inputs](docs/multimodal.md).

## 📦 Installation

See [Installation](docs/installation.md) for environment requirements, SGLang, Transformers, and MLX dependencies, and uv or pip installation.

## 📖 Getting Started

See the [Usage guide](docs/usage.md) for complete examples:

- [Offline: Python API](docs/usage.md#offline-python-api)
- [MLX backend on Apple Silicon](docs/usage.md#offline-python-api)
- [Online: HTTP service](docs/usage.md#online-http-service)
- [Choosing between `staged` and `all`](docs/usage.md#choosing-between-staged-and-all)
- [Multimodal inputs](docs/multimodal.md)

<a id="demos"></a>

## 🎮 Demos

<table>
  <tr>
    <td align="center" valign="middle" width="67%"><img src="assets/web-demo.gif" alt="LLM2Jev web demo" width="100%"></td>
    <td align="center" valign="middle" width="33%"><img src="assets/snake.gif" alt="LLM2Jev Snake demo" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><a href="demos/web/README.md"><strong>Web demo</strong></a></td>
    <td align="center"><a href="demos/snake.py"><strong>Snake demo</strong></a></td>
  </tr>
  <tr>
    <td align="center" valign="middle"><img src="assets/mujoco.gif" alt="MuJoCo pick-and-place demo" width="100%"><br><a href="demos/pick_place/README.md"><strong>MuJoCo pick-and-place demo</strong></a></td>
    <td></td>
  </tr>
</table>


## 📊 Benchmarks

See [Performance benchmarks](docs/shared-prefix-benchmarks.md) for the Qwen3-1.7B / RTX 5090 measurements, test conditions, and comparison of `staged` and `all` across cold and warm caches. Gains depend on input length, candidate count, and cache state.

## 🗺️ Roadmap

- [ ] More benchmarks across model sizes, datasets, and workloads, covering decision quality, latency, and throughput.
- [x] An interactive web demo for submitting questions and inspecting probabilities.
- [x] Initial local-image support for Transformers and SGLang.
- [ ] More multimodal tasks and demos.

## 🧪 Tests

```bash
python -m unittest discover -s tests -v
```

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
