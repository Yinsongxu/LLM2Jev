  <p align="center">
    <img src="assets/llm2jev-banner.jpeg" alt="LLM2Jev" width="100%">
  </p>

<div align="center">

# 🧠 LLM2Jev：给你的 LLM / MLLM 装上决策引擎
<br/>

[![Python](https://img.shields.io/badge/python-3.12%2B-blue?style=flat-square)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square)](LICENSE)
[![Jev API](https://img.shields.io/badge/API-%2Fv1%2Fsystemone%20compatible-orange?style=flat-square)](docs/zh/usage.md)
[![Backend](https://img.shields.io/badge/Backend-SGLang|MLX|Transformers-yellow?style=flat-square)](docs/zh/usage.md)

[项目主页](https://yinsongxu.github.io/LLM2Jev/zh/) · [English](README.md)

**零训练，让你的 LLM / MLLM 直接拥有 Jev 风格的决策能力。**

直接使用已有模型权重，从文字与图片中完成分类、评分和动作选择，仅需 prefill 即可得到决策概率。支持 SGLang、Transformers 和 MLX 多种后端，沿用 OpenAI Python SDK 的调用方式；服务端同时支持聊天和决策时，两类请求可共用同一个 client。





</div>


> LLM2Jev 是一个独立的开源项目，与 Jev 或 TypeSafe 没有关联，也未获得其认可或授权。

## 📰 最新动态

- **10月15日** - **[Box 图片定位](docs/zh/multimodal.md#box定位图片中的目标)**：支持使用 Qwen3.5 的 0–1000 归一化坐标定位图片目标。
- **10月11日** - **[v0.1.0](docs/zh/releases/v0.1.0.md)**：零训练，为 LLM / MLLM 加上 Jev 风格的决策能力。支持 SGLang、Transformers、MLX 多种后端，提供扩展自 OpenAI SDK 的 Python client。
- **10月8日** - **[Valen 推箱子评测](examples/valen_sokoban.py)**：新增视觉推箱子 demo 和单步动作评测，基于 [Valen](https://github.com/Liuziyu77/Valen) 项目。

<details>
<summary>展开历史动态</summary>

- **9月26日** - **[JevBench 测评](docs/zh/jevbench.md)**： LLM2Jev 在 231 个公开题目上的准确率和延时，p50延时不到Jev官方的1/10。
- **9月23日** - **[MLX后端](docs/zh/usage.md#offline-python-api)**：新增图文评分、候选批量执行、有界前缀缓存和兼容的 System One HTTP 服务。
- **9月22日** - **[多模态输入](docs/zh/multimodal.md)**：SGLang、Transformers 和 System One HTTP API 均支持图文请求。
- **9月21日** - **[网页和贪吃蛇 demo](#demos)**：新增用于组合多种问题和模型决策的交互式示例。
- **9月21日** - **冷启动前缀复用**：新增分阶段候选提交以复用 SGLang Radix Cache，并提供[架构说明](docs/zh/request-to-model.md)、[使用文档](docs/zh/shared-prefix-cache.md)和[性能测评](docs/zh/shared-prefix-benchmarks.md)。
- **9月20日** - **SGLang 与 System One API**：新增 SGLang 评分后端，以及兼容的 [`POST /v1/systemone`](docs/zh/usage.md#online-http-服务) 接口。

</details>



## ✨ 核心特性

- **无需训练或微调**：直接使用现有模型权重，将 SGLang、Transformers 或 Apple Silicon 上 MLX 支持的模型用于 Jev 风格的决策。
- **聊天与决策共用服务和模型**：使用 SGLang 后端时，聊天接口与 `/v1/systemone` 共用推理引擎和已加载的模型。
- **聊天与决策共用 client**：`Client` 与 `AsyncClient` 在 OpenAI Python SDK 上新增 `systemone.create()`，保留原有接口、认证、超时和重试机制；接口是否可用取决于后端。
- **仅需 prefill**：读取 yes/no logits 并直接组装概率，无需生成答案或解析模型生成的 JSON。
- **图文决策**：在 `state` 或 `instructions` 中组合文字与图片，用于分类、评估与动作选择。
- **一次请求，三类问题**：混合提交 `Choice`、`Score` 与 `Noul`，以类型化 Python 对象或 JSON 获取选项概率、量表评分和是非概率。
- **共享前缀复用**：通过 SGLang Radix Cache 或 MLX 显式前缀预热，减少候选之间的重复计算，冷请求内也能复用。

所有候选共享 `state`，同一道题的候选还共享 `instructions`。SGLang 后端先评分一个真实的 `criteria` 候选来建立前缀缓存，再提交能够复用它的其他候选。每个候选只评分一次，减少长输入、多候选场景中的重复计算。MLX后端会先显式预热共享前缀，再评分候选后缀。

![候选分阶段评分，通过 SGLang Radix Cache 复用 state 和题目的 instructions。](assets/shared-prefix-stages.svg)

了解工作原理：[从 Jev Request 到 LLM Request](docs/zh/request-to-model.md) → [共享前缀设计](docs/zh/shared-prefix-cache.md)。


## 🚀 快速开始

在配有受支持 NVIDIA GPU 的 Linux 环境中，启动一个 SGLang 服务，同时提供聊天与决策接口：

```bash
git clone https://github.com/Yinsongxu/LLM2Jev.git
cd LLM2Jev
uv sync --extra sglang
source .venv/bin/activate
llm2jev-serve --backend sglang --model-path /path/to/model \
  --served-model-name local-model --host 127.0.0.1 --port 30000
```

请将 `/path/to/model` 替换为 SGLang 支持的模型目录。在另一个终端中使用同一 Python 环境，通过一个 client 调用两个接口：

```python
from llm2jev import Client, Choice

client = Client(base_url="http://localhost:30000/v1", api_key="unused")

decision = client.systemone.create(
    model="local-model",
    state="我的信用卡被重复扣款了。",
    questions={"department": Choice(criteria={
        "billing": "扣款和账单", "shipping": "物流配送",
    })},
)
print(decision.choices["department"].probabilities)

chat = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "如何申请退款？"}],
)
print(chat.choices[0].message.content)
```

已启用认证时，请使用服务配置的 API key。异步调用、client 生命周期、offline 推理及 MLX、Transformers 服务见[使用指南](docs/zh/usage.md)。图片服务见[多模态输入](docs/zh/multimodal.md)。

## 📦 安装

环境要求、SGLang、Transformers 与 MLX 依赖，以及 uv、pip 安装方式见[安装指南](docs/zh/installation.md)。

## 📖 使用入门

完整示例见[使用指南](docs/zh/usage.md)：

- [Offline：Python API](docs/zh/usage.md#offline-python-api)
- [Online：HTTP 服务](docs/zh/usage.md#online-http-服务)
- [Online：Python client](docs/zh/usage.md#python-client)
- [`staged` 与 `all` 的选择](docs/zh/usage.md#如何选择-staged-或-all)
- [多模态输入](docs/zh/multimodal.md)

<a id="demos"></a>

## 🎮 Demos

<table>
  <tr>
    <td align="center" valign="middle" width="67%"><img src="assets/web-demo.gif" alt="LLM2Jev 网页 demo" width="100%"></td>
    <td align="center" valign="middle" width="33%"><img src="assets/snake.gif" alt="LLM2Jev 贪吃蛇 demo" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><a href="demos/web/README.md"><strong>网页 demo</strong></a></td>
    <td align="center"><a href="demos/snake.py"><strong>贪吃蛇 demo</strong></a></td>
  </tr>
  <tr>
    <td align="center" valign="middle"><img src="assets/mujoco.gif" alt="MuJoCo 机械臂抓取放置 demo" width="100%"><br><a href="demos/pick_place/README.md"><strong>MuJoCo 机械臂抓取放置 demo</strong></a></td>
    <td align="center" valign="middle"><img src="assets/valen-sokoban.gif" alt="Valen Sokoban 决策 demo" width="100%"><br><a href="examples/valen_sokoban.py"><strong>推箱子 demo</strong></a></td>
  </tr>
</table>


## 📊 性能测评

[性能测评](docs/zh/shared-prefix-benchmarks.md)记录了 Qwen3-1.7B / RTX 5090 的测试条件、实测数据，以及冷缓存和热缓存下 `staged` 与 `all` 的比较。收益取决于输入长度、候选数量和缓存状态。

### JevBench 公开题目准确率

本次测试包含JevBench 中的 231 个公开题目。LLM2Jev 使用 Qwen3.5-4B 测得准确率为 **76.2%**。

![JevBench 公开题目准确率对比](assets/jevbench-accuracy.png)

本次记录的 LLM2Jev 延时为 **P50 48 ms  / P95 346 ms**，Jev 1.13.0 的对应数值为 **P50 652 ms / P95 722 ms**。具体数据和复现命令见 [JevBench 报告](docs/zh/jevbench.md)。

### Valen-Eval-Game

[单步评测脚本](examples/valen_sokoban_single_step.py)对 [Valen](https://github.com/Liuziyu77/Valen) 数据集中的 500 条图像 Sokoban 题计算并列最优动作准确率，使用多模态 `Choice` 流程和 SGLang 后端。

## 🗺️ 近期规划

- [ ] 更多 benchmark：覆盖不同模型规模、数据集和工作负载，评估判断质量、延迟与吞吐量。
- [x] 网页 demo：交互式提交问题并查看概率结果。
- [x] Transformers 和 SGLang 的首版本地图片支持。
- [x] 更多多模态任务及 demo。

## 🧪 测试

```bash
python -m unittest discover -s tests -v
```

## 📄 许可证

本项目基于 [Apache License 2.0](LICENSE) 发布。
