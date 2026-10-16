<script setup lang="ts">
import { computed, ref } from 'vue'
import { useData, withBase } from 'vitepress'
import news from './news.json'

const { lang, site } = useData()
const release = computed(() => site.value.themeConfig.release)
const zh = computed(() => lang.value.startsWith('zh'))
const page = (name: string) => withBase(`${zh.value ? '/zh/' : '/'}${name}`)
const active = ref('Choice')
const copied = ref(false)
const types = ['Choice', 'Score', 'Noul']
const text = computed(() => zh.value ? {
  release: `${release.value} · 现已就绪`, eyebrow: '为你的模型，加上决策力',
  title: ['给你的', 'LLM / MLLM', '装上决策引擎'],
  intro: '零训练，直接获得 Jev 风格的决策能力。从文字与图片中完成分类、评分和动作选择，用 Box 定位图片目标。',
  start: '开始使用', github: '查看 GitHub', backend: '选择你熟悉的后端',
  evidence: '输入证据', input: '我的信用卡被重复扣款了。', prompt: '哪个部门应当处理？',
  model: '已有模型权重', prefill: '一次前向，读取 yes / no logits', output: '结构化决策',
  billing: '账单问题', shipping: '物流配送', example: '示例概率 · 非实时推理',
  featureLabel: '更少的部署，直接的决策', featureTitle: '把已有模型的能力用起来。',
  features: [
    { number: '01', title: '零训练，直接接入', detail: '保留现有模型权重，用 Choice、Score、Noul 获取结构化决策和概率，用 Box 返回图片目标的归一化边界框。', tag: 'NO FINE-TUNING' },
    { number: '02', title: '决策仅需 prefill', detail: 'Choice、Score、Noul 从 yes/no logits 直接计算概率，共享前缀进一步减少候选间的重复计算；Box 通过坐标生成完成定位。', tag: 'PREFILL SCORING' },
    { number: '03', title: '聊天与决策，共用 client', detail: '沿用 OpenAI SDK 的调用方式。SGLang 部署下，聊天与决策还共用服务、推理引擎和模型。', tag: 'OPENAI SDK EXTENSION' },
  ],
  codeLabel: '熟悉的调用方式，新的决策能力', codeTitle: '几行代码，让模型做决定。',
  codeIntro: '构造一个 client，提交一个问题，拿到答案与概率。支持同步、异步以及图文输入。',
  codeLink: '阅读完整使用指南', command: '先安装你需要的后端', copy: '复制代码', copied: '已复制',
  demosLabel: '从屏幕到真实任务', demosTitle: '看见、判断、行动。', demosIntro: '同一套决策接口，连接不同的任务。',
  demos: [
    { title: '视觉推箱子', subtitle: '根据棋盘图像，选择下一步动作。', image: 'valen-sokoban.gif', link: 'examples/valen_sokoban.py', tag: 'VISION → ACTION' },
    { title: '机械臂抓取', subtitle: '用结构化决策驱动 MuJoCo 控制。', image: 'mujoco.gif', link: 'demos/pick_place/README.md', tag: 'DECISION → CONTROL' },
    { title: '贪吃蛇', subtitle: '每一步方向，都是一次 Choice 决策。', image: 'snake.gif', link: 'demos/snake.py', tag: 'STATE → CHOICE' },
  ],
  newsLabel: '持续构建', newsTitle: '最新动态', allNews: '查看全部动态',
  closing: '让你的模型，开始做决策。', closingDetail: '开源 · Apache 2.0 · SGLang / Transformers / MLX',
} : {
  release: `${release.value} · Ready to explore`, eyebrow: 'YOUR MODEL. NEW POSSIBILITIES.',
  title: ['Give your', 'LLM / MLLM', 'a decision engine.'],
  intro: 'Jev-style decisions. Zero training. Classify, score, and choose actions from text and images. Locate image targets with Box.',
  start: 'Get started', github: 'Explore on GitHub', backend: 'RUN WITH THE BACKEND YOU KNOW',
  evidence: 'Evidence', input: 'My card was charged twice.', prompt: 'Which department should handle this?',
  model: 'Your existing model', prefill: 'Prefill → yes / no logits', output: 'Structured decision',
  billing: 'Billing', shipping: 'Shipping', example: 'Illustrative probabilities · not live inference',
  featureLabel: 'LESS SETUP. MORE DECISIONS.', featureTitle: 'Put your existing model to work.',
  features: [
    { number: '01', title: 'Zero training. Ready to adapt.', detail: 'Keep your model weights. Get structured decisions and probabilities with Choice, Score, and Noul, and normalized image bounding boxes with Box.', tag: 'NO FINE-TUNING' },
    { number: '02', title: 'Decisions at prefill.', detail: 'Choice, Score, and Noul read yes/no logits directly, with shared prefixes reducing repeated computation. Box generates coordinates to locate targets.', tag: 'PREFILL SCORING' },
    { number: '03', title: 'Chat & decisions. Same client.', detail: 'Use the familiar OpenAI SDK interface. With SGLang, chat and decisions also share a service, inference engine, and loaded model.', tag: 'OPENAI SDK EXTENSION' },
  ],
  codeLabel: 'FAMILIAR INTERFACE. NEW CAPABILITIES.', codeTitle: 'A few lines. A clear decision.',
  codeIntro: 'Create a client, ask a question, get an answer with probabilities. Sync, async, text, and images are all supported.',
  codeLink: 'Read the usage guide', command: 'Install your preferred backend first', copy: 'Copy code', copied: 'Copied',
  demosLabel: 'FROM INPUT TO IMPACT', demosTitle: 'See. Decide. Act.', demosIntro: 'One decision interface. Different worlds to act in.',
  demos: [
    { title: 'Visual Sokoban', subtitle: 'Read the board. Choose the next move.', image: 'valen-sokoban.gif', link: 'examples/valen_sokoban.py', tag: 'VISION → ACTION' },
    { title: 'Robotic pick & place', subtitle: 'Structured decisions meet MuJoCo control.', image: 'mujoco.gif', link: 'demos/pick_place/README.md', tag: 'DECISION → CONTROL' },
    { title: 'Snake', subtitle: 'Every direction is a Choice decision.', image: 'snake.gif', link: 'demos/snake.py', tag: 'STATE → CHOICE' },
  ],
  newsLabel: 'BUILDING IN THE OPEN', newsTitle: 'Latest updates', allNews: 'All project news',
  closing: 'Your model. Ready to decide.', closingDetail: 'Open source · Apache 2.0 · SGLang / Transformers / MLX',
})

const request = computed(() => ({
  Choice: 'Choice(criteria={\n            "billing": "Charges and billing",\n            "shipping": "Delivery",\n        })',
  Score: 'Score(\n            instructions="How urgent is this issue?",\n            criteria=["Low", "Medium", "High"],\n        )',
  Noul: 'Noul(\n            instructions="Is this a billing issue?",\n        )',
}[active.value]))
const code = computed(() => `from llm2jev import Client, ${active.value}

client = Client(
    base_url="http://localhost:30000/v1",
    api_key="unused",
)

response = client.systemone.create(
    model="local-model",
    state="My card was charged twice.",
    questions={
        "decision": ${request.value},
    },
)
print(response.answers["decision"])`)
const result = computed(() => active.value === 'Choice'
  ? { type: 'choice', choice: 'billing', probabilities: { billing: 0.92, shipping: 0.08 } }
  : active.value === 'Score'
    ? { type: 'score', score: 1.72, probabilities: { '0': 0.04, '1': 0.2, '2': 0.76 } }
    : { type: 'noul', noul: 0.96 })
const prompt = computed(() => active.value === 'Choice' ? text.value.prompt
  : active.value === 'Score' ? (zh.value ? '这个问题有多紧急？' : 'How urgent is this issue?')
    : (zh.value ? '这是账单问题吗？' : 'Is this a billing issue?'))
const updates = computed(() => news[zh.value ? 'zh' : 'en'].slice(0, 3))

async function copyCode() {
  try {
    await navigator.clipboard.writeText(code.value)
    copied.value = true
    setTimeout(() => { copied.value = false }, 2000)
  } catch {
    // The selectable code remains available if clipboard permission is denied.
  }
}
</script>

<template>
  <div class="project-home">
    <section class="hero section-wrap">
      <div class="hero-copy">
        <a class="release-pill" :href="page(`releases/${release}`)"><span class="status-dot" />{{ text.release }}<span aria-hidden="true">↗</span></a>
        <p class="eyebrow">{{ text.eyebrow }}</p>
        <h1><span>{{ text.title[0] }}</span><span class="model-title">{{ text.title[1] }}</span><span>{{ text.title[2] }}</span></h1>
        <p class="hero-description">{{ text.intro }}</p>
        <div class="hero-actions">
          <a class="button button-primary" :href="page('installation')">{{ text.start }} <span aria-hidden="true">↗</span></a>
          <a class="button button-secondary" href="https://github.com/Yinsongxu/LLM2Jev">{{ text.github }} <span aria-hidden="true">→</span></a>
        </div>
      </div>

      <div class="decision-lab" aria-label="Decision example">
        <div class="lab-top"><span class="lab-mark">L2J<span> / </span>DECISION FLOW</span><span class="lab-live"><span class="status-dot" />PREFILL ONLY</span></div>
        <div class="evidence-card"><span class="card-kicker"><span aria-hidden="true">↳</span> {{ text.evidence }}</span><p>“{{ text.input }}”</p><span class="evidence-prompt">{{ prompt }}</span></div>
        <div class="flow-connector"><span /><span aria-hidden="true">↓</span><span /></div>
        <div class="engine-card"><div class="engine-icon" aria-hidden="true"><i /><i /><i /><i /></div><div><strong>{{ text.model }}</strong><p>{{ text.prefill }}</p></div><span class="engine-badge">0<span>TRAINING</span></span></div>
        <div class="flow-connector"><span /><span aria-hidden="true">↓</span><span /></div>
        <div class="result-card">
          <div class="result-top"><span class="card-kicker">{{ text.output }}</span><div class="type-tabs" aria-label="Question type"><button v-for="type in types" :key="type" :class="{ active: active === type }" :aria-pressed="active === type" @click="active = type">{{ type }}</button></div></div>
          <div v-if="active === 'Choice'" class="probabilities"><div class="probability"><span>{{ text.billing }}</span><div><i style="width: 92%" /></div><strong>0.92</strong></div><div class="probability muted"><span>{{ text.shipping }}</span><div><i style="width: 8%" /></div><strong>0.08</strong></div></div>
          <div v-else-if="active === 'Score'" class="score-example"><strong>1.72<span>/ 2</span></strong><div class="score-track"><i style="width: 86%" /></div><span>Low → Medium → High</span></div>
          <div v-else class="noul-example"><span>P(yes)</span><strong>0.96</strong><span class="yes-badge">YES</span></div>
        </div>
        <p class="lab-caption">{{ text.example }}</p>
        <div class="lab-corner" aria-hidden="true">✳</div>
      </div>
    </section>

    <div class="backend-strip section-wrap"><span>{{ text.backend }}</span><div><strong>SGLang</strong><span class="backend-cross">+</span><strong>Transformers</strong><span class="backend-cross">+</span><strong>MLX</strong></div><span class="backend-note">TEXT + VISION</span></div>

    <section class="features-section section-wrap">
      <p class="eyebrow">{{ text.featureLabel }}</p><h2>{{ text.featureTitle }}</h2>
      <div class="feature-grid"><article v-for="feature in text.features" :key="feature.number" class="feature"><span class="feature-number">{{ feature.number }} /</span><h3>{{ feature.title }}</h3><p>{{ feature.detail }}</p><span class="feature-tag">{{ feature.tag }}</span></article></div>
    </section>

    <section class="code-section section-wrap">
      <div class="code-intro"><p class="eyebrow">{{ text.codeLabel }}</p><h2>{{ text.codeTitle }}</h2><p>{{ text.codeIntro }}</p><a class="text-link" :href="page('usage#python-client')">{{ text.codeLink }} <span aria-hidden="true">↗</span></a><div class="install-command"><span>{{ text.command }}</span><code>uv sync --extra sglang</code></div></div>
      <div class="code-window"><div class="code-window-top"><div class="window-dots" aria-hidden="true"><i /><i /><i /></div><span>decision.py</span><button @click="copyCode" :aria-label="text.copy">{{ copied ? text.copied : text.copy }} <span aria-hidden="true">{{ copied ? '✓' : '⧉' }}</span></button></div><div class="code-type-tabs" aria-label="Code example type"><button v-for="type in types" :key="type" :class="{ active: active === type }" :aria-pressed="active === type" @click="active = type">{{ type }}</button></div><pre><code>{{ code }}</code></pre><div class="code-output"><span>↳ OUTPUT · {{ text.example }}</span><pre>{{ JSON.stringify(result, null, 2) }}</pre></div></div>
    </section>

    <section id="demos" class="demos-section section-wrap">
      <div class="section-heading"><div><p class="eyebrow">{{ text.demosLabel }}</p><h2>{{ text.demosTitle }}</h2></div><p>{{ text.demosIntro }}</p></div>
      <div class="demo-grid"><a v-for="demo in text.demos" :key="demo.image" class="demo" :href="`https://github.com/Yinsongxu/LLM2Jev/blob/main/${demo.link}`"><div class="demo-image"><img :src="withBase(`/assets/${demo.image}`)" :alt="demo.title" loading="lazy" width="520" height="325" /><span>{{ demo.tag }}</span></div><div class="demo-title"><h3>{{ demo.title }}</h3><span aria-hidden="true">↗</span></div><p>{{ demo.subtitle }}</p></a></div>
    </section>

    <section class="news-section section-wrap"><div class="section-heading"><div><p class="eyebrow">{{ text.newsLabel }}</p><h2>{{ text.newsTitle }}</h2></div><a class="text-link" :href="page('news')">{{ text.allNews }} <span aria-hidden="true">↗</span></a></div><div class="news-list"><article v-for="(update, index) in updates" :key="update.date + index"><span class="news-index">0{{ index + 1 }}</span><time>{{ update.date }}</time><div v-html="update.html" /><span class="news-arrow" aria-hidden="true">↗</span></article></div></section>

    <section class="closing section-wrap"><div><p class="eyebrow">LLM2JEV / OPEN SOURCE</p><h2>{{ text.closing }}</h2><p>{{ text.closingDetail }}</p></div><a class="button button-primary" :href="page('installation')">{{ text.start }} <span aria-hidden="true">↗</span></a></section>
  </div>
</template>
