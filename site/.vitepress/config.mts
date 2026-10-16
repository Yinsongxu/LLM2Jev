import { readFileSync } from 'node:fs'
import { defineConfig } from 'vitepress'

const repository = 'https://github.com/Yinsongxu/LLM2Jev'
const project = readFileSync(new URL('../../pyproject.toml', import.meta.url), 'utf8')
const version = project.match(/^version\s*=\s*"([^"]+)"/m)?.[1]
if (!version) throw new Error('Missing project version in pyproject.toml')
const release = `v${version}`
const sidebar = (zh = false) => {
  const prefix = zh ? '/zh/' : '/'
  return [
    {
      text: zh ? '开始使用' : 'Get started',
      items: [
        { text: zh ? '安装' : 'Installation', link: `${prefix}installation` },
        { text: zh ? '使用指南' : 'Usage guide', link: `${prefix}usage` },
        { text: zh ? '图文输入' : 'Text & images', link: `${prefix}multimodal` },
      ],
    },
    {
      text: zh ? '工作原理' : 'Under the hood',
      items: [
        { text: zh ? '从请求到决策' : 'From request to decision', link: `${prefix}request-to-model` },
        { text: zh ? '共享前缀' : 'Shared prefixes', link: `${prefix}shared-prefix-cache` },
      ],
    },
    {
      text: zh ? '评测与动态' : 'Results & updates',
      items: [
        { text: 'JevBench', link: `${prefix}jevbench` },
        { text: zh ? '前缀缓存性能' : 'Prefix-cache benchmarks', link: `${prefix}shared-prefix-benchmarks` },
        { text: zh ? '最新动态' : 'News', link: `${prefix}news` },
        { text: release, link: `${prefix}releases/${release}` },
        { text: 'v0.1.0', link: `${prefix}releases/v0.1.0` },
      ],
    },
  ]
}

export default defineConfig({
  title: 'LLM2Jev',
  titleTemplate: 'LLM2Jev | :title',
  description: 'Give your LLM / MLLM a decision engine. Zero training required.',
  base: '/LLM2Jev/',
  srcDir: './content',
  cleanUrls: true,
  lastUpdated: true,
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/LLM2Jev/favicon.svg' }],
    ['meta', { name: 'theme-color', content: '#faf9f6' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'LLM2Jev — Give your LLM / MLLM a decision engine' }],
  ],
  sitemap: { hostname: 'https://yinsongxu.github.io/LLM2Jev/' },
  locales: {
    root: {
      label: 'English', lang: 'en',
      themeConfig: {
        nav: [{ text: 'Overview', link: '/' }, { text: 'Docs', link: '/installation' }, { text: 'News', link: '/news' }],
        sidebar: sidebar(),
        footer: { message: 'Independent open source. Not affiliated with Jev or TypeSafe.', copyright: 'Released under the Apache 2.0 License.' },
      },
    },
    zh: {
      label: '简体中文', lang: 'zh-CN', link: '/zh/',
      description: '给你的 LLM / MLLM 装上决策引擎。零训练，直接获得决策能力。',
      themeConfig: {
        nav: [{ text: '项目介绍', link: '/zh/' }, { text: '使用文档', link: '/zh/installation' }, { text: '最新动态', link: '/zh/news' }],
        sidebar: sidebar(true),
        outline: { label: '本页目录' },
        docFooter: { prev: '上一篇', next: '下一篇' },
        lastUpdated: { text: '最后更新' },
        sidebarMenuLabel: '文档导航', returnToTopLabel: '回到顶部',
        langMenuLabel: '切换语言', darkModeSwitchLabel: '外观',
        footer: { message: '独立开源项目，与 Jev 或 TypeSafe 无关联。', copyright: '基于 Apache 2.0 许可证发布。' },
      },
    },
  },
  themeConfig: {
    release,
    logo: '/favicon.svg',
    siteTitle: 'LLM2Jev',
    socialLinks: [{ icon: 'github', link: repository }],
    search: {
      provider: 'local',
      options: {
        locales: {
          zh: { translations: {
            button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
            modal: { noResultsText: '没有找到相关结果', resetButtonTitle: '清除搜索',
              footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' } },
          } },
        },
      },
    },
  },
})
