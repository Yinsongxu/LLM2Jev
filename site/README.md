# Project website / 项目网站

The website uses VitePress and deploys to **https://yinsongxu.github.io/LLM2Jev/** through GitHub Pages.

## Develop locally

Use Node.js 22 or newer:

```bash
cd site
npm ci
npm run dev
```

Build and preview the production site:

```bash
npm run build
npm run preview
```

Open the `/LLM2Jev/` URL printed by the server. The base path matches GitHub's project Pages URL.

## Content

- English documentation: `docs/en/`.
- Chinese documentation: `docs/zh/`. Translated pages use the same filename, so the language menu can switch to the corresponding page.
- News: update the News sections in `README.md` and `README_zh.md`; `prepare.mjs` generates the website's news pages and homepage updates from them.
- Homepages: `site/pages/` and `site/.vitepress/theme/ProjectHome.vue`.
- Navigation, search, and deployment base path: `site/.vitepress/config.mts`.
- The homepage release badge and latest-release navigation read the package version from `pyproject.toml`. Add matching release notes under both `docs/en/releases/` and `docs/zh/releases/` when bumping the version.

`prepare.mjs` copies documentation into the ignored `site/content/` directory and adjusts links for the website. Edit the source documents, not the generated copies. Re-run the dev command after changing source documentation or News. Fonts and project media are served locally; the published site needs no application server or API key.

## GitHub Pages

In the repository's **Settings → Pages → Build and deployment**, set **Source** to **GitHub Actions**. Push the website changes to `main`, or run **Deploy project website** from the Actions tab. The workflow builds the site and uploads only `site/.vitepress/dist/`.

Publishing the website does not create a GitHub Release or publish the Python package. Release notes live under `docs/en/releases/` and `docs/zh/releases/`.

## 中文维护说明

文档分别维护在 `docs/en/`、`docs/zh/`，中英文页面使用相同文件名。News 直接读取两份 README，不需要额外编辑网站副本。修改文档或 News 后重新运行 `npm run dev`；构建命令会自动准备最新内容。

首页版本徽标和最新版本导航读取 `pyproject.toml` 中的包版本。升级版本时，请在 `docs/en/releases/` 和 `docs/zh/releases/` 中添加对应发布说明；修改包版本也会触发网站部署。

首次部署在仓库 **Settings → Pages** 中选择 **GitHub Actions**，再推送网站改动。之后，相关文件推送到 `main` 会自动构建和发布。网站发布与 GitHub Release、Python 包发布相互独立。
