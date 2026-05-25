# AGENTS.md - Portfolio

## Commands

```bash
pnpm dev              # Dev server (hot reload, port 4321)
pnpm start            # Same as dev
pnpm build            # Build to dist/
pnpm preview          # Preview production build
pnpm astro check      # Type check (Astro + TS)
pnpm md2docx <file>   # Convert MD/MDX to DOCX
```

## Setup

- `pnpm install` runs `postinstall`: `playwright install` (browser req for asciinema)
- `.npmrc` sets `shamefully-hoist=true` and sharp Taobao mirrors (China-friendly)

## TypeScript Path Aliases

```json
"@components/*" -> "src/components/*"
"@layouts/*"    -> "src/layouts/*"
```

## Paths

- **Images**: `public/image/` (NOT `src/assets/`; use direct `<img src="/image/...">` or `import { Image } from 'astro:assets'`)
- **Site config**: `src/config.ts` (SITE, SITE_TITLE, SITE_DESCRIPTION, TRANSITION_API)
- **Content schemas**: `src/content/config.ts` (blog, store collections)
- **Layouts**: `src/layouts/*.astro`
- **Components**: `src/components/*.astro`
- **CV components**: `src/components/cv/`
- **Pages**: `src/pages/` (file-based routing)

## Content Collections

Schema in `src/content/config.ts`:

- `blog`: title, description, pubDate, updatedDate?, heroImage?, badge?, tags[]
- `store`: title, description, custom_link_label, custom_link?, updatedDate, pricing?, oldPricing?, badge?, checkoutUrl?, heroImage?

Add blog posts as `.md` or `.mdx` in `src/content/blog/`.

## Layouts

- `BaseLayout`: Standard pages (includes SideBar, Header, Footer)
- `PostLayout`: Blog articles (wraps content in `prose` class, uses `Image` from astro:assets)
- `StoreItemLayout`: Product pages
- `CVLayout` / `CVLayout_en`: Resume pages (independent, bilingual)

## MDX Features

- Code highlighting: built-in via astro-expressive-code (line numbers via plugin)
- Mermaid diagrams: use ` ```mermaid ` code blocks (via rehype-mermaid)
- Supported diagram types: flowchart, stateDiagram-v2, sequenceDiagram, pie, er, gantt, classDiagram
- **NOT supported**: quadrantChart, tree, mindmap, gitgraph, xychart
- TOC: auto-generated from h1-h3 headings (via rehype-toc + rehype-slug)
- Heading anchors: prepended via rehype-autolink-headings
- Styling: use `prose prose-lg` classes in PostLayout

## Important URLs

- RSS: `/rss.xml`
- Sitemap: `/sitemap-index.xml`
- Site: `https://portfolio.jianzhang.site`
