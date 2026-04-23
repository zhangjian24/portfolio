# AGENTS.md - Portfolio

## Commands

```bash
pnpm dev          # Dev server (hot reload)
pnpm build        # Build to dist/
pnpm preview      # Preview build
pnpm astro check  # Type check
pnpm md2docx <file>  # Convert MD/MDX to DOCX
```

## Paths

- **Images**: `public/image/` (NOT `src/assets/`)
- **Site config**: `src/config.ts` (SITE, SITE_TITLE, SITE_DESCRIPTION)
- **Content schemas**: `src/content/config.ts` (blog, store collections)
- **Layouts**: `src/layouts/*.astro`
- **Components**: `src/components/*.astro`
- **Pages**: `src/pages/` (file-based routing)

## Content Collections

Schema in `src/content/config.ts`:
- `blog`: title, description, pubDate, heroImage, badge, tags
- `store`: title, description, pricing, oldPricing, checkoutUrl, custom_link

Add blog posts as `.md` or `.mdx` in `src/content/blog/`.

## Layouts

- `BaseLayout`: Standard pages (includes SideBar, Header, Footer)
- `PostLayout`: Blog articles (wraps content in `prose` class)
- `StoreItemLayout`: Product pages
- `CVLayout` / `CVLayout_en`: Resume pages (independent)

## Image Usage

```astro
<!-- For images in public/image/ - direct path reference -->
<img src="/image/logo.png" alt="Logo" />

<!-- For Astro optimization - import from astro:assets -->
import { Image } from 'astro:assets';
```

## MDX Features

- Code highlighting: built-in via astro-expressive-code
- Mermaid diagrams: use ` ```mermaid ` code blocks
- Supported diagram types: flowchart, stateDiagram-v2, sequenceDiagram, pie, er, gantt, classDiagram
- **NOT supported**: quadrantChart, tree, mindmap, gitgraph, xychart
- TOC: auto-generated from h1-h3 headings
- Styling: use `prose prose-lg` classes in PostLayout

## Important URLs

- RSS: `/rss.xml`
- Sitemap: `/sitemap-index.xml`
