# AGENTS.md - Portfolio Development Guide

## Project Overview

This is an Astro-based personal portfolio website using:
- **Framework**: Astro 4.x with TypeScript
- **Styling**: TailwindCSS with DaisyUI
- **Package Manager**: pnpm
- **Type Checking**: TypeScript (strictNullChecks enabled)

---

## Build Commands

```bash
# Install dependencies
pnpm install

# Start development server (hot reload)
pnpm dev
pnpm start        # alias for astro dev

# Build for production
pnpm build        # outputs to dist/

# Preview production build
pnpm preview

# Type check
pnpm astro check  # runs astro with --check flag
```

**Note**: This project has no test framework configured. To add tests, consider installing Vitest or Playwright.

---

## Code Style Guidelines

### General Rules

- Use 2 spaces for indentation
- Use single quotes for strings in JavaScript/TypeScript
- Use kebab-case for file names (e.g., `side-bar.astro`, not `SideBar.astro`)
- Use PascalCase for component names in code (e.g., `Card.astro`)
- Use trailing commas in arrays and objects
- Maximum line length: 120 characters

### TypeScript

- Enable strictNullChecks in tsconfig (already enabled)
- Use explicit types for function parameters and return types
- Avoid `any` - use `unknown` when type is truly unknown
- Use optional chaining (`?.`) and nullish coalescing (`??`) when appropriate

```typescript
// Good
function getTitle(title: string | undefined): string {
  return title ?? 'Default Title';
}

// Avoid
function getTitle(title: string | undefined): string {
  return title || 'Default Title';
}
```

### Astro Components

- Props interface should be defined using TypeScript
- Frontmatter should come first in the file
- Imports from `astro:assets` for images

```astro
---
interface Props {
  title: string;
  img: ImageMetadata;
  desc?: string;
  url?: string;
  badge?: string;
  tags?: string[];
  target?: string;
}

const { title, img, desc = '', url, badge, tags, target = '_blank' } = Astro.props;
import { Image } from 'astro:assets';
---
```

### Imports

Order imports consistently:
1. Astro built-ins (`astro:assets`, `astro:content`, etc.)
2. External libraries
3. Internal aliases (`@components/*`, `@layouts/*`)
4. Relative imports

```typescript
import { Image } from 'astro:assets';
import type { CollectionEntry } from 'astro:content';
import BaseHead from '../components/BaseHead.astro';
```

### TailwindCSS / DaisyUI

- Use utility classes in component markup
- Use DaisyUI components when available
- Keep custom CSS minimal (use global.css for site-wide styles)

```astro
<!-- Good -->
<div class="card bg-base-100 shadow-xl hover:scale-[102%] transition">
  <div class="card-body">
    <h2 class="card-title">{title}</h2>
  </div>
</div>
```

### File Organization

```
src/
├── components/     # Reusable UI components (.astro)
├── layouts/        # Page layouts (.astro)
├── pages/          # Route-based pages (.astro)
├── content/        # Markdown/MDX content (blog, store)
├── lib/            # Utility functions (.ts)
├── styles/         # Global styles
└── config.ts       # Global site configuration
```

### Naming Conventions

- **Components**: `ComponentName.astro` (PascalCase)
- **Utilities**: `camelCase.ts`
- **Constants**: `SCREAMING_SNAKE_CASE` in config.ts
- **Props**: Use `camelCase` and type with interface

### Error Handling

- Use try/catch for async operations in API routes
- Return appropriate error responses
- Log errors for debugging
- Handle edge cases gracefully with fallback UI

### Content Collections

- Define schemas in `src/content/config.ts`
- Use Zod for validation (Astro built-in)
- Follow frontmatter conventions in markdown files

---

## TypeScript Path Aliases

Configured in `tsconfig.json`:
- `@components/*` -> `src/components/*`
- `@layouts/*` -> `src/layouts/*`

Use these instead of relative paths when possible.

---

## Working with Images

Use Astro's built-in image optimization:

```astro
import { Image } from 'astro:assets';

<Image 
  width={750} 
  height={422} 
  format="webp" 
  src={img} 
  alt={title} 
/>
```

Place images in `src/assets/` or use existing images from `public/`.

---

## Development Workflow

1. Create new pages in `src/pages/`
2. Create reusable components in `src/components/`
3. Add content to `src/content/blog/` or `src/content/store/`
4. Update `src/config.ts` for site-wide configuration
5. Test with `pnpm dev` before building

Build完成后运行 `pnpm build` 并用 `pnpm preview` 验证输出。
