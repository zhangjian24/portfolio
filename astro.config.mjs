import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwind from "@astrojs/tailwind";
import rehypeToc from '@jsdevtools/rehype-toc'
import rehypeSlug from 'rehype-slug'
import rehypeAutolinkHeadings from 'rehype-autolink-headings'

import expressiveCode from 'astro-expressive-code';
import { pluginLineNumbers } from '@expressive-code/plugin-line-numbers'

// https://astro.build/config
export default defineConfig({
  site: 'https://portfolio.jianzhang.site',
  integrations: [expressiveCode({
    plugins: [pluginLineNumbers()],
  }), mdx(), sitemap(), tailwind(), ],
  markdown: {
		rehypePlugins: [
      rehypeSlug,
      [rehypeAutolinkHeadings, { behavior: 'prepend' }],
			[rehypeToc, { headings: ['h1','h2', 'h3'] }],
		],
	},
});