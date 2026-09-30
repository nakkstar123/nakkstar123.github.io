# brain-dump

Learning notes. Built with [Hugo](https://gohugo.io); pushing to `main` deploys to GitHub Pages.

## Writing

```sh
make new name=some-slug     # creates content/notes/some-slug.md with draft: true
make serve                  # preview at http://localhost:1313, drafts included
```

A note is published when you set `draft: false` (or remove the line) and push.

- **Math:** `$...$` inline, `$$...$$` display. It's rendered at build time with KaTeX, so a typo in the LaTeX fails the build instead of shipping broken math. Shared macros live in `data/katex_macros.toml`.
- **Front matter:** `tags`, plus optional `confidence` and `status` shown under the title. When you revise a view, add `lastmod: YYYY-MM-DD` and say what changed; don't silently rewrite.
- **Private material:** `drafts/` is gitignored. Put raw journal entries there. A note with `draft: true` in `content/` does not appear on the site, but its source **is** visible in this public repo.
- **Search engines:** set `noindex = true` in `hugo.toml` to ask crawlers to stay away. The site stays reachable by URL.

`make` uses `.bin/hugo` if it exists (a pinned binary, gitignored), otherwise `hugo` on your PATH. CI pins the version in `.github/workflows/deploy.yml`; keep them in sync.
