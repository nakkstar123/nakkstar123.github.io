# brain-dump

Learning notes, live at <https://nakkstar123.github.io/>. Built with [Hugo](https://gohugo.io); pushing to `main` deploys.

## Writing (Obsidian)

Open `content/` as an Obsidian vault.

- **Write:** new notes land in `drafts/`, which never leaves this machine (gitignored, and Hugo skips it). Insert the `note` template to add the properties (`date`, `tags`, `confidence`, `status`).
- **Publish:** drag the note into `notes/`, then run **Obsidian Git: Commit and sync** from the command palette. The site updates a minute later.
- **Unpublish:** drag it back to `drafts/` and sync.

The filename is the title, and the URL is the filename in lowercase with dashes (`Why PRFs matter.md` becomes `/notes/why-prfs-matter/`). Renaming a published note changes its URL.

- **Math:** `$...$` inline, `$$...$$` display. KaTeX renders it at build time, so a LaTeX typo fails the build instead of shipping. Shared macros are in `data/katex_macros.toml`.
- **Links and images:** normal Obsidian links to other notes and pasted images (saved to `attachments/`) work on the site. A link to a note that's still in `drafts/` shows as plain text.
- **Changing your mind:** add `lastmod: YYYY-MM-DD` and update `status` instead of silently rewriting.
- **Search engines:** set `noindex = true` in `hugo.toml` to ask crawlers to stay away. The site stays reachable by URL.

`make serve` previews at <http://localhost:1313>. `make` uses `.bin/hugo` if present (pinned, gitignored), else `hugo` on your PATH. CI pins the version in `.github/workflows/deploy.yml`.
