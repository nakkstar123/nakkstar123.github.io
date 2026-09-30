---
date: 2026-09-29
tags: ["crypto and ML", "paper notes"]
confidence: "moderate: haven't checked the proof in the appendix"
status: "thinking out loud"
---

This lives in `drafts/`, so it never leaves your machine. To see it on the site, drag it into `notes/` and run `make serve`. Delete it once you've seen how things render.

Links between notes use normal Obsidian links, like [the Notes page](../notes/_index.md). Pasted images land in `attachments/` and just work.

Inline math works with dollar signs: an adversary $\A$ wins with probability $\tfrac12 + \negl(\lambda)$. Display math works too. For $k \getsr \{0,1\}^\lambda$ and a random function $f$,

$$
\Big|\Pr\big[\A^{F_k(\cdot)}(1^\lambda) = 1\big] - \Pr\big[\A^{f(\cdot)}(1^\lambda) = 1\big]\Big| \le \negl(\lambda).
$$

Learning-theory flavored, with the `\E` and `\norm` macros from `data/katex_macros.toml`:

$$
\E_{(x,y)\sim\D}\big[\ell(h(x), y)\big] \le \widehat{L}_S(h) + O\!\left(\sqrt{\frac{d \log(n/d) + \log(1/\delta)}{n}}\right), \qquad \norm{w}_2 \le B.
$$

> Quotes from people whose stances you're tracking go in blockquotes.

Footnotes work for asides.[^1]

[^1]: Like this one.
