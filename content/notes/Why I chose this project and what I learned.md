---
title: "Why I chose this project and what I learned"
date: 2026-11-14
slug: in-context-retrieval-process
draft: false
research: true
tags: [mechanistic-interpretability]
description: "Why I chose a small in-context retrieval investigation, how I used AI, and what I learned."
---

## Why reasoning

I wanted my first interpretability investigation to be about reasoning. My primary motivation is safety. It's fortunate that we can learn so much about models' behavior by reading their chains of thought, but it seems unlikely that reasoning will remain so accessible through human-interpretable written explanations. Even now, we cannot assume that those explanations faithfully reflect the model's computation. I expect there to be optimization pressure to make reasoning more efficient, which could lead models to perform more computation internally or use representations that are harder for us to read. That makes understanding the internal computation seem useful even as models and the ways we interact with them change.

Additionally, my background is in cryptography, and I find it incredible that we have built machines that can do cutting-edge math research (and are now better than me at proving theorems). I think they now show signs of what I'd call *creativity* in solving problems. In math, finding the right intermediate lemma or the right way to set up a proof can require a deep insight, while carrying out the argument once that idea is available can feel quite mechanical. A further motivation for me to study reasoning is a desire to understand whether there is some analogous distinction inside a model, and what it would take to investigate it.

## Why in-context learning

Within reasoning, I am interested in how much a model can learn and adapt from information supplied in its context, without changing its weights. Working on an unfamiliar problem involves using new facts, examples and previous attempts. This seems particularly relevant for tasks that involve a long history of work on a problem or knowledge specific to a person or project. I wonder how much of the adaptation we want from continual learning might be possible through context alone. Could this make models more reliable at a broader range of real-world tasks, including ones with little task-specific training data or outcomes that are difficult to verify?

I am especially interested in the safety implications: even if recursive self-improvement first produces models that excel at AI research but remain unreliable at many other jobs, that gap need not persist if they can learn a new job in context from the same limited instruction and experience available to a human. They could then automate much of human labor without needing continual weight updates or large training datasets for each job.

Admittedly, retrieving an association supplied in a prompt (the focus of my investigation) is a very small part of that in-context reasoning ability. For example, if we tell a model that “the brick is in Box Y” and ask where the brick is, it has to use an assignment provided for this particular problem. We can change the assignment and see whether it follows. This gives us something concrete to study inside the model, with a clear expected answer, and seems like a reasonable first step toward understanding how it uses context.

## Why this paper and this extension

I settled on this smaller retrieval task after exploring a few other ideas. I started by looking at error recovery in interrupted chains of thought, then explored whether supplying a useful intermediate lemma or invariant helped a model solve a problem. From the start, I was very excited by how much implementation I could delegate to coding agents, and tried to move very fast, even delegating most of the work of proposing experiments and interpreting results. I was worried that a small replication would add little when an agent could already produce it.

But I was making experimental choices faster than I was developing the background to assess them, which was hurting my own learning. Compute costs also mattered, since I was funding the experiments myself. Eventually, I decided to work through a smaller published experiment and try a small extension that I could understand in detail.

What interested me most about *Mixing Mechanisms* was its methodology: how the authors distinguished explanations that would otherwise predict the same behavior. On an ordinary prompt, retrieving an association by its position or by an entity's name can produce the same correct answer. The authors construct paired prompts and replace selected internal activations so that the proposed mechanisms predict different answers. As I worked through the paper and its code, this became one of my main takeaways: we can design an intervention under which plausible explanations disagree, then inspect the model's behavior to distinguish them.

The paper's causal model describes how the retrieval mechanisms combine, while leaving open which attention heads or other components implement them. I wondered whether different attention heads might implement different forms of retrieval, with later computation combining their contributions. This seemed like a natural question to try. The simple task, released code and ability to run experiments on a small model made it practical to inspect examples and run repeated interventions. It also gave me a reason to learn NNsight, a library for inspecting and intervening on model activations.

I did not establish the hoped-for separation between heads. I eventually narrowed the investigation to what two particular head outputs could contribute to an answer, and what happened when I altered them individually and together. The [main post](/notes/investigating-attention-heads-in-in-context-retrieval/) describes those experiments and their limitations.

## How I used AI

I used ChatGPT and coding agents extensively throughout the project. They helped me explore directions, understand papers and code, propose experiments and controls, and interpret results. Coding agents wrote most of the experimental code, ran the GPU jobs and refactored the notebook to make it easier for me and others to follow. They also drafted and revised prose, including this note, using my notes, bullet points and our conversations. At several points I left agents running follow-up experiments before I had worked through the implementation in detail, then came back to the results and asked questions about them.

I prioritized reading and understanding the original paper and authors' code, working through the experiments, and questioning design choices that didn't seem well motivated. For example, I questioned why successive experiments used different interventions when a simpler, consistent intervention seemed sufficient to illustrate the main point. Some decisions made by my coding agents (Codex GPT-6.1 Sol/Astra) seemed quite arbitrary and introduced unnecessary complexity, which honestly surprised me. I hadn't expected to contribute much research taste given how new I am to empirical ML, yet my questions ended up substantially changing the design of the final experiments. As part of this process, I asked for the work to be consolidated into a self-contained notebook and went through the code and results line by line. I am comfortable explaining that code, although I would still use help to write parts of it from scratch.

## Main takeaways

My main takeaway was that it's very important to be on the lookout for possible confounds in experimental setups. I'm starting to appreciate the art of setting up clever experiments.

Working through these choices made me more careful about accurately interpreting the results of an intervention. For example, a head attending strongly to the answer token gives us a reason to inspect it, but this alone doesn't tell us whether its output promotes that answer, suppresses it or does something else entirely. Also, replacing an attention head's output both removes the original contribution and introduces another, so a replacement result needs a deletion comparison if we want to distinguish those effects.

When removing the later head's output mattered more after I changed the earlier head's output, it was tempting to describe the later head as compensating for the misleading message. But it could also have continued to support the correct answer much as it did ordinarily, with that contribution becoming more important once the earlier output was misleading. To distinguish these explanations, I had to check what each intervention allowed to change. In the joint replacement, both supplied vectors were constructed by combining activations from unmodified runs, so the later supplied output could not reflect a response to the earlier alteration. That result left the compensation question unresolved, even though it changed the model's answer.

This made progress slower than I had expected. Agents made it easy to write code and run experiments, but each plausible explanation raised further questions about what needed to be held fixed or compared. I often needed another control, or had to narrow the claim because the experiments did not distinguish the possibilities. Understanding the setup and first few experiments took me several days. I eventually cut down the notebook so I could understand and explain the experiments I kept in detail, leaving some follow-up questions for later. I would be interested in feedback on which follow-up would best distinguish the explanations left open here.
