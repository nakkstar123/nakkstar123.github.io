---
title: "Investigating Attention Heads in In-Context Retrieval"
date: 2026-11-13
slug: investigating-attention-heads-in-in-context-retrieval
draft: false
research: true
tags: [mechanistic-interpretability]
description: "An exploratory investigation of two attention heads in in-context retrieval, using Qwen2.5-3B-Instruct."
---

In [Mixing Mechanisms](https://arxiv.org/html/2510.06182v2), the authors investigate how language models retrieve associations (like object bindings) supplied in context. Among the mechanisms they identify are positional and lexical retrieval. A typical prompt says “the brick is in Box Y, the television is in Box Q, ...” and then asks “Which box is the brick in?” Looking up the brick by the position of its clause (which letter is in the first clause) or by its name (which letter is in the same clause as "brick") would both lead to `Y`, so the correct answer here is compatible with either mechanism, and therefore insufficient to distinguish between the two.

I was interested in how the authors distinguish these possibilities experimentally. In short, they run paired original and counterfactual prompts, then replace the original run's residual stream vector at the final token with the counterfactual's at a selected layer. They design the prompts so that the proposed mechanisms predict *different* answers after this interchange intervention.

For example, suppose the counterfactual includes “... the brick is in Box I, ...” as its sixth clause, followed by “Which box is the brick in?” The receiving prompt instead says “... the brick is in Box Y, ... the television is in Box Q, ...”, with the brick clause fourth and the television clause sixth (and asks about a third object). Transferring a reference to the sixth clause would retrieve `Q` from the receiving prompt (positional). A request about “brick” would retrieve `Y` (lexical). Without the patch, both mechanisms would retrieve their own prompt's correct answer.

| Mechanism | Lookup in the receiving prompt after the patch | Predicted answer |
| --- | --- | --- |
| Positional | Follow the sixth-clause position to “the television is in Box Q.” | `Q` |
| Lexical | Find “brick” in “the brick is in Box Y.” | `Y` |

The authors interpret the transferred state as a *retrieval instruction* that the receiving run uses in its own context. They fit a causal model of how the retrieval mechanisms combine to predict the output distribution. [Paper, §§3–4.](https://arxiv.org/html/2510.06182v2) Their model also includes reflexive retrieval, which is more nuanced, so I won't discuss it here. The causal model leaves open how the LM actually *implements* these mechanisms under the hood. 

I wanted to understand how particular components might implement this computation. One possibility was that different attention heads implement different retrieval mechanisms, with later computation combining their answers. Heads seemed a reasonable place to look because they read from earlier token positions and write into the state used to predict the answer. (Of course, the causal model is compatible with other implementations too.)

This was a time-bounded investigation, and I didn't make much progress toward that goal. In particular, the mapping between mechanisms and heads remains unresolved. However, I did identify two heads whose outputs contribute to the answer, and decided to investigate them. The rest of this write-up will go over these results. At a high level, I examined whether changing the values read by each head (while keeping its attention pattern fixed) increased the probability of a particular (predictable) letter, then tested how the two interventions combined. I describe those experiments below.

## Selecting head 31.7

I used Qwen2.5-3B-Instruct with prompts containing 20 object–box assignments, supplying the unfinished assistant response `Box` after each question. I measured the next-token distribution over the full vocabulary. We number blocks and heads from zero, so 31.7 means head 7 in block 31.

I first worked on identifying a layer to look for heads involved in retrieval. Following the intervention above, I replaced the residual stream at the final `Box` token with the corresponding vector from a donor (counterfactual) prompt. I tested one transformer-block boundary at a time, letting the receiving run continue from there.

The goal was to locate a point in the residual stream where the retrieval instruction, but not the final answer, has been formed. In particular, I looked at the probability of the letter corresponding to the donor's question being answered using the recipient's context (this is the answer predicted by lexical retrieval, according to the original paper's terminology). For example, suppose the recipient says “... the brick is in Box Y, ...” and asks about another object, while the donor says “... the brick is in Box I, ...” and asks about the brick. After the patch, answering `Y` would fit the donor's question using the recipient's context; copying the donor's answer would instead produce `I`.

We call the patched run the **hybrid run**. I compare it with a **query control** that uses the recipient's prompt context but explicitly asks the donor's question, with no activation patch. The **target answer** is the answer to that question using the recipient's declarations. 

Here is a simplified illustration, the actual experiments use 20 clauses.

| | Recipient | Donor | Query control |
| --- | --- | --- | --- |
| First clause | The **brick** is in Box **Y**. | The disk is in Box P. | The **brick** is in Box **Y**. |
| Second clause | The disk is in Box P. | The **brick** is in Box **I**. | The disk is in Box P. |
| Third clause | The **television** is in Box **Q**. | The television is in Box Y. | The television is in Box Q. |
| Fourth clause | The book is in Box I. | The book is in Box Q. | The book is in Box I. |
| Written question | Which box is the **television** in? | Which box is the **brick** in? | Which box is the **brick** in? |
| Correct answer to that question | **`Q`**, the recipient answer | **`I`**, the donor answer | **`Y`**, the target answer |

*Names such as “donor” and “hybrid” identify prompts or intervention runs. Letters, such as `Q` or `Y`, are literal box labels. The box letter playing each answer role can change between problems.*

To form the hybrid run, we keep the recipient's question about the **television**, but replace the residual vector at the final assistant `Box` token with the donor's vector at one transformer block's input, before input normalization. The query control instead gets the question about the **brick** directly in its prompt and runs without a patch.

| Reference answer in the hybrid run | What it would correspond to in this example |
| --- | --- |
| **Recipient answer: `Q`** | Answering the recipient's written question about the television. |
| **Donor answer: `I`** | Producing the donor's answer to its question about the brick. |
| **Target answer: `Y`** | Answering the donor's question about the brick using the recipient's declarations. |

The target answer `Y` is correct for the query control's written question. In the hybrid run, it would answer a question about the brick even though the written question still asks about the television, and it differs from the donor answer `I`. I looked for this combination while varying where the vector was transplanted, tracking the probabilities of the recipient, donor and target answers over the full vocabulary. The sweep below uses the eight actual 20-clause problems, rather than the shortened illustration.

![Residual transplants across block inputs preserve the recipient answer early, favor the donor question answered from recipient context around inputs 27–31, and transfer the donor answer later.](/attachments/binding-heads/residual-sweep-plot.png)

*Figure 1. Each point replaces the final `Box` residual at the indicated block input. Grey tracks the recipient answer, blue the target answer, and orange the donor answer. In the illustration these are `Q`, `Y` and `I`, respectively; the actual box letters vary across problems. Curves are means over eight problems; shaded bands are 95% bootstrap intervals. The dotted line marks layer 31.*

We can see that the target answer, obtained by answering the donor's question using the recipient's context, becomes likely across inputs 27–31. I decided to inspect block 31's input, which is at the end of this interval and before the donor's answer dominates at input 32. The hope is that this corresponds to a fully formed retrieval instruction. The sweep does not establish that donor-answer information is absent from the vector.

The next natural step was to ask whether a few heads in this block were especially important for making the target answer likely, giving me candidates to investigate more closely. Keeping the residual patch at block 31, I set each of that block's 16 head outputs at the final `Box` position to zero one by one, and also repeated those deletions in the query control. Deleting 31.7 caused the largest average decrease in the target answer's probability under the patch.

![Head-by-head deletions in block 31 show the largest mean loss of the hybrid target answer for 31.7, with uneven effects across problems.](/attachments/binding-heads/opening-head-screen-plot.png)

*Figure 2. Green corresponds to the hybrid run and orange to the query control. The upper panel gives the decrease in the target answer's probability after deleting a head. The lower panel gives the decrease in its logit relative to the recipient answer.*

Based on these results, it seemed reasonable to select 31.7 for further inspection, although the deletion effects varied across the eight problems, with one contributing much of the mean. Note that compared to the hybrid, the query control showed a much smaller probability change, even though the logit difference in the lower panel did change. This made 31.7 a candidate in the hybrid, but didn't establish a unique role in ordinary retrieval. Next, I wanted to test what this head's output can contribute during ordinary retrieval. Before trying to assign it a mechanism (which I was ultimately unable to do within the time I allocated for this investigation), I wanted to understand how changing its output affected the answer in the query control. I used that setup in the next experiment.

## Recombining attention and values in 31.7

For this experiment, the receiving run is the query control, with no residual patch. The donor prompt asks about the same object, described in a different clause and assigned a different box letter. Holding the question fixed lets us examine which positions the head attends to and which values it reads there.

In the accompanying notebook's worked example, the query control includes “... the brick is in Box Y, ... the television is in Box Q, ...”. The donor prompt includes “... the disk is in Box P, ... the brick is in Box I, ...”. In each, the two displayed clauses occupy the fourth and sixth positions in the full prompt, as illustrated in the table below.

| | Query control | Donor |
| --- | --- | --- |
| Fourth clause | The **brick** is in Box **`Y`**. | The disk is in Box **`P`**. |
| Sixth clause | The television is in Box `Q`. | The **brick** is in Box **`I`**. |
| Written question | Which box is the **brick** in? | Which box is the **brick** in? |
| Correct answer | **`Y`** | **`I`** |

Both ask where the brick is. The query control should answer `Y` and the donor should answer `I`. In each run, 31.7 attends strongly to the token containing its own correct answer.

![Head 31.7 attends mainly to box Y in the query control and box I in the donor prompt.](/attachments/binding-heads/first-head-attention.png)

*Figure 3. Each bar gives the attention weight from the final `Box` token to a box-letter token in one example. “Other” sums the weights at all remaining positions. The following intervention tests how changing the values read with these weights affects the answer.*

I wanted to test whether 31.7 helps pass on the letter at the position it attends to, which seems like it would be one important mechanism implemented by some attention head for such retrieval. 

There are two parts to this hypothesis: where the head attends, and what information is carried by the value vectors at those positions. I wanted to change the information it reads while holding its attention pattern fixed. If we copied the donor's whole output, we would vary both at once, since it was computed using the donor's attention and values. I therefore constructed a new output for 31.7 by combining the query control's attention weights with the donor's value vectors at the corresponding positions.

![Query-control attention weights and donor values form a crossed output, which replaces only head 31.7's final-Box output before the shared output projection in the query-control run.](/attachments/binding-heads/attention-value-crossing-diagram.svg)

 (It's worth pausing here for a second to understand the experimental setup, and why I previously patched the donor's residual into the recipient, but now combine the query control's attention with donor values and patch the resulting head output into the query control.)

In the query control, the head attends mainly to the position containing `Y`, where the donor has `P`. If the head passes on the letter it reads, this intervention should increase the probability of `P`, dubbed the **predicted wrong answer**: the particular letter predicted by the crossing, which is wrong for the query control's question. It differs from both the target answer `Y` and the donor answer `I`.

Across 32 examples, the mean probability of the predicted wrong answer rose from near zero to 0.199. The target answer, which is correct for the query control's question, remained the most probable token on 27/32 problems.

I then checked whether replacing the value at the *answer position alone* (as opposed to all token positions) could produce a similar effect. The increase was similar on average, with some differences between individual problems. 

![Head 31.7's full crossing and single-position value control increase the predicted wrong answer's probability, while each leaves the target answer top-ranked on 27 of 32 problems.](/attachments/binding-heads/first-head-content-plot.png)

*Figure 4. Both interventions retain the query control's attention and replace only 31.7's final-position output. “Answer position only” substitutes the donor value at the query control's answer position. The left panel plots the predicted wrong answer's full-vocabulary probability for all 32 problems. The right panel counts problems where the target answer is the top prediction. Both interventions leave 27/32 targets top-ranked.*

Recall that we hold attention fixed in both interventions. It's still unclear how the head originally selected those positions for the lookup (it could be using positional or lexical retrieval or a combination of both).

## Testing heads in block 32

With the correct answer still ranked first on 27 of 32 problems, I wondered what other heads were contributing to its score. I decided to search the next layer i.e. block 32, immediately after block 31, so I could test later heads while the earlier output was altered. I mostly decided to do this because I had limited time and thought it would be interesting to test a head in a different layer to investigate sequential dependencies and maybe identify some interesting interactions. Of course, components elsewhere in the model could also contribute.

I first repeated the head-by-head deletions in the query control and hybrid runs, with 31.7 computing its output normally. The hybrid still has the residual transplant at block 31 (the previous layer), the query control has no such patch. The effects were small or variable, and the target answer remained the most probable token after every single-head deletion in the query control.

I measured changes in both probabilities and logits because when the correct answer's probability is close to one, a substantial change in its logit relative to another letter's can produce only a small probability change.

![Block-32 head deletions in the hybrid and query control runs, before supplying a crossed output through 31.7.](/attachments/binding-heads/next-block-baselines-plot.png)

*Figure 5. The first block-32 screen. Green is the hybrid run and orange is the query control. The upper panel gives the decrease in the target answer's probability; the lower panel gives the decrease in its logit relative to the predicted wrong answer defined by the crossing.*

Since these results didn't single out a clear candidate, I repeated the screen in the query control with 31.7's *crossed output from the previous experiment fixed*, labeled **31.7 crossed** in the next figure. This tests which later heads matter when the earlier head supplies the "misleading" output. I deleted each block-32 head's final-position output in turn and compared it with the same setup with that head left intact.

![Deleting head 32.7 has a much larger effect with 31.7 crossed than in the hybrid or query control runs.](/attachments/binding-heads/challenged-screen-plot.png)

*Figure 6. Each head was deleted on the same 32 problems. “Hybrid” has the residual patch; “Query control” has no background intervention; “31.7 crossed” is the query control with the earlier head's output replaced. The upper panel gives the decrease in the target answer's probability; the lower panel gives the decrease in its logit relative to the predicted wrong answer.*

With 31.7 crossed, deleting 32.7 caused the largest decrease in the target answer's probability on all 32 problems, averaging 0.607. Deleting it reduced the target answer's logit relative to the predicted wrong answer by about 0.31 in the query control and 5.28 with 31.7 crossed. This was quite interesting and surprising to me! Removing 32.7's output mattered much more once 31.7 was altered. This could reflect a change in 32.7's own output or in the way the rest of the model uses its output.

I next tested whether 32.7 also passes on the letter at the query control's answer position. If so, applying the same crossing to 32.7 should favor `P` in the example. I returned to the query control and crossed only 32.7. The predicted wrong answer's logit rose relative to the target answer's by about 6.2, compared with 19.4 when only 31.7 was crossed. All 32 answers remained correct. Replacing the value at just the answer position again produced a similar change. Both heads therefore responded to the value replacement in the predicted direction. 

![The same attention/value crossing increases the predicted wrong answer's logit relative to the correct answer for both heads, with unequal probability changes.](/attachments/binding-heads/matched-content-plot.png)

*Figure 7. Each crossing is applied alone in the query control, compared with recombining that head's own attention and values. The left panel gives the change in the predicted wrong answer's probability; the right gives the change in its logit minus the target answer's logit. The small probability change for 32.7 accompanies a positive logit shift.*

*The single-position controls produce the following changes in the predicted wrong answer's logit relative to the target answer. Brackets give 95% bootstrap intervals across the same 32 problems.*

| Head | Full crossing | Answer position only |
| --- | --- | --- |
| 31.7 | 19.4 [17.9, 21.0] | 19.6 [18.1, 21.2] |
| 32.7 | 6.2 [5.4, 7.1] | 6.7 [5.8, 7.6] |

## Joint crossings and deletion controls

Since most answers remained correct under either crossing alone, and since 32.7's output helped preserve the correct answer when 31.7 was altered, I tested the intervention on both together. I constructed both vectors from the unmodified query control and donor runs at their respective layers, then inserted them together. The supplied 32.7 vector therefore cannot reflect a response to 31.7's altered message. Surprisingly, the predicted wrong answer now became the most probable token on 32/32 problems, with mean probability 0.992!

![Crossing both heads raises the predicted wrong answer's probability far more than crossing either head alone.](/attachments/binding-heads/joint-crossing-plot.png)

*Figure 8. The predicted wrong answer's probability when crossing 31.7 alone, 32.7 alone, or both. Bars average the same 32 problems; dots are individual problems and error bars are 95% bootstrap intervals. The two single-head means are 0.199 and approximately 0.000053, compared with 0.992 for the joint crossing.*

I identified one possible confound here: A replacement (1) removes whatever output the head would otherwise compute in that run and (2) inserts a different one. To check how much of the change could arise from removal alone, I compared each replacement with setting that output to zero. For each head, I either let its final-position output compute normally, set it to zero, or supplied the crossed output, giving nine combinations.

![The full grid of normal, zero and crossed final-position outputs for heads 31.7 and 32.7.](/attachments/binding-heads/message-grid-plot.png)

*Figure 9. The vertical axis specifies the intervention on 31.7 and the horizontal axis specifies the intervention on 32.7. The left grid gives the mean probability of the predicted wrong answer. The right grid gives its logit minus the target answer's logit, where positive values favor the predicted wrong answer. When 32.7 computes normally, it can respond to the altered state; its crossed output is a fixed vector from the unmodified source runs.*

With 31.7 crossed, deleting 32.7 raised the predicted wrong answer's mean probability from 0.199 to 0.806. The deleted output is what 32.7 computes after 31.7 has been altered. Replacing that output with the fixed crossed vector raised the probability further to 0.992. So much of the increase was already present after removing 32.7's output, without adding the crossed vector. This seems consistent with both heads helping pass on answer letters, though they could select those letters in different ways. Their contributions could affect the final scores without a computation that explicitly compares their outputs.

However, even these two final-position outputs together do not seem necessary to produce the correct answer on these problems. When I zeroed both final-position outputs, the model still answered all 32 query control prompts correctly, with some changes in confidence. So we see an interesting asymmetry here: the remaining computation was sufficient to produce the correct answer on these problems, even though supplying the two crossed outputs could change it.

| Final-position outputs in the query control | Target answer is top prediction | Predicted wrong answer is top prediction |
| --- | --- | --- |
| Both computed normally | 32/32 | 0/32 |
| Both set to zero | 32/32 | 0/32 |
| Both replaced by crossed outputs | 0/32 | 32/32 |

## Interpretation and limitations

I had hoped to connect the paper's competing retrieval mechanisms to separate heads, but some initial experiments that I did weren't conclusive, so I instead took this approach of identifying active heads and investigating how they interacted. The original question remains unresolved: a head could use several cues, and several heads could perform similar computations. Recall that the attention/value interventions above hold position selection fixed (potentially determined by earlier components), so they leave these possibilities open.

For the two heads I examined, we can see that combining attention and values from different prompts increased the probability of the letter predicted by that combination. Deleting 32.7 had a larger effect on the answer after I altered 31.7, and replacing both outputs produced different answers from deleting them.

Since 32.7 comes after 31.7, it could change its output in response to the earlier head's message. In the joint crossing, however, I replaced both outputs with fixed vectors constructed from unmodified runs, so the supplied 32.7 output could not reflect a response to the altered 31.7 message. The joint result alone therefore does not establish whether 32.7 adapts to the earlier head's message, and I think this might be another useful thing to investigate in the future. 

*The self-contained [NNsight notebook](/notebooks/binding-heads/) includes the code, saved GPU outputs, and CircuitsVis attention views.*
