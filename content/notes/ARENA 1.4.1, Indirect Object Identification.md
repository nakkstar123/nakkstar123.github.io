---
date: 2026-10-23
tags:
  - arena
confidence: ""
status: ""
---
_AI disclosure: ChatGPT used to extract takeaways from my original notes. Substantial clarifications and additions appear in labeled blockquotes._

These notes cover my exploration of the first three sections of [ARENA’s IOI exercises](https://github.com/ARENA-education/ARENA_materials/blob/main/chapter1_transformer_interp/instructions/pages/21_%5B1.4.1%5D_Indirect_Object_Identification.md), built around _Interpretability in the Wild_.[^paper] I looked at logit attribution and activation patching; the fuller path-patching analysis remains a follow-up.

The basic task is completing “When Mary and John went to the store, John gave a drink to” with “ Mary.” Here, **IO** is Mary, **S1** and **S2** are the first and second occurrences of John, and **END** is the final “to,” where the next-token prediction is made. Head labels below use the model’s zero-based layer and head indices.

## Key takeaways and insights

- **Trying to break the simple algorithm was useful before tracing the circuit.** My initial picture was roughly “find names, identify the duplicate, predict the other name.” GPT-2 handled some variations, including repeating the recipient and replacing a repeated subject with “he.” But it struggled with:
    
    “John bought milk for Mary and juice for Tom. John handed the juice to”
    
    It preferred Mary to the intended answer, Tom. I initially considered a first-mentioned-name bias, but switching the order didn’t fix the problem. Changing a name to an Indian name changed the preference too. Interesting and weird, but the first simple explanation had already failed.
    
    > **ChatGPT clarification:** With multiple possible recipients, identifying the non-subject name no longer determines the answer. The model must preserve the relation between the object and its recipient. These examples expose limits of the simple account; they do not yet isolate a particular name-bias mechanism. Names can change tokenization and learned priors as well as their role in the sentence.
    
- **SVD gives a way to organize a head’s read and write directions without changing its function.** The useful intuition for me was: a linear map detects alignment with input directions, scales those components, and writes along corresponding output directions. The “rotation” language initially confused me when the output matrix wasn’t square.
    
    > **ChatGPT mathematical clarification:** In column-vector notation, ignoring biases, write
    > 
    > \[ M_{OV}=W_OW_V=L\Sigma R^\top. \]
    > 
    > We can refactor it as \(\widetilde W_O=L\) and \(\widetilde W_V=\Sigma R^\top\), using orthonormal columns of \(L\). Then
    > 
    > \[ L^\top L=I,\qquad \|Lz\|_2=\|z\|_2. \]
    > 
    > A tall matrix with this property is a length-preserving embedding, or isometry, rather than a square rotation. All stretching is assigned to the value map. The coordinates of a head’s residual contribution \(o\) in that output subspace are \(L^\top o\).
    > 
    > The rank bound is **at most** \(d_{\mathrm{head}}\), not strictly less. The analogous QK refactoring balances singular-value scaling between query and key factors. These changes organize the coordinates; they do not guarantee that every singular direction has a clean semantic interpretation, or make destination and source roles interchangeable.
    
- **Choosing the metric is part of choosing what computation to investigate.** The IO-versus-subject logit difference focuses on which name the model prefers, rather than its general willingness to produce a name instead of an article or pronoun. I also liked the role-swapping and averaging tricks for separating the task from fixed preferences for particular names. These are the experimental-design details I want to develop intuition for.
    
    > **ChatGPT clarification:** For logits \(\ell\),
    > 
    > \[ \Delta=\ell_{\mathrm{IO}}-\ell_{\mathrm S} =\log\frac{p(\mathrm{IO})}{p(\mathrm S)}. \]
    > 
    > The softmax normalizer cancels. After the relevant normalization, this is a dot product with the readout direction \(d=u_{\mathrm{IO}}-u_{\mathrm S}\), plus any bias difference.
    > 
    > That bias difference does not vanish just because we subtract two logits. A fixed name-specific bias cancels when we average matched prompts with the answer roles swapped. Other context-dependent lexical effects can remain.
    
- **To decompose a normalized sum, use the normalization of the whole sum.** This was a cute implementation detail I’m glad I double-clicked on. If the final residual is a sum of component contributions, normalizing each contribution separately does not give a decomposition of the normalized residual. We need the final run’s common denominator.
    
    > **ChatGPT mathematical context:** Let \(r=\sum_m c_m\), let \(P\) subtract the coordinate mean, and let
    > 
    > \[ \sigma(r)=\sqrt{\|Pr\|_2^2/D+\epsilon}. \]
    > 
    > After folding LayerNorm’s learned affine parameters into the readout, the observed logit difference decomposes as
    > 
    > \[ \Delta = \sum_m d^\top\frac{Pc_m}{\sigma(r)} +\Delta b. \]
    > 
    > For this fixed forward pass, that is an exact accounting identity. A causal intervention can change the denominator and later computations, so the attribution is not automatically the effect of removing the component.
    > 
    > The exercise’s accumulated-residual plot uses the full run’s final LayerNorm scale. It therefore need not equal the predictions of a literally truncated model that recomputes normalization from each intermediate residual.
    
- **Information can matter before it becomes directly readable as the answer.** The attribution plots initially tempted me to say that the interesting work only happens around layers 7–9, and that MLPs aren’t solving the task. But I’d already noticed why we can’t just remove the earlier layers: they might prepare information that later heads use. The same reasoning applies to MLPs.
    
    > **ChatGPT clarification, from our discussion:** An MLP could write “this name is duplicated” in a direction with zero direct alignment to the answer’s logit-difference direction. A later head could still use it through QK to choose the answer.
    > 
    > Keep three restrictions in view: this is **direct** attribution, at the **final position**, on a **particular prompt distribution**. It hides upstream preparation, computations at other positions, and behavior on different examples.
    
- **A head can attend to the correct name while directly suppressing its prediction.** I initially expected the negative-contributing heads to attend to John instead of Mary. Seeing some of them attend to Mary too was a useful reminder that the attention pattern isn’t the whole computation. The write direction matters. Only a few heads stood out strongly in these attribution plots, which also made heads seem like a useful level to investigate on these examples.
    
    > **ChatGPT context:** Follow-up work on GPT-2 head 10.7 identifies a broader copy-suppression role: it can suppress a token that earlier components already predict and that also appears in the context. This helps explain why a head can look unhelpful on a narrow task while serving a useful calibration role more broadly.[^suppression] The sign of its attribution alone does not establish that it is a bug.
    
- **Patching is a counterfactual intervention, and its conclusion depends on the counterfactual.** Using an activation from a real alternative prompt feels more motivated than arbitrarily flipping numbers. But I still don’t think “it came from a valid forward pass” automatically guarantees that combining it with another run leaves everything else normal.
    
    > **ChatGPT clarification:** That concern is right: a real donor activation can still form an atypical combination with the recipient run.
    > 
    > In denoising, a clean activation is inserted into a corrupted run. Recovery shows that this replacement was sufficient to restore the measured behavior **in that context**, with the rest of the model still operating. It does not show that the component alone solves IOI.
    > 
    > In noising, an alternative activation replaces a clean one. Damage establishes disruption under that intervention, rather than unique indispensability or task specificity.[^patching]
    > 
    > For distinct clean and corrupted baselines, the normalized recovery score
    > 
    > \[ R= \frac{\Delta_{\mathrm{patched}}-\Delta_{\mathrm{corrupt}}} {\Delta_{\mathrm{clean}}-\Delta_{\mathrm{corrupt}}} \]
    > 
    > measures restoration of this scalar metric. It is not a probability, need not stay between zero and one, and does not measure the fraction of the entire algorithm we have explained.
    
- **The residual-patching heatmap helped localize where the clean/corrupted distinction becomes important.** In the tested prompts, restoring S2 helped strongly in earlier layers, while restoring END mattered later, with a striking transition around layers 7–8. That made information movement much more concrete than the equations alone.
    
    > **ChatGPT correction:** This localizes sensitivity to the chosen contrast; it does not establish that all relevant computation follows one route without detours. An activation that is the same in both runs can be essential while showing no patching effect. Small effects can also depend on redundancy and the particular metric.
    > 
    > The encouraging observation is that the contrast admits a relatively localized explanation. “We have mapped all the computation” would be a stronger claim.
    
- **Patching a branch output is different from patching the accumulated residual stream.** This was a very important clarification. The comparison is not simply between several nearby points on the same stream: patching an attention or MLP output replaces one contribution.
    
    > **ChatGPT addition:** For a pre-norm block,
    > 
    > \[ r_{\mathrm{mid}}=r+A(\operatorname{LN}(r)), \qquad r_{\mathrm{next}} =r_{\mathrm{mid}}+M(\operatorname{LN}(r_{\mathrm{mid}})). \]
    > 
    > Patching the MLP output changes the second additive term while retaining the recipient’s \(r_{\mathrm{mid}}\). Patching the next layer’s residual input replaces the whole sum. They can therefore give very different results.
    > 
    > Similarly, MLP0’s strong patching effect can reflect its role in constructing token representations. The course relates that role to GPT-2’s tied embeddings. It does not establish that MLP0 alone stores the IOI answer or literally overwrites the embedding.
    
- **Value patching versus key patching helps distinguish changing the message from changing the route.** For the middle heads, value patching mattered much more in the explored contrast, and their attention patterns already connected END to S2. The useful question became: does the clean prompt help because the head reads a different place, or because it reads different information from the same place?
    
    > **ChatGPT correction:** This supports a message-content explanation for the large difference between those paired runs. It does not prove that the keys never matter or that the head’s routing is independent of token identity.
    > 
    > A particularly useful distinction is **“I am a duplicate”** versus **“I duplicate the token at this position / with this identity.”** Both the clean and corrupted S2 can be duplicates, while differing in which name is duplicated. A duplicate flag used by the keys might therefore change little under this corruption, while the payload changes substantially.
    > 
    > The course’s fuller account explicitly includes some K-composition carrying such a flag. A weak coarse key-patching result does not rule it out.[^arena]
    
- **The circuit diagram became more convincing once I separated those kinds of information.** The simplified story is: early heads identify a repeated token at S2; middle heads move information about the duplicate to END; later name-mover heads use that information through their queries to suppress attention to the subject and retrieve IO instead. The middle heads’ role is not simply to copy the final answer.
    
    > **ChatGPT context:** That is the course’s provisional mechanism, informed by the original IOI analysis. The exploratory plots give evidence for parts of it; targeted path-patching experiments are used to test particular connections. Activation patching changes a component’s output broadly, while path patching tries to isolate its influence through specified receiver pathways.[^arena]
    
- **Manually tracing this is already a lot of work at this scale.** I can see why language models might be useful for scaling the investigation, especially with algorithmic guidance and tools that encode what we already understand. What I found most valuable here was repeatedly correcting the scope of my conclusions as the experiments became more specific.
    

## Open questions and things to revisit

- **Revisit the SVD detour.** I understand the direction–scale–direction picture, but want the refactoring to feel intuitive rather than just algebraically valid.
    
- **Return to the adversarial examples.** Test the name substitutions and ordering changes more systematically. Also check whether the late negative contributions behave differently on cases where the simple duplicate-suppression strategy is inadequate.
    
- **Extend the metric beyond two candidate names.** A pairwise logit difference is clear in the controlled task. What should I measure when there are several plausible recipients?
    
- **Skim Neel’s IOI walkthrough** to get a clearer view of the paper’s results and the stronger tests supporting them.
    
- **Deferred:** Work through the path-patching section, then the replication or extension exercises. The diagram walkthrough was marked done in my raw notes; that did not mean I had completed this more detailed circuit validation.
    

## References

[^paper]: Wang et al., [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593).  
[^arena]: [ARENA 1.4.1 source material](https://github.com/ARENA-education/ARENA_materials/blob/main/chapter1_transformer_interp/instructions/pages/21_%5B1.4.1%5D_Indirect_Object_Identification.md).  
[^suppression]: McDougall et al., [Copy Suppression: Comprehensively Understanding an Attention Head](https://arxiv.org/abs/2310.04625).  
[^patching]: Heimersheim and Nanda, [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255).