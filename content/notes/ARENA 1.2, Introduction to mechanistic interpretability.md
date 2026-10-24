---
date: 2026-10-21
tags:
  - arena
confidence: ""
status: ""
---
_AI disclosure: ChatGPT was used to extract takeaways from my original notes. Substantial clarifications and additions appear in labeled blockquotes._

Before starting the exercises, I talked to ChatGPT about the interpretability tools Neel Nanda lists as essential. I initially grouped them together as “making sense of activations.” Working through the induction-head examples helped me distinguish what each tool actually tells me. The hands-on work here mainly covered sections 1–3; the final reverse-engineering section remains a follow-up.

## Key takeaways and insights

- **The tools answer different questions, even when they inspect the same activations.** Max-activating examples suggest what a component responds to. Probes ask what information we can read out. Steering asks whether an activation change moves behavior. Patching and ablation ask what a specified intervention causes. Those are different kinds of evidence.
    
    > **ChatGPT clarification:** A linear probe’s weights are fitted from labeled activation vectors. We don’t need to manually guess the readout direction. The experimental choices are the property, activation site, position, and dataset. Successful decoding still does not establish that the model uses the decoded information.
    > 
    > Similarly, a difference-of-means steering vector is a candidate control direction, not a guarantee of selective control. A direction that predicts a property need not be a direction that changes only that property.
    
- **The important part of the SAE picture is sparsity, not just having more directions than dimensions.** The three-directions-in-a-plane example was very useful for me. Three independently varying coefficients cannot generally be recovered from their two-dimensional sum. But if only one of three distinct directions is active at a time, the vector can reveal which one it is. The hypothesis is that activations admit this kind of sparse representation in a larger dictionary.
    
    > **ChatGPT mathematical context:** A basic SAE uses
    > 
    > \[ z=\operatorname{ReLU}(W_{\mathrm{enc}}h+b_{\mathrm{enc}}), \qquad \hat h=Dz+b_{\mathrm{dec}}, \]
    > 
    > with more dictionary columns in \(D\) than dimensions in \(h\). One common objective combines reconstruction error with an \(L_1\) sparsity penalty:
    > 
    > \[ \mathbb E\left[ \|h-\hat h\|_2^2+\lambda\|z\|_1 \right], \]
    > 
    > with decoder-column norms controlled. The dictionary is overcomplete, so it is not a basis in the ordinary linear-algebra sense.[^sae]
    
- **“A sparse representation exists” is weaker than “training an SAE recovers the right features.”** My initial instinct was that reconstruction plus sparsity should converge to the right thing if that representation exists. 
    
    > **ChatGPT correction:** There are at least three questions: does such a representation exist, is it identifiable from the data, and does the training procedure recover it? Existence alone establishes neither of the latter two.
    > 
    > Even with a known dictionary, sparse recovery needs conditions. For example, if every set of at most \(2k\) dictionary columns is linearly independent, a vector has at most one representation using at most \(k\) columns: two different representations would give a dependence among at most \(2k\) columns. Learning the dictionary itself adds further ambiguities and optimization difficulties.
    > 
    > A learned feature being sparse and useful for reconstruction also does not, by itself, make its human interpretation correct.
    
- **I like explanations that account for the network’s history of learning.** Nick Cammarata’s recursive-generalization picture was interesting because it tries to explain why one detector ends up responding to examples that initially seem unrelated. An existing detector provides a starting point for learning something related, and later learning builds on that extension. Understanding the neuron means finding those connections.
    
    > **ChatGPT context:** _Neural Networks Are Already Clean_ presents this as a working hypothesis and argues that it explains important apparent examples of polysemanticity.[^clean] That is an empirical disagreement worth investigating, not a general mathematical result against sparse representations. Native-neuron interpretability and an overcomplete representation of a residual vector are also distinct claims.
    > 
    > These notes do not establish which account best explains a particular model. The comparison remains an open question rather than a settled conclusion.
    
- **“Circuit” here means a proposed mechanism involving components and their connections.** I was initially imagining an actual circuit in the TCS sense somehow hidden inside the matrix multiplications. The useful thing to identify is the sequence of computations that produces a behavior.
    
    > **ChatGPT clarification:** Mechanistic-interpretability circuits are usually descriptions of model components, representations, and causal connections at a chosen level of abstraction. They are not generally literal Boolean circuits extracted from the weights. An induction circuit, for instance, can be described through two heads and the information one writes for the other to read.
    
- **The induction task distinguishes learning a general rule from memorizing a particular continuation.** With a repeated random token sequence, the second copy becomes much easier to predict. The model can use information supplied in the current context: after seeing \(A\,B\,\ldots\,A\), predict \(B\).
    
    The previous-token head writes “I follow \(A\)” at \(B\)’s position. A later head matches that information through its keys and copies \(B\) toward the prediction. It needs both the attention pattern and an OV circuit that promotes the appropriate token.
    
    > **ChatGPT clarification:** The novel object can be the random sequence or token pairing, rather than an entirely unseen vocabulary item. An unfamiliar word can also be made from familiar tokenizer pieces. Generalizing to a new sequence supports an algorithmic account, while the weights still contain the learned machinery implementing it.
    > 
    > The two-stage mechanism requires sequential attention layers, which need not be adjacent. This does not establish that the same circuit occurs in every model or explains all in-context learning. The induction-heads paper explicitly separates stronger causal evidence in small attention-only models from more indirect evidence in larger models.[^induction]
    
- **Q-, K-, and V-composition became concrete once you think about how the later head reads an earlier update.** K-composition lets a candidate source advertise context the earlier head wrote there. Q-composition lets an earlier computation help frame the destination’s question. V-composition passes along information supplied by an earlier component.
    
    > **ChatGPT addition:** Ignoring normalization and biases, an earlier contribution \(u\) enters the later head as
    > 
    > \[ q_i=W_Qh_i+W_Qu_i,\qquad k_j=W_Kh_j+W_Ku_j,\qquad v_j=W_Vh_j+W_Vu_j. \]
    > 
    > These give three possible routes for composition; all three need not be functionally important in a particular circuit.
    > 
    > V-composition can implement a virtual two-positions-back head by composing two previous-token retrievals. The authors’ exercises also describe a Q-composition induction mechanism: retrieve the position of an earlier matching token, then use that pointer to look one position ahead.[^exercises]
    
- **The residual stream can hold working state, not just facts or a draft answer.** The induction diagrams made this especially clear. “I am token \(A\),” “an earlier \(A\) was at position \(p\),” and “predict \(B\)” are different pieces of information with different downstream uses. Some subspaces really do look like registers or working memory in this picture.
    
    > **ChatGPT clarification:** The colored subspaces in these diagrams describe an idealized mechanism. They are not proof that a trained model allocates fixed, mutually orthogonal registers to each quantity.
    > 
    > There is also an architecture-specific distinction: ARENA’s toy model uses Shortformer-style positional inputs to queries and keys, without directly adding those vectors to the value stream. That is not RoPE, and it prevents the direct copying of explicit position vectors used by the pointer-arithmetic construction. Position-dependent routing can still leave indirect position information in activations.[^arena]
    
- **Hooks are access points in an actual forward pass, not simulations of an activation.** This is where the tooling started to click: inspect an activation, process it, or replace it and let the rest of the model continue. The remaining awkwardness for me was mostly the callback signature and _functools.partial_.
    
    > **ChatGPT implementation clarification:** A hook receives the activation tensor and a hook-point object containing metadata. The hook’s name selects where to run it; the name itself is not passed as the activation.
    > 
    > For a head-output tensor shaped \((B,T,H,d_{\mathrm{head}})\), a minimal example is:
    > 
    > ```
    > def zero_head(activation, hook, head):
    >     activation[:, :, head, :] = 0
    >     return activation
    > 
    > head_hook = functools.partial(zero_head, head=7)
    > # fwd_hooks=[("blocks.0.attn.hook_z", head_hook)]
    > ```
    > 
    > Binding the head index leaves the two-argument callback the framework expects.[^arena]
    
- **Direct logit attribution tells you what a component writes toward the output, not its entire contribution to the computation.** A later induction head can have high direct attribution while an earlier previous-token head has low direct attribution. That doesn’t make the earlier head unimportant: its effect can be mediated through the later head.
    
    > **ChatGPT clarification:** An unembedding readout direction has dimension \(D=d_{\mathrm{model}}\), not \(d_{\mathrm{head}}\). For a component contribution \(c\), a simple direct contribution to token \(t\)’s logit is \(u_t^\top c\), with the model’s normalization accounted for where applicable.
    > 
    > Common shifts in all logits do not affect probabilities. A logit difference between candidate answers therefore avoids one ambiguity of attributing a single logit. Also, “sum of contributions” is the right description; it is not a claim that the components occupy disjoint subspaces.
    
- **Ablation gave a different kind of evidence from the attention pattern and DLA.** On the repeated-sequence test, the previous-token head and the induction-like heads had large effects on loss when ablated. That connects the proposed mechanism to an intervention, rather than just observing a compatible pattern.
    
    > **ChatGPT correction:** An ablation is not guaranteed to increase loss. It can improve performance on a chosen distribution, remove an unhelpful contribution, or trigger compensation. A large negative loss change warrants investigation, but is not automatically an implementation error.
    > 
    > Mean ablation can avoid some artifacts of replacing a nonzero activation with zero, but an average activation can still be atypical, and it can retain task-relevant information. The replacement distribution and the evaluation dataset are part of the experiment.[^patching]
    
- **Playing with the mechanism helped me generate the questions the formal exercises were asking.** I’ve now seen this pattern several times: an initially unmotivated question becomes natural after I step back and try to build or explain an example myself. The Q-composition and virtual-head exercises were particularly good examples of this.
    

## Open questions and things to revisit

- **Deferred:** Section 4, reverse-engineering the induction circuit from the weights, plus the bonus exercises. I moved to IOI before completing those.
    
- **Get more comfortable with hooks and partial functions.** Writing examples is helping, but I still want the syntax to feel natural.
    
- **Training history vs. depth:** Is the recursive refinement of relationships during training also reflected in increasingly complex relationships across layers? Those are different axes, and I don’t yet know how they relate.
    
- **How much of the residual space is used for each kind of information?** The diagrams suggest relatively small working subspaces. I’d like a precise way to test that, rather than inferring it from the picture.
    
- **Sharp formation of induction behavior raised an alignment question for me.** How much warning is visible before a capability becomes clear in a behavioral metric?
    
    > **ChatGPT clarification:** The reported training transition is evidence about the models and measurements studied. It does not establish that there are no detectable precursors, or that future capabilities must emerge without warning.[^induction]
    

## References

[^arena]: [ARENA 1.2 source material](https://github.com/ARENA-education/ARENA_materials/blob/main/chapter1_transformer_interp/instructions/pages/02_%5B1.2%5D_Intro_to_Mech_Interp.md).  
[^sae]: Cunningham et al., [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600).  
[^clean]: Cammarata, [Neural Networks Are Already Clean](https://already-clean.goodfire.com/).  
[^induction]: Olsson et al., [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895).  
[^exercises]: [Transformer Circuit Exercises](https://transformer-circuits.pub/2021/exercises/index.html).  
[^patching]: Heimersheim and Nanda, [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255).