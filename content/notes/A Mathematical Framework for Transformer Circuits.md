---
date: 2026-10-20
tags:
  - paper
confidence: ""
status: ""
---
_AI disclosure: I used ChatGPT to extract takeaways from my original notes. Substantial clarifications and additions appear in labeled blockquotes._

These are my takeaways from [Neel’s walkthrough](https://www.youtube.com/watch?v=KV5gbOmHbjU) of [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html). The setting of the paper is mostly small, attention-only transformers. I already understood the basic architecture; this paper gave me a slightly better sense of the constraints it imposes and the algorithms those constraints still allow.

## Key takeaways and insights

- **QK and OV have the same matrix dimensions but fundamentally different jobs.** QK is a bilinear form: it takes two residual-stream vectors and produces a scalar compatibility score. OV is a linear map from one residual-stream vector to another. 
    
    > **ChatGPT clarification:** Using column vectors and setting aside biases and normalization,
    > 
    > \[ M_{QK}=W_Q^\top W_K,\qquad M_{OV}=W_OW_V, \]\[ s_{ij}=\frac{h_i^\top M_{QK}h_j}{\sqrt{d_{\mathrm{head}}}}, \qquad \Delta h_i=\sum_{j\leq i}a_{ij}M_{OV}h_j. \]
    > 
    > Here \(i\) is the destination and \(j\) a source. QK controls the attention weights \(a_{ij}\); OV controls the messages being mixed. These are distinct roles, although they interact through shared activations and are learned together.
    
- **Once a source is selected, one head sends the same message from that source to every destination, up to a scalar weight.** This was a parameter-sharing constraint I hadn’t really internalized. Different destinations can attend to a source by different amounts, but they don’t get to ask that head for different transformations of its value.
    
    > **ChatGPT context:** For a fixed source representation \(h_j\), the vector \(M_{OV}h_j\) is shared across destinations. This helps explain the paper’s “skip-trigram bugs”: a head can learn to attend to “keep” from both “in” and “at,” while its value map promotes continuations associated with “keep” without independently choosing “mind” for one destination and “bay” for the other.
    > 
    > This is a restriction on that head’s contribution. Other heads and paths can compensate.
    
- **It often makes more sense to interpret the composed maps than the individual matrices.** I was already comfortable multiplying the matrices, but hadn’t fully appreciated that their factorization introduces arbitrary internal coordinates. Why assign a semantic meaning to an individual value coordinate if another parameterization gives exactly the same computation?
    
    > **ChatGPT addition:** For any invertible change of head coordinates \(R\),
    > 
    > \[ W_V'=RW_V,\qquad W_O'=W_OR^{-1} \]
    > 
    > leaves \(W_OW_V\) unchanged. Likewise,
    > 
    > \[ W_Q'=RW_Q,\qquad W_K'=R^{-\top}W_K \]
    > 
    > preserves query–key dot products. This explains why the products can be more meaningful than a particular factorization. It does not mean that queries, keys, and values are useless to inspect; they remain useful activation sites for experiments.
    
- **“Attention is almost linear” needs a precise qualification.** I found that intuition a little ungrounded when it sounded like attention’s nonlinearity wasn’t doing much interesting computation. The useful distinction is between mixing values with fixed weights and computing those weights from the input.
    
    > **ChatGPT correction:** If the attention pattern is held fixed, the value-mixing operation is linear in the source representations, ignoring biases. The full head is nonlinear because its scores depend on the input and pass through softmax. Conditional routing is precisely what makes mechanisms such as induction interesting. 
    
- **Attending to a token means reading its current residual representation, not necessarily its original token identity.** A position can carry information about earlier context that another component stored there. Seeing a head attend to “ley” therefore doesn’t, by itself, tell me that the head is interested in the literal token “ley.” That position might be carrying information saying that it followed “Durs.”
    
- **The residual stream is a communication channel between components, as well as an accumulation of outputs.** An earlier head can write something that a later head reads through its query, key, or value map. That makes interpreting the entire stream difficult, but gives a more focused question: what does this component write, and which later computation reads it?
    
    The induction example finally made this concrete. In a sequence \(A\,B\,\ldots\,A\), an earlier previous-token head writes information about \(A\) at the position of \(B\). A later induction head uses that information in \(B\)’s key, letting the later \(A\) attend to \(B\). Its OV circuit then promotes \(B\) as the next token.
    
    > **ChatGPT clarification:** This particular mechanism requires sequential attention layers: a later head reads a representation updated by an earlier one. Parallel heads in one layer cannot consume one another’s newly computed outputs.
    
- **A privileged basis is an architectural reason to take particular coordinates seriously.** I didn’t initially understand why a nonlinearity might privilege a basis while a linear map wouldn’t. This matters for deciding whether to inspect neurons or search for directions. There's work showing that Adam optimizer is what leads to privileged bases, but I haven't looked into it.
    
    > **ChatGPT addition:** A coordinatewise nonlinearity such as ReLU generally satisfies
    > 
    > \[ \operatorname{ReLU}(Rx)\neq R\,\operatorname{ReLU}(x) \]
    > 
    > for an arbitrary rotation \(R\). Rotating the hidden coordinates changes which quantities are individually thresholded. In a purely linear composition, a basis change can instead be absorbed into adjacent maps. This privileges the MLP’s neuron axes. It does not by itself explain why particular residual-stream coordinates become outliers: the MLP reads the residual stream through a learned linear map, so its neuron axes need not align with the residual coordinates.[^residual-basis]
    > Adam can also privilege coordinates through training: it rescales parameter updates using coordinatewise gradient statistics and is not rotation equivariant. Following Anthropic’s hypothesis, Caples and Neuhaus compared 12M-parameter TinyStories transformers trained with Adam and SGD with momentum. At comparable loss, Adam produced strong residual-stream outlier channels, while the SGD model’s excess kurtosis remained near zero.[^adam-basis] 
    > This provides causal evidence for Adam’s role in those outliers, without establishing that every form of basis privilege comes from Adam.
    > Neither source of basis privilege guarantees one clean human concept per coordinate.
    > This privileges neuron axes without guaranteeing that each neuron represents one clean human concept. PCA finds high-variance directions in a dataset; it does not automatically recover the model’s computational features or an architecturally privileged basis.[^superposition]
    
- **Superposition is a tradeoff between capacity and interference.** The model benefits from representing many features, but also needs to read them accurately enough to predict. Sparsity helps because features that rarely occur together interfere less often. I find this more motivating than just saying “there are more features than dimensions.”
    
    > **ChatGPT clarification:** An overcomplete collection of feature directions is not a basis in the linear-algebra sense. Not every possible set of coefficients can be uniquely recovered from their sum. Sparsity and other structure are what can make particular representations recoverable. Also, some residual-stream information can support intermediate computation without directly pointing toward an output token.
    
- **A composition score is a useful way to propose connections, but I still want to distinguish that from evidence that the connection is used.** The normalized Frobenius-norm product gives a concrete proxy for how an earlier head’s writing aligns with a later head’s reading. That was a more interesting claim to me than simply expanding the architecture into paths.
    
    > **ChatGPT mathematical clarification:** For K-composition, a relevant product in this notation is \(M_{QK}^{(2)}M_{OV}^{(1)}\). A corresponding score is
    > 
    > \[ C= \frac{\left\|M_{QK}^{(2)}M_{OV}^{(1)}\right\|_F} {\left\|M_{QK}^{(2)}\right\|_F \left\|M_{OV}^{(1)}\right\|_F}, \]
    > 
    > for nonzero factors. This is a normalized product norm, not the usual cosine similarity obtained from a matrix inner product. It measures a potential route through the weights. Actual use also depends on the activations, attention patterns, and input distribution; targeted interventions provide additional evidence.[^composition]
    

## Open questions and things to revisit

- **Build the SVD intuition behind composition scores.** I noted this as a possible mathematical detour, especially the connection between singular directions and reader–writer alignment.
    
- **Understand the evidence for the induction mechanism in detail.** Which interventions establish the previous-token head’s contribution to the later head’s keys, rather than just showing that their patterns fit the story?
    
    > **ChatGPT context:** The later _In-context Learning and Induction Heads_ paper distinguishes strong causal evidence in small attention-only models from correlational evidence in larger models with MLPs. It is useful to keep those levels of evidence separate.[^induction]
    
- **Revisit the usual dimension ordering \(d_{\mathrm{MLP}}>D>d_{\mathrm{head}}\).** What changes if these budgets are allocated differently? The term “projection” was initially confusing because the MLP’s first map expands dimension.
    
    > **ChatGPT clarification:** “Projection” is often used loosely here for a learned linear map. The MLP expands into more nonlinear units before writing back; each attention head uses a smaller communication space. Those are architectural choices, not mathematical requirements.
    
- **Related reading mentioned in the lecture:** _Toy Models of Superposition_. I haven’t worked through its arguments in these notes.
    

## References

[^superposition]: Elhage et al., [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html).  
[^residual-basis]: Elhage, Lasenby, and Olah, [Privileged Bases in the Transformer Residual Stream](https://transformer-circuits.pub/2023/privileged-basis/index.html).  
[^adam-basis]: Caples and Neuhaus, [Adam Optimizer Causes Privileged Basis in Transformer LM Residual Stream](https://www.lesswrong.com/posts/yrhu6MeFddnGRSLtQ/adam-optimizer-causes-privileged-basis-in-transformer-lm).  
[^composition]: [ARENA’s composition-score explanation](https://github.com/ARENA-education/ARENA_materials/blob/main/chapter1_transformer_interp/instructions/pages/02_%5B1.2%5D_Intro_to_Mech_Interp.md), which expands on the framework paper’s metric.  
[^induction]: Olsson et al., [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895).