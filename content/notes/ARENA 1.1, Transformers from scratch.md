---
date: 2026-10-18
tags:
  - arena
confidence: ""
status: ""
---
_AI disclosure: I used ChatGPT to extract takeaways from my original notes. Substantial clarifications and additions appear in labeled blockquotes._

I already felt comfortable with the math of transformers, but I wanted one coding exercise where I implemented a model myself. I was mostly just confused about some of the tensor dimensions and how they interact. This exercise was useful for clearing that up, and for starting to connect the implementation to interpretability.

## Key takeaways and insights

- **Implementing the model clarified which dimensions interact, and where parameters are shared.** The residual stream has shape \((B,T,D)\): batch, sequence position, model dimension. Different batch elements are processed independently. Attention mixes positions within a sequence; the MLP operates on each position independently and **in exactly the same way**, reusing the same matrices. The weights don’t have a separate copy for every batch element or position, even though the activations have those dimensions.
    
    > **ChatGPT clarification:** “Independently” describes the forward computation. The examples still share parameters, and their gradients are combined during training. Also, an MLP’s input can already contain contextual information gathered by earlier attention. In this architecture, attention brings information between positions, and MLPs can then process it.
    
- **Einsum clicked when I thought of it as nested loops and summation indices.** Which dimensions are carried along independently, which are summed over, and which appear in the output? Once that’s clear, the tensor operation follows. This also made Karpathy’s emphasis on tensor operations make more sense to me. But I find it insane to use any other library for tensor manipulation, specifically why bother learning the quirks of PyTorch broadcasting, etc?
    
    For example, projecting into queries:
    
    ```
    queries = einops.einsum(
        W_Q, resid,
        "head d_model d_head, batch pos d_model -> batch pos head d_head",
    )
    queries = queries + b_Q
    ```
    
    The model dimension is summed over. Batch and position are carried along, and we get a separate projection for each head. The head-specific bias broadcasts across batch elements and positions.
    
- **The residual stream as accumulated updates made the logit lens much less mysterious.** My initial reaction was: wait, the unembedding is only trained on the final output, so why would applying it halfway through give anything meaningful? Thinking of the residual stream as a shared backbone, with successive components adding updates, helped this click.
    
    > **ChatGPT clarification:** A logit lens applies the model’s output readout, including the appropriate final normalization, to an intermediate residual state. Residual connections make the accumulated-update picture natural, but they do not guarantee that every intermediate state is aligned with the final readout. Logit-lens predictions can be useful and can also be misleading or brittle.[^lens]
    
- **MLPs as key-value memories give me something more concrete than “this is where the model thinks.”** The input matrix checks alignment with various input directions. The nonlinearity determines how strongly those directions activate. Those activations then weight output directions that get written back into the residual stream. The input directions act like keys, and the output directions like values.
    
    > **ChatGPT context:** This is the interpretation developed in the _Key-Value Memories_ paper.[^mlp] It is a useful view of the computation, rather than a claim that every neuron stores one clean fact or that all knowledge lives exclusively in MLPs.
    
- **MLPs can also perform memory management.** The example of erasing a component from the residual stream was quite interesting. A neuron could detect a direction and then write in the opposite direction. So an additive update can remove information as well as introduce it.
    
    > **ChatGPT addition: the idealized example**
    > 
    > Ignoring biases and normalization, take a unit vector \(v\) and a ReLU neuron whose input direction is \(v\) and output direction is \(-v\). Its residual update is
    > 
    > \[ x'=x-v\,\operatorname{ReLU}(v^\top x). \]
    > 
    > If \(v^\top x>0\), this removes the component along \(v\); otherwise it leaves \(x\) unchanged. GPT-2 uses GeLU, so this exact positive-component deletion is a toy example, not an exact description of a general GPT-2 neuron.
    
- **Separating QK and OV gives one two distinct questions to ask about an attention head.** QK determines which source positions the destination attends to. OV determines what information from those positions gets written to the destination’s residual stream. The attention plots made the first part concrete, for example a head looking for previous occurrences of the same token. Importantly, the QK and OV circuits (along a specific path) interact only via the one value i.e. attention score handed off from the QK to the OV circuit. Looking ahead, this introduces so-called skip trigram bugs. 
    
    > **ChatGPT clarification:** An attention plot alone does not establish what information a head copies or how it affects predictions. The value and output maps matter too.
    
- **The dimensions of LayerNorm’s learned parameters are different from the dimensions of its computed statistics.** This was an important thing I missed until I counted parameters. We compute one mean and variance across the \(D\) features of each token vector, but learn \(D\) scales and \(D\) biases, shared across positions and batch elements. The learned parameters are scales and biases, not means and variances.[^ln]
    
    > **ChatGPT correction: what can be absorbed into the next linear layer?**
    > 
    > The learned affine transformation can be folded into a following linear map. If \(\hat x\) denotes the centered and normalized vector,
    > 
    > \[ W(\gamma\odot\hat x+\beta)+b = \bigl(W\operatorname{diag}(\gamma)\bigr)\hat x+(W\beta+b). \]
    > 
    > This removes the need to handle those affine parameters separately. The input-dependent normalization still remains.
    > 
    > Centering is linear; division by the input-dependent standard deviation is the nonlinear part. Holding that denominator fixed gives a linear approximation to normalization before the learned affine transformation. Whether that approximation is good depends on the intervention’s effect on the denominator.
    
- **The tied-embedding discussion raised a question I didn’t initially resolve.** The direct path from embedding to unembedding seems like it should capture bigram statistics. But why would that make tying the matrices problematic? And wouldn’t the same argument apply if both were trained together?
    
    > **ChatGPT clarification: joint training vs. tying**
    > 
    > Joint training just means both matrices are optimized. Tying imposes an additional constraint: the unembedding must be the transpose of the embedding.
    > 
    > In column-vector notation, let \(E\in\mathbb R^{D\times V}\) embed tokens and \(U\in\mathbb R^{V\times D}\) unembed them. Setting aside positions and LayerNorm, the direct-path scores are \(UEe_i\). Tying gives \(U=E^\top\), so the score matrix becomes the symmetric Gram matrix \(E^\top E\). Untied matrices can represent a broader class of directional score patterns.
    > 
    > The probabilities after softmax need not themselves be symmetric, so “bigram probabilities are asymmetric” alone is not the full argument.
    > 
    > Also, in a full transformer, the other components can compensate. The direct path is not independently forced to be an optimal bigram model, and tying is not simply a bad design. GPT-2 itself uses tied input and output embeddings.[^gpt2] 
    > 
    > *Nakul's response:* Yes, but this (potentially) leads to weird quirks like MLP0 working as an extending embedding layer to break the symmetry. 
    
- **Greedy next-token choices don’t necessarily give the most probable complete sequence.** This was a sampling point I’d overlooked. Sequence probability is a product of conditional probabilities, so a less likely choice now might lead to a much more likely continuation. Beam search keeps several candidate prefixes to explore that possibility.
    
    > **ChatGPT correction:** Finite-width beam search is approximate. It need not find the globally most probable sequence, and it does not universally beat greedy decoding in final sequence probability: it can prune the prefix that greedy would have followed.
    > 
    > For example, suppose the first-token probabilities are \(0.6\) for \(A\) and \(0.4\) for \(B\). After \(A\), the most likely next token has probability \(0.3\), giving a greedy prefix of probability \(0.18\). After \(B\), two tokens each have probability \(0.5\), producing two prefixes of probability \(0.2\). A width-two beam keeps those and drops the greedy prefix. If each retained prefix’s best final token has probability \(0.1\), beam search finishes at \(0.02\), while a certain final token after the greedy prefix would give \(0.18\).
    > 
    > More fundamentally, high sequence probability and good open-ended text are different goals. Repetition can remain a problem even when the search better optimizes the model’s score.[^nucleus]
    
- **Top-\(k\) and top-\(p\) made more sense as two different ways to choose how much of the distribution to retain.** Top-\(k\) keeps a fixed number of tokens. Top-\(p\) keeps the smallest set of highest-probability tokens whose cumulative probability reaches the threshold, so the number retained adapts to how concentrated the distribution is.
    
    > **ChatGPT clarification:** Both methods renormalize the retained probabilities. They preserve relative probabilities among the surviving tokens while changing the overall distribution. Temperature changes relative probabilities too: lowering it sharpens the distribution, but does not guarantee more coherent text.[^generation]
    

## Open questions and things to revisit

- **Compressibility of different kinds of knowledge.** Comparing model size with training-data size made me curious about the compressibility of things like human social behavior, the history of the World Wars, or mathematics. Interesting from an information-theory point of view, if nothing else. I haven’t yet turned this into a precise question or proposed experiment.
    
    > **ChatGPT correction:** A model being smaller than its training corpus does not establish that it isn’t memorizing. It can exploit regularities and still memorize particular passages. Verbatim training-data extraction has been demonstrated on GPT-2.[^memorization] A compression question also needs a specified prediction task and tolerance for error before “how compressible is this domain?” becomes measurable.
    
- **Adding vs. concatenating positional embeddings.** I understand the shared-residual-stream motivation for adding them. I’m still curious whether separating content and position through concatenation would improve positional understanding.
    
    > **ChatGPT clarification:** Superposition is a motivation for sharing the space, not a proof that addition is necessary or superior. A comparison would need to specify the resulting width and projections, since concatenation otherwise changes the architecture and parameter budget.
    
- **Deferred implementation work:** I mostly skipped the training-loop and logging details. Revisit them when I’m actually training a model and need those tools.
    
- **Conditional follow-up:** Skim Neel Nanda’s Transformers 1 and 2 alongside the exercises if I need additional clarification.
    

## References

[^lens]: Belrose et al., [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112).  
[^mlp]: Geva et al., [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913).  
[^ln]: [PyTorch LayerNorm documentation](https://docs.pytorch.org/docs/2.10/generated/torch.nn.LayerNorm.html).  
[^gpt2]: [OpenAI’s GPT-2 implementation](https://github.com/openai/gpt-2/blob/master/src/model.py), which uses the token embedding table again for the final output projection.  
[^generation]: [Hugging Face’s guide to generation methods](https://huggingface.co/blog/how-to-generate). The beam-search counterexample above corrects the guide’s overly strong claim that beam search always finds a higher-probability output than greedy search.  
[^nucleus]: Holtzman et al., [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751).  
[^memorization]: Carlini et al., [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805).