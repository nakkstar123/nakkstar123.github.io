---
date: 2026-10-16
tags:
  - 3b1b
confidence: ""
status: ""
---
*AI disclosure: ChatGPT was used to edit my original commentary, check the mathematics, and add technical context. Substantial additions and corrections are marked in blockquotes. The gradient addendum transcribes and expands my handwritten notes and follow-up discussion.*

Notes while watching 3Blue1Brown’s [transformer overview](https://www.3blue1brown.com/lessons/gpt/) and [attention lecture](https://www.3blue1brown.com/lessons/attention/).

## The overall picture

The original transformer in _Attention Is All You Need_ was developed for sequence-to-sequence tasks such as translation. GPT uses a decoder-only variant for next-token prediction.

**Question: how do we decide where to break words in tokenization?**

The tokens become vectors, and those vectors can represent semantic similarities. So I’m guessing the embedding matrix is learned along with the rest of the model.

> **ChatGPT addition: tokenization and embeddings are separate**
> 
> Usually, the tokenizer is trained or chosen first, then held fixed while the language model is trained. For example, byte-pair encoding builds a vocabulary by repeatedly merging frequent adjacent pieces. The resulting tokens need not be whole words or meaningful linguistic units. [This explanation of BPE](https://huggingface.co/docs/course/chapter6/5) gives the details.
> 
> The embedding matrix is then learned through the language-model objective. In the column-vector convention used here,
> 
> \[ W_E\in\mathbb R^{D\times|\mathcal V|}, \qquad e_t=W_E\,\mathrm{onehot}(x_t). \]
> 
> There is one learned column per token type. Token boundaries and learned embedding geometry are different choices.

An attention block updates these token representations. What do the activations correspond to now?

I like thinking of the vectors as having the capacity to soak in context. Their activations throughout the forward pass can represent features of that token within its context.

In the transformer’s MLP, the vectors at different positions don’t interact directly. They are processed independently, using the same operation in parallel. Here the activations make sense to me as features: “Is it English?”, “Is it a noun?”, “Is it part of a bigger word?”, etc.

> **ChatGPT clarification: which MLP, and what counts as a feature?**
> 
> This is the position-wise MLP **inside a transformer**. The earlier makemore MLP concatenated several embeddings and mixed their information.
> 
> Also, a feature need not correspond to one coordinate or a clean yes/no question. Useful information can be distributed across directions and combinations of activations. After attention has mixed context into a position, its MLP can compute with that contextual information even though the MLP itself does not communicate between positions.

There are normalization steps in the middle that Grant glosses over. What are those? I’ll come back to them below.

My initial shorthand was that the system prompt converts next-token prediction into a chatbot.

> **ChatGPT correction**
> 
> A system prompt conditions the model’s behavior. Chat formatting and post-training, such as supervised instruction tuning and preference-based training, also help produce an assistant that follows instructions. A prompt alone does not explain that transformation. [InstructGPT](https://arxiv.org/abs/2203.02155) is an early example of this distinction.

Deep learning seems to scale remarkably well. But what allows this, and why doesn’t increasing the number of parameters just lead to overfitting?

> **ChatGPT correction**
> 
> Deep networks can overfit. Successful scaling depends on the data, architecture, optimization, and regularization. Gradient descent and backpropagation are central tools, but gradient-based optimization is also used outside deep learning. They do not provide a general guarantee against overfitting.

## Embeddings, hidden states, and parameter counts

GPT-3 has about 175B parameters. In Grant’s accounting, these are organized into roughly 28K matrices with eight broad roles. More things to keep track of now! Though the earlier architectures also had distinct embedding, hidden-layer, and output maps.

His convention is to give the embedding matrix one column per token. For the largest GPT-3 model, the numbers are roughly 50K tokens and 12K dimensions per vector. That’s about 0.6B weights just in the embedding matrix.

> **ChatGPT addition: notation and the historical example**
> 
> The GPT-3 dimensions used throughout these lectures are
> 
> \[ D=12{,}288,\qquad |\mathcal V|=50{,}257, \qquad d_k=d_v=128. \]
> 
> It has 96 layers, 96 attention heads per layer, and a training context of 2048 tokens. These are numbers for that model, not universal transformer dimensions. Counts of separate “matrices” also depend on whether implementations pack several maps into one tensor.
> 
> The [GPT-3 paper](https://arxiv.org/abs/2005.14165) describes pre-normalization and alternating dense and locally banded sparse attention. The lecture’s dense-attention picture is a useful simplification.

The embedding examples illustrate that, at least in certain situations, directions have interpretable semantic meanings. Sometimes this is clear enough to predict where a related word’s embedding might lie.

> **ChatGPT qualification**
> 
> Some of the lecture’s illustrations use word-vector models to demonstrate the idea. They are not a guarantee that every transformer representation has the same geometry, or that every semantic distinction is a single direction.
> 
> It also helps to distinguish the initial token embedding from a later **hidden state**, often called a residual-stream vector. They have the same width \(D\), but later states are input-dependent computational representations, not simply more precise dictionary entries.

The model has a supported context length: how many token vectors it can process together. GPT-3’s was 2K. Apparently, in practice there is some flexibility to use a different inference length from the training length. How?

> **ChatGPT addition**
> 
> Attention weights are shared across positions, so the learned projection matrices do not inherently require a particular sequence length. Positional encoding, memory, compute, and training determine the usable range.
> 
> Shorter inputs are straightforward. Longer ones can require additional training or positional modifications; simply fitting a longer input into memory does not guarantee useful behavior. There is a concrete long-context example below.

At the end, another learned map takes the final hidden state from 12K dimensions to roughly 50K logits.

It takes only the last position when generating the next token, even though there are thousands of other vectors. But during training, every position can be treated as the end of its own prompt.

## Parallel training, sequential generation

This is actually the usual autoregressive objective. Even in makemore, we treated one name as multiple prefix-prediction examples.

The efficiency comes from computing the overlapping tasks together in one forward pass. Each position can use its preceding context, but can’t inspect the token it is supposed to predict.

The technical term here is the **causal mask**. Bidirectional models such as BERT use a different setup.

> **ChatGPT addition: the objective and why the mask matters**
> 
> For input tokens \(x_1,\ldots,x_T\), with the corresponding next-token targets available,
> 
> \[ \mathcal L = -\frac1T\sum_{t=1}^{T} \log p_\theta(x_{t+1}\mid x_1,\ldots,x_t). \]
> 
> The output map is applied at each position during training:
> 
> \[ z_t=W_Uh_t^{\mathrm{final}}. \]
> 
> A final normalization may precede this map. The unembedding \(W_U\) is not generally an inverse of \(W_E\); some architectures tie their weights.
> 
> The causal computation ensures that \(h_t^{\mathrm{final}}\) depends only on the prefix through \(t\). All those prefixes are known during training, so positions can be processed in parallel within a layer. Layers still depend on previous layers. During ordinary autoregressive generation, the next sampled token must become available before processing it to predict another.

Ah, he also discusses softmax with temperature here. That’s why I remembered it while going through makemore :)

> **ChatGPT addition**
> 
> For logits \(z\) and temperature \(\tau>0\),
> 
> \[ p_j=\frac{\exp(z_j/\tau)}{\sum_r\exp(z_r/\tau)}. \]
> 
> Lower temperature concentrates probability on larger logits. The limit as \(\tau\to0^+\) concentrates on the maximizers; implementations commonly treat a temperature of zero as greedy decoding.

## What attention is trying to do

The motivating picture is that attention calculates a correction to a generic token representation, like “mole,” based on context, moving it towards a more specific meaning: for example, chemistry rather than an animal.

**Question: do pieces of a compound word converge to the same representation?**

Suppose tokenization splits something like “carbon dioxide” or “Eiffel Tower” into several pieces. Over successive attention blocks, do all those pieces converge towards the same context-rich embedding of the entity?

The answer from my follow-up discussion was no. First, causal attention only gives a position access to the prefix through that position. An early piece can anticipate what might follow, but it can’t inspect the actual later pieces.

Second, think about each position’s role in the computation. The early pieces have different next-token prediction tasks. They can also preserve different information about position, grammatical structure, and being part of a larger expression.

So it’s possible that some feature of several states relates to “Eiffel Tower-ness,” but there’s no reason for the full vectors to become equal.

> **ChatGPT qualification**
> 
> Those splits are illustrative, not claims about a particular tokenizer. The architecture also does not require intermediate states to approach a final semantic prototype. They are optimized to support the prediction computation.

## One attention head

Starting with a single head:

- Position is available to the model, though it need not be added directly to the token embedding; RoPE is a different approach discussed below.
- Query and key matrices map the current hidden states into a smaller space, such as 128 dimensions.
- Their individual meanings aren’t very clear to me, but together they have an explicit purpose: the inner product determines a compatibility score.
- For a receiving position, softmax turns the allowed source scores into weights.
- A value map determines what information each source contributes.
- The resulting update is added back to the original hidden state. That’s the residual, or skip, connection.

> **ChatGPT addition: the forward pass in column-vector notation**
> 
> Let \(h_i\in\mathbb R^D\) be the receiving state. For sources \(j\leq i\),
> 
> \[ q_i=W_Qh_i,\qquad k_j=W_Kh_j,\qquad v_j=W_Vh_j, \]\[ s_{ji}=\frac{k_j^\top q_i}{\sqrt{d_k}}, \qquad \alpha_{ji}=\frac{e^{s_{ji}}}{\sum_{r\leq i}e^{s_{ri}}}, \]\[ o_i=\sum_{j\leq i}\alpha_{ji}v_j, \qquad y_i=h_i+W_Oo_i. \]
> 
> Here
> 
> \[ W_Q,W_K\in\mathbb R^{d_k\times D},\quad W_V\in\mathbb R^{d_v\times D},\quad W_O\in\mathbb R^{D\times d_v}. \]
> 
> The **query position \(i\) attends to the key/value position \(j\)**. Keys do not attend to queries. A standard causal mask allows self-attention, \(j=i\), as well as earlier positions.
> 
> Biases, normalization, positional transformations, and dropout are omitted here.

The attention score array is context length by context length. That seems like one reason why scaling context is so difficult.

> **ChatGPT correction: this is not the KV cache**
> 
> The \(T\times T\) array contains query-key scores or attention weights. The keys and values themselves have shapes \(d_k\times T\) and \(d_v\times T\) per head.
> 
> Dense attention has quadratic interaction cost in \(T\). The KV cache stores keys and values, with storage growing linearly in \(T\) for fixed model dimensions. Efficient kernels can avoid materializing the entire score array without removing the dense pairwise computation.

### Why smaller query, key, and value spaces?

If the effective value map sent a \(D\)-dimensional state directly to a \(D\)-dimensional update, it would be a \(D\times D\) matrix. Instead, we factor it into a projection down and a projection up.

I prefer keeping these maps separate when thinking about what they do. Packing them into the usual multi-head matrix notation initially made the operation harder for me to see.

I still want to understand why we choose low rank here, or why queries and keys need a much smaller dimension. I understand that compatibility is a different task from representing all the information in a token, but that doesn’t determine the dimension.

> **ChatGPT addition: what the factorization buys**
> 
> For a fixed head,
> 
> \[ W_O\sum_j\alpha_{ji}W_Vh_j = \sum_j\alpha_{ji}(W_OW_V)h_j. \]
> 
> The effective value map \(W_OW_V\) has rank at most \(d_v\). Each head writes updates through a limited set of output directions.
> 
> Likewise,
> 
> \[ s_{ji} = \frac{h_j^\top W_K^\top W_Qh_i}{\sqrt{d_k}}, \]
> 
> so its bilinear compatibility map has rank at most \(d_k\). That does not mean the softmax attention-weight matrix must have low rank.
> 
> Smaller head dimensions make multiple distinct attention patterns affordable. They do not constitute a theorem about the intrinsic dimension of language. Multiple heads can collectively write into a much larger space.

He also divides score logits by \(\sqrt{d_k}\). I initially filed this under numerical stability, but why that particular factor?

> **ChatGPT correction: score scale and softmax saturation**
> 
> Under the simplifying assumption that query and key coordinates are independent, mean zero, and unit variance,
> 
> \[ \operatorname{Var}(k^\top q)=d_k. \]
> 
> Dividing by \(\sqrt{d_k}\) keeps the variance near one under those assumptions. Otherwise, larger head dimensions would tend to produce larger score differences and a more saturated softmax, with small routing gradients.
> 
> This scale adjustment is distinct from subtracting the maximum score before exponentiating to prevent overflow. The variance calculation is a motivation, not a claim that trained coordinates remain independent and unit variance.

**Brief detour: cross-attention.**

> **ChatGPT addition**
> 
> In self-attention, queries, keys, and values come from the same sequence. In cross-attention, queries can come from a decoder and keys and values from an encoder’s representation of another sequence. Translation is a natural example: an output position can retrieve information from the source sentence. The score array then has source positions along one axis and receiving positions along the other.

### Multiple heads

GPT-3 has 96 heads in one attention sublayer. Each has its own maps and computes its update in parallel. Then we add their proposed changes to the residual stream.

“Different ways context changes meaning” is a helpful way to hold this in my head.

> **ChatGPT addition: summing heads versus concatenating them**
> 
> If \(o_i^{(r)}\) is head \(r\)’s low-dimensional output, then
> 
> \[ y_i=h_i+\sum_r W_O^{(r)}o_i^{(r)}. \]
> 
> This is the same as concatenating the head outputs and applying a single output matrix whose column blocks are the \(W_O^{(r)}\).
> 
> Heads compute in parallel within the sublayer, but are trained jointly through their combined effect on loss. They need not each learn one clean, distinct linguistic relation.

With the lecture’s dimensions, that’s about 6.3M weights per head and 600M per attention sublayer. Across 96 layers, about 58B parameters come from attention, only around a third of the total.

> **ChatGPT addition: the arithmetic**
> 
> Ignoring biases and normalization:
> 
> |Component|Weight count|
> |---|---|
> |One head’s four maps|\(4D\cdot128=6{,}291{,}456\)|
> |96 heads|\(603{,}979{,}776\)|
> |Attention across 96 layers|\(57{,}982{,}058{,}496\)|
> |Two-layer MLPs of width \(4D\), across 96 layers|\(96\cdot8D^2\approx116\) billion|
> 
> The one-third comparison is specific to this configuration. It is a parameter comparison, not an attention-runtime comparison.

## KV caching becomes much more obvious now

Once I understand each object’s role, the caching idea feels natural.

A newly appended token cannot affect the older representations in an ordinary causal transformer. That should hold across all layers. So we don’t need to recompute the old states just to work out how they influence the new one.

The two inference phases are called **prefill** and **decode**.

> **ChatGPT addition: what is reused**
> 
> Prefill processes the prompt and stores each layer’s keys and values. After sampling a token, a decoding pass processes that new token, computes its queries, keys, and values at each layer, and attends to the stored history. Its new keys and values are appended to the cache.
> 
> Old queries are not needed for the new position’s attention. Old keys and values remain valid because appending a token does not change the old causal prefixes. This assumes unchanged earlier tokens and positions and a compatible deterministic inference computation.
> 
> For a dense head, the new position’s attention work grows roughly as \(O(T(d_k+d_v))\), in addition to its projections and other blocks. Caching avoids rerunning the entire prefix; it does not make reading that history free.

This also clarifies why training can make all the prefix predictions together, while ordinary generation remains sequential. The same causal dependency structure gives us two different kinds of efficiency.

## What relation is a head learning?

I found a helpful counterexample to treating queries as just “noun-ness” and keys as just “adjective-ness”:

“A green forest surrounded the blue creature.”

For the intended adjective-retrieval example, the query at “creature” should favor “blue” over “green.” Recognizing nouns and adjectives alone doesn’t tell us which adjective belongs with which noun.

So it seems more useful to think about the relation a head might pick up on, rather than assign a simple meaning to each map separately.

> **ChatGPT addition: even the query/key coordinates are not unique**
> 
> For any invertible matrix \(B\), replacing
> 
> \[ q\mapsto Bq,\qquad k\mapsto B^{-\top}k \]
> 
> leaves \(k^\top q\) unchanged. In this simplified setting, the attention scores alone cannot identify a unique query/key basis.
> 
> This does not prevent interpretation, but it reinforces the point that their joint computation matters. A head’s attention pattern also does not by itself specify the effect of the information it writes through the value and output maps.

Other useful examples to hold in mind are copying from a previous position and induction: recognizing what followed an earlier occurrence of the current token. In-context learning in a very simple form!

> **ChatGPT addition**
> 
> A classic induction pattern is
> 
> \[ A\,B\,\ldots\,A\quad\longrightarrow\quad B. \]
> 
> A circuit can attend to the earlier continuation and increase its probability. This can involve cooperation between heads and layers. The context changes the activations without requiring a parameter update during that forward pass.
> 
> [Induction-head research](https://arxiv.org/abs/2209.11895) gives evidence for this mechanism in particular settings. It is not a complete account of all in-context learning.

## Detour: how does next-token loss teach a head where to attend?

This question from my discussion with ChatGPT seems worth keeping:

“How can next-token prediction error teach a head where to attend, when nobody supplies labels saying ‘look at this token’?”

The language used to describe the gradients helps me avoid latching onto the noun/adjective analogy too literally. [The follow-up chat is here](https://chatgpt.com/share/6ad26a7e-8cac-83e8-866b-9ab70ddeb097). I also worked through it in my handwritten notes.

The important new expression was

\[ \frac{\partial o}{\partial s_j} = \alpha_j(v_j-o). \]

Increasing a score doesn’t simply add more of \(v_j\). It moves the current average towards \(v_j\), at a rate proportional to how much attention it already receives.

There’s a lot of intuition packed into that. I’ve put the derivation, the gradient interpretation, and the compact matrix version in the addendum below.

_Meta:_ That final matrix expression packs in so much information. I need to get better at reading matrix operations as many related computations happening in parallel, without losing the individual meanings.

## What changes in less simplified models?

Some differences between this picture and published decoder architectures:

- States are often normalized before entering attention.
- Positional information can be introduced by rotating queries and keys.
- Query heads may share key/value heads.
- The MLPs have their own nonlinear computation, sometimes with gates.

The rotation idea seems pretty wild to me. It feels like its own rabbit hole.

> **ChatGPT addition: normalization and residual blocks**
> 
> A pre-normalized block can be written schematically as
> 
> \[ \widetilde H=\operatorname{Norm}_1(H),\qquad H'=H+\operatorname{MHA}(\widetilde H), \]\[ H_{\mathrm{next}} = H'+\operatorname{MLP}(\operatorname{Norm}_2(H')). \]
> 
> LayerNorm normalizes across one token state’s feature coordinates, using its mean and variance. RMSNorm rescales using its root-mean-square magnitude without subtracting the mean. Neither uses batch statistics in the way makemore’s BatchNorm did.
> 
> Pre-normalization was already part of GPT-3; it is something omitted by the simplified picture, not exclusively a later innovation. The gradient addendum below omits normalization, so a real block also backpropagates through those normalization maps.

> **ChatGPT addition: RoPE**
> 
> With column vectors, rotary positional embeddings can take the form
> 
> \[ q_i=R_iW_Qx_i,\qquad k_j=R_jW_Kx_j. \]
> 
> Consequently,
> 
> \[ k_j^\top q_i = x_j^\top W_K^\top R_j^\top R_iW_Qx_i. \]
> 
> For the standard rotary construction, the middle rotation depends on the relative position \(i-j\). This puts position directly into compatibility scoring. It is not necessary to add a positional vector to the token embedding to achieve that.
> 
> Rotations preserve lengths, but change relative alignments. Supporting substantially longer distances still requires care.

> **ChatGPT addition: other concrete differences**
> 
> Grouped-query attention lets several query heads share key/value heads, reducing cache storage and KV projections. For example, the [Llama 3 report](https://arxiv.org/abs/2407.21783) specifies GQA, RoPE, and SwiGLU MLPs.
> 
> A SwiGLU-style MLP has a gated form such as
> 
> \[ f(x)=W_2\bigl[\operatorname{SiLU}(W_1x)\odot(W_3x)\bigr]. \]
> 
> These are published examples, not a specification of every deployed model. Sharing introduces additional shared-gradient contributions, and normalization or RoPE changes the paths leading into the simplified \(Q,K,V\) computation.

### How do million-token models manage the context?

The lecture lists sparse attention, blockwise attention, Linformer, Reformer, Ring Attention, Longformer, and adaptive attention span. Which of these are actually being used?

> **ChatGPT addition: these names address different problems**
> 
> Sparse or local attention changes which interactions are computed. Low-rank or other approximate methods change the computation differently. FlashAttention computes exact dense attention with much better memory access and without storing the whole attention array. Ring Attention distributes blockwise computation across devices; it need not approximate attention.
> 
> A disclosed million-token example is [Qwen2.5-1M](https://qwenlm.github.io/blog/qwen2.5-1m/). Its report describes training through 256K tokens, length extrapolation with Dual Chunk Attention, and an inference framework using sparse attention plus chunked prefill and kernel optimizations.
> 
> So long context can combine training, positional methods, attention structure, and systems work. An advertised context length alone does not identify the mechanism or imply equally good use of every part of that context.

**TODO:** Watch the next chapter on transformer MLPs, superposition, and interpretability.

It’s still quite arbitrary to me why we alternate these particular blocks and what we gain over other natural approaches. Is there an insightful way to see the architecture as a natural choice for language modeling?

> **ChatGPT context for the next step**
> 
> A useful division of labor is input-dependent communication between positions through attention, followed by shared nonlinear computation at each position through the MLP. The updated features can then affect what the next attention layer retrieves.
> 
> Attention is already nonlinear; the MLP adds a different kind of computation. This division helps motivate the design, but it does not uniquely derive today’s architectures or settle the empirical choices of widths, heads, normalization, and gates.

## Mathematical addendum: the gradient detour

> **ChatGPT addition: transcription, corrections, and expansion of my handwritten notes**
> 
> These calculations use column vectors throughout. They describe one causal head without normalization, RoPE, dropout, or biases. Keeping \(W_O\) explicit lets us distinguish the low-dimensional value space from the residual-stream space.
> 
> ### 1. Set up one receiving position
> 
> Fix position \(i\), abbreviate \(q=q_i\), and range over allowed sources \(j\leq i\):
> 
> \[ q=W_Qh_i,\quad k_j=W_Kh_j,\quad v_j=W_Vh_j, \]\[ s_j=\frac{k_j^\top q}{\sqrt{d_k}},\quad \alpha=\operatorname{softmax}(s),\quad o=\sum_j\alpha_jv_j,\quad y_i=h_i+W_Oo. \]
> 
> Let
> 
> \[ g_y=\nabla_{y_i}\mathcal L,\qquad g=W_O^\top g_y=\nabla_o\mathcal L. \]
> 
> Here \(g\) is the loss gradient. The direction that decreases loss is \(-g\).
> 
> My handwritten version sometimes folds the output map into the value map. That corresponds to using residual-space values directly; then \(W_O=I\) in these formulas.
> 
> ### 2. Updating what a source contributes
> 
> Holding the attention weights fixed,
> 
> \[ \nabla_{v_j}\mathcal L=\alpha_jg. \]
> 
> Therefore the value-path contribution to the input-state gradient is
> 
> \[ \left.\nabla_{h_j}\mathcal L\right|_{\mathrm{value}} = \alpha_jW_V^\top g. \]
> 
> Since \(W_V\) is reused across sources, its gradient from this receiving position is
> 
> \[ \left.\nabla_{W_V}\mathcal L\right|_i = \sum_j\alpha_jg h_j^\top = g\left(\sum_j\alpha_jh_j\right)^\top. \]
> 
> **Correction to the handwritten derivation:** the factor \(\alpha_j\) must remain when differentiating through \(v_j=W_Vh_j\). For one source, the parameter contribution is \(\alpha_jg h_j^\top\).
> 
> This has the familiar outer-product interpretation: update each column of the map according to the corresponding input coordinate, the upstream gradient, and how strongly that source was attended to. Gradient descent takes the negative of this gradient.
> 
> ### 3. Why changing a score moves the average towards a value
> 
> If \(s_j\) increases by a small \(\epsilon\), its unnormalized weight \(e^{s_j}\) is multiplied by
> 
> \[ e^\epsilon=1+\epsilon+O(\epsilon^2). \]
> 
> Relative to the old normalization, the numerator for the weighted average gains \(\epsilon\alpha_jv_j\), while the denominator gains \(\epsilon\alpha_j\). Thus
> 
> \[ o' = \frac{o+\epsilon\alpha_jv_j+O(\epsilon^2)} {1+\epsilon\alpha_j+O(\epsilon^2)}, \]
> 
> and
> 
> \[ \boxed{\Delta o = \epsilon\alpha_j(v_j-o)+O(\epsilon^2).} \]
> 
> The \(v_j\) term adds information from that source. The subtraction of \(o\) accounts for the attention mass that must be taken away from the existing mixture.
> 
> If \(\alpha_j\) is tiny, the source has little influence. If it is near one, then \(o\) is already close to \(v_j\), so the change is again small.
> 
> More precisely, holding the values and all other scores fixed, define the other sources’ normalized average
> 
> \[ \bar v_{-j} = \frac{\sum_{r\ne j}\alpha_rv_r}{1-\alpha_j}. \]
> 
> Then
> 
> \[ \frac{\partial o}{\partial s_j} = \alpha_j(1-\alpha_j)(v_j-\bar v_{-j}). \]
> 
> With at least two allowed sources, its norm as we vary only \(s_j\) is maximized at \(\alpha_j=\tfrac12\), unless the value difference is zero. This claim holds the other sources’ relative weights fixed; it is not a universal optimum for useful attention.
> 
> ### 4. How prediction error teaches routing
> 
> Applying the upstream gradient gives
> 
> \[ \boxed{ \beta_j := \frac{\partial\mathcal L}{\partial s_j} = \alpha_j g^\top(v_j-o). } \]
> 
> If moving towards \(v_j\) would reduce loss, then \(g^\top(v_j-o)<0\). Gradient descent increases \(s_j\), because it subtracts the negative \(\beta_j\).
> 
> So routing receives a signal about a source’s usefulness **relative to the current mixture**. No label saying “look at token \(j\)” is needed.
> 
> Small attention can mean a weak routing-learning signal, even when the source could be useful. It does not prove that the source is irrelevant.
> 
> ### 5. Redistribution and geometric constraints
> 
> The score gradients sum to zero:
> 
> \[ \sum_j\beta_j = g^\top\left(\sum_j\alpha_jv_j-o\right)=0. \]
> 
> This matches softmax’s invariance to adding a common constant to every score.
> 
> If \(V=[v_1\ \cdots\ v_m]\), then
> 
> \[ \frac{\partial o}{\partial s} = V\left[\operatorname{diag}(\alpha)-\alpha\alpha^\top\right]. \]
> 
> Changing only scores keeps the output in the convex hull of the fixed values. Infinitesimal changes lie in the span of \(v_j-o\), equivalently in the span of value differences.
> 
> If all values are equal, changing attention has no effect on this output. A real parameter update can also change the values and input states, so this is not a constraint on the entire model’s update.
> 
> ### 6. Back through queries and keys
> 
> \[ \nabla_{k_j}\mathcal L = \frac{\beta_j}{\sqrt{d_k}}q, \qquad \nabla_q\mathcal L = \frac1{\sqrt{d_k}}\sum_j\beta_jk_j. \]
> 
> For this query, gradient descent moves a key towards \(q\) when its score should increase and away when its score should decrease. The query combines the learning signals from all candidate keys.
> 
> Because \(\sum_j\beta_j=0\), we can center the keys:
> 
> \[ \nabla_q\mathcal L = \frac1{\sqrt{d_k}} \sum_j\alpha_j (k_j-\bar k)(v_j-o)^\top g, \qquad \bar k=\sum_j\alpha_jk_j. \]
> 
> The weighted key/value cross-covariance maps the output loss gradient back into query space. In particular, this query gradient lies in the span of key differences. Also,
> 
> \[ \sum_j\nabla_{k_j}\mathcal L=0 \]
> 
> for the contributions from this one query. These identities refer to the simplified projected keys and queries, before positional transformations.
> 
> The projection-matrix contributions follow the same outer-product rule:
> 
> \[ \left.\nabla_{W_Q}\mathcal L\right|_i = (\nabla_q\mathcal L)h_i^\top, \qquad \left.\nabla_{W_K}\mathcal L\right|_i = \sum_j(\nabla_{k_j}\mathcal L)h_j^\top. \]
> 
> Actual parameter gradients sum over all receiving positions and examples. A state can also receive learning signals through several roles: as a query input, a key input, a value input, and through its residual connection.
> 
> ### 7. The compact matrix expression
> 
> Put the input states in columns:
> 
> \[ H=[h_1\ \cdots\ h_T]\in\mathbb R^{D\times T}, \]\[ Q=W_QH,\quad K=W_KH,\quad V=W_VH. \]
> 
> Define
> 
> \[ S=\frac{K^\top Q}{\sqrt{d_k}}+M,\qquad A=\operatorname{softmax}_{\mathrm{columns}}(S), \]
> 
> where \(M_{ji}=0\) for \(j\leq i\) and \(-\infty\) otherwise. Thus column \(i\) is the receiving query’s distribution over source rows \(j\).
> 
> \[ O=VA,\qquad Y=H+W_OO. \]
> 
> Given the upstream gradient \(G=\nabla_Y\mathcal L\), define
> 
> \[ B=W_O^\top G,\qquad E=V^\top B, \]\[ R_{ji} = A_{ji}\left(E_{ji}-\sum_rA_{ri}E_{ri}\right). \]
> 
> Here \(R\) is the gradient with respect to the finite score entries; masked entries are zero. Each column contains the \(\beta_j\) values for one query.
> 
> Then
> 
> \[ \boxed{ \nabla_H\mathcal L = \underbrace{G}_{\mathrm{residual}} + \underbrace{\frac{W_Q^\top KR}{\sqrt{d_k}}}_{\mathrm{query}} + \underbrace{\frac{W_K^\top QR^\top}{\sqrt{d_k}}}_{\mathrm{key}} + \underbrace{W_V^\top BA^\top}_{\mathrm{value}}. } \]
> 
> The parameter gradients are
> 
> \[ \begin{aligned} \nabla_{W_Q}\mathcal L &=\frac{KRH^\top}{\sqrt{d_k}},\\ \nabla_{W_K}\mathcal L &=\frac{QR^\top H^\top}{\sqrt{d_k}},\\ \nabla_{W_V}\mathcal L &=BA^\top H^\top,\\ \nabla_{W_O}\mathcal L &=GO^\top. \end{aligned} \]
> 
> If the value map already returns residual-space vectors and \(W_O=I\), then \(B=G\). This recovers the simpler expression in the screenshot.
> 
> The compact form is the same collection of local computations, summed wherever a state or parameter is reused. Additional heads contribute additional branches; normalization and positional operations require their own chain-rule factors.

## Sources and follow-ups

- [3b1b: transformer overview](https://www.3blue1brown.com/lessons/gpt/) and [attention](https://www.3blue1brown.com/lessons/attention/).
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and [Language Models Are Few-Shot Learners](https://arxiv.org/abs/2005.14165).
- [Tokenization: byte-pair encoding](https://huggingface.co/docs/course/chapter6/5).
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155).
- [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895).
- [Layer Normalization](https://arxiv.org/abs/1607.06450), [RMSNorm](https://arxiv.org/abs/1910.07467), and [RoFormer / RoPE](https://arxiv.org/abs/2104.09864).
- [GQA](https://arxiv.org/abs/2305.13245), [GLU variants](https://arxiv.org/abs/2002.05202), and [the Llama 3 report](https://arxiv.org/abs/2407.21783).
- [FlashAttention](https://arxiv.org/abs/2205.14135), [Ring Attention](https://arxiv.org/abs/2310.01889), and [Qwen2.5-1M](https://qwenlm.github.io/blog/qwen2.5-1m/).
- [My attention-gradient discussion with ChatGPT](https://chatgpt.com/share/6ad26a7e-8cac-83e8-866b-9ab70ddeb097).