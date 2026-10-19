---
date: 2026-10-17
tags:
  - makemore
confidence: ""
status: ""
---
_AI disclosure: ChatGPT used for editing and technical additions to my original notes. Substantial additions and corrections appear in labeled blockquotes._

These are my notes while watching [Karpathy’s “Let’s build GPT”](https://www.youtube.com/watch?v=kCc8FmEb1nY), with some detours into questions that came up along the way. As with the previous posts, I’m assuming the reader has watched the lecture.

## Setup

He trains a character-level GPT on Tiny Shakespeare, a roughly 1 MB text file.

> **ChatGPT clarification: the dataset and nanoGPT**
> 
> Tiny Shakespeare contains a selection of Shakespeare’s writing, rather than his complete works. It has roughly 1.1 million characters. Here, each character is a token.[^dataset]
> 
> The lecture builds a small GPT from scratch, then connects it to Karpathy’s more complete nanoGPT implementation. The GPT-2 comparison is that nanoGPT can reproduce the smallest GPT-2 model’s performance when trained at the appropriate scale on OpenWebText. The Shakespeare model built here is much smaller.[^nanogpt]

Makemore is a soft prerequisite for this video.

He starts with a block size of 8. Nine characters implicitly contain eight next-character prediction examples. I want to keep this distinct from batch size.

> **ChatGPT addition: block size vs. batch size**
> 
> Given nine consecutive characters \(c_1,\ldots,c_9\), one training sequence supplies
> 
> \[ \text{input}=(c_1,\ldots,c_8), \qquad \text{target}=(c_2,\ldots,c_9). \]
> 
> At input position \(i\), the model predicts \(c_{i+1}\) using the available prefix \(c_1,\ldots,c_i\). A causal mask lets us compute all eight predictions in one forward pass without letting a position read its answer from a later input position.
> 
> Block size \(T\) controls sequence length and the maximum context available within a training block. Batch size \(B\) counts how many such sequences we process together. The loss averages \(BT\) next-token prediction terms:
> 
> \[ \mathcal L = -\frac{1}{BT} \sum_{b=1}^{B}\sum_{i=1}^{T} \log p_\theta(c_{b,i+1}\mid c_{b,1:i}). \]
> 
> Those \(BT\) terms are not statistically independent examples: predictions within a sequence share overlapping context.

Starts with bigram again!

This time, he trains it with PyTorch’s AdamW optimizer.

> **ChatGPT addition: where AdamW fits**
> 
> Backprop still computes the gradients. AdamW changes how the optimizer uses them: it maintains moving averages of gradients and squared gradients to adapt update sizes, and applies weight decay separately from that adaptation.[^adamw]

## Prefix averaging as a stepping stone to attention

He starts with a “mathematical trick in self-attention” where, at position \(i\), we average the embeddings of characters up to and including \(i\). He describes this as the naivest way to absorb context from previous embeddings. Of course, this loses valuable information, such as which earlier character occurred where.

We want to do this across all examples in a batch, efficiently. Oh, he rewrites prefix averaging as multiplication by a matrix whose \(i\)th row is \(1/i\) for the first \(i\) entries and zero for the rest.

> **ChatGPT addition: the averaging matrix**
> 
> Using one-based indices, define
> 
> \[ A_{ij}= \begin{cases} 1/i,&j\leq i,\\ 0,&j>i. \end{cases} \qquad A= \begin{pmatrix} 1&0&0\\ 1/2&1/2&0\\ 1/3&1/3&1/3 \end{pmatrix} \quad\text{when }T=3. \]
> 
> In the code, token vectors are stored as rows, so this is \(AX\). Keeping my column-vector convention, with  
> \(H=[h_1\ \cdots\ h_T]\), the same operation is
> 
> \[ \bar H=HA^\top, \qquad \bar h_i=\frac1i\sum_{j\leq i}h_j. \]
> 
> At one position, this average does not record which earlier token occurred where. There is a useful distinction, though: the collection of _all_ prefix averages is invertible, since
> 
> \[ h_i=i\bar h_i-(i-1)\bar h_{i-1}, \qquad \bar h_0=0. \]
> 
> So the limitation is what each individual position receives from this aggregation, rather than irreversible information loss from the full sequence of outputs.

Funny way to get this matrix using a lower-triangular mask, \(-\infty\), and softmax. But the nice thing is that this already has the structure we will need for attention!

> **ChatGPT addition: why masked softmax gives prefix averages**
> 
> Start with scores of zero at allowed positions and \(-\infty\) at forbidden positions:
> 
> \[ S_{ij}= \begin{cases} 0,&j\leq i,\\ -\infty,&j>i. \end{cases} \]
> 
> Apply softmax across each row. The first \(i\) entries have equal exponentials, and the rest have exponential zero, giving exactly \(A_{ij}=1/i\) for \(j\leq i\).
> 
> The mask specifies which positions may communicate. The scores determine how much weight each permitted position receives.

Right now, the affinities are uniform. Eventually, we will learn how to compute them through the key and query matrices, then use them to combine values. At this preliminary stage, we’re just averaging the embeddings themselves.

I like this stepping stone. It introduces a very simplified variant that generalizes naturally to what attention actually does. Funny that he motivated it through computational efficiency, but viewing it this way also makes the connection to keys, queries, and values clearer.

> **ChatGPT clarification: what gets learned**
> 
> The uniform averaging example corresponds to using the identity as the value map, \(W_V=I\). The values are the embeddings, not an identity matrix.
> 
> In learned attention, the attention coefficients are computed afresh from the input. The learned parameters are the projection matrices used to produce those coefficients and the values.

## Keys, queries, values, and positions

He explicitly introduces a position embedding table, mapping \(0,\ldots,\text{block size}-1\) to the same embedding dimension as the token embeddings. Then the input is just

\[ h_i=e(c_i)+p_i. \]

I don’t love that this feels like one extra arbitrary design choice, but it’s not completely unreasonable.

Again, I’m not satisfied with the explanation that every embedding emits a query vector, meaning “what am I looking for?”, and a key vector, meaning “what do I contain?” I want a more precise explanation of what those phrases correspond to mathematically.

> **ChatGPT addition: a precise version of the query/key explanation**
> 
> Treat each \(h_i\in\mathbb R^D\) as a column vector. A head computes
> 
> \[ q_i=W_Qh_i,\qquad k_i=W_Kh_i,\qquad v_i=W_Vh_i, \]
> 
> where \(q_i,k_i\in\mathbb R^{d_k}\) and \(v_i\in\mathbb R^{d_v}\). Karpathy uses equal key, query, and value dimensions within a head.
> 
> For receiving position \(i\) and source position \(j\), the score is
> 
> \[ s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_k}} =\frac{h_i^\top W_Q^\top W_Kh_j}{\sqrt{d_k}}. \]
> 
> After masking and softmax across source positions,
> 
> \[ a_{ij} = \frac{\exp(s_{ij})} {\sum_{\ell\leq i}\exp(s_{i\ell})}, \quad j\leq i, \qquad z_i=\sum_{j\leq i}a_{ij}v_j. \]
> 
> The “looking for” interpretation means that the representation at \(i\) determines which features at other positions raise its attention scores. The key determines how a source position participates in that comparison; the value determines what information it sends.
> 
> These are useful roles, but they do not imply that a key literally lists everything a token contains, or that a query has an independently readable meaning. Training learns a compatibility function through the interaction of the two maps.

A single head has a head-size hyperparameter. The attention weights now have shape \((B,T,T)\).

An alternative design could have just learned a \(T\times T\) matrix of weights. But decomposing matrices, or restricting their rank, seems useful as a general principle, as with the bigram model when we introduced embeddings.

> **ChatGPT addition: the low-rank connection, and an important difference**
> 
> With \(Q=[q_1\ \cdots\ q_T]\) and \(K=[k_1\ \cdots\ k_T]\), the unmasked score matrix is
> 
> \[ S=\frac{Q^\top K}{\sqrt{d_k}}, \qquad \operatorname{rank}(S)\leq d_k. \]
> 
> So the factorization connection is real. But a fixed learned \(T\times T\) matrix would assign the same mixing pattern to every sequence. Attention computes a different pattern depending on the tokens and their current representations. The shared projection matrices also let us compute scores for different sequence lengths without learning a separate parameter for every pair of positions.
> 
> The rank bound applies to the scores before masking and softmax. It does not generally apply to the resulting attention matrix. In fact, causal attention before dropout gives a lower-triangular matrix with positive diagonal entries, so it has full rank \(T\), even when the unmasked scores have low rank.
> 
> The restriction is therefore on how we parameterize input-dependent compatibility, rather than a requirement that the final mixing matrix have low rank.

We don’t aggregate tokens directly. We aggregate the values of their representations. Here, the values live in a smaller, 16-dimensional space, so we can’t add the head’s output directly to the original embedding. We would need to project it back up first.

“Attention is a communication mechanism.” I like this framing. Here, position \(i\) attends to itself and previous positions.

> **ChatGPT clarification: attention and graphs**
> 
> A mask can specify a general pattern of permitted communication, which need not be a DAG. Bidirectional self-attention, for example, lets two positions attend to one another. There is still no circular computation within the layer: every output is computed from the input representations supplied to that layer.
> 
> The causal mask used here is the particular pattern that prevents future information from leaking into next-token predictions.

Oh true, computing keys and queries from token embeddings alone doesn’t explicitly tell the model where a token is. So introducing positional information does seem natural. I guess an attention head could specifically cater to positionality, but could it do that without being given position information?

> **ChatGPT addition: can a head infer position on its own?**
> 
> Without positional information or a mask, self-attention is permutation equivariant: rearranging the input tokens just rearranges the output representations. A head cannot distinguish two orders purely by deciding to specialize in position.
> 
> Causal masking complicates that statement because different positions can access different prefixes. This breaks the full permutation symmetry and can provide indirect position cues. It would therefore be too strong to say that a causal model can never infer position without explicit embeddings.
> 
> Learned position embeddings make position directly available. Adding them to token embeddings is one architectural choice; it is not forced by the definition of attention. Other designs supply positional information through different mechanisms.

Across the batch dimension, everything is done independently.

> **ChatGPT clarification**
> 
> Each sequence attends only within itself. The sequences share model parameters, and their gradients are combined during training.

## Encoder, decoder, and cross-attention

In an encoder’s self-attention block, we don’t use this future mask. For something like sentiment analysis, reading the whole input makes sense. Here we use a decoder-style causal block so that later characters don’t give away the answer.

I don’t love this terminology.

Self-attention means the same source produces keys, values, and queries. We could also have keys and values from a different source than the queries, as in language translation. Andrej mentions cross-attention to an encoder, which we won’t implement. What exactly is this doing?

> **ChatGPT addition: reading the original encoder–decoder diagram**
> 
> In the original translation Transformer, the encoder reads the source sentence using bidirectional self-attention. The decoder generates the translation autoregressively. Each decoder block has three sublayers: causal self-attention over the translation so far, cross-attention to the encoder’s outputs, and a tokenwise MLP.[^transformer]
> 
> For decoder representation \(h_i\) and encoder outputs \(u_j\), cross-attention uses
> 
> \[ q_i=W_Qh_i,\qquad k_j=W_Ku_j,\qquad v_j=W_Vu_j. \]
> 
> The decoder’s current state determines which parts of the source sentence to retrieve information from. It cannot read future translation tokens, but it can read the whole source sentence.
> 
> GPT keeps the causal self-attention and MLP parts and omits the separate encoder and its cross-attention sublayer. “Encoder” and “decoder” refer to these architectural roles, rather than just being alternative names for unmasked and masked attention.

## Scaling and multiple heads

Attention scores are scaled before softmax. This seems important because scaling all the scores up would sharpen the distribution toward the largest entry, potentially making it nearly one-hot. Similar to temperature, I suppose.

> **ChatGPT addition: why divide by \(\sqrt{d_k}\)?**
> 
> If the coordinates of \(q\) and \(k\) are independent, centered, and have variance one, then
> 
> \[ \operatorname{Var}(q^\top k) = \operatorname{Var}\left(\sum_{r=1}^{d_k}q_rk_r\right) =d_k. \]
> 
> Dividing by \(\sqrt{d_k}\) keeps the score variance around one under those assumptions. This controls the growth of score magnitudes with head dimension; it does not guarantee unit statistics throughout the attention block.
> 
> The softmax Jacobian is \(\operatorname{diag}(a)-aa^\top\). When a distribution is almost one-hot, its entries become small, making it harder for gradients through softmax to change the routing. The temperature analogy is appropriate.

An important empirical observation: a single head only improves the loss from roughly 2.5 to 2.4. The text is still pretty bad.

This next part didn’t initially make sense to me: he reduces the dimension of each head when he increases the number of heads. Maybe this is to preserve parameter count and illustrate the point, but in practice these hyperparameters should be independently tunable, no?

> **ChatGPT addition: head count vs. head size**
> 
> Yes. Head count and head dimension can be tuned separately. A common convention with \(m\) heads is
> 
> \[ d_k=d_v=D/m, \]
> 
> so concatenating the head outputs gives width \(D\). Ignoring biases, all the query, key, and value projections together then have
> 
> \[ 3mD(D/m)=3D^2 \]
> 
> parameters, independent of \(m\). Changing head count at fixed total width changes how the computation is divided among separate attention distributions.
> 
> But the particular transition in these notes is not a comparison at fixed total head width: one 16-dimensional head becomes four 8-dimensional heads, for total width 32. That also increases the combined size of the query, key, and value projections. The observed loss improvement cannot be attributed solely to adding separate heads.
> 
> Multiple heads are useful because each computes its own attention distribution. A single head has to use the same routing weights for all its value coordinates; several heads can retrieve different information from different positions.

This brings the loss down to about 2.28. Nice! Tokens have a lot to talk about.

## Communication, then computation

The feedforward MLP acts on each token independently. Andrej describes it as: first the tokens talk to each other, then they think individually about the gathered information.

> **ChatGPT addition: what “independently” means here**
> 
> The same MLP is applied at every position, with shared parameters:
> 
> \[ \operatorname{MLP}(h_i) = W_2\,\operatorname{ReLU}(W_1h_i+b_1)+b_2. \]
> 
> In the completed lecture model, the hidden width is \(4D\), then the second linear map returns to width \(D\). This sublayer does not mix positions. Its input already contains contextual information gathered by attention, so it can perform nonlinear computation on that information.

With a feedforward layer, loss goes down to around 2.24.

As we stack multiple blocks, he says the network starts suffering from optimization issues because it’s quite deep. Two things help with depth.

### Residual connections

Skip / residual connections give us a gradient pathway that’s unimpeded by the transformations in the branch. We do this for both attention and the feedforward sublayer.

> **ChatGPT addition: the gradient pathway**
> 
> For \(y=x+F(x)\),
> 
> \[ \nabla_x\mathcal L = \nabla_y\mathcal L+ J_F(x)^\top\nabla_y\mathcal L. \]
> 
> The identity term provides a direct route for the gradient. It helps avoid forcing every signal through a long product of branch Jacobians, though it does not guarantee that gradients can never vanish or explode.

“Projection back into the residual path”: I thought the architecture already did this. It seems like an unnecessary extra linear layer, but maybe I’m missing something.

> **ChatGPT addition: what the output projection contributes**
> 
> After concatenating head outputs, multi-head attention computes
> 
> \[ \Delta h_i = W_O \begin{bmatrix} z_i^{(1)}\\ \vdots\\ z_i^{(m)} \end{bmatrix} +b_O. \]
> 
> If the concatenated width differs from \(D\), a projection makes the dimensions compatible with the residual addition.
> 
> If the widths already match, the projection is not required for dimensional compatibility. It still learns how to mix information from the different heads and write it into the residual representation. Without it, each head writes directly into its own fixed slice of the concatenation.
> 
> Similarly, the MLP’s second linear map both returns from its expanded width to \(D\) and learns what update to write back.

This gets the loss down to about 2.08. Much better, and the samples are starting to look like English.

### LayerNorm

LayerNorm looks similar to BatchNorm, but we normalize across the features of an individual token representation. We don’t need running means and variances, and it works naturally for a single sequence at inference.

**Question / connection to the earlier lectures:** I want the gradient interpretation here too. I reckon it will be similar to BatchNorm’s, but across a different dimension. What’s the intuition for the difference between the two?

> **ChatGPT addition: what gets normalized together?**
> 
> For activations with shape \((B,T,D)\):
> 
> |Normalization|Which values share a mean and variance?|
> |---|---|
> |BatchNorm, in the usual shared-channel sequence setup|Values of one feature across the batch and token positions|
> |LayerNorm used here|The \(D\) features of one token in one sequence|
> 
> BatchNorm compares the same feature across examples. LayerNorm compares features within one representation. They share a centering-and-rescaling operation, but they couple different quantities.
> 
> For one token vector \(h\in\mathbb R^D\),
> 
> \[ \mu=\frac1D\sum_{r=1}^{D}h_r, \qquad \sigma=\sqrt{\frac1D\sum_{r=1}^{D}(h_r-\mu)^2+\epsilon}, \qquad \hat h=\frac{h-\mu\mathbf1}{\sigma}, \]
> 
> followed by
> 
> \[ \operatorname{LN}(h)=\gamma\odot\hat h+\beta. \]
> 
> Here \(\gamma,\beta\in\mathbb R^D\): there is a learned scale and bias for each feature. Each LayerNorm has its own pair of vectors, shared across token positions and batch elements. The statistics are computed from the current token in both training and evaluation.[^layernorm]
> 
> This controls the overall offset and scale of the input seen by the next sublayer. It does not require each feature to have the same distribution across examples, or assume the activations are Gaussian. The usefulness of this particular normalization remains an architectural and empirical question.

So here we normalize the vector across its embedding dimensions. Why should this help, and how does it affect the gradient?

> **ChatGPT mathematical detour: the LayerNorm gradient**
> 
> Let \(g=\nabla_y\mathcal L\) be the upstream gradient at \(y=\gamma\odot\hat h+\beta\), and set \(u=\gamma\odot g\). Write \(\overline{u}=D^{-1}\sum_r u_r\), and similarly for other coordinatewise averages. Then
> 
> \[ \boxed{ \nabla_h\mathcal L = \frac1\sigma \left[ u-\overline{u}\mathbf1 -\hat h\,\overline{u\odot\hat h} \right]. } \]
> 
> To see the geometry, temporarily take \(\epsilon=0\) and assume nonzero variance. Let
> 
> \[ P=I-\frac{\mathbf1\mathbf1^\top}{D}, \qquad z=Ph=h-\mu\mathbf1. \]
> 
> The Jacobian of the normalization before the affine transformation is
> 
> \[ J_{\hat h} = \frac1\sigma \left( P-\frac{zz^\top}{\lVert z\rVert^2} \right). \]
> 
> The matrix in parentheses projects onto the subspace perpendicular to both \(\mathbf1\) and \(z\). First we remove the mean direction, then the centered radial direction, and rescale by \(1/\sigma\).
> 
> This corresponds to two invariances: shifting every coordinate equally does not change the normalized vector, and multiplying the centered vector by a positive scalar does not change it either. The gradient through normalization therefore has no component in those directions.
> 
> With \(\epsilon>0\), the first boxed formula remains exact. Shift invariance remains exact, while scale invariance is approximate, so the radial component is attenuated rather than removed completely.
> 
> This is the same local geometry as normalizing a vector of BatchNorm activations. The difference is which activations form that vector. BatchNorm couples examples contributing to a feature’s statistics; LayerNorm couples the features within one token.
> 
> In this pre-norm Transformer, normalization happens inside a residual branch. The residual stream can carry the original representation past it. Removing scale and offset from what the branch reads does not remove them from the entire network state.

We deviate from the original Transformer formulation by doing pre-norm rather than post-norm.

> **ChatGPT addition: why placement matters for gradients**
> 
> In the original post-norm layout,
> 
> \[ y=\operatorname{LN}(x+F(x)). \]
> 
> In the pre-norm layout used here,
> 
> \[ y=x+F(\operatorname{LN}(x)). \]
> 
> Their forward Jacobians are, respectively,
> 
> \[ J_{\text{post}} = J_{\operatorname{LN}}(I+J_F), \qquad J_{\text{pre}} = I+J_FJ_{\operatorname{LN}}, \]
> 
> evaluated at the appropriate inputs. In pre-norm, the direct identity path also bypasses the LayerNorm Jacobian. This helps explain why the location of normalization matters for optimization, beyond simply controlling activation statistics.[^prenorm]

He also adds dropout. All these changes must meaningfully affect the gradient, and I’m starting to lose intuition. I want to think more about what each one is doing.

> **ChatGPT correction and context: dropout is regularization**
> 
> Dropout is not another normalization method. During training, it randomly zeros selected activations or attention coefficients. With drop probability \(p\), its coordinatewise operation is
> 
> \[ \tilde h_r=\frac{m_r}{1-p}h_r, \qquad m_r\sim\operatorname{Bernoulli}(1-p). \]
> 
> Conditional on \(h\), the expected output equals \(h\). For a sampled mask, the backward pass multiplies the gradient by the same \(m_r/(1-p)\). At evaluation, dropout is disabled.[^dropout]
> 
> Its purpose is to discourage reliance on particular features or connections during training. LayerNorm standardizes the input to a sublayer; residual connections provide direct signal paths; dropout injects training noise for regularization.
> 
> One small consequence: attention coefficients after dropout need not sum to one in an individual training pass.

After scaling up the network, the loss reaches about 1.48. Wow, the text is much better.

> **ChatGPT context for the loss numbers**
> 
> These are approximate observations from the successive versions in the lecture, not controlled estimates of the benefit of each component. Width, depth, context length, and other settings change along the way.
> 
> The cross-entropy here uses natural logarithms, so the loss is measured in nats per character. Comparisons with models using a different tokenizer require care.

## From the small GPT to an assistant

He walks through nanoGPT, and it’s pretty similar. Training something like this on a large chunk of the internet would be the pretraining part of building a model like ChatGPT.

Andrej’s model has around 10 million parameters. GPT-3’s largest model had 175 billion parameters and was trained for 300 billion tokens.

> **ChatGPT correction: corpus size vs. training tokens**
> 
> Tiny Shakespeare has roughly 1.1 million character tokens, with 90% used for training in the lecture code. The completed model has about 10.8 million parameters.[^lecture-code]
> 
> That corpus size differs from the number of token positions processed over training, since we sample blocks repeatedly. GPT-3’s 300 billion training-token figure counts tokens processed during training, and its tokenizer uses subword units rather than individual characters.[^gpt3]

**Aside:** The Wikipedia page on LLMs is actually impressive for its broad coverage of related topics. It seems useful for familiarizing myself with jargon I might not encounter during a mech-interp deep dive.

After pretraining, there is further training to make the model behave like an assistant. Yes, it’s not just adding a system prompt and asking it to complete.

The first two steps in the pipeline he shows are clear to me:

1. Supervised fine-tuning, using labeled examples of the responses we want.
2. Training a reward model from human comparisons of model responses.

It’s unclear to me how we then use the reward model to improve the language model. Basically, I don’t know what PPO is.

> **ChatGPT context: the pipeline being discussed**
> 
> This is the InstructGPT-style pipeline: supervised fine-tuning, reward-model training, then policy optimization using PPO. It is a particular post-training recipe, not a necessary description of every assistant’s training.[^instructgpt]
> 
> In the third step, the language model generates responses and the reward model scores them. Those scores supply a training signal for changing the language model’s response probabilities.

## Detour with ChatGPT: policy gradients

Some very cute math here. We can compute the gradient of expected reward even though sampling a response is discrete.

> **ChatGPT addition: the log-derivative trick**
> 
> Fix a prompt \(x\). Let \(\pi_\theta(y\mid x)\) be the probability of a complete response \(y\), and let \(R(x,y)\) be its reward. Initially, assume the reward function itself does not depend on \(\theta\).
> 
> We want to maximize
> 
> \[ J_x(\theta) = \mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[R(x,y)]. \]
> 
> Under the usual conditions allowing differentiation under the sum,
> 
> \[ \begin{aligned} \nabla_\theta J_x &= \sum_y R(x,y)\nabla_\theta\pi_\theta(y\mid x)\\ &= \sum_y \pi_\theta(y\mid x)R(x,y) \nabla_\theta\log\pi_\theta(y\mid x)\\ &= \mathbb E_{y\sim\pi_\theta} \left[ R(x,y)\nabla_\theta\log\pi_\theta(y\mid x) \right]. \end{aligned} \]
> 
> The identity is \(\nabla p=p\nabla\log p\), for positive \(p\). The reward need not be differentiable with respect to the sampled text.
> 
> Intuitively, \(\nabla\log\pi_\theta(y\mid x)\) measures how parameter changes affect the response’s probability in relative terms. Multiplying by the reward tells us which probability changes improve the average score. We differentiate the probabilities of outcomes, rather than differentiating through the act of sampling them.
> 
> A batch of sampled responses gives a Monte Carlo estimate of this expectation. Averaging over prompts then estimates the gradient of the overall objective.

We can also subtract any baseline \(b(x)\) independent of the sampled response. The log trick converts a gradient of an expectation into an expectation we can estimate from samples. There must be an intuition for why this works more generally that’s worth developing, but I can follow the algebra.

> **ChatGPT addition: why the baseline is allowed**
> 
> For fixed \(x\),
> 
> \[ \begin{aligned} \mathbb E_y \left[b(x)\nabla_\theta\log\pi_\theta(y\mid x)\right] &= b(x)\sum_y\nabla_\theta\pi_\theta(y\mid x)\\ &= b(x)\nabla_\theta 1=0. \end{aligned} \]
> 
> Therefore
> 
> \[ \boxed{ \nabla_\theta J_x = \mathbb E_y \left[ (R(x,y)-b(x)) \nabla_\theta\log\pi_\theta(y\mid x) \right]. } \]
> 
> Probability mass must sum to one. Adding the same reward to every possible response changes the objective’s value but supplies no reason to prefer one response over another.
> 
> A useful baseline reduces the variance of the sampled gradient estimate. In practice, a learned value function can estimate expected future reward from a prompt or partial response. At a particular token decision, that baseline may depend on the prefix already generated, but not on the next sampled token.
> 
> The baseline is treated as fixed when differentiating the policy loss. Subtracting the mean reward of the very same batch, including each response’s own reward, introduces a finite-batch bias unless corrected; a learned baseline or a leave-one-out estimate avoids that particular issue.

This is like ordinary backprop on \((x,y)\), except we scale the gradient by a signed value. It tells us whether to treat this as an example to learn from or an example whose probability we want to reduce. Interesting! A bad example can still be useful to the gradient-based training procedure: reverse the sign and scale the strength of the feedback.

> **ChatGPT addition: the connection to supervised learning**
> 
> Set \(A=R(x,y)-b(x)\). For a sampled response, construct
> 
> \[ \mathcal L_{\mathrm{PG}} = -A\log\pi_\theta(y\mid x), \]
> 
> holding the response and \(A\) fixed while differentiating. A gradient-descent step on this loss gives the policy-gradient ascent direction.
> 
> Since
> 
> \[ \log\pi_\theta(y\mid x) = \sum_t\log\pi_\theta(y_t\mid x,y_{<t}), \]
> 
> this uses the same next-token log-probability calculation as supervised training, multiplied by a signed weight. At token \(t\), with predicted probability vector \(p_t\) and sampled-token one-hot vector \(e_{y_t}\),
> 
> \[ \nabla_{z_t}\mathcal L_{\mathrm{PG}} = A(p_t-e_{y_t}), \]
> 
> where \(z_t\) is the vector of output logits.
> 
> Positive advantage reinforces the sampled response; negative advantage reverses that local gradient. This is relative to the baseline, not necessarily to reward zero. It is not a guarantee of selectively “unlearning” the response’s content, since shared parameters affect many other responses too.
> 
> The reward model can remain frozen during these policy updates. We use its score as a weight on the language model’s own log-probability gradient.

## Reusing samples: where the ratio comes from

There’s another nice trick for taking a batch of samples from an old snapshot of the model and reusing that batch for multiple updates.

Naively, we would sample, update, then need to resample because the expectation is now over a new distribution. I need to understand this better, but to take an expectation using samples from the snapshot, we need reweighting. That’s where the ratio comes in.

> **ChatGPT addition: importance weighting**
> 
> Let \(\pi_{\mathrm{old}}\) be the policy that generated a batch. Define the complete-response probability ratio
> 
> \[ \rho_\theta(y\mid x) = \frac{\pi_\theta(y\mid x)} {\pi_{\mathrm{old}}(y\mid x)}. \]
> 
> Provided the old distribution has support wherever needed,
> 
> \[ \mathbb E_{y\sim\pi_\theta}[f(y)] = \mathbb E_{y\sim\pi_{\mathrm{old}}} [\rho_\theta(y\mid x)f(y)]. \]
> 
> A response that is now twice as probable as under the old policy gets twice the weight. This corrects for drawing samples from a different distribution.
> 
> With fixed \(A=R-b\), differentiating the sampled objective \(\rho_\theta A\) gives
> 
> \[ \nabla_\theta(\rho_\theta A) = \rho_\theta A \nabla_\theta\log\pi_\theta(y\mid x). \]
> 
> At the old parameters, \(\rho_\theta=1\), giving the familiar policy-gradient expression. Farther away, the ratio adjusts the contribution.
> 
> The expectation identity is exact, but a finite batch can give a very noisy estimate when the policies differ substantially. A few enormous ratios can dominate, and important responses under the new policy may be poorly represented in the old batch.

## What PPO clips

There’s a clipping mechanism too, because we don’t want drastic changes based on a finite batch. These choices seem intuitively motivated rather than forced by some deep mathematical fact.

> **ChatGPT addition: from the ratio to PPO**
> 
> Standard token-level PPO uses the state \(s_t=(x,y_{<t})\), the sampled next token \(y_t\), and the ratio
> 
> \[ \rho_t(\theta) = \frac{\pi_\theta(y_t\mid s_t)} {\pi_{\mathrm{old}}(y_t\mid s_t)}. \]
> 
> It holds the old sampled prefixes and advantage estimates \(\hat A_t\) fixed during an update round. The clipped objective to **maximize** is
> 
> \[ L^{\mathrm{clip}}(\theta) = \widehat{\mathbb E}_t \left[ \min\left( \rho_t\hat A_t,\, \operatorname{clip}(\rho_t,1-\varepsilon,1+\varepsilon)\hat A_t \right) \right]. \]
> 
> A gradient-descent implementation minimizes its negative.[^ppo]
> 
> With \(\varepsilon=0.2\):
> 
> - If \(\hat A_t>0\), increasing the sampled token’s probability helps until the ratio reaches \(1.2\). Beyond that, this term gives no further incentive to increase it.
> - If \(\hat A_t<0\), decreasing its probability helps until the ratio reaches \(0.8\). Beyond that, this term gives no further incentive to decrease it.
> 
> The minimum is important: changes in the harmful direction still worsen the objective. This is not simply clipping every ratio and ignoring everything outside the interval.
> 
> Clipping removes an incentive for excessive changes; it does not impose a hard bound on the actual probabilities, parameter updates, or KL divergence. Other samples and shared parameters can still move a ratio beyond the interval.[^ppo-guide]
> 
> There is also a distinction from the exact complete-response importance-weighting identity above. A complete response has a ratio equal to the product of its token ratios. PPO instead uses individual token ratios with advantages and prefixes from the old policy. This is a local surrogate objective, not exact reweighting of the full new-policy response distribution.
> 
> Your distinction between an algebraic identity and an optimization design choice is useful here. Importance weighting follows from an identity. This particular clipped surrogate is a design for making repeated updates on sampled data more stable.

## Why also penalize divergence from a reference model?

At the same time, we penalize too much divergence from a reference model. Okay, this is starting to feel like we’re throwing things in, rather than following one perfectly natural derivation. It seems like its own thing that I’ll have to study later.

> **ChatGPT addition: the old policy and the reference policy have different jobs**
> 
> It helps to distinguish the objects involved:
> 
> |Object|Role|
> |---|---|
> |Current policy \(\pi_\theta\)|The language model being updated|
> |Old policy \(\pi_{\mathrm{old}}\)|A snapshot that generated the current batch; refreshed for a new round of sampling|
> |Reference policy \(\pi_{\mathrm{ref}}\)|A fixed anchor, typically the supervised fine-tuned model in this pipeline|
> |Reward model \(R_\phi\)|Scores responses; held fixed during the policy optimization phase|
> |Value function / critic \(V_\psi\)|Learns to predict future reward, helping estimate token-level advantages|
> 
> PPO’s ratio and clipping concern movement from the **old policy** during one update round. The RLHF reference penalty concerns cumulative departure from a separate **reference policy** across training. Their roles remain different even if the policies start with identical weights.

> **ChatGPT mathematical addition: a coherent objective for the reference penalty**
> 
> For a fixed prompt, a simplified RLHF objective is
> 
> \[ J_x(\theta) = \mathbb E_{y\sim\pi_\theta} [R_\phi(x,y)] - \beta D_{\mathrm{KL}} \left( \pi_\theta(\cdot\mid x) \,\Vert\, \pi_{\mathrm{ref}}(\cdot\mid x) \right), \qquad \beta>0. \]
> 
> This says: improve the learned reward while paying for departure from the reference distribution. InstructGPT used a reference-model KL penalty to reduce overoptimization of the reward model.[^instructgpt]
> 
> Expanding the divergence gives
> 
> \[ J_x(\theta) = \mathbb E_{y\sim\pi_\theta} \left[ R_\phi(x,y) - \beta\log \frac{\pi_\theta(y\mid x)} {\pi_{\mathrm{ref}}(y\mid x)} \right]. \]
> 
> So the KL term can be represented as a modification to the reward. For an autoregressive response, its log-ratio decomposes into a sum of token-level log-ratios. Implementations can use those penalties when computing returns and advantages.
> 
> The log-ratio for one sampled response can be negative; its expectation is the nonnegative KL divergence. The penalty concerns the overall distribution, not a requirement that every sample receive a positive charge.

> **ChatGPT further context: why this objective has a natural mathematical form**
> 
> Suppose we could optimize directly over all response distributions, rather than through a neural network’s parameters. With fixed reward and reference distribution, and assuming the normalizing sum is finite, the optimum is
> 
> \[ \boxed{ \pi^*(y\mid x) = \frac{ \pi_{\mathrm{ref}}(y\mid x) \exp(R_\phi(x,y)/\beta) }{Z(x)}. } \]
> 
> The optimized distribution exponentially reweights the reference toward higher-reward responses. To verify this, substitute the boxed expression into
> 
> \[ J_x(\pi) = \beta\log Z(x) - \beta D_{\mathrm{KL}}(\pi\Vert\pi^*). \]
> 
> Since KL divergence is minimized at zero, the maximum occurs at \(\pi=\pi^*\), within the reference distribution’s support.
> 
> This gives a clean interpretation of the objective. It does not make PPO’s clipping rule mathematically inevitable, or guarantee that the learned reward captures what we actually want.
> 
> One subtlety in connecting this back to policy gradients: the modified reward contains \(\pi_\theta\), so it depends on the policy parameters. For this particular KL objective, differentiating explicitly gives
> 
> \[ \nabla_\theta J_x = \mathbb E_{y\sim\pi_\theta} \left[ \left( R_\phi(x,y) - \beta\log\frac{\pi_\theta(y\mid x)} {\pi_{\mathrm{ref}}(y\mid x)} -b(x) \right) \nabla_\theta\log\pi_\theta(y\mid x) \right]. \]
> 
> The extra derivative of the log-ratio has expectation  
> \(-\beta\,\mathbb E[\nabla_\theta\log\pi_\theta]=0\).  
> That cancellation justifies the expression here; treating every parameter-dependent reward as fixed would not generally be valid.
> 
> At this level, the pieces have separate purposes: policy gradients turn sampled scores into parameter updates; importance ratios connect old samples to an updated policy; PPO clipping limits the incentive to overuse those samples; and the reference penalty specifies how much departure from the starting behavior we are willing to reward. A full implementation also needs the critic, advantage estimation, and practical training details.

## Sources for the technical additions

[^dataset]: [TensorFlow Datasets: Tiny Shakespeare](https://www.tensorflow.org/datasets/catalog/tiny_shakespeare).  
[^nanogpt]: [Karpathy’s nanoGPT repository](https://github.com/karpathy/nanoGPT).  
[^lecture-code]: [Karpathy’s lecture implementation](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py). The repository has subsequent edits, including an initialization change explicitly identified as outside the original lecture.  
[^adamw]: Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).  
[^transformer]: Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).  
[^layernorm]: Ba, Kiros, and Hinton, [Layer Normalization](https://arxiv.org/abs/1607.06450); [PyTorch LayerNorm documentation](https://docs.pytorch.org/docs/2.10/generated/torch.nn.LayerNorm.html).  
[^prenorm]: Xiong et al., [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745).  
[^dropout]: [PyTorch Dropout documentation](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Dropout.html).  
[^gpt3]: Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165).  
[^instructgpt]: Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155).  
[^ppo]: Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).  
[^ppo-guide]: [OpenAI Spinning Up: Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html).