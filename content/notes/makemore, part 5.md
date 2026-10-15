---
date: 2026-10-15
tags:
  - makemore
confidence: ""
status: ""
---
*AI disclosure: ChatGPT was used for editing and technical additions to my original free-write. Substantial additions and explanations from my follow-up chat are marked.*

## Why a more complex architecture?

- Take more than 3 inputs.
- Don’t feed everything into a single hidden layer because it “squashes” information too quickly. Instead, bring that information in progressively. Not clear what this means yet.
- This process will lead us to WaveNet, an autoregressive model developed at DeepMind in 2016 for generating audio, predicting the next audio sample.
- Tree-like structure.

The main idea is progressive combination. Fuse pairs of characters into bigrams, then pairs of bigrams into fourgrams, etc. The receptive field grows exponentially with depth, and the computations within each layer can happen in parallel.

> **ChatGPT addition: the tree and the convolution**
> 
> The lecture builds a simplified hierarchy inspired by [WaveNet](https://arxiv.org/abs/1609.03499). It is useful to distinguish this grouping tree from the original model’s dilated causal convolutions.
> 
> The tree combines disjoint pairs and reduces the number of groups at each layer. A dilated convolution instead applies a shared local operation across the sequence, leaving positions in place but increasing the spacing between the inputs it combines. “Causal” means that a prediction cannot depend on future samples. Dilation determines the pattern of connections, so it is more than an implementation optimization.

In words, my initial interpretation was: “To have \(N\) characters in context, we need a \(V^N\) representation of these characters. We also need not combine them all together, but can do it step by step.” I don’t yet understand the benefit of this approach intuitively, though.

> **ChatGPT correction: possible contexts versus representation size**
> 
> There are \(V^N\) possible contexts, but we do not need \(V^N\) coordinates to represent a context. The previous MLP already avoids an exponential lookup table by concatenating \(N\) learned embeddings, giving an input of dimension \(Nd\).
> 
> The hierarchy introduces a different way to compose features. Weight sharing then lets it reuse those composition rules. Neither is necessary simply to avoid enumerating every possible context.

## What is actually different from the MLP?

Helpful reminder: in the MLP, the concatenated context is \(x=x_1\Vert x_2\Vert x_3\), and the linear map has fan-in \(3d\). Its output then goes into the tanh.

This is equivalent to splitting the weight matrix into position-specific blocks and putting

\[ A_1x_1+A_2x_2+A_3x_3+b \]

into the tanh. So each embedding has its own separate “map to preactivations” matrix, and that’s how position is encoded.

[My follow-up chat with ChatGPT](https://chatgpt.com/share/6ace8a73-1390-83e8-b1e7-723c5e790a85) helped here.

> **ChatGPT explanation from the follow-up**
> 
> The hierarchy makes it easier to build complex patterns out of reusable simpler patterns. The previous MLP can already learn interactions between characters. The potential advantage is that the hierarchical model can reuse those interactions as ingredients in further computation.
> 
> For four embeddings \(e_1,\ldots,e_4\in\mathbb R^d\), compare the flat MLP
> 
> \[ h=\tanh(A_1e_1+A_2e_2+A_3e_3+A_4e_4+b), \qquad z=Rh+c \]
> 
> with a hierarchy:
> 
> \[ \begin{aligned} h_L&=\tanh(Ae_1+Be_2+b),\\ h_R&=\tanh(Ae_3+Be_4+b),\\ h&=\tanh(Ph_L+Qh_R+c),\\ z&=Rh+r. \end{aligned} \]
> 
> Here \(A,B\in\mathbb R^{H\times d}\), while \(P,Q\in\mathbb R^{H\times H}\). The second level combines representations of pairs, rather than directly combining individual character embeddings.

Interesting repeat of \(A\) and \(B\) here.

The intuitive explanation that “for names, bigrams are useful clusters that inform future decisions” makes sense, but the comparison with the old approach isn’t immediate.

> **ChatGPT explanation from the follow-up: sharing is separate from depth**
> 
> Both pair representations use the same \(A,B,b\). Once the model learns a useful way to interpret a pair in one group position, that same computation is available in another group position.
> 
> The previous model already shared character embeddings. This architecture also shares the rule for combining characters into contextual features. It uses a different shared transformation at the next level.

Okay, this makes sense. It is an assumption about the local meaningfulness of relationships between characters and their reusability across positions within the sequence. Also, more training evidence for each detector, then.

Later layers can still treat those local features differently depending on where they occur.

> **ChatGPT addition: what sharing preserves**
> 
> The separate matrices \(A\) and \(B\) preserve order within each pair. Sharing does not make “ab” and “ba” equivalent.
> 
> There is also a fixed grouping boundary: characters 1 and 2 combine immediately, but characters 2 and 3 first meet higher in the tree. Reusing a processor across groups is a more specific assumption than treating every shift of a pattern identically.

There is some napkin math showing that parameters grow like \(O(\log N)\), versus \(O(N)\) for a dense MLP, when keeping hidden dimension \(H\) and embedding dimension \(d\) fixed and using equal depth \(L=\log_2N\). But there is an important qualification: the hierarchy assumes reusable representation mappings. I’ve put the precise comparison in the mathematical addendum below.

It still performs the local matrix calculations at every group. It is sharing the matrices.

> **ChatGPT clarification**
> 
> That statement compares a shared tree with the same tree using separate weights at every node. Sharing reduces parameters without removing those applications. Compared with a flat MLP, the number and sizes of operations also change, so equal parameter counts do not imply equal computation.

## Parameters versus intermediate representations

Andrej’s intuition here is that we’re crushing way too much information way too fast, so the network doesn’t make sense. I think this is useful intuition beyond the weight-sharing explanation.

> **ChatGPT explanation from the follow-up**
> 
> We’re forcing all eight characters to share one fixed-size representation immediately, when it may be useful to develop their local features separately first.
> 
> The hierarchy maintains more intermediate representational capacity early on. Weight sharing makes that affordable in parameters.

Basically, the difference is keeping \(H\) activations in each hidden layer versus going from \(4H\) to \(2H\) to \(H\) activations. With this reuse, it becomes important to separate the indexing of the matrices from the indexing of the different inputs they process.

So we are kinda increasing the number of intermediate activations, and therefore the scope for feature representations, without increasing parameters in the same way.

> **ChatGPT qualification**
> 
> This is an intuition about the organization of computation and the representations that may be easier to learn. It is not a formal information-capacity bound obtained by counting real-valued coordinates. The final representation is still \(H\)-dimensional.
> 
> Also, depth, grouping, and weight sharing are distinct changes. A comparison that changes all three cannot tell us which one caused an improvement.

The surprising thing about language modeling is that we don’t even know how much, or how little, information there truly is in language, right? Or do we? Are we learning more about the underlying dataset here, or about the architecture we’re using to model it?

> **ChatGPT addition**
> 
> For a fixed dataset and evaluation procedure, loss measures how well a particular model predicts held-out data. It does not by itself identify the data’s intrinsic complexity. Model limitations, optimization, finite data, and the structure of the data all affect that loss.

If the task requires interactions between distant characters immediately, a flat architecture might be a better fit. The local hierarchy is an assumption whose usefulness depends on the task.

Important note, btw: increasing the context from 3 to 8 characters in the plain MLP already improved performance on makemore.

## Connection: this makes attention feel more naturally motivated

A useful distinction came out of the chat:

> **ChatGPT explanation from the follow-up**
> 
> - **Parameters:** reusable rules the model learns.
> - **Activations:** representations those rules produce for a particular input.
> - **Connections:** which representations can exchange information.
> 
> Transformers reuse learned processing rules across positions while maintaining a representation at each position. Attention makes the strength of information exchange depend on the input.

Seems like another step in this direction from MLP to WaveNet. Except the connections now seem “dynamic” in that they depend on the input, so fundamentally a bit different.

And transformers keep an \(H\)-dimensional intermediate representation for each token through the layers!

> **ChatGPT addition: the precise connection**
> 
> Within a standard transformer layer, attention projections and the position-wise MLP share weights across token positions. Different layers ordinarily have different weights. Increasing sequence length therefore increases the number of activations without increasing the core block’s parameter count. A table of learned positional embeddings is one possible exception outside that core.
> 
> For one attention head,
> 
> \[ Q=XW_Q,\qquad K=XW_K,\qquad U=XW_V, \]\[ \operatorname{Attention}(X) = \operatorname{softmax} \left(\frac{QK^\top}{\sqrt{d_k}}+M\right)U. \]
> 
> The mask \(M\) determines which positions are allowed to communicate. The input-dependent attention weights determine how strongly they communicate. In a causal model, future positions remain forbidden.
> 
> More activations still require more computation. In particular, ordinary dense attention forms pairwise scores between positions. Parameter count and runtime cost are different quantities.

Now the problem of deciding which connections to use feels more naturally motivated: we have lots of representations but want to reuse the processing machinery, and attention lets relevance depend on the input.

Remember, my goal was to naturally motivate parts of modern architectures as “this is the obvious thing to do,” and hopefully identify and prune, or more deeply understand, the decisions that currently seem arbitrary.

Transformers relax the commitment to a particular local grouping while keeping the commitment to reusing processing machinery across positions.

**Meta: a communication-computation analogy**

If we think of parameters as the “communication” we allow between features, and the number of activations as the “computation” needed to progress through the network, could this be a communication-computation tradeoff? :) I wonder if that’s a useful perspective and whether anyone has explored it.

> **ChatGPT clarification**
> 
> There is a useful distinction here, but those labels need care. Parameters specify stored rules; activations are input-dependent states. Neither is itself a measure of communication or computation.
> 
> A closer analogy is the cost of describing a rule versus the cost of applying it many times. Weight sharing lets a compact description support a much larger computation. Actual communication costs, for example across devices, depend on how parameters, activations, and gradients are distributed and transferred.

## Detour: gradient flow through shared rules

Now it seems important to understand gradient flow through WaveNets, and then transformers.

I wonder if there’s a way to design a course around:

- Find desirable properties of gradient updates.
- Reverse engineer an architecture that gives you those properties.
- Build up from the bigram model to the transformer this way.

This would be fun to do!

ChatGPT connected this to [neural tangent kernels](https://arxiv.org/abs/1806.07572), then prepared a course outline that I’m not yet ready to understand.

> **ChatGPT addition: one concrete connection to the NTK**
> 
> For logits \(f_\theta(x)\), let \(J_\theta(x)\) be their Jacobian with respect to the parameters. A small gradient step on example \(x\) changes the logits on another example \(x'\), to first order, by
> 
> \[ \Delta f_\theta(x') \approx -\eta \underbrace{J_\theta(x')J_\theta(x)^\top}_{K_\theta(x',x)} \nabla_f\ell(f_\theta(x),y). \]
> 
> So \(K_\theta(x',x)\) describes how learning from one input locally changes predictions on another. This gives a mathematical way to ask what kinds of generalization an architecture’s gradient updates encourage.
> 
> This first-order relation applies to differentiable models at their current parameters. The stronger theory in which the kernel remains fixed during training requires additional assumptions. It is one connection to the proposed course, rather than a complete recipe for designing architectures.

It’s actually very simple to derive the gradient update to a shared lower-layer matrix, say \(A\). The interpretation is: “How much should input coordinate \(q\) influence local feature \(p\), taking both occurrences into account?”

The answer is what you’d expect: sum the gradient contributions from the different uses.

> **ChatGPT addition: the shared-weight gradient**
> 
> Using the four-character hierarchy above, and temporarily leaving out batch normalization, let \(\delta_*\) be the loss gradient with respect to the parent’s preactivation. Then
> 
> \[ \begin{aligned} \delta_L&=(P^\top\delta_*)\odot(1-h_L^2),\\ \delta_R&=(Q^\top\delta_*)\odot(1-h_R^2). \end{aligned} \]
> 
> The gradients of the shared matrices are
> 
> \[ \nabla_A\ell=\delta_Le_1^\top+\delta_Re_3^\top, \qquad \nabla_B\ell=\delta_Le_2^\top+\delta_Re_4^\top. \]
> 
> In coordinates,
> 
> \[ \frac{\partial\ell}{\partial A_{pq}} = (\delta_L)_p(e_1)_q+(\delta_R)_p(e_3)_q. \]
> 
> The same parameter receives learning signals from every place where it was used. Those contributions can reinforce or oppose one another.

There’s useful intuition in the extreme cases too: even if two pairs produce the same activations, \(h_L=h_R\), their gradients can differ because of their roles in the larger context.

So the forward computation can be local while the learning signal reflects the global prediction task.

Doubling receptive fields lets distant characters participate through relatively few stages, although those stages can still shrink or amplify gradients.

## Detour: reusable rules for selecting rules?

We went from “reuse a local feature detector” to “reuse a rule for selecting relevant information.” Seems like we’re going one level higher.

Can we generalize this to multiple levels higher, or is there a limit? Earlier, we wanted local features to save parameters. Now the information need not be local, but the rule for deciding what is relevant is reusable. What if that rule isn’t reusable either, but the rule for selecting that rule is?

Would that motivate a different architecture?

ChatGPT brought up mixture of experts and hypernetworks. So there are ways to do this. Transformers also seem to have some implicit flexibility through their layers.

> **ChatGPT clarification and mathematical context**
> 
> Ordinary attention primarily selects **which information to gather**. Selecting **which processing function to apply** is a further architectural choice.
> 
> A mixture of experts can make that choice explicitly:
> 
> \[ y=\sum_{e=1}^{E}g_e(x)f_e(x), \]
> 
> where a learned router supplies weights \(g_e(x)\), often activating only a few experts. The experts and the router are themselves shared learned functions.
> 
> A hypernetwork can instead generate weights:
> 
> \[ W(x)=H_\psi(x),\qquad y=F(x;W(x)). \]
> 
> Here the reusable object is the rule for generating a processing rule. The conditioning input need not be the same \(x\); it could describe a task or some other context.
> 
> These are examples of the principle you’re pointing at: when sharing one fixed rule seems too restrictive, look for reusable structure in how that rule should vary.

I guess it ultimately depends on the underlying data. Transformers, perhaps with MoE, seem like a useful level of meta-rule-making for language models.

> **ChatGPT qualification**
> 
> That is an empirical hypothesis, rather than a reason to conclude that this is the optimal level or that there is a principled ceiling on further levels.

## Back to the lecture: tensor shapes and weight reuse

Andrej motivates this through tensor dimensions and how the linear layer operates across them. I don’t know if introducing it this way is necessary, and I don’t find it very useful.

I think it also makes the parameter reuse less explicit. Incrementally flattening tensors is “slick,” but the presentation feels pedagogically unhelpful to me.

> **ChatGPT addition: separate the reshape from the learned operation**
> 
> For a tensor of shape \((B,T,C)\), grouping consecutive pairs reshapes it to
> 
> \[ (B,T/2,2C). \]
> 
> That rearranges the same numbers. It does not itself combine features or discard information.
> 
> The linear layer then applies the **same** map \(\mathbb R^{2C}\to\mathbb R^H\) at every batch and group position. Matrix multiplication over the last dimension implements this reuse. Keeping those two operations conceptually separate makes the architecture easier to see.

The dimensions do line up cleanly, and you can see where the total number of activations decreases. Four groups of character pairs, each with 200 activations, become two groups of four characters, again with 200 activations each, then one group of eight characters.

So it’s \(800\to400\to200\) hidden activations per context.

> **ChatGPT addition: the shapes**
> 
> |Stage|Shape|Activations per context|
> |---|---|---|
> |Character embeddings|\((B,8,d)\)|\(8d\)|
> |Pair representations|\((B,4,H)\)|\(4H\)|
> |Four-character representations|\((B,2,H)\)|\(2H\)|
> |Eight-character representation|\((B,H)\)|\(H\)|
> |Output logits|\((B,V)\)|\(V\)|

The linear layers do the “compressing,” except the first one here: \(H\) is larger than \(2d\), so each pair’s embedding actually expands into a larger feature representation.

He adjusts the hidden dimension to 68 to keep the parameter count similar to the previous network. Performance is then about the same as the eight-character MLP. He points out that we haven’t done hyperparameter tuning and that there is a potential batch normalization bug.

## Batch normalization: what counts as a channel?

The question is: should batch norm maintain separate statistics for the different groups?

My first thought was yes, because one training example consists of these groupings. But actually, I don’t know. I think the matrix sharing is the thing to focus on.

He describes a “channel” as the same activation coordinate across the groups. So for \(4\times68\) activations, he describes 68 channels.

This is a different perspective, and it makes sense with parameter reuse. We’re treating each group as another input to the same local processor.

> **ChatGPT correction: multiple responses, not independent examples**
> 
> The four groups still belong to one context with one next-character target. They are not four independent training examples.
> 
> But the same processor produces a 68-dimensional feature vector at each of those four locations. Each coordinate is one channel, with four responses per context. Across a batch of 32 contexts, that gives 128 responses per channel for estimating normalization statistics. Those responses can be correlated.

The takeaway is that if we want a reusable local processor, its normalization should also be shared across the places where we reuse it. Position-specific normalization is a legitimate architectural choice, but it isn’t the choice we want here.

Think of each channel as a learned feature detector, or at least a learned feature coordinate. One eight-character input gives four responses from that detector.

> **ChatGPT addition: the reduction axes**
> 
> For preactivations \(a\) of shape \((B,T,H)\), the lecture’s revised normalization pools over both the batch and group axes:
> 
> \[ \mu_h=\frac{1}{BT}\sum_{b,t}a_{bth}, \]\[ s_h^2=\frac{1}{BT-1}\sum_{b,t}(a_{bth}-\mu_h)^2, \]\[ \widehat a_{bth} = \gamma_h\frac{a_{bth}-\mu_h}{\sqrt{s_h^2+\epsilon}}+\beta_h. \]
> 
> The variance expression follows the custom implementation’s use of Bessel’s correction.
> 
> Reducing only over the batch axis gives separate statistics for each group position. Reducing over both axes gives one mean and variance per channel. The learned scale and offset are also shared across those positions.
> 
> During training, pooling these statistics couples the groups through normalization. The processor is therefore not completely independent at each location, even though its linear map is shared.

Btw, fixing this issue itself doesn’t improve the result much.

To show a stronger result, he scales from about 22K parameters to about 76K. He also acknowledges that the current testing isn’t very good.

> **ChatGPT addition: what that comparison establishes**
> 
> The larger model’s improvement does not isolate the effect of hierarchy from the effect of additional parameters or different settings. The matched-parameter result and the larger-model result answer different questions.
> 
> A comparison would be easier to interpret with a consistent evaluation harness: fixed data splits and context lengths, parameter counts and compute reported, comparable tuning, and repeated runs where the differences are small.

## Residual connections, convolutions, and what remains

He skipped the paper’s residual and skip connections. Is there useful intuition I’m missing before building up to transformers?

> **ChatGPT addition: residual and skip connections**
> 
> A residual block has the form
> 
> \[ h_{\ell+1}=h_\ell+F_\ell(h_\ell), \qquad \frac{\partial h_{\ell+1}}{\partial h_\ell} = I+J_{F_\ell}(h_\ell). \]
> 
> It lets a layer learn a change to the existing representation, with an identity path available for both activations and gradients. That helps explain why residual connections are useful when building deep networks, including transformers. It does not guarantee that gradients cannot vanish or explode.
> 
> WaveNet also uses skip connections that send intermediate features toward the output, giving the prediction access to features computed at multiple depths.

He also didn’t implement the convolution optimization. Looking at the paper’s picture, all those top prediction nodes can be computed in parallel! And computations for overlapping contexts can be reused.

> **ChatGPT addition: parallel training versus sequential generation**
> 
> For a width-two causal convolution with dilations \(1,2,4,\ldots,2^{L-1}\), the receptive field is
> 
> \[ R_L=1+\sum_{\ell=0}^{L-1}2^\ell=2^L. \]
> 
> For example, a simplified layer can combine states at \(t\) and \(t-2^\ell\). Unlike the downsampling tree, the convolution computes features at every sequence position. Overlapping prediction windows can therefore share intermediate computations.
> 
> During training, the observed sequence is available, so predictions for all target positions can be computed in parallel within the causal graph. Layers still depend on earlier layers. During ordinary autoregressive generation, later samples depend on earlier generated samples, so generation remains sequential.
> 
> Efficient kernels implement these operations, but parallel computation and reuse are properties of the computational structure, not just of moving a Python loop into CUDA.

Still haven’t set up an experimental/evaluation harness yet.

## Mathematical addendum: parameters and computation

> **ChatGPT addition**
> 
> Here is the napkin comparison made precise. Consider the simplified pair-grouping tree, with:
> 
> - \(N=2^L\) input characters;
> - embedding dimension \(d\), hidden width \(H\), and vocabulary size \(V\);
> - separate weights at different levels, shared across groups within each level;
> - biases and batch normalization omitted from the count.
> 
> The first level has a \(2d\to H\) map, and each of the remaining \(L-1\) levels has a \(2H\to H\) map. Including embeddings and the output map,
> 
> \[ P_{\text{tree}} = Vd+2dH+2(L-1)H^2+HV. \]
> 
> A flat MLP with the same number of hidden layers, all of width \(H\), has
> 
> \[ P_{\text{flat}} = Vd+NdH+(L-1)H^2+HV. \]
> 
> For fixed \(V,d,H\), this gives the \(O(\log N)\) versus \(O(N)\) parameter comparison. It depends on sharing and on this particular choice of widths and depths.
> 
> But each tree map is applied at many groups. Counting dense multiply-accumulate operations for one context, excluding embedding lookup and nonlinearities, gives
> 
> \[ C_{\text{tree}} = NdH+(N-2)H^2+HV, \]
> 
> whereas the matched-depth flat MLP uses
> 
> \[ C_{\text{flat}} = NdH+(L-1)H^2+HV. \]
> 
> So the tree’s smaller parameter count does not imply less computation. Its benefit is that many applications reuse a small set of parameters while preserving separate intermediate representations.
> 
> These formulas describe the lecture’s simplified tree. They are not general complexity formulas for the original WaveNet or for transformers.

## Sources for the ChatGPT additions

- [Karpathy’s makemore part 5 notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb)
- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572)
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [HyperNetworks](https://arxiv.org/abs/1609.09106)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)