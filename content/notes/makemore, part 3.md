---
date: 2026-10-11
tags:
  - makemore
confidence: ""
status: ""
---
*AI disclosure: I used ChatGPT to edit my original free-write and add technical context. Substantive AI additions are marked throughout.*

*Longer calculations are collected in a mathematical addendum at the end.*

## What this lecture is trying to solve

Btw, the neural language-model paper was from 2003, wow. Next steps are RNNs, GRUs, and then transformers, but this video is still focused on MLPs.

> **ChatGPT addition (historical context):** The 2003 paper is Bengio et al.'s _A Neural Probabilistic Language Model_, which motivates the embedding-plus-MLP language model. MLPs themselves are older; 2003 is not when the architecture was invented.[^1]

The perspective of “these networks are not easy to optimize” leads us to studying activations and the gradients flowing through them. And future architectures work on this too, which is interesting. Right, if this model is already nonlinear, why not just scale it totally? Why new architectures after this?

> **ChatGPT addition (context):** Expressiveness, trainability, and efficient use of context are separate questions. Increasing an MLP's size doesn't automatically make it easy to optimize or give it an efficient way to handle long sequences. The later architectures change how information is represented and combined, as well as how gradients propagate.

### A question about randomness

Recall the mini-batch idea: a noisy gradient approximation, with a computational advantage in certain cases, using a batch B much smaller than the dataset.

But this introduces new randomness into training, beyond initialization. And of course we use randomness in inference for sampling. I remember reading about watermarking methods that embed a statistical signal in inference sampling, and found that an interesting place to replace randomness with structured pseudorandomness. Is there any point in replacing training randomness with structured pseudorandomness?

> **ChatGPT addition (clarification):** Ordinary software training already uses seeded pseudorandom generators for initialization and sampling. Changing the seed changes the sequence of batches and can change training outcomes. Watermarking adds a particular keyed structure to generation; using pseudorandomness in training does not by itself create a watermark.[^2]

## Initialization: what loss should we expect?

What loss should we expect at initialization? Compare with predicting uniformly at random. There are 27 characters, so that's less than five bits of surprise when seeing the next character.

Eeks, he's using natural logs, so this is nats, not bits, which is why he got 3.3 instead.

> **ChatGPT addition (the baseline):**
> 
> \[ -\log(1/27)=\log 27\approx3.296\text{ nats} \approx4.755\text{ bits}. \]
> 
> This is the exact loss of a uniform next-character predictor, not an upper bound on arbitrary models' losses. A model that confidently assigns very low probability to the correct character can do much worse.

The initial network can be very confidently wrong if there are large differences in logits. Remember, a difference in logits is a log-ratio of probabilities. So any common constant value for the logits gives a uniform distribution; let's standardize to a goal of logits close to zero initially.

**Question:** This seems to save the very easy steps at the start that bring logits closer to one value. How valuable is it really?

> **ChatGPT addition:** A reasonable output scale avoids starting with arbitrary extreme preferences. But the more consequential problem comes next: if hidden units are saturated, the gradients needed to improve the representations can be very small. Initialization affects which parts of the model can learn, not just the initial loss.

How should we initialize the weights to achieve this? Start by making the output weights very small. Why not set them to zero? There's a risk, to be found later.

> **ChatGPT addition (distinguishing layers):** Making the final layer small and keeping hidden-layer weights at an appropriate scale are different choices. Making every layer tiny can also shrink the signal too much. Initializing all hidden units identically can preserve symmetry. Setting only the final matrix to zero is a different case: it initially blocks the gradient to earlier layers, but a subsequent update can unblock it if the hidden features are already distinct.

## Tanh saturation and vanishing gradients

Another issue is that the activations are close to +1 or −1, meaning they're saturated. This happens when the preactivations aren't restricted to an appropriate scale.

Renormalizing all gradient updates to some fixed size doesn't entirely solve this. Some early weights might be affected by saturated units while later weights aren't. A global rescaling can still mostly update the later weights, leaving the early, poorly initialized part learning very slowly.

I explored this in [a follow-up conversation](https://chatgpt.com/share/6ac82a27-6408-83e8-95ba-92757fe5acf9). A helpful toy optimization example would give me intuition for how this could slow convergence from something like n² to n³ in a contrived setting.

> **ChatGPT addition (the local sensitivity):** For \(h=\tanh(a)\),
> 
> \[ \frac{dh}{da}=1-h^2=\operatorname{sech}^2(a). \]
> 
> Large \(|a|\) makes this derivative tiny. Normalizing the whole parameter gradient changes its length, but not the relative sizes of its components. Normalizing individual blocks can help in some cases, but it doesn't automatically recover useful directions when their components have been attenuated differently. A toy conditioning calculation appears in the addendum below.

### “Dead” neurons

The concept of a “dead neuron”: its activation is saturated on ALL inputs. I see, because we use the gradients from the inputs in the batch to measure the next step. If the unit is unsaturated on some examples, those examples can still give it a learning signal.

A pretty simple initialization solution is to make the weights feeding into the preactivations appropriately smaller.

> **ChatGPT addition (terminology):** A finite tanh preactivation has a nonzero derivative mathematically, so “dead” here usually means effectively stuck because the gradient is extremely small, or numerically zero. This differs from a ReLU unit that is strictly negative on every relevant input and has exactly zero activation derivative there. Saturation on a single batch also doesn't establish saturation on the entire dataset.

## Working backward from the desired activation scale

The more principled way to control activations is to do it mechanistically. Given the distributions from which x and W are sampled, we can compute the scale of Wx using probability tricks.

This might blow up the standard deviation of Wx compared with x. So divide the weight scale by sqrt(fan-in), because that's how the standard deviation grows. We work backward from the desired preactivation scale to choose an appropriately smaller weight initialization.

And we also want to think about the effect of the nonlinearity. For ReLU, something roughly halves, so we use sqrt(2/n).

> **ChatGPT addition (the variance calculation):** For one preactivation,
> 
> \[ a=\sum_{j=1}^{n}w_jx_j, \]
> 
> assume independent zero-mean weights, independent of the inputs, with \(\mathbb E[w_j^2]=\sigma_w^2\), and inputs with a common second moment \(\mathbb E[x_j^2]=q\). Then
> 
> \[ \operatorname{Var}(a)=n\sigma_w^2q. \]
> 
> Choosing \(\sigma_w^2=1/n\) preserves this scale across a linear layer. A gain g gives \(\sigma_w=g/\sqrt n\).
> 
> For a symmetric preactivation distribution,
> 
> \[ \mathbb E[\operatorname{ReLU}(a)^2]=\tfrac12\mathbb E[a^2]. \]
> 
> It is the **second moment** that halves, not generally the variance, because ReLU has a positive mean. This motivates \(\sigma_w^2=2/n\) for forward signal propagation. The independence assumptions are initialization approximations, not identities that remain true throughout training.[^3]

Actually, we don't just want the activations to have a sensible scale. The reason to investigate this was to make sure gradients also propagate sensibly. So we want a mechanistic interpretation of that too.

The [Kaiming initialization paper](https://arxiv.org/abs/1502.01852), which Andrej said isn't necessarily accessible, analyzes this.

> **ChatGPT addition (forward versus backward):** He et al. analyze both directions. Preserving forward second moments uses fan-in; the corresponding backward calculation involves fan-out and activation derivatives. A choice that behaves well in the forward pass does not automatically guarantee well-conditioned backward propagation, especially when layer widths differ or the network is deep.[^3]

### Why a tanh gain of 5/3?

Tanh gain is 5/3? That's an interesting number. ChatGPT initially told me that “no one knows why”, which made this feel even more mysterious.

> **ChatGPT addition (correcting that earlier answer):** “No one knows why” is too strong. PyTorch recommends \(5/3\) for tanh, but that recommendation is not an exact, distribution-independent variance-preservation identity. For a proposed input distribution, the output second moment can be studied through
> 
> \[ \mathbb E[\tanh^2(gX)]. \]
> 
> It depends on the distribution and scale of X, and preserving forward moments and backward sensitivities are different objectives. The documentation establishes the recommendation, not a universal theorem deriving this particular number.[^4]

**Meta: robustness versus understanding.** Back in the day, getting initialization exactly right was extremely important. Now we have residual connections, batch norm, and optimizers like RMSprop and Adam.

Nice, this seems like a more systematic way to make training work. Or wait, maybe these innovations are so robust to the details that we no longer need the same mechanistic understanding for good performance, which I guess isn't entirely good news :(

## Batch normalization

Wait, one of the batch-norm authors is Christian Szegedy, the same person who wrote the mathematics-graduating essay!!!

> **ChatGPT addition (reference):** The authors are Sergey Ioffe and Christian Szegedy. The essay's title is _Is Mathematics Over, or Just Graduating?_[^5]

### What distribution do we actually want?

This paper helped make deep-network training more reliable. But I want more math supporting why the desired activation distribution is a good idea in the first place.

It seems a bit arbitrary, almost a cop-out, to say we want them normal. Do we want them to always be roughly normal during training, or just at initialization? If it's just initialization, why do this every step? Why not normalize once using the training data and then let training continue normally?

Seems like a question worth getting to the bottom of: “What is the right distribution of activations we want, why, and how do we achieve it?”

I guess the idea is that we want preactivations around a sensible mean and variance, rather than requiring normality. For tanh, we want to avoid everything being saturated, but we also don't want the network to behave almost linearly everywhere.

At initialization, sums can look approximately Gaussian. Really, it's the scale of the distribution that we care about, and the appropriate scale depends on the activation function.

> **ChatGPT addition (two corrections):** Approximate Gaussianity of sums is a central-limit-theorem intuition, not the law of large numbers. Batch norm itself does not make a distribution Gaussian: it centers and rescales it. Nor is mean zero and variance one a universally optimal distribution for every activation. It provides a standardized intermediate representation with a learned adjustment afterward.

### The batch-norm equations

> **ChatGPT addition (equations corresponding to the screenshot):** For one feature's preactivations \(a_1,\ldots,a_B\) across a batch,
> 
> \[ \mu_{\mathcal B}=\frac1B\sum_i a_i,\qquad \sigma_{\mathcal B}^2=\frac1B\sum_i(a_i-\mu_{\mathcal B})^2, \]\[ \widehat a_i=\frac{a_i-\mu_{\mathcal B}} {\sqrt{\sigma_{\mathcal B}^2+\epsilon}}, \qquad y_i=\gamma\widehat a_i+\beta. \]
> 
> The operation is applied separately to each feature. Batch statistics are computed from the current batch; \(\gamma,\beta\) are learned parameters shared across batches. The normalized feature has mean zero and variance \(\sigma_{\mathcal B}^2/(\sigma_{\mathcal B}^2+\epsilon)\), rather than exactly one when \(\epsilon>0\).

### A connection to conditioning in linear regression

**Connection: ISLP and optimization.** Once again, the idea is to get some standardization across dimensions. A nice example is linear regression: if coordinates have very different variances, some directions can be steep and others shallow. Interestingly, this comes from looking at the Hessian, so again I need better intuition for these second-order effects.

> **ChatGPT addition (the calculation from the other screenshot):**
> 
> \[ L(w)=\tfrac12\mathbb E[(w^\top x-y)^2], \qquad \nabla^2L(w)=\mathbb E[xx^\top]. \]
> 
> If inputs are centered and their coordinates are uncorrelated,
> 
> \[ \nabla^2L(w) =\operatorname{diag}\bigl(\operatorname{Var}(x_1),\ldots, \operatorname{Var}(x_d)\bigr). \]
> 
> Different variances give different curvatures. A learning rate small enough for the steep directions can make progress slow in the shallow ones. Rescaling nonconstant coordinates to unit variance makes this Hessian the identity under these assumptions.
> 
> This benefit needs no Gaussian assumption. If coordinates are correlated, standardizing their individual variances does not make the Hessian the identity; batch norm likewise does not whiten all feature correlations.

My initialization intuition was indeed pointing toward a real idea: a related method called LSUV rescales layers to have unit output variance before training. It doesn't require a pass over every example.

> **ChatGPT addition (context):** LSUV combines orthonormal initialization with data-based layer rescaling using a batch. It adjusts the starting point; batch norm remains part of the computation during training. These are related strategies, not equivalent algorithms.[^6]

### Why normalize repeatedly?

Another thing batch norm does is maintain an invariant during training: shifting a feature across the whole batch, or positively rescaling it, gets canceled out.

Why is this something we want? Maybe because we care about differences within the batch rather than similarities? Hmm, let me finish seeing the algorithm before committing to that.

> **ChatGPT addition (the invariant):** Ignoring \(\epsilon\), for \(c>0\),
> 
> \[ \widehat{(ca+d\mathbf 1)}=\widehat a. \]
> 
> The downstream computation is insensitive to these changes in the incoming feature's offset and scale. A one-time normalization doesn't preserve this property as weights change. With \(\epsilon>0\), shift invariance remains exact, while scale invariance is approximate.

“Research has found it makes the loss landscape smoother.” Bruh, what does this even mean? Is there a more mathematically precise way of saying it?

> **ChatGPT addition (what “smoother” means here):** A standard quantitative notion is a bound on changes in the gradient:
> 
> \[ \|\nabla L(\theta+\delta)-\nabla L(\theta)\| \le K\|\delta\|. \]
> 
> Smaller K means the local gradient remains predictive over larger moves. Santurkar et al. study related measures empirically and derive bounds under specified conditions. Their result is not that batch norm universally lowers one global smoothness constant for every network and parameter setting.[^7]

### Why put scale and shift back in?

Ah yes, Andrej says we don't want to force fixed unit statistics all the way through training. Normalizing without the learned scale and shift imposes a restriction.

Worth doing an experiment to see how models learn with and without that final scale and shift. Might need a more complicated setup than one hidden layer to see the difference.

Ohhh, I was making a mistake: \(\beta,\gamma\) are shared across batches, whereas the mean and variance used for normalization come from the current batch.

ALSO, these parameters are learned via gradient descent rather than observed from the data. They are chosen to improve prediction. So even with one batch containing all training data, this could be useful.

> **ChatGPT addition (clarification):** For each batch, the transformed feature has mean \(\beta\) and variance
> 
> \[ \gamma^2\frac{\sigma_{\mathcal B}^2} {\sigma_{\mathcal B}^2+\epsilon}. \]
> 
> Thus \(\gamma,\beta\) give a shared, learned adjustment of feature scale and offset. They are not estimates of the raw feature's global dataset statistics. Learning can deliberately move the activation into a different operating regime; standardization does not force it to remain at zero mean and unit variance.

The follow-up explanation helped me separate how the method was developed from what we can now explain about it. There are useful mathematical mechanisms, but not a complete prediction of its effect in an arbitrary deep network.

### Batch-dependent noise

One weird, apparently beneficial thing: the forward pass on an input now depends on its batch statistics. Which other examples it is sampled with creates some jitter.

This can act as a “regularizer”, somewhat like data augmentation: another way to make overfitting harder.

> **ChatGPT addition (scope):** Given the same complete batch and parameters, this forward computation is deterministic. It is the representation of one example that changes when its batch companions change. That variability can regularize training, but it is not automatically beneficial in every setting.

## What the derivative tells us

A few things I learned through [this conversation](https://chatgpt.com/share/6acbc00d-da2c-83e8-b2a4-8b565e7c6415):

Gradients through batch norm can be interpreted as removing particular ineffective changes and appropriately rescaling what remains.

Its derivative removes motion that changes only the center or length of the batch's preactivation vector. The signal carried backward then focuses on how to change the “pattern”, while \(\gamma,\beta\) can handle scale and offset. Why does separating things out like this help? Idk.

Wait, the math is so pretty! It's a rescaling of the upstream gradient with projections onto two perpendicular directions subtracted: the all-ones direction and the radial direction of the centered vector.

> **ChatGPT addition (the projection formula):** Let \(g_i=\partial L/\partial y_i\), \(s=\sqrt{\sigma_{\mathcal B}^2+\epsilon}\), and use a bar for the batch average. Then
> 
> \[ \frac{\partial L}{\partial a_i} =\frac{\gamma}{s} \left(g_i-\overline g-\widehat a_i\, \overline{g\widehat a}\right). \]
> 
> When \(\epsilon=0\) and the variance is nonzero, \(\widehat a\) is perpendicular to \(\mathbf1\) and has squared length B. In vector form,
> 
> \[ \nabla_aL= \frac{\gamma}{s} \left(I-\frac{\mathbf1\mathbf1^\top}{B} -\frac{\widehat a\widehat a^\top}{B}\right)g. \]
> 
> This is the exact projection interpretation in the idealized case: remove common-shift and centered-scale components. For \(\epsilon>0\), the first scalar formula remains exact, but the radial component is not completely projected away.[^7]

There's also an effective learning-rate change through batch norm, and we can prove convergence in some cases.

The [Cai, Li, and Shen paper](https://proceedings.mlr.press/v97/cai19a.html) seems worth understanding. In deep networks, keeping unit variance isn't enough: there's singular-value stuff I haven't looked into, and architectural changes like residual connections can help.

> **ChatGPT addition (scope of the theory):** Cai et al. analyze batch-normalized gradient descent on ordinary least squares. Arora, Li, and Lyu analyze automatic rate tuning for scale-invariant parameters under their assumptions. These are specific optimization results, not guarantees of global convergence for arbitrary deep networks. The effective-rate mechanism is derived in the addendum below.[^8]
> 
> Individual features having controlled variance does not ensure that products of layer Jacobians preserve every direction. Some directions can shrink while others expand, even when a simple variance summary looks healthy.

## Training and inference are different here

One trick to keep inference deterministic is to estimate mean and variance from training data and then use fixed values at inference.

Tbh this seems very important for efficiency as well: I want to run inference on a single input! You can also estimate these statistics in a running manner during training.

Ah, an interesting efficiency subtlety: we can remove the bias from the layer immediately before batch norm, because it's canceled out.

> **ChatGPT addition (implementation context):** Standard batch-norm inference uses frozen running statistics; calibration with training data is another option. In evaluation mode, normalization no longer depends on the current batch companions. Autoregressive sampling can still be random afterward.
> 
> A bias added directly before batch norm cancels during training's mean subtraction, so it has zero data-loss gradient there. This motivates omitting it from the architecture. Deleting a bias from an already trained model with frozen statistics is a different operation and needs a compensating adjustment.[^9]

He showed that ResNets use similar ingredients: an affine layer, batch norm, a nonlinearity, and residual connections, which haven't yet been covered.

Also, the PyTorch example used weights drawn from a uniform distribution instead of a normal one. I think that doesn't matter much for this first scale calculation, because what we're focusing on is the statistics of the activations.

> **ChatGPT addition:** The second-moment calculation can apply to either distribution with suitable centering and scale. This doesn't mean their higher moments and tails never matter.

Oh, interesting: group normalization and layer normalization are alternatives, and batch norm has some footguns.

> **ChatGPT addition (context):** Layer norm and group norm compute statistics within an example, avoiding dependence on batch companions. Batch norm needs care with small batches, training/evaluation modes, and running statistics. The alternatives address different settings; this is not a universal ranking of normalization methods.[^10]

Next, recurrent neural networks, and he says “unrolling the loop” motivates all this activation analysis. Interested to see what he means.

> **ChatGPT addition:** Unrolling represents successive recurrent steps as a chain of computations with shared parameters. Backpropagation through that chain multiplies successive Jacobians, making gradient growth and decay important over time as well as depth.

## Stacking more layers and looking at diagnostics

In the PyTorch-esque API walkthrough, he builds a roughly 50K-parameter model by stacking layers.

Interesting experiment: with six layers and WITHOUT batch norm, removing the gain adjustment makes the activation scale collapse toward zero.

> **ChatGPT addition (correcting “saturation converges to zero”):** This is shrinking activation scale, not saturation at \(\pm1\). Near zero, tanh behaves almost linearly, but it still compresses magnitudes slightly. Repeating the layers can progressively shrink the signal. Vanishing activations and saturated activations are different failure modes.

So before batch norm, the gain was important to set correctly, and its effect was magnified by depth.

He also talks about a deep network with only linear layers. Even though the forward pass collapses into one linear map, backprop makes the training dynamics interesting. He didn't say much more about that.

> **ChatGPT addition (connection to the part 2 addendum):** The computed function can be linear while its factorized parameterization has nonlinear training dynamics. Gradients for one matrix depend on the other matrices, just as the updates to C and U depended on each other in the simplified embedding model.

### Update-to-parameter ratios

The 0.001 or 0.01 update-to-data ratio still seems like a magic number. I have no idea why those are specifically the right orders of magnitude. There must be some math here, no?

> **ChatGPT addition (what the diagnostic measures):** In the lecture's standard-deviation version, the ratio is roughly
> 
> \[ r=\frac{\operatorname{std}(\Delta\theta)} {\operatorname{std}(\theta)}, \qquad \Delta\theta=-\eta\nabla_\theta L \]
> 
> for a parameter tensor. RMS-based versions are also used. Values near 0.001 or 0.01 describe small relative updates; they are heuristics, not universal optimal constants. The ratio is ill-defined when its denominator is zero and needs care for tensors with nearly zero spread.
> 
> It measures relative parameter movement, not directly the change in predictions. For scale-invariant weights, a large weight norm can lower this ratio and reduce the effective directional step.

Nice, batch norm makes these experiments much more robust to bad initialization, including gain. But the scale of updates is still affected, and I'm trying to understand what that measure tells us.

He acknowledges that initialization and backpropagation aren't simply “solved” at this point.

> **ChatGPT addition (qualification):** Batch norm helps with several problems; it doesn't fix everything. Learning rates, depth, batch statistics, and the learned gains can still affect optimization. Robust performance in this experiment is not a guarantee for every network.

## Connection: mechanistic interpretability

Wow, [this paper](https://arxiv.org/abs/2301.05217) looks insanely interesting. Mech interp is trying to reverse-engineer the circuits within a model. So a transformer architecture can learn circuits for tasks like addition mod p, and in this example it uses Fourier structure rather than a binary adder.

What other circuit problems might be natural enough to investigate this way? Using ablations to test whether an interpretation is faithful is quite nice. A good thing about digital intelligence is that you can do surgery on it :)

I've also been reading Neel Nanda's arguments about prioritizing safety-relevant interpretability work rather than assuming toy problems are the best research direction. I would bet he still considers them helpful pedagogical tools, though!

> **ChatGPT addition (scope of the example):** The modular-addition paper reverse-engineers a particular learned mechanism using Fourier features and trigonometric identities. It doesn't establish that every transformer solves modular addition this way or that interpreting this toy task resolves a safety problem. Pedagogical usefulness and research priority are separate judgments.[^11]

## Mathematical addendum

### A toy example of slow progress from a small sensitivity

> **ChatGPT addition (illustrative calculation):** Consider a local linearized prediction
> 
> \[ r(\theta)=(\varepsilon\theta_1,\theta_2), \qquad F(\theta)=\tfrac12\|r(\theta)\|^2 =\tfrac12(\varepsilon^2\theta_1^2+\theta_2^2), \]
> 
> where \(0<\varepsilon<1\). The small sensitivity \(\varepsilon\) is analogous to a small activation derivative.
> 
> With ordinary gradient descent and learning rate 1,
> 
> \[ \theta_1^{(t)}=(1-\varepsilon^2)^t\theta_1^{(0)}, \qquad \theta_2^{(1)}=0. \]
> 
> Reducing the first coordinate by a factor \(\delta\in(0,1)\) takes
> 
> \[ t=\Theta\!\left(\varepsilon^{-2}\log(1/\delta)\right) \]
> 
> as \(\varepsilon\) becomes small. Choosing \(\varepsilon=n^{-1}\) gives n² scaling; choosing \(\varepsilon=n^{-3/2}\) gives n³ scaling.
> 
> This makes the sensitivity-versus-curvature issue quantitative for ordinary gradient descent. It is not a proof that normalized-gradient methods cannot help: once only one nonzero component remains, normalization can amplify it. The concern with a single global normalization is that it preserves the relative suppression while multiple components are competing.

### Batch norm and the effective directional learning rate

> **ChatGPT addition (scale-invariant parameterization):** Consider incoming weights w immediately followed by batch normalization, with other parameters fixed. Ignore \(\epsilon\), weight decay, and numerical effects, and assume nonzero batch variance. Positive scaling leaves the loss unchanged:
> 
> \[ L(cw)=L(w),\qquad c>0. \]
> 
> Differentiating with respect to c gives
> 
> \[ w^\top\nabla_wL=0. \]
> 
> An ordinary gradient step therefore satisfies
> 
> \[ \|w-\eta\nabla_wL\|^2 =\|w\|^2+\eta^2\|\nabla_wL\|^2. \]
> 
> Let \(r=\|w\|\), \(v=w/r\), and let \(G(v)\) be the tangent gradient of the loss as a function of direction on the unit sphere. Scale invariance gives \(\nabla_wL=G(v)/r\). Consequently,
> 
> \[ w^+=r\left(v-\frac{\eta}{r^2}G(v)\right), \]
> 
> and the new direction is
> 
> \[ v^+= \frac{v-(\eta/r^2)G(v)} {\|v-(\eta/r^2)G(v)\|}. \]
> 
> Thus the coefficient of the directional gradient before renormalization is
> 
> \[ \eta_{\mathrm{effective}}=\frac{\eta}{\|w\|^2}. \]
> 
> Larger updates grow the weight norm, which reduces this coefficient on later steps. This is an automatic feedback mechanism, not a claim that every such adjustment is optimal. Finite \(\epsilon\) makes scale invariance approximate; momentum, Adam, and weight decay change these exact plain-gradient-descent identities.[^8]

## References for ChatGPT additions

[^1]: Bengio et al., [_A Neural Probabilistic Language Model_](https://jmlr.org/papers/v3/bengio03a.html), 2003.  
[^2]: PyTorch, [_Reproducibility_](https://docs.pytorch.org/docs/stable/notes/randomness.html).  
[^3]: He et al., [_Delving Deep into Rectifiers_](https://arxiv.org/abs/1502.01852), especially the forward and backward initialization analysis.  
[^4]: PyTorch, [_torch.nn.init_](https://docs.pytorch.org/docs/stable/nn.init.html), including the recommended tanh gain.  
[^5]: Ioffe and Szegedy, [_Batch Normalization_](https://arxiv.org/abs/1502.03167); Szegedy, [_Is Mathematics Over, or Just Graduating?_](https://docs.google.com/document/d/e/2PACX-1vTSh-pyNP3Gi99WMmsinnLmE9V5CDI0HEm6WGbIPMNt3V5SGlClHF8-BetotKHOImrvQDSXmFdiOw8D/pub).  
[^6]: Mishkin and Matas, [_All you need is a good init_](https://arxiv.org/abs/1511.06422).  
[^7]: Santurkar et al., [_How Does Batch Normalization Help Optimization?_](https://arxiv.org/abs/1805.11604), including the gradient-through-normalization formula.  
[^8]: Cai, Li, and Shen, [_A Quantitative Analysis of the Effect of Batch Normalization on Gradient Descent_](https://proceedings.mlr.press/v97/cai19a.html); Arora, Li, and Lyu, [_Theoretical Analysis of Auto Rate-Tuning by Batch Normalization_](https://arxiv.org/abs/1812.03981).  
[^9]: PyTorch, [_BatchNorm1d_](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html). The original batch-norm paper also discusses frozen inference statistics and the redundant preceding bias.  
[^10]: Ba, Kiros, and Hinton, [_Layer Normalization_](https://arxiv.org/abs/1607.06450); Wu and He, [_Group Normalization_](https://arxiv.org/abs/1803.08494).  
[^11]: Nanda et al., [_Progress measures for grokking via mechanistic interpretability_](https://arxiv.org/abs/2301.05217).

Lecture materials: Karpathy's [makemore part 3 notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb).