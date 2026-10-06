---
date: 2026-10-06
tags:
  - makemore
confidence: ""
status: ""
---
*AI disclosure: I used ChatGPT to edit my original free-write and add technical context. Substantive AI additions are marked throughout.*
## Recap and embeddings

Recap: we used one-hot encodings and saw the model converge toward the counting model. It's not a difficult proof that this is what we'd expect in terms of minimal loss. The encoding treated an input as selecting a row and outputting the next-character probability distribution. Cross-entropy had the intuition of two parts: the target's own entropy and the error in modeling it with an approximation.

> **ChatGPT addition (technical clarification):** This comparison concerns the unregularized bigram model. Its loss is convex in the logits, so optimization is relatively simple, but convergence still depends on the learning rate and other choices. Exact zero probabilities are only attained as limits of finite softmax logits.

To generalize to taking in multiple characters to predict the next one, this would require a really large encoding/output-probability matrix. So instead we map inputs to a high (but not so high) dimensional vector space where similarity can allow sharing context and learning a representation.

Ofc this works when the underlying data has lots of correlations in context, although I don't even know what that means precisely. Right, the data has examples of sentences where one word is fungible with another. And this gives its meaning in the first place. The coding would be much different, no?

Ah yes, reminder that we are working at the character level in makemore.

> **ChatGPT addition (context):** The word-level example motivates learning representations from usage. Here the same idea is applied to characters: learning how a character behaves in different name contexts can help across many contexts. The embedding space is typically much smaller than the one-hot space. Useful similarity is learned through the prediction objective; Euclidean proximity is not explicitly enforced.

Okay, this clearly has more potential than “counting occurrences in training data” in terms of “for a given prediction, how much information in the dataset did we use?” Now we can extract information relevant to this example from related examples. So I guess the idea is to stop treating each explicit context separately and instead first learn some patterns in the input itself?

**Technical takeaway:** The one-hot picking pattern still exists. It's just separated from the actual computation into an embedding matrix of size |vocabulary| × |embedding dimension|. So the parameters explicitly store the embedded vector of each character.

While writing this, I also realized that it doesn't seem to make sense to instead have some log|vocabulary|-sized representation a priori that is linearly meaningful. Maybe this initial embedding dictionary is the most natural and sensible way to proceed.

> **ChatGPT addition (qualification and notation):** A binary code of length \(\lceil\log_2 V\rceil\) can identify each of \(V\) symbols, so it is possible to use one. But its bit patterns impose an arbitrary structure, and a linear map of those bits cannot freely assign an independent vector to every symbol. A lookup table directly learns those vectors:
> 
> \[ C\in\mathbb R^{V\times d},\qquad e_i=C_{i,:}. \]
> 
> For a context of \(k\) characters, concatenate the embeddings:
> 
> \[ u=[e_{x_1};\ldots;e_{x_k}]\in\mathbb R^{kd}. \]
> 
> This preserves the order of the characters. The remaining network is shared across contexts.

## Hidden layer and nonlinearity

**Question:** The role of the hidden layer, which takes the embedded context and shrinks (?) it, is very unclear and unintuitive to me right now. I need details. Hmm, or maybe it expands! With 30-dimensional embeddings and a three-character context, that's a 90-dimensional input, but the hidden layer could output 100 dimensions.

Okay, in this architecture there is still a fixed “block size”, like three, that determines the length of context.

It's also not clear to me why a tanh is required before feeding into W2. Is it to “normalize” values before entering the next matrix, so that the weights W2 converges to are also “normal” looking? And W2 of course has its output dimension determined by the need to produce logits for all characters.

> **ChatGPT addition (technical explanation):** Using column-vector notation, the network is
> 
> \[ h=\tanh(W_1u+b_1),\qquad z=W_2h+b_2, \]
> 
> with \(W_1\in\mathbb R^{m\times kd}\) and \(W_2\in\mathbb R^{V\times m}\). The hidden width \(m\) can be smaller or larger than \(kd\); it is not inherently a compression step. The 90-to-100 example above is hypothetical.
> 
> Tanh bounds its outputs, but its essential role here is **nonlinearity**, not normalization. Without it,
> 
> \[ z=W_2W_1u+(W_2b_1+b_2), \]
> 
> which is just one affine map of the concatenated embeddings. Hidden units can instead learn nonlinear responses to combinations of context features. Tanh is one choice; the need for a nonlinearity does not uniquely justify tanh.

**Follow-up question:** If we didn't add nonlinearity but still imposed an initial compression matrix \(C\in\mathbb R^{V\times d}\), would that still achieve our intended effect of sharing context? Would it have hope of improving over the bigram language model?

> **ChatGPT addition (parameter sharing without a hidden nonlinearity):** Yes to both. Collapsing the linear layers does not remove the learned embeddings or the use of multiple context positions. The resulting logits have the form
> 
> \[ z(x_1,\ldots,x_k)=b+\sum_{t=1}^{k}A_t e_{x_t}, \]
> 
> where \(A_t=W_2W_{1,t}\) and \(W_{1,t}\) is the part of the first-layer matrix acting on position \(t\). The same embedding and positional map are reused across many contexts. Each position's character-to-logit table is
> 
> \[ M_t=CA_t^\top,\qquad \operatorname{rank}(M_t)\le d. \]
> 
> So there is still sharing through learned, low-rank **logit tables**. The probability tables need not have the same rank after softmax. The model is affine in the concatenated embeddings, but learning both \(C\) and the subsequent matrices is still a factorized optimization problem; it is not generally jointly convex in those parameters.
> 
> It can also exploit information a bigram cannot. In the interior of a long sequence
> 
> \[ \texttt{aabbaabbaabb}\ldots, \]
> 
> either next character is equally likely given only the previous character. Given the previous two, the next character is the opposite of the older one. The linear model can represent that rule by using the older character and ignoring the newer one. No hidden nonlinearity is needed.
> 
> The restriction is **additivity of contributions to the logits**. For two possible outputs,
> 
> \[ \log\frac{P(a\mid x_1,x_2)}{P(b\mid x_1,x_2)} =b'+f(x_1)+g(x_2). \]
> 
> Changing the first character changes these log-odds by the same amount regardless of the second character. Softmax remains nonlinear, so this is a restriction on logits and log-odds, not a claim that probabilities are additive.
> 
> For example, suppose the desired next character is \(a\) when the two context characters match, and \(b\) when they differ. Writing the log-odds as \(s_{ij}\), additivity forces
> 
> \[ s_{aa}+s_{bb}=s_{ab}+s_{ba}. \]
> 
> The desired rule requires both terms on the left to be positive and both on the right negative, which is impossible. A sufficiently wide nonlinear hidden layer can represent this interaction.
> 
> The embedding bottleneck already provides parameter sharing; the hidden nonlinearity expands the kinds of contextual combinations the model can represent. Neither guarantees better generalization. With one-character context, compression restricts the full bigram model; with longer context, the additional information can more than compensate for that restriction.

## Why softmax?

**Question:** Why do softmax in the first place? Why not keep logits as they are and expect the model to learn the underlying probability distribution when directly computing loss via cross-entropy?

Ah, because the mathematical cross-entropy formula expects probabilities. Otherwise it could make the logits (fake probabilities) all super high, like 100, without penalizing anything else. What we actually want is that raising the probability of one class takes away from the others.

There might be another way to enforce nonnegativity and a sum of one, but softmax is a smooth way of doing this. Is it the best in any sense? It is natural if you want these outputs to represent log-probabilities, up to an additive constant, because softmax exponentiates.

> **ChatGPT addition (equations and implementation distinction):**
> 
> \[ p_j=\frac{e^{z_j}}{\sum_\ell e^{z_\ell}},\qquad \log p_j=z_j-\log\sum_\ell e^{z_\ell}. \]
> 
> For target class \(y\),
> 
> \[ \ell(z,y)=-\log p_y=-z_y+\log\sum_j e^{z_j}. \]
> 
> The normalization term prevents us from reducing the loss simply by raising all logits together. In PyTorch, the cross_entropy function already accepts logits and performs the equivalent log-softmax/NLL calculation internally; we should not apply softmax first when using that function.[^1]

### The surprisingly clean gradient

Softmax plus cross-entropy also has a surprisingly clean gradient: for the correct class, increase its logit by \(1-\text{predicted probability}\); for the others, decrease the logit by the predicted probability. Very interesting. Almost seems like a dimension mismatch.

> **ChatGPT addition (precise version):**
> 
> \[ \frac{\partial\ell}{\partial z_j}=p_j-\mathbf 1\{j=y\}. \]
> 
> So a gradient-descent step **on the logits themselves**, with learning rate \(\eta\), would be
> 
> \[ z_j\leftarrow z_j+\eta\bigl(\mathbf 1\{j=y\}-p_j\bigr). \]
> 
> The gradient components sum to zero, consistent with softmax ignoring a constant shift of all scores. The network actually updates its parameters, rather than each example's logits independently, so the resulting logit changes also depend on the network's Jacobian.

From my understanding, this is a kind of counterbalancing force. Loss with respect to the correct-class probability has a gradient inversely proportional to that probability, so it wants to really bump up low probabilities. But the gradient of probability with respect to score, via softmax, also involves the probability, so it cancels out. The gradient of loss with respect to score is finally a balanced correction.

More generally, for exponential-family models, it's natural to see these score gradients comparing an expected outcome with an actual outcome.

> **ChatGPT addition (the cancellation):** For a hard target \(y\),
> 
> \[ \frac{\partial\ell}{\partial p_y}=-\frac1{p_y}, \qquad \frac{\partial p_y}{\partial z_j} =p_y\bigl(\mathbf 1\{j=y\}-p_j\bigr). \]
> 
> Multiplying gives the gradient above. The categorical softmax model is an exponential family: the gradient compares the model's expected one-hot outcome \(p\) with the observed one-hot outcome. This doesn't mean every weight in a deep network has this simple gradient.

### Temperature as an entropy coefficient

Another deep reason for scores → probabilities via softmax is that softmax solves an optimization problem: choose a probability distribution such that expected score + entropy is maximized.

That's very deep. What an interesting quantity to maximize. Surely you want to give a higher score a higher probability, but also you want the distribution to have high entropy, and therefore enough diversity? Why specifically this sum? What about weighting these two factors? Is that what temperature is?

Wow, T as a coefficient on entropy is EXACTLY the temperature. What a great guess.

> **ChatGPT addition (the variational statement):** For fixed scores \(z\), \(T>0\), and entropy \(H(p)=-\sum_jp_j\log p_j\),
> 
> \[ \underset{p\in\Delta_V}{\arg\max} \left\{\sum_jp_jz_j+T H(p)\right\} =\operatorname{softmax}(z/T), \]
> 
> where \(\Delta_V\) is the probability simplex. If \(q=\operatorname{softmax}(z/T)\), then
> 
> \[ \sum_jp_jz_j+T H(p) =T\log\sum_j e^{z_j/T}-T D_{\mathrm{KL}}(p\|q). \]
> 
> This proves the result immediately: the maximum occurs at \(p=q\). It explains softmax under this particular objective, rather than establishing that softmax is universally the best probability map. Also, this entropy tradeoff describes converting fixed scores into probabilities; it doesn't imply that cross-entropy training explicitly rewards output entropy.

For the second interpretation, it's a kind of answer to “choose the highest-entropy distribution consistent with a specified expected score”. Favor high scores, but retain as much uncertainty as possible.

> **ChatGPT addition:** That constrained maximum-entropy problem gives a Gibbs/softmax distribution too, with a multiplier chosen to meet the score constraint. The expected-score-plus-entropy formulation above is the corresponding penalized form. Positive temperature corresponds to favoring higher scores.

More details on these two connections are in my [follow-up conversation with ChatGPT](https://chatgpt.com/share/6ac53198-97d4-83e8-b6fa-52e13539cc77).

## How does this learn useful features?

**Question:** It really is a leap to say that “correcting an error in the final prediction teaches the hidden layers what features are worth representing”, from a generalization-goal point of view. It doesn't make sense to me why this would generalize. Even if the underlying data truly has some complicated relationship, how is this structure picking it up?

> **ChatGPT addition (mechanism versus guarantee):** Backpropagation supplies a mechanism for changing representations, not a guarantee of useful generalization. Writing \(a=W_1u+b_1\), the gradient returning to the hidden layer is
> 
> \[ \frac{\partial\ell}{\partial a} =\bigl(W_2^\top(p-e_y)\bigr)\odot(1-h^2). \]
> 
> This tells each hidden unit how changing its activation would affect prediction error. The gradient then flows to the embeddings. Because the same embeddings and weights participate in many examples, their updates must serve many contexts. That creates an opportunity to learn reusable structure. Whether it generalizes depends on the data, architecture, optimization, and regularization; lowering training loss alone doesn't establish it.

## Hard labels, distributions, and distillation

It feels like cross-entropy as a concept is being underutilized when it's used for a probability distribution versus a deterministic output, rather than an ideal distribution.

**Question:** If we rearrange the data where the same input appears multiple times into an empirical distribution, and then use cross-entropy, is that the same loss? Ah yes, we can treat the labels as an empirical distribution equivalently.

> **ChatGPT addition (grouping repeated contexts):** Let \(n_{xj}\) count occurrences of context \(x\) with next character \(j\), let \(n_x=\sum_jn_{xj}\), and define \(\widehat Q(j\mid x)=n_{xj}/n_x\). Then
> 
> \[ -\frac1N\sum_{x,j}n_{xj}\log P_\theta(j\mid x) =\sum_x\frac{n_x}{N} H\bigl(\widehat Q(\cdot\mid x),P_\theta(\cdot\mid x)\bigr). \]
> 
> The weighting by \(n_x/N\) is essential. Giving every distinct context equal weight would generally change the objective. A single hard label is already a point-mass target distribution, so cross-entropy is not being used incorrectly in the ungrouped version.

**Connection: distillation.** Model distillation uses this more fully. There we minimize KL divergence from a teacher's distribution to the student model. The teacher is the target, and we sample or take expectations according to it, but use the student's probabilities to judge the surprise.

In RL, the averaging is over the policy being trained, with a fixed reference model appearing in the KL penalty. I don't quite understand why it's done this way, but I'll worry about that when I come to RL.

Why this difference? They are answering two different questions.

Distillation: on answers the teacher produces, how much probability does the student assign? If the teacher frequently produces an answer the student never produces, it gets learned. The student “judges” the teacher here, which is counterintuitive.

RL: how good are the policy's answers? The reference “judges” the policy here, which feels more natural.

> **ChatGPT addition (important correction):** In common distillation, matching the teacher's full soft distribution minimizes \(H(Q_{\text{teacher}},P_{\text{student}})\), equivalently \(D_{\mathrm{KL}}(Q_{\text{teacher}}\|P_{\text{student}})\), since the teacher's entropy is fixed. We can compute an expectation over classes directly instead of sampling teacher answers.[^2]
> 
> In a common KL-regularized RLHF objective,
> 
> \[ \max_\theta\; \mathbb E_{y\sim P_\theta}[r(y)] -\beta D_{\mathrm{KL}}(P_\theta\|P_{\mathrm{ref}}), \]
> 
> with the prompt suppressed for readability. The **reward**, not the reference model, judges response quality. The KL term constrains drift and contains both the policy's and reference's probabilities; it is not merely expected surprise under the reference.[^3]

Hmm, maybe the better distinction is “learn to predict demonstrations from this source” versus “improve chosen behavior”. The two goals are different:

1. Imitation: “Learn to predict demonstrations from this source.” Average over the source's demonstrations.
2. Reward optimization: “Improve the consequences of your own choices.” Average over your own choices.

The two choices are now clearer, but I still don't necessarily see why one is right over the other in certain situations. Maybe the answer is that it isn't that crystal clear.

> **ChatGPT addition (framing from the follow-up):** The teacher/student roles don't force the KL direction. The goal, covering demonstrations versus improving chosen behavior, is what makes each direction natural. Distillation and RL aren't defined by opposite KL directions, and either setting can use other objectives.

## Implementation and overfitting

### Floating-point stability

**Implementation detail:** Given logits, “normalize” them by subtracting the maximum before exponentiating. Large positive numbers can become infinity when exponentiated.

> **ChatGPT addition (why subtracting the maximum helps):** Softmax is unchanged by a constant shift:
> 
> \[ \frac{e^{z_j}}{\sum_k e^{z_k}} =\frac{e^{z_j-M}}{\sum_k e^{z_k-M}}, \qquad M=\max_k z_k. \]
> 
> All exponentials are then at most 1, and at least one is exactly 1, preventing exponential overflow and a zero denominator. Very negative values can still underflow to zero. Stable log-softmax/cross-entropy avoids unnecessarily computing a tiny probability and then taking its logarithm.[^1]

### Too many parameters, too little data

Important aside: many parameters and very few examples make overfitting the training data easy.

**Connection: ISLP.** Our confidence in estimates from so little data would be low. Remember that even overfitting need not make loss zero if there's entropy in the input data. For the starting “...”, multiple different characters can follow. So we're bottlenecked by entropy anyway.

> **ChatGPT addition (making these two points precise):** The ISLP connection is about statistical uncertainty and sensitivity to the sample. Ordinary coefficient confidence intervals do not automatically carry over to neural-network weights, whose parameterizations can be non-identifiable.
> 
> If the same visible context has different labels, a deterministic context-to-distribution model cannot assign probability 1 to all of them. Its minimum empirical average log-loss, over unrestricted conditional distributions, is
> 
> \[ \widehat H(Y\mid X) =\sum_x\frac{n_x}{N}H\bigl(\widehat Q(\cdot\mid x)\bigr). \]
> 
> An MLP may not attain this lower bound. The uncertainty is relative to the chosen context: a longer context may distinguish some cases. Overfitting can still mean memorizing the empirical frequencies rather than learning population frequencies.

**Meta: pedagogy.** A useful example would be teaching a class where you explain all this without the idea of entropy and just give formulas, motivating them with reasonable enough math like smoothing, normalization, and gradients pointing in the correct direction. Then separately motivate it with concepts of entropy.

This seems like a good illustration of the difference between understanding the math and being able to solve problems with the formulas.

## SGD and mini-batching

This is where he introduces SGD. The intuition for why the “loss” from sampling a smaller set is offset by the speed gain seems like a very good opportunity for deeper mathematical intuition. I also thought there might be an advantage from somehow smoothing the loss through random sampling.

Answer from the follow-up: the mini-batch gradient is the true gradient + random error. This error is described by the covariance of the gradient (it's a random variable, so this should sound reasonable), and we can study an inequality describing how the error affects loss progress.

Okay, there's some beautiful theory here. Takeaway: studying variance is once again incredibly important. The error affects loss progress, and we can show a certain amount of expected progress when the signal (size of the gradient) is large compared with the noise, factoring in batch size and curvature.

> **ChatGPT addition (a quantitative statement):** Let \(f(\theta)\) be average training loss and \(g_B\) the mean gradient of \(B\) examples sampled independently and uniformly, with replacement. At the current parameters,
> 
> \[ g_B=\nabla f(\theta)+\varepsilon,\qquad \mathbb E[\varepsilon\mid\theta]=0,\qquad \operatorname{Cov}(\varepsilon\mid\theta)=\frac{\Sigma(\theta)}B, \]
> 
> where \(\Sigma(\theta)\) is the covariance of one randomly sampled example's gradient.
> 
> Assume \(f\) has \(L\)-Lipschitz gradient in the region of the step. Applying the smoothness bound to \(\theta^+=\theta-\eta g_B\) gives
> 
> \[ \mathbb E[f(\theta^+)\mid\theta] \le f(\theta) -\eta\left(1-\frac{L\eta}{2}\right)\|\nabla f(\theta)\|^2 +\frac{L\eta^2}{2B}\operatorname{tr}\Sigma(\theta). \]
> 
> For \(0<\eta<2/L\), the bound guarantees expected descent when the gradient term exceeds the variance penalty. This is a one-step statement, not a general convergence or computation guarantee.[^4]

> **ChatGPT explanation from the follow-up:** The most revealing feature is that noise cancels in the first-order term, but survives in the second-order term. An unbiased noisy direction gives the same average linear improvement as the true gradient. But curvature charges for the length of the step, and random errors make that step longer on average.

> **ChatGPT addition (the algebra behind that intuition):**
> 
> \[ \mathbb E\|g_B\|^2 =\|\nabla f(\theta)\|^2+\frac{\operatorname{tr}\Sigma(\theta)}B. \]
> 
> This accounts for the extra term in the smoothness bound. It does not mean mini-batching literally smooths the objective: the expected mini-batch loss is still the full training loss. Noise can affect optimization trajectories without changing that mean objective.

More details are in the [same follow-up conversation](https://chatgpt.com/share/6ac53198-97d4-83e8-b6fa-52e13539cc77), including the question of when this reduces total computation rather than just computation per update.

> **ChatGPT addition (a simple example where it really saves computation):** Take
> 
> \[ f_i(\theta)=\tfrac12(\theta-a_i)^2,\qquad f(\theta)=\frac1N\sum_i f_i(\theta). \]
> 
> Let \(\mu\) and \(\sigma^2\) be the mean and variance of the \(a_i\)'s in the dataset. A full-gradient step with learning rate 1 reaches \(\theta^*=\mu\) in one update, using all \(N\) examples. A mini-batch step with the same learning rate reaches the batch mean \(\bar a_B\), and
> 
> \[ \mathbb E[f(\bar a_B)-f(\mu)] =\tfrac12\mathbb E[(\bar a_B-\mu)^2] =\frac{\sigma^2}{2B}. \]
> 
> For target expected excess loss \(\epsilon>0\), a batch of size \(B\ge\sigma^2/(2\epsilon)\) suffices. If that is much smaller than \(N\), we meet the target with much less example-processing. This illustrates an advantage at a specified accuracy, not an unconditional advantage at every accuracy or on every objective.

Lowkey need a vector-calculus refresher to reintuit all these things about bounds on small nudges. They seem quite unintuitive to me, or at least only superficially intuitive, in a way that I wouldn't dispute but wouldn't try firsthand.

### What changes in one update?

Worth stopping to appreciate the scaling. First, we used ALL input data: ran a forward pass on all examples, computed cross-entropy against all outputs, and used a backward pass to compute one gradient update on the weights. Then repeated this again and again.

Hmm, do we get a natural decomposition of how each individual input point affected the weight, or does the chain rule implicitly add the contributions from individual points?

Wait, I'm confusing things. Each data point individually contributes to loss, and then the average loss is taken, either across all data or across the mini-batch. Therefore the gradient is the average of the gradients due to each data point. This point is resolved.

> **ChatGPT addition (the decomposition):**
> 
> \[ \nabla_\theta\left(\frac1B\sum_{i\in\mathcal B}\ell_i(\theta)\right) =\frac1B\sum_{i\in\mathcal B}\nabla_\theta\ell_i(\theta). \]
> 
> All gradients here are evaluated at the same parameters, followed by one update. This is different from updating after each example, since those later gradients would be evaluated at changed parameters.

What we change with mini-batching is that instead of using all the data, we use a random subset for each round of weight updates. But we still update the model's weights.

> **ChatGPT addition (small qualification):** The update is not restricted to a chosen subset of parameters, but some coordinates can have zero gradient. For example, embedding rows for characters absent from the batch receive no data-loss gradient in this lookup architecture.

## Validation and hyperparameters

Training, dev, test split. Need some more theory here as well, but ISLP might be a better bet for this, at least initially.

When underfitting, there is scope for improvement, so use more parameters. Train and dev loss being similar seems like a clue here.

**Question:** How do we optimize hyperparameters? How do we know when to increase the hidden-layer size versus the embedding dimension?

> **ChatGPT addition (diagnosis and the two knobs):** Similar train/dev losses indicate a small observed generalization gap, not necessarily underfitting. If both losses are high relative to a useful baseline or achievable performance, capacity may be limiting, but optimization or insufficient context may also be responsible.
> 
> Embedding dimension controls the size of each character's reusable representation; hidden width controls the number of nonlinear features computed from the combined context. To choose between them, compare controlled changes using dev loss, while accounting for training budget and variation across runs. There is no universal rule that equal train/dev loss means “increase parameters”.

### Looking at the embeddings

Vowels (and y!!) clustered together, yay! Just from a dataset of names :)

> **ChatGPT addition (interpretability qualification):** This is suggestive evidence that prediction training learned similar usage patterns for those characters. The geometry is not uniquely determined, though: an invertible linear transformation of the embeddings, compensated for in the first layer, can leave predictions unchanged while changing distances. A cluster in this learned coordinate system is an observation, not a unique semantic interpretation.

**Meta: automation.** Hyperparameter optimization definitely seems like something that can be auto-researched really well: a small set of tunable knobs, with a clear validation score for feedback.

> **ChatGPT addition:** Repeatedly choosing settings against the same dev set can itself overfit that set. The test set should be reserved for evaluation after model selection.

## Parallelization

Within one update, each example's forward and backward pass can run independently in parallel. There's a caveat for batch norm, which we'll see next time, I guess.

> **ChatGPT addition (scope):** For this MLP with an example-separable loss, the example computations can be parallelized and their gradients averaged before one shared parameter update. The operations within an example still have dependencies. Batch normalization couples examples through batch statistics, so the “independent per-example” description needs modification when it is introduced.

## References for ChatGPT additions

[^1]: PyTorch, [CrossEntropyLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).  
[^2]: Hinton, Vinyals, and Dean, [_Distilling the Knowledge in a Neural Network_](https://arxiv.org/abs/1503.02531), §2.  
[^3]: Ouyang et al., [_Training language models to follow instructions with human feedback_](https://arxiv.org/abs/2203.02155), §3.5.  
[^4]: Cornell CS4787, [_Minibatching and Decreasing Step Sizes_](https://www.cs.cornell.edu/courses/cs4787/2021sp/notebooks/Slides6.html). The covariance form and quadratic example above are written out here to make the assumptions and computation comparison explicit.

Lecture materials: Karpathy's [makemore part 2 notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb).