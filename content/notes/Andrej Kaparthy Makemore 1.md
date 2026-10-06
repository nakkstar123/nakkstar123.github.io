*AI disclosure: I used GPT-6.1 Sol High to edit my original free-write and add technical context. Substantive AI additions are marked throughout.*
## Context

In transitioning my research focus to AI safety, I am simultaneously adopting top-down and bottom-up approaches. I am reading popular research perspectives on why AI poses an existential threat and what technical approaches are being attempted, as well as learning the fundamental details of how frontier LLMs work. For the latter, my first step is studying Andrej Karpathy's "Neural Networks: Zero to Hero" playlist.

I have skimmed through some of these videos in the past, taken Intro ML and Deep Learning courses at UCLA, and previously engaged with 3b1b's neural networks playlist, so I'm already entering with some background. I also have an above-average background in linear algebra, calculus, and probability from my math undergrad/masters and PhD. Nevertheless, I think it's valuable for me to think deeply about the fundamentals leading up to how frontier models work.

A significant amount of my time while "consuming" this content was spent talking to ChatGPT about follow-ups that piqued my curiosity. I'm also trying to question all arbitrary design decisions—even ones that are somewhat motivated by intuition—jumping ahead of myself, and trying to get more mathematical than Andrej intended so I can get more value from this exercise. In particular, I'm attempting to understand all models introduced from an interpretability POV from the get-go, which I know isn't how research in ML typically happens.

My hope is that my research background, and eventual goal of working on AI safety, might mean that this approach to learning leads to simple but original insights at some point. Or, at the very least, that it helps me make deeper connections that were historically only discovered a bit later, but are natural to think about early on—saving me time and the cognitive effort of recall if I were to learn them only much later via a typical study path.

I am aware that this approach may be a time sink, consist of many detours that drain or slow me, and will most likely not lead to novel ideas. But I am certain these detours will lead to a better understanding, and I'm taking a bet that improved understanding of frontier models is important enough that a potentially inefficient way to get there is still worth it.

I will describe things such as the problem setup briefly, but not spend too much time on them, as I don't intend this for anyone who hasn't watched the lectures.

The problem setup is training a character-level language model to produce human-sounding names.
## Bigram model

### Counting

If we are forced to use at most one previous token of context to predict the next one, it seems very intuitive that the best fit to the training data is counting all bigrams. When you have character c in context, use the frequencies of bigrams "c?" to define a probability distribution for sampling the next character, and keep repeating. It is a nice sanity check that the gradient-descent formulation also converges to this solution.

> **ChatGPT addition — what “best” means:** Counting gives the maximum-likelihood fit within the unrestricted bigram model. This is a statement about fitting the training data, not necessarily predicting unseen data. Smoothing can improve the latter. The gradient-descent comparison assumes no regularization, with a caveat about zero probabilities below.

What follows was my live commentary/note-taking produced while watching the lecture, edited for clarity and with some mathematical details added. Some of the detours came from follow-up conversations with ChatGPT.

Starting with bigrams: an incredibly tiny context window of just one character predicting the next.

Idea: count all the different bigrams. There are only (vocabulary size)² possible pairs, so store them as a matrix where `m[l1][l2]` has the number of occurrences of the bigram (l1,l2). Note that this is not symmetric.

Super easy generation method: sample each next letter, starting with a start marker, until you hit the end marker. Probabilities are weighted according to the bigram counts for the current character.

> **ChatGPT addition — notation:** Karpathy uses the same "." symbol for both boundaries. With 26 letters, that gives 27 symbols. Writing the counts as \(N_{ij}\), the next-character distribution is
> 
> \[ \widehat P(j\mid i)=\frac{N_{ij}}{\sum_k N_{ik}}, \]
> 
> for a context \(i\) with observed transitions.

**Question — temperature.** Hmm, there's no fancy smoothing/spiking transformation from original scores (logits) to probabilities here. Does that mean there's no temperature setting?

> **ChatGPT addition:** Temperature can also be applied directly to probabilities:
> 
> \[ p_j^{(T)}=\frac{p_j^{1/T}}{\sum_k p_k^{1/T}},\qquad T>0. \]
> 
> It doesn't require a neural network. Zero probabilities remain zero, though.

**Reaction to the samples.** The generated names were absolutely horrible btw. But visibly better than uniform noise, so I guess that's still saying something good. Maybe my expectations are also a bit too high now.

**Technical takeaway.** Nice abstraction: constructing the probability matrix P was "training" the model, and sampling repeatedly is "inference". Now we want some way of measuring the performance of our model on a dataset, i.e. loss.

### Likelihood, surprise, and compression

Big idea: look at the probability of the entire dataset assigned by the trained model. If the model is good at fitting the training data, we shouldn't be surprised if the actual training data is likely, and we should be surprised if it's unlikely.

The transformation from probability to negative log-likelihood seems motivated by information/entropy. Any intuition there would carry over, I suppose. I think I'm just lacking enough intuition for why information is the right framework—in the mathematical sense of entropy; ofc intuitively it makes sense.

If "surprised" is defined correctly, maximum likelihood is equivalent to choosing the model that is least surprised by the observed dataset.

Okay, the bits interpretation is starting to make sense now: it is the idealized number of bits required to encode the dataset. So we wish to compress the data as well as possible.

> **ChatGPT addition:** With base-2 logarithms, an event assigned probability \(p\) has information content \(-\log_2 p\) bits. This gives an idealized coding cost under the model. It assumes the decoder already has the model; it does not include the cost of describing the model itself.

"For autoregression, it is the idealized number of bits to encode the sequence token by token." What is autoregression? Oh, it means predicting a sequence token by token, using what came before it. GPT is a fancy way to estimate these conditional distributions.

The idealized number of bits to encode the sequence is the sum of the surprise of each next token.

> **ChatGPT addition — the equation behind this:**
> 
> \[ P(x_1,\ldots,x_n)=\prod_{t=1}^{n}P(x_t\mid x_{<t}), \]
> 
> so
> 
> \[ -\log_2 P(x_1,\ldots,x_n) =\sum_{t=1}^{n}-\log_2 P(x_t\mid x_{<t}). \]
> 
> This is the probability chain rule, not an independence assumption. An autoregressive model can use restricted context: the bigram model replaces the full history with just the previous character.

### Cross-entropy and KL

Seems like the deepest interpretation is: "Make the model distribution close to the true distribution in KL."

**Connection — 3b1b.** Ohhh, KL divergence makes so much more sense now that I've watched the 3b1b video: how much additional surprise does it cost you to use distribution P to approximate distribution Q, averaged over events occurring according to Q?

"How many extra bits per observation do I need because I'm compressing data using the wrong probability model?"

Entropy(Q): average surprise when sampling from Q and using Q's probabilities.

Cross-entropy(Q, P): average surprise when sampling from Q, but using P's distribution to predict.

KL(Q, P): extra surprise, on average, when using P instead of Q to model Q.

> **ChatGPT addition — putting these together:**
> 
> \[ \begin{aligned} H(Q)&=\mathbb E_{x\sim Q}[-\log Q(x)],\\ H(Q,P)&=\mathbb E_{x\sim Q}[-\log P(x)],\\ D_{\mathrm{KL}}(Q\|P)&=\mathbb E_{x\sim Q}\!\left[\log\frac{Q(x)}{P(x)}\right]. \end{aligned} \]
> 
> Therefore,
> 
> \[ H(Q,P)=H(Q)+D_{\mathrm{KL}}(Q\|P). \]

**Connection — ISLP.** Similar to ISLP: total prediction error = irreducible uncertainty when predicting Q + error from using the wrong model.

> **ChatGPT addition — a qualification:** For next-token prediction, the identity applies to conditional distributions, averaged over contexts. “Irreducible” is relative to the information available: uncertainty after seeing one character may disappear after seeing a longer context.

**Question — what is the “true distribution” of language?** For a language model, ofc we don't know Q, which we believe "exists" as some distribution about how language works everywhere. Hmm, I think this assumption itself doesn't respect diversity of thought. But we have an empirical distribution, and training minimizes its cross-entropy with P_theta, which means minimizing the corresponding KL.

> **ChatGPT addition:** A distribution can represent diversity: different people, styles, and situations can all be part of a mixture. The question is which mixture the corpus represents. Also, minimizing empirical cross-entropy is not the same as minimizing population cross-entropy; that gap is a generalization question.

### A detour into forward/reverse KL and RLHF

The next thing that caught my attention was mode-covering vs mode-seeking, depending on the forward vs reverse KL direction. Reverse KL seems less intuitive to me, and kinda dangerous, but maybe will be useful in some contexts in the future.

In the forward direction, we are saying: "Sample from the real world, and make sure that, for actual human text, the model doesn't declare it too crazy." So the model is "judging" the actual data, and we want it to be such that it isn't too surprised by that data.

**Connection — RLHF.** Then came the connection to RL. We start with some original distribution and train a new distribution. We want the new distribution trained in a way such that, WHEN JUDGED AGAINST THE OLD DISTRIBUTION, it doesn't seem too crazy.

Wow, this feels like an insane philosophical difference. Should the judge be the "underlying data" or the trained model? Idek if this is an RL vs no-RL thing or just a matter of judgment.

Ahh, a better way to interpret the two KL directions: it's about where you average. Do you average according to where reality goes or where my model goes?

> **Technical context · ChatGPT addition — separating the two comparisons:** When comparing a model \(P\) with a target \(Q\),
> 
> \[ D_{\mathrm{KL}}(Q\|P)=\mathbb E_{x\sim Q}\log\frac{Q(x)}{P(x)} \]
> 
> averages over the target's samples, whereas
> 
> \[ D_{\mathrm{KL}}(P\|Q)=\mathbb E_{x\sim P}\log\frac{P(x)}{Q(x)} \]
> 
> averages over the model's samples. In a common RLHF setup, the second argument is instead a **reference model**, not the underlying data distribution:
> 
> \[ \max_\theta\; \mathbb E_{y\sim P_\theta(\cdot\mid x)}[r(x,y)] -\beta D_{\mathrm{KL}}\!\left( P_\theta(\cdot\mid x)\|P_{\mathrm{ref}}(\cdot\mid x) \right). \]
> 
> This is shown for one prompt \(x\); training also averages over prompts. The reward encourages preferred responses, while the KL term constrains drift from the reference. This is the setup used in InstructGPT, rather than a definition of RL in general.[^1]

Forward KL tries its best to "not miss anything reality has"; reverse KL instead says, roughly, "Whenever you generate something, make sure it's plausible." Combining this with post-training RLHF seems smart!

> **ChatGPT addition — limits of the intuition:** “Mode-covering” and “mode-seeking” are tendencies under imperfect approximation, not universal rules. Forward KL also penalizes wasting probability outside the target's support, and reverse KL needn't collapse to one mode.

Btw, if the model is expressive enough to represent P = Q and we reach that solution, both KLs are zero. So this distinction matters especially when we acknowledge that the model isn't expressive enough.

> **ChatGPT addition:** Limited data and optimization can also make the direction matter, even if the model could theoretically represent \(Q\).

> **Illustrative example · ChatGPT addition from the follow-up discussion:**
> 
> Imagine the true distribution of human-written answers to some prompt contains 1,000 qualitatively different excellent answers.
> 
> Model A covers all 1,000 styles, but also occasionally generates garbage.
> 
> Model B produces only 20 of the 1,000 styles, but those 20 are consistently excellent.
> 
> Which is the better model?
> 
> If you're studying human linguistic diversity, probably A.
> 
> If you're shipping an assistant people actually use, perhaps B.
> 
> Pretraining says approximately: learn everything humans write.
> 
> Post-training asks: of everything you could say, concentrate probability on the subset we actually want you to say.
> 
> This illustrates the difference in objectives, rather than a consequence of KL direction alone. Post-training can also include supervised learning.

**Takeaway.** "Model the world" and "act well in the world" are different objectives.

**Meta — following curiosity.** I think following curiosity helps you make connections among different concepts as you go, rather than having them individually and then having to graph-search to find the paths/connections.

### Back to the bigram loss

Average negative log-likelihood, as he computed it, is average surprise when encountering a transition in the dataset. So it's not average surprise when seeing a whole word in the OG dataset—that's higher—but average surprise when seeing each individual "next" character. For a word, we sum those surprises.

> **ChatGPT addition:** The average is over transitions, including boundary transitions. The sum follows from the chain rule above, rather than independence of characters. The code uses natural logs, so its loss is in nats; dividing by \(\log 2\) converts it to bits.

Loss is infinite if we evaluate a particular bigram that has zero training occurrences and therefore zero assigned probability. You can do model smoothing and add fake counts (+1) to everything. The more you add, the more uniform the distributions become.

> **ChatGPT addition — making the smoothing explicit:** If \(N_i=\sum_j N_{ij}\) and the vocabulary has \(V\) symbols, adding a pseudocount \(\alpha>0\) gives
> 
> \[ P_\alpha(j\mid i)=\frac{N_{ij}+\alpha}{N_i+\alpha V}. \]
> 
> For an observed context, this can be written as
> 
> \[ P_\alpha(\cdot\mid i) =\frac{N_i}{N_i+\alpha V}\widehat P(\cdot\mid i) +\frac{\alpha V}{N_i+\alpha V}U, \]
> 
> where \(U\) is uniform. So it is literally a mixture of observed frequencies and uniformity, with more smoothing for contexts that have fewer observations.

## Neural network

### Learning the same bigram distributions

Now a new method of training, i.e. a neural network, but again a bigram-level model—that's a decision about what past data to look at/ignore.

The neural net thinks of each bigram as a training example (x, y), where x is ix1 and y is ix2, lol. Can see how this would generalize to trigrams. We want this input x to instead be a vector and then do some linear operations on it. Uses that to motivate one-hot encoding (OHE).

**Interpretability takeaway.** Okay, so in making your training data these (x, y) pairs, you've already forced it to be a bigram and forgotten everything further back in the past.

Hmmm, maybe one way to motivate this is to try to force the neural network to learn the bigram probability distributions. It takes a one-hot encoding and multiplies it by a 27 × 27 matrix. The output at position j is the inner product of column j with the one-hot input. So if the input is OHE(i), this is just W(i, j).

Oh, I guess we'll do a post-processing step of softmax, so the weights will be logits. Then softmax(W[i, :]) gives the probabilities, trained by cross-entropy.

> **ChatGPT addition — what the matrix multiplication does:**
> 
> \[ e_i^\top W=W_{i,:},\qquad P_\theta(j\mid i)=\frac{e^{W_{ij}}}{\sum_k e^{W_{ik}}}. \]
> 
> The one-hot input simply selects a row. At this stage there is no sharing of statistical strength between different input characters: each row is a separately learned next-character distribution.

Based on the data, I'm pretty sure matching the bigram frequencies is the optimal thing to do. It should converge toward this with gradient descent, I'd assume.

This means there are multiple choices of optimal logits: one is log p*(j|i), but adding any constant c_i to the whole row also works.

> **ChatGPT addition — the optimum and its caveat:** For a row whose empirical probabilities are all positive,
> 
> \[ W_{ij}=\log\widehat P(j\mid i)+c_i \]
> 
> reproduces the counting solution. Finite softmax logits cannot represent exact zeros, so for zero-count transitions the unregularized optimum is approached as a limit. Convergence also depends on optimization choices, and regularization changes the optimum.

### Why should this help with longer contexts?

And this neural network approach might work better for trigrams and such, but I need better intuition for why that might be the case. Right now, I know the naive approach has too many parameters. If we want to shrink the number of parameters without a NN, it's not clear what's the best way to compress them, and maybe a neural network figures that out itself as well...?

Ah, wait, it's not neural network magic; it's a step before that. If you choose to do OHE for all trigram inputs, i.e. pairs of characters, that's already 27² possible inputs, with 27 output logits associated with each. So you want to refuse to give every possible context its own independent set of parameters.

> **ChatGPT addition:** A full table for context length \(k\) has \(V^k\) rows and \(V\) outputs per row: \(V^{k+1}\) entries. The issue becomes much more severe as context grows; a trigram table alone is still quite manageable for 27 symbols.

So instead of treating pairs as entirely separate trigram inputs, still think of individual characters and "learn" things from the surrounding context as well, without forgetting your specific context. Seems like a pretty hard problem tbh, and seems lossy. How one-sided the tradeoff can be in favor of sharing parameters is not something I would've expected.

Don't need to store a representation for each input/output pair—or here, input/input/output triple. Rather, have reusable character representations e_i, e_j (so presumably not just one-hot vectors?), a reusable way of combining them, and a reusable way to map the combined vector to the output.

Still feels significantly weaker, but I guess that's because I'm imagining trying to fit noise. The point is that the data has reusable structure.

> **ChatGPT addition — the tradeoff:** At a fixed small parameter budget, a shared model generally cannot express every arbitrary lookup table. But observations in one context can inform predictions in another, which is valuable for rare or unseen contexts. The combining function also needs to preserve position/order, since “ab” and “ba” are different contexts.

**Connection — later models.** Nice abstraction: a central thing that will change with more advanced models is how to map inputs to output logits. Yes, that's all the work, but when you see the code, some of the intimidating stuff is just the loss loop, updates, etc.

### Smoothing, initialization, and regularization

The limit of smoothing in the probability-counting approach is uniform. That made me think about initialization in gradient descent. If all weights are equal—easy to think of them as zero, but I wonder if that hides some subtle difference—then it starts with uniform probabilities for all!

Then there's adding the sum of squares of entries of W to the loss. But shouldn't you do the variance within each row instead? Instead of forcing toward zero, just force toward a constant.

GPT said I'm right about the softmax invariance: softmax(z) = softmax(z + c), where c is added to every entry. So shrinking toward zero kinda recenters anyway.

> **Technical explanation · ChatGPT addition — why L2 still works here:** Let \(\bar W_i\) be the mean logit in row \(i\). Then
> 
> \[ \sum_j W_{ij}^2 =\sum_j(W_{ij}-\bar W_i)^2+V\bar W_i^2. \]
> 
> The likelihood ignores the row mean because softmax is invariant to constant shifts. L2 therefore selects a zero-mean representative of each distribution and penalizes the remaining variation. For this standalone bigram matrix, L2 and a centered squared-logit penalty give the same optimal predicted distributions, with matching scaling. This isn't a general equivalence for arbitrary neural-network weights.

Also, mathematically speaking, the +α method for probabilities is a different regularization procedure from this one. They're both regularizations, but they don't give exactly the same solution.

Btw, two different points here: initializing weights to the same constant vs adding a regularization term.

> **ChatGPT addition:** Initialization gives uniform predictions at the start; regularization keeps exerting pressure during training. Neither is identical to adding pseudocounts. Also, zero initialization is workable for this bigram matrix, but can be problematic in networks with hidden layers, where identically initialized units can remain symmetric.

The hidden-layer initialization issue is to be discovered soon!

[^1]: Ouyang et al., [_Training language models to follow instructions with human feedback_](https://arxiv.org/abs/2203.02155), §3.5.