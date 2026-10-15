---
date: 2026-10-12
tags: []
confidence: ""
status: ""
---
*AI disclosure: I used ChatGPT to edit my original free-write and add technical context. Substantive AI additions are marked throughout.*

Nice, so he does think it's valuable to develop intuition for how gradients flow through the layers, which is what I was trying to do in the previous lectures.

He calls backprop a “leaky abstraction”, and yeah, that seems pretty obvious: you can't separate the optimization technique from the architecture, lol. He motivates this from an engineer POV, but I think it's even more important from a researcher POV.

From the overview, it seems he's going to do much of what I already did in my detours over the last two lectures. He works through backprop at the level of individual operations in cross-entropy, and then derives the simplified expression analytically. I'm quite familiar with this now, so these notes are shorter.

## Observations

### Subtracting the maximum logit

We subtract the maximum logit for numerical stability, but this doesn't change the final gradients with respect to logits. As discussed earlier, the downstream gradient is insensitive to constant shifts in logits.

> **ChatGPT addition (making the cancellation precise):** For one example, \(h=\nabla_z\ell=p-e_y\), so \(\sum_jh_j=0\). If \(u=z-m(z)\mathbf1\), where \(m(z)=\max_jz_j\), the extra chain-rule contribution from differentiating the maximum is proportional to \(\sum_jh_j\), so it cancels. The shift depends on z, but that dependence does not change the final gradient in exact arithmetic.

### Vector derivatives and intuition

**Meta: pedagogy.** He didn't give the intuition for vector derivatives that I think is VERY important. He expanded the derivatives out, then collapsed them back into a matrix multiplication.

I think every time there's a cleaner-than-expected result, it's an opportunity to find a simpler or more intuitive explanation. This was what I was trying to do with the direction/coefficient interpretation in the part 2 addendum.

### Bessel's correction

Nice, he added Bessel's correction to the variance estimate.

> **ChatGPT addition (a convention worth distinguishing):** The notebook divides the centered sum of squares by \(B-1\):
> 
> \[ s_{\mathrm{sample}}^2=\frac1{B-1}\sum_{i=1}^{B}(a_i-\bar a)^2. \]
> 
> This is an unbiased population-variance estimator under the usual i.i.d. sampling assumptions. It doesn't follow that this convention is always better for normalization. PyTorch's batch-norm training forward pass divides by B, while its running-variance update uses the unbiased estimate.[^1]

### Summing and broadcasting are dual

Another interesting observation is the duality between summing and broadcasting in the forward and backward passes. This is just a calculus thing, but it's nice to notice: sum in the forward pass becomes broadcast in the backward pass, and broadcast in the forward pass becomes sum in the backward pass.

> **ChatGPT addition (the small example):**
> 
> \[ y=\sum_i x_i \quad\Longrightarrow\quad \frac{\partial L}{\partial x_i}=\frac{\partial L}{\partial y}. \]
> 
> Conversely,
> 
> \[ y_i=x \quad\Longrightarrow\quad \frac{\partial L}{\partial x}=\sum_i\frac{\partial L}{\partial y_i}. \]
> 
> A shared input accumulates contributions from every place it is used. More generally, backward propagation applies the transpose of the forward operation's Jacobian. Summation and replication are transposes of each other.

### Analytical backprop as understanding

**Meta: pedagogy.** He also motivates “analytical backprop” through efficiency and simplicity rather than understanding :/ I think understanding is a very valuable pedagogical motivation here too.

But he did give intuition for the logit gradient: its total across output guesses is zero, so there's a redistribution mechanism.

> **ChatGPT addition (the identity):** For one example,
> 
> \[ \nabla_z\ell=p-e_y,\qquad \mathbf1^\top\nabla_z\ell=0. \]
> 
> A gradient-descent step raises the target logit and lowers the others. For the batch-mean loss, each example's logit gradient is additionally divided by B. The zero-sum identity is also the gradient-level expression of the constant-shift invariance above.

### Batch norm: a missed opportunity for geometric intuition

Oh no! A big pedagogical mistake, imo: he does the full batch-norm gradient derivation very mechanically, without giving the geometric intuition for the result.

The result is also presented in a way that makes it harder to see what the variables mean. Vectorized notation would at least make the projection connection easier to see. He even says it's nontrivial to convert his expression to vectorized form.

I also think the broadcasting details overcomplicate the conceptual task. I'd prefer to understand the transformation on one feature's vector across a batch first, then handle the tensor shapes separately.

> **ChatGPT addition (the projection, using this lecture's variance convention):** For one feature, let \(a\in\mathbb R^B\) contain its values across the batch. Define
> 
> \[ s=\sqrt{s_{\mathrm{sample}}^2+\epsilon},\qquad \widehat a=\frac{a-\bar a\mathbf1}{s},\qquad y=\gamma\widehat a+\beta\mathbf1. \]
> 
> If \(g=\nabla_yL\), the exact input gradient is
> 
> \[ \nabla_aL= \frac{\gamma}{s} \left( I-\frac{\mathbf1\mathbf1^\top}{B} -\frac{\widehat a\widehat a^\top}{B-1} \right)g. \]
> 
> When \(\epsilon=0\) and the variance is nonzero, \(\widehat a\perp\mathbf1\) and \(\|\widehat a\|^2=B-1\). The two subtracted terms are orthogonal projections onto the common-shift and centered-scale directions. The backward pass removes those components and rescales what remains.
> 
> With \(\epsilon>0\), the formula still holds, but the scale-direction component is not completely removed. The \(B-1\) denominator matches the notebook; using the B-denominator variance convention changes that term to B.
> 
> This is the same geometry explored in the earlier notes. The longer expression is not just algebra to simplify: it describes which changes normalization makes ineffective.

## References

[^1]: PyTorch, [_BatchNorm1d_](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html), on training-time and running-variance conventions.

Lecture materials: Karpathy's [makemore part 4 notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb). The equations above use its \(B-1\) variance convention.