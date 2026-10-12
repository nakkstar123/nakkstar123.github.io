---
date: 2026-10-12
tags:
  - makemore
confidence: ""
status: ""
---
AI disclosure: I used ChatGPT to edit my original free-write and add technical context. Substantive AI additions are marked throughout.*

*These notes were written before watching part 3. They revisit the embeddings, hidden layer, and backpropagation introduced in part 2.*

## What exactly is being learned?

Recap: there was a big conceptual jump in the previous video as we introduced low-dimensional embeddings and a hidden layer with nonlinearity. Roughly, it seems that the motivations for these two were (1) learning shared representations between inputs and (2) learning nonlinear structure in the data.

It's still quite black-box to me how this architecture, through gradient descent and backprop, learns shared representations. In particular, how do you interpret the weights and their updates now? The bigram model was very interpretable: each row of the matrix predicted output logits corresponding to a specific input character. Now we aren't selecting a row of output logits directly; there's more meaningful linear algebra, and I'm losing some intuition.

My [ChatGPT detour on this](https://chatgpt.com/g/g-p-6a90c8dcffd08191af9957b58875e31a-recruiting/c/6ac5375d-cecc-83e8-aea4-f3dc35264332).

**Meta: tracing the computation.** There's sufficient (but not yet too much) complexity at this step that I think it's the perfect thing to decompose manually and trace through all the steps. I'm noticing that my linear algebra and calculus lack intuition right now, and things that I think could feel obvious don't yet feel obvious. It's also useful to separate out the roles of variables and operations, like one input versus one gradient update.

A good reframing of my initial statement is: it is clear how this architecture introduces the opportunity to learn shared structure, and this can be made explicit by examining the training algorithm. What kinds of shared structure are learned through the weights, and whether that shared structure is useful for prediction, are different questions altogether and depend on the data.

One thing that's difficult for me to interpret, maybe because it's a “two-direction” optimization, is that the task being solved here is: “Learn useful shared prediction directions, and learn how each input should combine them.”

### Features and nonlinearity

The nonlinearity's essential role is still counterintuitive to me. Yes, I understand the “necessity” of it because otherwise the logits are linear in the concatenated embeddings and can't capture certain interactions. But I don't at all understand how this is sufficient for picking out rich nonlinear structure. I also don't have a good idea yet of what the simplest nonlinear structure looks like in language modeling.

The “grouping” aspect does make sense: now the coefficient vector that picks the learned output-logit directions is linked to the entire context, in a nonlinear way, rather than a single character. But this doesn't address my concern above.

So the decomposition is: characters → features → a linear combination of output-logit prediction directions. Technically, there's a step before that which represents the characters as d-dimensional vectors in the first place, and backprop adjusts these so that the computation of features is convenient? I guess idk.

> **ChatGPT addition (making the decomposition explicit):** With row-vector notation, concatenate the embeddings of the context characters into \(x\), then compute
> 
> \[ a=xW_1+b_1,\qquad h=\tanh(a),\qquad z=hU_2+b_2. \]
> 
> If the hidden layer has \(m\) units, each row \(u_r\) of \(U_2\in\mathbb R^{m\times V}\) is a vector of output-logit contributions:
> 
> \[ z=b_2+\sum_{r=1}^{m}h_r u_r. \]
> 
> The hidden activations are the coefficients combining those vectors. Embeddings help determine the coefficients through the first layer. Nonlinearity makes richer interactions representable; it does not guarantee that gradient descent finds useful ones.

Interpretability here is lost, or at least made a lot more difficult than before, because the architecture doesn't describe which features are to be learned.

It seems like the next lecture might focus on how gradient descent can effectively change parameters, especially with the slope complications that tanh introduces. That feels like a different question from interpreting what the model learns. I want to understand this computation better before starting the lecture.

## Updating shared prediction directions

I want intuition for how these shared output-logit-changing directions update during gradient descent, and how an update from example i affects the prediction for example j in a way proportional to an inner product between their representations.

The first part is actually very simple: imagine fixing the embedding matrix and updating only the directions. This isn't what's actually done; they're updated simultaneously, so I need to check whether that breaks the logic.

The predicted logits on input i are C_i × U, a linear combination of the directions weighted by the embedding coefficients.

> **ChatGPT addition (notation and scope):** Start with a simplified model that has no hidden layer:
> 
> \[ C\in\mathbb R^{V\times d},\qquad U\in\mathbb R^{d\times V},\qquad z_i=c_iU, \]
> 
> where \(c_i=C_{i,:}\) is a row vector. We omit bias for now. Let \(u_r=U_{r,:}\) denote the \(r\)-th prediction direction, so \(z_i=\sum_r c_{ir}u_r\).
> 
> For one training example \((i,y)\), set
> 
> \[ p_i=\operatorname{softmax}(z_i),\qquad g_i=-\nabla_{z_i}\ell=e_y-p_i. \]
> 
> Here \(g_i\) is the **negative** logit gradient, the desired direction of correction. The ordinary gradient is \(p_i-e_y\). All updates below use the old parameter values and a learning rate \(\eta\).

Not required for this specific reasoning, but the gradient of loss on example (i, y) with respect to logits is p − one-hot(y). So the correction raises the correct logit by an amount proportional to 1 − its predicted probability, and lowers each incorrect logit by an amount proportional to its predicted probability. I would also like to see how one such update affects the probabilities after going through softmax; the scaling might be informative.

Given this upstream correction, we would change each direction u_r in proportion to its coefficient C_{i,r}. But when we look at how this affects the prediction on another input i', that same direction is weighted by C_{i',r}. So its contribution to the change is proportional to C_{i,r} C_{i',r} g. Summing over directions gives an inner product.

> **ChatGPT addition (the exact update with representations fixed):**
> 
> \[ U^+=U+\eta c_i^\top g_i,\qquad \Delta u_r=\eta c_{ir}g_i. \]
> 
> For another input \(i'\),
> 
> \[ \Delta z_{i'} =c_{i'}(U^+-U) =\eta\langle c_{i'},c_i\rangle g_i. \]
> 
> This is exact for this update, not just a first-order approximation. Aligned embeddings transfer the same correction; orthogonal embeddings transfer none; a negative inner product transfers the opposite correction. Whether that helps prediction on \(i'\) depends on what its correct answer is.

The intuition is: example i's weight update changes each direction in proportion to how strongly example i uses it. But example i' also predicts using a weighted linear combination, according to how strongly it uses each direction. Combining these, it's clear that changing U according to example i affects prediction on example i' in a way that measures how closely aligned the two representations are.

This was also a useful exercise in tracing transposes, and reaffirms how much confusion rows and columns can introduce just for the sake of compactness.

**Algebraic aside:** One thing that took me time to intuit: if z is a linear combination of vectors u_r, and we change ALL of the u_r by multiples of some common vector g, then z changes by a multiple of g too. How much is an inner product of the original combination coefficients and the update coefficients.

### Another way to see the inner product

This gives one way to intuit the inner product as an alignment of embeddings. Another geometric perspective makes it feel inevitable: consider some output logit n. It's an inner product between c_i and the column of U corresponding to output n.

To increase that logit fastest, we would move that column in the direction of c_i. Its effect on the same output logit for another input i' then depends on the inner product of c_i and c_i'.

> **ChatGPT addition (correcting the indexing):** If \(v_n=U_{:,n}\), then
> 
> \[ z_{i,n}=\langle c_i,v_n\rangle,\qquad \Delta v_n=\eta g_{i,n}c_i^\top, \]
> 
> hence
> 
> \[ \Delta z_{i',n} =\eta g_{i,n}\langle c_{i'},c_i\rangle. \]
> 
> The factor \(g_{i,n}\) belongs in the **update**, not in the definition of the original logit.

In the original bigram model, the representations were one-hot vectors. Different input characters were perpendicular, so an update on one input didn't change predictions on another.

> **ChatGPT addition:** Taking \(c_i=e_i^\top\) gives \(\langle c_{i'},c_i\rangle=\mathbf 1\{i'=i\}\). That recovers the independent-row updates of the bigram table.

### What happens to the probabilities?

> **ChatGPT addition (answering the earlier question):** For the same input i, keeping its embedding fixed gives
> 
> \[ z_i^+=z_i+\alpha g_i,\qquad \alpha=\eta\|c_i\|^2. \]
> 
> The updated probabilities are exactly
> 
> \[ p_{i,n}^+ =\frac{p_{i,n}e^{\alpha g_{i,n}}} {\sum_k p_{i,k}e^{\alpha g_{i,k}}}. \]
> 
> For a small step,
> 
> \[ p_i^+-p_i =\alpha\left( p_i\odot g_i-\langle p_i,g_i\rangle p_i \right)+O(\alpha^2). \]
> 
> This follows from the softmax Jacobian. The target probability increases for \(\alpha>0\) when it is below 1. All incorrect logits decrease, but not necessarily every individual incorrect probability: normalization can redistribute probability among the incorrect classes too. Also, \(\|c_i\|^2\) sets the scale of this logit correction, so different embeddings can induce different effective step sizes even with the same \(\eta\).

## Updating the embedding coefficients

I also want to intuit how the weights of the embedding matrix are updated. Presumably there's similar reasoning about vectors and changes.

There is very useful intuition in the updates to U and C: to make the prediction move in direction g, update each prediction direction in proportion to its current influence. Similarly, update the coefficient vector by strengthening its use of directions already aligned with g and weakening its use of directions pointing against g.

> **ChatGPT addition (the coefficient update):**
> 
> \[ c_i^+=c_i+\eta g_iU^\top,\qquad \Delta c_{ir}=\eta\langle g_i,u_r\rangle. \]
> 
> If \(u_r\) points toward the desired correction, its coefficient increases; if it points against the correction, its coefficient decreases. Coefficients can be negative, so “stronger influence” should be understood as changing the signed contribution.
> 
> With U fixed, the resulting logit change is
> 
> \[ \Delta z_i=\eta g_iU^\top U. \]
> 
> Unlike the U-only update, this need not be a scalar multiple of \(g_i\): the available prediction directions shape which corrections coefficient changes can produce.

### Does simultaneous updating break the argument?

What is less intuitive is why the “joint” change that's optimal, in some constrained sense, is to simultaneously update both the direction vectors and the coefficient vectors.

> **ChatGPT addition (separating the effects):** In the simplified lookup model, one example's data-loss gradient updates only row \(c_i\) of C. For a distinct input \(i'\), row \(c_{i'}\) is unchanged. Consequently, its logit change remains exactly
> 
> \[ \Delta z_{i'}=\eta\langle c_{i'},c_i\rangle g_i, \qquad i'\ne i, \]
> 
> even when C and U are updated simultaneously.
> 
> For input i itself, both changes matter:
> 
> \[ \Delta z_i =\eta\|c_i\|^2g_i+\eta g_iU^\top U+O(\eta^2). \]
> 
> The first two terms are the individual first-order effects; the remainder comes from multiplying the two parameter changes. This analysis is for one example, ordinary gradient descent, no bias, and no regularization. Mini-batches and the full MLP introduce additional shared updates.

**Connection: ISLP.** This reminds me of the weighted-variance material from ISLP, and I wonder if there's a connection.

“Gradient descent chooses the greatest first-order decrease in loss for a fixed budget of parameter movement, measured by the sum of squared changes to all parameters.” Maybe this sum-of-squares budget is why similar variance behavior shows up!

> **ChatGPT addition (the precise optimization statement):** Treat all parameters as one vector \(\theta\). For a fixed Euclidean movement budget \(\rho>0\), if the gradient is nonzero,
> 
> \[ \underset{\|\delta\|\le\rho}{\arg\min} \;\langle\nabla_\theta\ell,\delta\rangle =-\rho\frac{\nabla_\theta\ell}{\|\nabla_\theta\ell\|}. \]
> 
> The norm includes every parameter block, so
> 
> \[ \|\delta\|^2=\|\Delta C\|_F^2+\|\Delta U\|_F^2 \]
> 
> in the simplified model. Equivalently, minimizing
> 
> \[ \langle\nabla_\theta\ell,\delta\rangle +\frac1{2\eta}\|\delta\|^2 \]
> 
> gives the ordinary step \(\delta=-\eta\nabla_\theta\ell\). “Optimal” here means optimal for the **first-order approximation under this metric**, not a globally optimal finite update or convergence to the best model.
> 
> The squared movement budget is not a weight-decay penalty on C or U. Its use of squared norms alone does not establish a connection to statistical variance minimization in ISLP.

## From embeddings to hidden representations

Worthwhile exercise to trace it all out, separately indexing the important variables. Remember, this was done entirely without the nonlinear layer and context packing. Each input i indexed a row C_i, which we treated as coefficients for the prediction directions.

But in the full MLP, we use the hidden output h for an entire context (i1, i2, i3), generated in a nonlinear way, as the coefficients combining output directions. So inner products between the hidden vectors of different contexts now control part of how their predictions change.

> **ChatGPT addition (the same argument at the output layer):** For two contexts s and t, if only \(U_2\) changes and their hidden representations stay fixed,
> 
> \[ \Delta U_2=\eta h_s^\top g_s,\qquad \Delta z_t=\eta\langle h_t,h_s\rangle g_s. \]
> 
> If the output bias is trained too, it contributes \(\eta g_s\), making the total \(\eta(\langle h_t,h_s\rangle+1)g_s\). Updating \(W_1\) and C also changes the hidden representations, so the hidden inner product describes the output-layer contribution rather than the entire network update.

### Tracing sensitivity backward through tanh

I found [another conversation](https://chatgpt.com/share/6ac56ae0-1318-83e8-b00e-169d251d3fa9) helpful for thinking about input sensitivity, though I initially couldn't understand it because of fatigue.

The negative gradient plays this role in output-logit space: how helpful is moving the logits in each available direction for minimizing loss?

The usefulness of the negative gradient on tunable parameters is that we can nudge the parameters in that direction.

We find the best direction by looking at each coordinate's influence on prediction. For example, in the factorized model, we update c_r in proportion to the inner product between u_r and the desired correction. If we have a limited budget of coefficient movement, we want to spend it on directions that matter and ignore those that don't.

Similarly, our goal now is to find the best direction to move the preactivation vector. We know how changing each postactivation coefficient moves the prediction, and how sensitive the postactivation is to a change in its preactivation. So we combine those to find the best direction to move the preactivations.

For example, if an activation is already saturated, changing the corresponding preactivation won't make much difference, even if that hidden unit's direction is useful for prediction. So we might not put much movement budget there.

> **ChatGPT addition (the backward equations):** In the row-vector MLP notation from above, let \(g_z=e_y-p\). Then the negative gradients are
> 
> \[ g_h=g_zU_2^\top,\qquad g_a=g_h\odot(1-h^2). \]
> 
> The first equation measures how useful each hidden coefficient would be to change. The second multiplies that usefulness by the local sensitivity of tanh. Saturation near either \(+1\) or \(-1\) makes \(1-h_r^2\) small.
> 
> These are gradients with respect to intermediate quantities. The network does not independently update a or h as free parameters; it uses their gradients to update the parameters that produce them.[^1]

For W1 updates, again apply the direction-update interpretation, but now analyze how it affects preactivations rather than final logits. For C updates, again apply the coefficient-update interpretation.

“Directions change in proportion to their use; coefficients change according to the usefulness of their directions.” Tbh this seems like intuition for matrix calculus on something like A1A2.

> **ChatGPT addition (completing that pattern):**
> 
> \[ \Delta W_1=\eta x^\top g_a,\qquad g_x=g_aW_1^\top. \]
> 
> Each row of \(W_1\) specifies a direction in preactivation space; its change is proportional to the corresponding input coefficient in x. Each component of \(g_x\) measures the usefulness of changing that coefficient. Split \(g_x\) into its embedding-sized blocks and add the corresponding corrections to the character rows of C. If a character appears more than once in the context, its row receives the sum of those contributions. Mini-batch contributions are averaged in the same way.

Helpful takeaway when moving to deeper models: preactivations play a similar role to logits for the purposes of tracing a linear map backward.

> **ChatGPT addition (scope of the analogy):** Both are outputs of affine maps, so the same matrix-gradient interpretation applies. Hidden preactivations are not themselves class logits: their downstream objective and activation are different.

## What can we prove about convergence?

I also found it helpful to write out gradient descent for the bigram model and work through convergence to the counting solution in [this conversation](https://chatgpt.com/share/6ac6b1d0-bf40-83e8-91c0-b9e28d2c7e92). The Hessian/variance argument is another place where my intuition feels too weak, but it seems important to understand. Maybe this is where we can start proving things about asymptotic efficiency.

> **ChatGPT addition (why the bigram case is simpler):** For one observed input row, let q be its empirical next-character distribution. Its average loss is
> 
> \[ \ell(z)=\log\sum_n e^{z_n}-q^\top z,\qquad \nabla\ell=p-q. \]
> 
> Its Hessian is
> 
> \[ H=\operatorname{diag}(p)-pp^\top. \]
> 
> For any vector v,
> 
> \[ v^\top H v=\operatorname{Var}_{n\sim p}(v_n)\ge0. \]
> 
> This proves convexity and makes the variance connection literal: curvature measures variation in a proposed logit direction under the current model. The constant direction has zero curvature because softmax ignores constant shifts.
> 
> When all q entries are positive, a finite optimum exists at \(z=\log q+c\mathbf 1\), and full-gradient descent with a sufficiently small constant step converges to the optimal predictions. If some counts are zero, the probability optimum is on the boundary and finite logits cannot attain it exactly. Optimization rates also depend on curvature and conditioning; they are not automatically statements about total training computation.[^2]

Interesting: once we learn both C and U, we have already reached the land where general claims of global optimality need stronger assumptions because of nonconvexity. But there are meaningful notions of optimality here too. It feels like an art to find the assumptions that lead to useful results, and to understand which are genuine restrictions and which are just choices of representation.

> **ChatGPT addition (a distinction worth keeping):** The loss is convex in the unrestricted logit table, and convex in either C or U when the other is fixed, but generally not jointly convex in both. Even if the best predictions can be represented with dimension d, that alone does not prove gradient descent will find them. Local steepest descent, global training optimality, and generalization to unseen data are three separate claims.

## An experiment I want to try

**Exercise:** Set up experiments that isolate the new features added in the jump from bigram to MLP:

1. Low-dimensional representations.
2. Nonlinearity.
3. Increased context.

These changes were motivated partly by saving parameters when increasing context, but I want to see the results and get some taste of how they actually generalize. This should help me understand the different sources of improvement.

> **ChatGPT addition (sharpening the comparison):** Start with the full bigram table, then a factorized one-character model \(CU\), then an additive linear model using longer context, and finally the nonlinear MLP using that same longer context. These comparisons isolate compression, additional context, and nonlinearity in a useful sequence.
> 
> Keep the data split and evaluation metric fixed; report training and dev loss, parameter count, and variation across runs. Capacity and optimization budget can confound the interpretation, so these are comparisons under specified settings rather than a unique decomposition of the improvement.

## References for ChatGPT additions

[^1]: Goodfellow, Bengio, and Courville, [_Deep Learning_, chapter 6, §6.5](https://www.deeplearningbook.org/contents/mlp.html), on backpropagation and Jacobian-gradient products.  
[^2]: Stanford STATS 305B, [_Multinomial models_](https://web.stanford.edu/class/stats305b/notes/GLM_V.html), on the multinomial covariance and Hessian. The simplified model updates and softmax-step calculations above are written out directly for this addendum.