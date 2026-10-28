---
date: 2026-10-26
tags:
  - arena
confidence: ""
status: ""
---
_AI disclosure: ChatGPT used to select and edit takeaways from my original notes and check the source material. Substantial clarifications and additions appear in labeled blockquotes._

These notes focus on the truth and deception sections of [ARENA 1.3.1](https://github.com/ARENA-education/ARENA_materials/blob/main/chapter1_transformer_interp/instructions/pages/11_%5B1.3.1%5D_Linear_Probes.md), especially _The Geometry of Truth_ and Apollo Research’s work on deception probes. The interesting parts for me were understanding what a probe actually establishes, why reading and steering can require different directions, and what the evaluation is really measuring.

## Key takeaways and insights

- **Successfully recovering a property with a linear probe is evidence for a linear representation, in the readout sense.** This was my initial connection to the linear representation hypothesis. If a linear classifier can recover truth labels from internal activations on held-out examples, then truth-relevant information is linearly accessible there. Whether the model uses that information when producing outputs is a separate question. The Geometry of Truth paper investigates both.
    
    > **ChatGPT clarification:** A probe establishes accessibility on the distributions tested. Cross-dataset transfer and causal interventions address stronger claims. Also, “truth” here refers to the truth values of simple factual statements with clear labels, rather than a general ability to identify objective truth or recover everything the model believes.[^truth]
    
- **The PCA result is more striking than I initially appreciated: there was no instruction to separate true from false.** I had mixed this up with the earlier patching experiments, which used output behavior to find interesting activation sites. PCA itself just looks for variation in the activations. Finding truth-related separation in its leading components is pretty cool!
    
    My intuition is the pancake cloud: PC1 points along its longest axis, PC2 along its width, perpendicular to PC1, and PC3 through its thickness. The projection should preserve as much of the spread as possible.
    
    > **ChatGPT mathematical context:** For activations \(h_i\), let \(\mu\) be their mean and \(S\) their covariance matrix. PCA chooses
    > 
    > \[ v_1\in\arg\max_{\|v\|=1} \operatorname{Var}\!\left[v^\top(h_i-\mu)\right] =\arg\max_{\|v\|=1}v^\top Sv. \]
    > 
    > Later components maximize the same objective subject to orthogonality to earlier ones. Centering makes the analysis about spread around the mean; the sign of each component is arbitrary.
    > 
    > Two components are convenient for plotting, not a restriction on probing. A separator in their plane corresponds to a linear probe in the original activation space. Conversely, useful information can lie outside that plane. Sparse features alone do not imply that only a few PCs suffice: many different features can vary across a dataset even when few are active on each example.
    > 
    > The PCA fit is unsupervised, but the overall investigation still involves choices about datasets, layers, and token positions. Here the paper motivates those sites with patching, and deliberately uses datasets with limited unrelated variation.[^truth]
    
- **A difference-of-means probe is incredibly simple.** Take the mean activation for true statements, the mean for false statements, and point from one to the other. Then take their midpoint as the origin and ask which way an example points. This felt much more intuitive than starting with logistic regression.
    
    > **ChatGPT mathematical context:** Writing
    > 
    > \[ \delta=\mu_+-\mu_-,\qquad m=\frac{\mu_++\mu_-}{2}, \]
    > 
    > the basic mass-mean (MM) classifier predicts the positive class when
    > 
    > \[ \delta^\top(h-m)>0. \]
    > 
    > This is also the rule for choosing the closer class mean in Euclidean distance. Explicitly including the midpoint matters when activations have not already been centered appropriately.[^arena]
    > 
    > Interpreting \(\delta\) as a particular feature direction needs an additional assumption: other features must have the same mean in both classes. Otherwise their mean differences contribute too. Taking a difference of means does not automatically remove confounds.
    
- **The best direction for reading a feature need not be the best direction for changing it.** This was probably the most useful geometric clarification for me. If feature directions aren’t orthogonal, a good classifier may compensate for interference from another feature. Its weights therefore need not point along the direction that originally encoded the feature I care about.
    
    > **ChatGPT example:** Suppose an activation is
    > 
    > \[ h=t\begin{pmatrix}1\\0\end{pmatrix} +z\begin{pmatrix}1\\1\end{pmatrix} =\begin{pmatrix}t+z\\z\end{pmatrix}. \]
    > 
    > Changing \(t\) while keeping \(z\) fixed means moving along \((1,0)^\top\). But reading \(t\) without interference from \(z\) uses \((1,-1)^\top\), since \((1,-1)h=t\). Adding the readout vector would change both coefficients. The distinction is already present in this tiny linear example.
    > 
    > This makes the paper’s MM-versus-LR comparison more understandable: optimizing classification and identifying a useful intervention direction are different tasks. The paper reports similar classification accuracies alongside stronger MM intervention effects in many conditions. Their agreement is something to test, rather than assume.[^truth]
    
- **The covariance correction has a nice mathematical motivation: maximize separation relative to within-class variation.** I initially read “multiply by the inverse covariance” as a mysterious way of canceling unrelated variation. What helped was writing down exactly what I want from a score \(w^\top h\).
    
    The difference between its class means is \(w^\top\delta\). Its variation within each class is also relevant: I don’t just want separated averages if individual examples fluctuate wildly around them. A natural objective is squared separation divided by within-class variance. The probe is still linear even though the objective used to choose it isn’t.
    
    > **ChatGPT mathematical clarification:** Let \(r=h-\mu_y\) be the class-centered residual, and let \(\Sigma=\operatorname{Cov}(r)\) be the pooled within-class covariance. Then
    > 
    > \[ \operatorname{Var}(w^\top r)=w^\top\Sigma w, \qquad J(w)=\frac{(w^\top\delta)^2}{w^\top\Sigma w}. \]
    > 
    > This is Fisher’s linear discriminant objective. If \(\Sigma\) is positive definite and \(\delta\ne0\), Cauchy–Schwarz gives
    > 
    > \[ (w^\top\delta)^2 \leq(w^\top\Sigma w)(\delta^\top\Sigma^{-1}\delta), \]
    > 
    > with equality when \(w\propto\Sigma^{-1}\delta\). That explains the inverse: directions with substantial within-class fluctuation are downweighted relative to their contribution to separation.
    > 
    > It does not remove all unrelated information or leave data varying only in truth. Equivalently, whitening by \(\Sigma^{-1/2}\) makes the within-class covariance the identity, and the corrected score is an ordinary inner product between the whitened activation and whitened mean difference. For singular covariance estimates, an inverse needs regularization or a pseudoinverse; ARENA uses a pseudoinverse.[^arena]
    
- **Cross-dataset testing makes “a truth direction” a much more interesting claim.** Training on cities and testing on translations or numerical comparisons asks whether the probe found something shared across topics. High accuracy on a single dataset could be exploiting a much narrower distinction.
    
    The truth-versus-likelihood issue was especially useful. I initially imagined subtracting out a “likely text” direction. My understanding changed: the paper instead trains separate likely/unlikely probes and checks how they transfer. Negated statements are a nice test because truth and likelihood can become anticorrelated. “Paris is not in France” can be a probable continuation while being false. The important finding is truth-relevant information beyond text plausibility, rather than just another successful classifier.[^truth]
    
- **CCS is a caution about what accuracy establishes, rather than a reason to dismiss the entire method.** The point I want to retain is that a probe performing well on one dataset does not establish a reliable, general truth direction. I also found it interesting that so much of the performance can come from constructing contrast pairs, before introducing a special loss.
    
    > **ChatGPT clarification:** Original Contrast-Consistent Search learns a probe using consistency and confidence objectives on paired inputs. PCA on contrast-pair differences is a simpler alternative discussed in ARENA and Emmons’s analysis, rather than the original CCS algorithm.[^ccs]
    > 
    > The Geometry of Truth paper does evaluate CCS transfer and interventions. The useful distinction is between success in particular experiments and reliability across the range of settings we care about.[^truth]
    
- **Localizing a useful activation site and identifying a useful direction are separate experimental steps.** Layer-and-position patching now feels like an intuitive first move: find where replacing activations affects the answer, then investigate the geometry there. I don’t yet have an intuition for why truth-related information should be so localized, but it is a reasonable empirical starting point.
    
    > **ChatGPT clarification:** The later direction intervention adds or subtracts a vector at selected sites; it does not copy an entire activation from a donor run. ARENA scales the truth intervention using the class-mean separation along the candidate direction, and evaluates its effect on \(P(\texttt{TRUE})-P(\texttt{FALSE})\). A direction’s classification accuracy and its effect on that output metric are separate measurements.[^arena]
    
- **The instructed-pairs design is clever, but interpreting the difference as “intent” requires care.** At first the setup seemed very weird: give the model an honest or deceptive instruction, then supply the same factual response prefix in both cases. Excluding the end of the fact made the motivation clearer. The idea is to look before the model commits to a truthful or false completion, while keeping the supplied content fixed.
    
    My update is that this is an appealing way to construct a candidate signal for intent. Whether the signal tracks actual deception, instruction following, a persona, or recognition of a deception-related situation then has to be investigated. The paper’s observation that honest responses in deception-related scenarios also score highly is particularly relevant here.[^deception]
    
- **The difference between the paper’s evaluation and the exercise matters.** My concern was specifically about classifying an identical supplied response under different system prompts. That does not seem like the same test as catching deception in something the model actually chose to generate.
    
    > **ChatGPT source clarification:** The ARENA evaluation loop does use the same supplied answer for the two instruction variants, with labels assigned by variant. It is useful as an instruction-contrast test, but cannot by itself establish detection of model-chosen deceptive behavior.[^arena]
    > 
    > The paper distinguishes several settings:
    > 
    > |Setting|Where the response comes from|
    > |---|---|
    > |Main instructed-pairs training|A supplied factual prefix under honest/deceptive instructions|
    > |Alternative roleplaying training|Model-generated responses labeled for behavior|
    > |Main evaluation|On-policy model-generated responses in the evaluation scenarios|
    > 
    > So supplied responses are part of the paper’s training methodology too. The stronger test is transfer to generated behavior. Extracting activations from a recorded response in a later forward pass is compatible with on-policy evaluation: “on-policy” concerns how the response was originally collected, not whether it is regenerated each time the probe scores it.[^deception]
    
- **AUROC and a useful deployment threshold answer different questions.** The intuitive interpretation of AUROC helped: pick a random honest response and a random deceptive response; how often does the deceptive one receive the higher score? But I also want to know what happens at a particular threshold, with a practically acceptable false positive rate.
    
    The paper calibrates a threshold on ordinary, largely deception-unrelated chat. I can imagine a probe giving a cooking answer a low score, an honest confession of insider trading a medium score, and a lie concealing the trading a high score. It could rank the trading responses perfectly while a threshold calibrated on cooking answers flags both. This makes the choice of negative examples really important, and the authors observe a related issue.[^deception]
    
    > **ChatGPT mathematical clarification:** For a score \(s\),
    > 
    > \[ \mathrm{AUROC} =P\!\left(s(D)>s(H)\right) +\tfrac12P\!\left(s(D)=s(H)\right), \]
    > 
    > where \(D\) and \(H\) are independently sampled deceptive and honest examples from the specified evaluation distributions. A 1% false positive rate on an ordinary-chat control set does not establish a 1% false positive rate on honest responses in deception-related situations.
    
- **How token scores are aggregated can change the conclusion.** The same probe reads activations at multiple response positions, then the paper averages the scores to get one response-level score. A short deceptive passage can be diluted by a long low-scoring introduction. Looking at the token-level behavior therefore seems useful even when the final output is a single number.
    
    > **ChatGPT clarification:** The paper averages real-valued probe scores, rather than necessarily averaging probabilities after a sigmoid. A mean also discards where the high scores occur. Replacing it with a maximum changes the failure modes, rather than automatically solving the problem.[^deception]
    
- **Norm-scaled steering gives the coefficient a concrete interpretation.** The deception exercise uses the MM direction to intervene during generation, which brings the reading-versus-changing distinction back into view. The unhealthy-supplement sales example was a striking illustration. I still want to understand how the coefficient is picked and how to evaluate the resulting behavior beyond a few examples.
    
    > **ChatGPT mathematical context:** With a unit steering vector \(\hat v\), the exercise applies
    > 
    > \[ h'=h+\alpha\|h\|\hat v. \]
    > 
    > Thus the added vector has norm \(|\alpha|\|h\|\): its size is a fraction of the original activation norm. This is a perturbation-scaling convention, not normalization of the resulting activation. The exercise illustrates several coefficients; it does not derive an optimal one.[^arena]
    
- **For familiar interventions, my learning is shifting toward experimental design.** Once I understood exactly what the intervention code was doing, writing it out myself felt less valuable. I expect to use AI for that implementation work. If there is a fundamentally different intervention, I’d still want to inspect the code to understand the operation.
    

## Open questions and things to revisit

- **Why can middle layers be better for these probes than later layers?** I want a better account of what changes in the representation as it gets closer to the output.
    
- **Try an experiment of my own.** The combination of patching, simple geometry, and interventions makes this feel much more approachable than it did before.
    
- **Understand steering coefficients and evaluation better.** How should I choose the intervention strength, and distinguish the intended behavioral change from a change in style or other side effects?
    
- **Come back to probing SAE features after understanding SAEs themselves.** I noted this comparison in the deception paper, but have deferred the details.
    
- **The safety question remains whether internal probes help when the deception is no longer obvious from the transcript.** The paper compares against a black-box baseline, but does not establish that advantage in the harder settings that motivate this work.[^deception]
    

## References

[^arena]: [ARENA 1.3.1: Linear Probes](https://github.com/ARENA-education/ARENA_materials/blob/main/chapter1_transformer_interp/instructions/pages/11_%5B1.3.1%5D_Linear_Probes.md), especially the MM probe, intervention, instructed-pairs, and evaluation implementations. Checked October 28, 2026.  
[^truth]: Samuel Marks and Max Tegmark, [The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets](https://arxiv.org/html/2310.06824v3), especially sections 3–6 and appendix E.  
[^ccs]: Scott Emmons, [Contrast Pairs Drive the Empirical Performance of Contrast Consistent Search (CCS)](https://www.lesswrong.com/posts/9vwekjD6xyuePX7Zr/contrast-pairs-drive-the-empirical-performance-of-contrast). The original method is described in Burns et al., [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827).  
[^deception]: Nicholas Goldowsky-Dill, Bilal Chughtai, Stefan Heimersheim, and Marius Hobbhahn, [Detecting Strategic Deception Using Linear Probes](https://arxiv.org/html/2502.03407v1), especially sections 3–5 and appendices B and D. The published version is titled [Detecting Strategic Deception with Linear Probes](https://proceedings.mlr.press/v267/goldowsky-dill25a.html).