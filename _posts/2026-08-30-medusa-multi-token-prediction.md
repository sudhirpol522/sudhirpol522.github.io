---
layout: post
title: "Medusa From the Tensors Up: Self-Speculative Decoding Without a Draft Model"
date: 2026-08-30
description: A tensor-level guide to Medusa heads, shifted cross-entropy, candidate trees, verification, and the difference between Medusa and recursive MTP.
tags: llm-inference speculative-decoding medusa multi-token-prediction
categories: technical-writing
math: true
---

In my [previous post]({% post_url 2026-06-22-speculative-decoding-math %}), we gave speculative decoding a separate draft model. The draft proposed a short continuation; the target model verified it in parallel. The mathematics was elegant, but the system now had to host two models.

Medusa asks a wonderfully practical question:

> **What if the target model could speculate for itself?**

Instead of loading a separate draft model, Medusa attaches a few lightweight prediction heads to the target model's final hidden state. The normal language-model head predicts the next token, while the extra heads predict tokens further into the future.

That one sentence hides most of the interesting details. What exactly is a "head"? Which parameters are trained? How are the labels shifted? Why are later heads less accurate? How can four heads create a verification input with roughly 250 tokens? And why is the recursive MTP implementation in vLLM Speculators not the same architecture as Medusa?

This post answers those questions from the tensors upward.

## Part 1: Start With One Hidden State, Not One Logits State

Consider the tokenized sentence:

> I like cats today .

Let the current position be the token `I`, at index $$t$$. After the Transformer processes the context through position $$t$$, it produces a final hidden state:

$$
h_t \in \mathbb{R}^{d}
$$

where $$d$$ is the model's hidden dimension. This is a hidden representation, not a vector of vocabulary logits.

The original LM head maps that representation into one score per vocabulary item:

$$
\ell_t^{(0)} = W_{\text{LM}}h_t \in \mathbb{R}^{V}
$$

and predicts the immediate next token, like:

$$
h_t \xrightarrow{\text{LM head}} x_{t+1}
$$

With three auxiliary Medusa heads, the same $$h_t$$ also produces three additional vocabulary distributions:

$$
\begin{aligned}
h_t \xrightarrow{\text{Medusa head 1}} &\ x_{t+2} && \text{(cats)} \\
h_t \xrightarrow{\text{Medusa head 2}} &\ x_{t+3} && \text{(today)} \\
h_t \xrightarrow{\text{Medusa head 3}} &\ x_{t+4} && \text{(.)}
\end{aligned}
$$

![Medusa's independent heads compared with recursive MTP]({{ '/assets/img/medusa-vs-recursive-mtp.svg' | relative_url }})

Medusa sends one backbone state into independent future-token heads. Recursive MTP feeds each MTP output into the next speculative step.

The distinction matters:

- There is one shared backbone state $$h_t$$.
- There are multiple logit tensors, one from each prediction head.
- Classic Medusa heads do not receive the predictions or activations of earlier heads.

That last property makes Medusa cheap and parallel, but it also explains why accuracy deteriorates for more distant positions.

## Part 2: What Is Inside a Medusa Head?

The Medusa paper implements each auxiliary head as a residual feed-forward transformation followed by a vocabulary projection:

$$
p_t^{(k)} = \operatorname{softmax}\left(
W_2^{(k)}
\left[
\operatorname{SiLU}\left(W_1^{(k)}h_t\right)+h_t
\right]
\right)
$$

For head $$k$$:

- $$W_1^{(k)} \in \mathbb{R}^{d \times d}$$ is its residual transformation.
- $$W_2^{(k)} \in \mathbb{R}^{V \times d}$$ projects into the vocabulary.
- $$p_t^{(k)} \in \mathbb{R}^{V}$$ is a complete probability distribution over the vocabulary.

Medusa initializes $$W_1^{(k)}$$ to zero and copies the original LM-head weights into $$W_2^{(k)}$$. At initialization, the residual branch contributes nothing, so every new head begins close to the already trained LM head instead of emitting random logits.

A minimal PyTorch version looks like this:

```python
class MedusaHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, base_lm_head):
        super().__init__()

        self.residual = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )
        self.vocab_projection = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
        )

        nn.init.zeros_(self.residual.weight)
        self.vocab_projection.weight.data.copy_(
            base_lm_head.weight.data
        )

    def forward(self, hidden_states):
        transformed = F.silu(self.residual(hidden_states))
        return self.vocab_projection(hidden_states + transformed)
```

If we configure three Medusa heads, we have three independent instances of this module, plus the original LM head. In Medusa, those auxiliary vocabulary projections have separate parameters, even though they start from the same initialization.

## Part 3: Medusa-1 Freezes the Backbone

Medusa-1 is a post-training recipe. We begin with an already trained target model and freeze it:

```python
for parameter in backbone.parameters():
    parameter.requires_grad = False
```

Only the newly attached Medusa heads receive optimizer updates:

```python
optimizer = torch.optim.AdamW(
    medusa_heads.parameters(),
    lr=1e-3,
)
```

The token embeddings, Transformer blocks, normalization layers, and original LM head are all part of the frozen backbone. We still run the backbone to obtain hidden states, but gradients do not update it.

This is the central operational advantage of Medusa-1:

- No separate draft model must be trained and hosted.
- The target model's normal next-token behavior cannot be damaged by training.
- The backbone may even be quantized while training the auxiliary heads.

Medusa-2 changes this recipe by jointly updating the backbone and auxiliary heads with a carefully balanced objective. For now, we will remain in the simpler Medusa-1 setting.

## Part 4: The Loss Is Just Shifted Cross-Entropy

Suppose our batch contains a sequence of token IDs:

$$
[x_0,x_1,x_2,x_3,x_4] = [\text{I},\text{like},\text{cats},\text{today},\text{.}]
$$

At every input position, each head is trained against a different future offset:

| Prediction source | Logits at position $$t$$ target | Shift |
| --- | --- | ---: |
| Original LM head | $$x_{t+1}$$ | 1 |
| Medusa head 0 | $$x_{t+2}$$ | 2 |
| Medusa head 1 | $$x_{t+3}$$ | 3 |
| Medusa head 2 | $$x_{t+4}$$ | 4 |

For head $$k$$, where Python indexing begins at zero:

$$
\text{shift}_k = k+2
$$

The cross-entropy loss is:

$$
-\frac{1}{N_k}
\sum_t
\log p_t^{(k)}\left(x_{t+k+2}\right)
$$

where $$N_k$$ is the number of valid, non-padding targets for that head.

The slicing operation is surprisingly small:

```python
def shifted_cross_entropy(logits, labels, shift, ignore_index=-100):
    """
    logits[:, t] predicts labels[:, t + shift].
    """
    if shift <= 0:
        raise ValueError("shift must be positive")
    if logits.size(1) <= shift:
        raise ValueError(
            f"sequence length {logits.size(1)} must exceed shift {shift}"
        )

    shifted_logits = logits[:, :-shift, :].contiguous()
    shifted_labels = labels[:, shift:].contiguous()

    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)).float(),
        shifted_labels.reshape(-1),
        ignore_index=ignore_index,
    )
```

For a sequence of length five:

| Head | Valid logits positions | Target positions | Number of training pairs |
| --- | --- | --- | ---: |
| Base, shift 1 | $$[0,1,2,3]$$ | $$[1,2,3,4]$$ | 4 |
| Medusa 0, shift 2 | $$[0,1,2]$$ | $$[2,3,4]$$ | 3 |
| Medusa 1, shift 3 | $$[0,1]$$ | $$[3,4]$$ | 2 |
| Medusa 2, shift 4 | $$[0]$$ | $$[4]$$ | 1 |

This reveals an easy-to-miss engineering detail: later heads receive fewer supervised positions near sequence boundaries. A sequence of length three can train only shift 1 and shift 2. Attempting shift 3 produces empty tensors unless the implementation guards against it.

### One Numerical Cross-Entropy Calculation

Take a toy vocabulary of five tokens:

$$
[\text{I}, \text{like}, \text{cats}, \text{dogs}, \text{today}]
$$

At position `I`, Medusa head 0 is trained with shift 2, so its target is `cats`. Suppose it emits:

$$
\ell=[0,0,1.5,0,0]
$$

The probability assigned to `cats` is:

$$
\frac{e^{1.5}}{e^{1.5}+4e^0}
=
\frac{4.4817}{8.4817}
\approx 0.5284
$$

Therefore this position contributes:

$$
-\log(0.5284)
\approx 0.6379
$$

PyTorch performs this calculation for every valid batch-position pair and averages the resulting negative log-probabilities.

### Weighting the Heads

Later positions are harder because they are more uncertain. The Medusa paper therefore downweights later-head losses:

$$
\sum_{k=1}^{K}\lambda_k\mathcal{L}_k,
\qquad
\lambda_k \approx 0.8^k
$$

In Python:

```python
def medusa1_loss(medusa_logits, labels, decay=0.8):
    losses = []

    for head_index, logits in enumerate(medusa_logits):
        loss = shifted_cross_entropy(
            logits,
            labels,
            shift=head_index + 2,
        )
        losses.append(loss)

    total = sum(
        decay ** (head_index + 1) * loss
        for head_index, loss in enumerate(losses)
    )

    return total, losses
```

Because the backbone is frozen, Medusa-1 does not need the original next-token loss to protect the base model. The base-model loss is useful as an evaluation metric, but it contributes no gradient to the frozen backbone.

## Part 5: Why Later Medusa Heads Usually Have Higher Loss

A common intuition is that the later head has "more context" because it predicts a later token. In classic Medusa, the opposite is true: every head receives the same context representation $$h_t$$.

Head 3 is asked to predict further into the future without observing the tokens that heads 1 and 2 are trying to predict.

Using our example:

$$
\begin{aligned}
p(x_{t+2}\mid x_{\le t}) &\quad \text{Head 1} \\
p(x_{t+3}\mid x_{\le t}) &\quad \text{Head 2} \\
p(x_{t+4}\mid x_{\le t}) &\quad \text{Head 3}
\end{aligned}
$$

The later distributions average over more possible unseen continuations. Their entropy tends to be higher, so the typical aggregate ordering is:

$$
\mathcal{L}_{\text{base}}
<
\mathcal{L}_{\text{head 1}}
<
\mathcal{L}_{\text{head 2}}
<
\mathcal{L}_{\text{head 3}}
$$

This is an empirical tendency, not a guarantee for every minibatch. But it motivates both the decaying loss weights and architectures that explicitly feed earlier future tokens into later predictions.

## Part 6: MTP Is a Family, Not a Single Architecture

"Multi-token prediction" describes the task: predict several future tokens from one observed prefix. It does not uniquely specify the network architecture.

### Original Parallel MTP

The original MTP paper by Gloeckle et al. trains the capability during pretraining. A shared Transformer trunk produces a representation $$z_t$$, and independent head-specific Transformer layers predict several future positions:

$$
\operatorname{softmax}\left(
f_u\left(f_{h_i}(z_t)\right)
\right)
$$

The head transformations $$f_{h_i}$$ are independent, but the expensive unembedding matrix $$f_u$$ is shared. This differs from Medusa's separate auxiliary vocabulary projections.

### Recursive FastMTP in vLLM Speculators

The current vLLM Speculators MTP implementation follows a recursive FastMTP-style design. It combines two inputs at each speculative step:

- The previous hidden representation.
- The embedding of the preceding future token.

For the sentence `I like cats today`, training proceeds as:

$$
z_0 = \operatorname{MTP}\left(h_t,E(\text{like})\right)
\quad\rightarrow\quad
\text{predict cats}
$$

$$
z_1 = \operatorname{MTP}\left(z_0,E(\text{cats})\right)
\quad\rightarrow\quad
\text{predict today}
$$

The same MTP layer is reused recursively. During training, the token embeddings come from ground-truth tokens -- teacher forcing. During inference, they come from previously predicted tokens.

With three speculative steps and the default decay parameter $$\beta=0.6$$, the normalized step weights are approximately:

$$
[\alpha_0,\alpha_1,\alpha_2]=[0.51,0.31,0.18]
$$

and the total loss is:

$$
0.51\mathcal{L}_0
+0.31\mathcal{L}_1
+0.18\mathcal{L}_2
$$

The verifier, embedding table, and LM head remain frozen and shared; only the MTP layer is fine-tuned.

| Property | Medusa-1 | Original parallel MTP | vLLM FastMTP-style MTP |
| --- | --- | --- | --- |
| Training stage | Post-training | Pretraining | Fine-tuning native MTP weights |
| Future-position modules | Independent residual heads | Independent Transformer heads | One recursively reused MTP layer |
| Earlier future tokens visible | No | No | Yes |
| Token embeddings used by predictor | No | No | Yes |
| Vocabulary projection | Separate per auxiliary head | Shared unembedding | Shared frozen LM head |
| Backbone updated | No | Yes | No |

Calling all three approaches "MTP" is reasonable at the task level. Calling their implementations identical is not.

## Part 7: Training Heads and Building a Tree Are Different Operations

Training uses one ground-truth label per future position. There is no Cartesian product, no candidate tree, and no tree-attention mask in the Medusa-1 loss.

The tree appears only during inference.

Each trained Medusa head emits $$V$$ logits. We retain only its highest-ranked candidates. Define:

$$
s_i = \text{number of top candidate tokens retained from head }i
$$

The values $$s_i$$ are inference hyperparameters. They are not learned weights and they are not the number of Medusa heads.

### A Two-Head Example

Suppose the base LM head selects `like`. Then:

- Medusa head 1 retains `cats` and `dogs`, so $$s_1=2$$.
- Medusa head 2 retains `today` and `too`, so $$s_2=2$$.

![A two-level Medusa candidate tree with two choices per head]({{ '/assets/img/medusa-candidate-tree.svg' | relative_url }})

Because classic Medusa head 2 did not condition on head 1, its same two token labels are attached below both first-level branches. They become distinct verification nodes because `today` after `like cats` has a different context from `today` after `like dogs`.

The four complete candidate continuations are:

```text
like cats today
like cats too
like dogs today
like dogs too
```

The number of complete candidates -- the leaves -- is:

$$
s_1s_2 = 2 \times 2 = 4
$$

But the verification tree contains both intermediate prefix nodes and leaves:

$$
q=s_1+s_1s_2=2+4=6
$$

Tree attention packs those six candidate-token positions into one target-model verification pass. Its attention mask ensures that each node can see only the prompt and the ancestors on its own branch.

### Where Does an "Additional Length" Near 250 Come From?

With $$K$$ Medusa heads, a regular Cartesian tree has:

$$
s_1
+s_1s_2
+s_1s_2s_3
+\cdots
+\prod_{i=1}^{K}s_i
$$

Suppose four heads retain:

$$
(s_1,s_2,s_3,s_4)=(5,5,3,2)
$$

Then the tree contains:

$$
\begin{aligned}
\text{depth 1:}&\quad 5 \\
\text{depth 2:}&\quad 5\times5=25 \\
\text{depth 3:}&\quad 5\times5\times3=75 \\
\text{depth 4:}&\quad 5\times5\times3\times2=150
\end{aligned}
$$

Therefore:

$$
q=5+25+75+150=255
$$

That is how four learned heads can create an additional verification length near 250. The model does not contain 250 Medusa heads. It contains four heads whose retained alternatives create 255 contextual token nodes.

## Part 8: Verification Is Where the Speedup Comes From

The Medusa heads are cheap, but their predictions are speculative. The target model must still decide which branch agrees with normal decoding.

Tree attention makes that verification efficient:

1. Flatten the candidate tree into token positions.
2. Assign position IDs according to tree depth.
3. Construct an attention mask so each node sees only its branch ancestors.
4. Run the target model over all tree nodes in parallel.
5. Accept the longest valid prefix under the chosen acceptance rule.

Increasing $$s_i$$ improves the probability that the target model finds a good branch, but it also increases $$q$$ and therefore verification cost. This creates the same kind of interior optimum we saw in standard speculative decoding: more speculation helps until verification overhead dominates.

The later Medusa report recommends training at most five heads and notes that optimized inference often needs only three or four. One illustrated optimized configuration uses four heads but only 64 candidate-token nodes after pruning low-value branches.

## Part 9: What I Would Measure as an MLE

A lower training loss is necessary, but it is not the final objective. An inference optimization should be evaluated end to end.

I would track:

| Metric | Why it matters |
| --- | --- |
| Per-head cross-entropy | Shows how uncertainty grows with prediction distance |
| Per-head top-1 and top-$$k$$ accuracy | Determines useful candidate branching factors |
| Mean accepted tokens per verification | Measures actual decoding-step compression |
| Candidate-tree node count $$q$$ | Captures verification workload |
| Target forward calls per generated token | Hardware-independent indicator of saved sequential work |
| Time to first token | Medusa mainly accelerates decode, not prompt prefill |
| Inter-token latency | The user-visible low-batch latency metric |
| Throughput under concurrency | Speculation can lose its advantage when the GPU is already saturated |
| Exact-match output under greedy decoding | Validates lossless verification behavior |

Wall-clock speedup is workload-dependent. Medusa is most compelling when autoregressive decoding is memory-bandwidth-bound and the serving batch is small enough that the GPU has spare arithmetic capacity. At high concurrency, ordinary batching may already use that capacity, making a larger verification tree less attractive.

## The Mental Model to Keep

Medusa-1 is easiest to understand as three separate stages:

1. **Training:** Freeze the target model. Train independent residual heads with shifted future-token cross-entropy.
2. **Candidate construction:** At inference, keep $$s_i$$ top tokens from each head and combine them into a tree.
3. **Verification:** Run the target model once over the tree with a branch-aware attention mask and accept the longest valid prefix.

Recursive MTP keeps the same high-level goal -- predict multiple future tokens -- but changes the predictor. Instead of independent parallel heads over one hidden state, it reuses an MTP layer and feeds previous speculative activations and token embeddings forward.

That distinction is more than terminology. It determines parameter sharing, exposure bias, loss alignment, serving integration, and how quickly prediction quality degrades with depth.

The broader engineering lesson is simple: "predict multiple tokens" describes an objective, not an implementation. The speedup lives in the interaction between the training architecture, the proposal structure, the acceptance rate, and the hardware cost of verification.

## References

- Cai et al., [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- FasterDecoding, [Official Medusa implementation](https://github.com/FasterDecoding/Medusa)
- Gloeckle et al., [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- vLLM Project, [Speculators MTP documentation](https://docs.vllm.ai/projects/speculators/en/stable/user_guide/algorithms/mtp/)
- vLLM Project, MTP training implementation.
