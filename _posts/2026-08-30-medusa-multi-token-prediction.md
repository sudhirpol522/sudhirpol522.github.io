---
layout: post
title: "Medusa From the Tensors Up: Self-Speculative Decoding Without a Draft Model"
date: 2026-08-30
description: A tensor-level guide to Medusa heads, shifted cross-entropy, candidate trees, verification, and the difference between Medusa and recursive MTP.
tags: llm-inference speculative-decoding medusa multi-token-prediction
categories: technical-writing
math: true
---

In my [previous post]({% post_url 2026-06-22-speculative-decoding-math %}), speculative decoding used a separate draft model. The draft proposed a short continuation, and the target model verified it in parallel. The mathematics was elegant, but the system now had to train, load, and serve two models.

Medusa asks a wonderfully practical question:

> **What if the target model could speculate for itself?**

Instead of loading a separate draft model, Medusa attaches a few lightweight prediction heads to the target model's final hidden state. The normal language-model head predicts the next token, while the extra heads predict tokens farther into the future.

That one sentence hides most of the interesting details. What exactly is a "head"? Which parameters are trained? How are the labels shifted? Why are later heads less accurate? How can four heads create a verification input with roughly 250 tokens? And why is the recursive MTP implementation in vLLM Speculators not the same architecture as Medusa?

This post answers those questions from the tensors upward.

## Part 1: One Hidden State, Not One Logits State

Consider the tokenized sentence:

> I like cats today .

Let the current position be the token `I`, at index $$t$$. After the Transformer processes the context through position $$t$$, it produces a final hidden state:

$$
h_t \in \mathbb{R}^{d},
$$

where $$d$$ is the model's hidden dimension. This is a hidden representation, not a vector of vocabulary logits.

The original LM head maps that representation to one score per vocabulary item:

$$
\ell_t^{(0)} = W_{\mathrm{LM}}h_t \in \mathbb{R}^{V},
$$

and predicts the immediate next token:

$$
h_t \xrightarrow{\text{LM head}} x_{t+1}.
$$

With three auxiliary Medusa heads, the same $$h_t$$ produces three additional vocabulary distributions:

$$
\begin{aligned}
h_t \xrightarrow{\text{Medusa head 1}} &\ x_{t+2} && \text{(cats)}, \\
h_t \xrightarrow{\text{Medusa head 2}} &\ x_{t+3} && \text{(today)}, \\
h_t \xrightarrow{\text{Medusa head 3}} &\ x_{t+4} && \text{(.)}.
\end{aligned}
$$

![Medusa's independent heads compared with recursive MTP]({{ '/assets/img/medusa-vs-recursive-mtp.svg' | relative_url }})

The distinction is important:

- There is one shared backbone state, $$h_t$$.
- There are multiple logit tensors, one from each prediction head.
- Classic Medusa heads do not receive the predictions or activations of earlier Medusa heads.

That last property makes the heads cheap and parallel, but it also explains why prediction quality generally deteriorates at more distant positions.

## Part 2: What Is Inside a Medusa Head?

The Medusa paper defines each auxiliary head as a residual feed-forward transformation followed by a vocabulary projection:

$$
p_t^{(k)} = \operatorname{softmax}\left(
W_2^{(k)}
\left[
\operatorname{SiLU}\left(W_1^{(k)}h_t\right)+h_t
\right]
\right).
$$

For head $$k$$:

- $$W_1^{(k)} \in \mathbb{R}^{d \times d}$$ is its residual transformation.
- $$W_2^{(k)} \in \mathbb{R}^{V \times d}$$ projects into the vocabulary.
- $$p_t^{(k)} \in \mathbb{R}^{V}$$ is a complete probability distribution over the vocabulary.

The paper initializes $$W_1^{(k)}$$ to zero and copies the original LM-head weights into $$W_2^{(k)}$$. At initialization, the residual branch contributes nothing, so every new head begins with the base model's next-token mapping rather than arbitrary logits.

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

If we configure three Medusa heads, we have three independent instances of this module plus the original LM head. The auxiliary projections have separate trainable parameters even though they share the same initialization.

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

This gives Medusa-1 three useful operational properties:

- No separate draft model must be trained and hosted.
- Training cannot alter the target model's original next-token behavior.
- The frozen backbone can be quantized while the auxiliary heads are trained, reducing memory use.

Medusa-2 changes the recipe by jointly updating the backbone and auxiliary heads with a carefully balanced objective. Here, I will stay with the simpler Medusa-1 setting.

## Part 4: The Loss Is Shifted Cross-Entropy

Suppose a batch contains this sequence of token IDs:

$$
[x_0,x_1,x_2,x_3,x_4]
=
[\text{I},\text{like},\text{cats},\text{today},\text{.}].
$$

At every input position, each head is trained against a different future offset:

| Prediction source | Logits at position $$t$$ target | Shift |
| --- | --- | ---: |
| Original LM head | $$x_{t+1}$$ | 1 |
| Medusa head 0 | $$x_{t+2}$$ | 2 |
| Medusa head 1 | $$x_{t+3}$$ | 3 |
| Medusa head 2 | $$x_{t+4}$$ | 4 |

For a Medusa head indexed from zero in Python:

$$
\operatorname{shift}_k = k+2.
$$

Its cross-entropy loss is:

$$
\mathcal{L}_k
=
-\frac{1}{N_k}
\sum_t
\log p_t^{(k)}\left(x_{t+k+2}\right),
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

For a sequence of length five, the valid pairs shrink as the shift grows:

| Head | Valid logits positions | Target positions | Training pairs |
| --- | --- | --- | ---: |
| Base, shift 1 | $$[0,1,2,3]$$ | $$[1,2,3,4]$$ | 4 |
| Medusa 0, shift 2 | $$[0,1,2]$$ | $$[2,3,4]$$ | 3 |
| Medusa 1, shift 3 | $$[0,1]$$ | $$[3,4]$$ | 2 |
| Medusa 2, shift 4 | $$[0]$$ | $$[4]$$ | 1 |

This reveals an easy-to-miss engineering detail: later heads receive fewer supervised positions near sequence boundaries. A sequence of length three can train shifts 1 and 2, but a shift of 3 produces empty tensors unless the implementation guards against it.

### One Numerical Cross-Entropy Calculation

Take a toy vocabulary with five tokens:

$$
[\text{I},\text{like},\text{cats},\text{dogs},\text{today}].
$$

At position `I`, Medusa head 0 is trained with shift 2, so its target is `cats`. Suppose it emits:

$$
\ell=[0,0,1.5,0,0].
$$

The probability assigned to `cats` is:

$$
\frac{e^{1.5}}{e^{1.5}+4e^0}
=
\frac{4.4817}{8.4817}
\approx 0.5284.
$$

Therefore, this position contributes:

$$
-\log(0.5284) \approx 0.6379.
$$

PyTorch performs this calculation for every valid batch-position pair and averages the negative log-probabilities.

### Weighting the Heads

Later positions are harder because they are more uncertain. The Medusa paper therefore proposes downweighting later-head losses:

$$
\mathcal{L}_{\text{Medusa-1}}
=
\sum_{k=1}^{K}\lambda_k\mathcal{L}_k,
\qquad
\lambda_k \approx 0.8^k.
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

Because the backbone is frozen, Medusa-1 does not need the original next-token loss to protect the base model. That loss can still be monitored as an evaluation metric, but it contributes no useful gradient to frozen parameters.

## Part 5: Why Later Heads Usually Have Higher Loss

A tempting intuition is that a later head has "more context" because it predicts a later token. In classic Medusa, the opposite is true: every auxiliary head receives the same context representation, $$h_t$$.

Head 3 must predict farther into the future without observing the tokens that heads 1 and 2 are trying to predict:

$$
\begin{aligned}
p(x_{t+2}\mid x_{\le t}) &\quad \text{Head 1}, \\
p(x_{t+3}\mid x_{\le t}) &\quad \text{Head 2}, \\
p(x_{t+4}\mid x_{\le t}) &\quad \text{Head 3}.
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
\mathcal{L}_{\text{head 3}}.
$$

This is an empirical tendency, not a guarantee for every minibatch. It motivates both decaying loss weights and architectures that explicitly feed earlier future tokens into later predictions.

## Part 6: MTP Is a Family, Not a Single Architecture

"Multi-token prediction" describes a task: predict several future tokens from one observed prefix. It does not uniquely specify the network architecture.

### Original Parallel MTP

The MTP work by Gloeckle et al. trains multi-token prediction during pretraining. A shared Transformer trunk produces a representation $$z_t$$, and independent head-specific Transformer layers predict several future positions:

$$
p_i
=
\operatorname{softmax}\left(
f_u\left(f_{h_i}(z_t)\right)
\right).
$$

The head transformations $$f_{h_i}$$ are independent, but the expensive unembedding matrix $$f_u$$ is shared. This differs from the separate auxiliary vocabulary projections described in the Medusa paper.

### Recursive MTP in vLLM Speculators

The current vLLM Speculators MTP implementation fine-tunes a model's native MTP layer in a recursive, FastMTP-style design. At each speculative step, the layer combines:

1. The previous hidden representation.
2. The embedding of the preceding future token.

For the sentence `I like cats today`, training proceeds conceptually as:

$$
z_0
=
\operatorname{MTP}\left(h_t,E(\text{like})\right)
\quad\longrightarrow\quad
\text{predict cats},
$$

$$
z_1
=
\operatorname{MTP}\left(z_0,E(\text{cats})\right)
\quad\longrightarrow\quad
\text{predict today}.
$$

The MTP layer is reused recursively. During training, the token embeddings come from ground-truth tokens through teacher forcing. During inference, they come from earlier predicted tokens.

With three speculative steps and the documented default decay parameter $$\beta=0.6$$, the normalized step weights are approximately:

$$
[\alpha_0,\alpha_1,\alpha_2]=[0.51,0.31,0.18],
$$

and the total loss is:

$$
0.51\mathcal{L}_0
+0.31\mathcal{L}_1
+0.18\mathcal{L}_2.
$$

The verifier, embedding table, and LM head remain frozen and shared; only the native MTP layer is fine-tuned.

| Property | Medusa-1 | Original parallel MTP | vLLM recursive MTP |
| --- | --- | --- | --- |
| Training stage | Post-training | Pretraining | Fine-tuning native MTP weights |
| Future-position modules | Independent residual heads | Independent Transformer heads | One recursively reused MTP layer |
| Earlier future tokens visible | No | No | Yes |
| Token embeddings used by predictor | No | No | Yes |
| Vocabulary projection | Separate auxiliary projections | Shared unembedding | Shared frozen LM head |
| Backbone updated | No | Yes | No |

Calling all three approaches "MTP" is reasonable at the task level. Calling their implementations identical is not.

## Part 7: Training Heads and Building a Tree Are Different Operations

Training uses one ground-truth label per future position. There is no Cartesian product, candidate tree, or tree-attention mask in the Medusa-1 training loss.

The tree appears only during inference.

Each trained Medusa head emits $$V$$ logits. We retain only its highest-ranked candidates. Define:

$$
s_i = \text{number of top candidate tokens retained from head }i.
$$

The values $$s_i$$ are inference hyperparameters. They are not learned weights, and they are not the number of Medusa heads.

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

The number of complete candidates, or leaves, is:

$$
s_1s_2 = 2 \times 2 = 4.
$$

But the verification tree contains both intermediate prefix nodes and leaves:

$$
q=s_1+s_1s_2=2+4=6.
$$

Tree attention packs those six candidate-token positions into one target-model verification pass. Its mask ensures that each node can see only the prompt and the ancestors on its own branch.

### Where Does an Additional Length Near 250 Come From?

With $$K$$ Medusa heads, a regular Cartesian tree has:

$$
q
=
s_1
+s_1s_2
+s_1s_2s_3
+\cdots
+\prod_{i=1}^{K}s_i.
$$

Suppose four heads retain:

$$
(s_1,s_2,s_3,s_4)=(5,5,3,2).
$$

Then the tree contains:

$$
\begin{aligned}
\text{depth 1:}&\quad 5, \\
\text{depth 2:}&\quad 5\times5=25, \\
\text{depth 3:}&\quad 5\times5\times3=75, \\
\text{depth 4:}&\quad 5\times5\times3\times2=150.
\end{aligned}
$$

Therefore:

$$
q=5+25+75+150=255.
$$

That is how four learned heads can create an additional verification length near 250. The model does not contain 250 heads. It contains four heads whose retained alternatives create 255 contextual token nodes.

In practice, the tree need not be a full Cartesian product. Low-value branches can be pruned, producing an irregular tree that spends the verification budget on more promising paths.

## Part 8: Verification Is Where the Speedup Comes From

The Medusa heads are cheap, but their outputs are only proposals. The target model still decides which continuation is valid.

Tree attention makes that verification efficient:

1. Flatten the candidate tree into token positions.
2. Assign position IDs according to tree depth.
3. Construct an attention mask so each node sees only its branch ancestors.
4. Run the target model over all tree nodes in parallel.
5. Accept the longest valid prefix under the selected acceptance rule.

Under greedy decoding, matching the target's greedy choices preserves the original output exactly. Rejection sampling can preserve the target distribution for sampling. Medusa's optional typical-acceptance rule can accept more tokens, but it relaxes exact distribution matching in exchange for additional speed while aiming to preserve generation quality.

Increasing $$s_i$$ makes it more likely that the target model finds a useful branch, but it also increases $$q$$ and therefore verification cost. This creates an interior optimum like the one in standard speculative decoding: more speculation helps until verification overhead dominates.

The Medusa paper's hardware model illustrates this trade-off clearly. In one simulated setting, speedup improves as the candidate-token count grows, peaks around 64 candidates, and then declines as extra verification work overwhelms the benefit.

## Part 9: What I Would Measure as an MLE

A lower training loss is necessary, but it is not the final objective. An inference optimization should be evaluated end to end.

| Metric | Why it matters |
| --- | --- |
| Per-head cross-entropy | Shows how uncertainty grows with prediction distance |
| Per-head top-1 and top-$$k$$ accuracy | Helps choose useful branching factors |
| Mean accepted tokens per verification | Measures actual decoding-step compression |
| Candidate-tree node count $$q$$ | Captures the verification workload |
| Target forward calls per generated token | Measures saved sequential work independently of hardware |
| Time to first token | Confirms that the optimization targets decode rather than prompt prefill |
| Inter-token latency | Captures the user-visible low-batch latency improvement |
| Throughput under concurrency | Reveals when ordinary batching already saturates the GPU |
| Greedy exact match or sampling-distribution tests | Validates the selected acceptance rule |

Wall-clock speedup is workload-dependent. Medusa is most compelling when autoregressive decoding is memory-bandwidth-bound and the serving batch is small enough that the GPU has spare arithmetic capacity. At high concurrency, ordinary batching may already use that capacity, making a larger verification tree less attractive.

## The Mental Model to Keep

Medusa-1 is easiest to understand as three separate stages:

1. **Training:** Freeze the target model and train independent residual heads with shifted future-token cross-entropy.
2. **Candidate construction:** At inference, keep selected top tokens from each head and combine them into a tree.
3. **Verification:** Run the target model once over that tree with a branch-aware attention mask and accept the longest valid prefix.

Recursive MTP keeps the same high-level goal, but changes the predictor. Instead of independent parallel heads over one hidden state, it reuses an MTP layer and feeds previous speculative activations and token embeddings forward.

That distinction is more than terminology. It determines parameter sharing, exposure bias, loss alignment, serving integration, and how quickly prediction quality degrades with depth.

The broader engineering lesson is simple: "predict multiple tokens" describes an objective, not an implementation. The speedup lives in the interaction between the training architecture, proposal structure, acceptance rate, and hardware cost of verification.

---

## References

- Cai et al., [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- FasterDecoding, [official Medusa implementation](https://github.com/FasterDecoding/Medusa)
- Gloeckle et al., [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- vLLM Project, [Speculators MTP documentation](https://docs.vllm.ai/projects/speculators/en/stable/user_guide/algorithms/mtp/)
