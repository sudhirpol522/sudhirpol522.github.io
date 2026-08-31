---
layout: post
title: "Medusa From Training to Tree Verification: A Complete Implementation Guide"
date: 2026-08-30 12:00:00 -0600
description: "A tensor-level guide to Medusa training, sparse candidate trees, exact verification, an official Vicuna-7B A100 trace, and the modern MTP taxonomy."
tags: llm-inference speculative-decoding medusa multi-token-prediction pytorch
categories: machine-learning
math: true
---

Autoregressive language models have an awkward hardware problem: a decoding step moves billions of parameters through the GPU and usually produces only one token. Speculative decoding improves that ratio by letting a cheap draft model propose several tokens and asking the expensive target model to verify them together. It works, but now the serving system owns two models.

Medusa asks a beautifully direct question:

What if the target model carried its own lightweight drafting mechanism?

It adds several future-token prediction heads to one LLM, turns their top predictions into a candidate tree, verifies that tree with the same LLM, and commits the longest accepted prefix. There is no separately hosted draft model.

That one-sentence description hides most of the details that matter in an implementation:

- Does every head receive the same hidden state?
- Which token is head 0 trained to predict?
- Is the target model frozen?
- Is the tree used during training?
- Why does the reference implementation calculate top-10 for every head but verify only 64 nodes?
- Where do 42 candidate paths come from?
- What is the difference between an accepted Medusa token and a committed token?
- If the heads are part of the model, who rejects their predictions?
- Is Medusa the same architecture as modern multi-token prediction, or MTP?

This article answers those questions in chronological order. We will train the heads before discussing inference, construct a regular tree before optimizing it, choose a fixed tree before filling it with runtime tokens, and generate candidates before verifying them. At the end, we will interpret an actual A100 trace from the official Vicuna-7B Medusa checkpoint.

The central mental model is:

> Medusa has one target LLM, lightweight auxiliary draft heads, and a target-verification pass. The draft heads propose; the original LM decides.

## A Short Historical Map

The terminology became confusing because related ideas arrived with different training and architectural choices.

| Year | Work | Important change |
| --- | --- | --- |
| 2018 | Blockwise Parallel Decoding | Used multiple output heads to predict future positions in parallel. |
| 2023 | Together AI's Medusa post | Introduced Medusa heads, tree attention, typical acceptance, and the head-configuration ablation. |
| 2024 | Medusa technical report | Formalized Medusa-1, Medusa-2, self-distillation, and optimized sparse trees. |
| 2024 | Better & Faster LLMs via MTP | Made parallel future-token prediction a pretraining objective with independent heads. |
| 2024-2025 | DeepSeek-V3 | Used sequential MTP modules that preserve a causal chain between future positions. |
| 2025 onward | FastMTP and vLLM Speculators MTP | Reused a native MTP prediction layer recursively and fine-tuned it for speculation. |

All of these methods predict multiple future tokens. They are not all the same network.

## Part 1: Why Decoding One Token at a Time Is Expensive

Let the current tokenized context be

$$
x_{\le t}=[x_1,x_2,\ldots,x_t].
$$

The Transformer backbone produces a final hidden state at the last position:

$$
h_t\in\mathbb{R}^{d}.
$$

The ordinary language-model head maps it to one score per vocabulary token:

$$
\ell_t^{(0)}=W_{\text{LM}}h_t\in\mathbb{R}^{V},
$$

where $$d$$ is the hidden width and $$V$$ is the vocabulary size. Greedy decoding chooses

$$
x_{t+1}=\arg\max_v \ell_{t,v}^{(0)}.
$$

The new token is appended, and another decoding step is needed for $$x_{t+2}$$. The dependency is sequential:

$$
x_{\le t}
\rightarrow x_{t+1}
\rightarrow x_{t+2}
\rightarrow x_{t+3}.
$$

At low batch sizes, decoding is commonly memory-bandwidth-bound: the GPU repeatedly reads the large model weights while doing too little arithmetic per read. The objective is not to make the target model cheaper. It is to obtain several useful tokens from one expensive target-model interaction.

Keep the tensor names distinct:

- $$h_t$$ is a hidden vector of length $$d$$.
- $$\ell_t$$ is a logit vector of length $$V$$.
- `argmax(logits)` is a token ID.
- Decoding that token ID produces a text fragment, which may be only part of a word.

This distinction becomes important when several heads all consume the same hidden state but emit different vocabulary distributions.

## Part 2: Vanilla Speculative Decoding Still Has a Target

Traditional speculative decoding introduces two autoregressive models:

- A cheap draft model $$q$$ proposes $$\gamma$$ tokens.
- The expensive target model $$p$$ evaluates those proposed positions in one forward pass.

For greedy generation, verification is easy to state. Starting at the first drafted position, accept tokens while

$$
\text{draft token}_j
=
\arg\max_v p(v\mid x_{\le t+j-1}).
$$

The first mismatch ends the accepted prefix. A later token cannot be accepted after an earlier rejection because its context contains the rejected token.

For sampling, exact speculative decoding uses a rejection-sampling correction involving both $$p$$ and $$q$$. That procedure can preserve the target distribution exactly. The mathematical details are covered in [Unpacking Speculative Decoding]({% post_url 2026-06-22-speculative-decoding-math %}).

Two guarantees are often mixed together:

- **Progress:** every iteration can commit at least one target-approved token.
- **Draft acceptance:** there is no guarantee that even one speculative token will be accepted.

If every draft token is wrong, generation still advances by one target token. Speculation changes how many target-approved tokens can be harvested from a verification step; it never removes the verifier.

The operational cost is that the draft model brings its own parameters, KV cache, checkpoint lifecycle, device placement, and distributed-serving configuration. Medusa removes that separately hosted model--not the verification step.

## Part 3: Medusa Puts the Draft Heads on the Target Model

Assume we attach $$K=3$$ Medusa heads. At position $$t$$, the prediction roles are

$$
\begin{aligned}
\text{Original LM head}(h_t)&\rightarrow x_{t+1},\\
\text{Medusa head 0}(h_t)&\rightarrow x_{t+2},\\
\text{Medusa head 1}(h_t)&\rightarrow x_{t+3},\\
\text{Medusa head 2}(h_t)&\rightarrow x_{t+4}.
\end{aligned}
$$

![One target hidden state feeding the original LM head and three independent Medusa heads]({{ '/assets/img/medusa-independent-heads.svg' | relative_url }})

There is still only one Transformer backbone. The four prediction modules receive the same final backbone representation.

This gives us three precise statements:

- `medusa_num_heads = 3` means three additional heads plus the original LM head.
- Head 1 does not consume the token selected by head 0.
- A later Medusa head does not receive more observed context than an earlier one.

Each head models a different marginal distribution from the same prefix:

$$
\begin{aligned}
p_1(x_{t+2}\mid x_{\le t}),\\
p_2(x_{t+3}\mid x_{\le t}),\\
p_3(x_{t+4}\mid x_{\le t}).
\end{aligned}
$$

The heads are parallel and conditionally independent given $$h_t$$ in the architecture. The future tokens themselves are not independent in natural language; that mismatch is why farther heads are harder to train.

### What is inside one Medusa head?

The Medusa report uses a residual feed-forward transformation followed by a vocabulary projection:

$$
\operatorname{softmax}\left(
W_2^{(k)}
\left[
\operatorname{SiLU}(W_1^{(k)}h_t)+h_t
\right]
\right).
$$

The shapes are

$$
W_1^{(k)}\in\mathbb{R}^{d\times d},
\qquad
W_2^{(k)}\in\mathbb{R}^{V\times d}.
$$

A compact PyTorch version is:

```python
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states):
        return hidden_states + F.silu(self.linear(hidden_states))

class MedusaHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, lm_head_weight):
        super().__init__()
        self.residual = ResidualBlock(hidden_size)
        self.vocab_projection = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
        )

        with torch.no_grad():
            self.vocab_projection.weight.copy_(lm_head_weight)

    def forward(self, hidden_states):
        return self.vocab_projection(self.residual(hidden_states))
```

The residual block starts as the identity because its learned branch is zero-initialized. The vocabulary projection is initialized from the original LM head, but every Medusa head owns and subsequently trains its own copy.

Three heads therefore mean three residual blocks and three vocabulary projections--not one projection evaluated at three offsets.

## Part 4: Medusa-1 and Medusa-2 Train Different Parameter Sets

The candidate tree does not exist during head training. Training first teaches future-token distributions; inference later turns those distributions into a tree.

### Medusa-1: freeze the complete target model

Medusa-1 starts from an already trained LLM and freezes:

- token embeddings,
- all Transformer layers,
- normalization layers,
- the original language-model head.

Only the new Medusa heads are trainable:

```python
for parameter in base_model.parameters():
    parameter.requires_grad_(False)

medusa_heads = nn.ModuleList(
    MedusaHead(
        hidden_size=base_model.config.hidden_size,
        vocab_size=base_model.config.vocab_size,
        lm_head_weight=base_model.get_output_embeddings().weight,
    )
    for _ in range(number_of_medusa_heads)
)

optimizer = torch.optim.AdamW(
    medusa_heads.parameters(),
    lr=1e-3,
)
```

The frozen model still runs a forward pass to produce $$h_t$$. Frozen means that its parameters receive no optimizer update; it does not mean the backbone is absent from training.

Because the original model weights do not change, its ordinary next-token behavior remains intact. This makes Medusa-1 a post-hoc, parameter-efficient augmentation.

### Medusa-2: update the target and the heads jointly

Medusa-2 jointly trains the target backbone and Medusa heads. The report uses a combined objective, differential learning rates, and head warm-up so the large auxiliary gradients do not damage next-token capability.

| Property | Medusa-1 | Medusa-2 |
| --- | --- | --- |
| Starting point | Existing target LLM | Existing or jointly fine-tuned LLM |
| Backbone | Frozen | Trainable |
| Token embeddings | Frozen | May be updated with backbone recipe |
| Original LM head | Frozen | Protected by ordinary LM loss |
| Auxiliary heads | Trained | Trained |
| Main reason to use it | Cheap, post-hoc augmentation | Better head accuracy and acceptance |
| Original checkpoint behavior preserved | Yes, because target weights are unchanged | Not byte-for-byte; the jointly trained model becomes the new target |

The report measured about $$2.18\times$$ wall-time speedup for Medusa-1 and $$2.83\times$$ for Medusa-2 in its Vicuna-7B case study. These are workload- and implementation-specific measurements, not universal multipliers.

## Part 5: The Loss Is Shifted Cross-Entropy

Suppose a token sequence is

$$
[\text{I},\text{like},\text{cats},\text{today},\text{.}].
$$

The logit-to-label alignment is:

| Prediction source | `logits[:, t]` predicts | Shift |
| --- | --- | ---: |
| Original LM head | `labels[:, t + 1]` | 1 |
| Medusa head 0 | `labels[:, t + 2]` | 2 |
| Medusa head 1 | `labels[:, t + 3]` | 3 |
| Medusa head 2 | `labels[:, t + 4]` | 4 |

For zero-based Python head index $$j$$,

$$
\operatorname{shift}(j)=j+2.
$$

The reusable function is:

```python
def shifted_cross_entropy(
    logits,
    labels,
    shift,
    ignore_index=-100,
):
    """logits[:, t] predicts labels[:, t + shift]."""
    if shift <= 0:
        raise ValueError("shift must be positive")
    if logits.size(1) <= shift:
        raise ValueError("sequence is too short for this head")

    shifted_logits = logits[:, :-shift, :].contiguous()
    shifted_labels = labels[:, shift:].contiguous()

    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)).float(),
        shifted_labels.reshape(-1),
        ignore_index=ignore_index,
    )
```

For sequence length $$T=5$$:

| Predictor | Logit positions | Label positions | Valid pairs |
| --- | --- | --- | ---: |
| Base, shift 1 | $$[0,1,2,3]$$ | $$[1,2,3,4]$$ | 4 |
| Medusa 0, shift 2 | $$[0,1,2]$$ | $$[2,3,4]$$ | 3 |
| Medusa 1, shift 3 | $$[0,1]$$ | $$[3,4]$$ | 2 |
| Medusa 2, shift 4 | $$[0]$$ | $$[4]$$ | 1 |

Farther heads lose more training positions at the right sequence boundary. With packed long sequences this is a small fraction; with toy sequences it is impossible to ignore.

### Full calculation: batch 2, sequence length 3, vocabulary 5

Use this vocabulary:

`0=<pad>, 1=I, 2=like, 3=cats, 4=dogs`

Our batch is

$$
\begin{bmatrix}
1&2&3\\
1&2&4
\end{bmatrix}
$$

with shape

[batch, sequence] = [2, 3].

Every Medusa head emits logits of shape

[batch, sequence, vocabulary] = [2, 3, 5].

For Medusa head 0, shift=2:

```python
shifted_logits = logits[:, :-2, :]  # [2, 1, 5]
shifted_labels = labels[:, 2:]      # [2, 1]
```

Only position $$t=0$$ is valid in each example:

Example 0: logits[0, 0] predicts labels[0, 2] = cats
Example 1: logits[1, 0] predicts labels[1, 2] = dogs

Assume the two relevant logit vectors are

$$
\ell_A=[0,0,0,2,0],
\qquad
\ell_B=[0,0,0,0,1].
$$

For example A, the probability of the target cats is

$$
\frac{e^2}{e^2+4e^0}
=
\frac{7.389}{11.389}
\approx0.6488.
$$

Its negative log-likelihood is

$$
L_A=-\log(0.6488)\approx0.4327.
$$

For example B:

$$
\frac{e^1}{e^1+4e^0}
=
\frac{2.718}{6.718}
\approx0.4046,
$$

and

$$
L_B=-\log(0.4046)\approx0.9048.
$$

PyTorch's default cross-entropy reduction averages the two valid batch-position pairs:

$$
\frac{0.4327+0.9048}{2}
\approx0.6687.
$$

Nothing is calculated for logits[:, 1] or logits[:, 2] for this head because the corresponding labels would lie beyond the sequence.

Medusa head 1 would require shift=3. With $$T=3$$,

```python
logits[:, :-3, :]  # empty
labels[:, 3:]      # empty
```

so that head cannot be trained from this toy block. A robust implementation must skip it or reject the batch rather than pass empty tensors into cross-entropy.

### Combining multiple head losses

Medusa-1 downweights more distant heads:

$$
\sum_{k=1}^{K}\lambda_k\mathcal{L}_k,
\qquad
\lambda_k\approx0.8^k.
$$

```python
def medusa1_loss(medusa_logits, labels, decay=0.8):
    per_head = []

    for head_index, logits in enumerate(medusa_logits):
        loss = shifted_cross_entropy(
            logits,
            labels,
            shift=head_index + 2,
        )
        per_head.append(loss)

    total = sum(
        decay ** (index + 1) * loss
        for index, loss in enumerate(per_head)
    )
    return total, per_head
```

Medusa-2 also protects next-token prediction with the ordinary LM loss:

$$
\mathcal{L}_{\text{LM}}
+
\lambda_0\mathcal{L}_{\text{Medusa-1}}.
$$

The original next-token loss is unnecessary for updating a frozen Medusa-1 backbone, although it can still be logged as a baseline metric.

### Why the later-head loss is usually higher

The later head does not have more context. All independent Medusa heads see $$h_t$$, which represents only $$x_{\le t}$$.

Head 0 models $$x_{t+2}$$ without seeing $$x_{t+1}$$. Head 2 models $$x_{t+4}$$ without seeing $$x_{t+1},x_{t+2},x_{t+3}$$. Conditional uncertainty generally grows with distance, so the aggregate tendency is

$$
\mathcal{L}_{\text{head 0}}
<
\mathcal{L}_{\text{head 1}}
<
\mathcal{L}_{\text{head 2}}.
$$

That is a dataset-level tendency, not a guarantee for every batch. Boundary effects also give farther heads fewer valid examples.

At this point training is complete. There has been no top-K selection, no candidate tree, and no rejection. Those are inference operations.

## Part 6: One Medusa Inference Iteration, Without Skipping Steps

Consider a prompt ending with:

The cat

Assume the current final hidden state is $$h_t$$. One forward pass produces:

```text
Original LM head, top-1:  sat

Medusa head 0, top-2:    on, slept
Medusa head 1, top-2:    the, .
```

The original head predicts the root token sat. The two Medusa heads independently predict the next two positions after that root.

The complete candidates are:

```text
sat on    the
sat on    .
sat slept the
sat slept .
```

![Two-level Medusa candidate tree with two retained ranks at each level]({{ '/assets/img/medusa-candidate-tree.svg' | relative_url }})

The implementation does not duplicate the prompt four times and run four ordinary target batches. It flattens the shared nodes into one tree and applies a special attention mask so each node sees only its own ancestors.

Suppose target verification says:

```text
After "The cat"         target wants "sat"
After "The cat sat"     target wants "on"
After "The cat sat on"  target wants "the"
```

Then the path

sat on the

matches for its entire represented depth. One verification iteration commits three tokens:

```text
root target token:       sat
accepted Medusa token 1: on
accepted Medusa token 2: the
```

The next iteration begins from the new prefix:

The cat sat on the

The selected verification node supplies the new final state and head logits. Those outputs produce new top-K lists, the same fixed tree topology is filled with new token IDs, and verification repeats.

If instead the target wanted slept after sat, the sat on ... paths would fail immediately. If one of the retained head-0 ranks contains slept, a path beginning with that rank can still win. If no retained rank matches, Medusa commits only sat and starts the next iteration.

This is the entire runtime loop:

```text
prefill prompt
-> produce root and head distributions
-> fill candidate tree
-> verify tree with target backbone
-> choose longest accepted prefix
-> copy only selected KV-cache states
-> repeat from committed prefix
```

The draft heads do not recursively rerun while creating this tree. They generate their distributions from one backbone state. The target backbone is then run for tree verification. After the initial prompt prefill, that verification pass also supplies the selected base and Medusa logits needed to construct the next tree; there is no extra standalone prompt pass between decode iterations.

## Part 7: Regular Tree Mathematics and the "250 Heads" Misreading

Let

$$
s_i=\text{number of top-ranked tokens retained from Medusa level }i.
$$

The notation is one-based: $$s_1$$ refers to Python medusa_head[0].

For two levels with $$s_1=s_2=2$$:

Depth 1 contains $$s_1=2$$ nodes.

Depth 2 contains $$s_1s_2=4$$ nodes.

There are $$s_1s_2=4$$ complete leaf continuations.

The number of additional non-root candidate nodes is

$$
q=s_1+s_1s_2=2+4=6.
$$

Including the root token from the original LM head:

$$
N_{\text{total}}=1+q=7.
$$

For $$K$$ uniform Cartesian levels:

$$
q=s_1
+s_1s_2
+s_1s_2s_3
+\cdots
+\prod_{i=1}^{K}s_i,
$$

and

$$
N_{\text{total}}=1+q.
$$

Therefore the expression

$$
1+s_1+s_1s_2+s_1s_2s_3+s_1s_2s_3s_4+s_1s_2s_3s_4s_5
$$

is exactly correct for a full five-level tree in which every parent at depth $$i-1$$ receives the same $$s_i$$ children.

### Where an "Additional Length" near 250 comes from

The appendix of the Together AI launch post plots speed against Additional Length for different head configurations. Additional length counts flattened candidate-token positions, not learned heads.

The post does not provide the exact $$s_i$$ tuple for every plotted point, so consider this illustrative four-head configuration:

$$
(s_1,s_2,s_3,s_4)=(5,5,3,2).
$$

The depths contain

$$
5,\quad25,\quad75,\quad150
$$

nodes, giving

$$
q=5+25+75+150=255.
$$

This means:

- 4 Medusa heads participate.
- 150 complete leaf continuations exist.
- 255 non-root candidate-token nodes are verified.
- 256 total nodes exist if the LM-head root is included.

It does not mean that the model has approximately 250 prediction heads.

### Why a full top-10 tree explodes

If four Medusa levels each retained ten children under every parent, then

$$
1+10+10^2+10^3+10^4
=
11{,}111.
$$

With five levels it would become

$$
111{,}111.
$$

Verifying that many nodes would destroy the latency benefit. This motivates a sparse, nonuniform tree.

## Part 8: How Medusa Chooses an Optimized Sparse Tree

The optimized tree is chosen offline, not separately for every prompt.

### Step 1: measure exact-rank accuracy

Run the trained Medusa heads on a calibration dataset. For head $$k$$ and rank $$i$$, measure

$$
P(\text{the correct future token occurs at exact rank }i).
$$

This is an exact-rank statistic. If rank numbering begins at one,

$$
P(\text{exact rank }i)
=
\text{top-}i\text{ accuracy}
-
\text{top-}(i-1)\text{ accuracy}.
$$

The top-1 candidate may be correct frequently, rank 1 less frequently, and rank 9 only rarely. Farther heads also tend to be less accurate.

### Step 2: score a rank path

A rank tuple such as

(0, 2, 1)

means:

- head 0 -> take its rank-0 token
- head 1 -> take its rank-2 token
- head 2 -> take its rank-1 token

Under the paper's independence approximation, the node score is

$$
a_1^{(0)}a_2^{(2)}a_3^{(1)}.
$$

It estimates how often that rank combination will correctly extend the accepted prefix to that depth.

### Step 3: grow a prefix-closed tree under a budget

Begin with the root. Repeatedly add the highest-scoring node whose parent is already present. Stop when the node budget is exhausted.

A toy calibration makes the behavior clear:

head 0: rank 0 = 0.60, rank 1 = 0.20
head 1: rank 0 = 0.50, rank 1 = 0.10

The candidate node scores include:

(0,)   = 0.60
(1,)   = 0.20
(0,0)  = 0.60 x 0.50 = 0.30
(0,1)  = 0.60 x 0.10 = 0.06
(1,0)  = 0.20 x 0.50 = 0.10

The greedy order can begin:

(0,) -> (0,0) -> (1,) -> (1,0) -> (0,1)

The search goes deeper under a reliable top-ranked prefix before spending many nodes on unlikely branches. This produces the characteristic left-heavy tree.

For a prefix-closed tree $$T$$, the paper approximates expected accepted length as

$$
\mathbb{E}[L]
\approx
\sum_{u\in T}\operatorname{score}(u).
$$

The optimization is therefore:

- maximize estimated accepted length,
- subject to a fixed node budget,
- and the rule that a node's parent must already exist.

The Medusa paper reports using AlpacaEval statistics to shape the sparse Vicuna-7B tree. It separately benchmarks tree sizes because a larger tree raises acceptance but also increases linear-layer and attention work. In its ablation, a sparse 64-node tree outperformed dense settings with 256 nodes. The optimum is a model-dataset-hardware property, not a universal constant.

### A sparse tree needs depth counts, not one global $$s_i$$

For a nonuniform tree, different parents retain different numbers of children. The number of nodes at depth $$d$$ is

$$
n_d=\sum_{u\in P_{d-1}}b(u),
$$

where $$P_{d-1}$$ is the retained parent set and $$b(u)$$ is the number of children assigned to parent $$u$$.

Then

$$
N_{\text{total}}=1+\sum_{d=1}^{D}n_d.
$$

The Cartesian product formula is the special case in which every parent at the same depth has the same branching factor.

## Part 9: The Real Vicuna-7B Reference Tree

The official loader selects the hard-coded vicuna_7b_stage2 topology for a Vicuna-7B model.

The checkpoint loader creates five trained Medusa heads. The candidate utility then computes

```python
TOPK = 10
top_tokens = torch.topk(medusa_logits, TOPK, dim=-1).indices
```

so each head initially contributes a pool of ten ranked token IDs. Five heads produce $$5\times10=50$$ pool entries.

`TOPK=10` is only the candidate-pool width. It is not the number of children attached below every tree node.

The static tree contains rank tuples of maximum length four. Its depth counts are:

| Tree depth | Prediction source | Retained nodes | Distinct ranks used at that depth |
| ---: | --- | ---: | --- |
| 0 | Original LM-head root | 1 | top-1 root |
| 1 | Medusa head 0 | 10 | 0-9 |
| 2 | Medusa head 1 | 23 | 0-9, repeated below different parents |
| 3 | Medusa head 2 | 23 | 0-8, repeated below different parents |
| 4 | Medusa head 3 | 7 | 0-3, repeated below different parents |
| 5 | Medusa head 4 | 0 | unused by this topology |

Therefore

$$
1+10+23+23+7=64
$$

total nodes are evaluated, of which 63 are non-root choices.

At depth 2, the ten depth-1 parents receive different child counts:

```text
depth-1 rank:       0  1  2  3  4  5  6  7  8  9
children retained: 10  5  2  2  1  1  1  1  0  0
```

Their sum is 23, not $$10\times10=100$$.

The sparse tree has 42 leaves:

```text
 2 paths end at depth 1
14 paths end at depth 2
19 paths end at depth 3
 7 paths end at depth 4
------------------------
42 root-to-leaf paths
```

Those paths are padded to a common width of five positions:

1 root + at most 4 Medusa tokens = width 5

This explains the reference trace:

```text
63 non-root choices
64 verified tree nodes
42 padded candidate paths
path width 5
```

Although the checkpoint carries five Medusa heads, this tree uses four future depths. The reference implementation still computes top-K values for all loaded heads before indexing the chosen tree; a tighter production kernel could skip a redundant head.

## Part 10: Static Rank Topology, Dynamic Token Values

The tree file contains ranks, not vocabulary IDs and not words.

Suppose one inference iteration produces:

```text
base LM root: "The"

head 0 ranks: 0=" cat", 1=" dog", 2=" model", ...
head 1 ranks: 0=" is",  1=" was", 2=" sat",   ...
head 2 ranks: 0=" a",   1=" on",  2=" very",  ...
```

The static tuple

(0, 2, 1)

is instantiated as

"The" + " cat" + " sat" + " on"

On the next iteration, (0, 2, 1) remains in the topology but maps to completely different token IDs because the head distributions changed.

This separates the two selection problems:

| Decision | When? | What decides it? |
| --- | --- | --- |
| Which rank tuples deserve tree nodes? | Offline | Calibration accuracies plus a node budget |
| Which actual token occupies each tuple position? | Every inference iteration | Current head top-K logits |
| Which candidate prefix is committed? | Every verification iteration | Current target-model logits |

The offline score never overrides the target model at runtime. It only decides which proposals are worth presenting to the target.

## Part 11: Tree Attention Verifies Shared Prefixes Once

If we expanded every leaf into a separate batch row, shared prefixes would be duplicated. Tree attention represents each distinct node once.

For the official Vicuna tree, the principal tensors are conceptually:

```text
tree_candidates:      [1, 64]
tree attention mask:  [1, 1, 64, 64]
tree position IDs:    [64]
retrieved paths:      [42, 5]
verifier logits:      [42, 5, vocabulary]
```

The exact dimensions can acquire singleton axes inside the implementation, but the semantics are stable.

The attention mask grants a node access to:

- every original prompt token,
- the root candidate,
- only the candidate nodes on its own ancestor path.

It cannot attend to siblings or tokens from another continuation. Position IDs are assigned by tree depth, so two siblings occupy the same logical sequence position even though they have different flattened indices.

After the target forward pass, retrieval indices reconstruct 42 padded root-to-leaf paths from the 64 node outputs. Only KV-cache entries belonging to the selected accepted path are copied into the contiguous decode cache. The other verified branches are discarded.

This is why production Medusa is neither:

four prompts copied into a normal batch

nor:

four separate target-model forward passes.

It is one tree-shaped target pass with shared prefix nodes and a custom causal mask.

## Part 12: Exact Greedy Verification and Rejection

Let

```python
candidates.shape       = [number_of_paths, path_width]
verifier_logits.shape  = [number_of_paths, path_width, vocabulary]
```

candidates[:, 0] is the root token already selected by the ordinary LM head. For every later position, the target logit at the previous position predicts the candidate token:

```python
target_tokens = verifier_logits[:, :-1].argmax(dim=-1)
matches = candidates[:, 1:] == target_tokens
prefix_matches = matches.int().cumprod(dim=1)
accepted_per_path = prefix_matches.sum(dim=1)
```

cumprod enforces the left-to-right rule:

```text
raw matches:     [True, False, True, True]
valid prefix:    [1,    0,     0,    0]
accepted length: 1
```

The later True values do not count after an earlier rejection.

The reference implementation chooses

```python
accept_length = accepted_per_path.max()
best_candidate = accepted_per_path.argmax()
```

and appends

`candidates[best_candidate, :accept_length + 1]`

The +1 includes the root token.

Define the metrics carefully:

$$
\text{accepted Medusa tokens}=\text{accept_length},
$$

$$
\text{committed tokens}=1+\text{accept_length}.
$$

Therefore:

| Verification outcome | Accepted Medusa tokens | Committed tokens |
| --- | ---: | ---: |
| Every proposal fails immediately | 0 | 1 root |
| One future token matches | 1 | 2 |
| Four future tokens match | 4 | 5 |

If all paths accept zero Medusa tokens, the implementation selects path 0 by convention. The choice is harmless because every path shares the same root and only that root is committed.

If several paths tie at a positive greedy prefix length, their accepted token prefix must agree with the same target argmax sequence. Their unaccepted suffixes may differ; choosing the first maximum does not change the committed text.

### Padded paths are not real completions

The 42 leaves have different depths but are stored at width five. Short paths use a sentinel retrieval index that maps to token ID 0 in the reference code. With the Vicuna tokenizer, ID 0 decodes as `<unk>`.

Consequently, a diagnostic trace may say that Medusa proposed `<unk>` immediately after an accepted prefix. Sometimes that is not a genuine head proposal--it means the selected leaf ended and the next tensor slot was padding. A correct trace should distinguish

first real candidate mismatch

from

selected path exhausted

rather than interpreting every padded ID as a proposal.

## Part 13: What Is Actually Guaranteed?

The phrase "Medusa is lossless" needs an acceptance rule attached to it.

### Greedy exact verification

When every accepted Medusa token must equal the target model's greedy argmax, the committed sequence matches ordinary greedy decoding from that same target. Medusa can change the number of tokens committed per step, not their values.

There is still no guarantee that a speculative token is accepted. The guarantee is:

at least one target root token is committed per iteration

not:

every Medusa head contributes one accepted token

### Distribution-preserving rejection sampling

For stochastic decoding, a proper speculative rejection-sampling correction can preserve the target distribution. This uses the draft and target probabilities, not a simple equality test.

### Typical acceptance

Medusa also proposes typical acceptance. A candidate token is considered plausible when its target probability exceeds an entropy-adaptive threshold of the form

$$
p_{\text{target}}(x)
>
\min\left(
\epsilon,
\delta e^{-H(p_{\text{target}})}
\right).
$$

The first root token is still selected greedily, and the longest typical prefix is committed. This can accept more tokens at nonzero temperature, but it deliberately relaxes exact distribution matching. "Similar generation quality" is not the same claim as "identical target distribution."

In every mode, the idea that Medusa has "no target" is incorrect. The same LLM that supplied the hidden state also acts as the verifier during the tree pass.

## Part 14: Running a Real Pretrained Medusa Checkpoint

The repository includes an instrumented inference runner for the official FasterDecoding/medusa-vicuna-7b-v1.3 checkpoint:

[Pretrained Medusa tree trace](https://github.com/sudhirpol522/sudhirpol522.github.io/blob/main/examples/medusa_pretrained_trace.py)

### Colab/A100 requirements

In a fresh Colab A100 runtime, clone this repository and install the pinned compatibility set:

```bash
python -m pip install -r examples/requirements-medusa-pretrained.txt
```

The [requirements file](https://github.com/sudhirpol522/sudhirpol522.github.io/blob/main/examples/requirements-medusa-pretrained.txt) installs the official Medusa repository together with its intentionally conservative Transformers stack. If your runtime already includes a CUDA-enabled PyTorch build, keep that GPU build rather than replacing it with a CPU wheel.

Keep Colab's CUDA-enabled PyTorch rather than pinning a second Torch wheel. Then run:

```bash
python examples/medusa_pretrained_trace.py \
    --prompt "Explain why speculative decoding is lossless." \
    --max-new-tokens 96 \
    --dtype float16 \
    --output-json medusa_trace.json
```

The script directly calls the official primitives:

- `initialize_medusa`
- `generate_candidates`
- `tree_decoding`
- `evaluate_posterior`
- `update_inference_inputs`

It independently reconstructs the greedy prefix mask and asserts that its accepted length agrees with the official utility.

### Why the loader says the heads were newly initialized

The checkpoint is loaded in three steps:

1. Load the Vicuna base model.
2. Construct Medusa-head modules.
3. Load `medusa_lm_head.pt` into those modules.

Transformers prints its "newly initialized" warning during step 1, before the separate head file is applied. In the official loading path, that warning is expected and does not by itself mean the final heads remain random.

The old uploaded configuration also reports too few heads. The official loader contains a compatibility workaround that constructs five heads before loading the separate state dictionary. This is why using `AutoModel` directly is not equivalent to using `MedusaModel.from_pretrained(...)`.

These other startup messages are also informational:

- `_register_pytree_node` is a compatibility deprecation warning from the older Transformers version.
- "legacy" Llama tokenization preserves the tokenizer behavior expected by Vicuna.
- The first run downloads the Vicuna base shards and the separate Medusa weights; later runs use the Hugging Face cache.

### What one A100 trace showed

One run used a 50-token prompt and generated at least 96 new tokens. Representative iterations were:

| Iteration | Match mask after root | Accepted Medusa tokens | Total committed | Committed text fragment |
| ---: | --- | ---: | ---: | --- |
| 1 | `[False]` | 0 | 1 | `Spe` |
| 2 | `[True, False]` | 1 | 2 | `culative` |
| 3 | `[True, True, True, True]` | 4 | 5 | `decoding is a technique` |
| 18 | `[True, True, True, False]` | 3 | 4 | `current frame. This` |

Iteration 1 illustrates the zero-acceptance case:

```text
root token from base LM:  "Spe" -> committed
head-0 candidate:         "spe"
target continuation:      "cul" -> mismatch
```

Iteration 3 illustrates a full-depth match. The target accepted four Medusa tokens after the root, so one verification committed five tokens.

Across all 45 iterations:

```text
accepted Medusa tokens:              52
total committed tokens:              97
mean accepted Medusa / iteration:   1.156
mean committed / iteration:         2.156
```

The acceptance-length distribution was:

| Accepted Medusa tokens | Number of iterations |
| ---: | ---: |
| 0 | 9 |
| 1 | 23 |
| 2 | 11 |
| 3 | 1 |
| 4 | 1 |

The first tree iteration took about 148 ms because it included warm-up effects. The remaining 44 iterations averaged approximately 40.47 ms with a median near 40.12 ms on that runtime.

### Tokens per verification step are not wall-time speedup

The ratio

$$
2.156
$$

is a decode-step compression ratio. It says how much sequence progress the average tree pass made.

It is not automatically a $$2.156\times$$ latency speedup. A 64-node tree pass is more expensive than a one-token decode pass. Wall-time speedup requires an ordinary greedy baseline with the same:

- target model,
- prompt set and output lengths,
- dtype and quantization,
- GPU,
- sampling configuration,
- warm-up procedure,
- cache implementation.

A useful approximation is

$$
\text{speedup}
\approx
\frac{\text{committed tokens per tree step}}
{\text{tree-step latency}/\text{ordinary-step latency}},
$$

but the final number should be measured rather than inferred.

### What should be verified against a baseline?

For greedy exact mode, compare the complete generated token-ID sequence against ordinary greedy generation from the base target. Text comparison alone can hide tokenizer cleanup differences.

The repository also contains a small educational implementation that trains Medusa-1 heads and performs this exact baseline-ID comparison:

[Small Medusa-1 training and verification demo](https://github.com/sudhirpol522/sudhirpol522.github.io/blob/main/examples/medusa_smol_lm_demo.py)

Install its small dependency set from [the demo requirements file](https://github.com/sudhirpol522/sudhirpol522.github.io/blob/main/examples/requirements-medusa-demo.txt). That teaching script expands candidate leaves into a normal batch so every probability and rejection is easy to inspect. It is not a production speed benchmark. The pretrained runner uses the official flattened tree, custom position IDs, tree mask, and KV-cache update.

## Part 15: Medusa and Multi-Token Prediction Share an Objective, Not Always an Architecture

The broad MTP objective is:

$$
\text{Given }x_{\le t},
\text{ predict several tokens }x_{t+1},x_{t+2},\ldots.
$$

Medusa implements that objective with post-hoc residual heads. Later work uses different modules and training lifecycles.

### Parallel MTP during pretraining

The 2024 Better & Faster LLMs via Multi-token Prediction work trains a shared Transformer trunk and $$n$$ independent output heads from the beginning:

$$
\operatorname{softmax}\left(
f_u(f_{h_i}(f_s(x_{\le t})))
\right).
$$

Here:

- $$f_s$$ is the shared trunk.
- $$f_{h_i}$$ is a head-specific Transformer layer.
- $$f_u$$ is a shared unembedding matrix.

The future heads are still parallel: they do not consume one another's predicted tokens. The main differences from Medusa-1 are:

- MTP is a pretraining objective, so the trunk learns from all future losses.
- Its head-specific modules share the large vocabulary unembedding.
- The paper's count $$n$$ includes ordinary next-token prediction.

Thus an $$n=4$$ MTP model and a Medusa model with three auxiliary heads both emit four position distributions, but their configuration counters differ.

At inference, the MTP model can discard auxiliary heads and decode normally, or use them for self-speculative decoding and Medusa-like tree attention.

### DeepSeek-style sequential MTP

DeepSeek-V3 explicitly contrasts its design with independent parallel heads. It uses $$D$$ sequential MTP modules, maintaining a causal chain across prediction depths.

At depth $$k$$, it combines:

- the previous-depth hidden representation $$h_i^{k-1}$$,
- the embedding of the relevant future token $$\operatorname{Emb}(t_{i+k})$$.

It projects their normalized concatenation and applies a depth-specific Transformer block:

$$
h_i^{\prime k}
=
M_k
\left[
\operatorname{RMSNorm}(h_i^{k-1});
\operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))
\right],
$$

$$
h_i^k=\operatorname{TRM}_k(h_i^{\prime k}).
$$

A shared output head predicts $$t_{i+k+1}$$. During pretraining, the future embedding comes from the ground-truth sequence, and the MTP losses are auxiliary to the main model loss.

This is more context than a Medusa head receives--but it is context in the MTP causal chain, not additional observed prompt context magically available to a parallel head.

### Recursive shared-weight MTP and vLLM Speculators

The current vLLM Speculators MTP trainer fine-tunes native MTP weights in a FastMTP-style loop. At training step $$k$$:

1. Take the current hidden representation.
2. Look up the ground-truth token embedding at the required shifted position.
3. Normalize, concatenate, and project them.
4. Apply the same MTP prediction layer.
5. Use the shared frozen LM head to predict the next shifted target.
6. Feed the MTP output hidden state into the next recursive step.

The essential implementation pattern is:

```python
current_hidden = verifier_hidden_states

for step in range(number_of_speculative_steps):
    token_embeddings = embed_tokens(shifted_ground_truth_ids)
    mtp_output = shared_mtp_layer(
        hidden_states=current_hidden,
        token_embeddings=token_embeddings,
    )
    logits = frozen_lm_head(mtp_output)
    loss = cross_entropy(logits, shifted_targets)
    current_hidden = mtp_output
```

During inference, ground truth does not exist. The recursive draft loop uses the token selected at the previous speculative step, then the target model still verifies the resulting draft. Recursive feedback improves draft conditioning; it does not turn draft tokens into automatically accepted target tokens.

Only the MTP layers are trainable in the current fine-tuning path. The verifier embedding table and LM head are frozen and shared. Per-step losses use normalized exponential decay; with default $$\beta=0.6$$ and three steps:

$$
[\alpha_1,\alpha_2,\alpha_3]
\approx
[0.51,0.31,0.18].
$$

This gives the earlier speculative positions more weight because an early rejection invalidates every later draft position.

Selecting `speculator_type=mtp` in that trainer does not train vanilla Medusa-1 heads. The Speculators project tracks Medusa training as a separate RFC.

![Independent Medusa heads compared with one recursively reused MTP layer]({{ '/assets/img/medusa-vs-recursive-mtp.svg' | relative_url }})

### The final taxonomy

| Property | Medusa-1 | Parallel MTP pretraining | DeepSeek sequential MTP | FastMTP-style/current vLLM trainer |
| --- | --- | --- | --- | --- |
| Training stage | Post-training | Pretraining | Pretraining | Fine-tuning native MTP weights |
| Future predictor | Independent residual heads | Independent Transformer heads | Several sequential modules | One prediction layer reused recursively |
| Every depth sees the same $$h_t$$ independently | Yes | Yes | No | No |
| Previous future information enters later depth | No | No | Yes | Yes |
| Weights shared across speculative depths | No | No | Generally no | Yes |
| Vocabulary projection | Separate per Medusa head | Shared unembedding | Shared main output head | Shared frozen verifier LM head |
| Backbone updated | No | Yes | Yes | No during speculator fine-tuning |
| Candidate tree used during training | No | No | No | No |
| Target verification still required for speculation | Yes | Yes | Yes | Yes |

The safest vocabulary is:

- **Medusa-1:** post-hoc independent residual heads over a frozen target.
- **Parallel MTP:** independent future heads jointly pretrained with the trunk.
- **Sequential MTP:** depth-specific modules preserve a future-token causal chain.
- **Recursive shared-weight MTP:** one native MTP layer is reused across draft steps.

"MTP" names the broad objective. It does not uniquely determine the architecture.

## Part 16: Production Measurements That Matter

Head loss is useful, but it does not establish inference speed. A complete evaluation should include:

| Metric | Why it matters |
| --- | --- |
| Per-head cross-entropy | Shows how uncertainty grows with prediction distance |
| Exact-rank and top-K accuracy | Supplies statistics for sparse-tree design |
| Accepted Medusa tokens per iteration | Measures useful speculative depth |
| Committed tokens per iteration | Includes guaranteed root progress |
| Acceptance-length histogram | Reveals whether averages hide frequent zero-accept steps |
| Tree node count and leaf count | Quantifies verification breadth and path coverage |
| Ordinary decode latency | Establishes the real denominator for speedup |
| Tree verification latency | Measures added compute from speculative nodes |
| Target calls per generated token | Measures reduction in sequential target work |
| Time to first token | Usually dominated by prefill rather than Medusa drafting |
| Inter-token latency | Captures the main low-batch user-visible benefit |
| Throughput under concurrency | Shows whether normal batching already saturates compute |
| Peak memory and KV-cache memory | Captures head, buffer, and serving overhead |
| Exact baseline token-ID match | Validates greedy losslessness |

Tree selection should be calibrated on output resembling production traffic. A code-tuned tree may waste nodes on chat, and a chat-tuned tree may underperform on structured generation. The optimum can also move between an A100, H100, and a highly concurrent server because the cost of verifying additional nodes changes.

The right optimization target is not maximum acceptance in isolation. It is approximately

$$
\frac{\text{expected committed tokens}}
{\text{measured verification latency}}.
$$

Adding a low-probability node can raise expected acceptance slightly while making every iteration slower. That is why the Medusa paper observed diminishing and eventually negative returns as trees grew.

## Common Misreadings, Corrected

| Misreading | Correct interpretation |
| --- | --- |
| "Additional length 250 means 250 heads." | It counts flattened candidate-token nodes; a few heads can create hundreds of nodes. |
| "Top-K is 10, so every node has ten children." | Ten is the candidate pool per head; the sparse topology selects only specific rank nodes under each parent. |
| "The 63 choices are selected from the current prompt." | The rank topology is selected offline from calibration statistics and reused. |
| "The tree contains 63 total tokens." | It contains 63 non-root choices plus one LM-head root, so 64 verification nodes. |
| "There are 64 candidate completions." | There are 64 nodes but 42 root-to-leaf candidate paths in this configuration. |
| "Five trained heads means five heads must be used." | The Vicuna tree reaches four Medusa depths; the fifth head is not indexed by this topology. |
| "Later Medusa heads have more context." | Independent heads see the same backbone state; later targets are farther away and generally harder. |
| "A frozen backbone is not run during training." | It still produces hidden states; its parameters are simply not updated. |
| "Because the heads belong to the LLM, their output is accepted automatically." | The original target backbone verifies the tree and can reject every speculative token. |
| "Accepted tokens and committed tokens are identical." | Committed tokens equal one root plus accepted Medusa tokens. |
| "At least one Medusa token is guaranteed." | Only one target root token is guaranteed; speculative acceptance may be zero. |
| "Typical acceptance is exactly lossless." | Greedy exact matching or proper rejection sampling can preserve target behavior; typical acceptance is a quality-preserving relaxation, not exact distribution equality. |
| "2.156 committed tokens per iteration means 2.156x faster." | It is sequence progress per verification call; wall speed also depends on tree-pass cost. |
| "vLLM Speculators MTP is Medusa-1." | Its current MTP path recursively fine-tunes native MTP weights; Medusa training is a distinct design. |

## The Entire Story in One Pass

1. An ordinary LLM uses one expensive decoding step to choose one next token.
2. Vanilla speculative decoding adds a cheap model that drafts several tokens for target verification.
3. Medusa removes the separate draft model by attaching lightweight future-token heads to the target LLM.
4. In Medusa-1, embeddings, backbone, and original LM head stay frozen; only auxiliary heads train.
5. Python Medusa head 0 predicts two positions ahead, so its labels use shift 2; head $$j$$ uses shift $$j+2$$.
6. Later independent heads do not receive earlier draft tokens, so uncertainty usually grows with depth.
7. At inference, every head produces a vocabulary distribution and the implementation extracts a top-K candidate pool.
8. A fixed rank topology maps selected rank combinations into actual token candidates for the current prompt.
9. A regular tree follows the Cartesian-product node formula; an optimized sparse tree retains only high-value prefix nodes.
10. The official Vicuna topology contains 63 non-root choices, one root, 64 verification nodes, 42 leaves, and four Medusa depths.
11. Tree attention verifies all distinct nodes in one target pass while preserving each candidate's causal ancestry.
12. Greedy verification counts consecutive equality matches and commits one root plus the longest accepted Medusa prefix.
13. Zero accepted Medusa tokens is valid: the iteration still commits its target root.
14. The observed A100 run averaged 1.156 accepted and 2.156 committed tokens per verification iteration; this is not itself a wall-time speedup measurement.
15. Parallel MTP, sequential MTP, and recursive FastMTP share the future-token objective but use different architectures and training lifecycles.

The cleanest summary remains:

> The heads create possibilities. The sparse tree decides which possibilities are worth checking. Tree attention checks them efficiently. The target model decides which prefix becomes real output.

## References

- [Stern et al., *Blockwise Parallel Decoding for Deep Autoregressive Models*](https://arxiv.org/abs/1811.03115)
- [Together AI, *Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads*](https://www.together.ai/blog/medusa)
- [Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*](https://arxiv.org/abs/2401.10774)
- [FasterDecoding, official Medusa implementation](https://github.com/FasterDecoding/Medusa)
- [FasterDecoding, Vicuna-7B Medusa checkpoint](https://huggingface.co/FasterDecoding/medusa-vicuna-7b-v1.3)
- [Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction*](https://arxiv.org/abs/2404.19737)
- [DeepSeek-AI, *DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437)
- [Cui et al., *FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction*](https://arxiv.org/abs/2509.18362)
- [vLLM Project, Speculators MTP documentation](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/mtp/)
- [vLLM Project, MTP training tutorial](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_mtp_online/)
- [vLLM Project, RFC: Add Medusa speculator training](https://github.com/vllm-project/speculators/issues/622)
