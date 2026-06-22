---
layout: page
title: "Unpacking Speculative Decoding: The Math Behind the Speedup"
permalink: /blog/speculative-decoding-math/
math: true
---

If you've been following my recent posts on optimizing LLM inference, you know that making models faster is an ongoing battle. We've discussed shrinking models using quantization techniques like AWQ and GPTQ to reduce memory bandwidth. But what if you want to speed up inference *without* changing the target model's weights at all?

In my previous post, [Speculative Decoding: The Clever Trick Making LLMs 2x Faster](https://medium.com/@sudhirpol522/speculative-decoding-the-clever-trick-making-llms-2-faster-69a2adee98a7), we looked at the high-level intuition. Today, we are opening the hood. We are going to derive the exact mathematical objective function that governs speculative decoding, prove why it guarantees zero degradation in quality, and reveal a brutal reality about token probabilities: **why failing early is devastating.**

## The Setup

Let's quickly define our variables. We have two autoregressive models:

* **Target Model $$p(x)$$**: Our large, expensive model. We must match its output distribution exactly.
* **Draft Model $$q(x)$$**: A small, cheap model that proposes tokens quickly.

In every iteration, the Draft model autoregressively proposes $$\gamma$$ tokens (the lookahead). The Target model then verifies all of them in one parallel forward pass.

---

## Part 1: The Single-Token Acceptance Probability ($$\beta$$)

This is the heart of speculative decoding. How do we decide if we keep a drafted token? The rule is:

1. Draft samples a token $$x \sim q(x)$$.
2. Accept $$x$$ with probability $$\min\left(1, \frac{p(x)}{q(x)}\right)$$.

To find the expected acceptance rate ($$\beta$$) across all possible tokens, we multiply the probability that the draft picks a token by the probability that we accept it, and sum this over the entire vocabulary:

$$\beta = \sum_x q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right)$$

Push $$q(x)$$ inside the $$\min$$ function, and it simplifies beautifully:

$$\beta = \sum_x \min(q(x), p(x))$$

**In simple English:** The acceptance probability is exactly equal to the "overlap mass" between the Target and Draft distributions. If they agree perfectly everywhere, $$\beta = 1$$.

### The Total Variation Distance Identity

We can rewrite this using a standard mathematical identity: $$\min(a,b) = \frac{a+b-|a-b|}{2}$$.

$$\sum_x \min(p,q) = \frac{1}{2}\sum_x \big(p(x) + q(x) - |p(x) - q(x)|\big)$$

Because $$p$$ and $$q$$ are valid probability distributions, they both sum to $$1$$.

$$= \frac{1}{2}\big(1 + 1 - \sum_x|p(x) - q(x)|\big)$$

By definition, Total Variation (TV) distance is $$D_{TV}(p,q) = \frac{1}{2}\sum_x|p(x) - q(x)|$$. Substituting this in gives us our final, elegant identity:

$$\beta = 1 - D_{TV}(p,q)$$

A better-aligned draft model has a smaller TV distance to the target, which directly raises the acceptance rate.

---

## Part 2: A Concrete Example (Vocabulary of 5)

Let's look at real numbers to see this identity hold true. Imagine a vocabulary of just 5 words. The Draft model proposes a sequence of 3 tokens. Let's calculate the acceptance probability ($$\beta_1$$) for the very first drafted token.

| Token 1 | Word A | Word B | Word C | Word D | Word E |
| --- | --- | --- | --- | --- | --- |
| **Target $$p$$** | 0.50 | 0.20 | 0.15 | 0.10 | 0.05 |
| **Draft $$q$$** | 0.38 | 0.25 | 0.20 | 0.10 | 0.07 |
| **$$\min(p,q)$$** | 0.38 | 0.20 | 0.15 | 0.10 | 0.05 |
| **$$\lvert p-q \rvert$$** | 0.12 | 0.05 | 0.05 | 0.00 | 0.02 |

Let's compute $$\beta_1$$ using both methods we just derived:

**Method 1: Overlap Mass**

$$\beta_1 = \sum \min(p,q) = 0.38 + 0.20 + 0.15 + 0.10 + 0.05 = \mathbf{0.88}$$

**Method 2: Total Variation Distance**
The sum of the absolute differences $$\sum \lvert p-q \rvert = 0.24$$.

$$D_{TV} = \frac{0.24}{2} = 0.12$$

$$\beta_1 = 1 - 0.12 = \mathbf{0.88}$$

Both methods perfectly yield **0.88**.

Now, assume we calculate the next two tokens in the drafted sequence. Token 2 happens to be very well-aligned, but Token 3 is poorly aligned:

* **Token 1:** $$\beta_1 = 0.88$$
* **Token 2:** $$\beta_2 = 0.96$$
* **Token 3:** $$\beta_3 = 0.65$$

---

## Part 3: Expected Tokens and The Domino Effect

Because speculative decoding strictly evaluates left-to-right, a token only counts if **every token before it was also accepted.** Using our actual per-position acceptance probabilities, we can compute exactly how many drafted tokens we expect to keep ($$\mathbb{E}[n]$$) from our 3-token sequence:

$$\mathbb{E}[n] = \beta_1 + (\beta_1 \cdot \beta_2) + (\beta_1 \cdot \beta_2 \cdot \beta_3)$$

$$\mathbb{E}[n] = 0.88 + (0.88 \times 0.96) + (0.88 \times 0.96 \times 0.65)$$

$$\mathbb{E}[n] = 0.88 + 0.8448 + 0.54912 = \mathbf{2.274 \text{ tokens}}$$

### The +1 Guaranteed Bonus Token

Here is a subtlety often missed: Every iteration yields one extra token beyond the accepted prefix.

* If a rejection happens, the Target model resamples a valid token from the residual distribution to replace it.
* If all tokens are accepted, the Target's parallel forward pass naturally yields the logit for the *next* unseen token.

Either way, you always get $$+1$$ token.

$$\mathbb{E}[X] = \mathbb{E}[n] + 1 \approx \mathbf{3.27 \text{ real tokens per iteration}}$$

### Why Failing Early is Devastating (Order Matters)

Notice that our weak draft ($$\beta_3 = 0.65$$) sat at the very end of the sequence. What if that weak token had been proposed first?

If $$\beta_1 = 0.65$$, $$\beta_2 = 0.96$$, and $$\beta_3 = 0.88$$, our expected draft tokens would drop dramatically:

$$\mathbb{E}[n] = 0.65 + (0.65 \times 0.96) + (0.65 \times 0.96 \times 0.88) = \mathbf{1.82 \text{ tokens}}$$

A wrong early token acts as a bottleneck, instantly rendering the rest of the drafted block as wasted compute. This is exactly why, when tuning draft models, accuracy on the *first* few positions matters significantly more than the end of the sequence.

*(Note: In literature, you will often see this mathematically approximated by averaging all $$\beta$$ values into a single $$\alpha$$, which allows you to use the geometric shortcut formula $$\mathbb{E}[X] = \frac{1-\alpha^{\gamma+1}}{1-\alpha}$$. While handy, it obscures the reality that token order matters!)*

---

## Part 4: The Objective Function (Calculating the Speedup)

We know how many tokens we get. Now, what is the wall-clock speedup compared to running the Target model alone?

Let $$c \in [0,1]$$ be the cost ratio: the time to draft one token divided by the time to verify one token.

**1. The Cost of Speculative Decoding**
In units of the Target model's time, one iteration requires:

* 1 parallel target verification pass: Cost = $$1$$
* $$\gamma$$ sequential draft passes: Cost = $$\gamma \cdot c$$
* Total Cost = $$\gamma c + 1$$

**2. The Speedup Ratio**
If the Target model worked alone, producing $$\mathbb{E}[X]$$ tokens would cost exactly $$\mathbb{E}[X]$$ time units. By dividing the baseline time by our speculative time, we get the objective function:

$$\text{Speedup}(\gamma) = \frac{\mathbb{E}[X]}{\gamma c + 1}$$

Substituting our geometric approximation for $$\mathbb{E}[X]$$ gives the standard theoretical formula:

$$\text{Speedup}(\gamma) = \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}$$

### Finding the Sweet Spot

This formula perfectly illustrates the trade-off you must manage as an ML Engineer:

* **The Numerator (Tokens gained):** Grows with $$\gamma$$, but saturates. Because of the domino effect of rejections, drafting deeper and deeper yields rapidly diminishing returns.
* **The Denominator (Compute cost):** Grows linearly. Every extra draft token costs real time, whether accepted or rejected.

Because saturating gains are fighting linear costs, there is always an interior optimum $$\gamma^\star$$. Drafting too far is actively wasteful. By understanding this math, you can dynamically tune your lookahead based on your Draft model's alignment and your hardware constraints, squeezing maximum performance out of your inference infrastructure.
