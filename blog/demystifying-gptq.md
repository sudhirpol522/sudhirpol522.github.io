---
layout: page
title: "Demystifying GPTQ: From Lagrange Multipliers to Vectorized PyTorch"
permalink: /blog/demystifying-gptq/
math: true
---

If you are optimizing inference engines or writing custom CUDA kernels for modern LLMs, you inevitably hit the memory wall. Serving a 70B parameter model in FP16 requires over 140GB of VRAM—forcing you into multi-GPU tensor parallelism just to load the weights.

Post-Training Quantization (PTQ) algorithms like GPTQ solve this by compressing weights down to 4-bits or 3-bits. But if you naively round the weights of a massive LLM, you destroy the model's perplexity.

GPTQ survives aggressive quantization by recognizing a fundamental truth: **weights do not act in isolation**. If you force one weight down, you can calculate exactly how much to push the neighboring weights up to absorb the structural damage.

In this post, we will derive the exact mathematics behind GPTQ from scratch using Lagrange multipliers, shadow-run a 2D example, and finally translate the derivation into highly optimized, vectorized PyTorch code.

---

## Part 1: The Problem Formulation

Imagine a single linear layer. We have an input matrix $$X$$ and a weight matrix $$W$$. We want to quantize a specific weight $$w_q$$ to a target integer value, $$quant(w_q)$$.

This introduces a forced change, or a quantization error, which we will call $$\Delta_q$$:

$$\Delta_q = quant(w_q) - w_q$$

Our goal is to find the optimal vector of weight updates, $$\delta$$, for the entire row of weights to compensate for this forced change. We want to minimize the mean squared error (MSE) of the layer's output.

Using a second-order Taylor expansion, the error introduced by changing the weights by $$\delta$$ is approximated by the Hessian matrix $$H$$:

$$E(\delta) = \frac{1}{2} \delta^T H \delta$$

Where $$H = 2 X X^T$$ (the uncentered covariance of the input features).

**The Constraint:** We are not minimizing this error freely. We are locked to a strict rule: the $$q$$-th element of our update vector $$\delta$$ *must* equal our forced quantization change $$\Delta_q$$. Let $$\mathbf{u}_q$$ be a unit vector (all zeros with a $$1$$ at index $$q$$). Our constraint is:

$$\mathbf{u}_q^T \delta = \Delta_q$$

---

## Part 2: The Derivation (Lagrange Multipliers)

We need to minimize a quadratic equation subject to a linear constraint. This is the textbook definition of a Lagrange multiplier problem.

### Step 1: Write the Lagrangian

We combine our objective and our constraint using a dummy variable, $$\lambda$$:

$$L(\delta, \lambda) = \frac{1}{2} \delta^T H \delta + \lambda (\mathbf{u}_q^T \delta - \Delta_q)$$

### Step 2: Set the gradient to zero

To find the bottom of the error curve, we take the derivative with respect to the vector $$\delta$$ and set it to zero:

$$\frac{\partial L}{\partial \delta} = H \delta + \lambda \mathbf{u}_q = 0$$

Solving for the optimal update direction $$\delta$$:

$$H \delta = -\lambda \mathbf{u}_q$$

$$\delta = -\lambda H^{-1} \mathbf{u}_q \quad \text{... (Equation i)}$$

> **The Insight:** Multiplying any matrix by a unit vector $$\mathbf{u}_q$$ extracts exactly the $$q$$-th column. This proves that the optimal way to update *all* the weights when you quantize weight $$q$$ is to move them in the exact proportions dictated by the $$q$$-th column of the inverse Hessian.

### Step 3: Solve for $$\lambda$$

We know our direction, but we need the exact scale. We substitute Equation (i) back into our strict constraint:

$$\mathbf{u}_q^T (-\lambda H^{-1} \mathbf{u}_q) = \Delta_q$$

$$-\lambda (\mathbf{u}_q^T H^{-1} \mathbf{u}_q) = \Delta_q$$

The term $$(\mathbf{u}_q^T H^{-1} \mathbf{u}_q)$$ extracts exactly the diagonal entry of the inverse Hessian at index $$(q, q)$$.

$$-\lambda [H^{-1}]_{qq} = \Delta_q$$

$$\lambda = \frac{-\Delta_q}{[H^{-1}]_{qq}}$$

### Step 4: The Final Update Formula

Substitute $$\lambda$$ back into Equation (i) to get the exact, final update vector:

$$\delta = \left( \frac{\Delta_q}{[H^{-1}]_{qq}} \right) \cdot H^{-1}[:, q]$$

---

## Part 3: A Shadow Run with Real Numbers

Let's ground this math. Suppose we have an inverse Hessian for a 2-feature input, and we are updating a single output neuron with 2 weights: $$w = [0.8, 0.6]^T$$.

$$H^{-1} = \begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix}$$

We want to quantize $$w_1$$ (currently **0.8**) to **1.0**.

* Our forced change is $$\Delta_1 = 0.2$$.
* We look at the first column of $$H^{-1}$$: $$[1, -1]^T$$.
* Our diagonal $$[H^{-1}]_{11}$$ is $$1$$.

**Calculate the update vector $$\delta$$:**

$$\delta = \left( \frac{0.2}{1} \right) \cdot \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 0.2 \\ -0.2 \end{bmatrix}$$

**The Result:** Weight 1 shifts by $$+0.2$$ (landing perfectly on our quantized integer $$1.0$$).
Because the input data is correlated, forcing weight 1 up damages the network. To perfectly absorb that structural damage, the Hessian tells us Weight 2 must shift by $$-0.2$$. The new unquantized weight becomes $$0.4$$.

---

## Part 4: The Secret of Row Independence

If we have a massive weight matrix of shape `[out_features, in_features]`, do we need a massive, uncomputable Hessian?

No. In a linear layer, **output neurons do not share weights**. When you change a weight in Row 1, you only damage the output of Neuron 1. Neuron 2 is completely unaffected.

Therefore, we only need the Hessian of the *input features* (shape `[in_features, in_features]`). We can apply this exact same small inverse Hessian to optimize every single row of the weight matrix independently.

Even better, we don't have to write a `for` loop across the rows. We can use vectorization to update every single output neuron simultaneously.

---

## Part 5: The Vectorized PyTorch Implementation

Here is the exact mathematical core of GPTQ translated into a high-throughput PyTorch execution.

Notice how the algorithm marches sequentially through the columns ($$q = 0, 1, 2...$$). Once a column is quantized, it is locked. The compensation update is *only* applied to the remaining columns to its right (`q+1:`). If we applied the update to the whole matrix, we would add floating-point noise back onto our already-quantized integers, ruining our previous work.

```python
import torch

def quantize_to_nearest(w):
    """
    Placeholder for symmetric/affine quantization.
    """
    return torch.round(w)

def gptq_core(W, H):
    """
    W: Weight matrix of shape [out_features, in_features]
    H: Hessian matrix of shape [in_features, in_features]
    """
    out_features, in_features = W.shape
    Q = torch.zeros_like(W)
    
    # 1. Hessian Preparation (Dampening for numerical stability)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    dampening = 0.01 * torch.mean(torch.diag(H))
    H = H + torch.eye(in_features) * dampening
    
    # In production, use Cholesky decomposition instead of raw inverse 
    # to prevent floating-point catastrophic cancellation on massive matrices.
    # H_inv = torch.linalg.cholesky(torch.linalg.inv(H), upper=True)
    H_inv = torch.linalg.inv(H)
    
    # 2. The Core Quantization Loop (Marching left to right)
    for q in range(in_features):
        
        # Grab the q-th column (active weight for all output neurons)
        w_q = W[:, q]
        
        # Quantize and lock into our final output matrix
        quantized_w_q = quantize_to_nearest(w_q)
        Q[:, q] = quantized_w_q
        
        # Step 1: Calculate forced change (shape: [out_features])
        delta_q = quantized_w_q - w_q 
        
        # Step 3: Calculate scale factor lambda (shape: [out_features])
        scale_factor = delta_q / H_inv[q, q] 
        
        # Step 4: The Vectorized Compensation
        # We grab the remaining slice of the inverse Hessian.
        # Note: We slice the row (H_inv[q, q+1:]) instead of the column 
        # because PyTorch memory is C-contiguous (row-major). Since the 
        # Hessian is perfectly symmetric, row == col, but row is vastly 
        # faster for the GPU memory bus.
        h_inv_slice = H_inv[q, q+1:]
        
        # torch.outer computes the update for ALL output neurons instantly
        update = torch.outer(scale_factor, h_inv_slice)
        
        # Apply the compensation ONLY to the remaining unquantized weights
        W[:, q+1:] = W[:, q+1:] + update
        
    return Q
```

### Engineering Polish: Why Cholesky?

If you run the standard `torch.linalg.inv` on a 70B model, the loop will fail. The inverse of million-element matrices introduces tiny floating-point rounding errors. By the time the loop reaches the final columns, those errors compound exponentially, and the weight updates become catastrophic.

Because the Hessian matrix is perfectly symmetrical and positive-definite, production quantization engines swap the raw inverse for a **Cholesky decomposition** of the inverse (`torch.linalg.cholesky`). This heavily optimized, triangular matrix perfectly handles floating-point stabilization, allowing the exact same loop to compress a massive LLM on a single GPU in hours.

---

### Conclusion

Algorithms like GPTQ and AWQ are what allow modern serving infrastructure to exist at scale. By understanding the underlying continuous mathematics—Lagrange multipliers and second-order Hessian approximations—we can write the discrete, vectorized matrix operations required to optimize the next generation of inference engines.
