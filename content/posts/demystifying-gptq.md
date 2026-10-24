---
title: "Demystifying GPTQ: From Lagrange Multipliers to Vectorized PyTorch"
date: 2026-06-14
description: "A derivation of the inverse Hessian update used by GPTQ, followed by a vectorized PyTorch implementation and 4-bit integer bit-packing."
tags: llm-inference quantization gptq
---

If you are optimizing inference engines or writing custom CUDA kernels for modern LLMs, you inevitably hit the memory wall. Serving a 70B parameter model in FP16 requires over 140GB of VRAM, forcing you into multi-GPU tensor parallelism just to load the weights.

Post-Training Quantization (PTQ) algorithms like GPTQ solve this by compressing weights down to 4-bits or 3-bits. But if you naively round the weights of a massive LLM, you destroy the model's perplexity.

GPTQ survives aggressive quantization by recognizing a fundamental truth: weights do not act in isolation. If you force one weight down, you can calculate exactly how much to push the neighboring weights up to absorb the structural damage.

In this post, we will derive the exact mathematics behind GPTQ from scratch using Lagrange multipliers, shadow-run a 2D example, translate the derivation into vectorized PyTorch code, and finally pack the resulting 4-bit integers into dense 32-bit checkpoint tensors.

---

## Part 1: The Problem Formulation

Imagine a single linear layer. We have an input matrix $X$ and a weight matrix $W$. We want to quantize a specific weight $w_q$ to a target integer value, $\text{quant}(w_q)$.

This introduces a forced change, or a quantization error, which we will call $\Delta_q$:

$$\Delta_q = \text{quant}(w_q) - w_q$$

Our goal is to find the optimal vector of weight updates, $\delta$, for the entire row of weights to compensate for this forced change. We want to minimize the mean squared error (MSE) of the layer's output.

Using a second-order Taylor expansion, the error introduced by changing the weights by $\delta$ is approximated by the Hessian matrix $H$:

$$E(\delta) = \frac{1}{2} \delta^T H \delta$$

Where $H = 2XX^T$ (the uncentered covariance of the input features).

**The Constraint:** We are not minimizing this error freely. We are locked to a strict rule: the $q$-th element of our update vector $\delta$ must equal our forced quantization change $\Delta_q$. Let $u_q$ be a unit vector (all zeros with a $1$ at index $q$). Our constraint is:

$$u_q^T \delta = \Delta_q$$

---

## Part 2: The Derivation (Lagrange Multipliers)

We need to minimize a quadratic equation subject to a linear constraint. This is the textbook definition of a Lagrange multiplier problem.

### Step 1: Write the Lagrangian

We combine our objective and our constraint using a dummy variable, $\lambda$:

$$L(\delta, \lambda) = \frac{1}{2} \delta^T H \delta + \lambda (u_q^T \delta - \Delta_q)$$

### Step 2: Set the gradient to zero

To find the bottom of the error curve, we take the derivative with respect to the vector $\delta$ and set it to zero:

$$\frac{\partial L}{\partial \delta} = H\delta + \lambda u_q = 0$$

Solving for the optimal update direction $\delta$:

$$H\delta = -\lambda u_q$$

$$\delta = -\lambda H^{-1} u_q \quad \dots \text{ (Equation i)}$$

**The Insight:** Multiplying any matrix by a unit vector $u_q$ extracts exactly the $q$-th column. This proves that the optimal way to update all the weights when you quantize weight $q$ is to move them in the exact proportions dictated by the $q$-th column of the inverse Hessian.

### Step 3: Solve for $\lambda$

We know our direction, but we need the exact scale. We substitute Equation (i) back into our strict constraint:

$$u_q^T (-\lambda H^{-1} u_q) = \Delta_q$$

$$-\lambda (u_q^T H^{-1} u_q) = \Delta_q$$

The term $(u_q^T H^{-1} u_q)$ extracts exactly the diagonal entry of the inverse Hessian at index $(q, q)$.

$$-\lambda [H^{-1}]_{qq} = \Delta_q$$

$$\lambda = \frac{-\Delta_q}{[H^{-1}]_{qq}}$$

### Step 4: The Final Update Formula

Substitute $\lambda$ back into Equation (i) to get the exact, final update vector:

$$\delta = \left(\frac{\Delta_q}{[H^{-1}]_{qq}}\right) \cdot H^{-1}_{[:, q]}$$

---

## Part 3: A Shadow Run with Real Numbers

Let's ground this math. Suppose we have an inverse Hessian for a 2-feature input, and we are updating a single output neuron with 2 weights: $w = [0.8, 0.6]^T$.

$$H^{-1} = \begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix}$$

We want to quantize $w_1$ (currently $0.8$) to $1.0$.

* Our forced change is $\Delta_1 = 0.2$.
* We look at the first column of $H^{-1}$: $[1, -1]^T$.
* Our diagonal $[H^{-1}]_{11}$ is $1$.

Calculate the update vector $\delta$:

$$\delta = \left(\frac{0.2}{1}\right) \cdot \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 0.2 \\ -0.2 \end{bmatrix}$$

**The Result:** Weight 1 shifts by $+0.2$ (landing perfectly on our quantized integer $1.0$). Because the input data is correlated, forcing weight 1 up damages the network. To perfectly absorb that structural damage, the Hessian tells us Weight 2 must shift by $-0.2$. The new unquantized weight becomes $0.4$.

---

## Part 4: The Secret of Row Independence

If we have a massive weight matrix of shape `[out_features, in_features]`, do we need a massive, uncomputable Hessian?

No. In a linear layer, output neurons do not share weights. When you change a weight in Row 1, you only damage the output of Neuron 1. Neuron 2 is completely unaffected.

Therefore, we only need the Hessian of the input features (shape `[in_features, in_features]`). We can apply this exact same small inverse Hessian to optimize every single row of the weight matrix independently.

Even better, we don't have to write a `for` loop across the rows. We can use vectorization to update every single output neuron simultaneously.

---

## Part 5: The Vectorized PyTorch Implementation

Here is the exact mathematical core of GPTQ translated into a high-throughput PyTorch execution.

Notice how the algorithm marches sequentially through the columns ($q = 0, 1, 2 \dots$). Once a column is quantized, it is locked. The compensation update is only applied to the remaining columns to its right (`q+1:`). If we applied the update to the whole matrix, we would add floating-point noise back onto our already-quantized integers, ruining our previous work.

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

If you run the standard `torch.linalg.inv` on a 70B model, the loop will fail—both numerically and mathematically. First, the inverse of million-element matrices introduces tiny floating-point rounding errors that compound exponentially by the final columns. Second, as columns $0 \dots q-1$ are locked, they can no longer absorb future error. A static $H^{-1}$ tries to route compensation back through already-frozen weights.

Because the Hessian matrix is perfectly symmetrical and positive-definite, production quantization engines swap the raw inverse for an upper-triangular Cholesky decomposition of the inverse (`torch.linalg.cholesky(torch.linalg.inv(H), upper=True)`). Reading the $q$-th row of this upper Cholesky factor automatically provides the exact conditional inverse of *only* the remaining unquantized columns while stabilizing floating-point arithmetic, allowing the exact same loop to compress a massive LLM on a single GPU in hours.

---

## Part 6: From Simulated Floats to 4-Bit Integer Codes (Scales & Zero-Points)

Look closely at `quantize_to_nearest(w)` in Part 5. Using `torch.round(w)` works as a conceptual placeholder, but real LLM weights live in tiny floating-point distributions (like $-0.8$ to $+0.7$). Calling `torch.round()` directly would crush almost every weight in the network to $0.0$.

Furthermore, `gptq_core` returns `Q`—a floating-point matrix with the exact same `[out_features, in_features]` shape and FP16 byte size as `W`. To actually shrink the model, we must map weights into **4-bit integer codes** (`0` to `15`) while tracking a **scale ($s$)** and **zero-point ($z$)** per group of weights.

### 1. Group-Wise Quantization Parameters

Instead of using a single scale for an entire row, GPTQ divides each row of length `in_features` into contiguous blocks of `group_size = 128`. For each group of 128 weights, we find the range $[a, b]$ containing zero:

$$a = \min(\min(w_{\text{group}}), 0), \qquad b = \max(\max(w_{\text{group}}), 0)$$

For a bit-width $B = 4$, the maximum integer code is $M = 2^4 - 1 = 15$. The **scale ($s$)** is the step size between adjacent representable weights, and the **zero-point ($z$)** is the integer code that represents exact floating-point $0.0$:

$$s = \frac{b - a}{15}, \qquad z = \text{clamp}\left(\text{round}\left(-\frac{a}{s}\right), 0, 15\right)$$

### 2. Encode, Reconstruct, and Propagate Error

Inside the GPTQ column loop, when we touch weight $w_q$, we perform a two-step **Encode-and-Reconstruct** dance:

1. **Encode to a 4-bit Integer Code ($c_q \in [0, 15]$):**

$$c_q = \text{clamp}\left(\text{round}\left(\frac{w_q}{s}\right) + z, 0, 15\right)$$


2. **Reconstruct the Simulated Float ($\hat{w}_q$):**

$$\hat{w}_q = s \cdot (c_q - z)$$



Suppose a 4-bit group has range $[-0.8, 0.7]$, giving $s = 0.1$ and $z = 8$:

| Working Weight ($w_q$) | 4-Bit Integer Code ($c_q$) | Reconstructed Float ($\hat{w}_q = 0.1(c_q - 8)$) | GPTQ Forced Change ($\Delta_q = \hat{w}_q - w_q$) |
| --- | --- | --- | --- |
| `-0.80` | `0` | `-0.80` | `0.00` |
| `0.00` | `8` | `0.00` | `0.00` |
| `0.26` | `11` | `0.30` | `+0.04` |
| `0.70` | `15` | `0.70` | `0.00` |

This distinction is critical:

* **We save the integer code `11**` to disk for memory compression.
* **We use the reconstructed float `0.30**` to compute $\Delta_q = 0.30 - 0.26 = +0.04$ so the Hessian compensation updates the remaining floating-point weights in their native coordinate space.

---

## Part 7: The "Empty Box" Problem — Bit-Packing `int4` into `int32`

Even after computing integer codes in the range `0..15`, we face a hardware problem: **GPU memory is byte-addressable, and PyTorch has no native `int4` tensor storage.**

If you store 16 quantized 4-bit codes inside a `torch.int32` tensor, PyTorch allocates 32 bits of memory for every single number. Look at what happens when the weight code `11` (`1011` in binary) sits inside a 32-bit `int32` container:

```text
[0000 0000 0000 0000 0000 0000 0000 | 1011]
 ^------------ 28 wasted zeros -----^   ^-- 4-bit weight (11)

```

Out of 32 bits, **28 bits are wasted zeros**. Storing 16 weights in `torch.int32` consumes $16 \times 32 = 512\text{ bits}$ (64 bytes)—which is actually *twice as large* as our original FP16 weights!

### How Bit-Packing Compresses 16 Weights into 2 `int32` Words

Since one `int32` container has 32 bits of space and each weight only needs 4 bits, we can pack **$32 \div 4 = 8$ weights** into a single `int32` word. Think of a 32-bit integer as an apartment building with **8 rooms**, each 4 bits wide:

```text
Bit position:  31..28   27..24   23..20   19..16   15..12   11..8    7..4     3..0
              [Room 7] [Room 6] [Room 5] [Room 4] [Room 3] [Room 2] [Room 1] [Room 0]

```

Let's walk through a concrete example. Suppose we have **16 quantized 4-bit weights** in `int32` that we want to compress into **2 packed `int32` words**:

```python
codes = [3, 10, 15, 0, 7, 12, 1, 9,   4, 14, 2, 8, 11, 5, 6, 13]
#        |------- Word 0 -------|     |------- Word 1 -------|

```

#### Step 1: Split into Groups of 8

We reshape the 16 codes into 2 groups of 8:

* **Group 0:** `[3, 10, 15, 0, 7, 12, 1, 9]`
* **Group 1:** `[4, 14, 2, 8, 11, 5, 6, 13]`

#### Step 2: Slide Each Weight Into Its 4-Bit Room (`<<`)

Right now, all 8 weights in Group 0 sit in Room 0 (bits `3..0`). If we added them together directly, they would collide. Instead, we slide each weight $k \in \{0 \dots 7\}$ to the left by $4k$ bits using the **left bit-shift operator (`<<`)**:

$$p = c_0 \;\vert\; (c_1 \ll 4) \;\vert\; (c_2 \ll 8) \;\vert\; \cdots \;\vert\; (c_7 \ll 28)$$

Look at the exact 32-bit binary layout of **Group 0** (`[3, 10, 15, 0, 7, 12, 1, 9]`) as each weight shifts into its designated room:

```text
                Room 7  Room 6  Room 5  Room 4  Room 3  Room 2  Room 1  Room 0
codes[0] =  3 <<  0 :   0000    0000    0000    0000    0000    0000    0000    0011
codes[1] = 10 <<  4 :   0000    0000    0000    0000    0000    0000    1010    0000
codes[2] = 15 <<  8 :   0000    0000    0000    0000    0000    1111    0000    0000
codes[3] =  0 << 12 :   0000    0000    0000    0000    0000    0000    0000    0000
codes[4] =  7 << 16 :   0000    0000    0000    0111    0000    0000    0000    0000
codes[5] = 12 << 20 :   0000    0000    1100    0000    0000    0000    0000    0000
codes[6] =  1 << 24 :   0000    0001    0000    0000    0000    0000    0000    0000
codes[7] =  9 << 28 :   1001    0000    0000    0000    0000    0000    0000    0000
------------------------------------------------------------------------------------   + (SUM / BITWISE OR)
Packed Word 0       :   1001    0001    1100    0111    0000    1111    1010    0011
                         (9)     (1)    (12)     (7)     (0)    (15)    (10)     (3)

```

Because each vertical column ("Room") only has non-zero bits in a single row, summing the 8 shifted integers (`shifted.sum(dim=-1)`) is identical to a Bitwise OR (`|`). Every 4-bit weight drops cleanly into its own 4-bit slot without any carry or collision.

In hexadecimal (where 1 hex character represents 4 bits), you can read our 8 weights right-to-left inside **Packed Word 0**: `0x91C70FA3` (`3`, `A`=10, `F`=15, `0`, `7`, `C`=12, `1`, `9`). Doing the exact same shift-and-sum on **Group 1** (`[4, 14, 2, 8, 11, 5, 6, 13]`) produces **Packed Word 1**: `0xD65B82E4`.

> **Why do packed `int32` weights look negative in Python?**
> If you print `Packed Word 0` (`0x91C70FA3`) as a signed `int32`, Python displays `-1849225309`. Because Bit 31 (the leftmost bit of Room 7, where `9` = `1001` lives) is `1`, Two's Complement interprets the 32-bit word as negative. That does not corrupt a single bit—the GPU kernel ignores the decimal sign and reads the raw 4-bit fields directly.

---

### How Unpacking Reverses the Process (`>>` and `& 0xF`)

When an inference kernel loads `Packed Word 0` (`0x91C70FA3`) from VRAM and wants to recover `codes[4]` (`7`, sitting in Room 4, bits `19..16`), it executes two single-cycle bitwise operations:

1. **Right-Shift by 16 bits (`word >> 16`):** Slides Room 4 all the way down into Room 0 (bits `3..0`), pushing Rooms 0–3 off the right edge:
```text
Room 7  Room 6  Room 5  Room 4  Room 3  Room 2  Room 1  Room 0
 0000    0000    0000    0000    1001    0001    1100   [0111]

```


2. **Bitwise AND with `15` (`& 0xF`):** The mask `15` in binary is `0000...00001111`. Bitwise AND zeroes out Rooms 1–7 and isolates only Room 0:
```text
  0000    0000    0000    0000    1001    0001    1100   [0111]  (Word 0 >> 16)
& 0000    0000    0000    0000    0000    0000    0000    1111   (Mask = 0xF)
---------------------------------------------------------------
= 0000    0000    0000    0000    0000    0000    0000   [0111]  = 7!

```



Here is the standalone PyTorch verification compressing 16 `int32` weights (64 bytes) into 2 `int32` words (8 bytes) and unpacking them losslessly:

```python
import torch

# 16 quantized 4-bit weights (0..15) stored in int32 (64 bytes)
codes = torch.tensor(
    [3, 10, 15, 0, 7, 12, 1, 9,  4, 14, 2, 8, 11, 5, 6, 13],
    dtype=torch.int32
)

shifts = torch.arange(8, dtype=torch.int64) * 4  # [0, 4, 8, 12, 16, 20, 24, 28]

# 1. PACK: [16] -> [2, 8] -> bit-shift into 8 rooms -> sum into [2] int32 words (8 bytes)
packed_words = (codes.to(torch.int64).reshape(2, 8) << shifts).sum(dim=-1).to(torch.int32)
# packed_words: tensor([-1849225309, -698645788], dtype=torch.int32) -> ['0x91c70fa3', '0xd65b82e4']

# 2. UNPACK: [2] -> right-shift all 8 rooms -> mask with 0xF (15) -> flatten to [16]
unsigned = packed_words.to(torch.int64) & 0xFFFFFFFF
unpacked = ((unsigned.unsqueeze(-1) >> shifts) & 0xF).reshape(-1).to(torch.int32)

assert torch.equal(codes, unpacked)  # Exact match: [3, 10, 15, 0, 7, 12, 1, 9, 4, 14, 2, 8, 11, 5, 6, 13]

```

---

## Part 8: Production Checkpoint Layout & End-to-End Implementation

When saving a quantized LLM layer to a `.safetensors` checkpoint for inference engines (like vLLM, Marlin, or ExLlama), we don't store weights in PyTorch's default `[out_features, in_features]` (`[N, K]`) orientation. Instead, GPTQ checkpoints transpose the matrix to `[in_features, out_features]` (`[K, N]`) so the input dimension $K$ is contiguous along columns.

For a standard `nn.Linear(4096, 4096)` layer with $K = 4096$ inputs, $N = 4096$ outputs, `bits = 4`, and `group_size = 128` ($4096 \div 128 = 32$ groups):

| Tensor Name | Shape (`[K=4096, N=4096]`) | Dtype | Memory Size | Packing Orientation |
| --- | --- | --- | --- | --- |
| **Original `weight**` | `[4096, 4096]` | `float16` | **33.55 MB** | Uncompressed dense `[N, K]` matrix |
| **`qweight`** | `[512, 4096]` (`[K // 8, N]`) | `int32` | **8.39 MB** | **Packed along $K$ (Axis 0):** 8 input rows per `int32` |
| **`qzeros`** | `[32, 512]` (`[groups, N // 8]`) | `int32` | **0.065 MB** | **Packed along $N$ (Axis 1):** 8 output channels per `int32` |
| **`scales`** | `[32, 4096]` (`[groups, N]`) | `float16` | **0.26 MB** | Unpacked FP16/BF16 scale per group and output channel |

Total memory drops from **33.55 MB to 8.71 MB** per layer—nearly a **4x reduction** including all metadata.

Here is the complete, end-to-end PyTorch implementation combining the Cholesky-stabilized GPTQ loop, dynamic group-wise `scales` and `zeros`, checkpoint bit-packing into `qweight` and `qzeros`, and the matching dequantization pass:

```python
import torch

def get_group_qparams(w_group, bits=4):
    """
    Computes asymmetric scale (s) and integer zero-point (z) per output row.
    w_group shape: [N, group_size]
    """
    max_code = (1 << bits) - 1  # 15 for 4-bit
    a = torch.minimum(w_group.min(dim=1, keepdim=True).values, torch.tensor(0.0, device=w_group.device))
    b = torch.maximum(w_group.max(dim=1, keepdim=True).values, torch.tensor(0.0, device=w_group.device))

    s = (b - a) / max_code
    s = torch.where(s == 0, torch.ones_like(s), s)
    z = torch.clamp(torch.round(-a / s), 0, max_code).to(torch.int32)
    return s.squeeze(1), z.squeeze(1)


def gptq_quantize_and_pack(W, H, bits=4, group_size=128):
    """
    Input:
      W: FP16/BF16 weight matrix of shape [N, K] ([out_features, in_features])
      H: FP32 Hessian matrix of shape [K, K] ([in_features, in_features])
    Output (Standard GPTQ Checkpoint Format):
      qweight: [K // 8, N]       int32   (8 input weights packed along K)
      qzeros:  [K // G, N // 8]  int32   (8 output zero-points packed along N)
      scales:  [K // G, N]       fp16    (1 scale per group and output neuron)
    """
    W = W.clone().float()
    N, K = W.shape
    pack = 32 // bits  # 8 values per int32
    num_groups = K // group_size
    max_code = (1 << bits) - 1

    codes = torch.zeros((N, K), dtype=torch.int32, device=W.device)
    scales = torch.zeros((num_groups, N), dtype=torch.float16, device=W.device)
    zeros = torch.zeros((num_groups, N), dtype=torch.int32, device=W.device)

    # 1. Hessian Dampening & Upper Cholesky Inverse
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    dampening = 0.01 * torch.mean(torch.diag(H))
    H = H + torch.eye(K, device=H.device) * dampening

    L = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(L)
    H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)

    # 2. Vectorized GPTQ Loop (Marching left to right across K)
    for q in range(K):
        if q % group_size == 0:
            g = q // group_size
            s, z = get_group_qparams(W[:, q : q + group_size], bits=bits)
            scales[g, :] = s.to(torch.float16)
            zeros[g, :] = z

        w_q = W[:, q]

        # Encode to 4-bit integer code [0..15] and reconstruct float w_hat
        code_q = torch.clamp(torch.round(w_q / s) + z, 0, max_code).to(torch.int32)
        w_hat = s * (code_q.float() - z.float())

        codes[:, q] = code_q

        # Calculate error using reconstructed float and scale by Cholesky diagonal
        delta_q = w_hat - w_q
        scale_factor = delta_q / H_inv_chol[q, q]

        # Compensate remaining unquantized columns
        if q + 1 < K:
            W[:, q + 1 :] += torch.outer(scale_factor, H_inv_chol[q, q + 1 :])

    # 3. Bit-Pack into GPTQ Checkpoint Layout
    shifts = torch.arange(pack, device=W.device, dtype=torch.int64) * bits

    # Transpose codes from [N, K] -> [K, N], then pack 8 rows along K into [K // 8, N]
    codes_kn = codes.t().contiguous().to(torch.int64)
    qweight = (codes_kn.reshape(K // pack, pack, N) << shifts[None, :, None]).sum(dim=1).to(torch.int32)

    # Pack 8 zero-points along N from [groups, N] into [groups, N // 8]
    qzeros = (zeros.to(torch.int64).reshape(num_groups, N // pack, pack) << shifts[None, None, :]).sum(dim=-1).to(torch.int32)

    return qweight, qzeros, scales


def unpack_and_dequantize_gptq(qweight, qzeros, scales, group_size=128):
    """
    Unpacks checkpoint tensors and reconstructs the [K, N] floating-point weight matrix.
    """
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)

    # 1. Unpack qweight: [K // 8, N] -> [K // 8, 8, N] -> [K, N]
    w_int4 = ((qweight[:, None, :] >> shifts[None, :, None]) & 0xF).reshape(-1, qweight.shape[1])

    # 2. Unpack qzeros: [groups, N // 8] -> [groups, N // 8, 8] -> [groups, N]
    z_int4 = ((qzeros[:, :, None] >> shifts[None, None, :]) & 0xF).reshape(qzeros.shape[0], -1)

    # 3. Expand group parameters across group_size=128 and dequantize: w_hat = s * (c - z)
    s_expanded = scales.repeat_interleave(group_size, dim=0)
    z_expanded = z_int4.repeat_interleave(group_size, dim=0).to(scales.dtype)

    return s_expanded * (w_int4.to(scales.dtype) - z_expanded)

```

---

## Conclusion

Algorithms like GPTQ and AWQ are what allow modern serving infrastructure to exist at scale. By understanding the full pipeline—from continuous Lagrange multipliers and second-order Cholesky Hessian updates down to discrete 4-bit integer grids and bitwise register packing—we can bridge the gap between theoretical model compression and the exact memory layouts required by high-performance CUDA kernels like Marlin.

*© 2026 Sudhir Pol · Email · GitHub · LinkedIn · Medium*