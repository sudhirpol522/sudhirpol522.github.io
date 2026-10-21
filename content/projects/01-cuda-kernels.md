---
title: FlashAttention-1 CUDA Kernels
description: A tiled, non-causal FlashAttention-1 forward kernel in CUDA, built and optimized as a PyTorch C++ extension.
area: LLM Inference
stack: CUDA, C++, PyTorch C++ extension
highlight: "Validated against PyTorch SDPA"
importance: 3
---

## Overview

Implemented a tiled, non-causal FlashAttention-1 forward kernel in CUDA with online softmax, using one thread block per (batch, head, query tile) and shared-memory staging for the Q, K, V, score, and output-accumulator tiles. Both kernel variants are exposed to PyTorch as a C++/CUDA extension via pybind11.

## Selected results

- Traced the first implementation's shared-memory bank conflicts in the QK<sup>T</sup> inner loop and removed them by padding the K tile to (d+1).
- Replaced a lane-0 serialized softmax reduction with warp-shuffle (`__shfl_xor_sync`) max and sum reductions across all 32 lanes.
- Built a correctness-gated benchmark harness that validates both kernels against `torch.nn.functional.scaled_dot_product_attention` to a max absolute error under 5e-2 (FP32) before timing, sweeping sequence lengths 128 to 1024 at batch 8, 12 heads, head dimension 64.

## Source

[View fa1-noncausal-benchmark on GitHub](https://github.com/sudhirpol522/fa1-noncausal-benchmark) — the repository doesn't have a README yet, so the kernel files are the best reference for now.
