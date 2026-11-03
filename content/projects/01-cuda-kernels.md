---
title: FlashAttention-1 CUDA Kernels
description: I wrote FlashAttention-1 kernels in CUDA, improved their shared-memory access and softmax reductions, and checked the outputs against PyTorch.
area: LLM Inference
stack: CUDA, C++, PyTorch C++ extension
highlight: "Validated against PyTorch SDPA"
importance: 2
---

## Overview

I implemented the non-causal FlashAttention-1 forward pass in CUDA, then worked through shared-memory bank conflicts and serialized softmax reductions in the first version. Both kernel variants run through a PyTorch C++/CUDA extension built with pybind11.

The implementation uses tiled attention and online softmax, with one thread block per batch, head, and query tile. Before timing either kernel, the benchmark checks its output against PyTorch scaled dot-product attention.

## Selected results

- Traced the first implementation's shared-memory bank conflicts in the QK<sup>T</sup> inner loop and removed them by padding the K tile to (d+1).
- Replaced a lane-0 serialized softmax reduction with warp-shuffle (`__shfl_xor_sync`) max and sum reductions across all 32 lanes.
- Built a correctness-gated benchmark harness that validates both kernels against `torch.nn.functional.scaled_dot_product_attention` to a max absolute error under 5e-2 (FP32) before timing, sweeping sequence lengths 128 to 1024 at batch 8, 12 heads, head dimension 64.

## Source

[View fa1-noncausal-benchmark on GitHub](https://github.com/sudhirpol522/fa1-noncausal-benchmark) — the repository doesn't have a README yet, so the kernel files are the best reference for now.
