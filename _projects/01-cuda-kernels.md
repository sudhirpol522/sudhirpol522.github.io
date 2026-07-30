---
layout: page
title: FlashAttention V2 from Scratch
description: CUDA implementation of FlashAttention with tiled attention, online softmax, and Nsight-guided optimization.
area: LLM Inference
stack: CUDA C++, PyTorch, Nsight Compute
importance: 1
category: featured
---

## Overview

Implemented the FlashAttention V1 and V2 forward pass in CUDA C++ from the papers, then profiled and optimized the kernel for memory bandwidth and warp-level efficiency.

## Selected results

- Built tiled attention with online softmax and matched PyTorch scaled dot-product attention to within 2e-6 in fp32.
- Improved kernel performance by 8.5× by removing redundant HBM traffic, clearing shared-memory bank conflicts, and parallelizing softmax with warp-shuffle reductions.
- Profiled every revision in Nsight Compute and documented each optimization step with measured impact.

## Source

[View CUDA AI Kernels on GitHub](https://github.com/sudhirpol522/CUDA-AI-Kernels)
