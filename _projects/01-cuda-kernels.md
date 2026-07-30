---
layout: page
title: CUDA Kernels for LLM Inference
description: FlashAttention kernels and GPU optimization work developed from first principles.
area: LLM Inference
stack: CUDA C++, PyTorch, Nsight Compute
importance: 1
category: featured
---

## Overview

This project builds CUDA kernels for LLM inference from the ground up, moving from memory hierarchy and
coalesced access patterns to optimized attention implementations.

## Selected results

- Implemented FlashAttention V1 and V2 forward passes with tiled attention and online softmax.
- Matched PyTorch scaled dot product attention to within 2e-6 in fp32.
- Improved kernel performance by 8.5 times by removing redundant HBM traffic, resolving shared memory bank conflicts, and parallelizing softmax with warp shuffle reductions.
- Documented profiling methods with CUDA events and Nsight Compute.

## Source

[View CUDA AI Kernels on GitHub](https://github.com/sudhirpol522/CUDA-AI-Kernels)
