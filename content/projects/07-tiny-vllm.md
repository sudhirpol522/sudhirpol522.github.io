---
title: "Tiny-vLLM: LLM Inference Engine"
description: A CPU and Apple Silicon LLM inference engine with paged KV caching, continuous batching, and an optional MLX backend, benchmarked against Hugging Face.
area: LLM Inference
stack: Python, PyTorch, MLX, FastAPI
highlight: "Paged KV cache + continuous batching"
importance: 1
---

## Overview

Tiny-vLLM is a CPU and Apple Silicon (MPS) inference engine built around the serving ideas behind production engines like vLLM: paged KV-cache management with prefix caching, continuous batching with chunked prefill, and OpenAI-compatible HTTP endpoints with streaming. The core engine components (KV cache manager, scheduler, reference attention, model definitions, and async server) are adapted from [tinyserve](https://github.com/priyammaz/tinyserve), used with the author's permission, with local additions for benchmarking, testing, and an optional MLX backend. This version does not include CUDA kernels or CUDA graphs.

## Selected results

- Implemented paged KV-cache management with prefix caching, continuous batching with chunked prefill, and greedy/temperature/top-k/top-p sampling from safetensors checkpoints.
- Added OpenAI-compatible completion and chat endpoints with SSE streaming and client-disconnect cleanup.
- Built an optional MLX backend for Apple GPUs; compiled decode reduced generation latency by 20.7–24.5% versus eager MLX on a local fixture.
- Benchmarked against Hugging Face's eager, padded-batch implementation on a small random Qwen3 fixture (Apple M5, float32, CPU and MPS, seven-run medians): Tiny-vLLM was faster on a single request (4,681 vs. 3,160 output tok/s on CPU) but slower than HF's static batch on multi-request, mixed-length and shared-prefix workloads on this fixture. These numbers describe the small test fixture; trained models would need separate measurement.
- Verified identical greedy token outputs against Hugging Face across chunking, prefix reuse, and preemption, with 96 passing tests.

## Source

[View Tiny-vLLM on GitHub](https://github.com/sudhirpol522/Tiny-vLLM)
