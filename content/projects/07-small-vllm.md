---
title: "small-vLLM: LLM Inference Engine"
description: I built an LLM inference engine that manages GPU memory, batches incoming requests, and streams generated text through an OpenAI-compatible API.
area: Model Serving
stack: Python, PyTorch, FlashAttention, CUDA Graphs, FastAPI
highlight: "Paged KV cache, continuous batching, CUDA Graphs, and INT4 weight loading"
importance: 1
---

## Overview

In small-vLLM, I built the pieces that let an inference engine serve multiple requests while working within a GPU memory budget. It uses a paged KV cache with reference-counted blocks to reuse shared prefixes, continuous batching to schedule requests, and chunked prefill and preemption to manage the work.

I connected those pieces to FlashAttention, CUDA Graphs, GPTQ and AWQ weight loading, and a streaming API. I also benchmarked the engine against vLLM on an RTX A6000; the measurements and workload settings are below.

## Selected results

- Integrated FlashAttention and a PyTorch reference attention backend, with CUDA Graphs for decode execution.
- Implemented paged KV caching with reference-counted blocks for prefix reuse, continuous batching, chunked prefill, and request preemption.
- Benchmarked steady-state generation on one RTX A6000 with a 4,096-token context limit, one request at a time, and 512 generated tokens per request. Measured 189.8 tokens/s on Qwen3-0.6B and 36.5 tokens/s on Qwen3-8B, compared with 393.9 and 42.4 tokens/s for vLLM, respectively, under the same benchmark conditions.
- Integrated GPTQ and AWQ INT4 weight loading, handling their distinct packed checkpoint layouts and routing execution through a shared Marlin backend.
- Exposed OpenAI-compatible completion and chat endpoints with streaming, alongside model discovery, health, and metrics endpoints in FastAPI.

## Source

[View small-vLLM on GitHub](https://github.com/sudhirpol522/small-vLLM)
