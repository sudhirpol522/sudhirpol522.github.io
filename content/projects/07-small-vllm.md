---
title: "small-vLLM: LLM Inference Engine"
description: A vLLM-style inference engine with paged KV caching, continuous batching, quantized weight loading, and OpenAI-compatible serving.
area: Model Serving
stack: Python, PyTorch, FlashAttention, CUDA Graphs, FastAPI
highlight: "Paged KV cache, continuous batching, CUDA Graphs, and INT4 weight loading"
importance: 1
---

## Overview

small-vLLM implements the core components of an LLM serving engine: paged KV-cache management, reference-counted blocks for prefix reuse, continuous batching, chunked prefill, and request preemption. The project connects scheduling and memory management with attention backends, quantized execution, and a streaming API.

## Selected results

- Integrated FlashAttention and a PyTorch reference attention backend, with CUDA Graphs for decode execution.
- Implemented paged KV caching with reference-counted blocks for prefix reuse, continuous batching, chunked prefill, and request preemption.
- Benchmarked steady-state generation on one RTX A6000 with a 4,096-token context limit, one request at a time, and 512 generated tokens per request. Measured 189.8 tokens/s on Qwen3-0.6B and 36.5 tokens/s on Qwen3-8B, compared with 393.9 and 42.4 tokens/s for vLLM, respectively, under the same benchmark conditions.
- Integrated GPTQ and AWQ INT4 weight loading, handling their distinct packed checkpoint layouts and routing execution through a shared Marlin backend.
- Exposed OpenAI-compatible completion and chat endpoints with streaming, alongside model discovery, health, and metrics endpoints in FastAPI.

## Source

[View small-vLLM on GitHub](https://github.com/sudhirpol522/small-vLLM)
