---
layout: page
title: Continuous Batching Inference Server
description: A minimal serving engine that demonstrates the scheduling ideas behind high throughput LLM inference.
area: Model Serving
stack: Python, PyTorch, LLM Inference
importance: 5
category: featured
---

## Overview

This project implements an iteration level scheduling loop for variable length requests and makes the tradeoffs
behind continuous batching visible in a compact codebase.

## Selected results

- Added dynamic request scheduling and iteration level batching.
- Managed a bounded KV cache budget under changing sequence lengths.
- Implemented preemption for lower priority sequences under memory pressure.
- Documented why sequential serving leaves accelerator capacity unused.

## Source

[View Continuous Batching on GitHub](https://github.com/sudhirpol522/Continuous-Batching)
