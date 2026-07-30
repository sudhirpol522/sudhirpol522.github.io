---
layout: page
title: QuantRoute
description: A quantization aware inference router designed around serving quality, latency, and cost.
area: Model Serving
stack: vLLM, llm-compressor, AWQ, GPTQ, NVIDIA A100
importance: 2
category: featured
---

## Overview

QuantRoute serves multiple precision lanes and learns when a request needs the more expensive path.
The project combines model compression, difficulty prediction, and offline evaluation into one serving decision.

## Selected results

- Quantized Qwen2.5 7B to 4 bit with AWQ and GPTQ.
- Reached 2.2 times faster decode and 56 percent lower serving cost on financial tasks.
- Built a learned difficulty classifier with an LLM as Judge escalation loop.
- Used offline oracle analysis to separate routing errors from model quality limits.
