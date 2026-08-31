---
layout: page
title: Speculative Decoding from Scratch
description: A complete draft and verify implementation with correctness derivations and performance benchmarks.
area: LLM Inference
stack: PyTorch, Hugging Face Transformers
importance: 4
category: featured
---

## Overview

This project implements speculative decoding end to end, including accept and reject sampling and the adjusted
residual distribution required to preserve the target model output distribution.

## Selected results

- Benchmarked multiple draft lengths and acceptance rates.
- Measured 3.4 times faster generation at a draft length of 4.
- Reached 95 tokens per second against a 28 token per second baseline.
- Connected the implementation to a full derivation of acceptance probability and expected speedup.

## Source

[View Speculative Decoding on GitHub](https://github.com/sudhirpol522/Speculative-Decoding)
