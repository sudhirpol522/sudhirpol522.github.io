---
title: "IntentShop: Session-Based Recommendation"
description: A two-stage e-commerce recommender combining session-aware neural retrieval with a residual reranker and chronological evaluation.
area: Applied Machine Learning
stack: Python, PyTorch, pandas, NumPy, Google Colab
highlight: "0.3356 NDCG@10 on chronological evaluation"
importance: 3
---

## Overview

IntentShop is a session-based recommendation system over 14,507 products. A session-aware neural retriever generates candidates, and an eight-feature residual reranker refines their ordering. The project evaluates next-product recommendations with chronological splits and explicit checks for data leakage.

## Selected results

- Trained the residual reranker on 91K next-product examples.
- Achieved 0.3356 NDCG@10 on 18,312 held-out chronological requests, improving 13.6% over neural retrieval ordering and 27.4% over an item-transition baseline.
- Established chronological splits, leakage checks, feature ablations, and ranker comparisons across three random seeds.
- Used paired session bootstrap evaluation; the primary NDCG gain had a 95% interval of +0.0365 to +0.0440.

## Source

[View IntentShop on GitHub](https://github.com/sudhirpol522/IntentShop)
