---
title: "IntentShop: Session-Based Recommendation"
description: I built a recommender that uses a shopping session to suggest the next product, then tested it on later sessions to measure how well it generalizes.
area: Applied Machine Learning
stack: Python, PyTorch, pandas, NumPy, Google Colab
highlight: "0.3356 NDCG@10 on chronological evaluation"
importance: 3
---

## Overview

With IntentShop, I built a system that recommends the next product based on a shopping session. It first retrieves candidates from a catalog of 14,507 products using a neural model, then uses an eight-feature residual reranker to refine their order.

I evaluated the recommendations on later requests, keeping the train and evaluation data in chronological order. Leakage checks, feature ablations, and comparisons across three random seeds helped me assess whether the reranker improved on retrieval alone.

## Selected results

- Trained the residual reranker on 91K next-product examples.
- Achieved 0.3356 NDCG@10 on 18,312 held-out chronological requests, improving 13.6% over neural retrieval ordering and 27.4% over an item-transition baseline.
- Established chronological splits, leakage checks, feature ablations, and ranker comparisons across three random seeds.
- Used paired session bootstrap evaluation; the primary NDCG gain had a 95% interval of +0.0365 to +0.0440.

## Source

[View IntentShop on GitHub](https://github.com/sudhirpol522/IntentShop)
