---
layout: page
title: Probabilistic Yard Gain Forecasting
description: Calibrated play-outcome modeling from NFL tracking data, framed as a full cumulative distribution rather than a point estimate.
area: Applied Machine Learning
stack: Python, Scikit-learn, XGBoost, Pandas, NumPy, Flask
importance: 6
category: featured
---

## Overview

Framed yard-gain prediction as a probabilistic problem and predicted the complete cumulative distribution of outcomes rather than a single point estimate.

## Selected results

- Engineered play-level spatial and football-specific features from tracking data.
- Benchmarked logistic regression, SVM, random forest, gradient boosted trees, and XGBoost on held-out data.
- Calibrated forecast probabilities with isotonic regression for threshold-based decisions.
- Served the model through a Flask application for interactive evaluation.

## Source

[View the project on GitHub](https://github.com/sudhirpol522/Heroku-NFL)
