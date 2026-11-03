---
title: DeepSpeech2 for Automatic Speech Recognition
description: A PyTorch speech-to-text implementation with masked convolutions, a bidirectional LSTM, and character-level CTC decoding.
area: Speech Recognition
stack: Python, PyTorch, LibriSpeech
highlight: "18.42% mean per-utterance WER on LibriSpeech"
importance: 4
---

## Overview

Implemented a DeepSpeech2 speech-to-text model in PyTorch with two masked convolutional layers, a bidirectional LSTM, and character-level connectionist temporal classification (CTC) loss. The model processes 80-bin mel spectrograms and handles variable-length LibriSpeech audio throughout the convolutional and recurrent stages.

## Selected results

- Downsampled the mel spectrograms by 2× along the time axis before the LSTM.
- Tracked valid sequence lengths through convolution and packed recurrent inputs to handle variable-length audio.
- Evaluated greedy CTC decoding across 5,323 LibriSpeech utterances from dev-clean and test-clean. The linked notebook reports 18.42% mean per-utterance word error rate (WER).

## Source

[View the DeepSpeech2 notebook on GitHub](https://github.com/sudhirpol522/DeepSpeech/blob/main/deepspeech2.ipynb)
