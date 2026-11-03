---
title: DeepSpeech2 for Automatic Speech Recognition
description: I implemented a DeepSpeech2 speech-to-text model in PyTorch and evaluated its transcriptions on LibriSpeech audio of varying lengths.
area: Speech Recognition
stack: Python, PyTorch, LibriSpeech
highlight: "18.42% mean per-utterance WER on LibriSpeech"
importance: 4
---

## Overview

I built a DeepSpeech2 speech-to-text model in PyTorch and evaluated the words it produced on LibriSpeech recordings. The model uses two masked convolutional layers, a bidirectional LSTM, and character-level connectionist temporal classification (CTC) loss.

The recordings vary in length, so I tracked valid sequence lengths through the convolutional layers and packed the recurrent inputs. The model takes 80-bin mel spectrograms, downsamples them along the time axis, and produces transcriptions using greedy CTC decoding.

## Selected results

- Downsampled the mel spectrograms by 2× along the time axis before the LSTM.
- Tracked valid sequence lengths through convolution and packed recurrent inputs to handle variable-length audio.
- Evaluated greedy CTC decoding across 5,323 LibriSpeech utterances from dev-clean and test-clean. The linked notebook reports 18.42% mean per-utterance word error rate (WER).

## Source

[View the DeepSpeech2 notebook on GitHub](https://github.com/sudhirpol522/DeepSpeech/blob/main/deepspeech2.ipynb)
