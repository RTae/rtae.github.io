+++
title = "DSpark: when speculative decoding starts caring about your batch"
date = "2026-08-03"
description = "A serving engineer's read of DSpark (arXiv:2607.05147) — how semi-autoregressive drafting fixes suffix decay, and why confidence-scheduled verification makes draft length a scheduling decision instead of a fixed hyperparameter."
tags = ["llm-inference", "llm-serving", "paper-review"]
+++

Most speculative decoding papers optimize a number that does not survive contact with a production serving system: tokens accepted per request, measured at batch size one. DSpark is interesting because it optimizes the thing you actually get billed for — throughput under concurrency — and in doing so it treats draft length as a scheduling decision rather than a hyperparameter.

<!--more-->

## Why decoding is the bottleneck


## What is speculative decoding?


## What's wrong with the prior methods?

### How DSpark works

## Results

## Demo with vLLM

## Demo with Deepinfer

## Takeaways

---

*Paper: [arXiv:2607.05147](https://arxiv.org/abs/2607.05147). All figures cited above are the authors'.*
