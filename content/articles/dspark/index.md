+++
title = "DSpark: when speculative decoding starts caring about your batch"
date = "2026-08-03"
description = "A serving engineer's read of DSpark (arXiv:2607.05147) — how semi-autoregressive drafting fixes suffix decay, and why confidence-scheduled verification makes draft length a scheduling decision instead of a fixed hyperparameter."
tags = ["llm-inference", "llm-serving", "paper-review"]
+++

Most speculative decoding papers optimize a number that does not survive contact with a production serving system: tokens accepted per request, measured at batch size one. DSpark is interesting because it optimizes the thing you actually get billed for — throughput under concurrency — and in doing so it treats draft length as a scheduling decision rather than a hyperparameter.

<!--more-->

## TLDR

## Why decoding is the bottleneck
In LLM arhitechture are two main process that model to do first is prefill process and second is decode process. Prefill process is the process that model generate the first token of the output sequence, and decode process is the process that model generate the rest of the output sequence. In prefill process, model can generate the first token in parallel for all request in a batch which in practice is very fast and compute bound. However, in decode process, model can only generate the next token after the previous token is generated which in practice is very slow and memory bound due to the speed of the bandwith memory is very difference from the speed of the compute. Therefore, decode process is the bottleneck of the LLM inference.

There are many method try resolve this issue such as disaggregated inference, model quantization, model distillation, speculative decoding and hardware specific to improve bandwidth memory (Like nvidia Rubin). In this article, we will focus on the speculative decoding method.

## What is speculative decoding?
Speculative decoding takes advantage of the gap described above. Because decode is memory-bound, a forward pass over k tokens costs roughly the same as a pass over one token. So instead of asking the large *target* model for one token at a time, we let a cheap *drafter* guess the next k tokens, then have the target check all k of them in a single forward pass, which works like a small prefill. We keep the longest prefix the target agrees with, plus one token from the target itself, so every step produces at least one token. With rejection sampling, the output distribution is exactly the same as the target model's. The method is lossless, unlike quantization or distillation.

The speedup depends on two things: how cheap the draft is, and how many draft tokens get accepted. The two main families of drafters trade these off differently.

1. **Autoregressive drafter** (e.g. a small LM, EAGLE): the drafter generates the k tokens one at a time, each conditioned on the previous one. Drafts are high quality because every token sees the tokens before it, but drafting cost grows linearly with k, so draft lengths stay short.

2. **Parallel drafter** (e.g. Medusa heads, block or diffusion drafters): the drafter predicts all k tokens in a single forward pass. Drafting cost stays roughly constant as k grows, but each position is predicted without seeing the actual tokens before it, so the acceptance rate drops at later positions in the block.

## What's wrong with the prior methods?
1. Previous methods for parallel drafter model inter-token dependencies with in a block leads to multi-modal and the accepted token rate is decay  at later positon

## How DSpark works

### Semi-autoregressive generation

### Confidence-scheduled verification
#### Confidence Head
#### Hardware-Aware Prefix Scheduler

## Results

### Experimental Setup
### Experimental Results
### Experimental Analysis
1. Why can Parallel drafter outperform Autoregressive drafter?
2. A little Autoregressive drafter goes a long way
3. Verify Smater, not Longer: The role of Confidence Scheduling
4. For production results.
   1. Scalability:
   2. Hareware-Aware Prefix Scheduler:
   3. High Throughput with Low Latency:
   4. Reult from real traffic:
### Limitations and Future Work

## Demo with vLLM

## Demo with Deepinfer

## Takeaways

---

*Paper: [arXiv:2607.05147](https://arxiv.org/abs/2607.05147). All figures cited above are the authors'.*
