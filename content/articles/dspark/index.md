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
LLM inference has two phases: **prefill** and **decode**. In prefill, the model processes the whole prompt in one forward pass, building the KV cache and producing the first output token. Every prompt token is processed in parallel, so each weight loaded from memory is reused across many tokens. Prefill is therefore compute-bound and fast per token.

Decode produces the rest of the output, one token per forward pass, because each token depends on the one before it. Every step has to stream the full model weights (and the KV cache) from GPU memory to produce just one token per request. There is very little math to do for each byte loaded, so the GPU spends most of its time waiting on memory bandwidth, which is far slower than its compute. Decode is memory-bound, and since it runs once per output token, it dominates end-to-end latency.

There are many ways to attack this: prefill/decode disaggregation, quantization, distillation, hardware with more memory bandwidth (such as NVIDIA Rubin), and speculative decoding. This article focuses on speculative decoding.

## What is speculative decoding?
Speculative decoding takes advantage of the gap described above. Because decode is memory-bound, a forward pass over k tokens costs roughly the same as a pass over one token. So instead of asking the large *target* model for one token at a time, we let a cheap *drafter* guess the next k tokens, then have the target check all k of them in a single forward pass, which works like a small prefill. We keep the longest prefix the target agrees with, plus one token from the target itself, so every step produces at least one token. With rejection sampling, the output distribution is exactly the same as the target model's. The method is lossless, unlike quantization or distillation.

The speedup depends on two things: how cheap the draft is, and how many draft tokens get accepted. The two main families of drafters trade these off differently.

1. **Autoregressive drafter** (e.g. a small LM, EAGLE): the drafter generates the k tokens one at a time, each conditioned on the previous one. Drafts are high quality because every token sees the tokens before it, but drafting cost grows linearly with k, so draft lengths stay short.

2. **Parallel drafter** (e.g. Medusa heads, block or diffusion drafters): the drafter predicts all k tokens in a single forward pass. Drafting cost stays roughly constant as k grows, but each position is predicted without seeing the actual tokens before it, so the acceptance rate drops at later positions in the block.

## What's wrong with the prior methods?
1. **Parallel drafters ignore dependencies inside the block.** Each position in the block is predicted independently, without knowing which token the earlier positions chose. When several continuations are plausible (the distribution is multi-modal), different positions can lock onto different ones. For example, after "I'm flying to", position 1 may guess "New" while position 2 guesses "Angeles", because both "New York" and "Los Angeles" are likely. Each token looks reasonable on its own, but together they are nonsense. Since verification only accepts a prefix, one wrong token throws away everything after it, so errors compound and acceptance decays at later positions. This is *suffix decay*: making the block longer adds draft tokens that almost never get accepted.

2. **Draft length is a fixed hyperparameter.** Both families draft and verify the same k tokens for every request at every step. At batch size one that is nearly free, because verification uses compute that would otherwise sit idle. Under real concurrency, decode becomes less memory-bound as the batch grows, so every draft token the target verifies and then rejects is wasted compute that could have served another request. A k tuned on single-request benchmarks drafts too few tokens for easy requests that would accept more, and wastes verification on hard ones.

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
