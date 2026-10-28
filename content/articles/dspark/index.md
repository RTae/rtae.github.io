+++
title = "DSpark: when speculative decoding starts caring about your batch"
date = "2026-08-03"
description = "A serving engineer's read of DSpark (arXiv:2607.05147) — how semi-autoregressive drafting fixes suffix decay, and why confidence-scheduled verification makes draft length a scheduling decision instead of a fixed hyperparameter."
tags = ["llm-inference", "llm-serving", "paper-review"]
+++

Most speculative decoding papers optimize a number that does not survive contact with a production serving system: tokens accepted per request, measured at batch size one. DSpark is interesting because it optimizes the thing you actually get billed for throughput under concurrency and in doing so it treats draft length as a scheduling decision rather than a hyperparameter.

<!--more-->

# TLDR
TBD

# 1. Why decoding is the bottleneck

{{< figure src="decode-bottleneck.png" link="decode-bottleneck.png" alt="Two panels. Prefill: the model weights are read from GPU memory once, all prompt tokens are computed together, and K,V for every prompt token is written to the KV cache, so compute stays busy. Decode: each step reads all the weights and the whole KV cache again, computes only the newest token, and writes its K,V back, so compute waits for the data most of each step." caption="Illustrative, not to scale: what moves between GPU memory and compute in each phase." >}}

LLM inference has two phases, **prefill** and **decode**. In prefill, the model processes the whole prompt in one forward pass, building the KV cache and producing the first output token. Every prompt token is processed in parallel, so each weight read from memory is reused across many tokens. Prefill is therefore compute-bound and fast per token.

Decode produces the rest of the output, one token per forward pass, because each token depends on the one before it. Every step has to read all the model weights and the KV cache from GPU memory to produce just one token per request, and since the KV cache grows with every token, these reads get longer as the output grows. There is very little computation to do for each byte read, so the GPU spends most of its time waiting on memory bandwidth, which is far slower than its compute. Decode is memory-bound, and since it runs once per output token, it dominates end-to-end latency.

There are many ways to tackle this problem, including prefill/decode disaggregation, quantization, hardware with more memory bandwidth (such as NVIDIA Rubin), and speculative decoding. This article focuses on speculative decoding.

# 2. What is speculative decoding?

Speculative decoding takes advantage of the gap described above. This is because decode is memory-bound, a forward pass over k tokens costs roughly the same as a pass over one token. So instead of asking the large *target* model for one token at a time, we let a cheap *drafter* guess the next k tokens, then have the target check all k of them in a single forward pass, which works like a small prefill. We keep the longest prefix the target agrees with, plus one token from the target itself, so every step produces at least one token. With rejection sampling, the output distribution is exactly the same as the target model's. The method is lossless, unlike quantization.

The speedup depends on two things. First, how cheap the draft is. Second, how many draft tokens get accepted. The two main families of drafters trade these off differently.

## 2.1 Autoregressive drafter

{{< figure src="drafter-ar.png" link="drafter-ar.png" alt="An autoregressive drafter runs four passes in sequence, each producing one draft token (fox, jumps, over, a) that feeds the next pass. The target verifies all four in one pass, keeps fox, jumps and over, and replaces the rejected a with its own token, the." caption="Autoregressive drafter: one pass per draft token. Illustrative." >}}

the drafter generates the k tokens one at a time, each conditioned on the previous one. Drafts are high quality because every token sees the tokens before it, but drafting cost grows linearly with k, so draft lengths stay short. Popular methods include EAGLE-3 and MTP, often paired with tree-based verification to make up for the short drafts.

## 2.2 Parallel drafter

{{< figure src="drafter-parallel.png" link="drafter-parallel.png" alt="A parallel drafter produces all four draft tokens in a single pass, with no links between positions. The target verifies them in one pass with the same outcome: fox, jumps and over are kept and the target's token replaces a." caption="Parallel drafter: one pass for the whole block. Illustrative." >}}

the drafter predicts all k tokens in a single forward pass. Drafting cost stays roughly constant as k grows, but each position is predicted without seeing the actual tokens before it, so the acceptance rate drops at later positions in the block. Popular methods include Medusa, PARD, DART, and DFlash.

# 3. What's wrong with the prior methods?

## 3.1 Autoregressive drafters

{{< figure src="prior-ar.png" link="prior-ar.png" alt="Left: drafter forward passes per block, k for an autoregressive drafter and 1 for a parallel drafter. Right: a bar chart of acceptance at position 1, EAGLE-3 versus DFlash: 0.81 vs 0.88 on math and 0.53 vs 0.72 on chat." caption="Left: drafter passes per block. Right: acceptance at position 1 (DSpark paper, Sec. 4.3.1)." >}}

Each draft token needs its own drafter forward pass, so drafting latency grows linearly with the block length. To keep that latency low, these drafters use short blocks and small, shallow networks. Tree-based verification tries to make up for the short blocks by verifying several candidate paths at once, but that sends many more tokens to the target model and reduces overall serving throughput. Therefore, this create a degrade impact on the first draft position, because the target only accepts a prefix, a miss there throws away the whole block. At position 1, EAGLE-3, an autoregressive drafter, is accepted less often than DFlash, a parallel drafter that can afford a much deeper network by 0.81 vs. 0.88 on math, and 0.53 vs. 0.72 on chat.

## 3.2 Parallel drafters

{{< figure src="prior-parallel.png" link="prior-parallel.png" alt="Left: with a parallel drafter, of or no at position 1 and course or problem at position 2 can combine into nonsense pairs, while an autoregressive drafter only produces matching pairs. Right: a slope chart of conditional acceptance on chat, where DFlash falls from 0.72 to 0.63 and EAGLE-3 rises from 0.53 to 0.74, so the lines cross." caption="Left: the paper's example. Right: conditional acceptance on chat (DSpark paper, Sec. 4.3.1)." >}}

A parallel drafter predicts every position in the block in one forward pass, so no position can see which token was actually sampled before it. Instead, each position averages over all the tokens that might come before it. As a result, when the context allows several continuations, different positions can lock onto different ones. For example, if both "of course" and "no problem" fit, a parallel drafter may propose "of problem" or "no course". The paper calls this issue a *multi-modal collision*. These collisions cause *suffix decay*, so even when every earlier token in the block was accepted, each later position is accepted less often than the one before. Therefore, for a pure parallel drafter, each extra position in a longer block is worth less than the one before. The paper measures *conditional acceptance* at each position such as how often a token is accepted, given that every token before it was. DFlash's conditional acceptance falls along the block, from 0.87 to 0.78 on code and from 0.72 to 0.63 on chat. However, EAGLE-3 does the opposite, because each of its tokens sees the ones before it, it holds steady or even improves deeper into the block (from 0.53 to 0.74 on chat).

Put together, the two families fail in opposite places. Autoregressive drafters are weak at the first position, and parallel drafters get weaker at later ones.

## 3.3 Verification length

{{< figure src="prior-verify.png" link="prior-verify.png" alt="Left: with a fixed length of six draft tokens, a code request keeps five and a chat request keeps one. Right: at light load a target step has idle slots, so rejected tokens are cheap; at heavy load every slot is full, rejected tokens take room, and a fifth request waits." caption="Illustrative: acceptance varies by request, and rejected tokens cost more under heavy load." >}}

Parallel drafting makes long blocks cheap to produce, but verifying every drafted token is not cheap. The ideal verification length changes along two axes.
  - **Data:** acceptance depends on the request. Structured text such as code sustains much higher acceptance than open-ended chat, so any single length is too short for some requests and too long for others.
  - **System load:** at light load, verifying an extra token is nearly free. The GPU is waiting on memory anyway, so even a rejected token costs almost nothing. Under heavy load, every token sent for verification adds to the batch the target model processes in that step, and the step gets slower as that batch grows. A token that is likely to be rejected takes batch capacity that could have served another request.

Prior methods either verify a fixed number of tokens or use a static confidence threshold to decide how many to verify. Neither looks at the current load. That works under single-request assumptions but is suboptimal in a high-concurrency production system. Before DSpark, DeepSeek's production serving used MTP-1, a single draft token, because deploying a static multi-token drafter (MTP-3 or MTP-5) degraded aggregate throughput under high concurrency.

# 4. How DSpark works

## 4.1 Semi-autoregressive generation

## 4.2 Confidence-scheduled verification

### 4.2.1 Confidence Head

### 4.2.2 Hardware-Aware Prefix Scheduler

# 5. Results

## 5.1 Experimental Setup

## 5.2 Experimental Results

## 5.3 Experimental Analysis

1. Why can Parallel drafter outperform Autoregressive drafter?
2. A little Autoregressive drafter goes a long way
3. Verify Smater, not Longer: The role of Confidence Scheduling
4. For production results.
   1. Scalability:
   2. Hareware-Aware Prefix Scheduler:
   3. High Throughput with Low Latency:
   4. Reult from real traffic:
## 5.4 Limitations and Future Work

# 6. Demo

## 6.1 Demo with vLLM

## 6.2 Demo with Deepinfer

# 7 Takeaways

---

*Paper: [arXiv:2607.05147](https://arxiv.org/abs/2607.05147). All figures cited above are the authors'.*
