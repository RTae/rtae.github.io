+++
title = "DSpark: when speculative decoding starts caring about your batch"
date = "2026-08-03"
description = "A serving engineer's read of DSpark (arXiv:2607.05147) — how semi-autoregressive drafting reduces suffix decay, and why confidence-scheduled verification makes draft length a scheduling decision instead of a fixed hyperparameter."
tags = ["llm-inference", "llm-serving", "paper-review"]
+++

Speculative decoding has a production problem. Verifying a few extra draft tokens may help one request, but under heavy load those tokens can consume GPU capacity that could have served other requests. DSpark solves this by estimating the chance that each draft token will be accepted and checking how busy the GPU is. When the GPU has spare capacity, it verifies more tokens. When the system is busy, it verifies fewer, because extra tokens would slow the whole batch. The goal is to keep total throughput high, rather than using a fixed number for every request.

<!--more-->

## TLDR

DSpark combines a parallel drafting backbone with a tiny sequential Markov head to reduce suffix decay without paying for a full autoregressive drafter. It then uses calibrated token confidence and profiled GPU step times to choose how many draft tokens each request should verify under the current load. The authors report higher accepted lengths across four target models and substantial production throughput gains over MTP-1, although the production results have not been independently reproduced.

## 1. Why decoding is the bottleneck

{{< figure src="decode-bottleneck.png" link="decode-bottleneck.png" alt="Two panels. Prefill: the model weights are read from GPU memory once, all prompt tokens are computed together, and K,V for every prompt token is written to the KV cache, so compute stays busy. Decode: each step reads all the weights and the whole KV cache again, computes only the newest token, and writes its K,V back, so compute waits for the data most of each step." caption="Illustrative, not to scale: what moves between GPU memory and compute in each phase." >}}

LLM inference has two phases, prefill and decode. In prefill, the model processes the whole prompt in one forward pass, building the KV cache and producing the first output token. Every prompt token is processed in parallel, so each weight read from memory is reused across many tokens. For sufficiently large prompts, prefill is generally compute-bound. It spends most of its time computing rather than waiting for data.

Decode produces the remaining tokens one step at a time until the model emits an end-of-sequence token or reaches the output's maximum length. Each token depends on the previous one, so every step reads the model weights and KV cache from GPU memory to produce the next token. At small batches, there is little computation for each byte read, and the KV-cache reads grow with the output, so the GPU spends much of its time waiting for data. Therefore, in this phase it's usually domoninate end-to-end latency.

There are many ways to tackle this problem, including prefill/decode disaggregation (so long prefills don't stall decoding), quantization (fewer bytes to read), hardware with more memory bandwidth (such as NVIDIA Rubin), and speculative decoding (more tokens per read). This article focuses on speculative decoding.

## 2. What is speculative decoding?

Speculative decoding leverages this memory-bound regime. At small batches, a target-model forward pass that verifies k draft tokens can cost roughly the same as a one-token decode step because both read the weights once, although attention, KV-cache traffic, and batching eventually make longer verification more expensive. Instead of querying the large target model for one token at a time, a cheaper drafter proposes the next k tokens, and the target evaluates them in one forward pass, functioning similarly to a small prefill.

With greedy decoding, the system keeps the longest prefix matching the target, replaces the first mismatch with the target's token, or adds one target token if every draft matches. With sampled decoding, speculative rejection sampling accepts tokens probabilistically and draws from a correction distribution after a rejection; if every draft is accepted, it adds one target token. Both procedures preserve the target model's output distribution, and every round produces at least one token.

Token counts are easy to conflate. Here, *draft length* excludes the target's token. The paper's *verification length* counts scheduled draft tokens, while its production *verification budget* also includes one target token per request. A five-token draft therefore gives a maximum budget of six. *Accepted length* counts tokens emitted per round, including the target's token.

The speedup depends on two things. First, how cheap the draft is. Second, how many draft tokens get accepted. The two main families of drafters trade these off differently.

### 2.1 Autoregressive drafter

{{< figure src="drafter-ar.png" link="drafter-ar.png" alt="An autoregressive drafter runs four passes in sequence, each producing one draft token (fox, jumps, over, a) that feeds the next pass. The target verifies all four in one pass, keeps fox, jumps and over, and replaces the rejected a with its own token, the." caption="Autoregressive drafter: one pass per draft token. Illustrative." >}}

The drafter generates the k tokens one at a time, each conditioned on the previous one. Drafts are high quality because every token sees the tokens before it, but drafting cost grows linearly with k, so draft lengths stay short. Popular methods include EAGLE-3 and MTP, often paired with tree-based verification to make up for the short drafts.

### 2.2 Parallel drafter

{{< figure src="drafter-parallel.png" link="drafter-parallel.png" alt="A parallel drafter produces all four draft tokens in a single pass, with no links between positions. The target verifies them in one pass with the same outcome: fox, jumps and over are kept and the target's token replaces a." caption="Parallel drafter: one pass for the whole block. Illustrative." >}}

The drafter predicts all k tokens in a single forward pass. Drafting cost stays roughly constant as k grows, but each position is predicted without seeing the actual tokens before it, so the acceptance rate drops at later positions in the block. Popular methods include Medusa, PARD, DART, and DFlash.

## 3. What's wrong with the prior methods?

### 3.1 Autoregressive drafters

{{< figure src="prior-ar.png" link="prior-ar.png" alt="Left: drafter forward passes per block, k for an autoregressive drafter and 1 for a parallel drafter. Right: a bar chart of acceptance at position 1, EAGLE-3 versus DFlash: 0.81 vs 0.88 on math and 0.53 vs 0.72 on chat." caption="Left: drafter passes per block. Right: acceptance at position 1 (DSpark paper, Sec. 4.3.1)." >}}

Each draft token needs its own drafter forward pass, so drafting latency grows linearly with the block length. To keep that latency low, these drafters use short blocks and shallow networks, then use tree-based verification to compensate for the short blocks by verifying several candidate paths at once. This sends more tokens to the target model and can reduce aggregate serving throughput under concurrency. A shallow network is also less accurate, and this matters most at the first draft position. This is because the target only accepts a prefix, so if the first draft token is wrong, the whole block is thrown away. At the first position, EAGLE-3 is accepted less often than DFlash, a parallel drafter that can afford a much deeper network with 0.81 vs. 0.88 on math, and 0.53 vs. 0.72 on chat.

### 3.2 Parallel drafters

{{< figure src="prior-parallel.png" link="prior-parallel.png" alt="Left: with a parallel drafter, of or no at position 1 and course or problem at position 2 can combine into nonsense pairs, while an autoregressive drafter only produces matching pairs. Right: a slope chart of conditional acceptance on chat, where DFlash falls from 0.72 to 0.63 and EAGLE-3 rises from 0.53 to 0.74, so the lines cross." caption="Left: the paper's example. Right: conditional acceptance on chat (DSpark paper, Sec. 4.3.1)." >}}

A parallel drafter predicts every position in the block in one forward pass, so no position can see which token was actually sampled before it. Instead, each position must account for all the tokens that might come before it. As a result, when the context allows several continuations, different positions can lock onto different ones. For example, if both "of course" and "no problem" fit, a parallel drafter may propose "of problem" or "no course". The paper calls this issue a *multi-modal collision*, and it causes *suffix decay*. Conditional acceptance tends to fall at later positions, even when every earlier token in the block was accepted. Therefore, for a pure parallel drafter, each extra position in a longer block tends to be worth less than the one before.

The paper measures *conditional acceptance* at each position, that is, how often a token is accepted given that every token before it was. DFlash starts high but falls along the block, from 0.87 to 0.78 on code and from 0.72 to 0.63 on chat. However, EAGLE-3 does the opposite. It starts lower (0.53 on chat, the same first-position number as in 3.1), but because each of its tokens sees the ones before it, it holds steady or even improves deeper into the block (up to 0.74 on chat).

### 3.3 Verification length

{{< figure src="prior-verify.png" link="prior-verify.png" alt="Left: with a fixed length of six draft tokens, a code request keeps five and a chat request keeps one. Right: at light load a target step has idle slots, so rejected tokens are cheap; at heavy load every slot is full, rejected tokens take room, and a fifth request waits." caption="Illustrative: acceptance varies by request, and rejected tokens cost more under heavy load." >}}

Parallel drafting makes long blocks cheap to produce, but verifying every drafted token is not cheap. The ideal verification length changes along two axes.

- **Data:** acceptance depends on the request. Structured text such as code sustains much higher acceptance than open-ended chat, so any single length is too short for some requests and too long for others.
- **System load:** at light load, verifying an extra token can be nearly free when the GPU has idle capacity and is waiting on memory anyway, so even a rejected token costs almost nothing. Under heavy load, every token sent for verification adds to the batch the target model processes in that step, and the step gets slower as that batch grows. A token that is likely to be rejected takes batch capacity that could have served another request.

Put together, the two families fail in opposite places. Autoregressive drafters are weak at the first position, and parallel drafters get weaker at later ones. The first position matters most, because a miss there throws away the whole block. In the paper's main Qwen comparisons, this advantage lets parallel drafters come out ahead despite their decay; the Gemma4-12B results in Section 5.2 are an exception.

DSpark targets this gap by retaining the parallel drafter's strong first position while reducing decay at later positions.

Prior methods also either verify a fixed number of tokens or use a static confidence threshold. Neither accounts for current system load. That can work under single-request assumptions, but it is suboptimal in a high-concurrency production system. Before DSpark, DeepSeek's production serving used MTP-1, a single draft token, because deploying a static multi-token drafter (MTP-3 or MTP-5) degraded aggregate throughput under high concurrency.

## 4. How DSpark works

DSpark addresses the problems from Section 3 with two parts. The first is *semi-autoregressive generation*, which keeps the strong first position of a parallel drafter and reduces suffix decay after it. The second is *confidence-scheduled verification*, which decides how many draft tokens are worth sending to the target, based on how confident the drafter is and how busy the GPU is.

### 4.1 Semi-autoregressive generation

{{< figure src="semi-ar.png" link="semi-ar.png" alt="The input is an anchor token followed by mask tokens. A parallel backbone runs one heavy forward pass over the whole block and gives a base guess for each position, such as of or no for position 1 and course or problem for position 2. Then a tiny Markov head runs left to right, adding a bias from the previously sampled token at each position, so after of it picks course." caption="Illustrative: one heavy parallel pass, then a tiny Markov head left to right (paper, Sec. 3.1)." >}}

DSpark splits drafting into two stages. In the *parallel stage*, a heavy backbone (DFlash in the paper) runs one forward pass over the whole block and gives a base guess for every position. This is where most of the drafting work happens, so drafting time still stays roughly constant as the block grows. In the *sequential stage*, a tiny *Markov head* walks through the block from left to right. At each position, it adds a bias to the base guess that depends on the token just sampled, then samples the next token. For example, if both "of course" and "no problem" fit, the backbone alone guesses "of / no" and "course / problem". Once position 1 samples "of", the Markov head boosts "course" and suppresses "problem" at position 2, so the pair stays coherent.

The Markov head has to be tiny because it runs once per position, one after another. The paper builds it as a low-rank lookup (rank 256 by default), so it costs almost nothing next to the backbone. The paper also tests an *RNN head* that remembers the whole prefix instead of only the previous token, but it only gives small extra gains. Therefore, the Markov head is the default.

### 4.2 Confidence-scheduled verification

{{< figure src="verify-round.png" link="verify-round.png" alt="One decoding round. After the context The quick brown fox, DSpark drafts jumps, over, a and lazy, each with a confidence. The prefix scheduler keeps jumps over a and drops lazy. The target verifies only the three kept tokens in one pass, accepts jumps and over, rejects a, and adds its own token the." caption="Illustrative: the scheduler keeps the confident prefix, and the target verifies only that (paper, Fig. 1)." >}}

Even with better drafts, verifying every drafted token is wasteful, because the last few tokens of a block are often rejected. DSpark adds two pieces to decide how much of each block to verify. A *confidence head* (4.2.1) gives each draft token a confidence, and a *hardware-aware prefix scheduler* (4.2.2) uses those confidences, together with the current GPU load, to choose how many tokens of each request to send to the target. In the figure, the scheduler keeps "jumps over a" and drops "lazy", so the target verifies only three tokens. The target then works like normal speculative decoding. It accepts "jumps over", rejects "a", and adds its own token "the". The paper shows that this scheduling keeps the output distribution exactly the same as the target model's.

#### 4.2.1 Confidence Head

{{< figure src="confidence-head.png" link="confidence-head.png" alt="Left: the backbone hidden state and the embedding of the previous draft token go through a linear layer and a sigmoid to give a confidence c_k. Right: with confidences 0.95, 0.90, 0.70 and 0.50, the prefix survival bars fall to 0.95, 0.85, 0.60 and 0.30. Below: Sequential Temperature Scaling calibrates the scores." caption="Bar values are illustrative; calibration numbers from the paper (Sec. 4.3.3)." >}}

The confidence head is a small linear layer with a sigmoid on top. For each draft position k, it looks at the backbone's hidden state and the embedding of the previous draft token, and outputs a score c_k. This is the chance that token k is accepted, given that every token before it was accepted. During training, it learns to match the real acceptance rate, which the paper computes from how far the draft distribution is from the target's.

Because the target only accepts a prefix, what the scheduler really needs is the *prefix survival*, that is, the chance that every token up to position k is accepted. This is simply c_1 × c_2 × … × c_k, so it is non-increasing along the block. The paper also points out that raw neural confidence scores are usually too high. Therefore, DSpark calibrates them with *Sequential Temperature Scaling (STS)*, which fits one temperature per position, from left to right, on held-out data. The paper describes STS as an order-preserving calibration of the confidence head's relative token rankings. The raw head already ranks tokens well (ROC-AUC 0.81–0.90), and STS brings the calibration error down from 3–8% to about 1% (Sec. 4.3.3).

#### 4.2.2 Hardware-Aware Prefix Scheduler

{{< figure src="prefix-scheduler.png" link="prefix-scheduler.png" alt="The same three requests, code, math and chat, each with five draft tokens and their prefix survival values. At light load the scheduler verifies 13 of the 15 tokens; at heavy load it verifies only 9, keeping the most confident ones. Below, the three steps the scheduler follows to choose the cut-off." caption="Illustrative survival values; production numbers from the paper (Sec. 5.4)." >}}

The scheduler looks at every request in the batch at once. It sorts all candidate draft tokens by prefix survival, then admits them one by one, from the most likely to be accepted to the least. Each admitted token adds expected accepted tokens, but it also makes the target's step bigger and slower. The step time comes from a table that is profiled once when the engine starts, and the scheduler stops at the batch size where the expected tokens per second is the highest. As a result, at light load extra tokens are almost free, so it verifies long prefixes, and at heavy load every extra token slows the whole batch, so only confident tokens make it. Because survival is non-increasing along each block, sorting by survival keeps each request's verified tokens as a prefix automatically.

In production, the paper makes two practical changes (Sec. 5.2). First, the real step time does not grow smoothly with batch size but jumps in steps, so the scheduler searches the whole table instead of stopping at the first drop. Second, the serving engine needs to know the next batch size before the current step finishes, so the scheduler uses confidences from two steps earlier to decide how many tokens fit, while still ranking the current tokens by their latest scores. The paper shows that this design stays lossless.

## 5. Results

*All results in this section are reported by the DSpark authors in the paper; I have not reproduced them. Section and table numbers in parentheses refer to the paper.*

### 5.1 Experimental Setup

| Setting | From the paper (Sec. 4.1) |
|---|---|
| Target models | Qwen3-4B, Qwen3-8B, Qwen3-14B, Gemma4-12B |
| Drafters | EAGLE-3 (autoregressive, 1 layer), DFlash (parallel, 5 layers), DSpark (5-layer backbone + Markov head) |
| Block size | 7 for every drafter |
| Training data | 1.3M prompts from Open-PerfectBlend (39.4% math, 38.9% code, 17.6% chat, 4.1% instruction following), with responses regenerated by each target model, 10 epochs |
| Benchmarks | Math: GSM8K, MATH500, AIME25. Code: MBPP, HumanEval, LiveCodeBench. Chat: MT-Bench, Alpaca, Arena-Hard |
| Decoding | Standard speculative decoding, temperature 1.0, chain drafting, non-thinking mode |
| Metric | Accepted length per round, including the target's own token (higher is better) |

All three drafters are retrained with the same framework and the same data. The confidence scheduler is turned off for these offline tests, so every drafter proposes a fixed block. This isolates the quality of the drafts.

### 5.2 Experimental Results

| Target | Drafter | GSM8K | MATH | AIME25 | MBPP | HumanEval | LCB | MT-Bench | Alpaca | Arena-Hard |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | EAGLE-3 | 5.14 | 4.62 | 3.92 | 3.69 | 4.16 | 3.77 | 2.39 | 2.26 | 2.55 |
| | DFlash | 5.40 | 4.85 | 4.15 | 4.40 | 4.74 | 4.18 | 3.07 | 2.96 | 2.83 |
| | **DSpark** | **6.11** | **5.70** | **4.89** | **5.13** | **5.38** | **4.86** | **3.64** | **3.54** | **3.29** |
| Qwen3-8B | EAGLE-3 | 5.30 | 4.77 | 3.91 | 3.96 | 4.33 | 4.17 | 2.66 | 2.54 | 2.54 |
| | DFlash | 5.33 | 4.91 | 4.07 | 4.36 | 4.64 | 4.39 | 3.11 | 2.98 | 2.81 |
| | **DSpark** | **6.17** | **5.78** | **5.01** | **5.16** | **5.52** | **5.17** | **3.72** | **3.58** | **3.21** |
| Qwen3-14B | EAGLE-3 | 5.24 | 4.60 | 3.71 | 3.81 | 4.14 | 4.01 | 2.62 | 2.47 | 2.48 |
| | DFlash | 5.41 | 4.84 | 3.98 | 4.44 | 4.59 | 4.33 | 3.10 | 2.94 | 2.72 |
| | **DSpark** | **6.21** | **5.74** | **4.94** | **5.26** | **5.43** | **5.02** | **3.70** | **3.58** | **3.13** |
| Gemma4-12B | EAGLE-3 | 5.87 | 5.46 | 4.83 | 4.72 | 5.37 | 4.16 | 3.19 | 3.06 | 2.72 |
| | DFlash | 5.45 | 5.04 | 4.22 | 4.39 | 4.95 | 3.70 | 2.98 | 2.84 | 2.59 |
| | **DSpark** | **6.05** | **5.78** | **5.12** | **5.11** | **5.64** | **4.51** | **3.49** | **3.35** | **2.92** |

*Accepted length per round, from the paper's Table 1. LCB is LiveCodeBench.*

DSpark has the longest accepted length on every benchmark and every target model. On Qwen3-4B, 8B and 14B, it improves the average accepted length over EAGLE-3 by 30.9%, 26.7% and 30.0%, and over DFlash by 16.3%, 18.4% and 18.3% (Sec. 4.2). The gains also hold on Gemma4-12B, which is a different model family.

Two more things stand out in the table. First, acceptance depends a lot on the domain. For Qwen3-4B, DSpark keeps 5.57 tokens per round on math and 5.12 on code, but only 3.49 on chat. This variance is exactly why a fixed verification length can waste compute on some requests and leave useful capacity unused on others, motivating the confidence scheduler. Second, from my reading of the table, EAGLE-3 beats DFlash on every Gemma4-12B benchmark, so a pure parallel drafter is not always the stronger baseline, but DSpark still beats both.

### 5.3 Experimental Analysis

#### 1. Position-wise acceptance confirms the design (Sec. 4.3.1)

The position-wise measurements support the mechanism described in Sections 3.1–3.2. DSpark retains the deep parallel backbone's strong start (0.93 conditional acceptance on math at position 1), while the Markov head keeps conditional acceptance high and stable through the rest of the block. This is the intended combination of parallel first-token capacity and sequential suffix coherence.

#### 2. A little autoregression goes a long way (Sec. 4.3.2, Qwen3-4B)

- **Depth:** with a block size of 7, DSpark gets better with every extra layer from 1 to 5, and the biggest jump is from 1 to 2 layers. A 2-layer DSpark already beats the 5-layer DFlash in every domain.
- **Block length:** DSpark beats DFlash at every block length the paper tests, and the gap grows with the block. With 7 proposed tokens, DSpark accepts 16% more on math, 15% more on code and 18% more on chat. With 15 proposed tokens, the gains grow to 30%, 26% and 22%. The RNN head adds only small extra gains, mainly at longer blocks.
- **Cost:** at a batch size of 128, growing the draft from 4 to 16 tokens adds only 0.2% to 1.3% to the full round latency compared with DFlash, while the accepted length improves by up to 30%.

#### 3. Verify smarter, not longer: the role of confidence scheduling (Sec. 4.3.3, Qwen3-4B)

The confidence-head ablation uses a static threshold to drop low-confidence draft tokens. As the threshold rises, the dropped tokens are mostly ones that would have been rejected anyway, so the acceptance rate among retained tokens increases. It rises from 45.7% to 95.7% on chat, from 76.9% to 92.5% on math, and from 67.6% to 92.0% on code. Chat gains the most because its later tokens are rejected the most. This does not by itself establish a speedup. Aggressive pruning can lower accepted length and require more rounds. A static threshold also ignores system load, which is why production uses the hardware-aware scheduler instead. Section 4.2.1 summarizes the separate calibration results.

#### 4. Production results (Sec. 5, DeepSeek-V4-Flash and V4-Pro previews, DSpark with up to 5 draft tokens against the previous MTP-1 setup)

- **Scalability:** the production drafter is three MoE layers with a sliding window attention of 128, plus the Markov head. Training it needs the target model's output distributions, so the paper adds two tricks. Workers exchange hidden states instead of full-vocabulary logits. Training also packs short blocks around sampled anchor positions, keeping the draft model's cost from growing with the target's long contexts.
- **Hardware-aware prefix scheduler:** the production adaptations from Section 4.2.2 let the scheduler fit the serving engine's zero-overhead scheduling and CUDA graphs while remaining lossless.
- **High throughput with low latency:** in DeepSeek's deployment, the batch stays well below the point where the GPU's compute is saturated, because of per-request KV cache limits and the available traffic. Therefore, higher total throughput and faster per-user speed go together instead of trading off. To verify a different number of tokens for each request in one batch, the kernels flatten all tokens and track which request each one belongs to. On DeepSeek-V4, only the index-attention and compress kernels needed changes.
- **Results from real traffic:** at moderate load (fewer than 200 concurrent requests on V4-Flash and 150 on V4-Pro), the average target verification budget is about 4–6 tokens per request, including the target's own token, against MTP-1's fixed budget of 2 (one draft token plus one target token). The budget shrinks as load grows. DSpark replaced MTP-1 in production two weeks after the DeepSeek-V4 preview release.

| Engine | Per-user speed target (SLA) | Throughput vs. MTP-1 | Per-user speed at matched throughput |
|---|---|---|---|
| V4-Flash | 80 tok/s/user | +51% | 60–85% faster |
| V4-Flash | 120 tok/s/user | +661% (nominal) | |
| V4-Pro | 35 tok/s/user | +52% | 57–78% faster |
| V4-Pro | 50 tok/s/user | +406% (nominal) | |

*From the paper's Sec. 5.4.* At the stricter speed targets, MTP-1 can only serve a very small batch, so the paper reads the +661% and +406% as DSpark extending what is reachable, not as typical speedups.

### 5.4 Limitations and Future Work

According to the paper, the scheduler cuts wasted verification on the target, but the drafter still pays a fixed cost to generate the full block with its parallel backbone. For hard requests with low acceptance, that drafting work is wasted. The authors suggest difficulty-aware early exit inside the drafter, so these requests can skip generating the full block.

From my reading, there are a few more things to keep in mind.

- **Hard to reproduce:** the production numbers come from DeepSeek's own engine and traffic, and they are compared only with MTP-1, so they cannot be checked outside DeepSeek or compared directly with other serving systems.
- **Offline results measure acceptance, not speed:** the main table reports accepted length with the scheduler turned off, so it shows draft quality rather than end-to-end speedup.
- **Integration cost:** the scheduler needs a step-time table profiled for each engine and kernels that can verify a different number of tokens per request, which is real work for any engine that wants to adopt it.

## 6. Takeaways

- **Draft architecture and verification policy are separate problems.** DSpark uses a heavy parallel backbone for cheap long drafts and a tiny Markov head for coherence, then independently decides how much of each draft deserves target compute.
- **Optimize system throughput under a latency target, not acceptance alone.** Higher acceptance or accepted length does not guarantee faster serving; verification cost and batch load determine the benefit.
- **The evidence is promising but uneven.** Offline tables show author-reported accepted-length gains across four targets, while the production gains come from DeepSeek's own engine and traffic and have not been independently reproduced.
- **Adoption is systems work.** Calibrated confidence, engine-specific step-time profiles, asynchronous scheduling, and variable-length verification kernels are all required.

---

*Paper: [arXiv:2607.05147](https://arxiv.org/abs/2607.05147). All figures are illustrative redraws; paper-specific examples and numerical results are based on the DSpark paper.*
