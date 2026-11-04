+++
title = "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"
date = "2026-11-01"
description = "A serving engineer's read of DSpark (arXiv:2607.05147) — how semi-autoregressive drafting reduces suffix decay, and why confidence-scheduled verification makes draft length a scheduling decision instead of a fixed hyperparameter."
tags = ["llm-inference", "llm-serving", "paper-review"]
+++

In speculative decoding, by verifying a few extra draft tokens may help one request, but under heavy load those tokens can consume GPU capacity that could have served other requests. DSpark solves this by estimating the chance that each draft token will be accepted and checking how busy the GPU is. When the GPU has capacity, it verifies more tokens. If not, it verifies fewer, because extra tokens would slow the whole batch. As a result, this method help to improve total throughput high.

<!--more-->

## TLDR

{{< figure src="dspark-overview.gif" link="dspark-overview.gif" alt="Animation of one DSpark decoding round. After the context Can you help? Yes, one parallel backbone pass guesses every draft position, with of or no at position 1 and course or problem at position 2. Picked separately, the guesses can clash into of problem, which the target rejects, wasting the rest of the block. A tiny Markov head instead picks left to right, so of steers position 2 to course and rules out problem, giving of course I can while each round runs only 0.2% to 1.3% slower than the parallel backbone alone. The confidence head gives prefix survival scores of 0.95, 0.86, 0.60, and 0.30. Under light GPU load all four tokens are kept, while under heavy load only of and course are kept. The target verifies them in one pass, accepts both, and adds its own token, an exclamation mark." caption="One DSpark round: a parallel pass guesses the block, a tiny Markov head keeps it coherent, and the current GPU load decides how many draft tokens the target verifies" >}}

DSpark combines a parallel drafting backbone with a tiny sequential Markov head to reduce suffix decay without paying for a full autoregressive drafter. It then uses calibrated token confidence and profiled GPU step times to choose how many draft tokens each request should verify under the current load. The authors report higher accepted lengths across four target models and substantial production throughput gains over MTP-1, although the production results have not been independently reproduced.

## 1. Why decoding is the bottleneck

{{< figure src="decode-bottleneck.png" link="decode-bottleneck.png" alt="Top: in prefill, model weights are read once from HBM and shared across all prompt tokens in one compute-bound pass. Bottom: in decode, every step re-reads the weights and the growing KV cache to produce a single token, leaving compute waiting on memory." caption="Prefill reuses one weight read across the whole prompt, while decode re-reads weights and the KV cache for each new token" >}}

LLM inference has two phases, prefill and decode. In prefill, shown at the top of the figure, the model processes the whole prompt in one forward pass, building the KV cache and producing the first output token. Every prompt token is processed in parallel, so each weight read from memory is reused across many tokens. For sufficiently large prompts, prefill is generally compute-bound. It spends most of its time computing rather than waiting for data.

Decode, shown at the bottom of the figure, produces the remaining tokens one step at a time until the model emits an end-of-sequence token or reaches the output's maximum length. Each token depends on the previous one, so every step reads the model weights and KV cache from GPU memory to produce the next token. At small batches, there is little computation for each byte read, and the KV-cache reads grow with the output, so the GPU spends much of its time waiting for data. As a result, this phase usually dominates end-to-end latency.

There are many ways to tackle this problem, including prefill/decode disaggregation (so long prefills don't stall decoding), quantization (fewer bytes to read), hardware with more memory bandwidth (such as NVIDIA Rubin), and speculative decoding (more tokens per read). This article focuses on speculative decoding.

## 2. What is speculative decoding?

Speculative decoding takes advantage of the idle compute during decode. Modern GPUs have plenty of compute, so a large model can verify several tokens from a small model in one forward pass for nearly the same cost as generating a single token. A smaller drafter proposes the next k tokens, called a *draft block*. The large model, called the *target model*, then checks them all in one pass. If a draft token is rejected, the system keeps the accepted prefix and uses a token from the large model instead. With sampling, acceptance is probabilistic, but the output still follows the same distribution as the large model on its own. Every round generates at least one token.

The speedup depends on two things. First, how cheap the draft is. Second, how many draft tokens get accepted. The two main families of drafters trade these off differently.

### 2.1 Autoregressive drafter

{{< figure src="drafter-ar.png" link="drafter-ar.png" alt="An autoregressive drafter runs four sequential passes (fox, jumps, over, a), and the target verifies all four in one pass, keeping fox, jumps, over and replacing the rejected a with the." caption="Autoregressive drafter: four sequential draft passes followed by one target verification pass (green = accepted draft, blue = target token)" >}}

The drafter generates the k tokens one at a time, each conditioned on the previous one. Drafts are high quality because every token sees the tokens before it, but drafting cost grows linearly with k, so draft lengths stay short. Popular methods include EAGLE-3 and MTP, often paired with tree-based verification to make up for the short drafts.

### 2.2 Parallel drafter

{{< figure src="drafter-parallel.png" link="drafter-parallel.png" alt="A parallel drafter predicts all four positions in one pass, and the target verifies them in one pass, keeping fox, jumps, over and replacing a with the." caption="Parallel drafter: one draft pass for the whole block followed by one target verification pass" >}}

The drafter predicts all k tokens in a single forward pass. Drafting cost stays roughly constant as k grows, but each position is predicted without seeing the actual tokens before it, so the acceptance rate drops at later positions in the block. Popular methods include Medusa, PARD, DART, and DFlash.

## 3. What's wrong with the prior methods?

### 3.1 Autoregressive drafters

{{< figure src="prior-ar.png" link="prior-ar.png" alt="Left: drafter passes grow linearly with block length k for an autoregressive drafter while staying at 1 for a parallel drafter. Right: first-position acceptance is 0.81 vs. 0.88 on math and 0.53 vs. 0.72 on chat for EAGLE-3 vs. DFlash." caption="Left: drafter passes per block. Right: acceptance at position 1, EAGLE-3 vs. DFlash" >}}

As shown on the left of the figure, each draft token needs its own drafter forward pass, so drafting latency grows linearly with the block length. To keep that latency low, these drafters use short blocks and shallow networks, then use tree-based verification to compensate for the short blocks by verifying several candidate paths at once. This sends more tokens to the target model and can reduce aggregate serving throughput under concurrency. A shallow network is also less accurate, and this matters most at the first draft position. The target only accepts a prefix, so if the first draft token is wrong, the whole block is thrown away. As shown on the right, EAGLE-3 is accepted less often at the first position than DFlash, a parallel drafter that can afford a much deeper network, with 0.81 vs. 0.88 on math and 0.53 vs. 0.72 on chat.

### 3.2 Parallel drafters

{{< figure src="prior-parallel.png" link="prior-parallel.png" alt="Left: a parallel drafter picks each position independently and can mix of course and no problem into of problem or no course. Right: conditional acceptance on chat falls from 0.72 to 0.63 for DFlash while rising from 0.53 to 0.74 for EAGLE-3." caption="Left: independent position picks can collide. Right: conditional acceptance on chat" >}}

A parallel drafter predicts every position in the block in one forward pass, so no position can see which token was actually sampled before it. Instead, each position must account for all the tokens that might come before it. As a result, when the context allows several continuations, different positions can lock onto different ones. For example, as shown on the left of the figure, if both "of course" and "no problem" fit, a parallel drafter may propose "of problem" or "no course". The paper calls this issue a *multi-modal collision*, and it causes *suffix decay*. The chance that a token is accepted tends to fall at later positions, even when every earlier token in the block was accepted. Therefore, for a pure parallel drafter, each extra position in a longer block tends to be worth less than the one before.

The right panel plots *conditional acceptance* at each position, that is, how often a token is accepted given that every token before it was. DFlash starts high but falls along the block, from 0.87 to 0.78 on code and from 0.72 to 0.63 on chat. However, EAGLE-3 does the opposite. It starts lower (0.53 on chat, the same first-position number as in 3.1), but because each of its tokens sees the ones before it, it holds steady or even improves deeper into the block (up to 0.74 on chat).

### 3.3 Verification length

{{< figure src="prior-verify.png" link="prior-verify.png" alt="Left: with six draft tokens each, a code request keeps five while a chat request keeps one. Right: one target step holds three requests. At light load, two requests leave a row of idle slots, so rejected tokens cost little. At heavy load, three requests fill every slot, mostly with rejected or discarded tokens, and a fourth request has to wait." caption="Left: acceptance varies by request type. Right: rejected tokens are cheap at light load but block other requests at heavy load" >}}

Parallel drafting makes long blocks cheap to produce, but verifying every drafted token is not cheap. The ideal verification length changes along two axes.

- **Data:** on the left of the figure, acceptance depends on the request. Structured text such as code sustains much higher acceptance than open-ended chat, so any single length is too short for some requests and too long for others.
- **System load:** on the right of the figure, verifying an extra token at light load can be nearly free when the GPU has idle capacity and is waiting on memory anyway, so even a rejected token costs almost nothing. Under heavy load, every token sent for verification adds to the batch the target model processes in that step, and the step gets slower as that batch grows. A token that is likely to be rejected takes batch capacity that could have served another request.

Put together, the two methods have a opposite disadvantage. Autoregressive drafters are weak at the first position, and parallel drafters get weaker at later ones. The first position matters most since a miss there throws away the whole block. DSpark targets this gap by retaining the parallel drafter's strong first position while reducing decay at later positions.

Also, prior methods also either verify a fixed number of tokens or use a static confidence threshold. Neither accounts for current system load that can work under single-request assumptions, but in a high-concurrency production system, there are many issue that need to addresses. Before DSpark, DeepSeek's production serving used MTP-1, a single draft token, because deploying a static multi-token drafter (MTP-3 or MTP-5) degraded aggregate throughput under high concurrency.

## 4. How DSpark works

DSpark addresses the problems from Section 3 with two parts. The first is *semi-autoregressive generation*, which keeps the strong first position of a parallel drafter and reduces suffix decay after it. The second is *confidence-scheduled verification*, which decides how many draft tokens are worth sending to the target based on how confident the drafter is and how busy the GPU is.

### 4.1 Semi-autoregressive generation

{{< figure src="semi-ar.png" link="semi-ar.png" alt="An anchor token and mask tokens pass through one parallel backbone to produce base guesses, then a tiny rank-256 Markov head runs left to right so sampling of at position 1 boosts course at position 2." caption="One heavy parallel pass produces base guesses for the whole block, and a tiny Markov head refines them left to right" >}}

DSpark splits drafting into two stages, as shown in the figure. In the *parallel stage*, a heavy backbone (DFlash in the paper) runs one forward pass over the whole block and gives a base guess for every position. This is where most of the drafting work happens, so drafting time stays roughly constant as the block grows. Then, in the *sequential stage*, a tiny *Markov head* walks through the block from left to right. At each position, it adds a bias to the base guess that depends on the token just sampled, then samples the next token. For example, if both "of course" and "no problem" fit, the backbone alone guesses "of / no" and "course / problem". Once position 1 samples "of", the Markov head boosts "course" and suppresses "problem" at position 2, so the pair stays coherent. In short, this acts like a parallel drafter for speed and uses a small sequential head to keep adjacent words coherent without sacrificing that speed.

The Markov head has to be tiny because it runs once per position, one after another, using a low-rank lookup table (rank 256 by default). The paper also tests an *RNN head* that remembers the whole prefix instead of only the previous token, but it only gives small extra gains.

### 4.2 Confidence-scheduled verification

{{< figure src="verify-round.png" link="verify-round.png" alt="After the context The quick brown fox, DSpark drafts jumps, over, a, and lazy with confidences 0.95, 0.90, 0.70, and 0.50. The scheduler drops lazy, and the target verifies the first three tokens in one pass, accepting jumps and over, rejecting a, and adding its own token the." caption="One decoding round: the scheduler drops the low-confidence tail token before the target pass" >}}

Even with better drafts, verifying every drafted token is wasteful because the last few tokens of a block are often rejected. DSpark adds two pieces to decide how much of each block to verify. A *confidence head* gives each draft token a confidence score, and a *hardware-aware prefix scheduler* uses those scores together with the current GPU load to choose how many tokens from each request to send to the target. In the figure, the scheduler keeps "jumps over a" and drops "lazy", so the target verifies only three tokens. The target then works like normal speculative decoding. It accepts "jumps over", rejects "a", and adds its own token "the". The paper shows that this scheduling keeps the output distribution exactly the same as the target model's.

#### 4.2.1 Confidence Head

{{< figure src="confidence-head.png" link="confidence-head.png" alt="Step 1: backbone state h_k and previous token embedding pass through a linear layer and sigmoid to produce raw c_k. Step 2: Sequential Temperature Scaling divides the logit by a position-specific temperature T_k, lowering an overconfident raw score of 0.76 to the true rate of 0.70. Step 3: calibrated c_k values 0.95, 0.90, 0.70, and 0.50 multiply into falling prefix survival bars of 0.95, 0.86, 0.60, and 0.30." caption="From per-token confidence to prefix survival: (1) computing raw c_k, (2) calibrating with STS, and (3) multiplying into prefix survival a_k" >}}

As shown in Step 1 of the figure, the confidence head is a small linear layer with a sigmoid on top. For each draft position k, it looks at the backbone's hidden state and the embedding of the previous draft token, and outputs a score c_k. This is the chance that token k is accepted, given that every token before it was accepted. During training, it learns to match the real acceptance rate, which the paper computes from how far the draft distribution is from the target's.

Raw neural network scores are usually overconfident, so DSpark calibrates them first. As shown in Step 2, it uses *Sequential Temperature Scaling (STS)*, fitting one temperature per position from left to right on held-out data. This rescales the scores to match real acceptance rates without changing their order, reducing the calibration error from 3–8% to about 1%. Each calibrated c_k still covers only one position, while the target model accepts draft tokens as a continuous prefix. In Step 3, DSpark multiplies the scores together into a *prefix survival* probability, c_1 × c_2 × … × c_k, which gives the chance that every token up to position k is accepted and never increases along the block.

#### 4.2.2 Hardware-Aware Prefix Scheduler

{{< figure src="prefix-scheduler.png" link="prefix-scheduler.png" alt="Step 1: three requests, code, math, and chat, each with five draft tokens and falling prefix survival. Step 2: pooling and sorting all 15 candidate tokens against profiled step time, expected tok/s peaks at 9 tokens under heavy load and 13 tokens under light load. Step 3: the prefixes kept under light load, 13 of 15, and heavy load, 9 of 15." caption="Hardware-aware prefix scheduler: (1) per-request prefix survival, (2) global ranking and step-time cutoff, and (3) selected prefixes under light vs. heavy load" >}}

With the survival scores ready, the scheduler decides how many draft tokens to send to the target model across the whole batch. Each extra draft token can add accepted output, but it also makes the target model's batch larger and slower to run. To balance the two, the scheduler pools the draft tokens from every request, ranks them from highest to lowest survival, and adds them one at a time. A later token in a request never has a higher survival score than an earlier one, so the ranking always picks tokens from the start of each request's draft. For each possible batch size, the scheduler looks up the step time in a table profiled at startup and picks the size that gives the most expected accepted tokens per second. In the figure, extra tokens barely slow the step under light load, so the scheduler verifies 13 of the 15 draft tokens. Under heavy load, every extra token slows the whole batch, so only the 9 tokens with the highest survival make the cut.

{{< figure src="scheduler-production.png" link="scheduler-production.png" alt="Left: GPU step time rises in plateaus with sharp jumps at tile boundaries, so expected tok/s dips before reaching its true peak. Early stopping picks the first local peak, while a full-table search finds the true one. Right: while Step t-1 runs on the GPU, the scheduler uses Step t-2 confidences to fix the batch budget B* early, then fills those slots with the latest scores after Step t-1 finishes." caption="Two production adjustments: searching the full step-time table past tile-boundary dips, and using two-step lookback to size the batch without stalling the GPU pipeline" >}}

Running this in a production serving engine takes two small adjustments, shown in the figure above. The first, on the left, deals with the shape of the step-time curve. GPU step time does not rise smoothly as the batch grows. It stays flat for a while and then jumps when a hardware tile fills up, which creates small dips in expected tokens per second. Instead of stopping at the first dip, the scheduler checks the whole step-time table to find the true peak. The second, on the right, deals with timing. The engine prepares the next batch while the GPU is still running the current one, so it cannot wait for the newest confidence scores to choose the batch size. DSpark uses the scores from two steps earlier to decide how many tokens fit in the batch, then fills those slots with the latest scores once they arrive.

## 5. Results

*All results in this section are reported by the DSpark paper. I have not reproduced them.*

### 5.1 Experimental Setup

| Setting | From the paper (Sec. 4.1) |
| --- | --- |
| Target models | Qwen3-4B, Qwen3-8B, Qwen3-14B, Gemma4-12B |
| Drafters | EAGLE-3 (autoregressive, 1 layer), DFlash (parallel, 5 layers), DSpark (5-layer backbone + Markov head) |
| Block size | 7 for every drafter |
| Training data | 1.3M prompts from Open-PerfectBlend (39.4% math, 38.9% code, 17.6% chat, 4.1% instruction following), with responses regenerated by each target model, 10 epochs |
| Benchmarks | Math: GSM8K, MATH500, AIME25. Code: MBPP, HumanEval, LiveCodeBench. Chat: MT-Bench, Alpaca, Arena-Hard |
| Decoding | Standard speculative decoding, temperature 1.0, chain drafting, non-thinking mode |
| Metric | Accepted length per round, including the target's own token (higher is better) |

All three drafters are trained with the same code and data in DeepSpec. The confidence scheduler is turned off for these offline tests so every drafter proposes a fixed 7-token block, which isolates draft quality from scheduling.

### 5.2 Experimental Results

| Target | Drafter | GSM8K | MATH | AIME25 | MBPP | HumanEval | LCB | MT-Bench | Alpaca | Arena-Hard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
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

DSpark reaches the longest accepted length on every benchmark and every target model. On Qwen3-4B, 8B, and 14B, it accepts 30.9%, 26.7%, and 30.0% more tokens per round than EAGLE-3, and 16.3%, 18.4%, and 18.3% more than DFlash. The same lead holds on Gemma4-12B, showing that the design carries over to another model family.

Two patterns in the table stand out. First, acceptance varies sharply by task. On Qwen3-4B, DSpark keeps 5.57 tokens per round on math and 5.12 on code, compared with 3.49 on chat. That gap is why a single fixed verification length wastes compute on hard prompts and underuses easy ones. Second, EAGLE-3 beats DFlash across every Gemma4-12B benchmark, so a pure parallel drafter does not always beat an autoregressive one, yet DSpark still comes out ahead of both.

### 5.3 Experimental Analysis

#### 1. Position-wise acceptance confirms the design

Tracking acceptance at each draft position backs up the behavior in Sections 3.1 and 3.2. DSpark keeps the deep parallel backbone's strong first token (0.93 acceptance on math at position 1), and the Markov head keeps acceptance high and steady across the remaining positions instead of decaying.

#### 2. A little autoregression goes a long way

- **Depth:** with a 7-token block, DSpark improves with each added layer from 1 to 5. The biggest jump happens between 1 and 2 layers, and a 2-layer DSpark already beats the 5-layer DFlash in every domain.
- **Block length:** DSpark beats DFlash at every block length tested, and the gap widens as blocks get longer. At 7 draft tokens, DSpark accepts 16% more on math, 15% more on code, and 18% more on chat. At 15 draft tokens, those gains grow to 30%, 26%, and 22%. Replacing the Markov head with an RNN head adds only small extra gains on longer blocks.
- **Cost:** at batch size 128, DSpark's sequential head makes each round only 0.2% to 1.3% slower than DFlash across draft lengths from 4 to 16 tokens, while accepting up to 30% more tokens.

#### 3. Verify smarter, not longer: the role of confidence scheduling

Before adding the full scheduler, the paper tests the confidence head on its own by dropping draft tokens below a fixed confidence cutoff. Raising that cutoff filters out tokens that would mostly be rejected anyway, pushing the acceptance rate of the kept tokens from 45.7% to 95.7% on chat, from 76.9% to 92.5% on math, and from 67.6% to 92.0% on code. Chat benefits the most because its later draft tokens miss more often. Still, a high acceptance rate on kept tokens does not guarantee a wall-clock speedup, since cutting too many tokens shrinks each round. A fixed cutoff also ignores GPU load, which is why production serving pairs the calibrated confidence head with the hardware-aware scheduler.

#### 4. Production results

- **Scalability:** the production drafter uses three MoE layers with a sliding window attention of 128 plus the Markov head. To keep training fast on long contexts, workers send hidden states instead of full-vocabulary logits and pack short training blocks around sampled anchor positions.
- **Hardware-aware prefix scheduler:** the two production tweaks from Section 4.2.2 let the scheduler run alongside zero-overhead scheduling and CUDA graphs without changing the output distribution.
- **High throughput with low latency:** in DeepSeek's deployment, KV-cache limits and live traffic keep batches below GPU compute saturation, so higher total throughput and faster per-user speed improve together. To verify a different number of tokens per request in the same batch, the engine flattens all tokens and tracks their request IDs, requiring changes to only two kernels (index-attention and compress).
- **Results from real traffic:** at moderate load (under 200 concurrent requests on V4-Flash and 150 on V4-Pro), the scheduler verifies about 4–6 tokens per request (including the target's own token) compared with MTP-1's fixed 2, and dials that budget down as load rises. DSpark replaced MTP-1 in production two weeks after the DeepSeek-V4 preview launch.

| Engine | Per-user speed target (SLA) | Throughput vs. MTP-1 | Per-user speed at matched throughput |
| --- | --- | --- | --- |
| V4-Flash | 80 tok/s/user | +51% | 60–85% faster |
| V4-Flash | 120 tok/s/user | +661% (nominal) | |
| V4-Pro | 35 tok/s/user | +52% | 57–78% faster |
| V4-Pro | 50 tok/s/user | +406% (nominal) | |

*From the paper's Sec. 5.4.* At the stricter speed targets, MTP-1 can only serve a very small batch, so the paper reads the +661% and +406% as DSpark extending what is reachable, not as typical speedups.

### 5.4 Limitations and Future Work

One limitation noted in the paper is that the scheduler only saves work on the target model. The drafter still runs its full parallel backbone over the entire block on every step, even on hard requests where the scheduler ends up dropping most of the draft tokens. The authors suggest adding difficulty-aware early exit inside the drafter so hard prompts can skip generating the full block.

From my perspective, a few more trade-offs are worth keeping in mind.

- **Offline tables measure draft quality, not speed:** Table 1 runs with the confidence scheduler turned off and a fixed block size of 7. That is a clean way to compare drafters, but real end-to-end speedup still depends on drafter overhead, batch size, and target step time.
- **Using scores from two steps ago can lag sudden changes:** In the paper's design, the batch size is picked from scores two steps earlier. That works well when acceptance stays steady from round to round, but if a request suddenly switches from predictable code to open-ended writing, the number of tokens the scheduler verifies takes two rounds to catch up.

## 6. Demo

[vLLM](https://docs.vllm.ai/en/stable/features/speculative_decoding/adaptive_verification/) already supports both parts of DSpark. Setting `"method": "dspark"` runs the semi-autoregressive drafter from Section 4.1, and setting `"enable_adaptive_verification": true` adds the confidence-scheduled verification from Section 4.2. With adaptive verification on, vLLM scores each draft token with the checkpoint's confidence head and uses a step-time table profiled at startup to decide how many tokens each request should verify.

### Step 1. Install vLLM

Install vLLM in a fresh Python environment. The `bench` extra adds the packages the benchmark tool needs to load datasets from Hugging Face. Then run `vllm download-kernels` to install FlashInfer's precompiled kernels. Even with Triton attention, vLLM uses FlashInfer for top-k and top-p sampling. Without the precompiled kernels, FlashInfer downloads and compiles them when the server starts, which can take several minutes.

{{< tabs >}}
{{< tab "GPU machine" >}}
```bash
pip install -U "vllm[bench]"
vllm download-kernels
```
{{< /tab >}}
{{< tab "Modal" >}}
[Modal](https://modal.com) runs the same install inside a container image, and every run gets its own H100, billed by the second. Modal needs a payment method on file before it lets you use GPUs. Install the Modal client and log in.

```bash
pip install modal
modal setup
```

Then create `dspark_modal.py`. Its `run` function plays both terminals. It starts the server commands in the background, waits until the server answers, runs the client commands, and then stops the server like Ctrl+C. The `!` in `H100!` keeps Modal from moving the run to an H200, so every benchmark runs on the same GPU.

```bash
cat > dspark_modal.py <<'EOF'
import os
import signal
import subprocess
import time
import urllib.request

import modal

app = modal.App("dspark-demo")

# Step 1 on Modal: a container image with vLLM installed
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("curl")
    .uv_pip_install("vllm[bench]")
    .run_commands("vllm download-kernels", gpu="H100")  # vllm needs a GPU to start
)

# Keep downloaded models and compiled kernels between runs
volumes = {
    "/root/.cache/huggingface": modal.Volume.from_name("dspark-hf-cache", create_if_missing=True),
    "/root/.cache/vllm": modal.Volume.from_name("dspark-vllm-cache", create_if_missing=True),
}


@app.function(image=image, gpu="H100!", volumes=volumes, timeout=60 * 60)
def run(server: str, client: str):
    # Terminal 1: start the server in the background
    server_process = subprocess.Popen(["bash", "-c", server], start_new_session=True)

    # Wait until the server answers, which happens after "Application startup complete"
    while True:
        try:
            urllib.request.urlopen("http://localhost:8000/health")
            break
        except OSError:
            if server_process.poll() is not None:
                raise RuntimeError("vllm serve exited before it was ready")
            time.sleep(5)

    # Terminal 2: run the client commands against the server
    subprocess.run(["bash", "-c", client], check=True)

    # Wait for the server's next stats log, then stop it like Ctrl+C
    time.sleep(15)
    os.killpg(server_process.pid, signal.SIGINT)
    server_process.wait()


@app.local_entrypoint()
def main(server: str, client: str):
    with open(server) as server_file, open(client) as client_file:
        run.remote(server_file.read(), client_file.read())
EOF
```
{{< /tab >}}
{{< /tabs >}}

### Step 2. Launch vLLM with DSpark

Point `--speculative-config` at the DSpark drafter that matches your target model. The released Qwen3 drafters were trained with thinking mode turned off, so the server turns it off too. Prefix caching is disabled so that repeated benchmark runs do not reuse cached prompts and inflate the results. Adaptive verification also needs an attention backend whose CUDA graphs can handle a different number of tokens for each request. On GPUs such as the A100, L40S and RTX PRO 6000, vLLM's default FlashAttention 2 backend cannot do this and fails at startup, so the server uses Triton attention instead.

{{< tabs >}}
{{< tab "GPU machine" >}}
```bash
# Terminal 1: start the server
vllm serve Qwen/Qwen3-8B \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{
    "method": "dspark",
    "model": "deepseek-ai/dspark_qwen3_8b_block7",
    "num_speculative_tokens": 7,
    "enable_adaptive_verification": true
  }'
```

Once the log prints `Application startup complete`, send a test request from a second terminal.

```bash
# Terminal 2: send a test request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "user", "content": "Write a Python function for binary search."}
    ],
    "max_tokens": 256,
    "temperature": 0
  }'
```
{{< /tab >}}
{{< tab "Modal" >}}
Save the server command and the test request as files, then run both on Modal. The first run takes about ten minutes because it builds the image and downloads both models. Later runs reuse them and start in about three minutes.

```bash
# Save the server command
cat > server.sh <<'EOF'
vllm serve Qwen/Qwen3-8B \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{
    "method": "dspark",
    "model": "deepseek-ai/dspark_qwen3_8b_block7",
    "num_speculative_tokens": 7,
    "enable_adaptive_verification": true
  }'
EOF
```

```bash
# Save the test request
cat > client.sh <<'EOF'
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "user", "content": "Write a Python function for binary search."}
    ],
    "max_tokens": 256,
    "temperature": 0
  }'
EOF
```

```bash
# Run both on a Modal H100
modal run dspark_modal.py --server server.sh --client client.sh
```
{{< /tab >}}
{{< /tabs >}}

The reply should look like any normal Qwen3-8B answer. In the server log, a line starting with `SpecDecoding metrics` reports the mean acceptance length, the average number of tokens each round produces, including the target model's own token.

### Step 3. Benchmark light and heavy load

Run the same two benchmarks against four servers. Every server runs the target model with Triton attention, so the differences come from speculative decoding alone. On a GPU machine, start each server in one terminal, run the benchmarks from a second terminal, and stop the server with Ctrl+C before starting the next one. On Modal, each `modal run` starts its server, runs both benchmarks and stops the server for you. Light load sends one request at a time, and heavy load keeps 64 requests in flight on the same GPU. The chat backend lets the server apply the chat template, so thinking stays off during the benchmark.

#### 3.1 No speculative decoding

This is the baseline, where the target model generates one token per step.

{{< tabs >}}
{{< tab "GPU machine" >}}
```bash
# Terminal 1: start the server
vllm serve Qwen/Qwen3-8B \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

```bash
# Terminal 2: light load, one request at a time
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-8B --skip-chat-template \
  --dataset-name hf --dataset-path philschmid/mt-bench \
  --num-prompts 80 --max-concurrency 1

# Terminal 2: heavy load, 64 requests at a time
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-8B --skip-chat-template \
  --dataset-name hf --dataset-path philschmid/mt-bench \
  --num-prompts 320 --max-concurrency 64
```
{{< /tab >}}
{{< tab "Modal" >}}
```bash
# Save the server command
cat > server.sh <<'EOF'
vllm serve Qwen/Qwen3-8B \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  --default-chat-template-kwargs '{"enable_thinking": false}'
EOF
```

```bash
# Save the light and heavy load benchmarks
cat > client.sh <<'EOF'
# light load, one request at a time
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-8B --skip-chat-template \
  --dataset-name hf --dataset-path philschmid/mt-bench \
  --num-prompts 80 --max-concurrency 1

# heavy load, 64 requests at a time
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-8B --skip-chat-template \
  --dataset-name hf --dataset-path philschmid/mt-bench \
  --num-prompts 320 --max-concurrency 64
EOF
```

```bash
# Run both on a Modal H100
modal run dspark_modal.py --server server.sh --client client.sh
```
{{< /tab >}}
{{< /tabs >}}

#### 3.2 Speculative decoding with a fixed draft length

Both servers below verify a fixed number of draft tokens for every request. The first uses EAGLE-3, a popular autoregressive drafter, and the second uses the DSpark drafter with adaptive verification turned off. Comparing the two shows how much DSpark's drafter helps on its own.

##### 3.2.1 EAGLE-3

EAGLE-3 drafts one token at a time, so each extra draft token costs another drafter pass. Red Hat's EAGLE-3 drafter for Qwen3-8B was trained with thinking turned off and recommends three draft tokens. Its training data differs from DeepSeek's drafters, so this compares the released drafters rather than reproducing the paper's Table 1.

{{< tabs >}}
{{< tab "GPU machine" >}}
```bash
# Terminal 1: start the server
vllm serve Qwen/Qwen3-8B \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{
    "method": "eagle3",
    "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
    "num_speculative_tokens": 3
  }'
```

```bash
# Terminal 2: light load, one request at a time
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-8B --skip-chat-template \
  --dataset-name hf --dataset-path philschmid/mt-bench \
  --num-prompts 80 --max-concurrency 1

# Terminal 2: heavy load, 64 requests at a time
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-8B --skip-chat-template \
  --dataset-name hf --dataset-path philschmid/mt-bench \
  --num-prompts 320 --max-concurrency 64
```
{{< /tab >}}
{{< tab "Modal" >}}
```bash
# Save the server command
cat > server.sh <<'EOF'
vllm serve Qwen/Qwen3-8B \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{
    "method": "eagle3",
    "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
    "num_speculative_tokens": 3
  }'
EOF
```

```bash
# Save the light and heavy load benchmarks
cat > client.sh <<'EOF'
# light load, one request at a time
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-8B --skip-chat-template \
  --dataset-name hf --dataset-path philschmid/mt-bench \
  --num-prompts 80 --max-concurrency 1

# heavy load, 64 requests at a time
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-8B --skip-chat-template \
  --dataset-name hf --dataset-path philschmid/mt-bench \
  --num-prompts 320 --max-concurrency 64
EOF
```

```bash
# Run both on a Modal H100
modal run dspark_modal.py --server server.sh --client client.sh
```
{{< /tab >}}
{{< /tabs >}}

##### 3.2.2 DSpark drafter with a fixed draft length

This uses the same DSpark drafter as Step 2 with adaptive verification turned off, so every request verifies all 7 draft tokens, just like standard speculative decoding.

{{< tabs >}}
{{< tab "GPU machine" >}}
```bash
# Terminal 1: start the server
vllm serve Qwen/Qwen3-8B \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{
    "method": "dspark",
    "model": "deepseek-ai/dspark_qwen3_8b_block7",
    "num_speculative_tokens": 7,
    "enable_adaptive_verification": false
  }'
```

```bash
# Terminal 2: light load, one request at a time
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-8B --skip-chat-template \
  --dataset-name hf --dataset-path philschmid/mt-bench \
  --num-prompts 80 --max-concurrency 1

# Terminal 2: heavy load, 64 requests at a time
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-8B --skip-chat-template \
  --dataset-name hf --dataset-path philschmid/mt-bench \
  --num-prompts 320 --max-concurrency 64
```
{{< /tab >}}
{{< tab "Modal" >}}
```bash
# Save the server command
cat > server.sh <<'EOF'
vllm serve Qwen/Qwen3-8B \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{
    "method": "dspark",
    "model": "deepseek-ai/dspark_qwen3_8b_block7",
    "num_speculative_tokens": 7,
    "enable_adaptive_verification": false
  }'
EOF
```

```bash
# Save the light and heavy load benchmarks
cat > client.sh <<'EOF'
# light load, one request at a time
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-8B --skip-chat-template \
  --dataset-name hf --dataset-path philschmid/mt-bench \
  --num-prompts 80 --max-concurrency 1

# heavy load, 64 requests at a time
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-8B --skip-chat-template \
  --dataset-name hf --dataset-path philschmid/mt-bench \
  --num-prompts 320 --max-concurrency 64
EOF
```

```bash
# Run both on a Modal H100
modal run dspark_modal.py --server server.sh --client client.sh
```
{{< /tab >}}
{{< /tabs >}}

#### 3.3 DSpark with adaptive verification

This is the same server as Step 2, with both the drafter and adaptive verification turned on.

{{< tabs >}}
{{< tab "GPU machine" >}}
```bash
# Terminal 1: start the server
vllm serve Qwen/Qwen3-8B \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{
    "method": "dspark",
    "model": "deepseek-ai/dspark_qwen3_8b_block7",
    "num_speculative_tokens": 7,
    "enable_adaptive_verification": true
  }'
```

```bash
# Terminal 2: light load, one request at a time
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-8B --skip-chat-template \
  --dataset-name hf --dataset-path philschmid/mt-bench \
  --num-prompts 80 --max-concurrency 1

# Terminal 2: heavy load, 64 requests at a time
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-8B --skip-chat-template \
  --dataset-name hf --dataset-path philschmid/mt-bench \
  --num-prompts 320 --max-concurrency 64
```
{{< /tab >}}
{{< tab "Modal" >}}
```bash
# Save the server command
cat > server.sh <<'EOF'
vllm serve Qwen/Qwen3-8B \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{
    "method": "dspark",
    "model": "deepseek-ai/dspark_qwen3_8b_block7",
    "num_speculative_tokens": 7,
    "enable_adaptive_verification": true
  }'
EOF
```

```bash
# Save the light and heavy load benchmarks
cat > client.sh <<'EOF'
# light load, one request at a time
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-8B --skip-chat-template \
  --dataset-name hf --dataset-path philschmid/mt-bench \
  --num-prompts 80 --max-concurrency 1

# heavy load, 64 requests at a time
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model Qwen/Qwen3-8B --skip-chat-template \
  --dataset-name hf --dataset-path philschmid/mt-bench \
  --num-prompts 320 --max-concurrency 64
EOF
```

```bash
# Run both on a Modal H100
modal run dspark_modal.py --server server.sh --client client.sh
```
{{< /tab >}}
{{< /tabs >}}

Each benchmark prints a summary. Compare the output token throughput, which measures total system speed, and the mean TPOT, which measures how fast each user sees new tokens. At light load, all three speculative versions should run well ahead of the baseline, since the GPU has spare compute and extra draft tokens are almost free. Comparing 3.2.1 with 3.2.2 shows how much DSpark's drafter adds on its own. At heavy load, both fixed-length versions still verify every draft token for every request, so their lead shrinks. Adaptive verification trims the low-survival tokens as the batch fills up, so 3.3 should keep more of its speedup. To see how acceptance changes across domains, change `--dataset-path` to `openai/gsm8k` or `openai/openai_humaneval`.

## 7. Takeaways

- **Drafting and verification solve different problems.** A deep parallel backbone with a tiny Markov head makes it cheap to draft long, coherent blocks, while a separate scheduler decides how many of those tokens are worth sending to the target model.
- **Accepted tokens per round is not the only metric that matters.** In production, the real goal is total system throughput within your latency target, which depends on batch load as much as draft accuracy.
- **Draft length works best as a live scheduling choice.** Checking more draft tokens pays off when the GPU has spare room, and checking fewer protects the batch when traffic is heavy.

---

*Paper: [arXiv:2607.05147](https://arxiv.org/abs/2607.05147). All figures are illustrative redraws; paper-specific examples and numerical results are based on the DSpark paper.*
