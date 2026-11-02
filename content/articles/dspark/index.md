+++
title = "DSpark: when speculative decoding starts caring about your batch"
date = "2026-11-01"
description = "A serving engineer's read of DSpark (arXiv:2607.05147) — how semi-autoregressive drafting reduces suffix decay, and why confidence-scheduled verification makes draft length a scheduling decision instead of a fixed hyperparameter."
tags = ["llm-inference", "llm-serving", "paper-review"]
+++

Speculative decoding has a production problem. Verifying a few extra draft tokens may help one request, but under heavy load those tokens can consume GPU capacity that could have served other requests. DSpark solves this by estimating the chance that each draft token will be accepted and checking how busy the GPU is. When the GPU has spare capacity, it verifies more tokens. When the system is busy, it verifies fewer, because extra tokens would slow the whole batch. The goal is to keep total throughput high, rather than using a fixed number for every request.

<!--more-->

## TLDR

DSpark combines a parallel drafting backbone with a tiny sequential Markov head to reduce suffix decay without paying for a full autoregressive drafter. It then uses calibrated token confidence and profiled GPU step times to choose how many draft tokens each request should verify under the current load. The authors report higher accepted lengths across four target models and substantial production throughput gains over MTP-1, although the production results have not been independently reproduced.

## 1. Why decoding is the bottleneck

{{< figure src="decode-bottleneck.png" link="decode-bottleneck.png" alt="Top: in prefill, model weights are read once from HBM and shared across all prompt tokens in one compute-bound pass. Bottom: in decode, every step re-reads the weights and the growing KV cache to produce a single token, leaving compute waiting on memory." caption="Prefill reuses one weight read across the whole prompt, while decode re-reads weights and the KV cache for each new token (illustrative, not to scale)." >}}

LLM inference has two phases, prefill and decode. In prefill (top of the figure), the model processes the whole prompt in one forward pass, building the KV cache and producing the first output token. Every prompt token is processed in parallel, so each weight read from memory is reused across many tokens. For sufficiently large prompts, prefill is generally compute-bound. It spends most of its time computing rather than waiting for data.

Decode (bottom of the figure) produces the remaining tokens one step at a time until the model emits an end-of-sequence token or reaches the output's maximum length. Each token depends on the previous one, so every step reads the model weights and KV cache from GPU memory to produce the next token. At small batches, there is little computation for each byte read, and the KV-cache reads grow with the output, so the GPU spends much of its time waiting for data. As a result, this phase usually dominates end-to-end latency.

There are many ways to tackle this problem, including prefill/decode disaggregation (so long prefills don't stall decoding), quantization (fewer bytes to read), hardware with more memory bandwidth (such as NVIDIA Rubin), and speculative decoding (more tokens per read). This article focuses on speculative decoding.

## 2. What is speculative decoding?

Speculative decoding takes advantage of how fast prefill is. Modern GPUs have plenty of compute, so a large model can verify several tokens from a small model in one forward pass for nearly the same cost as generating a single token. A smaller drafter proposes the next k tokens, and the large model checks them all in one pass. If a draft token is rejected, the system keeps the accepted prefix and uses a token from the large model instead. With sampling, acceptance is probabilistic, but the final output still matches what the large model would have produced. Every round generates at least one token.

The speedup depends on two things. First, how cheap the draft is. Second, how many draft tokens get accepted. The two main families of drafters trade these off differently.

### 2.1 Autoregressive drafter

{{< figure src="drafter-ar.png" link="drafter-ar.png" alt="An autoregressive drafter runs four sequential passes (fox, jumps, over, a), and the target verifies all four in one pass, keeping fox, jumps, over and replacing the rejected a with the." caption="Autoregressive drafter: four sequential draft passes followed by one target verification pass (green = accepted draft, blue = target token)." >}}

The drafter generates the k tokens one at a time, each conditioned on the previous one. Drafts are high quality because every token sees the tokens before it, but drafting cost grows linearly with k, so draft lengths stay short. Popular methods include EAGLE-3 and MTP, often paired with tree-based verification to make up for the short drafts.

### 2.2 Parallel drafter

{{< figure src="drafter-parallel.png" link="drafter-parallel.png" alt="A parallel drafter predicts all four positions in one pass, and the target verifies them in one pass, keeping fox, jumps, over and replacing a with the." caption="Parallel drafter: one draft pass for the whole block followed by one target verification pass." >}}

The drafter predicts all k tokens in a single forward pass. Drafting cost stays roughly constant as k grows, but each position is predicted without seeing the actual tokens before it, so the acceptance rate drops at later positions in the block. Popular methods include Medusa, PARD, DART, and DFlash.

## 3. What's wrong with the prior methods?

### 3.1 Autoregressive drafters

{{< figure src="prior-ar.png" link="prior-ar.png" alt="Left: drafter passes grow linearly with block length k for an autoregressive drafter while staying at 1 for a parallel drafter. Right: first-position acceptance is 0.81 vs. 0.88 on math and 0.53 vs. 0.72 on chat for EAGLE-3 vs. DFlash." caption="Left: drafter passes per block. Right: acceptance at position 1, EAGLE-3 vs. DFlash (paper, Sec. 4.3.1)." >}}

Each draft token needs its own drafter forward pass (left panel), so drafting latency grows linearly with the block length. To keep that latency low, these drafters use short blocks and shallow networks, then use tree-based verification to compensate for the short blocks by verifying several candidate paths at once. This sends more tokens to the target model and can reduce aggregate serving throughput under concurrency. A shallow network is also less accurate, and this matters most at the first draft position. The target only accepts a prefix, so if the first draft token is wrong, the whole block is thrown away. As shown on the right, EAGLE-3 is accepted less often at the first position than DFlash, a parallel drafter that can afford a much deeper network (0.81 vs. 0.88 on math, and 0.53 vs. 0.72 on chat).

### 3.2 Parallel drafters

{{< figure src="prior-parallel.png" link="prior-parallel.png" alt="Left: a parallel drafter picks each position independently and can mix of course and no problem into of problem or no course. Right: conditional acceptance on chat falls from 0.72 to 0.63 for DFlash while rising from 0.53 to 0.74 for EAGLE-3." caption="Left: independent position picks can collide. Right: conditional acceptance on chat (paper, Sec. 4.3.1)." >}}

A parallel drafter predicts every position in the block in one forward pass, so no position can see which token was actually sampled before it. Instead, each position must account for all the tokens that might come before it. As a result, when the context allows several continuations, different positions can lock onto different ones. For example, as shown on the left of the figure, if both "of course" and "no problem" fit, a parallel drafter may propose "of problem" or "no course". The paper calls this issue a *multi-modal collision*, and it causes *suffix decay*. Conditional acceptance tends to fall at later positions, even when every earlier token in the block was accepted. Therefore, for a pure parallel drafter, each extra position in a longer block tends to be worth less than the one before.

The right panel plots *conditional acceptance* at each position, that is, how often a token is accepted given that every token before it was. DFlash starts high but falls along the block, from 0.87 to 0.78 on code and from 0.72 to 0.63 on chat. However, EAGLE-3 does the opposite. It starts lower (0.53 on chat, the same first-position number as in 3.1), but because each of its tokens sees the ones before it, it holds steady or even improves deeper into the block (up to 0.74 on chat).

### 3.3 Verification length

{{< figure src="prior-verify.png" link="prior-verify.png" alt="Left: with six draft tokens, a code request keeps five while a chat request keeps one. Right: at light load rejected tokens use idle slots, while at heavy load they fill the batch and delay a fifth request." caption="Left: acceptance varies by request type. Right: rejected tokens are cheap at light load but block other requests at heavy load." >}}

Parallel drafting makes long blocks cheap to produce, but verifying every drafted token is not cheap. The ideal verification length changes along two axes.

- **Data (left panel):** acceptance depends on the request. Structured text such as code sustains much higher acceptance than open-ended chat, so any single length is too short for some requests and too long for others.
- **System load (right panel):** at light load, verifying an extra token can be nearly free when the GPU has idle capacity and is waiting on memory anyway, so even a rejected token costs almost nothing. Under heavy load, every token sent for verification adds to the batch the target model processes in that step, and the step gets slower as that batch grows. A token that is likely to be rejected takes batch capacity that could have served another request.

Put together, the two families fail in opposite places. Autoregressive drafters are weak at the first position, and parallel drafters get weaker at later ones. The first position matters most since a miss there throws away the whole block. DSpark targets this gap by retaining the parallel drafter's strong first position while reducing decay at later positions.

Prior methods also either verify a fixed number of tokens or use a static confidence threshold. Neither accounts for current system load. That can work under single-request assumptions, but it is suboptimal in a high-concurrency production system. Before DSpark, DeepSeek's production serving used MTP-1, a single draft token, because deploying a static multi-token drafter (MTP-3 or MTP-5) degraded aggregate throughput under high concurrency.

## 4. How DSpark works

DSpark addresses the problems from Section 3 with two parts. The first is *semi-autoregressive generation*, which keeps the strong first position of a parallel drafter and reduces suffix decay after it. The second is *confidence-scheduled verification*, which decides how many draft tokens are worth sending to the target based on how confident the drafter is and how busy the GPU is.

### 4.1 Semi-autoregressive generation

{{< figure src="semi-ar.png" link="semi-ar.png" alt="An anchor token and mask tokens pass through one parallel backbone to produce base guesses, then a tiny rank-256 Markov head runs left to right so sampling of at position 1 boosts course at position 2." caption="One heavy parallel pass produces base guesses for the whole block, and a tiny Markov head refines them left to right (paper, Sec. 3.1)." >}}

DSpark splits drafting into two stages, as shown in the figure. In the *parallel stage*, a heavy backbone (DFlash in the paper) runs one forward pass over the whole block and gives a base guess for every position. This is where most of the drafting work happens, so drafting time still stays roughly constant as the block grows. Then, in the *sequential stage*, a tiny *Markov head* walks through the block from left to right. At each position, it adds a bias to the base guess that depends on the token just sampled, then samples the next token. For example, if both "of course" and "no problem" fit, the backbone alone guesses "of / no" and "course / problem". Once position 1 samples "of", the Markov head boosts "course" and suppresses "problem" at position 2, so the pair stays coherent. In short, this acts like a parallel drafter for speed and uses a small sequential head to keep adjacent words coherent without sacrificing that speed.

The Markov head has to be tiny because it runs once per position, one after another, using a low-rank lookup table (rank 256 by default). The paper also tests an *RNN head* that remembers the whole prefix instead of only the previous token, but it only gives small extra gains.

### 4.2 Confidence-scheduled verification

{{< figure src="verify-round.png" link="verify-round.png" alt="After context The quick brown fox, DSpark drafts jumps, over, a, and lazy with confidences 0.95, 0.90, 0.60, and 0.20. The scheduler drops lazy, and the target verifies the first three tokens in one pass." caption="One decoding round: the scheduler drops the low-confidence tail token before the target pass (paper, Fig. 1)." >}}

Even with better drafts, verifying every drafted token is wasteful because the last few tokens of a block are often rejected. DSpark adds two pieces to decide how much of each block to verify. A *confidence head* gives each draft token a confidence score, and a *hardware-aware prefix scheduler* uses those scores together with the current GPU load to choose how many tokens from each request to send to the target. In the figure, the scheduler keeps "jumps over a" and drops "lazy", so the target verifies only three tokens. The target then works like normal speculative decoding. It accepts "jumps over", rejects "a", and adds its own token "the". The paper shows that this scheduling keeps the output distribution exactly the same as the target model's.

#### 4.2.1 Confidence Head

{{< figure src="confidence-head.png" link="confidence-head.png" alt="Step 1: backbone state h_k and previous token embedding pass through a linear layer and sigmoid to produce raw c_k. Step 2: Sequential Temperature Scaling (STS) divides logits by position-specific temperatures T_k to align overconfident raw scores with true acceptance. Step 3: calibrated c_k values (0.95, 0.90, 0.70, 0.50) multiply into falling prefix survival bars (0.95, 0.85, 0.60, 0.30)." caption="From per-token confidence to prefix survival: (1) computing raw c_k, (2) calibrating with STS, and (3) multiplying into prefix survival a_k (illustrative values; calibration error from Sec. 4.3.3)." >}}

As shown in Step 1 of the figure, the confidence head is a small linear layer with a sigmoid on top. For each draft position k, it looks at the backbone's hidden state and the embedding of the previous draft token, and outputs a score c_k. This is the chance that token k is accepted, given that every token before it was accepted. During training, it learns to match the real acceptance rate, which the paper computes from how far the draft distribution is from the target's.

Each score c_k only covers one position, while the target model accepts draft tokens as a continuous prefix. However, raw neural network scores are usually overconfident, which would make prefix estimates too high. As shown in Step 2, DSpark calibrates them with *Sequential Temperature Scaling (STS)*, fitting one temperature per position from left to right on held-out data. This rescales the scores to match real acceptance rates without changing their order, reducing the calibration error from 3–8% to about 1%. In Step 3, DSpark multiplies these calibrated scores together into a *prefix survival* probability, c_1 × c_2 × … × c_k, which gives the chance that every token up to position k is accepted and never increases along the block.

#### 4.2.2 Hardware-Aware Prefix Scheduler

{{< figure src="prefix-scheduler.png" link="prefix-scheduler.png" alt="Step 1: three requests (code, math, chat) with five draft tokens each and decreasing prefix survival. Step 2: pooling and sorting all candidate tokens against profiled step time shows expected tok/s peaking at 9 tokens under heavy load and 13 tokens under light load. Step 3: the resulting kept prefixes under light load (13 of 15) and heavy load (9 of 15)." caption="Hardware-aware prefix scheduler: (1) per-request prefix survival, (2) global ranking and step-time cutoff, and (3) selected prefixes under light vs. heavy load (illustrative)." >}}

With the survival scores ready, the scheduler has to decide how many draft tokens to send to the target model across the whole batch (Step 1 in the figure). Sending another draft token can produce more accepted output, but it also makes the target model's batch larger and slower to run. To find the best balance, the scheduler pools all draft tokens from every request, ranks them by their survival score from highest to lowest, and adds them one by one (Step 2). A later token in a request never has a higher survival score than an earlier one, so this ranking always takes tokens from the start of each request's draft. For each batch size, the scheduler looks up the step time from a table profiled at startup and picks the point that yields the most expected accepted tokens per second. As shown in Step 3, when load is light (left), extra tokens barely slow the step down, so the scheduler verifies 13 of the 15 draft tokens. When load is heavy (right), every extra token slows the whole batch, so only the 9 highest-survival tokens make the cut.

{{< figure src="scheduler-production.png" link="scheduler-production.png" alt="Left: GPU step time rises in flat plateaus and sharp jumps at tile boundaries, creating a local dip in expected tok/s before the true global peak. Right: while Step t-1 runs on the GPU, the CPU uses Step t-2 confidences to lock the batch size budget B* early, then fills those B* slots with the latest Step t scores." caption="Two production adjustments: (1) searching the full step-time table past tile-boundary dips, and (2) using two-step lookback to size the batch without stalling the GPU pipeline (paper, Sec. 5.2)." >}}

Running this in a production serving engine takes two small adjustments, illustrated above. First (left panel), GPU step time does not rise smoothly as the batch grows. It stays flat across a plateau and then spikes up when a hardware tile fills, which creates local dips in expected tokens per second. Instead of stopping at the first dip, the scheduler checks the whole step-time table to find the true peak. Second (right panel), the engine prepares the next batch while the GPU is still running the current one, so it cannot wait for the newest confidence scores to pick the batch size. DSpark gets around this by using scores from two steps earlier (`Step t-2`) to lock in the batch budget `B*` early, then filling those `B*` slots with the latest scores (`Step t`) once they arrive.

## 5. Results

*All results in this section are reported by the DSpark paper. I have not reproduced them.*

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

All three drafters are trained with the same code and data in DeepSpec. The confidence scheduler is turned off for these offline tests so every drafter proposes a fixed 7-token block, which isolates draft quality from scheduling.

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

DSpark reaches the longest accepted length on every benchmark and every target model. On Qwen3-4B, 8B, and 14B, it accepts 30.9%, 26.7%, and 30.0% more tokens per round than EAGLE-3, and 16.3%, 18.4%, and 18.3% more than DFlash (Sec. 4.2). The same lead holds on Gemma4-12B, showing that the design carries over to another model family.

Two patterns in the table stand out. First, acceptance varies sharply by task. On Qwen3-4B, DSpark keeps 5.57 tokens per round on math and 5.12 on code, compared with 3.49 on chat. That gap is why a single fixed verification length wastes compute on hard prompts and underuses easy ones. Second, EAGLE-3 beats DFlash across every Gemma4-12B benchmark, so a pure parallel drafter does not always beat an autoregressive one, yet DSpark still comes out ahead of both.

### 5.3 Experimental Analysis

#### 1. Position-wise acceptance confirms the design (Sec. 4.3.1)

Tracking acceptance at each draft position backs up the behavior in Sections 3.1 and 3.2. DSpark keeps the deep parallel backbone's strong first token (0.93 acceptance on math at position 1), and the Markov head keeps acceptance high and steady across the remaining positions instead of decaying.

#### 2. A little autoregression goes a long way (Sec. 4.3.2, Qwen3-4B)

- **Depth:** with a 7-token block, DSpark improves with each added layer from 1 to 5. The biggest jump happens between 1 and 2 layers, and a 2-layer DSpark already beats the 5-layer DFlash in every domain.
- **Block length:** DSpark beats DFlash at every block length tested, and the gap widens as blocks get longer. At 7 draft tokens, DSpark accepts 16% more on math, 15% more on code, and 18% more on chat. At 15 draft tokens, those gains grow to 30%, 26%, and 22%. Replacing the Markov head with an RNN head adds only small extra gains on longer blocks.
- **Cost:** at batch size 128, growing the draft from 4 to 16 tokens adds just 0.2% to 1.3% to total round latency over DFlash while increasing accepted length by up to 30%.

#### 3. Verify smarter, not longer: the role of confidence scheduling (Sec. 4.3.3, Qwen3-4B)

Before adding the full scheduler, the paper tests the confidence head on its own by dropping draft tokens below a fixed confidence cutoff. Raising that cutoff filters out tokens that would mostly be rejected anyway, pushing the acceptance rate of the kept tokens from 45.7% to 95.7% on chat, from 76.9% to 92.5% on math, and from 67.6% to 92.0% on code. Chat benefits the most because its later draft tokens miss more often. Still, a high acceptance rate on kept tokens does not guarantee a wall-clock speedup, since cutting too many tokens shrinks each round. A fixed cutoff also ignores GPU load, which is why production serving pairs the calibrated confidence head with the hardware-aware scheduler.

#### 4. Production results (Sec. 5, DeepSeek-V4-Flash and V4-Pro previews, DSpark with up to 5 draft tokens against the previous MTP-1 setup)

- **Scalability:** the production drafter uses three MoE layers with a sliding window attention of 128 plus the Markov head. To keep training fast on long contexts, workers send hidden states instead of full-vocabulary logits and pack short training blocks around sampled anchor positions.
- **Hardware-aware prefix scheduler:** the two production tweaks from Section 4.2.2 let the scheduler run alongside zero-overhead scheduling and CUDA graphs without changing the output distribution.
- **High throughput with low latency:** in DeepSeek's deployment, KV-cache limits and live traffic keep batches below GPU compute saturation, so higher total throughput and faster per-user speed improve together. To verify a different number of tokens per request in the same batch, the engine flattens all tokens and tracks their request IDs, requiring changes to only two kernels (index-attention and compress).
- **Results from real traffic:** at moderate load (under 200 concurrent requests on V4-Flash and 150 on V4-Pro), the scheduler verifies about 4–6 tokens per request (including the target's own token) compared with MTP-1's fixed 2, and dials that budget down as load rises. DSpark replaced MTP-1 in production two weeks after the DeepSeek-V4 preview launch.

| Engine | Per-user speed target (SLA) | Throughput vs. MTP-1 | Per-user speed at matched throughput |
|---|---|---|---|
| V4-Flash | 80 tok/s/user | +51% | 60–85% faster |
| V4-Flash | 120 tok/s/user | +661% (nominal) | |
| V4-Pro | 35 tok/s/user | +52% | 57–78% faster |
| V4-Pro | 50 tok/s/user | +406% (nominal) | |

*From the paper's Sec. 5.4.* At the stricter speed targets, MTP-1 can only serve a very small batch, so the paper reads the +661% and +406% as DSpark extending what is reachable, not as typical speedups.

### 5.4 Limitations and Future Work

One limitation noted in the paper is that the scheduler only saves work on the target model. The drafter still runs its full parallel backbone over the entire block on every step, even on hard requests where the scheduler ends up dropping most of the draft tokens. The authors suggest adding difficulty-aware early exit inside the drafter so hard prompts can skip generating the full block.

From my perspective, a few more trade-offs are worth keeping in mind.

- **Offline tables measure draft quality, not wall-clock speed:** Table 1 runs with the confidence scheduler turned off and a fixed block size of 7. That is a clean way to isolate drafter accuracy, but end-to-end speedup still depends on drafter overhead, batch size, and target step time.
- **A 1D step-time table assumes balanced contexts:** Profiling step time only by verified token count `B` works well in DeepSeek's disaggregated decode setup where context lengths are balanced across ranks. In smaller deployments where short chat prompts and long-context RAG requests share one GPU batch, step time depends on both token count `B` and KV-cache bytes read, so a 1D table can misjudge the cutoff.
- **Two-step lookback lags sudden shifts in difficulty:** Sizing step `t`'s batch budget from step `t-2` confidences keeps CUDA graphs and zero-overhead scheduling intact, but if a request suddenly shifts from predictable boilerplate to open-ended reasoning, the batch budget trails that change by two rounds.
- **Production gains are tied to DeepSeek's stack:** The live serving numbers come from DeepSeek's internal engine and traffic against MTP-1. Bringing the full system to open-source engines like vLLM or SGLang takes real systems work, including per-model confidence calibration, hardware step-time profiling, and ragged verification kernels.

## 6. Demo

DeepSeek released the pretrained DSpark drafter checkpoints on [Hugging Face](https://huggingface.co/collections/deepseek-ai/deepspec) along with the [DeepSpec](https://github.com/deepseek-ai/DeepSpec) training and evaluation code, and vLLM supports DSpark drafting through `--speculative-config`. On a single 48 GB RTX 6000, `Qwen/Qwen3-8B` paired with its 7-token DSpark drafter (`deepseek-ai/dspark_qwen3_8b_block7`) fits comfortably in `bf16` (or you can use `Qwen/Qwen3-4B` with `deepseek-ai/dspark_qwen3_4b_block7` on smaller GPUs).

Note that open-source vLLM currently runs DSpark's semi-autoregressive drafter from Section 4.1 with a fixed block size, while the load-aware prefix scheduler from Section 4.2.2 is part of DeepSeek's internal engine.

### Step 1. Pull the vLLM nightly image

DSpark support landed in vLLM nightly, so start by pulling the nightly container (or installing the nightly wheel in a fresh Python environment).

```bash
docker pull vllm/vllm-openai:nightly
```

### Step 2. Launch vLLM with DSpark enabled

Pass `--speculative-config` with `"method": "dspark"` and point `"model"` at the matching DSpark checkpoint. The released Qwen3 drafters were trained with thinking mode turned off, so disable thinking in the chat template to match the training distribution.

```bash
docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:nightly \
  --model Qwen/Qwen3-8B \
  --served-model-name qwen3-dspark \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method":"dspark","model":"deepseek-ai/dspark_qwen3_8b_block7","num_speculative_tokens":7,"attention_backend":"FLASH_ATTN"}'
```

To compare against plain decoding on the same RTX 6000, run the exact same command without the `--speculative-config` line.

### Step 3. Send a test request and compare speed

Once the server is ready, send an OpenAI-compatible chat completion request to port `8000`. Structured prompts like math or code show the largest jump in accepted draft tokens and output speed, while open-ended chat sees a smaller but steady gain.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-dspark",
    "messages": [
      {"role": "user", "content": "Write a Python function for binary search and explain its time complexity."}
    ],
    "max_tokens": 512,
    "temperature": 0.0
  }'
```

## 7. Takeaways

- **Drafting and verification solve different problems.** A deep parallel backbone with a tiny Markov head makes it cheap to draft long, coherent blocks, while a separate scheduler decides how many of those tokens are worth sending to the target model.
- **Accepted tokens per round is not the only metric that matters.** In production, the real goal is total system throughput within your latency target, which depends on batch load as much as draft accuracy.
- **Draft length works best as a live scheduling choice.** Checking more draft tokens pays off when the GPU has spare room, and checking fewer protects the batch when traffic is heavy.
- **The drafter is easy to try today, while the scheduler is engine work.** You can run the open-source DSpark drafters in vLLM on a single GPU right now, whereas getting the high-concurrency serving gains from the paper requires confidence calibration, profiled step-time tables, and variable-length verification kernels.

---

*Paper: [arXiv:2607.05147](https://arxiv.org/abs/2607.05147). All figures are illustrative redraws; paper-specific examples and numerical results are based on the DSpark paper.*
