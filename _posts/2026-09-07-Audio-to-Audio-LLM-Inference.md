---
layout: post
title: "How Audio-to-Audio LLM Inference Actually Works"
date: 2026-09-07
slug: audio-to-audio-llm-inference
description: "A ground-up explanation of how audio LLMs tokenize, reason over, and synthesize speech."
tags: [LLM, Inference, Audio, On-device]
categories: [Field Notes, ai-engineering]
toc:
  sidebar: left
---

# About Multi-Model LLMs - Text, Audio, Vision

As LLMs are getting mature with time, it is becoming increasingly evident of the possibilities of using multi-modals models like text, audio, and vision together, on the edge while still hitting the key performance metrics.
As of writing this, the latest release from Nvidia - [Nemotron VoiceChat 11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B) disrupts the traditional voice assistant pipeline. In a traditional voice assistants systems, the architecture contains cascaded modules for speech recognition, natural language understanding, dialogue management, and speech synthesis. I do not want to derive on the benefits of the edge AI solutions but rather would like to bring focus on the basic working of the audio-to-audio LLM inference pipeline.

So in the following part of the article I will cover on the basic notes from my learning about the audio models working. I've been learning about the Audio LLM. While debugging performance, I ended up with detailed logs of every phase of the inference pipeline, from model load through speech synthesis. This post is my attempt to write down what I learned, from first principles.

If you've used text LLMs but never thought about how voice gets in and out — this is for you.

## The Core Problem: Audio needs extra steps to Tokenize

LLMs are token machines. They take a sequence of integer IDs, run a forward pass, and output probabilities over the next integer ID. That's it. Text is easy to tokenize — split into sub-word chunks using BPE, map each to an integer. Done.

Audio is not. A 0.6-second voice clip at 16 kHz is 9,600 raw samples. A sequence of 9,600 floating point numbers has no natural token boundary. You can't just feed PCM into an LLM.

So audio-language models have to solve two bridging problems:

1. **Audio → tokens:** Convert the waveform into something the LLM backbone can reason over.
2. **Tokens → audio:** Convert the model's output tokens back into a playable waveform.

Both bridges are themselves neural networks. And both add substantial compute cost on top of the core LLM inference.

## What a Token Actually Is

Before going into audio specifics, let me nail the basics.

A token is a sub-word chunk. The BPE (Byte Pair Encoding) tokenizer merges common character sequences:

```
"octopuses" → ["oct", "op", "uses"]   ← 3 tokens, not 1 word
```

The model I was working with (LFM2.5-Audio) has a vocabulary of 65,536 possible tokens. At every decode step, the model assigns a probability to all 65,536 and you sample the next one.

LLM inference is fundamentally this loop:

```
Input:  "Sure, did you know octopuses"
Model:  [probability over 65,536 tokens]
Sample: "have"

Input:  "Sure, did you know octopuses have"
Model:  [new probabilities]
Sample: "three"
...
```

Each iteration is one **token decode step** — one full forward pass through the model. Everything else in this post is built on top of this loop.


## The Basics which are similar to text-based LLMs

### The Two Phases: Prefill and Decode

Every LLM inference run has two distinct phases.

**Prefill (prompt eval):** You have an existing context — system prompt, user input, audio embeddings. All of it exists simultaneously. The model processes all tokens in parallel as a batch, populating the KV cache. Fast per token because of parallelism.

**Decode (autoregressive generation):** You generate one token at a time. Feed the last token in, get the next one out. Repeat. Inherently serial. This is where most inference time goes.

Here are some akward numbers from one of the recent run:

```
prompt eval time = 1,833 ms / 41 tokens   (44.72 ms/token)
eval time        = 69,947 ms / 149 runs   (469.45 ms/token)
```

Prefill: 44 ms/token. Decode: 469 ms/token. Decode is 10× slower per token because you lose the batching parallelism.

### What Is Attention and Why Does It Matter

The transformer's core operation is **attention** — the mechanism by which every token can look at every previous token and decide how much to "draw from" it.

Each token produces three vectors:
- **Q (Query):** What am I looking for?
- **K (Key):** What information do I carry?
- **V (Value):** What should I pass forward?

The attention score between tokens A and B is `dot_product(Q_A, K_B)`. High score = A pays attention to B. Those scores are normalized and used to weight-sum all Values across the sequence.

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

This runs at every layer, for every token, against every past token. As context length grows, the `Q × K^T` matrix grows quadratically. That is the compute bottleneck.

### The KV Cache

You don't recompute K and V for tokens you've already processed. They get cached. On every new decode step, you only compute Q, K, V for the new token, then attend over the cached K and V from all previous positions. The cache is allocated once at model load and stays warm for the session.

### Flash Attention

Normally, attention requires the model to build and store the full `Q × K^T` matrix in GPU/CPU memory — a matrix that grows quadratically with context length. For a 4096-token context, that is 4096 × 4096 = 16 million scores just for one layer, one head. This intermediate matrix is written to memory, then read back to apply softmax, then read again to multiply with V.

Flash Attention computes attention in small tiles that fit inside fast on-chip memory (SRAM), doing the softmax accumulation incrementally as it scans through the tiles. The result is identical to standard attention — same outputs, same math — but the GPU/CPU never reads or writes the giant intermediate matrix to main memory. Memory usage drops from O(n²) to O(n), and the operation runs dramatically faster purely because it makes fewer round-trips to slow memory. This is why Flash Attention is not an approximation — it is a memory-access optimization that is especially effective when context is long.

## Step 1: Audio → Tokens (The Encoder)

Usually the models is trained on certain type of audio input, such as 16 kHz mono PCM waveforms. We have to convert our raw audio into this format before feeding it into the model. The first problem is converting this into something the LLM backbone understands.

For eg, for the audio model from [liquid AI expects](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) 16 kHz mono PCM waveforms as input. The encoder converts the raw audio into a sequence of embedding vectors. This is the **audio projector** — `mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf`, a 14-block Conformer network.

A Conformer is a hybrid of CNNs and multi-head attention, originally designed for speech recognition. It is good at capturing both local patterns (phonemes, individual sounds) and long-range temporal structure (prosody, rhythm) simultaneously. CNNs are efficient at the local structure; attention handles long-range.

The process:

```
Raw PCM: samples @ 16 kHz
    ↓
Mel filterbank: extract log-frequency features per 25ms window
    ↓
14-block Conformer (CNN + attention, 512-dim)
    ↓
17 audio embedding vectors (each 2048-dim — matches backbone's n_embd)
```

Those 17 vectors are injected into the backbone as if they were regular token embeddings. The backbone never sees raw audio. It sees learned representations — compressed, semantically rich summaries of the sound. Each of the 17 vectors is 2048-dimensional.

Then the backbone processes those embeddings through its own attention layers:

```
decoding audio batch 1/1, n_tokens_batch = 17
audio decoded (batch 1/1)
```

## Step 2: LLM Backbone Inference

The backbone is LFM2.5-Audio-1.5B — a 1.17B parameter model, quantized to Q4_0 (4.74 bits per weight, 661 MiB on disk).

LFM2 is a **hybrid architecture**. Not a pure transformer. It mixes:
- **SSM layers (Mamba-style):** Fast recurrent processing, O(1) per step, good for sequential patterns
- **Attention layers:** Selective attention at specific layers for global context
- Attention only at layers 2, 5, 8, 10, 11, 12, 13, 15 (the rest are SSM)

This is why only 6 layers have KV cache entries and why V shapes are non-uniform — the architecture is deliberately heterogeneous. Every decode step:

1. Reads most of the 661 MiB of weights from RAM
2. Runs Q4 dequantize + NEON DOTPROD matmul through 16 layers
3. Samples the next token

The bottleneck is memory bandwidth — the CPU has to read hundreds of megabytes of model weights per step. There is no way around this at CPU speeds.

The model interleaves text and audio output tokens. When it generates text tokens you get transcript words. When it switches to audio modality, those tokens feed the vocoder pipeline instead.

The model literally switches between "speak these words as text" and "speak these words as sound" mid-generation. Interleaved multimodal output.

---

## Step 3: Tokens → Audio (The Vocoder Pipeline)

This is the most expensive part of the whole pipeline — 76% of total inference time. And it is the least obvious step.

The backbone generates audio tokens — integers from the vocabulary space (65,536 possible values). An integer is not a sound wave. You need two more networks to get from integer to PCM.

### Stage A: Audio Detokenizer (70M params)

`tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf` — a smaller 8-layer LFM2 model (70M parameters).

Its job: take the backbone's audio token IDs and convert them into **spectral feature vectors** — continuous representations in the frequency domain.

```
Audio token IDs: [52341, 48221, 61034, ...]   ← integers from backbone vocab
    ↓  8-layer LFM2 transformer (70M params, flash attention enabled)
Spectral feature vectors: [...continuous f32...]   ← frequency-domain representation
```

### Stage B: Vocoder (iSTFT synthesis)

`vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf` — 85 tensors, a depthformer + iSTFT network.

The spectral feature vectors from the detokenizer represent the sound in the frequency domain — essentially a mel-spectrogram-like representation. The vocoder converts this back to a time-domain waveform using **inverse Short-Time Fourier Transform (iSTFT)**:

1. Treat each feature vector as magnitude + phase of frequency bins
2. Apply inverse Fourier transform per frame: freq domain → time domain
3. Overlap-add successive frames to produce a continuous PCM waveform

```
Spectral feature vectors (frequency domain)
    ↓  Depthformer + iSTFT
PCM samples @ 24 kHz
```

Each audio decode step produces 1920 samples = **80 ms of audio at 24 kHz**. From the logs:

```
audio decode step 1  → 1440 samples (total  1,440)
audio decode step 2  → 1920 samples (total  3,360)
audio decode step 3  → 1920 samples (total  5,280)
...
audio decode step 115 → total 218,400 samples
```

218,400 samples ÷ 24,000 Hz = **9.1 seconds of speech**. The response "Sure, did you know octopuses have three hearts? Two pump blood to the gills, and the third sends it to the rest of the body."

## The Complete Pipeline

Here is the full chain with actual timing from the logs:

```
WAV input
  │  Resample: 44.1kHz float32 → 16kHz s16le
  ↓
Audio Encoder / mmproj (14-block Conformer)          
  │   audio samples → 17 embedding vectors (2048-dim)
  ↓
Backbone prefill (LFM2 1.17B, 16 layers)
  │  System prompt + audio embeddings → KV cache (41 positions)
  ↓
Autoregressive decode loop 
  │
  ├─ TEXT tokens ────────────► transcript
  │
  └─ AUDIO tokens 
        ↓
     Audio Detokenizer (LFM2 70M, 8 layers)
        │  Token IDs → spectral vectors
        ↓
     Vocoder (iSTFT, 85 tensors) 
        │  Spectral → PCM frames @ 24kHz
        ↓
     Output ring → wav output
```

## What The First few Seconds Are Doing on bootup

The inference doesn't start until the model is loaded. From the logs, model load takes 6.45 seconds wall time:

| Sub-phase| What happens |
|---|---|
| Unload previous model | Free VLM weights from memory |
| Backbone GGUF parse | Read metadata, build tokenizer caches |
| Tensor repacking (CPU_REPACK) | Rearrange Q4_0 blocks for NEON DOTPROD |
| KV cache + context alloc | Allocate 48 MiB KV cache, compute graphs |
| Backbone warmup | One empty forward pass to prime JIT caches |
| mmproj load | Parse + mmap 650 Conformer tensors |
| Vocoder + detokenizer load | Load two more GGUF files, warmup |

The `CPU_REPACK` phase — is when the runtime (llama.cpp in my case) is rearranging Q4_0 quantized weights into a cache-friendly layout for ARM NEON. 652 MiB of weights being reorganized in-place. You pay this cost once per model load. The warmup pass is good optimization to do before any real inference begins. Without it, the first actual inference call would be slow and unpredictable as JIT caches cold-start mid-inference.

## The KV Cache Growth Problem

One subtle issue visible in the logs: the context is capped at 4096 tokens for the backbone.

```
llama_context: n_ctx = 4096
n_ctx_seq (4096) < n_ctx_train (128000) — the full capacity of the model will not be utilized
```

The model was trained on 128,000 token contexts. It is running with a 4096-token window to conserve the 48 MiB KV cache. Each attention step is `O(n)` in context length — longer context means more memory and more compute per token. On a device with 8 GB RAM and no GPU offload, 4096 is a pragmatic cap.

The audio detokenizer, interestingly, gets the full 128,000-token context:

```
llama_context: n_ctx = 128000  (detokenizer)
```

This is because the audio token sequences can be long and the detokenizer needs to see the full generated audio context to maintain coherent speech output.

## What I Take Away From This

Audio-to-audio LLM inference is genuinely harder than it looks from the outside. You are not running one model — you are running four (encoder, backbone, detokenizer, vocoder), chained sequentially, each with its own memory footprint and compute profile.

The backbone is not even the bottleneck. The vocoder is. 76% of inference time goes into converting audio token IDs into PCM frames, one 80ms chunk at a time, one neural network forward pass per chunk.

On CPU without GPU acceleration, real-time audio synthesis from a neural vocoder is essentially impossible for a 1.5B-parameter model. You are 10× below real-time. The architecture is right, the implementation is correct — the hardware constraint is the wall.

When the OpenCL backend lights up on the actual hardware, everything changes. 20× faster decode means the vocoder drops from 830 ms/frame to ~40 ms/frame — comfortably faster than real-time. That is the gap between "demo that works" and "production assistant you can actually talk to."
