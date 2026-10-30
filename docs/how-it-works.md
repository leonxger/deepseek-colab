# How it works

DeepSeek V4.1 Flash has 552 billion parameters in 40 layers, plus a 196-billion-parameter Engram lookup table. Even at 2 bits that is 249 GB, and the biggest single GPU on Colab has 96 GB. This page explains how it still runs at over 100 tokens a second.

## The files

| What | Source (pinned revision) | Size |
| --- | --- | --- |
| Backbone, GSQ-RCO IQ2_XXS (2.0 bits per weight on average) | ISTA-DASLab, shards 1-5 | 138.2 GB |
| Engram table, IQ4_NL | ISTA-DASLab, shard 6 | 110.6 GB |
| Uncensored tensors (82: `attn_output_b` and `ffn_down_shexp` of all 40 layers, and 2 `engram_wkv`) | AtomicChat, by HTTP range reads | 1.7 GB |
| DSpark draft model, Q8_0 | unsloth | 8.0 GB |
| Image encoder (mmproj, BF16) | ISTA-DASLab | 1.0 GB |

AtomicChat's uncensored files are ISTA-DASLab's shards in an older layout, with exactly those 82 tensors replaced by tensors of the same type and size. `vm/mkpatch.py` reads only those bytes from AtomicChat's repo, and `vm/applypatch.py` writes them over ISTA's tensors at their offsets, checking every sha256. The result is the uncensored model in the current layout without a second 249 GB download. `UNCENSOR=0` skips the splice.

## Where every byte lives

The G4 VM has an RTX PRO 6000 Blackwell (96 GB), 190 GB of RAM and a local disk that writes at only about 250 MB/s. So the weights go into a RAM disk, never onto the disk.

| GPU, 96 GB | RAM, 190 GB |
| --- | --- |
| attention, dense layers, shared experts | the Engram table (110.6 GB) |
| expert cache: 7,814 hottest experts, pinned (66.8 GB) | the other experts, pinned for fast copies (65.2 GB) |
| dynamic cache slots for misses (110 per group) | |
| draft model, KV cache for 2 x 262K, compute buffers | |
| image encoder (when image input is on) | at least 2.1 GB stays free |

Each token picks 6 of the 384 routed experts in every layer, and the choice is skewed. The cache is filled in the order of `vm/profile-IQ2_XXS.txt`, a hotness ranking measured on real traffic. Once those experts are on the GPU, their RAM copies are punched out of the RAM disk (`vm/punch.py`), which is what makes room for the Engram table. On a miss, the expert is copied from pinned host memory into a dynamic slot and computed on the GPU, so all expert math happens on the GPU.

RAM is the binding limit. The cache size and slot count of the two layouts in `LAYOUTS` (`llm.py`) were measured so that RAM never drops below 2 GB, even with a 245K-token prompt.

## The boot sequence

`vm/boot.sh` writes a timeline that `llm.py logs` shows:

1. In parallel: download the backbone (64 parallel range requests, about 3.5 GB/s from Hugging Face's CDN) and the draft model, fetch the uncensored tensors and the image encoder, and build llama.cpp (`vm/build.sh`, about 2 minutes on the VM's CPUs).
2. Check the shard sizes and apply the splice. The Engram file starts as an empty sparse file with its real header.
3. Start llama-server. Once it is healthy, verify that the weights sit on the GPU, then free their RAM copies and the resident experts.
4. Pin the cold experts with `cudaHostRegister` (`vm/pin.c`, preloaded into the server), so misses copy at PCIe speed.
5. Download the Engram table into the hole that the punching left.
6. Send one warmup request, then start `vm/gate.py` on port 18080. The tunnel from `llm.py` can connect at any time, but only reaches the server after this point, so no request ever sees a half-filled Engram table.

## Why decoding is fast

- **Speculative decoding.** The DSpark draft model proposes 2 tokens. The big model verifies both in one pass and keeps the ones it agrees with, on average 2.25 tokens per step. Verification is exact.
- **One CUDA graph per step.** The expert-cache lookup runs on the GPU, so a verify step has no host round trip per layer.
- **Fewer, fused kernels.** The hyper-connection chain, the gating weights and several gathers are fused.
- **Copies that read whole memory lines.** GGUF aligns tensors to 32 bytes, so 86% of the cold expert bytes start in the middle of a 128-byte line, and copies from such addresses ran at 33 GB/s. Copies that start on line boundaries run at 43-54 GB/s.
- **Copy engine for misses** (text-only layout only). The DMA engine copies misses from aligned sources while the GPU computes. With the image encoder loaded, an image prompt stalls this path, so the other layout uses the copy kernel instead.

Prefill uses DeepSeek's encoder-decoder structure (CED). Prompt tokens before the last 2,414 positions only need the 20 encoder layers, which makes the first token 2-2.5x faster. The result is exact: 2,414 tokens is the full reach of the stacked sliding windows.

## The patches

`vm/build.sh` applies `vm/patches/series` in order to llama.cpp b11963 at a pinned commit. The step times are milliseconds per verify step on identical texts, measured on one VM each.

| Patch | What it does | Measured |
| --- | --- | --- |
| `apply_pr30966.py` | upstream PR #30966: a GPU cache for experts that live in host memory | base of the layout |
| `patch_e.py` | fills the cache in hotness order and pins those experts, lists them for punching | base of the layout |
| `patch_ced.py` | exact encoder-only prefill for early prompt tokens | first token 2-2.5x faster |
| `patch_cedspec.py` | keeps that prefill on with the draft model | 32K first token 45 to 28 s |
| `patch_dsparkfix.py` | stops a draft cleanly on an empty candidate list | crash fix |
| `patch_fuse.py`, `patch_mmvf.py`, `patch_fuse2.py` | verify-step fusions, split-K mat-vec for long rows | 3,368 to 2,723 kernels per step |
| `patch_prefetch.py`, `patch_predict.py`, `patch_peakdump.py` | expert prefetch, prediction, buffer diagnostics; off at runtime | kept because later patches build on them |
| `patch_copybatch.py` | one gather kernel per expert copy in large batches | prefill |
| `patch_devcache.py` | expert-cache lookup on the GPU, one graph per step | +4.1% decode |
| `patch_hcfuse.py` | hyper-connection chain in 2 kernels instead of 7 | -2.75% per step |
| `patch_cedmm.py` | keeps the fast prefill on for text prompts while the image encoder is loaded | 32K first token 38.7 to 24.8 s |
| `patch_cbdiet.py` | 2.4 GB smaller compute buffer, which became more cache slots | +4.6% decode |
| `patch_asyncin.py`, `patch_chunk1.py`, `patch_seqrm.py` | no host syncs at the step boundary | 20.50 to 20.20 ms |
| `patch_align.py` | expert copies read whole 128-byte lines | 20.19 to 19.83 ms |
| `patch_dma.py`, `patch_dmaalign.py` | misses by the copy engine from aligned sources (`--fast`) | 19.72 to 18.79 ms |

## What "lossless" means here

Every speedup was measured against the build before it, on the same weights:
- a greedy comparison of 25 prompts (KL divergence of the next-token distributions, at most 0.0025, the noise floor of two parallel slots);
- a 10-task coding quality check (10/10);
- six fixed long answers, which must be token-identical or differ only at near-ties.

The 2-bit quantization itself is the one loss against the original FP8/FP4 model. ISTA-DASLab's IQ2_XXS matched the original on their AIME25 (100.0), GPQA-Diamond (90.9) and LiveCodeBench v6 (89.7) runs at reasoning effort 50, before the uncensored splice.

## Limits

- Two requests run at a time, each with up to 262,144 tokens (131,072 with `--fast`).
- Decode gets slower with context: about 4% per step at 126K and 7% at 250K, because the indexer scores every compressed row each step.
- Every start downloads again: a Colab VM keeps nothing between sessions.
- The model files come from third-party Hugging Face repos at pinned revisions. If one disappears, starts fail at the download step.
