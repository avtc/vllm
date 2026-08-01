# DeepSeek-V4-Flash on Ampere (SM8x) — this branch

This branch (`feat/patched-vllm-0.25-dsv4-ampere`) is **vLLM main @ `5c4db60f`
(v0.25.0 line)** plus the **Ampere (SM8x) backend** ported from
[Lasimeri/vllm-dsv4-ampere](https://github.com/Lasimeri/vllm-dsv4-ampere)
(main-port overlay). It lets **DeepSeek-V4-Flash** run on Ampere GPUs
(A100 SM80 / RTX 30xx SM86) — which upstream vLLM does **not** support, because
DSv4-Flash depends on Hopper-only kernels (FlashMLA-sparse, cutedsl, DeepGEMM
FP8/FP4, TileLang MHC, native fp8e4nv). The `ampere/` backend replaces each of
those with an SM86-compatible Triton kernel or PyTorch reference.

It can serve **two checkpoints**:
1. **`deepseek-ai/DeepSeek-V4-Flash`** — the native **FP4 + FP8 + BF16** model (~158 GiB).
2. **`Intel/DeepSeek-V4-Flash-W4A16-AutoRound`** — **INT4 W4A16** (GPTQ experts, ~145 GiB).

> **Status: the FP4+FP8 path is the author-validated flagship (on 8x RTX 3080
> 20GB). The INT4 path is structurally supported on this tree but UNVALIDATED at
> runtime — see "INT4 notes" below.** Both are tuned here for **8x RTX 3090 24GB
> (192 GB total)** with all weights resident.

---

## 1. Prerequisites (on the Linux 8x3090 server)

vLLM does **not** build/serve on Windows — do the build and serving on the Linux
box with the GPUs. (This repo can be cloned/committed on Windows, but build+run
is Linux.)

- **GPU**: 8x Ampere SM8x (RTX 3090 SM86 / A100 SM80), ~192 GB total VRAM.
- **NVIDIA driver** supporting CUDA 12.x; **CUDA toolkit 12.x** (`nvcc`).
- **Python 3.12**, **`uv`** (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **`clang`** (FlashInfer's runtime JIT rejects gcc>=15 headers): `apt install clang`.
- **`ninja`** (FlashInfer JIT build tool) — pulled in by the venv, ensure it's on `PATH`.
- **`git`**, and a HF token only if you hit rate limits (both models are public).

## 2. Get this branch onto the server & build (Python-only, ~minutes)

```bash
# Option A: if you already have a vllm clone, fetch this branch.
git clone <your-vllm-remote> vllm && cd vllm
git checkout feat/patched-vllm-0.25-dsv4-ampere

# Option B: from the Windows checkout, push to a remote the server can reach,
#           then checkout on the server. (This branch is NOT on upstream.)

# This branch pins torch == 2.11.0 (verified in pyproject.toml build-system and
# CMakeLists.txt TORCH_SUPPORTED_VERSION_CUDA). For a FIXED-DRIVER / CUDA-12.8 box,
# do a SOURCE build (below) rather than VLLM_USE_PRECOMPILED=1: the prebuilt
# binaries are pinned to the CI's torch/driver and can mismatch yours. The ampere
# backend is pure Python + Triton, so this only compiles the vLLM *core* C++/CUDA
# extension (fast).
python3.12 -m venv .venv
source .venv/bin/activate

# 1. torch 2.11.0 from the CUDA 12.8 index (matches CUDA 12.8 toolkit/driver)
pip install --force-reinstall --no-deps \
     torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
     --index-url https://download.pytorch.org/whl/cu128

# 2. build tools (versions required by vLLM's pyproject)
pip install "cmake>=3.26" ninja "setuptools>=77,<81" setuptools-scm setuptools-rust

# 3. build the core extension for SM 8.6 ONLY (3090) to cut build time
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=8          # raise if you have >128 GB RAM
pip install -e . --no-build-isolation
```

Because vLLM is installed **editable** and the ampere files are committed into
the tree, they are live — no rebuild is needed when you switch checkpoints.

> Need a different torch/CUDA? This commit supports `torch == 2.11.0` exactly.
> `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto` is a faster
> alternative (no compile) but uses CI-pinned torch/driver binaries — avoid it on
> a fixed-driver box.

## 3. INT4 note: you do NOT need `auto_round_kernel` on NVIDIA

The W4A16 checkpoint declares `quant_method: "auto-round"`, which vLLM's
`INCConfig` claims. But on **CUDA/NVIDIA**, the INC scheme routes every
quantized layer straight to **Marlin** (`AutoGPTQLinearMethod` for linears,
`AutoGPTQMoEMethod` for experts) — verified in
`vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_scheme.py`
(the ARK/`auto_round_kernel` path is gated behind `is_xpu()`/`is_cpu()` only).
So **do not** `pip install auto_round_kernel`: it pins `torch==2.9.1` and will
**downgrade your torch 2.11.0 → 2.9.1, breaking vLLM**. No extra install is
needed for INT4 on this branch.

---

## 4. Serve the **FP4+FP8** checkpoint (author-validated path)

```bash
bash serve_flash_fp4.sh
# first launch downloads ~158 GiB from HuggingFace (set HF_HOME if needed)
```

What it does: TP=8, expert-parallel, fp8 KV cache, block 256, `--cpu-offload-gb 0`,
`--gpu-memory-utilization 0.93`, FULL decode cudagraph (`FULL_DECODE_ONLY`),
chunked prefill, 32k context. The SM8x dispatch activates automatically
(`deepseek_v4/__init__.py` routes `capability < 9.0` → `ampere/`).

Test it:
```bash
curl http://localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "DeepSeek-V4-Flash",
  "messages": [{"role":"user","content":"Say hello."}]
}'
```

## 5. Serve the **INT4 (W4A16 AutoRound)** checkpoint

```bash
bash serve_flash_int4.sh        # no extra install needed (Marlin on NVIDIA; see §3)
```
Same shape as the FP4 script but targets `Intel/DeepSeek-V4-Flash-W4A16-AutoRound`.

---

## 6. Tuning knobs (env vars before the serve script)

| Var | Default | Effect |
| --- | --- | --- |
| `CPU_OFFLOAD_GB` | `0` | GiB of weights offloaded to CPU **per GPU**. Raise if the profiling phase OOMs (each GiB/GPU adds ~16 ms/token → ~0.5 tok/s). |
| `GPU_MEM_UTIL` | `0.93` | `--gpu-memory-utilization`. Lower if you OOM; 0.88-0.95 typical. |
| `MAXLEN` | `32768` | `--max-model-len`. Lower to free KV; raise toward 1M only with DCP. |
| `CG_MODE` | `FULL` | `FULL` = whole decode step as one cudagraph (fastest). `BREAKABLE` = upstream piecewise (safer). |
| `EAGER` | `0` | `1` = `--enforce-eager` (no cudagraphs). Bisect first if output looks wrong. |
| `EAGER=1` then `CG_MODE=BREAKABLE` then `CG_MODE=FULL` | — | Order to bisect cudagraph-capture problems. |
| `MAXBATCH` | `1024` (int4 script) | `--max-num-batched-tokens`. **Biggest KV lever.** The activation peak reserved during profiling scales ~linearly with this. On 8x3090 INT4, 512 -> ~49K KV tokens; **256 -> ~150K**, 128 -> ~220K. Trade: slower prefill (smaller chunks); decode speed unchanged. |
| `TRITON_SPARSE_MLA` | `0` | `1` = route **decode** attention through the fused vLLM-Moet Triton sparse-MLA port (reads fp8_ds_mla pages, dequants in-register, no flat bf16 workspace) instead of the default two-stage ampere kernel. Same output, different speed/VRAM profile - A/B compare with a fixed prompt. |
| `ENABLE_NCCL_TUNE` | `0` | `1` = trim NCCL/cuBLAS buffers (`NCCL_BUFFSIZE=2MB`, 1 channel). Reclaims ~0.05-0.10 GiB/GPU -> ~6-12K extra KV tokens. Most non_torch (~0.45 GiB) is Triton CUmodules + CUDA context (irreducible). |
| `DCP` | `1` | `>1` enables decode-context-parallel (`--decode-context-parallel-size N --dcp-comm-backend a2a`) for long context. Experimental. |

**OOM during profiling:** the first step is to raise `CPU_OFFLOAD_GB` in steps of
4-8. On 8x3090 you have ~20-34 GB (FP4+FP8) / ~47 GB (INT4) of headroom after
weights, so offload=0 should hold for 32k; longer context may need a few GB offload.

### Measured on 8x RTX 3090 (INT4, block-size 256, GMU 0.965, FULL capture, MAXBATCH 512)
- Weights ~19.1 GiB/GPU; cudagraph ACTUAL = **0.07 GiB** (the 1.44 GiB estimate is wildly off;
  this branch's `[VRAM]` logging reads the real capture cost). KV pool budget ~0.40 GiB ->
  **~49K KV tokens**.
- The packed KV block is ~1 MiB (79% main MLA, 17% indexer/compressor, 3% SWA) - near-optimal
  for DSv4's multi-cache architecture; cannot improve without dropping layer types.
- **`non_torch` ~1.4 GiB is mostly irreducible**: 0.43 GiB CUDA context + ~0.45 GiB Triton
  CUmodules + ~0.56 GiB NCCL/cuBLAS/JIT during forward. NCCL tuning reclaims only ~0.05-0.10 GiB.
- **The dominant lever is `MAXBATCH`** (the activation peak, ~1.86 GiB at 512): 512->256 halves
  it and ~3x's the KV pool. Do this before anything else.
- **vLLM-Moet Triton sparse-MLA port** (`TRITON_SPARSE_MLA=1`): fused decode kernel, may be
  faster and lower-VRAM than the default two-stage ampere decode. Layout-compatible (same
  fp8_ds_mla 584-byte packed cache). A/B compare on a fixed prompt - measure tok/s and check
  the `[VRAM]` available_kv line for a KV difference too.


## 7. Expected performance (honest, unvalidated on 8x3090)

- **Author-validated reference (8x RTX 3080 20GB, offload 15):** FP4+FP8 ≈ **4.4 tok/s**
  (FULL capture). That config streams ~80% of weights over PCIe every step.
- **On 8x3090 with offload=0:** decode should be **notably faster** — the per-step
  PCIe weight-streaming floor is removed, and the 3090 has ~23% more memory
  bandwidth (936 vs 760 GB/s). No measured number is available; expect single-digit
  to low-teens tok/s, bounded by the emulated sparse-MLA/indexer attention (which
  has no native Ampere kernel and is the decode floor on every branch).
- **INT4 vs FP4+FP8:** on the legacy 8x3080 measurement, INT4 (Marlin int4 experts,
  no software dequant) edged out FP4+FP8. Expect similar or slightly faster on 8x3090.
- Prefill is **slow** on this port (Python-vectorized gather); TTFT will be high.

## 8. Caveats

- Not upstream-mergeable (SM<90 is explicitly out of scope for vLLM).
- Pinned to vLLM `5c4db60f` (v0.25.0). Do **not** rebase onto newer main without
  re-reviewing the ~26 modified files — `gpu_model_runner.py` and the MLA/KV
  internals churn weekly upstream.
- `get_dcp_local_seq_lens` was relocated `v1/attention/backends/utils.py` →
  `v1/worker/cp_utils.py` by this branch; one stale test import was fixed.
- Two latent footguns if you modify the KV/attention path (documented upstream in
  the source repo): `.reshape(-1)` on the pooled KV-cache view silently drops
  writes; `-inf` sentinels NaN-poison online softmax on fully-masked tiles.

---

## INT4 notes (why no separate int4 code patch was applied)

The legacy `int4-autoround-support` branch's patches target an **older vLLM
layout** (single-file `model_executor/models/deepseek_v4.py`, base `c2fb0133`,
2026-04-30) and **cannot apply** here. Investigation showed the int4-specific
value is already absorbed into this branch:

| Legacy int4 patch contribution | Present on this branch |
| --- | --- |
| SM86 indexer logits kernel (K12) | `ampere/ampere_indexer_logits.py` |
| SM86 split-K sparse MLA decode (K13) | `ampere/ampere_sparse_splitk.py` |
| o-proj Marlin bypass | `model_executor/layers/quantization/fp8.py` (`is_bmm` guard) |
| ue8m0 scale_fmt for AutoRound | hardcoded `"ue8m0"` in `models/deepseek_v4/attention.py:716` |

The W4A16 checkpoint's `quant_method: "auto-round"` is claimed by vLLM-core
`INCConfig.override_quantization_method()` → experts + quantized linears run on
hardware Marlin WNA16 (int4); the custom DeepSeek-V4 layers (wo_a, compressor,
indexer weights) are kept bf16 by the checkpoint's `extra_config`, so no
DeepSeek-V4-specific quant handling is required. The checkpoint's GPTQ keys
(`.qweight`/`.qzeros`/`.scales`, plural — verified from its safetensors index)
bypass the model's fp4 weight-mapper (which only matches `.scale$` singular).

**What to check on first INT4 load:** if it fails in weight loading, the likely
culprit is the interaction between the fp4 weight-name mapper and AutoRound's
key layout — that is the one statically-unverified seam. If it loads but output
is wrong, set `EAGER=1` first (bisect: eager → breakable → full).
