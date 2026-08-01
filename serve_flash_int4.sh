#!/usr/bin/env bash
# serve_flash_int4.sh — Serve Intel/DeepSeek-V4-Flash-W4A16-AutoRound (INT4, ~145 GiB)
# on the ampere (SM8x) backend of this patched vLLM tree.
#
# Target: 8x RTX 3090 24GB (192 GB total). All weights resident (offload=0).
#
# ##################################################################################
# STATUS: UNVALIDATED. Read DSV4_AMPERE.md before relying on it.
# ##################################################################################
# The legacy int4 patches' ampere-specific value (K12 indexer logits, K13 split-K
# sparse decode, o-proj Marlin bypass, ue8m0 scale_fmt) is ALREADY in this tree's
# ampere/ backend — so no extra code patches are needed for int4. The W4A16
# checkpoint loads via vLLM-core: quant_method="auto-round" is claimed by
# INCConfig.override_quantization_method() -> the INC scheme, which runs experts +
# quantized linears on hardware Marlin WNA16 (int4). The custom DeepseekV4 layers
# (wo_a, compressor/indexer weights) are kept bf16 by the checkpoint's
# extra_config, so no DeepseekV4-specific quant handling is required.
#
# PREREQ: NONE beyond the main build. On NVIDIA/CUDA the W4A16 checkpoint runs
# entirely on Marlin (AutoGPTQ linear + MoE methods); the INC scheme only touches
# auto_round_kernel on XPU/CPU. Do NOT `pip install auto_round_kernel` — it pins
# torch==2.9.1 and will downgrade your torch 2.11.0, breaking vLLM. See DSV4_AMPERE.md §3.
#
# KNOWN RISK (unverified): the model's fp4 weight-name mapper is selected because
# the W4A16 config still declares expert_dtype="fp4". The checkpoint's GPTQ keys
# (.qweight/.qzeros/.scales, PLURAL) bypass the mapper's `.scale$` (singular)
# regex, and the prefix map (layers. -> model.layers.) still applies — so loading
# should be clean. First launch is the real test.
# ##################################################################################
set -u

cd "$(git rev-parse --show-toplevel)"
export PATH=".venv/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$HOME/.local/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CXX=clang++ CC=clang
export HF_HUB_DISABLE_XET=1
export VLLM_MXFP4_USE_MARLIN=1            # harmless for pure int4; needed if any MXFP4 path

EP_FLAG="--enable-expert-parallel"; [ "${EXPERT_PARALLEL:-1}" = "0" ] && EP_FLAG=""
PC_FLAG="";  [ "${PREFIX_CACHE:-1}" = "0" ] && PC_FLAG="--no-enable-prefix-caching"
ASYNC_FLAG=""; [ "${ASYNC_SCHED:-1}" = "0" ] && ASYNC_FLAG="--no-async-scheduling"
EAGER_FLAG=""; [ "${EAGER:-0}" = "1" ] && EAGER_FLAG="--enforce-eager"
CC_FLAG=""
if [ "${CG_MODE:-FULL}" = "FULL" ]; then
  export VLLM_USE_BREAKABLE_CUDAGRAPH=0
  # No cudagraph_capture_sizes override: vLLM already bounds the capture set to
  # the actual max decode batch. For --max-num-seqs 1 that is just [1, 2]
  # (max_cudagraph_capture_size = min(max_num_seqs*2, 512) = 2); the docstring's
  # "[1,2,4,8,...,248]" is outdated -- the real code filters every step by
  # max_cudagraph_capture_size. So nothing to trim here.
  CC_FLAG='--compilation-config={"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
fi

MODEL="${MODEL_PATH:-Intel/DeepSeek-V4-Flash-W4A16-AutoRound}"
OFFLOAD="${CPU_OFFLOAD_GB:-0}"
GMU="${GPU_MEM_UTIL:-0.93}"

exec .venv/bin/vllm serve "$MODEL" --trust-remote-code \
  --served-model-name DeepSeek-V4-Flash \
  --tokenizer-mode deepseek_v4 \
  --tensor-parallel-size 8 $EP_FLAG $PC_FLAG $ASYNC_FLAG \
  --kv-cache-dtype fp8 --block-size 256 \
  --gpu-memory-utilization "$GMU" --cpu-offload-gb "$OFFLOAD" \
  --max-num-seqs 1 \
  --max-model-len "${MAXLEN:-32768}" \
  --max-num-batched-tokens "${MAXBATCH:-1024}" \
  --enable-chunked-prefill \
  $EAGER_FLAG $CC_FLAG \
  --host 0.0.0.0 --port "${PORT:-8001}"
