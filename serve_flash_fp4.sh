#!/usr/bin/env bash
# serve_flash_fp4.sh — Serve deepseek-ai/DeepSeek-V4-Flash (native FP4+FP8, ~158 GiB)
# on the ampere (SM8x) backend of this patched vLLM tree.
#
# Target: 8x RTX 3090 24GB (192 GB total). All weights resident (offload=0).
#
# STATUS: adapted from the main-port launch_e4.sh for 8x3090 (offload 0, GMU 0.93).
# The author validated the ampere backend on 8x RTX 3080 20GB at ~4.4 tok/s with
# offload 15 (80/20 CPU/GPU). With offload=0 on 8x3090, decode should be faster
# (no per-step PCIe weight streaming) but is UNVALIDATED — tune to your box.
set -u

cd "$(git rev-parse --show-toplevel)"
export PATH=".venv/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$HOME/.local/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CXX=clang++ CC=clang               # FlashInfer JIT needs clang (gcc>=15 rejects torch headers)
export HF_HUB_DISABLE_XET=1               # Xet CAS can stall; classic HTTP is reliable
export VLLM_MXFP4_USE_MARLIN=1            # Marlin is the only MXFP4 backend < SM90 (DeepGEMM/TRTLLM need SM>=90)
export VLLM_SM86_SINK="${VLLM_SM86_SINK:-0}"       # post-hoc attention-sink fold (debug)
export VLLM_SM86_NAN_PROBE="${VLLM_SM86_NAN_PROBE:-0}"

EP_FLAG="--enable-expert-parallel"; [ "${EXPERT_PARALLEL:-1}" = "0" ] && EP_FLAG=""
PC_FLAG="";  [ "${PREFIX_CACHE:-1}" = "0" ] && PC_FLAG="--no-enable-prefix-caching"
ASYNC_FLAG=""; [ "${ASYNC_SCHED:-1}" = "0" ] && ASYNC_FLAG="--no-async-scheduling"
EAGER_FLAG=""; [ "${EAGER:-0}" = "1" ] && EAGER_FLAG="--enforce-eager"
CC_FLAG=""
if [ "${CG_MODE:-FULL}" = "FULL" ]; then
  export VLLM_USE_BREAKABLE_CUDAGRAPH=0
  # With --max-num-seqs 1, decode is always batch=1, so capture ONLY that size.
  # The vLLM default ([1,2,4,8,16,...,248] ~= 33 graphs) wastes 2-3 GiB/GPU of
  # cudagraph memory on batch sizes that never run -- memory better spent on KV.
  # Override via CG_CAPTURE_SIZES (JSON array) + matching CG_MAX_CAPTURE_SIZE.
  CCS="${CG_CAPTURE_SIZES:-[1]}"
  MCS="${CG_MAX_CAPTURE_SIZE:-1}"
  CC_FLAG="--compilation-config={"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":${CCS},"max_cudagraph_capture_size":${MCS}}"
fi

MODEL="${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Flash}"
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
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  $EAGER_FLAG $CC_FLAG \
  --host 0.0.0.0 --port "${PORT:-8001}"
