# FP8 KV-cache support for inkling Triton rel-attention — investigation findings

Status: investigated 2026-08-04 (todo 1). These decisions gate the implementation
todos (2–6).

## Cache layout (CONFIRMED)
- `get_kv_cache_spec` returns `FullAttentionSpec`/`SlidingWindowSpec` with
  `dtype=self.kv_cache_torch_dtype`.
- `kv_cache_dtype_str_to_dtype("fp8")` → `torch.uint8` (v1 stores float8_e4m3fn
  typed as uint8). Layout is the standard contiguous form, NOT transposed:
  `kv_cache: [num_blocks, num_kv_heads, block_size, 2*head_size]`; `_split_kv_cache`
  does `transpose(1,2).split(head_size, dim=-1)` → K,V each
  `[num_blocks, block_size, num_kv_heads, head_size]`. Matches flex/flash layout.
- Plain `"fp8"` uses a **per-tensor scalar** scale (no inline per-token scales —
  those are `fp8_per_token_head`). inkling's scalar `k_scale`/`v_scale` buffers
  are the right shape.

## Scale passing — CAPTURE-SAFE (DECIDED)
- Pass `k_scale`/`v_scale` as **device tensor pointers**; load inside the kernel
  with `tl.load(k_scale)`. NO `.item()` host sync → cudagraph-safe (user runs
  enforce_eager=False).
- Canonical reference: `vllm/v1/attention/ops/chunked_prefill_paged_decode.py:195`
  ```python
  K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
  ```
- Use compile-time `K_load.dtype.is_fp8()` constexpr branch so the dequant path
  is pruned for the bf16 codepath (zero overhead, bit-identical non-fp8 behavior).

## READ-side dequant (triton_rel_attention.py)
- Decode `_inkling_rel_attn_decode_partial`: K load (~L137) then `.to(tl.float32)`,
  multiply by `tl.load(K_SCALE)` when fp8. V load (~L146) MUST be cast to fp32
  always (currently left in cache dtype) and multiplied by `tl.load(V_SCALE)` when
  fp8 — needed so the `tl.sum(p[:,None]*v)` PV math is float on SM8x.
- Prefill `_inkling_rel_attn_prefill`: K/V load (~L370-379) dequant+scale; fix
  L423 `tl.dot(p.to(v.dtype), v)` → `tl.dot(p, v)` with V in fp32.
- Add `KV_IS_FP8` tl.constexpr + `K_SCALE_PTR`/`V_SCALE_PTR` args. Non-fp8 branch
  must be unchanged.

## WRITE-side quantize (qkvr_prep.py `fused_qkvr_prep`)
- inkling writes K/V inside the fused Triton kernel (`fused_qkvr_prep`,
  `tl.store(key_cache_ptr...)` ~L386 and ~L596), NOT via reshape_and_cache_flash.
- Canonical fp8 write: `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:110`
  ```python
  key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)
  # tl.store implicit-casts to fp8 from key_cache_ptr.dtype
  ```
- Change both store sites: if `CACHE_IS_FP8` constexpr, compute
  `k_store = k_normalized / tl.load(K_SCALE)` (V likewise) before `tl.store`; the
  pointer dtype (uint8 viewed as fp8) does the cast. Non-fp8 path unchanged.
- Pass k_scale/v_scale tensors from attention.py `forward()` into fused_qkvr_prep.

## Scale calibration (RESOLVED: default 1.0, no calibration needed)
- MODERN vLLM: `cache_config.calculate_kv_scales` is DEPRECATED
  (config/cache.py:266 validator warns; removed in v0.19). New behavior:
  scales loaded from checkpoint if available, else **default to 1.0**.
- `gpu_model_runner.init_fp8_kv_scales` only processes
  `isinstance(Attention, MLAAttention)` and resets to 1.0; inkling extends
  AttentionLayerBase so it's skipped — but it only resets to 1.0 anyway, and
  inkling's `k_scale`/`v_scale` buffers are already `torch.ones(...)` (1.0).
- DECISION: reuse inkling's existing 1.0 default buffers — NO calibration
  machinery added. Matches modern vLLM fp8 behavior. K/V are post-RMSNorm
  (bounded magnitude), so scale=1.0 casts to fp8 with graceful clipping to
  e4m3 max=448, the standard behavior.
- If a user later needs calibrated scales, that's checkpoint-loading scope,
  not runtime calibration. Per-tensor scalar (not per-token) — matches buffers.

Verified: inkling k_scale/v_scale = torch.ones((), float32) scalar buffers at
attention.py:185-186; passed to both fused_qkvr_prep (write) and
inkling_fa4_rel_attention (read). No code change needed for todo 5 beyond
confirming the buffers exist and default to 1.0 (they do).

## fp8 math note
- e4m3 max = 448 (FP8_MAX from get_fp8_min_max). K_RANGE=200 → scale chosen so
  max|K|/200 maps into fp8 range with headroom; the exact constant is vLLM's
  convention, reuse as-is for consistency.

## Tests
- Extend tests/test_fa4_rel_attention.py: force_triton_fallback fixture + an fp8
  kv path (kv_cache_torch_dtype=uint8, kv_cache_dtype="fp8", populate scales via
  max(|k|)/200). Compare Triton output vs _ref_rel_attn on an fp8 round-tripped
  cache. Cover decode, chunked prefill, sliding window.
- No CUDA on Windows dev box → verify via ast.parse locally; user runs GPU tests.
