# KVarN + inkling relative-attention viability assessment

Status: investigated 2026-08-04 (todo 8). Verdict: **NOT VIABLE without major
work — effectively a new backend, disproportionate for the user's goal.**

## What the user asked
Serve inkling-small on 8x RTX 3090 with `--kv-cache-dtype kvarn_k4v2_g128`.
Current failure (log-3): the Triton prefill PV matmul `tl.dot(p.to(v.dtype), v)`
crashes with "only int8 supported!" because KVarN stores the KV cache as packed
uint8 tile records, which inkling reads raw with no dequant.

## How KVarN actually works (verified in both repos)
KVarN is a *complete attention backend*, not a drop-in cache format:
- **Cache layout**: packed tile record per (block, head). For head_dim=128,
  k4/v2/g128: ~17920 bytes/block = packed K(8192) + K scales/zp + V(8192) +
  V scales/zp, reinterpreted as `(num_blocks, block_size=128, num_kv_heads,
  140)`. NOT the `(num_blocks, block_size, num_kv_heads, head_dim)` float
  layout inkling uses.
- **Write model**: NOT per-token. `do_kv_cache_update` buffers incoming fp16
  K/V in a per-block staging dict; when a block fills to 128 tokens it runs
  Hadamard rotation + Sinkhorn variance-normalization + asymmetric RTN
  (`kvarn_store_tile_{k,v}`) and writes the packed record. (Per-token writes
  are impossible — the tile is quantized as a unit.)
- **Read/decode** (`kvarn_decode_attention`, triton_kvarn_decode.py:811):
  1. Rotate Q: q·H.
  2. Dequant cached blocks → fp16 K', V' (rotated frame) + sink/tail.
  3. `flash_attn_varlen_func(q', K', V', causal=False)` — **plain, NO bias**.
  4. Un-rotate output: out·H.
- **Backend selection**: inkling's `InklingAttention` extends
  `AttentionLayerBase`, hardcodes `get_attn_backend()→FlashAttentionBackend`,
  allocates its own float paged cache, and runs its own FA4/Triton kernel.
  The KVarN backend is **never selected** for inkling layers.

## The relative-bias question: is it mathematically compatible?
**Yes.** Hadamard H is orthonormal, so QK^T is rotation-invariant:
`(qH)·(kH)^T = q·k^T`. The rel bias depends only on (q_pos, k_pos), not on the
K basis, so applying it in the rotated frame preserves the math:
`softmax(scale·QK^T + rel_bias)·(vH) = [softmax(...)·v]·H`, un-rotated → exact.
So a hybrid (dequant KVarN tiles → fp16 K'/V' → run inkling rel-attn → un-rotate)
is *mathematically* sound. (No prior art in either repo: grep for relative/score_mod/
bias in the KVarN fork returns only the generic flex_attention backend.)

## Hard blockers (why "not viable now")
1. **block_size constraint.** KVarN asserts `block_size == cfg.group ∈ {64,128}`
   (kvarn_attn.py:236; `get_supported_kernel_block_sizes`→{64,128}). Inkling's
   sliding/local layers and the SW block-skip optimization rely on block_size=16.
   KVarN sliding support (`TQSlidingWindowSpec`, the merged code) still forces
   block_size=128. Unverified whether 128-block SW even works for inkling.
2. **Write model mismatch.** Inkling writes K/V per-token inside `fused_qkvr_prep`
   (fused sconv+rmsnorm+cache-write). KVarN needs 128-token tile buffering +
   flush. These cannot coexist in inkling's fused kernel — KVarN's write would
   have to replace fused_qkvr_prep's cache-write stage entirely.
3. **No bias hook in KVarN decode.** `kvarn_decode_attention` calls plain
   `flash_attn_varlen_func` (causal=False, no bias). Adding rel bias needs
   either FA4 score_mod (SM9+ only — the 3090 is SM8x, the whole reason the
   Triton fallback exists) OR replacing the flash-attn call with a custom
   rel-attention kernel over the dequantized fp16 K'/V'.
4. **Cache layout rewrite.** Inkling's `_split_kv_cache`, the Triton kernels,
   and `get_kv_cache_spec` all assume the float paged layout. KVarN's packed
   tile layout needs a different alloc shape, a tile-write path, and a
   dequant-on-read path — i.e. a new attention backend for inkling.

## The only theoretically-working approach (large)
A new "KVarN-aware relative-attention" path for inkling:
- Allocate the KVarN packed-tile cache for inkling's layers.
- Replace fused_qkvr_prep's cache-write with KVarN's tile-buffer + store
  (Hadamard+Sinkhorn+RTN), keeping inkling's sconv/rmsnorm/q-prep intact.
- For read: dequant tiles → fp16 K'/V' (rotated) + sink/tail, then run inkling's
  Triton rel-attention kernel (rotated Q, rel bias) on the fp16 buffers, then
  un-rotate. Reconcile block_size=128 for local layers.
This is a multi-week, multi-file feature spanning a new backend + rewrite of
inkling's fused write kernel + KVarN tile plumbing. Disproportionate for
"reduce KV-cache memory on a 3090".

## Recommendation
- **Do NOT implement KVarN for inkling now.** Use bf16 (working) or fp8
  (just added, ~2x KV-cache compression, capture-safe) KV cache.
- If KV-cache memory is the real goal, fp8 KV cache (`--kv-cache-dtype fp8`,
  commit f1821677f7) is the practical answer on SM8x.
- KVarN for inkling should be a separate, scoped project (and likely needs
  upstream KVarN changes + an SM8x bias-aware decode kernel) — not a follow-up
  to this task.

## Sources
- E:/Work/Git/KVarN: kvarn_attn.py, triton_kvarn_decode.py (kvarn_decode_attention
  L811; flash_attn call L984 causal=False no bias), kvarn_decode.py (dequant ref),
  config.py.
- E:/Work/Git/vllm (in-tree, identical): same files + kvarn_attn.py:236 block_size
  assert, get_supported_kernel_block_sizes→{64,128}.
- Failure log: E:/sync/unique/AIServer/tmp/ink-small-awq-load-log-3.txt.
- Session memory: [593c5a6674db] (root-cause verdict), [f007bb562e07]
  (Triton path is public/general).
