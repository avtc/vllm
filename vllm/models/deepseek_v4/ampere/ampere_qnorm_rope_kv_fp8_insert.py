# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ampere Triton replacement for fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.

Does: Q per-head RMSNorm + GPT-J RoPE, KV GPT-J RoPE + UE8M0 FP8 quant + insert.
Uses the existing quantize_and_insert_k_cache for the FP8 portion.
"""

import torch

from vllm.triton_utils import tl, triton

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HALF_ROPE = ROPE_DIM // 2


# [DSv4-ampere debug] one-shot flag for the ROPE_SHAPE log.
_ampere_qnorm_rope_shape_logged = type("S", (), {"done": False})()


# [DSv4-ampere debug] BEFORE-quantize snapshot of the SWA cache target slots.
# Reads the bf16-RoPE region [448:576] of each slot to be written, logging any
# pre-existing non-finite. Compared with the AFTER read (SWA_WRITE probe), this
# shows whether quantize INTRODUCES the NaN or it pre-existed (stale block /
# partial prior write). Gated by VLLM_SM86_NAN_PROBE=1, rank0, one-shot.
_SWA_BEFORE_FIRED = False


def _swa_before_snapshot(swa_kv_cache, slot_mapping):
    import os

    global _SWA_BEFORE_FIRED
    if os.environ.get("VLLM_SM86_NAN_PROBE") != "1":
        return
    if _SWA_BEFORE_FIRED:
        return
    slots = slot_mapping.detach()
    valid = slots >= 0
    if not bool(valid.any().item()):
        return
    sv = slots[valid].to(torch.int64)
    head_bytes = swa_kv_cache.shape[-1]  # 584
    flat = swa_kv_cache.reshape(-1)
    row_ptrs = sv * head_bytes + 448
    offs = torch.arange(128, device=sv.device, dtype=torch.int64)
    gathered = flat[row_ptrs[:, None] + offs[None, :]].view(torch.bfloat16).float()
    n_nan = int(torch.isnan(gathered).sum().item())
    n_inf = int(torch.isinf(gathered).sum().item())
    if n_nan or n_inf:
        _SWA_BEFORE_FIRED = True
        bad_rows = torch.where(torch.isnan(gathered).any(1) | torch.isinf(gathered).any(1))[0]
        bad_slots = sv[bad_rows][:8].tolist()
        print(
            f"[SWA_BEFORE quantize] target slots ALREADY have non-finite! "
            f"nan={n_nan} inf={n_inf} bad_slots={bad_slots} "
            f"n_target_slots={int(sv.numel())} head_bytes={head_bytes}",
            flush=True,
        )


@triton.jit
def _xpu_qnorm_rope_kernel(
    q_ptr,  # [num_tokens, num_heads, HEAD_DIM]
    kv_ptr,  # [num_tokens, HEAD_DIM]
    kv_out_ptr,  # [num_tokens, HEAD_DIM] bf16 (RoPE-applied kv for cache insert)
    position_ids_ptr,
    cos_sin_cache_ptr,
    eps: tl.constexpr,
    num_tokens,
    num_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    """Apply per-head RMSNorm + GPT-J RoPE on Q, GPT-J RoPE on KV.

    GPT-J interleaved format: pairs are (data[2i], data[2i+1]).
    cos_sin_cache layout: [max_pos, ROPE_DIM] with first HALF_ROPE=cos,
    second HALF_ROPE=sin.
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if token_idx >= num_tokens:
        return

    pos = tl.load(position_ids_ptr + token_idx).to(tl.int64)

    # Load cos/sin for this position
    rope_pair_idx = tl.arange(0, HALF_ROPE)
    cos_val = tl.load(cos_sin_cache_ptr + pos * ROPE_DIM + rope_pair_idx).to(tl.float32)
    sin_val = tl.load(
        cos_sin_cache_ptr + pos * ROPE_DIM + HALF_ROPE + rope_pair_idx
    ).to(tl.float32)

    if head_idx < num_heads:
        # ========== Q: per-head RMSNorm + GPT-J RoPE ==========
        q_base = q_ptr + token_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM

        # Load full head
        offs = tl.arange(0, HEAD_DIM)
        q_vals = tl.load(q_base + offs).to(tl.float32)

        # RMSNorm (no weight)
        sq_sum = tl.sum(q_vals * q_vals, axis=0)
        rms = tl.rsqrt(sq_sum / HEAD_DIM + eps)
        q_vals = q_vals * rms

        # Store ONLY the NoPE portion (positions 0..NOPE_DIM-1)
        nope_mask = offs < NOPE_DIM
        tl.store(q_base + offs, q_vals.to(q_ptr.type.element_ty), mask=nope_mask)

        # GPT-J interleaved RoPE on the last ROPE_DIM dimensions:
        even_offs = NOPE_DIM + rope_pair_idx * 2
        odd_offs = NOPE_DIM + rope_pair_idx * 2 + 1

        # Re-load original values at rope positions and normalize
        q_even = tl.load(q_base + even_offs).to(tl.float32) * rms
        q_odd = tl.load(q_base + odd_offs).to(tl.float32) * rms

        new_even = q_even * cos_val - q_odd * sin_val
        new_odd = q_even * sin_val + q_odd * cos_val

        # Store rotated RoPE values
        tl.store(q_base + even_offs, new_even.to(q_ptr.type.element_ty))
        tl.store(q_base + odd_offs, new_odd.to(q_ptr.type.element_ty))
    else:
        # ========== KV: GPT-J RoPE only ==========
        kv_base = kv_ptr + token_idx * HEAD_DIM
        kv_out_base = kv_out_ptr + token_idx * HEAD_DIM

        # Copy full KV unchanged first
        offs = tl.arange(0, HEAD_DIM)
        kv_full = tl.load(kv_base + offs)
        tl.store(kv_out_base + offs, kv_full)

        # GPT-J interleaved RoPE on the last ROPE_DIM dimensions
        even_offs = NOPE_DIM + rope_pair_idx * 2
        odd_offs = NOPE_DIM + rope_pair_idx * 2 + 1

        kv_even = tl.load(kv_base + even_offs).to(tl.float32)
        kv_odd = tl.load(kv_base + odd_offs).to(tl.float32)

        new_even = kv_even * cos_val - kv_odd * sin_val
        new_odd = kv_even * sin_val + kv_odd * cos_val

        tl.store(kv_out_base + even_offs, new_even.to(kv_out_ptr.type.element_ty))
        tl.store(kv_out_base + odd_offs, new_odd.to(kv_out_ptr.type.element_ty))


def ampere_qnorm_rope_kv_fp8_insert(
    q: torch.Tensor,  # [num_tokens, num_heads, HEAD_DIM] bf16, in-place
    kv: torch.Tensor,  # [num_tokens, HEAD_DIM] bf16
    swa_kv_cache: torch.Tensor,  # [num_blocks, block_size, 584] or flat uint8
    slot_mapping: torch.Tensor,  # [num_tokens] int64
    positions: torch.Tensor,  # [num_tokens] int64
    cos_sin_cache: torch.Tensor,  # [max_pos, ROPE_DIM]
    eps: float,
    block_size: int,
):
    """Ampere Triton: qnorm+rope on Q, rope on KV, then FP8 UE8M0 quant+insert."""
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )

    num_tokens = q.shape[0]
    num_heads = q.shape[1]

    # Allocate temp buffer for RoPE-applied KV
    kv_roped = torch.empty_like(kv)

    # Grid: one program per (token, head_or_kv)
    # head_idx < num_heads: process Q head
    # head_idx == num_heads: process KV
    grid = (num_tokens, num_heads + 1)
    _xpu_qnorm_rope_kernel[grid](
        q,
        kv,
        kv_roped,
        positions,
        cos_sin_cache,
        eps,
        num_tokens,
        num_heads=num_heads,
        HEAD_DIM=HEAD_DIM,
        ROPE_DIM=ROPE_DIM,
        NOPE_DIM=NOPE_DIM,
        HALF_ROPE=HALF_ROPE,
    )

    # [DSv4-ampere debug] BISECT: is the NaN in kv_roped (=> _xpu_qnorm_rope_kernel
    # produced it) or introduced by quantize_and_insert_k_cache? kv_roped is
    # torch.empty_like (UNINITIALIZED) — if the RoPE kernel skips/partially-writes
    # any token, the stale garbage (incl. NaN bit patterns) flows to the cache.
    import os as _os_bisect
    if _os_bisect.environ.get("VLLM_SM86_NAN_PROBE") == "1":
        try:
            if not (torch.distributed.is_available() and torch.distributed.is_initialized()
                    and torch.distributed.get_rank() != 0):
                # PRIME SUSPECT: num_tokens mismatch. The RoPE kernel writes
                # kv_roped[0:q.shape[0]] (num_tokens=q.shape[0]), but
                # quantize_and_insert_k_cache reads kv_roped[0:slot_mapping.shape[0]].
                # If slot_mapping is LONGER than q/kv, quantize reads PAST kv_roped's
                # end => uninitialized memory => NaN. Log the shapes once.
                q_nt = q.shape[0]
                kv_nt = kv.shape[0]
                sm_nt = slot_mapping.shape[0]
                kvr_nt = kv_roped.shape[0]
                if not getattr(_ampere_qnorm_rope_shape_logged, "done", False):
                    _ampere_qnorm_rope_shape_logged.done = True
                    print(
                        f"[ROPE_SHAPE] q_nt={q_nt} kv_nt={kv_nt} "
                        f"slot_mapping_nt={sm_nt} kv_roped_nt={kvr_nt} num_heads={num_heads}",
                        flush=True,
                    )
                if sm_nt > kvr_nt:
                    print(
                        f"[ROPE_OOB_READ] slot_mapping_nt={sm_nt} > kv_roped_nt={kvr_nt}! "
                        f"quantize reads {sm_nt - kvr_nt} rows PAST kv_roped (uninitialized "
                        f"=> NaN). q_nt={q_nt} kv_nt={kv_nt}",
                        flush=True,
                    )
                rope_out = kv_roped[..., NOPE_DIM:].detach().float()  # [N, ROPE_DIM]
                if rope_out.numel():
                    bad = torch.where(torch.isnan(rope_out).any(1) | torch.isinf(rope_out).any(1))[0]
                    if bad.numel():
                        # also check the INPUT kv RoPE portion for comparison
                        kv_in = kv[..., NOPE_DIM:].detach().float()
                        in_bad = "yes" if (torch.isnan(kv_in).any().item() or torch.isinf(kv_in).any().item()) else "no"
                        print(
                            f"[ROPE_BISECT] kv_roped RoPE portion has non-finite! "
                            f"n_bad_tokens={int(bad.numel())} first_tokens={bad[:8].tolist()} "
                            f"input_kv_rope_nonfinite={in_bad} num_tokens={num_tokens} "
                            f"num_heads={num_heads}",
                            flush=True,
                        )
        except Exception as _e:  # noqa: BLE001
            if _os_bisect.environ.get("VLLM_SM86_PROBE_VERBOSE") == "1":
                print(f"[ROPE_BISECT] skipped: {_e}", flush=True)

    # FP8 UE8M0 quant + paged insert (reuse existing Triton kernel)
    # swa_kv_cache may be [num_blocks, block_size, 584] or [num_blocks, flat]
    # quantize_and_insert_k_cache expects [num_blocks, block_bytes] uint8
    cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)

    # [DSv4-ampere debug] BEFORE snapshot: read the target slots' bf16-RoPE
    # region BEFORE quantize, so the AFTER read (SWA_WRITE probe) can tell whether
    # quantize INTRODUCED the NaN or it pre-existed (stale/partial-write).
    import os as _os_before
    if _os_before.environ.get("VLLM_SM86_NAN_PROBE") == "1":
        try:
            if not (torch.distributed.is_available() and torch.distributed.is_initialized()
                    and torch.distributed.get_rank() != 0):
                _swa_before_snapshot(swa_kv_cache, slot_mapping)
        except Exception:
            pass

    quantize_and_insert_k_cache(
        kv_roped,
        cache_2d,
        slot_mapping,
        block_size=block_size,
    )
