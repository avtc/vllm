# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-based relative attention fallback for architectures without FA4 score_mod.

Used as a fallback on SM8x GPUs (e.g., RTX 3090) where FA4's custom score_mod
is not supported. Implements paged attention with per-distance relative bias.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

_BLOCK_M = 64
_BLOCK_N = 32


@triton.jit
def _inkling_rel_attn_decode(
    Q,
    K_CACHE,
    V_CACHE,
    OUT,
    REL_LOGITS,
    BLOCK_TABLE,
    CACHE_SEQLENS,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kbt,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbt,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    stride_rlt,
    stride_rlhn,
    stride_rle,
    stride_bts,
    sm_scale,
    kv_block_size: tl.constexpr,
    kv_group_num: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SW_LEFT: tl.constexpr,
    SW_RIGHT: tl.constexpr,
    USE_SW: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Decode kernel: single query token per program, iterates all KV blocks."""
    req_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    kv_head_idx = head_idx // kv_group_num
    kv_len = tl.load(CACHE_SEQLENS + req_idx).to(tl.int64)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    # Load single query token
    q = tl.load(
        Q + req_idx * stride_qt + head_idx * stride_qh + offs_d * stride_qd,
        mask=offs_d < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)[None, :]

    # Init softmax state
    m_i = -float("inf")
    l_i = 1.0
    acc = tl.zeros([1, HEAD_DIM], dtype=tl.float32)

    block_table_offset = req_idx * stride_bts
    num_blocks = tl.cdiv(kv_len, kv_block_size)

    for blk_idx in range(0, num_blocks):
        phys_block = tl.load(BLOCK_TABLE + block_table_offset + blk_idx).to(tl.int64)
        blk_start = blk_idx * kv_block_size
        if blk_start >= kv_len:
            break

        remaining = kv_len - blk_start
        k_mask_n = offs_n < remaining
        d_mask = offs_d < HEAD_DIM

        # Load K: (HEAD_DIM, BLOCK_N), V: (BLOCK_N, HEAD_DIM)
        k = tl.load(
            K_CACHE
            + phys_block * stride_kbt
            + kv_head_idx * stride_kh
            + offs_d[:, None] * stride_kd
            + offs_n[None, :] * stride_kbs,
            mask=k_mask_n[None, :] & d_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        v = tl.load(
            V_CACHE
            + phys_block * stride_vbt
            + kv_head_idx * stride_vh
            + offs_n[:, None] * stride_vbs
            + offs_d[None, :] * stride_vd,
            mask=k_mask_n[:, None] & d_mask[None, :],
            other=0.0,
        )

        # QK^T: (1, D) dot (D, N) = (1, N)
        qk = tl.dot(q, k) * sm_scale

        # Key positions
        k_pos = blk_start + offs_n
        attn_mask = k_pos < kv_len

        # Sliding window (decode: query at end, SW_LEFT is past, SW_RIGHT is future)
        if USE_SW and SW_LEFT > 0:
            attn_mask &= k_pos >= kv_len - 1 - SW_LEFT
        if USE_SW and SW_RIGHT > 0:
            attn_mask &= k_pos <= kv_len - 1 + SW_RIGHT

        # Relative distance and bias
        rel_dist = kv_len - k_pos
        rel_idx = tl.where(rel_dist >= 0, rel_dist, 0)
        rel_idx = tl.minimum(rel_idx, REL_EXTENT - 1)

        rel_bias = tl.load(
            REL_LOGITS
            + req_idx * stride_rlt
            + head_idx * stride_rlhn
            + rel_idx * stride_rle,
            mask=attn_mask,
            other=0.0,
        ).to(tl.float32)

        qk = tl.where(attn_mask, qk + rel_bias, -1.0e8)

        # Online softmax
        m_ij = tl.max(qk, axis=1)
        qk = qk - m_ij
        p = tl.exp(qk)
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha + tl.dot(p[None, :].to(v.dtype), v)
        m_i = m_ij

    # Store output
    acc = acc / l_i
    tl.store(
        OUT + req_idx * stride_ot + head_idx * stride_oh + offs_d * stride_od,
        acc[0, :],
        mask=offs_d < HEAD_DIM,
    )


@triton.jit
def _inkling_rel_attn_prefill(
    Q,
    K_CACHE,
    V_CACHE,
    OUT,
    REL_LOGITS,
    BLOCK_TABLE,
    CACHE_SEQLENS,
    CU_SEQLENS_Q,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kbt,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbt,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    stride_rlt,
    stride_rlhn,
    stride_rle,
    stride_bts,
    sm_scale,
    kv_block_size: tl.constexpr,
    kv_group_num: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SW_LEFT: tl.constexpr,
    SW_RIGHT: tl.constexpr,
    USE_SW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Prefill kernel: one program per (request, head, q_block)."""
    req_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    m_block = tl.program_id(2)

    kv_head_idx = head_idx // kv_group_num

    q_start = tl.load(CU_SEQLENS_Q + req_idx)
    q_end = tl.load(CU_SEQLENS_Q + req_idx + 1)
    q_len = q_end - q_start
    kv_len = tl.load(CACHE_SEQLENS + req_idx)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    q_mask = offs_m < q_len
    d_mask = offs_d < HEAD_DIM

    # Load Q: (BLOCK_M, HEAD_DIM)
    q = tl.load(
        Q
        + (q_start + offs_m[:, None]) * stride_qt
        + head_idx * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Init softmax state
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    block_table_offset = req_idx * stride_bts
    num_blocks = tl.cdiv(kv_len, kv_block_size)

    for blk_idx in range(0, num_blocks):
        phys_block = tl.load(BLOCK_TABLE + block_table_offset + blk_idx).to(tl.int64)
        blk_start = blk_idx * kv_block_size
        if blk_start >= kv_len:
            break

        remaining = kv_len - blk_start
        n_mask = offs_n < remaining

        # Load K: (HEAD_DIM, BLOCK_N), V: (BLOCK_N, HEAD_DIM)
        k = tl.load(
            K_CACHE
            + phys_block * stride_kbt
            + kv_head_idx * stride_kh
            + offs_d[:, None] * stride_kd
            + offs_n[None, :] * stride_kbs,
            mask=n_mask[None, :] & d_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        v = tl.load(
            V_CACHE
            + phys_block * stride_vbt
            + kv_head_idx * stride_vh
            + offs_n[:, None] * stride_vbs
            + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

        # QK^T: (M, D) dot (D, N) = (M, N)
        qk = tl.dot(q, k) * sm_scale

        # Positions
        q_local = offs_m[:, None]
        k_pos = blk_start + offs_n[None, :]

        # Masks
        attn_mask = k_pos < kv_len
        if IS_CAUSAL:
            attn_mask &= q_local >= k_pos
        if USE_SW and SW_LEFT > 0:
            attn_mask &= (k_pos - q_local) <= SW_LEFT
        if USE_SW and SW_RIGHT > 0:
            attn_mask &= (q_local - k_pos) <= SW_RIGHT

        # Relative bias
        rel_dist = q_local - k_pos
        rel_idx = tl.where(rel_dist >= 0, rel_dist, 0)
        rel_idx = tl.minimum(rel_idx, REL_EXTENT - 1)

        # Global query position for rel_logits indexing
        q_global = (q_start + q_local) * stride_rlt

        rel_bias = tl.load(
            REL_LOGITS + q_global + head_idx * stride_rlhn + rel_idx * stride_rle,
            mask=q_mask[:, None] & attn_mask,
            other=0.0,
        ).to(tl.float32)

        qk = tl.where(attn_mask, qk + rel_bias, -1.0e8)

        # Online softmax
        m_ij = tl.max(qk, axis=1)
        qk = qk - m_ij[:, None]
        p = tl.exp(qk)
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # Store output
    acc = acc / l_i[:, None]
    tl.store(
        OUT
        + (q_start + offs_m[:, None]) * stride_ot
        + head_idx * stride_oh
        + offs_d[None, :] * stride_od,
        acc,
        mask=q_mask[:, None] & d_mask[None, :],
    )


def inkling_triton_rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    rel_extent: int,
    rel_logits: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton paged attention with relative bias for SM8x GPUs.

    Args:
        q: (num_tokens, num_heads, head_dim) query tensor.
        key_cache: (num_blocks, block_size, num_kv_heads, head_dim) KV cache.
        value_cache: same shape as key_cache.
        block_table: (num_reqs, max_blocks) page table.
        cache_seqlens: (num_reqs,) KV lengths per request.
        cu_seqlens_q: (num_reqs + 1,) cumulative query offsets.
        max_seqlen_q: max query length in the batch.
        softmax_scale: scaling factor for QK^T.
        causal: whether to apply causal masking.
        window_size: (left, right) sliding window; (-1, -1) for infinite.
        rel_extent: number of relative distance buckets.
        rel_logits: (num_tokens, num_heads, rel_extent) relative bias logits.
        out: pre-allocated output buffer.

    Returns:
        Attention output (num_tokens, num_heads, head_dim).
    """
    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = key_cache.shape[2]
    num_reqs = cache_seqlens.shape[0]
    kv_block_size = key_cache.shape[1]

    if out is None:
        out = torch.empty_like(q)

    assert q.dtype == out.dtype
    assert rel_logits.shape[0] == num_tokens
    assert rel_logits.shape[1] == num_heads
    assert rel_logits.shape[2] == rel_extent

    kv_group_num = num_heads // num_kv_heads
    use_sw = window_size != (-1, -1)
    sw_left = window_size[0] if use_sw and window_size[0] >= 0 else 0
    sw_right = window_size[1] if use_sw and window_size[1] >= 0 else 0

    is_decode = max_seqlen_q <= 1

    if is_decode:
        grid = (num_reqs, num_heads)
        _inkling_rel_attn_decode[grid](
            q,
            key_cache,
            value_cache,
            out,
            rel_logits,
            block_table,
            cache_seqlens,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            rel_logits.stride(0),
            rel_logits.stride(1),
            rel_logits.stride(2),
            block_table.stride(0),
            softmax_scale,
            kv_block_size,
            kv_group_num,
            HEAD_DIM=head_dim,
            REL_EXTENT=rel_extent,
            IS_CAUSAL=causal,
            SW_LEFT=sw_left,
            SW_RIGHT=sw_right,
            USE_SW=use_sw,
            BLOCK_N=_BLOCK_N,
        )
    else:
        num_m_blocks = (num_tokens + _BLOCK_M - 1) // _BLOCK_M
        grid = (num_reqs, num_heads, num_m_blocks)
        _inkling_rel_attn_prefill[grid](
            q,
            key_cache,
            value_cache,
            out,
            rel_logits,
            block_table,
            cache_seqlens,
            cu_seqlens_q,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            rel_logits.stride(0),
            rel_logits.stride(1),
            rel_logits.stride(2),
            block_table.stride(0),
            softmax_scale,
            kv_block_size,
            kv_group_num,
            HEAD_DIM=head_dim,
            REL_EXTENT=rel_extent,
            IS_CAUSAL=causal,
            SW_LEFT=sw_left,
            SW_RIGHT=sw_right,
            USE_SW=use_sw,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
        )

    return out
