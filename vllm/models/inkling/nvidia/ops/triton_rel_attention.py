# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton relative-attention backend for Inkling's per-distance bias.

Implements paged attention with the per-distance relative bias that FA4
expresses via ``score_mod``. It is selected automatically on devices where
FA4's custom ``score_mod`` is unavailable (SM8x and below) and is a general,
device-agnostic backend that runs efficiently on any CUDA GPU (SM8x through
Blackwell+). The numerics match FA4 exactly (see
``tests/models/inkling/test_fa4_rel_attention.py::_ref_rel_attn``)::

    rel_dist = global_q_pos - k_pos
    rel_bias = rel_logits[global_q_idx, h, rel_dist]   if 0 <= rel_dist < rel_extent
             = 0                                       otherwise

with ``global_q_pos = q_local + (kv_len - q_len)`` (== ``kv_len - 1`` for decode).

Decode uses split-KV (flash-decoding): the split count adapts to the host GPU's
SM count, so long-context generation parallelizes across the full SM array
instead of serially scanning the cache.
"""

from __future__ import annotations

from functools import cache

import torch

from vllm.triton_utils import tl, triton

_BLOCK_M = 64


def _next_pow2(n: int) -> int:
    return 1 << max(0, (max(1, n) - 1).bit_length())


@cache
def _num_sms(device_index: int) -> int:
    """Multiprocessor count of ``device_index`` (cached; device-agnostic GPU sizing)."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@triton.jit
def _inkling_rel_attn_decode_partial(
    Q,
    K_CACHE,
    V_CACHE,
    REL_LOGITS,
    BLOCK_TABLE,
    CACHE_SEQLENS,
    PARTIAL_M,
    PARTIAL_L,
    PARTIAL_O,
    stride_qt,
    stride_qh,
    stride_kbt,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbt,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_rlt,
    stride_rlhn,
    stride_bts,
    stride_pm_r,
    stride_pm_h,
    stride_pm_s,
    stride_po_r,
    stride_po_h,
    stride_po_s,
    stride_po_d,
    sm_scale,
    kv_block_size: tl.constexpr,
    kv_group_num: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    SW_LEFT: tl.constexpr,
    SW_RIGHT: tl.constexpr,
    USE_SW: tl.constexpr,
    BLOCKS_PER_SPLIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One split of one (request, head): partial online-softmax over a KV range.

    Writes unnormalised (m, l, acc=Σ p·v). A split whose block range is empty
    (short sequence / beyond the cache) writes the identity (m=-inf, l=0, 0).
    """
    req_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    kv_head_idx = head_idx // kv_group_num
    kv_len = tl.load(CACHE_SEQLENS + req_idx)
    num_phys_blocks = tl.cdiv(kv_len, kv_block_size)

    blk_lo = split_idx * BLOCKS_PER_SPLIT
    blk_hi = tl.minimum(blk_lo + BLOCKS_PER_SPLIT, num_phys_blocks)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    dmask = offs_d < HEAD_DIM

    # Single query vector: (HEAD_DIM,)
    q = tl.load(
        Q + req_idx * stride_qt + head_idx * stride_qh + offs_d,
        mask=dmask,
        other=0.0,
    ).to(tl.float32)

    m_i = -1.0e9
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    q_seq = kv_len - 1  # decode query sits at the latest cached position
    bt_base = req_idx * stride_bts

    for blk in range(blk_lo, blk_hi):
        phys = tl.load(BLOCK_TABLE + bt_base + blk).to(tl.int64)
        blk_start = blk * kv_block_size
        # Local layers: skip physical blocks entirely before the sliding window.
        # The window for the decode query (at q_seq) is [q_seq - SW_LEFT, q_seq].
        if USE_SW and SW_LEFT > 0 and (blk_start + kv_block_size - 1 < q_seq - SW_LEFT):
            continue
        remaining = kv_len - blk_start
        n_mask = offs_n < remaining

        # K, V: (BLOCK_N, HEAD_DIM)
        k = tl.load(
            K_CACHE
            + phys * stride_kbt
            + kv_head_idx * stride_kh
            + offs_n[:, None] * stride_kbs
            + offs_d[None, :] * stride_kd,
            mask=n_mask[:, None] & dmask[None, :],
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_CACHE
            + phys * stride_vbt
            + kv_head_idx * stride_vh
            + offs_n[:, None] * stride_vbs
            + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None] & dmask[None, :],
            other=0.0,
        )

        # QK^T for a single query: (BLOCK_N,)
        qk = tl.sum(k * q[None, :], axis=1) * sm_scale

        k_pos = blk_start + offs_n
        attn_mask = n_mask & (k_pos < kv_len)
        if USE_SW and SW_LEFT > 0:
            attn_mask &= k_pos >= q_seq - SW_LEFT
        if USE_SW and SW_RIGHT > 0:
            attn_mask &= k_pos <= q_seq + SW_RIGHT

        # Relative bias (zero when out of range, matching FA4 score-mod).
        rel_dist = q_seq - k_pos
        in_range = (rel_dist >= 0) & (rel_dist < REL_EXTENT)
        rel_idx = tl.where(in_range, rel_dist, 0)
        rel_bias = tl.load(
            REL_LOGITS + req_idx * stride_rlt + head_idx * stride_rlhn + rel_idx,
            mask=in_range,
            other=0.0,
        ).to(tl.float32)

        qk = tl.where(attn_mask, qk + rel_bias, -1.0e9)

        m_ij = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new)
        p = tl.where(attn_mask, p, 0.0)
        l_ij = tl.sum(p, axis=0)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    pm_ptr = (
        PARTIAL_M
        + req_idx * stride_pm_r
        + head_idx * stride_pm_h
        + split_idx * stride_pm_s
    )
    tl.store(pm_ptr, m_i)
    tl.store(
        PARTIAL_L
        + req_idx * stride_pm_r
        + head_idx * stride_pm_h
        + split_idx * stride_pm_s,
        l_i,
    )
    tl.store(
        PARTIAL_O
        + req_idx * stride_po_r
        + head_idx * stride_po_h
        + split_idx * stride_po_s
        + offs_d * stride_po_d,
        acc,
        mask=dmask,
    )


@triton.jit
def _inkling_rel_attn_combine(
    PARTIAL_M,
    PARTIAL_L,
    PARTIAL_O,
    OUT,
    stride_pm_r,
    stride_pm_h,
    stride_pm_s,
    stride_po_r,
    stride_po_h,
    stride_po_s,
    stride_po_d,
    stride_ot,
    stride_oh,
    stride_od,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Merge the per-split partials (online-softmax merge) and write the output."""
    req_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_d = tl.arange(0, HEAD_DIM)
    dmask = offs_d < HEAD_DIM

    pm_base = PARTIAL_M + req_idx * stride_pm_r + head_idx * stride_pm_h
    pl_base = PARTIAL_L + req_idx * stride_pm_r + head_idx * stride_pm_h
    po_base = PARTIAL_O + req_idx * stride_po_r + head_idx * stride_po_h

    m_g = -1.0e9
    l_g = 0.0
    acc_g = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for s in tl.static_range(0, NUM_SPLITS):
        m_s = tl.load(pm_base + s * stride_pm_s)
        l_s = tl.load(pl_base + s * stride_pm_s)
        o_s = tl.load(
            po_base + s * stride_po_s + offs_d * stride_po_d,
            mask=dmask,
            other=0.0,
        )
        m_new = tl.maximum(m_g, m_s)
        alpha = tl.exp(m_g - m_new)
        beta = tl.exp(m_s - m_new)
        l_g = l_g * alpha + l_s * beta
        acc_g = acc_g * alpha + o_s * beta
        m_g = m_new

    out = tl.where(l_g > 0, acc_g / l_g, 0.0)
    tl.store(
        OUT + req_idx * stride_ot + head_idx * stride_oh + offs_d * stride_od,
        out,
        mask=dmask,
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
    """One program per (request, head, query block)."""
    req_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    m_block = tl.program_id(2)

    kv_head_idx = head_idx // kv_group_num
    q_start = tl.load(CU_SEQLENS_Q + req_idx)
    q_end = tl.load(CU_SEQLENS_Q + req_idx + 1)
    q_len = q_end - q_start
    kv_len = tl.load(CACHE_SEQLENS + req_idx)

    q_offset = kv_len - q_len  # local q idx -> absolute sequence position

    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    q_mask = offs_m < q_len
    dmask = offs_d < HEAD_DIM

    # Absolute sequence position of each query row: (BLOCK_M,)
    q_seq = offs_m + q_offset

    # Load Q: (BLOCK_M, HEAD_DIM)
    q = tl.load(
        Q
        + (q_start + offs_m[:, None]) * stride_qt
        + head_idx * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=q_mask[:, None] & dmask[None, :],
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full([BLOCK_M], -1.0e9, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Causal early-exit: keys beyond the furthest query in this block are all
    # masked, so stop at the KV block containing it.
    num_phys_blocks = tl.cdiv(kv_len, kv_block_size)
    q_max_seq = tl.max(tl.where(q_mask, q_seq, 0), axis=0)
    last_blk = tl.minimum(num_phys_blocks, tl.cdiv(q_max_seq + 1, kv_block_size))
    # Local layers: earliest relevant key is min(query) - SW_LEFT; blocks fully
    # before that are outside every query's window and can be skipped.
    if USE_SW and SW_LEFT > 0:
        q_min_seq = tl.min(tl.where(q_mask, q_seq, 1 << 30), axis=0)
        first_blk = tl.maximum(0, (q_min_seq - SW_LEFT) // kv_block_size)
    else:
        first_blk = 0

    bt_base = req_idx * stride_bts
    q_tok = q_start + offs_m  # global token index for rel_logits: (BLOCK_M,)

    for blk in range(first_blk, last_blk):
        phys = tl.load(BLOCK_TABLE + bt_base + blk).to(tl.int64)
        blk_start = blk * kv_block_size
        remaining = kv_len - blk_start
        n_mask = offs_n < remaining

        # K: (HEAD_DIM, BLOCK_N), V: (BLOCK_N, HEAD_DIM)
        k = tl.load(
            K_CACHE
            + phys * stride_kbt
            + kv_head_idx * stride_kh
            + offs_d[:, None] * stride_kd
            + offs_n[None, :] * stride_kbs,
            mask=dmask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_CACHE
            + phys * stride_vbt
            + kv_head_idx * stride_vh
            + offs_n[:, None] * stride_vbs
            + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None] & dmask[None, :],
            other=0.0,
        )

        # QK^T: (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k) * sm_scale

        k_pos = blk_start + offs_n  # (BLOCK_N,)
        attn_mask = n_mask[None, :] & (k_pos[None, :] < kv_len) & q_mask[:, None]
        if IS_CAUSAL:
            attn_mask &= q_seq[:, None] >= k_pos[None, :]
        if USE_SW and SW_LEFT > 0:
            attn_mask &= (q_seq[:, None] - k_pos[None, :]) <= SW_LEFT
        if USE_SW and SW_RIGHT > 0:
            attn_mask &= (k_pos[None, :] - q_seq[:, None]) <= SW_RIGHT

        # Relative bias (zero when out of range, matching FA4 score-mod).
        rel_dist = q_seq[:, None] - k_pos[None, :]  # (BLOCK_M, BLOCK_N)
        in_range = (rel_dist >= 0) & (rel_dist < REL_EXTENT)
        rel_idx = tl.where(in_range, rel_dist, 0)
        rel_bias = tl.load(
            REL_LOGITS
            + q_tok[:, None] * stride_rlt
            + head_idx * stride_rlhn
            + rel_idx * stride_rle,
            mask=q_mask[:, None] & in_range,
            other=0.0,
        ).to(tl.float32)

        qk = tl.where(attn_mask, qk + rel_bias, -1.0e9)

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new[:, None])
        p = tl.where(attn_mask, p, 0.0)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    out = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)
    tl.store(
        OUT
        + (q_start + offs_m[:, None]) * stride_ot
        + head_idx * stride_oh
        + offs_d[None, :] * stride_od,
        out,
        mask=q_mask[:, None] & dmask[None, :],
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
    max_kv_len: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton paged attention with the Inkling relative bias (device-agnostic).

    Selected automatically when FA4's score_mod is unavailable; runs on any
    CUDA GPU (SM8x through Blackwell+).

    Args:
        q: (num_tokens, num_heads, head_dim) query tensor.
        key_cache: (num_blocks, block_size, num_kv_heads, head_dim) KV cache.
        value_cache: same shape as key_cache.
        block_table: (num_reqs, max_blocks) page table.
        cache_seqlens: (num_reqs,) KV lengths per request (incl. current token).
        cu_seqlens_q: (num_reqs + 1,) cumulative query offsets.
        max_seqlen_q: bucketed max query length in the batch.
        softmax_scale: scaling factor for QK^T.
        causal: whether to apply causal masking.
        window_size: (left, right) sliding window; (-1, -1) for infinite.
        rel_extent: number of relative distance buckets.
        rel_logits: (num_tokens, num_heads, rel_extent) relative bias logits.
        max_kv_len: static upper bound on KV length (avoids a host sync; falls
            back to ``cache_seqlens.max()`` when ``None``).
        out: pre-allocated output buffer.

    Returns:
        Attention output (num_tokens, num_heads, head_dim).
    """
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = key_cache.shape[2]
    num_reqs = cache_seqlens.shape[0]
    kv_block_size = key_cache.shape[1]

    if out is None:
        out = torch.empty_like(q)

    kv_group_num = num_heads // num_kv_heads
    use_sw = window_size != (-1, -1)
    sw_left = window_size[0] if (use_sw and window_size[0] >= 0) else 0
    sw_right = window_size[1] if (use_sw and window_size[1] >= 0) else 0

    is_decode = max_seqlen_q <= 1

    if is_decode:
        # ---- split-KV decode (flash-decoding) ----
        if max_kv_len is None:
            max_kv_len = int(cache_seqlens.max().item())
        max_phys_blocks = (max_kv_len + kv_block_size - 1) // kv_block_size

        # Split count adapts to the host GPU: aim for ~8 waves of CTAs across the
        # SM array, capped and rounded to a power of two so the combine kernel
        # can use a constexpr tile/loop. Empty splits (short sequences) write the
        # softmax identity and are merged out.
        sms = _num_sms(q.device.index if q.device.index is not None else 0)
        base_ctas = num_reqs * num_heads
        desired = max(1, (sms * 8) // max(1, base_ctas))
        desired = min(desired, 128, max(1, max_phys_blocks))
        num_splits = _next_pow2(desired)
        blocks_per_split = max(1, (max_phys_blocks + num_splits - 1) // num_splits)

        partial_m = torch.empty(
            (num_reqs, num_heads, num_splits), dtype=torch.float32, device=q.device
        )
        partial_l = torch.empty_like(partial_m)
        partial_o = torch.empty(
            (num_reqs, num_heads, num_splits, head_dim),
            dtype=torch.float32,
            device=q.device,
        )

        grid = (num_reqs, num_heads, num_splits)
        _inkling_rel_attn_decode_partial[grid](
            q,
            key_cache,
            value_cache,
            rel_logits,
            block_table,
            cache_seqlens,
            partial_m,
            partial_l,
            partial_o,
            q.stride(0),
            q.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
            rel_logits.stride(0),
            rel_logits.stride(1),
            block_table.stride(0),
            partial_m.stride(0),
            partial_m.stride(1),
            partial_m.stride(2),
            partial_o.stride(0),
            partial_o.stride(1),
            partial_o.stride(2),
            partial_o.stride(3),
            softmax_scale,
            kv_block_size,
            kv_group_num,
            HEAD_DIM=head_dim,
            REL_EXTENT=rel_extent,
            SW_LEFT=sw_left,
            SW_RIGHT=sw_right,
            USE_SW=use_sw,
            BLOCKS_PER_SPLIT=blocks_per_split,
            BLOCK_N=kv_block_size,
        )
        _inkling_rel_attn_combine[(num_reqs, num_heads)](
            partial_m,
            partial_l,
            partial_o,
            out,
            partial_m.stride(0),
            partial_m.stride(1),
            partial_m.stride(2),
            partial_o.stride(0),
            partial_o.stride(1),
            partial_o.stride(2),
            partial_o.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            HEAD_DIM=head_dim,
            NUM_SPLITS=num_splits,
        )
    else:
        # ---- prefill (and mixed prefill+decode batches) ----
        num_m_blocks = (max_seqlen_q + _BLOCK_M - 1) // _BLOCK_M
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
            BLOCK_N=kv_block_size,
        )

    return out
