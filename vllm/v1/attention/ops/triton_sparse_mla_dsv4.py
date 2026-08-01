# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton port of FlashInfer's SM120 sparse-MLA paged attention (DSv4).

FlashInfer's ``trtllm_batch_decode_sparse_mla_dsv4`` resolves its backend
from the device architecture and refuses everything it has no kernel for
(``_resolve_dsv4_sparse_mla_backend``: SM100/103 -> trtllm-gen,
SM120/121 -> sparse, anything else -> ValueError; field-hit on Ada at
decode graph capture, AFTER 41 s of weight loading and a clean prefill
profile). This module is an arch-portable implementation of the
``sparse`` backend's exact call surface and semantics, so DSv4 sparse
attention can run on GPUs FlashInfer does not cover. Ada / sm_89 is the
tested target; the kernel itself only assumes Triton.

Semantics are LIFTED from flashinfer 0.6.14, not re-derived:

* ``flashinfer/mla/_core.py`` (``_trtllm_batch_decode_sparse_mla_dsv4_sm120``
  + ``_normalize_sparse_mla_*``) for the call surface, and
* ``include/flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh``
  (``KVCacheTraits<ModelType::DSV4>``) for the packed cache layout,

cross-checked against vLLM's cache writer
(``csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu``).

The contract:

* query: BF16 ``[T, H, 512]`` (or ``[B, Q, H, 512]``), H <= 128;
  512 = 448 nope + 64 rope. K and V are the SAME 512-dim cache row.
* packed ``fp8_ds_mla`` cache: the uint8 tensor ``[pages, pbs, 584]`` is
  a byte container, NOT row-major 584-byte rows. Physical page layout:

  - token data at   ``page*pbs*584 + tok*576``:
    448 B FP8-E4M3 nope + 64 x BF16 rope (128 B)
  - scale footer at ``page*pbs*584 + pbs*576 + tok*8``:
    7 UE8M0 bytes (one per 64-wide nope tile) + 1 pad byte
  - dequant: ``nope_fp8 * 2^(scale_byte - 127)``; rope is stored BF16.

* plain cache: rows ``[pages, pbs, 512]`` in the query dtype (BF16).
* indices: INT32 global token slots (``page = idx // pbs``); entries at
  or past the per-token length, and negative entries, are masked out.
* two segments (SWA + optional compressed) share ONE softmax.
* sinks: natural-log virtual logit per head; contributes softmax mass
  to the denominator only (value 0). Padded heads carry -inf. The sink
  is NOT scaled by ``bmm1_scale``.
* out: BF16 ``[T, H, 512]``. A token whose segments are entirely masked
  produces zeros (with or without a sink), matching the SM120 kernel.

Env knobs (read once, at first use):

* ``VLLM_DSV4_SPARSE_MLA_FORCE_TRITON=1`` - route to this port even on
  architectures FlashInfer supports (A/B numerics + perf on SM120/SM100
  silicon). Read by ``vllm/utils/flashinfer.py``.
* ``VLLM_DSV4_SPARSE_MLA_SELFTEST=0``     - skip the init-time on-device
  self-test (triage escape hatch; the test costs < 100 ms once).
"""

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)

# DSv4 dimensions, mirrored from KVCacheTraits<ModelType::DSV4>.
_D = 512  # d_qk == d_v
_D_NOPE = 448
_D_ROPE = 64
_QUANT_TILE = 64
_NUM_SCALES = 7
_BPT_PACKED = 584  # logical bytes/token of the packed fp8_ds_mla row
_TOKEN_DATA_BYTES = 576  # physical: 448 fp8 + 64 * bf16
_SCALE_BYTES = 8  # footer: 7 UE8M0 + 1 pad

# Finite sentinel for "-inf" in the online softmax. Keeps m/l arithmetic
# NaN-free for all-masked tokens (the SM120 kernel uses the same trick).
_NEG_INF = -1.0e30
_LOG2E = 1.4426950408889634

# Set once at import: TRITON_INTERPRET is read by Triton itself at import
# time, so a consistent snapshot here is correct by construction.
_IS_INTERPRET = os.getenv("TRITON_INTERPRET", "0") == "1"


@functools.cache
def _module_build_id() -> str:
    """Short content hash of THIS file, logged at boot.

    'Which build produced this log' burned a deploy cycle on 2026-07-19:
    a stale image re-raised an already-fixed self-test error and the log
    alone could not prove which code was running (only the traceback
    line numbers gave it away). The self-test line now carries this id;
    compare it to sha256 of the file in the repo when in doubt.
    """
    import hashlib

    with open(__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


@triton.jit
def _dsv4_gather_k(
    kv_ptr,
    idx,  # [BLOCK_N] int64 global slots (invalid lanes pre-zeroed)
    valid,  # [BLOCK_N] bool
    pbs,  # page block size (tokens per page)
    page_stride,  # elements between page starts (uint8 bytes for packed,
                  # query-dtype elements for plain). >= pbs * width.
                  # Read from kv_cache.stride()[0] so padded (alignment-padded)
                  # and contiguous layouts are accepted equally.
    d_offs,  # [D] arange
    IS_PACKED: tl.constexpr,
    D: tl.constexpr,
    D_NOPE: tl.constexpr,
    QUANT_TILE: tl.constexpr,
):
    """Gather a [BLOCK_N, D] fp32 K/V tile from a paged DSv4 cache."""
    if IS_PACKED:
        page = idx // pbs
        tok = idx - page * pbs
        page_base = page * page_stride
        data_base = page_base + tok * 576  # [BLOCK_N]
        is_nope = d_offs < D_NOPE  # [D]

        # FP8-E4M3 nope bytes, decoded with integer math (portable to the
        # Triton interpreter; the writer saturates so 0x7F/0xFF never occur).
        nope_addr = data_base[:, None] + d_offs[None, :]
        nope_mask = valid[:, None] & is_nope[None, :]
        b = tl.load(kv_ptr + nope_addr, mask=nope_mask, other=0).to(tl.uint32)
        sign = (b >> 7) & 1
        expo = ((b >> 3) & 0xF).to(tl.float32)
        mant = (b & 0x7).to(tl.float32)
        mag = tl.where(
            expo == 0.0,
            mant * 0.001953125,  # subnormal: mant/8 * 2^-6 = mant * 2^-9
            tl.exp2(expo - 7.0) * (1.0 + mant * 0.125),
        )
        nope_f32 = tl.where(sign == 1, -mag, mag)

        # UE8M0 tile scales from the page footer, gathered per column.
        tile_id = tl.minimum(d_offs // QUANT_TILE, 6)
        scale_addr = (page_base[:, None] + pbs * 576 + tok[:, None] * 8 +
                      tile_id[None, :])
        s = tl.load(kv_ptr + scale_addr, mask=nope_mask, other=127)
        scale = tl.exp2(s.to(tl.float32) - 127.0)

        # BF16 rope halves -> f32 via bit assembly.
        rope_col = d_offs - D_NOPE
        rope_addr = data_base[:, None] + D_NOPE + 2 * rope_col[None, :]
        rope_mask = valid[:, None] & (d_offs[None, :] >= D_NOPE)
        lo = tl.load(kv_ptr + rope_addr, mask=rope_mask, other=0).to(tl.uint32)
        hi = tl.load(kv_ptr + rope_addr + 1, mask=rope_mask,
                     other=0).to(tl.uint32)
        rope_f32 = ((hi << 24) | (lo << 16)).to(tl.float32, bitcast=True)

        return tl.where(is_nope[None, :], nope_f32 * scale, rope_f32)
    else:
        page = idx // pbs
        tok = idx - page * pbs
        addr = (page[:, None] * page_stride + tok[:, None] * D +
                d_offs[None, :])
        k = tl.load(kv_ptr + addr, mask=valid[:, None], other=0.0)
        return k.to(tl.float32)


@triton.jit
def _sparse_mla_dsv4_kernel(
    q_ptr,
    out_ptr,
    kv_ptr,
    idx_ptr,
    len_ptr,
    extra_kv_ptr,
    extra_idx_ptr,
    extra_len_ptr,
    sink_ptr,
    sm_scale,
    pbs,
    extra_pbs,
    page_stride,
    extra_page_stride,
    K_main,
    K_extra,
    H,
    stride_qt,
    stride_qh,
    stride_ot,
    stride_oh,
    stride_it,
    extra_stride_it,
    HAS_EXTRA: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_LENS: tl.constexpr,
    HAS_EXTRA_LENS: tl.constexpr,
    IS_PACKED: tl.constexpr,
    EXTRA_IS_PACKED: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    D_NOPE: tl.constexpr,
    QUANT_TILE: tl.constexpr,
    DOT_BF16: tl.constexpr,
):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    h_offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    h_ok = h_offs < H
    d_offs = tl.arange(0, D)

    # Keep dot operands bf16 (f32 accumulate): Ada has 99 KiB smem/SM and
    # f32 operand tiles for a D=512 dot do not fit (AOT-compile measured
    # 295 KiB with f32 operands; bf16 + BLOCK_N=32 + num_stages=1 fits).
    # Precision matches the native SM120 kernel, which runs fp8 MMA.
    q = tl.load(
        q_ptr + t * stride_qt + h_offs[:, None] * stride_qh + d_offs[None, :],
        mask=h_ok[:, None],
        other=0.0,
    )

    # Online softmax state, log2 domain.
    m = tl.full([BLOCK_H], -1.0e30, tl.float32)
    l = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D], tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    if HAS_LENS:
        n_main = tl.load(len_ptr + t)
        n_main = tl.maximum(tl.minimum(n_main, K_main), 0)
    else:
        n_main = K_main
    for start in range(0, n_main, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        in_len = offs < n_main
        idx = tl.load(idx_ptr + t * stride_it + offs, mask=in_len, other=-1)
        valid = in_len & (idx >= 0)
        idx0 = tl.where(valid, idx, 0).to(tl.int64)
        k = _dsv4_gather_k(kv_ptr, idx0, valid, pbs, page_stride, d_offs,
                           IS_PACKED, D, D_NOPE, QUANT_TILE)
        if DOT_BF16:
            k = k.to(tl.bfloat16)
            s = tl.dot(q, tl.trans(k)) * qk_scale
        else:
            s = tl.dot(q.to(tl.float32), tl.trans(k)) * qk_scale
        s = tl.where(valid[None, :], s, -1.0e30)
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp2(m - m_new)
        p = tl.where(valid[None, :], tl.exp2(s - m_new[:, None]), 0.0)
        l = l * alpha + tl.sum(p, axis=1)
        if DOT_BF16:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), k)
        else:
            acc = acc * alpha[:, None] + tl.dot(p, k)
        m = m_new

    if HAS_EXTRA:
        if HAS_EXTRA_LENS:
            n_extra = tl.load(extra_len_ptr + t)
            n_extra = tl.maximum(tl.minimum(n_extra, K_extra), 0)
        else:
            n_extra = K_extra
        for start in range(0, n_extra, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            in_len = offs < n_extra
            idx = tl.load(extra_idx_ptr + t * extra_stride_it + offs,
                          mask=in_len,
                          other=-1)
            valid = in_len & (idx >= 0)
            idx0 = tl.where(valid, idx, 0).to(tl.int64)
            k = _dsv4_gather_k(extra_kv_ptr, idx0, valid, extra_pbs,
                               extra_page_stride, d_offs,
                               EXTRA_IS_PACKED, D, D_NOPE, QUANT_TILE)
            if DOT_BF16:
                k = k.to(tl.bfloat16)
                s = tl.dot(q, tl.trans(k)) * qk_scale
            else:
                s = tl.dot(q.to(tl.float32), tl.trans(k)) * qk_scale
            s = tl.where(valid[None, :], s, -1.0e30)
            m_new = tl.maximum(m, tl.max(s, axis=1))
            alpha = tl.exp2(m - m_new)
            p = tl.where(valid[None, :], tl.exp2(s - m_new[:, None]), 0.0)
            l = l * alpha + tl.sum(p, axis=1)
            if DOT_BF16:
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), k)
            else:
                acc = acc * alpha[:, None] + tl.dot(p, k)
            m = m_new

    if HAS_SINK:
        # The sink is a virtual candidate with value 0: it adds softmax
        # mass to the denominator only (FlashMLA V4 semantics; padded
        # heads carry -inf and are a no-op). Natural-log -> log2.
        snk = tl.load(sink_ptr + h_offs, mask=h_ok, other=-1.0e30)
        snk = tl.maximum(snk * 1.4426950408889634, -1.0e30)
        m_new = tl.maximum(m, snk)
        alpha = tl.exp2(m - m_new)
        l = l * alpha + tl.exp2(snk - m_new)
        acc = acc * alpha[:, None]

    # All-masked tokens (l == 0) produce zeros, matching the SM120 kernel.
    l_safe = tl.where(l > 0.0, l, 1.0)
    o = acc / l_safe[:, None]
    tl.store(
        out_ptr + t * stride_ot + h_offs[:, None] * stride_oh + d_offs[None, :],
        o.to(out_ptr.dtype.element_ty),
        mask=h_ok[:, None],
    )


def _normalize_kv_cache(
    kv_cache: torch.Tensor,
    kv_layout: str,
    query_dtype: torch.dtype,
    name: str,
) -> tuple[torch.Tensor, int, bool, int]:
    """Return ``(cache, page_block_size, is_packed, page_stride)``.

    Mirrors flashinfer's ``_check_sm120_dsv4_kv_cache_layout`` +
    ``_packed_kv_page_block_size``: 3-D ``[pages, pbs, w]`` or 4-D with a
    singleton KV-head axis (NHD: dim 2, HND: dim 1); ``w`` is 584 packed
    uint8 or 512 in the query dtype.

    ``page_stride`` is the distance between page starts in elements of
    ``kv_cache.dtype`` (uint8 bytes for the packed path, query-dtype
    elements for the plain path). The upstream allocator may pad pages to
    alignment (e.g. 576-byte alignment for ``fp8_ds_mla``), producing a
    non-contiguous ``as_strided`` view where ``stride(0) > pbs * width``.
    The kernel reads ``page_stride`` from this tensor's own strides and
    uses it for the gather, so both padded and contiguous layouts are
    accepted with no copy.
    """
    if kv_cache.ndim == 4:
        if kv_layout == "NHD":
            if kv_cache.shape[2] != 1:
                raise ValueError(
                    f"{name}: NHD layout needs a singleton KV-head axis in "
                    f"dim 2, got shape {tuple(kv_cache.shape)}")
            pbs = int(kv_cache.shape[1])
        elif kv_layout == "HND":
            if kv_cache.shape[1] != 1:
                raise ValueError(
                    f"{name}: HND layout needs a singleton KV-head axis in "
                    f"dim 1, got shape {tuple(kv_cache.shape)}")
            pbs = int(kv_cache.shape[2])
        else:
            raise ValueError(f"{name}: kv_layout must be NHD or HND, got "
                             f"{kv_layout!r}")
    elif kv_cache.ndim == 3:
        pbs = int(kv_cache.shape[1])
    else:
        raise ValueError(
            f"{name} must have ndim 3 or 4, got {kv_cache.ndim}")
    # Accept either contiguous or alignment-padded layouts. The page stride
    # (distance between page starts in elements of kv_cache.dtype) is read
    # from the tensor's own strides; the kernel uses it instead of assuming
    # pbs * width. The minimum valid stride covers one page of data; reject
    # anything smaller as a malformed cache.
    page_stride = int(kv_cache.stride()[0])
    width = int(kv_cache.shape[-1])
    min_stride = pbs * width
    if page_stride < min_stride:
        raise ValueError(
            f"{name}: page stride {page_stride} < minimum {min_stride} "
            f"(pbs={pbs}, width={width}, dtype={kv_cache.dtype}); cache "
            "layout inconsistent with the spec")

    if kv_cache.dtype == torch.uint8:
        if width != _BPT_PACKED:
            raise ValueError(
                f"Expected packed DSV4 {name} head dim {_BPT_PACKED}, got "
                f"{width}")
        return kv_cache, pbs, True, page_stride
    if kv_cache.dtype != query_dtype:
        raise ValueError(f"{name} dtype must match query dtype, got "
                         f"{kv_cache.dtype} and {query_dtype}")
    if width != _D:
        raise ValueError(f"Expected {name} head dim {_D}, got {width}")
    return kv_cache, pbs, False, page_stride


def _normalize_indices(
    indices: torch.Tensor,
    lens: torch.Tensor | None,
    num_tokens: int,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Flatten ``[T, K]`` / ``[T, 1, K]`` index tensors; validate dtypes.

    Mirrors flashinfer's ``_normalize_sparse_mla_indices_and_lens`` for the
    flat-token case (batch = T, q_len = 1).
    """
    if indices.ndim == 3:
        if indices.shape[0] * indices.shape[1] != num_tokens:
            raise ValueError(
                f"{name}: shape {tuple(indices.shape)} does not cover "
                f"{num_tokens} query tokens")
        indices = indices.reshape(num_tokens, indices.shape[-1])
    elif indices.ndim == 2:
        if indices.shape[0] != num_tokens:
            raise ValueError(
                f"{name}: shape {tuple(indices.shape)} does not cover "
                f"{num_tokens} query tokens")
    else:
        raise ValueError(f"{name} must have ndim 2 or 3, got {indices.ndim}")
    if indices.shape[-1] <= 0:
        raise ValueError(f"{name} requires top-k > 0")
    if indices.dtype != torch.int32:
        raise ValueError(f"{name} must be int32, got {indices.dtype}")
    if indices.stride(-1) != 1:
        indices = indices.contiguous()

    if lens is not None:
        lens = lens.reshape(-1)
        if lens.numel() != num_tokens:
            raise ValueError(f"{name}_lens must have {num_tokens} entries, "
                             f"got {lens.numel()}")
        if lens.dtype != torch.int32:
            raise ValueError(f"{name}_lens must be int32, got {lens.dtype}")
        if not lens.is_contiguous():
            lens = lens.contiguous()
    return indices, lens


def triton_sparse_mla_dsv4(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor | None = None,
    sparse_indices: torch.Tensor | None = None,
    compressed_kv_cache: torch.Tensor | None = None,
    sparse_topk_lens: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    sinks: torch.Tensor | None = None,
    kv_layout: str = "HND",
    cum_seq_lens_q: torch.Tensor | None = None,
    max_q_len: int | None = None,
    enable_pdl: bool | None = None,
    swa_topk_lens: torch.Tensor | None = None,
    extra_sparse_indices: torch.Tensor | None = None,
    extra_sparse_topk_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Arch-portable DSv4 sparse-MLA attention (FlashInfer ``sparse`` mode).

    The signature mirrors flashinfer 0.6.14's
    ``trtllm_batch_decode_sparse_mla_dsv4`` so the dispatcher in
    ``vllm/utils/flashinfer.py`` can pass calls through unchanged.
    ``workspace_buffer`` and ``enable_pdl`` are accepted and ignored (the
    kernel needs neither); the trtllm-gen-only calling convention
    (``sparse_topk_lens``/``seq_lens``/``cum_seq_lens_q``) is refused with
    the remedy named.

    CUDA-graph safe: no host synchronization, no data-dependent shapes;
    lengths are consumed on-device.
    """
    del workspace_buffer, enable_pdl, max_q_len  # not needed by the port
    if sparse_indices is None:
        raise ValueError("sparse_indices is required")
    if (sparse_topk_lens is not None or seq_lens is not None
            or cum_seq_lens_q is not None):
        raise NotImplementedError(
            "DSv4 sparse MLA (Triton port) implements FlashInfer's "
            "SM120-mode calling convention (swa_topk_lens / "
            "extra_sparse_indices); the trtllm-gen convention "
            "(sparse_topk_lens / seq_lens / cum_seq_lens_q) is SM100-only. "
            "Route this call through the SM120-mode call sites "
            "(DeepseekV4FlashInferSM120Attention).")
    if isinstance(bmm1_scale, torch.Tensor) or isinstance(
            bmm2_scale, torch.Tensor):
        raise ValueError(
            "DSv4 sparse MLA (Triton port) expects float bmm scales")
    if float(bmm2_scale) != 1.0:
        raise ValueError("DSv4 sparse MLA does not support bmm2_scale")
    if query.dtype != torch.bfloat16:
        raise ValueError(
            f"DSv4 sparse MLA (Triton port) only supports BF16 query, got "
            f"{query.dtype}")

    orig_ndim = query.ndim
    if orig_ndim == 4:
        b, qlen, num_heads, head_dim = query.shape
        query_flat = query.reshape(b * qlen, num_heads, head_dim)
    elif orig_ndim == 3:
        query_flat = query
        num_heads, head_dim = query.shape[-2:]
    else:
        raise ValueError(f"Expected query.ndim == 3 or 4, got {query.ndim}")
    if head_dim != _D:
        raise ValueError(f"Expected DSv4 query head dim {_D}, got {head_dim}")
    if num_heads > 128:
        raise ValueError(f"Expected num_heads <= 128, got {num_heads}")
    if query_flat.stride(-1) != 1:
        query_flat = query_flat.contiguous()
    num_tokens = query_flat.shape[0]

    kv_main, pbs_main, packed_main, page_stride_main = _normalize_kv_cache(
        swa_kv_cache, kv_layout, query.dtype, "swa_kv_cache")
    idx_main, lens_main = _normalize_indices(sparse_indices, swa_topk_lens,
                                             num_tokens, "sparse_indices")

    has_extra = extra_sparse_indices is not None
    if has_extra != (extra_sparse_topk_lens is not None):
        raise ValueError(
            "extra_sparse_indices and extra_sparse_topk_lens must be "
            "provided together")
    if has_extra:
        if compressed_kv_cache is None:
            raise ValueError("compressed_kv_cache is required when "
                             "extra_sparse_indices is provided")
        kv_extra, pbs_extra, packed_extra, page_stride_extra = (
            _normalize_kv_cache(
                compressed_kv_cache, kv_layout, query.dtype,
                "compressed_kv_cache"))
        idx_extra, lens_extra = _normalize_indices(extra_sparse_indices,
                                                   extra_sparse_topk_lens,
                                                   num_tokens,
                                                   "extra_sparse_indices")
    else:
        kv_extra, pbs_extra, packed_extra, page_stride_extra = (
            kv_main, pbs_main, packed_main, page_stride_main)
        idx_extra, lens_extra = idx_main, lens_main

    expected_out_shape = ((b, qlen, num_heads, _D) if orig_ndim == 4 else
                          (num_tokens, num_heads, _D))
    if out is None:
        out = torch.empty(expected_out_shape,
                          dtype=torch.bfloat16,
                          device=query.device)
    else:
        if tuple(out.shape) != expected_out_shape:
            raise ValueError(f"out shape {tuple(out.shape)} != expected "
                             f"{expected_out_shape}")
        if out.dtype != torch.bfloat16:
            raise ValueError(f"out must be bf16, got {out.dtype}")
    out_flat = out.view(num_tokens, num_heads, _D)
    if out_flat.stride(-1) != 1:
        raise ValueError("out must have a unit stride head dim")

    if sinks is not None:
        if sinks.dtype != torch.float32 or sinks.numel() < num_heads:
            raise ValueError(
                f"sinks must be fp32 with >= {num_heads} entries, got "
                f"{sinks.dtype} x {sinks.numel()}")
        if not sinks.is_contiguous():
            sinks = sinks.contiguous()

    logger.info_once(
        "DSv4 sparse-MLA Triton port geometry: q%s %s, swa cache %s %s "
        "(pbs=%d, packed=%s, page_stride=%d), K_main=%d, extra=%s, "
        "K_extra=%s, extra_page_stride=%d, sinks=%s",
        tuple(query_flat.shape),
        query.dtype,
        tuple(swa_kv_cache.shape),
        swa_kv_cache.dtype,
        pbs_main,
        packed_main,
        page_stride_main,
        idx_main.shape[-1],
        has_extra,
        idx_extra.shape[-1] if has_extra else 0,
        page_stride_extra,
        sinks is not None,
    )

    block_h = 16
    grid = (num_tokens, triton.cdiv(num_heads, block_h))
    if num_tokens > 0:
        _sparse_mla_dsv4_kernel[grid](
            query_flat,
            out_flat,
            kv_main,
            idx_main,
            lens_main if lens_main is not None else idx_main,
            kv_extra,
            idx_extra,
            lens_extra if lens_extra is not None else idx_extra,
            sinks if sinks is not None else query_flat,
            float(bmm1_scale),
            pbs_main,
            pbs_extra,
            page_stride_main,
            page_stride_extra,
            idx_main.shape[-1],
            idx_extra.shape[-1] if has_extra else 0,
            num_heads,
            query_flat.stride(0),
            query_flat.stride(1),
            out_flat.stride(0),
            out_flat.stride(1),
            idx_main.stride(0),
            idx_extra.stride(0),
            HAS_EXTRA=has_extra,
            HAS_SINK=sinks is not None,
            HAS_LENS=lens_main is not None,
            HAS_EXTRA_LENS=has_extra and lens_extra is not None,
            IS_PACKED=packed_main,
            EXTRA_IS_PACKED=packed_extra,
            BLOCK_H=block_h,
            BLOCK_N=32,
            D=_D,
            D_NOPE=_D_NOPE,
            QUANT_TILE=_QUANT_TILE,
            # The Triton interpreter (numpy) cannot emulate bf16 dots; the
            # GPU build MUST use bf16 operands to fit Ada's 99 KiB smem
            # (f32 operand tiles measured 295 KiB via AOT compile). The
            # interpreter tier validates semantics in f32; the init-time
            # self-test validates the shipped bf16 path on real silicon.
            DOT_BF16=not _IS_INTERPRET,
            num_warps=4,
            num_stages=1,
        )
    return out


def dequant_dsv4_packed_cache(kv_cache: torch.Tensor,
                              pbs: int) -> torch.Tensor:
    """Decode a packed fp8_ds_mla cache to f32 rows ``[pages*pbs, 512]``.

    Reference-path helper (tests, self-test, torch fallback). Operates on
    the byte container in its NATIVE storage layout - data first, scale
    footer last - before any logical reshape.
    """
    pages = kv_cache.shape[0]
    flat = kv_cache.reshape(pages, -1)
    assert flat.shape[1] == pbs * _BPT_PACKED
    data = flat[:, :pbs * _TOKEN_DATA_BYTES].reshape(pages, pbs,
                                                     _TOKEN_DATA_BYTES)
    nope = data[..., :_D_NOPE].view(torch.float8_e4m3fn).to(torch.float32)
    rope = data[..., _D_NOPE:].contiguous().view(torch.bfloat16).to(
        torch.float32)
    scales = (flat[:, pbs * _TOKEN_DATA_BYTES:].reshape(
        pages, pbs, _SCALE_BYTES)[..., :_NUM_SCALES].to(torch.int32) - 127)
    factor = torch.ldexp(torch.ones_like(scales, dtype=torch.float32),
                         scales)
    factor = factor.repeat_interleave(_QUANT_TILE, dim=-1)
    k = torch.cat([nope * factor, rope], dim=-1)
    return k.reshape(pages * pbs, _D)


def sparse_mla_dsv4_torch_ref(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    *,
    compressed_kv_cache: torch.Tensor | None = None,
    swa_topk_lens: torch.Tensor | None = None,
    extra_sparse_indices: torch.Tensor | None = None,
    extra_sparse_topk_lens: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    sinks: torch.Tensor | None = None,
    kv_layout: str = "NHD",
) -> torch.Tensor:
    """Pure-torch reference with identical semantics (fp32 math)."""
    assert float(bmm2_scale) == 1.0
    orig_shape = query.shape
    query_flat = query.reshape(-1, query.shape[-2], query.shape[-1])
    num_tokens, num_heads, _ = query_flat.shape

    def rows(cache: torch.Tensor) -> torch.Tensor:
        cache3, pbs, packed, _ = _normalize_kv_cache(cache, kv_layout,
                                                     query.dtype, "cache")
        cache3 = cache3.contiguous().reshape(cache3.shape[0], -1)
        if packed:
            return dequant_dsv4_packed_cache(cache3, pbs)
        return cache3.reshape(-1, _D).to(torch.float32)

    segments = [(rows(swa_kv_cache), sparse_indices, swa_topk_lens)]
    if extra_sparse_indices is not None:
        assert compressed_kv_cache is not None
        segments.append((rows(compressed_kv_cache), extra_sparse_indices,
                         extra_sparse_topk_lens))

    q = query_flat.to(torch.float32)
    all_scores = []
    all_values = []
    for k_rows, indices, lens in segments:
        idx, lens = _normalize_indices(indices, lens, num_tokens, "indices")
        idx = idx.to(torch.int64)
        valid = idx >= 0
        if lens is not None:
            width = idx.shape[-1]
            pos = torch.arange(width, device=idx.device)
            valid &= pos[None, :] < lens.to(torch.int64)[:, None]
        k = k_rows[idx.clamp(min=0)]  # [T, K, D]
        s = torch.einsum("thd,tkd->thk", q, k) * float(bmm1_scale)
        s = s.masked_fill(~valid[:, None, :], float("-inf"))
        all_scores.append(s)
        all_values.append(torch.where(valid[..., None], k, 0.0))
    scores = torch.cat(all_scores, dim=-1)  # [T, H, Ktot]
    values = torch.cat(all_values, dim=-2)  # [T, Ktot, D]

    if sinks is not None:
        sink = sinks[:num_heads].to(torch.float32)
        scores = torch.cat(
            [scores,
             sink.view(1, num_heads, 1).expand(num_tokens, -1, -1)],
            dim=-1)
        values = torch.cat(
            [values, torch.zeros_like(values[:, :1])], dim=-2)

    m = scores.amax(dim=-1, keepdim=True).clamp_min(_NEG_INF)
    p = torch.exp(scores - m)
    p = torch.nan_to_num(p, nan=0.0)  # -inf - m for all-masked rows
    denom = p.sum(dim=-1, keepdim=True)
    o = torch.einsum("thk,tkd->thd", p, values) / denom.clamp_min(1e-38)
    o = torch.where(denom > 0, o, 0.0)
    result = o.to(torch.bfloat16).reshape(*orig_shape[:-1], _D)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _self_test_case(device: torch.device) -> float:
    """Build and run the canonical self-test case; return worst_rel.

    Device-agnostic on purpose: CI executes this EXACT boot-time code
    path under ``TRITON_INTERPRET=1`` on CPU. The original self-test
    called the kernel with positional arguments and was only reachable
    with CUDA present - after the public signature grew
    ``workspace_buffer`` in third position (to mirror flashinfer), the
    indices landed in ``workspace_buffer`` and the first REAL boot died
    at layer init with 'sparse_indices is required' (field-hit on the
    4090, 2026-07-19). Everything here is keyword-only for that reason,
    and the test suite runs this function on CPU so the boot path can
    never again be the one untested path.
    """
    # Every float tensor here carries an EXPLICIT dtype: this function
    # executes inside the model loader's set_default_dtype(bf16) context
    # at boot, so default-dtype allocations silently come out bf16 there
    # (second field hit on the 4090: the sinks tensor). The CPU tier
    # reruns this function under a bf16 ambient default to keep it true.
    torch.manual_seed(0)
    pages, pbs, t, h = 6, 64, 4, 16
    kv_f32 = torch.randn(pages * pbs, _D, device=device,
                         dtype=torch.float32) * 2.0
    packed = pack_dsv4_reference_cache(kv_f32, pbs)
    plain = torch.randn(pages, pbs, _D, device=device,
                        dtype=torch.bfloat16)
    q = torch.randn(t, h, _D, device=device, dtype=torch.bfloat16)
    idx = torch.randint(0, pages * pbs, (t, 128),
                        device=device,
                        dtype=torch.int32)
    idx[0, 5] = -1  # index padding
    lens = torch.tensor([128, 100, 0, 64], device=device, dtype=torch.int32)
    eidx = torch.randint(0, pages * pbs, (t, 1, 256),
                         device=device,
                         dtype=torch.int32)
    elens = torch.tensor([256, 0, 17, 256],
                         device=device,
                         dtype=torch.int32)
    sinks = torch.full((h,), -float("inf"), device=device,
                       dtype=torch.float32)
    sinks[:8] = torch.randn(8, device=device, dtype=torch.float32)
    kwargs = dict(
        query=q,
        swa_kv_cache=plain.unsqueeze(-2),
        sparse_indices=idx,
        compressed_kv_cache=packed.view(pages, pbs, 1, _BPT_PACKED),
        swa_topk_lens=lens,
        extra_sparse_indices=eidx,
        extra_sparse_topk_lens=elens,
        bmm1_scale=_D**-0.5,
        sinks=sinks,
        kv_layout="NHD",
    )
    got = triton_sparse_mla_dsv4(**kwargs)
    want = sparse_mla_dsv4_torch_ref(**kwargs)
    return _worst_row_rel(got, want)


def _worst_row_rel(got: torch.Tensor, want: torch.Tensor) -> float:
    """Worst error per (token, head) output row, relative to row magnitude.

    NOT per-element max-rel with a small clamp floor: the shipped GPU
    path rounds the softmax weights to bf16 before the value dot, so
    near-zero output elements (heavy +/- cancellation across candidates)
    carry absolute noise ~1e-3 while the row's real magnitude is O(1-5).
    Per-element max-rel with a 1e-2 floor scored that noise at 2e-1 and
    failed a CORRECT kernel on the 4090 (2026-07-19, boot 4;
    worst_rel=1.984e-01 on-device, 2.6e-01 reproduced off-GPU by
    emulating the kernel's exact rounding chain in torch - which is also
    how this metric's 3e-2 threshold was calibrated: emulated bf16 path
    scores 6.1e-3, f32 interpreter path scores lower still, real bugs
    like a wrong scale decode score O(1))."""
    got = got.to(torch.float32)
    want = want.to(torch.float32)
    row_ref = want.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    return ((got - want).abs() / row_ref).max().item()


@functools.cache
def dsv4_sparse_mla_self_test() -> None:
    """One-shot on-device self-test: Triton port vs. the torch reference.

    Runs at attention-layer init on architectures that take the port (so a
    codegen/driver regression fails the BOOT, attributed, instead of
    corrupting outputs). Covers: packed + plain caches, both segments,
    sinks, len-0 tokens and -1 index padding.
    """
    if os.getenv("VLLM_DSV4_SPARSE_MLA_SELFTEST", "1") in ("0", "false",
                                                           "no", "off"):
        logger.warning_once(
            "DSv4 sparse-MLA Triton port self-test SKIPPED "
            "(VLLM_DSV4_SPARSE_MLA_SELFTEST=0)")
        return
    if not (HAS_TRITON and torch.cuda.is_available()):
        return
    device = torch.device("cuda", torch.cuda.current_device())
    worst_row_rel = _self_test_case(device)
    cap = torch.cuda.get_device_capability(device)
    logger.info(
        "DSv4 sparse-MLA Triton port self-test on sm_%d%d (%s): "
        "worst_row_rel=%.3e vs torch reference (expected ~1e-2 for the "
        "bf16 path; module build %s, torch %s, triton %s)",
        cap[0], cap[1], torch.cuda.get_device_name(device), worst_row_rel,
        _module_build_id(), torch.__version__, triton.__version__)
    if worst_row_rel > 3e-2:
        raise RuntimeError(
            f"DSv4 sparse-MLA Triton port self-test FAILED on this device: "
            f"worst_row_rel={worst_row_rel:.3e} vs torch reference "
            f"(threshold 3e-2, expected ~1e-2; module build "
            f"{_module_build_id()}). This is a codegen/driver regression, "
            f"not a model problem. Set VLLM_DSV4_SPARSE_MLA_SELFTEST=0 "
            f"only to triage.")


def pack_dsv4_reference_cache(kv_rows: torch.Tensor,
                              pbs: int) -> torch.Tensor:
    """Encode f32 rows ``[pages*pbs, 512]`` into the packed fp8_ds_mla
    byte layout (reference writer for tests/self-test; mirrors
    ``fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu``: per-64-tile
    power-of-two UE8M0 scales, FP8-E4M3 saturating quant, BF16 rope)."""
    assert kv_rows.shape[-1] == _D and kv_rows.shape[0] % pbs == 0
    pages = kv_rows.shape[0] // pbs
    nope = kv_rows[:, :_D_NOPE].reshape(-1, _NUM_SCALES, _QUANT_TILE)
    amax = nope.abs().amax(dim=-1).clamp_min(2.0**-126)
    # Power-of-two scale covering amax/448 (mirror exp2f(ceil(log2)))
    exponent = torch.ceil(torch.log2(amax / 448.0))
    scale_byte = (exponent + 127).clamp(0, 254).to(torch.uint8)
    inv_scale = torch.exp2(-exponent)
    quant = (nope * inv_scale[..., None]).clamp(-448, 448).to(
        torch.float8_e4m3fn)
    rope = kv_rows[:, _D_NOPE:].to(torch.bfloat16)

    # Assemble per-page byte regions, then concatenate: [data | footer].
    # (Slice-views of the final tensor are non-contiguous; build forward.)
    quant_u8 = quant.reshape(pages, pbs, _D_NOPE).view(torch.uint8)
    rope_u8 = rope.reshape(pages, pbs, _D_ROPE).contiguous().view(torch.uint8)
    data_bytes = torch.cat([quant_u8, rope_u8],
                           dim=-1).reshape(pages, pbs * _TOKEN_DATA_BYTES)
    pad = torch.zeros(pages, pbs, 1, dtype=torch.uint8,
                      device=kv_rows.device)
    footer_bytes = torch.cat(
        [scale_byte.reshape(pages, pbs, _NUM_SCALES), pad],
        dim=-1).reshape(pages, pbs * _SCALE_BYTES)
    return torch.cat([data_bytes, footer_bytes],
                     dim=1).view(pages, pbs, _BPT_PACKED)
