# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ampere sparse decode for DeepSeek V4 with FP8 KV cache.

Strategy: dequantize FP8 UE8M0 KV cache pages to BF16 on the fly,
then reuse the BF16 sparse MLA attention kernel (xpu_sparse_mla_bf16).
This keeps the external KV cache layout identical to CUDA/ROCm.
"""

import os

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.xpu_mla_sparse import (
    triton_bf16_mla_sparse_interface,
)

# FP8 DS MLA cache layout constants
TOKEN_FP8_DIM = 448  # NoPE portion in FP8
TOKEN_BF16_DIM = 64  # RoPE portion in BF16
TOKEN_SCALE_DIM = 8  # UE8M0 scales per token
QUANT_BLOCK_SIZE = 64  # Elements per quant block
OUTPUT_DIM = 512  # = TOKEN_FP8_DIM + TOKEN_BF16_DIM after dequant
TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2  # 576 bytes per token

# [DSv4-ampere debug] One-shot non-finite probe for the decode kernel internals.
# Logs (once per process PER PROBE NAME) the first step at which a tensor inside
# the decode path becomes non-finite, so the divergence point (dequant overflow
# vs softmax overflow vs apply_attn_sink) is visible. Gated by
# VLLM_SM86_NAN_PROBE=1.
_DECODE_PROBE_FIRED = {}


def _decode_probe(name: str, t) -> None:
    import os

    if os.environ.get("VLLM_SM86_NAN_PROBE") != "1":
        return
    if t is None:
        return
    tf = t.detach().float()
    if not tf.numel():
        return
    has_nonfinite = bool(
        torch.isnan(tf).any().item() or torch.isinf(tf).any().item()
    )
    if not has_nonfinite:
        return
    if _DECODE_PROBE_FIRED.get(name):
        return
    _DECODE_PROBE_FIRED[name] = True
    n_nan = int(torch.isnan(tf).sum().item())
    n_inf = int(torch.isinf(tf).sum().item())
    amax = float(tf.abs().amax().item())
    print(
        f"[DECODE_PROBE ### {name} ###] nan={n_nan} inf={n_inf} "
        f"absmax={amax:.6e} shape={tuple(t.shape)}",
        flush=True,
    )


def _nan_slot_diagnostic(topk_buf_3d, topk_idx_2d, topk_lens, kv_cache):
    """When topk_dequant has NaN, read the offending slot's ACTUAL stored bytes
    directly from cache storage (fp8 NoPE + bf16 RoPE + UE8M0 scales) and report
    whether storage genuinely holds NaN there.

    This is the DECISIVE discriminator between the two remaining hypotheses:
      (A) DATA-NaN: the slot's storage is actually NaN (corrupted between its
          write ~token 2010 and this read ~token 2053 by an OOB write) -> the
          per-write readback (which found 0 from-finite-input NaN) was correct;
          the corruption is a different write touching this slot.
      (B) GATHER-NaN: storage is CLEAN but the gather/dequant produced NaN
          (wrong offset / bad scale / kernel bug) -> read-side bug.
    Also pinpoints WHICH region (fp8 NoPE vs bf16 RoPE) and, for fp8, the stored
    byte + its scale (to catch scale-overflow -> Inf, or fp8 NaN-encoding).

    topk_buf_3d: (num_tokens, max_topk, OUTPUT_DIM=512) dequantized gathered
    topk_idx_2d: (num_tokens, max_topk) global slot IDs used for the gather
    kv_cache:    (num_blocks, block_size, 584) the cache we gathered from
    """
    if os.environ.get("VLLM_SM86_NAN_PROBE") != "1":
        return
    if _DECODE_PROBE_FIRED.get("nan_slot_diag"):
        return
    try:
        nan_mask = torch.isnan(topk_buf_3d.float()).any(
            2)  # (num_tokens, max_topk)
        if not bool(nan_mask.any().item()):
            return
        _DECODE_PROBE_FIRED["nan_slot_diag"] = True
        bad = torch.where(nan_mask)
        tok_i = int(bad[0][0].item())
        slot_j = int(bad[1][0].item())
        slot_id = int(topk_idx_2d[tok_i, slot_j].item())
        # Which OUTPUT dim is NaN? (0..447 = dequantized fp8 NoPE;
        # 448..511 = bf16 RoPE)
        row = topk_buf_3d[tok_i, slot_j].float()  # (512,)
        row_nan = torch.isnan(row)
        nan_cols = torch.where(row_nan)[0]
        first_nan_col = int(nan_cols[0].item()) if nan_cols.numel() else -1
        n_nan = int(row_nan.sum().item())
        region = ("BF16_ROPE" if first_nan_col >= 448
                  else "FP8_NoPE" if first_nan_col >= 0 else "?")

        # ---- Read the slot's ACTUAL stored bytes from cache storage ----
        # Matches the kernel layout EXACTLY:
        #   block_idx = slot_id // 64; pos = slot_id % 64
        #   token_data @ storage_offset + block*stride0 + pos*576
        #     fp8 NoPE [+0:+448]; bf16 RoPE [+448:+576]
        #   scales    @ storage_offset + block*stride0 + 64*576 + pos*8 (8 bytes)
        block_size = kv_cache.shape[1]
        s0 = kv_cache.stride(0)
        st = kv_cache.untyped_storage()
        n_elem = st.nbytes() // kv_cache.element_size()
        flat = torch.empty(0, dtype=kv_cache.dtype, device=kv_cache.device)
        flat.set_(st, 0, (n_elem,), (1,))
        base = kv_cache.storage_offset()
        block_idx = slot_id // block_size
        pos = slot_id % block_size
        tok_base = base + block_idx * s0 + pos * 576
        scale_base = base + block_idx * s0 + block_size * 576 + pos * 8

        # fp8 NoPE bytes (448) -> check for NaN/Inf encoding (e4m3fn: 0x7F,0xFF)
        fp8_bytes = flat[tok_base:tok_base + 448].to(torch.uint8)
        fp8_nan_byte_mask = (fp8_bytes == 0x7F) | (fp8_bytes == 0xFF)
        fp8_nan_bytes = int(fp8_nan_byte_mask.sum().item())
        # DISAMBIGUATION: is this slot UNWRITTEN (all-zero except NaN scribbles)
        # or WRITTEN-WITH-NaN-INPUT (many nonzero fp8 + scale 0 from NaN exponent)?
        # A NaN-input write produces fp8 bytes 0x7F/0xFF for MOST elements; an
        # unwritten slot is all 0x00 except sparse OOB scribbles. Count nonzero
        # (non-NaN-encoding) fp8 bytes + nonzero bf16 RoPE values.
        fp8_nonzero = int(((fp8_bytes != 0x00) & (~fp8_nan_byte_mask)).sum().item())
        fp8_zero = int((fp8_bytes == 0x00).sum().item())
        # bf16 RoPE (64 values)
        rope_vals = flat[tok_base + 448:tok_base + 576].view(torch.bfloat16).float()
        rope_nf = bool((torch.isnan(rope_vals) | torch.isinf(rope_vals)).any().item())
        rope_nan = int(torch.isnan(rope_vals).sum().item())
        rope_inf = int(torch.isinf(rope_vals).sum().item())
        rope_nonzero = int(((rope_vals != 0.0) & torch.isfinite(rope_vals)).sum().item())
        # scales (8 bytes) -> flag overflow (byte 255 -> 2^128)
        scale_bytes = flat[scale_base:scale_base + 8].to(torch.uint8)
        scale_list = scale_bytes.tolist()
        has_overflow_scale = any(b >= 250 for b in scale_list)
        all_scales_zero = all(b == 0 for b in scale_list)

        # If the NaN col is fp8 NoPE, report that exact stored byte + its scale
        fp8_detail = ""
        if 0 <= first_nan_col < 448:
            sb = int(fp8_bytes[first_nan_col].item())
            sc_idx = first_nan_col // 64
            sc = int(scale_bytes[sc_idx].item()) if sc_idx < len(scale_list) else -1
            fp8_detail = (f" stored_fp8_byte_at_col{first_nan_col}=0x{sb:02x} "
                          f"scale[{sc_idx}]={sc}->2^{sc-127}")
        rope_detail = ""
        if first_nan_col >= 448:
            ri = first_nan_col - 448
            rv = float(rope_vals[ri].item()) if ri < rope_vals.numel() else None
            rope_detail = f" stored_bf16_at_col{first_nan_col}[{ri}]={rv}"

        storage_verdict = ("STORAGE_HAS_NaN" if (fp8_nan_bytes or rope_nan)
                           else "STORAGE_CLEAN")
        # DISAMBIGUATION verdict: unwritten vs written-with-NaN-input.
        # all_scales_zero AND (fp8 nearly all zero) => UNWRITTEN (OOB scribble).
        # nonzero fp8/rope data present => WRITTEN (NaN input => downstream).
        slot_kind = ("UNWRITTEN" if (all_scales_zero and fp8_nonzero == 0
                                      and rope_nonzero == 0)
                     else "WRITTEN-DATA-PRESENT")
        if storage_verdict == "STORAGE_HAS_NaN":
            if slot_kind == "UNWRITTEN":
                interp = ("(A1) DATA-NaN in UNWRITTEN slot: zero-init slot had "
                          "NaN SCRIBBLED by an OOB write (sparse: only "
                          f"{fp8_nan_bytes} fp8 + {rope_nan} rope NaN bytes). "
                          "Root = OOB write + indexer selecting unwritten slot.")
            else:
                interp = ("(A2) DATA-NaN in WRITTEN slot: nonzero fp8/rope data "
                          "present => compressor WROTE this slot with NaN INPUT "
                          "(residual already NaN) => DOWNSTREAM symptom, not root.")
        else:
            interp = ("(B) GATHER-NaN: storage clean; gather/dequant kernel "
                      "produced NaN from clean data -> read-side kernel bug")
        print(
            f"[NAN_SLOT_DIAG] slot_id={slot_id} (block={block_idx} pos={pos}) "
            f"gathered_nNaN={n_nan} first_nan_col={first_nan_col} region={region}\n"
            f"  -> STORAGE: {storage_verdict} | slot_kind={slot_kind} | "
            f"fp8_nan_bytes={fp8_nan_bytes} fp8_nonzero={fp8_nonzero} "
            f"fp8_zero={fp8_zero} rope_nan={rope_nan} rope_inf={rope_inf} "
            f"rope_nonzero={rope_nonzero} scales={scale_list} "
            f"all_scales_zero={all_scales_zero} overflow_scale={has_overflow_scale}"
            f"{fp8_detail}{rope_detail}\n"
            f"  -> {interp}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[NAN_SLOT_DIAG] failed: {e}", flush=True)


@triton.jit
def _dequant_gather_slots_kernel(
    # Output workspace: [total_slots, OUTPUT_DIM] bf16
    out_ptr,
    # FP8 paged cache base pointer (uint8 flat)
    cache_ptr,
    # Global slot indices: [total_slots] int32
    indices_ptr,
    # Cache geometry
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,  # 576
    block_stride: tl.int64,  # k_cache.stride(0) — total uint8 per block
    fp8_dim: tl.constexpr,  # 448
    bf16_dim: tl.constexpr,  # 64
    scale_dim: tl.constexpr,  # 8
    quant_block: tl.constexpr,  # 64
    output_dim: tl.constexpr,  # 512
    n_quant_blocks: tl.constexpr,  # 7
):
    """Dequantize scattered FP8 slots into a flat BF16 workspace.

    Grid: [total_slots] — one program per slot to gather.

    Cache block layout (block_size tokens):
      [0, block_size*576): Token data, each token 448 FP8 + 128 BF16
      [block_size*576, block_size*576 + block_size*8): Scales
    """
    pid = tl.program_id(0)

    # Load global slot index
    slot_idx = tl.load(indices_ptr + pid).to(tl.int64)

    # Output pointer for this slot
    out_row_ptr = out_ptr + pid * output_dim

    # Handle invalid slots (index < 0): write zeros
    if slot_idx < 0:
        zero = tl.zeros([quant_block], dtype=tl.bfloat16)
        for i in tl.static_range(0, 512, 64):
            offsets = i + tl.arange(0, quant_block)
            mask = offsets < output_dim
            tl.store(out_row_ptr + offsets, zero, mask=mask)
        return

    # Compute block and position within block
    block_idx = slot_idx // cache_block_size
    pos_in_block = slot_idx % cache_block_size

    # Block base pointer
    block_base = cache_ptr + block_idx * block_stride

    # Token data: at offset pos_in_block * token_data_size within block
    token_data_ptr = block_base + pos_in_block * token_data_size

    # Scale: after all token data, at offset
    # cache_block_size * token_data_size + pos_in_block * scale_dim
    scale_region_offset = tl.cast(cache_block_size, tl.int64) * token_data_size
    token_scale_ptr = block_base + scale_region_offset + pos_in_block * scale_dim

    # ========== Dequantize FP8 portion (448 elements) ==========
    for qblock_idx in tl.static_range(n_quant_blocks):
        qblock_start = qblock_idx * quant_block
        offsets = qblock_start + tl.arange(0, quant_block)
        mask = offsets < fp8_dim

        # Load FP8 e4m3fn bytes. SM8x Triton cannot bitcast to fp8e4nv, so
        # decode the e4m3fn format manually (sign.4-bit exp.3-bit mantissa,
        # bias 7): normal = (1 + m/8) * 2^(e-7); subnormal (e==0) = m/8 * 2^-6.
        x_uint8 = tl.load(token_data_ptr + offsets, mask=mask, other=0)
        xi = x_uint8.to(tl.int32)
        sign = (xi >> 7) & 1
        exp = (xi >> 3) & 0xF
        mant = (xi & 0x7).to(tl.float32)
        normal = (1.0 + mant * 0.125) * tl.exp2(exp.to(tl.float32) - 7.0)
        subnorm = mant * 0.125 * tl.exp2(-6.0)
        x_float = tl.where(exp == 0, subnorm, normal) * (
            1.0 - 2.0 * sign.to(tl.float32)
        )

        # Load UE8M0 scale: scale = 2^(stored_value - 127)
        encoded_scale = tl.load(token_scale_ptr + qblock_idx)
        exponent = encoded_scale.to(tl.float32) - 127.0
        # Overflow guard: fp8 e4m3 max is 448 (~2^9.8); float32 max is ~2^127.8,
        # so a scale exponent above ~118 overflows float32 on dequant
        # (448 * 2^128 > float32 max -> Inf -> NaN in softmax). Clamp the
        # exponent to a safe ceiling; this is a pure safety clamp (normal
        # RMSNorm'd KV values never approach it) that prevents stale/garbage
        # scale bytes (e.g. 254-255) from generating Inf/NaN at block
        # boundaries where newly-allocated blocks may hold stale data.
        scale = tl.exp2(tl.minimum(exponent, 118.0))

        # Dequantize and store as bf16
        x_dequant = x_float * scale
        tl.store(out_row_ptr + offsets, x_dequant.to(tl.bfloat16), mask=mask)

    # ========== Copy BF16 portion (64 elements) directly ==========
    bf16_src_ptr = (token_data_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))
    bf16_out_ptr = (out_row_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))

    for j in tl.static_range(bf16_dim // 16):
        chunk_offsets = j * 16 + tl.arange(0, 16)
        bf16_vals = tl.load(bf16_src_ptr + chunk_offsets)
        tl.store(bf16_out_ptr + chunk_offsets, bf16_vals)


def apply_attn_sink(
    out: torch.Tensor,  # [T, H, Dv]
    lse: torch.Tensor,  # [T, H] natural-log logsumexp from the BF16 kernel
    attn_sink: torch.Tensor,  # [H] (or padded) learned per-head sink logit
) -> torch.Tensor:
    """Fold the learned per-head attention sink into a sink-less kernel output.

    ``triton_bf16_mla_sparse_interface`` has no sink slot; the DeepSeek-V4 sink
    is a per-head logit that contributes ``exp(s_h - LSE)`` to the softmax
    denominator with no value vector. Correct post-hoc:
    ``out' = out / (1 + exp(s_h - LSE))``, broadcast per (token, head) over the
    value dim. ``attn_sink`` shares the scaled-logit units of ``LSE``; padded or
    ``-inf`` sinks are inert (``exp -> 0`` leaves the row unchanged).
    """
    import os

    if os.environ.get("VLLM_SM86_SINK", "1") != "1":
        return out
    sink = attn_sink[: out.shape[1]].to(lse.dtype)
    delta = sink.unsqueeze(0) - lse
    # -inf sink and/or -inf LSE (empty row) can yield inf-inf = NaN; treat any
    # non-finite delta as -inf so exp(delta)=0 leaves the row unchanged.
    delta = torch.where(torch.isfinite(delta), delta, torch.full_like(delta, float("-inf")))
    denom = 1.0 + torch.exp(delta)
    return out / denom.unsqueeze(-1)


def dequant_gather_slots(
    out: torch.Tensor,  # [total_slots, 512] bf16, pre-allocated
    cache: torch.Tensor,  # [num_blocks, block_size, head_bytes] uint8
    indices: torch.Tensor,  # [total_slots] int32, global slot IDs
    cache_block_size: int,  # block_size for this cache
) -> None:
    """Dequantize FP8 UE8M0 pages at scattered slot indices into bf16."""
    total_slots = indices.shape[0]
    if total_slots == 0:
        return

    block_stride = cache.stride(0)

    _dequant_gather_slots_kernel[(total_slots,)](
        out,
        cache,
        indices,
        cache_block_size=cache_block_size,
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=block_stride,
        fp8_dim=TOKEN_FP8_DIM,
        bf16_dim=TOKEN_BF16_DIM,
        scale_dim=TOKEN_SCALE_DIM,
        quant_block=QUANT_BLOCK_SIZE,
        output_dim=OUTPUT_DIM,
        n_quant_blocks=7,
    )


def ampere_sparse_decode_fp8(
    q: torch.Tensor,  # [num_tokens, num_heads, head_dim]
    kv_cache: torch.Tensor | None,  # [num_blocks, block_size, head_bytes] uint8
    swa_kv_cache: torch.Tensor,  # [num_blocks, swa_block_size, head_bytes] uint8
    swa_only: bool,
    topk_indices: torch.Tensor | None,  # [num_tokens, 1, topk] global slot IDs
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,  # [num_tokens, 1, swa_k] global slot IDs
    swa_lens: torch.Tensor,
    softmax_scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ampere decode: dequant FP8 pages to BF16, then BF16 sparse MLA attention.

    Keeps external FP8 KV cache layout identical to CUDA/ROCm.
    Returns ``(out, lse)`` -- the raw sink-less attention output plus the
    exact natural-log LSE; the caller applies the attention sink and/or the
    DCP LSE merge.
    """
    num_tokens = q.shape[0]
    device = q.device

    # Determine max topk and swa widths
    if not swa_only and topk_indices is not None:
        topk_idx_2d = (
            topk_indices.squeeze(1) if topk_indices.dim() == 3 else topk_indices
        )
        max_topk = topk_idx_2d.shape[1]
    else:
        topk_idx_2d = None
        max_topk = 0

    swa_idx_2d = swa_indices.squeeze(1) if swa_indices.dim() == 3 else swa_indices
    max_swa = swa_idx_2d.shape[1]

    K_total = max_topk + max_swa

    # Allocate flat workspace: [num_tokens * K_total, 512] bf16
    workspace = torch.empty(
        (num_tokens * K_total, OUTPUT_DIM), dtype=torch.bfloat16, device=device
    )
    ws_3d = workspace.view(num_tokens, K_total, OUTPUT_DIM)

    # Dequant+gather topk slots from compressed cache
    if not swa_only and topk_idx_2d is not None and kv_cache is not None:
        topk_flat = topk_idx_2d.reshape(-1).to(torch.int32)
        topk_buf = torch.empty(
            (num_tokens * max_topk, OUTPUT_DIM), dtype=torch.bfloat16, device=device
        )
        compressed_block_size = kv_cache.shape[1]
        dequant_gather_slots(topk_buf, kv_cache, topk_flat, compressed_block_size)
        ws_3d[:, :max_topk, :] = topk_buf.view(num_tokens, max_topk, OUTPUT_DIM)
        _decode_probe("topk_dequant", topk_buf)
        # If topk_dequant has NaN, identify WHICH gathered slot produced it
        # and whether that slot index is out of the valid written range. A slot
        # index beyond the last written slot reads UNWRITTEN buffer memory (which
        # in the strided cross-layer cache is not guaranteed zeroed -> may hold
        # NaN bit patterns). This is the prime suspect for the token-~2053 NaN.
        # NOTE: _nan_slot_diagnostic self-gates on VLLM_SM86_NAN_PROBE (a later
        # `import os` in this function makes `os` local, so we must NOT reference
        # os.environ here before that import).
        if topk_idx_2d is not None:
            _nan_slot_diagnostic(
                topk_buf.view(num_tokens, max_topk, OUTPUT_DIM),
                topk_idx_2d, topk_lens, kv_cache)
            _decode_probe("topk_indices", topk_idx_2d)
            _decode_probe("topk_lens", topk_lens)

    # Dequant+gather SWA slots
    swa_flat = swa_idx_2d.reshape(-1).to(torch.int32)
    swa_buf = torch.empty(
        (num_tokens * max_swa, OUTPUT_DIM), dtype=torch.bfloat16, device=device
    )
    swa_block_size = swa_kv_cache.shape[1]
    dequant_gather_slots(swa_buf, swa_kv_cache, swa_flat, swa_block_size)

    ws_3d[:, max_topk:, :] = swa_buf.view(num_tokens, max_swa, OUTPUT_DIM)
    _decode_probe("swa_dequant", swa_buf)

    # Build combined indices into the flat workspace, FIXED-WIDTH layout:
    # token t's row = [topk slots 0..max_topk) | swa slots 0..max_swa) | pad],
    # invalid entries = -1. The BF16 kernel masks (indices >= 0) per entry and
    # iterates the full index width, so holes are a no-op; nothing requires
    # contiguous packing. This keeps the build free of .item()/host syncs and
    # Python loops, making the decode path cudagraph-capturable.
    _BLOCK_N = 16
    K_padded = ((K_total + _BLOCK_N - 1) // _BLOCK_N) * _BLOCK_N
    combined_indices = torch.full(
        (num_tokens, K_padded),
        fill_value=-1,
        dtype=torch.int32,
        device=device,
    )

    token_offsets = (
        torch.arange(num_tokens, device=device, dtype=torch.int32) * K_total
    )  # [B]

    if not swa_only and topk_lens is not None:
        topk_range = torch.arange(
            max_topk, device=device, dtype=torch.int32
        ).unsqueeze(0)
        topk_valid = topk_range < topk_lens.unsqueeze(1)
        combined_indices[:, :max_topk] = torch.where(
            topk_valid, token_offsets.unsqueeze(1) + topk_range, -1
        )
    swa_range = torch.arange(max_swa, device=device, dtype=torch.int32).unsqueeze(0)
    swa_valid = swa_range < swa_lens.unsqueeze(1)
    combined_indices[:, max_topk:K_total] = torch.where(
        swa_valid, token_offsets.unsqueeze(1) + max_topk + swa_range, -1
    )

    # Call BF16 sparse MLA kernel. The base kernel launches one program per
    # (token, head-tile) -- a single program at batch-1 decode with 8 local
    # heads. Split-K partitions the index columns across ~n_tiles programs
    # (exact LSE merge, parity-tested), filling the SMs. VLLM_SM86_SPLITK=0
    # restores the base kernel.
    import os

    if os.environ.get("VLLM_SM86_SPLITK", "1") == "1":
        from vllm.models.deepseek_v4.ampere.ampere_sparse_splitk import (
            triton_bf16_mla_sparse_splitk,
        )

        out_attn, _, lse = triton_bf16_mla_sparse_splitk(
            q=q,
            kv=workspace.unsqueeze(1),
            indices=combined_indices.unsqueeze(1),
            sm_scale=softmax_scale,
            d_v=q.shape[-1],
        )
    else:
        out_attn, _, lse = triton_bf16_mla_sparse_interface(
            q=q,
            kv=workspace.unsqueeze(1),
            indices=combined_indices.unsqueeze(1),
            sm_scale=softmax_scale,
            d_v=q.shape[-1],
            block_dpe=0,
        )
    _decode_probe("kernel_out_attn", out_attn)
    _decode_probe("kernel_lse", lse)
    return out_attn, lse
