# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to vLLM project
"""Fused comb_mix (softmax + Sinkhorn) Triton kernel for the mHC head.

The mHC pre-block computes comb_mix = sinkhorn(softmax(comb_logits)) over a tiny
[hc_mult, hc_mult] = [4, 4] = 16-element matrix. The reference (mhc_pre_torch)
runs the Sinkhorn loop in Python: 19 iterations x (row-sum + add + div + col-sum
+ add + div) = ~114 tiny kernel launches per mhc_pre call, ~86 calls/step
(43 layers x 2) = ~9800 tiny kernels/step -- the dominant "glue" in DSv4 decode
(~73% of GPU time is tiny reduce/elementwise kernels). This kernel fuses softmax
+ initial col-norm + the full Sinkhorn loop into ONE launch (16 floats held in
registers, all iterations in-kernel), collapsing ~118 kernels -> 1.

Enabled by default on the mhc_pre_torch path (gated by VLLM_DSV4_FUSE_SINKHORN,
default "1"); set to "0" to restore the Python loop for A/B verification.
"""
import os

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_comb_mix_kernel(
    mixes_comb_ptr,   # [num_tokens, HC_MULT*HC_MULT] fp32 (the comb slice of mixes)
    base_comb_ptr,    # [HC_MULT*HC_MULT] fp32
    out_ptr,          # [num_tokens, HC_MULT*HC_MULT] fp32
    hc_scale_comb,    # scalar fp32 (comb scale = hc_scale[2])
    hc_eps,           # sinkhorn eps (added after softmax)
    HC_MULT: tl.constexpr,
    SINKHORN_REPEAT: tl.constexpr,
):
    """One program per token. Holds comb_mix [HC_MULT, HC_MULT] in registers,
    does softmax + eps + col-norm + (repeat-1) Sinkhorn iterations entirely
    in-kernel. Matches mhc_pre_torch lines 75-82 bit-for-bit (fp32 math)."""
    n = tl.program_id(0)
    # offsets for a [HC_MULT, HC_MULT] block belonging to token n
    row = tl.arange(0, HC_MULT)[:, None]   # [HC_MULT, 1]
    col = tl.arange(0, HC_MULT)[None, :]   # [1, HC_MULT]
    blk = row * HC_MULT + col              # [HC_MULT, HC_MULT]
    base = n * (HC_MULT * HC_MULT)

    # comb_logits = mixes_comb * scale + base_comb
    logits = (
        tl.load(mixes_comb_ptr + base + blk) * hc_scale_comb
        + tl.load(base_comb_ptr + blk)
    )
    # softmax over last dim (col, axis=1): max-subtract, exp, sum, div
    m = tl.max(logits, axis=1)            # [HC_MULT] per-row max
    e = tl.exp(logits - m[:, None])
    s = tl.sum(e, axis=1)                  # [HC_MULT] per-row sum
    comb = e / s[:, None] + hc_eps
    # initial col-norm: divide by sum over rows (axis=0, dim=-2)
    comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)
    # Sinkhorn loop (repeat-1 iterations): row-norm (axis=1, dim=-1) then
    # col-norm (axis=0, dim=-2). Done entirely in registers; static_range so
    # the loop unrolls at JIT time (SINKHORN_REPEAT is a constexpr).
    for _ in tl.static_range(SINKHORN_REPEAT - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + hc_eps)   # row-norm
        comb = comb / (tl.sum(comb, axis=0)[None, :] + hc_eps)   # col-norm
    tl.store(out_ptr + base + blk, comb)


def _fused_comb_mix(
    mixes_comb: torch.Tensor,   # [num_tokens, hc_mult*hc_mult] fp32
    base_comb: torch.Tensor,    # [hc_mult*hc_mult] fp32
    hc_scale_comb: float,
    hc_eps: float,
    sinkhorn_repeat: int,
    hc_mult: int,
) -> torch.Tensor:
    """Launcher for _fused_comb_mix_kernel. Returns comb_mix [N, hc_mult*hc_mult]."""
    num_tokens = mixes_comb.shape[0]
    out = torch.empty(
        (num_tokens, hc_mult * hc_mult), dtype=torch.float32, device=mixes_comb.device
    )
    _fused_comb_mix_kernel[(num_tokens,)](
        mixes_comb,
        base_comb,
        out,
        hc_scale_comb,
        hc_eps,
        HC_MULT=hc_mult,
        SINKHORN_REPEAT=sinkhorn_repeat,
    )
    return out


def _fuse_sinkhorn_enabled() -> bool:
    return os.environ.get("VLLM_DSV4_FUSE_SINKHORN", "1") == "1"


def _fuse_layer_input_enabled() -> bool:
    return os.environ.get("VLLM_DSV4_FUSE_LAYER_INPUT", "1") == "1"


def _fuse_mhc_post_enabled() -> bool:
    return os.environ.get("VLLM_DSV4_FUSE_MHC_POST", "1") == "1"


# ---------------------------------------------------------------------------
# Fused layer_input: out[n,h] = sum_i pre_mix[n,i] * residual[n,i,h] -> bf16.
# Reference: torch.sum(pre_mix.unsqueeze(-1) * residual.float, dim=1).to(bf16)
# ~5 kernels (unsqueeze, cast, mul, sum, cast) -> 1.
# ---------------------------------------------------------------------------
@triton.jit
def _layer_input_kernel(
    pre_mix_ptr,    # [num_tokens, HC_MULT] fp32
    residual_ptr,   # [num_tokens, HC_MULT, HIDDEN] bf16
    out_ptr,        # [num_tokens, HIDDEN] bf16
    stride_pm_n, stride_pm_m,
    stride_r_n, stride_r_m, stride_r_h,
    stride_o_n, stride_o_h,
    HIDDEN: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    n = tl.program_id(0)
    hb = tl.program_id(1)
    h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = h < HIDDEN
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for i in tl.static_range(HC_MULT):
        pm = tl.load(pre_mix_ptr + n * stride_pm_n + i * stride_pm_m)
        r = tl.load(
            residual_ptr + n * stride_r_n + i * stride_r_m + h * stride_r_h,
            mask=hmask, other=0.0,
        ).to(tl.float32)
        acc += pm * r
    tl.store(out_ptr + n * stride_o_n + h * stride_o_h, acc.to(tl.bfloat16), mask=hmask)


def fused_layer_input(
    pre_mix: torch.Tensor,    # [N, hc_mult] fp32
    residual: torch.Tensor,   # [N, hc_mult, hidden] bf16
    hc_mult: int,
) -> torch.Tensor:
    n, _, hidden = residual.shape
    out = torch.empty((n, hidden), dtype=torch.bfloat16, device=residual.device)
    BLOCK_H = 256
    _layer_input_kernel[(n, triton.cdiv(hidden, BLOCK_H))](
        pre_mix, residual, out,
        pre_mix.stride(0), pre_mix.stride(1),
        residual.stride(0), residual.stride(1), residual.stride(2),
        out.stride(0), out.stride(1),
        HIDDEN=hidden, HC_MULT=hc_mult, BLOCK_H=BLOCK_H,
    )
    return out


# ---------------------------------------------------------------------------
# Fused mhc_post: out[n,j,h] = sum_i comb_res_mix[n,i,j]*residual[n,i,h]
#                                + post_layer_mix[n,j]*x[n,h]  -> bf16.
# Reference: einsum(...ij,...ih->...jh) + post_term, ~8 kernels -> 1.
# ---------------------------------------------------------------------------
@triton.jit
def _mhc_post_kernel(
    comb_res_mix_ptr,   # [num_tokens, HC_MULT, HC_MULT] fp32
    residual_ptr,       # [num_tokens, HC_MULT, HIDDEN] bf16
    post_layer_mix_ptr, # [num_tokens, HC_MULT] fp32
    x_ptr,              # [num_tokens, HIDDEN] bf16
    out_ptr,            # [num_tokens, HC_MULT, HIDDEN] bf16
    stride_c_n, stride_c_i, stride_c_j,
    stride_r_n, stride_r_i, stride_r_h,
    stride_p_n, stride_p_j,
    stride_x_n, stride_x_h,
    stride_o_n, stride_o_j, stride_o_h,
    HIDDEN: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    n = tl.program_id(0)
    hb = tl.program_id(1)
    h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = h < HIDDEN
    for j in tl.static_range(HC_MULT):
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for i in tl.static_range(HC_MULT):
            c = tl.load(comb_res_mix_ptr + n * stride_c_n + i * stride_c_i + j * stride_c_j)
            r = tl.load(
                residual_ptr + n * stride_r_n + i * stride_r_i + h * stride_r_h,
                mask=hmask, other=0.0,
            ).to(tl.float32)
            acc += c * r
        plm = tl.load(post_layer_mix_ptr + n * stride_p_n + j * stride_p_j)
        xv = tl.load(x_ptr + n * stride_x_n + h * stride_x_h, mask=hmask, other=0.0).to(tl.float32)
        acc += plm * xv
        tl.store(
            out_ptr + n * stride_o_n + j * stride_o_j + h * stride_o_h,
            acc.to(tl.bfloat16), mask=hmask,
        )


def fused_mhc_post(
    comb_res_mix: torch.Tensor,   # [N, hc_mult, hc_mult] fp32
    residual: torch.Tensor,       # [N, hc_mult, hidden] bf16
    post_layer_mix: torch.Tensor, # [N, hc_mult] fp32
    x: torch.Tensor,              # [N, hidden] bf16
    hc_mult: int,
) -> torch.Tensor:
    n, _, hidden = residual.shape
    out = torch.empty((n, hc_mult, hidden), dtype=torch.bfloat16, device=residual.device)
    BLOCK_H = 256
    _mhc_post_kernel[(n, triton.cdiv(hidden, BLOCK_H))](
        comb_res_mix, residual, post_layer_mix, x, out,
        comb_res_mix.stride(0), comb_res_mix.stride(1), comb_res_mix.stride(2),
        residual.stride(0), residual.stride(1), residual.stride(2),
        post_layer_mix.stride(0), post_layer_mix.stride(1),
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        HIDDEN=hidden, HC_MULT=hc_mult, BLOCK_H=BLOCK_H,
    )
    return out
