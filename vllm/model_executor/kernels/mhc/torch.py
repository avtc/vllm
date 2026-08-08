# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.kernels.mhc.sinkhorn import (
    _fuse_sinkhorn_enabled,
    _fused_comb_mix,
    _fuse_layer_input_enabled,
    _fuse_mhc_post_enabled,
    _fuse_mhc_norm_enabled,
    fused_layer_input,
    fused_mhc_post,
    fused_mhc_norm_sigmoid,
)


def mhc_pre_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """

    # Validate shapes
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    x = residual_flat.view(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn_flat.t())
    # Fused sqrsum + RMSNorm-scale mixes + pre/post sigmoid (~10 kernels -> 1).
    # Outputs the SCALED mixes (comb slice feeds the Sinkhorn kernel). Set
    # VLLM_DSV4_FUSE_MHC_NORM=0 to restore the torch path.
    if _fuse_mhc_norm_enabled():
        pre_mix, post_mix, mixes = fused_mhc_norm_sigmoid(
            x, mixes, hc_scale, hc_base, rms_eps, hc_pre_eps,
            hc_post_mult_value, hc_mult, hidden_size,
        )
    else:
        sqrsum = x.square().sum(dim=-1, keepdim=True)
        mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

        pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
        pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

        post_logits = (
            mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult]
        )
        post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    # Fused softmax + Sinkhorn: the reference runs a 19-iter Python loop of tiny
    # sum/div kernels (~114 launches) on a 16-element matrix -- a dominant source
    # of decode glue. Fuse into one Triton kernel (16 floats in registers).
    # Set VLLM_DSV4_FUSE_SINKHORN=0 to restore the Python loop (A/B verify).
    if _fuse_sinkhorn_enabled():
        comb_mix_flat = _fused_comb_mix(
            mixes[:, 2 * hc_mult :],          # [N, hc_mult*hc_mult]
            hc_base[2 * hc_mult :],           # [hc_mult*hc_mult]
            hc_scale[2],
            hc_sinkhorn_eps,
            sinkhorn_repeat,
            hc_mult,
        )
        comb_mix = comb_mix_flat.view(num_tokens, hc_mult, hc_mult)
    else:
        comb_logits = (
            mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult) * hc_scale[2]
            + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
        )
        comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
        for _ in range(sinkhorn_repeat - 1):
            comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
            comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    # Fused layer_input: weighted sum of the hc_mult residual rows -> bf16.
    # Reference: torch.sum(pre_mix.unsqueeze(-1) * residual.float, dim=1).to(bf16)
    # (~5 kernels -> 1). VLLM_DSV4_FUSE_LAYER_INPUT=0 restores the torch path.
    if _fuse_layer_input_enabled():
        layer_input = fused_layer_input(pre_mix, residual_flat, hc_mult)
    else:
        layer_input = torch.sum(
            pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1
        ).to(torch.bfloat16)
    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def mhc_post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    # Fused mhc_post: einsum(...ij,...ih->...jh) + post_term + add -> bf16 in one
    # kernel (~8 kernels -> 1). VLLM_DSV4_FUSE_MHC_POST=0 restores the torch path.
    if _fuse_mhc_post_enabled() and residual.dim() >= 2:
        hc_mult = residual.shape[-2]
        return fused_mhc_post(
            comb_res_mix.to(torch.float32), residual, post_layer_mix.to(torch.float32),
            x, hc_mult,
        )
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.to(torch.float32),
        residual.to(torch.float32),
    )
    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return (mixed_residual + post_term).to(residual.dtype)
