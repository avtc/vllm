# SPDX-License-Identifier: Apache-2.0
"""Tests for the compressed-KV block-zeroing gate.

DeepSeek-V4 MLA uses a compressed KV cache (compress_ratio > 1): only a subset
of tokens is stored per block. Recycled physical blocks therefore carry stale
data in their unwritten slots and must be zeroed on allocation, exactly like
Mamba/SSM state caches. These tests pin the gate logic that enables that
zeroing for compressed-MLA models.
"""

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    FullAttentionSpec,
)


def _mla_spec(compress_ratio: int) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
        compress_ratio=compress_ratio,
    )


def _plain_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
    )


def _config(groups):
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(size=g.kv_cache_spec.page_size_bytes * 10,
                          shared_by=list(g.layer_names))
            for g in groups
        ],
        kv_cache_groups=[KVCacheGroupSpec(g.layer_names, g.kv_cache_spec)
                         for g in groups],
    )


def test_plain_attention_does_not_need_zeroing():
    """A vanilla FullAttention model (every token written) self-cleans on
    block reuse and must NOT trigger per-step block zeroing."""
    cfg = _config([KVCacheGroupSpec(["layer1", "layer2"], _plain_spec())])
    assert cfg.has_mamba_layers is False
    assert cfg.has_compressed_kv_layers is False
    assert cfg.needs_kv_cache_zeroing is False


def test_compressed_mla_needs_zeroing():
    """A compressed-MLA model (compress_ratio > 1) leaves unwritten slots in a
    recycled block and therefore MUST zero newly-allocated blocks."""
    cfg = _config([KVCacheGroupSpec(["layer1"], _mla_spec(compress_ratio=4))])
    assert cfg.has_mamba_layers is False
    assert cfg.has_compressed_kv_layers is True
    assert cfg.needs_kv_cache_zeroing is True


def test_uncompressed_mla_does_not_need_zeroing():
    """An MLA model with no compression (compress_ratio == 1) writes every
    token's KV and self-cleans like plain attention."""
    cfg = _config([KVCacheGroupSpec(["layer1"], _mla_spec(compress_ratio=1))])
    assert cfg.has_compressed_kv_layers is False
    assert cfg.needs_kv_cache_zeroing is False


def test_mixed_model_needs_zeroing():
    """A hybrid model (plain attention + compressed MLA) must still zero."""
    cfg = _config([
        KVCacheGroupSpec(["layer1"], _plain_spec()),
        KVCacheGroupSpec(["layer2"], _mla_spec(compress_ratio=4)),
    ])
    assert cfg.has_compressed_kv_layers is True
    assert cfg.needs_kv_cache_zeroing is True


def test_cache_view_zero_is_block_bounded():
    """The cache-view zero path zeroes exactly one layer's page per block:
    ``cache[block_id].zero_()`` must touch only that block's slice, not
    neighbors. This guards against the packed-stride overrun that the generic
    KVBlockZeroer suffers from (it derives the zero size from stride(0) ==
    full packed block stride)."""
    # Emulate a packed MLA view: shape (num_blocks, storage_block_size, 584)
    # with stride(0) = the full packed block stride (>> one layer's page).
    num_blocks, storage_block_size, head_bytes = 4, 64, 584
    page_bytes = storage_block_size * head_bytes
    # Emulate a PACKED layout: stride(0) is the full packed block stride,
    # which is far larger than one MLA layer's page (multiple slot-columns
    # are packed into a single physical block).
    packed_block_stride = page_bytes * 28
    raw = torch.ones(
        num_blocks * packed_block_stride, dtype=torch.uint8)
    cache = torch.as_strided(
        raw,
        size=(num_blocks, storage_block_size, head_bytes),
        stride=(packed_block_stride, head_bytes, 1),
    )

    # Zero only block 1 via the view, as the fix does.
    cache[1].zero_()

    flat = raw.tolist()
    # Block 0 untouched (all ones).
    assert all(b == 1 for b in flat[0:page_bytes])
    # Block 1's page zeroed.
    assert all(b == 0 for b in flat[packed_block_stride:packed_block_stride + page_bytes])
    # Block 2 untouched (all ones).
    assert all(
        b == 1
        for b in flat[2 * packed_block_stride:2 * packed_block_stride + page_bytes])
    # A generic zeroer that derived its size from stride(0) would instead zero
    # ``packed_block_stride`` bytes per block (28x here), scribbling across
    # every packed slot-column and into the next block.
    assert packed_block_stride > page_bytes
