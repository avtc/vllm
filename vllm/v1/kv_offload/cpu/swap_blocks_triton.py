# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel + tuned constants for the ``swap_blocks_batch`` fast path."""

from __future__ import annotations

from collections import deque

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import tl, triton

# Constants tuned empirically on H100 (PCIe Gen5):
#   NUM_SMS         - smallest SM slice within 5% of peak bandwidth at the
#                     8-32 KB block sizes that matter in practice
#   THRESHOLD_BYTES - max payload per descriptor where Triton beats DMA; above
#                     this the C++ cuMemcpyBatchAsync path takes the lead
#   MIN_N           - minimum batch size where Triton's per-launch cost is
#                     amortized; below this DMA wins
NUM_SMS = 12
THRESHOLD_BYTES = 28 * 1024
MIN_N = 16

# (completion event, descriptor upload tensors) pairs; pruned once the
# kernel finishes so the uploads never die before their kernel.
_keepalive: deque = deque()


@triton.jit
def _swap_blocks_kernel(
    src_addrs,
    dst_addrs,
    sizes,
    n_jobs,  # type: ignore[name-defined]
    BYTES_PER_CHUNK: tl.constexpr,  # type: ignore[name-defined]
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    WORDS_PER_CHUNK: tl.constexpr = BYTES_PER_CHUNK // 8
    offsets = tl.arange(0, WORDS_PER_CHUNK)
    job = pid
    while job < n_jobs:
        src = tl.load(src_addrs + job).to(tl.pointer_type(tl.int64))
        dst = tl.load(dst_addrs + job).to(tl.pointer_type(tl.int64))
        words = tl.load(sizes + job) // 8
        for start in range(0, words, WORDS_PER_CHUNK):
            idx = start + offsets
            mask = idx < words
            data = tl.load(src + idx, mask=mask, other=0)
            tl.store(dst + idx, data, mask=mask)
        job += num_progs


def swap_blocks_batch(
    src_addrs: torch.Tensor,
    dst_addrs: torch.Tensor,
    sizes: torch.Tensor,
    is_src_access_order_any: bool = False,
    *,
    bytes_per_chunk: int,
) -> None:
    """Triton implementation of ``swap_blocks_batch`` for small CPU->GPU batches."""
    n = src_addrs.numel()
    # Too few descriptors to amortize Triton's launch cost.
    if n < MIN_N:
        ops.swap_blocks_batch(
            src_addrs,
            dst_addrs,
            sizes,
            is_src_access_order_any=is_src_access_order_any,
        )
        return
    # Descriptor upload MUST be synchronous (or explicitly kept alive
    # past kernel completion). The temporaries returned by .to() are freed
    # at function return; with non_blocking=True the kernel can execute
    # against allocator-REUSED device memory holding stale pointer values
    # from earlier transfers - the kernel then copies from wrong
    # addresses (observed: GPU->GPU copies via stale GPU pointers in the
    # src array). The arrays are tiny (8B/op); sync upload is negligible.
    temps = (
        src_addrs.to("cuda", non_blocking=False),
        dst_addrs.to("cuda", non_blocking=False),
        sizes.to("cuda", non_blocking=False),
    )
    _swap_blocks_kernel[(min(NUM_SMS, n),)](
        temps[0],
        temps[1],
        temps[2],
        n,
        BYTES_PER_CHUNK=bytes_per_chunk,
    )
    # Keep the descriptor copies alive until the kernel completes: the
    # async kernel may otherwise outlive the temps, whose memory the
    # caching allocator can hand to another (cross-stream) allocation
    # while the kernel still dereferences it.
    ev = torch.cuda.Event()
    ev.record()
    _keepalive.append((ev, temps))
    while _keepalive and _keepalive[0][0].query():
        _keepalive.popleft()
