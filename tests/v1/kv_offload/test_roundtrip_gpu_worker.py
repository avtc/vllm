# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end GPU<->CPU transfer roundtrip tests for the CPU offload handler.

These tests are the deterministic bug-catcher for the byte-placement
arithmetic in SingleDirectionOffloadingHandler.transfer_async — including
the straddling-chunk case (a load whose destination starts mid-chunk, the
code path only real requests exercise via CPU KV loads).

Every logical GPU block is tagged with a unique pattern; after a
store->load roundtrip each destination block must hold exactly the
pattern of its logical position. Any offset/skip bug in the descriptor
construction shows up as a pattern mismatch.

Requires CUDA (runs on the serving box; skipped elsewhere).
"""

import pytest
import torch

from vllm.v1.kv_offload.base import CanonicalKVCacheRef, GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler

NUM_GPU_BLOCKS = 256
PAGE = 4096  # bytes per GPU block per tensor
BLOCKS_PER_CHUNK = 8  # e.g. 128-token chunks with 16-token GPU blocks
NUM_CPU_BLOCKS = 64


def _pattern_matrix(num_blocks: int) -> torch.Tensor:
    """Vectorized unique per-block pattern: value depends on (block, byte)."""
    b = torch.arange(num_blocks, dtype=torch.int32).view(num_blocks, 1)
    j = torch.arange(PAGE, dtype=torch.int32).view(1, PAGE)
    return ((b * 131 + j * 7 + 11) & 0xFF).to(torch.int8)


@pytest.fixture(scope="module")
def tensors():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for roundtrip tests")
    dev = torch.cuda.current_device()
    pattern = _pattern_matrix(NUM_GPU_BLOCKS)
    gpu = torch.zeros(NUM_GPU_BLOCKS, PAGE, dtype=torch.int8, device=dev)
    gpu.copy_(pattern, non_blocking=False)
    cpu = torch.zeros(
        NUM_CPU_BLOCKS, PAGE * BLOCKS_PER_CHUNK, dtype=torch.int8
    ).pin_memory()
    yield gpu, cpu


def _make_handler(tensors, gpu_to_cpu: bool) -> SingleDirectionOffloadingHandler:
    gpu, cpu = tensors
    refs = [[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=PAGE)]]
    return SingleDirectionOffloadingHandler(
        gpu_tensors=[gpu],
        cpu_tensors=[cpu],
        blocks_per_chunk=BLOCKS_PER_CHUNK,
        kv_cache_groups_data_refs=refs,
        gpu_to_cpu=gpu_to_cpu,
    )


def _store(store_h, job_id, gpu_blocks, cpu_chunks):
    src = GPULoadStoreSpec(
        gpu_blocks, group_sizes=[len(gpu_blocks)], block_indices=[0]
    )
    dst = CPULoadStoreSpec(cpu_chunks)
    assert store_h.transfer_async(job_id, src, dst)
    store_h.wait({job_id})
    results = store_h.get_finished()
    assert results and results[0].success


def test_roundtrip_chunk_aligned(tensors):
    """Store chunks 0..3, load them back to different GPU blocks at a
    chunk-aligned destination start."""
    store_h = _make_handler(tensors, gpu_to_cpu=True)
    load_h = _make_handler(tensors, gpu_to_cpu=False)
    gpu, _ = tensors

    num_blocks = 4 * BLOCKS_PER_CHUNK
    _store(store_h, 1, list(range(num_blocks)), [0, 1, 2, 3])

    start = 100  # physical destination blocks (arbitrary); logical start is 0
    ldst = GPULoadStoreSpec(
        list(range(start, start + num_blocks)),
        group_sizes=[num_blocks],
        block_indices=[0],
    )
    assert load_h.transfer_async(2, CPULoadStoreSpec([0, 1, 2, 3]), ldst)
    load_h.wait({2})
    assert load_h.get_finished()

    expected = _pattern_matrix(num_blocks)
    got = gpu[start : start + num_blocks].cpu()
    assert torch.equal(got, expected), "chunk-aligned load misplaced bytes"


def test_roundtrip_straddling_start(tensors):
    """Load whose destination begins MID-chunk: the request already holds
    logical blocks 0..skip-1 on GPU; only blocks skip..N load. The source
    CPU chunks must be entered with the correct sub-block skip."""
    store_h = _make_handler(tensors, gpu_to_cpu=True)
    load_h = _make_handler(tensors, gpu_to_cpu=False)
    gpu, _ = tensors

    num_blocks = 6 * BLOCKS_PER_CHUNK  # chunks 0..5
    _store(store_h, 1, list(range(num_blocks)), list(range(6)))

    skip = BLOCKS_PER_CHUNK + 3  # 3 GPU blocks into chunk 1 — NOT aligned
    pending = num_blocks - skip
    ldst = GPULoadStoreSpec(
        list(range(pending)), group_sizes=[pending], block_indices=[skip]
    )
    lsrc = CPULoadStoreSpec([1, 2, 3, 4, 5])  # chunks covering skip..end
    assert load_h.transfer_async(2, lsrc, ldst)
    load_h.wait({2})
    assert load_h.get_finished()

    # destination GPU blocks were allocated fresh (ids 0..pending-1 here);
    # logical block skip+k must appear at destination k.
    expected = _pattern_matrix(num_blocks)[skip : skip + pending]
    got = gpu[:pending].cpu()
    assert torch.equal(got, expected), (
        f"straddling load misplaced bytes (skip={skip})"
    )


def test_integrity_crc_detects_mutation(tensors, monkeypatch):
    """VLLM_KV_OFFLOAD_INTEGRITY=1: store records per-range CRCs; mutating
    a stored CPU byte between store and load must be detected and the
    corrupted range evicted from the trusted-CRC map."""
    monkeypatch.setenv("VLLM_KV_OFFLOAD_INTEGRITY", "1")
    _, cpu = tensors
    shared_map: dict[int, int] = {}
    refs = [[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=PAGE)]]

    def _handler(gpu_to_cpu: bool) -> SingleDirectionOffloadingHandler:
        gpu, _cpu = tensors
        return SingleDirectionOffloadingHandler(
            gpu_tensors=[gpu],
            cpu_tensors=[_cpu],
            blocks_per_chunk=BLOCKS_PER_CHUNK,
            kv_cache_groups_data_refs=refs,
            gpu_to_cpu=gpu_to_cpu,
            integrity_map=shared_map,
        )

    store_h = _handler(True)
    load_h = _handler(False)

    num_blocks = 2 * BLOCKS_PER_CHUNK
    _store(store_h, 1, list(range(num_blocks)), [7, 8])
    assert store_h._store_crcs, "CRCs not recorded at store completion"

    # Corrupt one byte inside stored chunk 7 (first page of the row).
    cpu[7, 123] ^= 0xFF
    # The address descriptors use: row start of chunk 7, page 0.
    row_addr = cpu.data_ptr() + 7 * cpu.stride(0)

    ldst = GPULoadStoreSpec(
        list(range(200, 200 + num_blocks)),
        group_sizes=[num_blocks],
        block_indices=[0],
    )
    assert load_h.transfer_async(2, CPULoadStoreSpec([7, 8]), ldst)
    load_h.wait({2})
    load_h.get_finished()

    # the corrupted first page of chunk 7 must have been dropped from the
    # trusted map (detected), while chunk 8's pages remain trusted.
    assert row_addr not in load_h._store_crcs, "corruption NOT detected"
    chunk8_addr = cpu.data_ptr() + 8 * cpu.stride(0)
    assert chunk8_addr in load_h._store_crcs, "clean range wrongly evicted"


def test_roundtrip_multi_group_straddling(tensors):
    """Production topology: FOUR cache groups with DIFFERENT logical start
    offsets (e.g. [13, 13, 13, 1] as logged in production) spanning TWO
    distinct tensors. Log-8 showed stores whose destination pages all held
    identical bytes (descriptor collapse) and loads reading zero pages -
    both only reachable through the multi-group descriptor loop, which the
    single-group tests cannot exercise."""
    gpu1, cpu1 = tensors  # fixture: unique per-block patterns, pinned cpu
    dev = gpu1.device

    # A second, independently-patterned tensor pair (production has many
    # canonical tensors; groups 2-3 here live in tensor 1).
    gpu2 = torch.zeros(NUM_GPU_BLOCKS, PAGE, dtype=torch.int8, device=dev)
    gpu2.copy_(_pattern_matrix(NUM_GPU_BLOCKS + 7)[7:], non_blocking=False)
    cpu2 = torch.zeros(
        NUM_CPU_BLOCKS, PAGE * BLOCKS_PER_CHUNK, dtype=torch.int8
    ).pin_memory()

    refs = [
        [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=PAGE)],
        [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=PAGE)],
        [CanonicalKVCacheRef(tensor_idx=1, page_size_bytes=PAGE)],
        [CanonicalKVCacheRef(tensor_idx=1, page_size_bytes=PAGE)],
    ]

    def _handler(gpu_to_cpu: bool) -> SingleDirectionOffloadingHandler:
        return SingleDirectionOffloadingHandler(
            gpu_tensors=[gpu1, gpu2],
            cpu_tensors=[cpu1, cpu2],
            blocks_per_chunk=BLOCKS_PER_CHUNK,
            kv_cache_groups_data_refs=refs,
            gpu_to_cpu=gpu_to_cpu,
        )

    store_h = _handler(True)
    load_h = _handler(False)

    block_indices = [13, 13, 13, 1]  # logical start per group (prod-like)
    span = 2 * BLOCKS_PER_CHUNK + 5  # blocks transferred per group
    group_sizes = [span] * 4

    # CPU chunks consumed per group = cdiv(span + dst_skip, bpc)
    def _chunks_needed(logical_start: int) -> int:
        skip = logical_start % BLOCKS_PER_CHUNK
        return (span + skip + BLOCKS_PER_CHUNK - 1) // BLOCKS_PER_CHUNK

    group_chunks: list[list[int]] = []
    next_chunk = 16  # arbitrary free region of the cpu tensors
    for g, start in enumerate(block_indices):
        n = _chunks_needed(start)
        group_chunks.append(list(range(next_chunk, next_chunk + n)))
        next_chunk += n

    # STORE: physical source blocks per group; logical start offsets via
    # block_indices. Group 0-1 read tensor0 rows, group 2-3 tensor1 rows.
    src_gpu_blocks: list[int] = []
    group_src_base = [50, 90, 33, 120]  # avoid rows 0..45 (earlier tests)
    for g in range(4):
        src_gpu_blocks.extend(range(group_src_base[g], group_src_base[g] + span))

    src = GPULoadStoreSpec(
        src_gpu_blocks, group_sizes=group_sizes, block_indices=block_indices
    )
    dst = CPULoadStoreSpec([c for grp in group_chunks for c in grp])
    assert store_h.transfer_async(1, src, dst)
    store_h.wait({1})
    res = store_h.get_finished()
    assert res and res[0].success

    # LOAD: same chunks, same logical offsets, fresh physical dst blocks.
    dst_gpu_blocks: list[int] = []
    group_dst_base = [160, 217, 45, 170]  # avoid rows 100..132, 200..216
    for g in range(4):
        dst_gpu_blocks.extend(range(group_dst_base[g], group_dst_base[g] + span))

    ldst = GPULoadStoreSpec(
        dst_gpu_blocks, group_sizes=group_sizes, block_indices=block_indices
    )
    lsrc = CPULoadStoreSpec([c for grp in group_chunks for c in grp])
    assert load_h.transfer_async(2, lsrc, ldst)
    load_h.wait({2})
    assert load_h.get_finished()

    pattern1 = _pattern_matrix(NUM_GPU_BLOCKS)
    pattern2 = _pattern_matrix(NUM_GPU_BLOCKS + 7)[7:]
    pos = 0
    for g in range(4):
        pattern = pattern1 if g < 2 else pattern2
        gpu_t = gpu1 if g < 2 else gpu2
        for k in range(span):
            got = gpu_t[group_dst_base[g] + k].cpu()
            want = pattern[group_src_base[g] + k].cpu()
            assert torch.equal(got, want), (
                f"group {g} logical pos {k}: bytes misplaced or zeroed"
            )
            pos += 1
    assert pos == sum(group_sizes)

def test_large_batch_swap_exact(tensors):
    """Production stores submit THOUSANDS of copy ops in one
    ops.swap_blocks_batch call (on-evict stores of many chunks x 4 groups x
    multiple layer refs). log-8 showed replicated + never-written pages -
    the signature of a driver-side large-batch indexing bug in the
    cuMemcpyBatchAsync emulation path (CUDA 12.8 driver on Ampere, where
    batch memcpy is driver-emulated, not hardware). Sweep batch sizes and
    both production directions (D2H store, H2D load) for exactness."""
    gpu, _ = tensors
    dev = gpu.device

    def _run(n_ops, page, direction):
        src_gpu = torch.zeros(n_ops, page, dtype=torch.int8, device=dev)
        ids = torch.arange(n_ops, dtype=torch.int32).view(n_ops, 1)
        offs = torch.arange(page, dtype=torch.int32).view(1, page)
        src_gpu.copy_(((ids * 131 + offs * 7 + 11) & 0xFF).to(torch.int8))

        if direction == "d2h":
            dst = torch.zeros(n_ops, page, dtype=torch.int8).pin_memory()
        else:
            dst = torch.zeros(n_ops, page, dtype=torch.int8, device=dev)

        perm = torch.randperm(n_ops)
        src_ptrs = torch.tensor(
            [src_gpu[i].data_ptr() for i in range(n_ops)],
            dtype=torch.int64,
        ).pin_memory()
        dst_ptrs = torch.tensor(
            [dst[int(p)].data_ptr() for p in perm], dtype=torch.int64
        ).pin_memory()
        sizes = torch.full((n_ops,), page, dtype=torch.int64).pin_memory()

        from vllm import _custom_ops as ops

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            ops.swap_blocks_batch(src_ptrs, dst_ptrs, sizes)
        ev = torch.cuda.Event()
        ev.record(s)
        ev.synchronize()

        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(n_ops)
        want = src_gpu[inv].cpu()
        if not torch.equal(dst, want):
            bad = (dst != want).any(dim=1)
            pytest.fail(
                f"{direction} batch n={n_ops} page={page}: "
                f"{int(bad.sum())}/{n_ops} pages corrupted"
            )

    for n_ops, page in ((256, 4096), (1024, 4096), (4096, 4096), (1024, 65536)):
        _run(n_ops, page, "d2h")
        _run(n_ops, page, "h2d")

