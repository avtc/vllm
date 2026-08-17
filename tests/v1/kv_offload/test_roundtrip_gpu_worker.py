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
    return ((b * 131 + j * 7 + 11) & 0xFF).to(torch.uint8)


@pytest.fixture(scope="module")
def tensors():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for roundtrip tests")
    dev = torch.cuda.current_device()
    pattern = _pattern_matrix(NUM_GPU_BLOCKS)
    gpu = torch.zeros(NUM_GPU_BLOCKS, PAGE, dtype=torch.uint8, device=dev)
    gpu.copy_(pattern, non_blocking=False)
    cpu = torch.zeros(
        NUM_CPU_BLOCKS, PAGE * BLOCKS_PER_CHUNK, dtype=torch.uint8
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
    assert store_h.submit_store(job_id, src, dst)
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

    start = 100  # chunk-aligned destination (100 % 8 == 0)
    ldst = GPULoadStoreSpec(
        list(range(start, start + num_blocks)),
        group_sizes=[num_blocks],
        block_indices=[start],
    )
    assert load_h.submit_load(2, CPULoadStoreSpec([0, 1, 2, 3]), ldst)
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
    assert load_h.submit_load(2, lsrc, ldst)
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
    assert load_h.submit_load(2, CPULoadStoreSpec([7, 8]), ldst)
    load_h.wait({2})
    load_h.get_finished()

    # the corrupted first page of chunk 7 must have been dropped from the
    # trusted map (detected), while chunk 8's pages remain trusted.
    assert row_addr not in load_h._store_crcs, "corruption NOT detected"
    chunk8_addr = cpu.data_ptr() + 8 * cpu.stride(0)
    assert chunk8_addr in load_h._store_crcs, "clean range wrongly evicted"
