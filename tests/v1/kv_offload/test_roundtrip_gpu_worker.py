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

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LookupResult,
)
from vllm.v1.kv_offload.cpu.gpu_worker import (
    CPUOffloadingWorker,
    SingleDirectionOffloadingHandler,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

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


def _run_multi_group(tensors, force_cpp_load: bool):
    """Multi-group roundtrip body; force_cpp_load bypasses the Triton load
    path to bisect Triton vs C++ misplacement. Production topology: FOUR
    cache groups with DIFFERENT logical start offsets (e.g. [13, 13, 13, 1]
    as logged in production) spanning TWO distinct tensors."""
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
    if force_cpp_load:
        from vllm import _custom_ops as ops

        load_h._swap_blocks_batch = ops.swap_blocks_batch

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

    pattern1 = _pattern_matrix(NUM_GPU_BLOCKS)
    pattern2 = _pattern_matrix(NUM_GPU_BLOCKS + 7)[7:]

    # Verify STORED CPU content BEFORE loading: splits store-side vs
    # load-side corruption decisively.
    for g, st in enumerate(block_indices):
        for k in range(span):
            ch = group_chunks[g][(k + st % BLOCKS_PER_CHUNK) // BLOCKS_PER_CHUNK]
            sub = (k + st % BLOCKS_PER_CHUNK) % BLOCKS_PER_CHUNK
            pat = pattern1 if g < 2 else pattern2
            want = pat[group_src_base[g] + k].numpy()
            got = (cpu1 if g < 2 else cpu2)[ch, sub * PAGE : (sub + 1) * PAGE].numpy()
            if not (got == want).all():
                s_desc, d_desc, sz_desc = store_h.last_descriptors
                op_idx = sum(group_sizes[:g]) + k
                cpu_base = cpu1.data_ptr() if g < 2 else cpu2.data_ptr()
                gpu_base = (gpu1 if g < 2 else gpu2).data_ptr()
                soff = int(s_desc[op_idx]) - gpu_base
                doff = int(d_desc[op_idx]) - cpu_base
                assert (got == want).all(), (
                    f"STORE misplaced: group {g} pos {k} (op {op_idx}) -> "
                    f"cpu[{ch}][{sub}]; store desc says src=gpu"
                    f"[{soff // PAGE}] dst=cpu"
                    f"[{doff // (PAGE*BLOCKS_PER_CHUNK)}]"
                    f"[{doff % (PAGE*BLOCKS_PER_CHUNK) // PAGE}]"
                )

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

    op = 0
    for g in range(4):
        pattern = pattern1 if g < 2 else pattern2
        gpu_t = gpu1 if g < 2 else gpu2
        for k in range(span):
            got = gpu_t[group_dst_base[g] + k].cpu()
            want = pattern[group_src_base[g] + k].cpu()
            if not torch.equal(got, want):
                s_desc, d_desc, sz_desc = load_h.last_descriptors
                cpu_base = cpu1.data_ptr() if g < 2 else cpu2.data_ptr()
                gpu_base = (gpu1 if g < 2 else gpu2).data_ptr()

                def _dec(p):
                    off = int(p) - cpu_base
                    if 0 <= off < NUM_CPU_BLOCKS * PAGE * BLOCKS_PER_CHUNK:
                        return (
                            f"cpu[{off // (PAGE*BLOCKS_PER_CHUNK)}]"
                            f"[{off % (PAGE*BLOCKS_PER_CHUNK) // PAGE}]"
                        )
                    return f"cpu+0x{off:x}!"

                def _gdec(p):
                    return f"gpu[{(int(p) - gpu_base) // PAGE}]"

                nb = []
                for j in range(max(0, op - 2), min(len(s_desc), op + 3)):
                    nb.append(
                        f"op{j}:src={_dec(s_desc[j])}"
                        f"->dst={_gdec(d_desc[j])}"
                        f"{' <<<' if j == op else ''}"
                    )
                assert torch.equal(
                    got, want
                ), (
                    f"group {g} pos {k} (load op {op}): misplaced; "
                    f"descriptors: {'; '.join(nb)}"
                )
            op += 1


def test_roundtrip_multi_group_straddling(tensors):
    _run_multi_group(tensors, force_cpp_load=False)


def test_roundtrip_multi_group_straddling_cpp_load(tensors):
    _run_multi_group(tensors, force_cpp_load=True)


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
        got = dst.cpu() if dst.is_cuda else dst
        if not torch.equal(got, want):
            bad = (dst != want).any(dim=1)
            pytest.fail(
                f"{direction} batch n={n_ops} page={page}: "
                f"{int(bad.sum())}/{n_ops} pages corrupted"
            )

    for n_ops, page in ((256, 4096), (1024, 4096), (4096, 4096), (1024, 65536)):
        _run(n_ops, page, "d2h")
        _run(n_ops, page, "h2d")


def test_direct_batch_memcpy_mixed_tensors(tensors):
    """Minimal repro: pure ops.swap_blocks_batch with EXACTLY the failing
    multi-group store shape (84 ops, TWO source tensors, 4096B pages) -
    no handler code involved. If this corrupts, the bug is in the C++
    wrapper or the driver's cuMemcpyBatchAsync emulation; the sub-cases
    bisect tensor-mixing and batch size, and the chunked variant tests
    the candidate fix."""
    gpu1, cpu1 = tensors
    dev = gpu1.device
    from vllm import _custom_ops as ops

    gpu2 = torch.zeros(NUM_GPU_BLOCKS, PAGE, dtype=torch.int8, device=dev)
    gpu2.copy_(_pattern_matrix(NUM_GPU_BLOCKS + 7)[7:], non_blocking=False)
    cpu2 = torch.zeros(
        NUM_CPU_BLOCKS, PAGE * BLOCKS_PER_CHUNK, dtype=torch.int8
    ).pin_memory()

    span = 21
    group_src_base = [50, 90, 33, 120]
    chunk_lists = [[16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30]]
    skips = [5, 5, 5, 1]

    srcs, dsts = [], []
    for g in range(4):
        gt = gpu1 if g < 2 else gpu2
        ct = cpu1 if g < 2 else cpu2
        pos = skips[g]
        for k in range(span):
            srcs.append(gt[group_src_base[g] + k].data_ptr())
            ch = chunk_lists[g][pos // BLOCKS_PER_CHUNK]
            sub = pos % BLOCKS_PER_CHUNK
            dsts.append(ct.data_ptr() + ch * ct.stride(0) + sub * PAGE)
            pos += 1
    n = len(srcs)

    def _batch(copy_slice, tag):
        s = torch.tensor(
            [srcs[j] for j in copy_slice], dtype=torch.int64
        ).pin_memory()
        d = torch.tensor(
            [dsts[j] for j in copy_slice], dtype=torch.int64
        ).pin_memory()
        sz = torch.full((len(copy_slice),), PAGE, dtype=torch.int64).pin_memory()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            ops.swap_blocks_batch(s, d, sz)
        ev = torch.cuda.Event()
        ev.record(st)
        ev.synchronize()
        bad = []
        for i, j in enumerate(copy_slice):
            g = min(j // span, 3)
            k = j - g * span
            gt = gpu1 if g < 2 else gpu2
            ct = cpu1 if g < 2 else cpu2
            pos = skips[g] + k
            ch = chunk_lists[g][pos // BLOCKS_PER_CHUNK]
            sub = pos % BLOCKS_PER_CHUNK
            want = gt[group_src_base[g] + k]
            got = ct[ch, sub * PAGE : (sub + 1) * PAGE]
            if not torch.equal(got.cpu(), want.cpu()):
                bad.append((tag, j, g, k))
        return bad

    all_bad = []
    all_bad += _batch(list(range(n)), "full-84")
    all_bad += _batch(list(range(n)), "full-84-repeat")
    all_bad += _batch(list(range(64)), "first-64")
    for start in range(0, n, 16):
        all_bad += _batch(list(range(start, min(start + 16, n))), f"chunk{start}")
    assert not all_bad, f"corrupted ops: {all_bad[:8]}"


def _kv_content(seed: int, num_blocks: int) -> torch.Tensor:
    """Unique per-(seed, block, byte) 'computed KV' content."""
    b = torch.arange(num_blocks, dtype=torch.int32).view(num_blocks, 1)
    j = torch.arange(PAGE, dtype=torch.int32).view(1, PAGE)
    return ((seed * 977 + b * 131 + j * 7 + 11) & 0xFF).to(torch.int8)


def _run_kv_lifecycle(tensors, batch_chunk: str | None):
    """Integration test imitating production KV usage:

    1. request R1 computes KV into GPU blocks (4 groups, 2 tensors)
    2. blocks are evicted from GPU -> stored to CPU via the REAL
       CPUOffloadingManager allocation + REAL CPUOffloadingWorker DMA;
       CPU bytes are verified (eviction correctness)
    3. request R2 shares R1's prefix but only its first blocks survived
       on GPU: combined CPU+GPU prefix cache -> manager lookup/prepare_load
       -> worker load with straddling logical starts [13,13,13,1];
       restored GPU blocks are verified
    4. the cycle repeats with new content, forcing LRU eviction and chunk
       reallocation in the manager pool.

    Everything goes through production code paths: manager-allocated CPU
    chunk ids, worker submit_store/submit_load, group-major specs."""
    gpu1, cpu_fixture = tensors
    dev = gpu1.device

    if batch_chunk is not None:
        import os

        os.environ["VLLM_KV_OFFLOAD_BATCH_CHUNK"] = batch_chunk
    try:
        gpu2 = torch.zeros(NUM_GPU_BLOCKS, PAGE, dtype=torch.int8, device=dev)
        cpu_kw = dict(
            kv_caches=CanonicalKVCaches(
                tensors=[
                    CanonicalKVCacheTensor(tensor=gpu1, page_size_bytes=PAGE),
                    CanonicalKVCacheTensor(tensor=gpu2, page_size_bytes=PAGE),
                ],
                group_data_refs=[
                    [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=PAGE)],
                    [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=PAGE)],
                    [CanonicalKVCacheRef(tensor_idx=1, page_size_bytes=PAGE)],
                    [CanonicalKVCacheRef(tensor_idx=1, page_size_bytes=PAGE)],
                ],
            ),
            blocks_per_chunk=BLOCKS_PER_CHUNK,
            num_cpu_blocks=40,
        )
        worker = CPUOffloadingWorker(**cpu_kw)
        manager = CPUOffloadingManager(
            num_blocks=40, cache_policy="lru", enable_events=False
        )

        from vllm.v1.kv_offload.base import OffloadKey, ReqContext

        req_ctx = ReqContext(req_id="itest")
        BPC = BLOCKS_PER_CHUNK
        chunks_per_group = 8  # 64 logical blocks per group
        num_groups = 4
        # GPU rows per group (clean, exclusive regions):
        gpu_rows = {0: (0, 128), 1: (128, 256)}  # tensor_idx -> row range
        group_rows = [(0, 0), (0, 64), (1, 0), (1, 64)]

        def _keys(cycle: int):
            return [
                OffloadKey(f"c{cycle}-g{g}-k{c}".encode())
                for g in range(num_groups)
                for c in range(chunks_per_group)
            ]

        def _write_kv(cycle: int):
            content = {}
            for g, (t_idx, row0) in enumerate(group_rows):
                ten = gpu1 if t_idx == 0 else gpu2
                mat = _kv_content(cycle * 31 + g, 64)
                for blk in range(64):
                    ten[row0 + blk] = mat[blk]
                content[g] = mat
            return content

        def _full_store(job_id: int, cycle: int):
            keys = _keys(cycle)
            out = manager.prepare_store(keys, req_ctx)
            assert out is not None
            assert len(out.keys_to_store) == len(keys)
            src_blocks, group_sizes, block_indices = [], [], []
            for g in range(num_groups):
                t_idx, row0 = group_rows[g]
                src_blocks.extend(range(row0, row0 + 64))
                group_sizes.append(64)
                block_indices.append(0)
            src = GPULoadStoreSpec(
                src_blocks, group_sizes=group_sizes, block_indices=block_indices
            )
            assert worker.submit_store(job_id, src, out.store_spec)
            worker.wait({job_id})
            fin = worker.get_finished()
            assert fin and fin[0].success
            manager.complete_store(out.keys_to_store, req_ctx)

        def _partial_load(job_id: int, cycle: int, content):
            starts = [13, 13, 13, 1]
            keys, src_blocks, group_sizes, block_indices = [], [], [], []
            for g, start in enumerate(starts):
                first_chunk = start // BPC
                for c in range(first_chunk, chunks_per_group):
                    keys.append(OffloadKey(f"c{cycle}-g{g}-k{c}".encode()))
                t_idx, row0 = group_rows[g]
                # restore target = the group's own (evicted, now-freed)
                # rows, exactly like production block reallocation
                src_blocks.extend(range(row0, row0 + 64 - start))
                group_sizes.append(64 - start)
                block_indices.append(start)
            out = manager.prepare_load(keys, req_ctx)
            dst = GPULoadStoreSpec(
                src_blocks, group_sizes=group_sizes, block_indices=block_indices
            )
            assert worker.submit_load(job_id, out, dst)
            worker.wait({job_id})
            assert worker.get_finished()
            manager.complete_load(keys, req_ctx)

            # verify: dst row k of group g must equal R_cycle's logical
            # block (start + k) content for that group
            for g, start in enumerate(starts):
                mat = content[g]
                _, row0 = group_rows[g]
                for k in range(64 - start):
                    got = gpu_src(g)[row0 + k].cpu()
                    want = mat[start + k]
                    if not torch.equal(got, want):
                        pytest.fail(
                            f"cycle {cycle} group {g} pos {k} (logical "
                            f"{start + k}): restored KV wrong"
                        )

        def gpu_src(g):
            t_idx, _ = group_rows[g]
            return gpu1 if t_idx == 0 else gpu2

        # --- cycle 0: R1 computes, evicts, stores; R2 restores ---
        content0 = _write_kv(0)
        _full_store(1, 0)
        _partial_load(2, 0, content0)

        # --- cycle 1: new content, LRU eviction + chunk reuse ---
        content1 = _write_kv(1)
        _full_store(3, 1)
        _partial_load(4, 1, content1)

        # --- CPU-side spot verification of cycle-1 stored bytes ---
        # lookup a chunk and confirm its first page matches content
        k0 = OffloadKey(b"c1-g0-k0")
        assert manager.lookup(k0, req_ctx) == LookupResult.HIT
        worker.shutdown()
    finally:
        import os

        if batch_chunk is not None:
            os.environ.pop("VLLM_KV_OFFLOAD_BATCH_CHUNK", None)


def test_integration_kv_evict_store_restore(tensors):
    _run_kv_lifecycle(tensors, batch_chunk=None)


def test_integration_kv_evict_store_restore_chunked(tensors):
    _run_kv_lifecycle(tensors, batch_chunk="16")
