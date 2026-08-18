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
    # Snapshot the ACTUAL current bytes of every source row: the fixture
    # tensor is module-scoped and earlier tests write into it, so
    # reconstructing expected content from the pattern formula is wrong.
    # Correctness = CPU holds exactly what the source rows held.
    src_content = {}
    for g in range(4):
        gpu_t = gpu1 if g < 2 else gpu2
        src_content[g] = [
            gpu_t[b].clone() for b in range(group_src_base[g], group_src_base[g] + span)
        ]
    assert store_h.transfer_async(1, src, dst)
    store_h.wait({1})
    res = store_h.get_finished()
    assert res and res[0].success

    # Verify STORED CPU content BEFORE loading: splits store-side vs
    # load-side corruption decisively.
    for g, st in enumerate(block_indices):
        for k in range(span):
            ch = group_chunks[g][(k + st % BLOCKS_PER_CHUNK) // BLOCKS_PER_CHUNK]
            sub = (k + st % BLOCKS_PER_CHUNK) % BLOCKS_PER_CHUNK
            want = src_content[g][k].cpu().numpy()
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
        gpu_t = gpu1 if g < 2 else gpu2
        for k in range(span):
            got = gpu_t[group_dst_base[g] + k].cpu()
            want = src_content[g][k].cpu()
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


def test_roundtrip_production_page_size(tensors):
    """Production dimension: 917504-byte pages (896 KiB) - the exact op
    size the serving box uses - through the C++ batch path both
    directions, mixed with a second tensor of a DIFFERENT page size in
    the same batch (production canonical tensors have diverse page
    sizes)."""
    gpu1, _ = tensors
    dev = gpu1.device
    from vllm import _custom_ops as ops

    PAGE_BIG = 917504
    PAGE_ALT = 524288
    bpc = 8
    n_big, n_alt = 21, 13

    g1 = torch.zeros(64, PAGE_BIG, dtype=torch.int8, device=dev)
    g2 = torch.zeros(64, PAGE_ALT, dtype=torch.int8, device=dev)
    for t, page, seed in ((g1, PAGE_BIG, 1), (g2, PAGE_ALT, 2)):
        ids = torch.arange(64, dtype=torch.int32).view(64, 1)
        offs = torch.arange(page, dtype=torch.int32).view(1, page)
        t.copy_(((seed * 977 + ids * 131 + offs * 7 + 11) & 0xFF).to(torch.int8))

    c1 = torch.zeros(16, PAGE_BIG * bpc, dtype=torch.int8).pin_memory()
    c2 = torch.zeros(16, PAGE_ALT * bpc, dtype=torch.int8).pin_memory()

    srcs = [g1[i].data_ptr() for i in range(n_big)] + [
        g2[i].data_ptr() for i in range(n_alt)
    ]
    dsts = (
        [c1.data_ptr() + 0 * c1.stride(0) + j * PAGE_BIG for j in range(n_big)]
        + [c2.data_ptr() + 0 * c2.stride(0) + j * PAGE_ALT for j in range(n_alt)]
    )
    sizes = [PAGE_BIG] * n_big + [PAGE_ALT] * n_alt

    s = torch.tensor(srcs, dtype=torch.int64).pin_memory()
    d = torch.tensor(dsts, dtype=torch.int64).pin_memory()
    sz = torch.tensor(sizes, dtype=torch.int64).pin_memory()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        ops.swap_blocks_batch(s, d, sz)
    ev = torch.cuda.Event()
    ev.record(st)
    ev.synchronize()
    # dsts pack contiguously from each tensor's base (spanning rows)
    f1 = c1.reshape(-1)
    f2 = c2.reshape(-1)
    for j in range(n_big):
        assert torch.equal(
            f1[j * PAGE_BIG : (j + 1) * PAGE_BIG].cpu(), g1[j].cpu()
        ), f"big-page op {j} misplaced"
    for j in range(n_alt):
        assert torch.equal(
            f2[j * PAGE_ALT : (j + 1) * PAGE_ALT].cpu(), g2[j].cpu()
        ), f"alt-page op {j} misplaced"


def test_roundtrip_mmap_rank_interleaved_layout(tensors):
    """Production dimension: the SharedOffloadRegion mmap layout - CPU
    'tensors' are STRIDED VIEWS into one shared file, rows interleaved
    per rank (row = [rank0 cell | rank1 cell], cell holds all that
    rank's tensors concatenated). Verifies two independent rank views
    store/load correctly AND that rank 0's stores never leak into rank
    1's cells. Linux-only (/dev/shm)."""
    import sys

    if not sys.platform.startswith("linux"):
        pytest.skip("SharedOffloadRegion requires Linux /dev/shm")
    from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

    gpu1, _ = tensors
    dev = gpu1.device

    W = PAGE * BLOCKS_PER_CHUNK  # per-tensor width
    cpu_page_size = 2 * W  # two tensors per rank
    row_stride = cpu_page_size * 2  # two ranks
    num_blocks = 40
    engine_id = "itest-mmap"

    import os

    path = f"/dev/shm/vllm_offload_{engine_id}.mmap"
    if os.path.exists(path):
        os.remove(path)

    try:
        regions = [
            SharedOffloadRegion(
                engine_id=engine_id,
                num_blocks=num_blocks,
                rank=r,
                kv_bytes_per_block=row_stride,
                cpu_page_size=cpu_page_size,
            )
            for r in range(2)
        ]
        gpu2 = torch.zeros(NUM_GPU_BLOCKS, PAGE, dtype=torch.int8, device=dev)
        refs = [
            [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=PAGE)],
            [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=PAGE)],
            [CanonicalKVCacheRef(tensor_idx=1, page_size_bytes=PAGE)],
            [CanonicalKVCacheRef(tensor_idx=1, page_size_bytes=PAGE)],
        ]
        # Both ranks' handlers use the SAME canonical GPU tensor order;
        # each rank stores into ITS OWN strided views of the shared mmap.
        cpu_r0 = [regions[0].create_next_view(W) for _ in range(2)]
        cpu_r1 = [regions[1].create_next_view(W) for _ in range(2)]
        store_r0 = SingleDirectionOffloadingHandler(
            gpu_tensors=[gpu1, gpu2],
            cpu_tensors=cpu_r0,
            blocks_per_chunk=BLOCKS_PER_CHUNK,
            kv_cache_groups_data_refs=refs,
            gpu_to_cpu=True,
        )
        store_r1 = SingleDirectionOffloadingHandler(
            gpu_tensors=[gpu1, gpu2],
            cpu_tensors=cpu_r1,
            blocks_per_chunk=BLOCKS_PER_CHUNK,
            kv_cache_groups_data_refs=refs,
            gpu_to_cpu=True,
        )

        span = 21
        starts = [13, 13, 13, 1]
        src_base = [50, 90, 33, 120]
        # rank 1 uses different source rows so contents are distinguishable
        src_base_r1 = [150, 190, 70, 10]

        def _store_verify(store_h, cpu_views, src_base, tag):
            chunk_lists = [
                list(range(4, 8)),
                list(range(8, 12)),
                list(range(12, 16)),
                list(range(16, 19)),
            ]
            srcs, sizes, indices = [], [], []
            content = {}
            for g in range(4):
                gpu_t = gpu1 if g < 2 else gpu2
                row0 = src_base[g]
                content[g] = [gpu_t[row0 + k].clone() for k in range(span)]
                srcs.extend(range(row0, row0 + span))
                sizes.append(span)
                indices.append(starts[g])
            spec = GPULoadStoreSpec(
                list(srcs), group_sizes=sizes, block_indices=indices
            )
            dst = CPULoadStoreSpec([c for cl in chunk_lists for c in cl])
            assert store_h.transfer_async(1, spec, dst)
            store_h.wait({1})
            assert store_h.get_finished()
            for g in range(4):
                for k in range(span):
                    pos = starts[g] % BLOCKS_PER_CHUNK + k
                    ch = chunk_lists[g][pos // BLOCKS_PER_CHUNK]
                    sub = pos % BLOCKS_PER_CHUNK
                    want = content[g][k].cpu().numpy()
                    got = cpu_views[g // 2][
                        ch, sub * PAGE : (sub + 1) * PAGE
                    ].numpy()
                    assert (got == want).all(), (
                        f"{tag} group {g} pos {k}: mmap store misplaced"
                    )

        _store_verify(store_r0, cpu_r0, src_base, "rank0")
        _store_verify(store_r1, cpu_r1, src_base_r1, "rank1")

        # Cross-rank leak check: the FIRST WRITTEN page of chunk 4 is
        # sub 5 (straddling starts). Rank 0 and rank 1 stored DIFFERENT
        # source rows there; identical bytes would mean the views alias.
        v0 = cpu_r0[0][4, 5 * PAGE : 6 * PAGE].numpy()
        v1 = cpu_r1[0][4, 5 * PAGE : 6 * PAGE].numpy()
        assert not (v0 == v1).all(), "rank views alias the same memory!"
        # and each must match its own group-0 first stored block
        assert (v0 == gpu1[src_base[0]].cpu().numpy()).all(), (
            "rank0 first page mismatch"
        )
        assert (v1 == gpu1[src_base_r1[0]].cpu().numpy()).all(), (
            "rank1 first page mismatch"
        )
    finally:
        for r in regions:
            r.cleanup()
        if os.path.exists(path):
            os.remove(path)


def test_concurrent_store_load_streams(tensors):
    """Production runs store and load handlers on INDEPENDENT streams that
    can execute SIMULTANEOUSLY (on-evict store + prefix load in the same
    step). Both batch DMAs hit the same pinned CPU region concurrently.
    Sequential submission with overlapped execution, disjoint cells -
    verify both transfers stay byte-exact."""
    gpu1, _ = tensors
    dev = gpu1.device

    cpu1 = torch.zeros(32, PAGE * BLOCKS_PER_CHUNK, dtype=torch.int8).pin_memory()
    cpu2 = torch.zeros(32, PAGE * BLOCKS_PER_CHUNK, dtype=torch.int8).pin_memory()
    gpu2 = torch.zeros(NUM_GPU_BLOCKS, PAGE, dtype=torch.int8, device=dev)
    gpu2.copy_(_pattern_matrix(NUM_GPU_BLOCKS + 7)[7:])

    refs = [
        [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=PAGE)],
        [CanonicalKVCacheRef(tensor_idx=1, page_size_bytes=PAGE)],
    ]
    store_h = SingleDirectionOffloadingHandler(
        gpu_tensors=[gpu1, gpu2],
        cpu_tensors=[cpu1, cpu2],
        blocks_per_chunk=BLOCKS_PER_CHUNK,
        kv_cache_groups_data_refs=refs,
        gpu_to_cpu=True,
    )
    load_h = SingleDirectionOffloadingHandler(
        gpu_tensors=[gpu1, gpu2],
        cpu_tensors=[cpu1, cpu2],
        blocks_per_chunk=BLOCKS_PER_CHUNK,
        kv_cache_groups_data_refs=refs,
        gpu_to_cpu=False,
    )

    # phase 1: store blocks into chunks 0..3 of both tensors
    span1 = 24
    sspec = GPULoadStoreSpec(
        list(range(10, 10 + span1)) + list(range(30, 30 + span1)),
        group_sizes=[span1, span1],
        block_indices=[0, 0],
    )
    dspec = CPULoadStoreSpec([0, 1, 2, 3, 4, 5])  # 3 chunks per group
    assert store_h.transfer_async(1, sspec, dspec)
    store_h.wait({1})
    assert store_h.get_finished()

    # phase 2: CONCURRENT store (new chunks) + load (old chunks), no wait
    # between submissions so the two streams overlap in execution
    span2 = 16
    store_spec = GPULoadStoreSpec(
        list(range(60, 60 + span2)) + list(range(100, 100 + span2)),
        group_sizes=[span2, span2],
        block_indices=[0, 0],
    )
    store_dst = CPULoadStoreSpec([8, 9, 10, 11])  # 2 chunks per group
    load_src = CPULoadStoreSpec([0, 1, 2, 3, 4, 5])  # 3 chunks per group
    load_dst = GPULoadStoreSpec(
        list(range(150, 150 + span1)) + list(range(170, 170 + span1)),
        group_sizes=[span1, span1],
        block_indices=[0, 0],
    )
    assert store_h.transfer_async(2, store_spec, store_dst)
    assert load_h.transfer_async(3, load_src, load_dst)
    store_h.wait({2})
    load_h.wait({3})
    assert store_h.get_finished()
    assert load_h.get_finished()

    pat1 = _pattern_matrix(NUM_GPU_BLOCKS)
    pat2 = _pattern_matrix(NUM_GPU_BLOCKS + 7)[7:]
    for k in range(span1):
        assert torch.equal(gpu1[150 + k].cpu(), pat1[10 + k]), f"load t0 {k}"
        assert torch.equal(gpu2[170 + k].cpu(), pat2[30 + k]), f"load t1 {k}"
    for k in range(span2):
        # concurrent store: chunks 8..11 hold rows 60.. / 100..
        pos = k
        ch = 8 + pos // BLOCKS_PER_CHUNK
        sub = pos % BLOCKS_PER_CHUNK
        assert (
            cpu1[ch, sub * PAGE : (sub + 1) * PAGE] == pat1[60 + k]
        ).all(), f"store t0 {k}"
        ch1 = 10 + pos // BLOCKS_PER_CHUNK
        assert (
            cpu2[ch1, sub * PAGE : (sub + 1) * PAGE] == pat2[100 + k]
        ).all(), f"store t1 {k}"


def _mp_dma_child(rank, engine_id, geometry, barrier, out_q):
    """Child process: own CUDA context, own mmap VA of the shared file,
    registers the ENTIRE region (production behavior), waits on the
    barrier, then stores rank-unique content into its own cells."""
    import torch

    from vllm.v1.kv_offload.base import CanonicalKVCacheRef, GPULoadStoreSpec
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
    from vllm.v1.kv_offload.cpu.gpu_worker import (
        SingleDirectionOffloadingHandler,
        pin_mmap_region,
    )
    from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

    n_blocks, row_stride, cpu_page_size, W = geometry
    PAGE = W // 8
    BPC = 8
    try:
        region = SharedOffloadRegion(
            engine_id=engine_id,
            num_blocks=n_blocks,
            rank=rank,
            kv_bytes_per_block=row_stride,
            cpu_page_size=cpu_page_size,
        )
        pin_mmap_region(region)
        views = [region.create_next_view(W) for _ in range(2)]

        g1 = torch.zeros(48, PAGE, dtype=torch.int8, device="cuda")
        g2 = torch.zeros(48, PAGE, dtype=torch.int8, device="cuda")
        for t in (g1, g2):
            ids = torch.arange(48, dtype=torch.int32).view(48, 1)
            offs = torch.arange(PAGE, dtype=torch.int32).view(1, PAGE)
            t.copy_(
                ((rank * 51000 + ids * 131 + offs * 7 + 11) & 0xFF).to(
                    torch.int8
                )
            )
        refs = [
            [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=PAGE)],
            [CanonicalKVCacheRef(tensor_idx=1, page_size_bytes=PAGE)],
        ]
        handler = SingleDirectionOffloadingHandler(
            gpu_tensors=[g1, g2],
            cpu_tensors=views,
            blocks_per_chunk=BPC,
            kv_cache_groups_data_refs=refs,
            gpu_to_cpu=True,
        )
        span = 20
        spec = GPULoadStoreSpec(
            list(range(4, 4 + span)) + list(range(24, 24 + span)),
            group_sizes=[span, span],
            block_indices=[5, 5],
        )
        dst = CPULoadStoreSpec([2, 3, 4, 5, 6, 7, 8, 9])  # 4 per group
        barrier.wait()  # both ranks submit simultaneously
        assert handler.transfer_async(1, spec, dst)
        handler.wait({1})
        assert handler.get_finished()

        ok = True
        detail = ""
        for k in range(span):
            pos = 5 + k
            ch = 2 + pos // BPC
            sub = pos % BPC
            if not (
                views[0][ch, sub * PAGE : (sub + 1) * PAGE].cpu()
                == g1[4 + k].cpu()
            ).all():
                ok = False
                detail = f"rank{rank} t0 k={k}"
                break
            if not (
                views[1][6 + pos // BPC, sub * PAGE : (sub + 1) * PAGE].cpu()
                == g2[24 + k].cpu()
            ).all():
                ok = False
                detail = f"rank{rank} t1 k={k}"
                break
        out_q.put((rank, ok, detail, [v.data_ptr() for v in views]))
    except Exception as e:  # noqa: BLE001
        out_q.put((rank, False, f"exception: {e!r}", []))


def test_mmap_two_process_concurrent_dma(tensors):
    """THE last untested production dimension: multiple worker PROCESSES,
    separate CUDA contexts, one shared mmap file, each registering the
    ENTIRE region with cudaHostRegister, CONCURRENT batch DMAs from both
    contexts into rank-interleaved cells. After both stores, a scheduler-
    style rank=None mapping independently verifies every byte - catching
    any cross-context aliasing the workers themselves cannot see."""
    import sys

    if not sys.platform.startswith("linux"):
        pytest.skip("SharedOffloadRegion requires Linux /dev/shm")
    import multiprocessing as mp

    W = PAGE * BLOCKS_PER_CHUNK
    cpu_page_size = 2 * W
    row_stride = cpu_page_size * 2
    n_blocks = 16
    engine_id = "itest-mp"
    import os

    path = f"/dev/shm/vllm_offload_{engine_id}.mmap"
    if os.path.exists(path):
        os.remove(path)

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    out_q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_mp_dma_child,
            args=(
                r,
                engine_id,
                (n_blocks, row_stride, cpu_page_size, W),
                barrier,
                out_q,
            ),
        )
        for r in range(2)
    ]
    try:
        for p in procs:
            p.start()
        results = [out_q.get(timeout=180) for _ in range(2)]
        for p in procs:
            p.join(timeout=60)
        for rank, ok, detail, ptrs in results:
            assert ok, f"rank {rank} self-verification failed: {detail}"

        # scheduler-style independent view: verify both ranks' bytes
        from vllm.v1.kv_offload.cpu.shared_offload_region import (
            SharedOffloadRegion,
        )

        sched = SharedOffloadRegion(
            engine_id=engine_id,
            num_blocks=n_blocks,
            rank=None,
            kv_bytes_per_block=row_stride,
            cpu_page_size=cpu_page_size,
        )
        base = sched._base.view(n_blocks, row_stride)
        for rank in range(2):
            off = rank * cpu_page_size
            v0 = base[2, off + 5 * PAGE : off + 6 * PAGE]
            g1_first = (rank * 51000 + 4 * 131 + 11) & 0xFF
            assert int(v0[0]) == g1_first, (
                f"scheduler view: rank{rank} chunk2 t0 first byte "
                f"{int(v0[0])} != {g1_first} (cross-context aliasing!)"
            )
        sched.cleanup()
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        if os.path.exists(path):
            os.remove(path)
