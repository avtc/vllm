# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Store-on-evict offloading (VLLM_KV_OFFLOAD_STORE_MODE=on_evict).

Population model: nothing is stored eagerly; prefix-cache blocks evicted by
the GPU are copied to the CPU tier at eviction time, before the reusing
forward pass overwrites them. See OffloadingConnectorScheduler.
"""

from collections import OrderedDict
from types import SimpleNamespace

import sys

if sys.platform == "win32":
    # uvloop is Unix-only; stub it so this pure-python test imports on Windows.
    # (On Linux/CI the real dependency is installed.)
    import types as _types

    sys.modules.setdefault("uvloop", _types.ModuleType("uvloop"))

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LookupResult,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

CTX = ReqContext(req_id="", kv_transfer_params=None)


def _h(block_hash: bytes, group_id: int = 0):
    return make_block_hash_with_group_id(block_hash, group_id)


def test_block_pool_collects_evictions_when_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_KV_OFFLOAD_STORE_MODE", "on_evict")
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=128)
    assert pool.evicted_blocks_collection_enabled
    assert pool.drain_evicted_blocks() == []

    blocks = pool.get_new_blocks(2)
    pool._insert_block_hash(_h(b"\x11" * 8), blocks[0], None)
    pool.free_blocks(blocks)
    reused = pool.get_new_blocks(2)
    assert {b.block_id for b in reused} == {b.block_id for b in blocks}

    evicted = pool.drain_evicted_blocks()
    assert len(evicted) == 1
    block_id, hashes = evicted[0]
    assert block_id == blocks[0].block_id
    assert len(hashes) == 1
    # drain clears the buffer
    assert pool.drain_evicted_blocks() == []


def test_block_pool_no_collection_when_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_KV_OFFLOAD_STORE_MODE", "eager")
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=128)
    assert not pool.evicted_blocks_collection_enabled
    blocks = pool.get_new_blocks(2)
    pool._insert_block_hash(_h(b"\x22" * 8), blocks[0], None)
    pool.free_blocks(blocks)
    pool.get_new_blocks(2)
    assert pool.drain_evicted_blocks() == []


def _make_fake_scheduler(manager: CPUOffloadingManager):
    cfg = SimpleNamespace(
        blocks_per_chunk=1,
        num_workers=1,
        kv_group_configs=[SimpleNamespace(group_idx=0)],
    )
    counter = [0]

    def gen_id():
        counter[0] += 1
        return counter[0]

    return SimpleNamespace(
        _store_mode="on_evict",
        _on_evict_max_blocks=32,
        config=cfg,
        manager=manager,
        _chain_parent=OrderedDict(),
        _evict_req_id="__on_evict_store__",
        _evict_req_context=ReqContext(
            req_id="__on_evict_store__", kv_transfer_params=None
        ),
        _jobs={},
        _generate_job_id=gen_id,
        _dropped_evict_keys=OrderedDict(),
        _dropped_evict_keys_cap=65_536,
    )


def test_build_evict_store_jobs_creates_job_and_stores_keys():
    manager = CPUOffloadingManager(
        num_blocks=8, cache_policy="lru", enable_events=False, store_threshold=0
    )
    fake = _make_fake_scheduler(manager)
    out = SimpleNamespace(
        evicted_cached_blocks=[
            (5, [_h(b"\xa1" * 8), _h(b"\xa2" * 8)]),
            (6, [_h(b"\xa3" * 8)]),
        ]
    )

    jobs = OffloadingConnectorScheduler._build_evict_store_jobs(fake, out)
    assert len(jobs) == 1
    (job_id, job), = jobs.items()
    # block 5 carries two aliased hashes -> two copy ops from block 5
    assert job.src_spec.block_ids.tolist() == [5, 5, 6]
    assert list(job.src_spec.group_sizes) == [3]
    assert list(job.src_spec.block_indices) == [0]
    assert fake._jobs[job_id].req_id == "__on_evict_store__"
    assert fake._jobs[job_id].is_store

    manager.complete_store(
        list(fake._jobs[job_id].keys), fake._evict_req_context
    )
    for b in (b"\xa1", b"\xa2", b"\xa3"):
        assert manager.lookup(make_offload_key(b * 8, 0), CTX) is LookupResult.HIT


def test_build_evict_store_jobs_dedups_already_stored():
    manager = CPUOffloadingManager(
        num_blocks=8, cache_policy="lru", enable_events=False, store_threshold=0
    )
    fake = _make_fake_scheduler(manager)
    first = SimpleNamespace(evicted_cached_blocks=[(5, [_h(b"\xa1" * 8)])])
    OffloadingConnectorScheduler._build_evict_store_jobs(fake, first)
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (  # noqa: E501
        TransferJobStatus,
    )

    status = next(iter(fake._jobs.values()))
    manager.complete_store(list(status.keys), fake._evict_req_context)

    second = SimpleNamespace(evicted_cached_blocks=[(9, [_h(b"\xa1" * 8)])])
    assert OffloadingConnectorScheduler._build_evict_store_jobs(fake, second) == {}


def test_build_evict_store_jobs_passes_chain_parents():
    manager = CPUOffloadingManager(
        num_blocks=8, cache_policy="lru", enable_events=False, store_threshold=0
    )
    fake = _make_fake_scheduler(manager)
    # b1 is the root; b2 extends it. b1 must be resident first, otherwise
    # the reachability filter correctly skips b2 as hole-fronted.
    first = SimpleNamespace(evicted_cached_blocks=[(6, [_h(b"\xb1" * 8)])])
    OffloadingConnectorScheduler._build_evict_store_jobs(fake, first)
    manager.complete_store(
        list(next(iter(fake._jobs.values())).keys), fake._evict_req_context
    )
    fake._jobs.clear()

    fake._chain_parent[make_offload_key(b"\xb2" * 8, 0)] = make_offload_key(
        b"\xb1" * 8, 0
    )
    seen = {}
    orig = manager.prepare_store

    def spy(keys, ctx, parent_map=None):
        seen.update(parent_map or {})
        return orig(keys, ctx, parent_map=parent_map)

    manager.prepare_store = spy
    out = SimpleNamespace(evicted_cached_blocks=[(7, [_h(b"\xb2" * 8)])])
    OffloadingConnectorScheduler._build_evict_store_jobs(fake, out)
    assert seen.get(make_offload_key(b"\xb2" * 8, 0)) == make_offload_key(
        b"\xb1" * 8, 0
    )


def test_build_evict_store_jobs_gated_by_mode():
    manager = CPUOffloadingManager(
        num_blocks=8, cache_policy="lru", enable_events=False, store_threshold=0
    )
    fake = _make_fake_scheduler(manager)
    fake._store_mode = "eager"
    out = SimpleNamespace(evicted_cached_blocks=[(5, [_h(b"\x33" * 8)])])
    assert OffloadingConnectorScheduler._build_evict_store_jobs(fake, out) == {}
    assert (
        OffloadingConnectorScheduler._build_evict_store_jobs(
            fake, SimpleNamespace(evicted_cached_blocks=None)
        )
        == {}
    )


def test_worker_submits_evict_jobs_without_deferral_and_waits():
    events: list[tuple] = []

    class FakeTransferWorker:
        def submit_store(self, jid, src, dst):
            events.append(("submit", jid))
            return True

        def submit_load(self, jid, src, dst):
            events.append(("load", jid))
            return True

        def wait(self, ids):
            events.append(("wait", tuple(sorted(ids))))

    wc = OffloadingConnectorWorker.__new__(OffloadingConnectorWorker)
    wc.worker = FakeTransferWorker()
    wc._unsubmitted_store_jobs = [
        ("prev_deferred", GPULoadStoreSpec([1], [1], [0]), "dst")
    ]
    evict_job = TransferJob(
        req_id="__on_evict_store__",
        src_spec=GPULoadStoreSpec([5], [1], [0]),
        dst_spec="dst",
    )
    meta = OffloadingConnectorMetadata(
        load_jobs={},
        store_jobs={},
        jobs_to_flush=None,
        evict_store_jobs={42: evict_job},
    )

    wc.start_kv_transfers(meta)
    # deferred (previous-step) store first, then the same-step evict store,
    # then a synchronous wait BEFORE the method returns (i.e. before forward).
    assert events == [("submit", "prev_deferred"), ("submit", 42), ("wait", (42,))]


def _key(b: bytes):
    return make_offload_key(b, 0)


def test_evict_store_captures_children_before_parents():
    """GPU eviction is coldest-first and prefix hits keep conversation
    heads hot, so children typically evict BEFORE their parents. A chunk
    whose parent is merely absent (still GPU-resident, will evict later)
    must be captured optimistically — the hole heals when the parent
    arrives."""
    manager = CPUOffloadingManager(
        num_blocks=8, cache_policy="lru", enable_events=False, store_threshold=0
    )
    fake = _make_fake_scheduler(manager)
    # chain: c1 -> c0, c2 -> c1, c3 -> c2 ; c0 is the root (still on GPU)
    fake._chain_parent[_key(b"\xc1" * 8)] = _key(b"\xc0" * 8)
    fake._chain_parent[_key(b"\xc2" * 8)] = _key(b"\xc1" * 8)
    fake._chain_parent[_key(b"\xc3" * 8)] = _key(b"\xc2" * 8)

    # Children evict first, parent c0 not evicted yet: ALL are captured.
    out = SimpleNamespace(
        evicted_cached_blocks=[
            (7, [_h(b"\xc3" * 8)]),
            (6, [_h(b"\xc2" * 8)]),
            (5, [_h(b"\xc1" * 8)]),
        ]
    )
    jobs = OffloadingConnectorScheduler._build_evict_store_jobs(fake, out)
    assert len(jobs) == 1
    status = next(iter(fake._jobs.values()))
    manager.complete_store(list(status.keys), fake._evict_req_context)
    for b in (b"\xc1", b"\xc2", b"\xc3"):
        assert manager.lookup(_key(b * 8), CTX) is LookupResult.HIT

    # The root evicts last and completes the chain.
    out_root = SimpleNamespace(evicted_cached_blocks=[(4, [_h(b"\xc0" * 8)])])
    assert len(OffloadingConnectorScheduler._build_evict_store_jobs(fake, out_root)) == 1


def test_evict_store_skips_descendants_of_dropped_ancestors():
    """A key dropped by cap overflow is gone forever; its descendants are
    permanently unreachable through the consecutive-from-start lookup and
    must be skipped. Re-capturing the dropped content heals the frontier."""
    manager = CPUOffloadingManager(
        num_blocks=8, cache_policy="lru", enable_events=False, store_threshold=0
    )
    fake = _make_fake_scheduler(manager)
    fake._on_evict_max_blocks = 1
    # chain: a (root), b -> a
    fake._chain_parent[_key(b"\xe1" * 8)] = _key(b"\xe0" * 8)

    # Burst of 2 with cap 1: the unrelated root x is taken, a is dropped.
    burst = SimpleNamespace(
        evicted_cached_blocks=[
            (1, [_h(b"\xf0" * 8)]),
            (2, [_h(b"\xe0" * 8)]),
        ]
    )
    jobs = OffloadingConnectorScheduler._build_evict_store_jobs(fake, burst)
    assert len(jobs) == 1  # only x stored
    assert _key(b"\xe0" * 8) in fake._dropped_evict_keys

    # b evicts later: its parent a was dropped -> skipped forever.
    child = SimpleNamespace(evicted_cached_blocks=[(3, [_h(b"\xe1" * 8)])])
    assert OffloadingConnectorScheduler._build_evict_store_jobs(fake, child) == {}

    # a is recomputed (content re-prefilled) and evicted again -> captured,
    # healing the drop; b evicted again afterwards -> captured too.
    heal = SimpleNamespace(evicted_cached_blocks=[(4, [_h(b"\xe0" * 8)])])
    assert len(OffloadingConnectorScheduler._build_evict_store_jobs(fake, heal)) == 1
    assert _key(b"\xe0" * 8) not in fake._dropped_evict_keys
    again = SimpleNamespace(evicted_cached_blocks=[(5, [_h(b"\xe1" * 8)])])
    assert len(OffloadingConnectorScheduler._build_evict_store_jobs(fake, again)) == 1
