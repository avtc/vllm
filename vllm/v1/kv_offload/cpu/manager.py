# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Collection, Iterable
from typing import Literal

from typing_extensions import override

from vllm.distributed.kv_events import MEDIUM_CPU
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.common import (
    CPULoadStoreSpec,
    CPUOffloadingMetrics,
)
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

_CACHE_POLICIES: dict[str, type[CachePolicy]] = {
    "lru": LRUCachePolicy,
    "arc": ARCCachePolicy,
}


class CPUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable CachePolicy (LRU or ARC).

    The manager owns all shared logic: ref-counting, event emission,
    block pool management, and the prepare_store/complete_store skeletons.
    Policy-specific block organization and eviction decisions are delegated
    to the CachePolicy implementation.
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: Literal["lru", "arc"] = "lru",
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
    ):
        self.medium: str = MEDIUM_CPU
        self._num_blocks: int = num_blocks
        self._num_allocated_blocks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = _CACHE_POLICIES.get(cache_policy)
        if policy_cls is None:
            raise ValueError(
                f"Unknown cache policy: {cache_policy!r}. "
                f"Supported: {list(_CACHE_POLICIES)}"
            )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_blocks)
        # Track the number of blocks in the cache that are evictable. i.e. ref_cnt 0.
        self._num_evictable_cache_blocks: int = 0
        # Track blocks with an in-flight store (ref_cnt -1, not yet completed).
        self._num_write_pending_blocks: int = 0

        self.store_threshold: int = store_threshold
        self.max_tracker_size: int = max_tracker_size
        self.stores_skipped_in_current_batch: int = 0
        self.allocation_sizes_in_current_batch: list[int] = []

        # Number of block references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

        # Session-LRU (prefix-chain) eviction: chunks are linked into
        # per-conversation chains via parent pointers supplied by the
        # scheduler at store time. Only chain leaves (chunks with no live
        # children) are evictable, so a paused session erodes from its TAIL
        # and its head (chunk 0) survives as long as possible. Without this,
        # plain LRU evicts the head first (stored/touched earliest = coldest),
        # and since lookups count consecutive hits from chunk 0, head loss
        # turns a 90%-resident session into a total miss.
        import os as _os

        self._session_lru: bool = (
            _os.environ.get("VLLM_KV_OFFLOAD_SESSION_LRU", "1") != "0"
        )
        # key -> parent key within the same prefix chain (None for roots).
        self._parent: dict[OffloadKey, OffloadKey | None] = {}
        # parent key -> live children that are still stored.
        self._children: dict[OffloadKey, set[OffloadKey]] = {}

    # --- session-chain registry ---

    def _link_chunk(self, key: OffloadKey, parent: OffloadKey | None) -> None:
        """Record `key` as a child of `parent` in its prefix chain."""
        self._parent[key] = parent
        if parent is not None:
            self._children.setdefault(parent, set()).add(key)

    def _unlink_chunk(self, key: OffloadKey) -> None:
        """Remove `key` from the chain registry (called when its block leaves
        the pool). The parent may become a new leaf for future eviction."""
        parent = self._parent.pop(key, None)
        if parent is not None:
            siblings = self._children.get(parent)
            if siblings is not None:
                siblings.discard(key)
                if not siblings:
                    del self._children[parent]
        # A key being evicted is always a leaf, but stay defensive.
        self._children.pop(key, None)

    def _evict_session_lru(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        """Evict `n` blocks by eroding prefix chains from their leaves.

        Iterates: evict currently-evictable leaves (LRU order within the
        policy), which exposes their parents as new leaves, until `n` blocks
        are freed. Keys with live children are never evicted, so heads are
        protected while any later chunk of the session survives.

        Returns the freed blocks, possibly fewer than requested `n` when
        protected ancestors pin the remaining chains; callers must then store
        a truncated prefix (never fail after eviction).
        """
        evicted_all: list[tuple[OffloadKey, BlockStatus]] = []
        remaining = n
        while remaining > 0:
            # Non-leaves: any stored key that still has live children.
            leaf_protected = protected | set(self._children.keys())
            # Evict ONE leaf at a time: policies return None when fewer
            # than `n` candidates exist, but each group iteration only has
            # as many leaves as chains — and evicting one exposes its parent
            # as the next leaf.
            evicted = self._policy.evict(1, leaf_protected)
            if not evicted:
                break
            key, block = evicted[0]
            self._unlink_chunk(key)
            evicted_all.append((key, block))
            remaining -= 1
        return evicted_all

    # --- block pool ---

    def _get_num_free_blocks(self) -> int:
        return len(self._free_list) + self._num_blocks - self._num_allocated_blocks

    def _allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
        num_fresh = min(len(keys), self._num_blocks - self._num_allocated_blocks)
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(self._num_allocated_blocks))
            self._num_allocated_blocks += 1

        # allocate reused blocks
        for _ in range(num_reused):
            blocks.append(BlockStatus(self._free_list.pop()))
        return blocks

    def _free_block(self, block: BlockStatus) -> None:
        self._free_list.append(block.block_id)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        return CPULoadStoreSpec([block.block_id for block in blocks])

    # --- OffloadingManager interface ---

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if self.counts is not None:
            if key in self.counts:
                self.counts.move_to_end(key)
                self.counts[key] += 1
            else:
                if len(self.counts) >= self.max_tracker_size:
                    self.counts.popitem(last=False)
                self.counts[key] = 1
        block = self._policy.get(key)
        if block is None:
            return LookupResult.MISS
        if not block.is_ready:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        blocks = []
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found in cache"
            assert block.is_ready, f"Block {key!r} is not ready for reading"
            if block.ref_cnt == 0:
                self._policy.mark_non_evictable(key)
                self._num_evictable_cache_blocks -= 1  # ref_cnt 0 -> 1
                assert self._num_evictable_cache_blocks >= 0
            block.ref_cnt += 1
            blocks.append(block)
        return self._get_load_store_spec(keys, blocks)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._policy.touch(keys, req_context)

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found"
            assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self._num_evictable_cache_blocks += 1  # ref_cnt 1 -> 0
                self._policy.mark_evictable(key)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        parent_map: dict[OffloadKey, OffloadKey | None] | None = None,
    ) -> PrepareStoreOutput | None:
        if self.counts is not None:
            num_keys = len(keys)
            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]
            self.stores_skipped_in_current_batch += num_keys - len(keys)
        # filter out blocks that are already stored
        keys_to_store = [k for k in keys if self._policy.get(k) is None]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        self.allocation_sizes_in_current_batch.append(len(keys_to_store))
        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()

        try:
            import os as _os
            if _os.environ.get("VLLM_KV_OFFLOAD_DEBUG") == "1" and keys_to_store:
                import logging as _lg
                _lg.getLogger("vllm.v1.kv_offload.cpu.manager").warning(
                    "[KV_OFFLOAD] STORE req_ctx=%s keys=%d free=%d/%d evictable=%d",
                    getattr(req_context, "request_id", "?"),
                    len(keys_to_store),
                    self._get_num_free_blocks(),
                    self._num_blocks,
                    self._num_evictable_cache_blocks,
                )
        except Exception:
            pass

        to_evict: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            if num_blocks_to_evict > self._num_evictable_cache_blocks:
                # Eviction will fail.
                try:
                    import os as _os
                    if _os.environ.get("VLLM_KV_OFFLOAD_DEBUG") == "1":
                        import logging as _lg
                        _lg.getLogger(
                            "vllm.v1.kv_offload.cpu.manager"
                        ).warning(
                            "[KV_OFFLOAD] STORE-FAIL: need %d blocks, "
                            "free=%d evictable=%d (pool too small / pinned) "
                            "-> KV NOT offloaded, will re-prefill on reuse",
                            len(keys_to_store),
                            self._get_num_free_blocks(),
                            self._num_evictable_cache_blocks,
                        )
                except Exception:
                    pass
                return None
            # There is a still a chance for eviction failure as some of the
            # idle blocks might be in the protected list.

            # Blocks from the original input are excluded from eviction candidates:
            # a block that was already stored must remain in the cache after this call.
            protected = set(keys)
            evicted: list[tuple[OffloadKey, BlockStatus]] | None
            if self._session_lru:
                evicted = self._evict_session_lru(num_blocks_to_evict, protected)
                if len(evicted) < num_blocks_to_evict:
                    # Protected ancestors pinned the rest: store the longest
                    # prefix that fits. The unstored trailing chunks become
                    # holes that later requests refill incrementally.
                    storable = self._get_num_free_blocks()
                    keys_to_store = keys_to_store[:storable]
                    if not keys_to_store:
                        return None
            else:
                evicted = self._policy.evict(num_blocks_to_evict, protected)
                if evicted is None:
                    return None

            # cache-policy removes only idle blocks.
            self._num_evictable_cache_blocks -= len(evicted)
            assert self._num_evictable_cache_blocks >= 0

            for key, block in evicted:
                self._free_block(block)
                to_evict.append(key)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        blocks = self._allocate_blocks(keys_to_store)
        assert len(blocks) == len(keys_to_store), (
            "Block pool did not allocate the expected number of blocks"
        )

        for key, block in zip(keys_to_store, blocks):
            self._policy.insert(key, block)
            if self._session_lru:
                self._link_chunk(
                    key, parent_map.get(key) if parent_map is not None else None
                )
        self._num_write_pending_blocks += len(keys_to_store)

        # build store specs for allocated blocks
        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        stored_keys: list[OffloadKey] = []

        if success:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    self._num_write_pending_blocks -= 1
                    self._num_evictable_cache_blocks += 1
                    self._policy.mark_evictable(key)
                    stored_keys.append(key)
        else:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    self._num_write_pending_blocks -= 1
                    self._policy.remove(key)
                    if self._session_lru:
                        self._unlink_chunk(key)
                    self._free_block(block)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    @override
    def reset_cache(self) -> None:
        # Clear ALL blocks unconditionally. The scheduler's _stale_job_threshold
        # guarantees that complete_load / complete_store are never called for
        # pre-reset jobs, so no lazy cleanup is needed. The scheduler also
        # flushes in-flight load job IDs to the workers before any new stores
        # can begin, preventing a cross-direction data race on reused offload block IDs.
        self._policy.clear()
        self._num_evictable_cache_blocks = 0
        self._num_write_pending_blocks = 0

        self._free_list.clear()
        self._num_allocated_blocks = 0
        self._parent.clear()
        self._children.clear()

    @override
    def debug_prefix_present(
        self, keys: Collection[OffloadKey]
    ) -> tuple[int, int, int]:
        """Diagnostics only: (leading-present run, total present, total keys)
        for the given chunk keys, regardless of readiness/loadability.
        Distinguishes 'prefix eroded from CPU pool' (0/N) from 'present but
        not loadable' (load-path bug)."""
        lead = 0
        present = 0
        leading = True
        for k in keys:
            if self._policy.get(k) is not None:
                present += 1
                if leading:
                    lead += 1
            else:
                leading = False
        return lead, present, len(list(keys))

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()

        # Compute cache usage.
        num_used = (
            self._num_allocated_blocks
            - len(self._free_list)
            - self._num_evictable_cache_blocks
        )
        usage = num_used / self._num_blocks if self._num_blocks > 0 else 0.0
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)

        for allocation_size in self.allocation_sizes_in_current_batch:
            stats.observe_histogram(
                CPUOffloadingMetrics.CPU_ALLOCATION_SIZE, allocation_size
            )
        self.allocation_sizes_in_current_batch.clear()

        write_usage = (
            self._num_write_pending_blocks / self._num_blocks
            if self._num_blocks > 0
            else 0.0
        )
        read_usage = max(usage - write_usage, 0.0)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC, write_usage)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC, read_usage)

        if self.store_threshold >= 2:
            stats.increase_counter(
                CPUOffloadingMetrics.STORES_SKIPPED,
                self.stores_skipped_in_current_batch,
            )
            self.stores_skipped_in_current_batch = 0

        return stats
