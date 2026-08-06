# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ampere DeepSeek-V4 attention subclass.

Subclasses the shared ``DeepseekV4Attention`` ABC and provides Ampere-native
Triton kernels for decode (FP8 dequant + BF16 attention) and prefill
(BF16 gathered KV + sparse attention).
"""

from typing import TYPE_CHECKING, cast

import os

import torch

from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import GroupCoordinator, get_dcp_group
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v4.common.ops import dcp_merge_flashmla_output
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.models.deepseek_v4.ampere.ampere_sparse_decode_fp8 import (
    ampere_sparse_decode_fp8,
    apply_attn_sink,
)
from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


# [DSv4-ampere debug] One-shot probe for the SWA (sliding-window) KV-cache WRITE.
# The SWA cache stores UN-NORMALIZED RoPE-applied KV; the bf16 RoPE portion is
# copied verbatim by quantize_and_insert_k_kernel. swa_dequant showed nan=64
# (a full RoPE row) at the divergence, so this catches whether the SWA write
# received a NaN input (kv) or wrote NaN from finite input. Gated by
# VLLM_SM86_NAN_PROBE=1; one-shot per prefix; rank0 only.
_SWA_WRITE_PROBE_FIRED: dict[str, bool] = {}


def _swa_write_nan_probe(
    prefix: str,
    kv: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if os.environ.get("VLLM_SM86_NAN_PROBE") != "1":
        return
    if _SWA_WRITE_PROBE_FIRED.get(prefix):
        return
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
    except Exception:
        pass

    # input-side: is the KV feeding the SWA write NaN?
    input_nan = False
    kvf = kv.detach().float()
    if kvf.numel():
        n_nan = int(torch.isnan(kvf).sum().item())
        n_inf = int(torch.isinf(kvf).sum().item())
        if n_nan or n_inf:
            input_nan = True
            amax = float(kvf.abs().amax().item())
            print(
                f"[SWA_WRITE ### {prefix} ###] INPUT kv NaN! nan={n_nan} inf={n_inf} "
                f"absmax={amax:.4e} shape={tuple(kv.shape)}",
                flush=True,
            )

    # output-side: read back the bf16 RoPE portion of the just-written slots.
    # ROBUST 3D layout: swa_kv_cache is [num_blocks, block_size, 584] uint8/int8.
    # Reshape to 1D and index by slot*584+448 (reshape copies if non-contiguous).
    try:
        fp8_dim = 448
        rope_bytes = 128
        slots = slot_mapping.detach()
        valid = slots >= 0
        if not bool(valid.any().item()):
            return
        sv = slots[valid].to(torch.int64)
        flat = swa_kv_cache.reshape(-1)
        head_bytes = swa_kv_cache.shape[-1]  # 584
        row_ptrs = sv * head_bytes + fp8_dim
        offs = torch.arange(rope_bytes, device=sv.device, dtype=torch.int64)
        gathered = flat[row_ptrs[:, None] + offs[None, :]].view(torch.bfloat16)
        g = gathered.float()
        n_nan = int(torch.isnan(g).sum().item())
        n_inf = int(torch.isinf(g).sum().item())
        if n_nan or n_inf:
            _SWA_WRITE_PROBE_FIRED[prefix] = True
            bad_rows = torch.where(torch.isnan(g).any(1) | torch.isinf(g).any(1))[0]
            bad_slots = sv[bad_rows][:8].tolist()
            tag = "OUTPUT-NaN-from-FINITE-input" if not input_nan else "OUTPUT-NaN (input also NaN)"
            print(
                f"[SWA_WRITE ### {prefix} ###] {tag}: WROTE NaN to SWA bf16-RoPE! "
                f"nan={n_nan} inf={n_inf} bad_slots={bad_slots} "
                f"n_valid_slots={int(sv.numel())}",
                flush=True,
            )
    except Exception as e:  # noqa: BLE001
        if os.environ.get("VLLM_SM86_PROBE_VERBOSE") == "1":
            print(f"[SWA_WRITE {prefix}] probe skipped: {e}", flush=True)


# [DSv4-ampere debug] Periodic FULL-CACHE NaN scan at decode time. The per-write
# readback probes silently failed on the packed-cache view, so they never
# confirmed whether a write produced NaN from finite input. This scan is
# LAYOUT-ROBUST: it indexes each cache's bf16-RoPE byte region [448:576] and
# views it as bf16, scanning ALL slots for NaN. Finds the exact decode step (and
# which cache/slot) at which the FIRST NaN appears — independent of which write
# path put it there. Gated by VLLM_SM86_NAN_PROBE=1; one-shot per (rank, name).
_CACHE_SCAN_FIRED: dict[str, bool] = {}


def _cache_nan_scan(name: str, cache, pass_no: int) -> None:
    import os

    if os.environ.get("VLLM_SM86_NAN_PROBE") != "1":
        return
    if cache is None:
        return
    if _CACHE_SCAN_FIRED.get(name):
        return
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
    except Exception:
        pass
    try:
        # bf16-RoPE region is bytes [448:576] of each token's row.
        rope = cache[..., 448:576]  # [..., 128] uint8/int8
        rows = rope.reshape(-1, 128)  # [num_slots, 128]
        rows_bf16 = rows.view(torch.bfloat16).float()  # [num_slots, 64]
        n_nan = int(torch.isnan(rows_bf16).sum().item())
        n_inf = int(torch.isinf(rows_bf16).sum().item())
        if n_nan or n_inf:
            _CACHE_SCAN_FIRED[name] = True
            bad = torch.where(torch.isnan(rows_bf16).any(1))[0]
            first_slot = int(bad[0].item()) if bad.numel() else -1
            print(
                f"[CACHE_SCAN ### {name} ### pass {pass_no}] NaN FOUND in cache! "
                f"nan={n_nan} inf={n_inf} first_slot={first_slot} "
                f"cache_shape={tuple(cache.shape)}",
                flush=True,
            )
    except Exception as e:  # noqa: BLE001
        if os.environ.get("VLLM_SM86_PROBE_VERBOSE") == "1":
            print(f"[CACHE_SCAN {name}] scan skipped: {e}", flush=True)


class DeepseekV4AmpereSparseBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "AMPERE_V4_MLA_SPARSE"

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        # SM8x (Ampere: A100 SM80, RTX 3080 SM86). The parent restricts to
        # Hopper/Blackwell (9/10); this backend uses the portable Triton
        # bf16 sparse-MLA kernel instead of FlashMLA/cutedsl, so it runs on 8.x.
        return capability.major == 8


def _maybe_gather_dcp_q(
    layer: "DeepseekV4AmpereAttention",
    q: torch.Tensor,
) -> tuple[torch.Tensor, int, "GroupCoordinator", bool]:
    """All-gather the real query heads across the DCP group.

    Mirrors the nvidia backend: each DCP rank attends its KV shard with ALL
    heads of the group, then the LSE merge redistributes per-head outputs.
    """
    dcp_group = get_dcp_group()
    if dcp_group.world_size == 1:
        return q, layer.n_local_heads, dcp_group, False
    q = dcp_group.all_gather(q[:, : layer.n_local_heads, :].contiguous(), dim=1)
    return q, q.shape[1], dcp_group, True


class DeepseekV4AmpereAttention(DeepseekV4Attention):
    """Ampere sparse MLA attention layer for DeepSeek V4."""

    backend_cls = DeepseekV4AmpereSparseBackend
    use_flashmla_fp8_layout = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        parallel_config = get_current_vllm_config().parallel_config
        if (
            parallel_config.decode_context_parallel_size > 1
            and parallel_config.dcp_comm_backend != "a2a"
        ):
            raise ValueError(
                "DeepseekV4 Ampere DCP requires dcp_comm_backend='a2a'."
            )
        self._sink_neg_inf: torch.Tensor | None = None

    def _merge_sink(self) -> torch.Tensor:
        """Sink tensor for the DCP merge: the trained sink when the env gate
        is on, an all--inf tensor (exact no-op in logaddexp) otherwise."""
        import os

        if os.environ.get("VLLM_SM86_SINK", "0") == "1":
            return self.attn_sink
        if self._sink_neg_inf is None:
            self._sink_neg_inf = torch.full_like(self.attn_sink, float("-inf"))
        return self._sink_neg_inf

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        from typing import cast

        if not isinstance(attn_metadata, dict):
            # Profile run: no-op, just return q (no padding needed on Ampere).
            return q

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        from vllm.models.deepseek_v4.ampere.ampere_qnorm_rope_kv_fp8_insert import (
            ampere_qnorm_rope_kv_fp8_insert,
        )

        ampere_qnorm_rope_kv_fp8_insert(
            q,
            kv,
            self.swa_cache_layer.kv_cache,
            swa_metadata.slot_mapping,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )

        # [DSv4-ampere debug] one-shot NaN probe on the SWA KV-cache write.
        _swa_write_nan_probe(
            self.prefix,
            kv,
            self.swa_cache_layer.kv_cache,
            swa_metadata.slot_mapping,
        )
        return q

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Ampere uses BF16 reference wo_a path (same as ROCm).
        from vllm.models.deepseek_v4.amd.rocm import rocm_inv_rope_einsum

        z = rocm_inv_rope_einsum(
            self.rotary_emb,
            o,
            positions,
            self.rope_head_dim,
            self.n_local_groups,
            self.o_lora_rank,
            self.wo_a,
        )
        return self.wo_b(z.flatten(1))

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: reserve workspace, skip actual kernels.
            swa_only = self.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # [DSv4-ampere debug] one-shot full-cache NaN scan BEFORE the attention
        # reads. Finds the exact decode step at which a NaN first appears in
        # either cache (main MLA or SWA), independent of which write path put
        # it there. Layout-robust: scans the bf16-RoPE region of every slot.
        global _DECODE_PASS_COUNTER
        try:
            _DECODE_PASS_COUNTER += 1
        except NameError:
            _DECODE_PASS_COUNTER = 1
        if kv_cache is not None:
            _cache_nan_scan(f"{self.prefix}.main_mla", kv_cache, _DECODE_PASS_COUNTER)
        _cache_nan_scan(
            f"{self.prefix}.swa", self.swa_cache_layer.kv_cache, _DECODE_PASS_COUNTER
        )

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                # Uniform-width buffer (PR #44573): under DCP the local topk
                # width varies per rank; keep the full buffer width, -1 padded.
                topk_indices = self.topk_indices_buffer[:num_decode_tokens]
                topk_indices.fill_(-1)
                topk_indices[:, : global_indices.shape[-1]].copy_(global_indices)
                topk_indices = topk_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        assert swa_indices is not None and swa_lens is not None
        q, num_real_heads, dcp_group, use_dcp = _maybe_gather_dcp_q(self, q)

        # Optional: route decode through the fused vLLM-Moet Triton sparse-MLA
        # port (reads fp8_ds_mla pages + dequants in-register, no flat bf16
        # workspace) for before/after comparison against the default ampere
        # two-stage (gather-dequant -> bf16 attention) decode kernel.
        # The fused kernel applies softmax + attention sink internally and
        # writes `output` directly, so the (out, lse) + apply_attn_sink /
        # dcp LSE-merge post-processing is skipped.
        # NOTE: DCP > 1 is NOT supported on this path (no lse to merge across
        # ranks); it auto-falls back to the default kernel.
        if (
            os.environ.get("VLLM_SM86_TRITON_SPARSE_MLA", "0") == "1"
            and not use_dcp
        ):
            from vllm.v1.attention.ops.triton_sparse_mla_dsv4 import (
                triton_sparse_mla_dsv4)

            swa_cache = self.swa_cache_layer.kv_cache
            extra_cache = kv_cache if not swa_only else None
            # Fused kernel does softmax + sink internally; returns output only.
            triton_sparse_mla_dsv4(
                query=q,
                swa_kv_cache=swa_cache,
                sparse_indices=swa_indices.to(torch.int32),
                compressed_kv_cache=extra_cache,
                out=output,
                bmm1_scale=self.scale,
                sinks=self.attn_sink,
                kv_layout="NHD",
                swa_topk_lens=swa_lens.to(torch.int32),
                extra_sparse_indices=(
                    topk_indices.to(torch.int32)
                    if topk_indices is not None else None),
                extra_sparse_topk_lens=(
                    topk_lens.to(torch.int32)
                    if topk_lens is not None else None),
            )
            return

        out_attn, lse = ampere_sparse_decode_fp8(
            q=q,
            kv_cache=kv_cache,
            swa_kv_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            softmax_scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
        )
        if use_dcp:
            dcp_merge_flashmla_output(
                out_attn[:, :num_real_heads, :],
                lse[:, :num_real_heads],
                self._merge_sink(),
                output,
                dcp_group,
            )
        else:
            output.copy_(apply_attn_sink(out_attn, lse, self.attn_sink))
        from vllm.models.deepseek_v4.ampere.ampere_sparse_decode_fp8 import (
            _decode_probe)
        _decode_probe("decode_output_after_sink", output)

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # [DSv4-ampere debug] one-shot cache scan at PREFILL START (before this
        # chunk writes anything). If NaN is ALREADY present here, it came from
        # a PRIOR chunk or the profile run (not this chunk's write). Combined
        # with the decode-start scan, this isolates WHICH prefill chunk first
        # plants the NaN at slots 256+ (the chunk-1 boundary).
        global _PREFILL_PASS_COUNTER
        try:
            _PREFILL_PASS_COUNTER += 1
        except NameError:
            _PREFILL_PASS_COUNTER = 1
        if compressed_k_cache is not None:
            _cache_nan_scan(f"{self.prefix}.main_mla.prefill_start", compressed_k_cache, _PREFILL_PASS_COUNTER)
        _cache_nan_scan(f"{self.prefix}.swa.prefill_start", swa_k_cache, _PREFILL_PASS_COUNTER)

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens

        q, num_real_heads, dcp_group, use_dcp = _maybe_gather_dcp_q(self, q)

        import os

        if (
            os.environ.get("VLLM_SM86_NAN_PROBE") == "1"
            and self.prefix.endswith("layers.0.attn")
        ):
            bt = swa_metadata.block_table[num_decodes:]
            print(
                f"[SWA_META] seq_lens={seq_lens.tolist() if seq_lens is not None else None} "
                f"gather_lens={gather_lens.tolist() if gather_lens is not None else None} "
                f"block_table_row0={bt[0, :6].tolist() if bt.numel() else '?'} "
                f"slot_map[:8]={swa_metadata.slot_mapping[:8].tolist()}",
                flush=True,
            )
        assert seq_lens is not None
        assert gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        chunk_size_const = self.PREFILL_CHUNK_SIZE
        num_chunks = (num_prefills + chunk_size_const - 1) // chunk_size_const

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((chunk_size_const, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size_const
            chunk_end = min(chunk_start + chunk_size_const, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                    cp_layout=self.cp_layout,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
                cp_layout=self.cp_layout,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
                cp_layout=self.cp_layout,
            )

            kv_ws = kv[:chunk_size].reshape(-1, 1, q.shape[-1])
            out, _, lse = triton_bf16_mla_sparse_interface(
                q=q[query_start:query_end],
                kv=kv_ws,
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                d_v=q.shape[-1],
                block_dpe=0,
            )
            if use_dcp:
                dcp_merge_flashmla_output(
                    out[:, :num_real_heads, :],
                    lse[:, :num_real_heads],
                    self._merge_sink(),
                    output[query_start:query_end],
                    dcp_group,
                )
            else:
                output[query_start:query_end] = apply_attn_sink(
                    out, lse, self.attn_sink
                )
