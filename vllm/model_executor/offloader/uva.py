# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UVA-based CPU offloading using Unified Virtual Addressing."""

from collections.abc import Generator

import torch
import torch.nn as nn
from torch.func import functional_call

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader, should_pin_memory
from vllm.utils.mem_utils import format_gib
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


def _gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def _mib(num_bytes: int) -> float:
    return num_bytes / (1024**2)


def _is_rank0() -> bool:
    try:
        from vllm.distributed.parallel_state import get_world_group
        return get_world_group().rank_in_group == 0
    except Exception:
        return True


def _probe_sticky_cuda_error(
    name: str, t: torch.Tensor, offloaded_so_far: int
) -> str | None:
    """Surface a sticky CUDA error *before* pin_memory().

    cudaHostAlloc raising "invalid argument" is almost always a *sticky* CUDA
    error left behind by an earlier failed op (e.g. an allocation that tripped
    during tight-VRAM weight loading), not a problem with this tensor. A tiny
    no-op CUDA op will re-raise the sticky error, exposing the real cause.
    Returns a warning string if a sticky error is found, else None.
    """
    if not torch.cuda.is_available():
        return None
    try:
        torch.cuda.synchronize()
        torch.zeros(1, device="cuda")  # no-op to surface a sticky error
    except Exception as probe_err:
        return (
            "[UVA-OFFLOAD] sticky CUDA error detected BEFORE pinning "
            f"param {name!r} (shape={tuple(t.shape)}, dtype={t.dtype}, "
            f"size={_mib(t.numel() * t.element_size()):.2f} MiB, offloaded so "
            f"far={_gib(offloaded_so_far):.2f} GiB): {probe_err!r}. "
            "=> the pin_memory() failure below is a symptom of this earlier "
            "error, not the param itself. Investigate the FIRST CUDA error "
            "earlier in the log."
        )
    return None


def _maybe_log_pin(name: str, t: torch.Tensor, offloaded_so_far: int) -> None:
    """Throttled progress log (rank0 only) of which tensors are being pinned."""
    if not _is_rank0():
        return
    this = t.numel() * t.element_size()
    # Log first 8 GiB worth plus an entry every 8 GiB thereafter.
    if offloaded_so_far < 8 * (1024**3) or (
        offloaded_so_far % (8 * (1024**3)) < this
    ):
        logger.info(
            "[UVA-OFFLOAD] pinning param %r (shape=%s, dtype=%s, %.2f MiB); "
            "total offloaded so far=%.2f GiB",
            name, tuple(t.shape), t.dtype, _mib(this), _gib(offloaded_so_far),
        )


def _format_pin_failure(
    name: str, t: torch.Tensor, offloaded_so_far: int, pin_err: BaseException
) -> str:
    """Detailed diagnostics for a pin_memory() failure."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        memlock = f"soft={soft} hard={hard}"
    except Exception:
        memlock = "n/a"
    return (
        "[UVA-OFFLOAD] pin_memory() FAILED on param "
        f"{name!r} (shape={tuple(t.shape)}, dtype={t.dtype}, "
        f"{_mib(t.numel() * t.element_size()):.2f} MiB); offloaded so far="
        f"{_gib(offloaded_so_far):.2f} GiB. error: {pin_err!r}. "
        f"RLIMIT_MEMLOCK (locked memory; needs 'unlimited' or >= offload "
        f"size): {memlock}. Hint: a prior sticky CUDA error usually causes "
        "'invalid argument' here -- scan up for the first CUDA error. Set "
        "VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 to bypass pinning (slower "
        "transfers) and isolate whether pinning itself is at fault."
    )


class UVAOffloader(BaseOffloader):
    """Offloader using Unified Virtual Addressing (UVA) for zero-copy access.

    This offloader moves parameters to pinned CPU memory and creates CUDA views
    using UVA. The GPU can then directly access the CPU memory without explicit
    transfers, at the cost of PCIe bandwidth (slower than GPU memory).

    When UVA is disabled via env var, falls back to a functional_call-based
    approach that moves parameters on-demand.

    Args:
        cpu_offload_max_bytes: Maximum bytes to offload to CPU.
        cpu_offload_params: Set of parameter name segments to selectively
            offload. If empty, all parameters are eligible up to the byte limit.
    """

    def __init__(
        self,
        cpu_offload_max_bytes: int,
        cpu_offload_params: set[str] | None = None,
    ):
        self.cpu_offload_max_bytes = cpu_offload_max_bytes
        self.cpu_offload_bytes = 0
        self.cpu_offload_params = cpu_offload_params or set()

        self.pin_memory = should_pin_memory()
        self.uva_offloading = (
            is_uva_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA
        )

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Wrap modules with UVA offloading."""
        modules = [self._maybe_offload_to_cpu(module) for module in modules_generator]
        if self.cpu_offload_bytes > 0:
            logger.info(
                "Total CPU offloaded parameters: %s",
                format_gib(self.cpu_offload_bytes),
            )
        return modules

    def _maybe_offload_to_cpu(self, module: nn.Module) -> nn.Module:
        """Offload module parameters to CPU using UVA if budget allows."""
        if (params := next(module.parameters(), None)) is None:
            return module

        device = params.device

        if device == torch.device("cpu"):
            return module

        if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
            return module

        # offload parameters to CPU
        # use pin_memory if possible, which helps cudagraph capture speed
        offloaded_parameters = False
        for name, p in module.named_parameters():
            if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
                # we use per-parameter offloading
                # one module might have some parameters offloaded and some not
                break

            if self.cpu_offload_params:
                # Check if parameter belongs to the offloading set
                # Add dots here to ensure we match full segments only
                # e.g., "experts.w2_weight" matches "mlp.experts.w2_weight"
                # but not "mlp.experts.w2_weight_scale"
                should_offload = any(
                    f".{param}." in f".{name}." for param in self.cpu_offload_params
                )
                if not should_offload:
                    continue

            cpu_data = p.data.to(device="cpu")
            if self.pin_memory:
                _probe = _probe_sticky_cuda_error(name, p.data, self.cpu_offload_bytes)
                if _probe is not None:
                    logger.warning(_probe)
                _maybe_log_pin(name, p.data, self.cpu_offload_bytes)
                try:
                    cpu_data = cpu_data.pin_memory()
                except Exception as pin_err:
                    logger.error(_format_pin_failure(name, p.data, self.cpu_offload_bytes, pin_err))
                    raise

            if not self.uva_offloading:
                p.data = cpu_data
            else:
                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                p._vllm_is_uva_offloaded = True

            self.cpu_offload_bytes += p.data.numel() * p.data.element_size()
            offloaded_parameters = True

        if offloaded_parameters and not self.uva_offloading:
            original_forward = module.forward

            def forward(*args, **kwargs):
                module.forward = original_forward
                device_state = {
                    # here we blindly call `to(device)`
                    # if the parameter is already on the device,
                    # it will be a no-op
                    k: v.to(device, non_blocking=True)
                    for k, v in module.state_dict().items()
                }

                # set `tie_weights=False` as tied weights in original model
                # become untied when calling .to(device) individually
                output = functional_call(
                    module,
                    device_state,
                    args=args,
                    kwargs=kwargs,
                    tie_weights=False,
                )
                module.forward = forward
                return output

            module.forward = forward

        return module
