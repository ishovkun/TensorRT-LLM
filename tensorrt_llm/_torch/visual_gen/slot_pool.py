# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Delivery slots: the storage a session result is written into and read from.

The executor reserves one slot before an advance, the pipeline writes its streams
into it, and the slot stays reserved until the client has moved past the result it
carries. K slots form the pool. Host-placed streams live in shared host memory that
the worker page-locks, so the pipeline's device-to-host copy is asynchronous and the
client maps the same bytes without a copy. Device-placed streams live in device
memory shared through CUDA IPC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm._torch.visual_gen.pipeline import SessionCapabilities, StreamSpec
from tensorrt_llm.logger import logger


@dataclass
class _StreamBuffer:
    spec: StreamSpec
    storage: torch.Tensor  # flat uint8 of spec.max_bytes
    handle: Dict[str, Any]  # what the client maps it from; made once, at allocation
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[torch.dtype] = None
    nbytes: int = 0


class DeliverySlot:
    """One slot: a buffer per declared stream plus the shapes written into them.

    The pipeline asks for ``buffer(name, shape, dtype)`` and copies into the view it
    gets back. The executor reads ``written()`` after the advance to build the result
    message and ``reset()`` when the slot is reserved again.
    """

    def __init__(self, slot_id: int, buffers: Dict[str, _StreamBuffer]) -> None:
        self.slot_id = slot_id
        self._buffers = buffers

    @property
    def stream_names(self) -> Tuple[str, ...]:
        return tuple(self._buffers)

    def device_of(self, name: str) -> torch.device:
        return self._buffers[name].storage.device

    def buffer(self, name: str, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """A tensor view of ``shape`` and ``dtype`` over this slot's storage for
        stream ``name``. Raises ``ValueError`` when the request exceeds what the
        stream declared."""
        buf = self._buffers[name]
        spec = buf.spec
        if dtype != spec.dtype:
            raise ValueError(f"stream {name!r}: {dtype} requested, {spec.dtype} declared")
        shape = tuple(int(d) for d in shape)
        if len(shape) != len(spec.max_shape) or any(a > b for a, b in zip(shape, spec.max_shape)):
            raise ValueError(f"stream {name!r}: shape {shape} exceeds declared {spec.max_shape}")
        nbytes = 1
        for d in shape:
            nbytes *= d
        nbytes *= dtype.itemsize
        buf.shape, buf.dtype, buf.nbytes = shape, dtype, nbytes
        return buf.storage[:nbytes].view(dtype).view(shape)

    def written(self) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        """Stream name to the shape and dtype written since the last reset."""
        return {
            name: (buf.shape, buf.dtype)
            for name, buf in self._buffers.items()
            if buf.shape is not None
        }

    def reset(self) -> None:
        for buf in self._buffers.values():
            buf.shape, buf.dtype, buf.nbytes = None, None, 0


class SlotPool:
    """K delivery slots for one session, allocated on the worker.

    ``export_handles()`` hands the client one shared-tensor handle per slot per
    stream; the client maps them once for the session. ``reserve()`` returns a free
    slot or ``None``; ``release(slot_id)`` returns it when the client has moved past
    its result. ``close()`` drops the storage.
    """

    def __init__(
        self,
        capabilities: SessionCapabilities,
        num_slots: int,
        device: torch.device,
        *,
        local: bool = False,
    ) -> None:
        if num_slots < 1:
            raise ValueError("a session needs at least one delivery slot")
        self.capabilities = capabilities
        self.num_slots = num_slots
        self.device = device
        self.local = local
        self._registered: List[torch.Tensor] = []
        self._slots: List[DeliverySlot] = []
        self._free: List[int] = []
        for slot_id in range(num_slots):
            buffers = {spec.name: self._allocate(spec) for spec in capabilities.streams}
            self._slots.append(DeliverySlot(slot_id, buffers))
            self._free.append(slot_id)
        self._free.reverse()  # reserve() pops from the end: slot 0 first

    def _allocate(self, spec: StreamSpec) -> _StreamBuffer:
        if spec.placement == "device":
            storage = torch.empty(spec.max_bytes, dtype=torch.uint8, device=self.device)
            handle = SharedTensorContainer.from_tensor(storage, local=self.local).dump_to_dict()
        else:
            storage = torch.empty(spec.max_bytes, dtype=torch.uint8)
            # Sharing moves the bytes into shared memory and changes the storage
            # address, so the handle is made before the address is page-locked.
            handle = SharedTensorContainer.from_tensor(storage, local=self.local).dump_to_dict()
            _page_lock(storage, self._registered)
        return _StreamBuffer(spec=spec, storage=storage, handle=handle)

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def nbytes(self) -> int:
        return self.num_slots * self.capabilities.max_result_bytes

    def slot(self, slot_id: int) -> DeliverySlot:
        return self._slots[slot_id]

    def reserve(self) -> Optional[DeliverySlot]:
        if not self._free:
            return None
        slot = self._slots[self._free.pop()]
        slot.reset()
        return slot

    def release(self, slot_id: int) -> None:
        if slot_id in self._free:
            raise ValueError(f"slot {slot_id} returned twice")
        if not 0 <= slot_id < self.num_slots:
            raise ValueError(f"slot {slot_id} does not exist")
        self._free.append(slot_id)

    def release_all(self) -> None:
        self._free = list(range(self.num_slots))[::-1]

    def export_handles(self) -> List[Dict[str, Dict[str, Any]]]:
        """One handle dict per slot per stream, for ``SessionOpened``. Each handle is
        consumed exactly once by the client's mapping."""
        return [
            {name: slot._buffers[name].handle for name in slot.stream_names}
            for slot in self._slots
        ]

    def close(self) -> None:
        for storage in self._registered:
            _page_unlock(storage)
        self._registered.clear()
        self._slots.clear()
        self._free.clear()


def _page_lock(storage: torch.Tensor, registered: List[torch.Tensor]) -> None:
    """Pin an existing host allocation so device-to-host copies into it are
    asynchronous. Shared memory cannot be allocated pinned, so it is registered
    after the fact; a failure leaves the copy synchronous and is logged once."""
    if not torch.cuda.is_available() or storage.numel() == 0:
        return
    try:
        rc = torch.cuda.cudart().cudaHostRegister(storage.data_ptr(), storage.numel(), 0)
    except Exception as exc:  # the runtime binding may be missing on some builds
        logger.warning(f"slot pool: cudaHostRegister unavailable ({exc}); copies stay synchronous")
        return
    if int(rc) != 0:
        logger.warning(
            f"slot pool: cudaHostRegister failed with {int(rc)}; copies stay synchronous"
        )
        return
    registered.append(storage)


def _page_unlock(storage: torch.Tensor) -> None:
    """Undo ``_page_lock`` before the storage is freed: a freed range that is
    still registered poisons the next allocation that lands on the same
    addresses (``cudaErrorHostMemoryAlreadyRegistered`` at its registration)."""
    try:
        rc = torch.cuda.cudart().cudaHostUnregister(storage.data_ptr())
    except Exception as exc:
        logger.warning(f"slot pool: cudaHostUnregister failed: {exc}")
        return
    if int(rc) != 0:
        logger.warning(f"slot pool: cudaHostUnregister failed with {int(rc)}")


class ClientSlotMap:
    """The client's mapping of a session's slot pool: one tensor per slot per stream,
    mapped once from the handles in ``SessionOpened`` and kept for the session."""

    def __init__(self, handles: List[Dict[str, Dict[str, Any]]]) -> None:
        self._views: List[Dict[str, torch.Tensor]] = []
        for per_stream in handles:
            mapped = {}
            for name, handle in per_stream.items():
                mapped[name] = SharedTensorContainer.from_dict(handle).get_local_view()
            self._views.append(mapped)

    @property
    def num_slots(self) -> int:
        return len(self._views)

    def view(
        self, slot_id: int, name: str, shape: Tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        nbytes = math.prod(int(d) for d in shape) * dtype.itemsize
        storage = self._views[slot_id][name]
        return storage[:nbytes].view(dtype).view(tuple(shape))

    def close(self) -> None:
        self._views.clear()


__all__ = [
    "ClientSlotMap",
    "DeliverySlot",
    "SlotPool",
]
