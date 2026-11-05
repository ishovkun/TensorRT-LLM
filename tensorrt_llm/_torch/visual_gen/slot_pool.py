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
from typing import Dict, List, Optional, Tuple

import torch

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm._torch.visual_gen.pipeline import SessionCapabilities, SessionStreamSpec
from tensorrt_llm._torch.visual_gen.session_protocol import SharedTensorHandle, SlotHandles
from tensorrt_llm.logger import logger


class StreamBuffer:
    """Fixed storage for one stream in one slot: a flat byte tensor allocated here,
    sized for the stream's largest result, exported once so the client can map it,
    page-locked when it lives on the host. Hands out views of a prefix in the
    shape of one result; a view is the storage, no copy."""

    def __init__(self, nbytes: int, device: torch.device, *, local: bool) -> None:
        self.storage = torch.empty(nbytes, dtype=torch.uint8, device=device)
        # The export comes first: sharing host memory moves it, and the client
        # must map the address that is then page-locked.
        self.handle: SharedTensorHandle = SharedTensorContainer.from_tensor(
            self.storage, local=local
        ).dump_to_dict()
        # Host pages are pinned (cudaHostRegister) so device-to-host copies into
        # them run as asynchronous DMA; undone in close(). Device memory needs none.
        self._page_locked = device.type != "cuda" and _page_lock(self.storage)

    @property
    def device(self) -> torch.device:
        return self.storage.device

    @property
    def nbytes(self) -> int:
        return self.storage.numel()

    def view(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """A tensor of ``shape`` and ``dtype`` over the start of the storage. Raises
        ``ValueError`` when it does not fit."""
        nbytes = math.prod(shape) * dtype.itemsize
        if nbytes > self.storage.numel():
            raise ValueError(f"{nbytes} bytes requested, the buffer holds {self.storage.numel()}")
        return self.storage[:nbytes].view(dtype).view(tuple(shape))

    def is_view_of(self, tensor: torch.Tensor) -> bool:
        """Whether ``tensor`` views this storage: what a pipeline may report as written."""
        return tensor.untyped_storage().data_ptr() == self.storage.untyped_storage().data_ptr()

    def close(self) -> None:
        """Undo the page lock before the storage is freed. Repeatable."""
        if self._page_locked:
            self._page_locked = False
            _page_unlock(self.storage)


class DeliverySlot:
    """One slot: a ``StreamBuffer`` per declared stream, by name. ``slot["video"]``
    is the buffer; the pipeline writes into a view of it and reports that view in
    its outcome. The slot keeps no record of writes."""

    def __init__(self, buffers: Dict[str, StreamBuffer]) -> None:
        self._buffers = buffers

    def __getitem__(self, name: str) -> StreamBuffer:
        return self._buffers[name]

    def __contains__(self, name: str) -> bool:
        return name in self._buffers

    @property
    def stream_names(self) -> Tuple[str, ...]:
        return tuple(self._buffers)


class SlotPool:
    """K delivery slots for one session, allocated on the worker.

    ``export_handles()`` hands the client one shared-tensor handle per slot per
    stream; the client maps them once for the session. ``reserve()`` returns a free
    slot with its index, the name the client knows it by, or ``None``;
    ``release(slot_id)`` returns it when the client has moved past its result.
    ``close()`` drops the storage. A slot does not know its index: the pool is the
    only owner of positions.
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
        self._slots: List[DeliverySlot] = []
        self._free: List[int] = []
        made: List[StreamBuffer] = []
        try:
            for slot_id in range(num_slots):
                buffers: Dict[str, StreamBuffer] = {}
                for spec in capabilities.streams:
                    buffers[spec.name] = self._allocate(spec)
                    made.append(buffers[spec.name])
                self._slots.append(DeliverySlot(buffers))
                self._free.append(slot_id)
        except BaseException:
            # A later allocation failed: every buffer made so far, a half-built
            # slot's included, is closed before the error reaches the caller.
            for buffer in made:
                buffer.close()
            self._slots.clear()
            self._free.clear()
            raise
        self._free.reverse()  # reserve() pops from the end: slot 0 first

    def _allocate(self, spec: SessionStreamSpec) -> StreamBuffer:
        device = self.device if spec.placement == "device" else torch.device("cpu")
        return StreamBuffer(spec.max_bytes, device, local=self.local)

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def nbytes(self) -> int:
        return self.num_slots * self.capabilities.max_result_bytes

    def slot(self, slot_id: int) -> DeliverySlot:
        return self._slots[slot_id]

    def reserve(self) -> Optional[Tuple[int, DeliverySlot]]:
        """A free slot with its index; ``None`` when every slot is out."""
        if not self._free:
            return None
        slot_id = self._free.pop()
        return slot_id, self._slots[slot_id]

    def release(self, slot_id: int) -> None:
        if slot_id in self._free:
            raise ValueError(f"slot {slot_id} returned twice")
        if not 0 <= slot_id < self.num_slots:
            raise ValueError(f"slot {slot_id} does not exist")
        self._free.append(slot_id)

    def release_all(self) -> None:
        self._free = list(range(self.num_slots))[::-1]

    def export_handles(self) -> List[SlotHandles]:
        """One handle dict per slot per stream, for ``SessionOpened``. Each handle is
        consumed exactly once by the client's mapping."""
        return [{name: slot[name].handle for name in slot.stream_names} for slot in self._slots]

    def close(self) -> None:
        for slot in self._slots:
            for name in slot.stream_names:
                slot[name].close()
        self._slots.clear()
        self._free.clear()


def _page_lock(storage: torch.Tensor) -> bool:
    """Pin an existing host allocation so device-to-host copies into it are
    asynchronous. Shared memory cannot be allocated pinned, so it is registered
    after the fact; a failure leaves the copy synchronous and is logged once.
    Returns whether the storage is now registered."""
    if not torch.cuda.is_available() or storage.numel() == 0:
        return False
    try:
        rc = torch.cuda.cudart().cudaHostRegister(storage.data_ptr(), storage.numel(), 0)
    except Exception as exc:  # the runtime binding may be missing on some builds
        logger.warning(f"slot pool: cudaHostRegister unavailable ({exc}); copies stay synchronous")
        return False
    if int(rc) != 0:
        logger.warning(
            f"slot pool: cudaHostRegister failed with {int(rc)}; copies stay synchronous"
        )
        return False
    return True


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

    def __init__(self, handles: List[SlotHandles]) -> None:
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
    "StreamBuffer",
]
