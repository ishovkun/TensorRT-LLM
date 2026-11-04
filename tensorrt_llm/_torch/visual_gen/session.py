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
"""Session state owned by the executor's worker loop.

A session is one generation whose model state stays alive on the worker while it
produces results one advance at a time. This module holds the input queues and
live values, the advance condition, the stream cursors and the messages that
travel between the coordinator and rank 0. The pipeline declares what it accepts
and produces through ``SessionCapabilities`` in ``pipeline.py`` and never sees
any of this.
"""

from __future__ import annotations

import itertools
import math
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Tuple

import torch

from .pipeline import SessionCapabilities

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest

# --------------------------------------------------------------------------- input size


def input_nbytes(value: Any, limit: float = math.inf, worst_case: bool = False) -> int:
    """The bytes a submitted value costs: the measure ``InputSpec.max_bytes``
    bounds. Counts what a value retains and what it expands to, whichever is
    larger: a tensor or NumPy array the larger of its backing storage (a narrow
    view keeps all of it alive; an array is followed down its ``base`` chain to
    the owner) and its logical size (a broadcast or expanded layout materializes
    to it in transport and validation); bytes, strings, numbers and containers
    their CPython object size (an arbitrary-precision int costs what it holds),
    containers plus every item, a repeated reference counted each time it
    appears. With ``worst_case`` the walk charges instead the most any value of
    this shape can cost: a number at the larger of a small int and a float, a
    list at its maximum over-allocation under CPython's growth policy
    (``n + n // 8 + 6`` slots); a declared bound computed that way from a
    canonical value admits every ordinary construction of it. Stops as soon
    as the total passes ``limit`` and never expands a container before its own
    size has been counted and checked, so the walk's memory and time are bounded
    by the limit, not by the value. Supported forms only, and only plain ones,
    since serialization carries whatever Python state a value drags along: the
    exact built-in types (no subclasses), ``torch.Tensor`` itself without
    autograd state or instance attributes, ``numpy.ndarray`` itself with a plain
    dtype (no fields, no metadata, no object references); every form the wire
    (pickle) carries as is. A decorated form, a container that contains itself,
    a memoryview (not picklable) or any other type raises ``ValueError`` (the
    bound could not be established)."""
    import numpy as np

    total = 0
    frames: List[Tuple[Optional[int], Any]] = [(None, iter((value,)))]  # (container id, items)
    on_path: set = set()
    while frames:
        if total > limit:
            return total
        container_id, items = frames[-1]
        try:
            v = next(items)
        except StopIteration:
            frames.pop()
            if container_id is not None:
                on_path.discard(container_id)
            continue
        kind = type(v)
        if kind is torch.Tensor:
            _check_plain_tensor_chain(v)
            total += max(v.untyped_storage().nbytes(), v.numel() * v.element_size())
        elif kind is np.ndarray:
            owner = _check_plain_array_chain(v)
            if isinstance(owner, np.ndarray):
                total += max(owner.nbytes, v.nbytes)
            else:
                total += v.nbytes  # the logical size; the owner is measured next
                frames.append((None, iter((owner,))))
        elif kind in (list, tuple, dict):
            if id(v) in on_path:
                raise ValueError("a container that contains itself is not a supported input form")
            if worst_case and kind is list:
                total += _SEQUENCE_HEADER + _SLOT * (len(v) + (len(v) >> 3) + 6)
            else:
                total += sys.getsizeof(v)
            if total > limit:
                return total  # not expanded: its slots alone are past the bound
            on_path.add(id(v))
            if isinstance(v, dict):
                frames.append((id(v), itertools.chain(v.keys(), v.values())))
            else:
                frames.append((id(v), iter(v)))
        elif kind in (bool, int, float):
            total += max(_NUMBER, sys.getsizeof(v)) if not worst_case else _NUMBER
        elif v is None or kind in (bytes, bytearray, str):
            total += sys.getsizeof(v)
        else:
            raise ValueError(f"{kind.__name__} is not a supported input form")
    return total


_NUMBER = max(sys.getsizeof(0), sys.getsizeof(0.0))  # a small int or a float, whichever is larger
_SEQUENCE_HEADER = sys.getsizeof([])  # a list with no items
_SLOT = 8  # one reference in a list


_OWNER_CHAIN_LIMIT = 64  # views of views; anything deeper is not a plain input


def _check_plain_tensor_chain(t: Any) -> None:
    """``t`` and every tensor it is a view of must be a plain ``torch.Tensor``:
    the exact type, no autograd state, no instance attributes. A view keeps its
    base alive, so whatever the base carries is retained and serialized."""
    for _ in range(_OWNER_CHAIN_LIMIT):
        if type(t) is not torch.Tensor:
            raise ValueError(f"{type(t).__name__} is not a supported input form")
        if t.requires_grad or t.__dict__:
            raise ValueError(
                "a tensor with autograd state or attributes is not a supported input form"
            )
        base = t._base
        if base is None:
            return
        t = base
    raise ValueError("a view chain this deep is not a supported input form")


def _check_plain_array_chain(a: Any) -> Any:
    """``a`` and every array it is a view of must be a plain ``numpy.ndarray``
    with a plain dtype; returns the owner at the end of the chain (an array, or
    a non-array exporter such as a tensor, whose own chain is checked)."""
    import numpy as np

    owner: Any = a
    for _ in range(_OWNER_CHAIN_LIMIT):
        if not isinstance(owner, np.ndarray):
            if isinstance(owner, torch.Tensor):
                _check_plain_tensor_chain(owner)
            return owner
        if type(owner) is not np.ndarray:
            raise ValueError(f"{type(owner).__name__} is not a supported input form")
        if not _plain_dtype(owner.dtype):
            raise ValueError(
                "a NumPy array with object references, fields or dtype metadata is not a "
                "supported input form"
            )
        if owner.base is None:
            return owner
        owner = owner.base
    raise ValueError("a view chain this deep is not a supported input form")


def _plain_dtype(dtype: Any) -> bool:
    """A NumPy dtype that carries nothing but numbers: no object references, no
    fields (checked down the nesting) and no metadata."""
    if dtype.hasobject or dtype.metadata is not None:
        return False
    if dtype.fields is not None:
        return False
    if dtype.subdtype is not None:
        return _plain_dtype(dtype.subdtype[0])
    return True


def check_input_size(spec: Any, value: Any) -> None:
    """Raise ``SessionQueueFull`` when ``value`` exceeds the input's declared
    bound (admission capacity), ``ValueError`` when its form is unsupported.
    ``spec`` is an ``InputSpec`` or its public dict."""
    limit = spec["max_bytes"] if isinstance(spec, dict) else spec.max_bytes
    name = spec["name"] if isinstance(spec, dict) else spec.name
    size = input_nbytes(value, limit)
    if size > limit:
        raise SessionQueueFull(
            f"input {name!r}: value holds at least {size} bytes, the declared bound is {limit}"
        )


class SessionError(RuntimeError):
    """Base class for session failures raised to the client."""


class SessionQueueFull(SessionError):
    """A queued input has no room; retry after the next advance."""


class SessionEnded(SessionError):
    """The session has a terminal result or is closing; no more input is admitted."""


class SessionLeaseExpired(SessionError):
    """A result's storage was returned to the worker; the view is no longer valid."""


# --------------------------------------------------------------------------- wire messages


@dataclass
class SessionOpen:
    request: "DiffusionRequest"


@dataclass
class SessionSubmit:
    session_id: int
    name: str
    value: Any


@dataclass
class SessionSlotReturn:
    session_id: int
    slot_id: int


@dataclass
class SessionClose:
    session_id: int


@dataclass
class SessionActivity:
    """The client asked for a result without returning a slot. Carries nothing;
    receiving it is what restarts the worker's idle clock."""

    session_id: int


@dataclass
class SessionOpened:
    session_id: int
    capabilities: Optional[Dict[str, Any]] = None
    slots: Optional[List[Dict[str, Dict[str, Any]]]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class SessionAdmitted:
    """The answer to the one submit a session has in flight."""

    session_id: int
    advance_index: Optional[int] = None
    error: Optional[str] = None
    error_kind: Optional[str] = None  # "invalid" | "queue_full" | "ended"


class ResultKind(Enum):
    """What an entry of a session's result stream is. The worker publishes
    ``MEDIA`` and ``TERMINAL``; ``ENDED`` and ``CLOSED`` exist only on the client,
    as the entry left at the head of its queue once the stream is over."""

    MEDIA = "media"  # a delivery slot holds the streams
    TERMINAL = "terminal"  # the session is over; ``error`` says why, ``None`` for normal completion
    ENDED = "ended"  # the terminal was taken: nothing more to read
    CLOSED = "closed"  # the client closed: the stream is discarded


@dataclass
class SessionResultMessage:
    """One entry of the result stream. For ``MEDIA``, ``streams`` maps a stream
    name to the ``(shape, dtype)`` written in ``slots[slot_id][name]`` and
    ``position`` maps it to ``(start, count)``. ``TERMINAL`` has ``slot_id=None``
    and no streams. The client-local kinds carry no sequence."""

    session_id: int
    sequence: Optional[int]
    advance_index: Optional[int]
    slot_id: Optional[int]
    streams: Dict[str, Tuple[Tuple[int, ...], str]]
    position: Dict[str, Tuple[int, int]]
    kind: ResultKind = ResultKind.MEDIA
    error: Optional[str] = None

    @classmethod
    def end(cls, session_id: int, kind: ResultKind) -> "SessionResultMessage":
        """The client-local entry that ends a stream: ``ENDED`` or ``CLOSED``."""
        if kind not in (ResultKind.ENDED, ResultKind.CLOSED):
            raise ValueError(f"{kind} is published by the worker, not an end of stream")
        return cls(session_id, None, None, None, {}, {}, kind=kind)


@dataclass
class SessionClosed:
    session_id: int


SessionReply = (SessionOpened, SessionAdmitted, SessionResultMessage, SessionClosed)


# --------------------------------------------------------------------------- worker state


class SessionState:
    """Executor-side bookkeeping for one open session on rank 0.

    Holds the queues, the live values, the counters and the terminal flag, and
    answers the advance condition. It does not touch slots: the caller passes the
    free-slot count in, because the pool is a separate object with its own lifetime.
    """

    def __init__(self, session_id: int, capabilities: SessionCapabilities) -> None:
        self.session_id = session_id
        self.capabilities = capabilities
        self.queues: Dict[str, Deque[Any]] = {s.name: deque() for s in capabilities.queued_inputs}
        self.admitted: Dict[str, int] = {s.name: 0 for s in capabilities.queued_inputs}
        self.live: Dict[str, Any] = {s.name: s.default for s in capabilities.live_inputs}
        self.next_advance = 0
        self.next_sequence = 0
        self.cursors: Dict[str, int] = {s.name: 0 for s in capabilities.streams}
        self.in_flight = False
        self.terminal: Optional[SessionResultMessage] = None
        self.closing = False
        self.initialized = False

    # ---- admission

    def admit(self, name: str, value: Any) -> int:
        """Validate and admit one value. Returns the advance index: for a queued input
        the advance that will consume the value; for a live input the next advance to
        be dispatched. Raises ``ValueError`` (size, then the validator),
        ``SessionQueueFull`` or ``SessionEnded``; a rejection changes nothing."""
        if self.terminal is not None or self.closing:
            raise SessionEnded(
                f"session {self.session_id}: "
                + ("closing" if self.closing else "a terminal result exists")
            )
        spec = self.capabilities.input(name)
        check_input_size(spec, value)
        if spec.validator is not None:
            spec.validator(value)
        if spec.policy == "live":
            self.live[name] = value
            return self.next_advance
        length = self.capabilities.length
        if length is not None and self.admitted[name] >= length:
            raise SessionEnded(
                f"input {name!r}: all {length} values of a finite session are already queued"
            )
        if len(self.queues[name]) >= spec.queue_depth:
            raise SessionQueueFull(
                f"input {name!r}: queue holds {spec.queue_depth} values; retry after the next advance"
            )
        index = self.admitted[name]
        self.queues[name].append(value)
        self.admitted[name] = index + 1
        return index

    # ---- advance

    def can_advance(self, free_slots: int) -> bool:
        if not self.initialized or self.closing or self.terminal is not None or self.in_flight:
            return False
        if free_slots <= 0:
            return False
        length = self.capabilities.length
        if length is not None and self.next_advance >= length:
            return False
        return all(self.queues[name] for name in self.queues)

    def exhausted(self) -> bool:
        """A finite session has run every advance; the terminal result follows."""
        length = self.capabilities.length
        return length is not None and self.next_advance >= length and not self.in_flight

    def take_inputs(self) -> Dict[str, Any]:
        """One value from every queue and the current value of every live input, as
        the dispatched advance sees them."""
        inputs = {name: queue.popleft() for name, queue in self.queues.items()}
        inputs.update(self.live)
        return inputs

    def dispatch(self) -> int:
        index = self.next_advance
        self.next_advance = index + 1
        self.in_flight = True
        return index

    # ---- results

    def media_result(
        self,
        advance_index: Optional[int],
        slot_id: int,
        written: Dict[str, Tuple[Tuple[int, ...], torch.dtype]],
    ) -> SessionResultMessage:
        """Build the message for what the slot recorded as written (stream name to
        shape and dtype). Every stream is checked against its declaration before
        anything moves: a rejected result publishes nothing and leaves every
        cursor where it was. Each stream's start is its cursor, which then moves
        by the extent along the stream axis."""
        from .slot_pool import check_extent

        caps = self.capabilities
        if not written:
            raise ValueError("a media result must carry at least one stream")
        checked: Dict[str, Tuple[Tuple[int, ...], str, int]] = {}
        for name, (shape, dtype) in written.items():
            spec = caps.stream(name)
            extents = check_extent(spec, shape, dtype)
            checked[name] = (extents, str(dtype), extents[spec.axis])
        streams: Dict[str, Tuple[Tuple[int, ...], str]] = {}
        position: Dict[str, Tuple[int, int]] = {}
        for name, (extents, dtype_name, count) in checked.items():
            streams[name] = (extents, dtype_name)
            start = self.cursors[name]
            position[name] = (start, count)
            self.cursors[name] = start + count
        message = SessionResultMessage(
            session_id=self.session_id,
            sequence=self.next_sequence,
            advance_index=advance_index,
            slot_id=slot_id,
            streams=streams,
            position=position,
        )
        self.next_sequence += 1
        return message

    def terminal_result(self, error: Optional[str] = None) -> SessionResultMessage:
        if self.terminal is not None:
            return self.terminal
        self.terminal = SessionResultMessage(
            session_id=self.session_id,
            sequence=self.next_sequence,
            advance_index=None,
            slot_id=None,
            streams={},
            position={},
            kind=ResultKind.TERMINAL,
            error=error,
        )
        self.next_sequence += 1
        return self.terminal

    def discard_inputs(self) -> None:
        for queue in self.queues.values():
            queue.clear()


def torch_dtype_from_str(name: str) -> torch.dtype:
    """``str(torch.uint8)`` is ``"torch.uint8"``; this is its inverse."""
    attr = name.split(".", 1)[1] if name.startswith("torch.") else name
    dtype = getattr(torch, attr, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"not a torch dtype: {name!r}")
    return dtype
