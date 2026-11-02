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
"""Session state shared by the executor's worker loop and its client channel.

A session is one generation whose model state stays alive on the worker while it
produces results one advance at a time. This module holds what the executor owns:
the pipeline's declared capabilities, the input queues and live values, the advance
condition, the stream cursors and the messages that travel between the coordinator
and rank 0. The pipeline owns its caches and never sees any of it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest


class InputPolicy(str, Enum):
    """How an input reaches an advance: ``QUEUED`` consumes one value per advance
    and waits when none is there; ``CURRENT`` reads the newest value and never waits."""

    QUEUED = "queued"
    CURRENT = "current"


class Placement(str, Enum):
    """Where a stream is written: shared page-locked host memory or device memory."""

    HOST = "host"
    DEVICE = "device"


QUEUED = InputPolicy.QUEUED
CURRENT = InputPolicy.CURRENT
HOST = Placement.HOST
DEVICE = Placement.DEVICE

DEFAULT_QUEUE_DEPTH = 8


class SessionError(RuntimeError):
    """Base class for session failures raised to the client."""


class SessionQueueFull(SessionError):
    """A queued input has no room; retry after the next advance."""


class SessionEnded(SessionError):
    """The session has a terminal result or is closing; no more input is admitted."""


class SessionLeaseExpired(SessionError):
    """A result's storage was returned to the worker; the view is no longer valid."""


# --------------------------------------------------------------------------- capabilities


@dataclass(frozen=True)
class InputSpec:
    """One named input the pipeline accepts during a session.

    ``validate`` runs on rank 0 before admission and raises ``ValueError`` for a
    bad value. ``queue_depth`` bounds a queued input.
    """

    name: str
    policy: InputPolicy = InputPolicy.QUEUED
    validate: Optional[Callable[[Any], None]] = None
    default: Any = None
    queue_depth: int = DEFAULT_QUEUE_DEPTH

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", InputPolicy(self.policy))
        if self.queue_depth < 1:
            raise ValueError(f"input {self.name!r}: queue_depth must be at least 1")


@dataclass(frozen=True)
class StreamSpec:
    """One named output stream the pipeline produces.

    ``max_shape`` is the largest tensor one result may carry for this stream; the
    slot pool sizes its buffers from it. ``axis`` is the dimension counted by
    ``position`` (frames, samples or predictions).
    """

    name: str
    kind: str
    dtype: torch.dtype
    max_shape: Tuple[int, ...]
    axis: int = 1
    rate: Optional[float] = None
    placement: Placement = Placement.HOST

    def __post_init__(self) -> None:
        object.__setattr__(self, "placement", Placement(self.placement))
        if not self.max_shape or any(int(d) <= 0 for d in self.max_shape):
            raise ValueError(f"stream {self.name!r}: max_shape must be non-empty and positive")
        if not 0 <= self.axis < len(self.max_shape):
            raise ValueError(f"stream {self.name!r}: axis {self.axis} outside max_shape")

    @property
    def max_bytes(self) -> int:
        return math.prod(int(d) for d in self.max_shape) * self.dtype.itemsize


@dataclass(frozen=True)
class SessionCapabilities:
    """What a pipeline declares for one session request.

    ``length`` is the number of advances for a finite session or ``None`` when the
    session is open-ended. ``lockstep`` is true when exactly one queued input exists
    and every advance yields one media result, which enables ``advance(value)`` on
    the client.
    """

    inputs: Tuple[InputSpec, ...]
    streams: Tuple[StreamSpec, ...]
    initial_result: bool = False
    length: Optional[int] = None
    lockstep: bool = False

    def __post_init__(self) -> None:
        names = [s.name for s in self.inputs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate input names: {names}")
        stream_names = [s.name for s in self.streams]
        if len(set(stream_names)) != len(stream_names):
            raise ValueError(f"duplicate stream names: {stream_names}")
        if not self.streams:
            raise ValueError("a session must declare at least one stream")
        if self.lockstep and len(self.queued_inputs) != 1:
            raise ValueError("a lockstep session has exactly one queued input")

    @property
    def queued_inputs(self) -> Tuple[InputSpec, ...]:
        return tuple(s for s in self.inputs if s.policy == QUEUED)

    @property
    def live_inputs(self) -> Tuple[InputSpec, ...]:
        return tuple(s for s in self.inputs if s.policy == CURRENT)

    @property
    def max_result_bytes(self) -> int:
        return sum(s.max_bytes for s in self.streams)

    def input(self, name: str) -> InputSpec:
        for spec in self.inputs:
            if spec.name == name:
                return spec
        raise ValueError(f"unknown input {name!r}; declared: {[s.name for s in self.inputs]}")

    def stream(self, name: str) -> StreamSpec:
        for spec in self.streams:
            if spec.name == name:
                return spec
        raise ValueError(f"unknown stream {name!r}; declared: {[s.name for s in self.streams]}")

    def public(self) -> Dict[str, Any]:
        """The picklable, callable-free form sent to the client."""
        return {
            "inputs": [
                {
                    "name": s.name,
                    "policy": s.policy.value,
                    "default": s.default,
                    "queue_depth": s.queue_depth,
                }
                for s in self.inputs
            ],
            "streams": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "dtype": str(s.dtype),
                    "max_shape": tuple(s.max_shape),
                    "axis": s.axis,
                    "rate": s.rate,
                    "placement": s.placement.value,
                }
                for s in self.streams
            ],
            "initial_result": self.initial_result,
            "length": self.length,
            "lockstep": self.lockstep,
        }


@dataclass
class AdvanceOutcome:
    """What ``pipeline.advance`` (and ``open_session``) hands back to the executor.

    ``event`` is a CUDA event recorded after every write into the slot, or ``None``
    when all writes have completed on return. ``done`` ends the session after this
    result. What was written is recorded by the slot itself.
    """

    event: Optional[torch.cuda.Event] = None
    done: bool = False


# --------------------------------------------------------------------------- wire messages


@dataclass
class SessionOpen:
    request: "DiffusionRequest"


@dataclass
class SessionSubmit:
    session_id: int
    seq: int
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
class SessionOpened:
    session_id: int
    capabilities: Optional[Dict[str, Any]] = None
    slots: Optional[List[Dict[str, Dict[str, Any]]]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class SessionAdmitted:
    session_id: int
    seq: int
    advance_index: Optional[int] = None
    error: Optional[str] = None
    error_kind: Optional[str] = None  # "invalid" | "queue_full" | "ended"


@dataclass
class SessionResultMessage:
    """One result. ``streams`` maps a stream name to the ``(shape, dtype)`` written
    in ``slots[slot_id][name]``; ``position`` maps it to ``(start, count)``. The
    terminal result has ``slot_id=None`` and no streams."""

    session_id: int
    sequence: int
    advance_index: Optional[int]
    slot_id: Optional[int]
    streams: Dict[str, Tuple[Tuple[int, ...], str]]
    position: Dict[str, Tuple[int, int]]
    done: bool = False
    error: Optional[str] = None


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
        be dispatched. Raises ``ValueError``, ``SessionQueueFull`` or ``SessionEnded``."""
        if self.terminal is not None or self.closing:
            raise SessionEnded(
                f"session {self.session_id}: "
                + ("closing" if self.closing else "a terminal result exists")
            )
        spec = self.capabilities.input(name)
        if spec.validate is not None:
            spec.validate(value)
        if spec.policy is InputPolicy.CURRENT:
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
        shape and dtype), checking each stream against its declaration; assigns each
        stream's start from its cursor and moves the cursor by the shape's extent
        along the stream axis."""
        caps = self.capabilities
        if not written:
            raise ValueError("a media result must carry at least one stream")
        streams: Dict[str, Tuple[Tuple[int, ...], str]] = {}
        position: Dict[str, Tuple[int, int]] = {}
        for name, (shape, dtype) in written.items():
            spec = caps.stream(name)
            if dtype != spec.dtype:
                raise ValueError(f"stream {name!r}: wrote {dtype}, declared {spec.dtype}")
            if len(shape) != len(spec.max_shape) or any(
                int(a) > int(b) for a, b in zip(shape, spec.max_shape)
            ):
                raise ValueError(f"stream {name!r}: shape {tuple(shape)} exceeds {spec.max_shape}")
            count = int(shape[spec.axis])
            streams[name] = (tuple(int(d) for d in shape), str(dtype))
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
            done=True,
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
