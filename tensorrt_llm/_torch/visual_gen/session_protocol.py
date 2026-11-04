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
"""The session protocol between the coordinator and rank 0.

A session is one generation whose model state stays alive on the worker while it
produces results one advance at a time. This module holds what both sides share:
the commands and replies that cross the sockets and the errors a caller can see.
The worker's own
bookkeeping is ``session_state.py``; the pipeline declares what it accepts and
produces through ``SessionCapabilities`` in ``pipeline.py`` and never sees any of
this.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest

# --------------------------------------------------------------------------- errors


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


def torch_dtype_from_str(name: str) -> torch.dtype:
    """``str(torch.uint8)`` is ``"torch.uint8"``; this is its inverse."""
    attr = name.split(".", 1)[1] if name.startswith("torch.") else name
    dtype = getattr(torch, attr, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"not a torch dtype: {name!r}")
    return dtype
