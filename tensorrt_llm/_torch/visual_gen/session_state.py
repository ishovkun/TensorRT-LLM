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
"""Session state owned by the executor's worker loop on rank 0: the input queues
and live values, the advance condition, the stream cursors and the terminal
result. Built from the pipeline's ``SessionCapabilities``; speaks to the client
only through the messages of ``session_protocol.py``.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import torch

from .pipeline import SessionCapabilities
from .session_protocol import ResultType, SessionEnded, SessionQueueFull, SessionResultMessage


class SessionState:
    """Executor-side bookkeeping for one open session on rank 0.

    Holds the queues, the live values, the counters and the terminal result, and
    answers the advance condition. The worker loop runs one advance at a time on
    one thread, so "in flight" and "initialized" are positions in that loop, not
    state kept here. It does not touch slots: the caller passes the
    free-slot count in, because the pool is a separate object with its own lifetime.
    """

    def __init__(self, session_id: int, capabilities: SessionCapabilities) -> None:
        self.session_id = session_id
        self.capabilities = capabilities
        # Per queue input: admitted values not yet consumed, oldest first.
        self.queue_values: Dict[str, Deque[Any]] = {
            s.name: deque() for s in capabilities.queue_inputs
        }
        # Per live input: the newest admitted value, the default until the first submit.
        self.live_values: Dict[str, Any] = {s.name: s.default for s in capabilities.live_inputs}
        # Advances dispatched so far; the next one gets this index.
        self.next_advance = 0
        # Results published so far, the terminal included; the next one gets this number.
        self.next_sequence = 0
        # Per stream: units (frames, samples) published so far along its axis.
        self.cursors: Dict[str, int] = {s.name: 0 for s in capabilities.streams}
        # The terminal result once the session is done or failed; built once.
        self.terminal: Optional[SessionResultMessage] = None
        # Set by the close command or the idle timeout; the loop ends at the next check.
        self.closing = False

    # ---- admission

    def admit(self, name: str, value: Any) -> int:
        """Validate and admit one value. Returns the advance index: for a queued input
        the advance that will consume the value; for a live input the next advance to
        be dispatched. Raises ``ValueError`` (the validator),
        ``SessionQueueFull`` or ``SessionEnded``; a rejection changes nothing."""
        if self.terminal is not None or self.closing:
            raise SessionEnded(
                f"session {self.session_id}: "
                + ("closing" if self.closing else "a terminal result exists")
            )
        spec = self.capabilities.input(name)
        if spec.validator is not None:
            spec.validator(value)
        if spec.policy == "live":
            self.live_values[name] = value
            return self.next_advance
        length = self.capabilities.length
        index = self.admitted_count(name)
        if length is not None and index >= length:
            raise SessionEnded(
                f"input {name!r}: all {length} values of a finite session are already queued"
            )
        if len(self.queue_values[name]) >= spec.queue_depth:
            raise SessionQueueFull(
                f"input {name!r}: queue holds {spec.queue_depth} values; retry after the next advance"
            )
        self.queue_values[name].append(value)
        return index

    def admitted_count(self, name: str) -> int:
        """Values of a queued input admitted so far: every advance consumed one, the
        rest are waiting. Also the advance index the next value gets."""
        return self.next_advance + len(self.queue_values[name])

    # ---- advance

    def can_advance(self, free_slots: int) -> bool:
        if self.closing or self.terminal is not None:
            return False
        if free_slots <= 0:
            return False
        length = self.capabilities.length
        if length is not None and self.next_advance >= length:
            return False
        return all(self.queue_values[name] for name in self.queue_values)

    def exhausted(self) -> bool:
        """A finite session has run every advance; the terminal result follows."""
        length = self.capabilities.length
        return length is not None and self.next_advance >= length

    def dispatch(self) -> Tuple[int, Dict[str, Any]]:
        """Start the next advance: its index, and its inputs, one value taken from
        every queue plus the current value of every live input."""
        index = self.next_advance
        self.next_advance = index + 1
        inputs = {name: queue.popleft() for name, queue in self.queue_values.items()}
        inputs.update(self.live_values)
        return index, inputs

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
        checked: Dict[str, Tuple[Tuple[int, ...], torch.dtype, int]] = {}
        for name, (shape, dtype) in written.items():
            spec = caps.stream(name)
            extents = check_extent(spec, shape, dtype)
            checked[name] = (extents, dtype, extents[spec.axis])
        streams: Dict[str, Tuple[Tuple[int, ...], torch.dtype]] = {}
        position: Dict[str, Tuple[int, int]] = {}
        for name, (extents, dtype, count) in checked.items():
            streams[name] = (extents, dtype)
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
            type=ResultType.TERMINAL,
            error=error,
        )
        self.next_sequence += 1
        return self.terminal

    def discard_inputs(self) -> None:
        for queue in self.queue_values.values():
            queue.clear()
