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
"""Streaming sessions for VisualGen.

A session is a generation whose model state stays alive on the worker while it
delivers results one advance at a time and takes input while it runs.

Open one with :meth:`tensorrt_llm.visual_gen.VisualGen.open_session`. The session
delivers :class:`SessionResult` objects through :meth:`VisualGenSession.results`;
each result is a lease on worker-owned storage that expires when the next result
is requested, when the iterator closes, or when the session closes. Call
:meth:`SessionResult.copy` before then to keep media.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

from tensorrt_llm._torch.visual_gen.session import (
    SessionEnded,
    SessionError,
    SessionLeaseExpired,
    SessionQueueFull,
    SessionResultMessage,
    torch_dtype_from_str,
)
from tensorrt_llm.llmapi.utils import set_api_status

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.executor import DiffusionRemoteClient, SessionChannel

__all__ = [
    "SessionEnded",
    "SessionError",
    "SessionLeaseExpired",
    "SessionQueueFull",
    "SessionResult",
    "SessionResults",
    "VisualGenSession",
]


@set_api_status("prototype")
class SessionResult:
    """One result of a session.

    ``outputs`` maps each stream the pipeline produced to a tensor that views the
    worker's delivery slot; it is valid until the next result is requested. A
    terminal result (``done=True``) carries no outputs and ``error`` is ``None`` for
    normal completion or a description. ``advance_index`` is ``None`` for an initial
    result and for the terminal result.
    """

    def __init__(
        self,
        request_id: int,
        sequence: int,
        advance_index: Optional[int],
        outputs: Dict[str, torch.Tensor],
        position: Dict[str, Tuple[int, int]],
        done: bool,
        error: Optional[str],
        *,
        owned: bool,
    ):
        self.request_id = request_id
        self.sequence = sequence
        self.advance_index = advance_index
        self.position = dict(position)
        self.done = done
        self.error = error
        self._outputs = dict(outputs)
        self._owned = owned
        self._expired = False

    @property
    def outputs(self) -> Dict[str, torch.Tensor]:
        if self._expired:
            raise SessionLeaseExpired(
                f"result {self.sequence} of session {self.request_id} expired; "
                "call copy() before requesting the next result to keep its media"
            )
        return self._outputs

    @property
    def video(self) -> Optional[torch.Tensor]:
        """``outputs["video"]`` when that stream exists."""
        return self.outputs.get("video")

    def copy(self) -> "SessionResult":
        """Return this result with its media in memory the caller owns.

        Does not expire the view it was copied from.
        """
        return SessionResult(
            self.request_id,
            self.sequence,
            self.advance_index,
            {name: tensor.clone() for name, tensor in self.outputs.items()},
            self.position,
            self.done,
            self.error,
            owned=True,
        )

    def _expire(self) -> None:
        if not self._owned:
            self._expired = True
            self._outputs = {}

    def __repr__(self) -> str:
        streams = {name: tuple(t.shape) for name, t in self._outputs.items()}
        return (
            f"SessionResult(sequence={self.sequence}, advance_index={self.advance_index}, "
            f"streams={streams}, position={self.position}, done={self.done}, error={self.error!r})"
        )


class _Consumer:
    """The one reader of a session's results.

    Holds the current lease and returns the slot when the next result is requested.
    """

    def __init__(self, session: "VisualGenSession"):
        self._session = session
        self._held: Optional[Tuple[SessionResult, Optional[int]]] = None
        self._finished = False

    def release_held(self) -> None:
        if self._held is None:
            return
        result, slot_id = self._held
        self._held = None
        result._expire()
        if slot_id is not None:
            self._session._return_slot(slot_id)

    def _take(self, message: SessionResultMessage) -> SessionResult:
        result = self._session._build_result(message)
        self._held = (result, message.slot_id)
        if message.done:
            self._finished = True
            self._session._mark_ended()
        return result

    def next(self) -> SessionResult:
        if self._finished:
            raise StopIteration
        self.release_held()
        return self._take(self._session._channel.results.get())

    async def anext(self) -> SessionResult:
        if self._finished:
            raise StopAsyncIteration
        self.release_held()
        loop = asyncio.get_running_loop()
        message = await loop.run_in_executor(None, self._session._channel.results.get)
        return self._take(message)


@set_api_status("prototype")
class SessionResults:
    """The results iterator of a session, sync or async.

    Use ``for r in results`` or ``async for r in results``. Requesting the next
    result expires the previous
    one and returns its slot to the worker. The terminal result is yielded once;
    the request after it ends the iteration. Closing the iterator (or leaving its
    context) returns the held slot and releases the consumer role without closing
    the session; a bare ``break`` does not.
    """

    def __init__(self, session: "VisualGenSession"):
        self._session = session
        self._consumer = _Consumer(session)
        self._closed = False

    def __iter__(self) -> "SessionResults":
        return self

    def __next__(self) -> SessionResult:
        self._check_open()
        return self._consumer.next()

    def __aiter__(self) -> "SessionResults":
        return self

    async def __anext__(self) -> SessionResult:
        self._check_open()
        return await self._consumer.anext()

    def _check_open(self) -> None:
        if self._closed:
            raise SessionError("results iterator is closed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._consumer.release_held()
        self._session._release_consumer(self)

    def __enter__(self) -> "SessionResults":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    async def __aenter__(self) -> "SessionResults":
        return self

    async def __aexit__(self, *exc) -> bool:
        self.close()
        return False


@set_api_status("prototype")
class VisualGenSession:
    """One open session on a model instance.

    Inputs are submitted by name. A queued input supplies one value per advance
    and the worker waits for it; a live input is a state value the worker reads
    at each advance. Results are consumed through :meth:`results` or, for a
    lockstep pipeline, one at a time through :meth:`advance`. ``close`` ends the
    session and is also the context manager's exit.
    """

    def __init__(self, request_id: int, client: "DiffusionRemoteClient", channel: "SessionChannel"):
        self.request_id = request_id
        self._client = client
        self._channel = channel
        self._capabilities: Dict[str, Any] = dict(channel.capabilities or {})
        self._streams: Dict[str, Dict[str, Any]] = {
            s["name"]: s for s in self._capabilities["streams"]
        }
        self._consumer: Optional[SessionResults] = None
        self._lockstep: Optional[_Consumer] = None
        self._lock = threading.Lock()
        self._closed = False
        self._ended = False

    # ---- introspection

    @property
    def capabilities(self) -> Dict[str, Any]:
        """What the pipeline declared for this session.

        ``inputs`` (name, policy, default, queue_depth, max_bytes), ``streams`` (name, kind, dtype,
        max_shape, axis, rate, placement), ``initial_result``, ``length``,
        ``lockstep``.
        """
        return self._capabilities

    @property
    def closed(self) -> bool:
        return self._closed

    # ---- inputs

    def submit(self, name: str, value: Any) -> int:
        """Admit one value and return an advance index.

        For a queued input the index is the advance that will consume the value; for
        a live input it is the next advance not yet
        dispatched when the value was admitted, so every advance from that index on
        reads this value or a later one for the same input. Raises ``ValueError``
        (invalid input), ``SessionQueueFull`` (retry after the next advance) or
        ``SessionEnded``.
        """
        self._check_live()
        return self._client.session_submit(self._channel, name, value).result()

    async def submit_async(self, name: str, value: Any) -> int:
        self._check_live()
        return await asyncio.wrap_future(self._client.session_submit(self._channel, name, value))

    # ---- results

    def results(self) -> SessionResults:
        """Take the consumer role and return the results iterator.

        Raises if an iterator is open or an ``advance`` call has a result outstanding.
        """
        self._check_live()
        with self._lock:
            if self._consumer is not None:
                raise SessionError("a results iterator is already open on this session")
            if self._lockstep is not None:
                raise SessionError("advance() holds the consumer role; results() is unavailable")
            self._consumer = SessionResults(self)
            return self._consumer

    def advance(self, value: Any) -> SessionResult:
        """Submit ``value`` to the queued input and return that advance's media result.

        Lockstep pipelines only. If the session has already
        ended, returns the terminal result once without submitting; if it ends
        before the submitted advance produces media, returns the terminal result
        instead. Later calls raise ``SessionEnded``.
        """
        consumer = self._lockstep_consumer()
        if consumer._finished:
            raise SessionEnded(f"session {self.request_id} has ended")
        if self._unread_media_pending():
            raise SessionError(
                "unread results exist; read them (a declared initial result through "
                "results(), then close that iterator) before calling advance()"
            )
        name = self._queued_input_name()
        try:
            self.submit(name, value)
        except SessionEnded:
            pass
        return self._lockstep_next(consumer)

    async def advance_async(self, value: Any) -> SessionResult:
        consumer = self._lockstep_consumer()
        if consumer._finished:
            raise SessionEnded(f"session {self.request_id} has ended")
        if self._unread_media_pending():
            raise SessionError(
                "unread results exist; read them (a declared initial result through "
                "results(), then close that iterator) before calling advance_async()"
            )
        name = self._queued_input_name()
        try:
            await self.submit_async(name, value)
        except SessionEnded:
            pass
        consumer.release_held()
        loop = asyncio.get_running_loop()
        message = await loop.run_in_executor(None, self._channel.results.get)
        return consumer._take(message)

    def _unread_media_pending(self) -> bool:
        """Whether a media result is waiting that ``advance`` would skip over.

        A waiting terminal result is fine, ``advance`` returns it.
        """
        results = self._channel.results
        with results.mutex:
            return any(not message.done for message in results.queue)

    def _lockstep_next(self, consumer: _Consumer) -> SessionResult:
        consumer.release_held()
        return consumer._take(self._channel.results.get())

    def _lockstep_consumer(self) -> _Consumer:
        self._check_live()
        if not self._capabilities.get("lockstep"):
            raise SessionError("advance() needs a lockstep pipeline; use submit() and results()")
        with self._lock:
            if self._consumer is not None:
                raise SessionError("a results iterator is open; close it before calling advance()")
            if self._lockstep is None:
                self._lockstep = _Consumer(self)
            return self._lockstep

    def _queued_input_name(self) -> str:
        for spec in self._capabilities["inputs"]:
            if spec["policy"] == "queued":
                return spec["name"]
        raise SessionError("lockstep session without a queued input")

    # ---- close

    def close(self) -> None:
        """Cancel the session.

        Queued inputs and undelivered results are discarded, in-flight work runs to
        its boundary, state is released. Idempotent.
        """
        if self._closed:
            return
        self._closed = True
        self._expire_all()
        fut = self._client.session_close(self._channel)
        try:
            fut.result()
        except RuntimeError:
            # The worker is gone; the client-side state is released regardless.
            pass

    async def close_async(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._expire_all()
        try:
            await asyncio.wrap_future(self._client.session_close(self._channel))
        except RuntimeError:
            pass

    def __enter__(self) -> "VisualGenSession":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    async def __aenter__(self) -> "VisualGenSession":
        return self

    async def __aexit__(self, *exc) -> bool:
        await self.close_async()
        return False

    # ---- internals used by the consumers

    def _check_live(self) -> None:
        if self._closed:
            raise SessionError(f"session {self.request_id} is closed")

    def _mark_ended(self) -> None:
        self._ended = True

    def _release_consumer(self, consumer: SessionResults) -> None:
        with self._lock:
            if self._consumer is consumer:
                self._consumer = None

    def _return_slot(self, slot_id: int) -> None:
        if not self._closed:
            self._client.session_return_slot(self._channel, slot_id)

    def _expire_all(self) -> None:
        with self._lock:
            consumer = self._consumer
            lockstep = self._lockstep
            self._consumer = None
        if consumer is not None:
            consumer._consumer.release_held()
            consumer._closed = True
        if lockstep is not None:
            lockstep.release_held()

    def _build_result(self, message: SessionResultMessage) -> SessionResult:
        outputs: Dict[str, torch.Tensor] = {}
        if message.slot_id is not None and self._channel.slots is not None:
            for name, (shape, dtype_name, nbytes) in message.streams.items():
                outputs[name] = self._channel.slots.view(
                    message.slot_id, name, tuple(shape), torch_dtype_from_str(dtype_name), nbytes
                )
        return SessionResult(
            self.request_id,
            message.sequence,
            message.advance_index,
            outputs,
            message.position,
            message.done,
            message.error,
            owned=False,
        )
