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
delivers :class:`SessionResult` objects through :meth:`VisualGenSession.results`
or, for an interactive pipeline, through :meth:`VisualGenSession.advance`. Each
result is a lease on worker-owned storage that expires when the next result is
requested, when the iterator closes, or when the session closes. Call
:meth:`SessionResult.copy` before then to keep media.

A session is driven from one thread or one asyncio task. The methods are not
reentrant: a second submit while one waits, or a second result request while one
waits, raises :class:`SessionError` instead of blocking.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
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
    "SessionResultsIterator",
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
        channel: Optional["SessionChannel"] = None,
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
        self._slot_id: Optional[int] = None
        # A view's storage belongs to the worker; once the session is closed or
        # the worker group has failed, the view is gone, whatever the lease said.
        # A result without storage (a terminal, a copy) has nothing to lose.
        self._channel = None if owned or not outputs else channel

    @property
    def outputs(self) -> Dict[str, torch.Tensor]:
        if self._expired:
            raise SessionLeaseExpired(
                f"result {self.sequence} of session {self.request_id} expired; "
                "call copy() before requesting the next result to keep its media"
            )
        if self._channel is not None and self._channel.lease_lost is not None:
            raise SessionLeaseExpired(
                f"result {self.sequence} of session {self.request_id} expired: "
                f"{self._channel.lease_lost}"
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


@set_api_status("prototype")
class SessionResultsIterator:
    """The session's results, as a sync or async iterator.

    Use ``for r in it`` or ``async for r in it``. Requesting the next result expires
    the previous one and returns its slot to the worker. The terminal result is
    yielded once; the request after it ends the iteration. Closing the iterator (or
    leaving its context) releases the held result without closing the session; a
    bare ``break`` does not.
    """

    def __init__(self, session: "VisualGenSession"):
        self._session = session
        self._closed = False

    def __iter__(self) -> "SessionResultsIterator":
        return self

    def __next__(self) -> SessionResult:
        if self._closed or self._session._ended:
            raise StopIteration
        return self._session._next_result()

    def __aiter__(self) -> "SessionResultsIterator":
        return self

    async def __anext__(self) -> SessionResult:
        if self._closed or self._session._ended:
            raise StopAsyncIteration
        return await self._session._next_result_async()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._session._release_held()

    def __enter__(self) -> "SessionResultsIterator":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    async def __aenter__(self) -> "SessionResultsIterator":
        return self

    async def __aexit__(self, *exc) -> bool:
        self.close()
        return False


@set_api_status("prototype")
class VisualGenSession:
    """One open session on a model instance.

    Inputs are submitted by name. A queued input supplies one value per advance
    and the worker waits for it; a live input is a state value the worker reads
    at each advance. Results arrive in order through :meth:`results`; an
    interactive pipeline also offers :meth:`advance`, which submits one value and
    returns that advance's result. The session holds one result at a time: asking
    for the next one expires the previous. ``close`` ends the session and is also
    the context manager's exit.
    """

    def __init__(self, request_id: int, client: "DiffusionRemoteClient", channel: "SessionChannel"):
        self.request_id = request_id
        self._client = client
        self._channel = channel
        self._capabilities: Dict[str, Any] = dict(channel.capabilities or {})
        # Busy flags, never waited on: a second caller while one is inside gets an error.
        self._submitting = threading.Lock()
        self._receiving = threading.Lock()
        self._held: Optional[Tuple[SessionResult, Optional[int]]] = None
        self._closed = False
        self._ended = False
        self._queued_names = {
            spec["name"]
            for spec in self._capabilities.get("inputs", [])
            if spec["policy"] == "queued"
        }

    # ---- introspection

    @property
    def capabilities(self) -> Dict[str, Any]:
        """What the pipeline declared for this session.

        ``inputs`` (name, policy, default, queue_depth), ``streams`` (name, kind,
        dtype, max_shape, axis, rate, placement), ``initial_result``, ``length``,
        ``lockstep``.
        """
        return self._capabilities

    @property
    def closed(self) -> bool:
        """Whether the session is over.

        Closed by ``close()``, closed by the worker (idle timeout), or lost with
        the worker group.
        """
        return self._closed or self._channel.closed.done()

    # ---- inputs

    def submit(self, name: str, value: Any) -> int:
        """Admit one value and return an advance index.

        For a queued input the index is the advance that will consume the value;
        for a live input it is the next advance not yet dispatched when the value
        was admitted, so every advance from that index on reads this value or a
        later one for the same input. Raises ``ValueError`` (invalid input),
        ``SessionQueueFull`` (retry after the next advance) or ``SessionEnded``.
        """
        fut = self._begin_submit(name, value)
        try:
            return fut.result()
        finally:
            self._submitting.release()

    async def submit_async(self, name: str, value: Any) -> int:
        fut = self._begin_submit(name, value)
        try:
            return await asyncio.wrap_future(fut)
        finally:
            self._submitting.release()

    def _begin_submit(self, name: str, value: Any, nothing_owed: bool = False) -> Future:
        self._check_live()
        if not self._submitting.acquire(blocking=False):
            raise SessionError("a submit is already in progress on this session")
        try:
            return self._client.session_submit(
                self._channel, name, value, nothing_owed=nothing_owed
            )
        except BaseException:
            self._submitting.release()
            raise

    # ---- results

    def results(self) -> SessionResultsIterator:
        """An iterator over the results not yet requested, in order."""
        self._check_live()
        return SessionResultsIterator(self)

    def advance(self, value: Any) -> SessionResult:
        """Submit ``value`` to the queued input and return that advance's result.

        Interactive sessions only: the pipeline declared one queued input and one
        result per advance. The result request is reserved for the whole call,
        admission included, so no other result request runs meanwhile. Refuses
        while a declared initial result or the media of an earlier queued value
        has not been taken, even if it has not arrived yet. If the session has
        already ended, returns the terminal result once without submitting; if
        it ends before the submitted advance produces media, returns the
        terminal result instead. Later calls raise ``SessionEnded``.
        """
        name = self._begin_advance()
        try:
            index = self._submit_for_advance(name, value) if name is not None else None
            fut = self._channel.request_next_result()
        except BaseException:
            self._receiving.release()
            raise
        try:
            message = fut.result()
        except BaseException:
            self._channel.abandon(fut)
            self._receiving.release()
            raise
        return self._end_advance(message, index)

    async def advance_async(self, value: Any) -> SessionResult:
        name = self._begin_advance()
        try:
            index = await self._submit_for_advance_async(name, value) if name is not None else None
            fut = self._channel.request_next_result()
        except BaseException:
            self._receiving.release()
            raise
        try:
            message = await asyncio.wrap_future(fut)
        except BaseException:
            self._channel.abandon(fut)
            self._receiving.release()
            raise
        return self._end_advance(message, index)

    def _begin_advance(self) -> Optional[str]:
        """Reserve the result request for one advance.

        Checks that nothing older is owed and expires the previous lease. Returns
        the queued input's name, or ``None`` when the next result is already a
        terminal: then nothing is submitted and the terminal is what comes back.
        """
        self._check_live()
        if not self._capabilities.get("lockstep"):
            raise SessionError(
                "advance() needs an interactive pipeline; use submit() and results()"
            )
        if self._ended:
            raise SessionEnded(f"session {self.request_id} has ended")
        if not self._receiving.acquire(blocking=False):
            raise SessionError("a result request is already in progress on this session")
        try:
            # A session that is gone answers with why, not with what it still owed.
            if self._channel.failure is not None:
                raise self._channel.failure
            if self._channel.terminal_next():
                self._release_held()
                return None
            if self._channel.closed_reason is not None:
                raise SessionEnded(self._channel.closed_reason)
            owed = self._channel.owed_media()
            if owed is not None:
                raise SessionError(f"{owed}; take it through results() before calling advance()")
            self._release_held()
            return next(iter(self._queued_names))
        except BaseException:
            self._receiving.release()
            raise

    def _submit_for_advance(self, name: str, value: Any) -> Optional[int]:
        """Submit for ``advance``.

        Returns the admission index, or ``None`` when the worker reports the
        session ended: the next result is then the terminal and nothing was
        submitted.
        """
        try:
            fut = self._begin_submit(name, value, nothing_owed=True)
        except SessionEnded:
            return None  # the session ended before the submit was even sent
        try:
            return fut.result()
        except SessionEnded:
            return None
        finally:
            self._submitting.release()

    async def _submit_for_advance_async(self, name: str, value: Any) -> Optional[int]:
        try:
            fut = self._begin_submit(name, value, nothing_owed=True)
        except SessionEnded:
            return None
        try:
            return await asyncio.wrap_future(fut)
        except SessionEnded:
            return None
        finally:
            self._submitting.release()

    def _end_advance(self, message: SessionResultMessage, index: Optional[int]) -> SessionResult:
        result = self._end_receive(message)
        if not result.done and result.advance_index != index:
            raise SessionError(
                f"advance() admitted index {index} but the next result is advance "
                f"{result.advance_index}; the worker broke result ordering"
            )
        return result

    def _next_result(self) -> SessionResult:
        fut = self._begin_receive()
        try:
            message = fut.result()
        except BaseException:
            self._channel.abandon(fut)
            self._receiving.release()
            raise
        return self._end_receive(message)

    async def _next_result_async(self) -> SessionResult:
        fut = self._begin_receive()
        try:
            message = await asyncio.wrap_future(fut)
        except BaseException:
            self._channel.abandon(fut)
            self._receiving.release()
            raise
        return self._end_receive(message)

    def _begin_receive(self) -> Future:
        self._check_live()
        if self._ended:
            raise SessionEnded(f"session {self.request_id} has ended")
        if not self._receiving.acquire(blocking=False):
            raise SessionError("a result request is already in progress on this session")
        try:
            if not self._release_held():
                # No slot goes back with this request, so nothing else tells the
                # worker the client is active.
                self._client.session_activity(self._channel)
            return self._channel.request_next_result()
        except BaseException:
            self._receiving.release()
            raise

    def _end_receive(self, message: SessionResultMessage) -> SessionResult:
        try:
            result = self._channel.settle(message, self._build_result)
            self._held = (result, result._slot_id)
            if result.done:
                self._ended = True
            return result
        finally:
            self._receiving.release()

    # ---- close

    def close(self) -> None:
        """Cancel the session.

        Queued inputs and undelivered results are discarded, in-flight work runs to
        its boundary, state is released. Idempotent.
        """
        if self._closed:
            return
        self._closed = True
        self._release_held()
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
        self._release_held()
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

    # ---- internals

    def _check_live(self) -> None:
        if self._closed:
            raise SessionError(f"session {self.request_id} is closed")

    def _release_held(self) -> bool:
        """Expire the held result and give its slot back to the worker.

        Returns whether a slot was returned.
        """
        if self._held is None:
            return False
        result, slot_id = self._held
        self._held = None
        result._expire()
        if slot_id is None or self.closed:
            return False
        self._client.session_return_slot(self._channel, slot_id)
        return True

    def _build_result(self, message: SessionResultMessage) -> SessionResult:
        outputs: Dict[str, torch.Tensor] = {}
        if message.slot_id is not None and self._channel.delivery_slots is not None:
            for name, (shape, dtype_name) in message.streams.items():
                outputs[name] = self._channel.delivery_slots.view(
                    message.slot_id, name, tuple(shape), torch_dtype_from_str(dtype_name)
                )
        result = SessionResult(
            self.request_id,
            message.sequence,
            message.advance_index,
            outputs,
            message.position,
            message.done,
            message.error,
            owned=False,
            channel=self._channel,
        )
        result._slot_id = message.slot_id
        return result
