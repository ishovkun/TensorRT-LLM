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
"""The session protocol end to end in one process: a rank-0 ``DiffusionExecutor``
over fake sockets, a ``DiffusionRemoteClient`` fed by them, the public
``VisualGenSession`` on top, and a synthetic pipeline that counts advances. No GPU,
no worker processes, no model."""

import asyncio
import queue
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import pytest
import torch

from tensorrt_llm._torch.visual_gen.executor import (
    DiffusionExecutor,
    DiffusionRemoteClient,
    DiffusionRequest,
)
from tensorrt_llm._torch.visual_gen.session import (
    CURRENT,
    QUEUED,
    AdvanceOutcome,
    InputSpec,
    SessionCapabilities,
    SessionEnded,
    SessionError,
    SessionLeaseExpired,
    SessionOpen,
    SessionQueueFull,
    StreamSpec,
)
from tensorrt_llm.visual_gen.session import VisualGenSession

pytestmark = pytest.mark.cpu_only

FRAMES, H, W = 4, 2, 3


class CountingPipeline:
    """Lockstep when ``queued`` is set: one queued input ``controls`` and a live
    input ``camera``; each advance writes a frame block whose pixels encode the
    advance index, the control value and the camera value. With ``queued`` unset it
    is autonomous and ends after ``length`` advances."""

    supports_sessions = True
    default_generation_params: Dict[str, Any] = {}
    extra_param_specs: Dict[str, Any] = {}

    def __init__(self, *, queued=True, initial=True, length=None, fail_at=None, open_fails=False):
        self.queued = queued
        self.initial = initial
        self.length = length
        self.fail_at = fail_at
        self.open_fails = open_fails
        self.opened = 0
        self.closed = 0
        self.advances = []

    def prepare_request(self, req):
        pass

    def classify_request_failure(self, exc):
        return "client" if isinstance(exc, ValueError) else None

    def session_capabilities(self, req):
        inputs = []
        if self.queued:
            inputs.append(InputSpec("controls", QUEUED, validate=_non_negative, queue_depth=2))
        inputs.append(InputSpec("camera", CURRENT, default=0))
        return SessionCapabilities(
            inputs=tuple(inputs),
            streams=(StreamSpec("video", "video", torch.uint8, (1, FRAMES, H, W, 3), axis=1),),
            initial_result=self.initial,
            length=self.length,
            lockstep=self.queued,
        )

    def open_session(self, req, initial_slot):
        if self.open_fails:
            raise ValueError("bad prompt")
        self.opened += 1
        state = SimpleNamespace(prompt=req.prompt[0], count=0)
        outcome = AdvanceOutcome()
        if initial_slot is not None:
            view = initial_slot.buffer("video", (1, 1, H, W, 3), torch.uint8)
            view.fill_(255)
            outcome.written = {"video": 1}
        return state, outcome

    def validate_input(self, state, name, value):
        pass

    def advance(self, state, inputs, slot):
        index = state.count
        state.count += 1
        if self.fail_at is not None and index == self.fail_at:
            raise RuntimeError("kernel exploded")
        self.advances.append(dict(inputs))
        control = inputs.get("controls", 0)
        camera = inputs.get("camera", 0)
        if slot is not None:
            view = slot.buffer("video", (1, FRAMES, H, W, 3), torch.uint8)
            view[..., 0] = index
            view[..., 1] = control
            view[..., 2] = camera
        done = self.length is not None and state.count >= self.length
        return AdvanceOutcome(written={"video": FRAMES}, done=done)

    def close_session(self, state):
        self.closed += 1


def _non_negative(value):
    if not isinstance(value, int) or value < 0:
        raise ValueError("controls must be a non-negative int")


class FakeRequestSocket:
    """What the worker sees as ``requests_ipc``: ``get(timeout)`` and ``drain()``."""

    def __init__(self):
        self.q: "queue.Queue[Any]" = queue.Queue()

    def get(self, timeout: Optional[float] = None):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            raise queue.Empty from None

    def drain(self):
        items = []
        while True:
            try:
                items.append(self.q.get_nowait())
            except queue.Empty:
                return items


def _make_executor(pipeline, session_config=None):
    ex = DiffusionExecutor.__new__(DiffusionExecutor)
    ex.pipeline = pipeline
    ex.rank = 0
    ex.world_size = 1
    ex.device_id = 0
    ex.in_client_process = True  # same process as the client: local handles
    ex.response_queue = queue.Queue()
    ex.requests_ipc = FakeRequestSocket()
    ex.session_config = session_config
    ex._deferred = []
    ex._shutdown_requested = False
    return ex


def _make_client():
    client = DiffusionRemoteClient.__new__(DiffusionRemoteClient)
    client._sessions = {}
    client._sessions_lock = threading.Lock()
    client.pending_requests = queue.Queue()
    client._worker_failure = None
    return client


class Harness:
    """Wires the client's outgoing queue to the executor's fake socket and the
    executor's responses to the client's router, each on its own thread, and runs
    the executor's session loop on a third."""

    def __init__(self, pipeline, session_config=None):
        self.pipeline = pipeline
        self.executor = _make_executor(pipeline, session_config)
        self.client = _make_client()
        self._stop = threading.Event()
        self.worker_error: Optional[BaseException] = None
        self.threads = [
            threading.Thread(target=self._forward_commands, daemon=True),
            threading.Thread(target=self._forward_replies, daemon=True),
            threading.Thread(target=self._worker, daemon=True),
        ]
        for t in self.threads:
            t.start()

    def _forward_commands(self):
        while not self._stop.is_set():
            try:
                item = self.client.pending_requests.get(timeout=0.02)
            except queue.Empty:
                continue
            self.executor.requests_ipc.q.put(item)

    def _forward_replies(self):
        while not self._stop.is_set():
            try:
                item = self.executor.response_queue.get(timeout=0.02)
            except queue.Empty:
                continue
            self.client._route_session_reply(item)

    def _worker(self):
        try:
            # serve_forever, reduced to the session path: one open, served to its close.
            while not self._stop.is_set():
                try:
                    message = self.executor.requests_ipc.get(timeout=0.02)
                except queue.Empty:
                    continue
                if message is None:
                    return
                if isinstance(message, SessionOpen):
                    self.executor._run_session(message.request)
                    if self.executor._shutdown_requested:
                        return
                else:
                    self.executor._answer_stray_command(message)
        except BaseException as e:  # surfaced by the test, not swallowed
            self.worker_error = e

    def open(self, prompt="p", timeout=10.0) -> VisualGenSession:
        req = DiffusionRequest(request_id=1, prompt=[prompt], params=SimpleNamespace())
        channel = self.client.open_session(req)
        channel.opened.result(timeout=timeout)
        return VisualGenSession(req.request_id, self.client, channel)

    def stop(self):
        self._stop.set()
        for t in self.threads:
            t.join(timeout=5.0)
        if self.worker_error is not None:
            raise self.worker_error

    def __enter__(self) -> "Harness":
        return self

    def __exit__(self, *exc) -> bool:
        # Threads end inside the test body: the thread-leak check runs before
        # fixture teardown.
        self.stop()
        return False


@pytest.fixture
def harness():
    return Harness


def _pixels(result) -> Tuple[int, int, int]:
    video = result.outputs["video"]
    return tuple(int(x) for x in video[0, 0, 0, 0])


# --------------------------------------------------------------------------- lockstep


def test_lockstep_session_delivers_one_result_per_advance(harness):
    pipe = CountingPipeline()
    with harness(pipe) as h:
        with h.open() as s:
            caps = s.capabilities
            assert caps["lockstep"] and caps["initial_result"] and caps["length"] is None
            with s.results() as results:
                first = next(results)  # the declared initial result
                assert first.advance_index is None and first.sequence == 0
                assert first.outputs["video"].shape == (1, 1, H, W, 3)
                assert int(first.outputs["video"][0, 0, 0, 0, 0]) == 255
                assert first.position == {"video": (0, 1)}
            for value in (5, 6, 7):
                r = s.advance(value)
                assert r.advance_index == value - 5
                assert r.position["video"] == (1 + FRAMES * (value - 5), FRAMES)
                assert _pixels(r) == (value - 5, value, 0)
            assert [a["controls"] for a in pipe.advances] == [5, 6, 7]
        assert pipe.opened == 1 and pipe.closed == 1


def test_submit_returns_indices_and_results_carry_them(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:
        with h.open() as s, s.results() as results:
            assert s.submit("controls", 1) == 0
            assert s.submit("controls", 2) == 1
            r0 = next(results)
            assert r0.advance_index == 0 and _pixels(r0) == (0, 1, 0)
            r1 = next(results)
            assert r1.advance_index == 1 and _pixels(r1) == (1, 2, 0)


def test_live_input_index_is_the_next_dispatch_and_newest_wins(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:
        with h.open() as s, s.results() as results:
            first_boundary = s.submit("camera", 3)
            second_boundary = s.submit("camera", 4)  # overwrites 3 before any dispatch
            assert first_boundary == second_boundary == 0
            s.submit("controls", 1)
            r = next(results)
            assert r.advance_index == 0 and _pixels(r) == (0, 1, 4)
            later = s.submit("camera", 9)
            assert later >= 1
            s.submit("controls", 2)
            r = next(results)
            assert _pixels(r) == (1, 2, 9)


def test_queue_full_and_invalid_input_errors(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:
        with h.open() as s:
            # The consumer never asks for a result, so after the first advance fills the
            # only free slot (K=2 with one taken by nothing: two advances run), the
            # queue of depth 2 fills up.
            s.submit("controls", 1)
            s.submit("controls", 2)
            time.sleep(0.2)  # let the worker consume what it can with 2 slots
            s.submit("controls", 3)
            s.submit("controls", 4)
            with pytest.raises(SessionQueueFull):
                s.submit("controls", 5)
            with pytest.raises(ValueError):
                s.submit("controls", -1)
            with pytest.raises(ValueError):
                s.submit("unknown", 1)


def test_backpressure_pauses_the_worker_until_a_slot_returns(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe, SimpleNamespace(delivery_slots=1, idle_timeout=None)) as h:
        with h.open() as s, s.results() as results:
            s.submit("controls", 1)
            s.submit("controls", 2)
            time.sleep(0.2)
            assert len(pipe.advances) == 1  # the single slot is published and unread
            r0 = next(results)
            assert r0.advance_index == 0
            time.sleep(0.2)
            assert len(pipe.advances) == 1  # r0 still holds the slot
            r1 = next(results)  # returns slot 0, advance 1 runs
            assert r1.advance_index == 1
            with pytest.raises(SessionLeaseExpired):
                r0.outputs


def test_copy_survives_the_next_request(harness):
    with harness(CountingPipeline(initial=False)) as h:
        with h.open() as s, s.results() as results:
            s.submit("controls", 1)
            s.submit("controls", 2)
            r0 = next(results)
            keep = r0.copy()
            next(results)
            assert _pixels(keep) == (0, 1, 0)
            with pytest.raises(SessionLeaseExpired):
                r0.video


# --------------------------------------------------------------------------- autonomous


def test_autonomous_session_runs_to_its_terminal(harness):
    pipe = CountingPipeline(queued=False, initial=False, length=3)
    with harness(pipe) as h:
        with h.open() as s:
            assert not s.capabilities["lockstep"]
            with pytest.raises(SessionError):
                s.advance(1)
            seen = []
            with s.results() as results:
                for r in results:
                    if r.done:
                        assert r.error is None
                        seen.append("done")
                        break
                    seen.append(r.advance_index)
            assert seen == [0, 1, 2, "done"]
            with pytest.raises(SessionEnded):
                s.submit("camera", 1)


def test_iterator_ends_after_the_terminal_result(harness):
    with harness(CountingPipeline(queued=False, initial=False, length=1)) as h:
        with h.open() as s:
            results = s.results()
            media = next(results)
            terminal = next(results)
            assert not media.done and terminal.done
            with pytest.raises(StopIteration):
                next(results)
            results.close()


def test_advance_returns_the_terminal_once_then_raises(harness):
    with harness(CountingPipeline(initial=False, length=1)) as h:
        with h.open() as s:
            r = s.advance(1)
            assert r.advance_index == 0
            terminal = s.advance(2)  # session already ended: terminal, nothing submitted
            assert terminal.done and terminal.error is None
            with pytest.raises(SessionEnded):
                s.advance(3)


# --------------------------------------------------------------------------- errors


def test_open_failure_raises_with_the_failure_class(harness):
    with harness(CountingPipeline(open_fails=True)) as h:
        with pytest.raises(ValueError, match="bad prompt"):
            h.open()


def test_execution_error_ends_the_session_with_a_terminal_error(harness):
    pipe = CountingPipeline(initial=False, fail_at=1)
    with harness(pipe) as h:
        with h.open() as s, s.results() as results:
            s.submit("controls", 1)
            s.submit("controls", 2)
            r0 = next(results)
            terminal = next(results)
            assert r0.advance_index == 0
            assert terminal.done and "kernel exploded" in terminal.error
            with pytest.raises(SessionEnded):
                s.submit("controls", 3)
        assert pipe.closed == 1


def test_idle_timeout_ends_an_abandoned_session(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe, SimpleNamespace(delivery_slots=2, idle_timeout=0.2)) as h:
        s = h.open()
        results = s.results()
        terminal = next(results)  # nothing submitted: the clock runs, the timeout fires
        assert terminal.done and "idle timeout" in terminal.error
        deadline = time.monotonic() + 5
        while pipe.closed == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pipe.closed == 1  # model instance released without a close from the client
        s.close()


def test_consumer_role_is_exclusive(harness):
    with harness(CountingPipeline()) as h:
        with h.open() as s:
            results = s.results()
            with pytest.raises(SessionError):
                s.results()
            with pytest.raises(SessionError):
                s.advance(1)  # iterator open
            next(results)  # initial
            results.close()
            s.advance(1)
            with pytest.raises(SessionError):
                s.results()  # advance() holds the role now


def test_advance_refuses_while_the_initial_result_is_unread(harness):
    with harness(CountingPipeline(initial=True)) as h:
        with h.open() as s:
            time.sleep(0.1)
            with pytest.raises(SessionError, match="unread"):
                s.advance(1)


def test_close_discards_queued_inputs_and_second_session_opens(harness):
    pipe = CountingPipeline(initial=False)
    with harness(pipe) as h:
        s = h.open()
        s.submit("controls", 1)
        s.close()
        s.close()  # idempotent
        with pytest.raises(SessionError):
            s.submit("controls", 2)
        deadline = time.monotonic() + 5
        while pipe.closed == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pipe.closed == 1
        # The model instance is free again.
        req = DiffusionRequest(request_id=2, prompt=["q"], params=SimpleNamespace())
        channel = h.client.open_session(req)
        channel.opened.result(timeout=10)
        VisualGenSession(2, h.client, channel).close()
        assert pipe.opened == 2


# --------------------------------------------------------------------------- async


def test_async_forms(harness):
    pipe = CountingPipeline(initial=False, length=2)
    with harness(pipe) as h:

        async def run():
            req = DiffusionRequest(request_id=1, prompt=["p"], params=SimpleNamespace())
            channel = h.client.open_session(req)
            await asyncio.wrap_future(channel.opened)
            s = VisualGenSession(1, h.client, channel)
            async with s, s.results() as results:
                assert await s.submit_async("controls", 1) == 0
                assert await s.submit_async("controls", 2) == 1
                seen = []
                async for r in results:
                    if r.done:
                        seen.append("done")
                        break
                    seen.append(_pixels(r))
                assert seen == [(0, 1, 0), (1, 2, 0), "done"]

        asyncio.run(run())
        assert pipe.closed == 1
