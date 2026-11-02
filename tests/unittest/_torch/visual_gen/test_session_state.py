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
"""Session bookkeeping on the worker: admission, the advance condition, result
assembly and the slot pool. No pipeline, no processes."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.session import (
    CURRENT,
    QUEUED,
    AdvanceOutcome,
    InputSpec,
    SessionCapabilities,
    SessionEnded,
    SessionQueueFull,
    SessionState,
    StreamSpec,
)
from tensorrt_llm._torch.visual_gen.slot_pool import ClientSlotMap, SlotPool

pytestmark = pytest.mark.cpu_only


def _video_stream(frames=8, h=4, w=6):
    return StreamSpec("video", "video", torch.uint8, (1, frames, h, w, 3), axis=1, rate=24.0)


def _positive(value):
    if not isinstance(value, int) or value < 0:
        raise ValueError("controls must be a non-negative int")


def lockstep_caps(initial=True, length=None):
    return SessionCapabilities(
        inputs=(InputSpec("controls", QUEUED, validate=_positive, queue_depth=2),),
        streams=(_video_stream(),),
        initial_result=initial,
        length=length,
        lockstep=True,
    )


def steered_caps():
    return SessionCapabilities(
        inputs=(
            InputSpec("controls", QUEUED, queue_depth=3),
            InputSpec("camera", CURRENT, default=(0.0, 0.0)),
        ),
        streams=(
            _video_stream(),
            StreamSpec("audio", "audio", torch.float32, (1, 2, 1024), axis=2),
        ),
    )


# --------------------------------------------------------------------------- capabilities


def test_capabilities_reject_bad_declarations():
    with pytest.raises(ValueError):
        InputSpec("x", "sometimes")
    with pytest.raises(ValueError):
        StreamSpec("v", "video", torch.uint8, (1, 2), placement="pinned")
    with pytest.raises(ValueError):
        StreamSpec("v", "video", torch.uint8, (0, 1))
    with pytest.raises(ValueError):
        StreamSpec("v", "video", torch.uint8, (1, 2), axis=5)
    with pytest.raises(ValueError):
        SessionCapabilities(inputs=(), streams=())
    with pytest.raises(ValueError):
        SessionCapabilities(inputs=(InputSpec("a"), InputSpec("a")), streams=(_video_stream(),))
    with pytest.raises(ValueError):
        SessionCapabilities(inputs=(), streams=(_video_stream(),), lockstep=True)


def test_public_form_is_picklable_and_callable_free():
    import pickle

    public = lockstep_caps().public()
    pickle.dumps(public)
    assert public["inputs"][0]["policy"] == "queued"
    assert public["streams"][0]["dtype"] == "torch.uint8"
    assert public["lockstep"] is True
    assert public["streams"][0]["placement"] == "host"
    assert "validate" not in public["inputs"][0]


# --------------------------------------------------------------------------- admission


def test_queued_admission_returns_consumption_index_and_bounds_the_queue():
    state = SessionState(7, lockstep_caps())
    state.initialized = True
    assert state.admit("controls", 1) == 0
    assert state.admit("controls", 2) == 1
    with pytest.raises(SessionQueueFull):
        state.admit("controls", 3)
    with pytest.raises(ValueError):
        state.admit("controls", -1)
    with pytest.raises(ValueError):
        state.admit("nope", 1)


def test_live_admission_returns_next_dispatch_and_newest_wins():
    state = SessionState(1, steered_caps())
    state.initialized = True
    assert state.admit("camera", (1.0, 0.0)) == 0
    assert state.admit("camera", (2.0, 0.0)) == 0
    state.admit("controls", "a")
    assert state.can_advance(free_slots=1)
    inputs = state.take_inputs()
    state.dispatch()
    assert inputs == {"controls": "a", "camera": (2.0, 0.0)}
    # Submitted while advance 0 is in flight: observed from advance 1 on.
    assert state.admit("camera", (3.0, 0.0)) == 1


def test_advance_condition():
    state = SessionState(1, steered_caps())
    assert not state.can_advance(free_slots=1)  # not initialized
    state.initialized = True
    assert not state.can_advance(free_slots=1)  # queue empty
    state.admit("controls", "a")
    assert not state.can_advance(free_slots=0)  # no slot
    assert state.can_advance(free_slots=1)
    state.take_inputs()
    state.dispatch()
    state.admit("controls", "b")
    assert not state.can_advance(free_slots=1)  # in flight
    state.in_flight = False
    assert state.can_advance(free_slots=1)
    state.closing = True
    assert not state.can_advance(free_slots=1)


def test_finite_session_exhausts_and_refuses_extra_values():
    state = SessionState(1, lockstep_caps(initial=False, length=2))
    state.initialized = True
    assert state.admit("controls", 1) == 0
    assert state.admit("controls", 2) == 1
    with pytest.raises(SessionEnded):
        state.admit("controls", 3)
    for _ in range(2):
        assert state.can_advance(1)
        state.take_inputs()
        state.dispatch()
        state.in_flight = False
    assert not state.can_advance(1)
    assert state.exhausted()


def test_admission_refused_after_terminal_and_while_closing():
    state = SessionState(1, lockstep_caps())
    state.initialized = True
    state.terminal_result()
    with pytest.raises(SessionEnded):
        state.admit("controls", 1)
    state = SessionState(2, lockstep_caps())
    state.initialized = True
    state.closing = True
    with pytest.raises(SessionEnded):
        state.admit("controls", 1)


# --------------------------------------------------------------------------- results


def test_media_result_validates_and_advances_cursors():
    state = SessionState(3, steered_caps())
    written = {
        "video": ((1, 8, 4, 6, 3), torch.uint8),
        "audio": ((1, 2, 1000), torch.float32),
    }
    first = state.media_result(0, 0, written)
    assert first.sequence == 0 and first.advance_index == 0 and first.slot_id == 0
    assert first.position == {"video": (0, 8), "audio": (0, 1000)}
    assert first.streams["video"] == ((1, 8, 4, 6, 3), "torch.uint8")
    second = state.media_result(1, 1, {"video": written["video"]})
    assert second.sequence == 1
    assert second.position == {"video": (8, 8)}  # audio omitted: its cursor stays
    with pytest.raises(ValueError):
        state.media_result(2, 0, {"video": ((1, 9, 4, 6, 3), torch.uint8)})  # over max_shape
    with pytest.raises(ValueError):
        state.media_result(2, 0, {"video": ((1, 8, 4, 6, 3), torch.float32)})  # wrong dtype
    with pytest.raises(ValueError):
        state.media_result(2, 0, {})
    terminal = state.terminal_result()
    assert terminal.done and terminal.error is None and terminal.sequence == 2
    assert state.terminal_result() is terminal  # exactly one


# --------------------------------------------------------------------------- slot pool


def test_slot_pool_reserve_release_and_views():
    caps = steered_caps()
    pool = SlotPool(caps, num_slots=2, device=torch.device("cpu"))
    assert pool.free_count == 2
    a = pool.reserve()
    b = pool.reserve()
    assert (a.slot_id, b.slot_id) == (0, 1)
    assert pool.reserve() is None
    view = a.buffer("video", (1, 8, 4, 6, 3), torch.uint8)
    view.fill_(7)
    assert a.written()["video"] == ((1, 8, 4, 6, 3), torch.uint8)
    with pytest.raises(ValueError):
        a.buffer("video", (1, 9, 4, 6, 3), torch.uint8)
    with pytest.raises(ValueError):
        a.buffer("video", (1, 8, 4, 6, 3), torch.float32)
    pool.release(0)
    with pytest.raises(ValueError):
        pool.release(0)
    assert pool.reserve().slot_id == 0
    pool.close()


def test_client_map_sees_the_worker_write_without_a_copy():
    caps = lockstep_caps()
    pool = SlotPool(caps, num_slots=1, device=torch.device("cpu"))
    handles = pool.export_handles()
    client = ClientSlotMap(handles)
    slot = pool.reserve()
    worker_view = slot.buffer("video", (1, 2, 4, 6, 3), torch.uint8)
    worker_view.fill_(0)
    worker_view[0, 1] = 9
    shape, dtype = slot.written()["video"]
    client_view = client.view(0, "video", shape, dtype)
    assert client_view.shape == (1, 2, 4, 6, 3)
    assert int(client_view[0, 1, 0, 0, 0]) == 9 and int(client_view[0, 0, 0, 0, 0]) == 0
    assert client_view.data_ptr() == worker_view.data_ptr()  # same process, same bytes
    client.close()
    pool.close()


def test_outcome_defaults():
    outcome = AdvanceOutcome()
    assert outcome.event is None and not outcome.done
