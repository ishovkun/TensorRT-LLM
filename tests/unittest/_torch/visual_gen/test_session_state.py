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

from tensorrt_llm._torch.visual_gen.pipeline import (
    AdvanceOutcome,
    InputSpec,
    SessionCapabilities,
    StreamSpec,
)
from tensorrt_llm._torch.visual_gen.session import (
    SessionEnded,
    SessionQueueFull,
    SessionState,
    input_nbytes,
)
from tensorrt_llm._torch.visual_gen.slot_pool import ClientSlotMap, SlotPool

pytestmark = pytest.mark.cpu_only

ONE_INT = input_nbytes(1)  # the object size of one small int
EIGHT_FLOATS = input_nbytes([1.0] * 8)
TWO_FLOATS = input_nbytes((0.0, 0.0))


def _video_stream(frames=8, h=4, w=6):
    return StreamSpec(
        name="video", kind="video", dtype=torch.uint8, max_shape=(1, frames, h, w, 3), rate=24.0
    )


def _positive(value):
    if not isinstance(value, int) or value < 0:
        raise ValueError("controls must be a non-negative int")


def lockstep_caps(initial=True, length=None):
    return SessionCapabilities(
        inputs=(InputSpec(name="controls", max_bytes=ONE_INT, validator=_positive, queue_depth=2),),
        streams=(_video_stream(),),
        initial_result=initial,
        length=length,
        lockstep=True,
    )


def steered_caps():
    return SessionCapabilities(
        inputs=(
            InputSpec(name="controls", max_bytes=EIGHT_FLOATS, queue_depth=3),
            InputSpec(name="camera", max_bytes=TWO_FLOATS, policy="live", default=(0.0, 0.0)),
        ),
        streams=(
            _video_stream(),
            StreamSpec(
                name="audio", kind="audio", dtype=torch.float32, max_shape=(1, 2, 1024), axis=2
            ),
        ),
    )


# --------------------------------------------------------------------------- capabilities


def test_capabilities_reject_bad_declarations():
    with pytest.raises(ValueError):
        InputSpec(name="x", max_bytes=8, policy="sometimes")
    with pytest.raises(ValueError):
        InputSpec(name="x", max_bytes=8, queue_depth=0)
    with pytest.raises(ValueError):
        InputSpec(name="x")  # the byte bound is declared, never defaulted
    with pytest.raises(ValueError):
        InputSpec(name="x", max_bytes=0)
    with pytest.raises(ValueError):
        StreamSpec(name="v", kind="video", dtype=torch.uint8, max_shape=(1, 2), placement="pinned")
    with pytest.raises(ValueError):
        StreamSpec(name="v", kind="video", dtype=torch.uint8, max_shape=(0, 1))
    with pytest.raises(ValueError):
        StreamSpec(name="v", kind="video", dtype=torch.uint8, max_shape=(1, 2), axis=5)
    with pytest.raises(ValueError):
        SessionCapabilities(inputs=(), streams=())
    with pytest.raises(ValueError):
        SessionCapabilities(
            inputs=(InputSpec(name="a", max_bytes=8), InputSpec(name="a", max_bytes=8)),
            streams=(_video_stream(),),
        )
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


# --------------------------------------------------------------------------- input byte bound


def test_input_nbytes_counts_what_a_value_retains_or_expands_to():
    import sys

    import numpy as np

    from tensorrt_llm._torch.visual_gen.session import _NUMBER

    assert input_nbytes(7) == _NUMBER == input_nbytes(7.5)
    assert input_nbytes(b"abc") == sys.getsizeof(b"abc")
    big_int = 1 << (8 * 1048576)
    assert input_nbytes(big_int) == sys.getsizeof(big_int)  # an arbitrary-precision int costs
    assert input_nbytes([1.0, 2.0]) == sys.getsizeof([1.0, 2.0]) + 2 * _NUMBER
    assert input_nbytes({"a": 1}) == sys.getsizeof({"a": 1}) + sys.getsizeof("a") + _NUMBER
    appended: list = []
    for i in range(17):
        appended.append(i)
    assert input_nbytes(appended) == sys.getsizeof(appended) + 17 * _NUMBER  # actual allocation
    # A worst-case bound from a canonical value admits every ordinary construction
    # of the same shape: slice, literal, append-built, comprehension, ints for floats.
    import json

    bound = input_nbytes([[0.0] * 7 for _ in range(8)], worst_case=True)
    loaded = json.loads(json.dumps([[0.1 * j for j in range(7)] for _ in range(16)]))
    built: list = []
    for _ in range(8):
        row: list = []
        for j in range(7):
            row.append(j)
        built.append(row)
    forms = (loaded[0:8], loaded[8:16], [[0] * 7] * 8, built, [list(range(7)) for _ in range(8)])
    for form in forms:
        assert input_nbytes(form) <= bound, form
    # The headroom is for construction variance, not for more rows: twice the rows is refused.
    assert input_nbytes([[0.0] * 7 for _ in range(16)]) > bound
    big = torch.zeros(262144, dtype=torch.float32)  # 1 MiB of storage
    assert input_nbytes(big) == 1 << 20
    assert input_nbytes(big[:1]) == 1 << 20  # a view keeps the whole storage alive
    assert input_nbytes(big[:1].numpy()) >= 1 << 20  # NumPy view whose base is the tensor
    arr = np.zeros(262144, dtype=np.float32)
    assert input_nbytes(arr[:1]) == 1 << 20  # NumPy view of a NumPy owner
    with pytest.raises(ValueError, match="memoryview"):
        input_nbytes(memoryview(bytearray(16)))  # not picklable: not a wire form
    assert input_nbytes([b""] * 100000) > 3_000_000  # a hundred thousand references
    # Small storage, huge logical extent: counted at what it expands to.
    assert input_nbytes(np.broadcast_to(np.array([1.0]), (1_000_000,))) >= 8_000_000
    assert input_nbytes(torch.zeros(1).expand(1_000_000)) >= 4_000_000
    assert input_nbytes(torch.zeros(1).expand(1_000_000).numpy()) >= 4_000_000
    # Shared rows are ordinary input, counted once per appearance; self-containment is not.
    row = [1.0, 2.0]
    assert input_nbytes([row, row]) == sys.getsizeof([row, row]) + 2 * input_nbytes(row)
    with pytest.raises(ValueError, match="contains itself"):
        cyc: list = []
        cyc.append(cyc)
        input_nbytes(cyc)
    with pytest.raises(ValueError, match="object references"):
        input_nbytes(np.array([b"x" * 16], dtype=object))
    structured = np.empty(1, dtype=[("payload", object)])
    structured["payload"][0] = b"x" * 1024
    with pytest.raises(ValueError, match="object references"):
        input_nbytes(structured)
    with pytest.raises(ValueError, match="not a supported input form"):
        input_nbytes(object())
    # Decorated forms carry arbitrary Python state into serialization: refused.
    decorated_array = np.zeros(1, dtype=np.dtype("uint8", metadata={"payload": b"x" * 4096}))
    with pytest.raises(ValueError, match="dtype metadata"):
        input_nbytes(decorated_array)
    decorated_tensor = torch.zeros(1, dtype=torch.uint8)
    decorated_tensor.payload = b"x" * 4096
    with pytest.raises(ValueError, match="attributes"):
        input_nbytes(decorated_tensor)
    with pytest.raises(ValueError, match="autograd"):
        input_nbytes(torch.zeros(1, requires_grad=True))

    class MyList(list):
        pass

    with pytest.raises(ValueError, match="MyList"):
        input_nbytes(MyList([1]))
    with pytest.raises(ValueError, match="not a supported input form"):
        input_nbytes(np.zeros(1).view(np.recarray))
    # Plain views of decorated owners retain the decoration: refused down the chain.
    decorated_base = torch.zeros(1)
    decorated_base.payload = b"x" * 4096
    with pytest.raises(ValueError, match="attributes"):
        input_nbytes(decorated_base[:])
    with pytest.raises(ValueError, match="attributes"):
        input_nbytes(decorated_base[:][:1])
    decorated_owner = np.zeros(1, dtype=np.dtype("uint8", metadata={"payload": b"x" * 4096}))
    with pytest.raises(ValueError, match="dtype metadata"):
        input_nbytes(decorated_owner.view(np.uint8))
    with pytest.raises(ValueError, match="dtype metadata"):
        input_nbytes(decorated_owner.view(np.uint8)[:1])
    # ``Tensor.numpy()`` roots the array at a fresh plain tensor over the same
    # storage, so the decorated Python object is not retained: measured, accepted.
    assert 4 <= input_nbytes(decorated_base[:].numpy()) <= 8
    # Plain views of plain owners keep working.
    plain = torch.zeros(4)
    assert input_nbytes(plain[1:3]) == 16 and input_nbytes(plain[1:3].numpy()[:1]) >= 16
    assert input_nbytes(np.zeros(4)[1:3]) == 32
    # Plain forms stay usable.
    assert input_nbytes(np.zeros(2, dtype=np.float32)) == 8
    assert input_nbytes(torch.zeros(2, dtype=torch.float32)) == 8
    with pytest.raises(ValueError, match="fields"):
        input_nbytes(np.zeros(2, dtype=[("a", np.float32), ("b", np.float32)]))


def test_input_nbytes_walk_is_bounded_by_the_limit_not_the_value():
    import sys
    import tracemalloc

    huge = [None] * 1_000_000
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    size = input_nbytes(huge, limit=64)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert size == sys.getsizeof(huge)  # the list object alone: not expanded
    assert peak - before < 64 * 1024  # the walk allocated no reference array for the items
    nested = [[0] * 1000 for _ in range(10_000)]
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    size = input_nbytes(nested, limit=1024)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert 1024 < size < sys.getsizeof(nested) + 3 * 10_000 and peak - before < 64 * 1024


def test_oversize_input_is_rejected_as_capacity_before_anything_changes():
    state = SessionState(1, steered_caps())
    state.initialized = True
    with pytest.raises(SessionQueueFull, match="declared bound"):
        state.admit("controls", [1.0] * 9)
    assert state.admitted["controls"] == 0 and not state.queues["controls"]
    assert state.admit("controls", [1.0] * 8) == 0  # exactly the bound is accepted
    with pytest.raises(SessionQueueFull, match="declared bound"):
        state.admit("camera", (1.0, 2.0, 3.0))
    assert state.live["camera"] == (0.0, 0.0)  # the live value is unchanged
    assert state.admit("camera", (1.0, 2.0)) == 0
    for sneaky in (
        torch.zeros(1024)[:1],
        torch.zeros(1024)[:1].numpy(),
    ):
        with pytest.raises(SessionQueueFull):
            state.admit("controls", sneaky)


def test_live_default_must_fit_the_bound():
    with pytest.raises(ValueError, match="default holds"):
        InputSpec(name="live", policy="live", max_bytes=16, default=bytes(1 << 20))
    spec = InputSpec(name="live", policy="live", max_bytes=TWO_FLOATS, default=(0.0, 0.0))
    assert spec.default == (0.0, 0.0)


def test_queue_full_and_size_rejections_leave_the_next_index_unchanged():
    state = SessionState(1, lockstep_caps(initial=False))
    state.initialized = True
    assert state.admit("controls", 1) == 0
    assert state.admit("controls", 2) == 1
    with pytest.raises(SessionQueueFull):
        state.admit("controls", 3)
    with pytest.raises(SessionQueueFull):
        state.admit("controls", [1, 2])
    assert state.admitted["controls"] == 2 and list(state.queues["controls"]) == [1, 2]
    state.take_inputs()
    assert state.admit("controls", 3) == 2  # the next index is the one after the admitted two


def test_supported_input_forms_survive_the_wire_serialization():
    """Every accepted form goes through what the request socket does to it
    (pickle) and comes back equal."""
    import pickle

    import numpy as np

    from tensorrt_llm._torch.visual_gen.session import SessionSubmit

    forms = [
        1,
        2.5,
        True,
        "text",
        b"bytes",
        bytearray(b"ba"),
        [1.0, 2.0, [3.0]],
        (1, 2),
        {"k": [1, 2]},
        torch.arange(6, dtype=torch.float32).reshape(2, 3)[:, 1:],
        np.arange(6, dtype=np.float32).reshape(2, 3)[:, 1:],
    ]
    for value in forms:
        assert input_nbytes(value) >= 0  # accepted
        back = pickle.loads(pickle.dumps(SessionSubmit(1, "x", value))).value
        if isinstance(value, torch.Tensor):
            assert torch.equal(back, value)
        elif isinstance(value, np.ndarray):
            assert np.array_equal(back, value)
        else:
            assert back == value
    with pytest.raises(TypeError):
        pickle.dumps(memoryview(b"abc"))  # which is why it is not a supported form


def test_the_bench_action_slice_fits_the_sim_style_bound():
    """What the native end-to-end run sends: rows loaded from JSON, sliced per
    advance; the Sim bound is the worst case of the canonical float rows."""
    import json

    width, rows_per_advance = 29, 8
    actions = json.loads(json.dumps([[float(i % 3) for _ in range(width)] for i in range(296)]))
    bound = input_nbytes([[0.0] * width for _ in range(rows_per_advance)], worst_case=True)
    for start in range(0, 296, rows_per_advance):
        assert input_nbytes(actions[start : start + rows_per_advance]) <= bound
    assert (
        input_nbytes(actions[0 : 2 * rows_per_advance]) > bound
    )  # the shape validator does the rest
