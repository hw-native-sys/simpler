# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sustained request scheduling and audited storage reuse."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

CASE = Path(__file__).resolve().parents[3] / "examples/a2a3/host_build_graph/qwen3_14b_serving_effective"
sys.path.insert(0, str(CASE))
from continuous_worker_submit import (  # noqa: E402
    _PRIVATE,
    _RETAINED,
    _arrival_predecessor,
    _assert_slot_storage,
    _close_resources,
    _joined_refills,
    _restore,
    _run_requests,
)


def _exercise(lengths, failure=None):
    active, live, retired, events = [], {}, [], []
    original = RuntimeError("injected failure")

    def start(state):
        assert state["slot"] not in live
        assert not active
        live[state["slot"]] = state["request"]
        events.append(("start", state["request"]))

    def submit(state, step):
        if failure == ("submit", state["request"], step):
            raise original
        assert len(active) < 4
        key = state["request"], step
        assert sum(request == key[0] for request, _ in active) < 2
        active.append(key)
        events.append(("submit", *key))

        def result(timeout):
            assert timeout == 120
            if failure == ("result", *key):
                raise original
            assert active.pop(0) == key
            events.append(("complete", *key))

        return SimpleNamespace(result=result)

    def finish(state, step, _handle):
        events.append(("read", state["request"], step))

    def checkpoint(states, finished):
        assert not active
        events.append(("checkpoint", len(retired)))
        for state in finished:
            assert state["submitted"] == state["completed"] == state["length"]
            retired.append(state["request"])
            del live[state["slot"]]

    if failure is None:
        _run_requests(lengths, start, submit, finish, checkpoint)
        assert not live and not active
        assert sorted(retired) == list(range(len(lengths)))
    else:
        with pytest.raises(RuntimeError) as caught:
            _run_requests(lengths, start, submit, finish, checkpoint)
        assert caught.value is original
    return events


@pytest.mark.parametrize("lengths", [[3, 7, 5], [2, 2, 2, 2], [5, 11, 7, 13, 9, 13]])
def test_staggered_requests_are_bounded_and_slots_retire_before_reuse(lengths):
    events = _exercise(lengths)
    assert sum(event[0] == "submit" for event in events) == sum(lengths)
    assert events.index(("submit", 0, 1)) < events.index(("complete", 0, 0))
    assert events.index(("submit", 1, 0)) < events.index(("complete", 0, 0))
    for request in range(2, len(lengths)):
        index = events.index(("start", request))
        assert events[index - 1][0] in {"checkpoint", "start"}


@pytest.mark.parametrize("first", range(2, 7))
@pytest.mark.parametrize("second", range(2, 7))
def test_odd_and_even_request_lengths_preserve_the_global_window(first, second):
    _exercise([first, second, 3, 6, 4])


@pytest.mark.parametrize("failure", [("submit", 0, 1), ("submit", 1, 0)])
def test_failed_submission_stops_refill_and_drains_accepted_runs(failure):
    events = _exercise([5, 11, 7], failure)
    assert not any(event == ("start", 2) for event in events)
    submitted = [event[1:] for event in events if event[0] == "submit"]
    completed = [event[1:] for event in events if event[0] == "complete"]
    assert submitted == completed


@pytest.mark.parametrize("lengths", [[], [2], [2, 2], [2, 0, 3], [2, -1, 3]])
def test_invalid_request_sequence_has_no_side_effects(lengths):
    def forbidden(*_args):
        pytest.fail("invalid request sequence reached a callback")

    with pytest.raises(ValueError):
        _run_requests(lengths, forbidden, forbidden, forbidden, forbidden)


def test_close_failure_retains_all_host_and_device_owners():
    events = []
    error = RuntimeError("close failed")

    def close():
        raise error

    worker = SimpleNamespace(close=close)
    hosts = [SimpleNamespace(close=lambda: events.append("host closed"))]
    slots = [object()]
    before = len(_RETAINED)
    report = {}
    try:
        with pytest.raises(RuntimeError) as caught:
            _close_resources(worker, hosts, slots, report)
        assert caught.value is error
        assert not events
        assert report["worker_closed"] is False
        assert _RETAINED[-1] == (worker, hosts, slots)
    finally:
        del _RETAINED[before:]


def test_successful_close_precedes_host_release_and_reports_release_failure():
    events = []
    error = RuntimeError("host release failed")

    def fail():
        events.append("failed host")
        raise error

    worker = SimpleNamespace(close=lambda: events.append("worker"))
    hosts = [SimpleNamespace(close=fail), SimpleNamespace(close=lambda: events.append("second host"))]
    report = {}
    with pytest.raises(RuntimeError) as caught:
        _close_resources(worker, hosts, [], report)
    assert caught.value is error
    assert events == ["worker", "failed host", "second host"]
    assert report["worker_closed"] is True


def test_initial_joined_edges_do_not_substitute_for_joined_refill():
    rows = [{"run_id": i, "refill": i >= 5} for i in range(1, 7)]
    with pytest.raises(AssertionError, match="refill"):
        _joined_refills(rows, {(4, 5), (5, 6)})
    assert _joined_refills(rows, {(4, 5)}) == 1


def test_arrival_requires_a_predecessor_that_has_not_been_drained():
    assert _arrival_predecessor([], set()) is None
    assert _arrival_predecessor([{"run_id": 7}], {7}) is None
    assert _arrival_predecessor([{"run_id": 7}], set()) == 7


def test_device_failure_stops_requests_and_still_waits_for_other_handles():
    submitted, waited = [], []
    error = RuntimeError("device failed")

    def submit(state, step):
        key = state["request"], step
        submitted.append(key)

        def result(timeout):
            waited.append(key)
            if key == (0, 0):
                raise error

        return SimpleNamespace(result=result)

    with pytest.raises(RuntimeError) as caught:
        _run_requests([5, 11, 7], lambda _: None, submit, lambda *_: None, lambda *_: None)
    assert caught.value is error
    assert submitted == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert set(waited) == set(submitted)


@pytest.mark.parametrize("kind", ["token", "metadata", "logits"])
def test_slot_isolation_includes_additional_token_metadata_and_output_frames(kind):
    def buffer(index):
        return SimpleNamespace(identity=SimpleNamespace(buffer_id=index), base=index * 4096, nbytes=64)

    def slot(base):
        return {
            "device": {name: buffer(base + i) for i, name in enumerate(_PRIVATE)},
            "host": {"sampled_ids": buffer(base + 10)},
            "tokens": [buffer(base + 20), buffer(base + 21)],
            "metadata": [{"seq_lens": buffer(base + 30)}, {"seq_lens": buffer(base + 31)}],
            "outputs": [
                {"next_hidden": buffer(base + 40), "out": buffer(base + 41)},
                {"next_hidden": buffer(base + 42), "out": buffer(base + 43)},
            ],
        }

    slots = [slot(1), slot(100)]
    _assert_slot_storage(slots)
    if kind == "token":
        slots[1]["tokens"][1] = slots[0]["tokens"][0]
    elif kind == "metadata":
        slots[1]["metadata"][1]["seq_lens"] = slots[0]["metadata"][0]["seq_lens"]
    else:
        slots[1]["outputs"][1]["out"] = slots[0]["outputs"][0]["out"]
    with pytest.raises(AssertionError, match="share private storage"):
        _assert_slot_storage(slots)


@pytest.mark.parametrize("fail_copy", [False, True])
def test_kv_restore_reuses_one_staging_buffer_and_releases_on_failure(fail_copy):
    allocated, released, uploads = [], [], []
    values = [torch.tensor([index, index + 1], dtype=torch.int32) for index in range(4)]

    def allocate(nbytes):
        host = SimpleNamespace(shm=SimpleNamespace(buf=bytearray(nbytes)), nbytes=nbytes)
        allocated.append(host)
        return host

    def copy_to(device, host, dst_offset=0):
        if fail_copy:
            raise RuntimeError("copy failed")
        uploads.append((device, dst_offset, bytes(host.shm.buf)))

    worker = SimpleNamespace(create_buffer=allocate, copy_to=copy_to, release_buffer=released.append)
    fixture = SimpleNamespace(
        layers=2,
        layer=lambda index: [("key", values[2 * index]), ("value", values[2 * index + 1])],
    )
    key, value = SimpleNamespace(nbytes=16), SimpleNamespace(nbytes=16)
    slot = {
        "device": {"k_cache": key, "v_cache": value},
        "host": {"sampled_ids": SimpleNamespace(shm=SimpleNamespace(buf=bytearray(16 * 8 * 4)))},
        "tokens": [],
    }
    if fail_copy:
        with pytest.raises(RuntimeError, match="copy failed"):
            _restore(worker, slot, fixture)
    else:
        _restore(worker, slot, fixture)
        assert [(device, offset) for device, offset, _payload in uploads] == [
            (key, 0),
            (value, 0),
            (key, 8),
            (value, 8),
        ]
        assert [list(torch.frombuffer(bytearray(payload), dtype=torch.int32)) for _, _, payload in uploads] == [
            list(row) for row in values
        ]
    assert len(allocated) == 1
    assert released == allocated
