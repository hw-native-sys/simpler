# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The clone runner's ownership and admission boundaries."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

CASE = Path(__file__).resolve().parents[3] / "examples/a2a3/host_build_graph/qwen3_14b_serving_effective"
sys.path.insert(0, str(CASE))
from clone_worker_submit import (  # noqa: E402
    _HOST_INPUTS,
    _PRIVATE,
    _assert_private_host_storage,
    _assert_private_storage,
    _first_join_evidence,
    _reset_sampled_ids,
    _run_decode_pairs,
)


def _buffers(start):
    return _buffers_for(_PRIVATE, start)


def _buffers_for(names, start):
    return {
        name: SimpleNamespace(identity=SimpleNamespace(buffer_id=start + i), base=(start + i) * 4096, nbytes=64)
        for i, name in enumerate(names)
    }


def test_request_private_buffers_cannot_alias_while_both_requests_are_live():
    first, second = _buffers(1), _buffers(100)
    _assert_private_storage(first, second)
    second["k_cache"].base = first["v_cache"].base + 32
    with pytest.raises(AssertionError, match="share request-private storage"):
        _assert_private_storage(first, second)


def test_request_private_host_staging_cannot_alias_while_both_requests_are_live():
    first = _buffers_for(_HOST_INPUTS, 1)
    second = _buffers_for(_HOST_INPUTS, 100)
    _assert_private_host_storage(first, second)
    second["seq_lens"].base = first["slot_mapping"].base + 32
    with pytest.raises(AssertionError, match="share request-private host storage"):
        _assert_private_host_storage(first, second)


def test_clone_requests_submit_each_pair_before_waiting_for_completion():
    events = []
    next_step = [0, 0]
    active = set()

    def prepare_pair():
        assert not active
        events.append(("reset", 0, next_step[0]))
        events.append(("reset", 1, next_step[1]))

    class Handle:
        done = False

    def submit(request):
        step = next_step[request]
        next_step[request] += 1
        handle = Handle()
        active.add(request)
        events.append(("submit", request, step))
        return SimpleNamespace(index=step), handle, {}

    def finish_pair(pair):
        for request, step, handle, _entry in pair:
            handle.done = True
            active.remove(request)
            events.append(("finish", request, step.index))

    _run_decode_pairs(2, prepare_pair, submit, finish_pair)
    assert events == [
        ("reset", 0, 0),
        ("reset", 1, 0),
        ("submit", 0, 0),
        ("submit", 1, 0),
        ("finish", 0, 0),
        ("finish", 1, 0),
        ("reset", 0, 1),
        ("reset", 1, 1),
        ("submit", 0, 1),
        ("submit", 1, 1),
        ("finish", 0, 1),
        ("finish", 1, 1),
    ]


def test_clone_requests_require_first_submission_to_remain_in_flight():
    class Handle:
        done = True

    with pytest.raises(RuntimeError, match="before the second request"):
        _run_decode_pairs(
            1, lambda: None, lambda _request: (SimpleNamespace(index=0), Handle(), {}), lambda _pair: None
        )


def test_sampled_ids_are_reset_before_each_device_submission():
    host = SimpleNamespace(shm=SimpleNamespace(buf=bytearray(16 * 8 * 4)))
    device = object()
    torch.frombuffer(host.shm.buf, dtype=torch.int32, count=16 * 8).fill_(11)
    copied = []

    _reset_sampled_ids(host, device, lambda target, source: copied.append((target, source)))

    assert torch.equal(
        torch.frombuffer(host.shm.buf, dtype=torch.int32, count=16 * 8),
        torch.full((16 * 8,), -1, dtype=torch.int32),
    )
    assert copied == [(device, host)]


def test_native_join_requires_attributed_ordered_device_boundaries():
    spans = [
        SimpleNamespace(pid=7, inv=10, name="chip.run", attrs="run_id=1 dispatch_id=20 slot_id=0"),
        SimpleNamespace(pid=7, inv=11, name="chip.run", attrs="run_id=2 dispatch_id=21 slot_id=1"),
        SimpleNamespace(
            pid=7,
            inv=11,
            name="chip.run.joined_launch",
            attrs="p_disp=20 p_slot=0 s_disp=21 s_slot=1 observed=1 unfired=1 rc=0",
        ),
        SimpleNamespace(
            pid=7,
            inv=10,
            name="chip.run.runner_run.device_boundaries",
            attrs="dev_id=0 ts_hz=1000000 wo_rc=0 wo_end=100 aic_rc=0 aic_start=10",
        ),
        SimpleNamespace(
            pid=7,
            inv=11,
            name="chip.run.runner_run.device_boundaries",
            attrs="dev_id=0 ts_hz=1000000 wo_rc=0 wo_end=200 aic_rc=0 aic_start=101",
        ),
    ]
    assert _first_join_evidence(spans, (1, 2))["serial_device_ticks"] == 1
    spans[-1].attrs = spans[-1].attrs.replace("aic_start=101", "aic_start=99")
    with pytest.raises(AssertionError, match="overlapped"):
        _first_join_evidence(spans, (1, 2))
