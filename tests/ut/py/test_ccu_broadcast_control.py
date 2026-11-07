# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-free validation of the CCU example's rank coordination failures."""

import ctypes
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[3] / "examples/a5/tensormap_and_ringbuffer/ccu_broadcast_demo/ccu_driver.py"
_SPEC = importlib.util.spec_from_file_location("ccu_example_control", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


class Reply:
    def __init__(self, message=None, ready=True, eof=False):
        self.message, self.ready, self.eof = message, ready, eof
        self.poll_timeouts = []

    def poll(self, timeout):
        self.poll_timeouts.append(timeout)
        return self.ready

    def recv(self):
        if self.eof:
            raise EOFError
        return self.message


@pytest.mark.parametrize("devices", [[], [0], [0, 0], [0, 1, 2], [-1, 1], [True, 2], [0.5, 1], [0, 2**40]])
def test_distinct_two_rank_devices_required(devices):
    with pytest.raises(ValueError, match="exactly two distinct"):
        driver.validate_request(devices, [64], [0], 1, 30)


@pytest.mark.parametrize("counts", [[], [0], [-1], [1025], [True], [1.5]])
def test_counts_fit_the_native_window(counts):
    with pytest.raises(ValueError, match="Element counts"):
        driver.validate_request([3, 7], counts, [0], 1, 30)


def test_valid_nonconsecutive_devices_and_both_roots():
    driver.validate_request([3, 7], [1, 64, 1024], [0, 1], 2, 30)
    assert ctypes.sizeof(driver.PeerInfo) == 32


@pytest.mark.parametrize("roots", [[], [-1], [2], [True], [1.0]])
def test_only_rank_ids_can_select_the_root(roots):
    with pytest.raises(ValueError, match="Broadcast roots"):
        driver.validate_request([3, 7], [64], roots, 1, 30)


@pytest.mark.parametrize("repeat", [0, -1, 101, True, 1.5])
def test_repetitions_are_bounded(repeat):
    with pytest.raises(ValueError, match="Repeat"):
        driver.validate_request([3, 7], [64], [0], repeat, 30)


@pytest.mark.parametrize("timeout", [0, -1, 601, True, float("inf"), float("nan")])
def test_timeout_has_a_finite_positive_budget(timeout):
    with pytest.raises(ValueError, match="Control timeout"):
        driver.validate_request([3, 7], [64], [0], 1, timeout)


def test_rank_failure_is_not_accepted_as_ready():
    with pytest.raises(RuntimeError, match="no UB link"):
        driver.receive_stage(Reply({"stage": "error", "rank": 1, "error": "no UB link"}), "armed", 1, 1)


def test_unexpected_stage_or_rank_is_rejected():
    with pytest.raises(RuntimeError, match="expected 'armed'"):
        driver.receive_stage(Reply({"stage": "window", "rank": 0}), "armed", 1, 0)
    with pytest.raises(RuntimeError, match="Invalid CCU control reply"):
        driver.receive_stage(Reply({"stage": "armed", "rank": 1}), "armed", 1, 0)


def test_missing_or_exited_rank_is_bounded():
    with pytest.raises(TimeoutError, match="control deadline"):
        driver.receive_stage(Reply(ready=False), "verified", 0.1, 1)
    with pytest.raises(RuntimeError, match="exited before"):
        driver.receive_stage(Reply(eof=True), "verified", 1, 1)


def test_all_ranks_share_one_stage_deadline(monkeypatch):
    ticks = iter([10, 10.25, 10.75])
    monkeypatch.setattr(driver.time, "monotonic", lambda: next(ticks))
    replies = [Reply({"stage": "armed", "rank": rank}) for rank in range(2)]
    result = driver.receive_all(replies, "armed", 1)
    assert len(result) == 2
    assert replies[0].poll_timeouts == [0.75]
    assert replies[1].poll_timeouts == [0.25]


def test_prepare_rejects_incomplete_peer_metadata_without_calling_sdk():
    native = driver.NativeSession.__new__(driver.NativeSession)
    with pytest.raises(ValueError, match="both ranks"):
        native.prepare([(1, 2, 3, 4)], 0)
