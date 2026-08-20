# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the L2 Worker retirement classifier in the root conftest.

The classifier decides whether a failed L2 test recycles its pooled Worker. A
miss leaves a Worker whose chip run lane is poisoned in the pool, and every
later test on that runtime fails with the poison message instead of running.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]

_LANE_POISON_TAIL = "(run_id=0 slot=0 generation=2 dispatch_id=0 run_epoch=2)"


def _load_root_conftest():
    spec = importlib.util.spec_from_file_location("_root_conftest_device_poison", _ROOT / "conftest.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "msg",
    [
        f"poll_native_run failed with code 42 {_LANE_POISON_TAIL}",
        f"finalize_native_run failed with code -100 {_LANE_POISON_TAIL}",
        f"chip run lane is poisoned: finalize_native_run failed with code -100 {_LANE_POISON_TAIL}",
        f"chip run lane is poisoned: poll_native_run failed with code 42 {_LANE_POISON_TAIL}",
        "chip run lane is poisoned: native-run token is stale or used in the wrong phase",
        "chip run lane is poisoned by an unknown native failure",
        f"prepare_native_run failed with code 507018 {_LANE_POISON_TAIL}",
        f"launch_native_run failed with code 507018 {_LANE_POISON_TAIL}",
        "simpler_init failed with code 507899",
        "device is already owned by another live ChipWorker in this process",
    ],
)
def test_retirement_messages_recycle_the_worker(msg):
    assert _load_root_conftest()._requires_l2_worker_retirement_msg(msg)


@pytest.mark.parametrize(
    "msg",
    [
        "Golden mismatch on 'out': max_diff=0.0625, rtol=0.005, atol=0.02",
        f"prepare_native_run failed with code 42 {_LANE_POISON_TAIL}",
    ],
)
def test_non_poisoning_messages_keep_the_worker_pooled(msg):
    assert not _load_root_conftest()._requires_l2_worker_retirement_msg(msg)
