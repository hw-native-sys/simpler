# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 two-rank CCU Broadcast triggered by a simpler AIV task."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


def _driver():
    path = Path(__file__).with_name("ccu_driver.py")
    spec = importlib.util.spec_from_file_location("ccu_broadcast_driver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(2)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_ccu_broadcast_demo(st_platform, st_device_ids):
    if st_platform != "a5":
        pytest.skip("CCU Broadcast requires real A5 CCU hardware")
    _driver().run_suite(list(st_device_ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a5"], default="a5")
    parser.add_argument("-d", "--devices", required=True, help="Two device IDs, for example 0,1")
    parser.add_argument("--counts", default="64,1024", help="Comma-separated float element counts")
    parser.add_argument("--roots", default="0,1", help="Comma-separated root rank IDs")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    _driver().run_suite(
        [int(device) for device in args.devices.split(",")],
        counts=[int(count) for count in args.counts.split(",")],
        roots=[int(root) for root in args.roots.split(",")],
        repeat=args.repeat,
        timeout=args.timeout,
    )
