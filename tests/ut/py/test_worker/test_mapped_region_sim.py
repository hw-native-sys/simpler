# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import subprocess
import sys
import textwrap
from array import array
from typing import Any

import pytest
import simpler.worker as worker_mod
from _task_interface import (  # pyright: ignore[reportMissingImports]
    CTRL_OFF_ARG0,
    CTRL_OFF_ARG1,
    CTRL_OFF_ARG2,
    CTRL_OFF_ARG3,
    CTRL_OFF_RESULT,
)
from simpler.task_interface import MappedRegionInfo
from simpler.worker import Worker


def _run_python_snippet(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_worker_mailbox_control_offsets_match_cpp_contract():
    assert (worker_mod._CTRL_OFF_ARG0, worker_mod._CTRL_OFF_ARG1, worker_mod._CTRL_OFF_ARG2) == (16, 24, 32)
    assert worker_mod._CTRL_OFF_ARG3 == 40
    assert worker_mod._CTRL_OFF_RESULT == 48
    assert (CTRL_OFF_ARG0, CTRL_OFF_ARG1, CTRL_OFF_ARG2, CTRL_OFF_ARG3, CTRL_OFF_RESULT) == (
        worker_mod._CTRL_OFF_ARG0,
        worker_mod._CTRL_OFF_ARG1,
        worker_mod._CTRL_OFF_ARG2,
        worker_mod._CTRL_OFF_ARG3,
        worker_mod._CTRL_OFF_RESULT,
    )


def test_worker_mapped_region_byte_view_accepts_bytes_for_non_byte_memoryview():
    raw = memoryview(array("I", [0, 0, 0, 0]))

    with pytest.raises(ValueError, match="different structures"):
        raw[:] = b"\x00" * raw.nbytes

    byte_view = worker_mod._mapped_region_byte_view(raw)
    byte_view[:] = b"\x5a" * raw.nbytes

    assert bytes(byte_view) == b"\x5a" * raw.nbytes


class FakeChipWorker:
    def __init__(self):
        self.calls: list[tuple[Any, ...]] = []

    def open_mapped_region(self, data_bytes, signal_count=1, flags=0):
        self.calls.append(("open", data_bytes, signal_count, flags))
        return 0xABC

    def mapped_region_info(self, handle):
        self.calls.append(("info", handle))
        return MappedRegionInfo(0, 0x1000, 16, 0, 0x2000, 2, 256, 0)

    def mapped_region_datacopy_h2region(self, handle, offset, data):
        self.calls.append(("h2region", handle, offset, data))

    def mapped_region_datacopy_region2h(self, handle, offset, nbytes):
        self.calls.append(("region2h", handle, offset, nbytes))
        return b"out"

    def mapped_region_notify(self, handle, signal_id, value):
        self.calls.append(("notify", handle, signal_id, value))

    def mapped_region_wait(self, handle, signal_id, target, timeout_us):
        self.calls.append(("wait", handle, signal_id, target, timeout_us))

    def close_mapped_region(self, handle):
        self.calls.append(("close", handle))


def make_l2_worker_with_fake_chip() -> tuple[Worker, FakeChipWorker]:
    worker = Worker(level=2)
    chip_worker = FakeChipWorker()
    worker._chip_worker = chip_worker
    worker._initialized = True
    return worker, chip_worker


def test_worker_l2_mapped_region_round_trips_to_chip_worker():
    worker, chip_worker = make_l2_worker_with_fake_chip()

    region = worker.open_mapped_region(16, signal_count=2)
    assert region.handle == 0xABC
    assert region.worker_id == 0
    assert region.data_bytes == 16
    assert region.signal_count == 2
    assert region.flags == 0
    assert region.closed is False

    info = worker.mapped_region_info(region)
    assert info.host_data_ptr == 0
    assert info.host_signal_ptr == 0
    assert info.device_data_ptr == 0x1000

    worker.mapped_region_datacopy_h2region(region, 4, b"abcd")
    assert worker.mapped_region_datacopy_region2h(region, 0, 3) == b"out"
    worker.mapped_region_notify(region, 1, 7)
    worker.mapped_region_wait(region, 1, 7, 100)
    worker.close_mapped_region(region)

    assert region.closed is True
    assert chip_worker.calls == [
        ("open", 16, 2, 0),
        ("info", 0xABC),
        ("h2region", 0xABC, 4, b"abcd"),
        ("region2h", 0xABC, 0, 3),
        ("notify", 0xABC, 1, 7),
        ("wait", 0xABC, 1, 7, 100),
        ("close", 0xABC),
    ]


def test_worker_mapped_region_rejects_mismatched_worker_id_and_closed_wrapper():
    worker, chip_worker = make_l2_worker_with_fake_chip()
    region = worker.open_mapped_region(16, signal_count=1)

    with pytest.raises(ValueError, match="worker_id"):
        worker.mapped_region_info(region, worker_id=1)

    worker.close_mapped_region(region)
    with pytest.raises(ValueError, match="closed"):
        worker.mapped_region_notify(region, 0, 1)

    worker.close_mapped_region(region)
    assert chip_worker.calls[-1] == ("close", 0xABC)
    assert chip_worker.calls.count(("close", 0xABC)) == 1


def test_worker_mapped_region_rejects_str_h2region_input():
    worker, _ = make_l2_worker_with_fake_chip()
    region = worker.open_mapped_region(16, signal_count=1)

    with pytest.raises(ValueError, match="bytes-like"):
        worker.mapped_region_datacopy_h2region(region, 0, "text")


@pytest.mark.parametrize("platform", ["a2a3sim", "a5sim"])
def test_worker_l2_mapped_region_sim_backend_round_trip(platform):
    _run_python_snippet(
        f"""
        from simpler.worker import Worker

        worker = Worker(level=2, platform="{platform}", runtime="tensormap_and_ringbuffer", build=True)
        worker.init()
        try:
            region = worker.open_mapped_region(8, signal_count=1)
            info = worker.mapped_region_info(region)
            assert info.host_data_ptr == 0
            assert info.host_signal_ptr == 0
            assert info.device_data_ptr != 0
            assert info.device_signal_ptr != 0
            assert info.data_bytes == 8
            assert info.signal_count == 1

            worker.mapped_region_datacopy_h2region(region, 2, b"abcd")
            assert worker.mapped_region_datacopy_region2h(region, 0, 8) == b"\\x00\\x00abcd\\x00\\x00"

            worker.mapped_region_wait(region, 0, 0, 0)
            try:
                worker.mapped_region_wait(region, 0, 1, 0)
                raise AssertionError("mapped_region_wait unexpectedly succeeded")
            except TimeoutError:
                pass
            worker.mapped_region_notify(region, 0, 3)
            worker.mapped_region_wait(region, 0, 3, 0)

            worker.close_mapped_region(region)
            try:
                worker.mapped_region_info(region)
                raise AssertionError("closed mapped region unexpectedly succeeded")
            except ValueError:
                pass
        finally:
            worker.close()
        """
    )


@pytest.mark.parametrize("platform", ["a2a3sim", "a5sim"])
def test_worker_l3_mapped_region_sim_backend_round_trip(platform):
    _run_python_snippet(
        f"""
        from simpler.worker import Worker

        worker = Worker(level=3, device_ids=[0], platform="{platform}", runtime="tensormap_and_ringbuffer", build=True)
        worker.init()
        try:
            region = worker.open_mapped_region(8192, signal_count=2, worker_id=0)
            info = worker.mapped_region_info(region)
            assert info.host_data_ptr == 0
            assert info.host_signal_ptr == 0
            assert info.device_data_ptr != 0
            assert info.device_signal_ptr != 0
            assert info.data_bytes == 8192
            assert info.signal_count == 2

            payload = bytes((i % 251 for i in range(5000)))
            worker.mapped_region_datacopy_h2region(region, 1024, payload)
            assert worker.mapped_region_datacopy_region2h(region, 1024, len(payload)) == payload

            try:
                worker.mapped_region_wait(region, 1, 1, 0)
                raise AssertionError("mapped_region_wait unexpectedly succeeded")
            except TimeoutError:
                pass
            worker.mapped_region_notify(region, 1, 9)
            worker.mapped_region_wait(region, 1, 9, 0)

            try:
                worker.mapped_region_info(region, worker_id=1)
                raise AssertionError("mismatched worker_id unexpectedly succeeded")
            except ValueError:
                pass
            worker.close_mapped_region(region)
            try:
                worker.mapped_region_info(region)
                raise AssertionError("closed mapped region unexpectedly succeeded")
            except ValueError:
                pass
        finally:
            worker.close()
        """
    )
