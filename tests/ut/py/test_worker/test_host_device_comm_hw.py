# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""a2a3 onboard host/device channel + shared-memory smoke tests."""

import pytest
import simpler.worker as worker_mod
from simpler.task_interface import CallConfig
from simpler.worker import Worker


@pytest.fixture
def l3_a2a3_worker(st_device_ids):
    device_id = int(st_device_ids[0])
    worker = Worker(
        level=3,
        platform="a2a3",
        runtime="tensormap_and_ringbuffer",
        device_ids=[device_id],
        num_sub_workers=0,
    )
    try:
        worker.init()
    except FileNotFoundError as e:
        pytest.skip(f"a2a3 runtime binaries unavailable: {e}")
    try:
        yield worker
    finally:
        worker.close()


def _test_ctrl(name: str) -> int:
    if not hasattr(worker_mod, name):
        pytest.fail(f"missing {name}; a2a3 onboard smoke needs child-side queue seeding helper")
    return int(getattr(worker_mod, name))


def _child_channel_send_l2(worker: Worker, channel: int, route: int, payload: bytes, correlation_id: int) -> None:
    worker._chip_control_payload(
        0,
        _test_ctrl("_CTRL_TEST_CHANNEL_SEND_L2"),
        arg0=channel,
        arg1=route,
        arg2=len(payload),
        arg3=correlation_id,
        payload=payload,
    )


def _child_channel_recv_l2(worker: Worker, channel: int, capacity: int = 64) -> tuple[bytes, int, int]:
    nbytes, payload, _arg1, _arg2, route, correlation_id = worker._chip_control_payload(
        0,
        _test_ctrl("_CTRL_TEST_CHANNEL_RECV_L2"),
        arg0=channel,
        arg1=capacity,
        arg2=1000,
        recv_capacity=capacity,
    )
    return payload[:nbytes], route, correlation_id


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(1)
def test_a2a3_onboard_channel_round_trip(l3_a2a3_worker):
    opened: dict[str, int] = {}

    def open_and_send(orch, _args, _cfg):
        channel = orch.open_channel(
            worker_id=0,
            cpu_to_l2_lanes=1,
            l2_to_cpu_lanes=1,
            lane_depth=4,
            max_message_bytes=64,
        )
        opened["channel"] = channel
        orch.channel_send(0, channel, route=7, data=b"cpu-to-l2", correlation_id=0x1234)

    l3_a2a3_worker.run(open_and_send, args=None, config=CallConfig())

    channel = opened["channel"]
    payload, route, correlation_id = _child_channel_recv_l2(l3_a2a3_worker, channel)
    assert payload == b"cpu-to-l2"
    assert route == 7
    assert correlation_id == 0x1234

    _child_channel_send_l2(l3_a2a3_worker, channel, route=3, payload=b"l2-to-cpu", correlation_id=0x5678)

    received: dict[str, tuple[bytes, int, int]] = {}

    def recv_and_close(orch, _args, _cfg):
        try:
            received["message"] = orch.channel_recv(0, channel, capacity=64, timeout_us=1000)
        finally:
            orch.close_channel(0, channel)

    l3_a2a3_worker.run(recv_and_close, args=None, config=CallConfig())

    data, route, correlation_id = received["message"]
    assert data == b"l2-to-cpu"
    assert route == 3
    assert correlation_id == 0x5678


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(1)
def test_a2a3_onboard_shared_memory_chunked_round_trip(l3_a2a3_worker):
    payload = bytes(i % 251 for i in range(worker_mod._CTRL_PAYLOAD_CAPACITY + 23))
    seen_info: dict[str, tuple[int, int, int, int, int]] = {}
    seen_readback: dict[str, bytes] = {}

    def shm_round_trip(orch, _args, _cfg):
        memory = orch.open_shared_memory(0, data_bytes=len(payload) + 16, signal_count=2, flags=7)
        try:
            seen_info["value"] = orch.shared_memory_info(0, memory)
            orch.shared_memory_write(0, memory, 5, payload)
            seen_readback["value"] = orch.shared_memory_read(0, memory, 5, len(payload))
            orch.shared_memory_notify(0, memory, signal_id=1, value=9)
            orch.shared_memory_wait(0, memory, signal_id=1, target=9, timeout_us=0)
        finally:
            orch.close_shared_memory(0, memory)

    l3_a2a3_worker.run(shm_round_trip, args=None, config=CallConfig())

    host_ptr, device_ptr, data_bytes, signal_count, flags = seen_info["value"]
    assert host_ptr == 0
    assert device_ptr != 0
    assert data_bytes == len(payload) + 16
    assert signal_count == 2
    assert flags == 7
    assert seen_readback["value"] == payload


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(1)
def test_a2a3_onboard_host_device_errors_are_diagnostic(l3_a2a3_worker):
    def invalid_open(orch, _args, _cfg):
        orch.open_channel(
            worker_id=0,
            cpu_to_l2_lanes=1,
            l2_to_cpu_lanes=1,
            lane_depth=3,
            max_message_bytes=64,
        )

    with pytest.raises(RuntimeError, match="open_channel.*invalid config"):
        l3_a2a3_worker.run(invalid_open, args=None, config=CallConfig())

    def oversized_send(orch, _args, _cfg):
        channel = orch.open_channel(
            worker_id=0,
            cpu_to_l2_lanes=1,
            l2_to_cpu_lanes=1,
            lane_depth=4,
            max_message_bytes=64,
        )
        try:
            orch.channel_send(
                0,
                channel,
                route=1,
                data=b"x" * (worker_mod._CTRL_PAYLOAD_CAPACITY + 1),
                correlation_id=0,
            )
        finally:
            orch.close_channel(0, channel)

    with pytest.raises(ValueError, match="payload too large"):
        l3_a2a3_worker.run(oversized_send, args=None, config=CallConfig())
