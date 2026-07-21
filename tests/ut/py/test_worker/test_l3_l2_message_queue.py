# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
import math
import struct
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import pytest
from simpler import l3_l2_orch_comm
from simpler import worker as worker_module
from simpler.l3_l2_message_queue import (
    L3L2_QUEUE_COUNTER_BYTES,
    L3L2_QUEUE_DESC_SLOT_BYTES,
    L3L2_QUEUE_L2_ABORT_FLAG_OFFSET,
    L3L2_QUEUE_L3_ABORT_FLAG_OFFSET,
    L3L2Queue,
    L3L2QueueMessage,
    L3L2QueueOpcode,
    make_l3_l2_queue_layout,
)
from simpler.l3_l2_orch_comm import (
    L3HostRegionMapping,
    L3L2OrchRegion,
    L3L2OrchRegionDesc,
    L3L2RegionAccessProfile,
    NotifyOp,
    WaitCmp,
)
from simpler.orchestrator import Orchestrator
from simpler.task_interface import DataType, Tensor, get_element_size
from simpler.worker import _IDLE, _OFF_STATE, Worker, _buffer_field_addr, _mailbox_store_i32


@dataclass(frozen=True)
class _FakeRequest:
    cmd: str
    op: int = 0
    region_id: int = 1
    payload_offset: int = 0
    host_ptr: int = 0
    payload_bytes: int = 0
    counter_bytes: int = 0
    counter_addr: int = 0
    counter_operand: int = 0


class _FakeCWorker:
    def __init__(self):
        self.next_region_id = 1

    def control_l3_l2_region_create(self, worker_id: int, request_shm_name: str, reply_shm_name: str) -> None:
        req_shm = SharedMemory(name=request_shm_name)
        reply_shm = SharedMemory(name=reply_shm_name)
        req_buf = req_shm.buf
        reply_buf = reply_shm.buf
        assert req_buf is not None
        assert reply_buf is not None
        try:
            req = l3_l2_orch_comm._REGION_CREATE_REQUEST.unpack_from(req_buf, 0)
            payload_bytes = int(req[2])
            counter_bytes = int(req[3])
            counter_offset = ((payload_bytes + 63) // 64) * 64
            region_id = self.next_region_id
            self.next_region_id += 1
            backing_name = f"queue-direct-{region_id}".encode()
            l3_l2_orch_comm._REGION_CREATE_REPLY.pack_into(
                reply_buf,
                0,
                0x4C334C3200020000,
                region_id,
                0x1000_0000,
                payload_bytes,
                0x1000_0000 + counter_offset,
                counter_bytes,
                int(L3L2RegionAccessProfile.SIM_POSIX_SHM),
                0,
                0,
                backing_name + b"\x00" * (l3_l2_orch_comm._CTRL_SHM_TOKEN_BYTES - len(backing_name)),
                counter_offset + counter_bytes,
                0,
            )
        finally:
            del req_buf
            del reply_buf
            req_shm.close()
            reply_shm.close()

    def control_l3_l2_region_release(self, worker_id: int, region_id: int) -> None:
        pass


class _FakeCOrch:
    def __init__(self):
        self._buffers = []
        self.fail_next_alloc = False

    def alloc(self, shape, dtype):
        if self.fail_next_alloc:
            self.fail_next_alloc = False
            raise RuntimeError("injected allocation failure")
        nbytes = math.prod(int(x) for x in shape) * int(get_element_size(dtype))
        storage_t = ctypes.c_uint8 * nbytes
        storage = storage_t()
        self._buffers.append(storage)
        return Tensor.make(ctypes.addressof(storage), tuple(int(x) for x in shape), dtype)


class _FakeClient:
    def __init__(self):
        self.requests: list[tuple[_FakeRequest, float]] = []
        self.payload_writes: list[tuple[int, bytes]] = []
        self.next_region_id = 1
        self.payload_base = 0x1000_0000
        self.counter_base = 0x1000_0000
        self.counter_mapping_offset = 0
        self.payload = bytearray()
        self.counters: dict[int, int] = {}
        self.peer_abort = False
        self.fail_next_cmd: Optional[str] = None
        self.original_helpers: list[tuple[object, str, object]] = []

    def import_region(self, _token: str, mapping_bytes: int) -> int:
        self.payload = bytearray(int(mapping_bytes))
        self.counters = {}
        self.counter_mapping_offset = int(mapping_bytes) - L3L2_QUEUE_COUNTER_BYTES
        self.counter_base = self.payload_base + self.counter_mapping_offset
        self.requests.append(
            (
                _FakeRequest(
                    cmd="alloc_region",
                    payload_bytes=self.counter_mapping_offset,
                    counter_bytes=L3L2_QUEUE_COUNTER_BYTES,
                ),
                0.0,
            )
        )
        return 1

    def payload_write(self, handle: int, offset: int, host_ptr: int, nbytes: int) -> None:
        request = _FakeRequest(
            cmd="payload_write",
            payload_offset=int(offset),
            host_ptr=int(host_ptr),
            payload_bytes=int(nbytes),
        )
        self.requests.append((request, 0.0))
        if self.fail_next_cmd == request.cmd:
            self.fail_next_cmd = None
            raise RuntimeError(f"injected failure for {request.cmd}")
        data = ctypes.string_at(int(host_ptr), int(nbytes))
        self.payload_writes.append((int(offset), data))
        begin = int(offset)
        self.payload[begin : begin + int(nbytes)] = data

    def payload_read(self, handle: int, offset: int, host_ptr: int, nbytes: int) -> None:
        request = _FakeRequest(
            cmd="payload_read",
            payload_offset=int(offset),
            host_ptr=int(host_ptr),
            payload_bytes=int(nbytes),
        )
        self.requests.append((request, 0.0))
        if request.cmd == "payload_write":
            raise AssertionError("unreachable")
        begin = int(offset)
        data = bytes(self.payload[begin : begin + int(nbytes)])
        ctypes.memmove(int(host_ptr), data, len(data))

    def counter_notify(self, handle: int, offset: int, value: int, op: int) -> None:
        logical_offset = int(offset) - self.counter_mapping_offset
        request = _FakeRequest(
            cmd="counter_notify",
            op=int(op),
            counter_addr=self.counter_base + logical_offset,
            counter_operand=int(value),
        )
        self.requests.append((request, 0.0))
        if int(op) == int(NotifyOp.Add):
            self.counters[logical_offset] = int(self.counters.get(logical_offset, 0)) + int(value)
        else:
            self.counters[logical_offset] = int(value)

    def counter_test(self, handle: int, offset: int, operand: int, cmp: int) -> tuple[bool, int]:
        logical_offset = int(offset) - self.counter_mapping_offset
        observed = (
            1
            if self.peer_abort and logical_offset == L3L2_QUEUE_L2_ABORT_FLAG_OFFSET
            else self.counters.get(logical_offset, 0)
        )
        request = _FakeRequest(
            cmd="counter_test",
            op=int(cmp),
            counter_addr=self.counter_base + logical_offset,
            counter_operand=int(operand),
        )
        self.requests.append((request, 0.0))
        return _test_compare_counter(observed, int(operand), int(cmp)), observed

    def counter_wait(self, handle: int, offset: int, operand: int, cmp: int, timeout_ns: int):
        matched, observed = self.counter_test(handle, offset, operand, cmp)
        if matched:
            return (0, 0, observed, True, "")
        return (-1, 7, observed, False, "SIGNAL_WAIT timed out")


def _test_compare_counter(observed: int, operand: int, cmp: int) -> bool:
    if cmp == int(WaitCmp.EQ):
        return observed == operand
    if cmp == int(WaitCmp.NE):
        return observed != operand
    if cmp == int(WaitCmp.GT):
        return observed > operand
    if cmp == int(WaitCmp.GE):
        return observed >= operand
    if cmp == int(WaitCmp.LT):
        return observed < operand
    if cmp == int(WaitCmp.LE):
        return observed <= operand
    return False


def _make_orchestrator() -> tuple[Orchestrator, Worker, SharedMemory, _FakeClient]:
    worker = Worker(level=3, device_ids=[0], platform="a2a3sim", runtime="tensormap_and_ringbuffer")
    shm = SharedMemory(create=True, size=4096)
    assert shm.buf is not None
    _mailbox_store_i32(_buffer_field_addr(shm.buf, _OFF_STATE), _IDLE)
    fake_client = _FakeClient()
    fake_client.original_helpers = [
        (worker_module, "_l3_host_mapped_region_import_sim", worker_module._l3_host_mapped_region_import_sim),
        (l3_l2_orch_comm, "_l3_host_mapped_payload_write", l3_l2_orch_comm._l3_host_mapped_payload_write),
        (l3_l2_orch_comm, "_l3_host_mapped_payload_read", l3_l2_orch_comm._l3_host_mapped_payload_read),
        (l3_l2_orch_comm, "_l3_host_mapped_counter_notify", l3_l2_orch_comm._l3_host_mapped_counter_notify),
        (l3_l2_orch_comm, "_l3_host_mapped_counter_test", l3_l2_orch_comm._l3_host_mapped_counter_test),
        (l3_l2_orch_comm, "_l3_host_mapped_counter_wait", l3_l2_orch_comm._l3_host_mapped_counter_wait),
        (l3_l2_orch_comm, "_l3_host_mapped_region_close", l3_l2_orch_comm._l3_host_mapped_region_close),
    ]
    worker._initialized = True
    worker._hierarchical_started = True
    worker._worker = _FakeCWorker()
    worker._chip_shms = [shm]
    worker._l3_l2_test_fake_client = fake_client
    worker_module._l3_host_mapped_region_import_sim = fake_client.import_region
    l3_l2_orch_comm._l3_host_mapped_payload_write = fake_client.payload_write
    l3_l2_orch_comm._l3_host_mapped_payload_read = fake_client.payload_read
    l3_l2_orch_comm._l3_host_mapped_counter_notify = fake_client.counter_notify
    l3_l2_orch_comm._l3_host_mapped_counter_test = fake_client.counter_test
    l3_l2_orch_comm._l3_host_mapped_counter_wait = fake_client.counter_wait
    l3_l2_orch_comm._l3_host_mapped_region_close = lambda _handle: None
    return Orchestrator(_FakeCOrch(), worker), worker, shm, fake_client


def _close(worker: Worker, shm: SharedMemory) -> None:
    worker._close_l3_l2_orch_comm()
    fake_client = getattr(worker, "_l3_l2_test_fake_client", None)
    if fake_client is not None:
        for module, name, helper in fake_client.original_helpers:
            setattr(module, name, helper)
    shm.close()
    shm.unlink()


def _publish_output(
    fake_client: _FakeClient,
    queue,
    *,
    seq: int = 1,
    payload: bytes = b"",
    opcode: int = int(L3L2QueueOpcode.DATA),
    payload_offset: Optional[int] = None,
) -> None:
    if payload_offset is None:
        payload_offset = queue.layout.output_arena_offset if payload else 0
    if payload:
        fake_client.payload[payload_offset : payload_offset + len(payload)] = payload
    desc = struct.pack("<4Q", seq, int(opcode), payload_offset, len(payload))
    desc_offset = queue.layout.output_desc_offset + ((seq - 1) & (queue.layout.depth - 1)) * L3L2_QUEUE_DESC_SLOT_BYTES
    fake_client.payload[desc_offset : desc_offset + L3L2_QUEUE_DESC_SLOT_BYTES] = desc
    fake_client.counters[queue.layout.output_desc_tail_offset] = seq


def test_layout_rejects_invalid_pr1_parameters():
    invalid_args = [
        (3, 128, 128),
        ((1 << 30) + 1, 128, 128),
        (4, 0, 128),
        (4, 127, 128),
        (4, 128, 0),
        (4, 128, 127),
    ]

    for depth, input_arena_bytes, output_arena_bytes in invalid_args:
        with pytest.raises(ValueError):
            make_l3_l2_queue_layout(depth, input_arena_bytes, output_arena_bytes)


def test_layout_rejects_uint64_overflow_to_match_cpp_helper():
    with pytest.raises(ValueError, match="overflowed uint64"):
        make_l3_l2_queue_layout(2, (1 << 64) - 64, 64)


@pytest.mark.parametrize(
    ("depth", "input_arena_bytes", "output_arena_bytes", "expected"),
    [
        (
            1,
            64,
            64,
            {
                "output_desc_offset": 32,
                "input_arena_offset": 64,
                "output_arena_offset": 128,
                "payload_bytes": 192,
            },
        ),
        (
            4,
            128,
            192,
            {
                "output_desc_offset": 128,
                "input_arena_offset": 256,
                "output_arena_offset": 384,
                "payload_bytes": 576,
            },
        ),
        (
            8,
            192,
            64,
            {
                "output_desc_offset": 256,
                "input_arena_offset": 512,
                "output_arena_offset": 704,
                "payload_bytes": 768,
            },
        ),
    ],
)
def test_layout_lockstep_cases_match_cpp_helper_expectations(depth, input_arena_bytes, output_arena_bytes, expected):
    layout = make_l3_l2_queue_layout(
        depth=depth,
        input_arena_bytes=input_arena_bytes,
        output_arena_bytes=output_arena_bytes,
    )

    assert layout.input_desc_offset == 0
    assert layout.output_desc_offset == expected["output_desc_offset"]
    assert layout.output_desc_offset == depth * L3L2_QUEUE_DESC_SLOT_BYTES
    assert layout.input_arena_offset == expected["input_arena_offset"]
    assert layout.output_arena_offset == expected["output_arena_offset"]
    assert layout.payload_bytes == expected["payload_bytes"]
    assert layout.input_arena_offset % 64 == 0
    assert layout.output_arena_offset % 64 == 0
    assert layout.input_desc_tail_offset == 0
    assert layout.input_desc_head_offset == 64
    assert layout.output_desc_tail_offset == 128
    assert layout.output_desc_head_offset == 192
    assert layout.l3_abort_flag_offset == L3L2_QUEUE_L3_ABORT_FLAG_OFFSET
    assert layout.l2_abort_flag_offset == L3L2_QUEUE_L2_ABORT_FLAG_OFFSET
    assert layout.counter_bytes == L3L2_QUEUE_COUNTER_BYTES


def test_create_l3_l2_queue_allocates_region_and_exposes_l2_task_scalars():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=192)

        alloc_req = fake_client.requests[0][0]
        assert alloc_req.cmd == "alloc_region"
        assert alloc_req.payload_bytes == queue.layout.payload_bytes
        assert alloc_req.counter_bytes == L3L2_QUEUE_COUNTER_BYTES
        assert queue.l2_task_arg_scalars() == [
            *queue.region.descriptor_scalars(),
            queue.magic_version,
            4,
            128,
            192,
            queue.layout.payload_bytes,
            queue.layout.counter_bytes,
        ]
        assert fake_client.counters == {
            queue.layout.input_desc_tail_offset: 0,
            queue.layout.input_desc_head_offset: 0,
            queue.layout.output_desc_tail_offset: 0,
            queue.layout.output_desc_head_offset: 0,
            queue.layout.l3_abort_flag_offset: 0,
            queue.layout.l2_abort_flag_offset: 0,
        }
    finally:
        _close(worker, shm)


def test_create_l3_l2_queue_frees_region_on_post_region_alloc_failure():
    orch, worker, shm, _fake_client = _make_orchestrator()
    original_alloc = orch._o.alloc

    def fail_alloc(_shape, _dtype):
        raise RuntimeError("injected alloc failure")

    orch._o.alloc = fail_alloc
    try:
        with pytest.raises(RuntimeError, match="injected alloc failure"):
            orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)

        assert len(worker._live_l3_l2_regions) == 1
        assert worker._live_l3_l2_regions[0]._released is True
    finally:
        orch._o.alloc = original_alloc
        _close(worker, shm)


def test_zero_byte_enqueue_skips_message_payload_write_and_publishes_descriptor():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        fake_client.requests.clear()
        fake_client.payload_writes.clear()

        queue.input.enqueue(None, nbytes=0, timeout=0.001)

        payload_write_offsets = [offset for offset, _data in fake_client.payload_writes]
        assert queue.layout.input_arena_offset not in payload_write_offsets
        assert queue.layout.input_desc_offset in payload_write_offsets
        notify_req = fake_client.requests[-1][0]
        assert notify_req.cmd == "counter_notify"
        assert notify_req.op == int(NotifyOp.Set)
        assert notify_req.counter_addr == queue.region.descriptor.counter_base + queue.layout.input_desc_tail_offset
        assert notify_req.counter_operand == 1
    finally:
        _close(worker, shm)


def test_enqueue_registered_tensor_uses_direct_payload_write_without_extra_alloc():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        host = orch.alloc([16], DataType.UINT8)
        alloc_count = len(orch._o._buffers)
        fake_client.requests.clear()
        fake_client.payload_writes.clear()

        queue.input.enqueue(host, nbytes=16, timeout=0.001)

        payload_write_offsets = [offset for offset, _data in fake_client.payload_writes]
        assert queue.layout.input_arena_offset in payload_write_offsets
        assert queue.layout.input_desc_offset in payload_write_offsets
        assert all(req.cmd != "alloc_region" for req, _timeout in fake_client.requests)
        assert len(orch._o._buffers) == alloc_count
    finally:
        _close(worker, shm)


def test_enqueue_replays_released_descriptors_before_reusing_input_arena():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        first = orch.alloc([80], DataType.UINT8)
        second = orch.alloc([80], DataType.UINT8)

        queue.input.enqueue(first, nbytes=80, timeout=0.001)
        fake_client.counters[queue.layout.input_desc_head_offset] = 1
        queue.input.enqueue(second, nbytes=80, timeout=0.001)

        payload_offsets = [offset for offset, data in fake_client.payload_writes if len(data) == 80]
        assert payload_offsets == [queue.layout.input_arena_offset, queue.layout.input_arena_offset]
    finally:
        _close(worker, shm)


def test_enqueue_accepts_ordinary_host_bytes_with_direct_payload_write():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        alloc_count = len(orch._o._buffers)
        fake_client.requests.clear()
        fake_client.payload_writes.clear()

        queue.input.enqueue(b"ordinary", nbytes=8, timeout=0.001)

        assert (queue.layout.input_arena_offset, b"ordinary") in fake_client.payload_writes
        assert queue.layout.input_desc_offset in [offset for offset, _data in fake_client.payload_writes]
        assert fake_client.counters[queue.layout.input_desc_tail_offset] == 1
        assert fake_client.counters.get(L3L2_QUEUE_L3_ABORT_FLAG_OFFSET, 0) == 0
        assert len(orch._o._buffers) == alloc_count
        assert queue.region.descriptor_scalars()[1] == 1
    finally:
        _close(worker, shm)


def test_l3_host_mapped_queue_ordinary_input_uses_direct_payload_write(monkeypatch):
    orch = _FakeCOrch()
    layout = make_l3_l2_queue_layout(4, 128, 128)
    desc = L3L2OrchRegionDesc(
        magic_version=0x4C334C3200020000,
        region_id=1,
        payload_base=0x1000_0000,
        payload_bytes=layout.payload_bytes,
        counter_base=0x1000_0000 + ((layout.payload_bytes + 63) // 64) * 64,
        counter_bytes=layout.counter_bytes,
    )
    region = L3L2OrchRegion(
        object(),
        0,
        desc,
        L3HostRegionMapping(
            worker_id=0,
            region_id=1,
            access_profile=L3L2RegionAccessProfile.SIM_POSIX_SHM,
            total_bytes=desc.counter_base - desc.payload_base + desc.counter_bytes,
            payload_offset=0,
            payload_bytes=desc.payload_bytes,
            counter_offset=desc.counter_base - desc.payload_base,
            counter_bytes=desc.counter_bytes,
            handle=44,
        ),
    )
    queue = L3L2Queue(
        orch,
        region,
        layout,
        orch.alloc([24], DataType.UINT8),
        orch.alloc([8], DataType.UINT8),
        orch.alloc([L3L2_QUEUE_DESC_SLOT_BYTES], DataType.UINT8),
    )
    alloc_count = len(orch._buffers)
    payload_writes: list[tuple[int, bytes]] = []
    counters: dict[int, int] = {}

    def payload_write(_handle: int, offset: int, src: int, nbytes: int) -> None:
        payload_writes.append((int(offset), ctypes.string_at(int(src), int(nbytes))))

    def counter_notify(_handle: int, offset: int, value: int, _op: int) -> None:
        counters[int(offset)] = int(value)

    monkeypatch.setattr(l3_l2_orch_comm, "_l3_host_mapped_payload_write", payload_write)
    monkeypatch.setattr(
        l3_l2_orch_comm,
        "_l3_host_mapped_counter_test",
        lambda _h, off, _v, _cmp: (False, counters.get(off, 0)),
    )
    monkeypatch.setattr(l3_l2_orch_comm, "_l3_host_mapped_counter_notify", counter_notify)

    queue.input.enqueue(b"ordinary", nbytes=8, timeout=0.001)

    assert (layout.input_arena_offset, b"ordinary") in payload_writes
    assert len(orch._buffers) == alloc_count
    assert counters[region._l3_host_mapping.counter_offset + layout.input_desc_tail_offset] == 1


def test_direct_mapped_ordinary_host_bytearray_does_not_allocate_queue_buffer():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        orch._o.fail_next_alloc = True
        fake_client.requests.clear()
        fake_client.payload_writes.clear()

        queue.input.enqueue(bytearray(b"ordinary"), nbytes=8, timeout=0.001)

        assert (queue.layout.input_arena_offset, b"ordinary") in fake_client.payload_writes
        assert fake_client.counters.get(queue.layout.input_desc_tail_offset, 0) == 1
        assert fake_client.counters.get(L3L2_QUEUE_L3_ABORT_FLAG_OFFSET, 0) == 0
        assert orch._o.fail_next_alloc is True
        assert queue.region.descriptor_scalars()[1] == 1
    finally:
        orch._o.fail_next_alloc = False
        _close(worker, shm)


def test_output_read_into_registered_tensor_uses_fast_path_and_release_notifies_head():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        _publish_output(fake_client, queue, payload=b"abcdefghijklmnop")
        output = orch.alloc([16], DataType.UINT8)

        handle = queue.output.peek(timeout=0.001)
        queue.output.read_into(handle, output)
        queue.output.release(handle)

        assert ctypes.string_at(int(output.data), 16) == b"abcdefghijklmnop"
        assert fake_client.counters[queue.layout.output_desc_head_offset] == 1
    finally:
        _close(worker, shm)


def test_dequeue_into_reads_and_releases_output():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        _publish_output(fake_client, queue, payload=b"abcdefghijklmnop")
        output = orch.alloc([16], DataType.UINT8)

        message = queue.output.dequeue_into(output, timeout=0.001)

        assert message.seq == 1
        assert message.opcode == L3L2QueueOpcode.DATA
        assert ctypes.string_at(int(output.data), 16) == b"abcdefghijklmnop"
        assert fake_client.counters[queue.layout.output_desc_head_offset] == 1
    finally:
        _close(worker, shm)


def test_output_error_opcode_is_delivered_without_poison():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        _publish_output(fake_client, queue, payload=b"error-detail", opcode=int(L3L2QueueOpcode.ERROR))
        output = orch.alloc([12], DataType.UINT8)

        message = queue.output.dequeue_into(output, timeout=0.001)

        assert message.opcode == L3L2QueueOpcode.ERROR
        assert ctypes.string_at(int(output.data), 12) == b"error-detail"
        assert fake_client.counters[queue.layout.output_desc_head_offset] == 1
        assert fake_client.counters.get(L3L2_QUEUE_L3_ABORT_FLAG_OFFSET, 0) == 0
    finally:
        _close(worker, shm)


def test_try_dequeue_into_empty_returns_none_without_abort():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        output = orch.alloc([16], DataType.UINT8)
        fake_client.requests.clear()

        assert queue.output.try_dequeue_into(output) is None

        assert fake_client.counters.get(queue.layout.output_desc_head_offset, 0) == 0
        assert all(
            not (
                req.cmd == "counter_notify"
                and req.counter_addr == queue.region.descriptor.counter_base + L3L2_QUEUE_L3_ABORT_FLAG_OFFSET
            )
            for req, _timeout in fake_client.requests
        )
    finally:
        _close(worker, shm)


def test_output_read_into_ordinary_buffer_uses_direct_payload_read():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        _publish_output(fake_client, queue, payload=b"abcdefghijklmnop")
        handle = queue.output.peek(timeout=0.001)
        output = bytearray(16)
        fake_client.requests.clear()

        queue.output.read_into(handle, output)
        queue.output.release(handle)

        assert bytes(output) == b"abcdefghijklmnop"
        assert any(req.cmd == "payload_read" for req, _timeout in fake_client.requests)
        assert fake_client.counters[queue.layout.output_desc_head_offset] == 1
        assert fake_client.counters.get(L3L2_QUEUE_L3_ABORT_FLAG_OFFSET, 0) == 0
    finally:
        _close(worker, shm)


def test_output_read_rejects_readonly_ordinary_buffer_before_release():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        _publish_output(fake_client, queue, payload=b"abcdefghijklmnop")
        handle = queue.output.peek(timeout=0.001)
        fake_client.requests.clear()

        with pytest.raises(ValueError, match="writable"):
            queue.output.read_into(handle, b"readonly-read-buf")

        assert all(req.cmd != "payload_read" for req, _timeout in fake_client.requests)
        assert fake_client.counters.get(queue.layout.output_desc_head_offset, 0) == 0
        assert fake_client.counters.get(L3L2_QUEUE_L3_ABORT_FLAG_OFFSET, 0) == 0
    finally:
        _close(worker, shm)


def test_output_release_inactive_handle_poisons_and_sets_l3_abort_flag():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        _publish_output(fake_client, queue, payload=b"abcdefghijklmnop")
        handle = queue.output.peek(timeout=0.001)
        wrong = L3L2QueueMessage(handle.seq + 1, handle.opcode, handle.payload_offset, handle.payload_nbytes)
        fake_client.requests.clear()

        with pytest.raises(RuntimeError, match="not active"):
            queue.output.release(wrong)

        assert fake_client.counters[L3L2_QUEUE_L3_ABORT_FLAG_OFFSET] == 1
        with pytest.raises(RuntimeError, match="poisoned"):
            queue.output.try_peek()
    finally:
        _close(worker, shm)


def test_output_stop_descriptor_poisons_and_sets_l3_abort_flag():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        _publish_output(fake_client, queue, opcode=int(L3L2QueueOpcode.STOP))

        with pytest.raises(RuntimeError, match="cannot be STOP"):
            queue.output.peek(timeout=0.001)

        assert fake_client.counters[L3L2_QUEUE_L3_ABORT_FLAG_OFFSET] == 1
    finally:
        _close(worker, shm)


def test_zero_byte_output_descriptor_with_nonzero_offset_poisons_and_sets_l3_abort_flag():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        _publish_output(fake_client, queue, payload_offset=queue.layout.output_arena_offset)

        with pytest.raises(RuntimeError, match="zero-byte.*nonzero"):
            queue.output.peek(timeout=0.001)

        assert fake_client.counters[L3L2_QUEUE_L3_ABORT_FLAG_OFFSET] == 1
    finally:
        _close(worker, shm)


def test_zero_byte_output_read_accepts_none_and_skips_payload_read():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        _publish_output(fake_client, queue, payload=b"")
        handle = queue.output.peek(timeout=0.001)
        fake_client.requests.clear()

        queue.output.read_into(handle, None)
        queue.output.release(handle)

        assert all(req.cmd != "payload_read" for req, _timeout in fake_client.requests)
        assert fake_client.counters[queue.layout.output_desc_head_offset] == 1
    finally:
        _close(worker, shm)


def test_try_enqueue_full_queue_returns_false_without_poison_or_publish():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=2, input_arena_bytes=128, output_arena_bytes=128)
        queue.input.enqueue(None, nbytes=0, timeout=0.001)
        queue.input.enqueue(None, nbytes=0, timeout=0.001)
        fake_client.requests.clear()
        fake_client.payload_writes.clear()

        assert queue.input.try_enqueue(None, nbytes=0) is False

        assert fake_client.payload_writes == []
        assert fake_client.counters[queue.layout.input_desc_tail_offset] == 2
        assert fake_client.counters.get(L3L2_QUEUE_L3_ABORT_FLAG_OFFSET, 0) == 0
    finally:
        _close(worker, shm)


def test_try_enqueue_full_queue_ordinary_buffer_does_not_stage():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=2, input_arena_bytes=128, output_arena_bytes=128)
        queue.input.enqueue(None, nbytes=0, timeout=0.001)
        queue.input.enqueue(None, nbytes=0, timeout=0.001)
        alloc_count = len(orch._o._buffers)
        fake_client.requests.clear()
        fake_client.payload_writes.clear()

        assert queue.input.try_enqueue(bytearray(b"x"), nbytes=1) is False

        assert fake_client.payload_writes == []
        assert len(orch._o._buffers) == alloc_count
        assert fake_client.counters[queue.layout.input_desc_tail_offset] == 2
        assert fake_client.counters.get(L3L2_QUEUE_L3_ABORT_FLAG_OFFSET, 0) == 0
    finally:
        _close(worker, shm)


def test_enqueue_after_stop_rejects_locally_without_polling_or_abort():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        queue.request_stop(timeout=0.001)
        fake_client.requests.clear()

        assert queue.input.try_enqueue(None, nbytes=0) is False
        with pytest.raises(RuntimeError, match="stopped"):
            queue.input.enqueue(None, nbytes=0, timeout=0.001)

        assert fake_client.requests == []
        assert fake_client.counters.get(L3L2_QUEUE_L3_ABORT_FLAG_OFFSET, 0) == 0
    finally:
        _close(worker, shm)


def test_try_enqueue_payload_larger_than_arena_returns_false_without_poison_or_publish():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        host = orch.alloc([256], DataType.UINT8)
        fake_client.requests.clear()
        fake_client.payload_writes.clear()

        assert queue.input.try_enqueue(host, nbytes=256) is False

        assert fake_client.payload_writes == []
        assert fake_client.counters.get(queue.layout.input_desc_tail_offset, 0) == 0
        assert fake_client.counters.get(L3L2_QUEUE_L3_ABORT_FLAG_OFFSET, 0) == 0
    finally:
        _close(worker, shm)


def test_try_enqueue_wraparound_arena_full_ordinary_buffer_does_not_stage_or_advance_tail():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        first = orch.alloc([112], DataType.UINT8)
        queue.input.enqueue(first, nbytes=112, timeout=0.001)
        alloc_count = len(orch._o._buffers)
        old_payload_tail = queue._input_payload_tail
        fake_client.requests.clear()
        fake_client.payload_writes.clear()

        assert queue.input.try_enqueue(bytearray(b"x" * 32), nbytes=32) is False

        assert fake_client.payload_writes == []
        assert len(orch._o._buffers) == alloc_count
        assert queue._input_payload_tail == old_payload_tail
        assert fake_client.counters[queue.layout.input_desc_tail_offset] == 1
        assert fake_client.counters.get(L3L2_QUEUE_L3_ABORT_FLAG_OFFSET, 0) == 0
    finally:
        _close(worker, shm)


def test_output_payload_offset_mismatch_poisons_before_payload_read():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        _publish_output(
            fake_client,
            queue,
            payload=b"abcdefghijklmnop",
            payload_offset=queue.layout.output_arena_offset + 16,
        )
        fake_client.requests.clear()

        with pytest.raises(RuntimeError, match="payload.*mismatch"):
            queue.output.peek(timeout=0.001)

        assert fake_client.counters[L3L2_QUEUE_L3_ABORT_FLAG_OFFSET] == 1
        assert all(
            not (req.cmd == "payload_read" and req.payload_offset == queue.layout.output_arena_offset + 16)
            for req, _timeout in fake_client.requests
        )
    finally:
        _close(worker, shm)


def test_enqueue_payload_write_failure_sets_l3_abort_flag():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        host = orch.alloc([16], DataType.UINT8)
        fake_client.fail_next_cmd = "payload_write"

        with pytest.raises(RuntimeError, match="injected failure"):
            queue.input.enqueue(host, nbytes=16, timeout=0.001)

        assert fake_client.counters[L3L2_QUEUE_L3_ABORT_FLAG_OFFSET] == 1
        with pytest.raises(RuntimeError, match="poisoned"):
            queue.input.try_enqueue(None, nbytes=0)
    finally:
        _close(worker, shm)


def test_timeout_without_peer_abort_flag_returns_timeout_and_keeps_queue_live():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        fake_client.requests.clear()

        with pytest.raises(TimeoutError, match="timed out"):
            queue.output.peek(timeout=0.0001)

        assert queue.region.descriptor_scalars()[1] == 1
        assert all(
            not (
                req.cmd == "counter_notify"
                and req.counter_addr == queue.region.descriptor.counter_base + L3L2_QUEUE_L3_ABORT_FLAG_OFFSET
            )
            for req, _timeout in fake_client.requests
        )
    finally:
        _close(worker, shm)


def test_timeout_with_peer_abort_flag_reports_remote_aborted_without_setting_own_flag():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        fake_client.peer_abort = True
        fake_client.requests.clear()

        with pytest.raises(RuntimeError, match="remote.*abort"):
            queue.output.peek(timeout=0.0001)

        with pytest.raises(RuntimeError, match="remote.*abort"):
            queue.input.try_enqueue(None, nbytes=0)
        assert all(
            not (
                req.cmd == "counter_notify"
                and req.counter_addr == queue.region.descriptor.counter_base + L3L2_QUEUE_L3_ABORT_FLAG_OFFSET
            )
            for req, _timeout in fake_client.requests
        )
    finally:
        _close(worker, shm)


def test_expired_queue_rejects_later_operations_without_abort_flag():
    orch, worker, shm, fake_client = _make_orchestrator()
    try:
        queue = orch.create_l3_l2_queue(worker_id=0, depth=4, input_arena_bytes=128, output_arena_bytes=128)
        queue.region._expire()
        fake_client.requests.clear()

        with pytest.raises(RuntimeError, match="expired"):
            queue.input.try_enqueue(None, nbytes=0)
        with pytest.raises(RuntimeError, match="expired"):
            queue.output.try_peek()

        assert fake_client.requests == []
        assert fake_client.counters.get(L3L2_QUEUE_L3_ABORT_FLAG_OFFSET, 0) == 0
    finally:
        _close(worker, shm)
