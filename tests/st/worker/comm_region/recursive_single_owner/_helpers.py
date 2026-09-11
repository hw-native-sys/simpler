# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Private L3/L4 kernel and host-handshake fixtures for delegated-region ST."""

from __future__ import annotations

import contextvars
import ctypes
import multiprocessing
import os
import struct
from collections.abc import Callable
from enum import IntEnum
from typing import Any

from simpler import comm_provider
from simpler.comm_endpoints import DEVICE_AICPU, HOST_CPU, RegionLayoutSpec, SingleOwner, at
from simpler.comm_provider import ProviderRegionStore, RegionPartKind
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, ChipCallable, CoreCallable, DataType, TaskArgs, scalar_to_uint64
from simpler.worker import Worker, attach_exception_note
from simpler.worker_chip_orch_comm import (
    NotifyOp,
    WaitCmp,
    WorkerChipOrchRegion,
    worker_chip_orch_region_desc_from_local_views,
)

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

_RUNTIME = "tensormap_and_ringbuffer"
_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH_SRC = os.path.join(_HERE, "kernels", "orchestration", "recursive_single_owner_orch.cpp")
_AIV_SRC = os.path.join(_HERE, "kernels", "aiv", "kernel_transform.cpp")
_HEADER = struct.Struct("<QII")
_HEADER_BYTES = 64
_NUMEL = 128 * 128
_NBYTES = _NUMEL * 4
_INPUT_OFFSET = _HEADER_BYTES
_OUTPUT_OFFSET = _INPUT_OFFSET + _NBYTES
_PAYLOAD_BYTES = _OUTPUT_OFFSET + _NBYTES
_DATA_READY_COUNTER = 0
_COMPLETION_COUNTER = 64
_COUNTER_BYTES = 128
_SCALAR = ctypes.c_float(7.0)

SUCCESS_PLATFORMS = ["a2a3sim", "a2a3", "a5sim", "a5"]
FAULT_PLATFORMS = ["a2a3sim", "a5sim"]


def build_chip_callable(platform: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    inc_dirs = kc.get_orchestration_include_dirs(_RUNTIME)
    extra_common = [str(kc.project_root / "src" / "common")]
    aiv = kc.compile_incore(
        _AIV_SRC,
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=inc_dirs,
    )
    if not platform.endswith("sim"):
        aiv = extract_text_section(aiv)
    orch = kc.compile_orchestration(
        runtime_name=_RUNTIME,
        source_path=_ORCH_SRC,
        extra_include_dirs=extra_common,
    )
    return ChipCallable.build(
        signature=[],
        func_name="recursive_single_owner_orchestration",
        binary=orch,
        children=[(0, CoreCallable.build(signature=[D.IN, D.OUT], binary=aiv))],
    )


def _float_view(handle):
    return (ctypes.c_float * _NUMEL).from_address(int(handle.base))


def _byte_view(handle):
    return (ctypes.c_uint8 * int(handle.nbytes)).from_address(int(handle.base))


def _write_header(header_tensor, seq: int, opcode: int) -> None:
    buf = _byte_view(header_tensor)
    for i in range(_HEADER_BYTES):
        buf[i] = 0
    _HEADER.pack_into(buf, 0, seq, opcode, 0)


def _fill_input(input_tensor, round_idx: int) -> list[float]:
    values = _float_view(input_tensor)
    expected = []
    for i in range(_NUMEL):
        value = float(round_idx * 1000 + (i % 251)) / 16.0
        values[i] = value
        expected.append(value + float(_SCALAR.value))
    return expected


def _assert_output_matches(output_tensor, expected: list[float]) -> None:
    values = _float_view(output_tensor)
    for i, want in enumerate(expected):
        got = float(values[i])
        assert abs(got - want) <= 1e-5, f"output[{i}] expected {want}, got {got}"


def _handshake(orch_handle, region, round_idx: int) -> None:
    header = orch_handle.alloc([_HEADER_BYTES], DataType.UINT8)
    host_input = orch_handle.alloc([_NUMEL], DataType.FLOAT32)
    host_output = orch_handle.alloc([_NUMEL], DataType.FLOAT32)
    data_ready = region.counter(_DATA_READY_COUNTER)
    completion = region.counter(_COMPLETION_COUNTER)
    expected = _fill_input(host_input, round_idx)
    region.payload_write(_INPUT_OFFSET, host_input, nbytes=_NBYTES)
    _write_header(header, round_idx, 1)
    region.payload_write(0, header, nbytes=_HEADER.size)
    data_ready.notify(round_idx, NotifyOp.Set)
    snapshot = completion.test(round_idx, WaitCmp.GE)
    if not snapshot.matched:
        completion.wait(round_idx, WaitCmp.GE, timeout=5.0)
    region.payload_read(_OUTPUT_OFFSET, host_output, nbytes=_NBYTES)
    _assert_output_matches(host_output, expected)
    stop_seq = round_idx + 1
    _write_header(header, stop_seq, 2)
    region.payload_write(0, header, nbytes=_HEADER.size)
    data_ready.notify(stop_seq, NotifyOp.Set)


def _chip_task_args(region) -> TaskArgs:
    task_args = TaskArgs()
    for scalar in region.descriptor_scalars():
        task_args.add_scalar(int(scalar))
    task_args.add_scalar(_INPUT_OFFSET)
    task_args.add_scalar(_OUTPUT_OFFSET)
    task_args.add_scalar(_NUMEL)
    task_args.add_scalar(DataType.FLOAT32.value)
    task_args.add_scalar(_NBYTES)
    task_args.add_scalar(scalar_to_uint64(_SCALAR))
    task_args.add_scalar(_DATA_READY_COUNTER)
    task_args.add_scalar(_COMPLETION_COUNTER)
    return task_args


def _stream_config() -> CallConfig:
    config = CallConfig()
    config.aicpu_thread_num = 2
    return config


def _install_release_probe(worker: Worker) -> list[dict[str, object]]:
    releases: list[dict[str, object]] = []
    original = worker._dispatch_delegated_release

    def _probe(*, session_instance_id, transaction_id, provider_path):
        releases.append(
            {
                "session_instance_id": bytes(session_instance_id),
                "transaction_id": int(transaction_id),
                "provider_path": bytes(provider_path),
            }
        )
        return original(
            session_instance_id=session_instance_id,
            transaction_id=transaction_id,
            provider_path=provider_path,
        )

    worker._dispatch_delegated_release = _probe
    return releases


def _assert_delegated_live_region(region, provider_path: bytes, transaction_id: int) -> None:
    instance = region._instance
    assert instance._delegated_release_edge is True
    assert instance._delegated_allocation_committed is True
    assert instance._delegated_provider_path == provider_path
    assert instance._delegated_transaction_id == transaction_id
    assert instance._payload_local_view is not None
    assert instance._counter_local_view is not None
    assert int(instance._payload_local_view.logical_bytes) == _PAYLOAD_BYTES
    assert int(instance._counter_local_view.logical_bytes) == _COUNTER_BYTES
    assert int(instance._payload_local_view.local_base) != int(instance._counter_local_view.local_base)


def _assert_clean_session(worker: Worker, *, next_transaction_id: int, releases: list[dict[str, object]]) -> None:
    registry = worker._region_instance_registry
    assert registry._next_delegated_transaction_id == next_transaction_id
    assert registry._instances == {}
    assert registry._delegated_admission_closed is False
    assert [item["transaction_id"] for item in releases] == list(range(1, next_transaction_id))
    assert worker._delegated_session_fatal is None


def create_live_region(orch_handle, provider_path: str):
    worker = orch_handle._worker
    if int(worker.level) == 3:
        return orch_handle.create_worker_chip_region(
            worker_id=0,
            payload_bytes=_PAYLOAD_BYTES,
            counter_bytes=_COUNTER_BYTES,
        )
    root_path = f"L{int(worker.level)}"
    provider = at(str(provider_path), DEVICE_AICPU)
    instance = worker._materialize_region_instance(
        (at(root_path, HOST_CPU), provider),
        SingleOwner(provider=provider),
        RegionLayoutSpec(payload_bytes=_PAYLOAD_BYTES, counter_bytes=_COUNTER_BYTES),
    )
    payload_view = instance.local_view(RegionPartKind.PAYLOAD)
    counter_view = instance.local_view(RegionPartKind.COUNTER)
    if payload_view is None or counter_view is None:
        raise RuntimeError("materialized instance is missing local views")
    desc = worker_chip_orch_region_desc_from_local_views(instance.provider_resource_id, payload_view, counter_view)
    return WorkerChipOrchRegion(worker, instance, desc)


def run_two_lifecycles(
    worker: Worker,
    *,
    provider_path: str,
    submit_chip: Callable[[Any, Any, CallConfig], None],
    recorder: _LifecycleRecorder,
) -> None:
    path = provider_path.encode("ascii")
    seen: list[int] = []
    releases = _install_release_probe(worker)

    def orch(orch_handle, _args, cfg):
        assert not hasattr(orch_handle, "create_region")
        region = create_live_region(orch_handle, provider_path)
        transaction_id = int(region._instance._delegated_transaction_id)
        seen.append(transaction_id)
        _assert_delegated_live_region(region, path, transaction_id)
        submit_chip(orch_handle, region, cfg)
        _handshake(orch_handle, region, 1)

    worker.run(orch, args=None, config=_stream_config())
    worker.run(orch, args=None, config=_stream_config())
    assert seen == [1, 2]
    _assert_clean_session(worker, next_transaction_id=3, releases=releases)
    _assert_lifecycle_events(recorder)


def make_l3_submit(chip_handle) -> Callable[[Any, Any, CallConfig], None]:
    def _submit(orch_handle, region, cfg):
        orch_handle.submit_next_level(chip_handle, _chip_task_args(region), cfg, worker=0)

    return _submit


def make_l3_forward(chip_handle):
    def l3_forward(orch, args, cfg):
        chip_args = TaskArgs()
        for index in range(args.scalar_count()):
            chip_args.add_scalar(int(args.scalar(index)))
        orch.submit_next_level(chip_handle, chip_args, cfg, worker=0)

    return l3_forward


def make_l4_submit(l3_handle, l3_worker_id: int) -> Callable[[Any, Any, CallConfig], None]:
    def _submit(orch_handle, region, cfg):
        orch_handle.submit_next_level(l3_handle, _chip_task_args(region), cfg, worker=int(l3_worker_id))

    return _submit


class _LifecycleEventKind(IntEnum):
    IMPORT_PAYLOAD = 1
    IMPORT_COUNTER = 2
    CLOSE_PAYLOAD = 3
    CLOSE_COUNTER = 4
    RELEASE_DISPATCH = 5
    RELEASE_ONCE_PAYLOAD = 6
    RELEASE_ONCE_COUNTER = 7
    RELEASE_RESULT = 8


_RECORDER_EVENT_LIMIT = 16
_RECORDER_RECORD_FIELDS = 3
_RECORDER_SLOT_COUNT = 1 + _RECORDER_EVENT_LIMIT * _RECORDER_RECORD_FIELDS
_RELEASE_RESOURCE_ID: contextvars.ContextVar[int] = contextvars.ContextVar(
    "recursive_single_owner_release_resource_id", default=0
)
_LIFECYCLE_KIND_ORDER = (
    _LifecycleEventKind.IMPORT_PAYLOAD,
    _LifecycleEventKind.IMPORT_COUNTER,
    _LifecycleEventKind.CLOSE_PAYLOAD,
    _LifecycleEventKind.CLOSE_COUNTER,
    _LifecycleEventKind.RELEASE_DISPATCH,
    _LifecycleEventKind.RELEASE_ONCE_PAYLOAD,
    _LifecycleEventKind.RELEASE_ONCE_COUNTER,
    _LifecycleEventKind.RELEASE_RESULT,
)


class _LifecycleRecorder:
    def __init__(self) -> None:
        self._storage = multiprocessing.Array(ctypes.c_uint64, _RECORDER_SLOT_COUNT, lock=True)
        self.restore: Callable[[], None] | None = None

    def append(self, kind: int, provider_resource_id: int, part_kind: int) -> None:
        with self._storage.get_lock():
            count = int(self._storage[0])
            if count >= _RECORDER_EVENT_LIMIT:
                raise AssertionError("lifecycle recorder overflow")
            base = 1 + count * _RECORDER_RECORD_FIELDS
            self._storage[base] = int(kind)
            self._storage[base + 1] = int(provider_resource_id)
            self._storage[base + 2] = int(part_kind)
            self._storage[0] = count + 1

    def events(self) -> list[tuple[int, int, int]]:
        with self._storage.get_lock():
            count = int(self._storage[0])
            records = []
            for index in range(count):
                base = 1 + index * _RECORDER_RECORD_FIELDS
                records.append((int(self._storage[base]), int(self._storage[base + 1]), int(self._storage[base + 2])))
            return records


class _LeaseProxy:
    def __init__(self, inner: Any, recorder: _LifecycleRecorder, resource_id: int, part: RegionPartKind) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_recorder", recorder)
        object.__setattr__(self, "_resource_id", int(resource_id))
        object.__setattr__(self, "_part", part)

    @property
    def handle(self) -> Any:
        inner = object.__getattribute__(self, "_inner")
        return getattr(inner, "handle", inner)

    @property
    def base(self) -> Any:
        inner = object.__getattribute__(self, "_inner")
        return getattr(inner, "base")

    @property
    def nbytes(self) -> Any:
        inner = object.__getattribute__(self, "_inner")
        return getattr(inner, "nbytes")

    def close(self) -> None:
        inner = object.__getattribute__(self, "_inner")
        closer = getattr(inner, "close", None)
        if closer is not None:
            closer()
        part = object.__getattribute__(self, "_part")
        kind = (
            _LifecycleEventKind.CLOSE_PAYLOAD if part is RegionPartKind.PAYLOAD else _LifecycleEventKind.CLOSE_COUNTER
        )
        object.__getattribute__(self, "_recorder").append(
            int(kind), int(object.__getattribute__(self, "_resource_id")), int(part)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)


class _ShellProxy:
    def __init__(self, inner: Any, part: RegionPartKind, recorder: _LifecycleRecorder) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_part", part)
        object.__setattr__(self, "_recorder", recorder)

    def materialize(self) -> Any:
        return object.__getattribute__(self, "_inner").materialize()

    def mapping_bytes(self) -> Any:
        return object.__getattribute__(self, "_inner").mapping_bytes()

    def import_capability(self) -> Any:
        return object.__getattribute__(self, "_inner").import_capability()

    def local_base(self) -> Any:
        return object.__getattribute__(self, "_inner").local_base()

    def zero_bytes(self, *args: Any, **kwargs: Any) -> Any:
        return object.__getattribute__(self, "_inner").zero_bytes(*args, **kwargs)

    def release_once(self) -> Any:
        resource_id = int(_RELEASE_RESOURCE_ID.get())
        if resource_id == 0:
            raise AssertionError("release_once observed a zero provider resource context")
        result = object.__getattribute__(self, "_inner").release_once()
        part = object.__getattribute__(self, "_part")
        kind = (
            _LifecycleEventKind.RELEASE_ONCE_PAYLOAD
            if part is RegionPartKind.PAYLOAD
            else _LifecycleEventKind.RELEASE_ONCE_COUNTER
        )
        object.__getattribute__(self, "_recorder").append(int(kind), resource_id, int(part))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)


def close_owned_workers(primary: BaseException | None, *workers: Any) -> None:
    first_cleanup: BaseException | None = None
    for worker in workers:
        if worker is None:
            continue
        try:
            worker.close()
        except BaseException as cleanup:
            if primary is not None:
                try:
                    attach_exception_note(primary, f"{type(cleanup).__name__}: {cleanup}")
                except BaseException:
                    pass
            elif first_cleanup is None:
                first_cleanup = cleanup
            else:
                try:
                    attach_exception_note(first_cleanup, f"{type(cleanup).__name__}: {cleanup}")
                except BaseException:
                    pass
    if primary is None and first_cleanup is not None:
        raise first_cleanup


def _lookup_release_resource_id(worker: Worker, session_instance_id: bytes, transaction_id: int) -> int:
    session = bytes(session_instance_id)
    tx = int(transaction_id)
    for instance in worker._region_instance_registry._instances.values():
        if instance._delegated_session_instance_id == session and int(instance._delegated_transaction_id) == tx:
            return int(instance._provider_resource_id)
    raise AssertionError("delegated release has no matching instance resource id")


def install_lifecycle_recorder(worker: Worker) -> _LifecycleRecorder:
    recorder = _LifecycleRecorder()
    original_import = worker._import_region_part_lease
    original_release = worker._dispatch_delegated_release
    original_store_release = ProviderRegionStore.release
    original_dispatcher = comm_provider._closed_part_dispatcher
    import_count = {"n": 0}

    def _import(worker_id: int, resource_id: int, export: Any, **kwargs) -> Any:
        lease = original_import(worker_id, resource_id, export, **kwargs)
        import_count["n"] += 1
        part = RegionPartKind.PAYLOAD if import_count["n"] % 2 == 1 else RegionPartKind.COUNTER
        kind = (
            _LifecycleEventKind.IMPORT_PAYLOAD if part is RegionPartKind.PAYLOAD else _LifecycleEventKind.IMPORT_COUNTER
        )
        recorder.append(int(kind), int(resource_id), int(part))
        return _LeaseProxy(lease, recorder, int(resource_id), part)

    def _dispatch(*, session_instance_id, transaction_id, provider_path):
        resource_id = _lookup_release_resource_id(worker, session_instance_id, transaction_id)
        recorder.append(int(_LifecycleEventKind.RELEASE_DISPATCH), resource_id, int(RegionPartKind.INVALID))
        result = original_release(
            session_instance_id=session_instance_id,
            transaction_id=transaction_id,
            provider_path=provider_path,
        )
        recorder.append(int(_LifecycleEventKind.RELEASE_RESULT), resource_id, int(RegionPartKind.INVALID))
        return result

    def _store_release(self, provider_resource_id: int):
        token = _RELEASE_RESOURCE_ID.set(int(provider_resource_id))
        try:
            return original_store_release(self, provider_resource_id)
        finally:
            _RELEASE_RESOURCE_ID.reset(token)

    def _dispatcher(context, part, spec):
        inner = original_dispatcher(context, part, spec)
        return _ShellProxy(inner, part, recorder)

    worker._import_region_part_lease = _import
    worker._dispatch_delegated_release = _dispatch
    ProviderRegionStore.release = _store_release  # type: ignore[method-assign]
    comm_provider._closed_part_dispatcher = _dispatcher

    def restore() -> None:
        worker._import_region_part_lease = original_import
        worker._dispatch_delegated_release = original_release
        ProviderRegionStore.release = original_store_release  # type: ignore[method-assign]
        comm_provider._closed_part_dispatcher = original_dispatcher

    recorder.restore = restore
    return recorder


def _assert_lifecycle_events(recorder: _LifecycleRecorder) -> None:
    events = recorder.events()
    assert len(events) == _RECORDER_EVENT_LIMIT
    first = events[:8]
    second = events[8:]
    expected_kinds = [int(kind) for kind in _LIFECYCLE_KIND_ORDER]
    assert [kind for kind, _resource, _part in first] == expected_kinds
    assert [kind for kind, _resource, _part in second] == expected_kinds
    assert first[0][1] != second[0][1]
    assert first[0][1] != 0 and second[0][1] != 0
    expected_parts = (
        int(RegionPartKind.PAYLOAD),
        int(RegionPartKind.COUNTER),
        int(RegionPartKind.PAYLOAD),
        int(RegionPartKind.COUNTER),
        int(RegionPartKind.INVALID),
        int(RegionPartKind.PAYLOAD),
        int(RegionPartKind.COUNTER),
        int(RegionPartKind.INVALID),
    )
    assert [part for _kind, _resource, part in first] == list(expected_parts)
    assert [part for _kind, _resource, part in second] == list(expected_parts)
    assert [resource for _kind, resource, _part in first] == [first[0][1]] * 8
    assert [resource for _kind, resource, _part in second] == [second[0][1]] * 8
