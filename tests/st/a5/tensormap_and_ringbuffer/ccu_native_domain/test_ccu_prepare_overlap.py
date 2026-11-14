# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hold actual CCU streams at trigger while preparing an independent successor."""

import ctypes
import subprocess
import sys
import time
from pathlib import Path

import pytest
from simpler.ccu import ccu_sdk_root, runtime_build_identity
from simpler.task_interface import ChipWorker

from simpler_setup.runtime_builder import RuntimeBuilder
from tests.st.a5.tensormap_and_ringbuffer.ccu_native_domain.test_ccu_native_domain import _build_callable


class _Workspace(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("generation", ctypes.c_uint64),
        ("controls", ctypes.c_uint64),
        ("stride", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("windows", ctypes.c_uint64),
        ("window_count", ctypes.c_uint32),
        ("window_size", ctypes.c_uint32),
    ]


class _Control(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32),
        ("id", ctypes.c_uint32),
        ("generation", ctypes.c_uint64),
        ("alignment", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("parameter", ctypes.c_uint64),
        ("parameter_size", ctypes.c_uint64),
        ("token", ctypes.c_uint64),
        ("ready", ctypes.c_uint64),
        ("ready_mask", ctypes.c_uint64),
        ("trigger", ctypes.c_uint64),
        ("trigger_mask", ctypes.c_uint64),
        ("done", ctypes.c_uint64),
        ("done_mask", ctypes.c_uint64),
    ]


def _until(predicate, message):
    deadline = time.monotonic() + 60
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(message)


def _rank(rank, device, directory):
    root = Path(directory)
    target = _build_callable(root / f"build-{rank}")
    bins = RuntimeBuilder("a5").get_binaries("tensormap_and_ringbuffer")
    chip = ChipWorker()
    chip.init(device, bins)
    handle = chip.register_callable(target)
    native = chip._impl
    comm = chip.comm_init(rank, 2, str(root / "rootinfo"))
    sdk, isa = runtime_build_identity()
    slot = chip._resolve_handle(handle).slot_id
    with pytest.raises(RuntimeError, match="registration failed"):
        native.register_ccu_callable(comm, 1, slot, [0, 1], rank, sdk, isa)
    with pytest.raises(RuntimeError, match="registration failed"):
        native.register_ccu_callable(comm, 1, slot, [0, 1], rank, bytes(32), isa, 8)
    committed = ctypes.c_uint64()
    native.comm_alloc_domain_windows(comm, 17, [0, 1], rank, 4 * 1024 * 1024, ctypes.addressof(committed), 8)
    assert committed.value == 1
    registration = native.register_ccu_callable(comm, 10 + rank, slot, [0, 1], rank, sdk, isa, 8, 17, 0)
    assert native.control_ccu_run(comm, registration, 8, 0) == 1
    helper = ctypes.CDLL(str(root / "event_access.so"))
    access = helper.ccu_test_event_access
    access.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    access.restype = ctypes.c_int
    binary = str(Path(bins.host_path).parent / "ccu_event_trigger.o").encode()

    def event(address, value=(1 << 64) - 1):
        result = ctypes.c_uint64()
        assert access(binary, address, value, ctypes.byref(result)) == 0
        return result.value

    def read(address, kind):
        result = kind()
        chip.copy_from(ctypes.addressof(result), address, ctypes.sizeof(result))
        return result

    def controls(address):
        workspace = read(address, _Workspace)
        assert workspace.size == 48 and workspace.count == 1 and workspace.stride == ctypes.sizeof(_Control) == 96
        return [read(workspace.controls + i * workspace.stride, _Control) for i in range(1)]

    output = chip.malloc(16)
    first = native.prepare_ccu_run(comm, registration, 1, 101, 17)
    a = controls(first)
    sentinels = []
    for i, control in enumerate(a):
        words = (ctypes.c_uint64 * 3)(output + 8 * i, rank * 100 + i + 1, 0)
        chip.copy_to(control.parameter, ctypes.addressof(words), ctypes.sizeof(words))
        sentinels.append(bytes(words))
    native.control_ccu_run(comm, registration, 1, 1)
    for control in a:
        _until(lambda c=control: event(c.ready) == 1, "CCU A did not reach READY/trigger gate")
    before = [(event(c.ready), event(c.trigger), event(c.done)) for c in a]
    second = native.prepare_ccu_run(comm, registration, 2, 102, 17)
    b = controls(second)
    assert first != second and all(x.parameter != y.parameter for x, y in zip(a, b))
    with pytest.raises(RuntimeError, match="control failed"):
        native.control_ccu_run(comm, registration, 1, 2)
    assert [(event(c.ready), event(c.trigger), event(c.done)) for c in a] == before
    assert [bytes(read(c.parameter, ctypes.c_uint64 * 3)) for c in a] == sentinels
    assert read(first, _Workspace).generation == 1
    (root / f"prepared-{rank}").touch()
    _until(lambda: (root / "release").exists(), "parent did not confirm both successor preparations")
    for generation, records in ((1, a), (2, b)):
        if generation == 2:
            native.control_ccu_run(comm, registration, 1, generation)
        for i, control in enumerate(records):
            _until(lambda c=control: event(c.ready) == 1, "local READY not observed")
            assert event(control.ready, 0) == 0
            words = (ctypes.c_uint64 * 3)(output + 8 * i, generation * 1000 + rank * 100 + i, int(generation == 2))
            chip.copy_to(control.parameter, ctypes.addressof(words), ctypes.sizeof(words))
            event(control.trigger, (1 << 63) | 1)
        for control in records:
            _until(lambda c=control: event(c.done) == 1, "CCU operation did not complete")
            assert event(control.done, 0) == 0
        if generation == 2:
            assert native.control_ccu_run(comm, registration, 9, generation) == 0
        else:
            _until(lambda g=generation: native.control_ccu_run(comm, registration, 3, g) == 0, "CCU stop unconfirmed")
        values = read(output, ctypes.c_uint64 * 1)
        assert list(values) == [generation * 1000 + rank * 100 + i for i in range(1)]
        native.control_ccu_run(comm, registration, 7, generation)
    assert helper.ccu_test_event_close() == 0
    chip.free(output)
    native.comm_release_domain_windows(comm, 17, 2, rank)
    chip.comm_destroy(comm)
    chip.finalize()


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
def test_prepared_successor_preserves_executing_ccu_resources(st_device_ids, tmp_path):
    sdk = ccu_sdk_root()
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            str(Path(__file__).with_name("event_access.cpp")),
            f"-I{sdk / 'include'}",
            f"-L{sdk / 'lib64'}",
            "-lascendcl",
            "-o",
            str(tmp_path / "event_access.so"),
        ],
        check=True,
    )
    processes = []
    streams = []
    try:
        for rank, device in enumerate(st_device_ids[:2]):
            stream = (tmp_path / f"rank-{rank}.log").open("w")
            streams.append(stream)
            processes.append(
                subprocess.Popen(
                    [sys.executable, __file__, str(rank), str(device), str(tmp_path)],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            )

        def prepared():
            for process in processes:
                assert process.poll() in (None, 0), "rank failed before prepare acknowledgement"
            return all((tmp_path / f"prepared-{rank}").exists() for rank in range(2))

        _until(prepared, "successor preparations were not acknowledged")
        (tmp_path / "release").touch()
        assert all(process.wait(timeout=90) == 0 for process in processes)
        print("CCU A held at trigger; prepare B preserved P, Events and generation; B rejected until confirmed A stop")
    finally:
        (tmp_path / "release").touch()
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for stream in streams:
            stream.close()
        for rank in range(len(streams)):
            print((tmp_path / f"rank-{rank}.log").read_text())


if __name__ == "__main__":
    _rank(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
