# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Mapping checks the loaded backend without changing ordinary Host allocation."""

import ctypes
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from simpler.task_interface import ArgDirection, CallConfig, DataType, TaskArgs, TensorTransfer
from simpler.worker import Worker, _Lifecycle


@pytest.fixture
def worker():
    worker = Worker(level=2, platform="a5", runtime="tensormap_and_ringbuffer")
    worker._lifecycle = _Lifecycle.READY
    worker._chip_worker = SimpleNamespace(
        _validate_execution_context=lambda: None,
        _host_ddr_mapping_supported=False,
        _execution_environment=None,
        _submit_chip_run_direct=Mock(),
    )
    worker._callable_registry[1] = SimpleNamespace(sig_count=1, sig=lambda _: ArgDirection.IN)
    yield worker
    worker._chip_worker = None
    worker.close()


def map_args(tensor):
    args = TaskArgs()
    args.add_tensor(tensor, transfer=TensorTransfer.MAP)
    return args


def test_default_keeps_shared_host_backing(worker):
    from simpler.task_interface import AddressSpace, DataType  # noqa: PLC0415

    buffer = worker.create_buffer(64)
    try:
        assert buffer.address_space == AddressSpace.HOST
        buffer.shm.buf[:4] = b"test"
        view = buffer.tensor((4,), DataType.INT32, byte_offset=16)
        assert view.byte_offset == 16
        assert bytes(buffer.shm.buf[:4]) == b"test"
    finally:
        worker.release_buffer(buffer)


@pytest.mark.parametrize("environment", [None, "ON", "1"])
def test_missing_mapping_support_rejected_during_prepare_before_native_submit(worker, monkeypatch, environment):
    if environment is None:
        monkeypatch.delenv("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", raising=False)
    else:
        monkeypatch.setenv("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", environment)
    buffer = worker.create_buffer(64)
    tensor = buffer.tensor((4,), DataType.INT32)
    allocate = Mock(side_effect=AssertionError("must not allocate"))
    monkeypatch.setattr(worker, "_create_buffer_locked", allocate)
    with pytest.raises(RuntimeError, match="A5 onboard runtime.*Host mapping"):
        worker._submit_l2_locked(1, map_args(tensor), CallConfig())
    allocate.assert_not_called()
    worker._chip_worker._submit_chip_run_direct.assert_not_called()
    assert list(worker._buffers.values()) == [buffer]
    worker.release_buffer(buffer)
    assert worker._active_ops == 0


def test_old_binding_without_capability_is_not_supported(worker, monkeypatch):
    tensor = worker.create_buffer(64).tensor((4,), DataType.INT32)
    del worker._chip_worker._host_ddr_mapping_supported
    monkeypatch.setenv("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", "ON")
    with pytest.raises(RuntimeError, match="A5 onboard runtime.*Host mapping"):
        worker._submit_l2_locked(1, map_args(tensor), CallConfig())


@pytest.mark.parametrize("level", [3, 4])
def test_host_ddr_refuses_unsupported_hierarchical_workers(level, worker):
    tensor = worker.create_buffer(64).tensor((4,), DataType.INT32)
    hierarchical = Worker(level=level)
    try:
        with pytest.raises(RuntimeError, match="L3 MAP requires local A5"):
            hierarchical._submit_l3_locked(None, map_args(tensor), CallConfig())
    finally:
        hierarchical.close()


def test_option_is_keyword_only(worker):
    with pytest.raises(TypeError):
        worker.create_buffer(64, True)


@pytest.mark.parametrize("option", [None, 1, "False", "ON"])
def test_allocation_policy_option_is_removed(worker, option):
    with pytest.raises(TypeError, match="unexpected keyword"):
        worker.create_buffer(64, device_readable=option)


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
@pytest.mark.parametrize(
    "platform",
    ["a5sim", pytest.param("a5", marks=[pytest.mark.requires_hardware, pytest.mark.platforms(["a5"])])],
)
def test_creation_with_real_loaded_runtime(platform, runtime, request, monkeypatch):
    device = int(str(request.config.getoption("--device")).split("-")[0].split(",")[0])
    worker = Worker(level=2, platform=platform, runtime=runtime, device_id=device)
    try:
        worker.init()
        root = Path(__file__).resolve().parents[3]
        variant = "sim" if platform.endswith("sim") else "onboard"
        library = ctypes.CDLL(str(root / "build/lib/a5" / variant / runtime / "libhost_runtime.so"))
        compiled = all(hasattr(library, symbol) for symbol in ("host_ddr_map_ctx", "host_ddr_unmap_ctx"))
        assert worker._chip_worker is not None
        assert worker._chip_worker._host_ddr_mapping_supported is compiled
        ordinary = worker.create_buffer(64)
        assert ordinary.shm is not None and ordinary.shm.buf is not None
        ordinary.shm.buf[:4] = b"data"
        worker.release_buffer(ordinary)
        for value in ("OFF", "ON"):
            monkeypatch.setenv("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", value)
            if worker._chip_worker._host_ddr_mapping_supported:
                readable = worker.create_buffer(64)
                registry = worker._l2_import_registry().host_mappings
                assert registry is not None
                args = map_args(readable.tensor((4,), DataType.INT32))
                registry.acquire(1, args)
                try:
                    materialized = worker._materialize_l2_args(args)
                    assert materialized.tensor(0).child_memory
                    assert materialized.tensor(0).data != 0
                    assert materialized.transfer(0) == TensorTransfer.NONE
                finally:
                    registry.release(1)
                worker.release_buffer(readable)
            else:
                ordinary = worker.create_buffer(64)
                try:
                    with pytest.raises(RuntimeError, match="A5 onboard runtime.*Host mapping"):
                        registry = worker._l2_import_registry().host_mappings
                        assert registry is not None
                        registry.acquire(1, map_args(ordinary.tensor((4,), DataType.INT32)))
                finally:
                    worker.release_buffer(ordinary)
            assert not worker._buffers
    finally:
        worker.close()
