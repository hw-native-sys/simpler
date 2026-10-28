# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Creation refuses unsupported Host DDR before allocating backing or an identity."""

import ctypes
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from simpler.worker import Worker, _Lifecycle


@pytest.fixture
def worker():
    worker = Worker(level=2, platform="a5", runtime="tensormap_and_ringbuffer")
    worker._lifecycle = _Lifecycle.READY
    worker._chip_worker = SimpleNamespace(
        _validate_execution_context=lambda: None,
        _host_ddr_mapping_supported=False,
    )
    yield worker
    worker._chip_worker = None
    worker.close()


@pytest.mark.parametrize("options", [{}, {"device_readable": False}])
def test_default_keeps_shared_host_backing(worker, options):
    from simpler.task_interface import AddressSpace, DataType  # noqa: PLC0415

    buffer = worker.create_buffer(64, **options)
    try:
        assert buffer.address_space == AddressSpace.HOST
        buffer.shm.buf[:4] = b"test"
        view = buffer.tensor((4,), DataType.INT32, byte_offset=16)
        assert view.byte_offset == 16
        assert bytes(buffer.shm.buf[:4]) == b"test"
    finally:
        worker.release_buffer(buffer)


@pytest.mark.parametrize("environment", [None, "ON", "1"])
def test_missing_mapping_support_rejected_before_allocation(worker, monkeypatch, environment):
    if environment is None:
        monkeypatch.delenv("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", raising=False)
    else:
        monkeypatch.setenv("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", environment)
    allocate = Mock(side_effect=AssertionError("must not allocate"))
    monkeypatch.setattr(worker, "_create_buffer_locked", allocate)
    with pytest.raises(RuntimeError, match="A5 onboard runtime.*Host mapping"):
        worker.create_buffer(64, device_readable=True)
    allocate.assert_not_called()
    assert not worker._buffers
    assert worker._active_ops == 0


def test_old_binding_without_capability_is_not_supported(worker, monkeypatch):
    del worker._chip_worker._host_ddr_mapping_supported
    monkeypatch.setenv("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", "ON")
    with pytest.raises(RuntimeError, match="A5 onboard runtime.*Host mapping"):
        worker.create_buffer(64, device_readable=True)


@pytest.mark.parametrize("level", [3, 4])
def test_host_ddr_refuses_hierarchical_workers(level):
    worker = Worker(level=level)
    try:
        with pytest.raises(TypeError, match="single-process level 2"):
            worker.create_buffer(64, device_readable=True)
    finally:
        worker.close()


def test_option_is_keyword_only(worker):
    with pytest.raises(TypeError):
        worker.create_buffer(64, True)


@pytest.mark.parametrize("option", [None, 1, "False", "ON"])
def test_option_requires_boolean(worker, option):
    with pytest.raises(TypeError, match="device_readable must be a bool"):
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
        ordinary = worker.create_buffer(64, device_readable=False)
        assert ordinary.shm is not None and ordinary.shm.buf is not None
        ordinary.shm.buf[:4] = b"data"
        worker.release_buffer(ordinary)
        for value in ("OFF", "ON"):
            monkeypatch.setenv("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", value)
            if worker._chip_worker._host_ddr_mapping_supported:
                readable = worker.create_buffer(64, device_readable=True)
                assert readable.device_readable
                worker.release_buffer(readable)
            else:
                with pytest.raises(RuntimeError, match="A5 onboard runtime.*Host mapping"):
                    worker.create_buffer(64, device_readable=True)
            assert not worker._buffers
    finally:
        worker.close()
