# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Base communicator generations are isolated across Worker lifetimes."""

import logging
import os
import subprocess
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import simpler.worker as worker_mod
from simpler.worker import Worker


def test_recycled_worker_address_does_not_reuse_rootinfo(monkeypatch, tmp_path):
    monkeypatch.setattr(worker_mod, "id", lambda _worker: 0x1234, raising=False)
    first = Worker(level=3, num_sub_workers=0)
    try:
        first_path = first._comm_plan_rootinfo_path()
        assert first._comm_plan_rootinfo_path() == first_path
    finally:
        first.close()
    del first

    stale = tmp_path / Path(first_path).name
    stale.write_bytes(b"predecessor rootinfo")
    marker = tmp_path / f"barrier_{stale.name}_rootinfo_ready_111_0.ready"
    marker.write_text("1")
    second = Worker(level=3, num_sub_workers=0)
    try:
        second_path = second._comm_plan_rootinfo_path()
        assert second_path != first_path
        assert not (tmp_path / Path(second_path).name).exists()
        assert second._comm_plan_rootinfo_path() == second_path
        assert stale.read_bytes() == b"predecessor rootinfo"
        assert marker.read_text() == "1"
    finally:
        second.close()


def test_live_workers_have_independent_rootinfo_paths(monkeypatch):
    monkeypatch.setattr(worker_mod, "id", lambda _worker: 0x1234, raising=False)
    first = Worker(level=3, num_sub_workers=0)
    second = Worker(level=3, num_sub_workers=0)
    try:
        assert first._comm_plan_rootinfo_path() != second._comm_plan_rootinfo_path()
    finally:
        second.close()
        first.close()


def test_base_comm_dispatch_shares_one_path_and_reuses_communicator():
    worker = Worker(level=3, num_sub_workers=0, device_ids=[0, 1])
    requests = {}

    def initialize(chip_idx, request_name):
        request = SharedMemory(name=request_name)
        try:
            request_buf = request.buf
            assert request_buf is not None
            payload = bytes(request_buf)
            rank, nranks = worker_mod._COMM_INIT_HEADER.unpack_from(payload)
            path = payload[worker_mod._COMM_INIT_HEADER.size :].split(b"\0", 1)[0].decode()
            assert chip_idx not in requests
            requests[chip_idx] = (rank, nranks, path)
        finally:
            request.close()

    worker._worker = cast(Any, SimpleNamespace(control_comm_init=initialize))
    try:
        path = worker._comm_plan_rootinfo_path()
        worker._ensure_comm_base()
        worker._ensure_comm_base()
        assert worker._comm_base_ready
        assert requests == {0: (0, 2, path), 1: (1, 2, path)}
    finally:
        worker._worker = None
        worker.close()


@pytest.fixture(scope="module", params=["a2a3", "a5"])
def bootstrap_binary(request, tmp_path_factory):
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    root = Path(__file__).resolve().parents[4]
    build = tmp_path_factory.mktemp("comm_bootstrap")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann"))
    if not (cann / "include/hccl/hccl_comm.h").exists():
        pytest.skip("bootstrap protocol fixture requires CANN headers, no device")
    for engine in ("sdma", "urma", "rdma"):
        header = build / f"pto/comm/async/{engine}/{engine}_workspace_manager.hpp"
        header.parent.mkdir(parents=True)
        header.write_text('#include "comm_workspace_managers.h"\n')
    log = build / "common/unified_log.h"
    log.parent.mkdir()
    log.write_text(
        "#pragma once\n#include <cstdio>\n"
        + "".join(
            f"#define LOG_{level}(...) std::fprintf(stderr, __VA_ARGS__)\n"
            for level in ("ERROR", "WARN", "INFO", "DEBUG")
        )
    )
    isa = Path(ensure_pto_isa_root())
    binary = build / "bootstrap"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-UNDEBUG",
            "-pthread",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            f"-I{build}",
            f"-I{isa}/include",
            f"-I{cann}/include",
            f"-I{cann}/pkg_inc/base",
            f"-I{root}/src/{request.param}/platform/include",
            f"-I{root}/src/common/platform/include",
            f"-I{root}/src/common",
            f"-I{root}/src/common/task_interface",
            f"-I{root}/src/common/worker",
            f"-I{root}/src/{request.param}/platform/onboard/host",
            f"-I{root}/tests/ut/py/support",
            str(root / "tests/ut/py/support/comm_bootstrap.cpp"),
            "-ldl",
            "-lrt",
            "-o",
            str(binary),
        ],
        check=True,
    )
    return binary


@pytest.mark.parametrize("first_rank", [0, 1])
def test_native_bootstrap_ignores_predecessor_files(bootstrap_binary, monkeypatch, tmp_path, first_rank):
    monkeypatch.setattr(worker_mod, "id", lambda _worker: 0x1234, raising=False)
    first = Worker(level=3, num_sub_workers=0)
    try:
        stale = tmp_path / Path(first._comm_plan_rootinfo_path()).name
    finally:
        first.close()
    del first
    second = Worker(level=3, num_sub_workers=0)
    try:
        current = tmp_path / Path(second._comm_plan_rootinfo_path()).name
        subprocess.run([str(bootstrap_binary), str(stale), str(current), str(first_rank)], check=True, timeout=20)
    finally:
        second.close()


def test_bootstrap_timeout_identifies_generation_and_missing_rank(bootstrap_binary, tmp_path):
    current = tmp_path / "rootinfo.bin"
    result = subprocess.run(
        [str(bootstrap_binary), str(current), str(current), "timeout"],
        check=True,
        timeout=10,
        capture_output=True,
        text=True,
    )
    assert "[comm rank 0]" in result.stderr
    assert "waiting for rank 1" in result.stderr
    assert "rootinfo_ready" in result.stderr
    assert f"path={current}" in result.stderr
    assert "token=222 nranks=2" in result.stderr


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(2)
def test_two_rank_worker_bootstrap_with_predecessor_residue(monkeypatch, tmp_path, caplog, st_platform, st_device_ids):
    caplog.set_level(logging.INFO, logger="simpler")
    original_path = Worker._comm_plan_rootinfo_path
    monkeypatch.setattr(worker_mod, "id", lambda _worker: 0x1234, raising=False)
    monkeypatch.setattr(Worker, "_comm_plan_rootinfo_path", lambda self: str(tmp_path / Path(original_path(self)).name))

    def make_worker():
        return Worker(
            level=3,
            num_sub_workers=0,
            device_ids=st_device_ids[:2],
            platform=st_platform,
            runtime="tensormap_and_ringbuffer",
        )

    assert len(st_device_ids) >= 2
    first = make_worker()
    try:
        first.init()
        first._ensure_comm_base()
        stale = Path(first._comm_plan_rootinfo_path())
        rootinfo = stale.read_bytes()
        markers = list(tmp_path.glob(f"barrier_{stale.name}_rootinfo_ready_*_0.ready"))
        assert len(markers) == 1
        marker = markers[0]
        marker_contents = marker.read_bytes()
    finally:
        first.close()
    del first
    stale.write_bytes(rootinfo)
    marker.write_bytes(marker_contents)

    second = make_worker()
    try:
        current = Path(second._comm_plan_rootinfo_path())
        assert current != stale
        second.init()
        second._ensure_comm_base()
        current_rootinfo = current.read_bytes()
        second._ensure_comm_base()
        assert second._comm_base_ready
        assert current.read_bytes() == current_rootinfo
        assert stale.read_bytes() == rootinfo
        assert marker.read_bytes() == marker_contents
    finally:
        second.close()
    assert not current.exists()
    assert not list(tmp_path.glob(f"barrier_{current.name}_*"))
