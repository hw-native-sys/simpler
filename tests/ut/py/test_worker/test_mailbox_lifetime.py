# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native mailbox lifetime after Python cleanup refuses to release a Worker."""

import subprocess
import sys
import textwrap
from multiprocessing.shared_memory import SharedMemory

import pytest
from simpler.worker import _OFF_SHUTDOWN, _mailbox_addr, _mailbox_extent, _mailbox_load_i32, _Worker, mailbox_size


@pytest.mark.parametrize("destroy", ["close", "destructor", "interpreter"])
def test_retained_worker_outlives_python_mailbox_mapping(destroy):
    script = textwrap.dedent(
        """
        import os
        from simpler.worker import Worker

        worker = Worker(level=3, num_sub_workers=1, heap_ring_size=65536)
        worker.init()
        def refuse_cleanup():
            raise RuntimeError("injected unconfirmed Host recovery")
        worker._close_host_group_mappings = refuse_cleanup
        try:
            worker.close()
        except RuntimeError as error:
            assert "injected unconfirmed Host recovery" in str(error)
        else:
            raise AssertionError("cleanup must refuse")
        native = worker._worker
        assert native is not None and worker._sub_pids
        for shm in worker._sub_shms:
            shm.close()
            shm.unlink()
        print("caller mapping released; native Worker retained", flush=True)
        """
    )
    if destroy == "close":
        script += "native.close()\n"
    elif destroy == "destructor":
        script += "worker._worker = None\nworker._orch = None\ndel native\n"
    if destroy != "interpreter":
        script += "\nfor pid in worker._sub_pids:\n    assert os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) == 0\n"
    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True, timeout=30)
    assert "caller mapping released; native Worker retained" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("endpoint", ["sub", "chip", "nested"])
def test_native_mailbox_mapping_survives_closed_descriptor(endpoint):
    original = SharedMemory(create=True, size=mailbox_size(1))
    observer = SharedMemory(name=original.name)
    native = _Worker(3, 65536)
    try:
        extent = _mailbox_extent(original)
        address = native._map_mailbox(original._fd, extent)
        assert address != _mailbox_addr(original)
        if endpoint == "sub":
            native.add_sub_worker(address, extent)
        elif endpoint == "chip":
            native.add_next_level_worker(address, extent)
        else:
            native.add_next_level_worker_at(7, address, extent)
        native.init()
        original.close()
        original.unlink()
        native.close()
        assert _mailbox_load_i32(_mailbox_addr(observer) + _OFF_SHUTDOWN) == 1
    finally:
        native.close()
        original.close()
        observer.close()
        try:
            original.unlink()
        except FileNotFoundError:
            pass


def test_native_mailbox_rejects_invalid_backing():
    original = SharedMemory(create=True, size=4096)
    native = _Worker(3, 65536)
    try:
        for extent in (0, original.size + 1):
            with pytest.raises(ValueError, match="extent"):
                native._map_mailbox(original._fd, extent)
        with pytest.raises(RuntimeError, match="fstat"):
            native._map_mailbox(-1, original.size)
        native.init()
        with pytest.raises(RuntimeError, match="after init"):
            native._map_mailbox(original._fd, original.size)
    finally:
        native.close()
        original.close()
        original.unlink()
