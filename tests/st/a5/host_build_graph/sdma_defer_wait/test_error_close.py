# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep communication backing isolated after an accepted SDMA request fails registration."""

import sys
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import _pto_isa_compile_cache_token, compile_chip_callable_spec, scene_level
from tests.st.a5.host_build_graph.async_completion.helpers import (
    close_after_failure,
    error_log_cursor,
    failure_evidence,
    read_host_tensor,
    stage_host_tensor,
)

HERE = Path(__file__).resolve().parent
ELEMENTS = 1024 * 1024
DOMAIN = "sdma_error_retention"


def compile_error_case(platform):
    spec = {
        "orchestration": {
            "source": str(HERE / "kernels/orchestration/error_after_post.cpp"),
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.INOUT, D.SCALAR, D.SCALAR, D.SCALAR],
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(HERE / "kernels/aiv/error_after_post.cpp"),
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT, D.SCALAR, D.SCALAR, D.SCALAR],
            }
        ],
    }
    return compile_chip_callable_spec(
        spec, platform, "host_build_graph", (str(HERE), "error_after_post", platform, _pto_isa_compile_cache_token())
    )


@pytest.mark.manual
@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("host_build_graph")
@pytest.mark.sdma(worker_workspace=False)
@pytest.mark.parametrize(
    "ranks",
    [pytest.param(2, marks=pytest.mark.device_count(2)), pytest.param(4, marks=pytest.mark.device_count(4))],
)
@scene_level(3)
def test_sdma_error_keeps_domain_unreclaimed(st_platform, st_device_ids, ranks, tmp_path, capfd, monkeypatch):
    assert len(st_device_ids) >= ranks
    device_logs = tmp_path / "ascend"
    device_logs.mkdir()
    monkeypatch.setenv("ASCEND_PROCESS_LOG_PATH", str(device_logs))
    monkeypatch.setenv("ASCEND_GLOBAL_LOG_LEVEL", "3")
    worker = Worker(
        level=3, platform=st_platform, runtime="host_build_graph", device_ids=st_device_ids[:ranks], num_sub_workers=0
    )
    handle = worker.register(compile_error_case(st_platform))
    closed = False
    try:
        worker.init()
        source = torch.arange(ELEMENTS, dtype=torch.float32) % 251
        sources = [worker.create_buffer(source.nbytes) for _ in range(ranks)]
        for rank, staging in enumerate(sources):
            stage_host_tensor(staging, source + rank * 1000)
        poison = worker.create_buffer(source.nbytes)
        stage_host_tensor(poison, torch.full_like(source, -1))
        readback = worker.create_buffer(source.nbytes)
        domains = []

        def submit(inject):
            def orchestrate(orch, _args, config):
                domain = orch.get_or_allocate_domain(
                    name=DOMAIN,
                    members=list(range(ranks)),
                    window_size=source.nbytes * 2,
                    buffers=[
                        CommBufferSpec(name=name, dtype="float32", count=ELEMENTS, nbytes=source.nbytes)
                        for name in ("source", "destination")
                    ],
                    retain_after_run=True,
                    workspace_engines=("sdma",),
                )
                domains.append(domain)
                for rank in range(ranks):
                    orch.copy_to(domain[rank].buffers["source"], sources[rank])
                    orch.copy_to(domain[rank].buffers["destination"], poison)
                for rank in range(ranks):
                    args = TaskArgs()
                    args.add_tensor(
                        domain[rank].buffers["source"].tensor((ELEMENTS,), DataType.FLOAT32), TensorArgType.INPUT
                    )
                    args.add_tensor(
                        domain[rank].buffers["destination"].tensor((ELEMENTS,), DataType.FLOAT32), TensorArgType.INOUT
                    )
                    args.add_scalar(domain.device_context(rank, workspace_engines=("sdma",)))
                    args.add_scalar(ELEMENTS)
                    args.add_scalar(int(inject))
                    orch.submit_next_level(handle, args, config, worker=rank)

            return worker.submit(orchestrate, config=CallConfig())

        submit(False).wait(timeout=30)
        domain = domains[0]
        for rank in range(ranks):
            worker.copy_from(readback, domain[rank].buffers["destination"])
            torch.testing.assert_close(
                read_host_tensor(readback, source), source + ((rank + 1) % ranks) * 1000, rtol=0, atol=0
            )
        capfd.readouterr()
        cursor = error_log_cursor()
        with pytest.raises(RuntimeError) as error:
            submit(True).wait(timeout=45)
        evidence = failure_evidence(error.value, capfd.readouterr(), cursor)
        (tmp_path / "sdma_error.log").write_text(evidence)
        with pytest.raises(RuntimeError, match="window reclamation is unconfirmed"):
            worker.close()
        closed = True
        assert domain.released and not domain.freed
        for device in st_device_ids[:ranks]:
            logs = list((device_logs / "debug" / f"device-{device}").glob("*.log"))
            assert logs, f"missing device diagnostics for {device}"
            native = "\n".join(path.read_text(errors="replace") for path in logs)
            assert "error=ASYNC_WAIT_OVERFLOW" in native, native
            assert "runtime failed with rc=-102" in native, native
            assert "resolved=0 expected=1" in native, native
            (tmp_path / f"device-{device}-error.log").write_text(native)
        with pytest.raises(RuntimeError, match=r"requires an initialized \(READY\) worker"):
            worker.create_buffer(4)
        (tmp_path / "retention.txt").write_text(f"{ranks} ranks: failed SDMA run retained its domain\n")
    finally:
        if not closed:
            if sys.exc_info()[0] is None:
                worker.close()
            else:
                close_after_failure(worker)
