# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep communication backing isolated after an accepted SDMA request fails registration."""

import ctypes
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, ChipCallable, CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import _pto_isa_compile_cache_token, compile_chip_callable_spec, scene_level
from tests.st.a5.host_build_graph.async_completion.helpers import (
    close_after_failure,
    error_log_cursor,
    failure_evidence,
    read_host_tensor,
    stage_host_tensor,
)
from tests.st.a5.host_build_graph.sdma_defer_wait._error_close_evidence import run_error_close_probe

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


def prepare_error_case(platform, ranks, output):
    output.mkdir(parents=True, exist_ok=True)
    compiled = compile_error_case(platform)
    (output / "callable.bin").write_bytes(ctypes.string_at(compiled.buffer_ptr(), compiled.buffer_size()))
    source = torch.arange(ELEMENTS, dtype=torch.float32) % 251
    sources = [source + rank * 1000 for rank in range(ranks)]
    torch.save(
        {
            "sources": sources,
            "golden": [sources[(rank + 1) % ranks] for rank in range(ranks)],
            "poison": torch.full_like(source, -1),
        },
        output / "inputs.pt",
    )


def run_error_case(st_platform, st_device_ids, output):
    ranks = len(st_device_ids)
    inputs = torch.load(output / "inputs.pt", map_location="cpu", weights_only=True)
    source = inputs["sources"][0]
    assert len(inputs["sources"]) == ranks
    worker = Worker(
        level=3, platform=st_platform, runtime="host_build_graph", device_ids=st_device_ids[:ranks], num_sub_workers=0
    )
    handle = worker.register(ChipCallable.from_bytes((output / "callable.bin").read_bytes()))
    closed = False
    try:
        worker.init()
        sources = [worker.create_buffer(source.nbytes) for _ in range(ranks)]
        for rank, staging in enumerate(sources):
            stage_host_tensor(staging, inputs["sources"][rank])
        poison = worker.create_buffer(source.nbytes)
        stage_host_tensor(poison, inputs["poison"])
        readback = worker.create_buffer(source.nbytes)
        domains = []

        def submit(inject):
            config = CallConfig()
            if inject:
                config.enable_chip_swimlane = 4
                config.output_prefix = str(output / "fault-capture")

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

            return worker.submit(orchestrate, config=config)

        submit(False).wait(timeout=30)
        domain = domains[0]
        for rank in range(ranks):
            worker.copy_from(readback, domain[rank].buffers["destination"])
            torch.testing.assert_close(read_host_tensor(readback, source), inputs["golden"][rank], rtol=0, atol=0)
        cursor = error_log_cursor()
        with pytest.raises(RuntimeError) as error:
            submit(True).wait(timeout=45)
        evidence = failure_evidence(error.value, SimpleNamespace(out="", err=""), cursor)
        (output / "sdma_error.log").write_text(evidence)
        with pytest.raises(RuntimeError, match="window reclamation is unconfirmed"):
            worker.close()
        closed = True
        assert domain.released and not domain.freed
        with pytest.raises(RuntimeError, match=r"requires an initialized \(READY\) worker"):
            worker.create_buffer(4)
        (output / "retention.txt").write_text(f"{ranks} ranks: failed SDMA run retained its domain\n")
    finally:
        if not closed:
            if sys.exc_info()[0] is None:
                worker.close()
            else:
                close_after_failure(worker)


@pytest.mark.manual
@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("host_build_graph")
@pytest.mark.sdma(worker_workspace=False)
@pytest.mark.parametrize(
    "ranks",
    [pytest.param(2, marks=pytest.mark.device_count(2)), pytest.param(4, marks=pytest.mark.device_count(4))],
)
@scene_level(3)
def test_sdma_error_keeps_domain_unreclaimed(st_platform, st_device_ids, ranks, request):
    assert len(st_device_ids) >= ranks
    report = request.config.getoption("xmlpath", default=None)
    artifacts = Path(report).resolve().parent if report else HERE.parents[4] / "outputs"
    directory = artifacts / "sdma-error-close"
    directory.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"ranks-{ranks}-", dir=directory))
    devices = st_device_ids[:ranks]
    prepare_error_case(st_platform, ranks, output)
    run_error_close_probe(
        [
            sys.executable,
            "-m",
            "tests.st.a5.host_build_graph.sdma_defer_wait.test_error_close",
            st_platform,
            ",".join(map(str, devices)),
            str(output),
        ],
        output,
        devices,
        cwd=HERE.parents[4],
    )


if __name__ == "__main__":
    run_error_case(sys.argv[1], [int(device) for device in sys.argv[2].split(",")], Path(sys.argv[3]))
