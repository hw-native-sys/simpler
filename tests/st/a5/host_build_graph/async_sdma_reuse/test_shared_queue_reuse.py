# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two AIV lanes share each default SDK queue through 66+2 retained submissions."""

import json
import warnings
from collections import defaultdict
from functools import cache
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import compile_chip_callable_spec
from tests.st.a5.host_build_graph.async_completion.helpers import error_log_cursor, failure_evidence

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels/shared_queue"
MAX_LANES, POSTS_PER_LANE, WORDS = 128, 33, 16
MAX_POSTS = MAX_LANES * POSTS_PER_LANE
LARGE, GUARD, STRIDE = 1024 * 1024, 16, 48
ALLOCATED = LARGE + 2 * GUARD + (MAX_POSTS - 1) * STRIDE
POISON, META_GUARD = -999.0, 0x13579BDF


def span(index):
    return (GUARD, LARGE) if index == 0 else (LARGE + 2 * GUARD + (index - 1) * STRIDE + GUARD, 8)


def source_values(rank, epoch):
    result = torch.full((ALLOCATED,), POISON, dtype=torch.float32)
    result[GUARD : GUARD + LARGE] = torch.arange(LARGE, dtype=torch.float32) % 251 + rank * 1000000 + epoch * 10000
    for index in range(1, MAX_POSTS):
        start, count = span(index)
        result[start : start + count] = (
            torch.arange(count, dtype=torch.float32) + rank * 1000000 + epoch * 10000 + index * 5
        )
    return result


def initial_metadata():
    data = torch.full((MAX_POSTS + 1, WORDS), META_GUARD, dtype=torch.int64)
    data[:, :8] = 0
    return data


def stage(buffer, value):
    torch.frombuffer(buffer.shm.buf, dtype=value.dtype, count=value.numel()).copy_(value.flatten())


def read(buffer, value):
    return torch.frombuffer(buffer.shm.buf, dtype=value.dtype, count=value.numel()).clone().reshape(value.shape)


@cache
def compile_reuse(platform):
    return compile_chip_callable_spec(
        {
            "orchestration": {
                "source": str(KERNELS / "orchestration/reuse.cpp"),
                "function_name": "aicpu_orchestration_entry",
                "signature": [D.IN, D.INOUT, D.INOUT, D.INOUT] + [D.SCALAR] * 4,
            },
            "incores": [
                {
                    "func_id": index,
                    "source": str(KERNELS / "aiv" / source),
                    "core_type": "aiv",
                    "signature": [D.IN, D.INOUT, D.INOUT, D.INOUT] + [D.SCALAR] * 5,
                }
                for index, source in enumerate(("submit.cpp", "consume.cpp"))
            ],
        },
        platform,
        "host_build_graph",
        (str(HERE), platform, "queue_reuse"),
    )


class QueueReuseRunner:
    def __init__(self, platform, devices, directory, diagnostics):
        self.ranks = len(devices)
        self.directory = directory
        directory.mkdir(parents=True)
        self.diagnostics = diagnostics
        self.closed = False
        self.epoch = 0
        self.identity = None
        self.last = [None] * self.ranks
        self.metadata = [initial_metadata().share_memory_() for _ in devices]
        self.outputs = [torch.full((ALLOCATED,), POISON, dtype=torch.float32).share_memory_() for _ in devices]
        self.worker = Worker(
            level=3, platform=platform, runtime="host_build_graph", device_ids=devices, num_sub_workers=0
        )
        try:
            self.callable = self.worker.register(compile_reuse(platform))
            self.worker.init()
            self.staging = [
                {name: self.worker.create_buffer(ALLOCATED * 4) for name in ("source", "destination")} for _ in devices
            ]
        except BaseException:
            self._close(failing=True)
            raise

    def _args(self, domain, rank, posts, inject_overflow):
        args = TaskArgs()
        args.add_tensor(domain[rank].buffers["source"].tensor((ALLOCATED,), DataType.FLOAT32), TensorArgType.INPUT)
        args.add_tensor(domain[rank].buffers["destination"].tensor((ALLOCATED,), DataType.FLOAT32), TensorArgType.INOUT)
        for data, dtype in ((self.outputs[rank], DataType.FLOAT32), (self.metadata[rank], DataType.INT64)):
            args.add_tensor(self.worker.make_tensor_arg(data, tuple(data.shape), dtype), TensorArgType.INOUT)
        args.add_scalar(domain.device_context(rank, workspace_engines=("sdma",)))
        args.add_scalar((rank + 1) % self.ranks)
        args.add_scalar(posts)
        args.add_scalar(int(inject_overflow))
        return args

    def run(self, posts, *, inject_overflow=False):
        self.epoch += 1
        source = [source_values(rank, self.epoch) for rank in range(self.ranks)]
        empty = torch.full((ALLOCATED,), POISON, dtype=torch.float32)
        capture = self.directory / f"epoch_{self.epoch}"
        capture.mkdir()
        for rank in range(self.ranks):
            self.metadata[rank].copy_(initial_metadata())
            self.outputs[rank].copy_(empty)
        config = CallConfig()
        config.enable_chip_swimlane = self.diagnostics
        config.output_prefix = str(capture)

        def orchestrate(orch, _args, _config):
            domain = orch.get_or_allocate_domain(
                name="sdma_queue_reuse",
                members=list(range(self.ranks)),
                window_size=ALLOCATED * 8,
                buffers=[
                    CommBufferSpec(name=name, dtype="float32", count=ALLOCATED, nbytes=ALLOCATED * 4)
                    for name in ("source", "destination")
                ],
                retain_after_run=True,
                workspace_engines=("sdma",),
            )
            identity = tuple(domain.device_context(rank, workspace_engines=("sdma",)) for rank in range(self.ranks))
            if self.identity is None:
                self.identity = identity
            else:
                assert identity == self.identity, "retained communication contexts changed"
            for rank in range(self.ranks):
                stage(self.staging[rank]["source"], source[rank])
                stage(self.staging[rank]["destination"], empty)
                orch.copy_to(domain[rank].buffers["source"], self.staging[rank]["source"])
                orch.copy_to(domain[rank].buffers["destination"], self.staging[rank]["destination"])
            for rank in range(self.ranks):
                rank_config = CallConfig()
                rank_config.enable_chip_swimlane = self.diagnostics
                rank_config.output_prefix = str(capture / f"rank_{rank}")
                Path(rank_config.output_prefix).mkdir()
                orch.submit_next_level(
                    self.callable, self._args(domain, rank, posts, inject_overflow), rank_config, worker=rank
                )

        self.worker.submit(orchestrate, config=config).wait(timeout=120)
        domain = self.worker.live_domains["sdma_queue_reuse"]
        report = []
        for rank in range(self.ranks):
            meta = self.metadata[rank]
            lanes = int(meta[-1, 0])
            assert 0 < lanes <= MAX_LANES and lanes % 2 == 0
            expected = initial_metadata()
            expected[-1, :4] = torch.tensor([lanes, posts, lanes, lanes])
            received, golden = empty.clone(), empty.clone()
            queues = defaultdict(list)
            for index in range(lanes * posts):
                start, count = span(index)
                incoming = source[(rank + 1) % self.ranks][start : start + count]
                received[start : start + count] = incoming
                golden[start : start + count] = incoming + 7
                address, post_id, cluster = map(int, meta[index, 1:4])
                assert address > 0 and address % 8 == 0 and post_id > 0
                expected[index, :8] = torch.tensor([1, address, post_id, cluster, index // posts, index % posts, 1, 0])
                queues[address].append((post_id, cluster, index // posts))
            torch.testing.assert_close(meta, expected, rtol=0, atol=0)
            torch.testing.assert_close(self.outputs[rank], golden, rtol=0, atol=0)
            for name, value in (("source", source[rank]), ("destination", received)):
                self.worker.copy_from(self.staging[rank][name], domain[rank].buffers[name])
                torch.testing.assert_close(read(self.staging[rank][name], value), value, rtol=0, atol=0)
            assert len(queues) == lanes // 2, "default queues must pair the two AIV lanes of every cluster"
            if self.last[rank] is not None:
                assert queues.keys() == self.last[rank].keys(), "retained queue record addresses changed"
            current = {}
            clusters = set()
            for address, rows in queues.items():
                ids = sorted(row[0] for row in rows)
                owners = {row[1] for row in rows}
                blocks = {row[2] for row in rows}
                assert len(rows) == 2 * posts and len(owners) == 1 and len(blocks) == 2
                assert ids == list(range(ids[0], ids[0] + 2 * posts)), "post IDs repeat or skip within a shared queue"
                if self.last[rank] is not None:
                    assert ids[0] == self.last[rank][address] + 1, "post IDs reset across retained runs"
                current[address] = ids[-1]
                clusters.update(owners)
            assert len(clusters) == lanes // 2
            self.last[rank] = current
            report.append(
                {
                    "rank": rank,
                    "lanes": lanes,
                    "posts_per_queue": 2 * posts,
                    "queues": {str(address): rows for address, rows in queues.items()},
                }
            )
        (capture / "queue-reuse.json").write_text(
            json.dumps(
                {
                    "epoch": self.epoch,
                    "ranks": report,
                    "limits": "66+2 posts per queue; concurrent in-flight count and SDK backpressure are not inferred.",
                },
                indent=2,
            )
            + "\n"
        )

    def _close(self, *, failing=False):
        if self.closed:
            return
        try:
            self.worker.close()
        except Exception as error:
            if not failing:
                raise
            warnings.warn(f"SDMA reuse cleanup also failed: {error}", RuntimeWarning, stacklevel=2)
        finally:
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _tb):
        self._close(failing=exc_type is not None)


def l3_case(function):
    function._st_level = 3
    return function


pytestmark = [
    pytest.mark.platforms(["a5"]),
    pytest.mark.runtime("host_build_graph"),
    pytest.mark.sdma(worker_workspace=False),
]


@pytest.mark.device_count(2)
@pytest.mark.parametrize("diagnostics", [0, 4])
@l3_case
def test_two_rank_queue_wrap_and_retention(st_platform, st_device_ids, tmp_path, diagnostics):
    with QueueReuseRunner(st_platform, st_device_ids[:2], tmp_path / "queues", diagnostics) as runner:
        runner.run(33)
        runner.run(1)


@pytest.mark.device_count(4)
@pytest.mark.parametrize("diagnostics", [0, 4])
@l3_case
def test_four_rank_queue_wrap_and_retention(st_platform, st_device_ids, tmp_path, diagnostics):
    with QueueReuseRunner(st_platform, st_device_ids[:4], tmp_path / "queues", diagnostics) as runner:
        runner.run(33)
        runner.run(1)


@pytest.mark.manual
@pytest.mark.device_count(2)
@l3_case
def test_registration_failure_after_real_sdma_retains_domain(st_platform, st_device_ids, tmp_path, capfd):
    with QueueReuseRunner(st_platform, st_device_ids[:2], tmp_path / "error", 4) as runner:
        runner.run(1)
        domain = runner.worker.live_domains["sdma_queue_reuse"]
        capfd.readouterr()
        cursor = error_log_cursor()
        with pytest.raises(RuntimeError) as error:
            runner.run(1, inject_overflow=True)
        evidence = failure_evidence(error.value, capfd.readouterr(), cursor)
        (tmp_path / "sdma-error.log").write_text(evidence)
        assert "ASYNC_WAIT_OVERFLOW" in evidence or "terminal_code=-102 " in evidence, evidence
        with pytest.raises(RuntimeError, match="window reclamation is unconfirmed"):
            runner._close()
        assert domain.released and not domain.freed, "failed SDMA run freed its communication storage"
