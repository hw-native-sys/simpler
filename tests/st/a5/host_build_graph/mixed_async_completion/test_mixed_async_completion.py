# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Mixed SDMA/COUNTER groups with real peers and shared Graph definitions."""

from copy import deepcopy

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _rehosted_buffer_for, _rehosted_ref_for

LINE = 32
META = 16
GUARD = 0x13579BDF
POISON = -1000000007
INSTANCES = 2


def executions(params):
    return INSTANCES * params["blocks"] * 3


def active(execution, mask):
    return bool(mask & (1 << (execution % 3)))


def source_values(params, rank):
    e, n = executions(params), params["elements"]
    result = torch.full((e, n + LINE), -77.0)
    result[:, :n] = (
        (torch.arange(n, dtype=torch.int32) % 251).float().unsqueeze(0)
        + torch.arange(e).float().unsqueeze(1) * 512
        + rank * 1000000
        + params["tag"] * 10000
    )
    return result.flatten()


def counter_values(params, rank, satisfied):
    result = torch.full((executions(params), params["ranks"], LINE), GUARD, dtype=torch.int32)
    result[:, :, 0] = 0
    if satisfied:
        for e in range(executions(params)):
            if active(e, params["mask"]):
                for peer in range(params["ranks"]):
                    if peer != rank:
                        result[e, peer, 0] = 1
    return result.flatten()


def notice_values(params, rank, published):
    result = torch.full((executions(params), params["ranks"], LINE), POISON, dtype=torch.int32)
    if published:
        for e in range(executions(params)):
            if active(e, params["mask"]):
                for peer in range(params["ranks"]):
                    if peer != rank:
                        result[e, peer] = params["tag"] * 100000 + peer * 1000 + e * LINE + torch.arange(LINE) + 1
    return result.flatten()


def ready_values(params, rank, published):
    result = torch.full((params["ranks"], LINE), GUARD, dtype=torch.int32)
    result[:, 0] = 0
    if published:
        for peer in range(params["ranks"]):
            if peer != rank:
                result[peer, 0] = 1
    return result.flatten()


def name(tag):
    return f"mixed_completion_{tag}"


def mixed_orchestration(orch, callables, values, config):
    ranks, e, n = int(values.ranks), int(values.executions), int(values.elements)
    counts = {"source": e * (n + LINE), "signals": e * ranks * LINE, "notices": e * ranks * LINE, "ready": ranks * LINE}
    domain = orch.get_or_allocate_domain(
        name=name(int(values.tag)),
        members=list(range(ranks)),
        window_size=sum(counts.values()) * 4 + 4096,
        buffers=[
            CommBufferSpec(name=key, dtype="float32" if key == "source" else "int32", count=count, nbytes=count * 4)
            for key, count in counts.items()
        ],
        retain_after_run=True,
        workspace_engines=("sdma",),
    )
    for rank in range(ranks):
        for key in counts:
            orch.copy_to(domain[rank].buffers[key], _rehosted_buffer_for(values, getattr(values, f"{key}_seed_{rank}")))
    for rank in range(ranks):
        args = TaskArgs()
        for key, count in counts.items():
            args.add_tensor(
                domain[rank].buffers[key].tensor((count,), DataType.FLOAT32 if key == "source" else DataType.INT32),
                TensorArgType.INOUT,
            )
        for key in ("received", "result", "notice_result", "registrations", "meta"):
            args.add_tensor(_rehosted_ref_for(values, getattr(values, f"{key}_{rank}")), TensorArgType.INOUT)
        args.add_scalar(domain.device_context(rank, workspace_engines=("sdma",)))
        for key in ("elements", "mask", "blocks", "graph", "late", "tag"):
            args.add_scalar(int(getattr(values, key)))
        orch.submit_next_level(callables.mixed, args, config, worker=rank)


class _MixedBase(SceneTestCase):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": mixed_orchestration,
        "callables": [
            {
                "name": "mixed",
                "orchestration": {
                    "source": "kernels/orchestration/mixed.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.INOUT] * 9 + [D.SCALAR] * 7,
                },
                "incores": [
                    {"func_id": index, "source": source, "core_type": core, "signature": [D.INOUT] * 9 + [D.SCALAR] * 5}
                    for index, source, core in (
                        (0, "kernels/aic/submit.cpp", "aic"),
                        (1, "kernels/aiv/submit.cpp", "aiv"),
                        (2, "kernels/aiv/control.cpp", "aiv"),
                        (3, "kernels/aiv/consume.cpp", "aiv"),
                    )
                ],
            }
        ],
    }

    def generate_args(self, params):
        e, ranks, n = executions(params), params["ranks"], params["elements"]
        args = []
        for rank in range(ranks):
            registrations = torch.full((e, LINE), GUARD, dtype=torch.int32)
            registrations[:, 0] = 0
            meta = torch.zeros((e, META), dtype=torch.int64)
            meta[:, -1] = GUARD
            args.extend(
                [
                    TensorArg(f"source_seed_{rank}", source_values(params, rank)),
                    TensorArg(f"signals_seed_{rank}", counter_values(params, rank, not params["late"])),
                    TensorArg(f"notices_seed_{rank}", notice_values(params, rank, not params["late"])),
                    TensorArg(f"ready_seed_{rank}", ready_values(params, rank, False)),
                    TensorArg(f"received_{rank}", torch.full((e * ranks * (n + LINE),), -17.0)),
                    TensorArg(f"result_{rank}", torch.full((e * ranks * (n + LINE),), -99.0)),
                    TensorArg(f"notice_result_{rank}", torch.full((e * ranks * LINE,), POISON, dtype=torch.int32)),
                    TensorArg(f"registrations_{rank}", registrations.flatten()),
                    TensorArg(f"meta_{rank}", meta.flatten()),
                ]
            )
        return TaskArgsBuilder(
            *args,
            Scalar("executions", e),
            *(Scalar(key, int(params[key])) for key in ("ranks", "elements", "mask", "blocks", "graph", "late", "tag")),
        )

    def compute_golden(self, args, params):
        e, ranks, n = executions(params), params["ranks"], params["elements"]
        for rank in range(ranks):
            receive = getattr(args, f"received_{rank}").view(e, ranks, n + LINE)
            output = getattr(args, f"result_{rank}").view(e, ranks, n + LINE)
            getattr(args, f"notice_result_{rank}").copy_(notice_values(params, rank, True))
            flags = getattr(args, f"registrations_{rank}").view(e, LINE)
            for execution in range(e):
                if not active(execution, params["mask"]):
                    continue
                flags[execution, 0] = 1
                if execution % 3 == 0:
                    continue
                for peer in range(ranks):
                    if peer == rank:
                        continue
                    source = getattr(args, f"source_seed_{peer}").view(e, n + LINE)[execution, :n]
                    receive[execution, peer, :n].copy_(source)
                    output[execution, peer, :n].copy_(source + 7)

    def compare_outputs(self, actual, golden, output_names, params):
        e, peers = executions(params), params["ranks"] - 1
        for rank in range(params["ranks"]):
            for key in ("received", "result", "notice_result", "registrations"):
                torch.testing.assert_close(
                    getattr(actual, f"{key}_{rank}"), getattr(golden, f"{key}_{rank}"), rtol=0, atol=0
                )
            meta = getattr(actual, f"meta_{rank}").view(e, META)
            assert torch.all(meta[:, -1] == GUARD), "metadata guard overwritten"
            for execution, row in enumerate(meta):
                if not active(execution, params["mask"]):
                    assert not torch.count_nonzero(row[:-1]), "inactive execution metadata changed"
                    continue
                sdma = peers if execution % 3 else 0
                assert row[:7].tolist() == [1, 1, peers + sdma, sdma, peers, 0, 0], (rank, execution, row)
                assert int(row[7]) >= 0
                for i in range(3):
                    address, expected = row[8 + i * 2 : 10 + i * 2].tolist()
                    assert (address != 0 and expected > 0) if i < sdma else (address == expected == 0)
                assert int(row[14]) == (0 if params["late"] else peers), "notifications preceded all registrations"

    def _run_and_validate(self, worker, callable_obj, case, **kwargs):
        rounds = kwargs.pop("rounds", 1)
        for iteration in range(rounds):
            current = deepcopy(case)
            current["params"]["tag"] += iteration * 100
            params = current["params"]
            try:
                super()._run_and_validate(worker, callable_obj, current, rounds=1, **kwargs)
                domain = worker.live_domains[name(params["tag"])]
                for rank in range(params["ranks"]):
                    expected = {
                        "source": source_values(params, rank),
                        "signals": counter_values(params, rank, True),
                        "notices": notice_values(params, rank, True),
                        "ready": ready_values(params, rank, params["late"]),
                    }
                    for key, value in expected.items():
                        buffer = worker.create_buffer(value.numel() * value.element_size())
                        worker.copy_from(buffer, domain[rank].buffers[key])
                        observed = torch.frombuffer(buffer.shm.buf, dtype=value.dtype, count=value.numel()).clone()
                        torch.testing.assert_close(observed, value, rtol=0, atol=0, msg=f"rank {rank} {key}")
                        worker.release_buffer(buffer)
            finally:
                domain = worker.live_domains.get(name(params["tag"]))
                if domain is not None:
                    domain.release()


def cases(ranks):
    shapes = [(2, 1), (2, 2), (3, 2), (5, 2), (6, 2), (7, 2)]
    specs = [(mask, blocks, graph, True) for graph in (False, True) for mask, blocks in shapes]
    specs += [(2, 1, graph, False) for graph in (False, True)]
    return [
        {
            "name": (
                f"{'graph' if graph else 'plain'}_mask{mask}_blocks{blocks}_"
                f"{'counter_late' if late else 'counter_ready'}"
            ),
            "platforms": ["a5"],
            "config": {"device_count": ranks, "num_sub_workers": 0},
            "params": {
                "ranks": ranks,
                "mask": mask,
                "blocks": blocks,
                "graph": graph,
                "late": late,
                "elements": 4096 if late else 1048576,
                "tag": index + 10,
            },
        }
        for index, (mask, blocks, graph, late) in enumerate(specs)
    ]


@pytest.mark.sdma(worker_workspace=False)
@scene_test(level=3, runtime="host_build_graph")
class TestMixedAsyncTwoRanks(_MixedBase):
    CASES = cases(2)


@pytest.mark.sdma(worker_workspace=False)
@scene_test(level=3, runtime="host_build_graph")
class TestMixedAsyncFourRanks(_MixedBase):
    CASES = cases(4)
