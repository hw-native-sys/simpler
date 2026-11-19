# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Signal-only waiters with independently owned receive payload arguments."""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _rehosted_buffer_for, _rehosted_ref_for

FIELDS = ("x", "scale", "weights", "routes")
MAX_BLOCKS = 128
ELEMENTS = 512
POISON = -17


def domain_name(tag):
    return f"signal_payload_{int(tag)}"


def payload_orchestration(orch, callables, task_args, config):
    ranks = int(task_args.ranks)
    epoch = int(task_args.epoch)
    retained = bool(task_args.retained)
    specs = [
        CommBufferSpec(name=name, dtype="int32", count=ranks * 32, nbytes=ranks * 128) for name in ("arrived", "ready")
    ] + [
        CommBufferSpec(
            name=name,
            dtype="int32" if name == "routes" else "float32",
            count=ranks * ELEMENTS,
            nbytes=ranks * ELEMENTS * 4,
        )
        for name in FIELDS
    ]
    domain = orch.get_or_allocate_domain(
        name=domain_name(task_args.domain_tag),
        members=list(range(ranks)),
        window_size=sum(spec.nbytes for spec in specs),
        buffers=specs,
        retain_after_run=retained,
    )
    if epoch == 1:
        for rank in range(ranks):
            for name in ("arrived", "ready"):
                orch.copy_to(domain[rank].buffers[name], _rehosted_buffer_for(task_args, task_args.signal_seed))
            for name in FIELDS:
                orch.copy_to(
                    domain[rank].buffers[name], _rehosted_buffer_for(task_args, getattr(task_args, f"seed_{name}"))
                )
    for rank in range(ranks):
        args = TaskArgs()
        for name in ("arrived", "ready"):
            args.add_tensor(domain[rank].buffers[name].tensor((ranks, 32), DataType.INT32), TensorArgType.INOUT)
        for name in FIELDS:
            dtype = DataType.INT32 if name == "routes" else DataType.FLOAT32
            args.add_tensor(domain[rank].buffers[name].tensor((ranks, ELEMENTS), dtype), TensorArgType.INOUT)
        for name in FIELDS:
            args.add_tensor(_rehosted_ref_for(task_args, getattr(task_args, f"out_{name}_{rank}")), TensorArgType.INOUT)
        args.add_tensor(_rehosted_ref_for(task_args, getattr(task_args, f"info_{rank}")), TensorArgType.INOUT)
        for value in (
            domain.device_context(rank),
            rank,
            ranks,
            epoch,
            int(task_args.dirty),
            int(task_args.consumer_type),
        ):
            args.add_scalar(value)
        orch.submit_next_level(callables.payload, args, config, worker=rank)
    if not retained:
        # End the caller's scope while accepted waiters and peer writes are outstanding.
        domain.release()


class _SignalPayloadBase(SceneTestCase):
    RTOL = 0
    ATOL = 0
    CALLABLE = {
        "orchestration": payload_orchestration,
        "callables": [
            {
                "name": "payload",
                "orchestration": {
                    "source": "kernels/orchestration/payload.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.INOUT] * 11 + [D.SCALAR] * 6,
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aic/warm.cpp",
                        "core_type": "aic",
                        "signature": [D.INOUT] * 5 + [D.SCALAR] * 2,
                    },
                    {
                        "func_id": 1,
                        "source": "kernels/aiv/warm.cpp",
                        "core_type": "aiv",
                        "signature": [D.INOUT] * 5 + [D.SCALAR] * 2,
                    },
                    {
                        "func_id": 2,
                        "source": "kernels/aic/gather.cpp",
                        "core_type": "aic",
                        "signature": [D.IN] * 4 + [D.INOUT] * 5 + [D.SCALAR],
                    },
                    {
                        "func_id": 3,
                        "source": "kernels/aiv/gather.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN] * 4 + [D.INOUT] * 5 + [D.SCALAR],
                    },
                    {
                        "func_id": 4,
                        "source": "kernels/aiv/waiter.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN] + [D.SCALAR] * 3,
                    },
                    {
                        "func_id": 5,
                        "source": "kernels/aiv/ready.cpp",
                        "core_type": "aiv",
                        "signature": [D.INOUT, D.SCALAR],
                    },
                    {
                        "func_id": 6,
                        "source": "kernels/aiv/publisher.cpp",
                        "core_type": "aiv",
                        "signature": [D.INOUT] * 5 + [D.SCALAR] * 2,
                    },
                    {
                        "func_id": 7,
                        "source": "kernels/aiv/gather_tile.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN] * 4 + [D.INOUT] * 5 + [D.SCALAR],
                    },
                ],
            }
        ],
    }

    def _run_and_validate(self, worker, callable_obj, case, **kwargs):
        rounds = kwargs.pop("rounds", 1)
        for _ in range(rounds):
            self._domain_sequence = getattr(self, "_domain_sequence", 0) + 1
            tag = self._domain_sequence
            epochs = (1, 2, 3) if case["params"]["retained"] else (1,)
            try:
                for epoch in epochs:
                    current = {**case, "params": {**case["params"], "epoch": epoch, "domain_tag": tag}}
                    super()._run_and_validate(worker, callable_obj, current, rounds=1, **kwargs)
            finally:
                handle = worker.live_domains.get(domain_name(tag))
                if handle is not None:
                    handle.release()

    def generate_args(self, params):
        ranks = params["ranks"]
        specs = [TensorArg("signal_seed", torch.zeros((ranks, 32), dtype=torch.int32))]
        for name in FIELDS:
            dtype = torch.int32 if name == "routes" else torch.float32
            specs.append(TensorArg(f"seed_{name}", torch.full((ranks, ELEMENTS), POISON, dtype=dtype)))
        for rank in range(ranks):
            for name in FIELDS:
                dtype = torch.int32 if name == "routes" else torch.float32
                specs.append(
                    TensorArg(f"out_{name}_{rank}", torch.full((MAX_BLOCKS, ranks, ELEMENTS), POISON, dtype=dtype))
                )
            specs.append(TensorArg(f"info_{rank}", torch.zeros((MAX_BLOCKS, 32), dtype=torch.int32)))
        return TaskArgsBuilder(
            *specs,
            *(
                Scalar(name, int(params[name]))
                for name in ("ranks", "epoch", "domain_tag", "dirty", "consumer_type", "retained")
            ),
        )

    def compute_golden(self, args, params):
        ranks, epoch = params["ranks"], params["epoch"]
        index = torch.arange(ELEMENTS, dtype=torch.int32)
        for rank in range(ranks):
            for peer in range(ranks):
                if peer == rank:
                    continue
                values = (
                    ((peer + 1) * 10000 + epoch * 100 + rank * 7 + index).float(),
                    (peer + 1) / 2 + epoch / 8 + (index % 4).float() / 4,
                    -((peer + 1) * 100 + epoch + rank) - index.float() / 16,
                    (peer + 1) * 100000 + epoch * 1000 + rank * 13 + index,
                )
                for name, value in zip(FIELDS, values):
                    getattr(args, f"out_{name}_{rank}")[:, peer, :].copy_(value)

    def compare_outputs(self, test_args, golden_args, output_names, params):
        for rank in range(params["ranks"]):
            info = getattr(test_args, f"info_{rank}")
            blocks = int(info[0, 1])
            assert 0 < blocks <= MAX_BLOCKS, f"invalid consumer width: {blocks}"
            assert torch.all(info[:blocks, 0] == 1), "missing consumer execution"
            assert torch.all(info[:blocks, 1] == blocks), "consumer width changed across executions"
            assert torch.all(info[:blocks, 2] == 1), "a consumer core was not warmed"
            assert torch.all(info[:blocks, 3] == blocks), "warm and consumer cohorts differ"
            assert not torch.count_nonzero(info[blocks:]), "inactive metadata was overwritten"
            for name in FIELDS:
                actual = getattr(test_args, f"out_{name}_{rank}")
                expected = getattr(golden_args, f"out_{name}_{rank}")
                torch.testing.assert_close(actual[:blocks], expected[:blocks], rtol=0, atol=0)
                assert torch.all(actual[blocks:] == POISON), f"inactive output overwritten: {name}"


def cases(ranks):
    return [
        {
            "name": (
                f"{('aic', 'aiv', 'aiv_tload')[core]}_{'dirty_release' if dirty else 'clean'}"
                f"_{'reuse' if retained else 'scope'}"
            ),
            "platforms": ["a5"],
            "manual": True,
            "config": {"device_count": ranks, "num_sub_workers": 0},
            "params": {"ranks": ranks, "consumer_type": core, "dirty": dirty, "retained": retained},
        }
        for core in (0, 1, 2)
        for dirty in (False, True)
        for retained in (False, True)
    ]


@scene_test(level=3, runtime="host_build_graph")
class TestSignalPayloadTwoRanks(_SignalPayloadBase):
    CASES = cases(2)


@scene_test(level=3, runtime="host_build_graph")
class TestSignalPayloadFourRanks(_SignalPayloadBase):
    CASES = cases(4)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
