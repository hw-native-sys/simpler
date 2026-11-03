#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scene test for distributed allreduce — one class per mode×rank.

NOTE: Each mode×nranks gets its own SceneTestCase subclass (not multiple CASES per
class) because multiple cases within a single class exhibit a Worker state-leak
that causes golden-mismatch failures on sequential worker.run() calls.
"""

import os
from multiprocessing.shared_memory import SharedMemory

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.worker import Worker

from simpler_setup import SceneTestCase, scene_test

from .._helpers import (
    CollectiveReduceOp,
    allreduce_expected_output,
    allreduce_orch_fn,
    make_allreduce_args,
)


def _orch_entry(source, func_name, config_name=""):
    return {
        "source": source,
        "function_name": func_name,
        "config_name": config_name,
        "signature": [D.IN, D.OUT, D.INOUT],
    }


def _incore(source):
    return {"func_id": 0, "source": source, "core_type": "aiv", "signature": [D.IN, D.OUT, D.INOUT]}


_ALLREDUCE_MODES = [
    {
        "name": "allreduce_onephase",
        "orchestration": _orch_entry(
            "kernels/orchestration/allreduce_onephase_orch.cpp",
            "allreduce_orchestration",
            "allreduce_orchestration_config",
        ),
        "incores": [_incore("kernels/aiv/allreduce_onephase_kernel.cpp")],
    },
    {
        "name": "allreduce_twophase",
        "orchestration": _orch_entry(
            "kernels/orchestration/allreduce_twophase_orch.cpp",
            "allreduce_twophase_orchestration",
            "allreduce_twophase_orchestration_config",
        ),
        "incores": [_incore("kernels/aiv/allreduce_twophase_kernel.cpp")],
    },
    {
        "name": "allreduce_ring",
        "orchestration": _orch_entry(
            "kernels/orchestration/allreduce_ring_orch.cpp",
            "allreduce_ring_orchestration",
            "allreduce_ring_orchestration_config",
        ),
        "incores": [_incore("kernels/aiv/allreduce_ring_kernel.cpp")],
    },
    {
        "name": "allreduce_bidirectional_ring",
        "orchestration": _orch_entry(
            "kernels/orchestration/allreduce_bidirectional_ring_orch.cpp",
            "allreduce_bidirectional_ring_orchestration",
            "allreduce_bidirectional_ring_orchestration_config",
        ),
        "incores": [_incore("kernels/aiv/allreduce_bidirectional_ring_kernel.cpp")],
    },
    {
        "name": "allreduce_ibing",
        "orchestration": _orch_entry(
            "kernels/orchestration/allreduce_ibing_orch.cpp",
            "allreduce_ibing_orchestration",
            "allreduce_ibing_orchestration_config",
        ),
        "incores": [_incore("kernels/aiv/allreduce_ibing_kernel.cpp")],
    },
]


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestAllreduceOnephaseP2(SceneTestCase):
    """Allreduce onephase — 2-rank."""

    CALLABLE = {"orchestration": allreduce_orch_fn, "callables": _ALLREDUCE_MODES}
    CASES = [
        {
            "name": "onephase",
            "platforms": ["a2a3sim", "a2a3", "a5sim", "a5"],
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 0, "reduce_op": 0},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"], params["reduce_op"])

    def compute_golden(self, args, params):
        expected = torch.tensor(
            allreduce_expected_output(params["nranks"], CollectiveReduceOp(params["reduce_op"])),
            dtype=torch.float32,
        )
        for rank in range(params["nranks"]):
            getattr(args, f"out_{rank}").copy_(expected)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestAllreduceTwophaseP2(SceneTestCase):
    """Allreduce twophase — 2-rank."""

    CALLABLE = {"orchestration": allreduce_orch_fn, "callables": _ALLREDUCE_MODES}
    CASES = [
        {
            "name": "twophase",
            "platforms": ["a2a3sim", "a2a3", "a5sim", "a5"],
            "manual": ["a2a3sim", "a5sim"],
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 1, "reduce_op": 0},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"], params["reduce_op"])

    def compute_golden(self, args, params):
        expected = torch.tensor(
            allreduce_expected_output(params["nranks"], CollectiveReduceOp(params["reduce_op"])),
            dtype=torch.float32,
        )
        for rank in range(params["nranks"]):
            getattr(args, f"out_{rank}").copy_(expected)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestAllreduceRingP2(SceneTestCase):
    """Allreduce ring — 2-rank."""

    CALLABLE = {"orchestration": allreduce_orch_fn, "callables": _ALLREDUCE_MODES}
    CASES = [
        {
            "name": "ring",
            "platforms": ["a2a3sim", "a2a3", "a5sim", "a5"],
            "manual": ["a2a3sim", "a5sim"],
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 2, "reduce_op": 0},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"], params["reduce_op"])

    def compute_golden(self, args, params):
        expected = torch.tensor(
            allreduce_expected_output(params["nranks"], CollectiveReduceOp(params["reduce_op"])),
            dtype=torch.float32,
        )
        for rank in range(params["nranks"]):
            getattr(args, f"out_{rank}").copy_(expected)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestAllreduceBidirectionalRingP2(SceneTestCase):
    """Allreduce bidirectional_ring — 2-rank."""

    CALLABLE = {"orchestration": allreduce_orch_fn, "callables": _ALLREDUCE_MODES}
    CASES = [
        {
            "name": "bidirectional_ring",
            "platforms": ["a2a3sim", "a2a3", "a5sim", "a5"],
            "manual": ["a2a3sim", "a5sim"],
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 3, "reduce_op": 0},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"], params["reduce_op"])

    def compute_golden(self, args, params):
        expected = torch.tensor(
            allreduce_expected_output(params["nranks"], CollectiveReduceOp(params["reduce_op"])),
            dtype=torch.float32,
        )
        for rank in range(params["nranks"]):
            getattr(args, f"out_{rank}").copy_(expected)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestAllreduceIbingP2(SceneTestCase):
    """Allreduce ibing — 2-rank."""

    CALLABLE = {"orchestration": allreduce_orch_fn, "callables": _ALLREDUCE_MODES}
    CASES = [
        {
            "name": "ibing",
            "platforms": ["a2a3sim", "a2a3", "a5sim", "a5"],
            "manual": ["a2a3sim", "a5sim"],
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 4, "reduce_op": 0},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"], params["reduce_op"])

    def compute_golden(self, args, params):
        expected = torch.tensor(
            allreduce_expected_output(params["nranks"], CollectiveReduceOp(params["reduce_op"])),
            dtype=torch.float32,
        )
        for rank in range(params["nranks"]):
            getattr(args, f"out_{rank}").copy_(expected)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestAllreduceOnephaseP4(SceneTestCase):
    """Allreduce onephase — 4-rank."""

    CALLABLE = {"orchestration": allreduce_orch_fn, "callables": _ALLREDUCE_MODES}
    CASES = [
        {
            "name": "onephase",
            "platforms": ["a2a3sim", "a2a3", "a5sim"],
            "manual": True,
            "config": {"device_count": 4},
            "params": {"nranks": 4, "mode_id": 0, "reduce_op": 0},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"], params["reduce_op"])

    def compute_golden(self, args, params):
        expected = torch.tensor(
            allreduce_expected_output(params["nranks"], CollectiveReduceOp(params["reduce_op"])),
            dtype=torch.float32,
        )
        for rank in range(params["nranks"]):
            getattr(args, f"out_{rank}").copy_(expected)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestAllreduceTwophaseP4(SceneTestCase):
    """Allreduce twophase — 4-rank."""

    CALLABLE = {"orchestration": allreduce_orch_fn, "callables": _ALLREDUCE_MODES}
    CASES = [
        {
            "name": "twophase",
            "platforms": ["a2a3sim", "a2a3", "a5sim"],
            "manual": True,
            "config": {"device_count": 4},
            "params": {"nranks": 4, "mode_id": 1, "reduce_op": 0},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"], params["reduce_op"])

    def compute_golden(self, args, params):
        expected = torch.tensor(
            allreduce_expected_output(params["nranks"], CollectiveReduceOp(params["reduce_op"])),
            dtype=torch.float32,
        )
        for rank in range(params["nranks"]):
            getattr(args, f"out_{rank}").copy_(expected)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestAllreduceRingP4(SceneTestCase):
    """Allreduce ring — 4-rank."""

    CALLABLE = {"orchestration": allreduce_orch_fn, "callables": _ALLREDUCE_MODES}
    CASES = [
        {
            "name": "ring",
            "platforms": ["a2a3sim", "a2a3", "a5sim"],
            "manual": True,
            "config": {"device_count": 4},
            "params": {"nranks": 4, "mode_id": 2, "reduce_op": 0},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"], params["reduce_op"])

    def compute_golden(self, args, params):
        expected = torch.tensor(
            allreduce_expected_output(params["nranks"], CollectiveReduceOp(params["reduce_op"])),
            dtype=torch.float32,
        )
        for rank in range(params["nranks"]):
            getattr(args, f"out_{rank}").copy_(expected)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestAllreduceBidirectionalRingP4(SceneTestCase):
    """Allreduce bidirectional_ring — 4-rank."""

    CALLABLE = {"orchestration": allreduce_orch_fn, "callables": _ALLREDUCE_MODES}
    CASES = [
        {
            "name": "bidirectional_ring",
            "platforms": ["a2a3sim", "a2a3", "a5sim"],
            "manual": True,
            "config": {"device_count": 4},
            "params": {"nranks": 4, "mode_id": 3, "reduce_op": 0},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"], params["reduce_op"])

    def compute_golden(self, args, params):
        expected = torch.tensor(
            allreduce_expected_output(params["nranks"], CollectiveReduceOp(params["reduce_op"])),
            dtype=torch.float32,
        )
        for rank in range(params["nranks"]):
            getattr(args, f"out_{rank}").copy_(expected)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestAllreduceIbingNranksError(SceneTestCase):
    """Negative test: ibing with nranks=4 raises ValueError."""

    CALLABLE = {"orchestration": allreduce_orch_fn, "callables": _ALLREDUCE_MODES}
    CASES = [
        {
            "name": "ibing_nranks_4",
            "platforms": ["a2a3sim", "a2a3", "a5sim"],
            "manual": True,
            "config": {"device_count": 4},
            "params": {"nranks": 4, "mode_id": 4, "reduce_op": 0},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"], params["reduce_op"])

    def compute_golden(self, args, params):
        pass

    def test_run(self, st_platform, st_worker, request):
        """Override: expect ValueError from orch_fn before any submission."""
        with pytest.raises(ValueError, match="ibing mode is only supported for nranks=2"):
            super().test_run(st_platform, st_worker, request)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestAllreduceOnephaseP2MaxMinProd(SceneTestCase):
    """Allreduce onephase — 2-rank, non-Sum reduce ops."""

    CALLABLE = {"orchestration": allreduce_orch_fn, "callables": _ALLREDUCE_MODES}
    CASES = [
        {
            "name": "onephase_max",
            "platforms": ["a2a3sim", "a2a3", "a5sim", "a5"],
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 0, "reduce_op": 1},
        },
        {
            "name": "onephase_min",
            "platforms": ["a2a3sim", "a2a3", "a5sim", "a5"],
            "manual": ["a2a3sim", "a5sim"],
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 0, "reduce_op": 2},
        },
        {
            "name": "onephase_prod",
            "platforms": ["a2a3sim", "a2a3", "a5sim", "a5"],
            "manual": ["a2a3sim", "a5sim"],
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 0, "reduce_op": 3},
        },
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"], params["reduce_op"])

    def compute_golden(self, args, params):
        expected = torch.tensor(
            allreduce_expected_output(params["nranks"], CollectiveReduceOp(params["reduce_op"])),
            dtype=torch.float32,
        )
        for rank in range(params["nranks"]):
            getattr(args, f"out_{rank}").copy_(expected)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestAllreduceTwophaseP2MaxMinProd(SceneTestCase):
    """Allreduce twophase — 2-rank, non-Sum reduce ops."""

    CALLABLE = {"orchestration": allreduce_orch_fn, "callables": _ALLREDUCE_MODES}
    CASES = [
        {
            "name": "twophase_max",
            "platforms": ["a2a3sim", "a2a3", "a5sim", "a5"],
            "manual": ["a2a3sim", "a5sim"],
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 1, "reduce_op": 1},
        },
        {
            "name": "twophase_min",
            "platforms": ["a2a3sim", "a2a3", "a5sim", "a5"],
            "manual": ["a2a3sim", "a5sim"],
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 1, "reduce_op": 2},
        },
        {
            "name": "twophase_prod",
            "platforms": ["a2a3sim", "a2a3", "a5sim", "a5"],
            "manual": ["a2a3sim", "a5sim"],
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 1, "reduce_op": 3},
        },
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"], params["reduce_op"])

    def compute_golden(self, args, params):
        expected = torch.tensor(
            allreduce_expected_output(params["nranks"], CollectiveReduceOp(params["reduce_op"])),
            dtype=torch.float32,
        )
        for rank in range(params["nranks"]):
            getattr(args, f"out_{rank}").copy_(expected)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestAllreduceRingP2MaxMinProd(SceneTestCase):
    """Allreduce ring — 2-rank, non-Sum reduce ops."""

    CALLABLE = {"orchestration": allreduce_orch_fn, "callables": _ALLREDUCE_MODES}
    CASES = [
        {
            "name": "ring_max",
            "platforms": ["a2a3sim", "a2a3", "a5sim", "a5"],
            "manual": ["a2a3sim", "a5sim"],
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 2, "reduce_op": 1},
        },
        {
            "name": "ring_min",
            "platforms": ["a2a3sim", "a2a3", "a5sim", "a5"],
            "manual": ["a2a3sim", "a5sim"],
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 2, "reduce_op": 2},
        },
        {
            "name": "ring_prod",
            "platforms": ["a2a3sim", "a2a3", "a5sim", "a5"],
            "manual": ["a2a3sim", "a5sim"],
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 2, "reduce_op": 3},
        },
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"], params["reduce_op"])

    def compute_golden(self, args, params):
        expected = torch.tensor(
            allreduce_expected_output(params["nranks"], CollectiveReduceOp(params["reduce_op"])),
            dtype=torch.float32,
        )
        for rank in range(params["nranks"]):
            getattr(args, f"out_{rank}").copy_(expected)


def _run_rejected_domain_case(case, platform, device_ids, request, monkeypatch):
    worker = Worker(
        level=3,
        platform=platform,
        runtime=case._st_runtime,
        device_ids=device_ids,
        num_sub_workers=0,
    )
    domains, release_calls = [], []
    allocate = worker._allocate_domain
    release = worker._release_domain_claimed

    def observe_allocate(*args, **kwargs):
        domain = allocate(*args, **kwargs)
        domains.append(domain)
        return domain

    def observe_release(domain):
        release_calls.append(domain)
        release(domain)

    monkeypatch.setattr(worker, "_allocate_domain", observe_allocate)
    monkeypatch.setattr(worker, "_release_domain_claimed", observe_release)
    try:
        compiled = case.build_callable(platform)
        handles = {name: value if name.endswith("_sig") else worker.register(value) for name, value in compiled.items()}
        monkeypatch.setattr(type(case), "_st_chip_handles", handles, raising=False)
        worker._st_device_id = device_ids[0]
        worker.init()
        pids = (*worker._chip_pids, *worker._sub_pids, *worker._next_level_pids)
        mailboxes = tuple(shm.name for shm in (*worker._chip_shms, *worker._next_level_shms))
        with pytest.raises(RuntimeError, match=r"reduce_op=1 unsupported|failed with code -5"):
            SceneTestCase.test_run(case, platform, worker, request)
        assert len(domains) == 1
        domain = domains[0]
        assert domain.released and not domain.freed
        assert not worker._cleanup_journal.empty
        for _ in range(2):
            with pytest.raises(RuntimeError, match="window reclamation is unconfirmed"):
                worker.close()
            assert not domain.freed and not worker._cleanup_journal.empty
            assert worker._domain_free_results[domain.allocation_id] is not None
            assert [(kind, identity) for kind, identity, _ in worker._cleanup_journal._entries] == [
                ("domain-retirement", f"True:{domain.allocation_id}")
            ]
            assert worker._worker is None
            assert not worker._chip_pids and not worker._sub_pids and not worker._next_level_pids
            for pid in pids:
                with pytest.raises(ChildProcessError):
                    os.waitpid(pid, os.WNOHANG)
            for name in mailboxes:
                with pytest.raises(FileNotFoundError):
                    SharedMemory(name=name)
        assert release_calls == [domain]
    finally:
        if worker._cleanup_journal.empty:
            worker.close()
        else:
            with pytest.raises(RuntimeError, match="window reclamation is unconfirmed"):
                worker.close()


@pytest.mark.device_count(2)
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestAllreduceBidirectionalRingRejectNonSum(SceneTestCase):
    """Negative test: bidirectional_ring rejects a non-Sum reduce_op via rt_report_fatal."""

    CALLABLE = {"orchestration": allreduce_orch_fn, "callables": _ALLREDUCE_MODES}
    CASES = [
        {
            "name": "bidirectional_ring_max_rejected",
            "platforms": ["a2a3sim", "a2a3", "a5sim", "a5"],
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 3, "reduce_op": 1},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"], params["reduce_op"])

    def compute_golden(self, args, params):
        pass

    def test_run(self, st_platform, st_device_ids, request, monkeypatch):
        """The device error leaves explicit retirement debt without stranding the worker tree."""
        _run_rejected_domain_case(self, st_platform, st_device_ids, request, monkeypatch)


@pytest.mark.device_count(2)
@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestAllreduceIbingRejectNonSum(SceneTestCase):
    """Negative test: ibing rejects a non-Sum reduce_op via rt_report_fatal."""

    CALLABLE = {"orchestration": allreduce_orch_fn, "callables": _ALLREDUCE_MODES}
    CASES = [
        {
            "name": "ibing_max_rejected",
            "platforms": ["a2a3sim", "a2a3", "a5sim", "a5"],
            "manual": ["a2a3sim", "a5sim"],
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 4, "reduce_op": 1},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"], params["reduce_op"])

    def compute_golden(self, args, params):
        pass

    def test_run(self, st_platform, st_device_ids, request, monkeypatch):
        """The device error leaves explicit retirement debt without stranding the worker tree."""
        _run_rejected_domain_case(self, st_platform, st_device_ids, request, monkeypatch)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
