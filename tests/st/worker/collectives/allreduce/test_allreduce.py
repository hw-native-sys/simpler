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

import pytest
import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, scene_test

from .._helpers import allreduce_expected_output, allreduce_orch_fn, make_allreduce_args


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
            "params": {"nranks": 2, "mode_id": 0},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"])

    def compute_golden(self, args, params):
        expected = torch.tensor(allreduce_expected_output(params["nranks"]), dtype=torch.float32)
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
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 1},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"])

    def compute_golden(self, args, params):
        expected = torch.tensor(allreduce_expected_output(params["nranks"]), dtype=torch.float32)
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
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 2},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"])

    def compute_golden(self, args, params):
        expected = torch.tensor(allreduce_expected_output(params["nranks"]), dtype=torch.float32)
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
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 3},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"])

    def compute_golden(self, args, params):
        expected = torch.tensor(allreduce_expected_output(params["nranks"]), dtype=torch.float32)
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
            "config": {"device_count": 2},
            "params": {"nranks": 2, "mode_id": 4},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"])

    def compute_golden(self, args, params):
        expected = torch.tensor(allreduce_expected_output(params["nranks"]), dtype=torch.float32)
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
            "config": {"device_count": 4},
            "params": {"nranks": 4, "mode_id": 0},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"])

    def compute_golden(self, args, params):
        expected = torch.tensor(allreduce_expected_output(params["nranks"]), dtype=torch.float32)
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
            "config": {"device_count": 4},
            "params": {"nranks": 4, "mode_id": 1},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"])

    def compute_golden(self, args, params):
        expected = torch.tensor(allreduce_expected_output(params["nranks"]), dtype=torch.float32)
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
            "config": {"device_count": 4},
            "params": {"nranks": 4, "mode_id": 2},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"])

    def compute_golden(self, args, params):
        expected = torch.tensor(allreduce_expected_output(params["nranks"]), dtype=torch.float32)
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
            "config": {"device_count": 4},
            "params": {"nranks": 4, "mode_id": 3},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"])

    def compute_golden(self, args, params):
        expected = torch.tensor(allreduce_expected_output(params["nranks"]), dtype=torch.float32)
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
            "config": {"device_count": 4},
            "params": {"nranks": 4, "mode_id": 4},
        }
    ]

    def generate_args(self, params):
        return make_allreduce_args(params["nranks"], params["mode_id"])

    def compute_golden(self, args, params):
        pass

    def test_run(self, st_platform, st_worker, request):
        """Override: expect ValueError from orch_fn before any submission."""
        with pytest.raises(ValueError, match="ibing mode is only supported for nranks=2"):
            super().test_run(st_platform, st_worker, request)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
