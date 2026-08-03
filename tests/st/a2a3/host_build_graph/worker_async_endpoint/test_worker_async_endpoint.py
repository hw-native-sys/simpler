#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Onboard validation for the generation-safe two-frame local endpoint."""

import atexit
import ctypes
import tempfile
import time
import uuid
from contextlib import suppress
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.worker import (
    _FRAME_STAGED,
    _OFF_ACCEPTED,
    _OFF_STATE,
    MAILBOX_FRAME_SIZE,
    _mailbox_load_i32,
)

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.scene_test import _build_l3_task_args

_VECTOR_KERNELS = "../vector_example/kernels/aiv"
_SIZE = 128 * 128
_CHAIN_LENGTH = 64


class _FileSignal:
    """Pickle-safe parent/child signal for the standalone scene runner."""

    def __init__(self, label: str):
        token = uuid.uuid4().hex
        self._path = Path(tempfile.gettempdir()) / f"simpler-worker-async-endpoint-{label}-{token}"

    def clear(self) -> None:
        with suppress(FileNotFoundError):
            self._path.unlink()

    def set(self) -> None:
        self._path.touch(exist_ok=True)

    def wait(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._path.exists():
                return True
            time.sleep(0.001)
        return self._path.exists()


_SUB_ENTERED = _FileSignal("entered")
_SUB_RELEASE = _FileSignal("release")


def _clear_signals() -> None:
    _SUB_ENTERED.clear()
    _SUB_RELEASE.clear()


atexit.register(_clear_signals)


def _wait_for_release(_args) -> None:
    _SUB_ENTERED.set()
    if not _SUB_RELEASE.wait(30.0):
        raise RuntimeError("two-frame endpoint test timed out waiting for the release fence")


@scene_test(level=3, runtime="host_build_graph")
class TestWorkerAsyncEndpoint(SceneTestCase):
    CALLABLE = {
        "callables": [
            {
                "name": "long_vector",
                "orchestration": {
                    "source": "kernels/orchestration/long_vector_orch.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.IN, D.IN, D.OUT],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": f"{_VECTOR_KERNELS}/kernel_add.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                    {
                        "func_id": 1,
                        "source": f"{_VECTOR_KERNELS}/kernel_add_scalar.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT],
                    },
                ],
            },
            {"name": "wait_for_release", "callable": _wait_for_release},
        ],
    }

    CASES = [
        {
            "name": "two_frame_sequential_execution",
            "platforms": ["a2a3"],
            "config": {"device_count": 1, "num_sub_workers": 1, "aicpu_thread_num": 4},
            "params": {},
        },
    ]

    @staticmethod
    def _tensor_from_host_buffer(worker, value):
        buffer = worker.create_host_buffer(_SIZE * torch.float32.itemsize)
        tensor = torch.frombuffer(buffer.buffer, dtype=torch.float32, count=_SIZE)
        tensor.fill_(value)
        return buffer, tensor

    def _run_and_validate_l3(  # noqa: PLR0913 -- mirror the scene-test runner hook
        self,
        worker,
        compiled_callables,
        sub_handles,
        case,
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=0,
        enable_dump_args=False,
        enable_pmu=0,
        enable_dep_gen=False,
        enable_scope_stats=False,
        output_prefix="",
    ):
        """Run the custom protocol assertions from the standalone scene runner."""
        del (
            rounds,
            skip_golden,
            enable_l2_swimlane,
            enable_dump_args,
            enable_pmu,
            enable_dep_gen,
            enable_scope_stats,
            output_prefix,
        )
        type(self)._st_chip_handles = compiled_callables
        type(self)._st_sub_handles = sub_handles
        platform = str(worker._config["platform"])  # noqa: SLF001 -- scene-test white-box validation
        assert platform in case["platforms"], f"worker platform {platform!r} is not enabled for this case"
        self._run_protocol_assertions(platform, worker)

    def _run_protocol_assertions(self, st_platform, st_worker):
        if st_platform != "a2a3":
            pytest.skip("the two-frame local endpoint is enabled on a2a3 onboard workers")

        _SUB_ENTERED.clear()
        _SUB_RELEASE.clear()
        buffers = []
        tensors = []
        first = None
        second = None
        try:
            for value in (2.0, 3.0, 0.0, 5.0, 7.0, 0.0):
                buffer, tensor = self._tensor_from_host_buffer(st_worker, value)
                buffers.append(buffer)
                tensors.append(tensor)
            first_a, first_b, first_out, second_a, second_b, second_out = tensors
            handle = type(self)._st_chip_handles["long_vector"]
            signature = type(self)._st_chip_handles["long_vector_sig"]
            sub_handle = type(self)._st_sub_handles["wait_for_release"]
            cfg = self._build_config(self.CASES[0]["config"])

            def submit_vector(orch, a, b, out, *, hold_open=False):
                builder = TaskArgsBuilder(Tensor("a", a), Tensor("b", b), Tensor("out", out))
                chip_args, _ = _build_l3_task_args(builder, signature)
                orch.submit_next_level(handle, chip_args, cfg, worker=0)
                if hold_open:
                    orch.submit_sub(sub_handle)

            def first_graph(orch, _args, _cfg):
                submit_vector(orch, first_a, first_b, first_out, hold_open=True)

            first = st_worker.submit(first_graph)
            assert _SUB_ENTERED.wait(30.0), "the predecessor never reached its SubTask fence"

            def second_graph(orch, _args, _cfg):
                submit_vector(orch, second_a, second_b, second_out)

            second = st_worker.submit(second_graph)

            shm_buf = st_worker._chip_shms[0].buf
            assert shm_buf is not None
            mailbox_addr = ctypes.addressof(ctypes.c_char.from_buffer(shm_buf))
            successor_frame_addr = mailbox_addr + 2 * MAILBOX_FRAME_SIZE
            successor_state_addr = successor_frame_addr + _OFF_STATE
            successor_accepted_addr = successor_frame_addr + _OFF_ACCEPTED

            saw_staged = False
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                state = _mailbox_load_i32(successor_state_addr)
                accepted = _mailbox_load_i32(successor_accepted_addr)
                if state == _FRAME_STAGED:
                    assert accepted == 0, "the successor crossed its launch fence before activation"
                    saw_staged = True
                    break
                assert accepted == 0, "the successor launched while its predecessor was still active"
                time.sleep(0.001)

            assert saw_staged, "the successor did not reach FRAME_STAGED behind its predecessor"
            assert not first.done, "the predecessor escaped its SubTask fence"

            _SUB_RELEASE.set()
            first.wait(30.0)
            second.wait(30.0)
            first_expected = first_a + first_b + _CHAIN_LENGTH
            second_expected = second_a + second_b + _CHAIN_LENGTH
            assert torch.allclose(first_out, first_expected), (
                f"first output mismatch: got={first_out[0].item()} expected={first_expected[0].item()}"
            )
            assert torch.allclose(second_out, second_expected), (
                f"second output mismatch: got={second_out[0].item()} expected={second_expected[0].item()}"
            )
        finally:
            _SUB_RELEASE.set()
            handles = [first, second]
            for run in handles:
                if run is not None:
                    with suppress(Exception):
                        run.wait(30.0)
            tensors.clear()
            tensor = None
            first_a = first_b = first_out = second_a = second_b = second_out = None
            submit_vector = first_graph = second_graph = None
            if all(run is None or run.done for run in handles):
                for buffer in buffers:
                    st_worker.free_host_buffer(buffer)
                _clear_signals()


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
