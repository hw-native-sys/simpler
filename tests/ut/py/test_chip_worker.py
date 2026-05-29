# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for CallConfig and ChipWorker state machine."""

import pytest
from _task_interface import CallConfig, _ChipWorker  # pyright: ignore[reportMissingImports]

# ============================================================================
# CallConfig tests
# ============================================================================


class TestCallConfig:
    def test_defaults(self):
        config = CallConfig()
        assert config.block_dim == 24
        assert config.aicpu_thread_num == 3
        assert config.enable_l2_swimlane == 0
        assert config.enable_dump_tensor is False
        assert config.enable_pmu == 0
        assert config.enable_dep_gen is False

    def test_setters(self):
        # enable_l2_swimlane accepts both an int perf_level (0-4) and a Python
        # bool. `True` maps to level 4 (preserves the pre-perf_level "fully on"
        # semantics for legacy callers); explicit ints select a specific level.
        config = CallConfig()
        config.block_dim = 32
        config.aicpu_thread_num = 4
        config.enable_l2_swimlane = True
        assert config.block_dim == 32
        assert config.aicpu_thread_num == 4
        assert config.enable_l2_swimlane == 4
        config.enable_l2_swimlane = 2
        assert config.enable_l2_swimlane == 2
        config.enable_l2_swimlane = False
        assert config.enable_l2_swimlane == 0

    def test_diagnostics_subfeatures_are_parallel(self):
        # Guard against drift: the four diagnostics sub-features under the
        # profiling umbrella must all round-trip through the nanobind surface.
        config = CallConfig()
        config.enable_l2_swimlane = True
        config.enable_dump_tensor = True
        config.enable_pmu = 2
        config.enable_dep_gen = True
        assert config.enable_l2_swimlane == 4
        assert config.enable_dump_tensor is True
        assert config.enable_pmu == 2
        assert config.enable_dep_gen is True
        r = repr(config)
        assert "enable_l2_swimlane=4" in r
        assert "enable_dump_tensor=True" in r
        assert "enable_pmu=2" in r
        assert "enable_dep_gen=True" in r

    def test_repr(self):
        config = CallConfig()
        r = repr(config)
        assert "block_dim=24" in r
        assert "enable_l2_swimlane=0" in r


# ============================================================================
# ChipWorker state machine tests
# ============================================================================


class TestChipWorkerStateMachine:
    def test_initial_state(self):
        worker = _ChipWorker()
        assert worker.initialized is False
        assert worker.device_id == -1

    def test_finalize_idempotent(self):
        worker = _ChipWorker()
        worker.finalize()
        worker.finalize()
        assert worker.initialized is False

    def test_init_after_finalize_raises(self):
        worker = _ChipWorker()
        worker.finalize()
        with pytest.raises(RuntimeError, match="finalized"):
            worker.init("/nonexistent/libfoo.so", "/dev/null", "/dev/null", device_id=0)

    def test_init_with_nonexistent_lib_raises(self):
        worker = _ChipWorker()
        with pytest.raises(RuntimeError, match="dlopen"):
            worker.init("/nonexistent/libfoo.so", "/dev/null", "/dev/null", device_id=0)

    def test_init_with_negative_device_id_raises(self):
        worker = _ChipWorker()
        with pytest.raises(RuntimeError, match="device_id"):
            worker.init("/nonexistent/libfoo.so", "/dev/null", "/dev/null", -1)

    def test_prepare_callable_before_init_raises(self):
        from _task_interface import ChipCallable  # noqa: PLC0415

        worker = _ChipWorker()
        callable_obj = ChipCallable.build(signature=[], func_name="test", binary=b"\x00", children=[])
        with pytest.raises(RuntimeError, match="not initialized"):
            worker.prepare_callable(0, callable_obj)

    def test_prepare_callable_from_blob_before_init_raises(self):
        # The from_blob overload shares the underlying ChipWorker::prepare_callable
        # entrypoint with the typed overload, so it must enforce the same
        # initialization guard. This protects the dynamic-register IPC handler
        # (which is the sole caller) from silently no-op'ing on a stale worker.
        from _task_interface import ChipCallable  # noqa: PLC0415

        worker = _ChipWorker()
        callable_obj = ChipCallable.build(signature=[], func_name="test", binary=b"\x00", children=[])
        with pytest.raises(RuntimeError, match="not initialized"):
            worker.prepare_callable_from_blob(0, callable_obj.buffer_ptr())

    def test_run_before_init_raises(self):
        from _task_interface import ChipStorageTaskArgs  # noqa: PLC0415

        worker = _ChipWorker()
        config = CallConfig()
        args = ChipStorageTaskArgs()
        with pytest.raises(RuntimeError, match="not initialized"):
            worker.run(0, args, config)

    def test_unregister_callable_before_init_raises(self):
        worker = _ChipWorker()
        with pytest.raises(RuntimeError, match="not initialized"):
            worker.unregister_callable(0)


# ============================================================================
# Python-level ChipWorker wrapper tests
# ============================================================================


class TestChipWorkerPython:
    def test_import(self):
        from simpler.task_interface import (  # noqa: PLC0415
            CallConfig as PyCallConfig,  # pyright: ignore[reportAttributeAccessIssue]
        )
        from simpler.task_interface import ChipWorker  # noqa: PLC0415  # pyright: ignore[reportAttributeAccessIssue]

        worker = ChipWorker()
        assert worker.initialized is False
        assert isinstance(PyCallConfig(), CallConfig)

    def test_cpp_chip_worker_exposes_role_keyed_init(self):
        worker = _ChipWorker()
        assert hasattr(worker, "init_roles")

    def test_init_accepts_cuda_role_only_runtime_binaries(self, monkeypatch, tmp_path):
        from simpler import task_interface as task_interface_module  # noqa: PLC0415
        from simpler.task_interface import ChipWorker  # noqa: PLC0415  # pyright: ignore[reportAttributeAccessIssue]

        class FakeLogInit:
            def __init__(self):
                self.calls = []

            def __call__(self, log_level, log_info_v):
                self.calls.append((log_level, log_info_v))
                return 0

        class FakeLogHandle:
            def __init__(self):
                self.simpler_log_init = FakeLogInit()

        class FakeImpl:
            def __init__(self):
                self.init_args = None
                self.init_roles_args = None

            def init(self, *args):
                self.init_args = args

            def init_roles(self, *args):
                self.init_roles_args = args

        class RoleOnlyBins:
            simpler_log_path = tmp_path / "libsimpler_log.so"
            sim_context_path = None

            def __init__(self):
                self.role_paths = {
                    "host": tmp_path / "libhost_runtime.so",
                    "device": tmp_path / "libcuda_device_runtime.so",
                }

            def path_for_role(self, role):
                return self.role_paths[role]

        fake_log_handle = FakeLogHandle()
        monkeypatch.setattr(task_interface_module, "_preload_global", lambda path: fake_log_handle)

        worker = ChipWorker()
        fake_impl = FakeImpl()
        worker._impl = fake_impl

        worker.init(0, RoleOnlyBins(), log_level=1, log_info_v=2)

        device_path = str(tmp_path / "libcuda_device_runtime.so")
        assert fake_log_handle.simpler_log_init.calls == [(1, 2)]
        assert fake_impl.init_args is None
        assert fake_impl.init_roles_args == (
            {
                "host": str(tmp_path / "libhost_runtime.so"),
                "device": device_path,
            },
            0,
        )
