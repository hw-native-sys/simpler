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
        assert config.enable_l2_swimlane is False
        assert config.enable_dump_tensor is False
        assert config.enable_pmu == 0

    def test_setters(self):
        config = CallConfig()
        config.block_dim = 32
        config.aicpu_thread_num = 4
        config.enable_l2_swimlane = True
        assert config.block_dim == 32
        assert config.aicpu_thread_num == 4
        assert config.enable_l2_swimlane is True

    def test_diagnostics_subfeatures_are_parallel(self):
        # Guard against drift: the three diagnostics sub-features under the
        # profiling umbrella must all round-trip through the nanobind surface.
        config = CallConfig()
        config.enable_l2_swimlane = True
        config.enable_dump_tensor = True
        config.enable_pmu = 2
        assert config.enable_l2_swimlane is True
        assert config.enable_dump_tensor is True
        assert config.enable_pmu == 2
        r = repr(config)
        assert "enable_l2_swimlane=True" in r
        assert "enable_dump_tensor=True" in r
        assert "enable_pmu=2" in r

    def test_repr(self):
        config = CallConfig()
        r = repr(config)
        assert "block_dim=24" in r
        assert "enable_l2_swimlane=False" in r


# ============================================================================
# ChipWorker state machine tests
# ============================================================================


class TestChipWorkerStateMachine:
    def test_initial_state(self):
        worker = _ChipWorker()
        assert worker.initialized is False
        assert worker.device_id == -1

    def test_run_before_init_raises(self):
        from _task_interface import ChipCallable, ChipStorageTaskArgs  # noqa: PLC0415

        worker = _ChipWorker()
        config = CallConfig()
        args = ChipStorageTaskArgs()

        # Build a minimal ChipCallable for the test
        callable_obj = ChipCallable.build(signature=[], func_name="test", binary=b"\x00", children=[])

        with pytest.raises(RuntimeError, match="not initialized"):
            worker.run(callable_obj, args, config)

    def test_finalize_idempotent(self):
        worker = _ChipWorker()
        worker.finalize()
        worker.finalize()
        assert worker.initialized is False

    def test_init_after_finalize_raises(self):
        worker = _ChipWorker()
        worker.finalize()
        with pytest.raises(RuntimeError, match="finalized"):
            worker.init(
                "/nonexistent/libfoo.so", "/dev/null", "/dev/null", "/nonexistent/libsimpler_log.so", device_id=0
            )

    def test_init_with_nonexistent_lib_raises(self):
        worker = _ChipWorker()
        with pytest.raises(RuntimeError, match="dlopen"):
            worker.init(
                "/nonexistent/libfoo.so", "/dev/null", "/dev/null", "/nonexistent/libsimpler_log.so", device_id=0
            )

    def test_init_with_negative_device_id_raises(self):
        worker = _ChipWorker()
        with pytest.raises(RuntimeError, match="device_id"):
            worker.init("/nonexistent/libfoo.so", "/dev/null", "/dev/null", "/nonexistent/libsimpler_log.so", -1)


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
