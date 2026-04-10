# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: E402
"""Tests for ChipCallConfig and ChipWorker state machine."""

import sys
from pathlib import Path

import pytest

# Ensure python/ is on the import path so _task_interface and task_interface resolve
_python_dir = str(Path(__file__).resolve().parent.parent.parent / "python")
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

from _task_interface import ChipCallConfig, _ChipWorker  # pyright: ignore[reportMissingImports]

# ============================================================================
# ChipCallConfig tests
# ============================================================================


class TestChipCallConfig:
    def test_defaults(self):
        config = ChipCallConfig()
        assert config.block_dim == 24
        assert config.aicpu_thread_num == 3
        assert config.enable_profiling is False

    def test_setters(self):
        config = ChipCallConfig()
        config.block_dim = 32
        config.aicpu_thread_num = 4
        config.enable_profiling = True
        assert config.block_dim == 32
        assert config.aicpu_thread_num == 4
        assert config.enable_profiling is True

    def test_repr(self):
        config = ChipCallConfig()
        r = repr(config)
        assert "block_dim=24" in r
        assert "enable_profiling=False" in r


# ============================================================================
# ChipWorker state machine tests
# ============================================================================


class TestChipWorkerStateMachine:
    def test_initial_state(self):
        worker = _ChipWorker()
        assert worker.initialized is False
        assert worker.device_set is False
        assert worker.device_id == -1

    def test_run_before_set_device_raises(self):
        from _task_interface import ChipCallable, ChipStorageTaskArgs  # noqa: PLC0415

        worker = _ChipWorker()
        config = ChipCallConfig()
        args = ChipStorageTaskArgs()

        # Build a minimal ChipCallable for the test
        callable_obj = ChipCallable.build(signature=[], func_name="test", binary=b"\x00", children=[])

        with pytest.raises(RuntimeError, match="device not set"):
            worker.run(callable_obj, args, config)

    def test_set_device_before_init_raises(self):
        worker = _ChipWorker()
        with pytest.raises(RuntimeError, match="not initialized"):
            worker.set_device(0)

    def test_reset_device_idempotent(self):
        worker = _ChipWorker()
        # reset_device() on an uninitialized worker should not raise
        worker.reset_device()
        worker.reset_device()
        assert worker.device_set is False

    def test_finalize_idempotent(self):
        worker = _ChipWorker()
        worker.finalize()
        worker.finalize()
        assert worker.initialized is False

    def test_init_after_finalize_raises(self):
        worker = _ChipWorker()
        worker.finalize()
        with pytest.raises(RuntimeError, match="finalized"):
            worker.init("/nonexistent/libfoo.so", "/dev/null", "/dev/null")

    def test_init_with_nonexistent_lib_raises(self):
        worker = _ChipWorker()
        with pytest.raises(RuntimeError, match="dlopen"):
            worker.init("/nonexistent/libfoo.so", "/dev/null", "/dev/null")


# ============================================================================
# Python-level ChipWorker wrapper tests
# ============================================================================


class TestChipWorkerPython:
    def test_import(self):
        from simpler.task_interface import (  # noqa: PLC0415
            ChipCallConfig as PyChipCallConfig,  # pyright: ignore[reportAttributeAccessIssue]
        )
        from simpler.task_interface import ChipWorker  # noqa: PLC0415  # pyright: ignore[reportAttributeAccessIssue]

        worker = ChipWorker()
        assert worker.initialized is False
        assert worker.device_set is False
        assert isinstance(PyChipCallConfig(), ChipCallConfig)
