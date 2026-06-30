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
from _task_interface import CallConfig, RuntimeEnv, _ChipWorker  # pyright: ignore[reportMissingImports]

# ============================================================================
# CallConfig tests
# ============================================================================


class TestCallConfig:
    def test_defaults(self):
        config = CallConfig()
        # 0 is the "auto" sentinel — DeviceRunner resolves it at run() time
        # to the max the AICore stream allows.
        assert config.block_dim == 0
        assert config.aicpu_thread_num == 3
        assert config.enable_l2_swimlane == 0
        assert config.enable_dump_tensor == 0
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
        # enable_dump_tensor is likewise a level (0=off, 1=partial, 2=full,
        # 3=full_json_only): `True` maps to level 1 (partial), explicit ints
        # select the level.
        config.enable_dump_tensor = True
        assert config.enable_dump_tensor == 1
        config.enable_dump_tensor = 2
        assert config.enable_dump_tensor == 2
        config.enable_dump_tensor = 3
        assert config.enable_dump_tensor == 3
        config.enable_dump_tensor = False
        assert config.enable_dump_tensor == 0

    def test_diagnostics_subfeatures_are_parallel(self):
        # Guard against drift: the four diagnostics sub-features under the
        # profiling umbrella must all round-trip through the nanobind surface.
        config = CallConfig()
        config.enable_l2_swimlane = True
        config.enable_dump_tensor = True
        config.enable_pmu = 2
        config.enable_dep_gen = True
        assert config.enable_l2_swimlane == 4
        assert config.enable_dump_tensor == 1
        assert config.enable_pmu == 2
        assert config.enable_dep_gen is True
        r = repr(config)
        assert "enable_l2_swimlane=4" in r
        assert "enable_dump_tensor=1" in r
        assert "enable_pmu=2" in r
        assert "enable_dep_gen=True" in r

    def test_repr(self):
        config = CallConfig()
        r = repr(config)
        assert "block_dim=0" in r
        assert "enable_l2_swimlane=0" in r
        # Ring sizing only shows in repr when set.
        assert "ring_heap" not in r

    def test_runtime_env_defaults_and_roundtrip(self):
        config = CallConfig()
        # Each resource reads back as a 4-entry list; unset = all zeros.
        assert config.runtime_env.ring_task_window == [0, 0, 0, 0]
        assert config.runtime_env.ring_heap == [0, 0, 0, 0]
        assert config.runtime_env.ring_dep_pool == [0, 0, 0, 0]
        # A scalar broadcasts to every ring...
        config.runtime_env.ring_task_window = 64
        assert config.runtime_env.ring_task_window == [64, 64, 64, 64]
        # ...a list sizes each scope-depth ring independently.
        config.runtime_env.ring_task_window = [16, 32, 128, 256]
        config.runtime_env.ring_heap = [
            10 * 1024 * 1024,
            64 * 1024 * 1024,
            1536 * 1024 * 1024,
            4 * 1024 * 1024 * 1024,
        ]
        config.runtime_env.ring_dep_pool = [64, 128, 256, 512]
        assert config.runtime_env.ring_task_window == [16, 32, 128, 256]
        assert config.runtime_env.ring_heap == [
            10 * 1024 * 1024,
            64 * 1024 * 1024,
            1536 * 1024 * 1024,
            4 * 1024 * 1024 * 1024,
        ]
        assert config.runtime_env.ring_dep_pool == [64, 128, 256, 512]
        config.validate()
        r = repr(config)
        assert "runtime_env.ring_task_window=[16, 32, 128, 256]" in r
        assert "runtime_env.ring_dep_pool=[64, 128, 256, 512]" in r

    def test_runtime_env_whole_object_assignment(self):
        re = RuntimeEnv()
        re.ring_heap = 1024  # scalar broadcasts to every ring
        config = CallConfig()
        config.runtime_env = re
        assert config.runtime_env.ring_heap == [1024, 1024, 1024, 1024]

        re2 = RuntimeEnv()
        re2.ring_heap = [1024, 2048, 3072, 4096]  # per-ring list
        config.runtime_env = re2
        assert config.runtime_env.ring_heap == [1024, 2048, 3072, 4096]

    def test_runtime_env_per_ring_length_validation(self):
        config = CallConfig()
        with pytest.raises(ValueError):
            config.runtime_env.ring_task_window = [16, 32, 64]  # must be exactly 4 entries

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("ring_task_window", 3),  # below min 4
            ("ring_task_window", 48),  # not a power of 2
            ("ring_heap", 512),  # below min 1024
            ("ring_dep_pool", 3),  # below min 4
            ("ring_dep_pool", 2**31),  # above INT32_MAX
        ],
    )
    def test_runtime_env_validate_rejects(self, field, value):
        config = CallConfig()
        setattr(config.runtime_env, field, value)
        with pytest.raises(ValueError):
            config.validate()

    def test_runtime_env_per_ring_validate_rejects(self):
        config = CallConfig()
        config.runtime_env.ring_task_window = [16, 32, 48, 64]  # 48 not a power of 2
        with pytest.raises(ValueError):
            config.validate()

        config = CallConfig()
        config.runtime_env.ring_heap = [1024, 512, 2048, 4096]  # 512 below min 1024
        with pytest.raises(ValueError):
            config.validate()

        config = CallConfig()
        config.runtime_env.ring_dep_pool = [4, 8, 2**31, 16]  # 2**31 above INT32_MAX
        with pytest.raises(ValueError):
            config.validate()


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
            worker.init("/nonexistent/libfoo.so", "/dev/null", "/dev/null", "", device_id=0)

    def test_init_with_nonexistent_lib_raises(self):
        worker = _ChipWorker()
        with pytest.raises(RuntimeError, match="dlopen"):
            worker.init("/nonexistent/libfoo.so", "/dev/null", "/dev/null", "", device_id=0)

    def test_init_with_negative_device_id_raises(self):
        worker = _ChipWorker()
        with pytest.raises(RuntimeError, match="device_id"):
            worker.init("/nonexistent/libfoo.so", "/dev/null", "/dev/null", "", -1)

    def test_register_callable_before_init_raises(self):
        from _task_interface import ChipCallable  # noqa: PLC0415

        worker = _ChipWorker()
        callable_obj = ChipCallable.build(signature=[], func_name="test", binary=b"\x00", children=[])
        with pytest.raises(RuntimeError, match="not initialized"):
            worker.register_callable(0, callable_obj)

    def test_register_callable_from_blob_before_init_raises(self):
        # The from_blob overload shares the underlying ChipWorker::register_callable
        # entrypoint with the typed overload, so it must enforce the same
        # initialization guard. This protects the dynamic-register IPC handler
        # (which is the sole caller) from silently no-op'ing on a stale worker.
        from _task_interface import ChipCallable  # noqa: PLC0415

        worker = _ChipWorker()
        callable_obj = ChipCallable.build(signature=[], func_name="test", binary=b"\x00", children=[])
        with pytest.raises(RuntimeError, match="not initialized"):
            worker.register_callable_from_blob(0, callable_obj.buffer_ptr())

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

    def test_l3_l2_orch_comm_init_before_init_raises(self):
        worker = _ChipWorker()
        with pytest.raises(RuntimeError, match="not initialized"):
            worker.l3_l2_orch_comm_init_from_addr(0x1000, 4096)

    def test_l3_l2_orch_comm_shutdown_before_init_raises(self):
        worker = _ChipWorker()
        with pytest.raises(RuntimeError, match="not initialized"):
            worker.l3_l2_orch_comm_shutdown()


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

    def test_public_wrapper_uses_handle_and_private_slot(self):
        from _task_interface import ChipCallable, ChipStorageTaskArgs  # noqa: PLC0415
        from simpler.callable_identity import CallableHandle  # noqa: PLC0415
        from simpler.task_interface import ChipWorker  # noqa: PLC0415  # pyright: ignore[reportAttributeAccessIssue]

        class FakeImpl:
            initialized = True
            device_id = 0

            def __init__(self):
                self.prepared = []
                self.runs = []
                self.unregistered = []
                self.aicpu_dlopen_count = 0
                self.host_dlopen_count = 0

            def register_callable(self, slot, callable_obj):
                self.prepared.append((slot, callable_obj))

            def run(self, slot, args, config):
                self.runs.append((slot, args, config))

            def unregister_callable(self, slot):
                self.unregistered.append(slot)

        worker = ChipWorker()
        fake = FakeImpl()
        worker._impl = fake
        callable_obj = ChipCallable.build(signature=[], func_name="test", binary=b"\x00", children=[])

        first = worker.register_callable(callable_obj)
        second = worker.register_callable(callable_obj)

        assert isinstance(first, CallableHandle)
        assert not isinstance(first, int)
        assert first.hashid == second.hashid
        assert fake.prepared == [(0, callable_obj)]

        args = ChipStorageTaskArgs()
        # run() returns None now; verify dispatch via the recorded call.
        assert worker.run(first, args, CallConfig()) is None
        assert fake.runs[0][0] == 0

        worker.unregister_callable(first)
        assert fake.unregistered == []
        worker.unregister_callable(second)
        assert fake.unregistered == [0]

    def test_public_wrapper_rejects_raw_slot_run(self):
        from _task_interface import ChipStorageTaskArgs  # noqa: PLC0415
        from simpler.task_interface import ChipWorker  # noqa: PLC0415  # pyright: ignore[reportAttributeAccessIssue]

        worker = ChipWorker()
        with pytest.raises(TypeError, match="CallableHandle returned by ChipWorker.register_callable"):
            worker.run(0, ChipStorageTaskArgs(), CallConfig())


# ============================================================================
# Mailbox CallConfig wire round-trip
# ============================================================================


class TestMailboxConfigRoundtrip:
    def test_config_roundtrip(self):
        # Guards the worker mailbox ABI: pack a CallConfig with _CFG_FMT, then
        # decode it with _read_config_from_mailbox and assert every field
        # survives. Catches field-order / offset drift in the packed layout
        # before it surfaces as a forked-worker failure.
        from simpler.worker import (  # noqa: PLC0415  # pyright: ignore[reportAttributeAccessIssue]
            _CFG_FMT,
            _OFF_CONFIG,
            _read_config_from_mailbox,
        )

        cfg = CallConfig()
        cfg.block_dim = 7
        cfg.aicpu_thread_num = 2
        cfg.enable_l2_swimlane = 3
        cfg.enable_dump_tensor = 2
        cfg.enable_pmu = 5
        cfg.enable_dep_gen = True
        cfg.enable_scope_stats = True
        cfg.runtime_env.ring_task_window = [16, 32, 128, 256]
        cfg.runtime_env.ring_heap = [1024, 2048, 4096, 8192]
        cfg.runtime_env.ring_dep_pool = [64, 128, 256, 512]
        cfg.output_prefix = "/tmp/out"

        buf = bytearray(_OFF_CONFIG + _CFG_FMT.size)
        _CFG_FMT.pack_into(
            buf,
            _OFF_CONFIG,
            cfg.block_dim,
            cfg.aicpu_thread_num,
            cfg.enable_l2_swimlane,
            int(cfg.enable_dump_tensor),
            cfg.enable_pmu,
            int(cfg.enable_dep_gen),
            int(cfg.enable_scope_stats),
            *cfg.runtime_env.ring_task_window,
            *cfg.runtime_env.ring_heap,
            *cfg.runtime_env.ring_dep_pool,
            cfg.output_prefix.encode(),
        )

        decoded = _read_config_from_mailbox(memoryview(buf))
        assert decoded.block_dim == 7
        assert decoded.aicpu_thread_num == 2
        assert decoded.enable_l2_swimlane == 3
        assert decoded.enable_dump_tensor == 2
        assert decoded.enable_pmu == 5
        assert decoded.enable_dep_gen is True
        assert decoded.enable_scope_stats is True
        assert decoded.runtime_env.ring_task_window == [16, 32, 128, 256]
        assert decoded.runtime_env.ring_heap == [1024, 2048, 4096, 8192]
        assert decoded.runtime_env.ring_dep_pool == [64, 128, 256, 512]
        assert decoded.output_prefix == "/tmp/out"
