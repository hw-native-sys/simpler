# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for CallConfig and ChipWorker state machine."""

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest
from _task_interface import CallConfig, RuntimeEnv, _ChipWorker  # pyright: ignore[reportMissingImports]

# ============================================================================
# CallConfig tests
# ============================================================================


class TestCallConfig:
    def test_defaults(self):
        config = CallConfig()
        # 0 is the "auto" sentinel for the per-architecture runtime default.
        assert config.aicpu_thread_num == 0
        assert config.enable_chip_swimlane == 0
        assert config.enable_dump_args == 0
        assert config.enable_pmu == 0
        assert config.enable_dep_gen is False

    def test_setters(self):
        # enable_chip_swimlane accepts both an int perf_level (0-4) and a Python
        # bool. `True` maps to level 4 (preserves the pre-perf_level "fully on"
        # semantics for legacy callers); explicit ints select a specific level.
        config = CallConfig()
        config.aicpu_thread_num = 4
        config.enable_chip_swimlane = True
        assert config.aicpu_thread_num == 4
        assert config.enable_chip_swimlane == 4
        config.enable_chip_swimlane = 2
        assert config.enable_chip_swimlane == 2
        config.enable_chip_swimlane = False
        assert config.enable_chip_swimlane == 0
        # enable_dump_args is likewise a level (0=off, 1=partial, 2=full,
        # 3=hybrid): `True` maps to level 1 (partial), explicit ints
        # select the level.
        config.enable_dump_args = True
        assert config.enable_dump_args == 1
        config.enable_dump_args = 2
        assert config.enable_dump_args == 2
        config.enable_dump_args = 3
        assert config.enable_dump_args == 3
        config.enable_dump_args = False
        assert config.enable_dump_args == 0

    def test_diagnostics_subfeatures_are_parallel(self):
        # Guard against drift: the four diagnostics sub-features under the
        # profiling umbrella must all round-trip through the nanobind surface.
        config = CallConfig()
        config.enable_chip_swimlane = True
        config.enable_dump_args = True
        config.enable_pmu = 2
        config.enable_dep_gen = True
        assert config.enable_chip_swimlane == 4
        assert config.enable_dump_args == 1
        assert config.enable_pmu == 2
        assert config.enable_dep_gen is True
        r = repr(config)
        assert "enable_chip_swimlane=4" in r
        assert "enable_dump_args=1" in r
        assert "enable_pmu=2" in r
        assert "enable_dep_gen=True" in r

    def test_repr(self):
        config = CallConfig()
        r = repr(config)
        assert "enable_chip_swimlane=0" in r
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


_KERNEL_LIFECYCLE_SYMBOLS = (
    "simpler_kernel_mode_init",
    "simpler_kernel_mode_prepare_callable",
    "simpler_kernel_mode_launch",
)


@pytest.fixture(scope="module")
def kernel_symbol_runtime(tmp_path_factory):
    """Build real DSOs with a selectable kernel capability and export surface."""
    compiler = shutil.which("c++") or shutil.which("g++")
    assert compiler is not None, "kernel symbol tests require a C++ compiler"
    build_dir = tmp_path_factory.mktemp("kernel_symbol_runtime")
    worker_headers = Path(__file__).resolve().parents[3] / "src" / "common" / "worker"
    unused_symbols = """
        device_malloc_ctx device_free_ctx committed_device_memory_ctx device_memory_info_ctx
        copy_to_device_ctx copy_from_device_ctx simpler_register_callable simpler_run
        simpler_prepare_run simpler_launch_run simpler_poll_run simpler_wait_run simpler_finalize_run
        supports_concurrent_native_prepare_ctx get_arena_bank_gm_heap_base_ctx get_retained_temp_addr_ctx
        simpler_unregister_callable get_aicpu_dlopen_count get_host_dlopen_count get_run_stream_set_create_count
        ensure_acl_ready_ctx create_comm_stream_ctx destroy_comm_stream_ctx comm_init comm_alloc_windows
        comm_get_local_window_base comm_get_window_size comm_derive_context comm_alloc_domain_windows
        comm_release_domain_windows comm_global_domain_prepare comm_global_domain_import
        comm_global_domain_release comm_barrier comm_destroy
    """.split()
    cache = {}

    def build(*, supported=0, missing=(), init_result=0):
        key = (supported, missing, init_result)
        if key in cache:
            return cache[key]
        source = build_dir / f"runtime_{len(cache)}.cpp"
        sentinels = source.with_suffix(".sentinels.cpp")
        library = source.with_suffix(".so")
        source.write_text(
            '#include "runtime_c_api.h"\n'
            "#include <cstdlib>\n"
            "static int live_contexts = 0;\n"
            "struct ContextLeakCheck {\n"
            "    ~ContextLeakCheck() { if (live_contexts != 0) std::abort(); }\n"
            "};\n"
            "static ContextLeakCheck context_leak_check;\n"
            "struct SimplerHostLogState;\n"
            'extern "C" int simpler_host_log_bind_state(SimplerHostLogState *) { return 0; }\n'
            "DeviceContextHandle create_device_context() { ++live_contexts; return new uint64_t{0}; }\n"
            "void destroy_device_context(DeviceContextHandle ctx) {\n"
            "    --live_contexts; delete static_cast<uint64_t *>(ctx);\n"
            "}\n"
            "int finalize_device(DeviceContextHandle) { return 0; }\n"
            "size_t get_runtime_size() { return sizeof(uint64_t); }\n"
            "size_t get_runtime_alignment() { return alignof(uint64_t); }\n"
            "const PipelineContract *get_pipeline_contract() {\n"
            "    static const PipelineContract contract{PTO_PIPELINE_CONTRACT_ABI_VERSION, 2, 1, {\n"
            "        {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},\n"
            "        {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0}}};\n"
            "    return &contract;\n"
            "}\n"
            "int simpler_init(DeviceContextHandle ctx, int, const uint8_t *, size_t, const uint8_t *, size_t,\n"
            "                 const uint8_t *, size_t, const CallConfig *, int, const void *, uint64_t) {\n"
            "    *static_cast<uint64_t *>(ctx) = 1;\n"
            f"    return {init_result};\n"
            "}\n"
            + (
                "int simpler_kernel_mode_supported(DeviceContextHandle ctx) {\n"
                "    if (ctx == nullptr || *static_cast<uint64_t *>(ctx) != 0) std::abort();\n"
                f"    return {supported};\n"
                "}\n"
                if "simpler_kernel_mode_supported" not in missing
                else ""
            ),
            encoding="utf-8",
        )
        # These symbols are only resolved during init; any invocation is a test failure.
        # A separate TU keeps the sentinels independent of the C ABI parameter lists.
        sentinels.write_text(
            "#include <cstdlib>\n"
            + "".join(
                f'extern "C" void {symbol}() {{ std::abort(); }}\n'
                for symbol in (*unused_symbols, *_KERNEL_LIFECYCLE_SYMBOLS)
                if symbol not in missing
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-shared",
                "-fPIC",
                "-I",
                str(worker_headers),
                str(source),
                str(sentinels),
                "-o",
                str(library),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        cache[key] = library
        return library

    return build


class TestChipWorkerKernelSymbols:
    def test_unsupported_runtime_without_lifecycle_symbols(self, kernel_symbol_runtime):
        runtime = kernel_symbol_runtime(missing=_KERNEL_LIFECYCLE_SYMBOLS)
        worker = _ChipWorker()
        try:
            worker.init(str(runtime), os.devnull, os.devnull, "", device_id=0)
            assert worker.initialized
            assert worker.device_id == 0
        finally:
            worker.finalize()
        assert not worker.initialized
        assert worker.runtime_slot_count == 0

    def test_supported_runtime_with_complete_lifecycle_symbols(self, kernel_symbol_runtime):
        runtime = kernel_symbol_runtime(supported=1)
        worker = _ChipWorker()
        try:
            worker.init(str(runtime), os.devnull, os.devnull, "", device_id=0)
            assert worker.initialized
        finally:
            worker.finalize()

    @pytest.mark.parametrize("missing", (*_KERNEL_LIFECYCLE_SYMBOLS, "simpler_kernel_mode_supported"))
    def test_missing_required_kernel_symbol_allows_retry(self, kernel_symbol_runtime, missing):
        incomplete = kernel_symbol_runtime(supported=1, missing=(missing,))
        unsupported = kernel_symbol_runtime(missing=_KERNEL_LIFECYCLE_SYMBOLS)
        worker = _ChipWorker()
        try:
            with pytest.raises(RuntimeError, match=f"dlsym failed for '{missing}'"):
                worker.init(str(incomplete), os.devnull, os.devnull, "", device_id=0)
            assert not worker.initialized
            assert worker.device_id == -1
            assert worker.runtime_slot_count == 0
            worker.init(str(unsupported), os.devnull, os.devnull, "", device_id=0)
            assert worker.initialized
        finally:
            worker.finalize()

    def test_program_init_failure_after_kernel_symbol_resolution_allows_retry(self, kernel_symbol_runtime):
        failing = kernel_symbol_runtime(supported=1, init_result=-1000)
        unsupported = kernel_symbol_runtime(missing=_KERNEL_LIFECYCLE_SYMBOLS)
        worker = _ChipWorker()
        try:
            with pytest.raises(RuntimeError, match="simpler_init failed with code -1000"):
                worker.init(str(failing), os.devnull, os.devnull, "", device_id=0)
            assert not worker.initialized
            assert worker.runtime_slot_count == 0
            worker.init(str(unsupported), os.devnull, os.devnull, "", device_id=0)
            assert worker.initialized
        finally:
            worker.finalize()


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
            worker.run(0, ChipStorageTaskArgs(), CallConfig())  # pyright: ignore[reportArgumentType]

    def test_public_wrapper_rejects_cross_thread_finalize(self):
        from simpler.task_interface import ChipWorker  # noqa: PLC0415  # pyright: ignore[reportAttributeAccessIssue]

        class FakeImpl:
            initialized = True
            device_id = 0

            def finalize(self):
                raise AssertionError("foreign thread reached native finalize")

        worker = ChipWorker()
        worker._impl = FakeImpl()
        worker._init_owner_thread = threading.current_thread()
        result = []

        def finalize_from_foreign_thread():
            try:
                worker.finalize()
            except BaseException as exc:  # noqa: BLE001
                result.append(exc)

        thread = threading.Thread(target=finalize_from_foreign_thread)
        thread.start()
        thread.join()

        assert len(result) == 1 and isinstance(result[0], RuntimeError)
        assert "thread that called ChipWorker.init" in str(result[0])

    def test_public_wrapper_rejects_finalize_during_init(self):
        from simpler.task_interface import ChipWorker  # noqa: PLC0415  # pyright: ignore[reportAttributeAccessIssue]

        class FakeImpl:
            initialized = False
            device_id = 0

            def finalize(self):
                raise AssertionError("finalize ran while init was in progress")

        worker = ChipWorker()
        worker._impl = FakeImpl()
        worker._init_in_progress = True
        with pytest.raises(RuntimeError, match=r"while ChipWorker\.init\(\) is in progress"):
            worker.finalize()

    def test_public_wrapper_flush_failure_does_not_skip_finalize_cleanup(self, monkeypatch, capsys):
        import simpler.task_interface as task_interface_mod  # noqa: PLC0415
        from _task_interface import ChipCallable  # noqa: PLC0415
        from simpler.task_interface import ChipWorker  # noqa: PLC0415  # pyright: ignore[reportAttributeAccessIssue]

        finalized = []

        class FakeImpl:
            initialized = True
            device_id = 0

            def finalize(self):
                finalized.append(True)

        def fail_flush(_timeout_ms):
            raise RuntimeError("injected host-log flush failure")

        worker = ChipWorker()
        worker._impl = FakeImpl()
        worker._callable_registry[0] = ChipCallable.build(signature=[], func_name="test", binary=b"\x00", children=[])
        worker._identity_registry[b"digest"] = object()
        worker._live_handles[1] = b"digest"
        monkeypatch.setattr(task_interface_mod, "_flush_host_log", fail_flush)

        worker.finalize()

        assert finalized == [True]
        assert worker._callable_registry == {}
        assert worker._identity_registry == {}
        assert worker._live_handles == {}
        expected_warning = (
            "WARNING: host-log flush failed during ChipWorker.finalize(): injected host-log flush failure"
        )
        assert expected_warning in capsys.readouterr().err

    def test_public_wrapper_flush_timeout_is_reported_with_loss_counters(self, monkeypatch, capsys):
        import simpler.task_interface as task_interface_mod  # noqa: PLC0415
        from simpler.task_interface import ChipWorker  # noqa: PLC0415  # pyright: ignore[reportAttributeAccessIssue]

        class FakeImpl:
            initialized = True
            device_id = 0

            def finalize(self):
                pass

        worker = ChipWorker()
        worker._impl = FakeImpl()
        monkeypatch.setattr(task_interface_mod, "_flush_host_log", lambda _timeout_ms: False)
        monkeypatch.setattr(task_interface_mod, "_host_log_pending_records", lambda: 7)
        monkeypatch.setattr(task_interface_mod, "_host_log_dropped_records", lambda: 3)

        worker.finalize()

        warning = capsys.readouterr().err
        assert "host-log flush timed out after 1000 ms during ChipWorker.finalize()" in warning
        assert "pending_records=7, dropped_records=3" in warning
        assert "accepted records may be lost" in warning


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
        cfg.aicpu_thread_num = 2
        cfg.enable_chip_swimlane = 3
        cfg.enable_dump_args = 2
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
            cfg.aicpu_thread_num,
            cfg.enable_chip_swimlane,
            int(cfg.enable_dump_args),
            cfg.enable_pmu,
            int(cfg.enable_dep_gen),
            int(cfg.enable_scope_stats),
            int(cfg.capture_clock_anchors),
            *cfg.runtime_env.ring_task_window,
            *cfg.runtime_env.ring_heap,
            *cfg.runtime_env.ring_dep_pool,
            cfg.output_prefix.encode(),
        )

        decoded = _read_config_from_mailbox(memoryview(buf))
        assert decoded.aicpu_thread_num == 2
        assert decoded.enable_chip_swimlane == 3
        assert decoded.enable_dump_args == 2
        assert decoded.enable_pmu == 5
        assert decoded.enable_dep_gen is True
        assert decoded.enable_scope_stats is True
        assert decoded.runtime_env.ring_task_window == [16, 32, 128, 256]
        assert decoded.runtime_env.ring_heap == [1024, 2048, 4096, 8192]
        assert decoded.runtime_env.ring_dep_pool == [64, 128, 256, 512]
        assert decoded.output_prefix == "/tmp/out"
        assert decoded.capture_clock_anchors is False

        ranked = _read_config_from_mailbox(memoryview(buf), chip_rank=2, capture_index=7)
        assert ranked.output_prefix == "/tmp/out/rank2/d7"
        assert ranked.capture_clock_anchors is True

    def test_rank_directory_covers_every_diagnostic_but_anchors_stay_swimlane_only(self):
        # rankN/dN separates one ChipWorker child's artifacts from its siblings',
        # which every diagnostic needs; capture_clock_anchors only turns on the
        # Host/Device clock anchors, which only the swimlane reader consumes.
        from simpler.worker import (  # noqa: PLC0415  # pyright: ignore[reportAttributeAccessIssue]
            _CFG_FMT,
            _OFF_CONFIG,
            _read_config_from_mailbox,
        )

        def decode(**flags):
            cfg = CallConfig()
            cfg.output_prefix = "/tmp/out"
            for name, value in flags.items():
                setattr(cfg, name, value)
            buf = bytearray(_OFF_CONFIG + _CFG_FMT.size)
            _CFG_FMT.pack_into(
                buf,
                _OFF_CONFIG,
                cfg.aicpu_thread_num,
                cfg.enable_chip_swimlane,
                int(cfg.enable_dump_args),
                cfg.enable_pmu,
                int(cfg.enable_dep_gen),
                int(cfg.enable_scope_stats),
                int(cfg.capture_clock_anchors),
                *cfg.runtime_env.ring_task_window,
                *cfg.runtime_env.ring_heap,
                *cfg.runtime_env.ring_dep_pool,
                cfg.output_prefix.encode(),
            )
            return _read_config_from_mailbox(memoryview(buf), chip_rank=1, capture_index=0)

        dep_gen_only = decode(enable_dep_gen=True)
        assert dep_gen_only.output_prefix == "/tmp/out/rank1/d0"
        assert dep_gen_only.capture_clock_anchors is False

        swimlane = decode(enable_chip_swimlane=4)
        assert swimlane.output_prefix == "/tmp/out/rank1/d0"
        assert swimlane.capture_clock_anchors is True

        # No diagnostic at all: nothing is written below output_prefix, so the
        # child leaves the case root alone.
        assert decode().output_prefix == "/tmp/out"

    def test_dispatch_identity_sidecar_uses_parent_dag_slot(self, tmp_path):
        from simpler.worker import (  # noqa: PLC0415  # pyright: ignore[reportAttributeAccessIssue]
            _TASK_PROTOCOL_VERSION,
            _write_dispatch_identity_sidecar,
        )

        _write_dispatch_identity_sidecar(
            str(tmp_path),
            frame_identity=(_TASK_PROTOCOL_VERSION, 17, 1, 23, 41, 5, 2, 4),
            chip_rank=2,
            capture_index=7,
            callable_digest=b"\xab" * 32,
        )

        identity = json.loads((tmp_path / "dispatch_identity.json").read_text())
        assert identity == {
            "schema_version": 1,
            "run_id": 17,
            "task_slot": 5,
            "group_index": 2,
            "group_size": 4,
            # The writer runs in the ChipWorker child, so this is that child's
            # own pid — the one whose `host.<pid>.log` brackets the capture.
            "host_pid": os.getpid(),
            "chip_rank": 2,
            "local_capture_index": 7,
            "endpoint_dispatch_id": 41,
            "pipeline_slot": 1,
            "pipeline_generation": 23,
            "callable_digest": "ab" * 32,
        }
