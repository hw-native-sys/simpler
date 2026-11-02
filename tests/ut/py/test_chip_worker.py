# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for CallConfig and ChipWorker state machine."""

import ctypes
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from _task_interface import CallConfig, RuntimeEnv, _ChipWorker  # pyright: ignore[reportMissingImports]
from simpler.buffer import Buffer
from simpler.task_interface import (
    ArgDirection,
    ChipCallable,
    ChipWorker,
    DataType,
    ExecutionContextRefused,
    TaskArgs,
    Tensor,
    TensorArgType,
)
from simpler.worker import Worker, _Lifecycle

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
        # enable_dump_args is likewise a level (0=off, 1=partial, 2=hybrid,
        # 3=full): `True` maps to level 1 (partial), explicit ints
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
        device_malloc_ctx device_free_ctx device_free_caller_buffer_ctx
        device_borrow_caller_buffers_ctx device_release_caller_buffers_ctx
        committed_device_memory_ctx device_memory_info_ctx
        copy_to_device_ctx copy_from_device_ctx simpler_register_callable simpler_run
        simpler_prepare_run simpler_launch_run simpler_launch_run_joined simpler_poll_run simpler_wait_run
        simpler_finalize_run supports_concurrent_native_prepare_ctx supports_joined_native_launch_ctx
        get_arena_bank_gm_heap_base_ctx get_retained_temp_addr_ctx
        simpler_unregister_callable get_aicpu_dlopen_count get_host_dlopen_count get_run_stream_set_create_count
        ensure_acl_ready_ctx create_comm_stream_ctx destroy_comm_stream_ctx comm_init comm_alloc_windows
        comm_get_local_window_base comm_get_window_size comm_derive_context comm_alloc_domain_windows
        comm_release_domain_windows comm_global_domain_prepare comm_global_domain_import
        comm_global_domain_release comm_global_domain_window comm_global_domain_copy comm_barrier comm_destroy
    """.split()
    cache = {}
    next_library_id = 0

    def build(  # noqa: PLR0913 -- independently selectable native lifecycle faults
        *,
        supported=0,
        missing=(),
        init_result=0,
        finalize_failures=0,
        borrowed=False,
        memory=False,
        early_failure=None,
        lifecycle_log=None,
        workspace=False,
        init_gate=False,
        unproven_cleanup=False,
        init_error=None,
    ):
        nonlocal next_library_id
        key = (
            supported,
            missing,
            init_result,
            finalize_failures,
            borrowed,
            memory,
            early_failure,
            lifecycle_log,
            workspace,
            init_gate,
            unproven_cleanup,
            init_error,
        )
        # Fault scripts retain their consumed failure budget in the loaded DSO.
        if finalize_failures == 0 and not init_gate and key in cache:
            return cache[key]
        source = build_dir / f"runtime_{next_library_id}.cpp"
        next_library_id += 1
        sentinels = source.with_suffix(".sentinels.cpp")
        library = source.with_suffix(".so")
        size = "0" if early_failure == "size" else "sizeof(uint64_t)"
        alignment = "3 * alignof(uint64_t)" if early_failure == "alignment" else "alignof(uint64_t)"
        releasable = (
            "0" if early_failure == "grant_cleanup" else "(context->initialized == 0 || context->cleanup_proven)"
        )
        source.write_text(
            '#include "runtime_c_api.h"\n'
            "#include <cstdlib>\n"
            "#include <cstdio>\n"
            "#include <memory>\n"
            "#include <atomic>\n"
            "#include <condition_variable>\n"
            "#include <chrono>\n"
            "#include <mutex>\n"
            f"static const char *lifecycle_log = {json.dumps(str(lifecycle_log)) if lifecycle_log else 'nullptr'};\n"
            "static void record(const char *event) {\n"
            "    if (!lifecycle_log) return;\n"
            '    FILE *file = std::fopen(lifecycle_log, "a");\n'
            "    if (!file) std::abort();\n"
            '    std::fprintf(file, "%s\\n", event);\n'
            "    std::fclose(file);\n"
            "}\n"
            "static std::atomic<int> live_contexts{0}, creates{0}, program_inits{0};\n"
            'extern "C" int test_creates() { return creates; }\n'
            'extern "C" int test_program_inits() { return program_inits; }\n'
            f"static bool gate_init = {'true' if init_gate else 'false'};\n"
            "static bool init_entered = false, init_released = false;\n"
            "static std::mutex init_mutex;\n"
            "static std::condition_variable init_changed;\n"
            'extern "C" int test_wait_init() {\n'
            "    std::unique_lock<std::mutex> lock(init_mutex);\n"
            "    return init_changed.wait_for(lock, std::chrono::seconds(5), [] { return init_entered; });\n"
            "}\n"
            'extern "C" void test_release_init() {\n'
            "    std::lock_guard<std::mutex> lock(init_mutex); init_released = true; init_changed.notify_all();\n"
            "}\n"
            "static void enter_init() {\n"
            "    std::unique_lock<std::mutex> lock(init_mutex);\n"
            "    if (!gate_init) return;\n"
            "    gate_init = false; init_entered = true; init_changed.notify_all();\n"
            "    init_changed.wait(lock, [] { return init_released; });\n"
            "}\n"
            "static int destroys = 0, closes = 0, borrowed_inits = 0, validation = 0;\n"
            'extern "C" int test_destroys() { return destroys; }\n'
            'extern "C" int test_closes() { return closes; }\n'
            'extern "C" int test_borrowed_inits() { return borrowed_inits; }\n'
            'extern "C" void test_validation(int rc) { validation = rc; }\n'
            "static int live_scratch = 0;\n"
            "struct Scratch {\n"
            '    Scratch() { ++live_scratch; record("scratch-created"); }\n'
            '    ~Scratch() { --live_scratch; record("scratch-destroyed"); }\n'
            "};\n"
            "struct Context { uint64_t initialized = 0, workspace_limit = 0; std::shared_ptr<void> scratch; "
            "bool cleanup_proven = false; };\n"
            "struct ContextLeakCheck {\n"
            "    ~ContextLeakCheck() {\n"
            "        if (live_contexts != 0 || live_scratch != 0) std::abort();\n"
            '        record("runtime-unloaded");\n'
            "    }\n"
            "};\n"
            "static ContextLeakCheck context_leak_check;\n"
            "struct SimplerHostLogState;\n"
            'extern "C" int simpler_host_log_bind_state(SimplerHostLogState *) { return 0; }\n'
            # ChipWorker binds the process's device-fault monitor into every
            # host runtime it loads, and does so strictly: a module that does
            # not export the setter is a stale build, not an opt-out. This stub
            # accepts the pointer and ignores it, as it does the log state.
            'extern "C" void simpler_bind_device_fault_monitor(void *) {}\n'
            "DeviceContextHandle create_device_context() { ++creates; ++live_contexts; return new Context; }\n"
            "void destroy_device_context(DeviceContextHandle ctx) {\n"
            "    ++destroys; --live_contexts; delete static_cast<Context *>(ctx);\n"
            '    record("context-destroyed");\n'
            "}\n"
            f"static int finalize_failures = {finalize_failures};\n"
            + (
                'extern "C" int simpler_execution_environment_releasable_ctx(DeviceContextHandle ctx) {\n'
                "    auto *context = static_cast<Context *>(ctx);\n"
                f"    return {releasable};\n"
                "}\n"
                if "simpler_execution_environment_releasable_ctx" not in missing
                else ""
            )
            + "int finalize_device(DeviceContextHandle ctx) {\n"
            "    ++closes;\n"
            "    if (finalize_failures > 0) { --finalize_failures; return -77; }\n"
            f"    static_cast<Context *>(ctx)->cleanup_proven = {'false' if unproven_cleanup else 'true'};\n"
            "    return 0;\n"
            "}\n"
            f"size_t get_runtime_size() {{ return {size}; }}\n"
            f"size_t get_runtime_alignment() {{ return {alignment}; }}\n"
            "const PipelineContract *get_pipeline_contract() {\n"
            "    static const PipelineContract contract{PTO_PIPELINE_CONTRACT_ABI_VERSION, 2, 1, {\n"
            "        {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},\n"
            "        {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0}}};\n"
            "    return &contract;\n"
            "}\n"
            "int simpler_init(DeviceContextHandle ctx, int, const uint8_t *, size_t, const uint8_t *, size_t,\n"
            "                 const uint8_t *, size_t, const CallConfig *, int, const void *, uint64_t) {\n"
            "    auto *context = static_cast<Context *>(ctx);\n"
            "    ++program_inits; enter_init();\n"
            "    context->initialized = 1;\n"
            "    context->scratch = std::make_shared<Scratch>();\n"
            f"    return {init_result};\n"
            "}\n"
            + (
                "int simpler_kernel_mode_supported(DeviceContextHandle ctx) {\n"
                "    if (ctx == nullptr || static_cast<Context *>(ctx)->initialized != 0) std::abort();\n"
                f"    return {supported};\n"
                "}\n"
                if "simpler_kernel_mode_supported" not in missing
                else ""
            ),
            encoding="utf-8",
        )
        if workspace:
            with source.open("a", encoding="utf-8") as output:
                output.write(
                    "int simpler_set_workspace_budget_ctx(DeviceContextHandle ctx, uint64_t limit) {\n"
                    "    if (static_cast<Context *>(ctx)->initialized != 0) std::abort();\n"
                    "    static_cast<Context *>(ctx)->workspace_limit = limit; return 0;\n"
                    "}\n"
                    "int simpler_get_workspace_report_ctx(DeviceContextHandle ctx, "
                    "SimplerWorkspaceReport *out, size_t) {\n"
                    "    const auto workspace_limit = static_cast<Context *>(ctx)->workspace_limit;\n"
                    "    *out = {}; out->schema = WORKSPACE_REPORT_SCHEMA;\n"
                    "    out->budget_enforced = workspace_limit != 0; out->limit_bytes = workspace_limit; return 0;\n"
                    "}\n"
                )
        if early_failure in ("grant", "grant_cleanup"):
            with source.open("a", encoding="utf-8") as output:
                output.write("int simpler_set_run_resource_grant_ctx(DeviceContextHandle, uint32_t) { return -89; }\n")
        if borrowed:
            with source.open("a", encoding="utf-8") as output:
                output.write(
                    "int simpler_kernel_mode_init(DeviceContextHandle ctx, int, const uint8_t *, size_t, "
                    "const uint8_t *, size_t, const uint8_t *, size_t, const CallConfig *cfg, uint64_t generation) {\n"
                    "    auto *context = static_cast<Context *>(ctx);\n"
                    "    enter_init();\n"
                    "    if (cfg == nullptr || generation == 0 || context->initialized != 0) std::abort();\n"
                    f"    ++borrowed_inits; context->initialized = {0 if init_result == -1001 else 2};\n"
                    f"    return {init_result};\n"
                    "}\n"
                    "int simpler_validate_execution_context(DeviceContextHandle) { return validation; }\n"
                    "static int registrations = 0, unregistrations = 0;\n"
                    'extern "C" int test_registrations() { return registrations; }\n'
                    'extern "C" int test_unregistrations() { return unregistrations; }\n'
                    "int simpler_register_callable(DeviceContextHandle, int32_t, const void *) {\n"
                    "    ++registrations; return validation;\n"
                    "}\n"
                    "int simpler_unregister_callable(DeviceContextHandle, int32_t) {\n"
                    "    ++unregistrations; return validation;\n"
                    "}\n"
                    "static int caller_validation = 0;\n"
                    'extern "C" void test_caller_validation(int rc) { caller_validation = rc; }\n'
                    "int simpler_validate_caller_stream(DeviceContextHandle, void *) { return caller_validation; }\n"
                    "int simpler_launch_run_on_stream(DeviceContextHandle, RuntimeHandle, void *, RuntimeHandle) "
                    "{ std::abort(); }\n"
                )

        memory_symbols = {
            "device_malloc_ctx",
            "device_free_ctx",
            "device_free_caller_buffer_ctx",
            "copy_to_device_ctx",
            "copy_from_device_ctx",
            "device_memory_info_ctx",
        }
        if "simpler_get_init_error_ctx" not in missing:
            with source.open("a", encoding="utf-8") as output:
                output.write(
                    'extern "C" const char *simpler_get_init_error_ctx(DeviceContextHandle ctx) {\n'
                    "    if (ctx == nullptr) std::abort();\n"
                    + ('    record("init-error-read");\n' if init_error is not None else "")
                    + f"    return {json.dumps(init_error) if init_error is not None else 'nullptr'};\n"
                    + "}\n"
                )
        if memory:
            with source.open("a", encoding="utf-8") as output:
                output.write(
                    "#include <cstring>\n"
                    "static int memory_calls = 0, allocated = 0;\n"
                    "static unsigned char backing[64][64];\n"
                    'extern "C" int test_memory_calls() { return memory_calls; }\n'
                    "void *device_malloc_ctx(DeviceContextHandle, size_t bytes) {\n"
                    "    ++memory_calls; if (bytes > 64 || allocated == 64) std::abort();\n"
                    "    return backing[allocated++];\n"
                    "}\n"
                    "void device_free_ctx(DeviceContextHandle, void *) { ++memory_calls; }\n"
                    "int device_free_caller_buffer_ctx(DeviceContextHandle, void *) { ++memory_calls; return 0; }\n"
                    "int copy_to_device_ctx(DeviceContextHandle, void *dst, const void *src, size_t bytes) {\n"
                    "    ++memory_calls; std::memcpy(dst, src, bytes); return 0;\n"
                    "}\n"
                    "int copy_from_device_ctx(DeviceContextHandle, void *dst, const void *src, size_t bytes) {\n"
                    "    ++memory_calls; std::memcpy(dst, src, bytes); return 0;\n"
                    "}\n"
                    "int device_memory_info_ctx(DeviceContextHandle, DeviceMemoryInfo *info) {\n"
                    "    ++memory_calls; *info = {}; return 0;\n"
                    "}\n"
                    "int simpler_flush_diagnostics_ctx(DeviceContextHandle, int32_t, char *, size_t) {\n"
                    "    ++memory_calls; return 0;\n"
                    "}\n"
                )

        # These symbols are only resolved during init; any invocation is a test failure.
        # A separate TU keeps the sentinels independent of the C ABI parameter lists.
        sentinels.write_text(
            "#include <cstdlib>\n"
            + "".join(
                f'extern "C" void {symbol}() {{ std::abort(); }}\n'
                for symbol in (*unused_symbols, *_KERNEL_LIFECYCLE_SYMBOLS)
                if symbol not in missing
                and not (memory and symbol in memory_symbols)
                and not (
                    borrowed
                    and symbol
                    in {"simpler_kernel_mode_init", "simpler_register_callable", "simpler_unregister_callable"}
                )
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
    @pytest.mark.parametrize("detail", ["", "actual operator timeout 4294967 us exceeds stream 4000 ms; adjust budget"])
    def test_program_init_failure_reports_context_detail_before_cleanup(self, kernel_symbol_runtime, tmp_path, detail):
        lifecycle_log = tmp_path / "init-error-lifecycle.log"
        runtime = kernel_symbol_runtime(init_result=-1000, init_error=detail, lifecycle_log=lifecycle_log)
        worker = _ChipWorker()
        try:
            with pytest.raises(RuntimeError) as error:
                worker.init(str(runtime), os.devnull, os.devnull, "", device_id=0)
            expected = "simpler_init failed with code -1000" + (f": {detail}" if detail else "")
            assert str(error.value) == expected
            assert not worker.initialized
            assert worker.device_id == -1
            assert lifecycle_log.read_text().splitlines() == [
                "scratch-created",
                "init-error-read",
                "scratch-destroyed",
                "context-destroyed",
                "runtime-unloaded",
            ]
            worker.init(str(kernel_symbol_runtime()), os.devnull, os.devnull, "", device_id=0)
            assert worker.initialized
        finally:
            worker.finalize()

    def test_failed_destructor_retains_runtime_code(self, kernel_symbol_runtime, tmp_path):
        lifecycle_log = tmp_path / "retained-lifecycle.log"
        runtime = kernel_symbol_runtime(lifecycle_log=lifecycle_log, finalize_failures=3)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent("""\
                    import gc
                    import os
                    import sys
                    from pathlib import Path
                    from _task_interface import _ChipWorker

                    worker = _ChipWorker()
                    worker.init(sys.argv[1], os.devnull, os.devnull, "", device_id=0)
                    try:
                        worker.finalize()
                    except RuntimeError as error:
                        assert "device teardown failed (-77)" in str(error)
                    else:
                        raise AssertionError("teardown unexpectedly succeeded")
                    del worker
                    gc.collect()
                    assert Path(sys.argv[2]).read_text().splitlines() == ["scratch-created"]
                    successor = _ChipWorker()
                    try:
                        successor.init(sys.argv[1], os.devnull, os.devnull, "", device_id=0)
                    except RuntimeError as error:
                        assert "device 0 already has an execution Worker" in str(error)
                    else:
                        raise AssertionError("failed destructor released the device reservation")
                    os._exit(0)
                    """),
                str(runtime),
                str(lifecycle_log),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "ChipWorker::~ChipWorker: teardown failed" in result.stderr
        assert lifecycle_log.read_text().splitlines() == ["scratch-created"]

    @pytest.mark.parametrize("exit_path", ["finalize", "init_failure", "finalize_retry"])
    def test_retained_scratch_is_destroyed_before_runtime_unload(self, kernel_symbol_runtime, tmp_path, exit_path):
        lifecycle_log = tmp_path / "lifecycle.log"
        runtime = kernel_symbol_runtime(
            lifecycle_log=lifecycle_log,
            init_result=-1000 if exit_path == "init_failure" else 0,
            finalize_failures=1 if exit_path == "finalize_retry" else 0,
        )
        worker = _ChipWorker()
        try:
            if exit_path == "init_failure":
                with pytest.raises(RuntimeError, match="simpler_init failed with code -1000"):
                    worker.init(str(runtime), os.devnull, os.devnull, "", device_id=0)
            else:
                worker.init(str(runtime), os.devnull, os.devnull, "", device_id=0)
                if exit_path == "finalize_retry":
                    with pytest.raises(RuntimeError, match=r"device teardown failed \(-77\)"):
                        worker.finalize()
                    assert lifecycle_log.read_text().splitlines() == ["scratch-created"]
                worker.finalize()
        finally:
            worker.finalize()
        assert lifecycle_log.read_text().splitlines() == [
            "scratch-created",
            "scratch-destroyed",
            "context-destroyed",
            "runtime-unloaded",
        ]

    def test_unsupported_runtime_still_exports_lifecycle_symbols(self, kernel_symbol_runtime):
        # A runtime that cannot run kernel mode reports so through supported()
        # and still exports the whole family, the same way the comm group ships
        # not-supported stubs rather than omitting symbols.
        runtime = kernel_symbol_runtime(supported=0)
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

    @pytest.mark.parametrize(
        "missing",
        (
            *_KERNEL_LIFECYCLE_SYMBOLS,
            "simpler_kernel_mode_supported",
            "simpler_execution_environment_releasable_ctx",
            "simpler_get_init_error_ctx",
        ),
    )
    def test_missing_required_runtime_symbol_allows_retry(self, kernel_symbol_runtime, missing):
        # Resolve the complete module contract before creating a context or
        # invoking either initialization path.
        incomplete = kernel_symbol_runtime(supported=1, missing=(missing,))
        complete = kernel_symbol_runtime(supported=0)
        observer = ctypes.CDLL(str(incomplete))
        before = observer.test_creates(), observer.test_program_inits(), observer.test_borrowed_inits()
        assert before == (0, 0, 0)
        worker = _ChipWorker()
        try:
            with pytest.raises(RuntimeError, match=f"dlsym failed for '{missing}'"):
                worker.init(str(incomplete), os.devnull, os.devnull, "", device_id=0)
            assert not worker.initialized
            assert worker.device_id == -1
            assert worker.runtime_slot_count == 0
            assert (observer.test_creates(), observer.test_program_inits(), observer.test_borrowed_inits()) == before
            worker.init(str(complete), os.devnull, os.devnull, "", device_id=0)
            assert worker.initialized
        finally:
            worker.finalize()

    def test_program_init_failure_after_kernel_symbol_resolution_allows_retry(self, kernel_symbol_runtime):
        failing = kernel_symbol_runtime(supported=1, init_result=-1000)
        complete = kernel_symbol_runtime(supported=0)
        worker = _ChipWorker()
        try:
            with pytest.raises(RuntimeError, match="simpler_init failed with code -1000"):
                worker.init(str(failing), os.devnull, os.devnull, "", device_id=0)
            assert not worker.initialized
            assert worker.runtime_slot_count == 0
            worker.init(str(complete), os.devnull, os.devnull, "", device_id=0)
            assert worker.initialized
        finally:
            worker.finalize()

    def test_native_finalize_failure_is_reported_and_retriable(self, kernel_symbol_runtime):
        runtime = kernel_symbol_runtime(finalize_failures=1)
        worker = _ChipWorker()
        worker.init(str(runtime), os.devnull, os.devnull, "", device_id=0)
        with pytest.raises(RuntimeError, match=r"device teardown failed \(-77\)"):
            worker.finalize()
        assert worker.initialized
        assert worker.device_id == 0
        worker.finalize()
        assert not worker.initialized


class TestWorkerNativeChip:
    @pytest.mark.parametrize("reject_prepare", [False, True])
    def test_chip_child_owns_native_runtime_and_finalizes_once(
        self, monkeypatch, kernel_symbol_runtime, reject_prepare
    ):
        import simpler.worker as worker_module  # noqa: PLC0415

        runtime = kernel_symbol_runtime(borrowed=True, memory=True)
        observer = ctypes.CDLL(str(runtime))
        registered, closed = observer.test_registrations(), observer.test_closes()
        target = ChipCallable.build(signature=[], func_name="child_native", binary=b"\0", children=[])
        bins = SimpleNamespace(host_path=runtime, aicpu_path=os.devnull, aicore_path=os.devnull)
        captured = []
        flushed = []

        def refuse_wrapper():
            pytest.fail("chip child must not construct the public ChipWorker wrapper")

        def serve(native, *_args, prepared, **_kwargs):
            assert isinstance(native, _ChipWorker)
            assert native.initialized
            assert native._execution_environment is not None
            assert native._execution_environment[4] is False
            assert prepared == {7}
            backing = native.malloc(16)
            payload = ctypes.create_string_buffer(b"0123456789abcdef")
            result = ctypes.create_string_buffer(16)
            native.copy_to(backing, ctypes.addressof(payload), 16)
            native.copy_from(ctypes.addressof(result), backing, 16)
            assert result.raw == payload.raw[:16]
            native.free(backing)
            captured.append(native)

        # `worker.py` reaches task_interface only through `from … import`, so a
        # reintroduced wrapper construction resolves against this module's globals
        # and nowhere else. The name is absent from them today, hence raising=False.
        monkeypatch.setattr(worker_module, "ChipWorker", refuse_wrapper, raising=False)
        monkeypatch.setattr(worker_module, "_run_chip_main_loop", serve)
        monkeypatch.setattr(worker_module, "_flush_host_log_or_warn", lambda context: flushed.append(context) or True)
        mailbox = memoryview(bytearray(worker_module.mailbox_size(2)))
        if reject_prepare:

            def fail_preparation(*_args, **_kwargs):
                raise RuntimeError("injected prepare failure")

            monkeypatch.setattr(worker_module, "_ensure_prepared", fail_preparation)
        worker_module._chip_process_loop(
            mailbox,
            bins,
            0,
            {7: target},
            {},
            {},
            worker_module.mint_owner_instance_id(),
            worker_module.mint_owner_instance_id(),
            worker_module.mint_owner_instance_id(),
            platform="a2a3sim",
            runtime="tensormap_and_ringbuffer",
        )
        assert observer.test_registrations() == registered + (not reject_prepare)
        assert observer.test_closes() == closed + 1
        assert len(flushed) == 1
        if reject_prepare:
            assert captured == []
            assert (
                worker_module._mailbox_load_i32(worker_module._buffer_field_addr(mailbox, worker_module._OFF_STATE))
                == worker_module._INIT_FAILED
            )
        else:
            assert len(captured) == 1
            assert not captured[0].initialized
            assert captured[0]._execution_environment is None

    def test_direct_worker_owns_native_chip_without_public_wrapper(self, monkeypatch, kernel_symbol_runtime):
        import simpler.worker as worker_module  # noqa: PLC0415

        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        runtime = kernel_symbol_runtime(borrowed=True, memory=True)
        observer = ctypes.CDLL(str(runtime))
        registered, unregistered, closed = (
            observer.test_registrations(),
            observer.test_unregistrations(),
            observer.test_closes(),
        )
        bins = SimpleNamespace(host_path=runtime, aicpu_path=os.devnull, aicore_path=os.devnull)
        monkeypatch.setattr(RuntimeBuilder, "get_binaries", lambda *_: bins)

        def refuse_wrapper():
            pytest.fail("direct L2 must not construct the public ChipWorker wrapper")

        monkeypatch.setattr(worker_module, "ChipWorker", refuse_wrapper, raising=False)
        owner = Worker(level=2, device_id=0, platform="a2a3sim", runtime="tensormap_and_ringbuffer")
        target = ChipCallable.build(signature=[], func_name="native_owner", binary=b"\0", children=[])
        handle = owner.register(target)
        try:
            owner.init()
            native = owner._chip_worker
            assert native is not None
            assert isinstance(native, _ChipWorker)
            assert native._execution_environment is not None
            assert native._execution_environment[4] is False
            assert owner._resolve_handle(handle).ref_count == 1
            assert observer.test_registrations() == registered + 1
            alias = owner.register(target)
            assert observer.test_registrations() == registered + 1
            backing = owner.malloc(16)
            data = bytearray(b"0123456789abcdef")
            owner.copy_to(backing, data)
            output = bytearray(16)
            owner.copy_from(output, backing)
            assert output == data
            owner.free(backing)
            source = Buffer.wrap(address=0x8000, nbytes=64, location=owner.device_location)
            source.close()
            owner.unregister(alias)
            assert observer.test_unregistrations() == unregistered
            owner.unregister(handle)
            assert observer.test_unregistrations() == unregistered + 1
        finally:
            owner.close()
        assert not native.initialized and native._execution_environment is None
        assert owner._chip_worker is None and observer.test_closes() == closed + 1

    @pytest.mark.parametrize("optional_paths", [False, True])
    def test_shared_initializer_maps_binary_paths_and_runtime_options(self, monkeypatch, optional_paths):
        import simpler.task_interface as task_module  # noqa: PLC0415

        from simpler_setup.runtime_builder import RuntimeBinaries  # noqa: PLC0415

        calls = []
        fake = SimpleNamespace(init=lambda *args: calls.append(args))
        log_levels = []
        monkeypatch.setattr(task_module, "_initialize_host_log", log_levels.append)
        bins = RuntimeBinaries(host_path=Path("host.so"), aicpu_path=Path("cpu.so"), aicore_path=Path("core.o"))
        if optional_paths:
            bins.dispatcher_path = Path("dispatcher.so")
            bins.sim_context_path = Path("sim.so")
            bins.sdma_warmup_path = Path("warmup.o")
        config = CallConfig()
        task_module._initialize_chip_runtime(
            fake,
            3,
            bins,
            log_level=30,
            prewarm_config=config,
            enable_sdma=True,
            collect_across_runs=True,
            workspace_budget_bytes=1234,
            manage_workspace=True,
            requested_pipeline_depth=4,
        )
        assert log_levels == [30]
        assert calls == [
            (
                "host.so",
                "cpu.so",
                "core.o",
                "dispatcher.so" if optional_paths else "",
                3,
                config,
                True,
                "sim.so" if optional_paths else "",
                "warmup.o" if optional_paths else "",
                True,
                1234,
                True,
                4,
            )
        ]


class TestPublicWorkerBorrowed:
    @pytest.fixture
    def owner(self, monkeypatch, kernel_symbol_runtime, request):
        from simpler_setup import runtime_builder  # noqa: PLC0415

        options = getattr(request, "param", {})
        if isinstance(options, int):
            options = {"finalize_failures": options}
        budget = options.get("budget", 0)
        runtime = kernel_symbol_runtime(
            borrowed=True, finalize_failures=options.get("finalize_failures", 0), workspace=bool(budget)
        )
        observer = ctypes.CDLL(str(runtime))
        bins = SimpleNamespace(
            host_path=runtime,
            aicpu_path=os.devnull,
            aicore_path=os.devnull,
            sdma_warmup_path=Path("/unread/sdma_warmup_kernel.o"),
        )
        monkeypatch.setattr(runtime_builder, "RuntimeBuilder", lambda *_: SimpleNamespace(get_binaries=lambda _: bins))
        worker = Worker(
            level=2,
            device_id=0,
            platform="a2a3",
            runtime="tensormap_and_ringbuffer",
            **({"workspace_budget_bytes": budget} if budget else {}),
        )
        try:
            target = ChipCallable.build(
                signature=[ArgDirection.IN], func_name="public_borrowed", binary=b"x", children=[]
            )
            handle = worker.register(target)
            worker.init(borrowed_context_generation=77)
            yield worker, handle, observer
        finally:
            observer.test_validation(0)
            observer.test_caller_validation(0)
            worker.close()

    def test_init_and_source_identity_use_the_borrowed_environment(self, owner):
        worker, _, observer = owner
        environment = worker._chip_worker._execution_environment
        assert environment[2:] == (77, 0, True)
        assert observer.test_borrowed_inits() >= 1
        source = Buffer.wrap(address=0x8000, nbytes=64, location=worker.device_location)
        assert source.to_descriptor().identity == source.identity
        source.close()
        native = worker._chip_worker
        worker.close()
        assert native._execution_environment is None
        assert worker._chip_worker is None

    @pytest.mark.parametrize("owner", [1], indirect=True)
    def test_failed_close_keeps_borrowed_binding_until_retry(self, owner):
        worker, _, observer = owner
        native = worker._chip_worker
        environment = native._execution_environment
        with pytest.raises(RuntimeError, match="teardown"):
            worker.close()
        assert worker._lifecycle is _Lifecycle.CLOSED
        assert native.initialized and native._execution_environment == environment
        assert observer.test_destroys() == 0
        worker.close()
        assert not native.initialized and native._execution_environment is None
        assert worker._cleanup_journal.empty
        assert observer.test_destroys() == 1

    @pytest.mark.parametrize("owner", [{"budget": 4096}], indirect=True)
    def test_workspace_budget_reaches_borrowed_native_init(self, owner):
        worker, _, _ = owner
        status, report = worker._chip_worker.workspace_report()
        assert status == "available"
        assert report["budget_enforced"] == 1 and report["limit_bytes"] == 4096

    def test_sdma_is_rejected_before_startup(self):
        worker = Worker(level=2, platform="a2a3", enable_sdma=True)
        with pytest.raises(ValueError, match="does not support SDMA"):
            worker.init(borrowed_context_generation=77)
        assert worker._lifecycle is _Lifecycle.NEW and worker._chip_worker is None
        worker.close()

    @pytest.mark.parametrize("stream", [None, 0, -1, True, 0.5, 1 << 64])
    def test_missing_or_invalid_stream_does_not_publish_a_run(self, owner, stream):
        worker, handle, _ = owner
        source = Buffer.wrap(address=0x8000, nbytes=64, location=worker.device_location)
        args = TaskArgs()
        args.add_tensor(Tensor(source, shapes=(4,), dtype=DataType.FLOAT32))
        try:
            with pytest.raises((TypeError, ValueError)):
                worker.submit(handle, args, caller_stream=stream)
            assert worker._chip_run_seq == 0
            assert not worker._chip_run_touched_identities and not worker._chip_source_uses
            assert not worker._source_context_failed and worker._chip_import_registry is None
        finally:
            source.close()

    def test_caller_refusal_precedes_mapping_and_source_reservation(self, owner):
        worker, handle, observer = owner
        source = Buffer.wrap(address=0x8000, nbytes=64, location=worker.device_location)
        args = TaskArgs()
        args.add_tensor(Tensor(source, shapes=(4,), dtype=DataType.FLOAT32))
        try:
            observer.test_caller_validation(-1004)
            with pytest.raises(RuntimeError, match="caller-stream validation"):
                worker.submit(handle, args, caller_stream=0x1234)
            assert worker._lifecycle is _Lifecycle.READY and worker._chip_run_seq == 0
            assert not worker._chip_source_uses and not worker._source_attachments
            assert not worker._source_context_failed and worker._chip_import_registry is None
        finally:
            observer.test_caller_validation(0)
            source.close()

    def test_h2d_request_is_refused_before_import(self, owner):
        from simpler.buffer import wrap_fork_inherited  # noqa: PLC0415

        worker, handle, _ = owner
        host = wrap_fork_inherited(0x1000, 64, worker._owner_instance_id, 55)
        args = TaskArgs()
        args.add_tensor(Tensor(host, shapes=(4,), dtype=DataType.FLOAT32))
        with pytest.raises(ValueError, match="does not copy HOST"):
            worker.submit(handle, args, caller_stream=0x1234)
        assert not worker._source_context_failed and worker._chip_import_registry is None
        assert worker._chip_run_seq == 0

    @pytest.mark.parametrize("generation", [0, -1, True, 1.5, 1 << 64])
    def test_invalid_generation_is_rejected_before_startup(self, generation):
        worker = Worker(level=2, platform="a2a3")
        with pytest.raises((TypeError, ValueError)):
            worker.init(borrowed_context_generation=generation)
        assert worker._lifecycle is _Lifecycle.NEW and worker._chip_worker is None
        worker.close()

    @pytest.mark.parametrize("options", [{"level": 3}, {"level": 2, "platform": "a2a3sim"}])
    def test_borrowed_context_does_not_cross_a_hierarchy_or_simulator(self, options):
        worker = Worker(**options)
        with pytest.raises(ValueError, match="direct L2|onboard"):
            worker.init(borrowed_context_generation=77)
        assert worker._lifecycle is _Lifecycle.NEW
        worker.close()

    def test_hierarchical_submit_refuses_a_process_local_stream(self):
        worker = Worker(level=3)
        with pytest.raises(ValueError, match="direct L2"):
            worker.submit(lambda *_: None, caller_stream=0x1234)
        assert worker._lifecycle is _Lifecycle.NEW
        worker.close()


class TestChipWorkerDeviceOwner:
    @pytest.mark.parametrize("first_borrowed", [False, True])
    @pytest.mark.parametrize("second_borrowed", [False, True])
    @pytest.mark.parametrize("different_runtime", [False, True])
    def test_same_device_refused_before_context_creation(
        self, kernel_symbol_runtime, first_borrowed, second_borrowed, different_runtime
    ):
        runtime = kernel_symbol_runtime(borrowed=True)
        other_runtime = kernel_symbol_runtime(borrowed=True, supported=1) if different_runtime else runtime
        observer = ctypes.CDLL(str(other_runtime))
        first, second, other_device = _ChipWorker(), _ChipWorker(), _ChipWorker()
        first_options = {"borrowed_context_generation": 77} if first_borrowed else {}
        second_options = {"borrowed_context_generation": 88} if second_borrowed else {}
        try:
            first.init(str(runtime), os.devnull, os.devnull, "", 0, **first_options)
            environment = first._execution_environment
            before = observer.test_creates(), observer.test_program_inits(), observer.test_borrowed_inits()
            with pytest.raises(RuntimeError, match="device 0 already has an execution Worker"):
                second.init(str(other_runtime), os.devnull, os.devnull, "", 0, **second_options)
            assert before == (observer.test_creates(), observer.test_program_inits(), observer.test_borrowed_inits())
            assert first._execution_environment == environment
            assert second._execution_environment is None
            with pytest.raises(RuntimeError, match="already initialized"):
                first.init(str(runtime), os.devnull, os.devnull, "", 0, **first_options)
            other_device.init(str(other_runtime), os.devnull, os.devnull, "", 1, **second_options)
            first.finalize()
            second.init(str(other_runtime), os.devnull, os.devnull, "", 0, **second_options)
            successor_environment = second._execution_environment
            first.finalize()
            contender = _ChipWorker()
            with pytest.raises(RuntimeError, match="device 0 already has an execution Worker"):
                contender.init(str(runtime), os.devnull, os.devnull, "", 0, **first_options)
            assert second._execution_environment == successor_environment
        finally:
            first.finalize()
            second.finalize()
            other_device.finalize()

    @pytest.mark.parametrize("borrowed", [False, True])
    def test_concurrent_init_reserves_device_before_bootstrap(self, kernel_symbol_runtime, borrowed):
        runtime = kernel_symbol_runtime(borrowed=True, init_gate=True)
        observer = ctypes.CDLL(str(runtime))
        first, second = _ChipWorker(), _ChipWorker()
        options = {"borrowed_context_generation": 77} if borrowed else {}
        errors = []

        def initialize():
            try:
                first.init(str(runtime), os.devnull, os.devnull, "", 0, **options)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=initialize)
        thread.start()
        try:
            assert observer.test_wait_init() == 1
            before = observer.test_creates()
            with pytest.raises(RuntimeError, match="device 0 already has an execution Worker"):
                second.init(str(runtime), os.devnull, os.devnull, "", 0, **options)
            assert observer.test_creates() == before
        finally:
            observer.test_release_init()
            thread.join(10)
            assert not thread.is_alive()
            first.finalize()
            second.finalize()
        assert not errors

    @pytest.mark.parametrize("borrowed", [False, True])
    def test_failed_close_keeps_reservation_until_retry(self, kernel_symbol_runtime, borrowed):
        runtime = kernel_symbol_runtime(borrowed=True, finalize_failures=1)
        healthy = kernel_symbol_runtime(borrowed=True)
        first, successor = _ChipWorker(), _ChipWorker()
        options = {"borrowed_context_generation": 77} if borrowed else {}
        try:
            first.init(str(runtime), os.devnull, os.devnull, "", 0, **options)
            with pytest.raises(RuntimeError, match="device teardown failed"):
                first.finalize()
            with pytest.raises(RuntimeError, match="device 0 already has an execution Worker"):
                successor.init(str(healthy), os.devnull, os.devnull, "", 0, **options)
            first.finalize()
            successor.init(str(healthy), os.devnull, os.devnull, "", 0, **options)
        finally:
            first.finalize()
            successor.finalize()

    @pytest.mark.parametrize("borrowed", [False, True])
    @pytest.mark.parametrize("failed_cleanup", [False, True])
    @pytest.mark.parametrize("detail", [None, "ACL setter failed; check SIMPLER_OP_EXECUTE_TIMEOUT_US"])
    def test_init_rollback_releases_only_after_proven_cleanup(
        self, kernel_symbol_runtime, borrowed, failed_cleanup, detail
    ):
        failing = kernel_symbol_runtime(
            borrowed=True, init_result=-66, finalize_failures=int(failed_cleanup), init_error=detail
        )
        healthy = kernel_symbol_runtime(borrowed=True)
        observer = ctypes.CDLL(str(failing))
        first, successor = _ChipWorker(), _ChipWorker()
        options = {"borrowed_context_generation": 77} if borrowed else {}
        try:
            before = observer.test_closes(), observer.test_destroys()
            with pytest.raises(
                RuntimeError, match="rollback failed" if failed_cleanup else "failed with code -66"
            ) as error:
                first.init(str(failing), os.devnull, os.devnull, "", 0, **options)
            if detail is not None:
                assert detail in str(error.value)
            assert observer.test_closes() == before[0] + 1
            if failed_cleanup:
                assert observer.test_destroys() == before[1]
                assert first._execution_environment is not None
                with pytest.raises(RuntimeError, match="device 0 already has an execution Worker"):
                    successor.init(str(healthy), os.devnull, os.devnull, "", 0, **options)
                first.finalize()
            else:
                assert observer.test_destroys() == before[1] + 1
                assert first._execution_environment is None
            successor.init(str(healthy), os.devnull, os.devnull, "", 0, **options)
        finally:
            first.finalize()
            successor.finalize()

    def test_simulator_classification_keeps_virtual_device_admission(self, kernel_symbol_runtime):
        runtime = kernel_symbol_runtime()
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent("""\
                    import os
                    import sys
                    from _task_interface import _ChipWorker

                    runtime = sys.argv[1]
                    first, second = _ChipWorker(), _ChipWorker()
                    try:
                        first.init(runtime, os.devnull, os.devnull, "", 0, sim_context_path=runtime)
                        second.init(runtime, os.devnull, os.devnull, "", 0, sim_context_path=runtime)
                        assert first._execution_environment != second._execution_environment
                    finally:
                        first.finalize()
                        second.finalize()
                    """),
                str(runtime),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.parametrize("failure_stage", ["none", "native", "grant"])
    @pytest.mark.parametrize("borrowed", [False, True])
    def test_logical_terminal_close_keeps_unproven_device_reserved(
        self, kernel_symbol_runtime, failure_stage, borrowed
    ):
        runtime = kernel_symbol_runtime(
            borrowed=True,
            init_result=-66 if failure_stage == "native" else 0,
            early_failure="grant_cleanup" if failure_stage == "grant" else None,
            unproven_cleanup=True,
            init_error="init diagnostic before unproven cleanup",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent("""\
                    import os
                    import sys
                    from _task_interface import _ChipWorker

                    runtime, failure_stage, borrowed = sys.argv[1], sys.argv[2], int(sys.argv[3])
                    options = {"borrowed_context_generation": 77} if borrowed else {}
                    first, successor = _ChipWorker(), _ChipWorker()
                    if failure_stage != "none":
                        try:
                            first.init(runtime, os.devnull, os.devnull, "", 0, **options)
                        except RuntimeError as error:
                            assert "rollback" in str(error)
                            expected = "refused a grant" if failure_stage == "grant" else "failed with code -66"
                            assert expected in str(error)
                            if failure_stage == "native":
                                assert "init diagnostic before unproven cleanup" in str(error)
                            assert first._execution_environment is not None
                        else:
                            raise AssertionError("failed init unexpectedly succeeded")
                    else:
                        first.init(runtime, os.devnull, os.devnull, "", 0, **options)
                    first.finalize()
                    assert first._execution_environment is None
                    first.finalize()
                    del first
                    try:
                        successor.init(runtime, os.devnull, os.devnull, "", 0, **options)
                    except RuntimeError as error:
                        assert "device 0 already has an execution Worker" in str(error)
                    else:
                        raise AssertionError("logical terminal close released an unproven reservation")
                    successor.finalize()
                    """),
                str(runtime),
                failure_stage,
                str(int(borrowed)),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


class TestChipWorkerExecutionEnvironment:
    @pytest.mark.parametrize("borrowed", [False, True])
    def test_environment_is_unique_and_survives_refusal(self, kernel_symbol_runtime, borrowed):
        runtime = kernel_symbol_runtime(borrowed=borrowed)
        observer = ctypes.CDLL(str(runtime))
        first, second = _ChipWorker(), _ChipWorker()
        assert first._execution_environment is None
        options = {"borrowed_context_generation": 77} if borrowed else {}
        try:
            first.init(str(runtime), os.devnull, os.devnull, "", 0, **options)
            second.init(str(runtime), os.devnull, os.devnull, "", 1, **options)
            environment = first._execution_environment
            pid, incarnation, generation, device_id, is_borrowed = environment
            assert pid == os.getpid() and incarnation > 0 and generation > 0
            assert device_id == 0 and is_borrowed is borrowed
            if borrowed:
                assert generation == 77
            assert environment != second._execution_environment
            if borrowed:
                observer.test_validation(-88)
                with pytest.raises(ExecutionContextRefused):
                    first._validate_execution_context()
                assert first._execution_environment == environment
                observer.test_validation(0)
            first.finalize()
            assert first._execution_environment is None
            assert second._execution_environment is not None
        finally:
            observer.test_validation(0)
            first.finalize()
            second.finalize()

    @pytest.mark.parametrize("borrowed", [False, True])
    @pytest.mark.parametrize(
        ("failure", "message"),
        [
            ("grant", "refused a grant .* code -89"),
            ("size", "unsupported native-run storage size/alignment"),
            ("alignment", "unsupported native-run storage size/alignment"),
        ],
    )
    def test_early_init_rollback_retires_environment_and_allows_retry(
        self, kernel_symbol_runtime, borrowed, failure, message
    ):
        failing = kernel_symbol_runtime(borrowed=borrowed, early_failure=failure)
        observer = ctypes.CDLL(str(failing))
        worker = _ChipWorker()
        options = {"borrowed_context_generation": 77} if borrowed else {}
        before = observer.test_destroys(), observer.test_closes(), observer.test_borrowed_inits()
        try:
            with pytest.raises(RuntimeError, match=message):
                worker.init(str(failing), os.devnull, os.devnull, "", 0, **options)
            assert observer.test_destroys() == before[0] + 1
            assert observer.test_closes() == before[1]
            assert observer.test_borrowed_inits() == before[2]
            assert not worker.initialized and worker.runtime_slot_count == 0
            assert worker.device_id == -1
            assert worker._execution_environment is None
            healthy = kernel_symbol_runtime(borrowed=borrowed)
            successor = _ChipWorker()
            successor.init(str(healthy), os.devnull, os.devnull, "", 0, **options)
            successor.finalize()
            worker.init(str(healthy), os.devnull, os.devnull, "", 0, **options)
            assert worker.initialized and worker.runtime_slot_count == 1
            environment = worker._execution_environment
            assert environment is not None and environment[0] == os.getpid()
            assert environment[2] > 0
            if borrowed:
                assert environment[2] == 77
        finally:
            worker.finalize()
        assert worker._execution_environment is None

    @pytest.mark.parametrize("borrowed", [False, True])
    def test_forked_native_owner_refuses_before_runtime_entry(self, kernel_symbol_runtime, borrowed):
        runtime = kernel_symbol_runtime(borrowed=borrowed, memory=True)
        worker = _ChipWorker()
        options = {"borrowed_context_generation": 77} if borrowed else {}
        worker.init(str(runtime), os.devnull, os.devnull, "", 0, **options)
        try:
            pid = os.fork()
            if pid == 0:
                signal.alarm(3)
                try:
                    for operation in (worker._validate_execution_context, lambda: worker.malloc(4), worker.finalize):
                        try:
                            operation()
                        except ExecutionContextRefused:
                            continue
                        os._exit(2)
                    child_owner = _ChipWorker()
                    child_owner.init(str(runtime), os.devnull, os.devnull, "", 0, **options)
                    assert child_owner._execution_environment[0] == os.getpid()
                    child_owner.finalize()
                except BaseException:
                    os._exit(3)
                os._exit(0)
            _, status = os.waitpid(pid, 0)
            assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, status
            worker._validate_execution_context()
            contender = _ChipWorker()
            with pytest.raises(RuntimeError, match="device 0 already has an execution Worker"):
                contender.init(str(runtime), os.devnull, os.devnull, "", 0, **options)
        finally:
            worker.finalize()

    @pytest.mark.parametrize("borrowed", [False, True])
    def test_source_expires_after_native_finalize_while_facade_stays_ready(self, kernel_symbol_runtime, borrowed):
        runtime = kernel_symbol_runtime(borrowed=borrowed)
        native = _ChipWorker()
        options = {"borrowed_context_generation": 77} if borrowed else {}
        native.init(str(runtime), os.devnull, os.devnull, "", 0, **options)
        owner = Worker(level=2)
        owner._chip_worker = native
        owner._lifecycle = _Lifecycle.READY
        try:
            location = owner.device_location
            source = Buffer.wrap(address=0x8000, nbytes=64, location=location)
            native.finalize()
            assert owner._lifecycle is _Lifecycle.READY
            with pytest.raises(ValueError, match="live device context"):
                Buffer.wrap(address=0x9000, nbytes=64, location=location)
            with pytest.raises(ValueError, match="live device context"):
                source.to_descriptor()
            source.close()
        finally:
            native.finalize()
            owner._lifecycle = _Lifecycle.CLOSED


class TestChipWorkerBorrowedInit:
    def test_context_preflight_does_not_consume_the_binding(self, kernel_symbol_runtime):
        runtime = kernel_symbol_runtime(borrowed=True)
        observer = ctypes.CDLL(str(runtime))
        worker = _ChipWorker()
        worker.init(str(runtime), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        try:
            before = observer.test_closes(), observer.test_destroys()
            observer.test_validation(-88)
            with pytest.raises(ExecutionContextRefused, match="context validation failed with code -88") as error:
                worker._validate_execution_context()
            assert isinstance(error.value, RuntimeError)
            assert worker.initialized and worker.runtime_slot_count == 1
            assert (observer.test_closes(), observer.test_destroys()) == before
            observer.test_validation(0)
            worker._validate_execution_context()
        finally:
            observer.test_validation(0)
            worker.finalize()

    @pytest.mark.parametrize("operation", ["wait", "result"])
    def test_native_wait_refusal_keeps_python_run_retryable(self, kernel_symbol_runtime, operation):
        runtime = kernel_symbol_runtime(borrowed=True)
        observer = ctypes.CDLL(str(runtime))
        native = _ChipWorker()
        native.init(str(runtime), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        owner = Worker(level=2)
        fences = []

        def wait(timeout):
            native._validate_execution_context()
            fences.append(timeout)
            return True

        owner._chip_worker = SimpleNamespace(
            _execution_environment=native._execution_environment,
            _validate_execution_context=native._validate_execution_context,
            _validate_caller_stream=native._validate_caller_stream,
            _submit_chip_run_direct=lambda *_, **kwargs: SimpleNamespace(wait=wait),
        )
        owner._lifecycle = _Lifecycle.READY
        try:
            source = Buffer.wrap(address=0x8000, nbytes=64, location=owner.device_location)
            args = TaskArgs()
            args.add_tensor(Tensor(source, shapes=(4,), dtype=DataType.FLOAT32), TensorArgType.INPUT)
            handle = owner._submit_l2_locked(3, args, CallConfig(), caller_stream=0x1234)
            run_id, keepalive = handle._run_id, handle._keepalive
            observer.test_validation(-88)
            for _ in range(2):
                with pytest.raises(ExecutionContextRefused, match="context validation failed with code -88"):
                    getattr(handle, operation)()
                assert not handle._terminal and not handle._wait_in_progress
                assert handle._run_id == run_id and handle._keepalive is keepalive
                assert run_id in owner._chip_runs and run_id in owner._chip_source_uses
                assert not owner._source_context_failed and not fences
                with pytest.raises(RuntimeError, match="in-flight or unproven"):
                    source.close()
            observer.test_validation(0)
            getattr(handle, operation)()
            assert handle.done and fences == [-1.0]
            assert not owner._chip_runs and not owner._chip_source_uses
            source.close()
            assert source.closed
        finally:
            owner._chip_runs.clear()
            owner._chip_run_touched_identities.clear()
            owner._close_chip_import_registry()
            owner._lifecycle = _Lifecycle.CLOSED
            observer.test_validation(0)
            native.finalize()

    @pytest.mark.parametrize("operation", ["malloc", "free", "copy_to", "copy_from", "device_memory_info", "flush"])
    def test_native_memory_refusal_precedes_runtime_entry(self, kernel_symbol_runtime, operation):
        runtime = kernel_symbol_runtime(borrowed=True, memory=True)
        observer = ctypes.CDLL(str(runtime))
        native = _ChipWorker()
        native.init(str(runtime), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        backing = native.malloc(16)
        host = ctypes.create_string_buffer(b"0123456789abcdef", 16)
        operations = {
            "malloc": lambda: native.malloc(16),
            "free": lambda: native.free(backing),
            "copy_to": lambda: native.copy_to(backing, ctypes.addressof(host), 16),
            "copy_from": lambda: native.copy_from(ctypes.addressof(host), backing, 16),
            "device_memory_info": native.device_memory_info,
            "flush": lambda: native.flush_diagnostics(0),
        }
        try:
            before = observer.test_memory_calls()
            observer.test_validation(-88)
            with pytest.raises(ExecutionContextRefused):
                operations[operation]()
            assert observer.test_memory_calls() == before
            observer.test_validation(0)
            operations[operation]()
            assert observer.test_memory_calls() == before + 1
        finally:
            observer.test_validation(0)
            native.finalize()

    def test_operation_lease_protects_unlisted_api_and_unwinds_refusal(self, kernel_symbol_runtime):
        runtime = kernel_symbol_runtime(borrowed=True)
        observer = ctypes.CDLL(str(runtime))
        native = _ChipWorker()
        native.init(str(runtime), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        owner = Worker(level=2)
        owner._chip_worker = native
        owner._lifecycle = _Lifecycle.READY
        try:
            observer.test_validation(-88)
            with pytest.raises(ExecutionContextRefused), owner._operation_lease("new_device_api"):
                pytest.fail("refused operation reached its body")
            assert owner._active_ops == 0 and not owner._lease_depth
            assert owner._lifecycle is _Lifecycle.READY
            # A source descriptor query owns no device-side operation.
            location = owner.device_location
            assert owner.device_location._source is location._source
            observer.test_validation(0)
            with owner._operation_lease("new_device_api"):
                assert owner._active_ops == 1
            assert owner._active_ops == 0 and not owner._lease_depth
        finally:
            owner._chip_worker = None
            owner._lifecycle = _Lifecycle.CLOSED
            observer.test_validation(0)
            native.finalize()

    @pytest.mark.parametrize("operation", ["malloc", "free", "copy_to", "copy_from", "create_buffer"])
    def test_public_memory_refusal_precedes_body(self, kernel_symbol_runtime, operation):
        runtime = kernel_symbol_runtime(borrowed=True, memory=True)
        observer = ctypes.CDLL(str(runtime))
        chip = ChipWorker()
        chip._impl.init(str(runtime), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        owner = Worker(level=2)
        owner._chip_worker = chip._impl
        owner._lifecycle = _Lifecycle.READY
        backing = owner.malloc(16)
        host = bytearray(b"0123456789abcdef")
        operations = {
            "malloc": lambda: owner.malloc(16),
            "free": lambda: owner.free(backing),
            "copy_to": lambda: owner.copy_to(backing, host),
            "copy_from": lambda: owner.copy_from(host, backing),
            "create_buffer": lambda: owner.create_buffer(16),
        }
        try:
            before = observer.test_memory_calls()
            registered = owner._child_alloc.get(backing.identity)
            observer.test_validation(-88)
            with pytest.raises(ExecutionContextRefused):
                operations[operation]()
            assert observer.test_memory_calls() == before
            assert owner._child_alloc.get(backing.identity) is registered
            assert len(owner._child_alloc) == 1 and not owner._buffers
            assert owner._active_ops == 0 and not owner._lease_depth
            observer.test_validation(0)
            result = operations[operation]()
            if operation in ("malloc", "create_buffer"):
                assert isinstance(result, Buffer)
                if operation == "malloc":
                    owner.free(result)
                else:
                    owner.release_buffer(result)
            if operation != "free":
                owner.free(backing)
            assert not owner._child_alloc
        finally:
            observer.test_validation(0)
            owner.close()
            chip.finalize()

    def test_operation_preflight_releases_lifecycle_lock(self):
        owner = Worker(level=2)
        started = threading.Event()
        acquired = threading.Event()
        errors = []

        def contender():
            if not started.wait(5):
                errors.append("validator was not entered")
                return
            with owner._hierarchical_start_cv:
                acquired.set()

        def validate():
            started.set()
            assert acquired.wait(5), "native validation holds the lifecycle lock"

        owner._chip_worker = SimpleNamespace(_validate_execution_context=validate)
        owner._lifecycle = _Lifecycle.READY
        thread = threading.Thread(target=contender)
        thread.start()
        try:
            with owner._operation_lease("new_device_api"):
                assert owner._active_ops == 1
            assert not errors and owner._active_ops == 0
        finally:
            started.set()
            thread.join(5)
            owner._chip_worker = None
            owner._lifecycle = _Lifecycle.CLOSED
        assert not thread.is_alive()

    @pytest.mark.parametrize("operation", ["register", "register_blob", "unregister"])
    def test_native_registration_refuses_context_before_runtime_entry(self, kernel_symbol_runtime, operation):
        runtime = kernel_symbol_runtime(borrowed=True)
        observer = ctypes.CDLL(str(runtime))
        native = _ChipWorker()
        native.init(str(runtime), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        target = ChipCallable.build(signature=[], func_name="control", binary=b"\0", children=[])
        try:
            if operation == "register":

                def action():
                    native.register_callable(0, target)

                calls = observer.test_registrations
            elif operation == "register_blob":

                def action():
                    native.register_callable_from_blob(0, target.buffer_ptr())

                calls = observer.test_registrations
            else:
                native.register_callable(0, target)

                def action():
                    native.unregister_callable(0)

                calls = observer.test_unregistrations
            before = calls()
            observer.test_validation(-88)
            with pytest.raises(ExecutionContextRefused):
                action()
            assert calls() == before and native.initialized
            observer.test_validation(0)
            action()
            assert calls() == before + 1
        finally:
            observer.test_validation(0)
            native.finalize()

    @pytest.mark.parametrize("operation", ["submit", "register", "unregister", "unregister_last", "close"])
    def test_public_worker_refusal_precedes_state_commit(self, kernel_symbol_runtime, operation):
        runtime = kernel_symbol_runtime(borrowed=True)
        observer = ctypes.CDLL(str(runtime))
        native = _ChipWorker()
        native.init(str(runtime), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        admissions = []

        class ControlledSubmission:
            def __getattr__(self, name):
                return getattr(native, name)

            def _submit_chip_run_direct(self, *args, **kwargs):
                native._validate_execution_context()
                admissions.append(args)
                return SimpleNamespace(wait=lambda _timeout: True, done=lambda: True)

        chip = ChipWorker()
        chip._impl = ControlledSubmission()
        owner = Worker(level=2)
        target = ChipCallable.build(signature=[], func_name="control", binary=b"\0", children=[])
        handle = owner.register(target)
        alias = None if operation == "unregister_last" else owner.register(target)
        references = 1 if alias is None else 2
        owner._chip_worker = chip._impl
        owner._lifecycle = _Lifecycle.READY
        chip._register_callable_at_slot(owner._resolve_handle(handle).slot_id, target)
        source = Buffer.wrap(address=0x8000, nbytes=64, location=owner.device_location)
        args = TaskArgs()
        args.add_tensor(Tensor(source, shapes=(4,), dtype=DataType.FLOAT32), TensorArgType.INPUT)
        operations = {
            "submit": lambda: owner.submit(handle, args),
            "register": lambda: owner.register(target),
            "unregister": lambda: owner.unregister(handle),
            "unregister_last": lambda: owner.unregister(handle),
            "close": owner.close,
        }
        before = observer.test_registrations(), observer.test_unregistrations(), observer.test_closes()
        try:
            observer.test_validation(-88)
            for _ in range(2):
                with pytest.raises(ExecutionContextRefused):
                    operations[operation]()
                assert owner._lifecycle is _Lifecycle.READY and owner._close_completion is None
                assert owner._resolve_handle(handle).ref_count == references
                if alias is not None:
                    assert owner._resolve_handle(alias).ref_count == references
                assert not owner._source_context_failed and not admissions
                assert not owner._chip_runs and not owner._chip_source_uses
                assert not owner._chip_run_touched_identities and owner._chip_run_seq == 0
                assert owner._active_ops == 0 and not owner._lease_depth
                assert native.initialized
                assert before == (
                    observer.test_registrations(),
                    observer.test_unregistrations(),
                    observer.test_closes(),
                )
            observer.test_validation(0)
            if operation == "submit":
                owner.submit(handle, args, caller_stream=0x1234).wait()
                assert len(admissions) == 1 and not owner._chip_source_uses
            elif operation == "register":
                registered = owner.register(target)
                assert owner._resolve_handle(registered).ref_count == 3
            elif operation in ("unregister", "unregister_last"):
                owner.unregister(handle)
                if alias is None:
                    assert not owner._identity_registry and not owner._live_handles
                else:
                    assert owner._resolve_handle(alias).ref_count == 1
            else:
                owner.close()
                assert owner._lifecycle is _Lifecycle.CLOSED and not native.initialized
            source.close()
            assert source.closed
        finally:
            observer.test_validation(0)
            owner.close()
            native.finalize()

    def test_accepted_close_failure_keeps_debt_across_context_refusal(self, kernel_symbol_runtime):
        runtime = kernel_symbol_runtime(borrowed=True, finalize_failures=1)
        observer = ctypes.CDLL(str(runtime))
        chip = ChipWorker()
        chip._impl.init(str(runtime), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        owner = Worker(level=2)
        owner._chip_worker = chip._impl
        owner._lifecycle = _Lifecycle.READY
        try:
            with pytest.raises(RuntimeError, match="teardown"):
                owner.close()
            assert owner._lifecycle is _Lifecycle.CLOSED
            assert chip.initialized and not owner._cleanup_journal.empty
            attempt = owner._close_completion
            assert attempt is not None and attempt.done and attempt.error is not None
            before = observer.test_closes(), observer.test_destroys()
            observer.test_validation(-88)
            with pytest.raises(ExecutionContextRefused):
                owner.close()
            assert owner._lifecycle is _Lifecycle.CLOSED and owner._close_completion is attempt
            assert chip.initialized and not owner._cleanup_journal.empty
            assert (observer.test_closes(), observer.test_destroys()) == before
            observer.test_validation(0)
            owner.close()
            assert not chip.initialized and owner._cleanup_journal.empty
            assert owner._lifecycle is _Lifecycle.CLOSED
            with pytest.raises(RuntimeError, match="requires an initialized"):
                owner.submit(None)
        finally:
            observer.test_validation(0)
            owner.close()
            chip.finalize()

    @pytest.mark.parametrize("references", [1, 2])
    @pytest.mark.parametrize("operation", ["register", "unregister"])
    def test_legacy_registration_refusal_keeps_live_handles(self, kernel_symbol_runtime, operation, references):
        runtime = kernel_symbol_runtime(borrowed=True)
        observer = ctypes.CDLL(str(runtime))
        chip = ChipWorker()
        chip._impl.init(str(runtime), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        target = ChipCallable.build(signature=[], func_name="control", binary=b"\0", children=[])
        handle = chip.register_callable(target)
        alias = chip.register_callable(target) if references == 2 else None
        try:
            observer.test_validation(-88)

            def action():
                if operation == "register":
                    chip.register_callable(target)
                else:
                    chip.unregister_callable(handle)

            for _ in range(2):
                with pytest.raises(ExecutionContextRefused):
                    action()
                assert chip._resolve_handle(handle).ref_count == references
                if alias is not None:
                    assert chip._resolve_handle(alias).ref_count == references
            observer.test_validation(0)
            if operation == "register":
                registered = chip.register_callable(target)
                assert chip._resolve_handle(registered).ref_count == references + 1
            else:
                chip.unregister_callable(handle)
                if alias is None:
                    assert not chip._identity_registry and not chip._live_handles
                else:
                    assert chip._resolve_handle(alias).ref_count == 1
        finally:
            observer.test_validation(0)
            chip.finalize()

    @pytest.mark.parametrize("borrowed", [False, True])
    def test_mode_cannot_change_during_or_after_worker_lifetime(self, kernel_symbol_runtime, borrowed):
        runtime = kernel_symbol_runtime(borrowed=True)
        observer = ctypes.CDLL(str(runtime))
        worker = _ChipWorker()
        options = {"borrowed_context_generation": 77} if borrowed else {}
        incompatible = {} if borrowed else {"borrowed_context_generation": 77}
        worker.init(str(runtime), os.devnull, os.devnull, "", 0, **options)
        environment = worker._execution_environment
        before = observer.test_borrowed_inits(), observer.test_closes(), observer.test_destroys()
        try:
            with pytest.raises(RuntimeError, match="already initialized"):
                worker.init(str(runtime), os.devnull, os.devnull, "", 0, **incompatible)
            assert worker._execution_environment == environment
            assert (observer.test_borrowed_inits(), observer.test_closes(), observer.test_destroys()) == before
        finally:
            worker.finalize()
        after = observer.test_borrowed_inits(), observer.test_closes(), observer.test_destroys()
        for init_options in (options, incompatible):
            with pytest.raises(RuntimeError, match="already finalized"):
                worker.init(str(runtime), os.devnull, os.devnull, "", 0, **init_options)
        assert worker._execution_environment is None
        assert (observer.test_borrowed_inits(), observer.test_closes(), observer.test_destroys()) == after

    @pytest.mark.parametrize("borrowed", [False, True])
    def test_auto_register_refuses_older_runtime_without_changing_binding(self, kernel_symbol_runtime, borrowed):
        runtime = kernel_symbol_runtime(borrowed=True)
        worker = _ChipWorker()
        options = {"borrowed_context_generation": 77} if borrowed else {}
        worker.init(str(runtime), os.devnull, os.devnull, "", 0, **options)
        environment = worker._execution_environment
        try:
            callable_image = ChipCallable.build(signature=[], func_name="test", binary=b"\x00", children=[])
            with pytest.raises(RuntimeError, match="does not support versioned"):
                worker.register_callable(callable_image)
            worker.register_callable(0, callable_image)
            assert worker.initialized
            assert worker._execution_environment == environment
        finally:
            worker.finalize()

    def test_borrowed_init_uses_shared_binding_and_can_close(self, kernel_symbol_runtime):
        runtime = kernel_symbol_runtime(borrowed=True)
        observer = ctypes.CDLL(str(runtime))
        worker = _ChipWorker()
        worker.init(str(runtime), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        assert worker.initialized
        assert worker._execution_environment is not None
        assert worker.runtime_slot_count == 1
        assert observer.test_borrowed_inits() == 1
        worker.finalize()
        assert not worker.initialized
        assert observer.test_closes() == 1
        assert observer.test_destroys() == 1
        assert worker._execution_environment is None

    def test_wrong_context_close_does_not_consume_the_binding(self, kernel_symbol_runtime):
        runtime = kernel_symbol_runtime(borrowed=True)
        observer = ctypes.CDLL(str(runtime))
        worker = _ChipWorker()
        worker.init(str(runtime), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        before_close, before_destroy = observer.test_closes(), observer.test_destroys()
        observer.test_validation(-88)
        try:
            with pytest.raises(RuntimeError, match="context validation failed with code -88"):
                worker.finalize()
            assert worker.initialized
            assert worker.runtime_slot_count == 1
            assert observer.test_closes() == before_close
            assert observer.test_destroys() == before_destroy
        finally:
            observer.test_validation(0)
            worker.finalize()

    def test_failed_borrowed_close_retains_for_retry(self, kernel_symbol_runtime):
        runtime = kernel_symbol_runtime(borrowed=True, finalize_failures=1)
        observer = ctypes.CDLL(str(runtime))
        worker = _ChipWorker()
        worker.init(str(runtime), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        environment = worker._execution_environment
        with pytest.raises(RuntimeError, match="device teardown failed"):
            worker.finalize()
        assert worker.initialized
        assert worker._execution_environment == environment
        assert worker.runtime_slot_count == 1
        assert observer.test_destroys() == 0
        worker.finalize()
        assert observer.test_destroys() == 1
        assert worker._execution_environment is None

    def test_failed_borrowed_init_closes_before_destroy_and_can_retry(self, kernel_symbol_runtime):
        failing = kernel_symbol_runtime(borrowed=True, init_result=-66)
        observer = ctypes.CDLL(str(failing))
        worker = _ChipWorker()
        with pytest.raises(RuntimeError, match="borrowed init failed with code -66"):
            worker.init(str(failing), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        assert not worker.initialized
        assert worker._execution_environment is None
        assert worker.runtime_slot_count == 0
        assert observer.test_closes() == 1
        assert observer.test_destroys() == 1
        complete = kernel_symbol_runtime(borrowed=True)
        worker.init(str(complete), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        worker.finalize()

    def test_unsupported_borrowed_init_does_not_finalize_unclaimed_context(self, kernel_symbol_runtime):
        runtime = kernel_symbol_runtime(borrowed=True, init_result=-1001)
        observer = ctypes.CDLL(str(runtime))
        worker = _ChipWorker()
        with pytest.raises(RuntimeError, match="borrowed init failed with code -1001"):
            worker.init(str(runtime), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        assert not worker.initialized
        assert worker.runtime_slot_count == 0
        assert observer.test_closes() == 0
        assert observer.test_destroys() == 1

    def test_failed_rollback_requires_explicit_finalize(self, kernel_symbol_runtime):
        failing = kernel_symbol_runtime(borrowed=True, init_result=-67, finalize_failures=1)
        observer = ctypes.CDLL(str(failing))
        worker = _ChipWorker()
        with pytest.raises(RuntimeError, match="rollback failed"):
            worker.init(str(failing), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        assert not worker.initialized
        environment = worker._execution_environment
        assert environment is not None and environment[2] == 77
        assert worker.runtime_slot_count == 1
        assert observer.test_destroys() == 0
        with pytest.raises(RuntimeError, match="incomplete rollback"):
            worker.init(str(failing), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        assert worker._execution_environment == environment
        worker.finalize()
        assert worker._execution_environment is None
        assert worker.runtime_slot_count == 0
        assert observer.test_destroys() == 1

    @pytest.mark.parametrize(
        "options",
        [
            {"borrowed_context_generation": 0},
            {"enable_sdma": True},
            {"sim_context_path": "not-a-simulator"},
            {"sdma_warmup_path": "not-a-warmup"},
        ],
    )
    def test_unsupported_policy_is_rejected_before_loading(self, options):
        options = {"borrowed_context_generation": 77, **options}
        worker = _ChipWorker()
        with pytest.raises(ValueError):
            worker.init("not-a-runtime", os.devnull, os.devnull, "", 0, **options)
        assert not worker.initialized
        assert worker.runtime_slot_count == 0

    @pytest.mark.parametrize("budget", [0, 4096])
    @pytest.mark.parametrize("manage", [False, True])
    def test_borrowed_workspace_policy_uses_existing_report(self, kernel_symbol_runtime, budget, manage):
        runtime = kernel_symbol_runtime(borrowed=True, workspace=True)
        worker = _ChipWorker()
        try:
            worker.init(
                str(runtime),
                os.devnull,
                os.devnull,
                "",
                0,
                borrowed_context_generation=77,
                workspace_budget_bytes=budget,
                manage_workspace=manage,
            )
            status, report = worker.workspace_report()
            assert status == "available"
            assert report["budget_enforced"] == bool(budget)
            assert report["limit_bytes"] == budget
        finally:
            worker.finalize()

    def test_older_runtime_is_refused_before_context_creation(self, kernel_symbol_runtime):
        runtime = kernel_symbol_runtime()
        observer = ctypes.CDLL(str(runtime))
        worker = _ChipWorker()
        before_destroy = observer.test_destroys()
        with pytest.raises(RuntimeError, match="requires native caller ordering"):
            worker.init(str(runtime), os.devnull, os.devnull, "", 0, borrowed_context_generation=77)
        assert not worker.initialized
        assert worker.runtime_slot_count == 0
        assert observer.test_destroys() == before_destroy
        worker.init(str(runtime), os.devnull, os.devnull, "", 0)
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

    # Every nonce the chip child can hand a submission. A caster expecting text
    # rejects the three byte strings outright, and `None` is the absent case
    # the binding's own default covers.
    _NONCE_CASES = (
        None,
        b"",
        # urandom output, so the nonce is neither text nor NUL-free.
        b"\x00\xff\x80\x01\xfe\x7f\x00\x2a",
        b"\xc3\x28\x00\x00\xed\xa0\x80\xff",
        # Wrong length states nothing rather than being padded, and is still a
        # legal argument.
        b"\x01\x02\x03",
    )

    @staticmethod
    def _submission_arguments(nonce):
        """One uninitialized worker and the arguments a submission takes.

        Slot 0 is the only slot the default `PipelineContract` admits — its
        `pipeline_depth` is 1 — so a wider slot would be refused by the lease
        check that runs before the lane is ever consulted, and the case would
        prove nothing about the nonce.
        """
        from _task_interface import ChipStorageTaskArgs  # noqa: PLC0415

        provenance = {} if nonce is None else {"owner_instance_id": nonce}
        return _ChipWorker(), ChipStorageTaskArgs(), CallConfig(), provenance

    # The two entry points are asserted in separate cases on purpose: one
    # covering both would stop at the first failure and leave the other route
    # unexercised, which is how a text-only caster survived on the staged side.
    @pytest.mark.parametrize("nonce", _NONCE_CASES)
    def test_the_serial_materialized_entry_takes_an_arbitrary_byte_nonce(self, nonce):
        # Reaching the lane's own guard proves the argument converted; a caster
        # expecting text fails before it with a TypeError, which is what this
        # match distinguishes.
        worker, args, config, provenance = self._submission_arguments(nonce)

        with pytest.raises(RuntimeError, match="run lane is not initialized"):
            worker.run_materialized(
                0, args, config, pipeline_slot=0, pipeline_generation=2, run_id=17, dispatch_id=41, **provenance
            )

    @pytest.mark.parametrize("nonce", _NONCE_CASES)
    def test_the_staged_materialized_entry_takes_an_arbitrary_byte_nonce(self, nonce):
        worker, args, config, provenance = self._submission_arguments(nonce)

        with pytest.raises(RuntimeError, match="run lane is not initialized"):
            worker._submit_chip_run_materialized(0, args, config, 0, 2, 17, 41, 0, 1, False, **provenance)


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

            def _validate_execution_context(self):
                pass

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

    def test_public_wrapper_keeps_registries_when_native_finalize_fails(self):
        from _task_interface import ChipCallable  # noqa: PLC0415
        from simpler.task_interface import ChipWorker  # noqa: PLC0415  # pyright: ignore[reportAttributeAccessIssue]

        class FakeImpl:
            initialized = True
            device_id = 0

            def finalize(self):
                raise RuntimeError("injected device teardown failure")

        worker = ChipWorker()
        worker._impl = FakeImpl()
        worker._callable_registry[0] = ChipCallable.build(signature=[], func_name="test", binary=b"\x00", children=[])
        worker._identity_registry[b"digest"] = object()
        worker._live_handles[1] = b"digest"

        with pytest.raises(RuntimeError, match="injected device teardown failure"):
            worker.finalize()

        # The registries name what the native side still holds. A teardown that
        # did not complete leaves those resources alive, so dropping the
        # registries would hide them from a retry and from the caller.
        assert list(worker._callable_registry) == [0]
        assert list(worker._identity_registry) == [b"digest"]
        assert worker._live_handles == {1: b"digest"}

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
            Worker,
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
        ranked = _read_config_from_mailbox(memoryview(buf), chip_rank=2, capture_index=7)
        assert ranked.output_prefix == "/tmp/out/rank2/d7"

        level_worker = Worker(level=3, num_sub_workers=0)
        level_worker._topology_worker_id = 5
        nested = _read_config_from_mailbox(
            memoryview(buf),
            chip_rank=2,
            capture_index=7,
            level_worker=level_worker,
        )
        assert nested.output_prefix == "/tmp/out/node5/rank2/d7"

    def test_level_directory_needs_both_a_prefix_and_parent_identity(self):
        from simpler.worker import Worker, _level_capture_prefix  # noqa: PLC0415

        unattached = Worker(level=3, num_sub_workers=0)
        attached = Worker(level=3, num_sub_workers=0)
        attached._topology_worker_id = 4

        assert _level_capture_prefix("/tmp/out", unattached) == "/tmp/out"
        assert _level_capture_prefix("", attached) == ""
        assert _level_capture_prefix("/tmp/out", attached) == "/tmp/out/node4"

    def test_rank_directory_covers_every_diagnostic(self):
        # rankN/dN separates one ChipWorker child's artifacts from its siblings',
        # which every diagnostic needs.
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
                *cfg.runtime_env.ring_task_window,
                *cfg.runtime_env.ring_heap,
                *cfg.runtime_env.ring_dep_pool,
                cfg.output_prefix.encode(),
            )
            return _read_config_from_mailbox(memoryview(buf), chip_rank=1, capture_index=0)

        dep_gen_only = decode(enable_dep_gen=True)
        assert dep_gen_only.output_prefix == "/tmp/out/rank1/d0"
        swimlane = decode(enable_chip_swimlane=4)
        assert swimlane.output_prefix == "/tmp/out/rank1/d0"

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
            owner_instance_id=b"\x0f\x1e\x2d\x3c\x4b\x5a\x69\x78",
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
            # The pairing namespace a retained artifact is matched in. Hex,
            # because the nonce is urandom output and JSON carries no bytes.
            "owner_instance_id": "0f1e2d3c4b5a6978",
        }
