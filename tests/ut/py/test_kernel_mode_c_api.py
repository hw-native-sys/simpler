# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Kernel-mode C ABI driven through a real loaded host runtime.

The class-level tests cover the state machines in isolation; these cover the
glue no other test reaches — argument validation as the loaded component
actually performs it, and, on hardware, the bring-up and teardown of a real
kernel context.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

PTO_RUNTIME_ERR_INTERNAL = -1000
PTO_RUNTIME_ERR_INVALID_ARGUMENT = -1004
PTO_RUNTIME_ERR_UNSUPPORTED = -1001
PTO_RUNTIME_ERR_INVALID_STATE = -1003
PTO_RUNTIME_ERR_CALLABLE_COUNT_EXCEEDED = -1005
PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED = -1006
PTO_RUNTIME_ERR_CALLABLE_NOT_RESIDENT = -1007
PTO_RUNTIME_ERR_CALLABLE_STALE = -1008

_ARCHES = ("a2a3", "a5")
_RUNTIMES = ("host_build_graph", "tensormap_and_ringbuffer")
_SIM_CASES = [
    pytest.param(arch, runtime, id=f"{arch}-sim-{runtime}", marks=pytest.mark.runtime(runtime))
    for arch in _ARCHES
    for runtime in _RUNTIMES
]
_ONBOARD_CASES = [
    pytest.param(
        arch,
        runtime,
        id=f"{arch}-onboard-{runtime}",
        marks=[
            pytest.mark.requires_hardware,
            pytest.mark.platforms([arch]),
            pytest.mark.runtime(runtime),
            pytest.mark.device_count(1),
        ],
    )
    for arch in _ARCHES
    for runtime in _RUNTIMES
]
_ONBOARD_TMR_CASES = [case for case in _ONBOARD_CASES if case.values[1] == "tensormap_and_ringbuffer"]


@pytest.fixture(scope="module")
def kernel_close_faults(tmp_path_factory):
    output = tmp_path_factory.mktemp("kernel-close-faults") / "faults.so"
    subprocess.run(
        [
            "c++",
            "-shared",
            "-fPIC",
            "-I" + str(_PROJECT_ROOT / "src/common/log/include"),
            str(Path(__file__).with_name("kernel_close_faults.cpp")),
            "-ldl",
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


@pytest.mark.parametrize(("arch", "runtime"), _ONBOARD_TMR_CASES)
@pytest.mark.parametrize(
    "scenario",
    [
        "repeat_init",
        "init_failure",
        "stream_close",
        "event_close",
        "persistent_free_close",
        "destroy_unclosed",
        "prepare",
        "prepare_capture_global",
        "prepare_capture_thread_local",
        "register_failure",
        "fatal_device",
    ],
)
def test_kernel_lifecycle_retry(arch, runtime, scenario, kernel_close_faults, request):
    _binaries(arch, runtime)
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(kernel_close_faults) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    env["PYTHONFAULTHANDLER"] = "1"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arch, runtime, device, scenario],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if scenario == "destroy_unclosed":
        assert "refusing to destroy an unclosed kernel context" in result.stdout + result.stderr


# RUNTIME_ENV_FIELD_GROUPS(3) * RUNTIME_ENV_RING_COUNT(4); the size assertion
# below is what catches a layout change rather than this constant.
_RUNTIME_ENV_UINT64_FIELDS = 12
_OUTPUT_PREFIX_BYTES = 1024


class CallConfig(ctypes.Structure):
    """Mirror of the packed CallConfig the C ABI takes by pointer."""

    _pack_ = 1
    _fields_ = [
        ("aicpu_thread_num", ctypes.c_int32),
        ("enable_chip_swimlane", ctypes.c_int32),
        ("enable_dump_args", ctypes.c_int32),
        ("enable_pmu", ctypes.c_int32),
        ("enable_dep_gen", ctypes.c_int32),
        ("enable_scope_stats", ctypes.c_int32),
        ("capture_clock_anchors", ctypes.c_int32),
        ("runtime_env", ctypes.c_uint64 * _RUNTIME_ENV_UINT64_FIELDS),
        ("output_prefix", ctypes.c_char * _OUTPUT_PREFIX_BYTES),
    ]


def _load(arch: str, variant: str, runtime: str) -> ctypes.CDLL:
    path = _PROJECT_ROOT / "build" / "lib" / arch / variant / runtime / "libhost_runtime.so"
    if not path.exists():
        pytest.skip(f"{path} not built")
    if variant == "sim":
        # A simulated host runtime resolves its device entries against the
        # simulator context, which the worker normally loads for it.
        sim_context = _PROJECT_ROOT / "build" / "lib" / "libcpu_sim_context.so"
        if not sim_context.exists():
            pytest.skip(f"{sim_context} not built")
        ctypes.CDLL(str(sim_context), mode=ctypes.RTLD_GLOBAL)  # its hooks resolve by dlsym(RTLD_DEFAULT)
    # RTLD_LOCAL, as the worker loads it: two runtimes export the same entry
    # names, so a globally-scoped load makes the second one's calls land in
    # the first.
    lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    lib.create_device_context.restype = ctypes.c_void_p
    lib.create_device_context.argtypes = []
    lib.destroy_device_context.argtypes = [ctypes.c_void_p]
    lib.finalize_device.argtypes = [ctypes.c_void_p]
    lib.finalize_device.restype = ctypes.c_int
    lib.committed_device_memory_ctx.argtypes = [ctypes.c_void_p]
    lib.committed_device_memory_ctx.restype = ctypes.c_size_t
    lib.simpler_kernel_mode_supported.argtypes = [ctypes.c_void_p]
    lib.simpler_kernel_mode_supported.restype = ctypes.c_int
    lib.simpler_kernel_mode_init.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(CallConfig), ctypes.c_uint64,
    ]  # fmt: skip
    lib.simpler_kernel_mode_init.restype = ctypes.c_int
    lib.simpler_kernel_mode_prepare_callable.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.simpler_kernel_mode_prepare_callable.restype = ctypes.c_int
    lib.simpler_kernel_mode_launch.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p]
    lib.simpler_kernel_mode_launch.restype = ctypes.c_int
    lib.simpler_init.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(CallConfig), ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64,
    ]  # fmt: skip
    lib.simpler_init.restype = ctypes.c_int
    return lib


def _minimal_callable_image() -> bytes:
    """A structurally valid ChipCallable image — the smallest input that gets
    past argument validation and reaches the ordering verdict."""
    import ctypes as _ctypes  # noqa: PLC0415

    from simpler.task_interface import ArgDirection, ChipCallable  # noqa: PLC0415

    chip = ChipCallable.build(signature=[ArgDirection.IN], func_name="f", binary=b"\x00", children=[])
    return _ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))


def _binaries(arch: str, runtime: str) -> tuple[bytes, bytes, bytes]:
    base = _PROJECT_ROOT / "build" / "lib" / arch / "onboard" / runtime
    dispatcher = _PROJECT_ROOT / "build" / "lib" / arch / "dispatcher" / "libsimpler_aicpu_dispatcher.so"
    core_name = "aicore_kernel_mode.o" if runtime == "tensormap_and_ringbuffer" else "aicore_kernel.o"
    files = (base / "libaicpu_kernel.so", base / core_name, dispatcher)
    for path in files:
        if not path.exists():
            pytest.skip(f"{path} not built")
    return tuple(path.read_bytes() for path in files)  # type: ignore[return-value]


@pytest.mark.parametrize(("arch", "runtime"), _SIM_CASES)
def test_kernel_init_rejects_malformed_arguments(arch: str, runtime: str):
    """Every component runs the same structural validation before its verdict.

    The checks live in one shared header precisely so a stub and a real
    implementation accept and reject the same arguments; that parity is only
    provable against the loaded component.
    """
    lib = _load(arch, "sim", runtime)
    config = CallConfig()
    payload = b"\x00\x01\x02\x03"
    ctx = lib.create_device_context()
    assert ctx
    try:
        # A binary and its size describe one object, so a pointer without a
        # size — or a size without a pointer — is structurally invalid.
        assert (
            lib.simpler_kernel_mode_init(
                ctx, 0, None, len(payload), payload, len(payload), payload, len(payload), ctypes.byref(config), 1
            )
            == PTO_RUNTIME_ERR_INVALID_ARGUMENT
        )
        # A negative device and a zero generation are both out of contract.
        for device_id, generation in ((-1, 1), (0, 0)):
            assert (
                lib.simpler_kernel_mode_init(
                    ctx,
                    device_id,
                    payload,
                    len(payload),
                    payload,
                    len(payload),
                    payload,
                    len(payload),
                    ctypes.byref(config),
                    generation,
                )  # fmt: skip
                == PTO_RUNTIME_ERR_INVALID_ARGUMENT
            )
        # A null config is rejected before anything else is read.
        assert (
            lib.simpler_kernel_mode_init(
                ctx, 0, payload, len(payload), payload, len(payload), payload, len(payload), None, 1
            )
            == PTO_RUNTIME_ERR_INVALID_ARGUMENT
        )
    finally:
        lib.destroy_device_context(ctx)


@pytest.mark.parametrize(("arch", "runtime"), _SIM_CASES)
def test_kernel_init_rejects_invalid_static_config_without_claim(arch: str, runtime: str):
    lib = _load(arch, "sim", runtime)
    config = CallConfig()
    payload = b"\x00\x01\x02\x03"
    ctx = lib.create_device_context()
    assert ctx
    try:
        for threads in (-1, 1, 6):
            config.aicpu_thread_num = threads
            assert (
                lib.simpler_kernel_mode_init(
                    ctx, 0, payload, len(payload), payload, len(payload), payload, len(payload), ctypes.byref(config), 1
                )
                == PTO_RUNTIME_ERR_INTERNAL
            )
        config.aicpu_thread_num = 0
        config.enable_dump_args = 1
        assert (
            lib.simpler_kernel_mode_init(
                ctx, 0, payload, len(payload), payload, len(payload), payload, len(payload), ctypes.byref(config), 1
            )
            == PTO_RUNTIME_ERR_UNSUPPORTED
        )
        config.enable_dump_args = 0
        assert (
            lib.simpler_kernel_mode_init(
                ctx, 0, payload, len(payload), payload, len(payload), payload, len(payload), ctypes.byref(config), 1
            )
            == PTO_RUNTIME_ERR_UNSUPPORTED
        )
        assert lib.committed_device_memory_ctx(ctx) == 0
        assert lib.simpler_kernel_mode_supported(ctx) == 0
    finally:
        lib.destroy_device_context(ctx)


@contextlib.contextmanager
def _caller_acl_device(lib, device):
    signatures = {
        "aclInit": [ctypes.c_char_p],
        "aclrtSetDevice": [ctypes.c_int],
        "aclrtResetDevice": [ctypes.c_int],
        "aclFinalize": [],
    }
    for symbol, argtypes in signatures.items():
        function = getattr(lib, symbol)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
    assert lib.aclInit(None) == 0
    try:
        assert lib.aclrtSetDevice(device) == 0
        try:
            yield
        finally:
            assert lib.aclrtResetDevice(device) == 0
    finally:
        assert lib.aclFinalize() == 0


def _run_lifecycle_retry(arch, runtime, device, scenario):
    lib = _load(arch, "onboard", runtime)
    faults = ctypes.CDLL(None)
    faults.bind_test_log.argtypes = [ctypes.c_void_p]
    faults.bind_test_log.restype = ctypes.c_int
    assert faults.bind_test_log(lib._handle) == 0
    faults.disarm_acl_guard.argtypes = []
    faults.disarm_acl_guard.restype = None
    # The caller initializes and later releases ACL outside the forbidden-call window.
    with _caller_acl_device(lib, device):
        try:
            _check_lifecycle_retry(lib, faults, arch, runtime, device, scenario)
        finally:
            faults.disarm_acl_guard()


def _finalize_after_quiescence(lib, ctx):
    """Close after external quiescence; preserve actual teardown errors."""
    return lib.finalize_device(ctx)


def _check_lifecycle_retry(lib, faults, arch, runtime, device, scenario):
    aicpu, aicore, dispatcher = _binaries(arch, runtime)
    config = CallConfig()
    ctx = lib.create_device_context()
    init_args = (
        ctx,
        device,
        aicpu,
        len(aicpu),
        aicore,
        len(aicore),
        dispatcher,
        len(dispatcher),
        ctypes.byref(config),
        1,
    )
    faults.arm_destroy_failure.argtypes = [ctypes.c_int]
    faults.arm_destroy_failure_after.argtypes = [ctypes.c_int, ctypes.c_int]
    faults.destroy_attempts.restype = ctypes.c_int
    lib.ensure_acl_ready_ctx.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ensure_acl_ready_ctx.restype = ctypes.c_int
    faults.acl_call_count.argtypes = [ctypes.c_int]
    faults.acl_call_count.restype = ctypes.c_int
    faults.arm_acl_guard()
    # Prove each interceptor counts and refuses calls before trusting zeros.
    forbidden_args = (
        ("aclInit", ctypes.c_char_p, None),
        ("aclrtSetDevice", ctypes.c_int, device),
        ("aclrtResetDevice", ctypes.c_int, device),
        ("aclrtResetDeviceForce", ctypes.c_int, device),
        ("aclFinalize", None, None),
        ("rtDeviceReset", ctypes.c_int, device),
        ("rtSetDevice", ctypes.c_int, device),
    )
    for i, (symbol, argtype, value) in enumerate(forbidden_args):
        fn = getattr(faults, symbol)
        fn.argtypes = [] if argtype is None else [argtype]
        fn.restype = ctypes.c_int
        assert (fn() if argtype is None else fn(value)) == -4322
        assert faults.acl_call_count(i) == 1
    faults.arm_acl_guard()
    if scenario == "init_failure":
        faults.arm_destroy_failure(3)
    finalized = False
    try:
        assert lib.simpler_kernel_mode_init(*init_args) == (-4321 if scenario == "init_failure" else 0)
        # These exported C++ methods use the Linux Itanium ABI. Resolve the
        # actual runtime's methods rather than adding production test hooks.
        force_reset = getattr(lib, "_ZN12DeviceRunner18force_reset_deviceEv")
        force_reset.argtypes = [ctypes.c_void_p]
        force_reset.restype = ctypes.c_int
        assert force_reset(ctx) == PTO_RUNTIME_ERR_INVALID_STATE
        assert lib.ensure_acl_ready_ctx(ctx, device) == PTO_RUNTIME_ERR_INVALID_STATE
        if scenario == "fatal_device":
            recover = getattr(lib, "_ZN12DeviceRunner31recover_device_or_mark_unusableEi")
            recover.argtypes = [ctypes.c_void_p, ctypes.c_int]
            recover.restype = None
            accepts = getattr(lib, "_ZNK12DeviceRunner14can_accept_runEv")
            accepts.argtypes = [ctypes.c_void_p]
            accepts.restype = ctypes.c_bool
            assert accepts(ctx)
            # Inject the drain result, not a real device fault or device reset.
            # The real recovery method sets device_unusable_ and real finalize
            # must then take the fatal-device branch.
            faults.arm_fatal_drain()
            recover(ctx, 507018)
            assert faults.fatal_drain_count() == 1
            assert not accepts(ctx)
            assert force_reset(ctx) == PTO_RUNTIME_ERR_INVALID_STATE
            assert lib.finalize_device(ctx) == 0
            finalized = True
        elif scenario == "destroy_unclosed":
            lib.destroy_device_context(ctx)
            assert lib.ensure_acl_ready_ctx(ctx, device) == PTO_RUNTIME_ERR_INVALID_STATE
            assert lib.finalize_device(ctx) == 0
            finalized = True
        elif scenario.startswith("prepare"):
            # The caller's packed buffer stops being configuration authority
            # when init returns; prepare uses its owned, validated snapshot.
            config.aicpu_thread_num = -1
            config.enable_dump_args = 1
            mode = {"prepare_capture_global": 0, "prepare_capture_thread_local": 1}.get(scenario)
            _check_prepare_reuse(lib, ctx, arch, runtime, faults=faults, capture_mode=mode)
            finalized = True
        elif scenario == "register_failure":
            _check_register_failure(lib, ctx, faults)
            finalized = True
        elif scenario == "persistent_free_close":
            _check_prepare_reuse(lib, ctx, arch, runtime, faults=faults, close=False)
            # The first rtFree releases K7 coordination storage after revoke.
            # Fail the next, owned by PersistentKernelArgs, to exercise the
            # owner -> finalize_common -> allocator retry chain.
            faults.arm_destroy_failure_after(4, 1)
            # The allocator path may translate an injected RTS status, but it
            # must never report success or forget the allocation before retry.
            assert _finalize_after_quiescence(lib, ctx) != 0
            first_attempts = faults.destroy_attempts()
            assert first_attempts > 0
            assert _finalize_after_quiescence(lib, ctx) == 0
            # A kernel context releases several device blocks, and the failure
            # stops the first pass partway, so the retry attempts the block
            # that failed plus everything the first pass never reached. The
            # invariant is that it re-attempts and completes, not a count.
            assert faults.destroy_attempts() > first_attempts
            assert lib.committed_device_memory_ctx(ctx) == 0
            finalized = True
        elif scenario in ("repeat_init", "init_failure"):
            assert lib.simpler_kernel_mode_init(*init_args) == PTO_RUNTIME_ERR_INVALID_STATE
            assert lib.ensure_acl_ready_ctx(ctx, device) == PTO_RUNTIME_ERR_INVALID_STATE
            assert lib.finalize_device(ctx) == 0
            finalized = True
        else:
            faults.arm_destroy_failure(1 if scenario == "stream_close" else 2)
            assert lib.finalize_device(ctx) == -4321
            first_attempts = faults.destroy_attempts()
            assert first_attempts > 0
            assert lib.finalize_device(ctx) == 0
            assert faults.destroy_attempts() == first_attempts + 1
            finalized = True
    finally:
        faults.finish_prepare_guard()
        if not finalized:
            assert _finalize_after_quiescence(lib, ctx) == 0
        lib.destroy_device_context(ctx)
        names = (
            "aclInit",
            "aclrtSetDevice",
            "aclrtResetDevice",
            "aclrtResetDeviceForce",
            "aclFinalize",
            "rtDeviceReset",
            "rtSetDevice",
        )
        assert {name: faults.acl_call_count(i) for i, name in enumerate(names)} == dict.fromkeys(names, 0)


def _check_register_failure(lib, ctx, faults):
    from simpler.task_interface import ChipCallable  # noqa: PLC0415

    # Host validation reads the image's sizes, offsets and names and never the
    # orchestration binary, so bytes that are not a loadable SO are admitted and
    # uploaded. Nothing dlopens them until a launch round asks for the
    # orchestration, so neither preparation nor a drain can refuse this image.
    chip = ChipCallable.build(
        signature=[], func_name="kernel_prepare_orchestration", binary=b"\x00" * 4096, children=[]
    )
    image = ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
    minted = ctypes.c_int32(99)
    assert _prepare_under_no_sync_guard(lib, ctx, faults, image, minted) == 0
    assert minted.value == 0
    committed = lib.committed_device_memory_ctx(ctx)
    assert committed > 0
    # Registration enqueues no device load, so a drain has nothing to refuse:
    # the refusal belongs to the first launch round, which this case never
    # reaches. A clean drain here is what proves the load is deferred.
    lib.aclrtSynchronizeDeviceWithTimeout.argtypes = [ctypes.c_int32]
    lib.aclrtSynchronizeDeviceWithTimeout.restype = ctypes.c_int
    assert lib.aclrtSynchronizeDeviceWithTimeout(60000) == 0
    # The registration's storage belongs to the context until an explicit close,
    # which still reclaims all of it.
    assert lib.committed_device_memory_ctx(ctx) == committed
    assert _finalize_after_quiescence(lib, ctx) == 0
    assert lib.committed_device_memory_ctx(ctx) == 0


def _prepare_under_no_sync_guard(lib, ctx, faults, image, minted):
    """Prepare with every synchronize entry refused.

    Preparation runs inside an ACLGraph capture, which no wait can sit in, so a
    refusal surfaces as prepare's own status instead of a capture that silently
    records one.
    """
    faults.arm_prepare_guard()
    try:
        return lib.simpler_kernel_mode_prepare_callable(ctx, image, len(image), ctypes.byref(minted))
    finally:
        assert faults.finish_prepare_guard() == 0, "prepare synchronized a stream or the device"


def _check_prepare_reuse(lib, ctx, arch, runtime, *, faults, close=True, capture_mode=None):
    import tempfile  # noqa: PLC0415

    from simpler.task_interface import ChipCallable  # noqa: PLC0415

    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="kernel-prepare-") as build_dir:
        binary = KernelCompiler(arch).compile_orchestration(
            runtime, str(Path(__file__).with_name("kernel_prepare_orchestration.cpp")), build_dir=build_dir
        )
    chip = ChipCallable.build(signature=[], func_name="kernel_prepare_orchestration", binary=binary, children=[])
    image = ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))
    lib.aclrtCreateStream.argtypes = [ctypes.c_void_p]
    lib.aclrtCreateStream.restype = ctypes.c_int
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    lib.aclrtDestroyStream.restype = ctypes.c_int
    lib.aclrtSynchronizeStreamWithTimeout.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.aclrtSynchronizeStreamWithTimeout.restype = ctypes.c_int
    caller_stream = ctypes.c_void_p()
    assert lib.aclrtCreateStream(ctypes.byref(caller_stream)) == 0
    # Prove the guard refuses and counts before trusting the zeros it reports.
    faults.arm_prepare_guard()
    assert faults.aclrtSynchronizeStream(None) == -4323
    assert faults.finish_prepare_guard() == 1
    model = ctypes.c_void_p()
    thread_mode = ctypes.c_int(capture_mode or 0)
    if capture_mode is not None:
        _bind_capture_entries(lib)
        # The thread override is independent of the mode Begin is given.
        assert lib.aclmdlRICaptureThreadExchangeMode(ctypes.byref(thread_mode)) == 0
        assert lib.aclmdlRICaptureBegin(caller_stream, capture_mode) == 0
    before = lib.committed_device_memory_ctx(ctx)
    minted = ctypes.c_int32(99)
    assert _prepare_under_no_sync_guard(lib, ctx, faults, image, minted) == 0
    assert minted.value == 0
    prepared = lib.committed_device_memory_ctx(ctx)
    assert prepared > before
    # Registration is pure: the same image again mints a second, distinct id
    # with its own upload. Committed device memory does not move, because the
    # code arena and its descriptor prefix are committed once on first use and
    # a second registration spends arena budget rather than new device memory.
    assert _prepare_under_no_sync_guard(lib, ctx, faults, image, minted) == 0
    assert minted.value == 1
    assert lib.committed_device_memory_ctx(ctx) == prepared
    if capture_mode is not None:
        # Each upload restored the mode it borrowed, so the capture the caller
        # opened is still the thread's own.
        restored = ctypes.c_int(2)
        assert lib.aclmdlRICaptureThreadExchangeMode(ctypes.byref(restored)) == 0
        assert restored.value == capture_mode
        assert lib.aclmdlRICaptureThreadExchangeMode(ctypes.byref(restored)) == 0
        assert lib.aclmdlRICaptureEnd(caller_stream, ctypes.byref(model)) == 0
        assert lib.aclmdlRIExecuteAsync(model, caller_stream) == 0
    assert lib.aclrtSynchronizeStreamWithTimeout(caller_stream, 60000) == 0
    # The uploads are context-owned state, not graph nodes: replaying the graph
    # they were issued under changes nothing they published.
    assert lib.committed_device_memory_ctx(ctx) == prepared
    if capture_mode is not None:
        assert lib.aclmdlRIDestroy(model) == 0
        assert lib.aclmdlRICaptureThreadExchangeMode(ctypes.byref(thread_mode)) == 0
    lib.simpler_unregister_callable.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.simpler_unregister_callable.restype = ctypes.c_int
    assert lib.simpler_unregister_callable(ctx, 0) == PTO_RUNTIME_ERR_INVALID_STATE
    if close:
        assert _finalize_after_quiescence(lib, ctx) == 0
        assert lib.committed_device_memory_ctx(ctx) == 0
        assert (
            lib.simpler_kernel_mode_prepare_callable(ctx, image, len(image), ctypes.byref(minted))
            == PTO_RUNTIME_ERR_INVALID_STATE
        )
        assert minted.value == -1
    assert lib.aclrtDestroyStream(caller_stream) == 0


def _bind_capture_entries(lib):
    lib.aclmdlRICaptureBegin.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.aclmdlRICaptureEnd.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.aclmdlRICaptureThreadExchangeMode.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.aclmdlRIExecuteAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.aclmdlRIDestroy.argtypes = [ctypes.c_void_p]
    for name in (
        "aclmdlRICaptureBegin",
        "aclmdlRICaptureEnd",
        "aclmdlRICaptureThreadExchangeMode",
        "aclmdlRIExecuteAsync",
        "aclmdlRIDestroy",
    ):
        getattr(lib, name).restype = ctypes.c_int


@pytest.mark.parametrize(("arch", "runtime"), _SIM_CASES)
def test_kernel_entries_reject_a_context_with_no_kernel_claim(arch: str, runtime: str):
    """Structurally valid calls on a context that never claimed kernel mode
    are an ordering error, not an argument error."""
    lib = _load(arch, "sim", runtime)
    image = _minimal_callable_image()
    stream = ctypes.byref((ctypes.c_uint8 * 8)())
    ctx = lib.create_device_context()
    assert ctx
    try:
        assert lib.simpler_kernel_mode_supported(ctx) == 0
        minted = ctypes.c_int32(99)
        assert (
            lib.simpler_kernel_mode_prepare_callable(ctx, image, len(image), ctypes.byref(minted))
            == PTO_RUNTIME_ERR_INVALID_STATE
        )
        assert minted.value == -1
        assert lib.simpler_kernel_mode_launch(ctx, 0, image, stream) == PTO_RUNTIME_ERR_INVALID_STATE
        # A truncated image and a missing out parameter are argument errors, so
        # the structural checks run before the ordering one.
        assert (
            lib.simpler_kernel_mode_prepare_callable(ctx, image, 1, ctypes.byref(minted))
            == PTO_RUNTIME_ERR_INVALID_ARGUMENT
        )
        assert (
            lib.simpler_kernel_mode_prepare_callable(ctx, image, len(image) - 1, ctypes.byref(minted))
            == PTO_RUNTIME_ERR_INVALID_ARGUMENT
        )
        assert (
            lib.simpler_kernel_mode_prepare_callable(ctx, image, len(image), None) == PTO_RUNTIME_ERR_INVALID_ARGUMENT
        )
        assert lib.simpler_kernel_mode_launch(ctx, -1, image, stream) == PTO_RUNTIME_ERR_INVALID_ARGUMENT
        assert lib.simpler_kernel_mode_launch(ctx, 0, image, None) == PTO_RUNTIME_ERR_INVALID_ARGUMENT
    finally:
        lib.destroy_device_context(ctx)


@pytest.mark.parametrize(("arch", "runtime"), _SIM_CASES)
def test_simulated_components_report_kernel_mode_unsupported(arch: str, runtime: str):
    """A component without kernel-mode execution still validates first, then
    reports unsupported — it never claims a mode it cannot honor."""
    lib = _load(arch, "sim", runtime)
    config = CallConfig()
    payload = b"\x00"
    ctx = lib.create_device_context()
    assert ctx
    try:
        assert (
            lib.simpler_kernel_mode_init(
                ctx, 0, payload, len(payload), payload, len(payload), payload, len(payload), ctypes.byref(config), 1
            )
            == PTO_RUNTIME_ERR_UNSUPPORTED
        )
        # The refused init took no claim, so the context is still free.
        image = _minimal_callable_image()
        minted = ctypes.c_int32(99)
        assert (
            lib.simpler_kernel_mode_prepare_callable(ctx, image, len(image), ctypes.byref(minted))
            == PTO_RUNTIME_ERR_INVALID_STATE
        )
    finally:
        lib.destroy_device_context(ctx)


@pytest.mark.parametrize(("arch", "runtime"), _ONBOARD_CASES)
@pytest.mark.parametrize("event_mode", [None, 0, 1], ids=["default", "software", "hardware"])
def test_kernel_context_init_respects_runtime_support_on_a_borrowed_device(
    arch: str, runtime: str, event_mode, request
):
    """Both onboard runtimes claim the borrowed device without owning its lifecycle."""
    lib = _load(arch, "onboard", runtime)
    aicpu, aicore, dispatcher = _binaries(arch, runtime)
    config = CallConfig()
    device_id = int(str(request.config.getoption("--device")).split("-")[0].split(",")[0])
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    lib.rtSetDevice.restype = ctypes.c_int
    assert lib.rtSetDevice(device_id) == 0
    if event_mode is not None:
        lib.rtEventWorkModeSet.argtypes = [ctypes.c_uint8]
        lib.rtEventWorkModeSet.restype = ctypes.c_int
        assert lib.rtEventWorkModeSet(event_mode) in (0, 207000)

    ctx = lib.create_device_context()
    assert ctx
    try:
        status = lib.simpler_kernel_mode_init(
                ctx,
                device_id,
                aicpu,
                len(aicpu),
                aicore,
                len(aicore),
                dispatcher,
                len(dispatcher),
                ctypes.byref(config),
                1,
            )  # fmt: skip
        assert status == 0
        mode = ctypes.c_uint8()
        lib.rtEventWorkModeGet.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
        lib.rtEventWorkModeGet.restype = ctypes.c_int
        mode_status = lib.rtEventWorkModeGet(ctypes.byref(mode))
        # CANN fixes hardware mode on chips that do not expose this query.
        assert mode_status in (0, 207000)
        if mode_status == 0:
            assert mode.value == (0 if event_mode == 0 else 1)
        assert lib.simpler_kernel_mode_supported(ctx) == 1
        # The claim is exclusive for the context's whole life.
        assert (
            lib.simpler_init(
                ctx,
                device_id,
                aicpu,
                len(aicpu),
                aicore,
                len(aicore),
                dispatcher,
                len(dispatcher),
                ctypes.byref(config),
                0,
                None,
                0,
            )  # fmt: skip
            == PTO_RUNTIME_ERR_INVALID_STATE
        )
        assert lib.finalize_device(ctx) == 0
    finally:
        # Destroying after a clean close is allowed; an unclosed kernel
        # context would be refused and leaked instead.
        lib.destroy_device_context(ctx)


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(1)
def test_kernel_eager_launch_executes_fresh_tensor_and_scalar_snapshots(request):
    _binaries("a2a3", "tensormap_and_ringbuffer")
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "a2a3", "tensormap_and_ringbuffer", device, "eager_values"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime("host_build_graph")
@pytest.mark.device_count(1)
@pytest.mark.parametrize("scenario", ["eager_values", "intermediate_queued"])
def test_hbg_kernel_eager_launch_executes_graph_snapshots(request, scenario):
    _binaries("a2a3", "host_build_graph")
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "a2a3", "host_build_graph", device, scenario],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(("arch", "runtime"), _ONBOARD_TMR_CASES)
@pytest.mark.parametrize("scenario", ["device_query_error", "device_mismatch"])
def test_kernel_device_query_rejections_keep_context_reusable(arch, runtime, scenario, kernel_close_faults, request):
    _binaries(arch, runtime)
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(kernel_close_faults) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arch, runtime, device, scenario],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _build_eager_callable(arch, runtime, intermediate=False):
    import tempfile  # noqa: PLC0415

    from simpler.task_interface import (  # noqa: PLC0415
        ArgDirection,
        ChipCallable,
        CoreCallable,
    )

    from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    compiler = KernelCompiler(arch)
    kernel = (
        _PROJECT_ROOT / f"examples/{arch}/tensormap_and_ringbuffer/vector_example/kernels/aiv/kernel_add_scalar.cpp"
    )
    with tempfile.TemporaryDirectory(prefix="kernel-eager-") as build_dir:
        orchestration = compiler.compile_orchestration(
            runtime, str(Path(__file__).with_name("kernel_eager_orchestration.cpp")), build_dir=build_dir
        )
        incore = compiler.compile_incore(
            str(kernel),
            core_type="aiv",
            pto_isa_root=ensure_pto_isa_root(),
            extra_include_dirs=compiler.get_orchestration_include_dirs(runtime),
            build_dir=build_dir,
        )
    signature = [ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR]
    child = CoreCallable.build(signature=signature, binary=extract_text_section(incore))
    return ChipCallable.build(
        signature=signature,
        func_name="kernel_eager_intermediate" if intermediate else "kernel_eager_orchestration",
        binary=orchestration,
        children=[(0, child)],
    )


def _check_device_query_rejection(lib, ctx, device, scenario, operation, *arguments):
    faults = ctypes.CDLL(None)
    faults.arm_device_query_override.argtypes = [ctypes.c_int, ctypes.c_int]
    faults.arm_device_query_override.restype = None
    faults.clear_device_query_override.argtypes = []
    faults.clear_device_query_override.restype = None
    faults.device_query_attempts.argtypes = []
    faults.device_query_attempts.restype = ctypes.c_int
    before = lib.committed_device_memory_ctx(ctx)
    query_result = -4323 if scenario == "device_query_error" else 0
    expected = query_result if query_result else PTO_RUNTIME_ERR_INVALID_STATE
    # Reporting a different ID exercises the guard without selecting another device.
    faults.arm_device_query_override(query_result, device + 1)
    try:
        assert operation(*arguments) == expected
        assert faults.device_query_attempts() == 1
        assert lib.committed_device_memory_ctx(ctx) == before
    finally:
        faults.clear_device_query_override()


def _allocate_eager_invocations(lib, host_array, scalars, allocations):
    count = host_array._length_
    bytes_per_tensor = ctypes.sizeof(host_array)
    invocations = []
    for round_index, scalar in enumerate(scalars):
        addresses = []
        for _ in range(2):
            address = ctypes.c_void_p()
            assert lib.aclrtMalloc(ctypes.byref(address), bytes_per_tensor, 0) == 0
            allocations.append(address)
            addresses.append(address)
        source, destination = addresses
        values = [float(i % 127 + round_index * 257) for i in range(count)]
        host_input = host_array(*values)
        host_output = host_array(*([-999.0] * count))
        assert lib.aclrtMemcpy(source, bytes_per_tensor, host_input, bytes_per_tensor, 1) == 0
        assert lib.aclrtMemcpy(destination, bytes_per_tensor, host_output, bytes_per_tensor, 1) == 0
        invocations.append((source, destination, scalar, values, host_output))
    return invocations


def _run_eager_values(arch, runtime, device, scenario="eager_values"):
    import struct  # noqa: PLC0415

    from simpler.task_interface import ChipStorageTaskArgs, ChipTensor, DataType  # noqa: PLC0415

    queued = scenario == "intermediate_queued"
    chip = _build_eager_callable(arch, runtime, intermediate=queued)
    check_device_query = scenario in ("device_query_error", "device_mismatch")
    lib = _load(arch, "onboard", runtime)
    acl_signatures = {
        "aclInit": [ctypes.c_char_p],
        "aclFinalize": [],
        "aclrtSetDevice": [ctypes.c_int],
        "aclrtResetDevice": [ctypes.c_int],
        "aclrtCreateStream": [ctypes.POINTER(ctypes.c_void_p)],
        "aclrtDestroyStream": [ctypes.c_void_p],
        "aclrtSynchronizeStreamWithTimeout": [ctypes.c_void_p, ctypes.c_int32],
        "aclrtMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int],
        "aclrtFree": [ctypes.c_void_p],
        "aclrtMemcpy": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
    }
    for symbol, argtypes in acl_signatures.items():
        function = getattr(lib, symbol)
        function.argtypes = argtypes
        function.restype = ctypes.c_int

    assert lib.aclInit(None) == 0
    caller_stream = ctypes.c_void_p()
    allocations = []
    ctx = None
    selected = False
    initialized = False
    try:
        assert lib.aclrtSetDevice(device) == 0
        selected = True
        assert lib.aclrtCreateStream(ctypes.byref(caller_stream)) == 0
        ctx = lib.create_device_context()
        assert ctx
        aicpu, aicore, dispatcher = _binaries(arch, runtime)
        config = CallConfig()
        for ring in range(4):
            config.runtime_env[ring] = 64
            config.runtime_env[4 + ring] = 1 << 20
            config.runtime_env[8 + ring] = 1024
        assert (
            lib.simpler_kernel_mode_init(
                ctx,
                device,
                aicpu,
                len(aicpu),
                aicore,
                len(aicore),
                dispatcher,
                len(dispatcher),
                ctypes.byref(config),
                71,
            )
            == 0
        )
        initialized = True
        assert lib.simpler_kernel_mode_supported(ctx) == 1
        minted = ctypes.c_int32(99)
        if check_device_query:
            _check_device_query_rejection(
                lib,
                ctx,
                device,
                scenario,
                lib.simpler_kernel_mode_prepare_callable,
                ctx,
                chip.buffer_ptr(),
                chip.buffer_size(),
                ctypes.byref(minted),
            )
        assert (
            lib.simpler_kernel_mode_prepare_callable(ctx, chip.buffer_ptr(), chip.buffer_size(), ctypes.byref(minted))
            == 0
        )
        assert minted.value == 0
        assert lib.aclrtSynchronizeStreamWithTimeout(caller_stream, 60000) == 0
        committed = lib.committed_device_memory_ctx(ctx)
        assert committed > 0

        count = 128 * 128
        host_array = ctypes.c_float * count
        bytes_per_tensor = ctypes.sizeof(host_array)
        results = []
        args = ChipStorageTaskArgs()
        invocations = _allocate_eager_invocations(lib, host_array, (1.25, -3.5) * (8 if queued else 1), allocations)
        for round_index, (source, destination, scalar, values, host_output) in enumerate(invocations):
            args.clear()
            args.add_tensor(ChipTensor.make(source.value, (count,), DataType.FLOAT32, child_memory=True))
            args.add_tensor(ChipTensor.make(destination.value, (count,), DataType.FLOAT32, child_memory=True))
            args.add_scalar(int.from_bytes(struct.pack("<f", scalar), "little"))
            if check_device_query:
                _check_device_query_rejection(
                    lib, ctx, device, scenario, lib.simpler_kernel_mode_launch, ctx, 0, args.__ptr__(), caller_stream
                )
                assert lib.aclrtSynchronizeStreamWithTimeout(caller_stream, 60000) == 0
                assert lib.aclrtMemcpy(host_output, bytes_per_tensor, destination, bytes_per_tensor, 2) == 0
                assert list(host_output) == [-999.0] * count
            launch_rc = lib.simpler_kernel_mode_launch(ctx, 0, args.__ptr__(), caller_stream)
            assert launch_rc == 0, f"launch round {round_index} returned {launch_rc}"
            # CANN owns the launch snapshot after enqueue returns.
            args.clear()
            expected = [value + scalar * (2 if queued else 1) for value in values]
            if not queued:
                assert lib.aclrtSynchronizeStreamWithTimeout(caller_stream, 60000) == 0
                assert lib.aclrtMemcpy(host_output, bytes_per_tensor, destination, bytes_per_tensor, 2) == 0
                assert list(host_output) == expected
            assert lib.committed_device_memory_ctx(ctx) == committed
            results.append((destination, expected))
            if check_device_query and round_index == 0:
                # The next round must still execute after a refused close.
                _check_device_query_rejection(lib, ctx, device, scenario, lib.finalize_device, ctx)

        assert lib.aclrtSynchronizeStreamWithTimeout(caller_stream, 60000) == 0
        first_output = host_array()
        for destination, expected in results:
            assert lib.aclrtMemcpy(first_output, bytes_per_tensor, destination, bytes_per_tensor, 2) == 0
            assert list(first_output) == expected
        assert _finalize_after_quiescence(lib, ctx) == 0
        initialized = False
        assert lib.committed_device_memory_ctx(ctx) == 0
    finally:
        if caller_stream:
            lib.aclrtSynchronizeStreamWithTimeout(caller_stream, 60000)
        if ctx:
            if initialized:
                _finalize_after_quiescence(lib, ctx)
            lib.destroy_device_context(ctx)
        for address in reversed(allocations):
            assert lib.aclrtFree(address) == 0
        if caller_stream:
            assert lib.aclrtDestroyStream(caller_stream) == 0
        if selected:
            assert lib.aclrtResetDevice(device) == 0
        assert lib.aclFinalize() == 0


if __name__ == "__main__":
    if sys.argv[4] in ("eager_values", "intermediate_queued", "device_query_error", "device_mismatch"):
        _run_eager_values(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        _run_lifecycle_retry(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
