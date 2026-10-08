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

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

PTO_RUNTIME_ERR_INTERNAL = -1000
PTO_RUNTIME_ERR_UNSUPPORTED = -1001
PTO_RUNTIME_ERR_INVALID_STATE = -1003
PTO_RUNTIME_ERR_INVALID_ARGUMENT = -1004

_ARCHES = ("a2a3", "a5")
_RUNTIMES = ("host_build_graph", "tensormap_and_ringbuffer")
_SIM_CASES = [pytest.param(arch, runtime, id=f"{arch}-sim-{runtime}") for arch in _ARCHES for runtime in _RUNTIMES]
# These lifecycle fault cases include TMR-specific callable device uploads.
# Per-invocation preparation below independently exercises both runtimes.
_KERNEL_CAPABLE_RUNTIMES = ("tensormap_and_ringbuffer",)
_ONBOARD_CASES = [
    pytest.param(
        arch,
        runtime,
        id=f"{arch}-onboard-{runtime}",
        marks=[pytest.mark.requires_hardware, pytest.mark.platforms([arch])],
    )
    for arch in _ARCHES
    for runtime in _KERNEL_CAPABLE_RUNTIMES
]


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


@pytest.mark.parametrize(("arch", "runtime"), _ONBOARD_CASES)
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
        "fatal_device",
    ],
)
def test_kernel_lifecycle_retry(arch, runtime, scenario, kernel_close_faults, request):
    _binaries(arch, runtime)
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(kernel_close_faults) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
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
    if scenario == "fatal_device":
        # The quarantine is observable and terminal: both closes report that
        # the resources stay pinned, and destruction is refused because they
        # are still owned.
        combined = result.stdout + result.stderr
        assert combined.count("quarantined — resources stay pinned for the process lifetime") == 2
        assert "refusing to destroy an unclosed kernel context" in combined


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
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_size_t,
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
    files = (base / "libaicpu_kernel.so", base / "aicore_kernel.o", dispatcher)
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


def _run_lifecycle_retry(arch, runtime, device, scenario):
    lib = _load(arch, "onboard", runtime)
    # Caller-owned binding precedes kernel init and the forbidden-call window.
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    lib.rtSetDevice.restype = ctypes.c_int
    assert lib.rtSetDevice(device) == 0
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
    faults = ctypes.CDLL(None)
    faults.bind_test_log.argtypes = [ctypes.c_void_p]
    faults.bind_test_log.restype = ctypes.c_int
    assert faults.bind_test_log(lib._handle) == 0
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
    assert lib.simpler_kernel_mode_init(*init_args) == (-4321 if scenario == "init_failure" else 0)
    finalized = False
    try:
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
            # A borrowed device gets no reset, and no reset means no confirmed
            # recovery: finalize quarantines and says so rather than reporting
            # a clean teardown it did not perform. The streams, events and
            # argument blocks are still live on the caller's device, so they
            # stay owned — which is why destruction is still refused, and why a
            # repeat finalize reaches them again instead of returning early on
            # a cleared device identity.
            assert lib.finalize_device(ctx) == PTO_RUNTIME_ERR_INVALID_STATE
            assert lib.finalize_device(ctx) == PTO_RUNTIME_ERR_INVALID_STATE
            assert not accepts(ctx)
            finalized = True
        elif scenario == "destroy_unclosed":
            lib.destroy_device_context(ctx)
            assert lib.ensure_acl_ready_ctx(ctx, device) == PTO_RUNTIME_ERR_INVALID_STATE
            assert lib.finalize_device(ctx) == 0
            finalized = True
        elif scenario == "prepare":
            _check_prepare_reuse(lib, ctx, arch, runtime)
            finalized = True
        elif scenario == "persistent_free_close":
            _check_prepare_reuse(lib, ctx, arch, runtime, close=False)
            # The first rtFree releases the callable upload. Fail the next
            # call, which is owned by PersistentKernelArgs, to cover the exact
            # owner -> finalize_common -> allocator retry chain.
            faults.arm_destroy_failure_after(4, 1)
            # The allocator path may translate an injected RTS status, but it
            # must never report success or forget the allocation before retry.
            assert lib.finalize_device(ctx) != 0
            first_attempts = faults.destroy_attempts()
            assert first_attempts > 0
            assert lib.finalize_device(ctx) == 0
            assert faults.destroy_attempts() == first_attempts + 1
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
        if not finalized:
            assert lib.finalize_device(ctx) == 0
        lib.destroy_device_context(ctx)
        names = (
            "aclInit",
            "aclrtSetDevice",
            "aclrtResetDevice",
            "aclrtResetDeviceForce",
            "aclFinalize",
            "rtDeviceReset",
        )
        assert {name: faults.acl_call_count(i) for i, name in enumerate(names)} == dict.fromkeys(names, 0)


def _prepared_callable_image(arch, runtime, *, signature=(), function="kernel_prepare_orchestration") -> bytes:
    """A real orchestration callable image — both registration paths validate
    and hash the whole image, and the AICPU prewarm loads its SO."""
    import tempfile  # noqa: PLC0415

    from simpler.task_interface import ChipCallable  # noqa: PLC0415

    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="kernel-prepare-") as build_dir:
        binary = KernelCompiler(arch).compile_orchestration(
            runtime, str(Path(__file__).with_name("kernel_prepare_orchestration.cpp")), build_dir=build_dir
        )
    chip = ChipCallable.build(signature=list(signature), func_name=function, binary=binary, children=[])
    return ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size()))


def _check_prepare_reuse(lib, ctx, arch, runtime, *, close=True):
    image = _prepared_callable_image(arch, runtime)
    before = lib.committed_device_memory_ctx(ctx)
    assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) == 0
    prepared = lib.committed_device_memory_ctx(ctx)
    assert prepared > before
    # Duplicate registration is rejected; it must not disturb the first ID.
    assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) != 0
    assert lib.committed_device_memory_ctx(ctx) == prepared
    # Identical bytes deduplicate the callable upload. The second ID also
    # reuses the context's persistent argument blocks, so neither adds GM.
    assert lib.simpler_kernel_mode_prepare_callable(ctx, 1, image, len(image)) == 0
    assert lib.committed_device_memory_ctx(ctx) == prepared
    if close:
        assert lib.finalize_device(ctx) == 0
        assert lib.committed_device_memory_ctx(ctx) == 0
        assert lib.simpler_kernel_mode_prepare_callable(ctx, 2, image, len(image)) == PTO_RUNTIME_ERR_INVALID_STATE


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
        assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) == PTO_RUNTIME_ERR_INVALID_STATE
        assert lib.simpler_kernel_mode_launch(ctx, 0, image, stream) == PTO_RUNTIME_ERR_INVALID_STATE
        # An out-of-range callable id and a truncated image are argument
        # errors, so the structural checks run before the ordering one.
        assert lib.simpler_kernel_mode_prepare_callable(ctx, -1, image, len(image)) == PTO_RUNTIME_ERR_INVALID_ARGUMENT
        assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, 1) == PTO_RUNTIME_ERR_INVALID_ARGUMENT
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
        assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) == PTO_RUNTIME_ERR_INVALID_STATE
    finally:
        lib.destroy_device_context(ctx)


@pytest.mark.parametrize(("arch", "runtime"), _ONBOARD_CASES)
def test_kernel_context_brings_up_and_closes_on_a_borrowed_device(arch: str, runtime: str, request):
    """A kernel context comes up on a device the caller owns, refuses the
    program-mode entry for the rest of its life, and releases everything it
    created on close."""
    lib = _load(arch, "onboard", runtime)
    aicpu, aicore, dispatcher = _binaries(arch, runtime)
    config = CallConfig()
    device_id = int(str(request.config.getoption("--device")).split("-")[0].split(",")[0])
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    lib.rtSetDevice.restype = ctypes.c_int
    assert lib.rtSetDevice(device_id) == 0

    ctx = lib.create_device_context()
    assert ctx
    try:
        assert (
            lib.simpler_kernel_mode_init(
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
            == 0
        )
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


def _run_program_callable_release(arch, runtime, device):
    """`chip_callable_buffers_` is shared with program registration, and a
    program context does not participate in the retained-owner protocol: its
    device is reset at finalize, so a record of a failed release would outlive
    the generation its address belonged to."""
    lib = _load(arch, "onboard", runtime)
    lib.simpler_register_callable.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    lib.simpler_register_callable.restype = ctypes.c_int
    lib.simpler_unregister_callable.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.simpler_unregister_callable.restype = ctypes.c_int
    aicpu, aicore, dispatcher = _binaries(arch, runtime)
    config = CallConfig()
    faults = ctypes.CDLL(None)
    faults.bind_test_log.argtypes = [ctypes.c_void_p]
    faults.bind_test_log.restype = ctypes.c_int
    assert faults.bind_test_log(lib._handle) == 0
    faults.arm_destroy_failure_after.argtypes = [ctypes.c_int, ctypes.c_int]
    faults.destroy_attempts.restype = ctypes.c_int
    faults.free_failure_hits.restype = ctypes.c_int
    faults.watched_alloc_found.restype = ctypes.c_int
    image = _prepared_callable_image(arch, runtime)
    ctx = lib.create_device_context()
    assert ctx
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
        0,
        None,
        0,
    )
    try:
        assert lib.simpler_init(*init_args) == 0
        empty = lib.committed_device_memory_ctx(ctx)
        assert lib.simpler_register_callable(ctx, 0, image) == 0
        uploaded = lib.committed_device_memory_ctx(ctx)
        assert uploaded > empty
        # `unregister_callable` performs exactly one rtFree, the callable
        # buffer's, and discards its result — so failing the next one fails
        # that release and the caller is told nothing.
        faults.arm_destroy_failure_after(4, 0)
        assert lib.simpler_unregister_callable(ctx, 0) == 0
        assert faults.destroy_attempts() == 1
        # Registering the same bytes must upload again: the failed release left
        # no entry, so the hash resolves to neither a freed address nor a block
        # awaiting a release retry. Committed memory is the discriminator — a
        # dedup hit would leave it unchanged.
        #
        # That re-upload is also the block the second half needs, so its free is
        # armed here: the callable buffer is the first allocation a registration
        # makes, and it stays registered, so finalize's cleanup loop is what
        # frees it.
        faults.arm_free_failure_for_next_alloc()
        assert lib.simpler_register_callable(ctx, 0, image) == 0
        assert faults.watched_alloc_found() == 1
        assert lib.committed_device_memory_ctx(ctx) > uploaded

        # The second branch: a failed free inside finalize's own cleanup loop,
        # which is not the release path above.
        assert lib.finalize_device(ctx) != 0
        assert faults.free_failure_hits() == 1
        # A failed teardown leaves an Unproven context. Clearing the callable
        # bookkeeping does not authorize reinitializing that handle.
        assert lib.simpler_init(*init_args) == PTO_RUNTIME_ERR_INVALID_STATE
    finally:
        lib.destroy_device_context(ctx)


def _run_loader_unload_retry(arch, runtime, device, scenario):
    """The AICPU loader's binary is a kernel-context owner: a failed
    `rtsBinaryUnload` keeps its handle, reports the failure out of close, and
    holds off destruction until a retry succeeds. A kernel context borrows its
    device and never resets it, so nothing else can retire that binary."""
    lib = _load(arch, "onboard", runtime)
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    lib.rtSetDevice.restype = ctypes.c_int
    assert lib.rtSetDevice(device) == 0
    aicpu, aicore, dispatcher = _binaries(arch, runtime)
    config = CallConfig()
    faults = ctypes.CDLL(None)
    faults.bind_test_log.argtypes = [ctypes.c_void_p]
    faults.bind_test_log.restype = ctypes.c_int
    assert faults.bind_test_log(lib._handle) == 0
    faults.arm_unload_failures.argtypes = [ctypes.c_int]
    faults.unload_call_count.restype = ctypes.c_int
    faults.arm_func_lookup_failure.argtypes = [ctypes.c_int]
    ctx = lib.create_device_context()
    assert ctx
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
    try:
        if scenario == "init_rollback":
            # Symbol resolution fails after the binary is loaded, so `Init`
            # rolls back — and its rollback unload fails too. The handle is the
            # only thing naming a loaded binary, so init must not clear it.
            faults.arm_func_lookup_failure(1)
            faults.arm_unload_failures(1)
            assert lib.simpler_kernel_mode_init(*init_args) != 0
            assert faults.unload_call_count() == 1
            # The retained binary keeps the context from being destroyed, and
            # the close that retries the unload is what releases it. Whatever
            # else partial init left behind is released on that same close.
            assert lib.finalize_device(ctx) == 0
            assert faults.unload_call_count() == 2
        else:
            assert lib.simpler_kernel_mode_init(*init_args) == 0
            faults.arm_unload_failures(1)
            # Close reports the unload failure instead of swallowing it, and
            # stops before the allocator finalize so every other owner is still
            # reachable for the retry.
            assert lib.finalize_device(ctx) != 0
            assert faults.unload_call_count() == 1
            # Destruction is refused while the binary is loaded; it is not a
            # resource a borrowed-device context can abandon.
            lib.destroy_device_context(ctx)
            assert lib.finalize_device(ctx) == 0
            assert faults.unload_call_count() == 2
            assert lib.committed_device_memory_ctx(ctx) == 0
    finally:
        lib.destroy_device_context(ctx)


def _run_program_loader_unload(arch, runtime, device):
    """A program close that resets its device ends the generation a retained
    binary handle belongs to, so the retention must not outlive the reset: the
    failure is reported and the handle is retired. The Unproven context
    refuses reinitialization; destruction must not unload the stale handle."""
    lib = _load(arch, "onboard", runtime)
    aicpu, aicore, dispatcher = _binaries(arch, runtime)
    config = CallConfig()
    faults = ctypes.CDLL(None)
    faults.bind_test_log.argtypes = [ctypes.c_void_p]
    faults.bind_test_log.restype = ctypes.c_int
    assert faults.bind_test_log(lib._handle) == 0
    faults.arm_unload_failures.argtypes = [ctypes.c_int]
    faults.unload_call_count.restype = ctypes.c_int
    ctx = lib.create_device_context()
    assert ctx
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
        0,
        None,
        0,
    )
    destroyed = False
    try:
        assert lib.simpler_init(*init_args) == 0
        faults.arm_unload_failures(1)
        # The failure is reported out of close on the program path too — it just
        # does not hold the context, because the reset below invalidates the
        # handle rather than leaving it retryable.
        assert lib.finalize_device(ctx) != 0
        assert faults.unload_call_count() == 1
        # The failed teardown leaves this handle Unproven, even though reset
        # retired the loader. It cannot start another lifecycle.
        assert lib.simpler_init(*init_args) == PTO_RUNTIME_ERR_INVALID_STATE
        assert faults.unload_call_count() == 1
        after_close = faults.unload_call_count()
        lib.destroy_device_context(ctx)
        destroyed = True
        # Nothing is left for `~LoadAicpuOp` to unload.
        assert faults.unload_call_count() == after_close
    finally:
        if not destroyed:
            lib.destroy_device_context(ctx)


def _run_program_loader_double_failure(arch, runtime, device, use_acl):
    """When a program close's unload *and* its device reset both fail, the
    retained handle has no reachable retry — `device_id_` is cleared and the
    next close returns early. It is therefore abandoned at that point, so
    nothing issues an unreported unload against a device whose reset never
    completed, and the context is still reusable.

    Covers the two soft-reset arms. `aclrtResetDeviceForce` is deliberately not
    injectable here: it belongs to the fatal branch, which this change does not
    touch and which has its own abandonment already."""
    lib = _load(arch, "onboard", runtime)
    lib.ensure_acl_ready_ctx.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ensure_acl_ready_ctx.restype = ctypes.c_int
    aicpu, aicore, dispatcher = _binaries(arch, runtime)
    config = CallConfig()
    faults = ctypes.CDLL(None)
    faults.bind_test_log.argtypes = [ctypes.c_void_p]
    faults.bind_test_log.restype = ctypes.c_int
    assert faults.bind_test_log(lib._handle) == 0
    faults.arm_unload_failures.argtypes = [ctypes.c_int]
    faults.unload_call_count.restype = ctypes.c_int
    faults.arm_reset_failures.argtypes = [ctypes.c_int]
    faults.reset_call_count.restype = ctypes.c_int
    ctx = lib.create_device_context()
    assert ctx
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
        0,
        None,
        0,
    )
    destroyed = False
    try:
        assert lib.simpler_init(*init_args) == 0
        if use_acl:
            # Brings `acl_ready_` up so finalize takes the aclrtResetDevice arm
            # instead of the bare rtDeviceReset one.
            assert lib.ensure_acl_ready_ctx(ctx, device) == 0
        faults.arm_unload_failures(1)
        faults.arm_reset_failures(1)
        assert lib.finalize_device(ctx) != 0
        assert faults.unload_call_count() == 1
        assert faults.reset_call_count() >= 1
        # `device_id_` is cleared regardless of the reset's result, so this
        # close is the last one that could reach anything. It reports the
        # failure; the next is idempotent over a context that owns nothing.
        assert lib.finalize_device(ctx) == 0
        assert faults.unload_call_count() == 1
        # A repeat no-op finalize does not turn an unproven reset into proof.
        assert lib.simpler_init(*init_args) == PTO_RUNTIME_ERR_INVALID_STATE
        assert faults.unload_call_count() == 1
        lib.destroy_device_context(ctx)
        destroyed = True
        # Destruction must not retry the abandoned loader against a device
        # whose reset was never confirmed.
        assert faults.unload_call_count() == 1
    finally:
        if not destroyed:
            lib.destroy_device_context(ctx)


@pytest.mark.parametrize("scenario", ["close_retry", "init_rollback"])
@pytest.mark.parametrize(("arch", "runtime"), _ONBOARD_CASES)
def test_loader_unload_failure_keeps_a_retryable_owner(arch, runtime, scenario, kernel_close_faults, request):
    """A failed `rtsBinaryUnload` must leave the loader holding its handle, on
    the close path and on the partial-init rollback path alike."""
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(kernel_close_faults) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arch, runtime, device, "loader_" + scenario],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if scenario == "close_retry":
        assert "refusing to destroy an unclosed kernel context" in result.stdout + result.stderr


@pytest.mark.parametrize(("arch", "runtime"), _ONBOARD_CASES)
def test_program_loader_unload_failure_does_not_outlive_its_device(arch, runtime, kernel_close_faults, request):
    """The same retention on a program context must not survive that context's
    device reset: the failure is reported, the stale handle is retired, and
    the unproven context refuses reinitialization."""
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(kernel_close_faults) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arch, runtime, device, "program_loader_unload"],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    # The discriminators are the unload count and refusal to reinitialize an
    # unproven context; the retirement itself logs at WARN,
    # which the test log threshold does not admit.
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(("arch", "runtime"), _ONBOARD_CASES)
def test_program_callable_release_failure_leaves_no_stale_record(arch, runtime, kernel_close_faults, request):
    """A program-mode callable release whose device free fails must stay
    recoverable within that lifetime: the block leaks, but the content hash
    is registerable again. A subsequent failed finalize forbids handle reuse.

    On tensormap_and_ringbuffer because that is where a callable image is
    uploaded to the device at all — host_build_graph registers a host dlopen
    handle and carries no `chip_buffer_hash` to release.
    """
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(kernel_close_faults) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arch, runtime, device, "program_callable_release"],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("reset_arm", ["rt", "acl"])
@pytest.mark.parametrize(("arch", "runtime"), _ONBOARD_CASES)
def test_program_loader_double_failure_leaves_nothing_to_unload(arch, runtime, reset_arm, kernel_close_faults, request):
    """A program close whose unload and device reset both fail must not leave a
    retryable-looking handle behind: there is no reachable retry once
    `device_id_` is cleared, so the destructor must issue no further unload."""
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(kernel_close_faults) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arch, runtime, device, "program_double_" + reset_arm],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class NativeRunDescriptor(ctypes.Structure):
    _fields_ = [
        ("pipeline_slot", ctypes.c_uint32),
        ("arena_bank", ctypes.c_uint32),
        ("run_id", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("dispatch_id", ctypes.c_uint64),
        ("run_epoch", ctypes.c_uint64),
        ("accepted_state", ctypes.POINTER(ctypes.c_int32)),
        ("accepted_value", ctypes.c_int32),
        ("joinable_boundary", ctypes.c_uint32),
    ]


def _run_borrowed_prepare(arch, runtime, device):
    from simpler.task_interface import (  # noqa: PLC0415
        AddressSpace,
        ArgDirection,
        ChipStorageTaskArgs,
        ChipTensor,
        DataType,
        TensorTransfer,
    )

    lib = _load(arch, "onboard", runtime)
    faults = ctypes.CDLL(None)
    faults.bind_test_log.argtypes = [ctypes.c_void_p]
    assert faults.bind_test_log(lib._handle) == 0
    faults.force_current_device.argtypes = [ctypes.c_int]
    faults.acl_call_count.argtypes = [ctypes.c_int]
    lib.rtSetDevice.argtypes = [ctypes.c_int]
    assert lib.rtSetDevice(device) == 0
    faults.arm_acl_guard()
    config = CallConfig()
    # No kernel is launched; two threads are the smallest legal explicit shape
    # for preparing either runtime without a ChipWorker's topology resolver.
    config.aicpu_thread_num = 2
    aicpu, aicore, dispatcher = _binaries(arch, runtime)
    ctx = lib.create_device_context()
    assert ctx
    run = None
    prepared = False
    try:
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
                1,
            )
            == 0
        )
        image = _prepared_callable_image(
            arch, runtime, signature=[ArgDirection.INOUT], function="kernel_call_orchestration"
        )
        assert lib.simpler_kernel_mode_prepare_callable(ctx, 0, image, len(image)) == 0
        for name in ("get_runtime_size", "get_runtime_alignment"):
            getattr(lib, name).restype = ctypes.c_size_t
        size, alignment = lib.get_runtime_size(), lib.get_runtime_alignment()
        storage = ctypes.create_string_buffer(size + alignment)
        run = (ctypes.addressof(storage) + alignment - 1) & -alignment
        lib.simpler_prepare_run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.POINTER(CallConfig),
            ctypes.POINTER(NativeRunDescriptor),
        ]
        for name in ("simpler_finalize_run", "simpler_launch_run"):
            getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        descriptor = NativeRunDescriptor(generation=1, run_epoch=1)

        def prepare(args):
            return lib.simpler_prepare_run(ctx, run, 0, args.__ptr__(), ctypes.byref(config), ctypes.byref(descriptor))

        def arguments(addr, space, transfer):
            args = ChipStorageTaskArgs()
            args.add_tensor(ChipTensor.make(addr, (1,), DataType.INT32, address_space=space), transfer=transfer)
            args.add_scalar(7)
            args.add_scalar(int(space == AddressSpace.HOST))
            return args

        # A forbidden host input is unreadable. Refusal must precede both its
        # consumption and any metadata publication or device allocation.
        invalid = arguments(1, AddressSpace.HOST, TensorTransfer.H2D)
        committed = lib.committed_device_memory_ctx(ctx)
        copies = faults.copy_call_count()
        assert prepare(invalid) == PTO_RUNTIME_ERR_INVALID_ARGUMENT
        assert lib.committed_device_memory_ctx(ctx) == committed
        assert faults.copy_call_count() == copies
        assert ctypes.string_at(run, 8) == bytes(8)

        # A DEVICE address is opaque to host preparation. No execution follows,
        # so this unmapped address must never be dereferenced by either side.
        device_args = arguments(0x1000, AddressSpace.DEVICE, TensorTransfer.NONE)
        faults.force_current_device(-1)
        assert prepare(device_args) == PTO_RUNTIME_ERR_INVALID_STATE
        assert ctypes.string_at(run, 8) == bytes(8)
        faults.force_current_device(-2)
        # A failed metadata copy unwinds the reservation; the same run storage
        # and slot must immediately accept another preparation.
        faults.arm_copy_failure()
        assert prepare(device_args) != 0
        assert ctypes.string_at(run, 8) == bytes(8)
        for epoch in (1, 2):
            descriptor.run_epoch = epoch
            assert prepare(device_args) == 0
            prepared = True
            assert lib.finalize_device(ctx) != 0
            assert lib.simpler_launch_run(ctx, run) == PTO_RUNTIME_ERR_INVALID_STATE
            faults.force_current_device(-1)
            assert lib.simpler_finalize_run(ctx, run) == PTO_RUNTIME_ERR_INVALID_STATE
            faults.force_current_device(-2)
            assert lib.simpler_finalize_run(ctx, run) == 0
            prepared = False
            assert ctypes.string_at(run, 8) == bytes(8)

        host_value = ctypes.c_int32(11)
        host_args = arguments(ctypes.addressof(host_value), AddressSpace.HOST, TensorTransfer.NONE)
        for expected in (18, 25):
            descriptor.run_epoch += 1
            rc = prepare(host_args)
            if runtime == "host_build_graph":
                assert rc == 0
                prepared = True
                assert host_value.value == expected  # fresh host orchestration on every call
                assert lib.simpler_finalize_run(ctx, run) == 0
                prepared = False
            else:
                assert rc != 0
                assert host_value.value == 11
        assert lib.simpler_kernel_mode_supported(ctx) == 0
    finally:
        faults.force_current_device(-2)
        if prepared:
            assert lib.simpler_finalize_run(ctx, run) == 0
        assert lib.finalize_device(ctx) == 0
        assert lib.committed_device_memory_ctx(ctx) == 0
        lib.destroy_device_context(ctx)
        assert [faults.acl_call_count(i) for i in range(8)] == [0] * 8


@pytest.mark.parametrize(
    ("arch", "runtime"),
    [
        pytest.param(arch, runtime, marks=[pytest.mark.requires_hardware, pytest.mark.platforms([arch])])
        for arch in _ARCHES
        for runtime in _RUNTIMES
    ],
)
def test_borrowed_context_uses_per_call_prepare_and_finalize(arch, runtime, kernel_close_faults, request):
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(kernel_close_faults) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arch, runtime, device, "borrowed_prepare"],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    if sys.argv[4] == "borrowed_prepare":
        _run_borrowed_prepare(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    elif sys.argv[4] == "program_callable_release":
        _run_program_callable_release(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    elif sys.argv[4] == "program_loader_unload":
        _run_program_loader_unload(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    elif sys.argv[4].startswith("program_double_"):
        _run_program_loader_double_failure(
            sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "program_double_acl"
        )
    elif sys.argv[4].startswith("loader_"):
        _run_loader_unload_retry(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4][len("loader_") :])
    else:
        _run_lifecycle_retry(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
