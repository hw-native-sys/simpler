# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real TMR kernel-mode eager and ACLGraph lifecycle/ordering coverage."""

import ctypes
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pytest

from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.graph_cases import run_graph_case
from tests.ut.py.test_kernel_mode_c_api import CallConfig, _binaries, _load

ROOT = Path(__file__).resolve().parents[5]
MODULE = "tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture"
RUNTIME = "tensormap_and_ringbuffer"
SCENARIOS = (
    "cold_synced",
    "close_fail_free",
    "host_submit_failure",
    "device_error_eager",
    "device_error_replay",
    "runtime_error_eager",
    "runtime_error_replay",
    "threaded_cross_stream_error",
    "cold_unsynced",
    "warm",
    "multi_callable",
    "cross_stream",
    "prepare_again",
    "prepare_after_capture",
    "prepare_in_capture",
    "stream_query_error",
    "stream_busy",
    "blocked_same",
    "replay_stream",
    "two_graphs",
    "fresh_inputs",
    "chain",
    "feedback_batch",
    "eager_replay",
    "init_fail_handshake",
    "eager_batch",
    "eager_multi_callable",
    "eager_rejections",
    "graph_recreate",
    "long_chain",
    "tmr_dag",
    "eager_dag",
)


@pytest.fixture(scope="module")
def capture_observer(tmp_path_factory):
    sdk = Path(os.environ["ASCEND_HOME_PATH"])
    include_dirs = [
        sdk / "include",
        sdk / f"{platform.machine()}-linux/pkg_inc",
        sdk / f"{platform.machine()}-linux/pkg_inc/runtime",
        sdk / f"{platform.machine()}-linux/pkg_inc/runtime/runtime",
        sdk / f"{platform.machine()}-linux/pkg_inc/profiling",
        ROOT / "src/a2a3/platform/include",
        ROOT / "src/common/platform/include",
        ROOT / "src/common",
    ]
    output = tmp_path_factory.mktemp("cold-capture-observer") / "observer.so"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            *[f"-I{path}" for path in include_dirs],
            str(Path(__file__).with_name("kernel_capture_observer.cpp")),
            str(Path(__file__).with_name("prepare_gate.cpp")),
            "-pthread",
            f"-L{sdk / 'lib64'}",
            f"-Wl,-rpath,{sdk / 'lib64'}",
            "-lascendcl",
            "-ldl",
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime(RUNTIME)
@pytest.mark.device_count(1)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_tmr_kernel_mode(st_platform, st_device_ids, scenario, capture_observer):
    _binaries(st_platform, RUNTIME)
    device = str(st_device_ids[0])
    artifacts = ROOT / "outputs" / "kernel_mode"
    artifacts.mkdir(parents=True, exist_ok=True)
    # Forked pytest workers can recreate basetemp between cases.
    tmp_path = Path(tempfile.mkdtemp(prefix=scenario + "-", dir=artifacts))
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(capture_observer) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    logs = tmp_path / "ascend"
    logs.mkdir()
    env["ASCEND_PROCESS_LOG_PATH"] = str(logs)
    with (tmp_path / "run.log").open("w") as log:
        try:
            result = subprocess.run(
                [sys.executable, "-m", MODULE + ".test_kernel_mode_capture", device, scenario, str(tmp_path)],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"capture scenario timed out; artifacts: {tmp_path}")
    output = (tmp_path / "run.log").read_text()
    assert result.returncode == 0, f"{output}\nArtifacts: {tmp_path}"
    if "_error_" in scenario:
        assert f"PASS {scenario} caller_error=1 failure_reported=1 context_error=1" in output
    elif scenario == "threaded_cross_stream_error":
        assert (
            "PASS threaded_cross_stream_error caller_error=1 failure_reported=1 host_reject=1 context_error=1" in output
        )
    elif scenario == "host_submit_failure":
        assert "PASS host_submit_failure host_status=-4334 retained=1" in output
    elif scenario == "close_fail_free":
        assert "PASS close_fail_free retained_then_retried=1" in output
    elif scenario == "init_fail_handshake":
        assert f"PASS {scenario} rejected_after_failure=1 forbidden_sync=0" in output
    elif scenario.startswith("eager_") and scenario != "eager_replay":
        assert f"PASS {scenario} eager=100 forbidden_sync=0" in output
    else:
        assert f"PASS {scenario} replays=100 forbidden_sync=0" in output


def _build_callable(build_dir, alternate=False, dag=False, execution_error=False):
    from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable  # noqa: PLC0415

    from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415
    from simpler_setup.kernel_compiler import KernelCompiler  # noqa: PLC0415
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    compiler = KernelCompiler("a2a3")
    build_dir.mkdir()
    source = (
        Path(__file__).with_name("kernel_capture_alternate.cpp")
        if alternate
        else ROOT / "tests/ut/py/kernel_eager_orchestration.cpp"
    )
    if dag:
        source = Path(__file__).with_name("kernel_tmr_chain.cpp")
    if execution_error:
        source = Path(__file__).with_name("kernel_execution_error.cpp")
    orchestration = compiler.compile_orchestration(RUNTIME, str(source), build_dir=str(build_dir))
    incore = compiler.compile_incore(
        str(ROOT / "examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels/aiv/kernel_add_scalar.cpp"),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs(RUNTIME),
        build_dir=str(build_dir),
    )
    signature = [ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR]
    child = CoreCallable.build(signature=signature, binary=extract_text_section(incore))
    return ChipCallable.build(
        signature=signature,
        func_name="kernel_execution_error"
        if execution_error
        else "kernel_tmr_chain"
        if dag
        else ("kernel_capture_alternate" if alternate else "kernel_eager_orchestration"),
        binary=orchestration,
        children=[(0, child)],
    )


def _bind_acl(lib):
    signatures = {
        "aclInit": [ctypes.c_char_p],
        "aclFinalize": [],
        "aclrtSynchronizeDevice": [],
        "aclrtSetDevice": [ctypes.c_int],
        "aclrtGetCurrentContext": [ctypes.POINTER(ctypes.c_void_p)],
        "aclrtSetCurrentContext": [ctypes.c_void_p],
        "aclrtResetDevice": [ctypes.c_int],
        "aclrtCreateStream": [ctypes.POINTER(ctypes.c_void_p)],
        "aclrtDestroyStream": [ctypes.c_void_p],
        "aclrtCreateEvent": [ctypes.POINTER(ctypes.c_void_p)],
        "aclrtRecordEvent": [ctypes.c_void_p, ctypes.c_void_p],
        "aclrtSynchronizeStreamWithTimeout": [ctypes.c_void_p, ctypes.c_int32],
        "aclrtMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int],
        "aclrtFree": [ctypes.c_void_p],
        "aclrtMemcpy": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
    }
    for name, arguments in signatures.items():
        getattr(lib, name).argtypes = arguments
        getattr(lib, name).restype = ctypes.c_int


def _config():
    config = CallConfig()
    for ring in range(4):
        config.runtime_env[ring] = 64
        config.runtime_env[ring + 4] = 1 << 20
        config.runtime_env[ring + 8] = 1024
    return config


def _bind_observer_guards(observer):
    for name in ("arm", "release", "blocked", "finish"):
        function = getattr(observer, "capture_gate_" + name)
        function.argtypes = []
        function.restype = None if name in ("arm", "release") else ctypes.c_int
    observer.capture_observer_guard_sync.argtypes = [ctypes.c_int]
    observer.capture_observer_guard_sync.restype = None
    observer.capture_observer_prepare_scope.argtypes = [ctypes.c_int]
    observer.capture_observer_prepare_scope.restype = None
    observer.capture_observer_invocation_scope.argtypes = [ctypes.c_int]
    observer.capture_observer_invocation_scope.restype = None
    observer.capture_observer_sync_calls.argtypes = []
    observer.capture_observer_sync_calls.restype = ctypes.c_uint64
    observer.capture_observer_override_query.argtypes = [ctypes.c_int]
    observer.capture_observer_override_query.restype = None
    observer.capture_observer_fail_prepare.argtypes = [ctypes.c_int]
    observer.capture_observer_fail_prepare.restype = None
    observer.capture_observer_fail_next_invocation.argtypes = []
    observer.capture_observer_fail_next_invocation.restype = None
    observer.capture_observer_failure_reported.argtypes = []
    observer.capture_observer_failure_reported.restype = ctypes.c_int
    for name in ("query_calls", "total_queries", "waits", "records", "clears", "prepare_failures"):
        function = getattr(observer, "capture_observer_" + name)
        function.argtypes = []
        function.restype = ctypes.c_uint64


def _submission_counts(observer):
    return tuple(
        getattr(observer, "capture_observer_" + name)()
        for name in (
            "waits",
            "records",
            "clears",
            "core_launches",
            "cpu_launches",
        )
    )


def _check_steady_launch(observer, launch, *arguments, **keywords):
    before = observer.capture_observer_total_queries()
    launch(*arguments, **keywords)
    assert observer.capture_observer_total_queries() == before, "steady same-caller launch queried an old event"


def _close(lib, ctx, allocations, streams, device, graphs=()):
    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _check  # noqa: PLC0415

    for graph in graphs:
        _check(lib.aclmdlRIDestroy(graph), "destroy graph")
    # All caller work is drained and graphs are destroyed before entering this
    # helper. Close waits only for its own metadata revocation.
    _check(lib.finalize_device(ctx), "finalize context (including revocation)")
    assert lib.committed_device_memory_ctx(ctx) == 0
    lib.destroy_device_context(ctx)
    for address in reversed(allocations):
        _check(lib.aclrtFree(address), "free tensor")
    for stream in streams:
        _check(lib.aclrtDestroyStream(stream), "destroy stream")
    _check(lib.aclrtResetDevice(device), "reset")
    _check(lib.aclFinalize(), "finalize ACL")


def _check_close_failure(context):
    lib, observer, ctx = context.lib, context.observer, context.handle
    observer.capture_observer_fail_large_free.argtypes = [ctypes.c_int]
    observer.capture_observer_failed_frees.restype = ctypes.c_uint64
    observer.capture_observer_fail_large_free(1)
    status = lib.finalize_device(ctx)
    assert observer.capture_observer_failed_frees() > 0
    assert status == -4334, f"failed free was lost: close={status}"
    retained = lib.committed_device_memory_ctx(ctx)
    assert retained > 0, "failed device free disappeared from the allocator ledger"
    assert lib.finalize_device(ctx) == -4334
    assert lib.committed_device_memory_ctx(ctx) == retained
    observer.capture_observer_fail_large_free(0)


def _fail():
    import traceback  # noqa: PLC0415

    traceback.print_exc()
    sys.stderr.flush()
    # Failed enqueue/capture does not establish graph-visible resource quiescence.
    os._exit(1)


def _check_init_failure(observer, lib, ctx, prepare, launch):
    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _check  # noqa: PLC0415

    # A context whose init failed keeps what it had committed and accepts no
    # further work; only close reclaims it.
    retained = lib.committed_device_memory_ctx(ctx)
    prepare(0, expected=-1003)
    submission = _submission_counts(observer)
    launch(0, expected=-1003)
    assert _submission_counts(observer) == submission
    assert observer.capture_observer_prepare_failures() == 1
    assert lib.committed_device_memory_ctx(ctx) == retained
    # Previously enqueued initialization still belongs to the refused context.
    _check(lib.aclrtSynchronizeDevice(), "external drain before refused-context close")
    assert lib.committed_device_memory_ctx(ctx) == retained


def _seed_tensors(io, chips):
    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _COUNT  # noqa: PLC0415

    pairs = [(io.allocate(), io.allocate()) for _ in chips]
    initial = [float(i % 127) for i in range(_COUNT)]
    for source, destination in pairs:
        io.write(source, initial)
        io.write(destination, [-999.0] * _COUNT)
    counter = io.allocate()
    io.write(counter, [0.0] * _COUNT)
    return pairs, initial, counter


def _initialize(device, scenario, build_dir):
    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import (  # noqa: PLC0415
        _bind_capture_functions,
        _check,
    )

    chips = [
        _build_callable(
            build_dir / "callable-a",
            dag=scenario in ("tmr_dag", "eager_dag"),
            execution_error=scenario.startswith("runtime_error_"),
        )
    ]
    if scenario in (
        "multi_callable",
        "prepare_again",
        "prepare_after_capture",
        "prepare_in_capture",
        "eager_multi_callable",
    ):
        chips.append(_build_callable(build_dir / "callable-b", alternate=True))
    lib = _load("a2a3", "onboard", RUNTIME)
    _bind_acl(lib)
    _check(lib.aclInit(None), "acl init")
    _check(lib.aclrtSetDevice(device), "device")
    streams = [ctypes.c_void_p(), ctypes.c_void_p()]
    for stream in streams:
        _check(lib.aclrtCreateStream(ctypes.byref(stream)), "create stream")
    caller = streams[0]
    ctx = lib.create_device_context()
    assert ctx
    config = _config()
    aicpu, aicore, dispatcher = _binaries("a2a3", RUNTIME)
    observer = _bind_capture_functions(lib)
    _bind_observer_guards(observer)
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
        71,
    )
    if scenario == "init_fail_handshake":
        # The context handshake is init's device work now, so a refused AICPU
        # launch is init's own status.
        observer.capture_observer_fail_prepare(1)
        observer.capture_observer_prepare_scope(1)
        try:
            rc = lib.simpler_kernel_mode_init(*init_args)
        finally:
            observer.capture_observer_prepare_scope(0)
        assert rc == -4333, f"kernel init rc={rc}, expected the injected -4333"
        assert observer.capture_observer_prepare_failures() == 1
        return chips, lib, streams, caller, ctx, observer
    _check(lib.simpler_kernel_mode_init(*init_args), "kernel init")
    return chips, lib, streams, caller, ctx, observer


def _prepare_initial(scenario, observer, prepare, launch):
    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _check  # noqa: PLC0415

    blocked = scenario == "blocked_same"
    prepare(0)
    if blocked:
        # The gate holds the first launch's AICPU work, so its serial tail stays
        # incomplete. A second launch on that same caller stream is ordered by
        # the stream's own FIFO, so it neither queries the tail nor waits for it
        # — the query belongs to the different-stream path alone.
        observer.capture_gate_arm()
        launch(0)
        assert observer.capture_gate_blocked() == 1
        before = observer.capture_observer_total_queries()
        launch(0)
        assert observer.capture_observer_total_queries() == before
        assert observer.capture_gate_blocked() == 1, "a same-stream launch waited for the one before it"
        observer.capture_gate_release()
        _check(observer.capture_gate_finish(), "finish launch gate")
    return blocked


def _verify_values(io, pairs, initial, counter, scenario):
    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _COUNT  # noqa: PLC0415

    for cid, (_, destination) in enumerate(pairs):
        io.verify(destination, [value + (1.25 if cid == 0 else -1.25) for value in initial])
    io.verify(counter, [187.5 if scenario == "two_graphs" else 125.0] * _COUNT)


@dataclass
class _Context:
    lib: ctypes.CDLL
    handle: int
    streams: list
    caller: ctypes.c_void_p
    observer: ctypes.CDLL


def _configure(context, scenario, prepare, launch, sync, io, pairs, initial):
    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _check  # noqa: PLC0415

    lib, observer, streams, caller = context.lib, context.observer, context.streams, context.caller
    if _prepare_initial(scenario, observer, prepare, launch):
        sync(caller)
        io.verify(pairs[0][1], [value + 1.25 for value in initial])
    if scenario in ("multi_callable", "eager_multi_callable"):
        prepare(1)
    if scenario != "cold_unsynced":
        _check(lib.aclrtSynchronizeDevice(), "external preparation drain")
    if scenario == "cross_stream":
        launch(0)
        sync(caller)
        caller = streams[1]
    if scenario in ("stream_query_error", "stream_busy"):
        if scenario == "stream_busy":
            observer.capture_gate_arm()
        launch(0, stream=caller)
        if scenario == "stream_query_error":
            sync(caller)
            observer.capture_observer_override_query(1)
        else:
            assert observer.capture_gate_blocked() == 1
        caller = streams[1]
        before = _submission_counts(observer)
        launch(0, stream=caller, expected=-4332 if scenario == "stream_query_error" else -1002)
        assert _submission_counts(observer) == before
        if scenario == "stream_busy":
            assert observer.capture_gate_blocked() == 1
            observer.capture_gate_release()
            _check(observer.capture_gate_finish(), "finish registration gate")
        sync(streams[0])
        launch(0, stream=caller)
        sync(caller)
    if scenario in ("warm", "prepare_again"):
        launch(0, stream=caller)
        sync(caller)
    if scenario == "prepare_again":
        prepare(1)
        _check(lib.aclrtSynchronizeDevice(), "external preparation drain")
    return caller


def _execute_eager(context, scenario, guarded, io, launch, sync, pairs, initial, cid):
    lib, ctx, caller, observer = context.lib, context.handle, context.caller, context.observer
    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.eager_cases import (  # noqa: PLC0415
        run_eager_case,
    )

    def reject(cid, rejected_args, null_stream, expected):
        before = _submission_counts(observer)
        observer.capture_observer_invocation_scope(1)
        try:
            guarded(
                lib.simpler_kernel_mode_launch,
                ctx,
                cid,
                rejected_args.__ptr__(),
                None if null_stream else caller,
                expected=expected,
            )
        finally:
            observer.capture_observer_invocation_scope(0)
        assert _submission_counts(observer) == before

    run_eager_case(
        scenario,
        io,
        launch,
        lambda: sync(caller),
        pairs,
        initial,
        reject,
        cid,
    )


def _guarded(observer, operation, *arguments, expected=0):
    """Run one launch with every synchronize refused.

    Launch is pure enqueue: a wait inside it is what an ACLGraph capture cannot
    contain, so the observer refuses each one rather than only counting it —
    a refusal surfaces as the operation's own status instead of a later hang.
    """
    observer.capture_observer_guard_sync(1)
    try:
        result = operation(*arguments)
    finally:
        observer.capture_observer_guard_sync(0)
    assert observer.capture_observer_sync_calls() == 0, "launch performed an internal sync"
    assert result == expected, f"guarded native operation rc={result}, expected={expected}"


def _prepared(observer, operation, *arguments, expected=0):
    """Run one registration with every synchronize refused.

    Registration is callable inside a caller's capture, so it may wait for
    nothing: it enqueues its AICPU work on the context's own stream, whose FIFO
    orders it ahead of every launch, and a device-side load failure is left for
    the caller's own warmup to surface. The scope also arms the registration
    fault injection, which applies to this call alone.
    """
    observer.capture_observer_prepare_scope(1)
    observer.capture_observer_guard_sync(1)
    try:
        result = operation(*arguments)
    finally:
        observer.capture_observer_guard_sync(0)
        observer.capture_observer_prepare_scope(0)
    assert observer.capture_observer_sync_calls() == 0, "prepare performed an internal sync"
    assert result == expected, f"prepared native operation rc={result}, expected={expected}"


def _replay(context, graph, stream=None):
    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _check  # noqa: PLC0415

    observer = context.observer
    before = observer.capture_observer_total_queries()
    observer.capture_observer_invocation_scope(1)
    try:
        _check(context.lib.aclmdlRIExecuteAsync(graph, context.caller if stream is None else stream), "replay")
    finally:
        observer.capture_observer_invocation_scope(0)
    assert observer.capture_observer_total_queries() == before


def _working_event_on_stream(lib, stream):
    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import _check  # noqa: PLC0415

    event = ctypes.c_void_p()
    _check(lib.aclrtCreateEvent(ctypes.byref(event)), "create independent stream event")
    _check(lib.aclrtRecordEvent(event, stream), "record independent stream event")
    _check(lib.aclrtSynchronizeStreamWithTimeout(stream, 10000), "independent stream before failure")
    return event


def _check_host_submission_failure(scenario, observer, prepare, launch):
    if scenario != "host_submit_failure":
        return
    prepare(0)
    observer.capture_observer_fail_next_invocation()
    started = time.monotonic()
    launch(0, expected=-4334)
    assert time.monotonic() - started < 1
    assert observer.capture_observer_core_launches() == 1
    assert observer.capture_observer_cpu_launches() == 0
    before = _submission_counts(observer)
    launch(0, expected=-1003)
    assert _submission_counts(observer) == before
    print("PASS host_submit_failure host_status=-4334 retained=1", flush=True)
    # AICore may be running without an AICPU task: retain until process recovery.
    os._exit(0)


def _check_device_failure(context, scenario, launch, record_nodes, replay):
    observer = context.observer
    independent_event = _working_event_on_stream(context.lib, context.streams[1])
    if scenario.startswith("device_error_"):
        observer.capture_observer_corrupt_next_invocation()
    if scenario.endswith("replay"):
        graph = record_nodes([(0, None, 1.25)])
        replay(graph)
    else:
        launch(0)
    started = time.monotonic()
    status = context.lib.aclrtSynchronizeStreamWithTimeout(context.caller, 10000)
    assert status != 0, "hidden AICPU error was not propagated to caller"
    assert time.monotonic() - started < 9, "failure only surfaced through timeout"
    elapsed_ms = (time.monotonic() - started) * 1000
    reported = observer.capture_observer_failure_reported()
    assert reported == 0, f"failed round diagnostic rc={reported}"
    assert observer.capture_observer_cpu_launches() == 1
    assert observer.capture_observer_core_launches() == 1
    independent_status = context.lib.aclrtRecordEvent(independent_event, context.streams[1])
    assert independent_status != 0, "unrelated stream accepted new work after context failure"
    print(
        f"PASS {scenario} caller_error=1 failure_reported=1 context_error=1 "
        f"caller_status={status} independent_status={independent_status} elapsed_ms={elapsed_ms:.3f}",
        flush=True,
    )
    # Failed streams/graphs retain their allocations until process teardown.
    # Stop-on-failure is not evidence that already running AICores have stopped.
    os._exit(0)


def _check_threaded_cross_stream_failure(
    context, guarded, prepare, launch, callable_ids, source, destination, io, initial
):
    from simpler.task_interface import ChipStorageTaskArgs, ChipTensor, DataType  # noqa: PLC0415

    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import (  # noqa: PLC0415
        _COUNT,
        _check,
    )

    lib, observer = context.lib, context.observer
    current = ctypes.c_void_p()
    _check(lib.aclrtGetCurrentContext(ctypes.byref(current)), "get current context")
    assert current.value

    def submit_on_second_caller(expected):
        outcome = []

        def submit():
            try:
                _check(lib.aclrtSetCurrentContext(current), "set worker context")
                args = ChipStorageTaskArgs()
                args.add_tensor(ChipTensor.make(source.value, (_COUNT,), DataType.FLOAT32, child_memory=True))
                args.add_tensor(ChipTensor.make(destination.value, (_COUNT,), DataType.FLOAT32, child_memory=True))
                args.add_scalar(ctypes.c_float(1.25))
                observer.capture_observer_invocation_scope(1)
                try:
                    guarded(
                        lib.simpler_kernel_mode_launch,
                        context.handle,
                        callable_id,
                        args.__ptr__(),
                        context.streams[1],
                        expected=expected,
                    )
                finally:
                    observer.capture_observer_invocation_scope(0)
                outcome.append(None)
            except BaseException as error:
                outcome.append(error)

        worker = threading.Thread(target=submit, daemon=True)
        worker.start()
        worker.join(timeout=10)
        assert not worker.is_alive(), "second caller Host submission stalled"
        assert outcome, "second caller did not report a result"
        if outcome[0] is not None:
            raise outcome[0]

    prepare(0)
    callable_id = callable_ids[0]
    observer.capture_gate_arm()
    launch(0)
    assert observer.capture_gate_blocked() == 1
    before = _submission_counts(observer)
    submit_on_second_caller(-1002)
    assert _submission_counts(observer) == before, "rejected caller submitted device work"
    observer.capture_gate_release()
    _check(observer.capture_gate_finish(), "finish first caller gate")
    _check(lib.aclrtSynchronizeStreamWithTimeout(context.streams[0], 10000), "first caller")
    io.verify(destination, [value + 1.25 for value in initial])

    first_caller_event = _working_event_on_stream(lib, context.streams[0])
    observer.capture_observer_corrupt_next_invocation()
    submit_on_second_caller(0)
    started = time.monotonic()
    status = lib.aclrtSynchronizeStreamWithTimeout(context.streams[1], 10000)
    elapsed_ms = (time.monotonic() - started) * 1000
    assert status != 0, "hidden AICPU error was not propagated to second caller"
    assert elapsed_ms < 9000, "second caller failure only surfaced through timeout"
    assert observer.capture_observer_failure_reported() == 0
    assert observer.capture_observer_cpu_launches() == 2
    assert observer.capture_observer_core_launches() == 2
    first_caller_status = lib.aclrtRecordEvent(first_caller_event, context.streams[0])
    assert first_caller_status != 0, "first caller accepted work after second caller's context failure"
    print(
        f"PASS threaded_cross_stream_error caller_error=1 failure_reported=1 host_reject=1 "
        f"context_error=1 sync_status={status} first_caller_status={first_caller_status} "
        f"latency_ms={elapsed_ms:.1f}",
        flush=True,
    )
    os._exit(0)


def _run_close_failure(context, prepare, launch, sync):
    prepare(0)
    launch(0)
    sync(context.caller)
    _check_close_failure(context)


@dataclass
class _GraphCase:
    """Everything a graph-recording scenario needs beyond its `_Context`."""

    io: object
    graphs: list
    record_nodes: object
    replay: object
    launch: object
    sync_caller: object
    # Read after the case runs: `launch` advances the caller's counter, so a
    # snapshot taken before would under-count.
    host_launches_now: object
    committed: int
    allocations: list
    device: int


def _run_graph_scenario(scenario, context, case):
    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import (  # noqa: PLC0415
        _check,
        _check_resident,
    )

    lib, ctx = context.lib, context.handle

    def destroy(graph):
        _check(lib.aclmdlRIDestroy(graph), "destroy graph")
        case.graphs.remove(graph)

    run_graph_case(scenario, case.io, case.record_nodes, case.replay, case.launch, case.sync_caller, destroy)
    host_launches = case.host_launches_now()
    _check_resident(context.observer, host_launches)
    assert lib.committed_device_memory_ctx(ctx) == case.committed
    _close(lib, ctx, case.allocations, context.streams, case.device, case.graphs)
    print(f"PASS {scenario} replays=100 forbidden_sync=0 host_launches={host_launches}", flush=True)


def _run(device, scenario, build_dir):
    from simpler.task_interface import ChipStorageTaskArgs, ChipTensor, DataType  # noqa: PLC0415

    from tests.st.a2a3.tensormap_and_ringbuffer.kernel_mode_capture.kernel_capture_values import (  # noqa: PLC0415
        _COUNT,
        _check,
        _check_resident,
        _TensorIO,
    )

    chips, lib, streams, caller, ctx, observer = _initialize(device, scenario, build_dir)
    context = _Context(lib, ctx, streams, caller, observer)
    allocations = []
    io = _TensorIO(lib, allocations)
    pairs, initial, counter = _seed_tensors(io, chips)
    args = ChipStorageTaskArgs()
    host_launches = 0
    graphs = []
    callable_ids = {}

    guarded = partial(_guarded, observer)

    def prepare(cid, expected=0):
        chip = chips[cid]
        minted = ctypes.c_int32(99)
        _prepared(
            observer,
            lib.simpler_kernel_mode_prepare_callable,
            ctx,
            chip.buffer_ptr(),
            chip.buffer_size(),
            ctypes.byref(minted),
            expected=expected,
        )
        if expected == 0:
            assert minted.value >= 0 and minted.value not in callable_ids.values()
            callable_ids[cid] = minted.value
        else:
            assert minted.value == -1

    def sync(stream):
        _check(lib.aclrtSynchronizeStreamWithTimeout(stream, 10000), "external sync")

    def launch(cid, tensors=None, expected=0, scalar=1.25, stream=None):
        nonlocal host_launches
        source, destination = pairs[cid] if tensors is None else tensors
        args.clear()
        args.add_tensor(ChipTensor.make(source.value, (_COUNT,), DataType.FLOAT32, child_memory=True))
        args.add_tensor(ChipTensor.make(destination.value, (_COUNT,), DataType.FLOAT32, child_memory=True))
        args.add_scalar(ctypes.c_float(scalar / 16 if scenario in ("tmr_dag", "eager_dag") else scalar))
        observer.capture_observer_invocation_scope(1)
        try:
            guarded(
                lib.simpler_kernel_mode_launch,
                ctx,
                callable_ids.get(cid, cid),
                args.__ptr__(),
                caller if stream is None else stream,
                expected=expected,
            )
        finally:
            observer.capture_observer_invocation_scope(0)
        args.clear()
        host_launches += int(expected == 0)

    def record_nodes(nodes, prepare_cid=None):
        graph = ctypes.c_void_p()
        _check(lib.aclmdlRICaptureBegin(caller, 0), "capture begin")
        if prepare_cid is not None:
            prepare(prepare_cid)
        for index, (cid, tensors, scalar) in enumerate(nodes):
            if index == 0:
                launch(cid, tensors, scalar=scalar)
            else:
                _check_steady_launch(observer, launch, cid, tensors, scalar=scalar)
        _check(lib.aclmdlRICaptureEnd(caller, ctypes.byref(graph)), "capture end")
        assert graph.value
        graphs.append(graph)
        return graph

    def record(cids, increment=1.25, prepare_cid=None):
        return record_nodes(
            [(cid, None, 1.25) for cid in cids] + [(0, (counter, counter), increment)], prepare_cid=prepare_cid
        )

    replay = partial(_replay, context)

    try:
        _check_host_submission_failure(scenario, observer, prepare, launch)
        if scenario.startswith(("device_error_", "runtime_error_")):
            prepare(0)
            _check_device_failure(context, scenario, launch, record_nodes, replay)
        if scenario == "threaded_cross_stream_error":
            _check_threaded_cross_stream_failure(
                context, guarded, prepare, launch, callable_ids, pairs[0][0], pairs[0][1], io, initial
            )
        if scenario == "close_fail_free":
            _run_close_failure(context, prepare, launch, sync)
            _close(lib, ctx, allocations, streams, device)
            print("PASS close_fail_free retained_then_retried=1", flush=True)
            return
        if scenario == "init_fail_handshake":
            _check_init_failure(observer, lib, ctx, prepare, launch)
            _close(lib, ctx, allocations, streams, device)
            print(f"PASS {scenario} rejected_after_failure=1 forbidden_sync=0", flush=True)
            return
        caller = _configure(context, scenario, prepare, launch, sync, io, pairs, initial)
        context.caller = caller
        committed = lib.committed_device_memory_ctx(ctx)
        if scenario.startswith("eager_") and scenario != "eager_replay":
            _execute_eager(
                context,
                scenario,
                guarded,
                io,
                launch,
                sync,
                pairs,
                initial,
                callable_ids[0],
            )
            _check_resident(observer, host_launches)
            assert lib.committed_device_memory_ctx(ctx) == committed
            _close(lib, ctx, allocations, streams, device)
            print(f"PASS {scenario} eager=100 forbidden_sync=0", flush=True)
            return
        if scenario in ("fresh_inputs", "chain", "feedback_batch", "eager_replay", "graph_recreate", "long_chain"):
            _run_graph_scenario(
                scenario,
                context,
                _GraphCase(
                    io,
                    graphs,
                    record_nodes,
                    replay,
                    launch,
                    lambda: sync(caller),
                    lambda: host_launches,
                    committed,
                    allocations,
                    device,
                ),
            )
            return
        if scenario == "prepare_in_capture":
            # Registration inside the capture publishes context state rather
            # than a graph node, so the replays below leave it where it is.
            record([0, 1, 0], prepare_cid=1)
            committed = lib.committed_device_memory_ctx(ctx)
        else:
            record([0, 1, 0] if len(callable_ids) == 2 else [0])
        if scenario == "prepare_after_capture":
            prepare(1)
            record([1])
            committed = lib.committed_device_memory_ctx(ctx)
        if scenario == "two_graphs":
            record([0], increment=2.5)
        _check_resident(observer, host_launches)
        replay_stream = streams[1] if scenario == "replay_stream" else caller
        for iteration in range(100):
            replay(graphs[iteration % len(graphs)], replay_stream)
        sync(replay_stream)
        _verify_values(io, pairs, initial, counter, scenario)
        _check_resident(observer, host_launches)
        assert lib.committed_device_memory_ctx(ctx) == committed
        _close(lib, ctx, allocations, streams, device, graphs)
    except BaseException:
        _fail()
    print(f"PASS {scenario} replays=100 forbidden_sync=0 host_launches={host_launches}", flush=True)


if __name__ == "__main__":
    _run(int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
