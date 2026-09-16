# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""UT for the Python kernel-mode surface: ChipWorker and the L2 Worker over it.

Two layers, and the split between them is what each part tests. ``ChipWorker``
is the native wrapper; ``Worker(level=2, execution_mode="kernel")`` is the public
object PyPTO holds, which fixes the mode at construction, dispatches init and
close to the kernel entries, and refuses the program surface. The Worker-level
mode and argument contract resolves before any device exists, so those cases run
anywhere; everything that binds a runtime needs a device or the simulator.

The hardware cases are the Python twin of
tests/ut/cpp/hardware/test_kernel_mode_entry.cpp, and the
inverse of test_platform_comm.py's contract: there ChipWorker owns ACL bring-up
and stream lifetime internally, here the *caller* owns both. That inversion is
what kernel mode is, so the test does its own device bind and stream creation
through ``_acl_bind_device`` / ``_acl_create_stream``. kernel_init runs on that
already-bound device and takes no stream; the stream's integer address goes
only to kernel_launch. In production that integer comes from the framework
instead — torch_npu.npu.current_stream().npu_stream.

tensormap_and_ringbuffer implements kernel mode, so its cases claim the borrowed
stream and check that a second claim on the same device is refused.
host_build_graph reports no kernel capability; the refusal case loads it, and
reaching the refusal is the assertion: it proves the Python call arrived at the
C ABI rather than being rejected on the way.

Each case runs in a forked subprocess: kernel_init binds a runtime library into
the process and the ACL device bind is per-thread, so a fresh process per case
keeps one case's state out of the next.

The ``runtime`` marker is what makes conftest's resource phase dispatch these
rather than deselect them, so it is load-bearing rather than descriptive — it
names the runtime whose kernel path the cases exercise. The refusal case also
loads host_build_graph, which every a2a3 build produces.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback

import pytest


def _run_case(case: str, device_id: int, platform: str, queue) -> None:
    """Subprocess body: stand up the caller's device + stream, then drive the
    kernel-mode surface and report what came back."""
    result: dict[str, object] = {"case": case, "stage": "start", "ok": False}
    stream = 0
    try:
        import _task_interface as native
        from simpler.task_interface import CallConfig, ChipWorker

        from simpler_setup.runtime_builder import RuntimeBuilder

        builder = RuntimeBuilder(platform=platform)
        bins = builder.get_binaries("tensormap_and_ringbuffer", build=False)

        # The caller's device and stream. simpler must not create, reset or
        # destroy any of this.
        native._acl_bind_device(device_id)
        stream = native._acl_create_stream()
        result["stream_nonzero"] = bool(stream)
        result["stage"] = "borrowed"

        worker = ChipWorker()
        config = CallConfig()

        if case == "init_refused":
            # host_build_graph reaching simpler_kernel_mode_init and being told
            # UNSUPPORTED is the pass: a call that never arrived would raise
            # something else.
            unsupported_bins = builder.get_binaries("host_build_graph", build=False)
            with pytest.raises(native.UnsupportedRuntimeOperation) as excinfo:
                worker.kernel_init(device_id, unsupported_bins, config)
            result["error"] = str(excinfo.value)
            result["code"] = excinfo.value.code
            result["reached_abi"] = excinfo.value.code == native.PTO_RUNTIME_ERR_UNSUPPORTED
            result["initialized_after"] = bool(worker._impl.initialized)
            worker.finalize()
            result["ok"] = bool(result["reached_abi"]) and not result["initialized_after"]

        elif case == "init_claims_borrowed_stream":
            # One kernel context per device and runtime: the first claim holds,
            # and a second worker is refused without disturbing the owner.
            worker.kernel_init(device_id, bins, config)
            result["stage"] = "kernel_init"
            result["initialized"] = bool(worker._impl.initialized)
            result["kernel_supported"] = bool(worker.kernel_mode_supported)
            refused = ChipWorker()
            try:
                refused.kernel_init(device_id, bins, config)
                result["second_claim_refused"] = False
            except RuntimeError as exc:
                result["second_claim_refused"] = True
                result["second_claim_error"] = str(exc)
            result["refused_initialized"] = bool(refused._impl.initialized)
            refused.finalize()
            result["owner_still_initialized"] = bool(worker._impl.initialized)
            worker.finalize()
            result["ok"] = (
                result["initialized"]
                and result["kernel_supported"]
                and result["second_claim_refused"]
                and not result["refused_initialized"]
                and result["owner_still_initialized"]
            )

        elif case == "null_stream_rejected":
            # kernel_launch is the only entry that takes a stream, so this is
            # where a null one has to be named. Rejecting at the Python boundary
            # names the argument instead of surfacing a bare ABI code.
            with pytest.raises(ValueError) as excinfo:
                worker.kernel_launch(0, None, 0)
            result["error"] = str(excinfo.value)
            result["ok"] = "caller_stream" in str(excinfo.value)

        elif case == "generation_is_unique":
            first = native._ChipWorker.next_kernel_context_generation()
            second = native._ChipWorker.next_kernel_context_generation()
            result["first"], result["second"] = first, second
            result["ok"] = first != 0 and second > first

        elif case == "uninitialized_surface_refuses":
            # No runtime is bound: supported() reports False, and the entries that
            # need a context refuse with INVALID_STATE before reaching the runtime.
            result["supported"] = worker.kernel_mode_supported
            chip_callable = native.ChipCallable.build(signature=[], func_name="probe", binary=b"\x00", children=[])
            codes = {}
            for label, fn in (
                ("prepare", lambda: worker.kernel_prepare_callable(chip_callable)),
                ("launch", lambda: worker.kernel_launch(0, native.ChipStorageTaskArgs(), stream)),
            ):
                try:
                    fn()
                    codes[label] = None
                except native.ChipWorkerError as exc:
                    codes[label] = exc.code
                    result[f"err_{label}"] = str(exc)
            result["codes"] = codes
            expected = native.PTO_RUNTIME_ERR_INVALID_STATE
            result["ok"] = (
                result["supported"] is False
                and codes == {"prepare": expected, "launch": expected}
                and worker._kernel_callables == {}
            )

        elif case == "program_init_still_works":
            # The program path must be unchanged by the kernel additions, and
            # the two identities must stay mutually exclusive on one worker.
            worker.init(device_id=device_id, bins=bins)
            result["stage"] = "program_init"
            result["initialized"] = bool(worker._impl.initialized)
            result["kernel_supported"] = bool(worker.kernel_mode_supported)
            try:
                worker.kernel_init(device_id, bins, config)
                result["second_init_refused"] = False
            except RuntimeError:
                result["second_init_refused"] = True
            worker.finalize()
            result["ok"] = result["initialized"] and result["kernel_supported"] and result["second_init_refused"]

        else:
            raise AssertionError(f"unknown case {case}")

        result["stage"] = "done"
    except BaseException as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        # Destroying the caller's stream after kernel_init and finalize() is the
        # evidence that neither reset the device nor finalized ACL, either of
        # which would invalidate a stream simpler was never given. Swallowing its
        # failure here would let such a teardown still report ok.
        if stream:
            try:
                import _task_interface as native

                native._acl_destroy_stream(stream)
                result["stream_destroyed"] = True
            except BaseException as exc:  # noqa: BLE001
                result["stream_destroyed"] = False
                result["stream_teardown_error"] = f"{type(exc).__name__}: {exc}"
                result["ok"] = False
        queue.put(result)


def _run_in_subprocess(case: str, device_id: int, platform: str) -> dict:
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run_case, args=(case, device_id, platform, queue))
    proc.start()
    proc.join(timeout=300)
    if proc.is_alive():
        # A hung child is non-daemon, so failing without reaping it would leave
        # the test process waiting on it until the CI job timeout.
        proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join()
        pytest.fail(f"case {case} did not exit within 300s")
    assert not queue.empty(), f"case {case} produced no result (exitcode={proc.exitcode})"
    return queue.get()


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.parametrize(
    "case",
    [
        "init_refused",
        "init_claims_borrowed_stream",
        "null_stream_rejected",
        "generation_is_unique",
        "uninitialized_surface_refuses",
        "program_init_still_works",
    ],
)
def test_kernel_mode_surface_on_borrowed_stream(case, st_platform, st_device_ids):
    """Drive one kernel-mode case against a stream the test itself owns."""
    assert st_device_ids, "device_count(1) fixture must yield at least one id"
    result = _run_in_subprocess(case, int(st_device_ids[0]), st_platform)
    assert result["ok"], f"case {case} failed: {result}"


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_borrowed_stream_survives_a_refused_kernel_init(st_platform, st_device_ids):
    """A refused kernel_init must leave the caller's device and stream usable.

    kernel_init never receives the stream, so what this checks is the device-level
    guarantee a framework caller depends on: simpler failing to come up neither
    resets the device nor finalizes ACL. Destroying the caller's stream after the
    refusal and finalize() is what proves the stream is still valid.
    """
    assert st_device_ids
    result = _run_in_subprocess("init_refused", int(st_device_ids[0]), st_platform)
    assert result.get("stream_nonzero"), f"test never obtained a stream: {result}"
    assert result["ok"], f"refused init did not behave per contract: {result}"
    # The assertion that makes this case distinct from init_refused: an
    # operation on the stream issued after the failed init has to succeed.
    assert result.get("stream_destroyed") is True, (
        f"the caller's stream did not survive a refused kernel_init: {result}"
    )
    assert "stream_teardown_error" not in result, f"post-init stream operation failed: {result}"


def _kernel_worker(platform: str, device_id: int = 0, runtime: str = "tensormap_and_ringbuffer"):
    from simpler.worker import Worker  # noqa: PLC0415

    return Worker(
        level=2,
        execution_mode="kernel",
        device_id=device_id,
        platform=platform,
        runtime=runtime,
    )


def _program_worker(platform: str = "a2a3", runtime: str = "tensormap_and_ringbuffer"):
    from simpler.worker import Worker  # noqa: PLC0415

    return Worker(level=2, device_id=0, platform=platform, runtime=runtime)


class TestWorkerExecutionMode:
    """The Worker-level contract that holds before any device is touched.

    ``execution_mode`` is fixed at construction and picks the dispatch surface,
    so everything here resolves from the constructor argument alone — no
    runtime binary, no device, no ACL.
    """

    def test_program_is_the_default(self):
        assert _program_worker().execution_mode == "program"

    def test_kernel_mode_is_level_2_only(self):
        from simpler.worker import Worker  # noqa: PLC0415

        for level in (3, 4):
            with pytest.raises(ValueError, match="requires level 2"):
                Worker(level=level, execution_mode="kernel")

    def test_an_unknown_mode_is_rejected(self):
        from simpler.worker import Worker  # noqa: PLC0415

        # Silently falling back to program mode is the failure this prevents:
        # nothing else validates **config keys, so a typo would come up READY
        # on the wrong surface.
        with pytest.raises(ValueError, match="execution_mode must be one of"):
            Worker(level=2, execution_mode="kernal", platform="a2a3", runtime="tensormap_and_ringbuffer")

    def test_kernel_init_requires_its_static_config(self):
        with pytest.raises(ValueError, match="requires config="):
            _kernel_worker("a2a3").init()

    def test_program_init_refuses_a_kernel_config(self):
        from simpler.task_interface import CallConfig  # noqa: PLC0415

        with pytest.raises(ValueError, match="kernel-mode only"):
            _program_worker().init(config=CallConfig())

    def test_kernel_init_refuses_a_prewarm_config(self):
        from simpler.task_interface import CallConfig  # noqa: PLC0415

        # The two configs are not interchangeable: prewarm is a ring-sizing hint
        # for a later run, config is the context's immutable sizing.
        with pytest.raises(ValueError, match="program-mode only"):
            _kernel_worker("a2a3").init(CallConfig(), config=CallConfig())

    def test_the_two_dispatch_surfaces_refuse_each_other(self):
        from simpler.task_interface import ChipStorageTaskArgs  # noqa: PLC0415

        kernel = _kernel_worker("a2a3")
        # run() is submit().wait(), so it is refused under submit's own name.
        for api, call in (
            ("register", lambda: kernel.register(None)),
            ("unregister", lambda: kernel.unregister(0)),
            ("submit", lambda: kernel.submit(None)),
            ("submit", lambda: kernel.run(None)),
        ):
            with pytest.raises(RuntimeError, match=f"Worker.{api}: requires execution_mode='program'"):
                call()

        program = _program_worker()
        for api, call in (
            ("kernel_prepare_callable", lambda: program.kernel_prepare_callable(None)),
            ("kernel_launch", lambda: program.kernel_launch(0, ChipStorageTaskArgs(), 1)),
        ):
            with pytest.raises(RuntimeError, match=f"Worker.{api}: requires execution_mode='kernel'"):
                call()

    def test_capability_is_false_before_init(self):
        # The answer belongs to a bound runtime, and neither worker has one yet.
        assert _kernel_worker("a2a3").kernel_mode_supported is False
        assert _program_worker().kernel_mode_supported is False

    def test_a_level_3_worker_reports_no_kernel_capability(self):
        from simpler.worker import Worker  # noqa: PLC0415

        # It binds no runtime of its own; its chip children each bind theirs.
        assert Worker(level=3, platform="a2a3", runtime="tensormap_and_ringbuffer").kernel_mode_supported is False


class TestWorkerKernelModeOnTheSimulator:
    """The simulator implements the four entries and refuses at init.

    Reaching that refusal is the assertion: it proves Worker.init dispatched
    into the kernel path and arrived at the C ABI, rather than being rejected
    on the way or silently taking the program path.
    """

    @staticmethod
    def _skip_without_sim_binaries():
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        try:
            RuntimeBuilder(platform="a2a3sim").get_binaries("tensormap_and_ringbuffer")
        except FileNotFoundError as e:
            pytest.skip(f"a2a3sim runtime binaries unavailable: {e}")

    def test_init_refusal_reaches_the_c_abi(self):
        import _task_interface as native  # noqa: PLC0415
        from simpler.task_interface import CallConfig  # noqa: PLC0415

        self._skip_without_sim_binaries()
        worker = _kernel_worker("a2a3sim")
        try:
            with pytest.raises(native.UnsupportedRuntimeOperation) as excinfo:
                worker.init(config=CallConfig())
            assert excinfo.value.code == native.PTO_RUNTIME_ERR_UNSUPPORTED
            assert worker.kernel_mode_supported is False
        finally:
            # A refused init must still leave a closeable Worker.
            worker.close()


def _kernel_context_config():
    """The context-static CallConfig the eager callable's rings need."""
    from simpler.task_interface import CallConfig  # noqa: PLC0415

    config = CallConfig()
    config.runtime_env.ring_task_window = [64] * 4
    config.runtime_env.ring_heap = [1 << 20] * 4
    config.runtime_env.ring_dep_pool = [1024] * 4
    return config


def _acl_memory_api(platform: str, runtime: str):
    """ctypes handles for the caller-side device memory this test owns.

    The device bind and the stream come from the nanobind ``_acl_*`` helpers
    above; allocation and copies have no such helper, so they are resolved off
    the same host runtime library, whose CANN dependencies export them.
    """
    import ctypes  # noqa: PLC0415

    from tests.ut.py.test_kernel_mode_c_api import _load  # noqa: PLC0415

    lib = _load(platform, "onboard", runtime)
    for symbol, argtypes in (
        ("aclrtMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]),
        ("aclrtFree", [ctypes.c_void_p]),
        ("aclrtMemcpy", [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]),
        ("aclrtSynchronizeStreamWithTimeout", [ctypes.c_void_p, ctypes.c_int32]),
    ):
        function = getattr(lib, symbol)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
    return lib


def _run_worker_eager_case(device_id: int, platform: str, queue) -> None:  # noqa: PLR0915 -- one linear caller script
    """Subprocess body: drive init -> prepare -> launch -> close through Worker.

    The Worker twin of ``test_kernel_mode_c_api``'s eager values case. There the
    caller is ctypes against the C ABI; here it is the public L2 Worker, which
    is what PyPTO holds. Everything outside simpler — device, stream, device
    memory — still belongs to this test.
    """
    import ctypes  # noqa: PLC0415

    result: dict[str, object] = {"case": "worker_eager", "stage": "start", "ok": False}
    stream = 0
    worker = None
    allocations: list = []
    lib = None
    runtime = "tensormap_and_ringbuffer"
    try:
        import _task_interface as native  # noqa: PLC0415
        from simpler.task_interface import ChipStorageTaskArgs, ChipTensor, DataType  # noqa: PLC0415

        from tests.ut.py.test_kernel_mode_c_api import _build_eager_callable  # noqa: PLC0415

        chip = _build_eager_callable(platform, runtime)
        lib = _acl_memory_api(platform, runtime)

        native._acl_bind_device(device_id)
        stream = native._acl_create_stream()
        result["stream_nonzero"] = bool(stream)
        result["stage"] = "borrowed"

        worker = _kernel_worker(platform, device_id=device_id, runtime=runtime)
        worker.init(config=_kernel_context_config())
        result["stage"] = "initialized"
        result["kernel_supported"] = bool(worker.kernel_mode_supported)

        # The program surface stays refused on a live kernel Worker.
        try:
            worker.submit(None)
            result["submit_refused"] = False
        except RuntimeError:
            result["submit_refused"] = True

        # Registration is pure: the same image registered twice mints two ids,
        # and both have to launch.
        first_id = worker.kernel_prepare_callable(chip)
        second_id = worker.kernel_prepare_callable(chip)
        result["ids"] = [int(first_id), int(second_id)]
        result["ids_distinct"] = first_id != second_id
        result["stage"] = "prepared"

        count = 128 * 128
        host_array = ctypes.c_float * count
        nbytes = ctypes.sizeof(host_array)
        rounds = []
        args = ChipStorageTaskArgs()
        for round_index, (callable_id, scalar) in enumerate(((first_id, 1.25), (second_id, -3.5))):
            addresses = []
            for _ in range(2):
                address = ctypes.c_void_p()
                assert lib.aclrtMalloc(ctypes.byref(address), nbytes, 0) == 0
                allocations.append(address)
                addresses.append(address)
            source, destination = addresses
            values = [float(i % 127 + round_index * 257) for i in range(count)]
            host_input = host_array(*values)
            host_output = host_array(*([-999.0] * count))
            assert lib.aclrtMemcpy(source, nbytes, host_input, nbytes, 1) == 0
            assert lib.aclrtMemcpy(destination, nbytes, host_output, nbytes, 1) == 0

            args.clear()
            args.add_tensor(ChipTensor.make(source.value, (count,), DataType.FLOAT32, child_memory=True))
            args.add_tensor(ChipTensor.make(destination.value, (count,), DataType.FLOAT32, child_memory=True))
            args.add_scalar(ctypes.c_float(scalar))
            worker.kernel_launch(callable_id, args, stream)
            # Enqueue took the snapshot, so the host-side args are already spent.
            args.clear()
            assert lib.aclrtSynchronizeStreamWithTimeout(stream, 60000) == 0
            assert lib.aclrtMemcpy(host_output, nbytes, destination, nbytes, 2) == 0
            expected = [value + scalar for value in values]
            rounds.append(list(host_output) == expected)
        result["rounds"] = rounds
        result["stage"] = "launched"

        worker.close()
        result["stage"] = "closed"
        # CLOSED is absorbing: it is what invalidates every id this context
        # minted, in place of a generation field on the id itself.
        try:
            worker.kernel_launch(first_id, ChipStorageTaskArgs(), stream)
            result["launch_after_close_refused"] = False
        except RuntimeError:
            result["launch_after_close_refused"] = True
        worker = None

        result["ok"] = (
            bool(result["kernel_supported"])
            and result["submit_refused"] is True
            and result["ids_distinct"] is True
            and all(rounds)
            and len(rounds) == 2
            and result["launch_after_close_refused"] is True
        )
        result["stage"] = "done"
    except BaseException as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if worker is not None:
            try:
                worker.close()
            except BaseException as exc:  # noqa: BLE001
                result["close_error"] = f"{type(exc).__name__}: {exc}"
                result["ok"] = False
        if lib is not None:
            for address in allocations:
                lib.aclrtFree(address)
        # Destroying the caller's stream after close() is the evidence that
        # neither the Worker nor its teardown reset the device or finalized ACL.
        if stream:
            try:
                import _task_interface as native  # noqa: PLC0415

                native._acl_destroy_stream(stream)
                result["stream_destroyed"] = True
            except BaseException as exc:  # noqa: BLE001
                result["stream_destroyed"] = False
                result["stream_teardown_error"] = f"{type(exc).__name__}: {exc}"
                result["ok"] = False
        queue.put(result)


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_worker_kernel_mode_eager_end_to_end(st_platform, st_device_ids):
    """The full public path — Worker init, prepare, launch, close — on silicon."""
    assert st_device_ids, "device_count(1) fixture must yield at least one id"
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run_worker_eager_case, args=(int(st_device_ids[0]), st_platform, queue))
    proc.start()
    proc.join(timeout=900)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join()
        pytest.fail("worker_eager did not exit within 900s")
    assert not queue.empty(), f"worker_eager produced no result (exitcode={proc.exitcode})"
    result = queue.get()
    assert result["ok"], f"worker_eager failed: {result}"
    assert result.get("stream_destroyed") is True, f"the caller's stream did not survive the Worker: {result}"
