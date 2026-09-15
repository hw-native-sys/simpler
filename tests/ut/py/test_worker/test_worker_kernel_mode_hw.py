# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Hardware UT for ``Worker(level=2, execution_mode="kernel")`` on a2a3.

The caller side is what kernel mode is: each case initializes ACL, makes the
device current, creates the stream and allocates the device tensors itself,
through ctypes on libascendcl, and lends them to the Worker. The Worker must
never create, reset or destroy any of it, so every case ends by having the
caller synchronize its stream, free its memory, destroy the stream, reset the
device and finalize ACL, and each of those must return 0.

The executed operator is the AIV vector-add-scalar kernel behind
``kernel_eager_orchestration.cpp`` (``y[i] = x[i] + scalar``), so numeric
checks are exact float32 comparisons after the caller's own synchronize.
No torch is involved.

Each case runs in a fresh interpreter started with ``python -m`` from the
repository root: kernel_init binds a runtime library into the process and the
ACL device binding is per-thread, so one case's state never reaches the next.

The ``runtime`` marker is what makes conftest's resource phase dispatch these
cases rather than deselect them, so it is load-bearing rather than descriptive.
The cases that need host_build_graph also load it; every a2a3 build produces it.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import signal
import struct
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_MODULE = "tests.ut.py.test_worker.test_worker_kernel_mode_hw"
_TMR = "tensormap_and_ringbuffer"
_HBG = "host_build_graph"
_CASE_TIMEOUT_S = 300
_SYNC_TIMEOUT_MS = 60000
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2
_COUNT = 128 * 128
_SENTINEL = -999.0

# Runtimes each case loads; a case whose runtime build is missing is skipped before any device work.
_CASE_RUNTIMES = {
    "supported_before_init": (_TMR, _HBG),
    "eager_numerics": (_TMR,),
    "prepare_twice": (_TMR,),
    "launch_refusals": (_TMR,),
    "second_worker_same_device": (_TMR,),
    "hbg_init_refused": (_HBG,),
}


# ---------------------------------------------------------------------------
# Caller side: ACL, device, stream and device tensors owned by the test
# ---------------------------------------------------------------------------


class _Caller:
    """The borrowing side of kernel mode: ACL, one current device, one stream and its device buffers."""

    def __init__(self, device: int) -> None:
        self.device = device
        self.acl = self._load_acl()
        self._stream = ctypes.c_void_p()
        self._allocations: list[ctypes.c_void_p] = []
        self._acl_initialized = False
        self._device_bound = False

    @staticmethod
    def _load_acl() -> ctypes.CDLL:
        lib = None
        for name in ("libascendcl.so", "libascendcl.so.1"):
            with contextlib.suppress(OSError):
                lib = ctypes.CDLL(name)
                break
        if lib is None:
            raise RuntimeError("libascendcl.so is not loadable; source the CANN set_env.sh first")
        signatures = {
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
        for symbol, argtypes in signatures.items():
            function = getattr(lib, symbol)
            function.argtypes = argtypes
            function.restype = ctypes.c_int
        return lib

    def open(self) -> None:
        assert self.acl.aclInit(None) == 0
        self._acl_initialized = True
        assert self.acl.aclrtSetDevice(self.device) == 0
        self._device_bound = True
        assert self.acl.aclrtCreateStream(ctypes.byref(self._stream)) == 0
        assert self._stream.value

    @property
    def stream(self) -> int:
        return int(self._stream.value or 0)

    def synchronize(self) -> int:
        return self.acl.aclrtSynchronizeStreamWithTimeout(self._stream, _SYNC_TIMEOUT_MS)

    def device_buffer(self, values: list[float]) -> int:
        """Allocate a float32 device buffer holding ``values``; freed by close()."""
        host = (ctypes.c_float * len(values))(*values)
        nbytes = ctypes.sizeof(host)
        address = ctypes.c_void_p()
        assert self.acl.aclrtMalloc(ctypes.byref(address), nbytes, 0) == 0
        self._allocations.append(address)
        assert self.acl.aclrtMemcpy(address, nbytes, host, nbytes, _ACL_MEMCPY_HOST_TO_DEVICE) == 0
        return int(address.value or 0)

    def read(self, address: int) -> list[float]:
        host = (ctypes.c_float * _COUNT)()
        nbytes = ctypes.sizeof(host)
        assert self.acl.aclrtMemcpy(host, nbytes, ctypes.c_void_p(address), nbytes, _ACL_MEMCPY_DEVICE_TO_HOST) == 0
        return list(host)

    def close(self) -> dict[str, int]:
        """Release everything the caller owns in reverse order; returns the first nonzero status per ACL call."""
        failures: dict[str, int] = {}

        def record(name: str, rc: int) -> None:
            if rc != 0:
                failures.setdefault(name, rc)

        if self._stream.value:
            record("aclrtSynchronizeStreamWithTimeout", self.synchronize())
        for address in reversed(self._allocations):
            record("aclrtFree", self.acl.aclrtFree(address))
        self._allocations.clear()
        if self._stream.value:
            record("aclrtDestroyStream", self.acl.aclrtDestroyStream(self._stream))
            self._stream = ctypes.c_void_p()
        if self._device_bound:
            record("aclrtResetDevice", self.acl.aclrtResetDevice(self.device))
            self._device_bound = False
        if self._acl_initialized:
            record("aclFinalize", self.acl.aclFinalize())
            self._acl_initialized = False
        return failures


@contextlib.contextmanager
def _caller_device(device: int):
    """Yield an opened _Caller; on a clean exit every caller-side teardown status must be 0.

    Leaving the block runs the stream synchronize, the frees, aclrtDestroyStream, aclrtResetDevice and
    aclFinalize after every Worker in the block has closed, which is the evidence that close() left the
    caller's device, stream and memory intact.
    """
    caller = _Caller(device)
    completed = False
    try:
        caller.open()
        yield caller
        completed = True
    finally:
        failures = caller.close()
        if completed:
            assert not failures, f"caller-side ACL teardown failed after the Worker closed: {failures}"
        elif failures:
            print(f"caller-side ACL teardown also failed: {failures}", file=sys.stderr)


@contextlib.contextmanager
def _kernel_worker(caller: _Caller, platform: str, runtime: str = _TMR):
    """Yield a NEW kernel-mode Worker on the caller's device; it is closed on exit, after a synchronize."""
    from simpler.worker import Worker

    worker = Worker(level=2, execution_mode="kernel", device_id=caller.device, platform=platform, runtime=runtime)
    try:
        yield worker
    finally:
        # close() requires the caller's enqueued launches to have drained; a closed Worker's close() is a no-op.
        caller.synchronize()
        worker.close()


# ---------------------------------------------------------------------------
# Callable, config and launch helpers
# ---------------------------------------------------------------------------


def _build_eager_callable(platform: str):
    """The ChipCallable ``kernel_eager_orchestration(x, y, scalar)`` submitting one AIV add-scalar task."""
    import tempfile

    from simpler.task_interface import ArgDirection, ChipCallable, CoreCallable

    from simpler_setup.elf_parser import extract_text_section
    from simpler_setup.kernel_compiler import KernelCompiler
    from simpler_setup.pto_isa import ensure_pto_isa_root

    compiler = KernelCompiler(platform)
    kernel = _PROJECT_ROOT / "examples" / platform / _TMR / "vector_example/kernels/aiv/kernel_add_scalar.cpp"
    orchestration_source = _PROJECT_ROOT / "tests/ut/py/kernel_eager_orchestration.cpp"
    with tempfile.TemporaryDirectory(prefix="worker-kernel-eager-") as build_dir:
        orchestration = compiler.compile_orchestration(_TMR, str(orchestration_source), build_dir=build_dir)
        incore = compiler.compile_incore(
            str(kernel),
            core_type="aiv",
            pto_isa_root=ensure_pto_isa_root(),
            extra_include_dirs=compiler.get_orchestration_include_dirs(_TMR),
            build_dir=build_dir,
        )
    signature = [ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR]
    child = CoreCallable.build(signature=signature, binary=extract_text_section(incore))
    return ChipCallable.build(
        signature=signature,
        func_name="kernel_eager_orchestration",
        binary=orchestration,
        children=[(0, child)],
    )


def _kernel_config():
    from simpler.task_interface import CallConfig

    config = CallConfig()
    config.runtime_env.ring_task_window = 64
    config.runtime_env.ring_heap = 1 << 20
    config.runtime_env.ring_dep_pool = 1024
    return config


def _input_values(seed: int) -> list[float]:
    # Every value and every value + scalar used below is exactly representable in float32.
    return [float(i % 127 + seed * 257) for i in range(_COUNT)]


def _launch_args(source: int, destination: int, scalar: float | None):
    """ChipStorageTaskArgs ``(x, y, scalar)``; ``scalar=None`` omits the scalar the signature requires."""
    from simpler.task_interface import ChipStorageTaskArgs, ChipTensor, DataType

    args = ChipStorageTaskArgs()
    args.add_tensor(ChipTensor.make(source, (_COUNT,), DataType.FLOAT32, child_memory=True))
    args.add_tensor(ChipTensor.make(destination, (_COUNT,), DataType.FLOAT32, child_memory=True))
    if scalar is not None:
        args.add_scalar(int.from_bytes(struct.pack("<f", scalar), "little"))
    return args


def _launch_and_check(caller: _Caller, worker, callable_id: int, scalar: float, seed: int):
    """Launch ``y = x + scalar`` on fresh buffers, synchronize the caller's stream and compare exactly.

    Returns ``(source, destination, expected)``.
    """
    values = _input_values(seed)
    source = caller.device_buffer(values)
    destination = caller.device_buffer([_SENTINEL] * _COUNT)
    args = _launch_args(source, destination, scalar)
    worker.kernel_launch(callable_id, args, caller_stream=caller.stream)
    # The launch snapshots its arguments at enqueue, so clearing them now cannot change the result.
    args.clear()
    assert caller.synchronize() == 0
    expected = [value + scalar for value in values]
    assert caller.read(destination) == expected
    return source, destination, expected


def _assert_closed_cleanly(worker, pin_finalizer) -> None:
    import simpler.worker as worker_module

    assert worker._lifecycle.name == "CLOSED"
    assert worker._chip_worker is None
    assert worker._kernel_callables == {}
    assert not pin_finalizer.alive
    assert worker_module._PINNED_KERNEL_CHIP_WORKERS == []


def _init_kernel_worker(worker):
    """init(config=...) and return the armed GC-pin finalizer."""
    worker.init(config=_kernel_config())
    assert worker._lifecycle.name == "READY"
    pin_finalizer = worker._kernel_pin_finalizer
    assert pin_finalizer is not None and pin_finalizer.alive
    return pin_finalizer


# ---------------------------------------------------------------------------
# Cases (subprocess bodies)
# ---------------------------------------------------------------------------


def _case_supported_before_init(platform: str, device: int) -> None:
    """kernel_mode_supported answers from the runtime build before init, with no device made current."""
    from simpler.worker import Worker

    for runtime, expected in ((_TMR, True), (_HBG, False)):
        kernel = Worker(level=2, execution_mode="kernel", device_id=device, platform=platform, runtime=runtime)
        program = Worker(level=2, device_id=device, platform=platform, runtime=runtime)
        for worker in (kernel, program):
            assert worker.kernel_mode_supported is expected, (runtime, worker._execution_mode)
            assert worker.kernel_mode_supported is expected, (runtime, worker._execution_mode)
            assert worker._lifecycle.name == "NEW"
            assert worker._chip_worker is None


def _case_eager_numerics(platform: str, device: int) -> None:
    """Prepare once, launch twice with different scalars, close, and leave the caller's device usable."""
    chip = _build_eager_callable(platform)
    with _caller_device(device) as caller, _kernel_worker(caller, platform) as worker:
        pin_finalizer = _init_kernel_worker(worker)
        assert worker.kernel_mode_supported is True
        callable_id = worker.kernel_prepare_callable(chip)
        with pytest.raises(RuntimeError, match="execution_mode"):
            worker.device_memory_info()
        committed = worker.committed_device_memory()
        assert committed > 0

        first = _launch_and_check(caller, worker, callable_id, scalar=1.25, seed=0)
        source, destination, _ = _launch_and_check(caller, worker, callable_id, scalar=-3.5, seed=1)
        # A launch commits no device memory, and the second launch wrote only its own output.
        assert worker.committed_device_memory() == committed
        assert caller.read(first[1]) == first[2]

        assert caller.synchronize() == 0
        worker.close()
        _assert_closed_cleanly(worker, pin_finalizer)
        with pytest.raises(RuntimeError, match="READY"):
            worker.kernel_launch(callable_id, _launch_args(source, destination, 1.0), caller_stream=caller.stream)
        with pytest.raises(RuntimeError, match="READY"):
            worker.kernel_prepare_callable(chip)
        assert caller.synchronize() == 0


def _case_prepare_twice(platform: str, device: int) -> None:
    """The same callable prepared twice takes two distinct ids, and both launch correctly."""
    chip = _build_eager_callable(platform)
    with _caller_device(device) as caller, _kernel_worker(caller, platform) as worker:
        pin_finalizer = _init_kernel_worker(worker)
        first_id = worker.kernel_prepare_callable(chip)
        second_id = worker.kernel_prepare_callable(chip)
        assert first_id >= 0 and second_id >= 0
        assert first_id != second_id
        assert set(worker._kernel_callables) == {first_id, second_id}

        _launch_and_check(caller, worker, second_id, scalar=0.75, seed=2)
        _launch_and_check(caller, worker, first_id, scalar=2.5, seed=3)

        assert caller.synchronize() == 0
        worker.close()
        _assert_closed_cleanly(worker, pin_finalizer)


def _case_launch_refusals(platform: str, device: int) -> None:
    """Refused launches enqueue nothing and leave the context launchable."""
    chip = _build_eager_callable(platform)
    with _caller_device(device) as caller, _kernel_worker(caller, platform) as worker:
        pin_finalizer = _init_kernel_worker(worker)
        callable_id = worker.kernel_prepare_callable(chip)
        values = _input_values(seed=4)
        source = caller.device_buffer(values)
        destination = caller.device_buffer([_SENTINEL] * _COUNT)
        args = _launch_args(source, destination, scalar=1.25)

        with pytest.raises(ValueError, match="kernel_prepare_callable"):
            worker.kernel_launch(callable_id + 1, args, caller_stream=caller.stream)
        with pytest.raises(ValueError, match="caller_stream"):
            worker.kernel_launch(callable_id, args, caller_stream=0)
        # Two tensors and no scalar disagree with the (IN, OUT, SCALAR) signature; the native launch refuses
        # them while encoding, before anything is enqueued on the caller's stream.
        with pytest.raises(RuntimeError, match="simpler_kernel_mode_launch failed"):
            worker.kernel_launch(callable_id, _launch_args(source, destination, None), caller_stream=caller.stream)
        assert caller.synchronize() == 0
        assert caller.read(destination) == [_SENTINEL] * _COUNT

        worker.kernel_launch(callable_id, args, caller_stream=caller.stream)
        assert caller.synchronize() == 0
        assert caller.read(destination) == [value + 1.25 for value in values]

        worker.close()
        _assert_closed_cleanly(worker, pin_finalizer)


def _case_second_worker_same_device(platform: str, device: int) -> None:
    """A second kernel Worker on a claimed device fails init without disturbing the owner; a successor inits."""
    chip = _build_eager_callable(platform)
    with _caller_device(device) as caller:
        with _kernel_worker(caller, platform) as owner:
            owner_pin = _init_kernel_worker(owner)
            owner_id = owner.kernel_prepare_callable(chip)
            with _kernel_worker(caller, platform) as rival:
                with pytest.raises(RuntimeError):
                    rival.init(config=_kernel_config())
                assert rival._lifecycle.name == "FAILED"
                assert rival._kernel_pin_finalizer is None
                rival.close()
                assert rival._lifecycle.name == "CLOSED"
                assert rival._chip_worker is None

            _launch_and_check(caller, owner, owner_id, scalar=1.25, seed=5)
            assert caller.synchronize() == 0
            owner.close()
            _assert_closed_cleanly(owner, owner_pin)

        with _kernel_worker(caller, platform) as successor:
            successor_pin = _init_kernel_worker(successor)
            successor_id = successor.kernel_prepare_callable(chip)
            _launch_and_check(caller, successor, successor_id, scalar=-3.5, seed=6)
            assert caller.synchronize() == 0
            successor.close()
            _assert_closed_cleanly(successor, successor_pin)


def _case_hbg_init_refused(platform: str, device: int) -> None:
    """host_build_graph has no kernel mode: init fails to FAILED, close() is clean, the caller's stream survives."""
    with _caller_device(device) as caller, _kernel_worker(caller, platform, runtime=_HBG) as worker:
        with pytest.raises(RuntimeError, match="kernel mode"):
            worker.init(config=_kernel_config())
        assert worker._lifecycle.name == "FAILED"
        assert worker._kernel_pin_finalizer is None
        assert caller.synchronize() == 0
        worker.close()
        assert worker._lifecycle.name == "CLOSED"
        assert worker._chip_worker is None


_CASES = {
    "supported_before_init": _case_supported_before_init,
    "eager_numerics": _case_eager_numerics,
    "prepare_twice": _case_prepare_twice,
    "launch_refusals": _case_launch_refusals,
    "second_worker_same_device": _case_second_worker_same_device,
    "hbg_init_refused": _case_hbg_init_refused,
}


# ---------------------------------------------------------------------------
# pytest side
# ---------------------------------------------------------------------------


def _require_prebuilt(platform: str, runtimes: tuple[str, ...]) -> None:
    from simpler_setup.runtime_builder import RuntimeBuilder

    for runtime in runtimes:
        try:
            RuntimeBuilder(platform=platform).get_binaries(runtime)
        except FileNotFoundError as exc:
            pytest.skip(str(exc))


def _run_case_in_subprocess(case: str, platform: str, device: int) -> None:
    env = dict(os.environ, PYTHONFAULTHANDLER="1")
    proc = subprocess.Popen(
        [sys.executable, "-m", _MODULE, case, platform, str(device)],
        cwd=str(_PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=_CASE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        # The case leads its own session, so this also kills the compiler processes it started.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        output, _ = proc.communicate()
        pytest.fail(f"case {case} did not exit within {_CASE_TIMEOUT_S}s:\n{output}")
    assert proc.returncode == 0, f"case {case} exited with {proc.returncode}:\n{output}"


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a2a3"])
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.parametrize("case", list(_CASE_RUNTIMES))
def test_worker_kernel_mode_on_caller_device(case, st_platform, st_device_ids):
    """Drive one Worker kernel-mode case against a device, stream and memory the case itself owns."""
    assert st_device_ids, "device_count(1) must yield at least one device id"
    _require_prebuilt(st_platform, _CASE_RUNTIMES[case])
    _run_case_in_subprocess(case, st_platform, int(st_device_ids[0]))


if __name__ == "__main__":
    _CASES[sys.argv[1]](sys.argv[2], int(sys.argv[3]))
