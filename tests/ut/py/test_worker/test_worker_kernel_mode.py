# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-free UT for ``Worker(level=2, execution_mode="kernel")``.

``simpler.worker.ChipWorker`` and ``simpler_setup.runtime_builder.RuntimeBuilder``
are replaced by fakes, so every case runs without an NPU or a built runtime. The
fake kernel ChipWorker records what the Worker forwards to it and fails loudly
on any program-mode entry, which is how a case proves a kernel-mode Worker never
reaches the program path. ``TestSimRuntime`` is the one case that loads a real
runtime build; it skips when the a2a3sim binaries are absent.
"""

from __future__ import annotations

import ctypes
import gc
import os
import re
import threading
import warnings
import weakref
from collections.abc import Callable
from typing import Any, cast

import pytest
import simpler.worker as worker_mod
from simpler.task_interface import CallConfig, ChipStorageTaskArgs, ChipWorkerError
from simpler.worker import Worker

import simpler_setup.runtime_builder as rb_mod

from ._harness import SIM_PLATFORM, SIM_RUNTIME, TEST_WALL_BUDGET_S, chip_callable, hard_timeout, install_fake_chip

_DEVICE_ID = 3
_STREAM = 0xABC0
_Lifecycle = worker_mod._Lifecycle

#: A stand-in for an argument of the wrong type; the refusals under test run before it is used.
_ANY: Any = object()


def _run_catch(fn: Callable[[], Any]) -> BaseException | None:
    """Run ``fn`` in a thread body, returning None on success or the exception."""
    try:
        fn()
        return None
    except BaseException as e:  # noqa: BLE001
        return e


class _KernelScript:
    """What the fake kernel ChipWorker and runtime builder do; a case sets it before the call it scripts."""

    def __init__(self) -> None:
        self.binaries = object()
        self.builder_calls: list[tuple[str, str]] = []
        self.builder_error: BaseException | None = None
        self.probe_result = True
        self.probe_calls: list[Any] = []
        self.kernel_init_error: BaseException | None = None
        # Ids kernel_prepare_callable returns, in order; once exhausted each chip counts up from 0.
        self.prepare_ids: list[int] = []
        self.prepare_entered = threading.Event()
        self.prepare_release: threading.Event | None = None
        self.launch_entered = threading.Event()
        self.launch_release: threading.Event | None = None
        self.launch_hook: Callable[[], None] | None = None
        # Number of finalize() calls that raise ChipWorkerError for a failed device teardown.
        self.failed_finalizes = 0
        self.chips: list[_FakeKernelChip] = []


class _FakeKernelImpl:
    """The native-handle half of :class:`_FakeKernelChip` (``chip._impl``)."""

    def __init__(self) -> None:
        self.initialized = False
        self.lane_closes = 0

    def _close_chip_run_lane(self) -> None:
        self.lane_closes += 1

    def register_callable_from_blob(self, *_a, **_k) -> None:
        raise AssertionError("kernel-mode Worker reached native register_callable_from_blob")

    def run_materialized(self, *_a, **_k) -> None:
        raise AssertionError("kernel-mode Worker reached native run_materialized")

    def _submit_chip_run_direct(self, *_a, **_k) -> None:
        raise AssertionError("kernel-mode Worker reached native _submit_chip_run_direct")


class _FakeKernelChip:
    """Stand-in for ``simpler.task_interface.ChipWorker`` driven through its kernel-mode surface."""

    pipeline_depth = 1
    committed_device_memory = 4096

    def __init__(self, script: _KernelScript) -> None:
        self._script = script
        self._impl = _FakeKernelImpl()
        self._next_id = 0
        self.kernel_init_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.prepared: list[tuple[int, Any]] = []
        self.launches: list[tuple[int, Any, int]] = []
        self.finalize_threads: list[threading.Thread] = []
        # len(finalize_threads) observed by each launch as it returns.
        self.finalizes_at_launch_return: list[int] = []
        script.chips.append(self)

    def kernel_init(self, *args: Any, **kwargs: Any) -> None:
        self.kernel_init_calls.append((args, kwargs))
        if self._script.kernel_init_error is not None:
            raise self._script.kernel_init_error
        self._impl.initialized = True

    def kernel_prepare_callable(self, chip_callable_obj: Any) -> int:
        script = self._script
        script.prepare_entered.set()
        if script.prepare_release is not None:
            assert script.prepare_release.wait(TEST_WALL_BUDGET_S)
        if script.prepare_ids:
            callable_id = script.prepare_ids.pop(0)
        else:
            callable_id = self._next_id
            self._next_id += 1
        self.prepared.append((callable_id, chip_callable_obj))
        return callable_id

    def kernel_launch(self, callable_id: int, args: Any, caller_stream: int) -> None:
        script = self._script
        if script.launch_hook is not None:
            script.launch_hook()
        script.launch_entered.set()
        if script.launch_release is not None:
            assert script.launch_release.wait(TEST_WALL_BUDGET_S)
        self.launches.append((callable_id, args, caller_stream))
        self.finalizes_at_launch_return.append(len(self.finalize_threads))

    def finalize(self) -> None:
        self.finalize_threads.append(threading.current_thread())
        # A kernel context clears initialized and records the owed teardown before it raises,
        # so a retry is a second finalize() rather than a re-init.
        self._impl.initialized = False
        if self._script.failed_finalizes > 0:
            self._script.failed_finalizes -= 1
            raise ChipWorkerError(-77, "ChipWorker::finalize: device teardown failed (-77)")

    def init(self, *_a, **_k) -> None:
        raise AssertionError("kernel-mode Worker reached ChipWorker.init")

    def malloc(self, *_a, **_k) -> int:
        raise AssertionError("kernel-mode Worker reached ChipWorker.malloc")

    def free(self, *_a, **_k) -> None:
        raise AssertionError("kernel-mode Worker reached ChipWorker.free")

    def copy_to(self, *_a, **_k) -> None:
        raise AssertionError("kernel-mode Worker reached ChipWorker.copy_to")

    def copy_from(self, *_a, **_k) -> None:
        raise AssertionError("kernel-mode Worker reached ChipWorker.copy_from")

    def device_memory_info(self, *_a, **_k) -> None:
        raise AssertionError("kernel-mode Worker reached ChipWorker.device_memory_info")

    def _register_callable_at_slot(self, *_a, **_k) -> None:
        raise AssertionError("kernel-mode Worker reached ChipWorker._register_callable_at_slot")

    def _unregister_slot(self, *_a, **_k) -> None:
        raise AssertionError("kernel-mode Worker reached ChipWorker._unregister_slot")

    def _run_slot(self, *_a, **_k) -> None:
        raise AssertionError("kernel-mode Worker reached ChipWorker._run_slot")


class _ObservedGate:
    """A kernel gate that reports when a blocking acquire starts; non-blocking attempts are not reported."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.blocking_acquire = threading.Event()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if blocking:
            self.blocking_acquire.set()
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *_exc_info: Any) -> None:
        self.release()


class _OtherProcessOs:
    """``os`` as a forked child of the test process sees it: another pid, everything else real."""

    def __init__(self, pid: int) -> None:
        self._pid = pid

    def getpid(self) -> int:
        return self._pid

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)


@pytest.fixture
def script(monkeypatch) -> _KernelScript:
    """Bind the fake kernel ChipWorker and runtime builder, and give the GC pin list a per-test instance."""
    kernel_script = _KernelScript()

    class _Chip(_FakeKernelChip):
        def __init__(self) -> None:
            super().__init__(kernel_script)

        @staticmethod
        def probe_kernel_mode_supported(bins: Any, log_level: int | None = None) -> bool:
            kernel_script.probe_calls.append(bins)
            return kernel_script.probe_result

    class _Builder:
        def __init__(self, platform: str, *_a, **_k) -> None:
            self._platform = platform

        def get_binaries(self, runtime: str, *_a, **_k) -> Any:
            kernel_script.builder_calls.append((self._platform, runtime))
            if kernel_script.builder_error is not None:
                raise kernel_script.builder_error
            return kernel_script.binaries

    monkeypatch.setattr(worker_mod, "ChipWorker", _Chip)
    monkeypatch.setattr(rb_mod, "RuntimeBuilder", _Builder)
    monkeypatch.setattr(worker_mod, "_PINNED_KERNEL_CHIP_WORKERS", [])
    return kernel_script


def _kernel_worker(**config: Any) -> Worker:
    return Worker(
        level=2,
        execution_mode="kernel",
        device_id=_DEVICE_ID,
        platform=SIM_PLATFORM,
        runtime=SIM_RUNTIME,
        **config,
    )


def _program_worker() -> Worker:
    return Worker(level=2, device_id=_DEVICE_ID, platform=SIM_PLATFORM, runtime=SIM_RUNTIME)


def _ready_kernel_worker(script: _KernelScript) -> tuple[Worker, _FakeKernelChip]:
    worker = _kernel_worker()
    worker.init(config=CallConfig())
    assert len(script.chips) == 1
    return worker, script.chips[0]


class _RejectingConfig:
    def validate(self) -> None:
        raise ValueError("injected CallConfig rejection")


class TestConstruction:
    def test_default_mode_is_program(self, script):
        assert _program_worker()._execution_mode == "program"
        explicit = Worker(level=2, execution_mode="program", platform=SIM_PLATFORM, runtime=SIM_RUNTIME)
        assert explicit._execution_mode == "program"

    def test_unknown_mode_raises(self, script):
        with pytest.raises(ValueError, match="execution_mode must be 'program' or 'kernel'"):
            Worker(level=2, execution_mode="graph", platform=SIM_PLATFORM, runtime=SIM_RUNTIME)

    def test_kernel_requires_level_2(self, script):
        with pytest.raises(ValueError, match=re.escape("execution_mode='kernel' requires level=2")):
            Worker(level=3, execution_mode="kernel", num_sub_workers=0)

    def test_kernel_refuses_sdma(self, script):
        with pytest.raises(ValueError, match="enable_sdma"):
            _kernel_worker(enable_sdma=True)
        assert _kernel_worker(enable_sdma=False)._execution_mode == "kernel"

    def test_construction_touches_no_runtime(self, script):
        _kernel_worker()
        assert script.chips == []
        assert script.builder_calls == []
        assert script.probe_calls == []


class TestInitArguments:
    """A refused init() argument spends no startup epoch: the Worker stays NEW and a correct init() follows."""

    def _assert_untouched(self, worker: Worker, script: _KernelScript) -> None:
        assert worker._lifecycle is _Lifecycle.NEW
        assert script.chips == []
        assert script.builder_calls == []

    def test_kernel_mode_requires_config(self, script):
        worker = _kernel_worker()
        with pytest.raises(ValueError, match=re.escape("config=CallConfig(...)")):
            worker.init()
        self._assert_untouched(worker, script)
        worker.init(config=CallConfig())
        assert worker._lifecycle is _Lifecycle.READY
        worker.close()

    def test_positional_config_binds_to_prewarm_config(self, script):
        worker = _kernel_worker()
        with pytest.raises(ValueError, match="positional argument binds to prewarm_config"):
            worker.init(CallConfig())
        self._assert_untouched(worker, script)
        worker.close()

    def test_kernel_mode_refuses_prewarm_config(self, script):
        worker = _kernel_worker()
        with pytest.raises(ValueError, match="takes no prewarm_config"):
            worker.init(CallConfig(), config=CallConfig())
        self._assert_untouched(worker, script)
        worker.close()

    def test_kernel_config_is_validated_before_startup(self, script):
        worker = _kernel_worker()
        with pytest.raises(ValueError, match="injected CallConfig rejection"):
            worker.init(config=cast(Any, _RejectingConfig()))
        self._assert_untouched(worker, script)
        worker.close()

    def test_program_mode_refuses_config(self, script):
        worker = _program_worker()
        with pytest.raises(ValueError, match="only accepted with execution_mode='kernel'"):
            worker.init(config=CallConfig())
        self._assert_untouched(worker, script)
        worker.close()


class TestInitRouting:
    def test_init_binds_through_kernel_init_only(self, script):
        config = CallConfig()
        worker = _kernel_worker()
        worker.init(config=config)
        try:
            assert worker._lifecycle is _Lifecycle.READY
            assert script.builder_calls == [(SIM_PLATFORM, SIM_RUNTIME)]
            (chip,) = script.chips
            assert worker._chip_worker is chip
            # No context_generation: ChipWorker.kernel_init mints it.
            assert chip.kernel_init_calls == [((_DEVICE_ID, script.binaries, config), {})]
            assert chip._impl.initialized
            assert worker._callable_registry == {}
            assert worker._identity_registry == {}
            assert chip.prepared == []
        finally:
            worker.close()

    def test_failed_kernel_init_rolls_back_to_failed(self, script):
        script.kernel_init_error = RuntimeError("injected kernel_init failure")
        worker = _kernel_worker()
        with pytest.raises(RuntimeError, match="injected kernel_init failure"):
            worker.init(config=CallConfig())
        (chip,) = script.chips
        assert worker._lifecycle is _Lifecycle.FAILED
        assert chip.finalize_threads == [threading.current_thread()]
        assert worker._chip_worker is None
        assert worker._kernel_pid is None
        assert worker._kernel_chip_pin is None
        assert worker._kernel_pin_finalizer is None

        worker.close()
        assert len(chip.finalize_threads) == 1
        with pytest.raises(RuntimeError, match="closed"):
            worker.init(config=CallConfig())

    def test_failed_kernel_init_arms_no_gc_pin(self, script):
        script.kernel_init_error = RuntimeError("injected kernel_init failure")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            worker = _kernel_worker()
            with pytest.raises(RuntimeError, match="injected kernel_init failure"):
                worker.init(config=CallConfig())
            # The script holds the raised exception, whose traceback reaches this Worker's init
            # frames; without dropping it the Worker stays reachable from the fixture.
            script.kernel_init_error = None
            ref = weakref.ref(worker)
            del worker
            gc.collect()
        assert ref() is None
        assert not [w for w in caught if issubclass(w.category, ResourceWarning)]
        assert worker_mod._PINNED_KERNEL_CHIP_WORKERS == []


class TestKernelModeSupported:
    @pytest.mark.parametrize("answer", [True, False])
    def test_probe_before_init_reports_runtime_answer(self, script, answer):
        script.probe_result = answer
        worker = _kernel_worker()
        assert worker.kernel_mode_supported is answer
        assert script.builder_calls == [(SIM_PLATFORM, SIM_RUNTIME)]
        assert script.probe_calls == [script.binaries]
        assert script.chips == []
        assert worker._lifecycle is _Lifecycle.NEW

    def test_probe_answer_is_cached(self, script):
        worker = _kernel_worker()
        assert worker.kernel_mode_supported is True
        script.probe_result = False
        assert worker.kernel_mode_supported is True
        assert len(script.probe_calls) == 1

    def test_program_mode_worker_probes_the_same_build(self, script):
        script.probe_result = True
        assert _program_worker().kernel_mode_supported is True
        assert script.probe_calls == [script.binaries]

    def test_level_3_is_false_without_probing(self, script):
        assert Worker(level=3, num_sub_workers=0).kernel_mode_supported is False
        assert script.builder_calls == []
        assert script.probe_calls == []

    def test_ready_kernel_worker_is_true_without_probing(self, script):
        script.probe_result = False
        worker, _chip = _ready_kernel_worker(script)
        try:
            assert worker.kernel_mode_supported is True
            assert script.probe_calls == []
        finally:
            worker.close()

    def test_missing_binaries_propagate_and_are_not_cached(self, script):
        script.builder_error = FileNotFoundError("injected missing runtime build")
        worker = _kernel_worker()
        with pytest.raises(FileNotFoundError, match="injected missing runtime build"):
            _ = worker.kernel_mode_supported
        assert script.probe_calls == []
        script.builder_error = None
        assert worker.kernel_mode_supported is True
        assert len(script.probe_calls) == 1


class TestPrepare:
    def test_same_callable_twice_takes_distinct_ids(self, script):
        worker, chip = _ready_kernel_worker(script)
        try:
            target = chip_callable()
            first = worker.kernel_prepare_callable(target)
            second = worker.kernel_prepare_callable(target)
            assert (first, second) == (0, 1)
            assert chip.prepared == [(0, target), (1, target)]
            assert worker._kernel_callables == {0: target, 1: target}
            assert worker._callable_registry == {}
            assert worker._identity_registry == {}
            assert worker._live_handles == {}
            assert worker._active_ops == 0
        finally:
            worker.close()
        assert worker._kernel_callables == {}

    def test_non_chip_callable_is_refused_before_native(self, script):
        worker, chip = _ready_kernel_worker(script)
        try:
            with pytest.raises(TypeError, match="expected a ChipCallable"):
                worker.kernel_prepare_callable(_ANY)
            assert chip.prepared == []
        finally:
            worker.close()

    def test_refused_before_init(self, script):
        worker = _kernel_worker()
        with pytest.raises(RuntimeError, match=r"requires an initialized \(READY\) worker"):
            worker.kernel_prepare_callable(chip_callable())
        assert script.chips == []
        worker.close()

    def test_refused_after_close(self, script):
        worker, chip = _ready_kernel_worker(script)
        worker.close()
        with pytest.raises(RuntimeError, match=r"requires an initialized \(READY\) worker"):
            worker.kernel_prepare_callable(chip_callable())
        assert chip.prepared == []

    def test_negative_id_raises_and_retains_the_callable(self, script):
        worker, chip = _ready_kernel_worker(script)
        try:
            script.prepare_ids = [-1]
            target = chip_callable()
            with pytest.raises(RuntimeError, match="invalid id -1"):
                worker.kernel_prepare_callable(target)
            assert chip.prepared == [(-1, target)]
            assert worker._kernel_callables == {-1: target}
            assert worker._active_ops == 0
            assert worker._kernel_gate.acquire(blocking=False)
            worker._kernel_gate.release()
            with pytest.raises(ValueError, match="was not returned by this Worker's kernel_prepare_callable"):
                worker.kernel_launch(-1, ChipStorageTaskArgs(), caller_stream=_STREAM)
            assert chip.launches == []
            assert worker._kernel_callables == {-1: target}
        finally:
            worker.close()
        assert worker._kernel_callables == {}


class TestLaunch:
    def test_launch_forwards_and_takes_no_lease(self, script):
        worker, chip = _ready_kernel_worker(script)
        try:
            callable_id = worker.kernel_prepare_callable(chip_callable())
            args = ChipStorageTaskArgs()
            observed: dict[str, Any] = {}
            script.launch_hook = lambda: observed.update(
                active_ops=worker._active_ops, lease_depth=dict(worker._lease_depth)
            )
            assert worker.kernel_launch(callable_id, args, caller_stream=_STREAM) is None
            assert len(chip.launches) == 1
            launched_id, launched_args, launched_stream = chip.launches[0]
            assert (launched_id, launched_stream) == (callable_id, _STREAM)
            assert launched_args is args
            assert observed == {"active_ops": 0, "lease_depth": {}}
            assert worker._chip_runs == {}
            assert worker._accepted_run_handles == set()
        finally:
            worker.close()

    def test_bad_arguments_are_refused_without_native_calls(self, script):
        worker, chip = _ready_kernel_worker(script)
        try:
            callable_id = worker.kernel_prepare_callable(chip_callable())
            args = ChipStorageTaskArgs()
            with pytest.raises(ValueError, match="non-null caller_stream"):
                worker.kernel_launch(callable_id, args, caller_stream=0)
            with pytest.raises(TypeError, match="args must be ChipStorageTaskArgs"):
                worker.kernel_launch(callable_id, _ANY, caller_stream=_STREAM)
            with pytest.raises(ValueError, match="was not returned by this Worker's kernel_prepare_callable"):
                worker.kernel_launch(callable_id + 7, args, caller_stream=_STREAM)
            with pytest.raises(TypeError):
                cast(Any, worker).kernel_launch(callable_id, args, _STREAM)
            with pytest.raises(TypeError):
                worker.kernel_launch(cast(Any, callable_id + 0.5), args, caller_stream=_STREAM)
            assert chip.launches == []
        finally:
            worker.close()

    def test_refused_before_init(self, script):
        worker = _kernel_worker()
        with pytest.raises(RuntimeError, match=r"requires an initialized \(READY\) worker"):
            worker.kernel_launch(0, ChipStorageTaskArgs(), caller_stream=_STREAM)
        assert script.chips == []
        worker.close()

    def test_refused_after_close(self, script):
        worker, chip = _ready_kernel_worker(script)
        callable_id = worker.kernel_prepare_callable(chip_callable())
        worker.close()
        with pytest.raises(RuntimeError, match=r"requires an initialized \(READY\) worker"):
            worker.kernel_launch(callable_id, ChipStorageTaskArgs(), caller_stream=_STREAM)
        assert chip.launches == []


_PROGRAM_ONLY_CALLS: dict[str, Callable[[Worker], Any]] = {
    "register": lambda w: w.register(chip_callable()),
    "unregister": lambda w: w.unregister(0),
    "submit": lambda w: w.submit(chip_callable()),
    "run": lambda w: w.run(chip_callable()),
    "malloc": lambda w: w.malloc(16),
    "free": lambda w: w.free(_ANY),
    "copy_to": lambda w: w.copy_to(_ANY, _ANY),
    "copy_from": lambda w: w.copy_from(_ANY, _ANY),
    "create_buffer": lambda w: w.create_buffer(16),
    "make_tensor_arg": lambda w: w.make_tensor_arg(_ANY, (1,), 0),
    "release_buffer": lambda w: w.release_buffer(_ANY),
    "device_memory_info": lambda w: w.device_memory_info(),
}


class TestModeGuards:
    @pytest.mark.parametrize("api", sorted(_PROGRAM_ONLY_CALLS))
    def test_program_only_api_refuses_in_kernel_mode(self, script, api):
        worker, chip = _ready_kernel_worker(script)
        try:
            with pytest.raises(RuntimeError, match=re.escape(f"Worker.{api}: requires execution_mode='program'")):
                _PROGRAM_ONLY_CALLS[api](worker)
            assert worker._callable_registry == {}
            assert chip.prepared == []
        finally:
            worker.close()

    def test_register_refuses_in_kernel_mode_at_new(self, script):
        worker = _kernel_worker()
        with pytest.raises(RuntimeError, match=re.escape("Worker.register: requires execution_mode='program'")):
            worker.register(chip_callable())
        assert worker._callable_registry == {}
        assert worker._identity_registry == {}
        worker.init(config=CallConfig())
        worker.close()

    def test_committed_device_memory_stays_available(self, script):
        worker, _chip = _ready_kernel_worker(script)
        try:
            assert worker.committed_device_memory() == _FakeKernelChip.committed_device_memory
        finally:
            worker.close()

    def test_kernel_apis_refuse_in_program_mode_at_new(self, script):
        worker = _program_worker()
        with pytest.raises(RuntimeError, match=re.escape("requires execution_mode='kernel'")):
            worker.kernel_prepare_callable(chip_callable())
        with pytest.raises(RuntimeError, match=re.escape("requires execution_mode='kernel'")):
            worker.kernel_launch(0, ChipStorageTaskArgs(), caller_stream=0)
        worker.close()

    def test_kernel_apis_refuse_on_a_ready_program_worker(self, script, monkeypatch):
        install_fake_chip(monkeypatch)
        worker = _program_worker()
        worker.init()
        try:
            with pytest.raises(RuntimeError, match=re.escape("requires execution_mode='kernel'")):
                worker.kernel_prepare_callable(chip_callable())
            with pytest.raises(RuntimeError, match=re.escape("requires execution_mode='kernel'")):
                worker.kernel_launch(0, ChipStorageTaskArgs(), caller_stream=_STREAM)
        finally:
            worker.close()


class TestKernelGate:
    def test_launch_during_blocked_prepare_fails_fast(self, script):
        with hard_timeout(TEST_WALL_BUDGET_S):
            worker, chip = _ready_kernel_worker(script)
            try:
                callable_id = worker.kernel_prepare_callable(chip_callable())
                args = ChipStorageTaskArgs()
                script.prepare_entered.clear()
                script.prepare_release = threading.Event()
                prepare_result: list[BaseException | None] = []
                preparer = threading.Thread(
                    target=lambda: prepare_result.append(
                        _run_catch(lambda: worker.kernel_prepare_callable(chip_callable("second")))
                    )
                )
                preparer.start()
                try:
                    assert script.prepare_entered.wait(TEST_WALL_BUDGET_S)
                    with pytest.raises(RuntimeError, match="a kernel prepare/launch/close is in progress"):
                        worker.kernel_launch(callable_id, args, caller_stream=_STREAM)
                    assert chip.launches == []
                finally:
                    script.prepare_release.set()
                    preparer.join(TEST_WALL_BUDGET_S)
                assert prepare_result == [None]
                worker.kernel_launch(callable_id, args, caller_stream=_STREAM)
                assert len(chip.launches) == 1
            finally:
                worker.close()

    def test_close_during_blocked_launch_finalizes_after_it_returns(self, script):
        with hard_timeout(TEST_WALL_BUDGET_S):
            worker = _kernel_worker()
            gate = _ObservedGate()
            worker._kernel_gate = gate
            worker.init(config=CallConfig())
            (chip,) = script.chips
            callable_id = worker.kernel_prepare_callable(chip_callable())
            gate.blocking_acquire.clear()
            script.launch_release = threading.Event()
            launch_result: list[BaseException | None] = []
            finalizes_before_release: list[int] = []

            def release_once_close_waits() -> None:
                if gate.blocking_acquire.wait(TEST_WALL_BUDGET_S):
                    finalizes_before_release.append(len(chip.finalize_threads))
                assert script.launch_release is not None
                script.launch_release.set()

            launcher = threading.Thread(
                target=lambda: launch_result.append(
                    _run_catch(lambda: worker.kernel_launch(callable_id, ChipStorageTaskArgs(), caller_stream=_STREAM))
                )
            )
            releaser = threading.Thread(target=release_once_close_waits)
            launcher.start()
            try:
                assert script.launch_entered.wait(TEST_WALL_BUDGET_S)
                releaser.start()
                worker.close()
            finally:
                script.launch_release.set()
                launcher.join(TEST_WALL_BUDGET_S)
                if releaser.ident is not None:
                    releaser.join(TEST_WALL_BUDGET_S)
            assert launch_result == [None]
            assert finalizes_before_release == [0]
            assert chip.finalizes_at_launch_return == [0]
            assert chip.finalize_threads == [threading.current_thread()]
            assert worker._chip_worker is None

    def test_close_past_the_gate_budget_keeps_the_context_for_retry(self, script, monkeypatch):
        monkeypatch.setattr(worker_mod, "_ROLLBACK_GRACEFUL_TIMEOUT_S", 0.2)
        with hard_timeout(TEST_WALL_BUDGET_S):
            worker, chip = _ready_kernel_worker(script)
            callable_id = worker.kernel_prepare_callable(chip_callable())
            script.launch_release = threading.Event()
            launch_result: list[BaseException | None] = []
            launcher = threading.Thread(
                target=lambda: launch_result.append(
                    _run_catch(lambda: worker.kernel_launch(callable_id, ChipStorageTaskArgs(), caller_stream=_STREAM))
                )
            )
            launcher.start()
            try:
                assert script.launch_entered.wait(TEST_WALL_BUDGET_S)
                with pytest.raises(TimeoutError, match=re.escape("close() again")):
                    worker.close()
                assert worker._lifecycle is _Lifecycle.CLOSED
                assert worker._chip_worker is chip
                assert chip.finalize_threads == []
                assert callable_id in worker._kernel_callables
            finally:
                script.launch_release.set()
                launcher.join(TEST_WALL_BUDGET_S)
            assert launch_result == [None]
            worker.close()
            assert len(chip.finalize_threads) == 1
            assert worker._chip_worker is None


class TestClose:
    def test_close_finalizes_once_on_the_init_thread(self, script):
        worker, chip = _ready_kernel_worker(script)
        worker.kernel_prepare_callable(chip_callable())
        finalizer = worker._kernel_pin_finalizer
        assert finalizer is not None and finalizer.alive
        worker.close()
        assert worker._lifecycle is _Lifecycle.CLOSED
        assert chip.finalize_threads == [threading.current_thread()]
        assert chip._impl.lane_closes == 1
        assert worker._chip_worker is None
        assert worker._kernel_callables == {}
        assert not finalizer.alive
        assert worker._kernel_pin_finalizer is None
        worker.close()
        assert len(chip.finalize_threads) == 1

    def test_non_owner_thread_close_raises(self, script):
        with hard_timeout(TEST_WALL_BUDGET_S):
            worker, chip = _ready_kernel_worker(script)
            try:
                close_result: list[BaseException | None] = []
                closer = threading.Thread(target=lambda: close_result.append(_run_catch(worker.close)))
                closer.start()
                closer.join(TEST_WALL_BUDGET_S)
                assert len(close_result) == 1
                assert isinstance(close_result[0], RuntimeError)
                assert "thread that init()'d it" in str(close_result[0])
                assert worker._lifecycle is _Lifecycle.READY
                assert chip.finalize_threads == []
                callable_id = worker.kernel_prepare_callable(chip_callable())
                worker.kernel_launch(callable_id, ChipStorageTaskArgs(), caller_stream=_STREAM)
            finally:
                worker.close()
            assert chip.finalize_threads == [threading.current_thread()]

    def test_failed_teardown_keeps_the_context_for_retry(self, script):
        worker, chip = _ready_kernel_worker(script)
        target = chip_callable()
        callable_id = worker.kernel_prepare_callable(target)
        script.failed_finalizes = 1

        with pytest.raises(ChipWorkerError, match="device teardown failed"):
            worker.close()
        assert worker._lifecycle is _Lifecycle.CLOSED
        assert len(chip.finalize_threads) == 1
        assert worker._chip_worker is chip
        assert worker._kernel_callables == {callable_id: target}
        finalizer = worker._kernel_pin_finalizer
        assert finalizer is not None and finalizer.alive
        with pytest.raises(RuntimeError, match=r"requires an initialized \(READY\) worker"):
            worker.kernel_launch(callable_id, ChipStorageTaskArgs(), caller_stream=_STREAM)
        with pytest.raises(RuntimeError, match=r"requires an initialized \(READY\) worker"):
            worker.kernel_prepare_callable(target)
        assert chip.launches == []

        worker.close()
        assert len(chip.finalize_threads) == 2
        assert worker._chip_worker is None
        assert worker._kernel_callables == {}
        assert not finalizer.alive
        worker.close()
        assert len(chip.finalize_threads) == 2


class TestGcPin:
    @staticmethod
    def _abandon_ready_kernel_worker(script: _KernelScript) -> tuple[weakref.ref, _FakeKernelChip]:
        worker, chip = _ready_kernel_worker(script)
        return weakref.ref(worker), chip

    def test_unclosed_ready_worker_pins_its_chip_instead_of_finalizing(self, script):
        with pytest.warns(ResourceWarning, match="garbage-collected without close"):
            ref, chip = self._abandon_ready_kernel_worker(script)
            gc.collect()
        try:
            assert ref() is None
            assert worker_mod._PINNED_KERNEL_CHIP_WORKERS == [chip]
            assert chip.finalize_threads == []
            assert chip._impl.initialized
        finally:
            # The pin takes a reference it never releases; returning it lets the fake chip, and the
            # CallConfig it recorded, go away with this test instead of outliving the process.
            ctypes.pythonapi.Py_DecRef(ctypes.py_object(chip))

    def test_closed_worker_is_not_pinned(self, script):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            worker, chip = _ready_kernel_worker(script)
            worker.close()
            ref = weakref.ref(worker)
            del worker
            gc.collect()
        assert ref() is None
        assert not [w for w in caught if issubclass(w.category, ResourceWarning)]
        assert worker_mod._PINNED_KERNEL_CHIP_WORKERS == []
        assert len(chip.finalize_threads) == 1


class TestProcessFence:
    def test_other_process_cannot_prepare_launch_or_close(self, script, monkeypatch):
        worker, chip = _ready_kernel_worker(script)
        callable_id = worker.kernel_prepare_callable(chip_callable())
        args = ChipStorageTaskArgs()
        with monkeypatch.context() as patch:
            patch.setattr(worker_mod, "os", _OtherProcessOs(os.getpid() + 1))
            with pytest.raises(RuntimeError, match="must not drive or tear it down"):
                worker.kernel_prepare_callable(chip_callable())
            with pytest.raises(RuntimeError, match="must not drive or tear it down"):
                worker.kernel_launch(callable_id, args, caller_stream=_STREAM)
            with pytest.raises(RuntimeError, match="must not drive or tear it down"):
                worker.close()
        assert len(chip.prepared) == 1
        assert chip.launches == []
        assert chip.finalize_threads == []
        assert worker._chip_worker is chip

        worker.close()
        assert chip.finalize_threads == [threading.current_thread()]
        assert worker._chip_worker is None


class TestSimRuntime:
    """The real a2a3sim tensormap_and_ringbuffer build, which has no kernel mode."""

    def test_sim_runtime_reports_no_kernel_mode_and_init_rolls_back(self):
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        try:
            RuntimeBuilder(platform=SIM_PLATFORM).get_binaries(SIM_RUNTIME)
        except FileNotFoundError as e:
            pytest.skip(f"{SIM_PLATFORM} runtime binaries unavailable: {e}")

        worker = Worker(level=2, execution_mode="kernel", platform=SIM_PLATFORM, runtime=SIM_RUNTIME)
        assert worker.kernel_mode_supported is False
        with pytest.raises(RuntimeError, match="kernel mode"):
            worker.init(config=CallConfig())
        assert worker._lifecycle is _Lifecycle.FAILED
        assert worker._chip_worker is None
        worker.close()
