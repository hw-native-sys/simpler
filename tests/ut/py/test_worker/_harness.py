# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-free chip-backed Worker trees for the Python worker unit tests.

A chip child is forked, so the ``ChipWorker`` bound on ``simpler.worker`` at
``Worker.init()`` time is the class the forked child instantiates. Replacing it
with :class:`FakeChipWorker` yields an L3 that carries ``device_ids``, reaches
READY, serves the CTRL_REGISTER / CTRL_UNREGISTER control path, and closes —
with no NPU and no device runtime driven. ``platform="a2a3sim"`` is a
construction-time requirement only: the parent reads the prebuilt sim runtime
binaries to build the fork payload and never executes them, which is what
:data:`requires_sim_binaries` gates on.

A fake chip is the real dispatch target for its L3, so a broadcast that reaches
it is observable from the parent: ``register_error`` makes the chip child's
native register raise, and the failure surfaces to the caller of
``Worker.register`` as REGISTER_PARTIAL_FAILURE.
"""

from __future__ import annotations

import ctypes
import signal
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from typing import Any

import pytest
import simpler.worker as worker_mod
from simpler.task_interface import ChipCallable
from simpler.worker import Worker

SIM_PLATFORM = "a2a3sim"
SIM_RUNTIME = "tensormap_and_ringbuffer"

#: Message raised by ``FakeChipWorker.init`` under the ``"raises"`` script.
CHIP_INIT_FAILURE = "injected chip init failure"

#: Wall budget for a single scenario — comfortably above the injected
#: ``startup_timeout_s`` values, well under any real hang.
TEST_WALL_BUDGET_S = 30.0

#: Longer than any test budget, so the ``"hangs"`` script is only ever ended by
#: the parent's startup deadline reaping the child.
_HANG_S = 3600.0

_SCRIPTS = ("ok", "raises", "hangs")


@contextmanager
def hard_timeout(seconds: float, msg: str = "startup did not return within the hard test budget"):
    """Abort the body with TimeoutError if it runs longer than ``seconds``.

    A backstop against a regression that reintroduces an unbounded startup
    spin: instead of hanging CI, the test fails with TimeoutError. Uses SIGALRM
    (pytest runs on the main thread), which interrupts the barrier's
    ``time.sleep`` poll.

    A single itimer backs this, so nesting two live ``hard_timeout`` scopes
    leaves only the inner deadline armed.
    """

    def _handler(_signum, _frame):
        raise TimeoutError(msg)

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def sim_binaries_available() -> bool:
    """True when the prebuilt ``a2a3sim`` runtime binaries a chip-carrying L3
    reads at construction are present."""
    try:
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        RuntimeBuilder(SIM_PLATFORM).get_binaries(SIM_RUNTIME)
        return True
    except Exception:  # noqa: BLE001
        return False


requires_sim_binaries = pytest.mark.skipif(
    not sim_binaries_available(), reason=f"{SIM_PLATFORM} runtime binaries not built"
)


def chip_callable(func_name: str = "x", binary: bytes = b"\x00") -> ChipCallable:
    """A minimal LOCAL_CHIP target; a fake chip never executes its payload."""
    return ChipCallable.build(signature=[], func_name=func_name, binary=binary, children=[])


class _FakeChipRun:
    """The live handle a direct submit returns (native ``_ChipRun``).

    A fake chip never executes its payload, so the run is already at its
    completion fence; what it still has to model is that the handle is the
    caller's, and that waiting on it is idempotent.
    """

    def __init__(self) -> None:
        self.wait_count = 0

    def done(self) -> bool:
        return True

    def wait(self, timeout: float = -1.0) -> bool:
        self.wait_count += 1
        return True

    def activate(self) -> None:
        pass

    def abandon(self) -> None:
        pass

    @property
    def launched(self) -> bool:
        return True

    @property
    def lane_poisoned(self) -> bool:
        return False


class _FakeChipImpl:
    """The native-handle half of :class:`FakeChipWorker` (``cw._impl``)."""

    supports_concurrent_native_prepare = False

    def __init__(self, register_error: str | None = None) -> None:
        self._register_error = register_error
        self.submitted: list[tuple[int, Any]] = []
        self.lane_closed = False
        self.finalized = False
        # Set when _close_chip_run_lane ran while the device was still up.
        # Draining after finalize would wait on a device that is already gone.
        self.lane_closed_before_finalize: bool | None = None

    def register_callable_from_blob(self, cid: int, addr: int) -> None:
        if self._register_error is not None:
            raise RuntimeError(self._register_error)

    def run_materialized(self, *_a, **_k) -> None:
        """An L2 submit reaches the native handle here, with its args already materialized."""

    def _submit_chip_run_direct(self, cid: int, args: Any, cfg: Any) -> _FakeChipRun:
        """A direct submit admits at capacity one and returns the run while it runs."""
        run = _FakeChipRun()
        self.submitted.append((cid, run))
        return run

    def _close_chip_run_lane(self) -> None:
        self.lane_closed = True
        if self.lane_closed_before_finalize is None:
            self.lane_closed_before_finalize = not self.finalized


class FakeChipWorker:
    """Stand-in for the native ChipWorker — no NPU touched.

    ``script`` selects what ``init`` does: ``"ok"`` returns, ``"raises"`` raises
    :data:`CHIP_INIT_FAILURE`, ``"hangs"`` blocks past any test budget.
    ``register_error``, orthogonal to the script, makes the post-READY native
    register raise that message on the chip child.

    Its "device" memory is a child-local host allocation, so the pointers
    ``malloc`` hands out are meaningful only in the process that made them —
    the same property a real device pointer has across the fork.

    Build one with :func:`fake_chip_worker`; ``ChipWorker`` is constructed with
    no arguments on both the L2 in-process and the forked chip-child paths.
    """

    pipeline_depth = 1
    committed_device_memory = 0

    def __init__(self, *, script: str = "ok", register_error: str | None = None) -> None:
        if script not in _SCRIPTS:
            raise ValueError(f"unknown fake-chip script {script!r}; expected one of {_SCRIPTS}")
        self.script = script
        self._impl = _FakeChipImpl(register_error)
        self._blocks: dict[int, Any] = {}

    def malloc(self, size: int) -> int:
        block = (ctypes.c_char * int(size))()
        ptr = ctypes.addressof(block)
        self._blocks[ptr] = block
        return ptr

    def free(self, ptr: int) -> None:
        self._blocks.pop(int(ptr), None)

    def copy_to(self, dst: int, src: int, size: int) -> None:
        ctypes.memmove(dst, src, size)

    def copy_from(self, dst: int, src: int, size: int) -> None:
        ctypes.memmove(dst, src, size)

    def init(self, *_a, **_k) -> None:
        if self.script == "raises":
            raise RuntimeError(CHIP_INIT_FAILURE)
        if self.script == "hangs":
            time.sleep(_HANG_S)

    def _register_callable_at_slot(self, _cid: int, _target) -> None:
        pass

    def _unregister_slot(self, _cid: int) -> None:
        pass

    def _run_slot(self, *_a, **_k) -> None:
        pass

    def finalize(self) -> None:
        self._impl.finalized = True


def fake_chip_worker(*, script: str = "ok", register_error: str | None = None):
    """A zero-argument factory producing :class:`FakeChipWorker` instances."""
    if script not in _SCRIPTS:
        raise ValueError(f"unknown fake-chip script {script!r}; expected one of {_SCRIPTS}")
    return partial(FakeChipWorker, script=script, register_error=register_error)


def install_fake_chip(monkeypatch, *, script: str = "ok", register_error: str | None = None) -> None:
    """Bind the fake as ``simpler.worker.ChipWorker`` for the rest of the test.

    Must precede ``Worker.init()``: the chip child resolves the name at fork
    time, and an L2 worker resolves it in-process during its own init.
    """
    monkeypatch.setattr(worker_mod, "ChipWorker", fake_chip_worker(script=script, register_error=register_error))


@contextmanager
def fake_chip_l3(
    monkeypatch,
    *,
    script: str = "ok",
    register_error: str | None = None,
    device_ids: Sequence[int] = (0,),
    num_sub_workers: int = 0,
    startup_timeout_s: float = 10.0,
    init: bool = True,
    timeout_s: float = TEST_WALL_BUDGET_S,
    **worker_kwargs,
) -> Iterator[Worker]:
    """Yield an L3 backed by ``len(device_ids)`` fake chips, closed on exit.

    With ``init=False`` the worker is yielded NEW, for a body that registers
    into the startup snapshot or asserts on ``init()`` itself. The whole body,
    including ``close()``, runs under :func:`hard_timeout` — do not nest another
    one inside it.
    """
    install_fake_chip(monkeypatch, script=script, register_error=register_error)
    worker = Worker(
        level=3,
        device_ids=list(device_ids),
        platform=SIM_PLATFORM,
        runtime=SIM_RUNTIME,
        num_sub_workers=num_sub_workers,
        startup_timeout_s=startup_timeout_s,
        **worker_kwargs,
    )
    with hard_timeout(timeout_s):
        try:
            if init:
                worker.init()
            yield worker
        finally:
            worker.close()
