#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3: a dispatch runs the callable it names, whichever ran on that worker before.

Two ChipCallables are built from the same orchestration by exchanging the AIV binaries at
``func_id`` 0 and 2. ``kernel_add`` and ``kernel_mul`` share an argument layout, so the
exchange is signature-safe, but the two callables compute different functions:

    X  func0=add, func2=mul   ->  f = (a+b+1) * (a+b+2) + (a+b)
    Y  func0=mul, func2=add   ->  c = a*b ; f = ((c+1) + (c+2)) * c

Different binaries per ``func_id`` give different hashids **and** different expected
values. The second half is what these cases turn on: the two callables in
``dynamic_register/test_dynamic_register.py`` differ only by an unused child entry and
therefore compute the same result, so a dispatch that ran the wrong one produces the value
that test asserts. Here the two results differ everywhere, and a wrong-callable dispatch is
reported as such rather than as an ordinary numeric mismatch.

Each case registers every callable and allocates every argument buffer before ``init()``,
which is the supported ordering: ``share_memory_()`` regions created in the parent after
the fork are not mapped in the chip child. Every dispatch gets its own input values, so a
stale-buffer read cannot pass as a correct result.
"""

from __future__ import annotations

import os

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, ChipCallable
from simpler.worker import Worker

from simpler_setup import TaskArgsBuilder, TensorArg
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.scene_test import _build_l3_task_args

_RUNTIME = "tensormap_and_ringbuffer"
_KERNELS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels",
)
_ORCH_SIG = [D.IN, D.IN, D.OUT]
_SIZE = 128 * 128
_TOL = 1e-4

#: One input pair per dispatch slot. Distinct per slot, and chosen so X's and Y's expected
#: values differ for every pair.
_INPUTS = [(2.0, 3.0), (5.0, 7.0), (1.5, 2.5), (4.0, 6.0), (3.0, 8.0), (2.5, 9.0)]


def _build_callable(platform: str, *, swap: bool) -> ChipCallable:
    """Compile the vector_example orchestration and its three AIV kernels.

    ``swap`` exchanges the binaries at ``func_id`` 0 and 2.
    """
    from simpler.task_interface import CoreCallable  # noqa: PLC0415

    from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    inc_dirs = kc.get_orchestration_include_dirs(_RUNTIME)
    orch_bytes = kc.compile_orchestration(
        runtime_name=_RUNTIME,
        source_path=os.path.join(_KERNELS, "orchestration", "example_orchestration.cpp"),
    )

    def _aiv(name: str) -> bytes:
        raw = kc.compile_incore(
            os.path.join(_KERNELS, "aiv", name),
            core_type="aiv",
            pto_isa_root=pto_isa_root,
            extra_include_dirs=inc_dirs,
        )
        return raw if platform.endswith("sim") else extract_text_section(raw)

    add = CoreCallable.build(signature=[D.IN, D.IN, D.OUT], binary=_aiv("kernel_add.cpp"))
    add_scalar = CoreCallable.build(signature=[D.IN, D.OUT], binary=_aiv("kernel_add_scalar.cpp"))
    mul = CoreCallable.build(signature=[D.IN, D.IN, D.OUT], binary=_aiv("kernel_mul.cpp"))

    first, third = (mul, add) if swap else (add, mul)
    return ChipCallable.build(
        signature=_ORCH_SIG,
        func_name="aicpu_orchestration_entry",
        binary=orch_bytes,
        children=[(0, first), (1, add_scalar), (2, third)],
    )


@pytest.fixture(scope="module")
def isolation_callables(st_platform):
    """``{"X": ChipCallable, "Y": ChipCallable}``, compiled once for the module."""
    return {"X": _build_callable(st_platform, swap=False), "Y": _build_callable(st_platform, swap=True)}


def _golden(kind: str, a: float, b: float) -> float:
    """Expected constant output of callable ``kind`` for constant inputs ``a``, ``b``."""
    if kind == "X":
        s = a + b
        return (s + 1.0) * (s + 2.0) + s
    c = a * b
    return ((c + 1.0) + (c + 2.0)) * c


def _check(got: torch.Tensor, kind: str, a: float, b: float, slot: str) -> None:
    """Assert ``got`` is callable ``kind``'s output, naming the other callable if it ran."""
    want = _golden(kind, a, b)
    if (got - want).abs().max().item() <= _TOL * max(1.0, abs(want)):
        return

    other_kind = "Y" if kind == "X" else "X"
    other = _golden(other_kind, a, b)
    if (got - other).abs().max().item() <= _TOL * max(1.0, abs(other)):
        pytest.fail(
            f"dispatch {slot} named callable {kind} but produced callable "
            f"{other_kind}'s result ({other:g}); expected {want:g}"
        )
    if not bool(torch.isfinite(got).all()):
        pytest.fail(f"dispatch {slot} ({kind}) produced non-finite output; expected {want:g}")
    if float(got.abs().max()) == 0.0:
        pytest.fail(f"dispatch {slot} ({kind}) left its output buffer untouched; expected {want:g}")
    pytest.fail(f"dispatch {slot} ({kind}) produced {got.flatten()[0].item():g}; expected {want:g}")


def _make_args(a: float, b: float) -> TaskArgsBuilder:
    return TaskArgsBuilder(
        TensorArg("a", torch.full((_SIZE,), a, dtype=torch.float32).share_memory_()),
        TensorArg("b", torch.full((_SIZE,), b, dtype=torch.float32).share_memory_()),
        TensorArg("f", torch.zeros(_SIZE, dtype=torch.float32).share_memory_()),
    )


def test_the_two_callables_are_distinguishable():
    """X and Y differ for every input pair, and ``_check`` names a wrong-callable result.

    Runs no kernel: it pins the property the hardware cases depend on. If X and Y ever
    computed the same value, every case below would pass whichever callable ran, which is
    the coverage gap these cases exist to close.
    """
    for a, b in _INPUTS:
        assert _golden("X", a, b) != _golden("Y", a, b), f"X and Y agree on ({a}, {b})"

    a, b = _INPUTS[0]
    _check(torch.full((16,), _golden("X", a, b)), "X", a, b, slot="0:X")

    with pytest.raises(BaseException, match="produced callable Y's result"):
        _check(torch.full((16,), _golden("Y", a, b)), "X", a, b, slot="0:X")
    with pytest.raises(BaseException, match="left its output buffer untouched"):
        _check(torch.zeros(16), "X", a, b, slot="0:X")
    with pytest.raises(BaseException, match="non-finite"):
        _check(torch.full((16,), float("nan")), "X", a, b, slot="0:X")


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.parametrize(
    "sequence",
    [
        pytest.param(["X"], id="single"),
        pytest.param(["X", "X"], id="same_twice"),
        pytest.param(["X", "Y"], id="two_callables"),
        pytest.param(["X", "Y", "X"], id="redispatch_after_other"),
        pytest.param(["X", "Y", "X", "Y", "X", "Y"], id="alternating"),
    ],
)
def test_dispatch_runs_the_named_callable(st_platform, st_device_ids, isolation_callables, sequence):
    """One worker serves several callables, including a callable dispatched again later."""
    device = int(st_device_ids[0])
    worker = Worker(level=3, device_ids=[device], num_sub_workers=0, platform=st_platform, runtime=_RUNTIME)
    # Guard from construction, not from init(): a register() or init() that raises
    # still owes teardown. init() "raises after a bounded rollback that reaps the
    # children it forked best-effort" and leaves the worker FAILED ("close this
    # Worker and create a new one"); the debt is re-driven only by a later close().
    # Skipping it would keep the device held and fail every later case on that card.
    try:
        handles = {kind: worker.register(isolation_callables[kind]) for kind in sorted(set(sequence))}
        if len(handles) > 1:
            hashids = {kind: h.hashid for kind, h in handles.items()}
            assert len(set(hashids.values())) == len(hashids), (
                f"callables that compute different functions share a hashid: {hashids}"
            )

        argsets = [_make_args(*_INPUTS[i % len(_INPUTS)]) for i in range(len(sequence))]
        chip_args = [_build_l3_task_args(a, _ORCH_SIG, worker)[0] for a in argsets]

        worker.init()
        for i, kind in enumerate(sequence):
            handle, args = handles[kind], chip_args[i]

            def orch(o, _args, _cfg, _handle=handle, _args_i=args):
                o.submit_next_level(_handle, _args_i, CallConfig(), worker=0)

            worker.run(orch)
            a, b = _INPUTS[i % len(_INPUTS)]
            _check(argsets[i].f, kind, a, b, slot=f"{i}:{kind}")
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_dispatch_runs_the_named_callable_fresh_worker_each(st_platform, st_device_ids, isolation_callables):
    """A fresh single-callable worker per dispatch produces each callable's own result.

    Baseline for the shared-worker cases above: it exercises the same three dispatches with
    no callable reuse, so a failure here is environmental rather than a reuse defect.
    """
    sequence = ["X", "Y", "X"]
    for i, kind in enumerate(sequence):
        worker = Worker(
            level=3, device_ids=[int(st_device_ids[0])], num_sub_workers=0, platform=st_platform, runtime=_RUNTIME
        )
        a, b = _INPUTS[i % len(_INPUTS)]
        argset = _make_args(a, b)
        try:
            handle = worker.register(isolation_callables[kind])
            args, _ = _build_l3_task_args(argset, _ORCH_SIG, worker)
            worker.init()

            def orch(o, _args, _cfg, _handle=handle, _args_i=args):
                o.submit_next_level(_handle, _args_i, CallConfig(), worker=0)

            worker.run(orch)
        finally:
            worker.close()
        _check(argset.f, kind, a, b, slot=f"{i}:{kind}")
