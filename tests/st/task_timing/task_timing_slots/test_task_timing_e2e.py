# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end ST for selective task-timing slots (issue #1325).

A two-task chain (`c = a + b` tagged slot 0, `out = c + b` tagged slot 1) is
run with chip swimlane OFF. The contract being verified:

    * both `chip.run.runner_run.device_wall.task_slot_0` and `..._1`
      [STRACE] markers are present with strictly positive duration — proving
      the whole path (Arg tag -> descriptor pad -> Scheduler dispatch/finish
      fold -> host H2D reset / D2H readback -> marker emit) works with the
      swimlane disabled;
    * the output is numerically correct, so the markers come from a real run
      and not an early-error path.

Reuses the sibling vector_add AIV kernel (out = src0 + src1) as func_id 0.
Markers go to stderr via the unified host logger, captured with ``capfd``.
"""

import os
import re

import pytest
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))

N_ROWS = 128
N_COLS = 128
N_ELEMS = N_ROWS * N_COLS
NBYTES = N_ELEMS * 4  # float32

_STRACE_RE = re.compile(r"\[STRACE\] .*\bname=(?P<name>\S+)\b.*\bts=(?P<ts>\d+)\b.*\bdur=(?P<dur>\d+)")


def _slot_spans(captured: str, slot: int) -> list:
    """Return (ts, dur) for every task_slot_<slot> [STRACE] marker."""
    name = f"chip.run.runner_run.device_wall.task_slot_{slot}"
    out = []
    for line in captured.splitlines():
        m = _STRACE_RE.search(line)
        if m and m["name"] == name:
            out.append((int(m["ts"]), int(m["dur"])))
    return out


def _build_chip_callable(
    platform: str, func_name: str = "task_timing_orchestration", runtime: str = "tensormap_and_ringbuffer"
) -> ChipCallable:
    kc = KernelCompiler(platform=platform)

    pto_isa_root = ensure_pto_isa_root()
    include_dirs = kc.get_orchestration_include_dirs(runtime)
    # The case owns its kernels: one names a single runtime's Tensor, and this helper
    # is driven under both. They also differ by architecture — a5's ccec/pto-isa needs
    # the qualified `pto::Stride` spelling, which the a2a3 copy does not use.
    arch = "a5" if platform.startswith("a5") else "a2a3"
    aiv_kernel = os.path.join(_HERE, "kernels", runtime, "aiv", f"vector_add_{arch}.cpp")
    kernel_bytes = kc.compile_incore(
        source_path=aiv_kernel,
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=include_dirs,
    )
    if not platform.endswith("sim"):
        from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415

        kernel_bytes = extract_text_section(kernel_bytes)

    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(_HERE, "kernels", runtime, "task_timing_orch.cpp"),
    )
    core_callable = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        binary=kernel_bytes,
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        func_name=func_name,
        binary=orch_bytes,
        children=[(0, core_callable)],
        config_name="task_timing_orchestration_config",
    )


def _drive(
    platform: str, device_id: int, func_name: str, b_multiplier: int, runtime: str = "tensormap_and_ringbuffer"
) -> None:
    """Run one orchestration; markers go to stderr for the caller's capfd.

    `b_multiplier` is how many times `b` is added to `a` by the chain, so the
    golden check ties the emitted markers to a real, correct run.
    """
    import torch  # noqa: PLC0415

    worker = Worker(level=2, platform=platform, runtime=runtime, device_id=device_id)
    chip_callable = _build_chip_callable(platform, func_name, runtime)
    chip_handle = worker.register(chip_callable)
    worker.init()
    try:
        host_a = torch.full((N_ROWS, N_COLS), 1.0, dtype=torch.float32)
        host_b = torch.full((N_ROWS, N_COLS), 2.0, dtype=torch.float32)
        expected = host_a + b_multiplier * host_b

        a_h = worker.malloc(NBYTES)
        b_h = worker.malloc(NBYTES)
        out_h = worker.malloc(NBYTES)
        worker.copy_to(a_h, host_a)
        worker.copy_to(b_h, host_b)

        args = TaskArgs()
        args.add_tensor(a_h.tensor(shapes=(N_ROWS, N_COLS), dtype=DataType.FLOAT32), TensorArgType.INPUT)
        args.add_tensor(b_h.tensor(shapes=(N_ROWS, N_COLS), dtype=DataType.FLOAT32), TensorArgType.INPUT)
        args.add_tensor(out_h.tensor(shapes=(N_ROWS, N_COLS), dtype=DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)

        config = CallConfig()
        config.enable_chip_swimlane = False  # slots must work with swimlane OFF

        assert worker.run(chip_handle, args, config) is None

        host_out = torch.zeros(N_ROWS, N_COLS, dtype=torch.float32)
        worker.copy_from(host_out, out_h)
        worker.free(a_h)
        worker.free(b_h)
        worker.free(out_h)
        assert torch.allclose(host_out, expected, rtol=1e-5, atol=1e-5), (
            f"{func_name} output diverged; max |expected - out| = "
            f"{float(torch.max(torch.abs(host_out - expected))):.3e}"
        )
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim", "a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(1)
@pytest.mark.manual(["a2a3sim", "a5sim"])
def test_distinct_slots_emit_markers(st_platform, st_device_ids, capfd, drain_host_log):
    # Two-task chain: t0 -> slot 0, t1 -> slot 1. out = a + 2b.
    _drive(st_platform, int(st_device_ids[0]), "task_timing_orchestration", 2)
    err = drain_host_log(capfd)

    slot0 = _slot_spans(err, 0)
    slot1 = _slot_spans(err, 1)
    assert slot0, "no task_slot_0 marker; dispatch/finish fold or host readback/emit regressed (swimlane OFF)."
    assert slot1, "no task_slot_1 marker; the second tagged task's slot was not emitted."
    assert slot0[0][1] > 0 and slot1[0][1] > 0, f"slot durations must be > 0: slot0={slot0}, slot1={slot1}"

    # t1 consumes t0's output, so t1's dispatch follows t0's finish.
    fin0 = slot0[0][0] + slot0[0][1]
    disp1 = slot1[0][0]
    assert disp1 >= fin0, f"expected dispatch(slot1)={disp1} >= finish(slot0)={fin0} (dependency ordering)"


@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim", "a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(1)
@pytest.mark.manual(["a2a3sim", "a5sim"])
def test_duplicate_slot_merges_window(st_platform, st_device_ids, capfd, drain_host_log):
    dev = int(st_device_ids[0])

    # A 3-task chain (t0 -> t1 -> t2) all tagged slot 0. out = a + 3b. The three
    # tagged tasks must fold their dispatch/finish into a SINGLE slot-0 window
    # (min dispatch .. max finish), not three separate markers, and no other slot
    # is touched.
    _drive(st_platform, dev, "task_timing_dup_orchestration", 3)
    err = drain_host_log(capfd)

    slot0 = _slot_spans(err, 0)
    assert len(slot0) == 1, f"expected exactly ONE merged task_slot_0 marker, got {len(slot0)}: {slot0}"
    assert not _slot_spans(err, 1) and not _slot_spans(err, 2), (
        "no other slot should be emitted (all tasks used slot 0)"
    )
    # A complete (dispatch < finish) merged window. We do NOT compare its magnitude
    # against a separate single-task run: on sim, per-task dispatch->finish absorbs
    # large, highly variable scheduling/cold-start overhead (observed 50k..1M ns on
    # either task across runs), so any cross-run magnitude comparison is inherently
    # flaky. The min/max fold math is covered deterministically by the C++ unit test
    # (tests/ut/cpp/a2a3/runtime/tensormap_and_ringbuffer/test_task_timing_slots.cpp) and per-task dispatch/finish
    # ordering by test_distinct_slots_emit_markers.
    assert slot0[0][1] > 0, f"merged task_slot_0 must be a complete window (dispatch < finish), got {slot0}"


# a2a3 host_build_graph exercises a distinct fold path from tensormap_and_ringbuffer:
# the RT2 orch is identical, but hbg builds the graph on the host and its Scheduler
# folds dispatch via the PublishHandle (scheduler_context.h) rather than the inline
# tensormap dispatch. These two cases give that path direct e2e coverage.
@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim", "a5"])
@pytest.mark.runtime("host_build_graph")
@pytest.mark.device_count(1)
@pytest.mark.manual(["a2a3sim", "a5sim"])
def test_hbg_distinct_slots_emit_markers(st_platform, st_device_ids, capfd, drain_host_log):
    # Same two-task chain as test_distinct_slots_emit_markers, on the hbg path.
    _drive(st_platform, int(st_device_ids[0]), "task_timing_orchestration", 2, runtime="host_build_graph")
    err = drain_host_log(capfd)

    slot0 = _slot_spans(err, 0)
    slot1 = _slot_spans(err, 1)
    assert slot0, "no task_slot_0 marker; hbg dispatch/finish fold or host readback/emit regressed (swimlane OFF)."
    assert slot1, "no task_slot_1 marker; the second tagged task's slot was not emitted."
    assert slot0[0][1] > 0 and slot1[0][1] > 0, f"slot durations must be > 0: slot0={slot0}, slot1={slot1}"

    # t1 consumes t0's output, so t1's dispatch follows t0's finish.
    fin0 = slot0[0][0] + slot0[0][1]
    disp1 = slot1[0][0]
    assert disp1 >= fin0, f"expected dispatch(slot1)={disp1} >= finish(slot0)={fin0} (dependency ordering)"


@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim", "a5"])
@pytest.mark.runtime("host_build_graph")
@pytest.mark.device_count(1)
@pytest.mark.manual(["a2a3sim", "a5sim"])
def test_hbg_duplicate_slot_merges_window(st_platform, st_device_ids, capfd, drain_host_log):
    # Same 3-task same-slot merge as test_duplicate_slot_merges_window, on the hbg
    # path: min(dispatch)/max(finish) must fold into a single slot-0 window.
    _drive(st_platform, int(st_device_ids[0]), "task_timing_dup_orchestration", 3, runtime="host_build_graph")
    err = drain_host_log(capfd)

    slot0 = _slot_spans(err, 0)
    assert len(slot0) == 1, f"expected exactly ONE merged task_slot_0 marker, got {len(slot0)}: {slot0}"
    assert not _slot_spans(err, 1) and not _slot_spans(err, 2), (
        "no other slot should be emitted (all tasks used slot 0)"
    )
    assert slot0[0][1] > 0, f"merged task_slot_0 must be a complete window (dispatch < finish), got {slot0}"


_MIXED_KERNELS = os.path.join(
    _HERE, "..", "..", "..", "..", "tests", "st", "a2a3", "tensormap_and_ringbuffer", "mixed_example", "kernels"
)
_MATMUL_SIZE = 128
_TILE_ELEMS = _MATMUL_SIZE * _MATMUL_SIZE  # 16384


def _build_mix_chip_callable(platform: str) -> ChipCallable:
    """A single AIC+AIV0+AIV1 mixed task (matmul + add + mul), reusing the
    committed mixed_example kernels. 9-tensor mix signature."""
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = kc.get_orchestration_include_dirs(runtime)

    def incore(rel, core_type):
        b = kc.compile_incore(
            source_path=os.path.join(_MIXED_KERNELS, rel),
            core_type=core_type,
            pto_isa_root=pto_isa_root,
            extra_include_dirs=include_dirs,
        )
        if not platform.endswith("sim"):
            from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415

            b = extract_text_section(b)
        return b

    sig9 = [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT] * 3
    matmul = CoreCallable.build(signature=sig9, binary=incore("aic/kernel_matmul.cpp", "aic"))
    add = CoreCallable.build(signature=sig9, binary=incore("aiv/kernel_add.cpp", "aiv"))
    mul = CoreCallable.build(signature=sig9, binary=incore("aiv/kernel_mul.cpp", "aiv"))

    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime, source_path=os.path.join(_HERE, "kernels", runtime, "task_timing_orch.cpp")
    )
    return ChipCallable.build(
        signature=sig9,
        func_name="task_timing_mix_orchestration",
        binary=orch_bytes,
        children=[(0, matmul), (1, add), (2, mul)],
        config_name="task_timing_mix_orchestration_config",
    )


@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(1)
@pytest.mark.manual(["a2a3sim"])
def test_mix_task_aggregates_across_subtasks(st_platform, st_device_ids, capfd, drain_host_log):
    # One MIX task (AIC matmul + AIV0 add + AIV1 mul) tagged slot 0. All three
    # subtasks fold their dispatch/finish into slot 0 -> one complete window.
    import torch  # noqa: PLC0415

    worker = Worker(
        level=2,
        platform=st_platform,
        runtime="tensormap_and_ringbuffer",
        device_id=int(st_device_ids[0]),
    )
    chip_handle = worker.register(_build_mix_chip_callable(st_platform))
    worker.init()
    try:
        torch.manual_seed(42)
        A = torch.randn(_MATMUL_SIZE, _MATMUL_SIZE, dtype=torch.float32) * 0.01
        B = torch.randn(_MATMUL_SIZE, _MATMUL_SIZE, dtype=torch.float32) * 0.01
        D = torch.randn(_TILE_ELEMS, dtype=torch.float32) * 0.01
        E = torch.randn(_TILE_ELEMS, dtype=torch.float32) * 0.01
        G = torch.randn(_TILE_ELEMS, dtype=torch.float32) * 0.01
        H = torch.randn(_TILE_ELEMS, dtype=torch.float32) * 0.01
        gold_F = D + E
        gold_I = G * H

        nb = _TILE_ELEMS * 4
        bufs = {n: worker.malloc(nb) for n in "ABCDEFGHI"}
        for name, t in {"A": A.flatten(), "B": B.flatten(), "D": D, "E": E, "G": G, "H": H}.items():
            worker.copy_to(bufs[name], t.contiguous())

        # Arg order matches the ChipCallable signature: A,B->C (matmul), D,E->F (add), G,H->I (mul).
        _shapes = {"A": (_MATMUL_SIZE, _MATMUL_SIZE), "B": (_MATMUL_SIZE, _MATMUL_SIZE)}
        _outputs = {"C", "F", "I"}
        args = TaskArgs()
        for n in "ABCDEFGHI":
            tag = TensorArgType.OUTPUT_EXISTING if n in _outputs else TensorArgType.INPUT
            args.add_tensor(bufs[n].tensor(shapes=_shapes.get(n, (_TILE_ELEMS,)), dtype=DataType.FLOAT32), tag)

        config = CallConfig()
        config.enable_chip_swimlane = False
        assert worker.run(chip_handle, args, config) is None

        out_C = torch.zeros(_TILE_ELEMS, dtype=torch.float32)
        out_F = torch.zeros(_TILE_ELEMS, dtype=torch.float32)
        out_I = torch.zeros(_TILE_ELEMS, dtype=torch.float32)
        worker.copy_from(out_C, bufs["C"])
        worker.copy_from(out_F, bufs["F"])
        worker.copy_from(out_I, bufs["I"])
        for b in bufs.values():
            worker.free(b)
        # The AIV0/AIV1 subtasks are element-wise (layout-agnostic): their correct
        # output proves both AIV subtasks of the mixed task executed. The AIC
        # matmul's numerical correctness (cube tile layout) is validated by the
        # mixed_example scene test and is out of scope for this timing check.
        assert torch.allclose(out_F, gold_F, rtol=1e-3, atol=1e-3), "add (AIV0 subtask) diverged"
        assert torch.allclose(out_I, gold_I, rtol=1e-3, atol=1e-3), "mul (AIV1 subtask) diverged"
    finally:
        worker.close()

    err = drain_host_log(capfd)
    slot0 = _slot_spans(err, 0)
    assert len(slot0) == 1, f"expected exactly one task_slot_0 marker for the MIX task, got {slot0}"
    assert slot0[0][1] > 0, f"MIX task_slot_0 must be a complete window across its subtasks, got {slot0}"


@pytest.mark.platforms(["a2a3sim", "a2a3", "a5sim", "a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(1)
@pytest.mark.manual(["a2a3sim", "a5sim"])
def test_spmd_task_aggregates_across_threads(st_platform, st_device_ids, capfd, drain_host_log):
    # One SPMD task (block_num=8) tagged slot 0; blocks dispatch across multiple
    # scheduler threads and must reduce to one complete slot. out = a + b.
    _drive(st_platform, int(st_device_ids[0]), "task_timing_spmd_orchestration", 1)
    err = drain_host_log(capfd)

    slot0 = _slot_spans(err, 0)
    assert len(slot0) == 1, f"expected exactly one task_slot_0 marker for the SPMD task, got {slot0}"
    assert slot0[0][1] > 0, (
        f"SPMD task_slot_0 must be a complete window (dispatch<finish), got {slot0}; "
        "cross-thread min/max reduction produced no valid span."
    )
