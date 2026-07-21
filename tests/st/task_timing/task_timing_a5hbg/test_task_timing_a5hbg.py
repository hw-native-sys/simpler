# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end ST for selective task-timing slots on a5 host_build_graph (#1325).

a5's host_build_graph uses the legacy add_task orchestration API and flat
Runtime::Task (no PTO2TaskDescriptor / RT2 scheduler), so it has its own
compatibility path for the timing slots. This drives a two-task chain
(`c = a + b` tagged slot 0, `out = c + b` tagged slot 1) with L2 swimlane OFF
and asserts both task_slot markers plus a correct golden — the legacy-path
counterpart of the RT2 e2e in tests/st/task_timing_slots/test_task_timing_e2e.py.

Args are CPU tensors: the host_build_graph orchestration owns device memory
(device_malloc + copy_to_device) and copies the output back via
record_tensor_pair, so no worker.malloc is used here.
"""

import os
import re

import pytest
from simpler.task_interface import ArgDirection, CallConfig, ChipCallable, ChipStorageTaskArgs, CoreCallable
from simpler.worker import Worker

from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

_HERE = os.path.dirname(os.path.abspath(__file__))
# Reuse the a5 host_build_graph dump_args add kernel (out = a + b, raw float* ABI
# matching this test's add_task orchestration).
_ADD_KERNEL = os.path.join(_HERE, "..", "..", "a5", "host_build_graph", "dump_args", "kernels", "aiv", "kernel_add.cpp")
_SIZE = 128 * 128
_STRACE_RE = re.compile(r"\[STRACE\] .*\bname=(?P<name>\S+)\b.*\bts=(?P<ts>\d+)\b.*\bdur=(?P<dur>\d+)")


def _slot_spans(captured: str, slot: int) -> list:
    name = f"simpler_run.runner_run.device_wall.task_slot_{slot}"
    out = []
    for line in captured.splitlines():
        m = _STRACE_RE.search(line)
        if m and m["name"] == name:
            out.append((int(m["ts"]), int(m["dur"])))
    return out


def _build_chip_callable(platform: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    runtime = "host_build_graph"
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = kc.get_orchestration_include_dirs(runtime)

    kernel_bytes = kc.compile_incore(
        source_path=_ADD_KERNEL, core_type="aiv", pto_isa_root=pto_isa_root, extra_include_dirs=include_dirs
    )
    if not platform.endswith("sim"):
        from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415

        kernel_bytes = extract_text_section(kernel_bytes)

    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime, source_path=os.path.join(_HERE, "kernels/orchestration/task_timing_a5hbg_orch.cpp")
    )
    add = CoreCallable.build(signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT], binary=kernel_bytes)
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT],
        func_name="build_task_timing_a5hbg_graph",
        binary=orch_bytes,
        children=[(0, add)],
        config_name="",  # host_build_graph has no orchestration config function
    )


@pytest.mark.platforms(["a5sim", "a5"])
@pytest.mark.runtime("host_build_graph")
@pytest.mark.device_count(1)
def test_a5hbg_task_timing_slots_emit_markers(st_platform, st_device_ids, capfd):
    import torch  # noqa: PLC0415

    worker = Worker(
        level=2, platform=st_platform, runtime="host_build_graph", device_id=int(st_device_ids[0]), aicpu_thread_num=3
    )
    chip_handle = worker.register(_build_chip_callable(st_platform))
    worker.init()
    try:
        a = torch.full((_SIZE,), 1.0, dtype=torch.float32)
        b = torch.full((_SIZE,), 2.0, dtype=torch.float32)
        out = torch.zeros(_SIZE, dtype=torch.float32)  # orch copies dev_out back here
        expected = a + 2 * b

        args = ChipStorageTaskArgs()
        from simpler_setup.torch_interop import make_tensor_arg  # noqa: PLC0415

        args.add_tensor(make_tensor_arg(a))
        args.add_tensor(make_tensor_arg(b))
        args.add_tensor(make_tensor_arg(out))

        config = CallConfig()
        config.enable_l2_swimlane = False  # slots must work with swimlane OFF
        config.block_dim = 3

        assert worker.run(chip_handle, args, config) is None
        assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5), (
            f"a5 hbg chain output diverged; max |expected - out| = {float(torch.max(torch.abs(out - expected))):.3e}"
        )
    finally:
        worker.close()

    err = capfd.readouterr().err
    slot0 = _slot_spans(err, 0)
    slot1 = _slot_spans(err, 1)
    assert slot0, "no task_slot_0 marker; a5 host_build_graph dispatch/finish fold or host readback/emit regressed."
    assert slot1, "no task_slot_1 marker; the second tagged task's slot was not emitted."
    assert slot0[0][1] > 0 and slot1[0][1] > 0, f"slot durations must be > 0: slot0={slot0}, slot1={slot1}"

    # t1 consumes t0's output, so t1's dispatch follows t0's finish.
    fin0 = slot0[0][0] + slot0[0][1]
    disp1 = slot1[0][0]
    assert disp1 >= fin0, f"expected dispatch(slot1)={disp1} >= finish(slot0)={fin0} (dependency ordering)"
