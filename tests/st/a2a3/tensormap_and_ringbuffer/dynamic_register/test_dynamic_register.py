#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end ST for post-init Worker.register(ChipCallable) at L3.

Exercises the _CTRL_REGISTER IPC path end-to-end: parent stages a
ChipCallable in shared memory after init(), broadcasts CTRL_REGISTER to
every chip child, the child mmaps + prepares, and the resulting cid is
indistinguishable from a pre-init registration when used in run().

The UT suite (tests/ut/py/test_worker/test_host_worker.py) already covers
the facade-level paths (lock guard, cid overflow, lambda rejection, run
race detection, shm name generator). This file's job is to prove the
bytes actually traverse shm to the chip child and prepare succeeds —
which only a real (sim or device) chip child can confirm.
"""

import os

import pytest
import torch
from _task_interface import MAX_REGISTERED_CALLABLE_IDS  # pyright: ignore[reportMissingImports]
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, ChipCallable
from simpler.worker import Worker

from simpler_setup import TaskArgsBuilder, Tensor
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.scene_test import _build_l3_task_args

_RUNTIME = "tensormap_and_ringbuffer"
_HERE = os.path.dirname(os.path.abspath(__file__))
_KERNELS = os.path.join(
    _HERE,
    "..",
    "..",
    "..",
    "..",
    "..",
    "examples",
    "a2a3",
    "tensormap_and_ringbuffer",
    "vector_example",
    "kernels",
)
_ORCH_SRC = os.path.join(_KERNELS, "orchestration", "example_orchestration.cpp")
_AIV_ADD = os.path.join(_KERNELS, "aiv", "kernel_add.cpp")
_AIV_ADD_SCALAR = os.path.join(_KERNELS, "aiv", "kernel_add_scalar.cpp")
_AIV_MUL = os.path.join(_KERNELS, "aiv", "kernel_mul.cpp")

_ORCH_SIG = [D.IN, D.IN, D.OUT]


def _build_vector_callable(platform: str) -> ChipCallable:
    """Compile the vector_example orchestration + 3 AIV kernels.

    Mirrors how SceneTestCase._compile_chip_callable_from_spec assembles
    a ChipCallable, but inline so the test can call register() on it both
    before and after init().
    """
    from simpler.task_interface import CoreCallable  # noqa: PLC0415

    from simpler_setup.elf_parser import extract_text_section  # noqa: PLC0415
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    kc = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    inc_dirs = kc.get_orchestration_include_dirs(_RUNTIME)

    orch_bytes = kc.compile_orchestration(runtime_name=_RUNTIME, source_path=_ORCH_SRC)

    def _aiv(path: str) -> bytes:
        raw = kc.compile_incore(path, core_type="aiv", pto_isa_root=pto_isa_root, extra_include_dirs=inc_dirs)
        return raw if platform.endswith("sim") else extract_text_section(raw)

    add = CoreCallable.build(signature=[D.IN, D.IN, D.OUT], binary=_aiv(_AIV_ADD))
    add_scalar = CoreCallable.build(signature=[D.IN, D.OUT], binary=_aiv(_AIV_ADD_SCALAR))
    mul = CoreCallable.build(signature=[D.IN, D.IN, D.OUT], binary=_aiv(_AIV_MUL))

    return ChipCallable.build(
        signature=_ORCH_SIG,
        func_name="aicpu_orchestration_entry",
        binary=orch_bytes,
        children=[(0, add), (1, add_scalar), (2, mul)],
    )


def _make_args(a: float, b: float) -> TaskArgsBuilder:
    size = 128 * 128
    return TaskArgsBuilder(
        Tensor("a", torch.full((size,), a, dtype=torch.float32).share_memory_()),
        Tensor("b", torch.full((size,), b, dtype=torch.float32).share_memory_()),
        Tensor("f", torch.zeros(size, dtype=torch.float32).share_memory_()),
    )


def _golden(a: float, b: float) -> float:
    # Matches the orchestration: f = (a+b+1) * (a+b+2) + (a+b)
    s = a + b
    return (s + 1) * (s + 2) + s


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_register_after_init_then_run(st_platform, st_device_ids):
    """Happy path: register one cid pre-init and one cid post-init, run both.

    Proves the post-init register path delivers a usable cid — the chip
    child mmapped the shm, prepared the ChipCallable, and dispatch finds
    it on the next submit. Both cids execute against the same kernel
    binaries and must produce numerically identical outputs.
    """
    chip_callable = _build_vector_callable(st_platform)

    worker = Worker(
        level=3,
        device_ids=[int(st_device_ids[0])],
        num_sub_workers=0,
        platform=st_platform,
        runtime=_RUNTIME,
    )
    cid_pre = worker.register(chip_callable)

    # Pre-allocate both runs' tensors BEFORE Worker.init() so the
    # share_memory_() mappings are inherited by the forked chip child.
    # share_memory_ regions created after fork in the parent are not visible
    # to the chip child, so dispatch on those would segfault.
    a, b = 2.0, 3.0
    expected = _golden(a, b)
    args_pre = _make_args(a, b)
    args_post = _make_args(a, b)
    chip_args_pre, output_names_pre = _build_l3_task_args(args_pre, _ORCH_SIG)
    chip_args_post, output_names_post = _build_l3_task_args(args_post, _ORCH_SIG)
    assert output_names_pre == ["f"] and output_names_post == ["f"]

    worker.init()
    try:
        config = CallConfig()
        config.block_dim = 3
        config.aicpu_thread_num = 4

        # 1. Run cid_pre once to force _start_hierarchical (forks chip
        #    children, runs the CTRL_PREPARE prewarm loop). This puts the
        #    chip child into _run_chip_main_loop, the only state in which
        #    a CTRL_REGISTER broadcast can be ACKed.
        def orch_pre(o, _args, _cfg):
            o.submit_next_level(cid_pre, chip_args_pre, config)

        worker.run(orch_pre)
        got_pre = args_pre.f
        assert torch.allclose(got_pre, torch.full_like(got_pre, expected), rtol=1e-5, atol=1e-5), (
            f"cid_pre={cid_pre}: expected {expected}, got {got_pre[:4].tolist()}..."
        )

        # 2. Now do the post-fork dynamic register. The parent stages bytes
        #    in shm and broadcasts CTRL_REGISTER; the child mmaps and calls
        #    prepare_callable_from_blob. cid_post is unknown to the
        #    CoW-inherited registry on the child side — only the IPC path
        #    can deliver it.
        cid_post = worker.register(chip_callable)
        assert cid_post != cid_pre

        # 3. Run with cid_post. If CTRL_REGISTER delivered correctly, the
        #    child has the cid prepared; otherwise dispatch will fail.
        def orch_post(o, _args, _cfg):
            o.submit_next_level(cid_post, chip_args_post, config)

        worker.run(orch_post)
        got_post = args_post.f
        assert torch.allclose(got_post, torch.full_like(got_post, expected), rtol=1e-5, atol=1e-5), (
            f"cid_post={cid_post}: expected {expected}, got {got_post[:4].tolist()}..."
        )
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.device_count(2)
@pytest.mark.runtime(_RUNTIME)
def test_register_after_init_parallel_broadcast(st_platform, st_device_ids):
    """Two chip children, post-init register broadcasts to both in parallel.

    Asserts that the registered cid runs successfully on each chip — proving
    the C++ broadcast (one std::thread per WorkerThread) delivers the bytes
    to every chip's mailbox and each prepare_callable_from_blob runs without
    racing against the others.
    """
    chip_callable = _build_vector_callable(st_platform)
    device_ids = [int(d) for d in st_device_ids[:2]]
    worker = Worker(
        level=3,
        device_ids=device_ids,
        num_sub_workers=0,
        platform=st_platform,
        runtime=_RUNTIME,
    )
    cid_pre = worker.register(chip_callable)
    a, b = 2.0, 3.0
    expected = _golden(a, b)
    # Pre-allocate args for each chip (chip_id = block group). The
    # vector_example orchestration partitions the input across cores, so a
    # single args bundle works for both chips' first-run trigger; the
    # second-run uses the post-registered cid.
    args_pre = _make_args(a, b)
    args_post = _make_args(a, b)
    chip_args_pre, _ = _build_l3_task_args(args_pre, _ORCH_SIG)
    chip_args_post, _ = _build_l3_task_args(args_post, _ORCH_SIG)

    worker.init()
    try:
        config = CallConfig()
        config.block_dim = 3
        config.aicpu_thread_num = 4

        def orch_pre(o, _a, _c):
            o.submit_next_level(cid_pre, chip_args_pre, config)

        worker.run(orch_pre)
        assert torch.allclose(args_pre.f, torch.full_like(args_pre.f, expected), rtol=1e-5, atol=1e-5)

        # Now broadcast CTRL_REGISTER to BOTH chip mailboxes in parallel.
        cid_post = worker.register(chip_callable)

        def orch_post(o, _a, _c):
            o.submit_next_level(cid_post, chip_args_post, config)

        worker.run(orch_post)
        assert torch.allclose(args_post.f, torch.full_like(args_post.f, expected), rtol=1e-5, atol=1e-5)
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_register_cid_overflow_post_init(st_platform, st_device_ids):
    """Saturate the cid table pre-init, then verify post-init register hits
    the same ``MAX_REGISTERED_CALLABLE_IDS`` ceiling.

    Confirms the cid bookkeeping is shared between the pre-init and the new
    dynamic-register entry points (and that the error message is
    protocol-aware so the operator sees the same diagnostic in both paths).
    """
    chip_callable = _build_vector_callable(st_platform)
    worker = Worker(
        level=3,
        device_ids=[int(st_device_ids[0])],
        num_sub_workers=0,
        platform=st_platform,
        runtime=_RUNTIME,
    )
    # Fill the registry pre-init with sub fn lambdas (cheap, no device cost).
    for _ in range(MAX_REGISTERED_CALLABLE_IDS - 1):
        worker.register(lambda args: None)
    cid_chip = worker.register(chip_callable)  # last slot
    assert len(worker._callable_registry) == MAX_REGISTERED_CALLABLE_IDS

    a, b = 2.0, 3.0
    args_pre = _make_args(a, b)
    chip_args_pre, _ = _build_l3_task_args(args_pre, _ORCH_SIG)

    worker.init()
    try:
        config = CallConfig()
        config.block_dim = 3
        config.aicpu_thread_num = 4

        def orch_pre(o, _a, _c):
            o.submit_next_level(cid_chip, chip_args_pre, config)

        worker.run(orch_pre)

        # The very next dynamic register hits the table ceiling.
        with pytest.raises(RuntimeError, match="MAX_REGISTERED_CALLABLE_IDS"):
            worker.register(chip_callable)
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_register_unregister_register_runs_each_time(st_platform, st_device_ids):
    """register → run → unregister → register again → run.

    Proves the IPC unregister path works end-to-end: after CTRL_UNREGISTER
    propagates to the chip child, the cid slot is freed and a subsequent
    dynamic register on a fresh cid prepares + dispatches correctly.
    """
    chip_callable = _build_vector_callable(st_platform)
    worker = Worker(
        level=3,
        device_ids=[int(st_device_ids[0])],
        num_sub_workers=0,
        platform=st_platform,
        runtime=_RUNTIME,
    )
    cid_pre = worker.register(chip_callable)

    a, b = 2.0, 3.0
    expected = _golden(a, b)
    # Three runs total — preallocate three args bundles BEFORE init() so
    # the share_memory_ mappings are inherited by the forked chip child.
    args_one = _make_args(a, b)
    args_two = _make_args(a, b)
    args_three = _make_args(a, b)
    chip_args_one, _ = _build_l3_task_args(args_one, _ORCH_SIG)
    chip_args_two, _ = _build_l3_task_args(args_two, _ORCH_SIG)
    chip_args_three, _ = _build_l3_task_args(args_three, _ORCH_SIG)

    worker.init()
    try:
        config = CallConfig()
        config.block_dim = 3
        config.aicpu_thread_num = 4

        # 1. Trigger fork via cid_pre to put the chip child into the main loop.
        def orch_one(o, _args, _cfg):
            o.submit_next_level(cid_pre, chip_args_one, config)

        worker.run(orch_one)
        assert torch.allclose(args_one.f, torch.full_like(args_one.f, expected), rtol=1e-5, atol=1e-5)

        # 2. Dynamic register a second cid, run it.
        cid_dyn = worker.register(chip_callable)

        def orch_two(o, _args, _cfg):
            o.submit_next_level(cid_dyn, chip_args_two, config)

        worker.run(orch_two)
        assert torch.allclose(args_two.f, torch.full_like(args_two.f, expected), rtol=1e-5, atol=1e-5)

        # 3. Unregister cid_dyn — CTRL_UNREGISTER hits the chip child,
        #    which calls cw.unregister_callable(cid_dyn).
        worker.unregister(cid_dyn)
        assert cid_dyn not in worker._callable_registry

        # 4. Re-register and run again. The freed slot is reused — cid
        #    allocation is `len(registry)` and the unregister popped one
        #    entry, so the next register lands on the same index as the
        #    just-unregistered cid. This is the slot-recycling property
        #    that unregister exists for.
        cid_again = worker.register(chip_callable)
        assert cid_again == cid_dyn, "unregister should free the cid slot for reuse"

        def orch_three(o, _args, _cfg):
            o.submit_next_level(cid_again, chip_args_three, config)

        worker.run(orch_three)
        assert torch.allclose(args_three.f, torch.full_like(args_three.f, expected), rtol=1e-5, atol=1e-5)
    finally:
        worker.close()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
