#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end ST for post-start Worker.register(ChipCallable) at L3.

Exercises the _CTRL_REGISTER IPC path end-to-end: parent stages a
ChipCallable in shared memory after child startup, broadcasts CTRL_REGISTER to
every chip child, the child mmaps + prepares, and the resulting
CallableHandle is indistinguishable from a pre-start preparation when used
in run().

The UT suite (tests/ut/py/test_worker/test_host_worker.py) already covers
the facade-level paths (lock guard, capacity overflow, lambda rejection, run
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


def _build_vector_callable(platform: str, *, extra_unused_child: bool = False) -> ChipCallable:
    """Compile the vector_example orchestration + 3 AIV kernels.

    Mirrors how SceneTestCase._compile_chip_callable_from_spec assembles
    a ChipCallable, but inline so the test can call register_callable() on it both
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

    children = [(0, add), (1, add_scalar), (2, mul)]
    if extra_unused_child:
        children.append((99, add))

    return ChipCallable.build(
        signature=_ORCH_SIG,
        func_name="aicpu_orchestration_entry",
        binary=orch_bytes,
        children=children,
    )


def _unique_py_callable(index: int):
    def fn(args, _index=index):
        return _index

    return fn


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
def test_prepare_new_identity_after_start_then_run(st_platform, st_device_ids):
    """Happy path: prepare one identity pre-start and another post-start.

    Proves the post-start control path delivers a usable handle for a
    previously unseen hashid. Both identities execute equivalent kernels and
    must produce numerically identical outputs.
    """
    chip_callable = _build_vector_callable(st_platform)
    post_callable = _build_vector_callable(st_platform, extra_unused_child=True)

    worker = Worker(
        level=3,
        device_ids=[int(st_device_ids[0])],
        num_sub_workers=0,
        platform=st_platform,
        runtime=_RUNTIME,
    )
    pre_handle = worker.register(chip_callable)

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

        # 1. Run pre_handle once against the snapshot-prepared chip callable.
        #    init() already forked the chip children and prepared the startup
        #    snapshot, so they are in _run_chip_main_loop — the only state in
        #    which the CTRL_REGISTER broadcast in step 2 can be ACKed.
        def orch_pre(o, _args, _cfg):
            o.submit_next_level(pre_handle, chip_args_pre, config)

        worker.run(orch_pre)
        got_pre = args_pre.f
        assert torch.allclose(got_pre, torch.full_like(got_pre, expected), rtol=1e-5, atol=1e-5), (
            f"pre_handle={pre_handle.hashid}: expected {expected}, got {got_pre[:4].tolist()}..."
        )

        # 2. Now do the post-start dynamic prepare. The parent stages bytes
        #    in shm and broadcasts CTRL_REGISTER; the child mmaps and calls
        #    prepare_callable_from_blob. post_handle is unknown to the
        #    CoW-inherited registry on the child side — only the IPC path
        #    can deliver it.
        post_handle = worker.register(post_callable)
        assert post_handle.hashid != pre_handle.hashid

        # 3. Run with post_handle. If CTRL_REGISTER delivered correctly, the
        #    child has the identity prepared; otherwise dispatch will fail.
        def orch_post(o, _args, _cfg):
            o.submit_next_level(post_handle, chip_args_post, config)

        worker.run(orch_post)
        got_post = args_post.f
        assert torch.allclose(got_post, torch.full_like(got_post, expected), rtol=1e-5, atol=1e-5), (
            f"post_handle={post_handle.hashid}: expected {expected}, got {got_post[:4].tolist()}..."
        )
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.device_count(2)
@pytest.mark.runtime(_RUNTIME)
def test_prepare_new_identity_after_start_parallel_broadcast(st_platform, st_device_ids):
    """Two chip children, post-start prepare broadcasts to both in parallel.

    Asserts that the prepared handle runs successfully on each chip — proving
    the C++ broadcast (one std::thread per WorkerThread) delivers the bytes
    to every chip's mailbox and each prepare_callable_from_blob runs without
    racing against the others.
    """
    chip_callable = _build_vector_callable(st_platform)
    post_callable = _build_vector_callable(st_platform, extra_unused_child=True)
    device_ids = [int(d) for d in st_device_ids[:2]]
    worker = Worker(
        level=3,
        device_ids=device_ids,
        num_sub_workers=0,
        platform=st_platform,
        runtime=_RUNTIME,
    )
    pre_handle = worker.register(chip_callable)
    a, b = 2.0, 3.0
    expected = _golden(a, b)
    # Pre-allocate args for each chip (chip_id = block group). The
    # vector_example orchestration partitions the input across cores, so a
    # single args bundle works for both chips' first-run trigger; the
    # second-run uses the post-start prepared handle.
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
            o.submit_next_level(pre_handle, chip_args_pre, config)

        worker.run(orch_pre)
        assert torch.allclose(args_pre.f, torch.full_like(args_pre.f, expected), rtol=1e-5, atol=1e-5)

        # Now broadcast CTRL_REGISTER to BOTH chip mailboxes in parallel.
        post_handle = worker.register(post_callable)

        def orch_post(o, _a, _c):
            o.submit_next_level(post_handle, chip_args_post, config)

        worker.run(orch_post)
        assert torch.allclose(args_post.f, torch.full_like(args_post.f, expected), rtol=1e-5, atol=1e-5)
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_prepare_capacity_overflow_post_start(st_platform, st_device_ids):
    """Saturate callable capacity pre-start, then verify post-start prepare hits
    the same ``MAX_REGISTERED_CALLABLE_IDS`` ceiling for a new hashid.

    Confirms the public capacity guard is shared between pre-start preparation
    and the post-start control path (and that the error message is
    protocol-aware so the operator sees the same diagnostic in both paths).
    """
    chip_callable = _build_vector_callable(st_platform)
    worker = Worker(
        level=3,
        device_ids=[int(st_device_ids[0])],
        # A sub worker gives the LOCAL_PYTHON fillers below a valid resolver, so
        # they exercise the shared capacity ceiling as eligible registrations
        # (not inert ones) alongside the chip callable.
        num_sub_workers=1,
        platform=st_platform,
        runtime=_RUNTIME,
    )
    # Fill the registry pre-start with distinct sub fn identities (cheap, no
    # device cost).
    for i in range(MAX_REGISTERED_CALLABLE_IDS - 1):
        worker.register(_unique_py_callable(i))
    chip_handle = worker.register(chip_callable)  # final capacity entry

    a, b = 2.0, 3.0
    args_pre = _make_args(a, b)
    chip_args_pre, _ = _build_l3_task_args(args_pre, _ORCH_SIG)

    worker.init()
    try:
        config = CallConfig()
        config.block_dim = 3
        config.aicpu_thread_num = 4

        def orch_pre(o, _a, _c):
            o.submit_next_level(chip_handle, chip_args_pre, config)

        worker.run(orch_pre)

        # The very next dynamic prepare of a new identity hits the capacity
        # ceiling. Re-preparing ``chip_callable`` itself would only create
        # another handle to the existing identity.
        with pytest.raises(RuntimeError, match="MAX_REGISTERED_CALLABLE_IDS"):
            worker.register(_build_vector_callable(st_platform, extra_unused_child=True))
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_duplicate_prepare_same_hashid_survives_one_unregister(st_platform, st_device_ids):
    """prepare same hashid twice, unregister one handle, run the other.

    This is the hashid-specific post-start path: the second
    ``register_callable(same_chip_callable)`` must return a distinct handle for
    the same hashid. Unregistering one handle must only drop that public
    handle; the remaining handle must still dispatch successfully.
    """
    chip_callable = _build_vector_callable(st_platform)
    worker = Worker(
        level=3,
        device_ids=[int(st_device_ids[0])],
        num_sub_workers=0,
        platform=st_platform,
        runtime=_RUNTIME,
    )
    pre_handle = worker.register(chip_callable)

    a, b = 2.0, 3.0
    expected = _golden(a, b)
    # Two runs total — preallocate both args bundles BEFORE init() so
    # the share_memory_ mappings are inherited by the forked chip child.
    args_one = _make_args(a, b)
    args_two = _make_args(a, b)
    chip_args_one, _ = _build_l3_task_args(args_one, _ORCH_SIG)
    chip_args_two, _ = _build_l3_task_args(args_two, _ORCH_SIG)

    worker.init()
    try:
        config = CallConfig()
        config.block_dim = 3
        config.aicpu_thread_num = 4

        # 1. Trigger fork via pre_handle to put the chip child into the main loop.
        def orch_one(o, _args, _cfg):
            o.submit_next_level(pre_handle, chip_args_one, config)

        worker.run(orch_one)
        assert torch.allclose(args_one.f, torch.full_like(args_one.f, expected), rtol=1e-5, atol=1e-5)

        # 2. Prepare the same callable after start. This returns another
        # public handle for the same hashid, not a new identity.
        duplicate_handle = worker.register(chip_callable)
        assert duplicate_handle.hashid == pre_handle.hashid
        assert duplicate_handle.digest == pre_handle.digest
        assert duplicate_handle._handle_id != pre_handle._handle_id

        # 3. Drop the first handle. The child must keep the prepared identity
        # alive for duplicate_handle.
        worker.unregister(pre_handle)

        with pytest.raises(KeyError, match="not live"):
            worker.run(lambda o, _args, _cfg: o.submit_next_level(pre_handle, chip_args_one, config))

        def orch_two(o, _args, _cfg):
            o.submit_next_level(duplicate_handle, chip_args_two, config)

        worker.run(orch_two)
        assert torch.allclose(args_two.f, torch.full_like(args_two.f, expected), rtol=1e-5, atol=1e-5)

        # 4. Dropping the final handle invalidates it through the public API.
        worker.unregister(duplicate_handle)
        with pytest.raises(KeyError, match="not live"):
            worker.run(lambda o, _args, _cfg: o.submit_next_level(duplicate_handle, chip_args_two, config))
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3sim"])
@pytest.mark.device_count(1)
@pytest.mark.runtime(_RUNTIME)
def test_unregister_last_handle_allows_reprepare_same_hashid(st_platform, st_device_ids):
    """prepare → run → unregister final handle → prepare same identity again.

    Proves the IPC unregister path works end-to-end: after CTRL_UNREGISTER
    propagates to the chip child, the old handle is invalid and a subsequent
    post-start prepare of that identity materializes a usable handle again.
    """
    chip_callable = _build_vector_callable(st_platform)
    post_callable = _build_vector_callable(st_platform, extra_unused_child=True)
    worker = Worker(
        level=3,
        device_ids=[int(st_device_ids[0])],
        num_sub_workers=0,
        platform=st_platform,
        runtime=_RUNTIME,
    )
    pre_handle = worker.register(chip_callable)

    a, b = 2.0, 3.0
    expected = _golden(a, b)
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

        def orch_one(o, _args, _cfg):
            o.submit_next_level(pre_handle, chip_args_one, config)

        worker.run(orch_one)
        assert torch.allclose(args_one.f, torch.full_like(args_one.f, expected), rtol=1e-5, atol=1e-5)

        dyn_handle = worker.register(post_callable)

        def orch_two(o, _args, _cfg):
            o.submit_next_level(dyn_handle, chip_args_two, config)

        worker.run(orch_two)
        assert torch.allclose(args_two.f, torch.full_like(args_two.f, expected), rtol=1e-5, atol=1e-5)

        worker.unregister(dyn_handle)
        with pytest.raises(KeyError, match="not live"):
            worker.run(lambda o, _args, _cfg: o.submit_next_level(dyn_handle, chip_args_two, config))

        # Re-prepare the same hashid after its final handle was dropped.
        again_handle = worker.register(post_callable)
        assert again_handle.hashid == dyn_handle.hashid
        assert again_handle.digest == dyn_handle.digest
        assert again_handle._handle_id != dyn_handle._handle_id

        def orch_three(o, _args, _cfg):
            o.submit_next_level(again_handle, chip_args_three, config)

        worker.run(orch_three)
        assert torch.allclose(args_three.f, torch.full_like(args_three.f, expected), rtol=1e-5, atol=1e-5)
    finally:
        worker.close()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
