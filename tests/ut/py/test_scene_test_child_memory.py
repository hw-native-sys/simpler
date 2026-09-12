# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Explicit L2 child memory has stable allocation and round semantics."""

import sys
from unittest.mock import patch

import pytest
import torch

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg

scene = sys.modules["simpler_setup.scene_test"]


def test_clone_preserves_child_memory_declarations():
    args = TaskArgsBuilder(TensorArg("x", torch.ones(4), child_memory=True))
    clone = args.clone()
    assert clone.specs[0].child_memory
    assert clone.x.data_ptr() != args.x.data_ptr()


def test_builder_add_tensor_accepts_child_memory():
    args = TaskArgsBuilder()
    args.add_tensor("x", torch.ones(4), child_memory=True)
    assert args.specs[0].child_memory


class FakeWorker:
    def __init__(self):
        from simpler.buffer import mint_owner_instance_id

        self.owner = mint_owner_instance_id()
        self.created = []
        self.freed = []
        self.uploads = []
        self.downloads = []
        self.data = {}
        self.fail_copy = False

    def malloc(self, size):
        from simpler.buffer import wrap_device_malloc

        buf = wrap_device_malloc(0x10000 + len(self.created) * 0x1000, size, self.owner, len(self.created) + 1, "L2")
        self.created.append(buf)
        return buf

    def free(self, buf):
        self.freed.append(buf)

    def copy_to(self, buf, host):
        if self.fail_copy:
            raise RuntimeError("copy failed")
        self.uploads.append(buf)
        self.data[buf.identity] = host.clone()

    def copy_from(self, host, buf):
        self.downloads.append(buf)
        host.copy_(self.data[buf.identity])

    def make_tensor_arg(self, host, *, shapes, dtype):
        from simpler.buffer import AccessMode, BackendKind, wrap_fork_inherited

        return wrap_fork_inherited(
            host.data_ptr(),
            host.numel() * host.element_size(),
            self.owner,
            100 + host.data_ptr(),
            access=AccessMode.READWRITE,
            backend_kind=BackendKind.FORK_SHM,
        ).tensor(shapes, dtype)


def test_child_memory_directions_empty_and_lifo():
    from simpler.task_interface import ArgDirection as D

    args = TaskArgsBuilder(
        TensorArg("x", torch.ones(4), True),
        TensorArg("y", torch.ones(4), True),
        TensorArg("z", torch.zeros(4), True),
        TensorArg("empty", torch.empty(0), True),
        TensorArg("staged", torch.ones(4)),
    )
    worker = FakeWorker()
    with scene._child_memory_args(worker, args, [D.IN, D.INOUT, D.OUT, D.IN, D.IN]) as child_args:
        assert len(worker.created) == 3
        assert worker.uploads == worker.created[:2]
        # Zero-shaped wire Tensors are rejected by the existing transport;
        # the owner skips their device allocation and leaves them host-staged.
        assert "empty" not in child_args.tensors
        nonempty = TaskArgsBuilder(*(spec for spec in args.specs if spec.name != "empty"))
        _chip_args, outputs = scene._build_l2_ref_args(nonempty, [D.IN, D.INOUT, D.OUT, D.IN], worker, child_args)
        assert outputs == ["y", "z"]
    assert worker.freed == worker.created[::-1]
    child_args.release()
    assert len(worker.freed) == 3


def test_build_args_rejects_a_count_the_signature_would_misalign():
    """A skipped empty tensor shifts every later argument, so reject rather than dispatch."""
    from simpler.task_interface import ArgDirection as D

    from simpler_setup.scene_test import ChildMemoryTaskArgs

    worker = FakeWorker()
    with ChildMemoryTaskArgs(worker) as child_args:
        child_args.add("x", torch.ones(4), D.IN)
        child_args.add("empty", torch.empty(0), D.IN)
        assert len(child_args.tensors) == 1
        with pytest.raises(ValueError, match="empty tensor cannot be child memory"):
            child_args.build_args(expected_count=2)
        assert child_args.build_args(expected_count=1).tensor_count() == 1


@pytest.mark.parametrize("rounds", [1, 2, 3])
@pytest.mark.parametrize("child_memory", [False, True])
@pytest.mark.parametrize("skip_golden", [False, True])
def test_round_state_and_final_copyback(rounds, child_memory, skip_golden):
    from simpler.task_interface import ArgDirection as D

    class Case(SceneTestCase):
        CALLABLE = {"orchestration": {"signature": [D.INOUT, D.OUT]}}
        CASES = []

        def generate_args(self, _params):
            self.args = TaskArgsBuilder(
                TensorArg("state", torch.ones(4), child_memory=child_memory),
                TensorArg("out", torch.zeros(4)),
            )
            return self.args

        def compute_golden(self, args, _params):
            args.state.add_(1)
            args.out.copy_(args.state)

    case = Case()
    worker = FakeWorker()
    worker.register = lambda _: 1
    runs = []

    def run(*_args, **_kwargs):
        state = worker.data[worker.created[0].identity] if child_memory else case.args.state
        assert torch.equal(case.args.out, torch.zeros(4))
        state.add_(1)
        case.args.out.copy_(state)
        runs.append(state.clone())

    worker.run = run
    with patch.object(Case, "_build_config", return_value=object()):
        case._run_and_validate_l2(worker, object(), {}, rounds=rounds, skip_golden=skip_golden)
    assert len(runs) == rounds
    assert torch.equal(runs[-1], torch.full((4,), float(rounds + 1 if child_memory else 2)))
    assert len(worker.created) == int(child_memory)
    assert len(worker.uploads) == int(child_memory)
    assert len(worker.downloads) == int(child_memory and not skip_golden)
    assert worker.freed == worker.created


def test_partial_construction_and_execution_failure_release():
    from simpler.task_interface import ArgDirection as D

    args = TaskArgsBuilder(TensorArg("x", torch.ones(4), True), TensorArg("bad", torch.ones(2, 3).T, True))
    worker = FakeWorker()
    with pytest.raises(ValueError, match="contiguous"):
        scene._child_memory_args(worker, args, [D.IN, D.IN])
    assert worker.freed == worker.created
    assert args.specs[0].value is args.x
    worker = FakeWorker()
    worker.fail_copy = True
    with pytest.raises(RuntimeError, match="copy failed"):
        scene._child_memory_args(worker, TaskArgsBuilder(args.specs[0]), [D.IN])
    assert worker.freed == worker.created
    worker = FakeWorker()
    with pytest.raises(RuntimeError, match="execution"):
        with scene._child_memory_args(worker, TaskArgsBuilder(args.specs[0]), [D.IN]):
            raise RuntimeError("execution")
    assert worker.freed == worker.created


def test_invalid_direction_and_alias_rejected():
    from simpler.task_interface import ArgDirection as D

    host = torch.ones(8)
    for args, sig, match in [
        (TaskArgsBuilder(TensorArg("x", host, True)), [D.SCALAR], "unsupported direction"),
        (TaskArgsBuilder(TensorArg("x", host[:4], True), TensorArg("y", host[2:])), [D.IN, D.IN], "alias"),
    ]:
        worker = FakeWorker()
        with pytest.raises(ValueError, match=match):
            scene._child_memory_args(worker, args, sig)
        assert not worker.created


def test_streaming_owner_does_not_retain_weights():
    import weakref

    from simpler.task_interface import ArgDirection as D

    from simpler_setup.scene_test import ChildMemoryTaskArgs

    worker = FakeWorker()
    with ChildMemoryTaskArgs(worker) as child_args:
        weight = torch.ones(4)
        reference = weakref.ref(weight)
        child_args.add("weight", weight, D.IN)
        del weight
        assert reference() is None
        assert child_args.build_args().tensor_count() == 1


def test_l3_rejects_child_memory_before_allocation():
    class Case(SceneTestCase):
        CASES = []

        def generate_args(self, params):
            return TaskArgsBuilder(TensorArg("x", torch.ones(4), child_memory=True))

    worker = FakeWorker()
    with pytest.raises(ValueError, match="require L2"):
        Case()._run_and_validate_l3(worker, {}, {}, {})
    assert not worker.created
