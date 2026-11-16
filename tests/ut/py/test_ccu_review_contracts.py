# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from types import SimpleNamespace

import pytest
from simpler.ccu import ChipCcuBinding
from simpler.task_interface import ChipCallable, CoreCallable


@pytest.mark.parametrize("index", [0, 1])
def test_extracted_child_repackages_without_padding_or_ccu_package(index):
    children = [(i, CoreCallable.build(signature=[], binary=bytes([i + 1]))) for i in range(2)]
    parent = ChipCallable.build(
        signature=[], func_name="parent", binary=b"orchestration", children=children, ccu_modules=[b"module"]
    )
    extracted = parent.child(index)
    rebuilt = ChipCallable.build(signature=[], func_name="rebuilt", binary=b"new", children=[(0, extracted)])
    assert rebuilt.child_count == 1
    assert rebuilt.child(0).binary_size == 1


def test_prepare_successor_preserves_active_generation_and_launch_waits_for_stop():
    operations = []
    native = SimpleNamespace(
        prepare_ccu_run=lambda comm, registration, gen, dispatch, allocation: operations.append(
            ("prepare", gen, dispatch)
        ),
        control_ccu_run=lambda comm, registration, op, gen: operations.append((op, gen)) or 0,
    )
    binding = ChipCcuBinding(native, 10, 9, 1)
    first = binding.prepare(100, 7)
    first.begin()
    second = binding.prepare(101, 7)
    assert operations == [("prepare", 1, 100), (1, 1), ("prepare", 2, 101)]
    with pytest.raises(RuntimeError, match="confirmed stop"):
        second.begin()
    assert operations[-1] == ("prepare", 2, 101)
    first.finish()
    second.begin()
    second.finish()
    assert operations[-5:] == [(3, 1), (7, 1), (1, 2), (3, 2), (7, 2)]


def test_unconfirmed_stop_blocks_prepared_successor():
    def control(comm, registration, op, gen):
        if op == 3:
            raise RuntimeError("stop unconfirmed")
        return 0

    binding = ChipCcuBinding(SimpleNamespace(prepare_ccu_run=lambda *args: 0, control_ccu_run=control), 10, 9, 1)
    first, second = binding.prepare(100, 7), binding.prepare(101, 7)
    first.begin()
    with pytest.raises(RuntimeError, match="stop unconfirmed"):
        first.finish()
    with pytest.raises(RuntimeError, match="confirmed stop"):
        second.begin()


@pytest.mark.parametrize("stop_confirmed", [True, False])
def test_exit_timeout_requires_stop_confirmation_before_admitting_successor(monkeypatch, stop_confirmed):
    operations = []

    def control(_comm, _registration, operation, generation):
        operations.append((operation, generation))
        return int(operation == 3 or (operation == 9 and not stop_confirmed))

    times = iter((0.0, 0.5, 1.0))
    monkeypatch.setattr("simpler.ccu.time", SimpleNamespace(monotonic=lambda: next(times)))
    binding = ChipCcuBinding(SimpleNamespace(prepare_ccu_run=lambda *_: 0, control_ccu_run=control), 10, 9, 1)
    first, successor = binding.prepare(100, 7), binding.prepare(101, 7)
    first.begin()

    if stop_confirmed:
        first.finish()
        assert operations == [(1, 1), (3, 1), (3, 1), (9, 1), (7, 1)]
        assert not binding.active
        successor.begin()
    else:
        with pytest.raises(RuntimeError, match="stop is unconfirmed"):
            first.finish()
        assert operations == [(1, 1), (3, 1), (3, 1), (9, 1), (4, 1)]
        assert binding.active
        with pytest.raises(RuntimeError, match="confirmed stop"):
            successor.begin()
