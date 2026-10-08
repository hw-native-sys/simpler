# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The onboard early-enqueue consumer supplies the shared HBG callable's host control."""

import importlib.util
import sys
from contextlib import ExitStack
from pathlib import Path

import pytest
from simpler.buffer import AddressSpace, create_host_shared_buffer, mint_owner_instance_id, wrap_device_malloc
from simpler.task_interface import ArgDirection, TensorArgType, TensorTransfer


@pytest.mark.parametrize("device_chain", [False, True])
def test_early_enqueue_passes_separate_host_control(monkeypatch, device_chain):
    path = Path(__file__).resolve().parents[2] / "st/a2a3/host_build_graph/early_enqueue/test_early_enqueue.py"
    spec = importlib.util.spec_from_file_location("_early_enqueue_argument_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    owner = mint_owner_instance_id()
    with ExitStack() as cleanup:
        buffers = []
        for i in range(3):
            if device_chain and i != 1:
                buffer = wrap_device_malloc(0x100000 * (i + 1), module._SIZE * 4, owner, i + 1)
            else:
                buffer = create_host_shared_buffer(module._SIZE * 4, owner, i + 1)
            cleanup.callback(buffer.close)
            buffers.append(buffer)
        signature = module._CALLABLES["callables"][0]["orchestration"]["signature"]
        args = module._chip_args(buffers, signature, 123)
        assert args.tensor_count() == 4
        assert signature == [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT, ArgDirection.IN]
        assert args.scalar_count() == 1
        assert args.scalar(0) == 123
        assert args.transfer(1) == TensorTransfer.H2D
        assert args.transfer(3) == TensorTransfer.NONE
        assert args.tag(3) == TensorArgType.INPUT
        assert args.tensor(3).buffer.address_space == AddressSpace.HOST
        assert args.tensor(3).buffer == args.tensor(1).buffer
        assert args.tensor(3).shapes == args.tensor(1).shapes
        for i in (0, 2):
            assert args.transfer(i) == (TensorTransfer.NONE if device_chain else TensorTransfer.H2D)
