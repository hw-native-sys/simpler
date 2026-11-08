# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
import struct

import pytest
from simpler.callable_identity import build_chip_callable_descriptor, compute_callable_hashid
from simpler.task_interface import CcuFunction, ChipCallable, CoreCallable


def module_bytes():
    binary = bytearray(64)
    binary[:7] = b"\x7fELF\x02\x01\x01"
    binary[16] = 3
    binary[20] = 1
    return bytes(binary)


def build(functions=()):
    return ChipCallable.build(
        signature=[],
        func_name="orch",
        binary=b"orch",
        children=[(0, CoreCallable.build([], b"core"))],
        ccu_functions=list(functions),
    )


def test_embedded_functions_round_trip_without_module_file():
    binary = module_bytes()
    chip = build([CcuFunction.build(binary, "copy"), CcuFunction.build(binary, "reduce")])
    raw = ctypes.string_at(chip.buffer_ptr(), chip.buffer_size())
    restored = ChipCallable.from_bytes(raw)
    assert restored.ccu_function_count == 2
    assert [(function.entry, function.binary) for function in restored.ccu_functions] == [
        ("copy", binary),
        ("reduce", binary),
    ]
    assert restored.child_count == 1
    assert restored.child_func_id(0) == 0
    assert restored.child(0).buffer_size() == chip.child(0).buffer_size()
    assert build().ccu_function_count == 0


def test_ccu_order_and_module_content_participate_in_identity():
    first = CcuFunction.build(module_bytes(), "copy")
    second = CcuFunction.build(module_bytes(), "reduce")
    changed = CcuFunction.build(module_bytes() + b"changed", "copy")
    identities = [
        compute_callable_hashid(build_chip_callable_descriptor(target=build(functions), platform="a5"))
        for functions in ([first, second], [second, first], [changed, second])
    ]
    assert len(set(identities)) == 3


@pytest.mark.parametrize("entry", ["", "../copy", "copy\x00", "copy\n", "0copy", "x" * 256])
def test_invalid_entry_rejected(entry):
    with pytest.raises(ValueError, match="CCU entry"):
        CcuFunction.build(module_bytes(), entry)


@pytest.mark.parametrize("binary", [b"", b"\x7fELF", b"not-a-module" * 10])
def test_invalid_binary_rejected(binary):
    with pytest.raises(ValueError, match="ELF64"):
        CcuFunction.build(binary, "copy")


def test_old_wire_version_and_truncated_payload_rejected():
    chip = build([CcuFunction.build(module_bytes(), "copy")])
    raw = ctypes.string_at(chip.buffer_ptr(), chip.buffer_size())
    for truncated in (raw[:10], raw[:-1]):
        with pytest.raises(ValueError):
            ChipCallable.from_bytes(truncated)
    legacy = bytearray(raw)
    struct.pack_into("<I", legacy, 17556, 0)
    with pytest.raises(ValueError, match="wire version"):
        ChipCallable.from_bytes(bytes(legacy))
