# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise sync selection against serialized Host submission records."""

import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HEADER = struct.Struct("<4IQ8i")


@pytest.fixture(scope="module")
def tool(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++ compiler required")
    directory = tmp_path_factory.mktemp("half-sync")
    production = ROOT / "tests/fdwic/dsv41_scheduler/production"
    includes = ["-I" + str(ROOT / "src/common"), "-I" + str(ROOT / "src/common/task_interface")]
    binary = directory / "record-tool"
    subprocess.run(
        [compiler, "-std=c++17", *includes, str(production / "record_tool.cpp"), "-o", str(binary)], check=True
    )
    probe = directory / "layout.cpp"
    probe.write_text(
        '#include <iostream>\n#include "record.h"\nint main() { '
        "static_assert(sizeof(dsv41::Header) == 56); "
        "std::cout << sizeof(dsv41::Operand); }\n"
    )
    subprocess.run(
        [compiler, "-std=c++17", *includes, "-I" + str(production), str(probe), "-o", str(directory / "layout")],
        check=True,
    )
    return binary, int(subprocess.check_output([str(directory / "layout")], text=True))


def capture(path, operand_size, tasks):
    data = struct.pack("<QQ", 0x4453563431523031, operand_size)
    data += HEADER.pack(0, 0, 0, 0, 0, -1, -1, -1, 1, 0, 0, -1, 0)
    for task_id, (blocks, sync, kernels, operation) in enumerate(tasks):
        data += HEADER.pack(operation, 0, 1, 1, task_id, *kernels, blocks, sync, 1, -1, 0)
        data += struct.pack("<QQ", 1234 + task_id, task_id)
    path.write_bytes(data)
    return data


def convert(tool, source, target, first, last, option="--half-sync-start"):
    return json.loads(
        subprocess.check_output([str(tool), option, str(source), str(first), str(last), str(target)], text=True)
    )


def test_half_sync_preserves_bytes_and_is_idempotent(tool, tmp_path):
    binary, size = tool
    vector = (-1, 7, -1)
    tasks = [(8, 0, vector, 2)] * 2 + [
        (8, 1, vector, 2),
        (128, 0, vector, 2),
        (33, 0, (5, -1, -1), 2),
        (33, 0, (5, 6, 7), 2),
        (32, 0, (5, 6, 7), 2),
        (64, 0, vector, 2),
        (8, 0, vector, 2),
        (1, 0, vector, 2),
        (8, 0, (-1, -1, -1), 3),
        (8, 0, vector, 2),
    ]
    source, target = tmp_path / "source.bin", tmp_path / "target.bin"
    original = capture(source, size, tasks)
    report = convert(binary, source, target, 1, 10)
    assert report["spmd"] == 8
    assert report["target"] == report["sync_start_after"] == 4
    assert report["sync_start_before"] == 1
    assert report["over_limit"] == 3
    assert report["changed"] == 3
    assert len(set(report["selected_task_ids"])) == 3
    assert set(report["selected_task_ids"]) <= {1, 6, 7, 8}
    expected = bytearray(original)
    for task_id in report["selected_task_ids"]:
        struct.pack_into("<i", expected, 16 + HEADER.size + task_id * (HEADER.size + 16) + 40, 1)
    assert target.read_bytes() == expected
    second = tmp_path / "second.bin"
    assert convert(binary, target, second, 1, 10)["changed"] == 0
    assert second.read_bytes() == target.read_bytes()
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "flags,blocks,expected",
    [
        ([0] * 5, [8] * 5, 3),
        ([1, 1, 1, 0], [8] * 4, 3),
        ([0] * 3, [128] * 3, 0),
        ([0] * 3, [1] * 3, 0),
    ],
)
def test_half_sync_rounding_and_constraints(tool, tmp_path, flags, blocks, expected):
    binary, size = tool
    source, target = tmp_path / "source.bin", tmp_path / "target.bin"
    tasks = [(block, flag, (-1, 7, -1), 2) for block, flag in zip(blocks, flags)]
    capture(source, size, tasks)
    report = convert(binary, source, target, 0, len(tasks) - 1)
    assert report["sync_start_after"] == expected
    if expected == sum(flags):
        assert source.read_bytes() == target.read_bytes()


def test_all_sync_cap64_preserves_scope_data_and_physical_capacity(tool, tmp_path):
    binary, size = tool
    vector = (-1, 7, -1)
    tasks = [
        (8, 0, vector, 2),  # outside the selected slice
        (63, 0, vector, 2),
        (64, 0, vector, 2),
        (65, 0, vector, 2),
        (32, 0, (5, 6, 7), 2),
        (33, 0, (5, 6, 7), 2),
        (1, 0, vector, 2),
        (8, 0, (-1, -1, -1), 3),
        (8, 1, vector, 2),
        (8, 0, vector, 2),  # outside the selected slice
    ]
    source, target = tmp_path / "source.bin", tmp_path / "all.bin"
    original = capture(source, size, tasks)
    option = "--all-sync-start-cap64"
    report = convert(binary, source, target, 1, 8, option)
    assert report["selected_task_ids"] == [1, 2, 4]
    assert report["sync_start_before"] == 1
    assert report["sync_start_after"] == report["target"] == 4
    assert report["over_limit"] == 2
    expected = bytearray(original)
    for task in (1, 2, 4):
        struct.pack_into("<i", expected, 16 + HEADER.size + task * (HEADER.size + 16) + 40, 1)
    assert target.read_bytes() == expected
    assert source.read_bytes() == original
    again = tmp_path / "again.bin"
    assert convert(binary, target, again, 1, 8, option)["changed"] == 0
    assert again.read_bytes() == target.read_bytes()
