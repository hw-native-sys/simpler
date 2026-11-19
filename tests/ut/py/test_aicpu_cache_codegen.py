# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check the onboard cache instructions emitted for each device architecture."""

import platform
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _compile_arm_assembly(tmp_path, source, arch):
    compiler = shutil.which("g++") if platform.machine() == "aarch64" else shutil.which("aarch64-linux-gnu-g++")
    if compiler is None:
        pytest.skip("AArch64 C++ compiler is unavailable")
    assembly = tmp_path / "cache_ops.s"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-S",
            f"-I{ROOT}",
            f"-I{ROOT / 'src/common'}",
            f"-I{ROOT / 'src' / arch / 'platform/include'}",
            f"-I{ROOT / 'src/common/platform/include'}",
            str(source),
            "-o",
            str(assembly),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return assembly.read_text()


@pytest.mark.parametrize("arch,needs_maintenance", [("a2a3", True), ("a5", False)])
def test_onboard_cache_maintenance_codegen(tmp_path, arch, needs_maintenance):
    code = _compile_arm_assembly(tmp_path, ROOT / "src" / arch / "platform/onboard/aicpu/cache_ops.cpp", arch)
    assert bool(re.search(r"\bdc\s+civac\b", code)) == needs_maintenance
    assert bool(re.search(r"\bdc\s+cvac\b", code)) == needs_maintenance
    assert bool(re.search(r"\bdsb\s+sy\b", code)) == needs_maintenance
    if not needs_maintenance:
        assert not re.search(r"\b(?:dsb|dmb|isb)\b", code)


@pytest.mark.parametrize(
    ("operation", "body", "instruction", "count"),
    [
        (
            "observe_counter",
            "int32_t value; CounterLocalOps{}.load_i32(address, value); return value;",
            r"\bldar\b",
            1,
        ),
        (
            "publish_counter",
            "CounterLocalOps{}.store_i32(address, 1); return 0;",
            r"\bstlr\b",
            1,
        ),
        (
            "flush_payload",
            "PayloadLocalOps{}.flush(reinterpret_cast<void *>(address), 64); return 0;",
            r"\b(?:ldar|stlr)\b",
            0,
        ),
        (
            "publish_result",
            "return aicpu_publish_run_terminal(address, 1, DeviceRunVerdict::Ok, 0, "
            "DeviceRunCodeSource::None, nullptr, 0);",
            r"\b(?:ldar|stlr)\b",
            0,
        ),
    ],
)
def test_a5_callers_use_only_protocol_ordering(tmp_path, operation, body, instruction, count):
    source = tmp_path / f"{operation}.cpp"
    source.write_text(
        '#include "aicpu/region_instance_view.h"\n'
        '#include "aicpu/device_run_result_aicpu.h"\n'
        '#include "src/a5/platform/onboard/aicpu/cache_ops.cpp"\n'
        f'extern "C" int32_t {operation}(uint64_t address) {{ {body} }}\n'
    )
    code = _compile_arm_assembly(tmp_path, source, "a5")
    assert len(re.findall(instruction, code)) == count
    assert not re.search(r"\bdc\s+(?:civac|cvac)\b|\b(?:dsb|dmb|isb)\b", code)
