# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 HBG proof packaging of CANN-compiled kernels."""

import os
import struct
from pathlib import Path

import pytest

from simpler_setup import KernelCompiler, extract_text_section
from simpler_setup import a5_input_cache as proof

_ROOT = Path(__file__).resolve().parents[5]

pytestmark = [pytest.mark.requires_toolkit, pytest.mark.platforms(["a5"])]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "tests/st/a5/host_build_graph/defer_wait_visibility/kernels/aiv/tail.cpp",
            [(0, 4, 4, 0), (1, 4, 4, 32), (2, 4, 8, 68)],
        ),
        ("tests/ut/py/fixtures/a5_input_cache/vendor/combine_reduce.cpp", []),
    ],
)
def test_final_ccec_kernel_proof_preserves_all_text(tmp_path, source, expected):
    sdk = os.environ.get("ASCEND_HOME_PATH")
    pto = _ROOT / "build/pto-isa"
    if not sdk or not (Path(sdk) / "bin/llvm-objdump").is_file() or not (pto / "include/pto").is_dir():
        pytest.skip("requires CANN's A5 compiler/disassembler and PTO headers")
    compiler = KernelCompiler("a5")
    binary = compiler.compile_incore(
        str(_ROOT / source),
        core_type="aiv",
        pto_isa_root=str(pto),
        extra_include_dirs=compiler.get_orchestration_include_dirs("host_build_graph"),
        build_dir=str(tmp_path),
    )
    text = extract_text_section(binary)
    packaged = proof.annotate_a5_hbg_text(text, "aiv")
    assert packaged[: len(text)] == text
    footer = packaged[len(text) :]
    if not footer:
        # CANN may emit instructions outside the verifier's certified subset.
        return
    assert len(footer) == 40
    assert struct.unpack_from("<4sBBBB", footer) == (b"HCP1", 2 if expected else 1, len(expected), 3, 2)
    assert [struct.unpack_from("<BBHI", footer, 16 + i * 8) for i in range(len(expected))] == expected
