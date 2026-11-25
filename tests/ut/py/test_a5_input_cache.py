# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 HBG proof transfer, conditional paths and exact executable packaging."""

import os
import struct
from pathlib import Path

import pytest

from simpler_setup import a5_input_cache as proof
from simpler_setup.a5_completion import entry_is_synchronous
from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler

_ROOT = Path(__file__).resolve().parents[3]
_TENSOR = ("LD_IO.b64 X1, [X0], #8, #0", "LD_IO.b64 X2, [X1], #0, #0")


def _instructions(program):
    table = {}
    for i, line in enumerate(program):
        name, _, args = line.partition(" ")
        table[i * 4] = proof._Instruction(name, tuple(args.split(", ")) if args else (), 4)
    return table


def _package(monkeypatch, program, vector=()):
    monkeypatch.setenv("ASCEND_HOME_PATH", "cann")
    monkeypatch.setattr(
        proof,
        "_decode",
        lambda tool, code, flags, *, deadline, vector=False, body=vector: _instructions(body if vector else program),
    )
    original = bytes(4 * len(program))
    annotated = proof.annotate_a5_hbg_text(original, "aiv")
    assert annotated.startswith(original)
    return annotated[len(original) :]


def test_mte_and_bypass_only_still_require_real_tensor_slots(monkeypatch):
    footer = _package(
        monkeypatch, [*_TENSOR, "MOV_OUT_TO_UB_ALIGN_V2.b32 [X4], [X2], X6, X7", "LD_DEV.b64 X3, [X2], #0", "RET"]
    )
    assert struct.unpack_from("<4sBBBB", footer) == (b"HCP1", 1, 0, 2, 2)


def test_scaled_view_covers_scalar_read_write_and_register_index(monkeypatch):
    footer = _package(
        monkeypatch,
        [
            *_TENSOR,
            "LD_IO.b64 X3, [X1], #24, #0",
            "LD.b32 X4, [X2], X3, #0",
            "SHLI.b64 X3, #2",
            "ADD.s64 X2, X2, X3",
            "ST_IO.b32 X4, [X2], #4, #0",
            "RET",
        ],
    )
    assert struct.unpack_from("<4sBBBB", footer) == (b"HCP1", 2, 1, 2, 2)
    assert struct.unpack_from("<BBHI", footer, 16) == (1, 4, 8, 0)


def test_ranges_over_multiple_cache_lines_and_both_branch_arms(monkeypatch):
    accesses = [f"LD_IO.b64 X4, [X2], #{i}, #0" for i in range(0, 80, 8)]
    footer = _package(
        monkeypatch, [*_TENSOR, "JUMPCI #20", "LD_IO.b32 X5, [X2], #128, #0", "JUMPI #24", "NOP", *accesses, "RET"]
    )
    assert struct.unpack_from("<4sBBBB", footer) == (b"HCP1", 2, 2, 2, 2)
    assert struct.unpack_from("<BBHIBBHI", footer, 16) == (1, 0, 80, 0, 1, 0, 4, 128)


@pytest.mark.parametrize(
    "body",
    [
        ["LD_IO.b64 X3, [X2], #0, #0", "LD_IO.b32 X4, [X3], #0, #0"],
        ["LD_IO.b64 X3, [X0], #360, #0", "LD_IO.b32 X4, [X3], #0, #0"],
        ["ADD.s64 X3, X2, X4", "LD_IO.b32 X4, [X3], #0, #0"],
        ["ST_IO.b64 X5, [X1], #0, #0"],
        ["ST_IO.b64 X1, [X2], #0, #0"],
        ["MOV_UB_TO_OUT_ALIGN_V2 [X2], [X4], X6, X7", "LD_IO.b64 X3, [X1], #0, #0"],
        ["JUMPCI #16", "JUMPI #20", "ASM_UNKNOWN"],
        ["JUMP X5"],
        ["MOVI X31, #8"],
        ["MOVI X3, #32", "VF X3, #1"],
        ["PUSH_PB X1, X0, X0, X0"],
        ["MOV SYS_VA_BASE, X2"],
        ["MOV CTRL, X2"],
        ["LD_IO.b32 X1, [X2], #0, #1"],
        ["DCCI X2, #1, #2"],
        ["GET_BUF.S X3"],
        [f"LD_IO.b32 X3, [X2], #{i}, #0" for i in (0, 64, 128, 192)],
    ],
)
def test_unproven_access_uses_plain_executable(monkeypatch, body):
    assert _package(monkeypatch, [*_TENSOR, *body, "RET"]) == b""


def test_vf_checks_complete_target_extent(monkeypatch):
    program = ["MOV X1, PC", "ADDI X1, X1, #16", "VF X1, #2", "RET", "NOP", "NOP"]
    assert _package(monkeypatch, program, ["VADD.f32 V0, V0, V1, P1", "SEND"])
    assert not _package(monkeypatch, program, ["ASM_UNKNOWN", "SEND"])
    assert not _package(monkeypatch, program, ["SEND", "ASM_UNKNOWN"])
    assert not _package(monkeypatch, program, ["VLOOPV2_V310 S2, #3, #1, #1", "SEND"])


def test_partial_stack_overwrite_loses_saved_provenance(monkeypatch):
    program = [
        *_TENSOR,
        "ST_IO.b64 X2, [X30], #-8, #0",
        "ST_IO.b32 X4, [X30], #-4, #0",
        "LD_IO.b64 X3, [X30], #-8, #0",
        "LD_IO.b32 X4, [X3], #0, #0",
        "RET",
    ]
    assert not _package(monkeypatch, program)


def test_tool_or_decoding_failure_keeps_original_bytes(monkeypatch, tmp_path):
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    original = bytes.fromhex("00f06140")
    assert proof.annotate_a5_hbg_text(original, "aiv") == original
    assert proof.annotate_a5_hbg_text(original, "sim") == original


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "tests/st/a5/host_build_graph/defer_wait_visibility/kernels/aiv/tail.cpp",
            [(0, 4, 4, 0), (1, 4, 4, 32), (2, 4, 8, 68)],
        ),
        ("tests/fdwic/dsv41_scheduler/kernels/pypto_lib_latest/vendor/aiv/combine_reduce.cpp", []),
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
    assert packaged[:-40] == text
    footer = packaged[-40:]
    assert struct.unpack_from("<4sBBBB", footer) == (b"HCP1", 2 if expected else 1, len(expected), 3, 2)
    assert [struct.unpack_from("<BBHI", footer, 16 + i * 8) for i in range(len(expected))] == expected


@pytest.mark.parametrize(("aic_program", "expected"), [(["RET"], 3), (["UNSUPPORTED"], 2)])
def test_untyped_export_masks_only_verified_decoder_families(monkeypatch, aic_program, expected):
    monkeypatch.setenv("ASCEND_HOME_PATH", "cann")
    monkeypatch.setattr(
        proof,
        "_decode",
        lambda tool, code, flags, *, deadline, vector=False: _instructions(
            aic_program if flags == proof._ELF_FLAGS["aic"] else ["RET"]
        ),
    )
    text = bytes(4)
    assert proof.annotate_a5_hbg_text(text)[len(text) + 7] == expected


def test_decoder_hang_uses_remaining_upload_budget(monkeypatch, tmp_path):
    import time  # noqa: PLC0415

    tool = tmp_path / "bin/llvm-objdump"
    tool.parent.mkdir()
    tool.write_text("#!/bin/sh\nexec sleep 15\n")
    tool.chmod(0o755)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    text = bytes.fromhex("00f06140")
    started = time.monotonic()
    assert proof.annotate_a5_hbg_text(text, deadline=started + 0.1) == text
    # Allow a loaded CI host to spawn and reap the process. A missed timeout
    # still waits for the full 15-second sleep and fails this assertion.
    assert time.monotonic() - started < 5


@pytest.mark.parametrize(
    ("read", "synchronous"),
    [
        ("%v = load i32, ptr addrspace(1) %ctx, align 4", True),
        ("%field = getelementptr i8, ptr addrspace(1) %ctx, i64 4", False),
        ("%word = load i64, ptr addrspace(1) %ctx, align 8", False),
        ("call void @helper(ptr addrspace(1) %ctx)", False),
        ("%escaped = ptrtoint ptr addrspace(1) %args to i64", False),
        ("%wide = load <2 x i64>, ptr addrspace(1) %loc", False),
        ("%slot = getelementptr inbounds i64, ptr addrspace(1) %args, i64 %id", False),
        ('call void asm sideeffect "ld", ""()', False),
    ],
)
def test_completion_proof_only_accepts_public_context_field(read, synchronous):
    ir = f"""
target triple = "hiipu64-hisilicon-cce"
define dso_local void @kernel_entry(ptr addrspace(1) %args) {{
  %loc = getelementptr inbounds i64, ptr addrspace(1) %args, i64 48
  %local = load i64, ptr addrspace(1) %loc, align 8
  %slot49 = getelementptr inbounds i64, ptr addrspace(1) %args, i64 49
  %word49 = load i64, ptr addrspace(1) %slot49, align 8
  %ctx = inttoptr i64 %word49 to ptr addrspace(1)
  {read}
  ret void
}}
"""
    assert entry_is_synchronous(ir) is synchronous


def test_compiler_completion_proof_is_independent_of_input_cache_decoder(tmp_path, monkeypatch):
    sdk = os.environ.get("ASCEND_HOME_PATH")
    pto = _ROOT / "build/pto-isa"
    if not sdk or not (Path(sdk) / "bin/bisheng").is_file() or not (pto / "include/pto").is_dir():
        pytest.skip("requires CANN's A5 compiler and PTO headers")
    compiler = KernelCompiler("a5")
    source = _ROOT / "tests/st/a5/host_build_graph/defer_wait_visibility/kernels/aiv/tail.cpp"
    includes = compiler.get_orchestration_include_dirs("host_build_graph")
    raw = extract_text_section(compiler.compile_incore(str(source), "aiv", str(pto), includes))
    monkeypatch.setattr(proof, "_decode", lambda *args, **kwargs: pytest.fail("registration ran decoder"))
    registered = compiler.compile_hbg_registration(str(source), "aiv", str(pto), includes)
    assert registered == proof.synchronous_registration(raw, "aiv")
