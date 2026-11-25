# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prove that an A5 HBG entry cannot reach its deferred-completion context."""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path

from .a5_input_cache import synchronous_registration
from .elf_parser import extract_text_section

_logger = logging.getLogger(__name__)
_VALUE = r"%[-a-zA-Z$._0-9]+"
_ENTRY = re.compile(r"^define [^\n]* void @kernel_entry\(ptr addrspace\(1\) [\w ]*(" + _VALUE + r")\)[^\n]*\{$", re.M)
_PTR = r"ptr addrspace\(1\) "
# HBG passes the private GlobalContext only in args[49]. Its first i32 is
# sub_block_id. Ordinary tensor, scalar and LocalContext arguments are disjoint.
_GLOBAL_SLOT = 49


def _has_opaque_assembly(ir: str) -> bool:
    if re.search(r"^module asm|@llvm\.(?:read_register|read_volatile_register|frameaddress|returnaddress)", ir, re.M):
        return True
    return any(
        not re.fullmatch(r'\s*(?:tail )?call void asm sideeffect "nop", ""\(\)(?: #[0-9]+)?(?:, !.*)?', line)
        for line in ir.splitlines()
        if re.search(r"\basm\b", line) and not line.startswith(("!", ";"))
    )


def _closed_ir_modules(modules: tuple[str, ...]) -> bool:
    # Only the linked caller and its supplied VF modules may supply implementations.
    # An external object or indirect callee could obtain context outside the modeled ABI.
    definitions = {name for ir in modules for name in re.findall(r"^define [^\n]*?@([-\w.$]+)\(", ir, re.M)}
    for ir in modules:
        if _has_opaque_assembly(ir):
            return False
        for call in re.finditer(
            r"^\s*(?:%[-\w.$]+ = )?(?:(?:musttail|notail|tail) )?(?:call|invoke|callbr) (.*)$", ir, re.M
        ):
            instruction = call[1]
            if 'asm sideeffect "nop", ""()' in instruction:
                continue
            callee = re.match(r"[^@\n]*@([-\w.$]+)\(", instruction)
            if callee is None or not (callee[1].startswith("llvm.") or callee[1] in definitions):
                return False
    return True


def _argument_uses(body: str) -> tuple[dict[str, list[str]], dict[str, tuple[int, int]]]:
    uses: dict[str, list[str]] = {}
    selections: dict[str, tuple[int, int]] = {}
    for raw in body.splitlines():
        line = raw.strip()
        if re.match(r"(?:tail )?call void @llvm\.dbg\.(?:declare|value|assign|label)\(", line):
            continue
        # Drop IR metadata, retaining every instruction operand.
        line = re.sub(r"(?:, ![\w.]+ ![0-9]+)+$", "", line)
        assignment = re.match(_VALUE + r" = (.*)$", line)
        rhs = assignment[1] if assignment else line
        if rhs.startswith("select "):
            choice = re.fullmatch(r"(" + _VALUE + r") = select i1 " + _VALUE + r", i64 (\d+), i64 (\d+)", line)
            if choice:
                selections[choice[1]] = (int(choice[2]), int(choice[3]))
        for name in re.findall(_VALUE, rhs):
            uses.setdefault(name, []).append(line)

    return uses, selections


def entry_is_synchronous(ir: str, vector_modules: tuple[str, ...] = ()) -> bool:
    entry = _ENTRY.search(ir)
    if entry is None or '\ntarget triple = "hiipu64-hisilicon-cce"' not in ir:
        return False
    end = ir.find("\n}", entry.end())
    if end == -1:
        return False
    body = ir[entry.end() : end]
    # Opaque assembly may access registers not named as IR operands.
    if not _closed_ir_modules((ir, *vector_modules)):
        return False
    uses, selections = _argument_uses(body)

    def loads(address: str, bits: int) -> list[str] | None:
        results = []
        for use in uses.get(address, ()):
            m = re.fullmatch(
                r"(" + _VALUE + rf") = load i{bits}, " + _PTR + re.escape(address) + r"(?:, align \d+)?", use
            )
            if m is None:
                return None
            results.append(m[1])
        return results

    args = entry[1]
    for use in uses.get(args, ()):
        # A direct i64 load is argument zero. All other accesses must start
        # from a constant, in-bounds argument slot. Reject escapes and wide reads.
        if re.fullmatch(_VALUE + r" = load i64, " + _PTR + re.escape(args) + r"(?:, align \d+)?", use):
            continue
        gep = re.fullmatch(
            r"("
            + _VALUE
            + r") = getelementptr inbounds i64, "
            + _PTR
            + re.escape(args)
            + r", i64 ("
            + _VALUE
            + r"|\d+)",
            use,
        )
        if gep is None:
            return False
        slots = (int(gep[2]),) if gep[2].isdigit() else selections.get(gep[2], ())
        if not slots or max(slots) > _GLOBAL_SLOT:
            return False
        values = loads(gep[1], 64)
        if values is None:
            return False
        if _GLOBAL_SLOT not in slots:
            continue
        for value in values:
            for user in uses.get(value, ()):
                cast = re.fullmatch(
                    r"(" + _VALUE + r") = inttoptr i64 " + re.escape(value) + r" to ptr addrspace\(1\)", user
                )
                if cast is None or loads(cast[1], 32) is None:
                    return False
    return True


def hbg_registration_text(elf: bytes, compiler: str, core: str, *, vector_modules: tuple[str, ...] = ()) -> bytes:
    """Attach the proof to registration bytes. Only executable text reaches the device."""
    text = extract_text_section(elf)
    bindir = Path(compiler).parent
    synchronous = False
    try:
        with tempfile.TemporaryDirectory(prefix="hbg-proof-") as directory:
            bc = Path(directory) / "entry.bc"
            # Inspect the linked image without modifying the input or retaining any bitcode.
            subprocess.run(
                [bindir / "llvm-objcopy", "--dump-section", f".llvmbc={bc}", "-", "-"],
                input=elf,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                timeout=30,
            )
            if bc.is_file():
                arch = "dav-c310-cube" if core == "aic" else "dav-c310-vec"
                ir = subprocess.run(
                    [
                        bindir / "bisheng",
                        "-cc1",
                        "-triple",
                        "hiipu64-hisilicon-cce",
                        "-target-cpu",
                        arch,
                        "-emit-llvm",
                        "-O1",
                        "-x",
                        "ir",
                        bc,
                        "-o",
                        "-",
                    ],
                    capture_output=True,
                    check=True,
                    timeout=30,
                )
                synchronous = entry_is_synchronous(ir.stdout.decode(), vector_modules)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        _logger.debug("HBG completion proof unavailable: %s", exc)
    return synchronous_registration(text, core) if synchronous else text
