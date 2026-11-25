# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check the A5 HBG Scalar DCache footprint of final, relocated incore instructions."""

from __future__ import annotations

import logging
import os
import re
import struct
import subprocess
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import NamedTuple, Union

_logger = logging.getLogger(__name__)
_ELF_FLAGS = {"aiv": 0x990000, "aic": 0x9A0000}
_CORE_MASK = {"aiv": 2, "aic": 1}
_LINE = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f]{8}|[0-9a-f]{16})\s+(\S+)(.*)$", re.MULTILINE)
_MASK = (1 << 64) - 1
# A5 host_build_graph/kernel_args.h, intrinsic.h and TensorData. Descriptor loads
# require min_tensor_count at bind, even when there are no Scalar data accesses.
_MAX_TENSORS = 32
_CONTEXT_SIZES = {48: 8, 49: 48}


class _Unproven(ValueError):
    pass


class _Instruction(NamedTuple):
    name: str
    args: tuple[str, ...]
    width: int


def _disassembly_image(code: bytes, flags: int, vector: bool) -> bytes:
    """BiSheng selects the scalar or micro-vector decoder via ELF mapping symbols."""
    strings = b"\0.text\0.strtab\0.symtab\0" + (b"$k.0\0" if vector else b"$x.0\0")
    symbol = bytes(24) + struct.pack("<IBBHQQ", strings.index(b"$"), 0, 0, 1, 0, 0)
    data = code + strings + symbol
    header = struct.pack(
        "<16sHHIQQQIHHHHHH", b"\x7fELF\x02\x01\x01", 1, 0x1029, 1, 0, 0, 64 + len(data), flags, 64, 0, 0, 64, 4, 2
    )
    section = struct.Struct("<IIQQQQIIQQ")
    return b"".join(
        (
            header,
            data,
            bytes(section.size),
            section.pack(1, 1, 6, 0, 64, len(code), 0, 0, 4, 0),
            section.pack(7, 3, 0, 0, 64 + len(code), len(strings), 0, 0, 1, 0),
            section.pack(15, 2, 0, 0, 64 + len(code) + len(strings), len(symbol), 2, 2, 8, 24),
        )
    )


def _time_left(deadline: float) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("A5 input cache analysis budget expired")
    return remaining


def _decode(tool: Path, code: bytes, flags: int, *, deadline: float, vector: bool = False) -> dict[int, _Instruction]:
    result = subprocess.run(
        [str(tool), "-d", "-z", "-"],
        input=_disassembly_image(code, flags, vector),
        capture_output=True,
        timeout=_time_left(deadline),
        check=True,
    )
    instructions = {}
    for match in _LINE.finditer(result.stdout.decode("ascii")):
        _time_left(deadline)
        address, word, op, operands = match.groups()
        pc, width = int(address, 16), len(word) // 2
        if pc + width > len(code) or int.from_bytes(code[pc : pc + width], "little") != int(word, 16):
            raise _Unproven("disassembly differs from executable bytes")
        operands = operands.partition(" <")[0].strip()
        instructions[pc] = _Instruction(op, tuple(operands.split(", ")) if operands else (), width)
    return instructions


def _reg(name: str) -> int:
    if not re.fullmatch(r"X(?:[12]?[0-9]|3[01])", name):
        raise _Unproven(f"expected Scalar GPR, got {name}")
    return int(name[1:])


def _imm(name: str) -> int:
    if not name.startswith("#"):
        raise _Unproven("expected immediate")
    return int(name[1:], 10)


@dataclass(frozen=True)
class _Address:
    kind: str
    slot: int = -1
    offset: int = 0
    scale: int = 0


# An integer is a known immediate, None an unconstrained value. Only a
# tagged PC survives relocation. An immediate that happens to look like a
# code address or a Scalar value equal to a buffer address proves nothing.
_Value = Union[_Address, int, None]
_ARGS = _Address("args")
_STACK = _Address("stack")
_RETURN = _Address("return")
_DYNAMIC_DATA = _Address("dynamic_data")
_LINE_ADDRESS = _Address("line")
_INDIRECT = _Address("loaded_value")
_CTRL = _Address("ctrl")


def _business(value: _Value) -> bool:
    return isinstance(value, _Address) and value.kind in ("data", "dynamic_data")


def _add(left: _Value, right: _Value) -> _Value:
    if _INDIRECT in (left, right):
        return _INDIRECT
    if isinstance(right, int):
        if isinstance(left, int):
            return (left + right) & _MASK
        if isinstance(left, _Address) and left.kind in ("args", "stack", "slot", "context", "data", "start", "pc"):
            offset = left.offset + right
            if -(1 << 31) <= offset <= (1 << 32) - 1:
                return replace(left, offset=offset)
        if left == _LINE_ADDRESS and right % 64 == 0:
            return _LINE_ADDRESS
        if left == _DYNAMIC_DATA:
            return left
        return None
    if isinstance(left, int):
        return _add(right, left)
    if isinstance(left, _Address) and isinstance(right, _Address):
        if left.kind == "start" and right.kind == "data":
            left, right = right, left
        if left.kind == "data" and right.kind == "start" and left.slot == right.slot and left.scale == 0:
            return replace(left, scale=right.scale, offset=left.offset + right.offset)
    # A variable tensor displacement is sufficient for an MTE operand. It
    # cannot justify a cached Scalar load or store.
    if _business(left) and (right is None or isinstance(right, _Address) and right.kind == "slot"):
        return _DYNAMIC_DATA
    if _business(right) and (left is None or isinstance(left, _Address) and left.kind == "slot"):
        return _DYNAMIC_DATA
    return None


def _scaled(value: _Value, factor: int) -> _Value:
    if isinstance(value, int):
        return (value * factor) & _MASK
    if isinstance(value, _Address) and value.kind == "start":
        return replace(value, scale=value.scale * factor, offset=value.offset * factor)
    return _INDIRECT if value == _INDIRECT else None


def _join_value(left: _Value, right: _Value) -> _Value:
    if left == right:
        return left
    if _INDIRECT in (left, right):
        return _INDIRECT
    return _DYNAMIC_DATA if _business(left) and _business(right) else None


@dataclass
class _State:
    regs: list[_Value]
    # Compiler-private 64-bit spills, indexed relative to the entry SP.
    stack: dict[int, _Value]
    wrote_gm: bool = False

    def copy(self):
        return _State(self.regs.copy(), self.stack.copy(), self.wrote_gm)

    def merged(self, other: _State):
        regs = [_join_value(a, b) for a, b in zip(self.regs, other.regs)]
        stack: dict[int, _Value] = {
            k: v for k, a in self.stack.items() if (v := _join_value(a, other.stack.get(k))) is not None
        }
        return _State(regs, stack, self.wrote_gm or other.wrote_gm)


class _Range(NamedTuple):
    slot: int
    scale: int
    offset: int
    size: int


# These micro-vector instructions operate on vector/predicate registers or
# UB. No instruction may leave the verified VF extent, fetch GM through the
# Scalar cache, or modify the calling Scalar register file.
_VECTOR_LOCAL = frozenset(
    """
    SMOV SMOVI VAG_V310 PLT VDUPS VSTI VST VLD VLDI VCVTFF VADD VMUL
    VMULS VADDS VMAX VMIN VSUB VDIV VABS VEXP VSQRT VRSQRT VMAXS
    SNOP SMAX SADD VOR VCI SMEM_BAR PSET
""".split()
)
_SCALAR_CALC = frozenset(
    """
    AND OR XOR NOT NEG MUL MULI DIV REM MIN MAX SHRI SHL SEL INSERT SIGNEXT ZEROEXT
    SBITSET0 SBITSET1 CMPN CONV
""".split()
)
_SIGNAL = frozenset(
    """
    NOP BAR DSB SET_FLAGI WAIT_FLAGI SET_FLAG WAIT_FLAG SET_FLAGI_V WAIT_FLAGI_V SET_FLAG_V WAIT_FLAG_V
    SET_INTRA_BLOCK WAIT_INTRA_BLOCK SET_INTRA_BLOCKI WAIT_INTRA_BLOCKI SET_INTRA_BLOCKI_V WAIT_INTRA_BLOCKI_V
    WAIT_INTRA_BLOCK_V
""".split()
)
_CORE_LOCAL = frozenset(
    """
    LOAD_L1_TO_L0A_2DV2 LOAD_L1_TO_L0B_2DV2 LOAD_L1_TO_L0A_MX_2DV2 LOAD_L1_TO_L0B_MX_2DV2
    MMAD MMAD_MX FIX_L0C_TO_UB MOV_UB_TO_L1 MOV_UB_TO_UB
""".split()
)
_MTE_READ = {
    "MOV_OUT_TO_L1_ALIGN_V2",
    "MOV_OUT_TO_L1_MULTI_ND2NZ",
    "MOV_OUT_TO_L1_MULTI_DN2NZ",
    "MOV_OUT_TO_UB_ALIGN_V2",
}
_MTE_WRITE = {"MOV_UB_TO_OUT_ALIGN_V2", "FIX_L0C_TO_OUT"}
# No writes to PC, stack configuration or memory translation are admissible.
_SPR = frozenset(
    """
    LOOP_SIZE_UBTOOUT LOOP1_STRIDE_UBTOOUT LOOP2_STRIDE_UBTOOUT
    LOOP_SIZE_OUTTOUB LOOP1_STRIDE_OUTTOUB LOOP2_STRIDE_OUTTOUB
    LOOP1_STRIDE_OUTTOL1 LOOP2_STRIDE_OUTTOL1 LOOP_SIZE_OUTTOL1
    ND_PARA MATRIX_FORMAT FMATRIX DEQSCALE FPC MTE2_NZ_PARA PADDING
""".split()
)


class _Verifier:
    def __init__(self, code: bytes, flags: int, tool: Path, deadline: float):
        self.code = code
        self.flags = flags
        self.tool = tool
        self.deadline = deadline
        self.instructions = _decode(tool, code, flags, deadline=deadline)
        self.vector_extents: set[tuple[int, int]] = set()
        self.min_tensors = 0
        self.ranges: set[_Range] = set()

    def vector_function(self, target: _Value, words: int) -> None:
        if not isinstance(target, _Address) or target.kind != "pc" or self.flags != _ELF_FLAGS["aiv"]:
            raise _Unproven("indirect VF target")
        start, size = target.offset, words * 4
        if start < 0 or start % 4 or words < 1 or start + size > len(self.code):
            raise _Unproven("VF extends outside executable")
        if (start, size) in self.vector_extents:
            return
        body = _decode(self.tool, self.code[start : start + size], self.flags, vector=True, deadline=self.deadline)
        for pos in range(0, size, 4):
            _time_left(self.deadline)
            inst = body.get(pos)
            if inst is None or inst.width != 4:
                raise _Unproven("incomplete SIMD disassembly")
            op = inst.name.partition(".")[0]
            if pos == size - 4:
                if inst.name != "SEND":
                    raise _Unproven("VF lacks terminal SEND")
            elif op == "VLOOPV2_V310":
                # Loop body length is encoded in words after the instruction.
                if (
                    len(inst.args) != 4
                    or inst.args[2:] != ("#1", "#1")
                    or not 0 < _imm(inst.args[1]) < (size - pos) // 4 - 1
                ):
                    raise _Unproven("VF loop exits the verified body")
            elif op not in _VECTOR_LOCAL:
                raise _Unproven(f"unsupported SIMD instruction {inst.name}")
        self.vector_extents.add((start, size))

    @staticmethod
    def pointer(state: _State, operand: str) -> _Value:
        if not operand.startswith("[") or not operand.endswith("]"):
            raise _Unproven("expected register-indirect address")
        return state.regs[_reg(operand[1:-1])]

    @staticmethod
    def spill(state: _State, pos: int, width: int, store: bool, value: _Value) -> _Value:
        if store:
            for k in list(state.stack):
                if k < pos + width and pos < k + 8:
                    del state.stack[k]
            if width == 8 and pos % 8 == 0 and value is not None:
                state.stack[pos] = value
        elif width == 8:
            return state.stack.get(pos)
        return None

    def memory(self, addr: _Value, width: int, state: _State, *, store=False, value=None, bypass=False) -> _Value:
        if not isinstance(addr, _Address):
            raise _Unproven("unknown memory address")
        pos = addr.offset
        if addr.kind == "stack" and -32768 <= pos and pos + width <= 0:
            return self.spill(state, pos, width, store, value)
        if state.wrote_gm and addr.kind in ("args", "slot", "context"):
            # A GM store through an alias could have changed a descriptor or
            # an args[] slot. Its previous pointer provenance is no longer valid.
            raise _Unproven("metadata loaded after a possible aliasing GM write")
        if addr.kind == "args" and not store and 0 <= pos and pos + width <= 400:
            slot = pos // 8
            if width == 8 and pos % 8 == 0:
                if slot < _MAX_TENSORS:
                    return _Address("slot", slot)
                if slot in _CONTEXT_SIZES:
                    return _Address("context", slot)
            return None
        if addr.kind == "slot" and not store and 0 <= pos and pos + width <= 96:
            self.min_tensors = max(self.min_tensors, addr.slot + 1)
            if width == 8 and pos == 0:
                return _Address("data", addr.slot)
            if width == 8 and pos == 24:
                return _Address("start", addr.slot, scale=1)
            return None
        if addr.kind == "context" and not store and 0 <= pos and pos + width <= _CONTEXT_SIZES[addr.slot]:
            return None
        if not _business(addr) or (store and not isinstance(value, int) and value != _INDIRECT):
            raise _Unproven("escaped or writable metadata")
        if not bypass:
            if addr.kind != "data" or addr.scale not in (0, 1, 2, 4, 8) or not 0 <= pos <= (1 << 32) - width:
                raise _Unproven("unbounded cached tensor access")
            self.ranges.add(_Range(addr.slot, addr.scale, pos, width))
        state.wrote_gm |= store
        return _INDIRECT

    def step(self, pc: int, ins: _Instruction, state: _State) -> tuple[int, ...]:
        op, _, subtype = ins.name.partition(".")
        a, regs = ins.args, state.regs
        following = pc + ins.width
        if ins.name == "RET" and not a and regs[31] == _RETURN:
            return ()
        if ins.name in ("JUMPI", "JUMPCI") and len(a) == 1:
            target = _imm(a[0])
            return (target,) if op == "JUMPI" else (target, following)
        if ins.name == "VF" and len(a) == 2:
            self.vector_function(regs[_reg(a[0])], _imm(a[1]))
        elif op in ("LD_IO", "ST_IO", "LD", "ST", "LDP", "STP", "LD_DEV", "ST_DEV"):
            self.scalar_memory(op, subtype, a, state)
        elif op == "DCCI":
            operand = regs[_reg(a[0])]
            if a[1:] != ("#0", "#2") or not isinstance(operand, _Address) or operand.kind not in ("data", "line"):
                raise _Unproven("unresolved or unsupported DCCI")
        elif op in _MTE_READ | _MTE_WRITE:
            # The GM operand is the source for MTE2 and the destination for MTE3/FIX.
            if not _business(self.pointer(state, a[1 if op in _MTE_READ else 0])):
                raise _Unproven("unresolved MTE GM operand")
            state.wrote_gm |= op in _MTE_WRITE
        elif op == "PUSH_PB":
            if len(a) != 4 or any(isinstance(regs[_reg(reg)], _Address) for reg in a):
                raise _Unproven("pointer escaped in VF parameters")
        elif op == "MOVEMASK":
            if len(a) != 2 or a[0] not in ("MASK[0]", "MASK[1]"):
                raise _Unproven("unsupported mask register update")
            if isinstance(regs[_reg(a[1])], _Address):
                raise _Unproven("pointer exported to a mask register")
        elif op not in _SIGNAL | _CORE_LOCAL | {"CMP"}:
            self.compute(pc, ins, regs)
        return (following,)

    @staticmethod
    def move(pc: int, a: tuple[str, ...], regs: list[_Value]) -> None:
        if a[0] == "CTRL":
            if regs[_reg(a[1])] != _CTRL:
                raise _Unproven("uncontrolled CTRL update")
        elif a[0] in _SPR:
            if isinstance(regs[_reg(a[1])], _Address):
                raise _Unproven("pointer exported to a special register")
        else:
            src = a[1]
            if src == "PC":
                value = _Address("pc", offset=pc)
            elif src == "CTRL":
                value = _CTRL
            elif src in _SPR or src in ("SUBBLOCKID", "CONDITION_FLAG"):
                value = None
            else:
                value = regs[_reg(src)]
            regs[_reg(a[0])] = value

    def compute(self, pc: int, ins: _Instruction, regs: list[_Value]) -> None:
        op, _, _ = ins.name.partition(".")
        a = ins.args
        if ins.name == "MOVI":
            regs[_reg(a[0])] = _imm(a[1])
        elif ins.name == "MOVK":
            dest, imm, shift = _reg(a[0]), _imm(a[1]), 16 * _imm(a[2])
            old = regs[dest]
            regs[dest] = (old & ~(65535 << shift)) | (imm << shift) if isinstance(old, int) else None
        elif ins.name == "MOV":
            self.move(pc, a, regs)
        elif ins.name in ("ADD.s64", "SUB.s64", "ADDI", "SUBI"):
            rhs = _imm(a[2]) if op.endswith("I") else regs[_reg(a[2])]
            if op.startswith("SUB"):
                rhs = -rhs if isinstance(rhs, int) else None
            regs[_reg(a[0])] = _add(regs[_reg(a[1])], rhs)
        elif ins.name == "SHLI.b64":
            dest, shift = _reg(a[0]), _imm(a[1])
            regs[dest] = _scaled(regs[dest], 1 << shift) if 0 <= shift < 64 else None
        elif ins.name == "NEG.s64":
            old = regs[_reg(a[1])]
            regs[_reg(a[0])] = (-old) & _MASK if isinstance(old, int) else _INDIRECT if old == _INDIRECT else None
        elif ins.name == "AND.b64":
            left, right = regs[_reg(a[1])], regs[_reg(a[2])]
            if (left == _MASK - 63 and isinstance(right, _Address) and right.kind == "data") or (
                right == _MASK - 63 and isinstance(left, _Address) and left.kind == "data"
            ):
                regs[_reg(a[0])] = _LINE_ADDRESS
            elif isinstance(left, int) and isinstance(right, int):
                regs[_reg(a[0])] = left & right
            else:
                regs[_reg(a[0])] = _INDIRECT if _INDIRECT in (left, right) else None
        elif op in ("SBITSET0", "SBITSET1") and regs[_reg(a[0])] == _CTRL:
            if regs[_reg(a[1])] not in (6, 7, 8, 46, 47, 56, 59, 60):
                raise _Unproven("unsupported CTRL mode")
        elif op in _SCALAR_CALC:
            # These instructions have exactly one GPR destination. A value
            # loaded from user GM cannot become a proven pointer via arithmetic.
            inputs = (regs[_reg(x)] for x in a if x.startswith("X"))
            regs[_reg(a[0])] = _INDIRECT if any(x == _INDIRECT for x in inputs) else None
        else:
            raise _Unproven(f"unsupported Scalar instruction {ins.name}")

    def scalar_memory(self, op: str, subtype: str, a: tuple[str, ...], state: _State) -> None:
        width = {"b8": 1, "b16": 2, "b32": 4, "b64": 8}.get(subtype)
        pair, store = op in ("LDP", "STP"), op.startswith("ST")
        count = 2 if pair else 1
        ordinary = op in ("LD", "ST", "LD_IO", "ST_IO")
        if width is None or len(a) != count + 2 + ordinary or ((ordinary or op.endswith("_DEV")) and a[-1] != "#0"):
            raise _Unproven("unsupported Scalar access mode")
        # LD/ST use a GPR index in elements, the IO and pair forms a signed
        # byte displacement. Only the non-updating (#0) form is accepted.
        index = a[count + 1]
        offset = _scaled(state.regs[_reg(index)], width) if op in ("LD", "ST") else _imm(index)
        base = _add(self.pointer(state, a[count]), offset)
        for i in range(count):
            reg = _reg(a[i])
            result = self.memory(
                _add(base, i * width), width, state, store=store, value=state.regs[reg], bypass=op.endswith("_DEV")
            )
            if not store:
                state.regs[reg] = result

    def verify(self) -> tuple[int, list[_Range]]:
        regs: list[_Value] = [None] * 32
        regs[0], regs[30], regs[31] = _ARGS, _STACK, _RETURN
        work = deque([0])
        states = {0: _State(regs, {})}
        while work:
            _time_left(self.deadline)
            pc = work.popleft()
            state = states[pc].copy()
            inst = self.instructions.get(pc)
            if inst is None:
                raise _Unproven("control leaves decoded Scalar code")
            for dest in self.step(pc, inst, state):
                previous = states.get(dest)
                merged = state if previous is None else state.merged(previous)
                if merged != previous:
                    states[dest] = merged
                    work.append(dest)
        return self.min_tensors, _bounded_ranges(self.ranges)


def _bounded_ranges(touches) -> list[_Range]:
    ranges: list[_Range] = []
    for current in sorted(touches):
        if ranges:
            last = ranges[-1]
            if current[:2] == last[:2] and current.offset <= last.offset + last.size:
                end = max(last.offset + last.size, current.offset + current.size)
                ranges[-1] = _Range(last.slot, last.scale, last.offset, end - last.offset)
                continue
        ranges.append(current)
    if len(ranges) > 3 or any(r.size > 65535 for r in ranges):
        raise _Unproven("Scalar footprint exceeds the bounded range budget")
    return ranges


def _pack_proof(text, mode, min_tensors, mask, ranges, deadline=None):
    prefix = struct.pack("<4sBBBB", b"HCP1", mode, len(ranges), min_tensors, mask)
    records = b"".join(struct.pack("<BBHI", r.slot, r.scale, r.size, r.offset) for r in ranges).ljust(24, b"\0")
    digest = 0xCBF29CE484222325
    for segment in (text, prefix, records):
        for offset in range(0, len(segment), 65536):
            if deadline is not None and monotonic() >= deadline:
                return text
            for byte in segment[offset : offset + 65536]:
                digest = ((digest ^ byte) * 0x100000001B3) & _MASK
    return text + prefix + struct.pack("<Q", digest) + records


def synchronous_registration(text: bytes, core: str) -> bytes:
    """Encode the compiler's per-family result independently of the cache decoder."""
    return _pack_proof(text, 0, 0, _CORE_MASK[core] << 2, [])


def annotate_a5_hbg_text(text: bytes, core_type: str | None = None, *, deadline: float | None = None) -> bytes:
    """Append an HCP1 proof for the A5 execution families verified on these exact bytes.

    A stripped export may have no core kind. Decode it independently under
    both instruction sets. If their combined footprint cannot fit, keep the
    first certified family. The binder checks the certificate against the
    actual AIC/AIV lane before accepting it.
    """
    sdk = os.environ.get("ASCEND_HOME_PATH")
    if not text or len(text) % 4 or not sdk:
        return text
    if core_type is not None and core_type not in _ELF_FLAGS:
        return text

    if deadline is None:
        deadline = monotonic() + 15.0
    original = text
    sync_mask = 0
    if len(text) >= 40 and text[-40:-36] == b"HCP1":
        # An offline exporter may already have a compiler completion proof. Verify
        # the original bytes before carrying its bits into the combined proof.
        if core_type is None or synchronous_registration(text[:-40], core_type) != text:
            return original
        text, sync_mask = text[:-40], text[-33]
    decoder = Path(sdk) / "bin" / "llvm-objdump"
    mask, min_tensors = 0, 0
    ranges: list[_Range] = []
    for family in _ELF_FLAGS if core_type is None else (core_type,):
        try:
            _time_left(deadline)
            required, touches = _Verifier(text, _ELF_FLAGS[family], decoder, deadline).verify()
            ranges = _bounded_ranges([*ranges, *touches])
        except (TimeoutError, subprocess.TimeoutExpired):
            break
        except (ValueError, IndexError, OSError, subprocess.SubprocessError) as exc:
            _logger.debug("A5 %s HBG input cache proof unavailable: %s", family, exc)
            continue
        min_tensors = max(required, min_tensors)
        mask |= _CORE_MASK[family]
    mode = 2 if ranges else 1
    if mask:
        enriched = _pack_proof(text, mode, min_tensors, mask | sync_mask, ranges, deadline)
        if len(enriched) > len(text):
            return enriched
    return original
