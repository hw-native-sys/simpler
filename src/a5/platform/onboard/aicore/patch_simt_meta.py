#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Patch bisheng's auto-emitted .ascend.meta.<funcname> section in an aicore
# ELF .o so that the SIMT-related TLVs carry SIMT-friendly values.
#
# Why this exists: bisheng emits a per-function meta section with TLV 12
# (AIV_TYPE_FLAG) defaulted to 1 (AIV_TYPE_NO_VF) and TLV 7
# (COMPILER_ALLOC_UB_SIZE) defaulted to 0 for kernels it cannot statically
# detect as SIMT. Our aicore_kernel entry is a SU dispatcher (vector ops are
# in callee TUs invoked through aicore_execute), so bisheng tags it NO_VF.
# Runtime's parser overwrites by section name (kernelInfoMap[name] = ...),
# so a hand-written sibling section is shadowed by bisheng's section. The
# only reliable fix is to patch bisheng's section in place.
#
# Usage:
#   patch_simt_meta.py <elf-path> <funcname> <ub_size> <aiv_type>
# Example:
#   patch_simt_meta.py kernel_aiv.o aicore_kernel_0_mix_aiv 8192 3
#
# Idempotent: re-running on an already-patched file is a no-op.

import struct
import sys

ELF_MAGIC = b"\x7fELF"

# TLV types that we patch (must match runtime/src/runtime/core/inc/kernel/elf.hpp)
TLV_COMPILER_ALLOC_UB_SIZE = 7
TLV_AIV_TYPE_FLAG = 12


def _read_elf64_section_table(buf):
    """Yield (name, offset, size) for each section in a 64-bit little-endian ELF."""
    if buf[:4] != ELF_MAGIC:
        raise ValueError("not an ELF file")
    if buf[4] != 2 or buf[5] != 1:
        raise ValueError("only ELF64 little-endian supported")

    e_shoff = struct.unpack_from("<Q", buf, 0x28)[0]
    e_shentsize = struct.unpack_from("<H", buf, 0x3A)[0]
    e_shnum = struct.unpack_from("<H", buf, 0x3C)[0]
    e_shstrndx = struct.unpack_from("<H", buf, 0x3E)[0]

    # Section header layout (64-bit):
    #   uint32 sh_name      offset 0x00
    #   uint32 sh_type      offset 0x04
    #   uint64 sh_flags     offset 0x08
    #   uint64 sh_addr      offset 0x10
    #   uint64 sh_offset    offset 0x18
    #   uint64 sh_size      offset 0x20
    shstr_hdr = e_shoff + e_shstrndx * e_shentsize
    shstr_offset = struct.unpack_from("<Q", buf, shstr_hdr + 0x18)[0]
    shstr_size = struct.unpack_from("<Q", buf, shstr_hdr + 0x20)[0]
    shstr_table = bytes(buf[shstr_offset : shstr_offset + shstr_size])

    for i in range(e_shnum):
        hdr = e_shoff + i * e_shentsize
        name_off = struct.unpack_from("<I", buf, hdr)[0]
        sh_offset = struct.unpack_from("<Q", buf, hdr + 0x18)[0]
        sh_size = struct.unpack_from("<Q", buf, hdr + 0x20)[0]
        end = shstr_table.find(b"\x00", name_off)
        name = shstr_table[name_off:end].decode("ascii", errors="replace")
        yield name, sh_offset, sh_size


def _patch_section(buf, sec_offset, sec_size, ub_size, aiv_type):
    """Walk TLV list at sec_offset for sec_size bytes; patch values for
    TLV 7 (ub_size) and TLV 12 (aiv_type). Returns count of TLVs patched."""
    patched = 0
    cursor = sec_offset
    end = sec_offset + sec_size
    while cursor + 4 <= end:
        tlv_type = struct.unpack_from("<H", buf, cursor)[0]
        tlv_len = struct.unpack_from("<H", buf, cursor + 2)[0]
        value_off = cursor + 4
        if value_off + tlv_len > end:
            break  # malformed; stop
        if tlv_type == TLV_AIV_TYPE_FLAG and tlv_len == 4:
            struct.pack_into("<I", buf, value_off, aiv_type)
            patched += 1
        elif tlv_type == TLV_COMPILER_ALLOC_UB_SIZE and tlv_len == 4:
            struct.pack_into("<I", buf, value_off, ub_size)
            patched += 1
        cursor = value_off + tlv_len
    return patched


def main():
    if len(sys.argv) != 5:
        print(
            "usage: patch_simt_meta.py <elf-path> <funcname> <ub_size> <aiv_type>",
            file=sys.stderr,
        )
        return 1

    elf_path, funcname, ub_size, aiv_type = sys.argv[1:]
    ub_size = int(ub_size, 0)
    aiv_type = int(aiv_type, 0)
    section_name = f".ascend.meta.{funcname}"

    with open(elf_path, "rb") as fp:
        buf = bytearray(fp.read())

    # Bisheng emits a section ≥ 16 bytes (we've observed 40 bytes). Our
    # hand-written sibling is exactly 16 bytes (two TLVs). Patch every
    # matching section so we cover both — the runtime overwrite means only
    # the last-parsed wins, but patching all of them costs nothing and
    # leaves no bait for future regressions.
    matches = [(off, sz) for name, off, sz in _read_elf64_section_table(buf) if name == section_name]
    if not matches:
        print(f"warning: no section named '{section_name}' found in {elf_path}", file=sys.stderr)
        return 0

    total_patched = 0
    for off, sz in matches:
        total_patched += _patch_section(buf, off, sz, ub_size, aiv_type)

    if total_patched == 0:
        print(
            f"warning: found '{section_name}' but no TLV 7/12 to patch in {elf_path}",
            file=sys.stderr,
        )
        return 0

    with open(elf_path, "wb") as fp:
        fp.write(buf)

    print(
        f"patched {total_patched} TLV(s) across {len(matches)} section(s) in {elf_path}: "
        f"ub_size={ub_size}, aiv_type={aiv_type}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
