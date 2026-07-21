# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Object File Parser for AICore Kernel Binaries

Pure Python implementation for extracting the ``.text`` section from
ELF64 or Mach-O ``.o`` files for direct execution on AICore.

The loader extracts only the literal ``.text`` section bytes; it does
NOT apply ELF relocations or merge ``.text._Z*`` COMDAT group sections
(out-of-line template instantiations). If a kernel ``.o`` contains
either, the loader rejects it with an actionable diagnostic — see
issue #900 and PR #830 / issue #831 for the failure modes that motivate
this check (CANN 507018 watchdog timeouts and silently-wrong partial
output, both caused by BL instructions in ``.text`` left with imm26=0).

The workaround on the kernel side is to mark every templated AICore
function in the call chain with ``__attribute__((always_inline))`` so
the compiler folds the call graph into a single ``.text`` section.
"""

import logging
import struct
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


# ELF Magic Numbers
ELFMAG0 = 0x7F
ELFMAG1 = ord("E")
ELFMAG2 = ord("L")
ELFMAG3 = ord("F")

# ELF64 layout constants
_SHDR_SIZE = 64
_RELA_SIZE = 24

# ELF section types
_SHT_PROGBITS = 1
_SHT_RELA = 4

# Mach-O Magic Numbers
MH_MAGIC_64 = 0xFEEDFACF

# Mach-O Load Command types
LC_SEGMENT_64 = 0x19


def extract_text_section(obj_input: Union[str, Path, bytes]) -> bytes:
    """
    Extract .text section from an ELF64 or Mach-O .o file.

    Args:
        obj_input: Either a path to the .o file (str/Path) or the binary data (bytes)

    Returns:
        Binary data of the .text section

    Raises:
        FileNotFoundError: If file path is provided and does not exist
        ValueError: If data is not a valid object file or .text section not found
    """
    # Handle input: either path or bytes
    if isinstance(obj_input, bytes):
        obj_data = obj_input
        source_name = "<bytes>"
    else:
        path = Path(obj_input)
        if not path.exists():
            raise FileNotFoundError(f"Object file not found: {obj_input}")
        with open(obj_input, "rb") as f:
            obj_data = f.read()
        source_name = str(obj_input)

    if len(obj_data) < 4:
        raise ValueError(f"Data too small to be a valid object file: {source_name}")

    # Detect format by magic number
    magic32 = struct.unpack("<I", obj_data[:4])[0]
    if magic32 == MH_MAGIC_64:
        return _extract_text_macho64(obj_data, source_name)

    if obj_data[0] == ELFMAG0 and obj_data[1] == ELFMAG1 and obj_data[2] == ELFMAG2 and obj_data[3] == ELFMAG3:
        return _extract_text_elf64(obj_data, source_name)

    raise ValueError(f"Not a valid ELF or Mach-O file: {source_name}")


def _extract_text_elf64(elf_data: bytes, source_name: str) -> bytes:
    """Extract .text section from ELF64 data; reject out-of-line code."""
    if len(elf_data) < 64:
        raise ValueError(f"Data too small to be a valid ELF: {source_name}")

    # Extract section header table info from ELF header
    e_shoff = struct.unpack("<Q", elf_data[40:48])[0]
    e_shnum = struct.unpack("<H", elf_data[60:62])[0]
    e_shstrndx = struct.unpack("<H", elf_data[62:64])[0]

    # Bounds-check the section header table against the buffer; subtract
    # from len() so the arithmetic stays safe on a 64-bit offset that the
    # bare addition would overflow on smaller-int architectures (issue #900
    # review feedback). The same shape repeats for the strtab and per-section
    # data slices below.
    if e_shnum == 0:
        raise ValueError(f"Invalid ELF: section header count is zero in {source_name}")
    if e_shstrndx >= e_shnum:
        raise ValueError(f"Invalid ELF: e_shstrndx ({e_shstrndx}) >= e_shnum ({e_shnum}) in {source_name}")
    if e_shoff > len(elf_data) - e_shnum * _SHDR_SIZE:
        raise ValueError(f"Invalid ELF: section header table is out of bounds in {source_name}")

    # Get string table section header
    shstr_offset = e_shoff + e_shstrndx * _SHDR_SIZE
    shstr_sh_offset = struct.unpack("<Q", elf_data[shstr_offset + 24 : shstr_offset + 32])[0]
    shstr_sh_size = struct.unpack("<Q", elf_data[shstr_offset + 32 : shstr_offset + 40])[0]
    if shstr_sh_size > len(elf_data) or shstr_sh_offset > len(elf_data) - shstr_sh_size:
        raise ValueError(f"Invalid ELF: section string table is out of bounds in {source_name}")
    strtab = elf_data[shstr_sh_offset : shstr_sh_offset + shstr_sh_size]

    # First pass: walk every section header, find .text, and collect any
    # out-of-line code sections that would make a literal .text extraction
    # produce broken code on device.
    text_data: Union[bytes, None] = None
    text_size = 0
    out_of_line: list[tuple[str, int]] = []
    text_relocs: list[tuple[str, int]] = []

    for i in range(e_shnum):
        section_offset = e_shoff + i * _SHDR_SIZE
        sh_name = struct.unpack("<I", elf_data[section_offset : section_offset + 4])[0]
        sh_type = struct.unpack("<I", elf_data[section_offset + 4 : section_offset + 8])[0]
        sh_offset = struct.unpack("<Q", elf_data[section_offset + 24 : section_offset + 32])[0]
        sh_size = struct.unpack("<Q", elf_data[section_offset + 32 : section_offset + 40])[0]
        if sh_name >= len(strtab):
            raise ValueError(f"Invalid ELF: section {i} name offset {sh_name} is out of strtab bounds in {source_name}")
        section_name = _extract_cstring(strtab, sh_name)

        if sh_type == _SHT_PROGBITS and section_name == ".text":
            if sh_size > len(elf_data) or sh_offset > len(elf_data) - sh_size:
                raise ValueError(f"Invalid ELF: .text section data is out of bounds in {source_name}")
            text_data = elf_data[sh_offset : sh_offset + sh_size]
            text_size = sh_size
        elif sh_type == _SHT_PROGBITS and section_name.startswith(".text."):
            # `.text._Z*` group sections hold out-of-line template instantiations
            # (and similar inline-but-not-inlined emissions). `.text.startup`
            # etc. land here too — none of them get loaded today.
            out_of_line.append((section_name, sh_size))
        elif sh_type == _SHT_RELA and section_name.startswith(".rela.text"):
            text_relocs.append((section_name, sh_size // _RELA_SIZE))

    if out_of_line or text_relocs:
        _raise_unresolved_text_error(source_name, out_of_line, text_relocs)

    if text_data is None:
        raise ValueError(f".text section not found in: {source_name}")

    logger.debug(f"Loaded .text section from {source_name} (size: {text_size} bytes)")
    return text_data


def _raise_unresolved_text_error(
    source_name: str,
    out_of_line: list[tuple[str, int]],
    text_relocs: list[tuple[str, int]],
) -> None:
    """Raise with an actionable diagnostic naming the offending sections.

    See issue #900 for context: the loader does not yet merge `.text._Z*`
    sections or apply `.rela.text*` relocations. Silently returning the
    raw `.text` bytes in that situation produces BL/B instructions with
    imm26=0, which on AICore manifests as CANN 507018 watchdog timeouts
    or silently-wrong partial output (e.g. PR #830 / issue #831).
    """
    detail_lines = []
    if out_of_line:
        detail_lines.append("Out-of-line code sections (likely template instantiations):")
        for name, size in out_of_line:
            detail_lines.append(f"  {name}  ({size} bytes)")
    if text_relocs:
        detail_lines.append("Unresolved relocations against .text:")
        for name, count in text_relocs:
            detail_lines.append(f"  {name}  ({count} entries)")
    detail = "\n".join(detail_lines)
    raise ValueError(
        f"AICore loader cannot extract a runnable payload from {source_name}: the .o file contains "
        f"out-of-line code or unresolved .text relocations that this loader does not yet apply "
        f"(see issue #900). On device the BL/B targets in .text would branch to garbage, "
        f"producing CANN 507018 watchdog timeouts or silently-wrong partial output "
        f"(historically PR #830 / issue #831).\n\n"
        f"{detail}\n\n"
        f"Workaround until the loader applies relocations: annotate every templated AICore "
        f"function in the call chain with __attribute__((always_inline)) so the compiler folds "
        f"the call graph into a single .text section. Verify with:\n"
        f"  readelf -S <file.o> | grep '\\.text'\n"
        f"  readelf -r <file.o>"
    )


def _extract_text_macho64(data: bytes, source_name: str) -> bytes:
    """Extract __text section from Mach-O 64-bit data."""
    # Mach-O 64-bit header: magic(4) + cputype(4) + cpusubtype(4) + filetype(4)
    #                        + ncmds(4) + sizeofcmds(4) + flags(4) + reserved(4) = 32 bytes
    if len(data) < 32:
        raise ValueError(f"Data too small to be a valid Mach-O: {source_name}")

    ncmds = struct.unpack("<I", data[16:20])[0]

    # Walk load commands starting at offset 32
    offset = 32
    for _ in range(ncmds):
        if offset + 8 > len(data):
            break
        cmd = struct.unpack("<I", data[offset : offset + 4])[0]
        cmdsize = struct.unpack("<I", data[offset + 4 : offset + 8])[0]

        if cmd == LC_SEGMENT_64:
            # segment_command_64: cmd(4) + cmdsize(4) + segname(16) + vmaddr(8)
            #   + vmsize(8) + fileoff(8) + filesize(8) + maxprot(4) + initprot(4)
            #   + nsects(4) + flags(4) = 72 bytes header
            nsects = struct.unpack("<I", data[offset + 64 : offset + 68])[0]

            # Sections start at offset+72, each section_64 is 80 bytes:
            # sectname(16) + segname(16) + addr(8) + size(8) + offset(4) + align(4)
            # + reloff(4) + nreloc(4) + flags(4) + reserved1(4) + reserved2(4) + reserved3(4)
            sect_base = offset + 72
            for s in range(nsects):
                sect_off = sect_base + s * 80
                sectname = data[sect_off : sect_off + 16].split(b"\x00")[0].decode("ascii")
                if sectname == "__text":
                    s_size = struct.unpack("<Q", data[sect_off + 40 : sect_off + 48])[0]
                    s_offset = struct.unpack("<I", data[sect_off + 48 : sect_off + 52])[0]
                    text_data = data[s_offset : s_offset + s_size]
                    logger.debug(f"Loaded __text section from {source_name} (size: {s_size} bytes)")
                    return text_data

        offset += cmdsize

    raise ValueError(f"__text section not found in: {source_name}")


def _extract_cstring(data: bytes, offset: int) -> str:
    """
    Extract a null-terminated C string from bytes.

    Args:
        data: Byte data
        offset: Starting offset

    Returns:
        Decoded string
    """
    end = data.find(b"\x00", offset)
    if end == -1:
        end = len(data)
    return data[offset:end].decode("ascii", errors="ignore")
