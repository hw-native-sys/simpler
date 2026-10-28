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

The loader extracts only the literal ``.text`` section bytes and jumps to
offset 0, so the payload must already be complete: no unapplied relocations,
no code or allocated read-only data stranded outside ``.text``, and
``kernel_entry`` first. Linked ``.rodata`` jump tables are not part of the
loaded payload and are rejected even when their relocations are resolved.

``KernelCompiler.compile_incore`` links each object with ``ld.lld`` before it
reaches this parser. The linker applies ``.rela.text`` and folds ``.text.*``
COMDAT groups into the output ``.text``, but may retain separate data sections.
The checks below therefore run *after* linking: reaching one means
the link did not produce a self-contained image, not that a kernel merely
needs inlining. Returning the raw bytes anyway is what issue #900 and
PR #830 / issue #831 record — CANN 507018 watchdog timeouts and
silently-wrong partial output from BL instructions left with imm26=0.
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
_SHT_SYMTAB = 2
_SHT_RELA = 4
_SHF_ALLOC = 2

# Hardware incore cache keys include the validation contract of extracted text.
ELF_TEXT_POLICY_VERSION = 2

# Elf64_Sym: st_name(4) st_info(1) st_other(1) st_shndx(2) st_value(8) st_size(8)
_SYM_SIZE = 24

# The AICore loader jumps to offset 0 of the payload, so this symbol must be
# the first thing in .text.
_ENTRY_SYMBOL = "kernel_entry"

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
    """Extract .text from ELF64 data; reject incomplete text-only images."""
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
    text_index = -1
    text_addr = 0
    out_of_line: list[tuple[str, int]] = []
    text_relocs: list[tuple[str, int]] = []
    read_only_data: list[tuple[str, int]] = []

    for i in range(e_shnum):
        section_offset = e_shoff + i * _SHDR_SIZE
        sh_name = struct.unpack("<I", elf_data[section_offset : section_offset + 4])[0]
        sh_type = struct.unpack("<I", elf_data[section_offset + 4 : section_offset + 8])[0]
        sh_flags = struct.unpack("<Q", elf_data[section_offset + 8 : section_offset + 16])[0]
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
            text_index = i
            text_addr = struct.unpack("<Q", elf_data[section_offset + 16 : section_offset + 24])[0]
        elif sh_type == _SHT_PROGBITS and section_name.startswith(".text."):
            # `.text._Z*` group sections hold out-of-line template instantiations
            # (and similar inline-but-not-inlined emissions). `.text.startup`
            # etc. land here too — none of them get loaded today.
            out_of_line.append((section_name, sh_size))
        elif sh_type == _SHT_RELA and section_name.startswith(".rela.text"):
            text_relocs.append((section_name, sh_size // _RELA_SIZE))
        elif (
            sh_type == _SHT_PROGBITS
            and sh_flags & _SHF_ALLOC
            and sh_size
            and (section_name == ".rodata" or section_name.startswith(".rodata."))
        ):
            read_only_data.append((section_name, sh_size))

    if out_of_line or text_relocs:
        _raise_unresolved_text_error(source_name, out_of_line, text_relocs)

    if read_only_data:
        sections = ", ".join(f"{name} ({size} bytes)" for name, size in read_only_data)
        raise ValueError(
            f"AICore image has allocated read-only data outside .text: {sections} in {source_name}. "
            "The text-only loader would discard it even after relocations are resolved. "
            "Use a self-contained .text image without external jump tables or constant data."
        )

    if text_data is None:
        raise ValueError(f".text section not found in: {source_name}")

    _check_entry_is_first(elf_data, source_name, e_shoff, e_shnum, text_index, text_addr)

    logger.debug(f"Loaded .text section from {source_name} (size: {text_size} bytes)")
    return text_data


def _check_entry_is_first(
    elf_data: bytes,
    source_name: str,
    e_shoff: int,
    e_shnum: int,
    text_index: int,
    text_addr: int,
) -> None:
    """Reject an image whose ``kernel_entry`` is not at the start of ``.text``.

    The loader jumps to offset 0 of the extracted payload. In an unlinked object
    that is ``kernel_entry`` by construction, but linking merges ``.text.*``
    COMDAT groups into ``.text`` and is free to order them, so a kernel with
    out-of-line code can end up with another function first. Objects carrying no
    symbol table, or none naming ``kernel_entry``, are left alone — orchestration
    and hand-built test fixtures both take that path.
    """
    for i in range(e_shnum):
        section_offset = e_shoff + i * _SHDR_SIZE
        if struct.unpack("<I", elf_data[section_offset + 4 : section_offset + 8])[0] != _SHT_SYMTAB:
            continue
        sym_offset = struct.unpack("<Q", elf_data[section_offset + 24 : section_offset + 32])[0]
        sym_size = struct.unpack("<Q", elf_data[section_offset + 32 : section_offset + 40])[0]
        link = struct.unpack("<I", elf_data[section_offset + 40 : section_offset + 44])[0]
        if link >= e_shnum or sym_size > len(elf_data) or sym_offset > len(elf_data) - sym_size:
            continue

        str_header = e_shoff + link * _SHDR_SIZE
        str_offset = struct.unpack("<Q", elf_data[str_header + 24 : str_header + 32])[0]
        str_size = struct.unpack("<Q", elf_data[str_header + 32 : str_header + 40])[0]
        if str_size > len(elf_data) or str_offset > len(elf_data) - str_size:
            continue
        symstr = elf_data[str_offset : str_offset + str_size]

        for s in range(sym_size // _SYM_SIZE):
            entry = sym_offset + s * _SYM_SIZE
            st_name = struct.unpack("<I", elf_data[entry : entry + 4])[0]
            if st_name >= len(symstr) or _extract_cstring(symstr, st_name) != _ENTRY_SYMBOL:
                continue
            st_shndx = struct.unpack("<H", elf_data[entry + 6 : entry + 8])[0]
            st_value = struct.unpack("<Q", elf_data[entry + 8 : entry + 16])[0]
            if st_shndx != text_index or st_value != text_addr:
                raise ValueError(
                    f"AICore loader cannot extract a runnable payload from {source_name}: "
                    f"{_ENTRY_SYMBOL} is not at the start of .text (symbol is at section "
                    f"{st_shndx} offset {st_value:#x}, .text is section {text_index} at "
                    f"{text_addr:#x}). The loader jumps to offset 0 of the payload, so it "
                    f"would enter the wrong function. This means the link placed out-of-line "
                    f"code ahead of the entry point. Inspect with:\n"
                    f"  readelf -SW <file> | grep '\\.text'\n"
                    f"  readelf -sW <file> | grep {_ENTRY_SYMBOL}"
                )
            return


def _raise_unresolved_text_error(
    source_name: str,
    out_of_line: list[tuple[str, int]],
    text_relocs: list[tuple[str, int]],
) -> None:
    """Raise with an actionable diagnostic naming the offending sections.

    `KernelCompiler.compile_incore` links before extraction, so a linked image
    reaching here still has code outside `.text` or relocations `ld.lld` could
    not apply. Silently returning the raw `.text` bytes in that situation
    produces BL/B instructions with imm26=0, which on AICore manifests as CANN
    507018 watchdog timeouts or silently-wrong partial output (issue #900,
    PR #830 / issue #831).
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
        f"AICore loader cannot extract a runnable payload from {source_name}: it contains "
        f"out-of-line code or relocations against .text that linking did not resolve "
        f"(see issue #900). On device the BL/B targets in .text would branch to garbage, "
        f"producing CANN 507018 watchdog timeouts or silently-wrong partial output "
        f"(historically PR #830 / issue #831).\n\n"
        f"{detail}\n\n"
        f"Incore objects are linked with ld.lld before extraction, so this is a link that "
        f"left the image incomplete — an undefined symbol the linker kept as a relocation, "
        f"or a section it could not place. Annotating the AICore call chain with "
        f"__attribute__((always_inline)) folds it into a single .text and sidesteps both. "
        f"Verify with:\n"
        f"  readelf -SW <file> | grep '\\.text'\n"
        f"  readelf -rW <file>"
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
