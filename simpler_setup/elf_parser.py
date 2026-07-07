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
_SHT_SYMTAB = 2
_SHT_NOBITS = 8

# ELF section flags
_SHF_ALLOC = 0x2

# ELF symbol table entry layout (Elf64_Sym)
_SYM_SIZE = 24

# Mach-O Magic Numbers
MH_MAGIC_64 = 0xFEEDFACF

# ---------------------------------------------------------------------------
# fully_distributed_within_core onboard orchestration blob
# ---------------------------------------------------------------------------
# The CCEC-compiled + cce-ld-linked orchestration is a position-independent
# AICore executable (see build_dist_orch_blob). We wrap its loadable image in
# a small fixed header so the AICPU stub can locate the AICore entry / bind
# symbols after the image is copied verbatim into device GM. Keep this struct
# byte-compatible with `DistOrchBlobHeader` in the AICPU executor.
DIST_ORCH_BLOB_MAGIC = 0x42524F44  # "DORB" little-endian
DIST_ORCH_BLOB_VERSION = 1
# 64-byte header keeps the image 64-byte aligned (CALLABLE_ALIGN) once copied
# after a device allocation whose base is already >= 64-byte aligned.
DIST_ORCH_BLOB_HEADER_SIZE = 64

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


def _parse_elf64_sections(elf_data: bytes, source_name: str) -> tuple[list[dict], bytes]:
    """Return (sections, shstrtab) for an ELF64 image.

    Each section dict carries name/type/flags/addr/offset/size/link/entsize.
    """
    if len(elf_data) < 64:
        raise ValueError(f"Data too small to be a valid ELF: {source_name}")
    e_shoff = struct.unpack("<Q", elf_data[40:48])[0]
    e_shnum = struct.unpack("<H", elf_data[60:62])[0]
    e_shstrndx = struct.unpack("<H", elf_data[62:64])[0]
    if e_shnum == 0 or e_shstrndx >= e_shnum:
        raise ValueError(f"Invalid ELF section header table in {source_name}")

    shstr_off = e_shoff + e_shstrndx * _SHDR_SIZE
    shstr_data_off = struct.unpack("<Q", elf_data[shstr_off + 24 : shstr_off + 32])[0]
    shstr_data_size = struct.unpack("<Q", elf_data[shstr_off + 32 : shstr_off + 40])[0]
    shstrtab = elf_data[shstr_data_off : shstr_data_off + shstr_data_size]

    sections = []
    for i in range(e_shnum):
        base = e_shoff + i * _SHDR_SIZE
        name_off = struct.unpack("<I", elf_data[base : base + 4])[0]
        sections.append(
            {
                "name": _extract_cstring(shstrtab, name_off),
                "type": struct.unpack("<I", elf_data[base + 4 : base + 8])[0],
                "flags": struct.unpack("<Q", elf_data[base + 8 : base + 16])[0],
                "addr": struct.unpack("<Q", elf_data[base + 16 : base + 24])[0],
                "offset": struct.unpack("<Q", elf_data[base + 24 : base + 32])[0],
                "size": struct.unpack("<Q", elf_data[base + 32 : base + 40])[0],
                "link": struct.unpack("<I", elf_data[base + 40 : base + 44])[0],
                "entsize": struct.unpack("<Q", elf_data[base + 56 : base + 64])[0],
            }
        )
    return sections, shstrtab


def _resolve_elf64_symbols(elf_data: bytes, sections: list[dict], names: set[str]) -> dict[str, int]:
    """Return {symbol_name: st_value} for the requested symbols in an ELF64."""
    symtab = next((s for s in sections if s["type"] == _SHT_SYMTAB), None)
    if symtab is None:
        return {}
    strtab = sections[symtab["link"]]
    strtab_bytes = elf_data[strtab["offset"] : strtab["offset"] + strtab["size"]]
    result: dict[str, int] = {}
    count = symtab["size"] // _SYM_SIZE
    for i in range(count):
        base = symtab["offset"] + i * _SYM_SIZE
        name_off = struct.unpack("<I", elf_data[base : base + 4])[0]
        value = struct.unpack("<Q", elf_data[base + 8 : base + 16])[0]
        name = _extract_cstring(strtab_bytes, name_off)
        if name in names:
            result[name] = value
    return result


def build_dist_orch_blob(linked_elf: bytes, entry_symbol: str, bind_symbol: str) -> bytes:
    """Package a linked, position-independent AICore orchestration executable.

    ``linked_elf`` is the output of linking the CCEC-compiled orchestration +
    common object with ``cce-ld -m aicorelinux -Ttext=0`` — a flat, PC-relative
    AICore executable whose ``.text``/``.rodata``/``.data`` sections reference
    each other PC-relative, so the whole image runs at any GM base as long as
    the sections keep their relative layout.

    Returns a byte blob = DIST_ORCH_BLOB_HEADER_SIZE header + contiguous image.
    The AICPU stub copies this verbatim into device GM, reads the header, and
    computes the AICore entry/bind addresses as ``gm_base + header + off``.
    """
    sections, _ = _parse_elf64_sections(linked_elf, "<orch-blob>")
    alloc = [s for s in sections if (s["flags"] & _SHF_ALLOC) and s["size"] > 0]
    if not alloc:
        raise ValueError("orchestration blob has no allocatable sections")

    lo = min(s["addr"] for s in alloc)
    hi = max(s["addr"] + s["size"] for s in alloc)
    image = bytearray(hi - lo)
    for s in alloc:
        start = s["addr"] - lo
        if s["type"] == _SHT_NOBITS:
            continue  # .bss stays zero-filled
        image[start : start + s["size"]] = linked_elf[s["offset"] : s["offset"] + s["size"]]

    syms = _resolve_elf64_symbols(linked_elf, sections, {entry_symbol, bind_symbol})
    if entry_symbol not in syms:
        raise ValueError(f"orchestration blob missing entry symbol '{entry_symbol}'")
    if bind_symbol not in syms:
        raise ValueError(f"orchestration blob missing bind symbol '{bind_symbol}'")
    entry_off = syms[entry_symbol] - lo
    bind_off = syms[bind_symbol] - lo

    header = struct.pack(
        "<IIQQQ",
        DIST_ORCH_BLOB_MAGIC,
        DIST_ORCH_BLOB_VERSION,
        entry_off,
        bind_off,
        len(image),
    )
    header = header.ljust(DIST_ORCH_BLOB_HEADER_SIZE, b"\x00")
    return bytes(header) + bytes(image)


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
