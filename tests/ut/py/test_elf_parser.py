# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for python/elf_parser.py - ELF64 and Mach-O .text extraction."""

import struct
import tempfile

import pytest

from simpler_setup.elf_parser import _extract_cstring, extract_text_section


def _build_elf64_with_text(text_data: bytes) -> bytes:
    """Build a minimal ELF64 .o file with a .text section."""
    # String table: \0.text\0.shstrtab\0
    strtab = b"\x00.text\x00.shstrtab\x00"
    text_name_offset = 1  # offset of ".text" in strtab
    shstrtab_name_offset = 7  # offset of ".shstrtab" in strtab

    # ELF header (64 bytes)
    e_shoff = 64  # section headers right after ELF header
    e_shnum = 3  # null + .text + .shstrtab
    e_shstrndx = 2  # .shstrtab is section 2

    elf_header = bytearray(64)
    elf_header[0:4] = b"\x7fELF"
    elf_header[4] = 2  # 64-bit
    elf_header[5] = 1  # little-endian
    elf_header[6] = 1  # version
    struct.pack_into("<H", elf_header, 18, 1)  # e_type = ET_REL
    struct.pack_into("<H", elf_header, 52, 64)  # e_shentsize
    struct.pack_into("<Q", elf_header, 40, e_shoff)
    struct.pack_into("<H", elf_header, 60, e_shnum)
    struct.pack_into("<H", elf_header, 62, e_shstrndx)

    # Data follows section headers: text_data then strtab
    data_offset = e_shoff + 64 * e_shnum  # after headers
    text_offset = data_offset
    strtab_offset = text_offset + len(text_data)

    # Section headers (64 bytes each)
    # Section 0: null
    sh_null = bytearray(64)

    # Section 1: .text
    sh_text = bytearray(64)
    struct.pack_into("<I", sh_text, 0, text_name_offset)  # sh_name
    struct.pack_into("<I", sh_text, 4, 1)  # SHT_PROGBITS
    struct.pack_into("<Q", sh_text, 24, text_offset)  # sh_offset
    struct.pack_into("<Q", sh_text, 32, len(text_data))  # sh_size

    # Section 2: .shstrtab
    sh_strtab = bytearray(64)
    struct.pack_into("<I", sh_strtab, 0, shstrtab_name_offset)  # sh_name
    struct.pack_into("<I", sh_strtab, 4, 3)  # SHT_STRTAB
    struct.pack_into("<Q", sh_strtab, 24, strtab_offset)  # sh_offset
    struct.pack_into("<Q", sh_strtab, 32, len(strtab))  # sh_size

    return bytes(elf_header) + bytes(sh_null) + bytes(sh_text) + bytes(sh_strtab) + text_data + strtab


def _build_macho64_with_text(text_data: bytes) -> bytes:
    """Build a minimal Mach-O 64-bit .o file with __text section."""
    # Header (32 bytes)
    header = bytearray(32)
    struct.pack_into("<I", header, 0, 0xFEEDFACF)  # magic
    struct.pack_into("<I", header, 4, 0x0100000C)  # cputype (ARM64)
    struct.pack_into("<I", header, 12, 1)  # filetype MH_OBJECT
    struct.pack_into("<I", header, 16, 1)  # ncmds

    # LC_SEGMENT_64 command
    segment_header = bytearray(72)
    struct.pack_into("<I", segment_header, 0, 0x19)  # LC_SEGMENT_64

    # One section: __text
    section = bytearray(80)
    section[0:6] = b"__text"
    section[16:22] = b"__TEXT"

    text_offset = 32 + 72 + 80  # after header + segment + section
    struct.pack_into("<Q", section, 40, len(text_data))  # size
    struct.pack_into("<I", section, 48, text_offset)  # offset

    cmdsize = 72 + 80
    struct.pack_into("<I", segment_header, 4, cmdsize)  # cmdsize
    struct.pack_into("<I", segment_header, 64, 1)  # nsects
    struct.pack_into("<I", header, 20, cmdsize)  # sizeofcmds

    return bytes(header) + bytes(segment_header) + bytes(section) + text_data


# =============================================================================
# ELF64 tests
# =============================================================================


class TestELF64:
    def test_extract_text(self):
        text_data = b"\x01\x02\x03\x04\x05"
        elf = _build_elf64_with_text(text_data)
        result = extract_text_section(elf)
        assert result == text_data

    def test_missing_text_section(self):
        # Build ELF with only null + .shstrtab (no .text)
        strtab = b"\x00.shstrtab\x00"
        e_shoff = 64
        e_shnum = 2
        e_shstrndx = 1

        elf_header = bytearray(64)
        elf_header[0:4] = b"\x7fELF"
        elf_header[4] = 2
        elf_header[5] = 1
        elf_header[6] = 1
        struct.pack_into("<Q", elf_header, 40, e_shoff)
        struct.pack_into("<H", elf_header, 60, e_shnum)
        struct.pack_into("<H", elf_header, 62, e_shstrndx)

        data_offset = e_shoff + 64 * e_shnum
        sh_null = bytearray(64)
        sh_strtab = bytearray(64)
        struct.pack_into("<I", sh_strtab, 0, 1)
        struct.pack_into("<I", sh_strtab, 4, 3)
        struct.pack_into("<Q", sh_strtab, 24, data_offset)
        struct.pack_into("<Q", sh_strtab, 32, len(strtab))

        elf = bytes(elf_header) + bytes(sh_null) + bytes(sh_strtab) + strtab
        with pytest.raises(ValueError, match=".text section not found"):
            extract_text_section(elf)

    def test_truncated_header(self):
        with pytest.raises(ValueError):
            extract_text_section(b"\x7fELF" + b"\x00" * 10)


# =============================================================================
# Mach-O tests
# =============================================================================


class TestMachO:
    def test_extract_text(self):
        text_data = b"\xaa\xbb\xcc\xdd"
        macho = _build_macho64_with_text(text_data)
        result = extract_text_section(macho)
        assert result == text_data

    def test_missing_text_section(self):
        # Header with no sections
        header = bytearray(32)
        struct.pack_into("<I", header, 0, 0xFEEDFACF)
        struct.pack_into("<I", header, 16, 0)  # ncmds = 0
        with pytest.raises(ValueError, match="__text section not found"):
            extract_text_section(bytes(header))


# =============================================================================
# Format detection
# =============================================================================


class TestFormatDetection:
    def test_unknown_format(self):
        with pytest.raises(ValueError, match="Not a valid ELF or Mach-O"):
            extract_text_section(b"\x00\x00\x00\x00" + b"\x00" * 60)

    def test_too_small(self):
        with pytest.raises(ValueError, match="too small"):
            extract_text_section(b"\x01\x02")

    def test_file_path(self):
        text_data = b"\xde\xad\xbe\xef"
        elf = _build_elf64_with_text(text_data)
        with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as f:
            f.write(elf)
            f.flush()
            result = extract_text_section(f.name)
            assert result == text_data

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            extract_text_section("/nonexistent/path.o")


# =============================================================================
# _extract_cstring
# =============================================================================


class TestExtractCString:
    def test_basic(self):
        data = b"hello\x00world\x00"
        assert _extract_cstring(data, 0) == "hello"
        assert _extract_cstring(data, 6) == "world"

    def test_no_null_terminator(self):
        data = b"unterminated"
        assert _extract_cstring(data, 0) == "unterminated"

    def test_empty_string(self):
        data = b"\x00rest"
        assert _extract_cstring(data, 0) == ""
