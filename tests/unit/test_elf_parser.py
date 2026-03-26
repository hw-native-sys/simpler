"""Unit tests for python/elf_parser.py — ELF64 and Mach-O binary parsing."""

import struct
import tempfile
from pathlib import Path

import pytest

from elf_parser import extract_text_section, _extract_cstring


# =============================================================================
# Helper: build minimal ELF64 binary with a .text section
# =============================================================================

def _build_minimal_elf64(text_content: bytes, include_text: bool = True) -> bytes:
    """Build a minimal ELF64 relocatable object with an optional .text section.

    Layout:
        [ELF header 64B]
        [.text section data]
        [string table data]
        [section headers: NULL + .text + .shstrtab]
    """
    # String table: \0 .text\0 .shstrtab\0
    if include_text:
        strtab = b"\x00.text\x00.shstrtab\x00"
        name_text = 1       # offset of ".text" in strtab
        name_shstrtab = 7   # offset of ".shstrtab" in strtab
    else:
        strtab = b"\x00.data\x00.shstrtab\x00"
        name_text = 1       # will name it ".data" instead
        name_shstrtab = 7

    text_size = len(text_content)
    strtab_size = len(strtab)

    # Offsets: header=64, then text data, then strtab, then section headers
    text_offset = 64
    strtab_offset = text_offset + text_size
    sh_offset = strtab_offset + strtab_size

    num_sections = 3  # NULL + .text/.data + .shstrtab
    shstrtab_index = 2

    # ELF header (64 bytes for ELF64)
    e_ident = bytes([
        0x7F, ord('E'), ord('L'), ord('F'),  # magic
        2,     # ELFCLASS64
        1,     # ELFDATA2LSB
        1,     # EV_CURRENT
        0,     # ELFOSABI_NONE
        0, 0, 0, 0, 0, 0, 0, 0  # padding
    ])
    header = e_ident
    header += struct.pack('<H', 1)       # e_type = ET_REL
    header += struct.pack('<H', 0x3E)    # e_machine = EM_X86_64
    header += struct.pack('<I', 1)       # e_version
    header += struct.pack('<Q', 0)       # e_entry
    header += struct.pack('<Q', 0)       # e_phoff
    header += struct.pack('<Q', sh_offset)  # e_shoff
    header += struct.pack('<I', 0)       # e_flags
    header += struct.pack('<H', 64)      # e_ehsize
    header += struct.pack('<H', 0)       # e_phentsize
    header += struct.pack('<H', 0)       # e_phnum
    header += struct.pack('<H', 64)      # e_shentsize
    header += struct.pack('<H', num_sections)  # e_shnum
    header += struct.pack('<H', shstrtab_index)  # e_shstrndx

    assert len(header) == 64

    # Section headers (each 64 bytes for ELF64)
    def _sh(name, sh_type, offset, size):
        sh = struct.pack('<I', name)     # sh_name
        sh += struct.pack('<I', sh_type) # sh_type
        sh += struct.pack('<Q', 0)       # sh_flags
        sh += struct.pack('<Q', 0)       # sh_addr
        sh += struct.pack('<Q', offset)  # sh_offset
        sh += struct.pack('<Q', size)    # sh_size
        sh += struct.pack('<I', 0)       # sh_link
        sh += struct.pack('<I', 0)       # sh_info
        sh += struct.pack('<Q', 1)       # sh_addralign
        sh += struct.pack('<Q', 0)       # sh_entsize
        return sh

    sh_null = _sh(0, 0, 0, 0)                                    # SHT_NULL
    sh_text = _sh(name_text, 1, text_offset, text_size)           # SHT_PROGBITS
    sh_strtab = _sh(name_shstrtab, 3, strtab_offset, strtab_size) # SHT_STRTAB

    return header + text_content + strtab + sh_null + sh_text + sh_strtab


# =============================================================================
# Helper: build minimal Mach-O 64-bit binary with __text section
# =============================================================================

def _build_minimal_macho64(text_content: bytes) -> bytes:
    """Build a minimal Mach-O 64-bit object with a __TEXT,__text section."""
    text_size = len(text_content)

    # Mach-O header (32 bytes)
    mh_magic = struct.pack('<I', 0xFEEDFACF)  # MH_MAGIC_64
    cputype = struct.pack('<I', 0x01000007)    # CPU_TYPE_X86_64
    cpusubtype = struct.pack('<I', 3)          # CPU_SUBTYPE_ALL
    filetype = struct.pack('<I', 1)            # MH_OBJECT
    ncmds = struct.pack('<I', 1)               # 1 load command
    # sizeofcmds will be filled below
    flags = struct.pack('<I', 0)
    reserved = struct.pack('<I', 0)

    # LC_SEGMENT_64 (72 bytes) + one section_64 (80 bytes)
    cmdsize = 72 + 80
    sizeofcmds = struct.pack('<I', cmdsize)

    macho_header = mh_magic + cputype + cpusubtype + filetype + ncmds + sizeofcmds + flags + reserved
    assert len(macho_header) == 32

    # segment_command_64 (72 bytes)
    cmd = struct.pack('<I', 0x19)          # LC_SEGMENT_64
    cmd_cmdsize = struct.pack('<I', cmdsize)
    segname = b'__TEXT'.ljust(16, b'\x00')
    vmaddr = struct.pack('<Q', 0)
    vmsize = struct.pack('<Q', text_size)

    # text data starts right after header + load command
    text_fileoff = 32 + cmdsize
    fileoff = struct.pack('<Q', text_fileoff)
    filesize = struct.pack('<Q', text_size)
    maxprot = struct.pack('<I', 7)
    initprot = struct.pack('<I', 5)
    nsects = struct.pack('<I', 1)
    seg_flags = struct.pack('<I', 0)

    segment = cmd + cmd_cmdsize + segname + vmaddr + vmsize + fileoff + filesize + maxprot + initprot + nsects + seg_flags
    assert len(segment) == 72

    # section_64 (80 bytes)
    sectname = b'__text'.ljust(16, b'\x00')
    sect_segname = b'__TEXT'.ljust(16, b'\x00')
    addr = struct.pack('<Q', 0)
    size = struct.pack('<Q', text_size)
    offset = struct.pack('<I', text_fileoff)
    align = struct.pack('<I', 0)
    reloff = struct.pack('<I', 0)
    nreloc = struct.pack('<I', 0)
    sect_flags = struct.pack('<I', 0)
    reserved1 = struct.pack('<I', 0)
    reserved2 = struct.pack('<I', 0)
    reserved3 = struct.pack('<I', 0)

    section = sectname + sect_segname + addr + size + offset + align + reloff + nreloc + sect_flags + reserved1 + reserved2 + reserved3
    assert len(section) == 80

    return macho_header + segment + section + text_content


# =============================================================================
# Tests
# =============================================================================

class TestExtractTextSection:
    """Tests for extract_text_section()."""

    def test_extract_text_elf64(self):
        """ELF64 .text section is correctly extracted."""
        text_data = b"\xDE\xAD\xBE\xEF" * 16
        elf = _build_minimal_elf64(text_data)
        result = extract_text_section(elf)
        assert result == text_data

    def test_extract_text_macho64(self):
        """Mach-O 64-bit __text section is correctly extracted."""
        text_data = b"\xCA\xFE\xBA\xBE" * 8
        macho = _build_minimal_macho64(text_data)
        result = extract_text_section(macho)
        assert result == text_data

    def test_invalid_magic_number(self):
        """Non-ELF/Mach-O data raises ValueError."""
        bad_data = b"\x00\x01\x02\x03" + b"\x00" * 100
        with pytest.raises(ValueError, match="Not a valid ELF or Mach-O"):
            extract_text_section(bad_data)

    def test_too_small_input(self):
        """Data shorter than 4 bytes raises ValueError."""
        with pytest.raises(ValueError, match="too small"):
            extract_text_section(b"\x7F")

    def test_missing_text_section(self):
        """Valid ELF without .text section raises ValueError."""
        # Build ELF with .data instead of .text
        elf = _build_minimal_elf64(b"\x00" * 16, include_text=False)
        with pytest.raises(ValueError, match=".text section not found"):
            extract_text_section(elf)

    def test_file_path_input(self, tmp_path):
        """File path input is correctly read and parsed."""
        text_data = b"\xAB\xCD" * 32
        elf = _build_minimal_elf64(text_data)

        obj_file = tmp_path / "test.o"
        obj_file.write_bytes(elf)

        result = extract_text_section(str(obj_file))
        assert result == text_data

    def test_file_not_found(self):
        """Non-existent file path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            extract_text_section("/nonexistent/path/to/file.o")


class TestExtractCstring:
    """Tests for _extract_cstring()."""

    def test_basic_extraction(self):
        """Extract null-terminated string at offset 0."""
        result = _extract_cstring(b"hello\x00world\x00", 0)
        assert result == "hello"

    def test_extraction_at_offset(self):
        """Extract string at non-zero offset."""
        result = _extract_cstring(b"hello\x00world\x00", 6)
        assert result == "world"

    def test_no_null_terminator(self):
        """String without null terminator reads to end."""
        result = _extract_cstring(b"abcdef", 0)
        assert result == "abcdef"
