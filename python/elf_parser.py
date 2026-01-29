"""
ELF64 Parser for AICore Kernel Binaries

Pure Python implementation for extracting .text section from ELF64 .o files.
Based on the C++ implementation in binary_loader.cpp.
"""

import struct
from pathlib import Path
from typing import Union


# ELF Magic Numbers
ELFMAG0 = 0x7F
ELFMAG1 = ord('E')
ELFMAG2 = ord('L')
ELFMAG3 = ord('F')


def extract_text_section(elf_input: Union[str, Path, bytes]) -> bytes:
    """
    Extract .text section from an ELF64 .o file.

    Args:
        elf_input: Either a path to the ELF .o file (str/Path) or the ELF binary data (bytes)

    Returns:
        Binary data of the .text section

    Raises:
        FileNotFoundError: If file path is provided and does not exist
        ValueError: If data is not a valid ELF or .text section not found
    """
    # Handle input: either path or bytes
    if isinstance(elf_input, bytes):
        elf_data = elf_input
        source_name = "<bytes>"
    else:
        path = Path(elf_input)
        if not path.exists():
            raise FileNotFoundError(f"ELF file not found: {elf_input}")
        with open(elf_input, 'rb') as f:
            elf_data = f.read()
        source_name = str(elf_input)

    if len(elf_data) < 64:
        raise ValueError(f"Data too small to be a valid ELF: {source_name}")

    # Step 1: Parse ELF64 Header (64 bytes)
    # Reference: https://en.wikipedia.org/wiki/Executable_and_Linkable_Format
    # Elf64_Ehdr structure:
    #   e_ident[16]     - Magic number and file class
    #   e_type          - Object file type (2 bytes)
    #   e_machine       - Architecture (2 bytes)
    #   e_version       - Version (4 bytes)
    #   e_entry         - Entry point (8 bytes)
    #   e_phoff         - Program header offset (8 bytes)
    #   e_shoff         - Section header offset (8 bytes) <-- at offset 40
    #   e_flags         - Flags (4 bytes)
    #   e_ehsize        - ELF header size (2 bytes)
    #   e_phentsize     - Program header entry size (2 bytes)
    #   e_phnum         - Program header count (2 bytes)
    #   e_shentsize     - Section header entry size (2 bytes)
    #   e_shnum         - Section header count (2 bytes) <-- at offset 60
    #   e_shstrndx      - String table index (2 bytes) <-- at offset 62

    # Verify ELF magic number
    if (elf_data[0] != ELFMAG0 or
        elf_data[1] != ELFMAG1 or
        elf_data[2] != ELFMAG2 or
        elf_data[3] != ELFMAG3):
        raise ValueError(f"Not a valid ELF file: {source_name}")

    # Extract section header table info from ELF header
    # Use little-endian ('<') format
    e_shoff = struct.unpack('<Q', elf_data[40:48])[0]      # Section header offset
    e_shnum = struct.unpack('<H', elf_data[60:62])[0]      # Section header count
    e_shstrndx = struct.unpack('<H', elf_data[62:64])[0]   # String table index

    # Step 2: Parse Section Headers
    # Each Elf64_Shdr is 64 bytes:
    #   sh_name      - Section name offset (4 bytes)
    #   sh_type      - Section type (4 bytes)
    #   sh_flags     - Section flags (8 bytes)
    #   sh_addr      - Section address (8 bytes)
    #   sh_offset    - File offset (8 bytes) <-- at +24
    #   sh_size      - Section size (8 bytes) <-- at +32
    #   sh_link      - Link to another section (4 bytes)
    #   sh_info      - Additional info (4 bytes)
    #   sh_addralign - Alignment (8 bytes)
    #   sh_entsize   - Entry size (8 bytes)

    # Get string table section header
    shstr_offset = e_shoff + e_shstrndx * 64
    shstr_sh_offset = struct.unpack('<Q', elf_data[shstr_offset+24:shstr_offset+32])[0]
    shstr_sh_size = struct.unpack('<Q', elf_data[shstr_offset+32:shstr_offset+40])[0]

    # Extract string table
    strtab = elf_data[shstr_sh_offset:shstr_sh_offset+shstr_sh_size]

    # Step 3: Find .text section
    for i in range(e_shnum):
        section_offset = e_shoff + i * 64

        # Parse section header
        sh_name = struct.unpack('<I', elf_data[section_offset:section_offset+4])[0]
        sh_offset = struct.unpack('<Q', elf_data[section_offset+24:section_offset+32])[0]
        sh_size = struct.unpack('<Q', elf_data[section_offset+32:section_offset+40])[0]

        # Get section name from string table
        section_name = _extract_cstring(strtab, sh_name)

        if section_name == '.text':
            # Extract .text section binary data
            text_data = elf_data[sh_offset:sh_offset+sh_size]
            print(f"Loaded .text section from {source_name} (size: {sh_size} bytes)")
            return text_data

    raise ValueError(f".text section not found in: {source_name}")


def _extract_cstring(data: bytes, offset: int) -> str:
    """
    Extract a null-terminated C string from bytes.

    Args:
        data: Byte data
        offset: Starting offset

    Returns:
        Decoded string
    """
    end = data.find(b'\x00', offset)
    if end == -1:
        end = len(data)
    return data[offset:end].decode('ascii', errors='ignore')
