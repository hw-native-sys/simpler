# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for ``simpler_setup.elf_parser.extract_text_section``.

The loader returns a literal ``.text`` section and jumps to offset 0, so the
image it is handed must be complete. ``KernelCompiler.compile_incore`` links
each object with ``ld.lld`` first, which applies ``.rela.text`` and folds
``.text._Z*`` group sections in; these tests cover what the parser must still
reject when that link leaves something behind — out-of-line code, surviving
relocations, allocated read-only data, or a ``kernel_entry`` that is not first. Returning such bytes
anyway causes CANN 507018 timeouts or silently-wrong partial output on device
(issue #900, PR #830 / issue #831).

The buffers are synthetic ELF64 so the tests don't depend on the CCEC
toolchain.
"""

import struct

import pytest

from simpler_setup.elf_parser import extract_text_section

_EHDR_SIZE = 64
_SHDR_SIZE = 64
_SHT_NULL = 0
_SHT_PROGBITS = 1
_SHT_SYMTAB = 2
_SHT_STRTAB = 3
_SHT_RELA = 4
_SHF_ALLOC = 2
_SHF_EXECINSTR = 4


def _pack_ehdr(*, e_shoff: int, e_shnum: int, e_shstrndx: int) -> bytes:
    e_ident = bytes([0x7F, ord("E"), ord("L"), ord("F"), 2, 1, 1, 0]) + b"\x00" * 8
    return e_ident + struct.pack(
        "<HHIQQQIHHHHHH",
        1,  # e_type ET_REL
        183,  # e_machine EM_AARCH64
        1,  # e_version
        0,  # e_entry
        0,  # e_phoff
        e_shoff,
        0,  # e_flags
        _EHDR_SIZE,
        0,  # e_phentsize
        0,  # e_phnum
        _SHDR_SIZE,
        e_shnum,
        e_shstrndx,
    )


def _pack_shdr(
    *,
    sh_name: int,
    sh_type: int,
    sh_flags: int = 0,
    sh_addr: int = 0,
    sh_offset: int = 0,
    sh_size: int = 0,
    sh_link: int = 0,
    sh_info: int = 0,
    sh_addralign: int = 1,
    sh_entsize: int = 0,
) -> bytes:
    return struct.pack(
        "<IIQQQQIIQQ",
        sh_name,
        sh_type,
        sh_flags,
        sh_addr,
        sh_offset,
        sh_size,
        sh_link,
        sh_info,
        sh_addralign,
        sh_entsize,
    )


def _pack_sym(*, st_name: int, st_shndx: int, st_value: int, st_size: int = 0) -> bytes:
    """Elf64_Sym: st_name(4) st_info(1) st_other(1) st_shndx(2) st_value(8) st_size(8)."""
    return struct.pack("<IBBHQQ", st_name, 0x12, 0, st_shndx, st_value, st_size)


def _strtab(strings: list[str]) -> tuple[bytes, list[int]]:
    """Build a string table; return bytes + per-entry name offsets."""
    buf = bytearray(b"\x00")
    offsets = []
    for s in strings:
        offsets.append(len(buf))
        buf.extend(s.encode("ascii") + b"\x00")
    return bytes(buf), offsets


class TestSingleTextNoRelocations:
    """Kernels whose call graph fully folds into ``.text`` (no
    ``.text._Z*``, no ``.rela.text``) must continue to load with
    byte-identical output — this is the regression guard for every
    kernel that currently works."""

    def test_returns_text_bytes_verbatim(self):
        text_bytes = b"\xd6\x5f\x03\xc0" * 4  # RET x 4; arbitrary payload, never executed
        elf = _build_single_text_elf(text_bytes)
        assert extract_text_section(elf) == text_bytes


class TestRejectsOutOfLineCodeSection:
    """``.text._Zfoo`` is what the compiler emits for an out-of-line
    template instantiation when it decides not to inline. The loader
    currently can't merge it, so we must fail loud rather than return a
    ``.text`` whose BLs target the unresolved symbol."""

    def test_raises_with_section_name_and_issue_link(self):
        elf = _build_elf_with_extra_text_section(extra_name=".text._Z3foo", extra_size=4)
        with pytest.raises(ValueError, match=r"issue #900"):
            extract_text_section(elf)

    def test_error_names_the_offending_section(self):
        elf = _build_elf_with_extra_text_section(extra_name=".text._Z3foo", extra_size=4)
        with pytest.raises(ValueError) as excinfo:
            extract_text_section(elf)
        assert ".text._Z3foo" in str(excinfo.value)

    def test_error_mentions_always_inline_workaround(self):
        elf = _build_elf_with_extra_text_section(extra_name=".text._Z3foo", extra_size=4)
        with pytest.raises(ValueError, match=r"always_inline"):
            extract_text_section(elf)


class TestRejectsTextRelocations:
    """A ``.rela.text`` section with any entries means ``.text`` has
    unresolved BL/B/ADRP targets. Even without ``.text._Z*`` (e.g. the
    target is an external symbol), returning ``.text`` verbatim is
    broken — must fail loud."""

    def test_raises_with_issue_link(self):
        elf = _build_elf_with_rela_text(reloc_count=2)
        with pytest.raises(ValueError, match=r"issue #900"):
            extract_text_section(elf)

    def test_error_reports_entry_count(self):
        elf = _build_elf_with_rela_text(reloc_count=2)
        with pytest.raises(ValueError) as excinfo:
            extract_text_section(elf)
        assert "2 entries" in str(excinfo.value)


@pytest.mark.parametrize("name", (".rodata", ".rodata.cst4", ".rodata.str1.1"))
@pytest.mark.parametrize("linked", (False, True))
def test_rejects_allocated_read_only_data_that_text_only_loading_would_drop(name, linked):
    elf = bytearray(_build_elf_with_extra_text_section(extra_name=name, extra_size=24))
    struct.pack_into("<Q", elf, _EHDR_SIZE + 2 * _SHDR_SIZE + 8, _SHF_ALLOC)
    if linked:
        struct.pack_into("<H", elf, 16, 2)  # ET_EXEC; its text relocations have already been resolved.
    with pytest.raises(ValueError, match=r"read-only data.*\.rodata"):
        extract_text_section(bytes(elf))


@pytest.mark.parametrize("size,flags", ((0, _SHF_ALLOC), (24, 0)))
def test_empty_or_nonallocated_rodata_does_not_change_the_text_payload(size, flags):
    elf = bytearray(_build_elf_with_extra_text_section(extra_name=".rodata", extra_size=size))
    struct.pack_into("<Q", elf, _EHDR_SIZE + 2 * _SHDR_SIZE + 8, flags)
    assert extract_text_section(bytes(elf)) == b"\x00" * 4


class TestEntryPointMustBeFirst:
    """The loader jumps to offset 0 of the payload. Linking merges ``.text.*``
    into ``.text`` and is free to order the pieces, so a kernel with out-of-line
    code can end up with another function ahead of ``kernel_entry`` — which
    would silently enter the wrong function."""

    def test_entry_at_start_of_text_loads(self):
        text_bytes = b"\xd6\x5f\x03\xc0" * 4
        elf = _build_elf_with_symtab(text_bytes, text_addr=0x1000, entry_value=0x1000, entry_shndx=1)
        assert extract_text_section(elf) == text_bytes

    def test_entry_offset_into_text_raises(self):
        elf = _build_elf_with_symtab(b"\x00" * 16, text_addr=0x1000, entry_value=0x1008, entry_shndx=1)
        with pytest.raises(ValueError, match=r"kernel_entry is not at the start of \.text"):
            extract_text_section(elf)

    def test_entry_in_another_section_raises(self):
        elf = _build_elf_with_symtab(b"\x00" * 16, text_addr=0x1000, entry_value=0x1000, entry_shndx=2)
        with pytest.raises(ValueError, match=r"kernel_entry is not at the start of \.text"):
            extract_text_section(elf)

    def test_object_without_symtab_is_left_alone(self):
        """Orchestration objects and hand-built fixtures carry no symbol table;
        the check must not turn those into failures."""
        text_bytes = b"\xd6\x5f\x03\xc0"
        assert extract_text_section(_build_single_text_elf(text_bytes)) == text_bytes


class TestMissingText:
    def test_no_text_section_raises(self):
        shstr_bytes, [shstr_name] = _strtab([".shstrtab"])
        e_shoff = _EHDR_SIZE
        shstr_off = e_shoff + 2 * _SHDR_SIZE
        ehdr = _pack_ehdr(e_shoff=e_shoff, e_shnum=2, e_shstrndx=1)
        shdrs = _pack_shdr(sh_name=0, sh_type=_SHT_NULL) + _pack_shdr(
            sh_name=shstr_name,
            sh_type=_SHT_STRTAB,
            sh_offset=shstr_off,
            sh_size=len(shstr_bytes),
        )
        elf = ehdr + shdrs + shstr_bytes
        with pytest.raises(ValueError, match=r"\.text section not found"):
            extract_text_section(elf)


class TestMalformedHeaderRejected:
    """The parser is the only thing between bytes-on-disk and a slice of
    ``.text`` that runs on AICore. Malformed headers must surface as
    clear ``ValueError``s rather than silent garbage or opaque
    ``struct.error``s — see issue #900 review feedback."""

    def test_shstrndx_past_shnum_raises(self):
        # e_shstrndx=2 but e_shnum=1 → string table index points past the table
        text_bytes = b"\x00\x00\x00\x00"
        e_shoff = _EHDR_SIZE
        ehdr = _pack_ehdr(e_shoff=e_shoff, e_shnum=1, e_shstrndx=2)
        shdrs = _pack_shdr(sh_name=0, sh_type=_SHT_NULL)
        elf = ehdr + shdrs + text_bytes
        with pytest.raises(ValueError, match=r"e_shstrndx"):
            extract_text_section(elf)

    def test_section_header_table_past_buffer_raises(self):
        # e_shoff points past the end of elf_data
        elf = _pack_ehdr(e_shoff=1 << 32, e_shnum=1, e_shstrndx=0)
        with pytest.raises(ValueError, match=r"section header table is out of bounds"):
            extract_text_section(elf)


# ---------------------------------------------------------------------------
# Synthetic ELF builders
# ---------------------------------------------------------------------------


def _build_single_text_elf(text_bytes: bytes) -> bytes:
    shstr_bytes, [text_name, shstr_name] = _strtab([".text", ".shstrtab"])

    shdr_count = 3
    e_shoff = _EHDR_SIZE
    data_start = e_shoff + shdr_count * _SHDR_SIZE

    text_off = data_start
    shstr_off = text_off + len(text_bytes)

    ehdr = _pack_ehdr(e_shoff=e_shoff, e_shnum=shdr_count, e_shstrndx=2)
    shdrs = (
        _pack_shdr(sh_name=0, sh_type=_SHT_NULL)
        + _pack_shdr(
            sh_name=text_name,
            sh_type=_SHT_PROGBITS,
            sh_flags=_SHF_ALLOC | _SHF_EXECINSTR,
            sh_offset=text_off,
            sh_size=len(text_bytes),
            sh_addralign=4,
        )
        + _pack_shdr(
            sh_name=shstr_name,
            sh_type=_SHT_STRTAB,
            sh_offset=shstr_off,
            sh_size=len(shstr_bytes),
        )
    )
    return ehdr + shdrs + text_bytes + shstr_bytes


def _build_elf_with_extra_text_section(*, extra_name: str, extra_size: int) -> bytes:
    text_bytes = b"\x00" * 4
    extra_bytes = b"\x00" * extra_size
    shstr_bytes, [text_name, extra_text_name, shstr_name] = _strtab([".text", extra_name, ".shstrtab"])

    shdr_count = 4
    e_shoff = _EHDR_SIZE
    cursor = e_shoff + shdr_count * _SHDR_SIZE

    def place(b: bytes) -> int:
        nonlocal cursor
        off = cursor
        cursor += len(b)
        return off

    text_off = place(text_bytes)
    extra_off = place(extra_bytes)
    shstr_off = place(shstr_bytes)

    ehdr = _pack_ehdr(e_shoff=e_shoff, e_shnum=shdr_count, e_shstrndx=3)
    shdrs = b"".join(
        [
            _pack_shdr(sh_name=0, sh_type=_SHT_NULL),
            _pack_shdr(
                sh_name=text_name,
                sh_type=_SHT_PROGBITS,
                sh_flags=_SHF_ALLOC | _SHF_EXECINSTR,
                sh_offset=text_off,
                sh_size=len(text_bytes),
                sh_addralign=4,
            ),
            _pack_shdr(
                sh_name=extra_text_name,
                sh_type=_SHT_PROGBITS,
                sh_flags=_SHF_ALLOC | _SHF_EXECINSTR,
                sh_offset=extra_off,
                sh_size=len(extra_bytes),
                sh_addralign=4,
            ),
            _pack_shdr(
                sh_name=shstr_name,
                sh_type=_SHT_STRTAB,
                sh_offset=shstr_off,
                sh_size=len(shstr_bytes),
            ),
        ]
    )
    return ehdr + shdrs + text_bytes + extra_bytes + shstr_bytes


def _build_elf_with_symtab(text_bytes: bytes, *, text_addr: int, entry_value: int, entry_shndx: int) -> bytes:
    """A linked-image shape: ``.text`` at a load address plus a symbol table
    naming ``kernel_entry``."""
    sym_bytes, [entry_name] = _strtab(["kernel_entry"])
    symtab_bytes = _pack_sym(st_name=0, st_shndx=0, st_value=0) + _pack_sym(
        st_name=entry_name, st_shndx=entry_shndx, st_value=entry_value, st_size=len(text_bytes)
    )
    shstr_bytes, [text_name, symtab_name, strtab_name, shstr_name] = _strtab(
        [".text", ".symtab", ".strtab", ".shstrtab"]
    )

    shdr_count = 5
    e_shoff = _EHDR_SIZE
    cursor = e_shoff + shdr_count * _SHDR_SIZE

    def place(b: bytes) -> int:
        nonlocal cursor
        off = cursor
        cursor += len(b)
        return off

    text_off = place(text_bytes)
    symtab_off = place(symtab_bytes)
    sym_str_off = place(sym_bytes)
    shstr_off = place(shstr_bytes)

    ehdr = _pack_ehdr(e_shoff=e_shoff, e_shnum=shdr_count, e_shstrndx=4)
    shdrs = b"".join(
        [
            _pack_shdr(sh_name=0, sh_type=_SHT_NULL),
            _pack_shdr(
                sh_name=text_name,
                sh_type=_SHT_PROGBITS,
                sh_flags=_SHF_ALLOC | _SHF_EXECINSTR,
                sh_addr=text_addr,
                sh_offset=text_off,
                sh_size=len(text_bytes),
                sh_addralign=4,
            ),
            _pack_shdr(
                sh_name=symtab_name,
                sh_type=_SHT_SYMTAB,
                sh_offset=symtab_off,
                sh_size=len(symtab_bytes),
                sh_link=3,
                sh_entsize=24,
            ),
            _pack_shdr(
                sh_name=strtab_name,
                sh_type=_SHT_STRTAB,
                sh_offset=sym_str_off,
                sh_size=len(sym_bytes),
            ),
            _pack_shdr(
                sh_name=shstr_name,
                sh_type=_SHT_STRTAB,
                sh_offset=shstr_off,
                sh_size=len(shstr_bytes),
            ),
        ]
    )
    return ehdr + shdrs + text_bytes + symtab_bytes + sym_bytes + shstr_bytes


def _build_elf_with_rela_text(*, reloc_count: int) -> bytes:
    text_bytes = b"\x00" * 4
    rela_bytes = b"\x00" * (24 * reloc_count)  # opaque; loader only counts entries
    shstr_bytes, [text_name, rela_name, shstr_name] = _strtab([".text", ".rela.text", ".shstrtab"])

    shdr_count = 4
    e_shoff = _EHDR_SIZE
    cursor = e_shoff + shdr_count * _SHDR_SIZE

    def place(b: bytes) -> int:
        nonlocal cursor
        off = cursor
        cursor += len(b)
        return off

    text_off = place(text_bytes)
    rela_off = place(rela_bytes)
    shstr_off = place(shstr_bytes)

    ehdr = _pack_ehdr(e_shoff=e_shoff, e_shnum=shdr_count, e_shstrndx=3)
    shdrs = b"".join(
        [
            _pack_shdr(sh_name=0, sh_type=_SHT_NULL),
            _pack_shdr(
                sh_name=text_name,
                sh_type=_SHT_PROGBITS,
                sh_flags=_SHF_ALLOC | _SHF_EXECINSTR,
                sh_offset=text_off,
                sh_size=len(text_bytes),
                sh_addralign=4,
            ),
            _pack_shdr(
                sh_name=rela_name,
                sh_type=_SHT_RELA,
                sh_offset=rela_off,
                sh_size=len(rela_bytes),
                sh_entsize=24,
            ),
            _pack_shdr(
                sh_name=shstr_name,
                sh_type=_SHT_STRTAB,
                sh_offset=shstr_off,
                sh_size=len(shstr_bytes),
            ),
        ]
    )
    return ehdr + shdrs + text_bytes + rela_bytes + shstr_bytes
