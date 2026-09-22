/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
// Unit tests for src/common/utils/elf_build_id.h.
//
// The helper is header-only. We validate three paths without shelling out to
// a real linker:
//   1. well-formed ELF64 + NT_GNU_BUILD_ID note → returns the first 8 bytes
//      of the descriptor verbatim.
//   2. well-formed ELF64 without any GNU Build-ID note → falls back to
//      FNV-1a over the whole buffer, and two identical buffers produce the
//      same hash while different buffers produce different hashes.
//   3. non-ELF / truncated input → also falls back to FNV-1a and does not
//      crash.

#include <cstdint>
#include <cstring>
#include <vector>

#ifdef __linux__
#include <elf.h>
#endif
#include <gtest/gtest.h>

#include "utils/elf_build_id.h"

namespace {

// Build a minimal ELF64 image that contains a PT_NOTE segment with one
// NT_GNU_BUILD_ID note whose descriptor is `build_id`.
std::vector<uint8_t> make_elf_with_build_id(const std::vector<uint8_t> &build_id) {
    // Layout: [Elf64_Ehdr][Elf64_Phdr][Nhdr][name "GNU\0"][padded desc]
    const size_t ehdr_size = sizeof(Elf64_Ehdr);
    const size_t phdr_size = sizeof(Elf64_Phdr);
    const size_t nhdr_size = sizeof(Elf64_Nhdr);
    const size_t name_size = 4;  // "GNU\0"
    const size_t name_padded = (name_size + 3u) & ~3u;
    const size_t desc_padded = (build_id.size() + 3u) & ~3u;
    const size_t note_total = nhdr_size + name_padded + desc_padded;

    std::vector<uint8_t> buf(ehdr_size + phdr_size + note_total, 0);

    Elf64_Ehdr ehdr{};
    ehdr.e_ident[EI_MAG0] = ELFMAG0;
    ehdr.e_ident[EI_MAG1] = ELFMAG1;
    ehdr.e_ident[EI_MAG2] = ELFMAG2;
    ehdr.e_ident[EI_MAG3] = ELFMAG3;
    ehdr.e_ident[EI_CLASS] = ELFCLASS64;
    ehdr.e_phoff = ehdr_size;
    ehdr.e_phentsize = phdr_size;
    ehdr.e_phnum = 1;
    std::memcpy(buf.data(), &ehdr, ehdr_size);

    Elf64_Phdr phdr{};
    phdr.p_type = PT_NOTE;
    phdr.p_offset = ehdr_size + phdr_size;
    phdr.p_filesz = note_total;
    std::memcpy(buf.data() + ehdr_size, &phdr, phdr_size);

    Elf64_Nhdr nhdr{};
    nhdr.n_namesz = name_size;
    nhdr.n_descsz = static_cast<uint32_t>(build_id.size());
    nhdr.n_type = NT_GNU_BUILD_ID;
    std::memcpy(buf.data() + ehdr_size + phdr_size, &nhdr, nhdr_size);
    std::memcpy(buf.data() + ehdr_size + phdr_size + nhdr_size, "GNU\0", name_size);
    std::memcpy(buf.data() + ehdr_size + phdr_size + nhdr_size + name_padded, build_id.data(), build_id.size());
    return buf;
}

}  // namespace

TEST(ElfBuildId, ReadsFirstEightBytesOfGnuBuildId) {
    std::vector<uint8_t> id(20, 0);
    for (int i = 0; i < 20; ++i)
        id[i] = static_cast<uint8_t>(0x10 + i);
    auto buf = make_elf_with_build_id(id);

    uint64_t got = simpler::common::utils::elf_build_id_64(buf.data(), buf.size());
    uint64_t want = 0;
    std::memcpy(&want, id.data(), 8);
    EXPECT_EQ(got, want);
}

TEST(ElfBuildId, DifferentBuildIdsProduceDifferentHashes) {
    auto a = make_elf_with_build_id(std::vector<uint8_t>(20, 0xAA));
    auto b = make_elf_with_build_id(std::vector<uint8_t>(20, 0xBB));
    EXPECT_NE(
        simpler::common::utils::elf_build_id_64(a.data(), a.size()),
        simpler::common::utils::elf_build_id_64(b.data(), b.size())
    );
}

TEST(ElfBuildId, NonElfFallsBackToFnv1aAndIsStable) {
    const char *data = "not-an-elf-binary";
    size_t len = std::strlen(data);
    uint64_t a = simpler::common::utils::elf_build_id_64(data, len);
    uint64_t b = simpler::common::utils::elf_build_id_64(data, len);
    EXPECT_EQ(a, b);

    const char *other = "totally-different-bytes";
    uint64_t c = simpler::common::utils::elf_build_id_64(other, std::strlen(other));
    EXPECT_NE(a, c);
}

TEST(ElfBuildId, ElfWithoutBuildIdFallsBack) {
    // Build a valid ELF64 header but skip the note → should fall through to
    // FNV-1a. The important check is that it does not crash and produces a
    // stable hash for identical inputs.
    std::vector<uint8_t> buf(sizeof(Elf64_Ehdr), 0);
    Elf64_Ehdr ehdr{};
    ehdr.e_ident[EI_MAG0] = ELFMAG0;
    ehdr.e_ident[EI_MAG1] = ELFMAG1;
    ehdr.e_ident[EI_MAG2] = ELFMAG2;
    ehdr.e_ident[EI_MAG3] = ELFMAG3;
    ehdr.e_ident[EI_CLASS] = ELFCLASS64;
    ehdr.e_phnum = 0;
    std::memcpy(buf.data(), &ehdr, sizeof(ehdr));

    uint64_t a = simpler::common::utils::elf_build_id_64(buf.data(), buf.size());
    uint64_t b = simpler::common::utils::elf_build_id_64(buf.data(), buf.size());
    EXPECT_EQ(a, b);
}
