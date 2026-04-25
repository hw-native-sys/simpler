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

#ifndef SIMPLER_COMMON_UTILS_ELF_BUILD_ID_H_
#define SIMPLER_COMMON_UTILS_ELF_BUILD_ID_H_

// Read the first 8 bytes of the ELF64 GNU Build-ID (NT_GNU_BUILD_ID) as a
// uint64_t. Falls back to FNV-1a over the full buffer when the note is
// missing (e.g. linker invoked without --build-id) or the input is not a
// well-formed ELF64 image.
//
// The Build-ID is a linker-computed hash written into `.note.gnu.build-id`
// whenever `-Wl,--build-id` is passed (the compiler default on GCC/Clang).
// Reading it is effectively free: we only touch the ELF header plus the
// notes section. Two SOs with identical Build-IDs are byte-identical by the
// linker's contract, so collisions only matter when Build-IDs are absent.

#include <cstddef>
#include <cstdint>
#include <cstring>

// <elf.h> is Linux-only. On other platforms (macOS, Windows) we embed the
// minimal subset of ELF64 types needed by this header.
#if defined(__linux__)
#include <elf.h>
#else
// Minimal ELF64 type definitions (subset of <elf.h>).
using Elf64_Half = uint16_t;
using Elf64_Word = uint32_t;
using Elf64_Off = uint64_t;
using Elf64_Addr = uint64_t;
using Elf64_Xword = uint64_t;

static constexpr int EI_MAG0 = 0;
static constexpr int EI_MAG1 = 1;
static constexpr int EI_MAG2 = 2;
static constexpr int EI_MAG3 = 3;
static constexpr int EI_CLASS = 4;
static constexpr unsigned char ELFMAG0 = 0x7f;
static constexpr unsigned char ELFMAG1 = 'E';
static constexpr unsigned char ELFMAG2 = 'L';
static constexpr unsigned char ELFMAG3 = 'F';
static constexpr unsigned char ELFCLASS64 = 2;
static constexpr Elf64_Word PT_NOTE = 4;
static constexpr Elf64_Word NT_GNU_BUILD_ID = 3;

struct Elf64_Ehdr {
    unsigned char e_ident[16];
    Elf64_Half e_type;
    Elf64_Half e_machine;
    Elf64_Word e_version;
    Elf64_Addr e_entry;
    Elf64_Off e_phoff;
    Elf64_Off e_shoff;
    Elf64_Word e_flags;
    Elf64_Half e_ehsize;
    Elf64_Half e_phentsize;
    Elf64_Half e_phnum;
    Elf64_Half e_shentsize;
    Elf64_Half e_shnum;
    Elf64_Half e_shstrndx;
};

struct Elf64_Phdr {
    Elf64_Word p_type;
    Elf64_Word p_flags;
    Elf64_Off p_offset;
    Elf64_Addr p_vaddr;
    Elf64_Addr p_paddr;
    Elf64_Xword p_filesz;
    Elf64_Xword p_memsz;
    Elf64_Xword p_align;
};

struct Elf64_Nhdr {
    Elf64_Word n_namesz;
    Elf64_Word n_descsz;
    Elf64_Word n_type;
};
#endif  // defined(__linux__)

namespace simpler::common::utils {

namespace detail {

inline uint64_t fnv1a_64(const void *data, std::size_t len) {
    constexpr uint64_t kPrime = 0x00000100000001b3ULL;
    uint64_t h = 0xcbf29ce484222325ULL;
    const auto *p = static_cast<const uint8_t *>(data);
    for (std::size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= kPrime;
    }
    return h;
}

}  // namespace detail

// Returns a 64-bit identifier derived from the ELF64 GNU Build-ID. Falls
// back to FNV-1a over the whole buffer when no Build-ID is available.
inline uint64_t elf_build_id_64(const void *data, std::size_t len) {
    if (data == nullptr || len < sizeof(Elf64_Ehdr)) {
        return detail::fnv1a_64(data, len);
    }
    const auto *base = static_cast<const uint8_t *>(data);
    Elf64_Ehdr ehdr{};
    std::memcpy(&ehdr, base, sizeof(ehdr));

    // Validate ELF magic and 64-bit class; otherwise fall back.
    if (ehdr.e_ident[EI_MAG0] != ELFMAG0 || ehdr.e_ident[EI_MAG1] != ELFMAG1 || ehdr.e_ident[EI_MAG2] != ELFMAG2 ||
        ehdr.e_ident[EI_MAG3] != ELFMAG3 || ehdr.e_ident[EI_CLASS] != ELFCLASS64) {
        return detail::fnv1a_64(data, len);
    }
    if (ehdr.e_phoff == 0 || ehdr.e_phentsize < sizeof(Elf64_Phdr)) {
        return detail::fnv1a_64(data, len);
    }
    // Guard against truncated / malformed inputs.
    std::size_t phdr_end = ehdr.e_phoff + static_cast<std::size_t>(ehdr.e_phnum) * ehdr.e_phentsize;
    if (phdr_end > len) {
        return detail::fnv1a_64(data, len);
    }

    for (std::size_t i = 0; i < ehdr.e_phnum; ++i) {
        Elf64_Phdr phdr{};
        std::memcpy(&phdr, base + ehdr.e_phoff + i * ehdr.e_phentsize, sizeof(phdr));
        if (phdr.p_type != PT_NOTE) {
            continue;
        }
        if (phdr.p_offset + phdr.p_filesz > len) {
            continue;  // Notes section lies beyond the buffer we were given.
        }
        const uint8_t *note = base + phdr.p_offset;
        const uint8_t *end = note + phdr.p_filesz;
        while (note + sizeof(Elf64_Nhdr) <= end) {
            Elf64_Nhdr nhdr{};
            std::memcpy(&nhdr, note, sizeof(nhdr));
            const uint8_t *name = note + sizeof(nhdr);
            const std::size_t name_aligned = (nhdr.n_namesz + 3u) & ~3u;
            const uint8_t *desc = name + name_aligned;
            const std::size_t desc_aligned = (nhdr.n_descsz + 3u) & ~3u;
            const uint8_t *next = desc + desc_aligned;
            if (next > end) {
                break;  // Malformed note entry.
            }
            if (nhdr.n_type == NT_GNU_BUILD_ID && nhdr.n_namesz == 4 && std::memcmp(name, "GNU\0", 4) == 0 &&
                nhdr.n_descsz >= sizeof(uint64_t)) {
                uint64_t id = 0;
                std::memcpy(&id, desc, sizeof(id));
                return id;
            }
            note = next;
        }
    }
    // No Build-ID found; the SO was likely linked without --build-id.
    return detail::fnv1a_64(data, len);
}

}  // namespace simpler::common::utils

#endif  // SIMPLER_COMMON_UTILS_ELF_BUILD_ID_H_
