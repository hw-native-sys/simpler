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
#pragma once

#include <elf.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace simpler::executor_abi {

inline constexpr const char *AICPU_ENTRY = "simpler_aicpu_exec_abi2";
inline constexpr const char *AIC_ENTRY = "aicore_kernel_abi2_0_mix_aic";
inline constexpr const char *AIV_ENTRY = "aicore_kernel_abi2_0_mix_aiv";

// Inspect the actual loaded image, not a sidecar, filename, or debug string.
// Unsupported ELF encodings and extended numbering are rejected, not guessed.
inline bool has_entry(const std::vector<uint8_t> &image, const char *name, uint32_t symbol_table_type) {
    const auto range = [&](uint64_t offset, uint64_t bytes) {
        return offset <= image.size() && bytes <= image.size() - offset;
    };
    if (!range(0, sizeof(Elf64_Ehdr))) return false;
    Elf64_Ehdr header{};
    std::memcpy(&header, image.data(), sizeof(header));
    if (std::memcmp(header.e_ident, ELFMAG, SELFMAG) != 0 || header.e_ident[EI_CLASS] != ELFCLASS64 ||
        header.e_ident[EI_DATA] != ELFDATA2LSB || header.e_ident[EI_VERSION] != EV_CURRENT ||
        header.e_version != EV_CURRENT || (header.e_type != ET_EXEC && header.e_type != ET_DYN) ||
        header.e_ehsize != sizeof(header) || header.e_shentsize != sizeof(Elf64_Shdr) || header.e_shnum == 0 ||
        !range(header.e_shoff, uint64_t(header.e_shnum) * sizeof(Elf64_Shdr)))
        return false;
    const auto section = [&](uint16_t index) {
        Elf64_Shdr value{};
        std::memcpy(&value, image.data() + header.e_shoff + uint64_t(index) * sizeof(value), sizeof(value));
        return value;
    };
    const size_t name_bytes = std::strlen(name) + 1;
    for (uint16_t index = 0; index < header.e_shnum; ++index) {
        const auto table = section(index);
        if (table.sh_type != symbol_table_type) continue;
        if (table.sh_entsize != sizeof(Elf64_Sym) || table.sh_size % sizeof(Elf64_Sym) != 0 ||
            table.sh_link >= header.e_shnum || !range(table.sh_offset, table.sh_size))
            return false;
        const auto strings = section(static_cast<uint16_t>(table.sh_link));
        if (strings.sh_type != SHT_STRTAB || !range(strings.sh_offset, strings.sh_size)) return false;
        for (uint64_t offset = 0; offset < table.sh_size; offset += sizeof(Elf64_Sym)) {
            Elf64_Sym symbol{};
            std::memcpy(&symbol, image.data() + table.sh_offset + offset, sizeof(symbol));
            if (ELF64_ST_TYPE(symbol.st_info) != STT_FUNC || ELF64_ST_BIND(symbol.st_info) != STB_GLOBAL ||
                ELF64_ST_VISIBILITY(symbol.st_other) != STV_DEFAULT || symbol.st_shndx == SHN_UNDEF ||
                symbol.st_shndx >= header.e_shnum || symbol.st_name >= strings.sh_size ||
                name_bytes > strings.sh_size - symbol.st_name)
                continue;
            if (std::memcmp(image.data() + strings.sh_offset + symbol.st_name, name, name_bytes) != 0) continue;
            const auto code = section(symbol.st_shndx);
            if (code.sh_type != SHT_PROGBITS ||
                (code.sh_flags & (SHF_ALLOC | SHF_EXECINSTR)) != (SHF_ALLOC | SHF_EXECINSTR) ||
                !range(code.sh_offset, code.sh_size) || symbol.st_value < code.sh_addr ||
                symbol.st_value - code.sh_addr >= code.sh_size || symbol.st_size == 0 ||
                symbol.st_size > code.sh_size - (symbol.st_value - code.sh_addr))
                return false;
            return true;
        }
    }
    return false;
}

inline void validate(const std::vector<uint8_t> &aicpu, const std::vector<uint8_t> &aicore) {
    if (!has_entry(aicpu, AICPU_ENTRY, SHT_DYNSYM) || !has_entry(aicore, AIC_ENTRY, SHT_SYMTAB) ||
        !has_entry(aicore, AIV_ENTRY, SHT_SYMTAB)) {
        throw std::invalid_argument(
            "executor ABI 2 requires defined AICPU, AIC and AIV entry symbols; rebuild all runtime binaries together"
        );
    }
}

}  // namespace simpler::executor_abi
