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

#include <initializer_list>
#include "host/executor_binary_abi.h"

namespace native_admission_test {
inline std::vector<uint8_t> executor_elf(uint32_t table_type, std::initializer_list<const char *> names) {
    std::vector<uint8_t> bytes(sizeof(Elf64_Ehdr) + 4 * sizeof(Elf64_Shdr));
    Elf64_Ehdr header{};
    std::memcpy(header.e_ident, ELFMAG, SELFMAG);
    header.e_ident[EI_CLASS] = ELFCLASS64;
    header.e_ident[EI_DATA] = ELFDATA2LSB;
    header.e_ident[EI_VERSION] = EV_CURRENT;
    header.e_type = ET_DYN;
    header.e_version = EV_CURRENT;
    header.e_ehsize = sizeof(header);
    header.e_shoff = sizeof(header);
    header.e_shnum = 4;
    header.e_shentsize = sizeof(Elf64_Shdr);
    Elf64_Shdr sections[4]{};
    sections[1].sh_type = SHT_PROGBITS;
    sections[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
    sections[1].sh_addr = 0x1000;
    sections[1].sh_offset = bytes.size();
    sections[1].sh_size = 16;
    bytes.resize(bytes.size() + 16);
    sections[3].sh_type = SHT_STRTAB;
    sections[3].sh_offset = bytes.size();
    std::vector<Elf64_Sym> symbols(1);
    bytes.push_back(0);
    for (const auto *name : names) {
        Elf64_Sym sym{};
        sym.st_name = bytes.size() - sections[3].sh_offset;
        sym.st_info = ELF64_ST_INFO(STB_GLOBAL, STT_FUNC);
        sym.st_shndx = 1;
        sym.st_value = 0x1000;
        sym.st_size = 1;
        symbols.push_back(sym);
        bytes.insert(bytes.end(), name, name + std::strlen(name) + 1);
    }
    sections[3].sh_size = bytes.size() - sections[3].sh_offset;
    sections[2].sh_type = table_type;
    sections[2].sh_offset = bytes.size();
    sections[2].sh_size = symbols.size() * sizeof(Elf64_Sym);
    sections[2].sh_entsize = sizeof(Elf64_Sym);
    sections[2].sh_link = 3;
    bytes.resize(bytes.size() + sections[2].sh_size);
    std::memcpy(bytes.data() + sections[2].sh_offset, symbols.data(), sections[2].sh_size);
    std::memcpy(bytes.data(), &header, sizeof(header));
    std::memcpy(bytes.data() + header.e_shoff, sections, sizeof(sections));
    return bytes;
}

}  // namespace native_admission_test
