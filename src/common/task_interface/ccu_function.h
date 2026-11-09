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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

inline constexpr uint32_t CCU_DEFINITIONS_MAGIC = 0x55434353;
inline constexpr uint32_t CCU_DEFINITIONS_VERSION = 1;
inline constexpr uint32_t CCU_BUILDER_ABI_VERSION = 2;
inline constexpr uint32_t CCU_MAX_FUNCTIONS = 4096;
enum class CcuModuleKind : uint32_t { HOST_BUILDER_ELF = 1 };

struct CcuFunction {
    std::vector<uint8_t> binary;
    std::string entry;

    void validate() const {
        if (binary.size() < 64 || std::memcmp(binary.data(), "\177ELF", 4) != 0 || binary[4] != 2 || binary[5] != 1 ||
            binary[6] != 1 || binary[16] != 3 || binary[17] != 0 || binary[20] != 1 || binary[21] != 0 ||
            binary[22] != 0 || binary[23] != 0) {
            throw std::invalid_argument("CCU module must be an ELF64 little-endian shared library");
        }
        if (binary.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("CCU module exceeds wire size limit");
        }
        auto letter = [](unsigned char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
        };
        if (entry.empty() || entry.size() > 255 || !letter(entry[0]) ||
            !std::all_of(entry.begin(), entry.end(), [&](unsigned char c) {
                return letter(c) || (c >= '0' && c <= '9');
            })) {
            throw std::invalid_argument("CCU entry must be a C symbol with at most 255 characters");
        }
    }
};

struct CcuDefinitionsHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t module_count;
    uint32_t function_count;
};
struct CcuModuleRecord {
    uint32_t offset;
    uint32_t size;
    CcuModuleKind kind;
    uint32_t abi_version;
};
struct CcuFunctionRecord {
    uint32_t module_index;
    uint32_t entry_offset;
    uint32_t entry_size;
    uint32_t reserved;
};
static_assert(std::is_trivially_copyable_v<CcuDefinitionsHeader> && std::is_standard_layout_v<CcuDefinitionsHeader>);
static_assert(std::is_trivially_copyable_v<CcuModuleRecord> && std::is_standard_layout_v<CcuModuleRecord>);
static_assert(std::is_trivially_copyable_v<CcuFunctionRecord> && std::is_standard_layout_v<CcuFunctionRecord>);
static_assert(sizeof(CcuDefinitionsHeader) == 16 && sizeof(CcuModuleRecord) == 16 && sizeof(CcuFunctionRecord) == 16);

inline std::vector<uint8_t> pack_ccu_functions(const std::vector<CcuFunction> &functions) {
    if (functions.empty()) return {};
    if (functions.size() > CCU_MAX_FUNCTIONS) throw std::invalid_argument("too many CCU functions");
    std::vector<const std::vector<uint8_t> *> modules;
    std::vector<CcuFunctionRecord> records;
    for (const auto &function : functions) {
        function.validate();
        auto found = std::find_if(modules.begin(), modules.end(), [&](const auto *module) {
            return *module == function.binary;
        });
        const auto index = static_cast<uint32_t>(found - modules.begin());
        if (found == modules.end()) modules.push_back(&function.binary);
        records.push_back({index, 0, static_cast<uint32_t>(function.entry.size()), 0});
    }
    size_t size = sizeof(CcuDefinitionsHeader) + modules.size() * sizeof(CcuModuleRecord) +
                  records.size() * sizeof(CcuFunctionRecord);
    std::vector<CcuModuleRecord> module_records;
    for (const auto *module : modules) {
        if (size + module->size() > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("CCU definitions exceed wire size limit");
        module_records.push_back(
            {static_cast<uint32_t>(size), static_cast<uint32_t>(module->size()), CcuModuleKind::HOST_BUILDER_ELF,
             CCU_BUILDER_ABI_VERSION}
        );
        size += module->size();
    }
    for (auto &record : records) {
        if (size + record.entry_size > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("CCU definitions exceed wire size limit");
        record.entry_offset = static_cast<uint32_t>(size);
        size += record.entry_size;
    }
    std::vector<uint8_t> data(size, 0);
    const CcuDefinitionsHeader header{
        CCU_DEFINITIONS_MAGIC, CCU_DEFINITIONS_VERSION, static_cast<uint32_t>(modules.size()),
        static_cast<uint32_t>(records.size())
    };
    std::memcpy(data.data(), &header, sizeof(header));
    std::memcpy(data.data() + sizeof(header), module_records.data(), module_records.size() * sizeof(CcuModuleRecord));
    std::memcpy(
        data.data() + sizeof(header) + module_records.size() * sizeof(CcuModuleRecord), records.data(),
        records.size() * sizeof(CcuFunctionRecord)
    );
    for (size_t i = 0; i < modules.size(); ++i)
        std::memcpy(data.data() + module_records[i].offset, modules[i]->data(), modules[i]->size());
    for (size_t i = 0; i < records.size(); ++i)
        std::memcpy(data.data() + records[i].entry_offset, functions[i].entry.data(), records[i].entry_size);
    return data;
}

inline std::vector<CcuFunction> unpack_ccu_functions(const void *raw, size_t size) {
    if (size == 0) return {};
    auto fail = [] {
        throw std::invalid_argument("invalid CCU definitions format, version or range");
    };
    if (raw == nullptr || size < sizeof(CcuDefinitionsHeader)) fail();
    const auto *data = static_cast<const uint8_t *>(raw);
    CcuDefinitionsHeader header{};
    std::memcpy(&header, data, sizeof(header));
    if (header.magic != CCU_DEFINITIONS_MAGIC || header.version != CCU_DEFINITIONS_VERSION ||
        header.function_count == 0 || header.function_count > CCU_MAX_FUNCTIONS || header.module_count == 0 ||
        header.module_count > header.function_count)
        fail();
    size_t offset = sizeof(header) + header.module_count * sizeof(CcuModuleRecord) +
                    header.function_count * sizeof(CcuFunctionRecord);
    if (offset > size) fail();
    std::vector<CcuModuleRecord> modules(header.module_count);
    std::memcpy(modules.data(), data + sizeof(header), modules.size() * sizeof(CcuModuleRecord));
    for (const auto &module : modules) {
        if (module.kind != CcuModuleKind::HOST_BUILDER_ELF || module.abi_version != CCU_BUILDER_ABI_VERSION ||
            module.offset != offset || module.size > size - offset)
            fail();
        offset += module.size;
    }
    std::vector<CcuFunction> functions;
    for (uint32_t i = 0; i < header.function_count; ++i) {
        CcuFunctionRecord record{};
        std::memcpy(
            &record, data + sizeof(header) + modules.size() * sizeof(CcuModuleRecord) + i * sizeof(CcuFunctionRecord),
            sizeof(record)
        );
        if (record.module_index >= modules.size() || record.reserved != 0 || record.entry_offset != offset ||
            record.entry_size > size - offset)
            fail();
        const auto &module = modules[record.module_index];
        CcuFunction function{
            {data + module.offset, data + module.offset + module.size},
            {reinterpret_cast<const char *>(data + offset), record.entry_size}
        };
        function.validate();
        functions.push_back(std::move(function));
        offset += record.entry_size;
    }
    if (offset != size) fail();
    return functions;
}
