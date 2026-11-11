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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

inline constexpr uint32_t CCU_MODULE_PACKAGE_MAGIC = 0x55434353U;
inline constexpr uint32_t CCU_MAX_MODULES = 4096U;

struct CcuModulePackageHeader {
    uint32_t magic;
    uint32_t struct_size;
    uint32_t module_record_size;
    uint32_t module_count;
    uint32_t total_size;
};

struct CcuModulePackageRecord {
    uint32_t struct_size;
    uint32_t module_offset;
    uint32_t module_size;
};

static_assert(
    std::is_trivially_copyable_v<CcuModulePackageHeader> && std::is_standard_layout_v<CcuModulePackageHeader>
);
static_assert(
    std::is_trivially_copyable_v<CcuModulePackageRecord> && std::is_standard_layout_v<CcuModulePackageRecord>
);
static_assert(sizeof(CcuModulePackageHeader) == 20);
static_assert(sizeof(CcuModulePackageRecord) == 12);

inline std::vector<uint8_t> pack_ccu_modules(const std::vector<std::vector<uint8_t>> &modules) {
    if (modules.empty()) return {};
    if (modules.size() > CCU_MAX_MODULES) throw std::invalid_argument("too many CCU modules");

    size_t total = sizeof(CcuModulePackageHeader) + modules.size() * sizeof(CcuModulePackageRecord);
    std::vector<CcuModulePackageRecord> records;
    records.reserve(modules.size());
    for (const auto &module : modules) {
        if (module.empty()) throw std::invalid_argument("CCU module bytes must not be empty");
        if (module.size() > std::numeric_limits<uint32_t>::max() ||
            total > std::numeric_limits<uint32_t>::max() - module.size())
            throw std::invalid_argument("CCU module package exceeds wire size limit");
        records.push_back(
            {sizeof(CcuModulePackageRecord), static_cast<uint32_t>(total), static_cast<uint32_t>(module.size())}
        );
        total += module.size();
    }

    const CcuModulePackageHeader header{
        CCU_MODULE_PACKAGE_MAGIC, sizeof(CcuModulePackageHeader), sizeof(CcuModulePackageRecord),
        static_cast<uint32_t>(modules.size()), static_cast<uint32_t>(total)
    };
    std::vector<uint8_t> result(total, 0);
    std::memcpy(result.data(), &header, sizeof(header));
    std::memcpy(result.data() + sizeof(header), records.data(), records.size() * sizeof(records.front()));
    for (size_t i = 0; i < modules.size(); ++i)
        std::memcpy(result.data() + records[i].module_offset, modules[i].data(), modules[i].size());
    return result;
}

inline uint32_t ccu_module_count(const void *raw, size_t size) {
    if (size == 0) return 0;
    const auto fail = [] {
        throw std::invalid_argument("invalid CCU module package");
    };
    if (raw == nullptr || size < sizeof(CcuModulePackageHeader)) fail();
    const auto *data = static_cast<const uint8_t *>(raw);
    CcuModulePackageHeader header{};
    std::memcpy(&header, data, sizeof(header));
    if (header.magic != CCU_MODULE_PACKAGE_MAGIC || header.struct_size != sizeof(CcuModulePackageHeader) ||
        header.module_record_size != sizeof(CcuModulePackageRecord) || header.module_count == 0 ||
        header.module_count > CCU_MAX_MODULES || header.total_size != size)
        fail();
    const size_t records_size = static_cast<size_t>(header.module_count) * sizeof(CcuModulePackageRecord);
    if (records_size > size - sizeof(header)) fail();
    size_t expected_offset = sizeof(header) + records_size;
    for (uint32_t i = 0; i < header.module_count; ++i) {
        CcuModulePackageRecord record{};
        std::memcpy(&record, data + sizeof(header) + i * sizeof(record), sizeof(record));
        if (record.struct_size != sizeof(CcuModulePackageRecord) || record.module_offset != expected_offset ||
            record.module_size == 0 || record.module_size > size - expected_offset)
            fail();
        expected_offset += record.module_size;
    }
    if (expected_offset != size) fail();
    return header.module_count;
}

inline std::vector<std::vector<uint8_t>> unpack_ccu_modules(const void *raw, size_t size) {
    const auto count = ccu_module_count(raw, size);
    const auto *data = static_cast<const uint8_t *>(raw);
    std::vector<std::vector<uint8_t>> modules;
    modules.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        CcuModulePackageRecord record{};
        std::memcpy(&record, data + sizeof(CcuModulePackageHeader) + i * sizeof(record), sizeof(record));
        modules.emplace_back(data + record.module_offset, data + record.module_offset + record.module_size);
    }
    return modules;
}
