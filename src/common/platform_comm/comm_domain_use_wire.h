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

#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "comm_domain_use.h"

// Host local-mailbox trailer following the unchanged TaskArgs blob. Neither
// this identity sidecar nor its header is part of Tensor or device Graph ABI.
struct CommDomainUsesHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t count;
    uint32_t row_bytes;
};
inline constexpr uint32_t COMM_DOMAIN_USES_MAGIC = 0x53554443;
inline constexpr uint32_t COMM_DOMAIN_USES_VERSION = 1;
static_assert(sizeof(CommDomainUsesHeader) == 16);
static_assert(sizeof(CommDomainUse) == 24);

inline void validate_comm_domain_uses(const std::vector<CommDomainUse> &uses) {
    for (size_t i = 0; i < uses.size(); ++i) {
        const auto &use = uses[i];
        if ((use.kind != CommDomainKind::LOCAL && use.kind != CommDomainKind::GLOBAL) || use.reserved != 0 ||
            use.id == 0 || use.device_context == 0)
            throw std::invalid_argument("invalid communication-domain use identity");
        for (size_t j = 0; j < i; ++j) {
            if ((uses[j].kind == use.kind && uses[j].id == use.id) || uses[j].device_context == use.device_context)
                throw std::invalid_argument("duplicate communication-domain use identity");
        }
    }
}

inline size_t comm_domain_uses_wire_size(const std::vector<CommDomainUse> &uses) {
    if (uses.size() > UINT32_MAX || uses.size() > (SIZE_MAX - sizeof(CommDomainUsesHeader)) / sizeof(CommDomainUse))
        throw std::length_error("communication-domain use sidecar is too large");
    return sizeof(CommDomainUsesHeader) + uses.size() * sizeof(CommDomainUse);
}

inline void write_comm_domain_uses(void *destination, size_t capacity, const std::vector<CommDomainUse> &uses) {
    validate_comm_domain_uses(uses);
    if (destination == nullptr || comm_domain_uses_wire_size(uses) > capacity)
        throw std::length_error("communication-domain use sidecar exceeds frame capacity");
    const CommDomainUsesHeader header{
        COMM_DOMAIN_USES_MAGIC, COMM_DOMAIN_USES_VERSION, static_cast<uint32_t>(uses.size()), sizeof(CommDomainUse)
    };
    std::memcpy(destination, &header, sizeof(header));
    if (!uses.empty())
        std::memcpy(static_cast<uint8_t *>(destination) + sizeof(header), uses.data(), uses.size() * sizeof(uses[0]));
}

inline std::vector<CommDomainUse> read_comm_domain_uses(const void *source, size_t capacity) {
    if (source == nullptr || capacity < sizeof(CommDomainUsesHeader))
        throw std::length_error("communication-domain use sidecar header is missing or truncated");
    CommDomainUsesHeader header{};
    std::memcpy(&header, source, sizeof(header));
    if (header.magic != COMM_DOMAIN_USES_MAGIC || header.version != COMM_DOMAIN_USES_VERSION ||
        header.row_bytes != sizeof(CommDomainUse))
        throw std::invalid_argument("unsupported communication-domain use sidecar format");
    if (header.count > (capacity - sizeof(header)) / sizeof(CommDomainUse))
        throw std::length_error("communication-domain use sidecar rows exceed frame capacity");
    std::vector<CommDomainUse> uses(header.count);
    if (!uses.empty())
        std::memcpy(uses.data(), static_cast<const uint8_t *>(source) + sizeof(header), uses.size() * sizeof(uses[0]));
    validate_comm_domain_uses(uses);
    return uses;
}
