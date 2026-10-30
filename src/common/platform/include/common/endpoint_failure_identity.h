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

#include <stddef.h>
#include <stdint.h>
#include <type_traits>

constexpr uint32_t kEndpointFailureKindNone = 0;
constexpr uint32_t kEndpointFailureKindRegion = 1;

struct EndpointFailureIdentity {
    uint32_t kind;
    uint32_t reserved;
    uint64_t session_instance_id_bits;
    uint64_t transaction_id;
};

static_assert(sizeof(EndpointFailureIdentity) == 24, "EndpointFailureIdentity must be 24 bytes");
static_assert(offsetof(EndpointFailureIdentity, kind) == 0, "kind offset");
static_assert(offsetof(EndpointFailureIdentity, reserved) == 4, "reserved offset");
static_assert(offsetof(EndpointFailureIdentity, session_instance_id_bits) == 8, "session offset");
static_assert(offsetof(EndpointFailureIdentity, transaction_id) == 16, "transaction offset");
static_assert(std::is_trivially_copyable_v<EndpointFailureIdentity>, "identity must be trivially copyable");
static_assert(std::is_standard_layout_v<EndpointFailureIdentity>, "identity must be standard layout");

inline bool endpoint_failure_identity_is_well_formed(const EndpointFailureIdentity &identity) {
    if (identity.reserved != 0) {
        return false;
    }
    if (identity.kind == 0) {
        return identity.session_instance_id_bits == 0 && identity.transaction_id == 0;
    }
    if (identity.kind != kEndpointFailureKindRegion) {
        return false;
    }
    if (identity.transaction_id == 0) {
        return identity.session_instance_id_bits == 0;
    }
    return true;
}
