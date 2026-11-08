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

#include "completion_token.h"
#include "platform_comm/ccu/ccu_device_access.h"

namespace simpler::ccu {

inline device::Reference completion_reference(uint64_t addr, uint64_t sequence, uint32_t binding) {
    return {reinterpret_cast<device::Workspace *>(addr), binding, sequence};
}

inline CompletionPollResult poll_request(uint64_t addr, uint64_t sequence, uint32_t binding) {
    switch (device::Protocol<>::poll(completion_reference(addr, sequence, binding))) {
    case device::Outcome::PENDING:
        return {CompletionPollState::PENDING, SIMPLER_ERROR_NONE};
    case device::Outcome::SUCCEEDED:
        return {CompletionPollState::READY, SIMPLER_ERROR_NONE};
    case device::Outcome::FAILED:
        return {CompletionPollState::FAILED, SIMPLER_ERROR_CCU_FAILED};
    default:
        return {CompletionPollState::UNCONFIRMED, SIMPLER_ERROR_CCU_UNCONFIRMED};
    }
}

inline bool retire_request(uint64_t addr, uint64_t sequence, uint32_t binding) {
    return device::Protocol<>::retire(completion_reference(addr, sequence, binding));
}

}  // namespace simpler::ccu
