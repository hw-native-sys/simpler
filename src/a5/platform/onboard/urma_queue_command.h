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

#include "../../runtime/urma/urma_cq.h"
#include "../../../common/platform_comm/urma_run_binding.h"

namespace simpler::urma {

enum class QueueAction : uint64_t { CheckPrefix, RingDoorbell };

struct QueueCommand {
    QueueAction action;
    uint64_t address, depth, completed, target, result_address;
};
static_assert(sizeof(QueueCommand) == 6 * sizeof(uint64_t));

template <typename Memory>
#if defined(__CCE_AICORE__)
[aicore]
#endif
    inline UrmaQueueState execute_queue_command(const Memory &memory, const QueueCommand &command) {
    if (command.action == QueueAction::RingDoorbell)
        return memory.write(command.address, static_cast<uint32_t>(command.target)) ? UrmaQueueState::READY :
                                                                                      UrmaQueueState::FAILED;
    if (command.action != QueueAction::CheckPrefix) return UrmaQueueState::FAILED;
    return inspect_cq_entries<UrmaQueueState>(
        memory, command.address, static_cast<uint32_t>(command.depth), static_cast<uint32_t>(command.completed),
        static_cast<uint32_t>(command.target)
    );
}

}  // namespace simpler::urma
