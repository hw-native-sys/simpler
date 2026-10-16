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

#include <atomic>
#include <thread>

#include "aicore_teardown.h"
#include "aicpu/platform_regs.h"

// A controllable answering core for platform tests. Constructing it permits
// responses; withholding it models a silent core even if COND retains EXITED.
// It owns no target storage and must be destroyed before that storage goes away.
class AicoreExitResponder {
public:
    explicit AicoreExitResponder(AicoreExitTarget target) :
        worker_([this, target] {
            uint32_t published = 0;
            while (!stop_.load(std::memory_order_acquire)) {
                const uint32_t request = __atomic_load_n(&target.teardown->exit_request, __ATOMIC_ACQUIRE);
                if (request != 0 && request != published &&
                    read_reg(target.reg_addr, RegId::DATA_MAIN_BASE) == AICORE_EXIT_SIGNAL) {
                    write_reg(target.reg_addr, RegId::COND, AICORE_EXITED_VALUE);
                    __atomic_store_n(&target.teardown->exit_ack, request, __ATOMIC_RELEASE);
                    published = request;
                }
                std::this_thread::yield();
            }
        }) {}

    ~AicoreExitResponder() {
        stop_.store(true, std::memory_order_release);
        worker_.join();
    }

    AicoreExitResponder(const AicoreExitResponder &) = delete;
    AicoreExitResponder &operator=(const AicoreExitResponder &) = delete;

private:
    std::atomic<bool> stop_{false};
    std::thread worker_;
};
