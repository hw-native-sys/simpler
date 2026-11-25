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
#include <atomic>
#include <cstdint>

#include "aicpu/aicpu_device_config.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu_executor.h"
#include "common/platform_config.h"
#include "supervisor.h"
#include "runtime.h"
#include "scheduler/scheduler_watchdog.h"
#include "spin_hint.h"

namespace {

// The same budget host_build_graph's own waits use, read from the same knob.
// Its copy of this is file-local to that translation unit.
uint64_t scheduler_timeout_cycles() {
    const int32_t timeout_ms = get_scheduler_timeout_ms();
    return timeout_ms > 0 ? static_cast<uint64_t>(timeout_ms) * (PLATFORM_PROF_SYS_CNT_FREQ / 1000) :
                            SCHEDULER_TIMEOUT_CYCLES;
}

}  // namespace

bool AicpuExecutor::schedules_on_aicore(const Runtime *runtime) const {
    return simpler::percounter::percounter_mode_enabled(runtime);
}

int32_t AicpuExecutor::post_handshake_init(Runtime *runtime) {
    if (is_chip_swimlane_enabled() && runtime->dev.chip_swimlane_level == 1)
        chip_swimlane_aicpu_use_self_scheduled_cores();
    return simpler::percounter::percounter_post_handshake_init(runtime, aicore_lifecycle_, scheduler_timeout_cycles());
}

void AicpuExecutor::publish_partition(Runtime *runtime, int32_t tidx, int32_t) {
    if (tidx != 0) return;
    auto *timing = simpler::percounter::percounter_aicpu_timing(simpler::percounter::percounter_control(runtime), tidx);
    if (timing != nullptr) timing->ready_publish_start_cycles = get_sys_cnt_aicpu();
    aicore_lifecycle_.publish_lifecycle_start();
    if (timing != nullptr) {
        timing->ready_publish_end_cycles = get_sys_cnt_aicpu();
        cache_flush_range(timing, sizeof(*timing));
    }
}

int32_t AicpuExecutor::supervise(Runtime *runtime, int32_t thread_idx) {
    return simpler::percounter::percounter_supervise(
        runtime, scheduler_timeout_cycles(), aicpu_thread_num_, thread_idx, [this, thread_idx] {
            return service_args_dump(thread_idx);
        }
    );
}

// Materialize Graph bodies before servicing diagnostics until shutdown.
void AicpuExecutor::await_shutdown(Runtime *runtime, int32_t thread_idx) {
    simpler::percounter::percounter_assist(runtime, thread_idx);
    auto *timing =
        simpler::percounter::percounter_aicpu_timing(simpler::percounter::percounter_control(runtime), thread_idx);
    if (timing != nullptr) timing->shutdown_start_cycles = get_sys_cnt_aicpu();
    uint32_t checks = 0;
    while (!shutdown_ready_.load(std::memory_order_acquire)) {
        service_args_dump(thread_idx);
        if (++checks == 64) {
            checks = 0;
            chip_swimlane_aicpu_poll_self_scheduled_cores(thread_idx, aicpu_thread_num_, runtime->get_worker_count());
        }
        SPIN_WAIT_HINT();
    }
    if (timing != nullptr) {
        timing->shutdown_end_cycles = get_sys_cnt_aicpu();
        cache_flush_range(timing, sizeof(*timing));
    }
}

int32_t AicpuExecutor::settle_run_error(Runtime *runtime) {
    return simpler::percounter::percounter_settle_run_error(runtime);
}
