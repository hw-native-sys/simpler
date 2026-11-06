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
/**
 * This runtime's seven AicpuExecutor steps.
 *
 * Everything around them — the thread rendezvous, the handshake barriers, the
 * completion gate, the terminal record, teardown — is host_build_graph's and is
 * compiled into this runtime unchanged. These definitions are strong, so the
 * linker takes them over the weak ones that translation unit carries.
 *
 * Three of them differ because percounter brings the cores up itself: it
 * publishes the discovered topology into the control block rather than handing
 * out per-worker contexts, and the lanes start from that block without the
 * bootstrap acknowledgement the resident scheduler waits for. The other three
 * are the run-time waits, which percounter's own supervisor bounds.
 */

#include <atomic>
#include <cstdint>

#include "aicpu/aicpu_device_config.h"
#include "aicpu_executor.h"
#include "common/platform_config.h"
#include "percounter/percounter_aicpu.h"
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

// A percounter graph is the only thing this runtime is built to run, so the
// legacy dispatch path behind a false here is never the right answer.
bool AicpuExecutor::drives_cores(const Runtime *runtime) const {
    return simpler::hbg::percounter::percounter_mode_enabled(runtime);
}

int32_t AicpuExecutor::post_handshake_init(Runtime *runtime) {
    return simpler::hbg::percounter::percounter_post_handshake_init(
        runtime, aicore_lifecycle_, scheduler_timeout_cycles()
    );
}

void AicpuExecutor::publish_partition(Runtime *runtime, int32_t tidx, int32_t nthreads) {
    simpler::hbg::percounter::percounter_publish_partition(runtime, aicore_lifecycle_, tidx, nthreads);
}

// The lanes take their start from the control block the host published before
// launch, so there is no bootstrap acknowledgement to wait for.
bool AicpuExecutor::awaits_bootstrap() const { return false; }

int32_t AicpuExecutor::supervise(Runtime *runtime) {
    return simpler::hbg::percounter::percounter_supervise(runtime, scheduler_timeout_cycles(), aicpu_thread_num_);
}

// This thread first materializes any graph bodies the lanes will run, then
// waits. The percounter supervisor's own progress watchdog bounds the wait;
// the ready watchdog's stamp is never advanced in this runtime.
bool AicpuExecutor::await_shutdown(Runtime *runtime) {
    simpler::hbg::percounter::percounter_assist(runtime);
    while (!shutdown_ready_.load(std::memory_order_acquire)) {
        SPIN_WAIT_HINT();
    }
    return false;
}

// The default settles the resident scheduler's run control, which this runtime
// does not have: its lanes report into the percounter control block instead.
int32_t AicpuExecutor::settle_run_error(Runtime *runtime) {
    return simpler::hbg::percounter::percounter_settle_run_error(runtime);
}
